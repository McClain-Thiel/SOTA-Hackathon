"""Host-side OpenRouter capture and concurrency gateway for Harbor/pi rollouts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import anyio
import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from repoopt.store.base import ArtifactStore, StoragePaths, sha256_bytes


@dataclass(frozen=True)
class AttemptIdentity:
    """Validated attempt routing identity parsed from container bearer token."""

    experiment_id: str
    task_id: str
    attempt_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.experiment_id, self.task_id, self.attempt_id)


def parse_attempt_token(authorization: str | None, x_attempt: str | None = None) -> AttemptIdentity:
    """Parse `Bearer attempt:<exp>:<task>:<attempt>` or `X-RepoOpt-Attempt` header."""
    raw = ""
    if x_attempt and x_attempt.strip():
        raw = x_attempt.strip()
        if raw.startswith("attempt:"):
            parts = raw.split(":", 3)
            if len(parts) == 4 and all(parts[1:]):
                return AttemptIdentity(parts[1], parts[2], parts[3])
        parts = raw.split("/", 2)
        if len(parts) == 3 and all(parts):
            return AttemptIdentity(parts[0], parts[1], parts[2])

    if authorization and authorization.strip():
        token = authorization.strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token.startswith("attempt:"):
            parts = token.split(":", 3)
            if len(parts) == 4 and all(parts[1:]):
                return AttemptIdentity(parts[1], parts[2], parts[3])

    raise HTTPException(
        status_code=401,
        detail="Missing or invalid attempt token (expected Bearer attempt:<exp>:<task>:<attempt>)",
    )


def format_attempt_token(experiment_id: str, task_id: str, attempt_id: str) -> str:
    """Format an attempt-scoped proxy tokensafe to inject into untrusted containers."""
    for label, val in (("experiment_id", experiment_id), ("task_id", task_id), ("attempt_id", attempt_id)):
        if not val or ":" in val or "/" in val:
            raise ValueError(f"{label} must be non-empty and cannot contain ':' or '/': {val!r}")
    return f"attempt:{experiment_id}:{task_id}:{attempt_id}"


@dataclass
class _AttemptState:
    identity: AttemptIdentity
    turn_counter: int = 0
    segment_counter: int = 0
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    flushed_segments: list[str] = field(default_factory=list)
    last_flush_monotonic: float = field(default_factory=time.monotonic)


class _SSEStreamAccumulator:
    """Reassembles OpenAI-compatible SSE chat completion chunks into a complete response record."""

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self.response_id: str | None = None
        self.model: str | None = None
        self.created: int | None = None
        self.role: str = "assistant"
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls_by_idx: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.saw_done: bool = False
        self._line_buffer: str = ""

    def feed_bytes(self, raw_bytes: bytes) -> None:
        text = raw_bytes.decode("utf-8", errors="replace")
        self._line_buffer += text
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._process_line(line.rstrip("\r"))

    def finalize(self) -> dict[str, Any]:
        if self._line_buffer.strip():
            self._process_line(self._line_buffer.strip())
            self._line_buffer = ""

        ordered_tool_calls = [
            self.tool_calls_by_idx[idx] for idx in sorted(self.tool_calls_by_idx)
        ]
        message: dict[str, Any] = {
            "role": self.role,
            "content": "".join(self.content_parts),
        }
        reasoning = "".join(self.reasoning_parts)
        if reasoning:
            message["reasoning_content"] = reasoning
        if ordered_tool_calls:
            message["tool_calls"] = ordered_tool_calls

        result: dict[str, Any] = {
            "id": self.response_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason,
                }
            ],
            "stream_chunk_count": len(self.chunks),
        }
        if self.usage is not None:
            result["usage"] = self.usage
        return result

    def _process_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload:
            return
        if payload == "[DONE]":
            self.saw_done = True
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        self.chunks.append(event)
        if self.response_id is None and isinstance(event.get("id"), str):
            self.response_id = event["id"]
        if self.model is None and isinstance(event.get("model"), str):
            self.model = event["model"]
        if self.created is None and isinstance(event.get("created"), int):
            self.created = event["created"]
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice = choices[0]
        if not isinstance(choice, dict):
            return
        if isinstance(choice.get("finish_reason"), str):
            self.finish_reason = choice["finish_reason"]

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        if isinstance(delta.get("role"), str):
            self.role = delta["role"]
        if isinstance(delta.get("content"), str):
            self.content_parts.append(delta["content"])
        for rkey in ("reasoning", "reasoning_content"):
            if isinstance(delta.get(rkey), str):
                self.reasoning_parts.append(delta[rkey])
        if isinstance(delta.get("reasoning_details"), list):
            for item in delta["reasoning_details"]:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    self.reasoning_parts.append(item["text"])

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc_delta in tool_calls:
                if not isinstance(tc_delta, dict):
                    continue
                idx = int(tc_delta.get("index", 0))
                entry = self.tool_calls_by_idx.setdefault(
                    idx,
                    {
                        "id": "",
                        "type": tc_delta.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if isinstance(tc_delta.get("id"), str) and tc_delta["id"]:
                    entry["id"] = tc_delta["id"]
                if isinstance(tc_delta.get("type"), str) and tc_delta["type"]:
                    entry["type"] = tc_delta["type"]
                fn = tc_delta.get("function")
                if isinstance(fn, dict):
                    if isinstance(fn.get("name"), str):
                        entry["function"]["name"] += fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        entry["function"]["arguments"] += fn["arguments"]


class CaptureGateway:
    """OpenAI-compatible host proxy that injects API keys, caps concurrency, and flushes traces."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        upstream_base_url: str = "https://openrouter.ai/api/v1",
        upstream_api_key: str | None = None,
        max_concurrency: int = 8,
        flush_interval_sec: float = 15.0,
        request_timeout_sec: float = 300.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.store = store
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.max_concurrency = max(1, max_concurrency)
        self.flush_interval_sec = max(0.05, flush_interval_sec)
        self.request_timeout_sec = request_timeout_sec
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._lock = asyncio.Lock()
        self._attempts: dict[tuple[str, str, str], _AttemptState] = {}
        self._active_inflight = 0
        self._peak_inflight = 0
        self._owns_client = http_client is None
        self._client = http_client
        self._flush_task: asyncio.Task[None] | None = None
        self.app = self._build_app()

    @property
    def peak_inflight(self) -> int:
        return self._peak_inflight

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.request_timeout_sec)
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.upstream_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="Host CaptureGateway has no OPENROUTER_API_KEY configured",
            )
        return key

    def _upstream_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in urlparse(self.upstream_base_url).netloc:
            headers["HTTP-Referer"] = "https://github.com/repoopt"
            headers["X-Title"] = "RepoOpt"
        return headers

    async def start(self) -> None:
        """Start the background periodic flush loop."""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    async def stop(self) -> None:
        """Stop the periodic flush loop, flush pending events, and close owned HTTP client."""
        if self._flush_task is not None:
            self._flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._flush_task
            self._flush_task = None
        await self.flush_all()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _periodic_flush_loop(self) -> None:
        poll_interval = min(1.0, max(0.05, self.flush_interval_sec / 2.0))
        while True:
            await asyncio.sleep(poll_interval)
            now = time.monotonic()
            async with self._lock:
                due = [
                    key
                    for key, st in self._attempts.items()
                    if st.pending_events and (now - st.last_flush_monotonic) >= self.flush_interval_sec
                ]
            for key in due:
                await self.flush_attempt(*key)

    async def _get_or_create_attempt(self, identity: AttemptIdentity) -> _AttemptState:
        async with self._lock:
            state = self._attempts.get(identity.key)
            if state is None:
                state = _AttemptState(identity=identity)
                self._attempts[identity.key] = state
            return state

    async def append_event(
        self,
        identity: AttemptIdentity,
        event_type: str,
        payload: dict[str, Any],
        *,
        flush_now: bool = True,
    ) -> None:
        """Append an event to the attempt buffer and optionally flush a segment immediately."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": identity.experiment_id,
            "task_id": identity.task_id,
            "attempt_id": identity.attempt_id,
            "event_type": event_type,
            "payload": payload,
        }
        async with self._lock:
            state = self._attempts.get(identity.key)
            if state is None:
                state = _AttemptState(identity=identity)
                self._attempts[identity.key] = state
            state.pending_events.append(event)
        if flush_now:
            await self.flush_attempt(*identity.key)

    async def flush_attempt(
        self, experiment_id: str, task_id: str, attempt_id: str
    ) -> str | None:
        """Flush buffered events for one attempt as an immutable content-hashed JSONL segment."""
        key = (experiment_id, task_id, attempt_id)
        async with self._lock:
            state = self._attempts.get(key)
            if state is None or not state.pending_events:
                return None
            batch = state.pending_events
            state.pending_events = []
            state.segment_counter += 1
            seq_idx = state.segment_counter
            state.last_flush_monotonic = time.monotonic()

        body = ("\n".join(json.dumps(item, ensure_ascii=False) for item in batch) + "\n").encode(
            "utf-8"
        )
        seg_key = StoragePaths.event_segment(
            experiment_id,
            task_id,
            attempt_id,
            seq_idx,
            sha256_bytes(body),
        )
        self.store.put_bytes(seg_key, body, immutable=True, content_type="application/x-ndjson")
        async with self._lock:
            if key in self._attempts:
                self._attempts[key].flushed_segments.append(seg_key)
        return seg_key

    async def flush_all(self) -> list[str]:
        """Flush all pending event buffers across every known attempt."""
        async with self._lock:
            keys = list(self._attempts.keys())
        flushed: list[str] = []
        for key in keys:
            seg = await self.flush_attempt(*key)
            if seg is not None:
                flushed.append(seg)
        return flushed

    async def _record_provider_turn(
        self,
        identity: AttemptIdentity,
        request_body: dict[str, Any],
        response_payload: dict[str, Any],
        *,
        status_code: int,
        latency_ms: float,
        streamed: bool,
        completed: bool,
        flush_now: bool = True,
    ) -> int:
        async with self._lock:
            state = self._attempts.get(identity.key)
            if state is None:
                state = _AttemptState(identity=identity)
                self._attempts[identity.key] = state
            state.turn_counter += 1
            turn_idx = state.turn_counter

        record = {
            "turn_index": turn_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": identity.experiment_id,
            "task_id": identity.task_id,
            "attempt_id": identity.attempt_id,
            "upstream_base_url": self.upstream_base_url,
            "streamed": streamed,
            "completed": completed,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "request": request_body,
            "response": response_payload,
        }
        raw_key = StoragePaths.raw_provider_event(
            identity.experiment_id,
            identity.task_id,
            identity.attempt_id,
            turn_idx,
        )
        self.store.put_json(raw_key, record, immutable=True)

        summary_event = {
            "turn_index": turn_idx,
            "raw_event_key": raw_key,
            "model": request_body.get("model"),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 3),
            "completed": completed,
            "usage": response_payload.get("usage"),
        }
        await self.append_event(identity, "model_turn", summary_event, flush_now=flush_now)
        return turn_idx

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RepoOpt Capture Gateway", version="0.4.0")

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {
                "status": "ok",
                "max_concurrency": self.max_concurrency,
                "active_inflight": self._active_inflight,
                "peak_inflight": self._peak_inflight,
            }

        @app.get("/v1/models")
        async def list_models(
            authorization: str | None = Header(default=None),
            x_repoopt_attempt: str | None = Header(default=None),
        ) -> Response:
            parse_attempt_token(authorization, x_repoopt_attempt)
            client = self._get_client()
            resp = await client.get(
                f"{self.upstream_base_url}/models",
                headers=self._upstream_headers(),
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_repoopt_attempt: str | None = Header(default=None),
        ) -> Response:
            identity = parse_attempt_token(authorization, x_repoopt_attempt)
            request_body = await request.json()
            if not isinstance(request_body, dict):
                raise HTTPException(status_code=400, detail="Request body must be a JSON object")

            is_stream = bool(request_body.get("stream", False))
            flush_immediately = request.headers.get("x-repoopt-defer-flush") != "1"
            client = self._get_client()
            upstream_url = f"{self.upstream_base_url}/chat/completions"

            if not is_stream:
                async with self._semaphore:
                    self._active_inflight += 1
                    self._peak_inflight = max(self._peak_inflight, self._active_inflight)
                    t0 = time.perf_counter()
                    try:
                        resp = await client.post(
                            upstream_url,
                            headers=self._upstream_headers(),
                            json=request_body,
                        )
                    finally:
                        self._active_inflight -= 1
                    latency_ms = (time.perf_counter() - t0) * 1000.0

                try:
                    parsed_resp = resp.json()
                except Exception:
                    parsed_resp = {"raw_text": resp.text}

                await self._record_provider_turn(
                    identity,
                    request_body,
                    parsed_resp,
                    status_code=resp.status_code,
                    latency_ms=latency_ms,
                    streamed=False,
                    completed=(200 <= resp.status_code < 300),
                    flush_now=flush_immediately,
                )
                return JSONResponse(content=parsed_resp, status_code=resp.status_code)

            # Streaming SSE mode: hold semaphore across stream lifetime and capture in finally.
            await self._semaphore.acquire()
            self._active_inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._active_inflight)
            t0 = time.perf_counter()
            req = client.build_request(
                "POST",
                upstream_url,
                headers=self._upstream_headers(),
                json=request_body,
            )
            try:
                upstream_resp = await client.send(req, stream=True)
            except Exception:
                self._active_inflight -= 1
                self._semaphore.release()
                raise

            if upstream_resp.status_code >= 400:
                err_bytes = await upstream_resp.aread()
                await upstream_resp.aclose()
                self._active_inflight -= 1
                self._semaphore.release()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                try:
                    parsed_err = json.loads(err_bytes.decode("utf-8", errors="replace"))
                except Exception:
                    parsed_err = {"raw_text": err_bytes.decode("utf-8", errors="replace")}
                await self._record_provider_turn(
                    identity,
                    request_body,
                    parsed_err,
                    status_code=upstream_resp.status_code,
                    latency_ms=latency_ms,
                    streamed=True,
                    completed=False,
                    flush_now=flush_immediately,
                )
                return JSONResponse(content=parsed_err, status_code=upstream_resp.status_code)

            acc = _SSEStreamAccumulator()

            async def event_stream() -> AsyncIterator[bytes]:
                stream_completed = False
                try:
                    async for chunk in upstream_resp.aiter_bytes():
                        acc.feed_bytes(chunk)
                        yield chunk
                    stream_completed = True
                finally:
                    with anyio.CancelScope(shield=True):
                        with contextlib.suppress(Exception):
                            await upstream_resp.aclose()
                        self._active_inflight -= 1
                        self._semaphore.release()
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        rebuilt = acc.finalize()
                        await self._record_provider_turn(
                            identity,
                            request_body,
                            rebuilt,
                            status_code=upstream_resp.status_code,
                            latency_ms=latency_ms,
                            streamed=True,
                            completed=bool(stream_completed and (acc.saw_done or acc.finish_reason is not None)),
                            flush_now=flush_immediately,
                        )

            return StreamingResponse(
                event_stream(),
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
            )

        return app


class RunningGateway:
    """Context-managed Uvicorn server running CaptureGateway on a local ephemeral port."""

    def __init__(self, gateway: CaptureGateway, *, host: str = "0.0.0.0", port: int = 0) -> None:
        self.gateway = gateway
        self.host = host
        self.requested_port = port
        self.bound_port: int | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def local_base_url(self) -> str:
        if self.bound_port is None:
            raise RuntimeError("Gateway is not started")
        return f"http://127.0.0.1:{self.bound_port}/v1"

    @property
    def container_base_url(self) -> str:
        if self.bound_port is None:
            raise RuntimeError("Gateway is not started")
        return f"http://host.docker.internal:{self.bound_port}/v1"

    async def __aenter__(self) -> "RunningGateway":
        await self.gateway.start()
        config = uvicorn.Config(
            self.gateway.app,
            host=self.host,
            port=self.requested_port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            if self._task.done():
                exc = self._task.exception()
                raise RuntimeError(f"Gateway server failed to start: {exc}")
            await asyncio.sleep(0.02)
        sockets = [
            sock for server in self._server.servers for sock in getattr(server, "sockets", [])
        ]
        if not sockets:
            raise RuntimeError("Gateway server bound no sockets")
        self.bound_port = int(sockets[0].getsockname()[1])
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self.gateway.stop()
