"""Tests for ArtifactStore (local + mocked S3) and Host CaptureGateway."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from repoopt.proxy.gateway import CaptureGateway, format_attempt_token
from repoopt.store.base import StoragePaths, sha256_bytes
from repoopt.store.local import LocalArtifactStore
from repoopt.store.s3 import S3ArtifactStore


class _FakeS3Client:
    """In-memory mock of boto3 S3 client for testing S3ArtifactStore."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"Metadata": obj["Metadata"], "ContentLength": len(obj["Body"])}

    def put_object(
        self, Bucket: str, Key: str, Body: bytes, ContentType: str = "application/octet-stream", Metadata: dict[str, str] | None = None
    ) -> dict[str, Any]:
        self.objects[(Bucket, Key)] = {
            "Body": Body,
            "ContentType": ContentType,
            "Metadata": dict(Metadata or {}),
        }
        return {"ETag": sha256_bytes(Body)}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        obj = self.objects.get((Bucket, Key))
        if obj is None:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Missing"}}, "GetObject")

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(obj["Body"])}

    def get_paginator(self, op_name: str) -> Any:
        assert op_name == "list_objects_v2"
        parent = self

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
                contents = [
                    {"Key": k}
                    for (b, k) in sorted(parent.objects)
                    if b == Bucket and k.startswith(Prefix)
                ]
                return [{"Contents": contents}]

        return _Paginator()


def test_local_and_s3_stores_layout_and_immutability(tmp_path) -> None:
    local = LocalArtifactStore(tmp_path / "store")
    s3 = S3ArtifactStore("repoopt", prefix="prefix", client=_FakeS3Client())

    for store in (local, s3):
        key = StoragePaths.raw_provider_event("exp1", "py-lj", "att1", 1)
        r1 = store.put_json(key, {"hello": "world"}, immutable=True)
        r2 = store.put_json(key, {"hello": "world"}, immutable=True)
        assert r1.sha256 == r2.sha256
        assert r2.skipped_existing is True
        assert store.get_json(key) == {"hello": "world"}

        with pytest.raises(FileExistsError):
            store.put_json(key, {"hello": "tampered"}, immutable=True)

        assert store.list_prefix("experiments/exp1/tasks/py-lj/attempts/att1") == [key]


def _make_mock_upstream_app(delay_sec: float = 0.05) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Any:
        body = await request.json()
        await asyncio.sleep(delay_sec)
        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": "chatcmpl-nonstream",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Checking code",
                                "reasoning_content": "Need to inspect kernel first",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                }
            )

        async def gen():
            chunks = [
                {
                    "id": "chatcmpl-stream-1",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning": "Profile "}, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl-stream-1",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning": "loop.",
                                "content": "Running read",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_stream_1",
                                        "type": "function",
                                        "function": {"name": "read", "arguments": '{"path": '},
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-stream-1",
                    "model": body.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"src/core.py"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                },
            ]
            abort_early = body.get("messages", [{}])[-1].get("content") == "abort-me"
            for idx, c in enumerate(chunks):
                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
                await asyncio.sleep(0.01)
                if abort_early and idx == 0:
                    return  # Simulate abrupt upstream stream termination without [DONE]
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def test_capture_gateway_streaming_concurrency_and_periodic_flush(tmp_path) -> None:
    async def _run() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        upstream_app = _make_mock_upstream_app(delay_sec=0.06)
        transport = httpx.ASGITransport(app=upstream_app)

        async with httpx.AsyncClient(
            transport=transport, base_url="http://mock-upstream", trust_env=False
        ) as upstream_client:
            gateway = CaptureGateway(
                store,
                upstream_base_url="http://mock-upstream/v1",
                upstream_api_key="sk-real-secret-never-in-container",
                max_concurrency=2,
                flush_interval_sec=0.15,
                http_client=upstream_client,
            )
            await gateway.start()
            try:
                token = format_attempt_token("exp0", "py-lj-forces", "att-001")
                gw_transport = httpx.ASGITransport(app=gateway.app)
                async with httpx.AsyncClient(
                    transport=gw_transport, base_url="http://local-gateway", timeout=10.0, trust_env=False
                ) as client:
                    # 1. Reject missing token
                    unauth = await client.post(
                        "/v1/chat/completions", json={"model": "z-ai/glm-5.3", "messages": []}
                    )
                    assert unauth.status_code == 401

                    # 2. Enforce concurrency=2 across 4 simultaneous calls + deferred periodic flush
                    async def one_call(idx: int) -> httpx.Response:
                        return await client.post(
                            "/v1/chat/completions",
                            headers={"Authorization": f"Bearer {token}", "X-RepoOpt-Defer-Flush": "1"},
                            json={"model": "z-ai/glm-5.3", "messages": [{"role": "user", "content": f"hi {idx}"}]},
                        )

                    resps = await asyncio.gather(*(one_call(i) for i in range(4)))
                    assert all(r.status_code == 200 for r in resps)
                    assert gateway.peak_inflight <= 2

                    # Wait briefly for periodic <=15s (0.15s here) flush ticker to persist segments
                    await asyncio.sleep(0.35)
                    seg_keys = store.list_prefix("experiments/exp0/tasks/py-lj-forces/attempts/att-001/events/")
                    assert len(seg_keys) >= 1

                    # 3. Streaming SSE reassembly (reasoning + chunked tool_calls)
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {token}"},
                        json={
                            "model": "z-ai/glm-5.3",
                            "stream": True,
                            "messages": [{"role": "user", "content": "optimize"}],
                        },
                    ) as sresp:
                        assert sresp.status_code == 200
                        raw_sse = (await sresp.aread()).decode("utf-8")
                        assert "data: [DONE]" in raw_sse

                    # 4. Mid-stream truncated SSE still records partial turn + flushes events
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {token}"},
                        json={
                            "model": "z-ai/glm-5.3",
                            "stream": True,
                            "messages": [{"role": "user", "content": "abort-me"}],
                        },
                    ) as sresp_abort:
                        assert sresp_abort.status_code == 200
                        await sresp_abort.aread()
            finally:
                await gateway.stop()

            # Check turn 5 raw provider JSON reassembly
            raw5 = store.get_json(
                StoragePaths.raw_provider_event("exp0", "py-lj-forces", "att-001", 5)
            )
            assert raw5["streamed"] is True
            assert raw5["completed"] is True
            msg = raw5["response"]["choices"][0]["message"]
            assert msg["reasoning_content"] == "Profile loop."
            assert msg["content"] == "Running read"
            assert msg["tool_calls"][0]["function"]["name"] == "read"
            assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"path": "src/core.py"}

            # Check turn 6 partial capture survived mid-stream interruption
            raw6 = store.get_json(
                StoragePaths.raw_provider_event("exp0", "py-lj-forces", "att-001", 6)
            )
            assert raw6["streamed"] is True
            assert raw6["completed"] is False
            assert raw6["response"]["choices"][0]["message"]["reasoning_content"] == "Profile "

    asyncio.run(_run())
