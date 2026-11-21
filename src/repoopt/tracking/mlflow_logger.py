"""Log RepoOpt rollouts (metrics, artifacts, and full multi-turn traces) to MLflow / Databricks."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import AssessmentSource, SpanStatus, SpanStatusCode, SpanType

from repoopt.config import TrackingConfig
from repoopt.store.base import ArtifactStore, StoragePaths

_MAX_TRACE_STR_CHARS = 16_000
_MAX_PARAM_CHARS = 500
_MIN_SPAN_NS = 1_000_000  # 1 ms minimum span width for UI waterfall clarity


def _truncate_str(value: Any, max_chars: int = _MAX_TRACE_STR_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    suffix = "...[truncated; full payload in S3]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix


def _truncate_param(value: Any) -> str:
    text = str(value)
    if len(text) <= _MAX_PARAM_CHARS:
        return text
    return text[: _MAX_PARAM_CHARS - 3] + "..."


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    return None


def _parse_iso_ns(ts: Any) -> int | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None


def _safe_get_json(store: ArtifactStore, key: str) -> Any | None:
    if not store.exists(key):
        return None
    return store.get_json(key)


def _safe_get_bytes(store: ArtifactStore, key: str) -> bytes | None:
    if not store.exists(key):
        return None
    return store.get_bytes(key)


def discover_store_attempts(store: ArtifactStore, experiment_id: str) -> list[tuple[str, str]]:
    """Return sorted (task_id, attempt_id) pairs found under an experiment in ArtifactStore."""
    prefix = f"experiments/{experiment_id}/tasks/"
    keys = store.list_prefix(prefix)
    found: set[tuple[str, str]] = set()
    for key in keys:
        parts = key.split("/")
        # experiments/<exp>/tasks/<task>/attempts/<attempt>/<file>
        if len(parts) >= 7 and parts[2] == "tasks" and parts[4] == "attempts" and parts[6] in {
            "trajectory.json",
            "evaluation.json",
            "completion.json",
        }:
            found.add((parts[3], parts[5]))
    return sorted(found)


def _extract_step_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _bound_raw_messages(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bounded: list[dict[str, Any]] = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        item: dict[str, Any] = {"role": str(msg.get("role", "user"))}
        if "content" in msg and msg["content"] is not None:
            item["content"] = _truncate_str(msg["content"])
        if msg.get("name"):
            item["name"] = str(msg["name"])
        if msg.get("tool_call_id"):
            item["tool_call_id"] = str(msg["tool_call_id"])
        if msg.get("tool_calls"):
            item["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": _truncate_str((tc.get("function") or {}).get("arguments") or "", 4000),
                }
                for tc in (msg.get("tool_calls") or [])
                if isinstance(tc, dict)
            ]
        bounded.append(item)
    return bounded


def _extract_token_counts(
    step: dict[str, Any] | None,
    raw_ev: dict[str, Any] | None,
) -> tuple[int, int, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    raw_usage = ((raw_ev or {}).get("response") or {}).get("usage")
    if isinstance(raw_usage, dict):
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        total_tokens = int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return prompt_tokens, completion_tokens, total_tokens

    metrics = (step or {}).get("metrics")
    if isinstance(metrics, dict):
        prompt_tokens = int(metrics.get("prompt_tokens") or 0)
        completion_tokens = int(metrics.get("completion_tokens") or 0)
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def _extract_raw_reasoning(raw_msg: dict[str, Any] | None) -> str:
    if not isinstance(raw_msg, dict):
        return ""
    for key in ("reasoning_content", "reasoning"):
        val = raw_msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _build_run_metadata(
    *,
    experiment_id: str,
    task_id: str,
    attempt_id: str,
    attempt_key: str,
    attempt_uri: str,
    traj_data: dict[str, Any] | None,
    eval_data: dict[str, Any],
    comp_data: dict[str, Any] | None,
    manifest_data: dict[str, Any] | None,
    raw_events: list[dict[str, Any]],
    patch_bytes: bytes | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    agent_info = (traj_data or {}).get("agent") or {}
    manifest_models = (manifest_data or {}).get("models") or {}
    manifest_exec = (manifest_data or {}).get("execution") or {}

    agent_name = str(agent_info.get("name") or manifest_exec.get("agent_name") or "pi")
    agent_version = str(agent_info.get("version") or "unknown")
    teacher_model = str(
        agent_info.get("model_name")
        or manifest_models.get("teacher_model")
        or ((raw_events[0].get("request") or {}).get("model") if raw_events else None)
        or "unknown"
    )
    environment_backend = str(manifest_exec.get("environment_backend") or "docker")
    schema_version = str((traj_data or {}).get("schema_version") or "missing")
    label = str(eval_data.get("label") or "INCONCLUSIVE")
    correct = bool(eval_data.get("correct", False))
    is_noop = bool(eval_data.get("is_noop_patch", patch_bytes is None or len(patch_bytes) == 0))

    steps = list((traj_data or {}).get("steps") or [])
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    tool_call_count = sum(len(s.get("tool_calls") or []) for s in agent_steps)

    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_latency_ms = 0.0

    if agent_steps:
        raw_idx = 0
        for step in agent_steps:
            has_llm_call = step.get("llm_call_count", 1) != 0
            raw_ev = raw_events[raw_idx] if (has_llm_call and raw_idx < len(raw_events)) else None
            if has_llm_call and raw_idx < len(raw_events):
                raw_idx += 1
            p_tok, c_tok, t_tok = _extract_token_counts(step, raw_ev)
            total_prompt += p_tok
            total_completion += c_tok
            total_tokens += t_tok
            if raw_ev is not None:
                lat = _finite_number(raw_ev.get("latency_ms"))
                if lat is not None:
                    total_latency_ms += lat
    else:
        for raw_ev in raw_events:
            p_tok, c_tok, t_tok = _extract_token_counts(None, raw_ev)
            total_prompt += p_tok
            total_completion += c_tok
            total_tokens += t_tok
            lat = _finite_number(raw_ev.get("latency_ms"))
            if lat is not None:
                total_latency_ms += lat

    llm_turns = (
        len([s for s in agent_steps if s.get("llm_call_count", 1) != 0])
        if agent_steps
        else len(raw_events)
    )

    tags = {
        "repoopt.attempt_key": attempt_key,
        "repoopt.experiment_id": experiment_id,
        "repoopt.task_id": task_id,
        "repoopt.attempt_id": attempt_id,
        "repoopt.outcome_label": label,
        "repoopt.correct": str(correct).lower(),
        "repoopt.is_noop_patch": str(is_noop).lower(),
        "repoopt.attempt_uri": attempt_uri,
        "repoopt.agent_name": agent_name,
        "repoopt.teacher_model": teacher_model,
    }

    params = {
        "experiment_id": _truncate_param(experiment_id),
        "task_id": _truncate_param(task_id),
        "attempt_id": _truncate_param(attempt_id),
        "agent_name": _truncate_param(agent_name),
        "agent_version": _truncate_param(agent_version),
        "teacher_model": _truncate_param(teacher_model),
        "storage_attempt_uri": _truncate_param(attempt_uri),
        "schema_version": _truncate_param(schema_version),
        "environment_backend": _truncate_param(environment_backend),
    }

    metrics: dict[str, float] = {}
    candidate_metrics: dict[str, Any] = {
        "reward": eval_data.get("reward", 0.0),
        "speedup": eval_data.get("speedup"),
        "ci95_low": eval_data.get("ci95_low"),
        "ci95_high": eval_data.get("ci95_high"),
        "median_baseline_ms": eval_data.get("median_baseline_ms"),
        "median_candidate_ms": eval_data.get("median_candidate_ms"),
        "baseline_cv": eval_data.get("baseline_cv"),
        "candidate_cv": eval_data.get("candidate_cv"),
        "max_abs_err": eval_data.get("max_abs_err"),
        "max_rel_err": eval_data.get("max_rel_err"),
        "patch_bytes": len(patch_bytes) if patch_bytes is not None else 0,
        "llm_turns": llm_turns,
        "total_steps": len(steps),
        "tool_call_count": tool_call_count,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_latency_ms": round(total_latency_ms, 3),
    }
    for k, v in candidate_metrics.items():
        num = _finite_number(v)
        if num is not None:
            metrics[k] = num

    return tags, params, metrics


def _log_structured_trace(
    *,
    client: MlflowClient,
    experiment_id_mlflow: str,
    run_id: str,
    task_id: str,
    attempt_id: str,
    attempt_key: str,
    attempt_uri: str,
    tags: dict[str, str],
    metrics: dict[str, float],
    traj_data: dict[str, Any] | None,
    eval_data: dict[str, Any],
    comp_data: dict[str, Any] | None,
    raw_events: list[dict[str, Any]],
    patch_bytes: bytes | None,
) -> tuple[str, list[str]]:
    """Build and persist the full hierarchical GenAI trace linked to run_id."""
    warnings: list[str] = []
    steps = list((traj_data or {}).get("steps") or [])

    # Extract top-level system prompt and initial user instruction for the root span.
    system_prompt = ""
    first_user_instruction = ""
    if steps:
        for s in steps:
            if s.get("source") == "system" and not system_prompt:
                system_prompt = _extract_step_text(s.get("message"))
            elif s.get("source") == "user" and not first_user_instruction:
                first_user_instruction = _extract_step_text(s.get("message"))
    if (not system_prompt or not first_user_instruction) and raw_events:
        first_msgs = (raw_events[0].get("request") or {}).get("messages") or []
        for rm in first_msgs:
            if rm.get("role") == "system" and not system_prompt:
                system_prompt = _extract_step_text(rm.get("content"))
            elif rm.get("role") == "user" and not first_user_instruction:
                first_user_instruction = _extract_step_text(rm.get("content"))

    now_ns = time.time_ns()
    candidate_starts = [
        ns
        for ns in (
            _parse_iso_ns((comp_data or {}).get("started_at")),
            _parse_iso_ns(steps[0].get("timestamp")) if steps else None,
            _parse_iso_ns(raw_events[0].get("timestamp")) if raw_events else None,
        )
        if ns is not None
    ]
    root_start_ns = min(candidate_starts) if candidate_starts else now_ns
    cursor_ns = root_start_ns

    total_p = int(metrics.get("total_prompt_tokens", 0))
    total_c = int(metrics.get("total_completion_tokens", 0))
    total_t = int(metrics.get("total_tokens", total_p + total_c))

    root_span = client.start_trace(
        name=f"repoopt.{task_id}",
        span_type=SpanType.AGENT,
        inputs={
            "task_id": task_id,
            "attempt_id": attempt_id,
            "system_prompt": _truncate_str(system_prompt),
            "instruction": _truncate_str(first_user_instruction),
        },
        attributes={
            "repoopt.agent_name": tags["repoopt.agent_name"],
            "repoopt.teacher_model": tags["repoopt.teacher_model"],
            "repoopt.attempt_uri": attempt_uri,
            "mlflow.chat.tokenUsage": {
                "input_tokens": total_p,
                "output_tokens": total_c,
                "total_tokens": total_t,
            },
        },
        tags={
            "repoopt.attempt_key": attempt_key,
            "repoopt.outcome_label": tags["repoopt.outcome_label"],
            "repoopt.attempt_uri": attempt_uri,
        },
        experiment_id=experiment_id_mlflow,
        start_time_ns=root_start_ns,
        run_id=run_id,
    )
    trace_id = root_span.trace_id
    root_span_id = root_span.span_id

    final_assistant_text = ""
    raw_idx = 0
    prev_raw_msg_count = 0
    pending_context_msgs: list[dict[str, Any]] = []

    if steps:
        for step in steps:
            source = step.get("source")
            if source in {"system", "user"}:
                pending_context_msgs.append(
                    {"role": str(source), "content": _truncate_str(_extract_step_text(step.get("message")))}
                )
                continue

            if source != "agent":
                continue

            has_llm_call = step.get("llm_call_count", 1) != 0
            raw_ev = raw_events[raw_idx] if (has_llm_call and raw_idx < len(raw_events)) else None
            if has_llm_call and raw_idx < len(raw_events):
                raw_idx += 1

            raw_req = (raw_ev or {}).get("request") or {}
            raw_resp = (raw_ev or {}).get("response") or {}
            raw_choice = (raw_resp.get("choices") or [{}])[0]
            raw_msg = raw_choice.get("message") if isinstance(raw_choice, dict) else None

            step_model = str(step.get("model_name") or raw_req.get("model") or tags["repoopt.teacher_model"])
            step_start_ns = max(
                cursor_ns,
                _parse_iso_ns(step.get("timestamp"))
                or _parse_iso_ns((raw_ev or {}).get("timestamp"))
                or cursor_ns,
            )
            lat_ms = _finite_number((raw_ev or {}).get("latency_ms"))
            duration_ns = max(_MIN_SPAN_NS, int(lat_ms * 1_000_000)) if lat_ms is not None else _MIN_SPAN_NS
            llm_end_ns = step_start_ns + duration_ns

            # Build delta messages for this model turn to stay well under trace size caps.
            if raw_req and isinstance(raw_req.get("messages"), list):
                req_msgs = raw_req["messages"]
                if prev_raw_msg_count > len(req_msgs):
                    # Context compaction shrank/rewrote the prefix; capture current prefix once.
                    delta_msgs = _bound_raw_messages(req_msgs)
                else:
                    delta_msgs = _bound_raw_messages(req_msgs[prev_raw_msg_count:])
                prev_raw_msg_count = len(req_msgs)
            else:
                delta_msgs = list(pending_context_msgs)
            pending_context_msgs = []

            reasoning_text = str(step.get("reasoning_content") or "") or _extract_raw_reasoning(raw_msg)
            assistant_text = _extract_step_text(step.get("message"))
            if assistant_text.strip():
                final_assistant_text = assistant_text

            step_tcs = list(step.get("tool_calls") or [])
            tc_summary = [
                {
                    "id": tc.get("tool_call_id"),
                    "name": tc.get("function_name"),
                    "arguments": tc.get("arguments"),
                }
                for tc in step_tcs
            ]

            p_tok, c_tok, t_tok = _extract_token_counts(step, raw_ev)
            llm_attrs: dict[str, Any] = {
                "mlflow.chat.model": step_model,
                "mlflow.chat.tokenUsage": {
                    "input_tokens": p_tok,
                    "output_tokens": c_tok,
                    "total_tokens": t_tok,
                },
                "repoopt.step_id": step.get("step_id"),
            }
            if lat_ms is not None:
                llm_attrs["repoopt.latency_ms"] = round(lat_ms, 3)

            llm_inputs: dict[str, Any] = {"delta_messages": delta_msgs}
            for skey in ("model", "temperature", "max_tokens", "reasoning_effort"):
                if skey in raw_req and raw_req[skey] is not None:
                    llm_inputs[skey] = raw_req[skey]
                elif skey == "reasoning_effort" and step.get("reasoning_effort") is not None:
                    llm_inputs[skey] = step["reasoning_effort"]

            llm_outputs: dict[str, Any] = {
                "reasoning": _truncate_str(reasoning_text),
                "message": _truncate_str(assistant_text),
                "tool_calls": tc_summary,
                "finish_reason": (raw_choice or {}).get("finish_reason") if isinstance(raw_choice, dict) else None,
            }

            step_label = step.get("step_id")
            llm_span = client.start_span(
                name=f"turn_{step_label}:{step_model}",
                trace_id=trace_id,
                parent_id=root_span_id,
                span_type=SpanType.CHAT_MODEL,
                inputs=llm_inputs,
                attributes=llm_attrs,
                start_time_ns=step_start_ns,
            )
            llm_status: SpanStatus | str = "OK"
            if raw_ev is not None and int(raw_ev.get("status_code", 200)) >= 400:
                llm_status = SpanStatus(SpanStatusCode.ERROR, f"HTTP {raw_ev.get('status_code')}")
            client.end_span(
                trace_id=trace_id,
                span_id=llm_span.span_id,
                outputs=llm_outputs,
                status=llm_status,
                end_time_ns=llm_end_ns,
            )
            cursor_ns = llm_end_ns

            # Pair tool calls with their ObservationResult(s) in chronological order.
            obs_results = list((step.get("observation") or {}).get("results") or [])
            obs_by_call_id = {
                res.get("source_call_id"): res
                for res in obs_results
                if res.get("source_call_id") is not None
            }
            for tc_idx, tc in enumerate(step_tcs):
                tc_id = tc.get("tool_call_id")
                fn_name = str(tc.get("function_name") or "tool")
                matched_res = obs_by_call_id.get(tc_id)
                if matched_res is None and tc_idx < len(obs_results):
                    matched_res = obs_results[tc_idx]

                obs_text = _extract_step_text((matched_res or {}).get("content"))
                res_extra = (matched_res or {}).get("extra") or {}
                is_err = bool(res_extra.get("is_error", False))

                t_start_ns = cursor_ns
                t_end_ns = t_start_ns + _MIN_SPAN_NS
                tool_span = client.start_span(
                    name=f"tool:{fn_name}",
                    trace_id=trace_id,
                    parent_id=root_span_id,
                    span_type=SpanType.TOOL,
                    inputs=tc.get("arguments") or {},
                    attributes={"tool_call_id": tc_id or ""},
                    start_time_ns=t_start_ns,
                )
                tool_status: SpanStatus | str = (
                    SpanStatus(SpanStatusCode.ERROR, _truncate_param(obs_text or "Tool execution failed"))
                    if is_err
                    else "OK"
                )
                client.end_span(
                    trace_id=trace_id,
                    span_id=tool_span.span_id,
                    outputs={
                        "content": _truncate_str(obs_text),
                        "source_call_id": tc_id,
                        "is_error": is_err,
                    },
                    status=tool_status,
                    end_time_ns=t_end_ns,
                )
                cursor_ns = t_end_ns
    else:
        # Fallback when Harbor terminated before emitting trajectory.json: reconstruct turns from raw events.
        for ev in raw_events:
            req = ev.get("request") or {}
            resp = ev.get("response") or {}
            choice = (resp.get("choices") or [{}])[0]
            msg = (choice.get("message") or {}) if isinstance(choice, dict) else {}
            req_msgs = req.get("messages") or []
            delta_msgs = _bound_raw_messages(req_msgs[prev_raw_msg_count:]) if isinstance(req_msgs, list) else []
            if isinstance(req_msgs, list):
                prev_raw_msg_count = len(req_msgs)

            turn_idx = ev.get("turn_index", 0)
            turn_model = str(req.get("model") or tags["repoopt.teacher_model"])
            t_start = max(cursor_ns, _parse_iso_ns(ev.get("timestamp")) or cursor_ns)
            lat_ms = _finite_number(ev.get("latency_ms"))
            t_dur = max(_MIN_SPAN_NS, int(lat_ms * 1_000_000)) if lat_ms is not None else _MIN_SPAN_NS
            t_end = t_start + t_dur
            p_tok, c_tok, t_tok = _extract_token_counts(None, ev)

            content_str = str(msg.get("content") or "")
            if content_str.strip():
                final_assistant_text = content_str

            span = client.start_span(
                name=f"turn_{turn_idx}:{turn_model}",
                trace_id=trace_id,
                parent_id=root_span_id,
                span_type=SpanType.CHAT_MODEL,
                inputs={"delta_messages": delta_msgs},
                attributes={
                    "mlflow.chat.model": turn_model,
                    "mlflow.chat.tokenUsage": {
                        "input_tokens": p_tok,
                        "output_tokens": c_tok,
                        "total_tokens": t_tok,
                    },
                },
                start_time_ns=t_start,
            )
            status_val: SpanStatus | str = (
                SpanStatus(SpanStatusCode.ERROR, f"HTTP {ev.get('status_code')}")
                if int(ev.get("status_code", 200)) >= 400
                else "OK"
            )
            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                outputs={
                    "reasoning": _truncate_str(_extract_raw_reasoning(msg)),
                    "message": _truncate_str(content_str),
                    "tool_calls": msg.get("tool_calls") or [],
                    "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                },
                status=status_val,
                end_time_ns=t_end,
            )
            cursor_ns = t_end

    # Independent verifier child span
    verifier_span_type = getattr(SpanType, "EVALUATOR", SpanType.CHAIN)
    v_start_ns = cursor_ns
    v_end_ns = v_start_ns + _MIN_SPAN_NS
    verifier_span = client.start_span(
        name="independent_verifier",
        trace_id=trace_id,
        parent_id=root_span_id,
        span_type=verifier_span_type,
        inputs={
            "task_id": task_id,
            "has_candidate_patch": bool(patch_bytes),
            "patch_bytes": len(patch_bytes or b""),
        },
        start_time_ns=v_start_ns,
    )
    label = tags["repoopt.outcome_label"]
    v_status: SpanStatus | str = (
        SpanStatus(SpanStatusCode.ERROR, _truncate_param(eval_data.get("reason") or label))
        if label in {"INCORRECT", "TIMEOUT", "ENVIRONMENT_FAILURE"}
        else "OK"
    )
    client.end_span(
        trace_id=trace_id,
        span_id=verifier_span.span_id,
        outputs=eval_data,
        status=v_status,
        end_time_ns=v_end_ns,
    )
    cursor_ns = v_end_ns

    finished_ns = _parse_iso_ns((comp_data or {}).get("finished_at"))
    root_end_ns = max(cursor_ns + _MIN_SPAN_NS, finished_ns or 0)
    trial_error = (comp_data or {}).get("trial_error")
    root_status: SpanStatus | str = (
        SpanStatus(SpanStatusCode.ERROR, _truncate_param(trial_error or label))
        if trial_error or label in {"TIMEOUT", "ENVIRONMENT_FAILURE"}
        else "OK"
    )
    client.end_trace(
        trace_id=trace_id,
        outputs={
            "final_message": _truncate_str(final_assistant_text),
            "label": label,
            "reward": eval_data.get("reward", 0.0),
            "speedup": eval_data.get("speedup"),
            "correct": bool(eval_data.get("correct", False)),
            "attempt_uri": attempt_uri,
        },
        status=root_status,
        end_time_ns=root_end_ns,
    )

    # Flush async trace queue before logging feedback so local/Databricks backends see the trace.
    mlflow.flush_trace_async_logging()

    rationale = (
        f"label={label}, speedup={eval_data.get('speedup')}, "
        f"ci95=[{eval_data.get('ci95_low')}, {eval_data.get('ci95_high')}], "
        f"max_abs_err={eval_data.get('max_abs_err')}"
    )
    fb_source = AssessmentSource(source_type="CODE", source_id="repoopt_verifier")
    feedback_items: list[tuple[str, Any]] = [
        ("outcome_label", label),
        ("reward", float(metrics.get("reward", 0.0))),
        ("verified_speedup", label == "VERIFIED_SPEEDUP"),
    ]
    for fb_name, fb_val in feedback_items:
        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=fb_name,
                value=fb_val,
                source=fb_source,
                rationale=rationale,
            )
        except Exception as exc:
            warnings.append(f"log_feedback({fb_name}) skipped: {type(exc).__name__}: {exc}")

    return trace_id, warnings


def log_attempt_to_mlflow(
    store: ArtifactStore,
    experiment_id: str,
    task_id: str,
    attempt_id: str,
    tracking_config: TrackingConfig,
    *,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Log one archived RepoOpt attempt (Run + Artifacts + GenAI Trace) to MLflow / Databricks."""
    traj_key = StoragePaths.attempt_trajectory(experiment_id, task_id, attempt_id)
    eval_key = StoragePaths.attempt_evaluation(experiment_id, task_id, attempt_id)
    comp_key = StoragePaths.attempt_completion(experiment_id, task_id, attempt_id)
    patch_key = StoragePaths.attempt_patch(experiment_id, task_id, attempt_id)
    manifest_key = StoragePaths.experiment_manifest(experiment_id)
    attempt_prefix = StoragePaths.attempt_prefix(experiment_id, task_id, attempt_id)

    traj_data = _safe_get_json(store, traj_key)
    eval_data = _safe_get_json(store, eval_key)
    comp_data = _safe_get_json(store, comp_key)
    if traj_data is None and eval_data is None and comp_data is None:
        raise FileNotFoundError(
            f"No attempt artifacts found for {experiment_id}/{task_id}/{attempt_id} in store"
        )

    if eval_data is None:
        eval_data = (comp_data or {}).get("evaluation") or {
            "label": "INCONCLUSIVE",
            "reward": 0.0,
            "correct": False,
            "is_noop_patch": True,
        }

    manifest_data = _safe_get_json(store, manifest_key)
    patch_bytes = _safe_get_bytes(store, patch_key)
    raw_keys = sorted(k for k in store.list_prefix(f"{attempt_prefix}/raw-provider-events/") if k.endswith(".json"))
    raw_events = [store.get_json(k) for k in raw_keys]

    uri = tracking_config.tracking_uri.strip() or "databricks"
    if uri.startswith("file:") or uri.startswith("/"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(experiment_name=tracking_config.experiment_name)
    client = mlflow.MlflowClient(tracking_uri=uri)

    attempt_key = f"{experiment_id}/{task_id}/{attempt_id}"
    attempt_uri = store.uri_for(attempt_prefix)

    if skip_existing:
        safe_attempt_key = attempt_key.replace("'", "\\'")
        existing_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`repoopt.attempt_key` = '{safe_attempt_key}'",
            max_results=1,
        )
        if existing_runs:
            return {
                "status": "skipped_existing",
                "experiment_id": exp.experiment_id,
                "run_id": existing_runs[0].info.run_id,
                "attempt_key": attempt_key,
            }

    tags, params, metrics = _build_run_metadata(
        experiment_id=experiment_id,
        task_id=task_id,
        attempt_id=attempt_id,
        attempt_key=attempt_key,
        attempt_uri=attempt_uri,
        traj_data=traj_data,
        eval_data=eval_data,
        comp_data=comp_data,
        manifest_data=manifest_data,
        raw_events=raw_events,
        patch_bytes=patch_bytes,
    )

    warnings: list[str] = []
    with mlflow.start_run(
        experiment_id=exp.experiment_id,
        run_name=f"{task_id}__{attempt_id}",
        tags=tags,
    ) as run:
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        if tracking_config.log_artifacts:
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    wrote_any = False
                    if traj_data is not None:
                        (tmp_path / "trajectory.json").write_text(
                            json.dumps(traj_data, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        wrote_any = True
                    (tmp_path / "evaluation.json").write_text(
                        json.dumps(eval_data, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    wrote_any = True
                    if comp_data is not None:
                        (tmp_path / "completion.json").write_text(
                            json.dumps(comp_data, indent=2, ensure_ascii=False), encoding="utf-8"
                        )
                        wrote_any = True
                    if patch_bytes is not None:
                        (tmp_path / "candidate.patch").write_bytes(patch_bytes)
                        wrote_any = True
                    if wrote_any:
                        mlflow.log_artifacts(str(tmp_path), artifact_path="repoopt")
            except Exception as exc:
                warnings.append(f"log_artifacts skipped: {type(exc).__name__}: {exc}")

        trace_id, trace_warnings = _log_structured_trace(
            client=client,
            experiment_id_mlflow=exp.experiment_id,
            run_id=run.info.run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_key=attempt_key,
            attempt_uri=attempt_uri,
            tags=tags,
            metrics=metrics,
            traj_data=traj_data,
            eval_data=eval_data,
            comp_data=comp_data,
            raw_events=raw_events,
            patch_bytes=patch_bytes,
        )
        warnings.extend(trace_warnings)

    return {
        "status": "logged",
        "experiment_id": exp.experiment_id,
        "run_id": run.info.run_id,
        "trace_id": trace_id,
        "attempt_key": attempt_key,
        "warnings": warnings,
    }


def sync_experiment_to_mlflow(
    store: ArtifactStore,
    experiment_id: str,
    tracking_config: TrackingConfig,
    *,
    attempts: list[tuple[str, str]] | None = None,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Sync all (or selected) attempts for an experiment from ArtifactStore to MLflow."""
    target_attempts = attempts if attempts is not None else discover_store_attempts(store, experiment_id)
    results: list[dict[str, Any]] = []
    logged = 0
    skipped_existing = 0
    failed = 0

    for task_id, attempt_id in target_attempts:
        try:
            res = log_attempt_to_mlflow(
                store,
                experiment_id,
                task_id,
                attempt_id,
                tracking_config,
                skip_existing=skip_existing,
            )
            results.append(res)
            if res.get("status") == "logged":
                logged += 1
            elif res.get("status") == "skipped_existing":
                skipped_existing += 1
        except Exception as exc:
            failed += 1
            results.append(
                {
                    "status": "failed",
                    "attempt_key": f"{experiment_id}/{task_id}/{attempt_id}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "experiment_id": experiment_id,
        "mlflow_experiment_name": tracking_config.experiment_name,
        "tracking_uri": tracking_config.tracking_uri,
        "total_attempts": len(target_attempts),
        "logged": logged,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "results": results,
    }
