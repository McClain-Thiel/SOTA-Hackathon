"""Strict ATIF-v1.8 parser, raw provider merger, and compaction window splitter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from harbor.models.trajectories import Trajectory
from harbor.utils.trajectory_validator import TrajectoryValidator
from tinker_cookbook.renderers.base import (
    ContentPart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolSpec,
)

DEFAULT_PI_TOOL_SPECS: list[ToolSpec] = [
    {
        "name": "read",
        "description": "Read the contents of a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
                "offset": {"type": "integer", "description": "1-indexed line offset."},
                "limit": {"type": "integer", "description": "Maximum number of lines to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write",
        "description": "Write or overwrite a file with new content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit",
        "description": "Replace an exact text snippet inside an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "oldText": {"type": "string", "description": "Exact text to replace."},
                "newText": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "oldText", "newText"],
        },
    },
    {
        "name": "bash",
        "description": "Execute a bash command inside the container environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "integer", "description": "Optional timeout in seconds."},
            },
            "required": ["command"],
        },
    },
]


class ATIFValidationError(ValueError):
    """Raised when an ATIF trajectory or associated raw provider event is malformed."""


@dataclass
class ConversationWindow:
    """One prefix-consistent conversation window ready for Qwen/Tinker rendering."""

    window_index: int
    system_prompt: str
    tools: list[ToolSpec]
    messages: list[Message]
    trainable_assistant_indices: list[int]
    compacted_from_prior: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedTrajectory:
    """Validated ATIF trajectory decomposed into trainable conversation windows."""

    trajectory: Trajectory
    windows: list[ConversationWindow]
    warnings: list[str] = field(default_factory=list)
    used_raw_provider_events: bool = False


def ensure_atif_v1_8(trajectory_data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an ATIF dictionary (upgrading ATIF-v1.7 from Harbor Pi to ATIF-v1.8)."""
    if not isinstance(trajectory_data, dict):
        raise ATIFValidationError("Trajectory payload must be a JSON object")

    upgraded = dict(trajectory_data)
    original_version = upgraded.get("schema_version")
    if original_version == "ATIF-v1.7":
        upgraded["schema_version"] = "ATIF-v1.8"
        extra = dict(upgraded.get("extra") or {})
        extra.setdefault("source_atif_version", "ATIF-v1.7")
        upgraded["extra"] = extra

    validator = TrajectoryValidator()
    if not validator.validate(upgraded, validate_images=False):
        errors = "; ".join(validator.get_errors())
        raise ATIFValidationError(f"Invalid ATIF trajectory: {errors}")

    return upgraded


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(str(getattr(item, "text") or ""))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                parts.append(str(getattr(item, "text") or ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _normalize_tool_specs(
    trajectory: Trajectory,
    raw_events: list[dict[str, Any]] | None,
) -> list[ToolSpec]:
    """Extract ToolSpecs from raw provider requests, ATIF tool_definitions, or Pi defaults."""
    if raw_events:
        for ev in reversed(raw_events):
            req = ev.get("request") or {}
            raw_tools = req.get("tools")
            if isinstance(raw_tools, list) and raw_tools:
                specs: list[ToolSpec] = []
                for item in raw_tools:
                    if not isinstance(item, dict):
                        continue
                    fn = item.get("function") if item.get("type") == "function" else item
                    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                        specs.append(
                            {
                                "name": fn["name"],
                                "description": str(fn.get("description") or ""),
                                "parameters": dict(fn.get("parameters") or {"type": "object", "properties": {}}),
                            }
                        )
                if specs:
                    return specs

    if trajectory.agent.tool_definitions:
        specs = []
        for td in trajectory.agent.tool_definitions:
            if isinstance(td, dict):
                fn = td.get("function") if td.get("type") == "function" else td
                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                    specs.append(
                        {
                            "name": fn["name"],
                            "description": str(fn.get("description") or ""),
                            "parameters": dict(fn.get("parameters") or {"type": "object", "properties": {}}),
                        }
                    )
        if specs:
            return specs

    return list(DEFAULT_PI_TOOL_SPECS)


def _validate_raw_provider_tool_calls(raw_events: list[dict[str, Any]]) -> None:
    """Reject malformed JSON tool-call arguments in raw provider records instead of coercing."""
    for idx, ev in enumerate(raw_events):
        resp = ev.get("response") or {}
        choices = resp.get("choices") or []
        if not isinstance(choices, list) or not choices:
            continue
        msg = (choices[0] or {}).get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ATIFValidationError(f"Raw provider event #{idx + 1} has non-list tool_calls")
        for tc in tool_calls:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name")
            args = fn.get("arguments")
            if not isinstance(name, str) or not name.strip():
                raise ATIFValidationError(f"Raw provider event #{idx + 1} has tool_call missing function.name")
            if isinstance(args, str):
                stripped = args.strip() or "{}"
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ATIFValidationError(
                        f"Malformed tool_call JSON arguments for {name!r} in raw event #{idx + 1}: {exc}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ATIFValidationError(
                        f"Tool_call arguments for {name!r} in raw event #{idx + 1} must decode to a JSON object"
                    )
            elif not isinstance(args, dict):
                raise ATIFValidationError(
                    f"Tool_call arguments for {name!r} in raw event #{idx + 1} must be JSON string or dict"
                )


def _extract_step_reasoning(
    step: Any,
    raw_msg: dict[str, Any] | None,
    warnings: list[str],
) -> str | None:
    """Return text reasoning if supported, or omit + flag unsupported/encrypted reasoning."""
    extra = step.extra or {}
    if extra.get("reasoning_encrypted") or extra.get("encrypted_reasoning"):
        warnings.append(
            f"Step {step.step_id}: omitted encrypted/opaque reasoning payload unsupported by {step.model_name or 'student'}"
        )
        return None

    if raw_msg is not None:
        details = raw_msg.get("reasoning_details")
        if isinstance(details, list):
            for entry in details:
                if isinstance(entry, dict) and entry.get("type") in {
                    "reasoning.encrypted",
                    "encrypted",
                    "redacted_thinking",
                }:
                    warnings.append(
                        f"Step {step.step_id}: omitted unsupported reasoning_details type={entry.get('type')!r}"
                    )
                    return None

        raw_r = raw_msg.get("reasoning_content")
        if raw_r is None:
            raw_r = raw_msg.get("reasoning")
        if raw_r is not None and not isinstance(raw_r, str):
            warnings.append(
                f"Step {step.step_id}: omitted non-string raw reasoning payload ({type(raw_r).__name__})"
            )
            return None
        if isinstance(raw_r, str) and raw_r.strip():
            return raw_r.strip()

    if step.reasoning_content is not None:
        if not isinstance(step.reasoning_content, str):
            warnings.append(
                f"Step {step.step_id}: omitted non-string ATIF reasoning_content ({type(step.reasoning_content).__name__})"
            )
            return None
        if step.reasoning_content.strip():
            return step.reasoning_content.strip()

    return None


def _build_assistant_message(
    *,
    text: str,
    reasoning: str | None,
    tool_calls: list[ToolCall],
) -> Message:
    parts: list[ContentPart] = []
    if reasoning:
        parts.append(ThinkingPart(type="thinking", thinking=reasoning))
    if text:
        parts.append(TextPart(type="text", text=text))
    msg: Message = {"role": "assistant", "content": parts if parts else ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _split_raw_compaction_windows(
    raw_events: list[dict[str, Any]],
    default_system_prompt: str,
    tools: list[ToolSpec],
) -> list[ConversationWindow] | None:
    """Split raw provider OpenAI request/response histories whenever context compaction shrinks/rewrites prefix."""
    completed = [
        ev for ev in raw_events
        if ev.get("completed", True)
        and isinstance((ev.get("request") or {}).get("messages"), list)
        and isinstance(((ev.get("response") or {}).get("choices") or [{}])[0].get("message"), dict)
    ]
    if not completed:
        return None

    def _msg_sig(m: dict[str, Any]) -> tuple[str, str]:
        return (str(m.get("role", "")), str(m.get("content", ""))[:200])

    # Group raw turns into prefix-monotonic segments; compaction resets the prefix.
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    prev_req_sigs: list[tuple[str, str]] | None = None

    for ev in completed:
        req_msgs = ev["request"]["messages"]
        req_sigs = [_msg_sig(m) for m in req_msgs]
        if prev_req_sigs is not None and (
            len(req_sigs) < len(prev_req_sigs) or req_sigs[: len(prev_req_sigs)] != prev_req_sigs
        ):
            groups.append(current_group)
            current_group = [ev]
        else:
            current_group.append(ev)
        prev_req_sigs = req_sigs
    if current_group:
        groups.append(current_group)

    if len(groups) <= 1:
        return None

    windows: list[ConversationWindow] = []
    for win_idx, group in enumerate(groups):
        last_ev = group[-1]
        req_msgs = last_ev["request"]["messages"]
        first_req_len = len(group[0]["request"]["messages"])

        sys_prompt = default_system_prompt
        convo: list[Message] = []
        trainable_indices: list[int] = []

        # Everything present in the first request of the compacted window is context prefix (weight=0),
        # except system prompt which goes into create_conversation_prefix_with_tools.
        for msg_idx, raw_m in enumerate(req_msgs):
            role = str(raw_m.get("role", "user"))
            if role == "system" and msg_idx == 0:
                sys_prompt = str(raw_m.get("content") or sys_prompt)
                continue

            is_prefix_context = msg_idx < first_req_len
            if role == "assistant":
                raw_tcs = raw_m.get("tool_calls") or []
                tcs = [
                    ToolCall(
                        id=str(tc.get("id") or f"call_{msg_idx}_{j}"),
                        function=ToolCall.FunctionBody(
                            name=str((tc.get("function") or {}).get("name") or ""),
                            arguments=json.dumps(
                                json.loads((tc.get("function") or {}).get("arguments") or "{}")
                            )
                            if isinstance((tc.get("function") or {}).get("arguments"), str)
                            else json.dumps((tc.get("function") or {}).get("arguments") or {}),
                        ),
                    )
                    for j, tc in enumerate(raw_tcs)
                ]
                r_text = raw_m.get("reasoning_content") or raw_m.get("reasoning")
                rendered = _build_assistant_message(
                    text=str(raw_m.get("content") or ""),
                    reasoning=r_text if isinstance(r_text, str) and r_text.strip() else None,
                    tool_calls=tcs,
                )
                convo.append(rendered)
                if not is_prefix_context:
                    trainable_indices.append(len(convo) - 1)
            elif role == "tool":
                tool_msg: Message = {
                    "role": "tool",
                    "content": str(raw_m.get("content") or ""),
                }
                if raw_m.get("tool_call_id"):
                    tool_msg["tool_call_id"] = str(raw_m["tool_call_id"])
                if raw_m.get("name"):
                    tool_msg["name"] = str(raw_m["name"])
                convo.append(tool_msg)
            else:
                convo.append({"role": "user", "content": str(raw_m.get("content") or "")})

        # Append final assistant completion from last_ev response
        final_resp_msg = last_ev["response"]["choices"][0]["message"]
        final_tcs = [
            ToolCall(
                id=str(tc.get("id") or f"call_final_{j}"),
                function=ToolCall.FunctionBody(
                    name=str((tc.get("function") or {}).get("name") or ""),
                    arguments=json.dumps(
                        json.loads((tc.get("function") or {}).get("arguments") or "{}")
                    )
                    if isinstance((tc.get("function") or {}).get("arguments"), str)
                    else json.dumps((tc.get("function") or {}).get("arguments") or {}),
                ),
            )
            for j, tc in enumerate(final_resp_msg.get("tool_calls") or [])
        ]
        final_r = final_resp_msg.get("reasoning_content") or final_resp_msg.get("reasoning")
        convo.append(
            _build_assistant_message(
                text=str(final_resp_msg.get("content") or ""),
                reasoning=final_r if isinstance(final_r, str) and final_r.strip() else None,
                tool_calls=final_tcs,
            )
        )
        trainable_indices.append(len(convo) - 1)

        if trainable_indices:
            windows.append(
                ConversationWindow(
                    window_index=win_idx,
                    system_prompt=sys_prompt,
                    tools=tools,
                    messages=convo,
                    trainable_assistant_indices=trainable_indices,
                    compacted_from_prior=(win_idx > 0),
                )
            )
    return windows or None


def parse_atif_trajectory(
    trajectory_data: dict[str, Any],
    *,
    raw_provider_events: list[dict[str, Any]] | None = None,
    default_system_prompt: str = (
        "You are an autonomous scientific software optimization engineer. "
        "Profile, optimize, and verify correctness and speedups using the available tools."
    ),
) -> ParsedTrajectory:
    """Validate an ATIF trajectory, merge raw provider context, and emit prefix-consistent windows."""
    normalized_dict = ensure_atif_v1_8(trajectory_data)
    trajectory = Trajectory.model_validate(normalized_dict)

    if raw_provider_events:
        _validate_raw_provider_tool_calls(raw_provider_events)

    # Extra strict ATIF sanity checks beyond Pydantic model constraints
    for step in trajectory.steps:
        if step.tool_calls:
            for tc in step.tool_calls:
                if not tc.tool_call_id.strip():
                    raise ATIFValidationError(f"Step {step.step_id}: empty tool_call_id")
                if not tc.function_name.strip():
                    raise ATIFValidationError(f"Step {step.step_id}: empty function_name")
                if not isinstance(tc.arguments, dict):
                    raise ATIFValidationError(
                        f"Step {step.step_id}: tool_call {tc.tool_call_id} arguments must be a dict"
                    )

    tools = _normalize_tool_specs(trajectory, raw_provider_events)
    warnings: list[str] = []

    # Check if raw provider events reveal a compaction boundary
    if raw_provider_events:
        raw_sys = default_system_prompt
        first_msgs = (raw_provider_events[0].get("request") or {}).get("messages") or []
        if first_msgs and first_msgs[0].get("role") == "system":
            raw_sys = str(first_msgs[0].get("content") or default_system_prompt)
        raw_windows = _split_raw_compaction_windows(raw_provider_events, raw_sys, tools)
        if raw_windows is not None:
            return ParsedTrajectory(
                trajectory=trajectory,
                windows=raw_windows,
                warnings=warnings,
                used_raw_provider_events=True,
            )

    # Otherwise reconstruct from ATIF steps (splitting on compaction / copied context markers)
    raw_agent_msgs: list[dict[str, Any]] = []
    if raw_provider_events:
        for ev in raw_provider_events:
            if not ev.get("completed", True):
                continue
            choices = (ev.get("response") or {}).get("choices") or []
            if choices and isinstance((choices[0] or {}).get("message"), dict):
                raw_agent_msgs.append(choices[0]["message"])

    windows: list[ConversationWindow] = []
    system_prompt = default_system_prompt
    cur_messages: list[Message] = []
    cur_trainable: list[int] = []
    cur_compacted = False
    raw_idx = 0

    for step in trajectory.steps:
        extra = step.extra or {}
        pi_event_type = extra.get("pi_event_type")

        # Explicit ATIF compaction marker starts a new window with the summary as non-trainable user context
        if pi_event_type in {"compaction", "branch_summary"}:
            if cur_messages and cur_trainable:
                windows.append(
                    ConversationWindow(
                        window_index=len(windows),
                        system_prompt=system_prompt,
                        tools=tools,
                        messages=cur_messages,
                        trainable_assistant_indices=cur_trainable,
                        compacted_from_prior=cur_compacted,
                        warnings=list(warnings),
                    )
                )
            summary_text = _extract_text_content(step.message)
            cur_messages = [
                {
                    "role": "user",
                    "content": f"[Compacted conversation summary]\n{summary_text}",
                }
            ]
            cur_trainable = []
            cur_compacted = True
            continue

        if step.source == "system":
            sys_text = _extract_text_content(step.message)
            if not cur_messages:
                system_prompt = sys_text or system_prompt
            else:
                cur_messages.append({"role": "system", "content": sys_text})
            continue

        if step.source == "user":
            cur_messages.append({"role": "user", "content": _extract_text_content(step.message)})
            continue

        # step.source == "agent"
        raw_msg = raw_agent_msgs[raw_idx] if raw_idx < len(raw_agent_msgs) else None
        if step.llm_call_count != 0:
            raw_idx += 1

        reasoning = _extract_step_reasoning(step, raw_msg, warnings)
        text = _extract_text_content(step.message)

        tinker_tcs: list[ToolCall] = []
        tc_names_by_id: dict[str, str] = {}
        if step.tool_calls:
            for tc in step.tool_calls:
                tc_names_by_id[tc.tool_call_id] = tc.function_name
                tinker_tcs.append(
                    ToolCall(
                        id=tc.tool_call_id,
                        function=ToolCall.FunctionBody(
                            name=tc.function_name,
                            arguments=json.dumps(tc.arguments),
                        ),
                    )
                )

        assistant_msg = _build_assistant_message(
            text=text,
            reasoning=reasoning,
            tool_calls=tinker_tcs,
        )
        cur_messages.append(assistant_msg)
        if step.is_copied_context is not True and step.llm_call_count != 0:
            cur_trainable.append(len(cur_messages) - 1)

        if step.observation and step.observation.results:
            for res in step.observation.results:
                obs_text = _extract_text_content(res.content)
                res_extra = res.extra or {}
                if res_extra.get("is_error") and obs_text and not obs_text.startswith("[ERROR]"):
                    obs_text = f"[ERROR]\n{obs_text}"
                tool_msg: Message = {
                    "role": "tool",
                    "content": obs_text,
                }
                if res.source_call_id is not None:
                    tool_msg["tool_call_id"] = res.source_call_id
                    if res.source_call_id in tc_names_by_id:
                        tool_msg["name"] = tc_names_by_id[res.source_call_id]
                if not tool_msg.get("name") and res_extra.get("tool_name"):
                    tool_msg["name"] = str(res_extra["tool_name"])
                cur_messages.append(tool_msg)

    if cur_messages and cur_trainable:
        windows.append(
            ConversationWindow(
                window_index=len(windows),
                system_prompt=system_prompt,
                tools=tools,
                messages=cur_messages,
                trainable_assistant_indices=cur_trainable,
                compacted_from_prior=cur_compacted,
                warnings=list(warnings),
            )
        )

    if not windows:
        raise ATIFValidationError("Trajectory contains zero trainable assistant turns")

    return ParsedTrajectory(
        trajectory=trajectory,
        windows=windows,
        warnings=warnings,
        used_raw_provider_events=bool(raw_provider_events),
    )
