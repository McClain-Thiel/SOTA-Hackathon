"""Section 9 Tinker conversion acceptance tests against real Qwen/Qwen3.5-9B tokenizer & qwen3_5 renderer."""

from __future__ import annotations

import json
from typing import Any

import pytest
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

from repoopt.store.base import StoragePaths
from repoopt.store.local import LocalArtifactStore
from repoopt.tinker_adapter.atif_parser import ATIFValidationError, ensure_atif_v1_8
from repoopt.tinker_adapter.converter import (
    convert_trajectory_dict_to_datums,
    export_attempts_to_tinker_dataset,
)


def _base_trajectory(steps: list[dict[str, Any]], *, schema_version: str = "ATIF-v1.8") -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "session_id": "sess-001",
        "agent": {
            "name": "pi",
            "version": "0.85.1",
            "model_name": "openrouter/z-ai/glm-5.3",
        },
        "steps": steps,
    }


def _decode_trained_text(datum: tinker.Datum) -> str:
    tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3.5-9B")
    targets = datum.loss_fn_inputs["target_tokens"].data
    weights = datum.loss_fn_inputs["weights"].data
    trained_ids = [int(tok) for tok, w in zip(targets, weights) if float(w) > 0.0]
    return tokenizer.decode(trained_ids, skip_special_tokens=False)


def _decode_full_targets(datum: tinker.Datum) -> str:
    tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3.5-9B")
    targets = [int(tok) for tok in datum.loss_fn_inputs["target_tokens"].data]
    return tokenizer.decode(targets, skip_special_tokens=False)


def test_1_simple_assistant_response() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Optimize ljforces/core.py"},
            {
                "step_id": 2,
                "source": "agent",
                "model_name": "z-ai/glm-5.3",
                "reasoning_content": "I should explain the plan first.",
                "message": "I will profile ljforces first.",
                "llm_call_count": 1,
            },
        ],
        schema_version="ATIF-v1.7",
    )
    res = convert_trajectory_dict_to_datums(traj, model_name="Qwen/Qwen3.5-9B", renderer_name="qwen3_5")
    assert len(res.datums) == 1
    datum = res.datums[0]
    trained_text = _decode_trained_text(datum)
    assert "I should explain the plan first." in trained_text
    assert "I will profile ljforces first." in trained_text
    assert "Optimize ljforces/core.py" not in trained_text
    assert res.provenance[0].masked_prompt_tokens > 0
    assert res.provenance[0].trained_tokens > 0


def test_2_assistant_tool_call_and_observation() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Benchmark the baseline."},
            {
                "step_id": 2,
                "source": "agent",
                "reasoning_content": "Run repoopt-bench.",
                "message": "",
                "llm_call_count": 1,
                "tool_calls": [
                    {
                        "tool_call_id": "call_1",
                        "function_name": "bash",
                        "arguments": {"command": "repoopt-bench"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_1",
                            "content": '{"correct": true, "speedup": 1.0}',
                            "extra": {"tool_name": "bash", "is_error": False},
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "Baseline took 240ms.",
                "llm_call_count": 1,
            },
        ]
    )
    res = convert_trajectory_dict_to_datums(traj)
    assert len(res.datums) == 2

    # Datum 0 trains only the tool-call assistant turn
    t0 = _decode_trained_text(res.datums[0])
    assert "<function=bash>" in t0
    assert "<parameter=command>" in t0
    assert "repoopt-bench" in t0
    assert "<tool_response>" not in t0

    # Datum 1 sees <tool_response> in masked prompt and trains only the final assistant answer
    full1 = _decode_full_targets(res.datums[1])
    trained1 = _decode_trained_text(res.datums[1])
    assert "<tool_response>" in full1
    assert '"speedup": 1.0' in full1
    assert "Baseline took 240ms." in trained1
    assert "<tool_response>" not in trained1


def test_3_multiple_sequential_tool_calls() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Speed up spatial_predict_loo."},
            {
                "step_id": 2,
                "source": "agent",
                "reasoning_content": "Read R source.",
                "message": "",
                "llm_call_count": 1,
                "tool_calls": [
                    {"tool_call_id": "c1", "function_name": "read", "arguments": {"path": "R/matern_krige.R"}}
                ],
                "observation": {
                    "results": [{"source_call_id": "c1", "content": "spatial_predict_loo <- function(...) {}"}]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "reasoning_content": "Replace loop with C++ Rcpp.",
                "message": "",
                "llm_call_count": 1,
                "tool_calls": [
                    {
                        "tool_call_id": "c2",
                        "function_name": "write",
                        "arguments": {"path": "src/fast.cpp", "content": "// Rcpp"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "c2", "content": "ok"}]},
            },
            {
                "step_id": 4,
                "source": "agent",
                "message": "Done optimizing.",
                "llm_call_count": 1,
            },
        ]
    )
    res = convert_trajectory_dict_to_datums(traj)
    assert len(res.datums) == 3
    assert "R/matern_krige.R" in _decode_trained_text(res.datums[0])
    assert "src/fast.cpp" in _decode_trained_text(res.datums[1])
    assert "Done optimizing." in _decode_trained_text(res.datums[2])
    # Historical <think> from step 2 is stripped in step 4 prompt by Qwen3_5Renderer
    full2 = _decode_full_targets(res.datums[2])
    assert "Read R source." not in full2


def test_4_failed_command_observation_preserved() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Compile extension."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "",
                "llm_call_count": 1,
                "tool_calls": [
                    {"tool_call_id": "c_fail", "function_name": "bash", "arguments": {"command": "R CMD INSTALL ."}}
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "c_fail",
                            "content": "error: expected ';' before '}' token",
                            "extra": {"tool_name": "bash", "is_error": True},
                        }
                    ]
                },
            },
            {
                "step_id": 3,
                "source": "agent",
                "reasoning_content": "Fix missing semicolon.",
                "message": "Fixed syntax error.",
                "llm_call_count": 1,
            },
        ]
    )
    res = convert_trajectory_dict_to_datums(traj)
    assert len(res.datums) == 2
    full1 = _decode_full_targets(res.datums[1])
    trained1 = _decode_trained_text(res.datums[1])
    assert "[ERROR]" in full1
    assert "expected ';' before '}' token" in full1
    assert "[ERROR]" not in trained1
    assert "Fixed syntax error." in trained1


def test_5_long_conversation_with_compaction_splits_windows() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Optimize ljforces."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "Pre-compaction turn 1.",
                "llm_call_count": 1,
            },
            {
                "step_id": 3,
                "source": "agent",
                "message": "Summary: baseline is 300ms, core loop is in ljforces/core.py.",
                "llm_call_count": 1,
                "extra": {"pi_event_type": "compaction", "tokens_before": 120000},
            },
            {
                "step_id": 4,
                "source": "agent",
                "reasoning_content": "Use Cython kernel now.",
                "message": "Applied Cython speedup.",
                "llm_call_count": 1,
            },
        ]
    )
    res = convert_trajectory_dict_to_datums(traj)
    assert len(res.datums) == 2
    assert res.provenance[0].window_index == 0
    assert res.provenance[0].compacted_from_prior is False
    assert res.provenance[1].window_index == 1
    assert res.provenance[1].compacted_from_prior is True

    # Post-compaction Datum sees summary in masked context (weight=0), not pre-compaction turn 1
    full_post = _decode_full_targets(res.datums[1])
    trained_post = _decode_trained_text(res.datums[1])
    assert "[Compacted conversation summary]" in full_post
    assert "Pre-compaction turn 1." not in full_post
    assert "[Compacted conversation summary]" not in trained_post
    assert "Applied Cython speedup." in trained_post


def test_6_unsupported_reasoning_explicitly_omitted_and_flagged() -> None:
    traj = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "Optimize."},
            {
                "step_id": 2,
                "source": "agent",
                "message": "Done.",
                "llm_call_count": 1,
                "extra": {"reasoning_encrypted": "opaque-ciphertext-blob"},
            },
        ]
    )
    raw_events = [
        {
            "turn_index": 1,
            "completed": True,
            "request": {"messages": [{"role": "user", "content": "Optimize."}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done.",
                            "reasoning_details": [{"type": "reasoning.encrypted", "data": "abc"}],
                        }
                    }
                ]
            },
        }
    ]
    res = convert_trajectory_dict_to_datums(traj, raw_provider_events=raw_events)
    assert len(res.datums) == 1
    assert any("encrypted" in w.lower() for w in res.warnings)
    trained = _decode_trained_text(res.datums[0])
    assert "opaque-ciphertext-blob" not in trained
    assert "Done." in trained


def test_7_malformed_atif_rejected_without_silent_coercion() -> None:
    # Bad step_id sequence + dangling source_call_id
    bad_atif = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "hi"},
            {
                "step_id": 99,
                "source": "agent",
                "message": "oops",
                "observation": {"results": [{"source_call_id": "missing_id", "content": "out"}]},
            },
        ]
    )
    with pytest.raises(ATIFValidationError, match="Invalid ATIF trajectory"):
        convert_trajectory_dict_to_datums(bad_atif)

    # Malformed raw tool-call JSON must be rejected, not coerced
    good_atif = _base_trajectory(
        [
            {"step_id": 1, "source": "user", "message": "hi"},
            {"step_id": 2, "source": "agent", "message": "", "llm_call_count": 1},
        ]
    )
    bad_raw = [
        {
            "turn_index": 1,
            "completed": True,
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{bad json"}}
                            ],
                        }
                    }
                ]
            },
        }
    ]
    with pytest.raises(ATIFValidationError, match="Malformed tool_call JSON"):
        convert_trajectory_dict_to_datums(good_atif, raw_provider_events=bad_raw)


def test_8_datum_roundtrip_and_store_export(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    traj = ensure_atif_v1_8(
        _base_trajectory(
            [
                {"step_id": 1, "source": "user", "message": "Accelerate kernel."},
                {
                    "step_id": 2,
                    "source": "agent",
                    "reasoning_content": "Use vectorized NumPy.",
                    "message": "",
                    "llm_call_count": 1,
                    "tool_calls": [
                        {"tool_call_id": "call_b", "function_name": "bash", "arguments": {"command": "pytest"}}
                    ],
                    "observation": {"results": [{"source_call_id": "call_b", "content": "1 passed"}]},
                },
                {
                    "step_id": 3,
                    "source": "agent",
                    "reasoning_content": "All tests green.",
                    "message": "Optimization complete.",
                    "llm_call_count": 1,
                },
            ],
            schema_version="ATIF-v1.7",
        )
    )
    store.put_json(StoragePaths.attempt_trajectory("exp1", "py-lj", "att1"), traj)
    store.put_json(
        StoragePaths.attempt_evaluation("exp1", "py-lj", "att1"),
        {"label": "VERIFIED_SPEEDUP", "reward": 2.5, "correct": True},
    )

    datums, report = export_attempts_to_tinker_dataset(
        store,
        "exp1",
        [("py-lj", "att1")],
        model_name="Qwen/Qwen3.5-9B",
        renderer_name="qwen3_5",
        require_verified_speedup=True,
    )
    assert report.trajectories_processed == 1
    assert report.datums_emitted == 2
    assert len(datums) == 2

    # Verify tinker.Datum types & Qwen3.5 renderer parse_response roundtrip on trained tokens
    tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3.5-9B")
    renderer = get_renderer("qwen3_5", tokenizer)
    for d in datums:
        assert isinstance(d, tinker.Datum)
        assert d.loss_fn_inputs["target_tokens"].dtype == "int64"
        assert d.loss_fn_inputs["weights"].dtype == "float32"
        assert d.model_input.length == len(d.loss_fn_inputs["target_tokens"].data)
        assert sum(d.loss_fn_inputs["weights"].data) == pytest.approx(1.0, abs=1e-4)

    # Check trained slice of Datum 0 parses cleanly back into ThinkingPart + ToolCall
    d0 = datums[0]
    w0 = d0.loss_fn_inputs["weights"].data
    t0 = d0.loss_fn_inputs["target_tokens"].data
    trained_slice = [int(tok) for tok, w in zip(t0, w0) if float(w) > 0.0]
    parsed_msg, termination = renderer.parse_response(trained_slice)
    assert termination.is_clean
    assert parsed_msg.get("tool_calls")
    assert parsed_msg["tool_calls"][0].function.name == "bash"
    assert json.loads(parsed_msg["tool_calls"][0].function.arguments) == {"command": "pytest"}

    assert store.exists(StoragePaths.training_conversion_report("exp1", "Qwen/Qwen3.5-9B"))
    assert store.exists(StoragePaths.training_manifest("exp1", "Qwen/Qwen3.5-9B"))
