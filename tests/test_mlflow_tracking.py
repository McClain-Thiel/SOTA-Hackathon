"""Offline tests for MLflow / Databricks run, artifact, and GenAI trace logging."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient
from mlflow.entities import SpanStatusCode, SpanType

from repoopt.cli import main as cli_main
from repoopt.config import ExperimentConfig, ExperimentManifest, TrackingConfig
from repoopt.store import LocalArtifactStore, StoragePaths
from repoopt.tracking.mlflow_logger import log_attempt_to_mlflow


def _seed_sample_attempt(store: LocalArtifactStore, exp_id: str, task_id: str, attempt_id: str) -> None:
    manifest = ExperimentManifest.create(ExperimentConfig(experiment_id=exp_id))
    store.put_json(StoragePaths.experiment_manifest(exp_id), manifest.model_dump(mode="json"))

    traj = {
        "schema_version": "ATIF-v1.8",
        "session_id": f"{task_id}__{attempt_id}",
        "agent": {
            "name": "pi",
            "version": "0.85.1",
            "model_name": "z-ai/glm-5.3",
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2026-11-21T16:00:00Z",
                "source": "system",
                "message": "You are an autonomous scientific optimization engineer.",
            },
            {
                "step_id": 2,
                "timestamp": "2026-11-21T16:00:01Z",
                "source": "user",
                "message": "Optimize the Lennard-Jones force kernel and verify with repoopt-bench.",
            },
            {
                "step_id": 3,
                "timestamp": "2026-11-21T16:00:02Z",
                "source": "agent",
                "model_name": "z-ai/glm-5.3",
                "reasoning_content": "First inspect baseline speed and edit kernel.",
                "message": "Running benchmark and updating implementation.",
                "llm_call_count": 1,
                "tool_calls": [
                    {
                        "tool_call_id": "call_1",
                        "function_name": "bash",
                        "arguments": {"command": "pytest -q missing_file.py"},
                    },
                    {
                        "tool_call_id": "call_2",
                        "function_name": "edit",
                        "arguments": {
                            "path": "lj.py",
                            "oldText": "slow()",
                            "newText": "fast_vectorized()",
                        },
                    },
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_1",
                            "content": "ERROR: file not found",
                            "extra": {"is_error": True},
                        },
                        {
                            "source_call_id": "call_2",
                            "content": "Updated lj.py successfully.",
                            "extra": {"is_error": False},
                        },
                    ]
                },
            },
            {
                "step_id": 4,
                "timestamp": "2026-11-21T16:00:05Z",
                "source": "agent",
                "model_name": "z-ai/glm-5.3",
                "reasoning_content": "Verify speedup and correctness using repoopt-bench.",
                "message": "Verified 10.93x speedup with full numerical accuracy.",
                "llm_call_count": 1,
                "tool_calls": [
                    {
                        "tool_call_id": "call_3",
                        "function_name": "bash",
                        "arguments": {"command": "repoopt-bench"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "call_3",
                            "content": json.dumps({"correct": True, "speedup": 10.93}),
                            "extra": {"is_error": False},
                        }
                    ]
                },
            },
        ],
    }

    raw_turn_1 = {
        "turn_index": 1,
        "timestamp": "2026-11-21T16:00:02Z",
        "latency_ms": 420.5,
        "status_code": 200,
        "request": {
            "model": "z-ai/glm-5.3",
            "temperature": 0.7,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": "You are an autonomous scientific optimization engineer."},
                {"role": "user", "content": "Optimize the Lennard-Jones force kernel and verify with repoopt-bench."},
            ],
        },
        "response": {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "Running benchmark and updating implementation.",
                        "reasoning_content": "First inspect baseline speed and edit kernel.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
        },
    }
    raw_turn_2 = {
        "turn_index": 2,
        "timestamp": "2026-11-21T16:00:05Z",
        "latency_ms": 310.0,
        "status_code": 200,
        "request": {
            "model": "z-ai/glm-5.3",
            "temperature": 0.7,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": "You are an autonomous scientific optimization engineer."},
                {"role": "user", "content": "Optimize the Lennard-Jones force kernel and verify with repoopt-bench."},
                {"role": "assistant", "content": "Running benchmark and updating implementation."},
                {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "ERROR: file not found"},
                {"role": "tool", "tool_call_id": "call_2", "name": "edit", "content": "Updated lj.py successfully."},
            ],
        },
        "response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Verified 10.93x speedup with full numerical accuracy.",
                        "reasoning_content": "Verify speedup and correctness using repoopt-bench.",
                    },
                }
            ],
            "usage": {"prompt_tokens": 190, "completion_tokens": 35, "total_tokens": 225},
        },
    }

    eval_payload = {
        "label": "VERIFIED_SPEEDUP",
        "reward": 3.45,
        "correct": True,
        "is_noop_patch": False,
        "speedup": 10.93,
        "ci95_low": 10.4,
        "ci95_high": 11.5,
        "median_baseline_ms": 110.0,
        "median_candidate_ms": 10.06,
        "baseline_cv": 0.02,
        "candidate_cv": 0.03,
        "max_abs_err": 1.2e-13,
        "max_rel_err": 1.0e-12,
    }
    comp_payload = {
        "experiment_id": exp_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "started_at": "2026-11-21T16:00:00Z",
        "finished_at": "2026-11-21T16:00:08Z",
        "trial_error": None,
        "trajectory_valid": True,
        "evaluation": eval_payload,
    }

    store.put_json(StoragePaths.attempt_trajectory(exp_id, task_id, attempt_id), traj)
    store.put_json(StoragePaths.raw_provider_event(exp_id, task_id, attempt_id, 1), raw_turn_1, immutable=True)
    store.put_json(StoragePaths.raw_provider_event(exp_id, task_id, attempt_id, 2), raw_turn_2, immutable=True)
    store.put_bytes(
        StoragePaths.attempt_patch(exp_id, task_id, attempt_id),
        b"diff --git a/lj.py b/lj.py\n--- a/lj.py\n+++ b/lj.py\n@@ -1 +1 @@\n-slow()\n+fast_vectorized()\n",
        content_type="text/x-diff",
    )
    store.put_json(StoragePaths.attempt_evaluation(exp_id, task_id, attempt_id), eval_payload)
    store.put_json(StoragePaths.attempt_completion(exp_id, task_id, attempt_id), comp_payload)


def test_log_attempt_to_mlflow_creates_run_artifacts_and_structured_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store = LocalArtifactStore(tmp_path / "artifacts")
    exp_id = "exp-mlflow-01"
    task_id = "py-lj-forces"
    attempt_id = "att-0001"
    _seed_sample_attempt(store, exp_id, task_id, attempt_id)

    tracking_uri = (tmp_path / "mlruns").as_uri()
    cfg = TrackingConfig(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name="repoopt-unit-test",
        log_artifacts=True,
    )

    receipt = log_attempt_to_mlflow(store, exp_id, task_id, attempt_id, cfg)
    assert receipt["status"] == "logged"
    assert receipt["attempt_key"] == f"{exp_id}/{task_id}/{attempt_id}"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(receipt["run_id"])
    assert run.data.tags["repoopt.outcome_label"] == "VERIFIED_SPEEDUP"
    assert run.data.tags["repoopt.attempt_uri"] == store.uri_for(
        StoragePaths.attempt_prefix(exp_id, task_id, attempt_id)
    )
    assert run.data.metrics["reward"] == pytest.approx(3.45)
    assert run.data.metrics["speedup"] == pytest.approx(10.93)
    assert run.data.metrics["ci95_low"] == pytest.approx(10.4)
    assert run.data.metrics["total_prompt_tokens"] == pytest.approx(310.0)
    assert run.data.metrics["total_completion_tokens"] == pytest.approx(80.0)
    assert run.data.metrics["total_tokens"] == pytest.approx(390.0)
    assert run.data.metrics["llm_turns"] == pytest.approx(2.0)
    assert run.data.metrics["tool_call_count"] == pytest.approx(3.0)

    artifact_names = {Path(a.path).name for a in client.list_artifacts(receipt["run_id"], "repoopt")}
    assert {"candidate.patch", "evaluation.json", "trajectory.json", "completion.json"} <= artifact_names

    trace = client.get_trace(receipt["trace_id"])
    found_traces = client.search_traces(locations=[receipt["experiment_id"]], run_id=receipt["run_id"])
    assert len(found_traces) == 1
    assert found_traces[0].info.trace_id == receipt["trace_id"]

    spans = trace.data.spans
    root_spans = [s for s in spans if s.span_type == SpanType.AGENT]
    llm_spans = [s for s in spans if s.span_type == SpanType.CHAT_MODEL]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    verifier_spans = [s for s in spans if s.name == "independent_verifier"]

    assert len(root_spans) == 1
    assert len(llm_spans) == 2
    assert len(tool_spans) == 3
    assert len(verifier_spans) == 1

    # Delta inputs: turn 2 should contain the assistant/tool delta rather than repeating the system message.
    turn2_delta_roles = [m["role"] for m in llm_spans[1].inputs["delta_messages"]]
    assert "system" not in turn2_delta_roles
    assert turn2_delta_roles == ["assistant", "tool", "tool"]
    assert llm_spans[0].outputs["reasoning"] == "First inspect baseline speed and edit kernel."
    assert llm_spans[0].attributes["mlflow.chat.tokenUsage"]["input_tokens"] == 120

    # Tool error propagation onto SpanStatus
    assert tool_spans[0].name == "tool:bash"
    assert tool_spans[0].status.status_code == SpanStatusCode.ERROR
    assert tool_spans[1].name == "tool:edit"
    assert tool_spans[1].status.status_code == SpanStatusCode.OK

    # Automated assessments attached to the trace
    assessment_names = {a.name for a in trace.info.assessments}
    assert {"outcome_label", "reward", "verified_speedup"} <= assessment_names


def test_sync_mlflow_cli_idempotency_and_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    store_root = tmp_path / "artifacts"
    store = LocalArtifactStore(store_root)
    exp_id = "exp-cli-sync"
    _seed_sample_attempt(store, exp_id, "r-spatial-kernel", "att-9999")

    tracking_uri = (tmp_path / "mlruns").as_uri()
    mlflow_exp_name = "repoopt-cli-sync-test"

    base_args = [
        "sync-mlflow",
        "--exp-id",
        exp_id,
        "--store-root",
        str(store_root),
        "--tracking-uri",
        tracking_uri,
        "--experiment-name",
        mlflow_exp_name,
    ]

    # 1. Initial sync logs 1 run + trace
    assert cli_main(base_args) == 0
    first_out = json.loads(capsys.readouterr().out)
    assert first_out["logged"] == 1
    assert first_out["skipped_existing"] == 0
    assert first_out["failed"] == 0

    # 2. Second sync skips existing run idempotently
    assert cli_main(base_args) == 0
    second_out = json.loads(capsys.readouterr().out)
    assert second_out["logged"] == 0
    assert second_out["skipped_existing"] == 1

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(mlflow_exp_name)
    assert exp is not None
    assert len(client.search_runs(experiment_ids=[exp.experiment_id])) == 1

    # 3. --force logs a fresh run
    assert cli_main([*base_args, "--force"]) == 0
    third_out = json.loads(capsys.readouterr().out)
    assert third_out["logged"] == 1
    assert len(client.search_runs(experiment_ids=[exp.experiment_id])) == 2
