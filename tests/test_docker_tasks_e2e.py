"""RL bridge / Ray config parity tests plus disk-guarded Docker + Harbor E2E suite."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from repoopt.bench.preflight_gate import verify_task_preflight
from repoopt.config import ExecutionConfig, ExperimentConfig, ModelConfig, StorageConfig
from repoopt.runner.harbor_runner import build_ray_job_spec, run_harbor_rollout
from repoopt.store.local import LocalArtifactStore
from repoopt.tinker_adapter.converter import export_attempts_to_tinker_dataset
from repoopt.tinker_adapter.rl_bridge import RepoOptEnvGroupBuilder, RepoOptPiTools, load_repoopt_rl_task

MIN_FREE_DOCKER_BYTES = int(2.5 * 1024**3)


def _docker_and_disk_ready() -> tuple[bool, str]:
    free_bytes = shutil.disk_usage("/").free
    if free_bytes < MIN_FREE_DOCKER_BYTES:
        return (
            False,
            f"Requires >= 2.5 GiB free host disk for Docker images; only {free_bytes / (1024**3):.2f} GiB free",
        )
    proc = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return False, "Docker daemon not reachable"
    return True, "ok"


def test_ray_job_spec_and_rl_bridge_tool_parity() -> None:
    task_dir = Path("tasks/py-lj-forces").resolve()
    exp_cfg = ExperimentConfig(
        experiment_id="ray-parity",
        models=ModelConfig(teacher_model="z-ai/glm-5.3"),
        execution=ExecutionConfig(environment_backend="ray"),
    )
    spec = build_ray_job_spec(
        task_dir=task_dir,
        exp_config=exp_cfg,
        ray_import_path="harbor_ray.environment:RayEnvironment",
        gateway_url="http://ray-head:8080/v1",
        attempt_token="attempt:ray-parity:py-lj-forces:att-1",
    )
    assert spec["environment"]["import_path"] == "harbor_ray.environment:RayEnvironment"
    assert spec["agent"]["name"] == "pi"
    assert spec["agent"]["model_name"] == "openrouter/z-ai/glm-5.3"
    assert spec["agent"]["kwargs"]["model_api"] == "openai-completions"

    rl_task = load_repoopt_rl_task(task_dir)
    assert "compute_lj_system" in rl_task.instruction
    builder = RepoOptEnvGroupBuilder(task=rl_task, group_size=1)
    assert builder.logging_tags() == ["repoopt", "py-lj-forces"]

    class _DummySandbox:
        sandbox_id = "dummy"

    tools = RepoOptPiTools(_DummySandbox())  # type: ignore[arg-type]
    assert [t.name for t in tools.as_tools()] == ["read", "write", "edit", "bash"]


def test_docker_preflight_and_harbor_pi_rollout_e2e(tmp_path: Path) -> None:
    ready, reason = _docker_and_disk_ready()
    if not ready:
        pytest.skip(reason)

    py_report = verify_task_preflight(Path("tasks/py-lj-forces"))
    assert py_report.passed is True
    assert py_report.reference_fast_check is not None
    assert py_report.reference_fast_check["label"] == "VERIFIED_SPEEDUP"

    r_report = verify_task_preflight(Path("tasks/r-spatial-kernel"))
    assert r_report.passed is True
    assert r_report.reference_fast_check is not None
    assert r_report.reference_fast_check["label"] == "VERIFIED_SPEEDUP"

    # Scripted mock OpenRouter server executing a deterministic Pi optimization sequence
    turn = {"n": 0}
    app = FastAPI()
    fast_code = (
        Path("tasks/py-lj-forces/environment/repo/ljforces/core.py").read_text(encoding="utf-8")
    )
    assert fast_code  # ensure readable

    @app.post("/v1/chat/completions")
    async def mock_chat(request: Request) -> JSONResponse:
        del request
        turn["n"] += 1
        idx = turn["n"]
        if idx == 1:
            msg = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "First benchmark the baseline with repoopt-bench.",
                "tool_calls": [
                    {
                        "id": "call_bench_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "repoopt-bench"}'},
                    }
                ],
            }
            finish = "tool_calls"
        elif idx == 2:
            msg = {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Apply fast vectorized LJ patch.",
                "tool_calls": [
                    {
                        "id": "call_patch_2",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": (
                                        "cat <<'EOF' > /tmp/fast.patch\n"
                                        + Path("tasks/py-lj-forces/solution/reference_fast.patch").read_text(encoding="utf-8")
                                        + "\nEOF\ngit apply /tmp/fast.patch && repoopt-bench"
                                    )
                                }
                            ),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            msg = {
                "role": "assistant",
                "content": "Optimization complete and verified via repoopt-bench.",
                "reasoning_content": "Speedup confirmed.",
            }
            finish = "stop"

        return JSONResponse(
            {
                "id": f"mock-{idx}",
                "model": "z-ai/glm-5.3",
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            }
        )

    async def _run() -> None:
        config = uvicorn.Config(app, host="0.0.0.0", port=0, log_level="warning")
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.02)
        port = int(server.servers[0].sockets[0].getsockname()[1])

        try:
            store = LocalArtifactStore(tmp_path / "artifacts")
            exp_cfg = ExperimentConfig(
                experiment_id="docker-e2e",
                models=ModelConfig(teacher_model="z-ai/glm-5.3", reasoning_effort=None),
                storage=StorageConfig(backend="local", local_root=tmp_path / "artifacts"),
                execution=ExecutionConfig(agent_timeout_sec=180.0, verifier_timeout_sec=180.0),
            )
            res = await run_harbor_rollout(
                Path("tasks/py-lj-forces"),
                store,
                exp_cfg,
                attempt_id="e2e-att1",
                trials_dir=tmp_path / "trials",
                upstream_base_url=f"http://127.0.0.1:{port}/v1",
                upstream_api_key="mock-key",
            )
            assert res["trajectory_valid"] is True
            assert res["evaluation"]["label"] == "VERIFIED_SPEEDUP"

            datums, conv_report = export_attempts_to_tinker_dataset(
                store,
                "docker-e2e",
                [("py-lj-forces", "e2e-att1")],
                model_name="Qwen/Qwen3.5-9B",
                renderer_name="qwen3_5",
                require_verified_speedup=True,
            )
            assert conv_report.trajectories_processed == 1
            assert len(datums) >= 1
        finally:
            server.should_exit = True
            await task

    asyncio.run(_run())
