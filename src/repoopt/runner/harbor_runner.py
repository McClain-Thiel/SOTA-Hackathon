"""Harbor rollout runner with Host CaptureGateway, ATIF-v1.8 harvesting, and Ray config export."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from harbor.models.environment_type import EnvironmentType
from harbor.models.model_api_format import ModelApiFormat
from harbor.models.task.config import TaskConfig as HarborTaskConfig
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
    EnvironmentConfig as HarborEnvConfig,
    TaskConfig as HarborTrialTaskConfig,
    TrialConfig as HarborTrialConfig,
    VerifierConfig as HarborVerifierConfig,
)
from harbor.trial.trial import Trial

from repoopt.bench.preflight_gate import sync_task_harness_assets
from repoopt.config import ExperimentConfig, ExperimentManifest
from repoopt.proxy.gateway import CaptureGateway, RunningGateway, format_attempt_token
from repoopt.store.base import ArtifactStore, StoragePaths
from repoopt.tinker_adapter.atif_parser import ATIFValidationError, ensure_atif_v1_8
from repoopt.tracking.mlflow_logger import log_attempt_to_mlflow


def _load_host_env() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path.home() / ".env", override=False)


def build_ray_job_spec(
    *,
    task_dir: Path,
    exp_config: ExperimentConfig,
    ray_import_path: str,
    gateway_url: str,
    attempt_token: str,
    ray_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pinned Harbor TrialConfig payload targeting a pre-release Ray environment backend."""
    return {
        "schema_version": 1,
        "task": {"path": str(task_dir.resolve())},
        "agent": {
            "name": exp_config.execution.agent_name,
            "model_name": exp_config.models.harbor_teacher_model,
            "base_url": gateway_url,
            "model_api_key_name": "OPENAI_API_KEY",
            "model_api_format": ModelApiFormat.OPENAI_CHAT_COMPLETIONS.value,
            "env": {
                "OPENAI_API_KEY": attempt_token,
                "NO_PROXY": "host.docker.internal,localhost,127.0.0.1",
            },
            "kwargs": {
                "model_api": "openai-completions",
                "skip_install": exp_config.execution.skip_agent_install,
                "max_turns": exp_config.execution.max_turns,
            },
        },
        "environment": {
            "type": None,
            "import_path": ray_import_path,
            "override_cpus": exp_config.execution.cpus,
            "override_memory_mb": exp_config.execution.memory_mb,
            "override_gpus": exp_config.execution.gpus,
            "kwargs": ray_kwargs or {"sandbox_backend": "gvisor"},
        },
    }


async def run_harbor_rollout(
    task_dir: Path,
    store: ArtifactStore,
    exp_config: ExperimentConfig,
    *,
    attempt_id: str | None = None,
    trials_dir: Path = Path("jobs/harbor_trials"),
    upstream_base_url: str | None = None,
    upstream_api_key: str | None = None,
    ray_import_path: str | None = None,
    ray_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one Harbor rollout via CaptureGateway and archive ATIF, patch, and evaluation."""
    _load_host_env()
    resolved_task = task_dir.resolve()
    sync_task_harness_assets(resolved_task)

    task_toml_cfg = HarborTaskConfig.model_validate_toml((resolved_task / "task.toml").read_text(encoding="utf-8"))
    task_id = resolved_task.name
    att_id = attempt_id or f"att-{uuid.uuid4().hex[:8]}"

    # Persist pinned experiment manifest
    manifest = ExperimentManifest.create(exp_config)
    store.put_json(StoragePaths.experiment_manifest(exp_config.experiment_id), manifest.model_dump(mode="json"))

    effective_upstream = upstream_base_url or exp_config.models.teacher_base_url
    effective_key = upstream_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    gateway = CaptureGateway(
        store,
        upstream_base_url=effective_upstream,
        upstream_api_key=effective_key,
        max_concurrency=exp_config.models.max_concurrency,
        flush_interval_sec=exp_config.storage.flush_interval_sec,
    )

    attempt_token = format_attempt_token(exp_config.experiment_id, task_id, att_id)
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_name = f"{task_id}__{att_id}"

    started_at = datetime.now(timezone.utc).isoformat()
    trial_result: Any = None
    trial_error: str | None = None
    trial_paths = None

    async with RunningGateway(gateway, host="0.0.0.0", port=0) as running_gw:
        base_url = (
            running_gw.container_base_url
            if exp_config.execution.environment_backend == "docker"
            else running_gw.local_base_url
        )

        agent_kwargs: dict[str, Any] = {
            "model_api": "openai-completions",
            "skip_install": exp_config.execution.skip_agent_install,
            "max_turns": exp_config.execution.max_turns,
        }
        if exp_config.models.reasoning_effort is not None and exp_config.execution.agent_name == "pi":
            agent_kwargs["reasoning_effort"] = exp_config.models.reasoning_effort

        if exp_config.execution.environment_backend == "ray":
            if not ray_import_path:
                raise ValueError("environment_backend='ray' requires ray_import_path (e.g. harbor_ray:RayEnvironment)")
            env_cfg = HarborEnvConfig(
                type=None,
                import_path=ray_import_path,
                override_cpus=task_toml_cfg.environment.cpus or exp_config.execution.cpus,
                override_memory_mb=task_toml_cfg.environment.memory_mb or exp_config.execution.memory_mb,
                override_gpus=task_toml_cfg.environment.gpus or exp_config.execution.gpus,
                kwargs=ray_kwargs or {"sandbox_backend": "gvisor"},
            )
        else:
            env_cfg = HarborEnvConfig(
                type=EnvironmentType.DOCKER,
                override_cpus=task_toml_cfg.environment.cpus or exp_config.execution.cpus,
                override_memory_mb=task_toml_cfg.environment.memory_mb or exp_config.execution.memory_mb,
            )

        trial_cfg = HarborTrialConfig(
            task=HarborTrialTaskConfig(path=resolved_task),
            trial_name=trial_name,
            trials_dir=trials_dir.resolve(),
            agent=HarborAgentConfig(
                name=exp_config.execution.agent_name,
                model_name=exp_config.models.harbor_teacher_model,
                base_url=base_url,
                model_api_key_name="OPENAI_API_KEY",
                model_api_format=ModelApiFormat.OPENAI_CHAT_COMPLETIONS,
                override_timeout_sec=exp_config.execution.agent_timeout_sec,
                env={
                    "OPENAI_API_KEY": attempt_token,
                    "NO_PROXY": "host.docker.internal,localhost,127.0.0.1",
                    "no_proxy": "host.docker.internal,localhost,127.0.0.1",
                },
                kwargs=agent_kwargs,
            ),
            environment=env_cfg,
            verifier=HarborVerifierConfig(
                override_timeout_sec=exp_config.execution.verifier_timeout_sec,
            ),
        )

        trial = await Trial.create(trial_cfg)
        trial_paths = trial.paths
        try:
            trial_result = await trial.run()
        except Exception as exc:
            trial_error = f"{type(exc).__name__}: {exc}"
        finally:
            await gateway.flush_attempt(exp_config.experiment_id, task_id, att_id)

    finished_at = datetime.now(timezone.utc).isoformat()
    archived: dict[str, Any] = {
        "experiment_id": exp_config.experiment_id,
        "task_id": task_id,
        "attempt_id": att_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "trial_dir": str(trial_paths.trial_dir) if trial_paths else None,
        "trial_error": trial_error,
        "trajectory_valid": False,
        "evaluation": None,
    }

    if trial_paths is not None:
        # 1. Harvest and validate ATIF trajectory (upgrading ATIF-v1.7 -> ATIF-v1.8)
        traj_file = trial_paths.agent_dir / "trajectory.json"
        if traj_file.exists():
            raw_traj = json.loads(traj_file.read_text(encoding="utf-8"))
            try:
                atif_v18 = ensure_atif_v1_8(raw_traj)
                archived["trajectory_valid"] = True
                store.put_json(
                    StoragePaths.attempt_trajectory(exp_config.experiment_id, task_id, att_id),
                    atif_v18,
                )
            except ATIFValidationError as exc:
                archived["trajectory_validation_error"] = str(exc)
                store.put_json(
                    StoragePaths.attempt_trajectory(exp_config.experiment_id, task_id, att_id),
                    raw_traj,
                )

        # 2. Harvest candidate.patch
        patch_file = trial_paths.verifier_dir / "candidate.patch"
        if patch_file.exists():
            store.put_bytes(
                StoragePaths.attempt_patch(exp_config.experiment_id, task_id, att_id),
                patch_file.read_bytes(),
                content_type="text/x-diff",
            )

        # 3. Harvest evaluation.json from verifier reward.json (or synthesize timeout/env error)
        reward_json = trial_paths.verifier_dir / "reward.json"
        if reward_json.exists():
            eval_payload = json.loads(reward_json.read_text(encoding="utf-8"))
        else:
            is_timeout = trial_error is not None and "timeout" in trial_error.lower()
            eval_payload = {
                "label": "TIMEOUT" if is_timeout else "ENVIRONMENT_FAILURE",
                "reward": 0.0,
                "correct": False,
                "is_noop_patch": True,
                "reason": trial_error or "Verifier produced no reward.json",
            }
        archived["evaluation"] = eval_payload
        store.put_json(
            StoragePaths.attempt_evaluation(exp_config.experiment_id, task_id, att_id),
            eval_payload,
        )

    if trial_result is not None and hasattr(trial_result, "model_dump"):
        archived["harbor_trial_result"] = trial_result.model_dump(mode="json")

    completion_key = StoragePaths.attempt_completion(exp_config.experiment_id, task_id, att_id)
    store.put_json(completion_key, archived)

    if exp_config.tracking.enabled:
        try:
            archived["mlflow"] = log_attempt_to_mlflow(
                store,
                exp_config.experiment_id,
                task_id,
                att_id,
                exp_config.tracking,
            )
        except Exception as exc:
            archived["mlflow"] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        store.put_json(completion_key, archived)

    return archived


def run_harbor_rollout_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Synchronous convenience wrapper around run_harbor_rollout."""
    return asyncio.run(run_harbor_rollout(*args, **kwargs))
