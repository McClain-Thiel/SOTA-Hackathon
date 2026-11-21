"""RepoOpt command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from repoopt.architect.builder import clone_and_squash_repo, scaffold_architect_workspace
from repoopt.bench.preflight_gate import verify_task_preflight
from repoopt.config import (
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    StorageConfig,
    TrackingConfig,
)
from repoopt.runner.harbor_runner import run_harbor_rollout_sync
from repoopt.store import ArtifactStore, StoragePaths, build_artifact_store
from repoopt.tinker_adapter.converter import (
    convert_trajectory_dict_to_datums,
    export_attempts_to_tinker_dataset,
)
from repoopt.tracking.mlflow_logger import sync_experiment_to_mlflow


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path.home() / ".env", override=False)


def _storage_from_args(args: argparse.Namespace) -> StorageConfig:
    _load_env()
    env_cfg = StorageConfig.from_env()
    if getattr(args, "s3_bucket", None):
        return StorageConfig(
            backend="s3",
            s3_bucket=args.s3_bucket,
            s3_prefix=getattr(args, "s3_prefix", None) or env_cfg.s3_prefix,
            s3_region=env_cfg.s3_region,
            s3_endpoint_url=env_cfg.s3_endpoint_url,
        )
    if getattr(args, "store_root", None) is not None:
        return StorageConfig(
            backend="local",
            local_root=Path(args.store_root),
        )
    if getattr(args, "s3_prefix", None) is not None and env_cfg.backend == "s3":
        return env_cfg.model_copy(update={"s3_prefix": args.s3_prefix})
    return env_cfg


def _tracking_from_args(args: argparse.Namespace, *, force_enable: bool = False) -> TrackingConfig:
    _load_env()
    env_cfg = TrackingConfig.from_env()
    mlflow_override = getattr(args, "mlflow", None)
    enabled = bool(mlflow_override) if mlflow_override is not None else (True if force_enable else env_cfg.enabled)
    tracking_uri = (
        getattr(args, "mlflow_tracking_uri", None)
        or getattr(args, "tracking_uri", None)
        or env_cfg.tracking_uri
    )
    experiment_name = (
        getattr(args, "mlflow_experiment", None)
        or getattr(args, "experiment_name", None)
        or env_cfg.experiment_name
    )
    log_artifacts = (
        not bool(getattr(args, "no_artifacts", False))
        if hasattr(args, "no_artifacts")
        else env_cfg.log_artifacts
    )
    return TrackingConfig(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        log_artifacts=log_artifacts,
    )


def _add_storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store-root", default=None)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)


def _discover_attempts(store: ArtifactStore, experiment_id: str) -> list[tuple[str, str]]:
    prefix = f"experiments/{experiment_id}/tasks/"
    keys = store.list_prefix(prefix)
    found: set[tuple[str, str]] = set()
    for key in keys:
        parts = key.split("/")
        # experiments/<exp>/tasks/<task>/attempts/<attempt>/trajectory.json
        if len(parts) >= 7 and parts[2] == "tasks" and parts[4] == "attempts" and parts[6] == "trajectory.json":
            found.add((parts[3], parts[5]))
    return sorted(found)


def _parse_attempt_selectors(items: list[str] | None) -> list[tuple[str, str]] | None:
    if not items:
        return None
    attempts: list[tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"--attempt must be TASK_ID:ATTEMPT_ID, got {item!r}")
        t_id, a_id = item.split(":", 1)
        attempts.append((t_id, a_id))
    return attempts


def _cmd_preflight_verify(args: argparse.Namespace) -> int:
    storage_cfg = _storage_from_args(args)
    store = build_artifact_store(storage_cfg) if args.exp_id else None
    report = verify_task_preflight(
        Path(args.task_dir),
        max_cv_threshold=args.max_cv,
        min_ref_speedup=args.min_ref_speedup,
        store=store,
        experiment_id=args.exp_id,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0 if report.passed else 1


def _cmd_architect(args: argparse.Namespace) -> int:
    work_repo = Path(args.work_dir or f"/tmp/repoopt-arch-{args.task_id}")
    squashed = clone_and_squash_repo(args.repo, args.commit, work_repo)
    out_dir = scaffold_architect_workspace(
        task_id=args.task_id,
        language=args.language,
        source_repo_dir=squashed,
        output_task_dir=Path(args.out),
    )
    print(json.dumps({"task_dir": str(out_dir), "architect_prompt": str(out_dir / "ARCHITECT_PROMPT.md")}, indent=2))
    return 0


def _cmd_rollout(args: argparse.Namespace) -> int:
    _load_env()
    storage_cfg = _storage_from_args(args)
    tracking_cfg = _tracking_from_args(args)
    store = build_artifact_store(storage_cfg)
    teacher_model = args.model or os.environ.get("REPOOPT_TEACHER_MODEL", "").strip() or "z-ai/glm-5.3"
    ray_import_path = args.ray_import_path or os.environ.get("REPOOPT_RAY_IMPORT_PATH", "").strip() or None
    exp_cfg = ExperimentConfig(
        experiment_id=args.exp_id,
        models=ModelConfig(
            teacher_model=teacher_model,
            max_concurrency=args.max_concurrency,
        ),
        storage=storage_cfg,
        execution=ExecutionConfig(
            environment_backend=args.backend,
            max_turns=args.max_turns,
            skip_agent_install=not args.no_skip_install,
        ),
        tracking=tracking_cfg,
    )
    result = run_harbor_rollout_sync(
        Path(args.task_dir),
        store,
        exp_cfg,
        attempt_id=args.attempt_id,
        upstream_base_url=args.base_url,
        ray_import_path=ray_import_path,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_sync_mlflow(args: argparse.Namespace) -> int:
    _load_env()
    storage_cfg = _storage_from_args(args)
    tracking_cfg = _tracking_from_args(args, force_enable=True)
    store = build_artifact_store(storage_cfg)
    attempts = _parse_attempt_selectors(args.attempt)
    summary = sync_experiment_to_mlflow(
        store,
        args.exp_id,
        tracking_cfg,
        attempts=attempts,
        skip_existing=not args.force,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


def _cmd_convert_tinker(args: argparse.Namespace) -> int:
    _load_env()
    storage_cfg = _storage_from_args(args)
    store = build_artifact_store(storage_cfg)
    student_model = args.model or os.environ.get("REPOOPT_STUDENT_MODEL", "").strip() or "Qwen/Qwen3.5-9B"
    attempts = _parse_attempt_selectors(args.attempt)
    if attempts is None:
        attempts = _discover_attempts(store, args.exp_id)

    datums, report = export_attempts_to_tinker_dataset(
        store,
        args.exp_id,
        attempts,
        model_name=student_model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        require_verified_speedup=args.require_verified_speedup,
    )
    print(
        json.dumps(
            {
                "datums_emitted": len(datums),
                "report": report.model_dump(mode="json"),
                "report_uri": store.uri_for(StoragePaths.training_conversion_report(args.exp_id, student_model)),
            },
            indent=2,
        )
    )
    return 0 if datums else 1


def _cmd_smoke_tinker(args: argparse.Namespace) -> int:
    _load_env()
    if not os.environ.get("TINKER_API_KEY"):
        print(
            json.dumps(
                {
                    "status": "missing_key",
                    "message": "TINKER_API_KEY is not set in environment or .env; skipping live Tinker API call.",
                },
                indent=2,
            )
        )
        return 2

    import tinker

    storage_cfg = _storage_from_args(args)
    store = build_artifact_store(storage_cfg)
    student_model = args.model or os.environ.get("REPOOPT_STUDENT_MODEL", "").strip() or "Qwen/Qwen3.5-9B"
    attempts = _discover_attempts(store, args.exp_id)
    if attempts:
        datums, _ = export_attempts_to_tinker_dataset(
            store,
            args.exp_id,
            attempts[:1],
            model_name=student_model,
            renderer_name=args.renderer,
            max_length=args.max_length,
            require_verified_speedup=False,
        )
    else:
        synthetic_atif = {
            "schema_version": "ATIF-v1.8",
            "session_id": "smoke-tinker",
            "agent": {"name": "pi", "version": "0.85.1", "model_name": "z-ai/glm-5.3"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Optimize scientific kernel in /workspace."},
                {
                    "step_id": 2,
                    "source": "agent",
                    "reasoning_content": "Run repoopt-bench first.",
                    "message": "",
                    "llm_call_count": 1,
                    "tool_calls": [
                        {"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "repoopt-bench"}}
                    ],
                    "observation": {"results": [{"source_call_id": "c1", "content": '{"correct": true, "speedup": 1.0}'}]},
                },
                {"step_id": 3, "source": "agent", "message": "Verified baseline.", "llm_call_count": 1},
            ],
        }
        datums = convert_trajectory_dict_to_datums(
            synthetic_atif, model_name=student_model, renderer_name=args.renderer, max_length=args.max_length
        ).datums

    if not datums:
        raise RuntimeError("No Datums available for Tinker forward_backward smoke test")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=student_model)
    fwdbwd_future = training_client.forward_backward([datums[0]], loss_fn="cross_entropy")
    fwdbwd_res = fwdbwd_future.result()
    print(
        json.dumps(
            {
                "status": "ok",
                "base_model": student_model,
                "datum_tokens": datums[0].model_input.length,
                "loss_fn_outputs": str(getattr(fwdbwd_res, "metrics", fwdbwd_res)),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="repoopt", description="RepoOpt Distributed Optimization Trace Factory")
    sub = parser.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight-verify", help="Build task image and run automated baseline/mutation/reference gates")
    pf.add_argument("task_dir", help="Path to Harbor task directory")
    pf.add_argument("--exp-id", default=None)
    _add_storage_args(pf)
    pf.add_argument("--max-cv", type=float, default=0.25)
    pf.add_argument("--min-ref-speedup", type=float, default=1.5)
    pf.set_defaults(func=_cmd_preflight_verify)

    arch = sub.add_parser("architect", help="Clone pinned repo, squash git history, and scaffold BenchArchitect task")
    arch.add_argument("--repo", required=True)
    arch.add_argument("--commit", required=True)
    arch.add_argument("--language", choices=("python", "r"), required=True)
    arch.add_argument("--task-id", required=True)
    arch.add_argument("--out", required=True)
    arch.add_argument("--work-dir", default=None)
    arch.set_defaults(func=_cmd_architect)

    ro = sub.add_parser("rollout", help="Run a Harbor rollout routed through Host CaptureGateway")
    ro.add_argument("--task-dir", required=True)
    ro.add_argument("--exp-id", default="local-pilot")
    ro.add_argument("--attempt-id", default=None)
    ro.add_argument("--model", default=None)
    ro.add_argument("--base-url", default=None)
    ro.add_argument("--backend", choices=("docker", "ray"), default="docker")
    ro.add_argument("--ray-import-path", default=None)
    ro.add_argument("--max-turns", type=int, default=40)
    ro.add_argument("--max-concurrency", type=int, default=4)
    ro.add_argument("--no-skip-install", action="store_true")
    ro.add_argument("--mlflow", action=argparse.BooleanOptionalAction, default=None)
    ro.add_argument("--mlflow-tracking-uri", default=None)
    ro.add_argument("--mlflow-experiment", default=None)
    _add_storage_args(ro)
    ro.set_defaults(func=_cmd_rollout)

    sm = sub.add_parser("sync-mlflow", help="Sync stored rollout runs, artifacts, and traces to MLflow / Databricks")
    sm.add_argument("--exp-id", default="local-pilot")
    sm.add_argument("--attempt", action="append", help="Optional TASK_ID:ATTEMPT_ID selector")
    sm.add_argument("--tracking-uri", default=None)
    sm.add_argument("--experiment-name", default=None)
    sm.add_argument("--no-artifacts", action="store_true")
    sm.add_argument("--force", action="store_true", help="Re-log even if an MLflow run with the same attempt_key exists")
    _add_storage_args(sm)
    sm.set_defaults(func=_cmd_sync_mlflow)

    ct = sub.add_parser("convert-tinker", help="Convert stored ATIF trajectories into Qwen/Tinker SFT Datums")
    ct.add_argument("--exp-id", default="local-pilot")
    ct.add_argument("--attempt", action="append", help="Optional TASK_ID:ATTEMPT_ID selector")
    ct.add_argument("--model", default=None)
    ct.add_argument("--renderer", default="qwen3_5")
    ct.add_argument("--max-length", type=int, default=32768)
    ct.add_argument("--require-verified-speedup", action="store_true")
    _add_storage_args(ct)
    ct.set_defaults(func=_cmd_convert_tinker)

    st = sub.add_parser("smoke-tinker", help="Run a real Tinker LoRA forward_backward smoke test")
    st.add_argument("--exp-id", default="local-pilot")
    st.add_argument("--model", default=None)
    st.add_argument("--renderer", default="qwen3_5")
    st.add_argument("--max-length", type=int, default=4096)
    _add_storage_args(st)
    st.set_defaults(func=_cmd_smoke_tinker)

    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
