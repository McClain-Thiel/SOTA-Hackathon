"""Automated Docker preflight verification gate for RepoOpt Harbor tasks."""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskConfig as HarborTaskConfig
from pydantic import BaseModel, Field

from repoopt.store.base import ArtifactStore, StoragePaths


class PreflightGateError(RuntimeError):
    """Raised when a Harbor task fails any automated preflight oracle gate."""


class PreflightReport(BaseModel):
    """Summary of automated baseline, mutation, and reference checks for one task."""

    task_id: str
    task_dir: str
    image_tag: str
    passed: bool
    checked_at: str
    in_container_bench: dict[str, Any] = Field(default_factory=dict)
    baseline_noop: dict[str, Any] = Field(default_factory=dict)
    mutation_check: dict[str, Any] | None = None
    reference_fast_check: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


def sync_task_harness_assets(task_dir: Path) -> None:
    """Copy canonical repoopt_bench.py and run_verifier.py into task_dir and ensure test.sh +x."""
    assets_dir = Path(__file__).resolve().parent / "harness_assets"
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(assets_dir / "repoopt_bench.py", env_dir / "repoopt_bench.py")
    shutil.copy2(assets_dir / "run_verifier.py", tests_dir / "run_verifier.py")
    test_sh = tests_dir / "test.sh"
    if test_sh.exists():
        test_sh.chmod(test_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run_verifier_in_container(
    image_tag: str,
    tests_dir: Path,
    *,
    patch_text: str | None = None,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="repoopt-pf-logs-") as td:
        logs_dir = Path(td)
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tests_dir.resolve()}:/tests:ro",
            "-v",
            f"{logs_dir.resolve()}:/logs/verifier",
        ]
        if patch_text is not None:
            patch_file = logs_dir / "_apply.patch"
            patch_file.write_text(patch_text, encoding="utf-8")
            script = (
                "git -C /workspace apply --whitespace=nowarn /logs/verifier/_apply.patch && "
                "rm -f /logs/verifier/_apply.patch && "
                "bash /tests/test.sh"
            )
            cmd.extend([image_tag, "bash", "-lc", script])
        else:
            cmd.extend([image_tag, "bash", "/tests/test.sh"])

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, check=False)
        reward_json_path = logs_dir / "reward.json"
        reward_txt_path = logs_dir / "reward.txt"
        if not reward_json_path.exists() or not reward_txt_path.exists():
            raise PreflightGateError(
                f"Verifier did not produce reward files (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        data = json.loads(reward_json_path.read_text(encoding="utf-8"))
        data["reward_txt_value"] = float(reward_txt_path.read_text(encoding="utf-8").strip())
        return data


def _run_in_episode_bench(image_tag: str, timeout_sec: int = 180) -> dict[str, Any]:
    proc = subprocess.run(
        ["docker", "run", "--rm", image_tag, "python3", "-I", "/usr/local/bin/repoopt-bench", "--suite", "dev"],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise PreflightGateError(
            f"In-container `repoopt-bench --suite dev` failed (rc={proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def verify_task_preflight(
    task_dir: Path,
    *,
    max_cv_threshold: float = 0.25,
    min_ref_speedup: float = 1.5,
    store: ArtifactStore | None = None,
    experiment_id: str | None = None,
) -> PreflightReport:
    """Validate task schema, build Docker image, and run baseline/mutation/reference gates."""
    resolved = task_dir.resolve()
    task_toml = resolved / "task.toml"
    if not task_toml.exists():
        raise PreflightGateError(f"Missing task.toml in {resolved}")

    harbor_cfg = HarborTaskConfig.model_validate_toml(task_toml.read_text(encoding="utf-8"))
    sync_task_harness_assets(resolved)

    env_dir = resolved / "environment"
    tests_dir = resolved / "tests"
    task_id = resolved.name
    image_tag = f"repoopt-preflight-{task_id.lower()}:{uuid.uuid4().hex[:8]}"

    build_proc = subprocess.run(
        ["docker", "build", "-t", image_tag, str(env_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if build_proc.returncode != 0:
        raise PreflightGateError(
            f"Docker build failed for {task_id}:\n{build_proc.stdout}\n{build_proc.stderr}"
        )

    warnings: list[str] = []
    try:
        # 1. Verify in-episode `repoopt-bench --suite dev` works identically for SFT & RL
        in_ep = _run_in_episode_bench(image_tag)
        if not in_ep.get("correct"):
            raise PreflightGateError(f"In-container repoopt-bench baseline failed: {in_ep}")

        # 2. Verify clean-room baseline no-op check (correct=True, is_noop=True, reward=0.0)
        noop_res = _run_verifier_in_container(image_tag, tests_dir, patch_text=None)
        if not noop_res.get("correct") or not noop_res.get("is_noop_patch"):
            raise PreflightGateError(f"Baseline no-op verifier gate failed: {noop_res}")
        if noop_res.get("label") != "CORRECT_NO_SPEEDUP" or float(noop_res.get("reward", -1.0)) != 0.0:
            raise PreflightGateError(
                f"Baseline no-op must score CORRECT_NO_SPEEDUP with reward=0.0, got: {noop_res}"
            )
        timing = noop_res.get("timing") or {}
        max_cv = max(float(timing.get("baseline_cv", 0.0)), float(timing.get("candidate_cv", 0.0)))
        if max_cv > max_cv_threshold:
            raise PreflightGateError(
                f"Baseline timing CV={max_cv:.4f} exceeds max_cv_threshold={max_cv_threshold}"
            )

        # 3. Synthetic mutation check: bad patch MUST be caught as INCORRECT with reward=0.0
        mutation_res: dict[str, Any] | None = None
        mutation_patch = tests_dir / "mutation_bad.patch"
        if mutation_patch.exists():
            mutation_res = _run_verifier_in_container(
                image_tag,
                tests_dir,
                patch_text=mutation_patch.read_text(encoding="utf-8"),
            )
            if mutation_res.get("correct") is not False or mutation_res.get("label") != "INCORRECT":
                raise PreflightGateError(
                    f"Mutation gate failed! Verifier did NOT reject bad patch: {mutation_res}"
                )
            if float(mutation_res.get("reward", -1.0)) != 0.0:
                raise PreflightGateError(
                    f"Mutation gate must receive reward=0.0, got {mutation_res.get('reward')}"
                )
        else:
            warnings.append("No tests/mutation_bad.patch supplied; skipped mutation gate")

        # 4. Reference fast patch check (if provided)
        ref_res: dict[str, Any] | None = None
        ref_patch = resolved / "solution" / "reference_fast.patch"
        if ref_patch.exists():
            ref_res = _run_verifier_in_container(
                image_tag,
                tests_dir,
                patch_text=ref_patch.read_text(encoding="utf-8"),
            )
            if ref_res.get("label") != "VERIFIED_SPEEDUP" or not ref_res.get("correct"):
                raise PreflightGateError(f"Reference fast patch failed verifier: {ref_res}")
            ref_speedup = float((ref_res.get("timing") or {}).get("speedup", 0.0))
            if ref_speedup < min_ref_speedup:
                raise PreflightGateError(
                    f"Reference fast patch speedup={ref_speedup:.3f}x < min_ref_speedup={min_ref_speedup}x"
                )
        else:
            warnings.append("No solution/reference_fast.patch supplied; skipped reference speedup gate")

        report = PreflightReport(
            task_id=task_id,
            task_dir=str(resolved),
            image_tag=image_tag,
            passed=True,
            checked_at=datetime.now(timezone.utc).isoformat(),
            in_container_bench=in_ep,
            baseline_noop=noop_res,
            mutation_check=mutation_res,
            reference_fast_check=ref_res,
            warnings=warnings,
        )

        if store is not None and experiment_id is not None:
            store.put_json(
                StoragePaths.task_benchmark(experiment_id, task_id),
                report.model_dump(mode="json"),
            )
            inspect_proc = subprocess.run(
                ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
                capture_output=True,
                text=True,
                check=False,
            )
            store.put_json(
                StoragePaths.task_image(experiment_id, task_id),
                {
                    "task_id": task_id,
                    "harbor_name": harbor_cfg.task.name if harbor_cfg.task else task_id,
                    "image_tag": image_tag,
                    "image_digest_id": inspect_proc.stdout.strip(),
                    "built_at": report.checked_at,
                },
            )

        return report
    finally:
        subprocess.run(["docker", "rmi", image_tag], capture_output=True, check=False)
