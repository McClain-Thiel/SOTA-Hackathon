#!/usr/bin/env python3
"""Clean-room Harbor verifier script invoked by tests/test.sh inside container.

Extracts text-only candidate.patch from /workspace, rejects binary file drops,
applies patch to pristine /opt/baseline_src in an isolated directory, runs
dev + unseen holdout accuracy and paired timing via /usr/local/bin/repoopt-bench,
and writes /logs/verifier/reward.txt and /logs/verifier/reward.json.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _classify_and_score(
    *,
    correct: bool,
    is_noop_patch: bool,
    build_ok: bool,
    tests_ok: bool,
    max_abs_err: float | None,
    max_rel_err: float | None,
    timing: dict[str, Any] | None,
    reason: str = "",
) -> dict[str, Any]:
    if not build_ok or not tests_ok or not correct:
        return {
            "label": "INCORRECT",
            "reward": 0.0,
            "correct": False,
            "is_noop_patch": is_noop_patch,
            "build_ok": build_ok,
            "tests_ok": tests_ok,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": timing,
            "reason": reason or "Failed build, unit tests, or numerical equivalence",
        }
    if is_noop_patch:
        return {
            "label": "CORRECT_NO_SPEEDUP",
            "reward": 0.0,
            "correct": True,
            "is_noop_patch": True,
            "build_ok": True,
            "tests_ok": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": timing,
            "reason": "No-op candidate patch receives 0.0 RL reward",
        }
    if timing is None:
        return {
            "label": "INCONCLUSIVE",
            "reward": 0.0,
            "correct": True,
            "is_noop_patch": False,
            "build_ok": True,
            "tests_ok": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": None,
            "reason": reason or "Missing timing data",
        }

    speedup = float(timing["speedup"])
    ci_low = float(timing["ci95_low"])
    ci_high = float(timing["ci95_high"])
    max_cv = max(float(timing["baseline_cv"]), float(timing["candidate_cv"]))

    if speedup > 1.05 and ci_low > 1.02:
        reward = round(min(4.0, max(0.0, math.log2(speedup))), 6)
        return {
            "label": "VERIFIED_SPEEDUP",
            "reward": reward,
            "correct": True,
            "is_noop_patch": False,
            "build_ok": True,
            "tests_ok": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": timing,
            "reason": f"Verified speedup {speedup:.3f}x (CI95 low {ci_low:.3f}x)",
        }

    if max_cv > 0.25:
        return {
            "label": "INCONCLUSIVE",
            "reward": 0.05,
            "correct": True,
            "is_noop_patch": False,
            "build_ok": True,
            "tests_ok": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": timing,
            "reason": f"Timing CV too high ({max_cv:.3f})",
        }

    if ci_high < 0.98:
        return {
            "label": "CORRECT_BUT_SLOWER",
            "reward": 0.05,
            "correct": True,
            "is_noop_patch": False,
            "build_ok": True,
            "tests_ok": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing": timing,
            "reason": f"Correct but slower ({speedup:.3f}x)",
        }

    return {
        "label": "CORRECT_NO_SPEEDUP",
        "reward": 0.05,
        "correct": True,
        "is_noop_patch": False,
        "build_ok": True,
        "tests_ok": True,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "timing": timing,
        "reason": f"Correct without statistically significant speedup ({speedup:.3f}x)",
    }


def _extract_patch(workspace_dir: Path) -> tuple[str, list[str]]:
    """Capture text patch and reject untracked/modified binary artifacts."""
    # Ignore compiled build outputs before git add -N
    subprocess.run(
        ["git", "-C", str(workspace_dir), "add", "-N", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    numstat = subprocess.run(
        ["git", "-C", str(workspace_dir), "diff", "--numstat", "HEAD", "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    binary_files: list[str] = []
    allowed_binary_ignore_suffixes = (".so", ".o", ".pyd", ".pyc", ".dll")
    text_paths: list[str] = []
    for line in (numstat.stdout or "").splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        if path.endswith(allowed_binary_ignore_suffixes) or "__pycache__" in path or ".egg-info" in path:
            continue
        if added == "-" or deleted == "-":
            binary_files.append(path)
        else:
            text_paths.append(path)

    if binary_files:
        return "", binary_files

    if not text_paths:
        return "", []

    diff = subprocess.run(
        ["git", "-C", str(workspace_dir), "diff", "HEAD", "--", *text_paths],
        capture_output=True,
        text=True,
        check=False,
    )
    return diff.stdout or "", []


def main() -> int:
    logs_dir = Path(os.environ.get("VERIFIER_LOGS_DIR", "/logs/verifier"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(os.environ.get("REPOOPT_WORKSPACE_DIR", "/workspace"))
    baseline_src = Path(os.environ.get("REPOOPT_BASELINE_SRC", "/opt/baseline_src"))
    bench_bin = os.environ.get("REPOOPT_BENCH_BIN", "/usr/local/bin/repoopt-bench")
    bench_cfg = os.environ.get("REPOOPT_BENCH_CONFIG", "/opt/bench/bench_config.json")

    patch_text, binary_files = _extract_patch(workspace_dir)
    (logs_dir / "candidate.patch").write_text(patch_text, encoding="utf-8")

    if binary_files:
        result = _classify_and_score(
            correct=False,
            is_noop_patch=False,
            build_ok=False,
            tests_ok=False,
            max_abs_err=None,
            max_rel_err=None,
            timing=None,
            reason=f"Rejected binary artifacts in workspace: {binary_files}",
        )
        (logs_dir / "reward.txt").write_text(f"{result['reward']}\n", encoding="utf-8")
        (logs_dir / "reward.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    is_noop = (patch_text.strip() == "")

    with tempfile.TemporaryDirectory(prefix="repoopt-clean-") as tmp_dir:
        clean_dir = Path(tmp_dir) / "candidate_src"
        shutil.copytree(baseline_src, clean_dir)
        if not is_noop:
            proc_apply = subprocess.run(
                ["git", "-C", str(clean_dir), "apply", "--whitespace=nowarn", "-"],
                input=patch_text,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_apply.returncode != 0:
                result = _classify_and_score(
                    correct=False,
                    is_noop_patch=False,
                    build_ok=False,
                    tests_ok=False,
                    max_abs_err=None,
                    max_rel_err=None,
                    timing=None,
                    reason=f"Clean-room git apply failed: {proc_apply.stderr}",
                )
                (logs_dir / "reward.txt").write_text(f"{result['reward']}\n", encoding="utf-8")
                (logs_dir / "reward.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result, indent=2))
                return 0

        bench_out = Path(tmp_dir) / "bench_out.json"
        proc_bench = subprocess.run(
            [
                sys.executable,
                "-I",
                bench_bin,
                "--config",
                bench_cfg,
                "--suite",
                "all",
                "--workspace",
                str(clean_dir),
                "--json-out",
                str(bench_out),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if not bench_out.exists():
            result = _classify_and_score(
                correct=False,
                is_noop_patch=is_noop,
                build_ok=False,
                tests_ok=False,
                max_abs_err=None,
                max_rel_err=None,
                timing=None,
                reason=f"repoopt-bench produced no output (rc={proc_bench.returncode}): {proc_bench.stderr}",
            )
        else:
            bdata = json.loads(bench_out.read_text(encoding="utf-8"))
            result = _classify_and_score(
                correct=bool(bdata.get("correct", False)),
                is_noop_patch=is_noop,
                build_ok=bool(bdata.get("build_ok", False)),
                tests_ok=bool(bdata.get("tests_ok", False)),
                max_abs_err=bdata.get("max_abs_err"),
                max_rel_err=bdata.get("max_rel_err"),
                timing=bdata.get("timing"),
                reason=str(bdata.get("reason", "")),
            )

    (logs_dir / "reward.txt").write_text(f"{result['reward']}\n", encoding="utf-8")
    (logs_dir / "reward.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
