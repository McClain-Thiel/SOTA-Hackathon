#!/usr/bin/env python3
"""Unified Python + R paired benchmark and numerical-equivalence runner for RepoOpt containers.

Runs under `python3 -I` using only the Python standard library so workspace files
cannot shadow imports or alter timing primitives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(os.environ.get("REPOOPT_BENCH_CONFIG", "/opt/bench/bench_config.json"))


def _single_thread_env(base_env: dict[str, str], allow_multithread: bool) -> dict[str, str]:
    env = dict(base_env)
    if not allow_multithread:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            env[var] = "1"
    return env


def _compare_numeric_trees(
    base_val: Any,
    cand_val: Any,
    *,
    rtol: float,
    atol: float,
    path: str = "root",
) -> tuple[bool, float, float, str]:
    """Recursively compare nested JSON numbers/lists/dicts with NaN/Inf exactness."""
    if isinstance(base_val, dict) and isinstance(cand_val, dict):
        if set(base_val.keys()) != set(cand_val.keys()):
            return (
                False,
                float("inf"),
                float("inf"),
                f"Dict key mismatch at {path}: base={sorted(base_val)} cand={sorted(cand_val)}",
            )
        max_abs = 0.0
        max_rel = 0.0
        for key in sorted(base_val.keys()):
            ok, a_err, r_err, reason = _compare_numeric_trees(
                base_val[key], cand_val[key], rtol=rtol, atol=atol, path=f"{path}.{key}"
            )
            max_abs = max(max_abs, a_err)
            max_rel = max(max_rel, r_err)
            if not ok:
                return False, max_abs, max_rel, reason
        return True, max_abs, max_rel, ""

    if isinstance(base_val, list) and isinstance(cand_val, list):
        if len(base_val) != len(cand_val):
            return (
                False,
                float("inf"),
                float("inf"),
                f"List length mismatch at {path}: base={len(base_val)} cand={len(cand_val)}",
            )
        max_abs = 0.0
        max_rel = 0.0
        for idx, (bv, cv) in enumerate(zip(base_val, cand_val)):
            ok, a_err, r_err, reason = _compare_numeric_trees(
                bv, cv, rtol=rtol, atol=atol, path=f"{path}[{idx}]"
            )
            max_abs = max(max_abs, a_err)
            max_rel = max(max_rel, r_err)
            if not ok:
                return False, max_abs, max_rel, reason
        return True, max_abs, max_rel, ""

    if isinstance(base_val, bool) or isinstance(cand_val, bool):
        if base_val is not cand_val:
            return False, float("inf"), float("inf"), f"Bool mismatch at {path}: {base_val!r} != {cand_val!r}"
        return True, 0.0, 0.0, ""

    if isinstance(base_val, (int, float)) and isinstance(cand_val, (int, float)):
        bf = float(base_val)
        cf = float(cand_val)
        if math.isnan(bf) or math.isnan(cf):
            if math.isnan(bf) and math.isnan(cf):
                return True, 0.0, 0.0, ""
            return False, float("inf"), float("inf"), f"NaN mismatch at {path}: base={bf}, cand={cf}"
        if math.isinf(bf) or math.isinf(cf):
            if bf == cf:
                return True, 0.0, 0.0, ""
            return False, float("inf"), float("inf"), f"Inf mismatch at {path}: base={bf}, cand={cf}"
        abs_err = abs(cf - bf)
        rel_err = abs_err / max(abs(bf), 1e-15)
        tol = atol + rtol * abs(bf)
        if abs_err > tol:
            return (
                False,
                abs_err,
                rel_err,
                f"Tolerance exceeded at {path}: base={bf:.12g}, cand={cf:.12g}, abs_err={abs_err:.3e} > tol={tol:.3e}",
            )
        return True, abs_err, rel_err, ""

    if base_val != cand_val:
        return False, float("inf"), float("inf"), f"Value mismatch at {path}: {base_val!r} != {cand_val!r}"
    return True, 0.0, 0.0, ""


def _decode_special_floats(obj: Any) -> Any:
    """Convert serialized __NAN__ / __INF__ / __NINF__ sentinel strings into IEEE floats."""
    if isinstance(obj, dict):
        return {k: _decode_special_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_special_floats(v) for v in obj]
    if obj == "__NAN__":
        return float("nan")
    if obj == "__INF__":
        return float("inf")
    if obj == "__NINF__":
        return float("-inf")
    return obj


def _role_exec_cmd(
    cfg: dict[str, Any],
    *,
    role: str,
    driver_path: str,
    extra_args: list[str],
    candidate_env_dir: Path,
) -> tuple[list[str], dict[str, str]]:
    language = cfg["language"]
    env = _single_thread_env(os.environ, bool(cfg.get("allow_multithread", False)))
    if language == "python":
        py_bin = cfg["baseline_python"] if role == "baseline" else str(candidate_env_dir / "bin" / "python")
        return [py_bin, "-I", driver_path, *extra_args], env
    if language == "r":
        r_lib = cfg["baseline_r_lib"] if role == "baseline" else str(candidate_env_dir)
        existing = env.get("R_LIBS_USER", "")
        env["R_LIBS_USER"] = f"{r_lib}:{existing}" if existing else r_lib
        return ["Rscript", "--vanilla", driver_path, *extra_args], env
    raise ValueError(f"Unsupported language in bench_config.json: {language!r}")


def _build_candidate(
    cfg: dict[str, Any],
    *,
    workspace_dir: Path,
    candidate_env_dir: Path,
) -> tuple[bool, str]:
    language = cfg["language"]
    env = dict(os.environ)
    if language == "python":
        if candidate_env_dir.exists():
            shutil.rmtree(candidate_env_dir)
        # Lightweight venv inheriting system/shared site packages (numpy, scipy, pytest, cython)
        proc_venv = subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(candidate_env_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_venv.returncode != 0:
            return False, f"Failed to create candidate venv: {proc_venv.stderr}"

        shared_sp = cfg.get("shared_site_packages")
        if shared_sp:
            sp_paths = [shared_sp] if isinstance(shared_sp, str) else list(shared_sp)
            cand_sp_dirs = list((candidate_env_dir / "lib").glob("python*/site-packages"))
            if cand_sp_dirs:
                (cand_sp_dirs[0] / "_repoopt_shared.pth").write_text(
                    "\n".join(str(p) for p in sp_paths) + "\n",
                    encoding="utf-8",
                )

        cand_py = str(candidate_env_dir / "bin" / "python")
        has_pip = subprocess.run([cand_py, "-m", "pip", "--version"], capture_output=True, check=False).returncode == 0
        if has_pip:
            cmd = [
                cand_py,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                str(workspace_dir),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "--python",
                cand_py,
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--no-index",
                str(workspace_dir),
            ]
        proc = subprocess.run(cmd, cwd=str(workspace_dir), capture_output=True, text=True, check=False, env=env)
        if proc.returncode != 0:
            return False, f"Python candidate build failed:\n{proc.stdout}\n{proc.stderr}"
        return True, ""

    if language == "r":
        if candidate_env_dir.exists():
            shutil.rmtree(candidate_env_dir)
        candidate_env_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["R", "CMD", "INSTALL", "--no-multiarch", f"--library={candidate_env_dir}", str(workspace_dir)],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            return False, f"R CMD INSTALL failed:\n{proc.stdout}\n{proc.stderr}"
        return True, ""

    return False, f"Unsupported language: {language!r}"


def _run_unit_tests(
    cfg: dict[str, Any],
    *,
    workspace_dir: Path,
    candidate_env_dir: Path,
) -> tuple[bool, str]:
    language = cfg["language"]
    env = _single_thread_env(os.environ, bool(cfg.get("allow_multithread", False)))
    if language == "python":
        cand_py = str(candidate_env_dir / "bin" / "python")
        proc = subprocess.run(
            [cand_py, "-m", "pytest", "-q"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            return False, f"pytest failed:\n{proc.stdout}\n{proc.stderr}"
        return True, proc.stdout.strip()

    if language == "r":
        existing = env.get("R_LIBS_USER", "")
        env["R_LIBS_USER"] = f"{candidate_env_dir}:{existing}" if existing else str(candidate_env_dir)
        proc = subprocess.run(
            [
                "Rscript",
                "--vanilla",
                "-e",
                f"res <- testthat::test_local({json.dumps(str(workspace_dir))}, reporter='summary', stop_on_failure=TRUE)",
            ],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            return False, f"testthat failed:\n{proc.stdout}\n{proc.stderr}"
        return True, proc.stdout.strip()

    return False, f"Unsupported language: {language!r}"


def _paired_summary(baseline_ms: list[float], candidate_ms: list[float], seed: int = 0) -> dict[str, Any]:
    ratios = [b / c for b, c in zip(baseline_ms, candidate_ms)]
    med_b = statistics.median(baseline_ms)
    med_c = statistics.median(candidate_ms)
    speedup = statistics.median(ratios)
    n = len(ratios)
    if n == 1:
        ci_lo = speedup
        ci_hi = speedup
    else:
        rng = random.Random(seed)
        n_boot = 2000
        boot = sorted(statistics.median([ratios[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
        ci_lo = boot[max(0, int(math.floor(0.025 * (n_boot - 1))))]
        ci_hi = boot[min(n_boot - 1, int(math.ceil(0.975 * (n_boot - 1))))]

    def _cv(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        m = statistics.fmean(vals)
        return (statistics.stdev(vals) / m) if m > 0 else 0.0

    return {
        "n_pairs": n,
        "baseline_ms": [round(x, 4) for x in baseline_ms],
        "candidate_ms": [round(x, 4) for x in candidate_ms],
        "median_baseline_ms": round(med_b, 4),
        "median_candidate_ms": round(med_c, 4),
        "speedup": round(speedup, 6),
        "ci95_low": round(ci_lo, 6),
        "ci95_high": round(ci_hi, 6),
        "baseline_cv": round(_cv(baseline_ms), 6),
        "candidate_cv": round(_cv(candidate_ms), 6),
    }


def run_benchmark(
    cfg: dict[str, Any],
    *,
    suite: str,
    workspace_dir: Path,
    warmup_override: int | None = None,
    pairs_override: int | None = None,
) -> dict[str, Any]:
    rtol = float(cfg.get("rtol", 1e-10))
    atol = float(cfg.get("atol", 1e-10))
    warmups = warmup_override if warmup_override is not None else int(cfg.get("warmup_runs", 1))
    n_pairs = pairs_override if pairs_override is not None else int(cfg.get("timed_pairs", 5))

    drivers: list[tuple[str, str]] = []
    if suite in ("dev", "all"):
        drivers.append(("dev", str(cfg["dev_driver"])))
    if suite in ("holdout", "all"):
        holdout = cfg.get("holdout_driver")
        if holdout and Path(holdout).exists():
            drivers.append(("holdout", str(holdout)))
        elif suite == "holdout":
            return {
                "build_ok": False,
                "tests_ok": False,
                "correct": False,
                "reason": f"Holdout driver not found: {holdout}",
            }

    with tempfile.TemporaryDirectory(prefix="repoopt-cand-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cand_env_dir = tmp_path / "cand_env"

        build_ok, build_msg = _build_candidate(cfg, workspace_dir=workspace_dir, candidate_env_dir=cand_env_dir)
        if not build_ok:
            return {
                "build_ok": False,
                "tests_ok": False,
                "correct": False,
                "reason": build_msg,
            }

        tests_ok, test_msg = _run_unit_tests(cfg, workspace_dir=workspace_dir, candidate_env_dir=cand_env_dir)
        if not tests_ok:
            return {
                "build_ok": True,
                "tests_ok": False,
                "correct": False,
                "reason": test_msg,
            }

        max_abs_err = 0.0
        max_rel_err = 0.0
        suite_checks: list[dict[str, Any]] = []

        # Accuracy comparison across requested drivers
        for driver_name, driver_path in drivers:
            base_out = tmp_path / f"{driver_name}_base.json"
            cand_out = tmp_path / f"{driver_name}_cand.json"
            for role, out_path in (("baseline", base_out), ("candidate", cand_out)):
                cmd, env = _role_exec_cmd(
                    cfg,
                    role=role,
                    driver_path=driver_path,
                    extra_args=["--phase", "accuracy", "--out", str(out_path)],
                    candidate_env_dir=cand_env_dir,
                )
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
                if proc.returncode != 0 or not out_path.exists():
                    return {
                        "build_ok": True,
                        "tests_ok": True,
                        "correct": False,
                        "reason": f"{driver_name} accuracy failed for {role}:\n{proc.stdout}\n{proc.stderr}",
                    }

            base_data = _decode_special_floats(json.loads(base_out.read_text(encoding="utf-8")))
            cand_data = _decode_special_floats(json.loads(cand_out.read_text(encoding="utf-8")))
            ok, abs_e, rel_e, err_msg = _compare_numeric_trees(
                base_data.get("outputs"),
                cand_data.get("outputs"),
                rtol=rtol,
                atol=atol,
                path=driver_name,
            )
            max_abs_err = max(max_abs_err, abs_e)
            max_rel_err = max(max_rel_err, rel_e)
            if not ok:
                return {
                    "build_ok": True,
                    "tests_ok": True,
                    "correct": False,
                    "max_abs_err": max_abs_err,
                    "max_rel_err": max_rel_err,
                    "reason": err_msg,
                }
            if cand_data.get("invariants_ok") is False:
                return {
                    "build_ok": True,
                    "tests_ok": True,
                    "correct": False,
                    "max_abs_err": max_abs_err,
                    "max_rel_err": max_rel_err,
                    "reason": f"{driver_name} candidate failed scientific invariants: {cand_data.get('invariant_msg', '')}",
                }
            suite_checks.append({"suite": driver_name, "cases": len(base_data.get("outputs", {}))})

        # Paired interleaved A/B timing on primary timing driver (holdout if suite=='all', else dev)
        timing_driver_name, timing_driver_path = drivers[-1]
        base_ms: list[float] = []
        cand_ms: list[float] = []
        for idx in range(warmups + n_pairs):
            order = ("baseline", "candidate") if idx % 2 == 0 else ("candidate", "baseline")
            pair_times: dict[str, float] = {}
            for role in order:
                t_out = tmp_path / f"time_{idx}_{role}.json"
                cmd, env = _role_exec_cmd(
                    cfg,
                    role=role,
                    driver_path=timing_driver_path,
                    extra_args=["--phase", "bench", "--out", str(t_out)],
                    candidate_env_dir=cand_env_dir,
                )
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
                if proc.returncode != 0 or not t_out.exists():
                    return {
                        "build_ok": True,
                        "tests_ok": True,
                        "correct": False,
                        "reason": f"Timing failed for {role} on {timing_driver_name}:\n{proc.stdout}\n{proc.stderr}",
                    }
                tdata = json.loads(t_out.read_text(encoding="utf-8"))
                pair_times[role] = float(tdata["elapsed_ms"])
            if idx >= warmups:
                base_ms.append(pair_times["baseline"])
                cand_ms.append(pair_times["candidate"])

        timing = _paired_summary(base_ms, cand_ms)
        return {
            "build_ok": True,
            "tests_ok": True,
            "correct": True,
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "timing_suite": timing_driver_name,
            "suite_checks": suite_checks,
            "timing": timing,
            "reason": f"OK: {timing['speedup']:.3f}x speedup (CI95 [{timing['ci95_low']:.3f}, {timing['ci95_high']:.3f}])",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RepoOpt paired benchmark tool")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suite", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    ws = (args.workspace or Path(cfg.get("workspace_dir", "/workspace"))).resolve()
    result = run_benchmark(
        cfg,
        suite=args.suite,
        workspace_dir=ws,
        warmup_override=args.warmup,
        pairs_override=args.pairs,
    )
    rendered = json.dumps(result, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result.get("correct") else 1


if __name__ == "__main__":
    raise SystemExit(main())
