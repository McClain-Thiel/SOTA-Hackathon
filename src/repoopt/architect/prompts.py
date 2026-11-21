"""System and task prompts for Agent 1 (BenchArchitect) task generation."""

from __future__ import annotations


def build_architect_instruction(
    *,
    repo_name: str,
    language: str,
    task_id: str,
    output_dir: str,
) -> str:
    """Build the Harbor task prompt for BenchArchitect (Agent 1)."""
    lang_notes = (
        "Use `python3` / `pytest` / `pip install --no-deps --no-build-isolation --no-index`. "
        "Emit `dev_cases.py` and `holdout_cases.py` with `--phase accuracy|bench --out <json>`."
        if language == "python"
        else "Use `R` / `Rscript --vanilla` / `testthat` / `jsonlite`. "
        "Emit `dev_cases.R` and `holdout_cases.R` with `--phase accuracy|bench --out <json>`."
    )
    return f"""You are **BenchArchitect** (Agent 1). Your job is to convert the scientific repository `{repo_name}` ({language}) located in `/workspace/repo` into a verified, leak-free Harbor optimization task at `{output_dir}` (`task_id={task_id}`).

## Required Task Layout at `{output_dir}`
- `task.toml`: Valid Harbor schema 1.4 (`[task].name = "repoopt/{task_id}"`).
- `instruction.md`: Clear optimization objective describing the target function(s), input/output contract, tolerances, and mentioning `repoopt-bench` + installed native compilers (`gcc`, `g++`, `cython3`, `Rcpp`, `OpenBLAS`).
- `environment/`:
  - `Dockerfile`: Start with the standard RepoOpt shared `ubuntu:24.04` + toolchain + `pi` base layer, install baseline into `/opt/baseline_*` and `/opt/baseline_src`, copy `bench_config.json`, `dev_cases.*`, `repoopt_bench.py` -> `/usr/local/bin/repoopt-bench`, and initialize `/workspace` as a single squashed commit (`git init && git add -A && git commit -m "baseline"`).
  - `repo/`: Clean squashed source tree (NO `.git` history!).
  - `dev_cases.*` & `bench_config.json`: Public dev workloads checking accuracy + invariants + regenerated-input timing (`~150-400ms` baseline).
- `tests/`:
  - `test.sh` & `run_verifier.py`: Clean-room rebuild + dev + holdout evaluation writing `/logs/verifier/reward.txt` & `reward.json`.
  - `holdout_cases.*`: Private unseen seeds, larger/smaller shapes, boundary/edge cases (NEVER copied into container image!).
  - `mutation_bad.patch`: Synthetic subtle accuracy bug that MUST be caught as `INCORRECT` by the verifier.

## Language Specifics ({language})
{lang_notes}

## Mandatory Verification Before Finishing
Run `repoopt preflight-verify {output_dir}` (or test `repoopt-bench` + baseline + mutation check) and confirm:
1. Baseline no-op scores `CORRECT_NO_SPEEDUP` (`reward == 0.0`).
2. `mutation_bad.patch` scores `INCORRECT` (`reward == 0.0`).
3. Timing CV `< 0.15`.
"""
