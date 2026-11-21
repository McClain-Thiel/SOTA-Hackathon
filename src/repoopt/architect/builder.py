"""BenchArchitect: repo preparation, squashing, scaffolding, and preflight gating."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from repoopt.architect.prompts import build_architect_instruction
from repoopt.bench.preflight_gate import PreflightReport, sync_task_harness_assets, verify_task_preflight
from repoopt.store.base import ArtifactStore

SHARED_BASE_DOCKERFILE_HEADER = """FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \\
    NO_PROXY=host.docker.internal,localhost,127.0.0.1 \\
    no_proxy=host.docker.internal,localhost,127.0.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    curl \\
    git \\
    gcc \\
    g++ \\
    libopenblas-dev \\
    python3 \\
    python3-dev \\
    python3-pip \\
    python3-venv \\
    python3-setuptools \\
    python3-wheel \\
    python3-numpy \\
    python3-scipy \\
    python3-pytest \\
    cython3 \\
    r-base-core \\
    r-base-dev \\
    r-cran-rcpp \\
    r-cran-testthat \\
    r-cran-jsonlite \\
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \\
    && . /root/.nvm/nvm.sh \\
    && nvm install 22 \\
    && nvm alias default 22 \\
    && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@0.85.1 \\
    && npm cache clean --force \\
    && pi --version
"""


def clone_and_squash_repo(
    source_url_or_path: str,
    commit: str,
    dest_dir: Path,
) -> Path:
    """Clone a repository at a pinned commit and strip .git history so Optimizer cannot inspect future/past PRs."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", source_url_or_path, str(dest_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(dest_dir), "checkout", "--detach", commit],
        check=True,
        capture_output=True,
        text=True,
    )
    git_dir = dest_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
    return dest_dir


def scaffold_architect_workspace(
    *,
    task_id: str,
    language: Literal["python", "r"],
    source_repo_dir: Path,
    output_task_dir: Path,
) -> Path:
    """Create a starter Harbor task skeleton with synced verifier/bench harness ready for BenchArchitect or customization."""
    env_dir = output_task_dir / "environment"
    tests_dir = output_task_dir / "tests"
    sol_dir = output_task_dir / "solution"
    repo_dest = env_dir / "repo"

    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    sol_dir.mkdir(parents=True, exist_ok=True)

    if repo_dest.exists():
        shutil.rmtree(repo_dest)
    shutil.copytree(source_repo_dir, repo_dest, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))

    sync_task_harness_assets(output_task_dir)

    (output_task_dir / "task.toml").write_text(
        f'''schema_version = "1.4"

[task]
name = "repoopt/{task_id}"
description = "Optimize {task_id} ({language}) while preserving numerical accuracy."
keywords = ["scientific-computing", "{language}", "optimization"]

[agent]
timeout_sec = 900.0
network_mode = "public"

[verifier]
timeout_sec = 300.0
environment_mode = "shared"

[environment]
cpus = 2
memory_mb = 4096
workdir = "/workspace"
network_mode = "public"
''',
        encoding="utf-8",
    )

    test_sh = tests_dir / "test.sh"
    test_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p /logs/verifier\npython3 -I /tests/run_verifier.py\n", encoding="utf-8")
    test_sh.chmod(0o755)

    ext = "py" if language == "python" else "R"
    bench_cfg: dict[str, object] = {
        "task_id": task_id,
        "language": language,
        "workspace_dir": "/workspace",
        "baseline_dir": "/opt/baseline_src",
        "dev_driver": f"/opt/bench/dev_cases.{ext}",
        "holdout_driver": f"/tests/holdout_cases.{ext}",
        "warmup_runs": 1,
        "timed_pairs": 5,
        "rtol": 1e-10 if language == "python" else 1e-9,
        "atol": 1e-10 if language == "python" else 1e-9,
        "allow_multithread": False,
    }
    if language == "python":
        bench_cfg["baseline_python"] = "/opt/baseline_venv/bin/python"
        baseline_install_block = """COPY repo /opt/baseline_src
RUN python3 -m venv --system-site-packages /opt/baseline_venv \\
    && /opt/baseline_venv/bin/pip install --no-deps --no-build-isolation --no-index /opt/baseline_src
"""
    else:
        bench_cfg["baseline_r_lib"] = "/opt/baseline_rlib"
        baseline_install_block = """COPY repo /opt/baseline_src
RUN mkdir -p /opt/baseline_rlib \\
    && R CMD INSTALL --no-multiarch --library=/opt/baseline_rlib /opt/baseline_src
"""

    (env_dir / "bench_config.json").write_text(json.dumps(bench_cfg, indent=2) + "\n", encoding="utf-8")
    (env_dir / "Dockerfile").write_text(
        f"""{SHARED_BASE_DOCKERFILE_HEADER}
{baseline_install_block}
COPY bench_config.json /opt/bench/bench_config.json
COPY dev_cases.{ext} /opt/bench/dev_cases.{ext}
COPY repoopt_bench.py /usr/local/bin/repoopt-bench
RUN chmod 0755 /usr/local/bin/repoopt-bench

RUN cp -a /opt/baseline_src /workspace \\
    && git config --global user.email "repoopt@local" \\
    && git config --global user.name "RepoOpt" \\
    && git -C /workspace init \\
    && git -C /workspace add -A \\
    && git -C /workspace commit -m "baseline"

WORKDIR /workspace
CMD ["sleep", "infinity"]
""",
        encoding="utf-8",
    )

    architect_prompt_file = output_task_dir / "ARCHITECT_PROMPT.md"
    architect_prompt_file.write_text(
        build_architect_instruction(
            repo_name=source_repo_dir.name,
            language=language,
            task_id=task_id,
            output_dir=str(output_task_dir),
        ),
        encoding="utf-8",
    )
    return output_task_dir


def onboard_task_with_preflight(
    task_dir: Path,
    *,
    store: ArtifactStore | None = None,
    experiment_id: str | None = None,
) -> PreflightReport:
    """Validate an authored Harbor task directory through the automated Docker preflight gate."""
    return verify_task_preflight(
        task_dir,
        store=store,
        experiment_id=experiment_id,
    )
