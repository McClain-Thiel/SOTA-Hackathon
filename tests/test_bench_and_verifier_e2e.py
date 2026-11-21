"""Host-level end-to-end verification of repoopt_bench.py and run_verifier.py against tasks/py-lj-forces."""

from __future__ import annotations

import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).resolve().parent.parent / "tasks" / "py-lj-forces"
REPOOPT_BENCH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "repoopt"
    / "bench"
    / "harness_assets"
    / "repoopt_bench.py"
)
RUN_VERIFIER = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "repoopt"
    / "bench"
    / "harness_assets"
    / "run_verifier.py"
)


@pytest.fixture(scope="module")
def lj_bench_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("lj-e2e")
    baseline_src = root / "baseline_src"
    shutil.copytree(TASK_DIR / "environment" / "repo", baseline_src)

    baseline_venv = root / "baseline_venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(baseline_venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    shared_sp = site.getsitepackages()
    base_sp = next((baseline_venv / "lib").glob("python*/site-packages"))
    (base_sp / "_repoopt_shared.pth").write_text("\n".join(shared_sp) + "\n", encoding="utf-8")

    base_py = baseline_venv / "bin" / "python"
    has_pip = subprocess.run([str(base_py), "-m", "pip", "--version"], capture_output=True, check=False).returncode == 0
    if has_pip:
        pip_cmd = [str(base_py), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index", str(baseline_src)]
    else:
        pip_cmd = [sys.executable, "-m", "pip", "--python", str(base_py), "install", "--no-deps", "--no-build-isolation", "--no-index", str(baseline_src)]
    subprocess.run(pip_cmd, check=True, capture_output=True, text=True)

    bench_cfg = root / "bench_config.json"
    bench_cfg.write_text(
        json.dumps(
            {
                "task_id": "py-lj-forces",
                "language": "python",
                "workspace_dir": str(root / "workspace"),
                "baseline_dir": str(baseline_src),
                "baseline_python": str(base_py),
                "shared_site_packages": shared_sp,
                "dev_driver": str(TASK_DIR / "environment" / "dev_cases.py"),
                "holdout_driver": str(TASK_DIR / "tests" / "holdout_cases.py"),
                "warmup_runs": 1,
                "timed_pairs": 3,
                "rtol": 1e-10,
                "atol": 1e-10,
                "allow_multithread": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "baseline_src": baseline_src,
        "bench_cfg": bench_cfg,
    }


def _fresh_workspace(baseline_src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(baseline_src, dest)
    subprocess.run(["git", "-C", str(dest), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.email", "test@repoopt"], check=True)
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(dest), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(dest), "commit", "-q", "-m", "baseline"], check=True)
    return dest


def _run_verifier(
    fixture: dict[str, Path],
    ws: Path,
    logs_dir: Path,
) -> dict:
    env = dict(os.environ)
    env["VERIFIER_LOGS_DIR"] = str(logs_dir)
    env["REPOOPT_WORKSPACE_DIR"] = str(ws)
    env["REPOOPT_BASELINE_SRC"] = str(fixture["baseline_src"])
    env["REPOOPT_BENCH_BIN"] = str(REPOOPT_BENCH)
    env["REPOOPT_BENCH_CONFIG"] = str(fixture["bench_cfg"])
    proc = subprocess.run(
        [sys.executable, "-I", str(RUN_VERIFIER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"run_verifier failed: {proc.stdout}\n{proc.stderr}"
    reward_txt = float((logs_dir / "reward.txt").read_text(encoding="utf-8").strip())
    reward_json = json.loads((logs_dir / "reward.json").read_text(encoding="utf-8"))
    assert reward_txt == pytest.approx(reward_json["reward"], abs=1e-9)
    return reward_json


def test_in_episode_repoopt_bench_dev(lj_bench_fixture: dict[str, Path], tmp_path: Path) -> None:
    ws = _fresh_workspace(lj_bench_fixture["baseline_src"], tmp_path / "ws_dev")
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(REPOOPT_BENCH),
            "--config",
            str(lj_bench_fixture["bench_cfg"]),
            "--suite",
            "dev",
            "--workspace",
            str(ws),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["correct"] is True
    assert data["tests_ok"] is True
    assert data["timing"]["median_baseline_ms"] > 0.0
    assert 0.5 < data["timing"]["speedup"] < 2.0


def test_verifier_noop_baseline_scores_zero_reward(lj_bench_fixture: dict[str, Path], tmp_path: Path) -> None:
    ws = _fresh_workspace(lj_bench_fixture["baseline_src"], tmp_path / "ws_noop")
    res = _run_verifier(lj_bench_fixture, ws, tmp_path / "logs_noop")
    assert res["correct"] is True
    assert res["is_noop_patch"] is True
    assert res["label"] == "CORRECT_NO_SPEEDUP"
    assert res["reward"] == 0.0


def test_verifier_catches_mutation_bad_patch(lj_bench_fixture: dict[str, Path], tmp_path: Path) -> None:
    ws = _fresh_workspace(lj_bench_fixture["baseline_src"], tmp_path / "ws_mut")
    bad_patch = (TASK_DIR / "tests" / "mutation_bad.patch").read_text(encoding="utf-8")
    subprocess.run(["git", "-C", str(ws), "apply", "-"], input=bad_patch, text=True, check=True)
    res = _run_verifier(lj_bench_fixture, ws, tmp_path / "logs_mut")
    assert res["correct"] is False
    assert res["label"] == "INCORRECT"
    assert res["reward"] == 0.0


def test_verifier_rewards_reference_fast_patch(lj_bench_fixture: dict[str, Path], tmp_path: Path) -> None:
    ws = _fresh_workspace(lj_bench_fixture["baseline_src"], tmp_path / "ws_fast")
    fast_patch = (TASK_DIR / "solution" / "reference_fast.patch").read_text(encoding="utf-8")
    subprocess.run(["git", "-C", str(ws), "apply", "-"], input=fast_patch, text=True, check=True)
    res = _run_verifier(lj_bench_fixture, ws, tmp_path / "logs_fast")
    assert res["correct"] is True
    assert res["is_noop_patch"] is False
    assert res["label"] == "VERIFIED_SPEEDUP"
    assert res["max_abs_err"] < 1e-10
    assert res["timing"]["speedup"] > 10.0
    assert res["reward"] > 3.0


def test_verifier_rejects_binary_file_smuggling(lj_bench_fixture: dict[str, Path], tmp_path: Path) -> None:
    ws = _fresh_workspace(lj_bench_fixture["baseline_src"], tmp_path / "ws_bin")
    (ws / "exploit.bin").write_bytes(b"\x00\xff\xfe\x89PNG\r\n\x1a\n")
    res = _run_verifier(lj_bench_fixture, ws, tmp_path / "logs_bin")
    assert res["correct"] is False
    assert res["label"] == "INCORRECT"
    assert res["reward"] == 0.0
    assert "Rejected binary artifacts" in res["reason"]
