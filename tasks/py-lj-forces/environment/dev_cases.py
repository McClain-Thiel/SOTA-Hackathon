"""Public dev accuracy and benchmark driver for py-lj-forces."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ljforces import compute_lj_system


def _perturbed_lattice(rng: np.random.Generator, n: int, box_length: float, jitter: float = 0.08) -> np.ndarray:
    side = int(math.ceil(n ** (1.0 / 3.0)))
    grid = np.linspace(0.5 * box_length / side, box_length * (1.0 - 0.5 / side), side)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])[:n]
    return np.mod(pts + rng.normal(0.0, jitter, size=pts.shape), box_length)


def _encode_result(res: dict) -> dict:
    return {
        "forces": np.asarray(res["forces"], dtype=np.float64).tolist(),
        "potential_energy": float(res["potential_energy"]),
        "virial": float(res["virial"]),
    }


def run_accuracy(out_path: Path) -> None:
    rng = np.random.default_rng(101)
    cases = {
        "dense_48": (_perturbed_lattice(rng, 48, 4.8), 4.8, 1.1, 0.95, 2.2),
        "liquid_96": (_perturbed_lattice(rng, 96, 6.4), 6.4, 0.9, 1.05, 2.5),
        "single_1": (np.array([[1.2, 2.3, 3.4]], dtype=np.float64), 5.0, 1.0, 1.0, 2.5),
    }
    outputs = {}
    invariants_ok = True
    invariant_msgs = []
    for name, (pos, box, eps, sig, rcut) in cases.items():
        res = compute_lj_system(pos.copy(), box_length=box, epsilon=eps, sigma=sig, r_cut=rcut)
        f = np.asarray(res["forces"], dtype=np.float64)
        if f.shape != pos.shape or not np.all(np.isfinite(f)):
            invariants_ok = False
            invariant_msgs.append(f"{name}: non-finite or bad force shape {f.shape}")
        net = float(np.max(np.abs(f.sum(axis=0)))) if f.size else 0.0
        scale = max(1.0, float(np.sum(np.abs(f))))
        if net > 1e-11 * scale:
            invariants_ok = False
            invariant_msgs.append(f"{name}: Newton 3rd law violated (max |sum F|={net:.3e}, scale={scale:.3e})")
        outputs[name] = _encode_result(res)

    out_path.write_text(
        json.dumps(
            {
                "outputs": outputs,
                "invariants_ok": invariants_ok,
                "invariant_msg": "; ".join(invariant_msgs),
            }
        ),
        encoding="utf-8",
    )


def run_bench(out_path: Path) -> None:
    rng = np.random.default_rng(202)
    base_pos = _perturbed_lattice(rng, 340, 8.2, jitter=0.1)
    # Distinct fresh inputs per call prevent input-pointer memoization
    inputs = [np.mod(base_pos + rng.normal(0.0, 1e-3, size=base_pos.shape), 8.2) for _ in range(4)]

    # One quick warmup inside subprocess
    _ = compute_lj_system(inputs[0].copy(), box_length=8.2, epsilon=1.15, sigma=1.0, r_cut=2.5)

    t0 = time.perf_counter()
    sink = 0.0
    for arr in inputs:
        res = compute_lj_system(arr, box_length=8.2, epsilon=1.15, sigma=1.0, r_cut=2.5)
        sink += float(res["potential_energy"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    out_path.write_text(json.dumps({"elapsed_ms": elapsed_ms, "sink": sink}), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("accuracy", "bench"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "accuracy":
        run_accuracy(args.out)
    else:
        run_bench(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
