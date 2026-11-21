"""Baseline 3D periodic minimum-image Lennard-Jones 12-6 force, energy, and virial solver."""

from __future__ import annotations

import math
import numpy as np


def compute_lj_system(
    positions: np.ndarray,
    box_length: float,
    epsilon: float = 1.0,
    sigma: float = 1.0,
    r_cut: float = 2.5,
) -> dict[str, np.ndarray | float]:
    """Compute pairwise LJ 12-6 forces, total potential energy, and scalar virial.

    Args:
        positions: Array of shape (N, 3) with particle coordinates.
        box_length: Cubic periodic simulation box side length (> 0).
        epsilon: Well depth parameter.
        sigma: Finite distance at which inter-particle potential is zero.
        r_cut: Spherical cutoff radius.

    Returns:
        Dictionary with:
            - "forces": (N, 3) float64 force array.
            - "potential_energy": float total potential energy.
            - "virial": float scalar virial sum_{i<j} r_ij . f_ij.
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) positions array, got {pos.shape}")
    if box_length <= 0.0:
        raise ValueError("box_length must be positive")
    if r_cut <= 0.0:
        raise ValueError("r_cut must be positive")

    n = pos.shape[0]
    forces = np.zeros((n, 3), dtype=np.float64)
    total_energy = 0.0
    total_virial = 0.0

    r_cut_sq = float(r_cut) * float(r_cut)
    sigma_sq = float(sigma) * float(sigma)
    inv_box = 1.0 / float(box_length)
    four_eps = 4.0 * float(epsilon)
    twenty_four_eps = 24.0 * float(epsilon)

    for i in range(n - 1):
        xi = float(pos[i, 0])
        yi = float(pos[i, 1])
        zi = float(pos[i, 2])
        for j in range(i + 1, n):
            dx = xi - float(pos[j, 0])
            dy = yi - float(pos[j, 1])
            dz = zi - float(pos[j, 2])

            dx -= box_length * round(dx * inv_box)
            dy -= box_length * round(dy * inv_box)
            dz -= box_length * round(dz * inv_box)

            r2 = dx * dx + dy * dy + dz * dz
            if 0.0 < r2 < r_cut_sq:
                sr2 = sigma_sq / r2
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6

                total_energy += four_eps * (sr12 - sr6)
                f_over_r = twenty_four_eps * (2.0 * sr12 - sr6) / r2
                fx = f_over_r * dx
                fy = f_over_r * dy
                fz = f_over_r * dz

                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[i, 2] += fz
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

                total_virial += dx * fx + dy * fy + dz * fz

    return {
        "forces": forces,
        "potential_energy": float(total_energy),
        "virial": float(total_virial),
    }
