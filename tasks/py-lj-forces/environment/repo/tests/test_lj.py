"""Unit tests for ljforces."""

import numpy as np

from ljforces import compute_lj_system


def test_dimer_analytical() -> None:
    pos = np.array([[0.0, 0.0, 0.0], [1.122462048309373, 0.0, 0.0]], dtype=np.float64)
    res = compute_lj_system(pos, box_length=10.0, epsilon=1.0, sigma=1.0, r_cut=3.0)
    # At r = 2^(1/6), potential energy is -epsilon and force is zero
    assert np.isclose(res["potential_energy"], -1.0, atol=1e-12)
    assert np.allclose(res["forces"], 0.0, atol=1e-12)


def test_periodic_wrap_and_newtons_third_law() -> None:
    rng = np.random.default_rng(7)
    box = 8.0
    pos = rng.uniform(0.0, box, size=(24, 3))
    res = compute_lj_system(pos, box_length=box, epsilon=1.2, sigma=0.95, r_cut=2.5)
    assert np.allclose(res["forces"].sum(axis=0), 0.0, atol=1e-11)

    # Shifting all particles periodically must leave forces and energy invariant
    pos_shifted = np.mod(pos + np.array([3.7, 5.1, 1.9]), box)
    res2 = compute_lj_system(pos_shifted, box_length=box, epsilon=1.2, sigma=0.95, r_cut=2.5)
    assert np.allclose(res["forces"], res2["forces"], atol=1e-11)
    assert np.isclose(res["potential_energy"], res2["potential_energy"], atol=1e-11)
    assert np.isclose(res["virial"], res2["virial"], atol=1e-11)


def test_force_matches_finite_difference_energy() -> None:
    pos = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.15, 1.2, 1.1],
            [1.3, 2.3, 1.05],
        ],
        dtype=np.float64,
    )
    res = compute_lj_system(pos, box_length=10.0, epsilon=1.3, sigma=1.0, r_cut=3.5)
    eps_fd = 1e-6
    pos_p = pos.copy()
    pos_m = pos.copy()
    pos_p[0, 0] += eps_fd
    pos_m[0, 0] -= eps_fd
    u_p = compute_lj_system(pos_p, box_length=10.0, epsilon=1.3, sigma=1.0, r_cut=3.5)["potential_energy"]
    u_m = compute_lj_system(pos_m, box_length=10.0, epsilon=1.3, sigma=1.0, r_cut=3.5)["potential_energy"]
    f_fd = -(u_p - u_m) / (2.0 * eps_fd)
    assert np.isclose(res["forces"][0, 0], f_fd, rtol=1e-6, atol=1e-7)
