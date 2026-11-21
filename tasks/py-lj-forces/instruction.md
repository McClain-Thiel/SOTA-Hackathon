# Optimize `ljforces.compute_lj_system`

Accelerate the 3D periodic minimum-image Lennard-Jones 12-6 force, potential energy, and virial kernel in `/workspace` (`ljforces/core.py`) while preserving full `float64` numerical accuracy and all existing public function signatures.

## Environment & Tooling
- Working directory: `/workspace` (initialized as a clean git repository).
- Run `pytest` to execute existing unit tests.
- Run `repoopt-bench` at any time via `bash` to measure numerical accuracy and paired speedup against the read-only `/opt/baseline` reference implementation on dev workloads.
- Native compilers and Python tools (`gcc`, `g++`, `cython3`, `numpy`, `scipy`) are preinstalled in the container. You may optimize algorithmically, vectorize with NumPy, or compile C/Cython extensions (remember to ensure `pip install --no-deps --no-build-isolation --no-index .` builds cleanly from source).

## Constraints
- Preserve the exact API and return dictionary structure (`forces`, `potential_energy`, `virial`) of `ljforces.compute_lj_system`.
- Maintain strict IEEE `float64` numerical agreement (`rtol <= 1e-10`, `atol <= 1e-10`) across arbitrary box sizes, cutoffs, particle counts, and periodic wraps.
- Do not hardcode inputs or drop cutoff/minimum-image checks.
