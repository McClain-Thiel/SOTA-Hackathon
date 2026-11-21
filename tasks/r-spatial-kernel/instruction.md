# Optimize `spatialkern::spatial_predict_loo`

Accelerate the anisotropic Matérn ($\nu = 3/2$) Gaussian process spatial prediction and leave-one-out cross-validation kernel in `/workspace` (`R/matern_krige.R`) while preserving full double-precision numerical accuracy and the exact public R API.

## Environment & Tooling
- Working directory: `/workspace` (initialized as a clean git repository).
- Run `Rscript --vanilla -e "testthat::test_local('/workspace')"` to run unit tests.
- Run `repoopt-bench` at any time via `bash` to measure numerical accuracy and paired speedup against `/opt/baseline_rlib` on dev workloads.
- `R`, `Rcpp`, `g++`, `gcc`, and `OpenBLAS` are preinstalled. You may optimize using vectorized R/BLAS linear algebra, mathematical identities, or compiled C/C++ via `Rcpp` (`src/*.cpp` + `DESCRIPTION` / `NAMESPACE`), ensuring `R CMD INSTALL .` builds cleanly from source.

## Constraints
- Preserve the exact signature and return structure (`list(predictions = ..., loo_residuals = ...)`) of `spatial_predict_loo(coords, values, target_coords, length_scales, nugget = 1e-4)`.
- Maintain numerical agreement (`rtol <= 1e-9`, `atol <= 1e-9`) across arbitrary training sizes, anisotropic length scales, and nugget values.
- Do not hardcode inputs or drop positive-definiteness/input validation checks.
