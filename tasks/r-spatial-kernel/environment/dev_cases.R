#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
phase <- NULL
out_path <- NULL
i <- 1L
while (i <= length(args)) {
  if (args[i] == "--phase" && i < length(args)) {
    phase <- args[i + 1L]
    i <- i + 2L
  } else if (args[i] == "--out" && i < length(args)) {
    out_path <- args[i + 1L]
    i <- i + 2L
  } else {
    i <- i + 1L
  }
}
if (is.null(phase) || is.null(out_path)) {
  stop("Usage: dev_cases.R --phase accuracy|bench --out <path>")
}

library(spatialkern)
library(jsonlite)

if (phase == "accuracy") {
  set.seed(101)
  c1_coords <- matrix(runif(45 * 3, min = 0.0, max = 5.0), nrow = 45, ncol = 3)
  c1_vals <- sin(c1_coords[, 1]) + 0.3 * cos(c1_coords[, 2]) - 0.2 * c1_coords[, 3]
  c1_tgt <- matrix(runif(20 * 3, min = 0.0, max = 5.0), nrow = 20, ncol = 3)
  r1 <- spatial_predict_loo(c1_coords, c1_vals, c1_tgt, length_scales = c(0.6, 1.4, 0.9), nugget = 1e-3)

  set.seed(102)
  c2_coords <- matrix(runif(24 * 2, min = -2.0, max = 2.0), nrow = 24, ncol = 2)
  c2_vals <- c2_coords[, 1]^2 - 0.5 * c2_coords[, 2]
  c2_tgt <- matrix(runif(10 * 2, min = -2.0, max = 2.0), nrow = 10, ncol = 2)
  r2 <- spatial_predict_loo(c2_coords, c2_vals, c2_tgt, length_scales = c(1.2, 0.5), nugget = 1e-2)

  ok <- all(is.finite(r1$predictions)) && all(is.finite(r1$loo_residuals)) &&
        all(is.finite(r2$predictions)) && all(is.finite(r2$loo_residuals))

  payload <- list(
    outputs = list(
      anisotropic_3d = list(predictions = as.numeric(r1$predictions), loo_residuals = as.numeric(r1$loo_residuals)),
      quadratic_2d = list(predictions = as.numeric(r2$predictions), loo_residuals = as.numeric(r2$loo_residuals))
    ),
    invariants_ok = ok,
    invariant_msg = if (ok) "" else "Non-finite predictions or LOO residuals"
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = 14), out_path)
} else if (phase == "bench") {
  set.seed(202)
  base_coords <- matrix(runif(130 * 3, min = 0.0, max = 6.0), nrow = 130, ncol = 3)
  base_vals <- sin(base_coords[, 1]) * cos(base_coords[, 2]) + 0.1 * base_coords[, 3]
  tgt <- matrix(runif(80 * 3, min = 0.0, max = 6.0), nrow = 80, ncol = 3)

  # Distinct fresh perturbations prevent memoization
  coords_list <- lapply(1:2, function(k) base_coords + matrix(rnorm(130 * 3, sd = 1e-3), nrow = 130, ncol = 3))
  vals_list <- lapply(1:2, function(k) base_vals + rnorm(130, sd = 1e-4))

  invisible(spatial_predict_loo(coords_list[[1]], vals_list[[1]], tgt, length_scales = c(0.8, 1.3, 0.7), nugget = 1e-3))

  t0 <- proc.time()[["elapsed"]]
  sink <- 0.0
  for (k in seq_along(coords_list)) {
    res <- spatial_predict_loo(coords_list[[k]], vals_list[[k]], tgt, length_scales = c(0.8, 1.3, 0.7), nugget = 1e-3)
    sink <- sink + sum(res$predictions) + sum(res$loo_residuals)
  }
  elapsed_ms <- (proc.time()[["elapsed"]] - t0) * 1000.0
  writeLines(toJSON(list(elapsed_ms = elapsed_ms, sink = sink), auto_unbox = TRUE), out_path)
} else {
  stop(paste("Unknown phase:", phase))
}
