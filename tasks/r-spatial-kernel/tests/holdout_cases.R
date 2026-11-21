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
  stop("Usage: holdout_cases.R --phase accuracy|bench --out <path>")
}

library(spatialkern)
library(jsonlite)

if (phase == "accuracy") {
  set.seed(777)
  h1_coords <- matrix(runif(55 * 4, min = -3.0, max = 3.0), nrow = 55, ncol = 4)
  h1_vals <- exp(-0.2 * h1_coords[, 1]^2) + 0.4 * h1_coords[, 4]
  h1_tgt <- matrix(runif(25 * 4, min = -3.0, max = 3.0), nrow = 25, ncol = 4)
  r1 <- spatial_predict_loo(h1_coords, h1_vals, h1_tgt, length_scales = c(0.4, 2.1, 0.9, 1.6), nugget = 5e-4)

  # Large-scale coordinates checking absence of catastrophic cancellation
  set.seed(778)
  h2_coords <- 1e4 + matrix(runif(20 * 2, min = 0.0, max = 1.0), nrow = 20, ncol = 2)
  h2_vals <- sin(h2_coords[, 1]) + cos(h2_coords[, 2])
  h2_tgt <- 1e4 + matrix(runif(10 * 2, min = 0.0, max = 1.0), nrow = 10, ncol = 2)
  r2 <- spatial_predict_loo(h2_coords, h2_vals, h2_tgt, length_scales = c(0.3, 0.7), nugget = 1e-2)

  ok <- all(is.finite(r1$predictions)) && all(is.finite(r1$loo_residuals)) &&
        all(is.finite(r2$predictions)) && all(is.finite(r2$loo_residuals))

  payload <- list(
    outputs = list(
      anisotropic_4d = list(predictions = as.numeric(r1$predictions), loo_residuals = as.numeric(r1$loo_residuals)),
      large_offset_2d = list(predictions = as.numeric(r2$predictions), loo_residuals = as.numeric(r2$loo_residuals))
    ),
    invariants_ok = ok,
    invariant_msg = if (ok) "" else "Non-finite predictions or LOO residuals in holdout"
  )
  writeLines(toJSON(payload, auto_unbox = TRUE, digits = 14), out_path)
} else if (phase == "bench") {
  set.seed(888)
  base_coords <- matrix(runif(140 * 3, min = 0.0, max = 8.0), nrow = 140, ncol = 3)
  base_vals <- sin(base_coords[, 1]) + 0.5 * cos(base_coords[, 3])
  tgt <- matrix(runif(90 * 3, min = 0.0, max = 8.0), nrow = 90, ncol = 3)

  coords_list <- lapply(1:2, function(k) base_coords + matrix(rnorm(140 * 3, sd = 1e-3), nrow = 140, ncol = 3))
  vals_list <- lapply(1:2, function(k) base_vals + rnorm(140, sd = 1e-4))

  invisible(spatial_predict_loo(coords_list[[1]], vals_list[[1]], tgt, length_scales = c(0.5, 1.8, 1.1), nugget = 1e-3))

  t0 <- proc.time()[["elapsed"]]
  sink <- 0.0
  for (k in seq_along(coords_list)) {
    res <- spatial_predict_loo(coords_list[[k]], vals_list[[k]], tgt, length_scales = c(0.5, 1.8, 1.1), nugget = 1e-3)
    sink <- sink + sum(res$predictions) + sum(res$loo_residuals)
  }
  elapsed_ms <- (proc.time()[["elapsed"]] - t0) * 1000.0
  writeLines(toJSON(list(elapsed_ms = elapsed_ms, sink = sink), auto_unbox = TRUE), out_path)
} else {
  stop(paste("Unknown phase:", phase))
}
