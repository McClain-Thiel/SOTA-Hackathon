library(testthat)
library(spatialkern)

test_that("spatial_predict_loo matches closed-form 2-point kriging and LOO", {
  coords <- matrix(c(0.0, 1.0, 0.0, 0.0), nrow = 2, ncol = 2)
  values <- c(2.0, 4.0)
  target <- matrix(c(0.0, 0.0), nrow = 1, ncol = 2)
  res <- spatial_predict_loo(coords, values, target, length_scales = c(1.0, 1.0), nugget = 1e-6)
  expect_length(res$predictions, 1)
  expect_length(res$loo_residuals, 2)
  expect_equal(res$predictions[1], 2.0, tolerance = 1e-4)
  expect_true(all(is.finite(res$loo_residuals)))
})

test_that("spatial_predict_loo respects anisotropic scaling and input validation", {
  set.seed(11)
  coords <- matrix(runif(30), nrow = 10, ncol = 3)
  values <- sin(coords[, 1]) + 0.5 * coords[, 2]
  target <- matrix(runif(12), nrow = 4, ncol = 3)
  res <- spatial_predict_loo(coords, values, target, length_scales = c(0.4, 1.2, 0.8), nugget = 1e-3)
  expect_equal(length(res$predictions), 4)
  expect_equal(length(res$loo_residuals), 10)
  expect_error(spatial_predict_loo(coords, values, target, length_scales = c(-1, 1, 1)))
})
