#' Anisotropic Matern 3/2 Spatial Kriging Prediction and Leave-One-Out Residuals
#'
#' @param coords Numeric matrix (N x D) of training coordinates.
#' @param values Numeric vector of length N of training responses.
#' @param target_coords Numeric matrix (M x D) of query coordinates.
#' @param length_scales Numeric vector of length D of positive anisotropic length scales.
#' @param nugget Positive scalar nugget regularization added to training diagonal.
#' @return Named list with `predictions` (length M) and `loo_residuals` (length N).
#' @export
spatial_predict_loo <- function(coords, values, target_coords, length_scales, nugget = 1e-4) {
  coords <- as.matrix(coords)
  target_coords <- as.matrix(target_coords)
  values <- as.numeric(values)
  length_scales <- as.numeric(length_scales)

  n <- nrow(coords)
  d <- ncol(coords)
  m <- nrow(target_coords)

  if (length(values) != n) {
    stop("length(values) must match nrow(coords)")
  }
  if (ncol(target_coords) != d) {
    stop("ncol(target_coords) must match ncol(coords)")
  }
  if (length(length_scales) != d || any(length_scales <= 0)) {
    stop("length_scales must be a positive vector of length ncol(coords)")
  }
  if (!is.finite(nugget) || nugget <= 0) {
    stop("nugget must be strictly positive")
  }

  sqrt3 <- sqrt(3.0)

  # Baseline pairwise training covariance matrix via explicit loops
  k_train <- matrix(0.0, nrow = n, ncol = n)
  for (i in seq_len(n)) {
    k_train[i, i] <- 1.0 + nugget
    if (i < n) {
      for (j in (i + 1L):n) {
        dist_sq <- 0.0
        for (dim_idx in seq_len(d)) {
          diff <- (coords[i, dim_idx] - coords[j, dim_idx]) / length_scales[dim_idx]
          dist_sq <- dist_sq + diff * diff
        }
        r <- sqrt3 * sqrt(dist_sq)
        val <- (1.0 + r) * exp(-r)
        k_train[i, j] <- val
        k_train[j, i] <- val
      }
    }
  }

  # Baseline cross-covariance matrix
  k_cross <- matrix(0.0, nrow = m, ncol = n)
  for (i in seq_len(m)) {
    for (j in seq_len(n)) {
      dist_sq <- 0.0
      for (dim_idx in seq_len(d)) {
        diff <- (target_coords[i, dim_idx] - coords[j, dim_idx]) / length_scales[dim_idx]
        dist_sq <- dist_sq + diff * diff
      }
      r <- sqrt3 * sqrt(dist_sq)
      k_cross[i, j] <- (1.0 + r) * exp(-r)
    }
  }

  alpha <- solve(k_train, values)
  predictions <- as.numeric(k_cross %*% alpha)

  # Naive leave-one-out residuals by solving each (N-1)x(N-1) subproblem explicitly
  loo_residuals <- numeric(n)
  for (i in seq_len(n)) {
    idx <- seq_len(n)[-i]
    sub_alpha <- solve(k_train[idx, idx, drop = FALSE], values[idx])
    pred_i <- sum(k_train[i, idx] * sub_alpha)
    loo_residuals[i] <- values[i] - pred_i
  }

  list(
    predictions = predictions,
    loo_residuals = loo_residuals
  )
}
