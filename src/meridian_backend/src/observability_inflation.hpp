#pragma once

#include <Eigen/Core>
#include <array>

#include "meridian/common/gaussian.hpp"
#include "meridian/common/observability.hpp"

namespace meridian::backend {

// Result of inflate_by_observability. `cov` keeps the input's rotation-first
// [rx,ry,rz,tx,ty,tz] axis order and feeds a noise model with no reordering; `lambda` holds
// the per-axis variance multipliers in that same rotation-first order.
struct InflationResult {
  Eigen::Matrix<double, 6, 6> cov;
  std::array<double, 6> lambda;
  bool any_locked = false;
};

// Inflates a rotation-first 6x6 covariance along the directions the front-end reports as
// poorly observed, so a degenerate axis cannot over-constrain the graph.
//
// Per axis: lambda_k = 1 + (rho_max - 1) * (1 - s_k)^gamma, with s_k clamped to [0,1]. When
// degenerate_lock is set and s_k < degenerate_thresh, the axis locks to lambda_k = rho_max
// and any_locked is raised. The translation-first scores (and, when present, the eigvec
// columns they refer to) are permuted onto the rotation-first axes of `cov`, and the
// inflation is applied in that basis:
//   Sigma' = R6 * L^{1/2} * (R6^T * Sigma * R6) * L^{1/2} * R6^T
// where R6 is the permuted eigvec matrix (identity when obs.eigvecs is unset) and L^{1/2}
// is diag(sqrt(lambda)).
//
// Preconditions: cov.M holds a covariance (not an information matrix); obs.frame is Body —
// any rotation between the report frame and the factor frame is resolved upstream and none
// is applied here; eigvec columns, when present, are orthonormal.
InflationResult inflate_by_observability(const GaussianBlock<6>& cov,
                                         const ObservabilityReport& obs, double rho_max,
                                         double gamma, double degenerate_thresh,
                                         bool degenerate_lock);

}  // namespace meridian::backend
