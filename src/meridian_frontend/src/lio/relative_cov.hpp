#pragma once

#include <Eigen/Core>

#include "meridian/common/pose.hpp"

namespace meridian::lio {

// Accumulate one more registration step into the running relative covariance:
// transport `cov_rel` across the step `delta` with the right-perturbation adjoint
// and add the step's own covariance. All blocks are translation-first [rho; phi]
// in the right/body tangent.
inline Eigen::Matrix<double, 6, 6> composeRelativeCov(const Eigen::Matrix<double, 6, 6>& cov_rel,
                                                      const Pose& delta,
                                                      const Eigen::Matrix<double, 6, 6>& cov_step) {
  // TODO(lio): implemented in a later lane
  (void)cov_rel;
  (void)delta;
  return cov_step;
}

}  // namespace meridian::lio
