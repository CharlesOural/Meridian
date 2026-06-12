#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "meridian/common/pose.hpp"

namespace meridian::lio {

// Accumulate one more registration step into the running relative covariance:
// transport `cov_rel` across the step `delta` with the right-perturbation adjoint
// and add the step's own covariance. All blocks are translation-first [rho; phi]
// in the right/body tangent.
inline Eigen::Matrix<double, 6, 6> composeRelativeCov(const Eigen::Matrix<double, 6, 6>& cov_rel,
                                                      const Pose& delta,
                                                      const Eigen::Matrix<double, 6, 6>& cov_step) {
  // X*exp(xi_rel) * D*exp(xi_step) = (X*D) * exp(Ad(D^-1)*xi_rel) * exp(xi_step), so to
  // first order the composed right-perturbation is Ad(D^-1)*xi_rel + xi_step and the
  // two independent covariances add after transporting cov_rel by Ad(delta^-1).
  const Eigen::Matrix<double, 6, 6> adj_delta_inv = Sophus::SE3d(delta.q, delta.t).inverse().Adj();
  return adj_delta_inv * cov_rel * adj_delta_inv.transpose() + cov_step;
}

}  // namespace meridian::lio
