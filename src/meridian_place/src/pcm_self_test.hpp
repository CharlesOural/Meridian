#pragma once

#include <Eigen/Core>

#include "meridian/common/pose.hpp"

namespace meridian {

// Inverse chi-square CDF at `confidence` for 6 degrees of freedom. Closed form for even DoF
// (chi-square with k=6 has CDF 1 - e^{-x/2}(1 + x/2 + (x/2)^2/2)), solved by bisection — no
// external special-function dependency, and matches the back-end's boost-based quantile so
// the L5 self-test and the L3 clique judge consistency on one threshold.
double chi2InvDof6(double confidence);

// The §8.2.1 single-loop self-test: does a candidate loop agree with the corrected odometry
// chain between its endpoints, within their combined uncertainty? Residual is the pose
// difference between the loop's transform and the odometry estimate of the same relative
// pose; it is whitened by the combined covariance and compared to the chi-square threshold.
// A heavily-drifted chain has a large chain covariance, so a genuine large-correction loop is
// NOT rejected — only one inconsistent with what the odometry could plausibly have drifted.
bool loopAgreesWithOdometry(const Pose& T_odom_from_to, const Pose& T_loop_from_to,
                            const Eigen::Matrix<double, 6, 6>& cov_combined,
                            double chi2_threshold, double* chi2_out = nullptr);

}  // namespace meridian
