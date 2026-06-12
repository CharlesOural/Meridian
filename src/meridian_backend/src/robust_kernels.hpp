#pragma once

#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Core>

namespace meridian::backend {

// Inverse chi-squared CDF: smallest x with P(chi^2_dof <= x) = alpha. Used to build
// gate thresholds (e.g. PCM, datum yaw) from a confidence level.
double chi2inv(double alpha, int dof);

// Huber-robustified Gaussian. cov_gtsam is the dense covariance in the variable's own
// tangent order (for a 6-DoF pose it must already be rotation-first to match Pose3).
// huber_k is the Huber threshold in whitened (sigma) units.
gtsam::SharedNoiseModel make_huber_noise(const Eigen::MatrixXd& cov_gtsam, double huber_k);

// Huber-robustified 3-DoF GNSS position noise from an ENU covariance [m^2].
gtsam::SharedNoiseModel make_gnss_noise(const Eigen::Matrix3d& cov_enu, double huber_k);

}  // namespace meridian::backend
