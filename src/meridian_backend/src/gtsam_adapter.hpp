#pragma once

#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>

#include "meridian/common/pose.hpp"

namespace meridian::backend {

// Exact pose conversions: rotation and translation are copied, never re-derived.
gtsam::Pose3 to_gtsam(const Pose& T);
Pose from_gtsam(const gtsam::Pose3& T);

// 6x6 permutation taking a Meridian-ordered tangent [tx,ty,tz,rx,ry,rz] to the GTSAM
// Pose3 order [rx,ry,rz,tx,ty,tz]: v_gtsam = P * v_meridian. P swaps the two 3-blocks,
// so it is symmetric and its own inverse.
const Eigen::Matrix<double, 6, 6>& P_meridian_gtsam();

// Conjugate a covariance/information block between the two tangent orderings.
Eigen::Matrix<double, 6, 6> reorder_meridian_to_gtsam(const Eigen::Matrix<double, 6, 6>& S);
Eigen::Matrix<double, 6, 6> reorder_gtsam_to_meridian(const Eigen::Matrix<double, 6, 6>& S);

// Symmetrizes S in place and raises any eigenvalue below min_eig up to min_eig.
// Returns true iff clamping was needed.
bool ensure_psd(Eigen::Matrix<double, 6, 6>& S, double min_eig = 1e-12);

}  // namespace meridian::backend
