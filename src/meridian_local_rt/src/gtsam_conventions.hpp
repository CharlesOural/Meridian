#pragma once

// Private GTSAM seam.  This header is intentionally not installed.

#include <array>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Key.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "meridian/local/imu.hpp"

namespace meridian::local::gtsam_api {

// Meridian navigation covariance: [R, V, P, Bg, Ba].
// GTSAM joint X,V,B tangent:      [R, P, V, Ba, Bg].
inline constexpr std::array<Eigen::Index, 15> kGtsamIndexToMeridian{
    0, 1, 2, 6, 7, 8, 3, 4, 5, 12, 13, 14, 9, 10, 11};

[[nodiscard]] gtsam::Pose3 toGtsamPose(const core::Pose3d& pose);
[[nodiscard]] core::Pose3d fromGtsamPose(const gtsam::Pose3& pose);
[[nodiscard]] gtsam::imuBias::ConstantBias toGtsamBias(
    const core::NavStateEstimate& state);
[[nodiscard]] core::NavStateEstimate fromGtsamState(
    const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
    const gtsam::imuBias::ConstantBias& bias);

[[nodiscard]] Eigen::Matrix<double, 15, 15> toGtsamNavigationCovariance(
    const NavigationCovariance& covariance);
[[nodiscard]] NavigationCovariance fromGtsamNavigationCovariance(
    const Eigen::Matrix<double, 15, 15>& covariance);

// Extracts the exact joint marginal represented by iSAM2's current linear
// Bayes tree.  This deliberately does not rebuild and relinearize the complete
// nonlinear factor graph: the Bayes tree is the posterior that produced the
// current incremental estimate.  Its already-linear Gaussian conditionals are
// marginalized once to retain every X/V/B cross block; the only dense solve is
// the final 15x15 joint.
[[nodiscard]] NavigationCovariance jointNavigationCovarianceFromBayesTree(
    const gtsam::ISAM2& solver, gtsam::Key pose_key,
    gtsam::Key velocity_key, gtsam::Key bias_key);

// Pose3 uses [rotation, translation] while Sophus SE3 uses
// [translation, rotation].  These helpers make that permutation reviewable and
// testable rather than relying on coincidentally compatible matrices.
[[nodiscard]] Eigen::Matrix<double, 6, 1> toGtsamPoseTangent(
    const Eigen::Matrix<double, 6, 1>& meridian_tangent);
[[nodiscard]] Eigen::Matrix<double, 6, 1> fromGtsamPoseTangent(
    const Eigen::Matrix<double, 6, 1>& gtsam_tangent);

}  // namespace meridian::local::gtsam_api
