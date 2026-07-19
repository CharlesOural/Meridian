#include "adjacent_boundary_factor_internal.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <utility>

namespace meridian::global::adjacent_internal {
namespace {

using FactorBase =
    gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::Vector3, gtsam::Vector3,
                             gtsam::Pose3, gtsam::Vector3, gtsam::Vector3, gtsam::Vector3>;
using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr std::size_t kCanonicalColumns = 30U;
constexpr double kGravityPreservationTolerance = 1.0e-10;

[[nodiscard]] AdjacentBoundaryAdapterError makeError(AdjacentBoundaryAdapterErrorCode code,
                                                     std::string detail) {
  return AdjacentBoundaryAdapterError{code, std::move(detail)};
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] gtsam::Pose3 toGtsamPose(const core::Pose3d& pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.so3().matrix()), pose.translation());
}

[[nodiscard]] const gtsam::Matrix6& rotationTranslationPermutation() {
  static const gtsam::Matrix6 permutation = [] {
    gtsam::Matrix6 value = gtsam::Matrix6::Zero();
    value.topRightCorner<3, 3>().setIdentity();
    value.bottomLeftCorner<3, 3>().setIdentity();
    return value;
  }();
  return permutation;
}

[[nodiscard]] bool gravityPreserving(const core::Pose3d& placement) noexcept {
  if (!finitePose(placement)) {
    return false;
  }
  const Eigen::Vector3d gravity_axis = Eigen::Vector3d::UnitZ();
  return (placement.so3().matrix() * gravity_axis - gravity_axis).norm() <=
         kGravityPreservationTolerance;
}

[[nodiscard]] std::array<gtsam::Key, 8> orderedKeys(const AdjacentBoundaryKeys& keys) noexcept {
  return {keys.anchor_from, keys.velocity_from, keys.gyro_bias_from, keys.accel_bias_from,
          keys.anchor_to,   keys.velocity_to,   keys.gyro_bias_to,   keys.accel_bias_to};
}

class ExactAdjacentBoundaryFactor final : public FactorBase {
public:
  ExactAdjacentBoundaryFactor(const core::CondensedBoundaryTransition& transition,
                              const core::Pose3d& C_from, const core::Pose3d& C_to,
                              const OdomEpochChartPlacement& placement,
                              const AdjacentBoundaryKeys& keys)
      : FactorBase(gtsam::noiseModel::Unit::Create(transition.boundary_factor.rows),
                   keys.anchor_from, keys.velocity_from, keys.gyro_bias_from, keys.accel_bias_from,
                   keys.anchor_to, keys.velocity_to, keys.gyro_bias_to, keys.accel_bias_to),
        rows_(transition.boundary_factor.rows, kCanonicalColumns),
        rhs_(transition.boundary_factor.rows),
        constant_squared_error_(transition.boundary_factor.constant_squared_error),
        center_from_(toGtsamPose(placement.H_map_odom * transition.from.T_odom_imu)),
        center_to_(toGtsamPose(placement.H_map_odom * transition.to.T_odom_imu)),
        C_from_(toGtsamPose(C_from)),
        C_to_(toGtsamPose(C_to)),
        velocity_basis_(placement.H_map_odom.so3().matrix().transpose()),
        velocity_center_from_(transition.from.velocity_odom),
        velocity_center_to_(transition.to.velocity_odom),
        gyro_bias_center_from_(transition.from.gyro_bias),
        gyro_bias_center_to_(transition.to.gyro_bias),
        accel_bias_center_from_(transition.from.accel_bias),
        accel_bias_center_to_(transition.to.accel_bias) {
    if (transition.boundary_factor.rows > 0U) {
      const Eigen::Map<const RowMajorMatrix> canonical_rows(
          transition.boundary_factor.row_major_A.data(),
          static_cast<Eigen::Index>(transition.boundary_factor.rows),
          static_cast<Eigen::Index>(kCanonicalColumns));
      rows_ = canonical_rows;
      rhs_ = Eigen::Map<const gtsam::Vector>(
          transition.boundary_factor.rhs.data(),
          static_cast<Eigen::Index>(transition.boundary_factor.rows));
    }
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& anchor_from, const gtsam::Vector3& velocity_from,
      const gtsam::Vector3& gyro_bias_from, const gtsam::Vector3& accel_bias_from,
      const gtsam::Pose3& anchor_to, const gtsam::Vector3& velocity_to,
      const gtsam::Vector3& gyro_bias_to, const gtsam::Vector3& accel_bias_to,
      boost::optional<gtsam::Matrix&> anchor_from_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> velocity_from_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> gyro_bias_from_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> accel_bias_from_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> anchor_to_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> velocity_to_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> gyro_bias_to_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> accel_bias_to_jacobian = boost::none) const override {
    gtsam::Matrix6 pose_from_coordinate_jacobian;
    gtsam::Matrix6 pose_to_coordinate_jacobian;
    const bool need_pose_from_jacobian = anchor_from_jacobian.has_value();
    const bool need_pose_to_jacobian = anchor_to_jacobian.has_value();

    Eigen::Matrix<double, kCanonicalColumns, 1> delta;
    delta.segment<6>(0) =
        poseCoordinates(center_from_, anchor_from, C_from_,
                        need_pose_from_jacobian ? &pose_from_coordinate_jacobian : nullptr);
    delta.segment<3>(6) = velocity_basis_ * velocity_from - velocity_center_from_;
    delta.segment<3>(9) = gyro_bias_from - gyro_bias_center_from_;
    delta.segment<3>(12) = accel_bias_from - accel_bias_center_from_;
    delta.segment<6>(15) =
        poseCoordinates(center_to_, anchor_to, C_to_,
                        need_pose_to_jacobian ? &pose_to_coordinate_jacobian : nullptr);
    delta.segment<3>(21) = velocity_basis_ * velocity_to - velocity_center_to_;
    delta.segment<3>(24) = gyro_bias_to - gyro_bias_center_to_;
    delta.segment<3>(27) = accel_bias_to - accel_bias_center_to_;

    if (anchor_from_jacobian) {
      *anchor_from_jacobian = rows_.block(0, 0, rows_.rows(), 6) * pose_from_coordinate_jacobian;
    }
    if (velocity_from_jacobian) {
      *velocity_from_jacobian = rows_.block(0, 6, rows_.rows(), 3) * velocity_basis_;
    }
    if (gyro_bias_from_jacobian) {
      *gyro_bias_from_jacobian = rows_.block(0, 9, rows_.rows(), 3);
    }
    if (accel_bias_from_jacobian) {
      *accel_bias_from_jacobian = rows_.block(0, 12, rows_.rows(), 3);
    }
    if (anchor_to_jacobian) {
      *anchor_to_jacobian = rows_.block(0, 15, rows_.rows(), 6) * pose_to_coordinate_jacobian;
    }
    if (velocity_to_jacobian) {
      *velocity_to_jacobian = rows_.block(0, 21, rows_.rows(), 3) * velocity_basis_;
    }
    if (gyro_bias_to_jacobian) {
      *gyro_bias_to_jacobian = rows_.block(0, 24, rows_.rows(), 3);
    }
    if (accel_bias_to_jacobian) {
      *accel_bias_to_jacobian = rows_.block(0, 27, rows_.rows(), 3);
    }
    return rows_ * delta - rhs_;
  }

  double error(const gtsam::Values& values) const override {
    return FactorBase::error(values) + 0.5 * constant_squared_error_;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new ExactAdjacentBoundaryFactor(*this));
  }

private:
  [[nodiscard]] static gtsam::Vector6 poseCoordinates(const gtsam::Pose3& center,
                                                      const gtsam::Pose3& anchor,
                                                      const gtsam::Pose3& C_submap_imu,
                                                      gtsam::Matrix6* anchor_jacobian) {
    const gtsam::Pose3 relative = center.inverse() * anchor * C_submap_imu;
    gtsam::Matrix6 log_jacobian;
    const gtsam::Vector6 rotation_first = anchor_jacobian
                                              ? gtsam::Pose3::Logmap(relative, log_jacobian)
                                              : gtsam::Pose3::Logmap(relative);
    if (anchor_jacobian != nullptr) {
      // GTSAM perturbations are right-local [rotation, translation]. Meridian
      // rows are right-local [translation, rotation]. Conjugation through the
      // immutable endpoint frame contributes Ad(C^-1).
      *anchor_jacobian =
          rotationTranslationPermutation() * log_jacobian * C_submap_imu.inverse().AdjointMap();
    }
    return rotationTranslationPermutation() * rotation_first;
  }

  gtsam::Matrix rows_;
  gtsam::Vector rhs_;
  double constant_squared_error_{};
  gtsam::Pose3 center_from_;
  gtsam::Pose3 center_to_;
  gtsam::Pose3 C_from_;
  gtsam::Pose3 C_to_;
  gtsam::Matrix3 velocity_basis_;
  gtsam::Vector3 velocity_center_from_;
  gtsam::Vector3 velocity_center_to_;
  gtsam::Vector3 gyro_bias_center_from_;
  gtsam::Vector3 gyro_bias_center_to_;
  gtsam::Vector3 accel_bias_center_from_;
  gtsam::Vector3 accel_bias_center_to_;
};

}  // namespace

core::Result<AdjacentBoundaryFactorPtr, AdjacentBoundaryAdapterError> makeAdjacentBoundaryFactor(
    const core::SparseSubmapSeal& predecessor, const core::SparseSubmapSeal& current,
    const OdomEpochChartPlacement& placement, const AdjacentBoundaryKeys& keys,
    const AdjacentBoundaryFactorLimits& limits) {
  using Result = core::Result<AdjacentBoundaryFactorPtr, AdjacentBoundaryAdapterError>;
  if (limits.maximum_rows == 0U || limits.maximum_coefficients == 0U) {
    return Result::failure(makeError(AdjacentBoundaryAdapterErrorCode::InvalidLimits,
                                     "adjacent boundary factor limits must be positive"));
  }
  if (!current.from_previous) {
    return Result::failure(makeError(AdjacentBoundaryAdapterErrorCode::InvalidSparseLink,
                                     "current seal has no incoming boundary transition"));
  }
  const core::CondensedBoundaryTransition& transition = current.from_previous->local_transition;
  // Bound work before the core link validator performs its rank-revealing
  // SVD. The adapter never accepts caller-supplied endpoint frames.
  if (transition.boundary_factor.rows > limits.maximum_rows ||
      transition.boundary_factor.row_major_A.size() > limits.maximum_coefficients ||
      transition.boundary_factor.rhs.size() > limits.maximum_rows ||
      transition.boundary_factor.layout.size() > 8U) {
    return Result::failure(
        makeError(AdjacentBoundaryAdapterErrorCode::CapacityExceeded,
                  "condensed transition exceeds the configured row or coefficient bound"));
  }
  if (!core::verifyCanonicalSparseSubmapSeal(predecessor) ||
      !core::verifyCanonicalSparseSubmapSeal(current) ||
      core::validateSparseSubmapLink(predecessor, current) !=
          core::SparseSubmapLinkValidationError::None) {
    return Result::failure(
        makeError(AdjacentBoundaryAdapterErrorCode::InvalidSparseLink,
                  "core sparse-seal link validation rejected the exact endpoints"));
  }
  if (!placement.odom_epoch.valid() || placement.odom_epoch != transition.odom_epoch) {
    return Result::failure(
        makeError(AdjacentBoundaryAdapterErrorCode::OdomEpochMismatch,
                  "odometry chart placement does not match the transition epoch"));
  }
  if (!gravityPreserving(placement.H_map_odom)) {
    return Result::failure(
        makeError(AdjacentBoundaryAdapterErrorCode::InvalidChartPlacement,
                  "odometry chart placement must be finite and preserve the gravity axis"));
  }
  const auto ordered_keys = orderedKeys(keys);
  const std::set<gtsam::Key> unique_keys(ordered_keys.begin(), ordered_keys.end());
  if (unique_keys.size() != ordered_keys.size()) {
    return Result::failure(
        makeError(AdjacentBoundaryAdapterErrorCode::DuplicateKey,
                  "adjacent boundary variables require eight distinct GTSAM keys"));
  }

  const core::Pose3d C_from =
      predecessor.T_odom_submap.inverse() * predecessor.boundary_navigation.T_odom_imu;
  const core::Pose3d C_to =
      current.T_odom_submap.inverse() * current.boundary_navigation.T_odom_imu;

  AdjacentBoundaryFactorPtr factor(
      new ExactAdjacentBoundaryFactor(transition, C_from, C_to, placement, keys));
  return Result::success(std::move(factor));
}

}  // namespace meridian::global::adjacent_internal
