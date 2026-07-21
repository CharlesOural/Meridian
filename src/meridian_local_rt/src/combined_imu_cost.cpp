#include "meridian/local_rt/combined_imu_cost.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <exception>
#include <memory>

#include "gtsam_inertial_internal.hpp"
#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt {
namespace {

using Matrix15x6 = Eigen::Matrix<double, 15, 6>;
using Matrix15x7 = Eigen::Matrix<double, 15, 7>;
using Matrix15x9 = Eigen::Matrix<double, 15, 9>;
using RowMajor15x7 = Eigen::Matrix<double, 15, 7, Eigen::RowMajor>;
using RowMajor15x9 = Eigen::Matrix<double, 15, 9, Eigen::RowMajor>;
using RowMajor6x7 = Eigen::Matrix<double, 6, 7, Eigen::RowMajor>;

bool pose(const double* value, gtsam::Pose3& destination) {
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> coefficients(value);
  if (!coefficients.array().isFinite().all()) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(value + 3);
  if (std::abs(quaternion.squaredNorm() - 1.0) > 1.0e-8) {
    return false;
  }
  destination = gtsam::Pose3(
      gtsam::Rot3::Quaternion(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()),
      gtsam::Point3(value[0], value[1], value[2]));
  return true;
}

bool motion(const double* value, gtsam::Vector3& velocity, gtsam::imuBias::ConstantBias& bias) {
  const Eigen::Map<const Eigen::Matrix<double, 9, 1>> vector(value);
  if (!vector.array().isFinite().all()) {
    return false;
  }
  velocity = vector.head<3>();
  bias = gtsam::imuBias::ConstantBias(vector.tail<3>(), vector.segment<3>(3));
  return true;
}

Matrix15x6 reorderPoseJacobian(const gtsam::Matrix& gtsam_jacobian) {
  Matrix15x6 result;
  result.leftCols<3>() = gtsam_jacobian.rightCols<3>();
  result.rightCols<3>() = gtsam_jacobian.leftCols<3>();
  return result;
}

Matrix15x9 motionJacobian(const gtsam::Matrix& velocity_jacobian,
                          const gtsam::Matrix& bias_jacobian) {
  Matrix15x9 result;
  result.leftCols<3>() = velocity_jacobian;
  result.middleCols<3>(3) = bias_jacobian.rightCols<3>();
  result.rightCols<3>() = bias_jacobian.leftCols<3>();
  return result;
}

}  // namespace

struct CombinedImuCost::Impl final {
  explicit Impl(const gtsam::PreintegratedCombinedMeasurements& pim)
      : factor(1U, 2U, 3U, 4U, 5U, 6U, pim) {}

  gtsam::CombinedImuFactor factor;
  RightSe3Manifold manifold;
};

CombinedImuCost::CombinedImuCost(const CombinedPreintegration& preintegration)
    : impl_(std::make_unique<Impl>(preintegration.impl_->pim)) {}

CombinedImuCost::~CombinedImuCost() = default;

bool CombinedImuCost::Evaluate(double const* const* parameters, double* residuals,
                               double** jacobians) const {
  if (parameters == nullptr || residuals == nullptr) {
    return false;
  }

  try {
    gtsam::Pose3 pose_i;
    gtsam::Pose3 pose_j;
    gtsam::Vector3 velocity_i;
    gtsam::Vector3 velocity_j;
    gtsam::imuBias::ConstantBias bias_i;
    gtsam::imuBias::ConstantBias bias_j;
    if (!pose(parameters[0], pose_i) || !motion(parameters[1], velocity_i, bias_i) ||
        !pose(parameters[2], pose_j) || !motion(parameters[3], velocity_j, bias_j)) {
      return false;
    }

    const bool need_jacobians =
        jacobians != nullptr && (jacobians[0] != nullptr || jacobians[1] != nullptr ||
                                 jacobians[2] != nullptr || jacobians[3] != nullptr);
    gtsam::Matrix H_pose_i;
    gtsam::Matrix H_velocity_i;
    gtsam::Matrix H_pose_j;
    gtsam::Matrix H_velocity_j;
    gtsam::Matrix H_bias_i;
    gtsam::Matrix H_bias_j;
    const gtsam::Vector unwhitened = impl_->factor.evaluateError(
        pose_i, velocity_i, pose_j, velocity_j, bias_i, bias_j,
        need_jacobians ? &H_pose_i : nullptr, need_jacobians ? &H_velocity_i : nullptr,
        need_jacobians ? &H_pose_j : nullptr, need_jacobians ? &H_velocity_j : nullptr,
        need_jacobians ? &H_bias_i : nullptr, need_jacobians ? &H_bias_j : nullptr);

    const gtsam::SharedNoiseModel& noise = impl_->factor.noiseModel();
    Eigen::Map<Eigen::Matrix<double, 15, 1>> residual_map(residuals);
    residual_map = noise->whiten(unwhitened);
    if (!need_jacobians) {
      return true;
    }

    H_pose_i = noise->Whiten(H_pose_i);
    H_velocity_i = noise->Whiten(H_velocity_i);
    H_pose_j = noise->Whiten(H_pose_j);
    H_velocity_j = noise->Whiten(H_velocity_j);
    H_bias_i = noise->Whiten(H_bias_i);
    H_bias_j = noise->Whiten(H_bias_j);

    std::array<double, 42> minus_storage{};
    if (jacobians[0] != nullptr) {
      if (!impl_->manifold.MinusJacobian(parameters[0], minus_storage.data())) {
        return false;
      }
      const Eigen::Map<const RowMajor6x7> minus_jacobian(minus_storage.data());
      const Matrix15x7 ambient = reorderPoseJacobian(H_pose_i) * minus_jacobian;
      Eigen::Map<RowMajor15x7> jacobian(jacobians[0]);
      jacobian = ambient;
    }
    if (jacobians[1] != nullptr) {
      Eigen::Map<RowMajor15x9> jacobian(jacobians[1]);
      jacobian = motionJacobian(H_velocity_i, H_bias_i);
    }
    if (jacobians[2] != nullptr) {
      if (!impl_->manifold.MinusJacobian(parameters[2], minus_storage.data())) {
        return false;
      }
      const Eigen::Map<const RowMajor6x7> minus_jacobian(minus_storage.data());
      const Matrix15x7 ambient = reorderPoseJacobian(H_pose_j) * minus_jacobian;
      Eigen::Map<RowMajor15x7> jacobian(jacobians[2]);
      jacobian = ambient;
    }
    if (jacobians[3] != nullptr) {
      Eigen::Map<RowMajor15x9> jacobian(jacobians[3]);
      jacobian = motionJacobian(H_velocity_j, H_bias_j);
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace meridian::local_rt
