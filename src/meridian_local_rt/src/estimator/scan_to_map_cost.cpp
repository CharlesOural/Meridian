#include "meridian/local_rt/estimator/scan_to_map_cost.hpp"

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt::estimator {
namespace {

using Matrix3x6d = Eigen::Matrix<double, 3, 6>;
using Matrix3x7d = Eigen::Matrix<double, 3, 7>;
using RowMajor3x7d = Eigen::Matrix<double, 3, 7, Eigen::RowMajor>;
using RowMajor6x7d = Eigen::Matrix<double, 6, 7, Eigen::RowMajor>;

struct PoseComponents final {
  Eigen::Matrix3d rotation;
  Eigen::Vector3d translation;
};

Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

bool decodePose(const double* parameters, PoseComponents& pose) {
  if (parameters == nullptr) {
    return false;
  }
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> vector(parameters);
  if (!vector.array().isFinite().all()) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(parameters + 3);
  if (std::abs(quaternion.squaredNorm() - 1.0) > 1.0e-8) {
    return false;
  }
  pose.rotation = quaternion.toRotationMatrix();
  pose.translation = vector.head<3>();
  return true;
}

Eigen::Matrix3d rotation(const core::Quaterniond& quaternion) {
  return Eigen::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z())
      .toRotationMatrix();
}

Eigen::Vector3d translation(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

int residualCount(std::size_t row_count) {
  constexpr std::size_t kResidualsPerRow = 3U;
  const std::size_t maximum_rows =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) / kResidualsPerRow;
  if (row_count == 0U || row_count > maximum_rows) {
    throw std::invalid_argument("scan-to-map cost requires a non-empty, representable row set");
  }
  return static_cast<int>(kResidualsPerRow * row_count);
}

void validateWeight(double sqrt_weight_over_sigma) {
  if (!std::isfinite(sqrt_weight_over_sigma) || sqrt_weight_over_sigma <= 0.0) {
    throw std::invalid_argument(
        "scan-to-map row sqrt_weight_over_sigma must be finite and positive");
  }
}

void validateRows(const std::vector<FinalizedLidarRow>& rows) {
  static_cast<void>(residualCount(rows.size()));
  for (const FinalizedLidarRow& row : rows) {
    if (!row.source_lidar.allFinite() || !row.target_odom.allFinite()) {
      throw std::invalid_argument("finalized scan-to-map row points must be finite");
    }
    validateWeight(row.sqrt_weight_over_sigma);
  }
}

void validateRows(const std::vector<ActiveLidarRow>& rows) {
  static_cast<void>(residualCount(rows.size()));
  for (const ActiveLidarRow& row : rows) {
    if (!row.source_lidar.allFinite() || !row.target_lidar.allFinite()) {
      throw std::invalid_argument("active-owner scan-to-map row points must be finite");
    }
    validateWeight(row.sqrt_weight_over_sigma);
  }
}

Eigen::Vector3d pointInImu(const Eigen::Matrix3d& R_imu_lidar, const Eigen::Vector3d& t_imu_lidar,
                           const lidar::Point3d& point_lidar) {
  return R_imu_lidar * point_lidar + t_imu_lidar;
}

Eigen::Vector3d pointInOdom(const PoseComponents& pose, const Eigen::Vector3d& point_imu) {
  return pose.rotation * point_imu + pose.translation;
}

Matrix3x6d sourceLocalJacobian(const PoseComponents& pose, const Eigen::Vector3d& point_imu,
                               double sqrt_weight_over_sigma) {
  Matrix3x6d jacobian;
  jacobian.leftCols<3>() = pose.rotation;
  jacobian.rightCols<3>() = -pose.rotation * skew(point_imu);
  return sqrt_weight_over_sigma * jacobian;
}

Matrix3x6d activeSourceLocalJacobian(const PoseComponents& source_pose,
                                     const PoseComponents& target_pose,
                                     const Eigen::Matrix3d& R_imu_lidar,
                                     const Eigen::Vector3d& source_point_imu,
                                     double sqrt_weight_over_sigma) {
  return R_imu_lidar.transpose() * target_pose.rotation.transpose() *
         sourceLocalJacobian(source_pose, source_point_imu, sqrt_weight_over_sigma);
}

Matrix3x6d activeTargetLocalJacobian(const Eigen::Matrix3d& R_imu_lidar,
                                     const Eigen::Vector3d& source_in_target_imu,
                                     double sqrt_weight_over_sigma) {
  Matrix3x6d jacobian;
  jacobian.leftCols<3>() = -Eigen::Matrix3d::Identity();
  jacobian.rightCols<3>() = skew(source_in_target_imu);
  return sqrt_weight_over_sigma * R_imu_lidar.transpose() * jacobian;
}

bool minusJacobian(const double* pose, RowMajor6x7d& result) {
  std::array<double, 42> storage{};
  RightSe3Manifold manifold;
  if (!manifold.MinusJacobian(pose, storage.data())) {
    return false;
  }
  result = Eigen::Map<const RowMajor6x7d>(storage.data());
  return true;
}

}  // namespace

FinalizedScanToMapCost::FinalizedScanToMapCost(std::vector<FinalizedLidarRow> rows,
                                               core::Pose3d T_imu_lidar)
    : rows_(std::move(rows)),
      R_imu_lidar_(rotation(T_imu_lidar.rotation())),
      t_imu_lidar_(translation(T_imu_lidar.translation())) {
  validateRows(rows_);
  set_num_residuals(residualCount(rows_.size()));
  mutable_parameter_block_sizes()->push_back(7);
}

bool FinalizedScanToMapCost::Evaluate(double const* const* parameters, double* residuals,
                                      double** jacobians) const {
  if (parameters == nullptr || residuals == nullptr) {
    return false;
  }
  PoseComponents source_pose;
  if (!decodePose(parameters[0], source_pose)) {
    return false;
  }

  const bool need_jacobian = jacobians != nullptr && jacobians[0] != nullptr;
  RowMajor6x7d lift;
  if (need_jacobian) {
    if (!minusJacobian(parameters[0], lift)) {
      return false;
    }
  }

  for (std::size_t index = 0U; index < rows_.size(); ++index) {
    const FinalizedLidarRow& row = rows_[index];
    const Eigen::Vector3d source_imu = pointInImu(R_imu_lidar_, t_imu_lidar_, row.source_lidar);
    const Eigen::Vector3d source_odom = pointInOdom(source_pose, source_imu);
    const Eigen::Index offset = static_cast<Eigen::Index>(3U * index);
    Eigen::Map<Eigen::Vector3d>(residuals + offset) =
        row.sqrt_weight_over_sigma * (source_odom - row.target_odom);
    if (need_jacobian) {
      const Matrix3x7d ambient =
          sourceLocalJacobian(source_pose, source_imu, row.sqrt_weight_over_sigma) * lift;
      Eigen::Map<RowMajor3x7d>(jacobians[0] + offset * 7) = ambient;
    }
  }
  return true;
}

ActiveOwnerScanToMapCost::ActiveOwnerScanToMapCost(std::vector<ActiveLidarRow> rows,
                                                   core::Pose3d T_imu_lidar)
    : rows_(std::move(rows)),
      R_imu_lidar_(rotation(T_imu_lidar.rotation())),
      t_imu_lidar_(translation(T_imu_lidar.translation())) {
  validateRows(rows_);
  set_num_residuals(residualCount(rows_.size()));
  mutable_parameter_block_sizes()->push_back(7);
  mutable_parameter_block_sizes()->push_back(7);
}

bool ActiveOwnerScanToMapCost::Evaluate(double const* const* parameters, double* residuals,
                                        double** jacobians) const {
  if (parameters == nullptr || residuals == nullptr) {
    return false;
  }
  PoseComponents source_pose;
  PoseComponents target_pose;
  if (!decodePose(parameters[0], source_pose) || !decodePose(parameters[1], target_pose)) {
    return false;
  }

  const bool need_source_jacobian = jacobians != nullptr && jacobians[0] != nullptr;
  const bool need_target_jacobian = jacobians != nullptr && jacobians[1] != nullptr;
  RowMajor6x7d source_lift;
  RowMajor6x7d target_lift;
  if ((need_source_jacobian && !minusJacobian(parameters[0], source_lift)) ||
      (need_target_jacobian && !minusJacobian(parameters[1], target_lift))) {
    return false;
  }

  for (std::size_t index = 0U; index < rows_.size(); ++index) {
    const ActiveLidarRow& row = rows_[index];
    const Eigen::Vector3d source_imu = pointInImu(R_imu_lidar_, t_imu_lidar_, row.source_lidar);
    const Eigen::Vector3d source_odom = pointInOdom(source_pose, source_imu);
    const Eigen::Vector3d source_in_target_imu =
        target_pose.rotation.transpose() * (source_odom - target_pose.translation);
    const Eigen::Vector3d source_in_target_lidar =
        R_imu_lidar_.transpose() * (source_in_target_imu - t_imu_lidar_);
    const Eigen::Index offset = static_cast<Eigen::Index>(3U * index);
    Eigen::Map<Eigen::Vector3d>(residuals + offset) =
        row.sqrt_weight_over_sigma * (source_in_target_lidar - row.target_lidar);
    if (need_source_jacobian) {
      Eigen::Map<RowMajor3x7d>(jacobians[0] + offset * 7) =
          activeSourceLocalJacobian(source_pose, target_pose, R_imu_lidar_, source_imu,
                                    row.sqrt_weight_over_sigma) *
          source_lift;
    }
    if (need_target_jacobian) {
      Eigen::Map<RowMajor3x7d>(jacobians[1] + offset * 7) =
          activeTargetLocalJacobian(R_imu_lidar_, source_in_target_imu,
                                    row.sqrt_weight_over_sigma) *
          target_lift;
    }
  }
  return true;
}

}  // namespace meridian::local_rt::estimator
