#pragma once

#include <ceres/cost_function.h>

#include <Eigen/Core>
#include <span>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/local_rt/lidar/voxel_target.hpp"

namespace meridian::local_rt::estimator {

// One frozen direct-LiDAR row against geometry expressed in a live target
// owner's LiDAR reference frame. The source pose is the first parameter block
// and the target-owner pose is the second parameter block.
struct ActiveLidarRow final {
  lidar::Point3d source_lidar;
  lidar::Point3d target_lidar;
  // Includes both the frozen robust square-root weight and metric whitening.
  double sqrt_weight_over_sigma{};
};

// One frozen direct-LiDAR row against finalized geometry expressed in odom.
struct FinalizedLidarRow final {
  lidar::Point3d source_lidar;
  lidar::Point3d target_odom;
  // Includes both the frozen robust square-root weight and metric whitening.
  double sqrt_weight_over_sigma{};
};

// Grouped unary direct point-to-point residuals. The sole parameter block is
// T_odom_imu encoded as [px, py, pz, qx, qy, qz, qw]. Every row contributes
// three residuals and uses the configured, fixed T_imu_lidar.
class FinalizedScanToMapCost final : public ceres::CostFunction {
public:
  FinalizedScanToMapCost(std::vector<FinalizedLidarRow> rows, core::Pose3d T_imu_lidar);

  FinalizedScanToMapCost(const FinalizedScanToMapCost&) = delete;
  FinalizedScanToMapCost& operator=(const FinalizedScanToMapCost&) = delete;

  [[nodiscard]] std::span<const FinalizedLidarRow> rows() const noexcept { return rows_; }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

private:
  std::vector<FinalizedLidarRow> rows_;
  Eigen::Matrix3d R_imu_lidar_;
  Eigen::Vector3d t_imu_lidar_;
};

// Grouped binary direct point-to-point residuals. Parameter block zero is the
// source T_odom_imu and block one is the active target owner's T_odom_imu;
// both use [p, qxyzw]. Target points remain in their owner's immutable LiDAR
// reference frame, so owner-pose revisions never require rebuilding geometry.
class ActiveOwnerScanToMapCost final : public ceres::CostFunction {
public:
  ActiveOwnerScanToMapCost(std::vector<ActiveLidarRow> rows, core::Pose3d T_imu_lidar);

  ActiveOwnerScanToMapCost(const ActiveOwnerScanToMapCost&) = delete;
  ActiveOwnerScanToMapCost& operator=(const ActiveOwnerScanToMapCost&) = delete;

  [[nodiscard]] std::span<const ActiveLidarRow> rows() const noexcept { return rows_; }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

private:
  std::vector<ActiveLidarRow> rows_;
  Eigen::Matrix3d R_imu_lidar_;
  Eigen::Vector3d t_imu_lidar_;
};

}  // namespace meridian::local_rt::estimator
