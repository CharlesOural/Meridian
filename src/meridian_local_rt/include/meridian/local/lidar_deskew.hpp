#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/imu.hpp"

namespace meridian::local {

enum class DeskewErrorCode {
  EmptySweep,
  EmptyTrajectory,
  InvalidTrajectory,
  ReferenceOutsideTrajectory,
  PointOutsideTrajectory,
  NonFiniteTransform,
};

struct DeskewError {
  DeskewErrorCode code{};
  std::string detail;
};

struct DeskewedSweep {
  core::MeasurementId source;
  core::FusionTime reference_time;
  core::Pose3d T_odom_imu_reference;
  core::LidarLayout layout;
  // Exclusive ownership prevents a retained mutable alias. Registration-cloud
  // construction validates selected rows against this provisional tracking
  // view, then discards it; accepted dense mapping starts again from raw data.
  std::unique_ptr<core::LidarPoints> points_in_reference_imu;
  std::vector<core::MeasurementId> imu_support;
  // Number of distinct point-time poses evaluated. Returns sharing the same
  // immutable time offset reuse one pose and reference-frame transform.
  std::size_t pose_interpolations{};
};

// Discrete-time deskew. `trajectory` is a table obtained from midpoint IMU
// propagation; interpolation is only between those fixed states and introduces
// no continuous-time optimization variables.
[[nodiscard]] core::Result<DeskewedSweep, DeskewError> deskewLidarSweep(
    const core::LidarSweep& sweep, core::FusionTime reference_time, const core::Pose3d& T_imu_lidar,
    const std::vector<TimedNavState>& trajectory,
    std::vector<core::MeasurementId> imu_support = {});

}  // namespace meridian::local
