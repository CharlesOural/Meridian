#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "meridian/core/observations.hpp"
#include "meridian/local_rt/lidar/point_to_point_registration.hpp"

namespace meridian::local_rt::initialization {

struct BootstrapOdometryOptions final {
  double downsample_voxel_size_m{};
  std::size_t minimum_points{};
  double maximum_accepted_rmse_m{};
  double maximum_accepted_condition_number{};
  lidar::VoxelTargetOptions target;
  lidar::PointToPointRegistrationOptions registration;
};

enum class BootstrapFrameOutcome : std::uint8_t {
  kAcceptedAnchor,
  kAcceptedRegistration,
  kRejectedTimestamp,
  kRejectedPointCount,
  kRejectedSourcePointLimit,
  kRejectedAnchorAdmission,
  kRejectedRegistration,
  kRejectedTargetAdmission,
};

struct BootstrapFrame final {
  core::MeasurementId measurement_id;
  core::TimeNs time;
  Sophus::SE3d initial_odom_from_lidar;
  Sophus::SE3d odom_from_lidar;
  lidar::PointCloud registration_points;
  lidar::VoxelDownsampleStats downsample;
  lidar::RegistrationQuality quality;
  std::optional<lidar::RegistrationStatus> registration_status;
  std::optional<lidar::VoxelTargetUpdateStats> target_update;
  BootstrapFrameOutcome outcome{BootstrapFrameOutcome::kRejectedPointCount};
  bool accepted{};
};

// Temporary metric LiDAR odometry used only by DYNAMIC initialization. The
// target and poses are discarded after initialization and never seed the
// production map directly.
class BootstrapOdometry final {
public:
  explicit BootstrapOdometry(BootstrapOdometryOptions options);

  [[nodiscard]] BootstrapFrame add(const core::LidarSweep& sweep,
                                   std::span<const lidar::Point3d> points_at_sweep_end);
  void reset();

private:
  struct AcceptedPose final {
    core::TimeNs time;
    Sophus::SE3d odom_from_lidar;
  };

  [[nodiscard]] Sophus::SE3d constantVelocityGuess(core::TimeNs time) const;

  BootstrapOdometryOptions options_;
  lidar::BoundedVoxelTarget target_;
  lidar::DirectPointToPointRegistration registration_;
  std::vector<AcceptedPose> accepted_poses_;
};

}  // namespace meridian::local_rt::initialization
