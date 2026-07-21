#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <sophus/se3.hpp>
#include <string>
#include <vector>

#include "meridian/core/initialization.hpp"
#include "meridian/core/observations.hpp"
#include "meridian/local_rt/initialization/bootstrap_odometry.hpp"

namespace meridian::local_rt {
class GtsamCombinedPreintegrator;
class ImuBuffer;
class ImuInterval;
}  // namespace meridian::local_rt

namespace meridian::local_rt::initialization {

struct DynamicInitializerOptions final {
  std::size_t target_sweeps{};
  std::int64_t maximum_support_ns{};
  std::int64_t lidar_time_offset_to_imu_ns{};
  double minimum_range_m{};
  double maximum_range_m{};
  double gravity_m_s2{};
  double gyroscope_finite_difference_step_rad_s{};
  double minimum_singular_value_ratio{};
  double maximum_condition_number{};
  double maximum_gyro_bias_correction_rad_s{};
  double maximum_gravity_magnitude_error_m_s2{};
  double maximum_alignment_residual_rms{};
  double maximum_held_out_rotation_error_rad{};
  double maximum_held_out_translation_error_m{};
  double maximum_refinement_rotation_change_rad{};
  double maximum_refinement_translation_change_m{};
  core::ImuBias calibrated_bias_prior;
  Eigen::Matrix3d gyroscope_bias_prior_covariance{Eigen::Matrix3d::Identity()};
  core::Pose3d base_from_imu;
  core::Pose3d imu_from_lidar;
  BootstrapOdometryOptions bootstrap;
};

struct BootstrapPoseSummary final {
  core::MeasurementId measurement_id;
  core::TimeNs measurement_time;
  core::Pose3d odom_from_lidar;
  std::size_t source_point_count{};
  std::size_t correspondence_count{};
  double point_rmse_m{};
  double hessian_condition_number{};
  bool accepted{};
};

struct DynamicInitializationUpdate final {
  core::InitializationStatus status{core::InitializationStatus::kCollecting};
  std::string reason;
  core::InitializationQuality quality;
  std::optional<BootstrapPoseSummary> bootstrap_pose;
  std::optional<core::InitializationResult> result;
};

// Fixed-calibration staged metric LiDAR–IMU alignment. The newest accepted
// LiDAR transition is excluded from both solves and used only for validation.
// One complete re-deskew/re-registration pass is the refinement ceiling.
class DynamicInitializer final {
public:
  explicit DynamicInitializer(DynamicInitializerOptions options);

  [[nodiscard]] DynamicInitializationUpdate add(const core::LidarSweep& sweep,
                                                const ImuBuffer& imu_buffer,
                                                const GtsamCombinedPreintegrator& preintegrator);
  [[nodiscard]] bool accepted() const noexcept { return accepted_result_.has_value(); }
  [[nodiscard]] std::span<const lidar::Point3d> acceptedAnchorPoints() const noexcept {
    return accepted_anchor_points_;
  }
  void reset();

private:
  struct AcceptedSweep final {
    core::LidarSweep sweep;
    BootstrapFrame bootstrap;
  };

  struct AlignmentEstimate final {
    core::ImuBias bias;
    Eigen::Vector3d gravity_bootstrap{Eigen::Vector3d::Zero()};
    std::vector<Eigen::Vector3d> velocities_bootstrap;
    core::InitializationQuality quality;
  };

  [[nodiscard]] std::optional<lidar::PointCloud> deskewRotationOnly(
      const core::LidarSweep& sweep, const ImuBuffer& imu_buffer, const core::ImuBias& bias) const;
  [[nodiscard]] std::optional<lidar::PointCloud> deskewWithEstimate(
      const AcceptedSweep& accepted, const AlignmentEstimate& estimate, std::size_t accepted_index,
      const ImuBuffer& imu_buffer, const GtsamCombinedPreintegrator& preintegrator) const;
  [[nodiscard]] std::optional<AlignmentEstimate> align(
      const std::vector<AcceptedSweep>& accepted, const ImuBuffer& imu_buffer,
      const GtsamCombinedPreintegrator& preintegrator, core::InitializationQuality& failure_quality,
      std::string& failure_reason) const;
  [[nodiscard]] DynamicInitializationUpdate attempt(
      const ImuBuffer& imu_buffer, const GtsamCombinedPreintegrator& preintegrator);

  DynamicInitializerOptions options_;
  BootstrapOdometry bootstrap_;
  std::deque<AcceptedSweep> accepted_sweeps_;
  std::uint32_t rejected_transitions_{};
  std::optional<core::InitializationResult> accepted_result_;
  lidar::PointCloud accepted_anchor_points_;
};

}  // namespace meridian::local_rt::initialization
