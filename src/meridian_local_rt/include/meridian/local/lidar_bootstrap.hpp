#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/lidar_registration.hpp"
#include "meridian/local/lidar_registration_cloud.hpp"
#include "meridian/local/motion_initializer.hpp"
#include "meridian/local/pipeline_observability.hpp"

namespace meridian::local {

// Bounded LiDAR odometry used only to make moving LiDAR--IMU initialization
// observable. The online proposal pass rotation-deskews every retained raw
// sweep from exact discrete IMU support. A private refinement pass then fully
// deskews the same raw points from the provisional batch states and rebuilds
// every adjacent registration. The LiDAR increments are therefore explicitly
// IMU-conditioned approximations, never independent duplicate evidence.
struct LidarBootstrapOdometryConfig {
  std::size_t maximum_sweeps{32U};
  LidarPreprocessConfig preprocessing;
  // Bootstrap compares one scan against one owner. Preserve substantially
  // more within-voxel geometry than the multi-owner tracking composite while
  // retaining the same bounded index implementation.
  LidarRegistrationConfig registration = [] {
    LidarRegistrationConfig config;
    config.maximum_composite_points_per_voxel = 64U;
    return config;
  }();
  double target_reuse_covariance_inflation{1.5};
  double imu_conditioning_covariance_inflation{4.0};
  double maximum_increment_translation_m{3.0};
  double maximum_increment_rotation_rad{0.8};
  double maximum_observable_condition{1.0e10};
};

struct LidarBootstrapInput {
  core::LidarSweep sweep;
  core::DerivedRecordId cloud_record;
  core::ObservationLineageId cloud_lineage;

  // Exact boundary-interpolated IMU support for the complete sweep
  // acquisition interval. It is mandatory even for the anchor scan.
  ImuInterval acquisition_imu;

  // Empty for the first sweep. Every later sweep must supply all three fields
  // for the exact previous-reference -> current-reference segment.
  std::optional<ImuInterval> between_reference_imu;
  std::optional<core::ObservationLineage> lidar_factor_lineage;
  std::optional<core::ObservationLineage> imu_factor_lineage;
};

enum class LidarBootstrapDisposition {
  AnchorCreated,
  IncrementCommitted,
};

struct LidarBootstrapCommit {
  LidarBootstrapDisposition disposition{LidarBootstrapDisposition::AnchorCreated};
  core::FusionTime reference_time;
  core::Pose3d T_bootstrap_imu;
  LidarPreprocessStats preprocessing;
  std::size_t deskew_pose_interpolations{};
  std::optional<LidarRegistrationResult> registration;
  std::optional<MotionInitializationSegment> segment;
  std::size_t retained_sweeps{};
};

struct LidarBootstrapRefinementDiagnostics {
  std::size_t sweeps{};
  std::size_t registrations{};
  std::size_t deskew_pose_interpolations{};
  std::size_t minimum_observable_rank{};
  double total_registration_cost{};
  double maximum_registration_cost{};
  double imu_conditioning_covariance_inflation{1.0};
  double applied_covariance_inflation{1.0};
};

struct LidarBootstrapRefinement {
  std::vector<MotionInitializationSegment> segments;
  std::shared_ptr<const LidarRegistrationCloud> final_cloud;
  core::ObservationLineage final_lineage;
  LidarBootstrapRefinementDiagnostics diagnostics;
};

enum class LidarBootstrapErrorCode {
  InvalidConfig,
  InvalidCalibration,
  InvalidSweep,
  InvalidAcquisitionImu,
  NonMonotonicSweep,
  Capacity,
  InvalidSegmentSupport,
  InvalidLineage,
  RegistrationCloudBuildFailed,
  RegistrationFailed,
  IncrementGateFailed,
  DeskewFailed,
  RefinementStateMismatch,
};

struct LidarBootstrapError {
  LidarBootstrapErrorCode code{};
  std::string detail;
  std::optional<LidarPreprocessError> preprocessing;
  std::optional<LidarRegistrationError> registration;
};

// Move-only, bounded, ROS-free single writer. Failed candidates do not replace
// the previously committed scan, pose, or constant-twist seed.
class LidarBootstrapOdometry {
public:
  [[nodiscard]] static core::Result<LidarBootstrapOdometry, LidarBootstrapError> create(
      core::CalibrationEpoch calibration, core::Pose3d T_imu_lidar, Eigen::Vector3d gravity_odom,
      Eigen::Vector3d gyro_bias_prior_mean_radps, LidarBootstrapOdometryConfig config = {});

  [[nodiscard]] static core::Result<LidarBootstrapOdometry, LidarBootstrapError> create(
      core::CalibrationEpoch calibration, core::Pose3d T_imu_lidar, Eigen::Vector3d gravity_odom,
      Eigen::Vector3d gyro_bias_prior_mean_radps, LidarBootstrapOdometryConfig config,
      std::shared_ptr<LocalPipelineTimingRecorder> timing);

  ~LidarBootstrapOdometry();
  LidarBootstrapOdometry(LidarBootstrapOdometry&&) noexcept;
  LidarBootstrapOdometry& operator=(LidarBootstrapOdometry&&) noexcept;
  LidarBootstrapOdometry(const LidarBootstrapOdometry&) = delete;
  LidarBootstrapOdometry& operator=(const LidarBootstrapOdometry&) = delete;

  [[nodiscard]] core::Result<LidarBootstrapCommit, LidarBootstrapError> add(
      LidarBootstrapInput input);

  // Pure second pass: does not mutate the committed proposal window. The
  // supplied states must correspond exactly to a contiguous retained sweep
  // subsequence, including its anchor and final source scan.
  [[nodiscard]] core::Result<LidarBootstrapRefinement, LidarBootstrapError> refine(
      const std::vector<TimedNavState>& reference_states) const;

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::size_t retainedSweeps() const noexcept;

private:
  struct Impl;
  explicit LidarBootstrapOdometry(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace meridian::local
