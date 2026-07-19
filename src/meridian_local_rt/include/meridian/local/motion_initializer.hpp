#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/imu.hpp"
#include "meridian/local/pipeline_observability.hpp"

namespace meridian::local {

// One relative IMU-pose observation produced by the IMU-conditioned LiDAR
// bootstrap lane. T_imu_start_imu_end maps points from the end IMU frame into
// the start IMU frame. Its information is expressed in Meridian's public
// right, translation-first pose tangent and may be rank deficient.
struct LidarBootstrapIncrement {
  core::FusionTime start_time;
  core::FusionTime end_time;
  core::Pose3d T_imu_start_imu_end;
  core::RankAwareInformation information;
  core::ObservationLineage lineage;

  // Rotation/full deskew conditions this LiDAR observation on raw IMU samples
  // that are also used by the batch IMU factor. The bounded initializer uses
  // a conservative covariance-inflation approximation rather than claiming
  // that the two observations are independent.
  double imu_conditioning_covariance_inflation{1.0};
  // Total inflation already applied to `information`, including target reuse
  // and the IMU-conditioning term above.
  double applied_covariance_inflation{1.0};
};

// The IMU support and LiDAR increment share exactly the same adjacent state
// times. The IMU lineage must name every raw measurement consumed by the
// interval as a primary residual exactly once.
struct MotionInitializationSegment {
  LidarBootstrapIncrement lidar;
  ImuInterval imu;
  core::ObservationLineage imu_lineage;
};

// Continuous-time densities used by the single batch preintegration path.
// Squaring a density yields the covariance density supplied to GTSAM.
struct MotionImuNoise {
  Eigen::Vector3d gravity_odom{0.0, 0.0, -9.80665};
  double accelerometer_noise_density_mps2_sqrt_hz{0.02};
  double gyroscope_noise_density_radps_sqrt_hz{0.0015};
  double integration_noise_density{1.0e-8};

  // One sensor-calibration prior is applied to the single shared batch bias.
  // The defaults match the AlphaSense-class OKVIS2 profile and the tracking
  // graph's first-bias model. They regularize the gravity/bias ambiguity
  // without duplicating a prior at every initialization state.
  Eigen::Vector3d accelerometer_bias_prior_mean_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyroscope_bias_prior_mean_radps{Eigen::Vector3d::Zero()};
  double accelerometer_bias_prior_sigma_mps2{0.1};
  double gyroscope_bias_prior_sigma_radps{0.01};
};

struct MotionInitializationRequest {
  std::vector<MotionInitializationSegment> segments;
  MotionImuNoise imu_noise;

  enum class Pass {
    RotationDeskewProposal,
    FullDeskewCommitCandidate,
  };
  Pass pass{Pass::FullDeskewCommitCandidate};
};

struct MotionInitializerConfig {
  std::size_t minimum_segments{4U};
  std::size_t maximum_segments{32U};
  std::size_t maximum_imu_knots{8'192U};
  core::Duration minimum_support{2'000'000'000LL};
  core::Duration maximum_support{10'000'000'000LL};
  core::Duration maximum_time_uncertainty{0};
  core::Duration maximum_raw_imu_gap{20'000'000LL};

  // A fast physical excitation pre-gate precedes the more authoritative
  // graph-rank test. Rotation or acceleration may make a batch eligible;
  // the Hessian still decides whether its actual factor geometry is
  // observable and well conditioned.
  // A lightweight pre-gate only; the data Hessian below is the authoritative
  // observability decision. Three centiradians rejects near-static scan noise
  // without excluding long, mostly translational platform starts.
  double minimum_rotation_excitation_rad{0.03};
  double minimum_acceleration_excitation_mps2{0.20};

  double information_basis_orthonormal_tolerance{1.0e-8};
  double information_zero_tolerance{1.0e-12};
  double hessian_absolute_rank_tolerance{1.0e-9};
  double hessian_relative_rank_tolerance{1.0e-10};
  double maximum_supported_hessian_condition{1.0e14};
  double covariance_symmetry_relative_tolerance{1.0e-8};
  double minimum_covariance_eigenvalue{-1.0e-10};

  double position_gauge_sigma_m{1.0e-6};
  double yaw_gauge_sigma_rad{1.0e-6};
  double maximum_bias_prior_mahalanobis{5.0};
  double minimum_imu_conditioning_covariance_inflation{4.0};

  // A moving batch generically has four sensor-only gauges: global
  // translation and yaw. Under weak attitude excitation, as many as two
  // additional roll/pitch--accelerometer-bias modes can remain unresolved by
  // the measurements. Those modes may be initialized from the single
  // calibration prior only when that prior restores the generic data rank;
  // every other missing direction remains a hard rejection.
  std::size_t maximum_prior_resolved_accel_tilt_modes{2U};

  // The most recent LiDAR increment is excluded from the commit-candidate
  // objective and evaluated as a bounded future-registration compatibility
  // check. IMU support remains in the batch through the final state.
  std::size_t holdout_lidar_segments{1U};
  double maximum_lidar_mean_squared_whitened_residual{9.0};
  double maximum_imu_mean_squared_whitened_residual{9.0};
  double maximum_reduced_chi_square{5.0};
  double maximum_holdout_mean_squared_whitened_residual{9.0};

  // Marginals from the idealized batch do not include scan-model mismatch.
  // Scale them by the measured residual dispersion and apply conservative
  // per-state floors before exporting the joint prior.
  double maximum_covariance_residual_inflation{5.0};
  double minimum_orientation_variance_rad2{3.046174197867086e-4};  // (1 deg)^2
  double minimum_velocity_variance_m2ps2{1.0e-2};                  // (0.1 m/s)^2
  double minimum_position_variance_m2{2.5e-3};                     // (0.05 m)^2
  double minimum_gyro_bias_variance_rad2ps2{2.5e-5};               // (0.005 rad/s)^2
  double minimum_accel_bias_variance_m2ps4{2.5e-3};                // (0.05 m/s^2)^2
  std::size_t maximum_solver_iterations{100U};
  double solver_relative_error_tolerance{1.0e-8};
  double solver_absolute_error_tolerance{1.0e-10};
};

enum class MotionInitializationObservabilityClass {
  SensorObservable,
  PriorResolvedAccelerometerTilt,
};

struct MotionInitializationDiagnostics {
  MotionInitializationRequest::Pass pass{
      MotionInitializationRequest::Pass::FullDeskewCommitCandidate};
  std::size_t segments{};
  std::size_t imu_knots{};
  core::Duration support{};
  double rotation_excitation_rad{};
  double acceleration_excitation_mps2{};
  std::size_t scalar_dimension{};
  std::size_t expected_data_rank{};
  std::size_t data_rank{};
  std::size_t calibrated_data_rank{};
  std::size_t full_rank{};
  std::size_t prior_resolved_accel_tilt_modes{};
  MotionInitializationObservabilityClass observability_class{
      MotionInitializationObservabilityClass::SensorObservable};
  double data_supported_condition{};
  double calibrated_data_supported_condition{};
  double full_hessian_condition{};
  double initial_error{};
  double final_error{};
  double lidar_error{};
  double imu_error{};
  double bias_prior_error{};
  double gauge_error{};
  std::size_t lidar_residual_dimension{};
  std::size_t imu_residual_dimension{};
  std::size_t bias_prior_residual_dimension{};
  std::size_t gauge_residual_dimension{};
  std::size_t total_residual_dimension{};
  std::size_t effective_degrees_of_freedom{};
  double lidar_mean_squared_whitened_residual{};
  double imu_mean_squared_whitened_residual{};
  double reduced_chi_square{};
  std::size_t holdout_lidar_segments{};
  std::size_t holdout_residual_dimension{};
  double holdout_error{};
  double holdout_mean_squared_whitened_residual{};
  bool statistically_compatible{};
  bool conditioned_lidar_imu_approximation{};
  double minimum_imu_conditioning_covariance_inflation{1.0};
  double minimum_applied_lidar_covariance_inflation{1.0};
  double covariance_residual_inflation{1.0};
  std::size_t deskew_solve_passes{1U};
  std::size_t refined_sweeps{};
  std::size_t refined_registrations{};
  std::size_t refined_deskew_pose_interpolations{};
  double refined_total_registration_cost{};
  double refined_maximum_registration_cost{};
  double bias_prior_mahalanobis{};
  std::size_t solver_iterations{};
};

enum class MotionInitializationLineageRole {
  FusedResidual,
  ValidationGate,
};

struct MotionInitializationLineageUse {
  core::ObservationLineageId lineage;
  MotionInitializationLineageRole role{MotionInitializationLineageRole::FusedResidual};
};

struct MotionInitialization {
  core::FusionTime exact_time;
  core::NavStateEstimate state;
  NavigationCovariance covariance;

  // Alignment from the LiDAR bootstrap odometry frame into the returned,
  // gravity-aligned odom frame. Its translation and yaw define the gauge.
  core::Pose3d T_odom_bootstrap;
  MotionInitializationDiagnostics diagnostics;
  // Discrete optimized states at every retained LiDAR reference time. This is
  // consumed privately by the full-deskew refinement pass; no bootstrap
  // factor is exported to the online graph.
  std::vector<TimedNavState> reference_states;
  std::vector<MotionInitializationLineageUse> lineage_uses;
};

enum class MotionInitializationErrorCode {
  InvalidConfig,
  SegmentCountOutOfBounds,
  SupportDurationOutOfBounds,
  NonContiguousSupport,
  InexactImuSupport,
  InvalidImuSupport,
  InvalidRelativePose,
  InvalidRankAwareInformation,
  InvalidLineage,
  LineageSupportMismatch,
  InsufficientExcitation,
  RankDeficientBatch,
  IllConditionedBatch,
  SolverFailure,
  SolverDidNotConverge,
  BiasPlausibilityFailed,
  StatisticalCompatibilityFailed,
  HoldoutCompatibilityFailed,
  NonFiniteSolution,
  MarginalCovarianceFailure,
};

struct MotionInitializationError {
  MotionInitializationErrorCode code{};
  std::optional<std::size_t> segment;
  std::string detail;
};

// ROS-free, bounded, one-shot LiDAR--IMU initialization. Optimizer types stay
// private to the implementation; the public result uses only framework APIs.
class MotionInitializer {
public:
  explicit MotionInitializer(MotionInitializerConfig config = {});
  MotionInitializer(MotionInitializerConfig config,
                    std::shared_ptr<LocalPipelineTimingRecorder> timing);

  [[nodiscard]] core::Result<MotionInitialization, MotionInitializationError> initialize(
      const MotionInitializationRequest& request) const;

private:
  MotionInitializerConfig config_;
  std::shared_ptr<LocalPipelineTimingRecorder> timing_;
};

}  // namespace meridian::local
