#include "meridian/local/lidar_bootstrap.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <utility>
#include <variant>
#include <vector>

#include "pipeline_timing_internal.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] LidarBootstrapError bootstrapError(
    LidarBootstrapErrorCode code, std::string detail,
    std::optional<LidarPreprocessError> preprocessing = std::nullopt,
    std::optional<LidarRegistrationError> registration = std::nullopt) {
  return LidarBootstrapError{code, std::move(detail), std::move(preprocessing),
                             std::move(registration)};
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool validPreprocessConfig(const LidarPreprocessConfig& config) noexcept {
  return config.parallel_worker_count >= 1U && config.parallel_worker_count <= 64U &&
         config.parallel_worker_count <=
             static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         std::isfinite(config.minimum_range_m) && std::isfinite(config.maximum_range_m) &&
         config.minimum_range_m >= 0.0 && config.maximum_range_m > config.minimum_range_m &&
         finitePositive(config.voxel_size_m) &&
         config.maximum_output_points > 0U;
}

[[nodiscard]] bool validConfig(const LidarBootstrapOdometryConfig& config) noexcept {
  const double covariance_inflation = config.target_reuse_covariance_inflation *
                                      config.imu_conditioning_covariance_inflation;
  return config.maximum_sweeps >= 2U &&
         config.maximum_sweeps < static_cast<std::size_t>(core::StateId::kInvalidValue) &&
         validPreprocessConfig(config.preprocessing) &&
         isValidLidarRegistrationConfig(config.registration) &&
         finitePositive(config.maximum_increment_translation_m) &&
         finitePositive(config.maximum_increment_rotation_rad) &&
         finitePositive(config.maximum_observable_condition) &&
         std::isfinite(config.target_reuse_covariance_inflation) &&
         config.target_reuse_covariance_inflation >= 1.0 &&
         std::isfinite(config.imu_conditioning_covariance_inflation) &&
         config.imu_conditioning_covariance_inflation >= 1.0 &&
         std::isfinite(covariance_inflation) && covariance_inflation > 1.0 &&
         finitePositive(config.registration.maximum_translation_information /
                        covariance_inflation);
}

struct CanonicalPhysicalInformation {
  core::RankAwareInformation information;
  double supported_condition{};
};

[[nodiscard]] core::Result<CanonicalPhysicalInformation, LidarBootstrapError>
canonicalPhysicalInformation(const LidarRegistrationResult& registration) {
  using Result = core::Result<CanonicalPhysicalInformation, LidarBootstrapError>;
  const std::size_t rank = registration.diagnostics.observable_rank;
  if (rank == 0U || rank > 6U || registration.diagnostics.physical_information.rank != rank ||
      !registration.projected_physical_information.allFinite()) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::IncrementGateFailed,
        "direct point ICP returned inconsistent physical-information rank or non-finite information"));
  }

  const Eigen::Matrix<double, 6, 6> symmetric =
      0.5 * (registration.projected_physical_information +
             registration.projected_physical_information.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
      !solver.eigenvectors().allFinite()) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::IncrementGateFailed,
                       "direct point ICP physical-information canonicalization eigensolver failed"));
  }

  CanonicalPhysicalInformation output;
  output.information.rank = rank;
  for (std::size_t output_index = 0U; output_index < 6U; ++output_index) {
    const Eigen::Index input_index = 5 - static_cast<Eigen::Index>(output_index);
    Eigen::Matrix<double, 6, 1> direction = solver.eigenvectors().col(input_index);
    Eigen::Index pivot = 0;
    direction.cwiseAbs().maxCoeff(&pivot);
    if (direction(pivot) < 0.0) {
      direction = -direction;
    }
    output.information.basis.col(static_cast<Eigen::Index>(output_index)) = direction;
    if (output_index < rank) {
      const double eigenvalue = solver.eigenvalues()(input_index);
      if (!finitePositive(eigenvalue)) {
        return Result::failure(
            bootstrapError(LidarBootstrapErrorCode::IncrementGateFailed,
                           "direct point ICP supported physical-information direction is non-positive"));
      }
      output.information.eigenvalues(static_cast<Eigen::Index>(output_index)) = eigenvalue;
    }
  }
  const double largest = output.information.eigenvalues(0);
  const double smallest =
      output.information.eigenvalues(static_cast<Eigen::Index>(output.information.rank - 1U));
  output.supported_condition = largest / smallest;
  if (!finitePositive(output.supported_condition)) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::IncrementGateFailed,
                       "direct point ICP supported physical-information condition is invalid"));
  }
  return Result::success(std::move(output));
}

[[nodiscard]] core::StateId bootstrapStateId(std::size_t retained_index) noexcept {
  return core::StateId{static_cast<std::uint64_t>(retained_index) + 1U};
}

[[nodiscard]] bool validSweep(const core::LidarSweep& sweep,
                              core::CalibrationEpoch calibration) noexcept {
  if (!sweep.id.valid() || !sweep.lidar.valid() || !sweep.stamp.source_epoch.valid() ||
      sweep.stamp.status == core::TimeMappingStatus::Discontinuous || !sweep.acquisition.valid() ||
      !sweep.layout.organized || sweep.layout.width == 0U || sweep.layout.height == 0U ||
      !sweep.points || sweep.points->empty() || sweep.header.direct_calibration != calibration) {
    return false;
  }
  return sweep.points->size() <= sweep.layout.sourcePointCount();
}

[[nodiscard]] bool finiteState(const core::NavStateEstimate& state) noexcept {
  return state.T_odom_imu.matrix().allFinite() && state.velocity_odom.allFinite() &&
         state.gyro_bias.allFinite() && state.accel_bias.allFinite();
}

[[nodiscard]] bool exactAcquisitionSupport(const ImuInterval& interval,
                                           const core::LidarSweep& sweep) noexcept {
  if (!interval.support.valid() || interval.support.start != sweep.acquisition.start ||
      interval.support.end != sweep.acquisition.end || interval.knots.size() < 2U ||
      interval.knots.front().time != sweep.acquisition.start ||
      interval.knots.back().time != sweep.acquisition.end || interval.raw_measurements.empty() ||
      interval.contains_saturation) {
    return false;
  }
  for (std::size_t index = 0U; index < interval.knots.size(); ++index) {
    const InterpolatedImuSample& knot = interval.knots[index];
    if (!knot.specific_force_mps2.allFinite() || !knot.angular_velocity_radps.allFinite() ||
        (index > 0U && knot.time <= interval.knots[index - 1U].time)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::vector<TimedNavState> rotationOnlyTrajectory(const ImuInterval& interval,
                                                                const Eigen::Vector3d& gyro_bias) {
  std::vector<TimedNavState> trajectory;
  trajectory.reserve(interval.knots.size());
  core::NavStateEstimate state;
  state.gyro_bias = gyro_bias;
  trajectory.push_back(TimedNavState{interval.knots.front().time, state});
  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const InterpolatedImuSample& previous = interval.knots[index - 1U];
    const InterpolatedImuSample& current = interval.knots[index];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    const Eigen::Vector3d omega =
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps) - gyro_bias;
    state.T_odom_imu = core::Pose3d(state.T_odom_imu.so3() * Sophus::SO3d::exp(omega * dt),
                                    Eigen::Vector3d::Zero());
    trajectory.push_back(TimedNavState{current.time, state});
  }
  return trajectory;
}

[[nodiscard]] core::Result<std::vector<TimedNavState>, LidarBootstrapError>
trajectoryEndingAtReference(const ImuInterval& interval, const core::NavStateEstimate& reference,
                            const Eigen::Vector3d& gravity_odom) {
  using Result = core::Result<std::vector<TimedNavState>, LidarBootstrapError>;
  if (!finiteState(reference) || !gravity_odom.allFinite() || interval.knots.empty() ||
      interval.knots.back().time != interval.support.end) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                       "full-deskew reference state, gravity, or acquisition support is invalid"));
  }
  std::vector<TimedNavState> trajectory(interval.knots.size());
  core::NavStateEstimate state = reference;
  trajectory.back() = TimedNavState{interval.knots.back().time, state};
  for (std::size_t right = interval.knots.size() - 1U; right > 0U; --right) {
    const InterpolatedImuSample& previous = interval.knots[right - 1U];
    const InterpolatedImuSample& current = interval.knots[right];
    const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
    if (!std::isfinite(dt) || dt <= 0.0) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::InvalidAcquisitionImu,
                         "full-deskew reverse propagation encountered a non-positive time step"));
    }
    const Eigen::Vector3d omega =
        0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps) - state.gyro_bias;
    const Sophus::SO3d rotation_current = state.T_odom_imu.so3();
    const Sophus::SO3d rotation_previous = rotation_current * Sophus::SO3d::exp(-omega * dt);
    const Eigen::Vector3d acceleration_previous =
        rotation_previous * (previous.specific_force_mps2 - state.accel_bias) + gravity_odom;
    const Eigen::Vector3d acceleration_current =
        rotation_current * (current.specific_force_mps2 - state.accel_bias) + gravity_odom;
    const Eigen::Vector3d acceleration_mid = 0.5 * (acceleration_previous + acceleration_current);
    const Eigen::Vector3d velocity_previous = state.velocity_odom - acceleration_mid * dt;
    const Eigen::Vector3d position_previous =
        state.T_odom_imu.translation() - velocity_previous * dt - 0.5 * acceleration_mid * dt * dt;
    state.velocity_odom = velocity_previous;
    state.T_odom_imu = core::Pose3d(rotation_previous, position_previous);
    if (!finiteState(state)) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::DeskewFailed,
                         "full-deskew reverse propagation produced a non-finite state"));
    }
    trajectory[right - 1U] = TimedNavState{previous.time, state};
  }
  return Result::success(std::move(trajectory));
}

[[nodiscard]] core::Result<DeskewedSweep, LidarBootstrapError> rotationDeskewedSweep(
    const core::LidarSweep& sweep, const core::Pose3d& T_imu_lidar,
    const ImuInterval& acquisition_imu, const Eigen::Vector3d& gyro_bias) {
  using Result = core::Result<DeskewedSweep, LidarBootstrapError>;
  auto deskewed = deskewLidarSweep(sweep, sweep.acquisition.end, T_imu_lidar,
                                   rotationOnlyTrajectory(acquisition_imu, gyro_bias),
                                   acquisition_imu.raw_measurements);
  if (!deskewed) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::DeskewFailed,
                       "bootstrap rotation deskew failed: " + deskewed.error().detail));
  }
  return Result::success(std::move(deskewed).value());
}

[[nodiscard]] core::ObservationLineage cloudLineage(core::MeasurementId source,
                                                    std::span<const core::MeasurementId> imu_support,
                                                    core::CalibrationEpoch calibration,
                                                    core::DerivedRecordId consumer,
                                                    core::ObservationLineageId lineage_id) {
  core::ObservationSlice slice;
  slice.root = source;
  slice.kind = core::SliceKind::Whole;
  slice.calibration = calibration;
  core::ObservationLineage lineage;
  lineage.id = lineage_id;
  lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::DerivedSummary,
                                                 consumer, std::nullopt, std::nullopt});
  std::set<core::MeasurementId> canonical_support(imu_support.begin(), imu_support.end());
  for (const core::MeasurementId measurement : canonical_support) {
    core::ObservationSlice conditioning;
    conditioning.root = measurement;
    conditioning.kind = core::SliceKind::Whole;
    conditioning.calibration = calibration;
    lineage.usage.push_back(core::ObservationUsage{
        conditioning, core::ObservationRole::ConditioningOnly, consumer, std::nullopt,
        std::nullopt});
  }
  return lineage;
}

[[nodiscard]] bool lineageUses(const core::ObservationLineage& lineage,
                               core::MeasurementId measurement, core::ObservationRole role) {
  return std::any_of(lineage.usage.begin(), lineage.usage.end(),
                     [&](const core::ObservationUsage& usage) {
                       return usage.role == role &&
                              std::holds_alternative<core::MeasurementId>(usage.slice.root) &&
                              std::get<core::MeasurementId>(usage.slice.root) == measurement;
                     });
}

[[nodiscard]] bool validFactorLineages(const core::ObservationLineage& lidar,
                                       const core::ObservationLineage& imu,
                                       core::MeasurementId previous, core::MeasurementId current,
                                       const ImuInterval& previous_acquisition,
                                       const ImuInterval& current_acquisition,
                                       const ImuInterval& segment_interval,
                                       double expected_covariance_inflation,
                                       double expected_information_cap) {
  const auto conditions_on = [&](const ImuInterval& support) {
    return std::all_of(support.raw_measurements.begin(), support.raw_measurements.end(),
                       [&](core::MeasurementId measurement) {
                         return lineageUses(lidar, measurement,
                                            core::ObservationRole::ConditioningOnly);
                       });
  };
  const auto declared_policy = [&](const core::ObservationUsage& usage) {
    if (usage.role != core::ObservationRole::ConditioningOnly || !usage.correlation_group) {
      return usage.role != core::ObservationRole::ConditioningOnly;
    }
    const auto declaration = std::find_if(lidar.correlations.begin(), lidar.correlations.end(),
                                          [&](const core::CorrelationDeclaration& candidate) {
                                            return candidate.group == *usage.correlation_group;
                                          });
    const double inflation_tolerance = 1.0e-12 * std::max(1.0, expected_covariance_inflation);
    const double cap_tolerance = 1.0e-12 * std::max(1.0, expected_information_cap);
    return declaration != lidar.correlations.end() &&
           declaration->policy == core::CorrelationPolicyRevision{1U} &&
           declaration->treatment ==
               core::CorrelationTreatment::CovarianceInflationAndInformationCap &&
           std::abs(declaration->covariance_inflation - expected_covariance_inflation) <=
               inflation_tolerance &&
           declaration->total_information_cap &&
           std::abs(*declaration->total_information_cap - expected_information_cap) <=
               cap_tolerance;
  };
  const bool conditioning_policy_matches =
      std::all_of(lidar.usage.begin(), lidar.usage.end(), declared_policy);
  return lidar.id.valid() && imu.id.valid() && lidar.id != imu.id &&
         core::validateLineage(lidar) == core::LineageValidationError::None &&
         core::validateLineage(imu) == core::LineageValidationError::None &&
         lineageUses(lidar, current, core::ObservationRole::PrimaryResidual) &&
         lineageUses(lidar, previous, core::ObservationRole::ConditioningOnly) &&
         conditions_on(previous_acquisition) && conditions_on(current_acquisition) &&
         conditions_on(segment_interval) && conditioning_policy_matches &&
         !core::lineagesAreIndependent(lidar, imu);
}

[[nodiscard]] bool exactSegmentSupport(const ImuInterval& interval, core::FusionTime start,
                                       core::FusionTime end) noexcept {
  return interval.support.start == start && interval.support.end == end &&
         interval.support.valid() && interval.knots.size() >= 2U &&
         interval.knots.front().time == start && interval.knots.back().time == end &&
         !interval.raw_measurements.empty() && !interval.contains_saturation;
}

}  // namespace

struct LidarBootstrapOdometry::Impl {
  struct RetainedSweep {
    core::LidarSweep sweep;
    core::DerivedRecordId cloud_record;
    core::ObservationLineageId cloud_lineage;
    ImuInterval acquisition_imu;
    std::shared_ptr<const LidarRegistrationCloud> rotation_deskewed_cloud;
    std::optional<MotionInitializationSegment> segment_from_previous;
  };

  core::CalibrationEpoch calibration;
  core::Pose3d T_imu_lidar;
  Eigen::Vector3d gravity_odom{0.0, 0.0, -9.80665};
  Eigen::Vector3d gyro_bias_prior_mean_radps{Eigen::Vector3d::Zero()};
  LidarBootstrapOdometryConfig config;
  std::size_t sweeps{};
  std::optional<core::MeasurementId> previous_measurement;
  std::optional<core::FusionTime> previous_time;
  std::shared_ptr<const LidarRegistrationCloud> previous_cloud;
  core::Pose3d T_bootstrap_imu;
  std::optional<core::Pose3d> previous_increment;
  std::optional<core::Duration> previous_increment_duration;
  std::vector<RetainedSweep> retained;
  std::shared_ptr<LocalPipelineTimingRecorder> timing;
};

LidarBootstrapOdometry::LidarBootstrapOdometry(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

LidarBootstrapOdometry::~LidarBootstrapOdometry() = default;
LidarBootstrapOdometry::LidarBootstrapOdometry(LidarBootstrapOdometry&&) noexcept = default;
LidarBootstrapOdometry& LidarBootstrapOdometry::operator=(LidarBootstrapOdometry&&) noexcept =
    default;

core::Result<LidarBootstrapOdometry, LidarBootstrapError> LidarBootstrapOdometry::create(
    core::CalibrationEpoch calibration, core::Pose3d T_imu_lidar, Eigen::Vector3d gravity_odom,
    Eigen::Vector3d gyro_bias_prior_mean_radps, LidarBootstrapOdometryConfig config) {
  return create(calibration, std::move(T_imu_lidar), std::move(gravity_odom),
                std::move(gyro_bias_prior_mean_radps), std::move(config), {});
}

core::Result<LidarBootstrapOdometry, LidarBootstrapError> LidarBootstrapOdometry::create(
    core::CalibrationEpoch calibration, core::Pose3d T_imu_lidar, Eigen::Vector3d gravity_odom,
    Eigen::Vector3d gyro_bias_prior_mean_radps, LidarBootstrapOdometryConfig config,
    std::shared_ptr<LocalPipelineTimingRecorder> timing) {
  using Result = core::Result<LidarBootstrapOdometry, LidarBootstrapError>;
  if (!validConfig(config)) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::InvalidConfig,
        "LiDAR bootstrap capacities, point-cloud preprocessing, direct ICP, or motion gates "
        "are invalid"));
  }
  if (!calibration.valid() || !T_imu_lidar.matrix().allFinite() || !gravity_odom.allFinite() ||
      !finitePositive(gravity_odom.norm()) || !gyro_bias_prior_mean_radps.allFinite()) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::InvalidCalibration,
        "LiDAR bootstrap requires valid calibration, gravity, gyro-bias mean, and T_imu_lidar"));
  }
  auto implementation = std::make_unique<Impl>();
  implementation->calibration = calibration;
  implementation->T_imu_lidar = std::move(T_imu_lidar);
  implementation->gravity_odom = std::move(gravity_odom);
  implementation->gyro_bias_prior_mean_radps = std::move(gyro_bias_prior_mean_radps);
  implementation->config = std::move(config);
  // Bootstrap is deliberately adjacent-only. A production rolling target may
  // bind several owned target poses, but a motion-initialization increment may
  // never merge or silently reuse older pose geometry.
  implementation->config.registration.maximum_targets = 1U;
  implementation->timing = std::move(timing);
  implementation->retained.reserve(implementation->config.maximum_sweeps);
  return Result::success(LidarBootstrapOdometry(std::move(implementation)));
}

core::Result<LidarBootstrapCommit, LidarBootstrapError> LidarBootstrapOdometry::add(
    LidarBootstrapInput input) {
  using Result = core::Result<LidarBootstrapCommit, LidarBootstrapError>;
  Impl& state = *implementation_;
  if (state.sweeps >= state.config.maximum_sweeps) {
    return Result::failure(bootstrapError(LidarBootstrapErrorCode::Capacity,
                                          "LiDAR bootstrap sweep capacity is exhausted"));
  }
  if (!validSweep(input.sweep, state.calibration) || !input.cloud_record.valid() ||
      !input.cloud_lineage.valid()) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::InvalidSweep,
        "bootstrap sweep, calibration, registration-cloud record, or lineage identity is invalid"));
  }
  if (!exactAcquisitionSupport(input.acquisition_imu, input.sweep)) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::InvalidAcquisitionImu,
        "bootstrap sweep requires finite exact IMU support over its full acquisition interval"));
  }
  const core::FusionTime reference_time = input.sweep.acquisition.end;
  if (state.previous_time &&
      (reference_time <= *state.previous_time || input.sweep.id <= *state.previous_measurement)) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::NonMonotonicSweep,
                       "bootstrap sweep identities and reference times must increase strictly"));
  }
  const bool first = state.sweeps == 0U;
  const bool segment_fields_present = input.between_reference_imu.has_value() &&
                                      input.lidar_factor_lineage.has_value() &&
                                      input.imu_factor_lineage.has_value();
  if ((first &&
       (input.between_reference_imu || input.lidar_factor_lineage || input.imu_factor_lineage)) ||
      (!first && !segment_fields_present)) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::InvalidSegmentSupport,
                       "the first bootstrap sweep has no segment; every later sweep requires exact "
                       "IMU support and two factor lineages"));
  }

  if (!first) {
    const ImuInterval& imu = *input.between_reference_imu;
    if (!exactSegmentSupport(imu, *state.previous_time, reference_time)) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::InvalidSegmentSupport,
                         "bootstrap IMU support must exactly span adjacent LiDAR reference times"));
    }
    if (!validFactorLineages(*input.lidar_factor_lineage, *input.imu_factor_lineage,
                             *state.previous_measurement, input.sweep.id,
                             state.retained.back().acquisition_imu, input.acquisition_imu, imu,
                             state.config.target_reuse_covariance_inflation *
                                 state.config.imu_conditioning_covariance_inflation,
                             state.config.registration.maximum_translation_information /
                                 (state.config.target_reuse_covariance_inflation *
                                  state.config.imu_conditioning_covariance_inflation))) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::InvalidLineage,
                         "bootstrap LiDAR lineage must bind source/target scans and declare every "
                         "reused raw IMU sample as conditioning data"));
    }
  }

  const ImuInterval* segment_imu = first ? nullptr : std::addressof(*input.between_reference_imu);
  core::Pose3d relative_seed;
  if (!first && state.previous_increment && state.previous_increment_duration &&
      state.previous_increment_duration->nanoseconds > 0) {
    const double scale =
        std::clamp(static_cast<double>(segment_imu->support.duration().nanoseconds) /
                       static_cast<double>(state.previous_increment_duration->nanoseconds),
                   0.25, 4.0);
    relative_seed = core::Pose3d::exp(scale * state.previous_increment->log());
  }
  const core::Pose3d current_pose_seed =
      first ? core::Pose3d{} : state.T_bootstrap_imu * relative_seed;

  core::PipelineWorkIdentity work;
  work.measurement = input.sweep.id;
  const core::ThreadCpuWallTimer acquisition_deskew_timer;
  auto rotation_deskewed = rotationDeskewedSweep(
      input.sweep, state.T_imu_lidar, input.acquisition_imu, state.gyro_bias_prior_mean_radps);
  detail::observeLocalPipelineTiming(
      state.timing, LocalPipelineTimingStage::BootstrapAcquisitionDeskew, acquisition_deskew_timer,
      rotation_deskewed ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
      work);
  if (!rotation_deskewed) {
    return Result::failure(rotation_deskewed.error());
  }
  // The rotation-only trajectory supplies the exact point-time transforms but
  // has an arbitrary local pose gauge. Registration identity must instead use
  // the committed bootstrap target pose and the constant-twist source seed.
  rotation_deskewed.value().T_odom_imu_reference = current_pose_seed;

  core::ObservationLineage current_lineage =
      cloudLineage(input.sweep.id, rotation_deskewed.value().imu_support, state.calibration,
                   input.cloud_record, input.cloud_lineage);
  if (core::validateLineage(current_lineage) != core::LineageValidationError::None) {
    return Result::failure(bootstrapError(LidarBootstrapErrorCode::InvalidLineage,
                                          "bootstrap registration-cloud ancestry is invalid"));
  }

  const std::size_t deskew_pose_interpolations = rotation_deskewed.value().pose_interpolations;
  const core::ThreadCpuWallTimer cloud_timer;
  auto cloud_result = buildLidarRegistrationCloud(
      std::move(rotation_deskewed).value(), current_lineage, state.config.preprocessing,
      LidarRegistrationIndexConfig{state.config.registration.target_voxel_resolution_m});
  detail::observeLocalPipelineTiming(
      state.timing, LocalPipelineTimingStage::RegistrationViewBuild, cloud_timer,
      cloud_result ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
      work);
  if (!cloud_result) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::RegistrationCloudBuildFailed,
        "LiDAR bootstrap registration-cloud construction failed: " + cloud_result.error().detail,
        cloud_result.error()));
  }
  auto current_cloud = std::move(cloud_result).value();

  LidarBootstrapCommit commit;
  commit.reference_time = reference_time;
  commit.preprocessing = current_cloud->stats;
  commit.deskew_pose_interpolations = deskew_pose_interpolations;
  if (first) {
    state.previous_measurement = input.sweep.id;
    state.previous_time = reference_time;
    state.previous_cloud = current_cloud;
    state.T_bootstrap_imu = core::Pose3d{};
    state.sweeps = 1U;
    state.retained.push_back(
        Impl::RetainedSweep{std::move(input.sweep), input.cloud_record, input.cloud_lineage,
                            std::move(input.acquisition_imu), state.previous_cloud, std::nullopt});
    commit.disposition = LidarBootstrapDisposition::AnchorCreated;
    commit.T_bootstrap_imu = state.T_bootstrap_imu;
    commit.retained_sweeps = state.sweeps;
    return Result::success(std::move(commit));
  }

  const ImuInterval& imu = *segment_imu;
  const core::StateId target_state = bootstrapStateId(state.sweeps - 1U);
  const core::StateId source_state = bootstrapStateId(state.sweeps);
  const std::array<LidarRegistrationTarget, 1U> targets{LidarRegistrationTarget{
      target_state, *state.previous_time, state.previous_cloud, state.T_bootstrap_imu,
      state.T_bootstrap_imu.inverse() * current_cloud->T_odom_imu_seed}};
  const core::ThreadCpuWallTimer registration_timer;
  auto registration =
      registerLidarScan(source_state, current_cloud, targets, state.config.registration);
  detail::observeLocalPipelineTiming(
      state.timing, LocalPipelineTimingStage::CorrespondenceRegistrationSolve, registration_timer,
      registration ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
      work);
  if (!registration) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::RegistrationFailed,
        "IMU-conditioned LiDAR bootstrap direct ICP failed: " + registration.error().detail,
        std::nullopt,
        registration.error()));
  }
  const core::Pose3d cumulative = registration.value().T_odom_source;
  const core::Pose3d increment = state.T_bootstrap_imu.inverse() * cumulative;
  auto canonical_information = canonicalPhysicalInformation(registration.value());
  if (!canonical_information) {
    return Result::failure(canonical_information.error());
  }
  const Eigen::Matrix<double, 6, 1> tangent = increment.log();
  if (increment.translation().norm() > state.config.maximum_increment_translation_m ||
      tangent.tail<3>().norm() > state.config.maximum_increment_rotation_rad ||
      canonical_information.value().supported_condition >
          state.config.maximum_observable_condition ||
      canonical_information.value().information.rank <
          state.config.registration.minimum_observable_rank) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::IncrementGateFailed,
        "LiDAR bootstrap increment exceeds motion, projected-rank, or projected-condition gates"));
  }

  const core::Duration segment_duration = imu.support.duration();
  core::RankAwareInformation effective_information = canonical_information.value().information;
  const double conditioning_inflation = state.config.target_reuse_covariance_inflation *
                                        state.config.imu_conditioning_covariance_inflation;
  effective_information.eigenvalues /= conditioning_inflation;
  MotionInitializationSegment segment;
  segment.lidar = LidarBootstrapIncrement{*state.previous_time,
                                          reference_time,
                                          increment,
                                          effective_information,
                                          std::move(*input.lidar_factor_lineage),
                                          state.config.imu_conditioning_covariance_inflation,
                                          conditioning_inflation};
  segment.imu = std::move(*input.between_reference_imu);
  segment.imu_lineage = std::move(*input.imu_factor_lineage);

  // The cloud retains the proposal/preprocessing seed that defined its frozen
  // geometry. The separately owned committed target pose is carried by the
  // bootstrap state and the next pose-aware target record.
  const std::shared_ptr<const LidarRegistrationCloud> committed_cloud = current_cloud;

  state.previous_measurement = input.sweep.id;
  state.previous_time = reference_time;
  state.previous_cloud = committed_cloud;
  state.T_bootstrap_imu = cumulative;
  state.previous_increment = increment;
  state.previous_increment_duration = segment_duration;
  ++state.sweeps;
  state.retained.push_back(
      Impl::RetainedSweep{std::move(input.sweep), input.cloud_record, input.cloud_lineage,
                          std::move(input.acquisition_imu), state.previous_cloud, segment});

  commit.disposition = LidarBootstrapDisposition::IncrementCommitted;
  commit.T_bootstrap_imu = cumulative;
  commit.registration = std::move(registration).value();
  commit.segment = std::move(segment);
  commit.retained_sweeps = state.sweeps;
  return Result::success(std::move(commit));
}

core::Result<LidarBootstrapRefinement, LidarBootstrapError> LidarBootstrapOdometry::refine(
    const std::vector<TimedNavState>& reference_states) const {
  using Result = core::Result<LidarBootstrapRefinement, LidarBootstrapError>;
  const Impl& state = *implementation_;
  core::PipelineWorkIdentity refinement_work;
  if (!state.retained.empty()) {
    refinement_work.measurement = state.retained.back().sweep.id;
  }
  detail::LocalPipelineTimingScope refinement_timing(
      state.timing, LocalPipelineTimingStage::MotionBatchSolveRefinement, refinement_work);
  if (reference_states.size() < 2U || reference_states.size() > state.retained.size()) {
    return Result::failure(bootstrapError(
        LidarBootstrapErrorCode::RefinementStateMismatch,
        "full-deskew refinement requires two or more states within the retained sweep bound"));
  }
  for (std::size_t index = 0U; index < reference_states.size(); ++index) {
    if (!finiteState(reference_states[index].state) ||
        (index > 0U && reference_states[index].time <= reference_states[index - 1U].time)) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                         "full-deskew refinement states must be finite and strictly ordered"));
    }
  }

  const auto first = std::find_if(
      state.retained.begin(), state.retained.end(), [&](const Impl::RetainedSweep& retained) {
        return retained.sweep.acquisition.end == reference_states.front().time;
      });
  if (first == state.retained.end()) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                       "first provisional state does not identify a retained bootstrap sweep"));
  }
  const std::size_t first_index =
      static_cast<std::size_t>(std::distance(state.retained.begin(), first));
  if (first_index + reference_states.size() > state.retained.size()) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                       "provisional state sequence extends beyond the retained bootstrap window"));
  }
  for (std::size_t index = 0U; index < reference_states.size(); ++index) {
    if (state.retained[first_index + index].sweep.acquisition.end != reference_states[index].time) {
      return Result::failure(bootstrapError(
          LidarBootstrapErrorCode::RefinementStateMismatch,
          "provisional state times do not match one contiguous retained sweep sequence"));
    }
  }

  LidarBootstrapRefinement output;
  output.segments.reserve(reference_states.size() - 1U);
  output.diagnostics.sweeps = reference_states.size();
  output.diagnostics.minimum_observable_rank = 6U;
  output.diagnostics.imu_conditioning_covariance_inflation =
      state.config.imu_conditioning_covariance_inflation;
  output.diagnostics.applied_covariance_inflation =
      state.config.target_reuse_covariance_inflation *
      state.config.imu_conditioning_covariance_inflation;

  std::shared_ptr<const LidarRegistrationCloud> previous_cloud;
  for (std::size_t index = 0U; index < reference_states.size(); ++index) {
    const Impl::RetainedSweep& retained = state.retained[first_index + index];
    auto trajectory = trajectoryEndingAtReference(
        retained.acquisition_imu, reference_states[index].state, state.gravity_odom);
    if (!trajectory) {
      return Result::failure(trajectory.error());
    }
    core::PipelineWorkIdentity work;
    work.measurement = retained.sweep.id;
    const core::ThreadCpuWallTimer acquisition_deskew_timer;
    auto deskewed =
        deskewLidarSweep(retained.sweep, retained.sweep.acquisition.end, state.T_imu_lidar,
                         trajectory.value(), retained.acquisition_imu.raw_measurements);
    detail::observeLocalPipelineTiming(
        state.timing, LocalPipelineTimingStage::BootstrapAcquisitionDeskew,
        acquisition_deskew_timer,
        deskewed ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed, work);
    if (!deskewed) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::DeskewFailed,
                         "full bootstrap deskew failed: " + deskewed.error().detail));
    }
    // Refinement identity is the supplied provisional state, not a cumulative
    // registration side effect. Every adjacent direct-ICP pair is rebuilt against
    // these fixed pose-owned cloud frames.
    deskewed.value().T_odom_imu_reference = reference_states[index].state.T_odom_imu;
    output.diagnostics.deskew_pose_interpolations += deskewed.value().pose_interpolations;
    output.final_lineage =
        cloudLineage(retained.sweep.id, deskewed.value().imu_support, state.calibration,
                     retained.cloud_record, retained.cloud_lineage);
    if (core::validateLineage(output.final_lineage) != core::LineageValidationError::None) {
      return Result::failure(bootstrapError(LidarBootstrapErrorCode::InvalidLineage,
                                            "full-deskew registration-cloud ancestry is invalid"));
    }
    const core::ThreadCpuWallTimer cloud_timer;
    auto cloud_result = buildLidarRegistrationCloud(
        std::move(deskewed).value(), output.final_lineage, state.config.preprocessing,
        LidarRegistrationIndexConfig{state.config.registration.target_voxel_resolution_m});
    detail::observeLocalPipelineTiming(
        state.timing, LocalPipelineTimingStage::RegistrationViewBuild, cloud_timer,
        cloud_result ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
        work);
    if (!cloud_result) {
      return Result::failure(bootstrapError(
          LidarBootstrapErrorCode::RegistrationCloudBuildFailed,
          "full-deskew registration-cloud construction failed: " + cloud_result.error().detail,
          cloud_result.error()));
    }
    auto current_cloud = std::move(cloud_result).value();
    output.final_cloud = current_cloud;
    if (index == 0U) {
      previous_cloud = std::move(current_cloud);
      continue;
    }

    const core::StateId target_state = bootstrapStateId(first_index + index - 1U);
    const core::StateId source_state = bootstrapStateId(first_index + index);
    const std::array<LidarRegistrationTarget, 1U> targets{LidarRegistrationTarget{
        target_state, reference_states[index - 1U].time, previous_cloud,
        reference_states[index - 1U].state.T_odom_imu,
        reference_states[index - 1U].state.T_odom_imu.inverse() * current_cloud->T_odom_imu_seed}};
    const core::ThreadCpuWallTimer registration_timer;
    auto registration =
        registerLidarScan(source_state, current_cloud, targets, state.config.registration);
    detail::observeLocalPipelineTiming(
        state.timing, LocalPipelineTimingStage::CorrespondenceRegistrationSolve, registration_timer,
        registration ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
        work);
    if (!registration) {
      std::ostringstream detail;
      detail << "full-deskew registration " << (index - 1U)
             << " failed: " << registration.error().detail;
      return Result::failure(bootstrapError(LidarBootstrapErrorCode::RegistrationFailed, detail.str(),
                                            std::nullopt, registration.error()));
    }
    const core::Pose3d increment = reference_states[index - 1U].state.T_odom_imu.inverse() *
                                   registration.value().T_odom_source;
    auto canonical_information = canonicalPhysicalInformation(registration.value());
    if (!canonical_information) {
      return Result::failure(canonical_information.error());
    }
    const Eigen::Matrix<double, 6, 1> tangent = increment.log();
    if (increment.translation().norm() > state.config.maximum_increment_translation_m ||
        tangent.tail<3>().norm() > state.config.maximum_increment_rotation_rad ||
        canonical_information.value().supported_condition >
            state.config.maximum_observable_condition ||
        canonical_information.value().information.rank <
            state.config.registration.minimum_observable_rank) {
      return Result::failure(bootstrapError(
          LidarBootstrapErrorCode::IncrementGateFailed,
          "full-deskew registration exceeds motion, projected-rank, or projected-condition "
          "gates"));
    }
    if (!retained.segment_from_previous) {
      return Result::failure(
          bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                         "retained source sweep has no first-pass segment metadata"));
    }
    MotionInitializationSegment refined = *retained.segment_from_previous;
    refined.lidar.T_imu_start_imu_end = increment;
    refined.lidar.information = canonical_information.value().information;
    refined.lidar.information.eigenvalues /= output.diagnostics.applied_covariance_inflation;
    refined.lidar.imu_conditioning_covariance_inflation =
        output.diagnostics.imu_conditioning_covariance_inflation;
    refined.lidar.applied_covariance_inflation = output.diagnostics.applied_covariance_inflation;
    output.segments.push_back(std::move(refined));
    ++output.diagnostics.registrations;
    output.diagnostics.minimum_observable_rank = std::min(
        output.diagnostics.minimum_observable_rank, canonical_information.value().information.rank);
    output.diagnostics.total_registration_cost += registration.value().final_robust_cost;
    output.diagnostics.maximum_registration_cost = std::max(
        output.diagnostics.maximum_registration_cost, registration.value().final_robust_cost);
    previous_cloud = std::move(current_cloud);
  }
  if (output.segments.size() + 1U != reference_states.size()) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                       "full-deskew refinement did not rebuild every adjacent registration"));
  }
  if (!output.final_cloud ||
      core::validateLineage(output.final_lineage) != core::LineageValidationError::None) {
    return Result::failure(
        bootstrapError(LidarBootstrapErrorCode::RefinementStateMismatch,
                       "full-deskew refinement produced no final registration cloud or valid lineage"));
  }
  refinement_timing.finish();
  return Result::success(std::move(output));
}

bool LidarBootstrapOdometry::empty() const noexcept {
  return implementation_->sweeps == 0U;
}

std::size_t LidarBootstrapOdometry::retainedSweeps() const noexcept {
  return implementation_->sweeps;
}

}  // namespace meridian::local
