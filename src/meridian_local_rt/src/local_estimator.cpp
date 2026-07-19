#include "meridian/local/local_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "identity_transaction.hpp"
#include "meridian/local/map_admission_gate.hpp"
#include "pipeline_timing_internal.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] LocalEstimatorError estimatorError(LocalEstimatorErrorCode code,
                                                 LocalEstimatorStage stage, std::string detail) {
  LocalEstimatorError error;
  error.code = code;
  error.stage = stage;
  error.detail = std::move(detail);
  return error;
}

[[nodiscard]] bool validSweep(const core::LidarSweep& sweep,
                              const core::CalibrationBundle& calibration) {
  return sweep.id.valid() && sweep.lidar == calibration.lidar().id() &&
         sweep.stamp.source_epoch.valid() &&
         sweep.stamp.status != core::TimeMappingStatus::Discontinuous &&
         sweep.acquisition.valid() && sweep.layout.width > 0U && sweep.layout.height > 0U &&
         sweep.points && !sweep.points->empty() &&
         sweep.header.direct_calibration == calibration.epoch();
}

[[nodiscard]] bool validConfig(const LocalEstimatorConfig& config,
                               const core::CalibrationBundle& calibration) {
  const double lidar_covariance_inflation = config.lidar_target_reuse_covariance_inflation *
                                            config.lidar_imu_conditioning_covariance_inflation;
  if (!config.odom_epoch.valid() || !config.first_state.valid() ||
      config.maximum_pending_lidar_sweeps == 0U || config.maximum_pending_imu_guards == 0U ||
      config.maximum_pending_finalized_lidar_sweeps == 0U ||
      config.finalized_lidar_prune_interval_sweeps == 0U ||
      config.maximum_recent_faulty_batches_to_remove == 0U ||
      config.maximum_recent_faulty_batches_to_remove >
          config.graph.maximum_factor_batches_per_removal_transaction ||
      config.maximum_recent_faulty_batches_to_remove >
          config.rolling_target.maximum_retained_sweeps ||
      config.minimum_lidar_factor_interval.nanoseconds < 0 || !config.pipeline_timing.valid() ||
      config.state_timeline.minimum_state_interval.nanoseconds < 0 ||
      config.state_timeline.maximum_navigation_states == 0U ||
      // A new sensor-neutral state must enter the timeline before the graph
      // transaction can enforce its live-state cap and publish the oldest
      // state's finality. Expressing graph + 1 as a strict comparison avoids
      // overflow when a hostile configuration uses size_t::max().
      config.state_timeline.maximum_navigation_states <= config.graph.maximum_navigation_states ||
      config.state_timeline.maximum_retained_requests == 0U ||
      !isValidLidarRegistrationConfig(config.lidar_registration) ||
      config.lidar_registration.maximum_targets >
          config.lidar_registration.maximum_composite_owners ||
      !std::isfinite(config.lidar_target_reuse_covariance_inflation) ||
      config.lidar_target_reuse_covariance_inflation < 1.0 ||
      !std::isfinite(config.lidar_imu_conditioning_covariance_inflation) ||
      config.lidar_imu_conditioning_covariance_inflation < 1.0 ||
      !std::isfinite(config.finalized_map_correlation_inflation_floor) ||
      config.finalized_map_correlation_inflation_floor <= 0.0 ||
      !std::isfinite(lidar_covariance_inflation) || lidar_covariance_inflation <= 1.0 ||
      !std::isfinite(config.lidar_registration.maximum_translation_information /
                     lidar_covariance_inflation) ||
      config.lidar_registration.maximum_translation_information / lidar_covariance_inflation <=
          0.0 ||
      config.finalized_lidar_target.query_voxel_size_m !=
          config.lidar_registration.target_voxel_resolution_m ||
      config.finalized_lidar_target.maximum_supported_query_distance_m !=
          config.lidar_registration.maximum_correspondence_distance_m ||
      config.lidar_registration.maximum_targets >
          config.graph.maximum_direct_lidar_factors_per_transaction) {
    return false;
  }
  std::set<std::uint64_t> enabled_cameras;
  for (const VisualCameraConfig& visual : config.visual_cameras) {
    if (!visual.camera.valid() || calibration.camera(visual.camera) == nullptr ||
        !enabled_cameras.insert(visual.camera.value()).second ||
        visual.lane.maximum_landmarks_per_attachment >
            config.graph.maximum_visual_landmarks_per_transaction ||
        visual.lane.maximum_factors_per_attachment >
            config.graph.maximum_visual_factors_per_transaction ||
        visual.lane.maximum_factor_retirements_per_attachment >
            config.graph.maximum_visual_factor_retirements_per_transaction) {
      return false;
    }
  }
  const bool has_zero_motion_prior = config.initialization.zero_motion_prior.has_value();
  if (has_zero_motion_prior &&
      (!config.initialization.zero_motion_prior->valid() ||
       config.initialization.zero_motion_prior->odom_epoch != config.odom_epoch)) {
    return false;
  }
  switch (config.initialization.mode) {
    case InitializationMode::StaticOnly:
      if (!has_zero_motion_prior) {
        return false;
      }
      break;
    case InitializationMode::DynamicOnly:
      if (has_zero_motion_prior) {
        return false;
      }
      break;
    case InitializationMode::SupervisedAuto:
      break;
  }
  return config.maximum_pending_lidar_sweeps > 0U &&
         config.stationary_retry_period.nanoseconds > 0 &&
         config.stationary_initializer.minimum_support.nanoseconds > 0 &&
         config.motion_initializer.minimum_segments > 0U &&
         config.lidar_bootstrap.maximum_sweeps > config.motion_initializer.minimum_segments &&
         config.imu_buffer.maximum_samples >= 2U &&
         config.imu_buffer.maximum_span >= std::max(config.stationary_initializer.minimum_support,
                                                    config.motion_initializer.minimum_support) &&
         calibration.epoch().valid() && calibration.imu().valid() &&
         calibration.lidar().id().valid();
}

[[nodiscard]] core::Duration nominalPeriod(const core::ImuCalibration& imu) {
  return core::Duration{static_cast<std::int64_t>(std::llround(1.0e9 / imu.nominalRateHz()))};
}

void applyCalibration(LocalEstimatorConfig& config, const core::CalibrationBundle& calibration) {
  const auto& imu = calibration.imu();
  const auto& noise = imu.noise();
  const core::Duration period = nominalPeriod(imu);
  config.stationary_initializer.nominal_period = period;
  config.stationary_initializer.gravity_mps2 = imu.gravityMagnitude();
  config.graph.imu.gravity_odom = Eigen::Vector3d{0.0, 0.0, -imu.gravityMagnitude()};
  config.graph.imu.accelerometer_noise_density_mps2_sqrt_hz = noise.accelerometerNoiseDensity();
  config.graph.imu.gyroscope_noise_density_radps_sqrt_hz = noise.gyroscopeNoiseDensity();
  config.graph.imu.accelerometer_bias_random_walk_mps3_sqrt_hz =
      noise.accelerometerBiasRandomWalk();
  config.graph.imu.gyroscope_bias_random_walk_radps2_sqrt_hz = noise.gyroscopeBiasRandomWalk();
  config.rolling_target.odom_epoch = config.odom_epoch;
  config.rolling_target.registration = config.lidar_registration;
  config.finalized_lidar_target.odom_epoch = config.odom_epoch;
  config.finalized_lidar_target.sensor =
      core::SensorInstanceId::lidar(calibration.lidar().id());
}

[[nodiscard]] core::ObservationLineage registrationCloudLineage(
    core::MeasurementId source, std::span<const core::MeasurementId> imu_support,
    core::CalibrationEpoch calibration, core::DerivedRecordId consumer,
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
    core::ObservationSlice conditioning_slice;
    conditioning_slice.root = measurement;
    conditioning_slice.kind = core::SliceKind::Whole;
    conditioning_slice.calibration = calibration;
    lineage.usage.push_back(core::ObservationUsage{conditioning_slice,
                                                   core::ObservationRole::ConditioningOnly,
                                                   consumer, std::nullopt, std::nullopt});
  }
  return lineage;
}

[[nodiscard]] bool sameObservationSlice(const core::ObservationSlice& lhs,
                                        const core::ObservationSlice& rhs) noexcept {
  return lhs.root == rhs.root && lhs.kind == rhs.kind && lhs.begin == rhs.begin &&
         lhs.end == rhs.end && lhs.source_checksum == rhs.source_checksum &&
         lhs.calibration == rhs.calibration;
}

[[nodiscard]] core::DirectionalObservability lidarDirectionalObservability(
    const core::RankAwareInformation& information, core::StateId target_state,
    core::FusionTime target_time, core::StateId source_state, core::FusionTime source_time) {
  core::DirectionalObservability result;
  // RankAwareInformation stores supported directions first in descending
  // order. FactorBatch metadata uses ascending eigenvalues so one threshold
  // has an unambiguous interpretation across solver implementations.
  for (Eigen::Index output_index = 0; output_index < 6; ++output_index) {
    const Eigen::Index input_index = 5 - output_index;
    result.basis.col(output_index) = information.basis.col(input_index);
    result.eigenvalues(output_index) = std::max(0.0, information.eigenvalues(input_index));
  }
  result.rank = static_cast<std::uint32_t>(information.rank);
  const std::size_t unsupported = 6U - information.rank;
  if (unsupported == 0U) {
    result.absolute_eigenvalue_threshold =
        std::max(std::numeric_limits<double>::min(), 0.5 * result.eigenvalues(0));
  } else {
    const double lower = result.eigenvalues(static_cast<Eigen::Index>(unsupported - 1U));
    const double upper = result.eigenvalues(static_cast<Eigen::Index>(unsupported));
    result.absolute_eigenvalue_threshold = lower + 0.5 * (upper - lower);
  }
  result.relative_eigenvalue_threshold = 0.0;
  result.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                core::DirectionalVariable::PoseRotation};
  result.endpoints = {{core::DirectionalEndpointRole::Target, target_state, target_time},
                      {core::DirectionalEndpointRole::Source, source_state, source_time}};
  return result;
}

[[nodiscard]] core::DirectionalObservability lidarUnaryDirectionalObservability(
    const core::RankAwareInformation& information, core::StateId source_state,
    core::FusionTime source_time) {
  core::DirectionalObservability result = lidarDirectionalObservability(
      information, source_state, source_time, source_state, source_time);
  result.endpoints = {
      {core::DirectionalEndpointRole::Unary, source_state, source_time}};
  return result;
}

[[nodiscard]] core::Result<core::ContentHash, LocalEstimatorError> lidarFactorLineageChecksum(
    const core::ObservationLineage& lineage) {
  using Result = core::Result<core::ContentHash, LocalEstimatorError>;
  auto checksum = recomputeAcceptedLidarLineageChecksum(lineage);
  if (!checksum || !core::contentHashPresent(checksum.value())) {
    return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                          LocalEstimatorStage::InternalInvariant,
                                          "could not seal canonical LiDAR factor lineage"));
  }
  return Result::success(checksum.value());
}

}  // namespace

struct LocalEstimator::Impl {
  struct PendingLidarSweep {
    core::LidarSweep sweep;
    StateAdmission admission;
  };

  struct PendingVisualKnot {
    std::size_t lane_index{};
    core::MeasurementId frame;
    StateAdmission admission;
  };

  struct PendingImuGuard {
    StateAdmission admission;
  };

  struct VisualIdentityState {
    std::map<VisualLandmarkId, VisualLandmarkId> landmarks;
    std::map<VisualLandmarkId, VisualLandmarkId> graph_to_local_landmarks;
    std::map<core::FactorId, core::FactorId> factors;
    std::map<core::FactorId, core::FactorId> graph_to_local_factors;
    std::map<core::FactorId, VisualLandmarkId> factor_landmarks;
    std::map<core::FactorId, core::ObservationLineageId> factor_lineages;
    std::map<core::FactorId, core::DerivedRecordId> factor_consumers;
    std::set<VisualLandmarkId> accepted_landmarks;
    std::set<core::FactorId> accepted_factors;
  };

  struct VisualLaneState {
    core::CameraId camera;
    VisualLane lane;
    VisualIdentityState identities;
    std::optional<core::FusionTime> previous_exposure;
  };

  struct PreparedVisualAttachment {
    std::size_t lane_index{};
    VisualAttachmentId lane_attachment;
    std::optional<VisualFactorBatch> graph_batch;
    std::vector<core::FactorId> graph_retirements;
    std::vector<VisualLandmarkId> local_landmarks;
    std::vector<core::FactorId> local_factors;
    std::vector<core::FactorId> local_retirements;
    std::optional<detail::IdentityTransaction<VisualIdentityState>> identity_transaction;
  };

  struct GraphAppendOutcome {
    LocalGraphCommit commit;
    std::optional<VisualGraphAttachmentReport> visual_attachment;
    std::optional<VisualGraphDegradationReport> visual_degradation;
  };

  struct PreparedLidarFactorBatch {
    LidarDirectFactorBatch batch;
    std::vector<DirectLidarPairReport> pairs;
    std::optional<DirectLidarFinalizedMapReport> finalized_map;
  };

  struct PendingFinalizedLidarSweep {
    SensorFactorBatchRef batch;
    core::FactorBatchMetadata accepted_batch_metadata;
    core::LocalGraphRevision admission_revision;
    MapAdmissionBatchKind admission_kind{MapAdmissionBatchKind::Regular};
    core::StateId state;
    core::FusionTime exact_time;
    core::CalibrationEpoch calibration;
    std::shared_ptr<const LidarRegistrationCloud> cloud;
    std::optional<LocalGraphFinalizedState> finality;
  };

  struct FailedBatchRemoval {
    std::vector<SensorFactorBatchRef> batches;
    std::optional<RollingLidarTargetRemovalStats> target_removal;
    std::optional<LocalGraphCommit> commit;
  };

  struct LidarHealthOutcome {
    SensorHealthUpdate update;
    FailedBatchRemoval removal;
  };

  struct AdmittedLidarCloud {
    RollingLidarTargetAddStats target_add;
    std::shared_ptr<const AcceptedLidarMapInput> map_input;
  };

  Impl(core::CalibrationBundle input_calibration, LocalEstimatorConfig input_config,
       RollingLidarTargetBuilder input_target, FinalizedLidarTargetMap input_finalized_target,
       LidarBootstrapOdometry input_bootstrap,
       SensorHealthRegistry input_sensor_health,
       std::shared_ptr<LocalPipelineTimingRecorder> input_timing)
      : calibration(std::move(input_calibration)),
        config(std::move(input_config)),
        nominal_imu_period(nominalPeriod(calibration.imu())),
        imu_buffer(config.imu_buffer),
        propagator(config.graph.imu.gravity_odom),
        pipeline_timing(std::move(input_timing)),
        graph(config.graph, pipeline_timing),
        rolling_target(std::move(input_target)),
        finalized_target(std::move(input_finalized_target)),
        map_admission(),
        sensor_health(std::move(input_sensor_health)),
        lidar_bootstrap(std::move(input_bootstrap)),
        motion_initializer(config.motion_initializer, pipeline_timing),
        state_timeline(config.state_timeline),
        next_state(config.first_state.value()) {}

  core::CalibrationBundle calibration;
  LocalEstimatorConfig config;
  core::Duration nominal_imu_period;
  ImuBuffer imu_buffer;
  MidpointImuPropagator propagator;
  std::shared_ptr<LocalPipelineTimingRecorder> pipeline_timing;
  LocalGraph graph;
  RollingLidarTargetBuilder rolling_target;
  FinalizedLidarTargetMap finalized_target;
  MapAdmissionGate map_admission;
  SensorHealthRegistry sensor_health;
  LidarBootstrapOdometry lidar_bootstrap;
  MotionInitializer motion_initializer;
  StateTimeline state_timeline;
  std::vector<VisualLaneState> visual_lanes;
  std::vector<MotionInitializationSegment> motion_segments;
  std::deque<PendingLidarSweep> pending_lidar;
  std::deque<PendingImuGuard> pending_imu_guards;
  std::deque<PendingVisualKnot> pending_visual_knots;
  std::deque<PendingFinalizedLidarSweep> pending_finalized_lidar;
  LocalEstimatorLifecycle lifecycle{LocalEstimatorLifecycle::AwaitingInitialization};
  LocalEstimatorStatistics statistics;
  FinalizedLidarTargetProcessReport finalized_target_process;
  std::vector<LocalGraphTransactionSolveReport> pending_graph_transactions;
  std::size_t finalized_insertions_since_prune{};
  bool finalized_target_frozen{};
  bool finalized_ready_blocked{};
  bool finalized_target_capacity_saturated{};
  std::size_t finalized_capacity_skips_since_retry{};
  bool lidar_map_initialized{};
  // Monotonic historical keyframe-rate anchor. Fault recovery may retract a
  // live batch, but it must not cause an immediate burst of replacement
  // keyframes at an already-serviced timestamp.
  std::optional<core::FusionTime> last_lidar_keyframe_time;
  std::optional<core::FusionTime> first_imu_time;
  std::optional<core::FusionTime> latest_imu_time;
  std::optional<core::RecordHeader> latest_imu_header;
  std::optional<core::FusionTime> last_initialization_attempt;
  std::optional<core::FusionTime> last_bootstrap_time;
  std::optional<core::MeasurementId> last_bootstrap_measurement;
  // Keeps the immutable raw payload by shared ownership only until moving
  // initialization either accepts this exact seed or resets the bootstrap.
  std::optional<core::LidarSweep> last_bootstrap_sweep;
  std::optional<ImuInterval> last_bootstrap_acquisition_imu;
  std::optional<core::MeasurementId> last_lidar_measurement;
  std::optional<core::FusionTime> last_lidar_start;
  std::uint64_t next_state;
  std::uint64_t next_lineage{1U};
  std::uint64_t next_derived{1U};
  std::uint64_t next_factor_group{1U};
  std::uint64_t next_correlation_group{1U};
  std::uint64_t next_knot_request{1U};
  std::uint64_t next_knot_resolution{1U};
  std::uint64_t next_visual_landmark{1U};
  std::uint64_t next_visual_factor{1U};
  std::uint64_t next_factor_batch{1U};
  std::size_t next_visual_attachment_lane{};

  template <typename Id>
  [[nodiscard]] std::optional<Id> allocate(std::uint64_t& counter) {
    if (counter == Id::kInvalidValue) {
      return std::nullopt;
    }
    const Id result{counter};
    ++counter;
    return result;
  }

  [[nodiscard]] detail::IdentityCounters identityCounters() const noexcept {
    return detail::IdentityCounters{next_lineage,         next_derived,
                                    next_factor_group,    next_correlation_group,
                                    next_visual_landmark, next_visual_factor};
  }

  void publishIdentityCounters(const detail::IdentityCounters& counters) noexcept {
    next_lineage = std::max(next_lineage, counters.next_lineage);
    next_derived = std::max(next_derived, counters.next_derived);
    next_factor_group = std::max(next_factor_group, counters.next_factor_group);
    next_correlation_group = std::max(next_correlation_group, counters.next_correlation_group);
    next_visual_landmark = std::max(next_visual_landmark, counters.next_visual_landmark);
    next_visual_factor = std::max(next_visual_factor, counters.next_visual_factor);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> createVisualLanes() {
    using Result = core::Result<bool, LocalEstimatorError>;
    visual_lanes.reserve(config.visual_cameras.size());
    for (const VisualCameraConfig& visual : config.visual_cameras) {
      auto lane = VisualLane::create(calibration, visual.camera, state_timeline, visual.lane);
      if (!lane) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
            "enabled visual lane is invalid: " + lane.error().detail));
      }
      visual_lanes.push_back(
          VisualLaneState{visual.camera, std::move(lane).value(), {}, std::nullopt});
    }
    return Result::success(true);
  }

  [[nodiscard]] std::optional<std::size_t> visualLaneIndex(core::CameraId camera) const noexcept {
    for (std::size_t index = 0U; index < visual_lanes.size(); ++index) {
      if (visual_lanes[index].camera == camera) {
        return index;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<VisualRotationPrior> visualRotationPrior(
      const VisualLaneState& lane, core::FusionTime current_exposure) const {
    if (!lane.previous_exposure || current_exposure <= *lane.previous_exposure ||
        !graph.initialized()) {
      return std::nullopt;
    }
    auto current = graph.estimate();
    if (!current || !latest_imu_time || current_exposure > *latest_imu_time) {
      return std::nullopt;
    }
    auto interval = imu_buffer.interval(core::TimeRange{*lane.previous_exposure, current_exposure},
                                        nominal_imu_period);
    if (!interval || interval.value().contains_saturation ||
        interval.value().inferred_missing_ticks != 0U ||
        interval.value().maximum_time_uncertainty.nanoseconds != 0) {
      return std::nullopt;
    }

    Sophus::SO3d R_previous_current_imu;
    for (std::size_t index = 1U; index < interval.value().knots.size(); ++index) {
      const InterpolatedImuSample& previous = interval.value().knots[index - 1U];
      const InterpolatedImuSample& current_knot = interval.value().knots[index];
      const double dt =
          static_cast<double>((current_knot.time - previous.time).nanoseconds) * 1.0e-9;
      const Eigen::Vector3d omega =
          0.5 * (previous.angular_velocity_radps + current_knot.angular_velocity_radps) -
          current.value().estimate.gyro_bias;
      R_previous_current_imu *= Sophus::SO3d::exp(omega * dt);
    }

    const core::CameraCalibration* camera = calibration.camera(lane.camera);
    if (camera == nullptr) {
      return std::nullopt;
    }
    const Sophus::SO3d& R_imu_camera = camera->extrinsics().T_imu_camera().so3();
    VisualRotationPrior prior;
    prior.R_current_previous =
        (R_imu_camera.inverse() * R_previous_current_imu.inverse() * R_imu_camera).matrix();
    prior.previous_exposure_midpoint = *lane.previous_exposure;
    prior.current_exposure_midpoint = current_exposure;
    prior.imu_support = interval.value().raw_measurements;
    prior.imu_calibration = calibration.epoch();
    if (prior.imu_support.empty()) {
      return std::nullopt;
    }
    return prior;
  }

  [[nodiscard]] core::Result<core::ObservationLineage, LocalEstimatorError> lidarRequestLineage(
      core::MeasurementId sweep, detail::IdentityCounters& counters) {
    using Result = core::Result<core::ObservationLineage, LocalEstimatorError>;
    const auto lineage_id = allocate<core::ObservationLineageId>(counters.next_lineage);
    const auto consumer = allocate<core::DerivedRecordId>(counters.next_derived);
    if (!lineage_id || !consumer) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                            LocalEstimatorStage::IdentityAllocation,
                                            "LiDAR knot-request lineage identity exhausted"));
    }
    core::ObservationSlice slice;
    slice.root = sweep;
    slice.calibration = calibration.epoch();
    core::ObservationLineage lineage;
    lineage.id = *lineage_id;
    lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::DerivedSummary,
                                                   *consumer, std::nullopt, std::nullopt});
    if (core::validateLineage(lineage) != core::LineageValidationError::None) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::InternalInvariant,
                                            "constructed LiDAR knot-request lineage is invalid"));
    }
    return Result::success(std::move(lineage));
  }

  [[nodiscard]] core::Result<StateAdmission, LocalEstimatorError> admitLidar(
      const core::LidarSweep& sweep) {
    using Result = core::Result<StateAdmission, LocalEstimatorError>;
    detail::IdentityTransaction<bool> request_identities(identityCounters(), false);
    std::uint64_t candidate_next_knot_request = next_knot_request;
    const auto request = allocate<core::KnotRequestId>(candidate_next_knot_request);
    if (!request) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                            LocalEstimatorStage::IdentityAllocation,
                                            "LiDAR knot-request identity exhausted"));
    }
    auto lineage = lidarRequestLineage(sweep.id, request_identities.counters());
    if (!lineage) {
      return Result::failure(lineage.error());
    }
    StateRequest state_request;
    state_request.header = sweep.header;
    state_request.id = *request;
    state_request.sensor = core::SensorInstanceId::lidar(calibration.lidar().id());
    state_request.purpose = StateRequestPurpose::LidarReference;
    state_request.exact_time = sweep.acquisition.end;
    state_request.lineage = std::move(lineage).value();
    auto admission = state_timeline.request(std::move(state_request));
    if (!admission) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::StateTimelineFailed, LocalEstimatorStage::StateTimeline,
          "LiDAR state request was rejected: " + admission.error().detail));
    }
    auto committed_identities = request_identities.commit();
    if (!committed_identities) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::IdentityAllocation,
          "admitted LiDAR request identities were already consumed"));
    }
    publishIdentityCounters(committed_identities.value().counters);
    next_knot_request = candidate_next_knot_request;
    return Result::success(std::move(admission).value());
  }

  [[nodiscard]] core::Result<PreparedVisualAttachment, LocalEstimatorError> prepareVisualAttachment(
      std::size_t lane_index, const ImuKnotAppend& navigation,
      const detail::IdentityCounters& candidate_counters) {
    using Result = core::Result<PreparedVisualAttachment, LocalEstimatorError>;
    VisualLaneState& lane = visual_lanes[lane_index];
    auto prepared = lane.lane.prepareGraphInput(navigation);
    if (!prepared) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::VisualLaneFailed, LocalEstimatorStage::VisualGraphAttachment,
          "visual lane could not prepare graph input: " + prepared.error().detail));
    }

    const auto reject_prepared = [&lane, &prepared]() {
      return lane.lane.acknowledgeGraphInputRejected(prepared.value().id);
    };
    PreparedVisualAttachment output;
    output.lane_index = lane_index;
    output.lane_attachment = prepared.value().id;
    output.identity_transaction.emplace(candidate_counters, lane.identities);
    auto& identities = output.identity_transaction->state();
    auto& counters = output.identity_transaction->counters();

    auto fail_translation = [&](std::string detail) {
      const auto rejected = reject_prepared();
      if (!rejected) {
        detail += "; visual in-flight acknowledgement also failed: " + rejected.error().detail;
      }
      return Result::failure(estimatorError(LocalEstimatorErrorCode::VisualReferenceUnavailable,
                                            LocalEstimatorStage::VisualGraphAttachment,
                                            std::move(detail)));
    };

    if (prepared.value().visual) {
      VisualFactorBatch batch = *prepared.value().visual;
      std::set<VisualLandmarkId> batch_landmarks;
      for (VisualLandmarkSeed& seed : batch.new_landmarks) {
        const VisualLandmarkId local = seed.landmark;
        if (!local.valid() || !batch_landmarks.insert(local).second ||
            identities.accepted_landmarks.contains(local)) {
          return fail_translation(
              "visual batch repeats an invalid, duplicate, or already accepted "
              "lane-local landmark identity");
        }
        auto global = identities.landmarks.find(local);
        if (global == identities.landmarks.end()) {
          const auto allocated = allocate<VisualLandmarkId>(counters.next_visual_landmark);
          if (!allocated) {
            return fail_translation("graph-visible visual landmark identity exhausted");
          }
          global = identities.landmarks.emplace(local, *allocated).first;
          identities.graph_to_local_landmarks.emplace(*allocated, local);
        } else {
          const auto reverse = identities.graph_to_local_landmarks.find(global->second);
          if (reverse == identities.graph_to_local_landmarks.end() || reverse->second != local) {
            return fail_translation(
                "lane-local visual landmark mapping has no unique graph-visible reverse entry");
          }
        }
        output.local_landmarks.push_back(local);
        seed.landmark = global->second;
      }

      std::set<core::FactorId> batch_factors;
      for (VisualReprojectionFactorSpec& factor : batch.factors) {
        const core::FactorId local_factor = factor.id;
        const VisualLandmarkId local_landmark = factor.landmark;
        if (!local_factor.valid() || !batch_factors.insert(local_factor).second ||
            identities.accepted_factors.contains(local_factor)) {
          return fail_translation(
              "visual batch repeats an invalid, duplicate, or already accepted "
              "lane-local factor identity");
        }
        const auto landmark = identities.landmarks.find(local_landmark);
        if (landmark == identities.landmarks.end()) {
          return fail_translation("visual factor refers to an unknown lane-local landmark");
        }
        auto global_factor = identities.factors.find(local_factor);
        if (global_factor == identities.factors.end()) {
          const auto allocated = allocate<core::FactorId>(counters.next_visual_factor);
          if (!allocated) {
            return fail_translation("graph-visible visual factor identity exhausted");
          }
          global_factor = identities.factors.emplace(local_factor, *allocated).first;
          identities.graph_to_local_factors.emplace(*allocated, local_factor);
        } else {
          const auto reverse = identities.graph_to_local_factors.find(global_factor->second);
          if (reverse == identities.graph_to_local_factors.end() ||
              reverse->second != local_factor) {
            return fail_translation(
                "lane-local visual factor mapping has no unique graph-visible reverse entry");
          }
        }
        const auto [factor_landmark, factor_landmark_inserted] =
            identities.factor_landmarks.emplace(local_factor, local_landmark);
        if (!factor_landmark_inserted && factor_landmark->second != local_landmark) {
          return fail_translation(
              "lane-local visual factor changed its immutable landmark dependency");
        }

        auto lineage = identities.factor_lineages.find(local_factor);
        if (lineage == identities.factor_lineages.end()) {
          const auto allocated = allocate<core::ObservationLineageId>(counters.next_lineage);
          if (!allocated) {
            return fail_translation("graph-visible visual factor lineage identity exhausted");
          }
          lineage = identities.factor_lineages.emplace(local_factor, *allocated).first;
        }
        auto consumer = identities.factor_consumers.find(local_factor);
        if (consumer == identities.factor_consumers.end()) {
          const auto allocated = allocate<core::DerivedRecordId>(counters.next_derived);
          if (!allocated) {
            return fail_translation("graph-visible visual factor consumer identity exhausted");
          }
          consumer = identities.factor_consumers.emplace(local_factor, *allocated).first;
        }

        factor.id = global_factor->second;
        factor.landmark = landmark->second;
        factor.lineage.id = lineage->second;
        for (core::ObservationUsage& usage : factor.lineage.usage) {
          usage.consumer = consumer->second;
          if (usage.role == core::ObservationRole::PrimaryResidual) {
            usage.factor_group = core::FactorGroupId(global_factor->second.value());
          } else {
            usage.factor_group.reset();
          }
        }
        if (core::validateLineage(factor.lineage) != core::LineageValidationError::None) {
          return fail_translation("global identity translation produced an invalid visual lineage");
        }
        output.local_factors.push_back(local_factor);
      }

      for (VisualTrackRetirement& retirement : batch.retired_tracks) {
        if (!retirement.landmark.valid()) {
          continue;
        }
        const auto global = identities.landmarks.find(retirement.landmark);
        if (global == identities.landmarks.end()) {
          return fail_translation(
              "visual track retirement refers to an unknown lane-local landmark");
        }
        retirement.landmark = global->second;
      }
      output.graph_batch = std::move(batch);
    }

    std::set<core::FactorId> retirements;
    for (const core::FactorId local : prepared.value().visual_factor_retirements) {
      if (!local.valid() || !retirements.insert(local).second ||
          !identities.accepted_factors.contains(local)) {
        return fail_translation(
            "visual retirement refers to an invalid, duplicate, stale, or "
            "unaccepted lane-local factor");
      }
      const auto global = identities.factors.find(local);
      if (global == identities.factors.end()) {
        return fail_translation("visual retirement has no graph-visible factor mapping");
      }
      output.local_retirements.push_back(local);
      output.graph_retirements.push_back(global->second);
    }
    return Result::success(std::move(output));
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> removeCommittedVisualRetirements(
      const PreparedVisualAttachment& attachment) {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto invariant = [this](std::string detail) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::VisualGraphAttachment,
                                            std::move(detail)));
    };
    if (attachment.local_retirements.size() != attachment.graph_retirements.size() ||
        attachment.lane_index >= visual_lanes.size()) {
      return invariant("committed visual retirement identity lists are inconsistent");
    }
    VisualIdentityState& identities = visual_lanes[attachment.lane_index].identities;
    for (std::size_t index = 0U; index < attachment.local_retirements.size(); ++index) {
      const core::FactorId local = attachment.local_retirements[index];
      const core::FactorId graph_factor = attachment.graph_retirements[index];
      const auto forward = identities.factors.find(local);
      const auto reverse = identities.graph_to_local_factors.find(graph_factor);
      if (!identities.accepted_factors.contains(local) || forward == identities.factors.end() ||
          forward->second != graph_factor || reverse == identities.graph_to_local_factors.end() ||
          reverse->second != local || !identities.factor_landmarks.contains(local)) {
        return invariant(
            "committed visual retirement lost its accepted bidirectional identity mapping");
      }
    }
    for (std::size_t index = 0U; index < attachment.local_retirements.size(); ++index) {
      const core::FactorId local = attachment.local_retirements[index];
      const core::FactorId graph_factor = attachment.graph_retirements[index];
      identities.accepted_factors.erase(local);
      identities.factors.erase(local);
      identities.graph_to_local_factors.erase(graph_factor);
      identities.factor_landmarks.erase(local);
      identities.factor_lineages.erase(local);
      identities.factor_consumers.erase(local);
    }
    return Result::success(true);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> reconcileVisualFinality(
      const LocalGraphCommit& commit) {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto invariant = [this](std::string detail) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::VisualGraphAttachment,
                                            std::move(detail)));
    };
    if (!commit.revision.valid()) {
      return invariant("committed graph finality has an invalid revision");
    }

    struct OwnedFactor {
      std::size_t lane{};
      core::FactorId graph;
      core::FactorId local;
    };
    struct OwnedLandmark {
      std::size_t lane{};
      VisualLandmarkId graph;
      VisualLandmarkId local;
    };
    std::vector<VisualFinalityUpdate> lane_updates(visual_lanes.size());
    for (VisualFinalityUpdate& update : lane_updates) {
      update.revision = commit.revision;
    }
    std::vector<OwnedFactor> owned_factors;
    std::vector<OwnedLandmark> owned_landmarks;
    owned_factors.reserve(commit.finalized_visual_factors.size());
    owned_landmarks.reserve(commit.finalized_visual_landmarks.size());

    for (std::size_t index = 0U; index < commit.finalized_visual_factors.size(); ++index) {
      const core::FactorId graph_factor = commit.finalized_visual_factors[index];
      if (!graph_factor.valid() ||
          (index > 0U && graph_factor <= commit.finalized_visual_factors[index - 1U])) {
        return invariant("committed automatic visual factor finality is not canonical");
      }
      std::optional<OwnedFactor> owner;
      for (std::size_t lane_index = 0U; lane_index < visual_lanes.size(); ++lane_index) {
        const VisualIdentityState& identities = visual_lanes[lane_index].identities;
        const auto reverse = identities.graph_to_local_factors.find(graph_factor);
        if (reverse == identities.graph_to_local_factors.end()) {
          continue;
        }
        const auto forward = identities.factors.find(reverse->second);
        if (owner || forward == identities.factors.end() || forward->second != graph_factor ||
            !identities.accepted_factors.contains(reverse->second) ||
            !identities.factor_landmarks.contains(reverse->second)) {
          return invariant("finalized visual factor does not have one accepted owning camera lane");
        }
        owner = OwnedFactor{lane_index, graph_factor, reverse->second};
      }
      if (!owner) {
        return invariant("finalized visual factor has no owning camera lane");
      }
      lane_updates[owner->lane].finalized_factors.push_back(owner->local);
      owned_factors.push_back(*owner);
    }

    for (std::size_t index = 0U; index < commit.finalized_visual_landmarks.size(); ++index) {
      const LocalGraphFinalizedVisualLandmark& finalized = commit.finalized_visual_landmarks[index];
      if (!finalized.landmark.valid() ||
          (index > 0U &&
           finalized.landmark <= commit.finalized_visual_landmarks[index - 1U].landmark) ||
          finalized.segment.segment.value() != finalized.landmark.value() ||
          finalized.segment.final_revision != commit.revision ||
          core::validateFinalizedLandmarkSegment(finalized.segment) !=
              core::FinalityValidationError::None) {
        return invariant("committed automatic visual landmark finality is not canonical");
      }
      std::optional<OwnedLandmark> owner;
      for (std::size_t lane_index = 0U; lane_index < visual_lanes.size(); ++lane_index) {
        const VisualIdentityState& identities = visual_lanes[lane_index].identities;
        const auto reverse = identities.graph_to_local_landmarks.find(finalized.landmark);
        if (reverse == identities.graph_to_local_landmarks.end()) {
          continue;
        }
        const auto forward = identities.landmarks.find(reverse->second);
        if (owner || forward == identities.landmarks.end() ||
            forward->second != finalized.landmark ||
            !identities.accepted_landmarks.contains(reverse->second)) {
          return invariant(
              "finalized visual landmark does not have one accepted owning camera lane");
        }
        owner = OwnedLandmark{lane_index, finalized.landmark, reverse->second};
      }
      if (!owner) {
        return invariant("finalized visual landmark has no owning camera lane");
      }
      lane_updates[owner->lane].finalized_landmarks.push_back(owner->local);
      owned_landmarks.push_back(*owner);
    }

    for (VisualFinalityUpdate& update : lane_updates) {
      std::sort(update.finalized_factors.begin(), update.finalized_factors.end());
      std::sort(update.finalized_landmarks.begin(), update.finalized_landmarks.end());
    }

    // A finalized parent must expose all of its still-accepted children in the
    // same commit. Verify this before mutating any lane.
    for (const OwnedLandmark& landmark : owned_landmarks) {
      const VisualIdentityState& identities = visual_lanes[landmark.lane].identities;
      const auto& finalized_factors = lane_updates[landmark.lane].finalized_factors;
      for (const auto& [factor, dependency] : identities.factor_landmarks) {
        if (dependency == landmark.local && identities.accepted_factors.contains(factor) &&
            !std::binary_search(finalized_factors.begin(), finalized_factors.end(), factor)) {
          return invariant(
              "finalized visual landmark omitted an accepted child factor from finality");
        }
      }
    }

    std::size_t tracks_pruned = 0U;
    std::size_t pending_factors_pruned = 0U;
    for (std::size_t lane_index = 0U; lane_index < visual_lanes.size(); ++lane_index) {
      auto reconciled =
          visual_lanes[lane_index].lane.reconcileGraphFinality(lane_updates[lane_index]);
      if (!reconciled) {
        return invariant("committed graph finality could not reconcile camera lane " +
                         std::to_string(visual_lanes[lane_index].camera.value()) + ": " +
                         reconciled.error().detail);
      }
      tracks_pruned += reconciled.value().factor_builder.accepted_tracks_pruned;
      pending_factors_pruned += reconciled.value().pending_factor_specs_pruned;
    }

    // Identity entries remain available for translation until every lane has
    // accepted this committed revision. Counters are monotonic and untouched.
    for (const OwnedFactor& factor : owned_factors) {
      VisualIdentityState& identities = visual_lanes[factor.lane].identities;
      identities.accepted_factors.erase(factor.local);
      identities.factors.erase(factor.local);
      identities.graph_to_local_factors.erase(factor.graph);
      identities.factor_landmarks.erase(factor.local);
      identities.factor_lineages.erase(factor.local);
      identities.factor_consumers.erase(factor.local);
    }
    for (const OwnedLandmark& landmark : owned_landmarks) {
      VisualIdentityState& identities = visual_lanes[landmark.lane].identities;
      identities.accepted_landmarks.erase(landmark.local);
      identities.landmarks.erase(landmark.local);
      identities.graph_to_local_landmarks.erase(landmark.graph);
    }

    statistics.visual_lane_finality_updates += visual_lanes.size();
    statistics.visual_finalized_landmarks += owned_landmarks.size();
    statistics.visual_finalized_factors += owned_factors.size();
    statistics.visual_finalized_tracks_pruned += tracks_pruned;
    statistics.visual_finality_pending_factors_pruned += pending_factors_pruned;
    return Result::success(true);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> requireFinalizedTargetCapacity() const {
    using Result = core::Result<bool, LocalEstimatorError>;
    if (pending_finalized_lidar.size() >= config.maximum_pending_finalized_lidar_sweeps) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::PendingFinalizedTargetCapacity,
          LocalEstimatorStage::FinalizedTarget,
          "persistent LiDAR target staging reached its configured hard capacity"));
    }
    return Result::success(true);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> stageFinalizedTargetSweep(
      const core::FactorBatchMetadata& metadata, const LocalGraphCommit& commit,
      MapAdmissionBatchKind admission_kind, core::StateId state,
      std::shared_ptr<const LidarRegistrationCloud> cloud) {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto capacity = requireFinalizedTargetCapacity();
    if (!capacity) {
      return capacity;
    }
    const core::SensorInstanceId lidar_sensor =
        core::SensorInstanceId::lidar(calibration.lidar().id());
    const auto state_pose = std::find_if(
        commit.navigation_poses.begin(), commit.navigation_poses.end(),
        [&](const LocalGraphPoseSnapshot& pose) { return pose.state == state; });
    if (!metadata.batch_id.valid() || metadata.odom_epoch != config.odom_epoch ||
        metadata.sensor != lidar_sensor || !metadata.map_eligible ||
        metadata.health.state != core::SensorHealthState::Active ||
        state_pose == commit.navigation_poses.end() || !cloud ||
        metadata.timing.measurement_timestamps.size() != 1U ||
        cloud->reference_time != metadata.timing.reference_time ||
        cloud->reference_time != metadata.timing.measurement_timestamps.front() ||
        cloud->reference_time != state_pose->exact_time) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::FinalizedTarget,
          "accepted persistent LiDAR target payload has inconsistent metadata, state, or cloud"));
    }
    const SensorFactorBatchRef batch{lidar_sensor, metadata.batch_id};
    if (!pending_finalized_lidar.empty()) {
      const PendingFinalizedLidarSweep& previous = pending_finalized_lidar.back();
      if (state <= previous.state || cloud->reference_time <= previous.exact_time ||
          batch.batch_id <= previous.batch.batch_id) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::FinalizedTarget,
            "persistent LiDAR target admissions are not strictly monotonic"));
      }
    }
    pending_finalized_lidar.push_back(PendingFinalizedLidarSweep{
        batch, metadata, commit.revision, admission_kind, state, cloud->reference_time,
        calibration.epoch(), std::move(cloud), std::nullopt});
    statistics.finalized_lidar_pending_high_watermark =
        std::max(statistics.finalized_lidar_pending_high_watermark,
                 pending_finalized_lidar.size());
    return Result::success(true);
  }

  [[nodiscard]] core::Result<std::size_t, LocalEstimatorError>
  removePendingFinalizedTargetBatches(std::span<const SensorFactorBatchRef> batches) {
    using Result = core::Result<std::size_t, LocalEstimatorError>;
    std::set<SensorFactorBatchRef> requested;
    const core::SensorInstanceId lidar_sensor =
        core::SensorInstanceId::lidar(calibration.lidar().id());
    for (const SensorFactorBatchRef& batch : batches) {
      if (batch.sensor != lidar_sensor || !batch.batch_id.valid() || !requested.insert(batch).second) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::FinalizedTarget,
            "persistent LiDAR rollback request has an invalid or duplicate batch identity"));
      }
    }
    for (const SensorFactorBatchRef& batch : requested) {
      const auto found = std::find_if(pending_finalized_lidar.begin(),
                                      pending_finalized_lidar.end(), [&](const auto& pending) {
                                        return pending.batch == batch;
                                      });
      if (found == pending_finalized_lidar.end() || found->finality) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::FinalizedTarget,
            "removable graph FactorBatch has no unique pre-finality persistent payload"));
      }
    }
    const std::size_t before = pending_finalized_lidar.size();
    std::erase_if(pending_finalized_lidar, [&](const auto& pending) {
      return requested.contains(pending.batch);
    });
    const std::size_t removed = before - pending_finalized_lidar.size();
    statistics.finalized_lidar_rollback_removals += removed;
    finalized_target_process.rollback_removals += removed;
    return Result::success(removed);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> drainFinalizedTarget() {
    using Result = core::Result<bool, LocalEstimatorError>;
    detail::LocalPipelineTimingScope timing_scope(
        pipeline_timing, LocalPipelineTimingStage::FinalizedTargetUpdate);
    const core::SensorInstanceId lidar_sensor =
        core::SensorInstanceId::lidar(calibration.lidar().id());
    auto health = sensor_health.snapshot(lidar_sensor);
    if (!health) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::SensorHealthFailed, LocalEstimatorStage::SensorHealth,
          "persistent LiDAR target cannot read LiDAR health: " + health.error().detail));
    }
    std::size_t ready = 0U;
    for (const PendingFinalizedLidarSweep& pending : pending_finalized_lidar) {
      ready += pending.finality.has_value() ? 1U : 0U;
    }
    if (health.value().health.state != core::SensorHealthState::Active) {
      if (ready > 0U && !finalized_ready_blocked) {
        ++statistics.finalized_lidar_freeze_events;
      }
      finalized_target_frozen = true;
      finalized_ready_blocked = ready > 0U;
      statistics.finalized_lidar_frozen_high_watermark =
          std::max(statistics.finalized_lidar_frozen_high_watermark, ready);
      timing_scope.finish();
      return Result::success(true);
    }
    finalized_target_frozen = false;
    finalized_ready_blocked = false;

    const auto make_sweep = [](const PendingFinalizedLidarSweep& pending) {
      FinalizedLidarSweep sweep;
      sweep.batch = pending.batch;
      sweep.accepted_batch_metadata = pending.accepted_batch_metadata;
      sweep.admission_revision = pending.admission_revision;
      sweep.admission_kind = pending.admission_kind;
      sweep.finalized_state = *pending.finality;
      sweep.calibration = pending.calibration;
      sweep.cloud = pending.cloud;
      return sweep;
    };
    const auto record_prune = [&](const FinalizedLidarTargetPruneStats& pruned) {
      finalized_target_process.prunes.push_back(pruned);
      if (pruned.removed_points > 0U) {
        ++statistics.finalized_lidar_prune_transactions;
        statistics.finalized_lidar_pruned_points += pruned.removed_points;
      }
    };
    const auto record_capacity_skip = [&](const PendingFinalizedLidarSweep& pending,
                                          FinalizedLidarTargetCapacitySkipReason reason) {
      const FinalizedLidarTargetMapStatistics& map = finalized_target.statistics();
      finalized_target_process.capacity_skips.push_back(FinalizedLidarTargetCapacitySkip{
          pending.batch,
          pending.state,
          pending.exact_time,
          pending.finality->final_revision,
          pending.cloud->source_sweep,
          pending.cloud->checksum,
          pending.cloud->points.size(),
          map.retained_points,
          finalized_target.config().hard_point_capacity,
          map.version,
          map.checksum,
          reason});
      ++statistics.finalized_lidar_capacity_skipped_sweeps;
      if (reason == FinalizedLidarTargetCapacitySkipReason::RetrySuppressedWhileSaturated) {
        ++statistics.finalized_lidar_capacity_retry_suppressions;
      }
      // The counter is bounded by the existing positive prune interval: once
      // it reaches that cadence, the next finalized payload is allowed one
      // insertion/prune/retry recovery attempt.
      if (finalized_capacity_skips_since_retry <
          config.finalized_lidar_prune_interval_sweeps) {
        ++finalized_capacity_skips_since_retry;
      }
    };

    std::optional<Eigen::Vector3d> latest_origin;
    while (!pending_finalized_lidar.empty() && pending_finalized_lidar.front().finality) {
      const PendingFinalizedLidarSweep& pending = pending_finalized_lidar.front();
      if (finalized_target_capacity_saturated &&
          finalized_capacity_skips_since_retry <
              config.finalized_lidar_prune_interval_sweeps) {
        record_capacity_skip(
            pending, FinalizedLidarTargetCapacitySkipReason::RetrySuppressedWhileSaturated);
        pending_finalized_lidar.pop_front();
        continue;
      }

      // A saturated target retries at most once per configured prune cadence.
      // Reset before the attempt so another failed recovery starts a fresh,
      // bounded suppression interval.
      if (finalized_target_capacity_saturated) {
        finalized_capacity_skips_since_retry = 0U;
      }
      auto inserted = finalized_target.insertFinalizedSweep(make_sweep(pending));
      bool capacity_recovery_attempted = false;
      if (!inserted &&
          inserted.error().code == FinalizedLidarTargetMapErrorCode::PointCapacity) {
        capacity_recovery_attempted = true;
        ++statistics.finalized_lidar_capacity_recovery_attempts;
        ++finalized_target_process.capacity_recovery_attempts;
        ++statistics.finalized_lidar_prune_attempts;
        const Eigen::Vector3d exact_final_origin =
            pending.finality->final_estimate.T_odom_imu.translation();
        auto pruned = finalized_target.pruneAround(exact_final_origin);
        if (!pruned) {
          return Result::failure(estimatorError(
              LocalEstimatorErrorCode::FinalizedTargetFailed,
              LocalEstimatorStage::FinalizedTarget,
              "capacity-recovery pruning of the finalized LiDAR target failed: " +
                  pruned.error().detail));
        }
        record_prune(pruned.value());
        finalized_insertions_since_prune = 0U;
        // The first failed insertion is semantically atomic, so this is an
        // identical immutable payload retry against only the explicitly
        // pruned map version.
        inserted = finalized_target.insertFinalizedSweep(make_sweep(pending));
      }
      if (!inserted) {
        if (inserted.error().code == FinalizedLidarTargetMapErrorCode::PointCapacity) {
          finalized_target_capacity_saturated = true;
          record_capacity_skip(
              pending, FinalizedLidarTargetCapacitySkipReason::RetryAfterPruneStillFull);
          pending_finalized_lidar.pop_front();
          continue;
        }
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::FinalizedTargetFailed,
            LocalEstimatorStage::FinalizedTarget,
            "finalized LiDAR target rejected an exact graph-final payload: " +
                inserted.error().detail));
      }
      if (capacity_recovery_attempted) {
        ++statistics.finalized_lidar_capacity_recovery_successes;
        ++finalized_target_process.capacity_recovery_successes;
      }
      if (!inserted.value().owner) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant,
            LocalEstimatorStage::FinalizedTarget,
            "successful finalized LiDAR admission did not expose its immutable owner"));
      }
      finalized_target_capacity_saturated = false;
      finalized_capacity_skips_since_retry = 0U;
      latest_origin = pending.finality->final_estimate.T_odom_imu.translation();
      finalized_target_process.insertions.push_back(inserted.value());
      ++statistics.finalized_lidar_insertions;
      statistics.finalized_lidar_inserted_points += inserted.value().admitted_points;
      ++finalized_insertions_since_prune;
      pending_finalized_lidar.pop_front();
    }

    if (latest_origin &&
        finalized_insertions_since_prune >= config.finalized_lidar_prune_interval_sweeps) {
      ++statistics.finalized_lidar_prune_attempts;
      auto pruned = finalized_target.pruneAround(*latest_origin);
      if (!pruned) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::FinalizedTargetFailed,
            LocalEstimatorStage::FinalizedTarget,
            "periodic finalized LiDAR target pruning failed: " + pruned.error().detail));
      }
      record_prune(pruned.value());
      finalized_insertions_since_prune = 0U;
    }
    timing_scope.finish();
    return Result::success(true);
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> reconcileFinalizedTarget(
      const LocalGraphCommit& commit) {
    using Result = core::Result<bool, LocalEstimatorError>;
    for (const LocalGraphFinalizedState& finalized : commit.finalized_states) {
      const auto found = std::find_if(pending_finalized_lidar.begin(),
                                      pending_finalized_lidar.end(), [&](const auto& pending) {
                                        return pending.state == finalized.state;
                                      });
      if (found == pending_finalized_lidar.end()) {
        continue;
      }
      if (found->finality || finalized.odom_epoch != config.odom_epoch ||
          finalized.exact_time != found->exact_time ||
          finalized.final_revision != commit.revision) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::FinalizedTarget,
            "graph finality does not exactly match its staged LiDAR target payload"));
      }
      found->finality = finalized;
      ++statistics.finalized_lidar_finality_matches;
      ++finalized_target_process.finality_matches;
    }
    return drainFinalizedTarget();
  }

  void refreshFinalizedTargetProcessSnapshot() noexcept {
    finalized_target_process.pending_sweeps = pending_finalized_lidar.size();
    finalized_target_process.pending_unfinalized_sweeps = 0U;
    finalized_target_process.finalized_ready_sweeps = 0U;
    for (const PendingFinalizedLidarSweep& pending : pending_finalized_lidar) {
      if (pending.finality) {
        ++finalized_target_process.finalized_ready_sweeps;
      } else {
        ++finalized_target_process.pending_unfinalized_sweeps;
      }
    }
    finalized_target_process.insertion_frozen = finalized_target_frozen;
    finalized_target_process.capacity_saturated = finalized_target_capacity_saturated;
    finalized_target_process.capacity_skips_since_retry =
        finalized_capacity_skips_since_retry;
    const core::SensorInstanceId lidar_sensor =
        core::SensorInstanceId::lidar(calibration.lidar().id());
    const auto health = sensor_health.snapshot(lidar_sensor);
    finalized_target_process.lidar_health = health ? health.value().health.state
                                                   : core::SensorHealthState::Failed;
    const FinalizedLidarTargetMapStatistics& map = finalized_target.statistics();
    finalized_target_process.retained_points = map.retained_points;
    finalized_target_process.map_version = map.version;
    finalized_target_process.map_checksum = map.checksum;
  }

  [[nodiscard]] core::Result<RollingLidarTargetPoseSynchronizationStats, LocalEstimatorError>
  synchronizeRollingTarget(const LocalGraphCommit& commit) {
    using Result = core::Result<RollingLidarTargetPoseSynchronizationStats, LocalEstimatorError>;
    RollingLidarTargetPoseSynchronizationStats rolling_stats;

    if (!rolling_target.empty()) {
      std::vector<RollingLidarTargetPose> poses;
      poses.reserve(commit.navigation_poses.size());
      for (const LocalGraphPoseSnapshot& pose : commit.navigation_poses) {
        poses.push_back(RollingLidarTargetPose{pose.state, pose.T_odom_imu});
      }
      std::vector<core::StateId> finalized_states;
      finalized_states.reserve(commit.finalized_states.size());
      for (const LocalGraphFinalizedState& finalized : commit.finalized_states) {
        finalized_states.push_back(finalized.state);
      }

      core::PipelineWorkIdentity work;
      work.state = commit.state;
      const core::ThreadCpuWallTimer synchronization_timer;
      auto synchronized = rolling_target.synchronizeCommittedPoses(
          commit.odom_epoch, poses, finalized_states);
      detail::observeLocalPipelineTiming(
          pipeline_timing, LocalPipelineTimingStage::TargetBuildUpdate, synchronization_timer,
          synchronized ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
          work);
      if (!synchronized) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::RollingTargetFailed, LocalEstimatorStage::RollingTarget,
            "committed graph poses could not synchronize the rolling target: " +
                synchronized.error().detail));
      }
      rolling_stats = synchronized.value();
      ++statistics.rolling_target_pose_synchronizations;
      statistics.rolling_target_sweeps_synchronized +=
          synchronized.value().matched_retained_sweeps;
      statistics.rolling_target_finalized_sweeps_evicted +=
          synchronized.value().finalized_sweeps_evicted;
      statistics.rolling_target_finalized_points_evicted +=
          synchronized.value().finalized_points_evicted;
    }
    auto reconciled = reconcileFinalizedTarget(commit);
    if (!reconciled) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(reconciled.error());
    }
    return Result::success(std::move(rolling_stats));
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> recordGraphPublishedState(
      const LocalGraphCommit& commit) {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto recorded = state_timeline.recordCommittedState(commit.state, commit.state_time);
    if (!recorded) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::StateTimelineFailed, LocalEstimatorStage::StateTimeline,
          "graph-published state could not bind to the state timeline: " +
              recorded.error().detail));
    }
    return Result::success(true);
  }

  [[nodiscard]] core::Result<GraphAppendOutcome, LocalEstimatorError> appendSensor(
      ImuKnotAppend navigation, bool* graph_commit_applied = nullptr,
      detail::IdentityCounters* staged_identity_counters = nullptr) {
    using Result = core::Result<GraphAppendOutcome, LocalEstimatorError>;
    if (graph_commit_applied != nullptr) {
      *graph_commit_applied = false;
    }
    std::optional<PreparedVisualAttachment> visual;
    for (std::size_t offset = 0U; offset < visual_lanes.size(); ++offset) {
      const std::size_t lane_index = (next_visual_attachment_lane + offset) % visual_lanes.size();
      const VisualLaneQueueState queues = visual_lanes[lane_index].lane.queueState();
      if (queues.pending_factor_batches == 0U && queues.pending_factor_retirements == 0U) {
        continue;
      }
      const detail::IdentityCounters candidate_counters =
          staged_identity_counters == nullptr ? identityCounters() : *staged_identity_counters;
      auto prepared = prepareVisualAttachment(lane_index, navigation, candidate_counters);
      if (!prepared) {
        return Result::failure(prepared.error());
      }
      visual = std::move(prepared).value();
      break;
    }

    SensorKnotAppend transaction;
    transaction.navigation = navigation;
    if (visual) {
      transaction.visual = visual->graph_batch;
      transaction.visual_factor_retirements = visual->graph_retirements;
    }
    auto appended = graph.appendSensorKnot(std::move(transaction));
    if (!appended) {
      if (visual) {
        const LocalGraphError visual_transaction_error = appended.error();
        // The graph writer is candidate-copy transactional. Retry the exact
        // same navigation append without any visual batch or retirement
        // to prove the rejected visual candidate did not mutate graph state.
        SensorKnotAppend navigation_retry;
        navigation_retry.navigation = std::move(navigation);
        auto recovered = graph.appendSensorKnot(std::move(navigation_retry));
        VisualLaneState& lane = visual_lanes[visual->lane_index];
        if (recovered) {
          if (graph_commit_applied != nullptr) {
            *graph_commit_applied = true;
          }
          GraphAppendOutcome outcome;
          outcome.commit = std::move(recovered).value();
          auto recorded = recordGraphPublishedState(outcome.commit);
          if (!recorded) {
            return Result::failure(recorded.error());
          }
          auto synchronized = synchronizeRollingTarget(outcome.commit);
          if (!synchronized) {
            return Result::failure(synchronized.error());
          }
          auto reconciled = lane.lane.acknowledgeGraphInputDegraded(visual->lane_attachment);
          if (!reconciled) {
            lifecycle = LocalEstimatorLifecycle::Faulted;
            return Result::failure(estimatorError(
                LocalEstimatorErrorCode::InternalInvariant,
                LocalEstimatorStage::VisualGraphAttachment,
                "committed navigation recovery could not reconcile its rejected visual "
                "attachment: " +
                    reconciled.error().detail));
          }

          auto finality = reconcileVisualFinality(outcome.commit);
          if (!finality) {
            return Result::failure(finality.error());
          }
          VisualGraphDegradationReport degradation;
          degradation.camera = lane.camera;
          degradation.lane_attachment = visual->lane_attachment;
          degradation.transaction_state = outcome.commit.state;
          degradation.committed_revision = outcome.commit.revision;
          degradation.rejected_graph_error_code = visual_transaction_error.code;
          degradation.rejection_detail = visual_transaction_error.detail;
          degradation.factor_batches_discarded = reconciled.value().factor_batches_discarded;
          degradation.landmark_seeds_discarded = reconciled.value().landmark_seeds_discarded;
          degradation.factors_discarded = reconciled.value().factors_discarded;
          degradation.stale_track_observations_discarded =
              reconciled.value().stale_track_observations_discarded;
          degradation.factor_retirements_preserved =
              reconciled.value().factor_retirements_preserved;
          ++statistics.visual_graph_degradations;
          statistics.visual_factor_batches_discarded += degradation.factor_batches_discarded;
          statistics.visual_landmark_seeds_discarded += degradation.landmark_seeds_discarded;
          statistics.visual_factor_specs_discarded += degradation.factors_discarded;
          statistics.visual_stale_track_observations_discarded +=
              degradation.stale_track_observations_discarded;
          next_visual_attachment_lane =
              visual_lanes.empty() ? 0U : (visual->lane_index + 1U) % visual_lanes.size();
          outcome.visual_degradation = std::move(degradation);
          return Result::success(std::move(outcome));
        }

        auto rejected = lane.lane.acknowledgeGraphInputRejected(visual->lane_attachment);
        if (!rejected) {
          return Result::failure(estimatorError(
              LocalEstimatorErrorCode::InternalInvariant,
              LocalEstimatorStage::VisualGraphAttachment,
              "failed navigation recovery could not release its visual attachment: " +
                  rejected.error().detail));
        }
        LocalEstimatorError error = estimatorError(
            LocalEstimatorErrorCode::GraphTransactionFailed, LocalEstimatorStage::LocalGraph,
            "navigation retry without visual failed: " + recovered.error().detail +
                "; original transaction with visual failed: " + visual_transaction_error.detail);
        error.graph_error_code = recovered.error().code;
        error.lidar_registration = recovered.error().lidar_registration;
        error.lidar_pairs = recovered.error().lidar_pairs;
        error.lidar_finalized_map = recovered.error().lidar_finalized_map;
        error.rejected_solve = recovered.error().rejected_solve;
        return Result::failure(std::move(error));
      }
      LocalEstimatorError error = estimatorError(
          LocalEstimatorErrorCode::GraphTransactionFailed, LocalEstimatorStage::LocalGraph,
          "atomic sensor graph transaction failed: " + appended.error().detail);
      error.graph_error_code = appended.error().code;
      error.lidar_registration = appended.error().lidar_registration;
      error.lidar_pairs = appended.error().lidar_pairs;
      error.lidar_finalized_map = appended.error().lidar_finalized_map;
      error.rejected_solve = appended.error().rejected_solve;
      return Result::failure(std::move(error));
    }

    GraphAppendOutcome outcome;
    outcome.commit = std::move(appended).value();
    if (graph_commit_applied != nullptr) {
      *graph_commit_applied = true;
    }
    auto recorded = recordGraphPublishedState(outcome.commit);
    if (!recorded) {
      return Result::failure(recorded.error());
    }
    auto synchronized = synchronizeRollingTarget(outcome.commit);
    if (!synchronized) {
      return Result::failure(synchronized.error());
    }
    if (visual) {
      VisualLaneState& lane = visual_lanes[visual->lane_index];
      auto committed_identities = visual->identity_transaction->commit();
      if (!committed_identities) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::IdentityAllocation,
            "committed graph visual identities were already consumed"));
      }
      if (staged_identity_counters == nullptr) {
        publishIdentityCounters(committed_identities.value().counters);
      } else {
        *staged_identity_counters = committed_identities.value().counters;
      }
      lane.identities = std::move(committed_identities).value().state;
      for (const VisualLandmarkId local : visual->local_landmarks) {
        lane.identities.accepted_landmarks.insert(local);
      }
      for (const core::FactorId local : visual->local_factors) {
        lane.identities.accepted_factors.insert(local);
      }
      auto accepted =
          lane.lane.acknowledgeGraphInputAccepted(visual->lane_attachment, outcome.commit.revision);
      if (!accepted) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::VisualGraphAttachment,
            "committed graph transaction could not acknowledge its visual "
            "attachment: " +
                accepted.error().detail));
      }
      auto retired_identities = removeCommittedVisualRetirements(*visual);
      if (!retired_identities) {
        return Result::failure(retired_identities.error());
      }

      VisualGraphAttachmentReport report;
      report.camera = lane.camera;
      report.lane_attachment = visual->lane_attachment;
      report.transaction_state = outcome.commit.state;
      report.committed_revision = outcome.commit.revision;
      if (visual->graph_batch) {
        for (const VisualLandmarkSeed& seed : visual->graph_batch->new_landmarks) {
          report.graph_landmarks.push_back(seed.landmark);
        }
        for (const VisualReprojectionFactorSpec& factor : visual->graph_batch->factors) {
          report.graph_factors.push_back(factor.id);
        }
      }
      report.graph_factor_retirements = visual->graph_retirements;
      ++statistics.visual_graph_attachments;
      statistics.visual_landmarks_attached += report.graph_landmarks.size();
      statistics.visual_factors_attached += report.graph_factors.size();
      statistics.visual_factors_retired += report.graph_factor_retirements.size();
      next_visual_attachment_lane =
          visual_lanes.empty() ? 0U : (visual->lane_index + 1U) % visual_lanes.size();
      outcome.visual_attachment = std::move(report);
    }
    auto finality = reconcileVisualFinality(outcome.commit);
    if (!finality) {
      return Result::failure(finality.error());
    }
    return Result::success(std::move(outcome));
  }

  [[nodiscard]] core::Result<core::ObservationLineage, LocalEstimatorError> graphLineage(
      const ImuInterval& interval, detail::IdentityCounters& staged_counters) {
    using Result = core::Result<core::ObservationLineage, LocalEstimatorError>;
    const auto lineage_id = allocate<core::ObservationLineageId>(staged_counters.next_lineage);
    const auto consumer = allocate<core::DerivedRecordId>(staged_counters.next_derived);
    const auto imu_group = allocate<core::FactorGroupId>(staged_counters.next_factor_group);
    if (!lineage_id || !consumer || !imu_group) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
          "observation lineage, derived record, or factor-group identity exhausted"));
    }

    core::ObservationLineage lineage;
    lineage.id = *lineage_id;
    std::unordered_set<std::uint64_t> admitted_imu;
    for (const core::MeasurementId measurement : interval.raw_measurements) {
      if (!measurement.valid() || !admitted_imu.insert(measurement.value()).second) {
        continue;
      }
      core::ObservationSlice slice;
      slice.root = measurement;
      slice.calibration = calibration.epoch();
      lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::PrimaryResidual,
                                                     *consumer, *imu_group, std::nullopt});
    }

    if (lineage.usage.empty() ||
        core::validateLineage(lineage) != core::LineageValidationError::None) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "constructed local factor ancestry is empty or invalid"));
    }
    return Result::success(std::move(lineage));
  }

  [[nodiscard]] core::Result<core::ObservationLineage, LocalEstimatorError> bootstrapLidarLineage(
      core::MeasurementId previous, core::MeasurementId current,
      const ImuInterval& previous_acquisition, const ImuInterval& current_acquisition,
      const ImuInterval& segment_imu, detail::IdentityCounters& counters) {
    using Result = core::Result<core::ObservationLineage, LocalEstimatorError>;
    const auto lineage_id = allocate<core::ObservationLineageId>(counters.next_lineage);
    const auto consumer = allocate<core::DerivedRecordId>(counters.next_derived);
    const auto factor_group = allocate<core::FactorGroupId>(counters.next_factor_group);
    const auto correlation = allocate<core::CorrelationGroupId>(counters.next_correlation_group);
    if (!lineage_id || !consumer || !factor_group || !correlation) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                            LocalEstimatorStage::IdentityAllocation,
                                            "bootstrap lineage identity exhausted"));
    }

    core::ObservationSlice source;
    source.root = current;
    source.calibration = calibration.epoch();
    core::ObservationSlice target;
    target.root = previous;
    target.calibration = calibration.epoch();

    core::ObservationLineage lineage;
    lineage.id = *lineage_id;
    lineage.usage.push_back(core::ObservationUsage{source, core::ObservationRole::PrimaryResidual,
                                                   *consumer, *factor_group, std::nullopt});
    lineage.usage.push_back(core::ObservationUsage{target, core::ObservationRole::ConditioningOnly,
                                                   *consumer, std::nullopt, *correlation});
    std::set<std::uint64_t> conditioned_imu;
    const auto admit_conditioning = [&](const ImuInterval& interval) {
      for (core::MeasurementId measurement : interval.raw_measurements) {
        if (!conditioned_imu.insert(measurement.value()).second) {
          continue;
        }
        core::ObservationSlice slice;
        slice.root = measurement;
        slice.calibration = calibration.epoch();
        lineage.usage.push_back(core::ObservationUsage{
            slice, core::ObservationRole::ConditioningOnly, *consumer, std::nullopt, *correlation});
      }
    };
    admit_conditioning(previous_acquisition);
    admit_conditioning(current_acquisition);
    admit_conditioning(segment_imu);
    lineage.correlations.push_back(core::CorrelationDeclaration{
        *correlation, core::CorrelationPolicyRevision{1U},
        core::CorrelationTreatment::CovarianceInflationAndInformationCap,
        config.lidar_bootstrap.target_reuse_covariance_inflation *
            config.lidar_bootstrap.imu_conditioning_covariance_inflation,
        config.lidar_bootstrap.registration.maximum_translation_information /
            (config.lidar_bootstrap.target_reuse_covariance_inflation *
             config.lidar_bootstrap.imu_conditioning_covariance_inflation)});
    if (core::validateLineage(lineage) != core::LineageValidationError::None) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::InternalInvariant,
                                            "constructed bootstrap LiDAR lineage is invalid"));
    }
    return Result::success(std::move(lineage));
  }

  [[nodiscard]] core::Result<core::ObservationLineage, LocalEstimatorError>
  motionInitializationLineage(const std::vector<MotionInitializationSegment>& segments,
                              const MotionInitialization& initialization,
                              detail::IdentityCounters& counters) {
    using Result = core::Result<core::ObservationLineage, LocalEstimatorError>;
    const auto lineage_id = allocate<core::ObservationLineageId>(counters.next_lineage);
    const auto consumer = allocate<core::DerivedRecordId>(counters.next_derived);
    const auto factor_group = allocate<core::FactorGroupId>(counters.next_factor_group);
    if (!lineage_id || !consumer || !factor_group) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                            LocalEstimatorStage::IdentityAllocation,
                                            "motion-initialization lineage identity exhausted"));
    }

    core::ObservationLineage lineage;
    lineage.id = *lineage_id;
    auto admit = [&](const core::ObservationLineage& source, MotionInitializationLineageRole role) {
      for (const core::ObservationUsage& usage : source.usage) {
        const bool primary = role == MotionInitializationLineageRole::FusedResidual &&
                             usage.role == core::ObservationRole::PrimaryResidual;
        const auto existing = std::find_if(lineage.usage.begin(), lineage.usage.end(),
                                           [&](const core::ObservationUsage& admitted) {
                                             return admitted.slice.overlaps(usage.slice);
                                           });
        if (existing != lineage.usage.end()) {
          if (primary && existing->role != core::ObservationRole::PrimaryResidual) {
            existing->role = core::ObservationRole::PrimaryResidual;
            existing->factor_group = *factor_group;
            existing->correlation_group.reset();
          }
          continue;
        }
        lineage.usage.push_back(core::ObservationUsage{
            usage.slice,
            primary ? core::ObservationRole::PrimaryResidual
                    : core::ObservationRole::ConditioningOnly,
            *consumer, primary ? std::optional<core::FactorGroupId>{*factor_group} : std::nullopt,
            std::nullopt});
      }
    };
    std::map<core::ObservationLineageId, MotionInitializationLineageRole> uses;
    for (const MotionInitializationLineageUse& use : initialization.lineage_uses) {
      uses.emplace(use.lineage, use.role);
    }
    for (const MotionInitializationSegment& segment : segments) {
      if (const auto found = uses.find(segment.lidar.lineage.id); found != uses.end()) {
        admit(segment.lidar.lineage, found->second);
      }
      if (const auto found = uses.find(segment.imu_lineage.id); found != uses.end()) {
        admit(segment.imu_lineage, found->second);
      }
    }
    if (lineage.usage.empty() ||
        core::validateLineage(lineage) != core::LineageValidationError::None) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::InternalInvariant,
                                            "motion-initialization ancestry is empty or invalid"));
    }
    return Result::success(std::move(lineage));
  }

  [[nodiscard]] MotionImuNoise motionImuNoise() const {
    MotionImuNoise noise;
    noise.gravity_odom = config.graph.imu.gravity_odom;
    noise.accelerometer_noise_density_mps2_sqrt_hz =
        config.graph.imu.accelerometer_noise_density_mps2_sqrt_hz;
    noise.gyroscope_noise_density_radps_sqrt_hz =
        config.graph.imu.gyroscope_noise_density_radps_sqrt_hz;
    noise.integration_noise_density = config.graph.imu.integration_noise_density;
    noise.accelerometer_bias_prior_mean_mps2 =
        config.graph.imu.initial_accelerometer_bias_mean_mps2;
    noise.gyroscope_bias_prior_mean_radps = config.graph.imu.initial_gyroscope_bias_mean_radps;
    noise.accelerometer_bias_prior_sigma_mps2 =
        config.graph.imu.initial_accelerometer_bias_sigma_mps2;
    noise.gyroscope_bias_prior_sigma_radps = config.graph.imu.initial_gyroscope_bias_sigma_radps;
    return noise;
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> resetBootstrap() {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto replacement = LidarBootstrapOdometry::create(
        calibration.epoch(), calibration.lidar().extrinsics().T_imu_lidar(),
        config.graph.imu.gravity_odom, config.graph.imu.initial_gyroscope_bias_mean_radps,
        config.lidar_bootstrap, pipeline_timing);
    if (!replacement) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::LidarBootstrap,
          "validated LiDAR bootstrap profile could not be recreated: " +
              replacement.error().detail));
    }
    lidar_bootstrap = std::move(replacement).value();
    motion_segments.clear();
    last_bootstrap_time.reset();
    last_bootstrap_measurement.reset();
    last_bootstrap_sweep.reset();
    last_bootstrap_acquisition_imu.reset();
    return Result::success(true);
  }

  [[nodiscard]] core::Result<GraphAppendOutcome, LocalEstimatorError> commitImuOnly(
      core::StateId state, core::FusionTime time, const ImuInterval& interval,
      bool* graph_commit_applied = nullptr,
      detail::IdentityCounters* staged_identity_counters = nullptr) {
    using Result = core::Result<GraphAppendOutcome, LocalEstimatorError>;
    if (graph_commit_applied != nullptr) {
      *graph_commit_applied = false;
    }
    std::optional<detail::IdentityTransaction<bool>> owned_identities;
    if (staged_identity_counters == nullptr) {
      owned_identities.emplace(identityCounters(), false);
      staged_identity_counters = &owned_identities->counters();
    }
    auto lineage = graphLineage(interval, *staged_identity_counters);
    if (!lineage) {
      if (owned_identities) {
        static_cast<void>(owned_identities->abort());
      }
      return Result::failure(lineage.error());
    }
    bool committed = false;
    auto commit = appendSensor(ImuKnotAppend{state, time, interval, std::move(lineage).value()},
                               &committed, staged_identity_counters);
    if (graph_commit_applied != nullptr) {
      *graph_commit_applied = committed;
    }
    if (!commit) {
      if (owned_identities && committed) {
        auto published = owned_identities->commit();
        if (!published) {
          lifecycle = LocalEstimatorLifecycle::Faulted;
          return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                                LocalEstimatorStage::IdentityAllocation,
                                                "committed IMU identities were already consumed"));
        }
        publishIdentityCounters(published.value().counters);
      } else if (owned_identities) {
        static_cast<void>(owned_identities->abort());
      }
      return Result::failure(commit.error());
    }
    if (owned_identities) {
      auto published = owned_identities->commit();
      if (!published) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
        return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                              LocalEstimatorStage::IdentityAllocation,
                                              "successful IMU identities were already consumed"));
      }
      publishIdentityCounters(published.value().counters);
    }
    return Result::success(std::move(commit).value());
  }

  [[nodiscard]] core::Result<SensorHealthUpdate, LocalEstimatorError> observeLidarHealth(
      core::FactorBatchId batch_id, core::FusionTime source_time, core::FusionTime support_end,
      SensorBatchHealthResult result) {
    using Result = core::Result<SensorHealthUpdate, LocalEstimatorError>;
    const core::SensorInstanceId sensor = core::SensorInstanceId::lidar(calibration.lidar().id());
    auto before = sensor_health.snapshot(sensor);
    if (!before) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::SensorHealthFailed, LocalEstimatorStage::SensorHealth,
          "LiDAR health snapshot is unavailable: " + before.error().detail));
    }
    core::FusionTime assessed_at = std::max(source_time + core::Duration{1LL}, support_end);
    if (assessed_at <= before.value().health.assessed_at) {
      if (before.value().health.assessed_at.nanoseconds ==
          std::numeric_limits<std::int64_t>::max()) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
        return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                              LocalEstimatorStage::SensorHealth,
                                              "LiDAR health assessment time is exhausted"));
      }
      assessed_at = before.value().health.assessed_at + core::Duration{1LL};
    }
    if (assessed_at.nanoseconds > std::numeric_limits<std::int64_t>::max() - 2LL) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                            LocalEstimatorStage::SensorHealth,
                                            "LiDAR health publication time is exhausted"));
    }
    const SensorBatchEvaluationMode mode =
        before.value().health.state == core::SensorHealthState::Active
            ? SensorBatchEvaluationMode::Primary
            : SensorBatchEvaluationMode::Shadow;
    auto update = sensor_health.observe(SensorHealthBatchObservation{
        sensor, batch_id, before.value().health.recovery_epoch, assessed_at, result, mode});
    if (!update) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::SensorHealthFailed, LocalEstimatorStage::SensorHealth,
          "LiDAR health update failed: " + update.error().detail));
    }
    if (mode == SensorBatchEvaluationMode::Shadow) {
      ++statistics.lidar_shadow_evaluations;
    }
    if (update.value().transitioned) {
      ++statistics.lidar_health_transitions;
    }
    return Result::success(std::move(update).value());
  }

  [[nodiscard]] core::Result<FailedBatchRemoval, LocalEstimatorError>
  removeRecentBatchesAfterFailure(const SensorHealthUpdate& health) {
    using Result = core::Result<FailedBatchRemoval, LocalEstimatorError>;
    FailedBatchRemoval output;
    if (!health.transitioned || health.after.state != core::SensorHealthState::Failed) {
      return Result::success(std::move(output));
    }
    struct Candidate {
      SensorFactorBatchRef batch;
      core::LocalGraphRevision revision;
    };
    std::vector<Candidate> candidates;
    const core::SensorInstanceId sensor = core::SensorInstanceId::lidar(calibration.lidar().id());
    for (const FactorBatchProvenance& provenance : graph.factorBatchJournal()) {
      if (provenance.status == FactorBatchJournalStatus::Active && provenance.removable &&
          provenance.batch.metadata.sensor == sensor) {
        candidates.push_back(
            Candidate{SensorFactorBatchRef{sensor, provenance.batch.metadata.batch_id},
                      provenance.inserted_revision});
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
      return rhs.revision < lhs.revision;
    });
    const std::size_t count =
        std::min(candidates.size(), config.maximum_recent_faulty_batches_to_remove);
    output.batches.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
      output.batches.push_back(candidates[index].batch);
    }
    if (output.batches.empty()) {
      return Result::success(std::move(output));
    }
    auto removed = graph.removeFactorBatches(
        FactorBatchRemovalRequest{output.batches, FactorBatchRemovalReason::SensorFailure});
    if (!removed) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::GraphTransactionFailed, LocalEstimatorStage::LocalGraph,
          "failed-sensor FactorBatch removal was rejected: " + removed.error().detail));
    }
    std::vector<core::FactorBatchId> removed_batch_ids;
    removed_batch_ids.reserve(output.batches.size());
    for (const SensorFactorBatchRef& batch : output.batches) {
      removed_batch_ids.push_back(batch.batch_id);
    }
    auto target_removal = rolling_target.removeFactorBatches(config.odom_epoch, removed_batch_ids);
    if (!target_removal) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::RollingTargetFailed, LocalEstimatorStage::RollingTarget,
          "failed-sensor target-payload removal was rejected: " + target_removal.error().detail));
    }
    output.target_removal = target_removal.value();
    auto pending_removal = removePendingFinalizedTargetBatches(output.batches);
    if (!pending_removal) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(pending_removal.error());
    }
    auto published = publishFactorOnlyCommit(removed.value());
    if (!published) {
      return Result::failure(published.error());
    }
    output.commit = std::move(removed).value();
    ++statistics.lidar_failure_removal_transactions;
    statistics.lidar_faulty_batches_removed += output.batches.size();
    statistics.lidar_faulty_target_sweeps_removed += target_removal.value().removed_sweeps;
    statistics.lidar_faulty_target_points_removed += target_removal.value().removed_points;
    return Result::success(std::move(output));
  }

  [[nodiscard]] core::Result<LidarHealthOutcome, LocalEstimatorError> processLidarHealth(
      core::FactorBatchId batch_id, core::FusionTime source_time, core::FusionTime support_end,
      SensorBatchHealthResult result) {
    using Result = core::Result<LidarHealthOutcome, LocalEstimatorError>;
    auto update = observeLidarHealth(batch_id, source_time, support_end, result);
    if (!update) {
      return Result::failure(update.error());
    }
    auto removal = removeRecentBatchesAfterFailure(update.value());
    if (!removal) {
      return Result::failure(removal.error());
    }
    auto drained = drainFinalizedTarget();
    if (!drained) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(drained.error());
    }
    return Result::success(
        LidarHealthOutcome{std::move(update).value(), std::move(removal).value()});
  }

  [[nodiscard]] core::Result<PreparedLidarFactorBatch, LocalEstimatorError> prepareLidarFactorBatch(
      const core::LidarSweep& sweep, core::FactorBatchId batch_id,
      const core::SensorHealthSnapshot& health, core::StateId source_state,
      core::FusionTime source_time, const std::shared_ptr<const LidarRegistrationCloud>& source,
      std::span<const LidarRegistrationTarget> selected_live_targets,
      LidarRegistrationResult registration,
      detail::IdentityCounters& staged_counters) {
    using Result = core::Result<PreparedLidarFactorBatch, LocalEstimatorError>;
    if (source_time.nanoseconds > std::numeric_limits<std::int64_t>::max() - 3LL) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
          "LiDAR factor timing cannot be represented without nanosecond overflow"));
    }
    const auto lineage_id = allocate<core::ObservationLineageId>(staged_counters.next_lineage);
    const auto consumer = allocate<core::DerivedRecordId>(staged_counters.next_derived);
    const auto factor_group = allocate<core::FactorGroupId>(staged_counters.next_factor_group);
    const auto correlation =
        allocate<core::CorrelationGroupId>(staged_counters.next_correlation_group);
    if (!batch_id.valid() || !lineage_id || !consumer || !factor_group || !correlation) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
          "LiDAR FactorBatch, lineage, consumer, factor-group, or correlation identity exhausted"));
    }

    if (!source || source->source_sweep != sweep.id || source->reference_time != source_time) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "LiDAR FactorBatch source cloud does not match its exact sweep and state time"));
    }
    const double base_covariance_inflation = config.lidar_target_reuse_covariance_inflation *
                                             config.lidar_imu_conditioning_covariance_inflation;
    const double raw_information_cap = config.lidar_registration.maximum_translation_information;
    if (!std::isfinite(base_covariance_inflation) || base_covariance_inflation <= 1.0) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
          "base LiDAR correlation inflation is invalid"));
    }

    const bool has_finalized_map = registration.finalized_map_snapshot != nullptr;
    const std::size_t expected_target_count =
        registration.target_snapshots.size() + (has_finalized_map ? 1U : 0U);
    if (registration.source_state != source_state || registration.source_time != source_time ||
        expected_target_count == 0U ||
        registration.diagnostics.target_count != expected_target_count ||
        registration.diagnostics.live_target_count != registration.target_snapshots.size() ||
        registration.diagnostics.finalized_map_target_count != (has_finalized_map ? 1U : 0U)) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "direct point ICP result source identity or mixed target count is inconsistent"));
    }
    std::sort(
        registration.target_snapshots.begin(), registration.target_snapshots.end(),
        [](const auto& lhs, const auto& rhs) { return lhs->targetState() < rhs->targetState(); });

    std::vector<const LidarRegistrationTarget*> matched_live_targets;
    matched_live_targets.reserve(registration.target_snapshots.size());
    for (const auto& snapshot : registration.target_snapshots) {
      const auto input = std::find_if(
          selected_live_targets.begin(), selected_live_targets.end(), [&](const auto& candidate) {
            return candidate.state == snapshot->targetState();
          });
      if (input == selected_live_targets.end() || !input->cloud ||
          input->time != snapshot->targetTime() ||
          input->cloud->source_sweep != snapshot->targetSweep() || snapshot->rows().empty()) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "direct point ICP snapshot is absent from its selected live target input"));
      }
      matched_live_targets.push_back(&*input);
    }

    double owner_pose_covariance_inflation = 1.0;
    std::vector<std::shared_ptr<const FinalizedLidarTargetOwner>> finalized_owners;
    if (has_finalized_map) {
      const auto& snapshot = registration.finalized_map_snapshot;
      if (snapshot->sourceState() != source_state || snapshot->sourceTime() != source_time ||
          snapshot->sourceSweep() != sweep.id ||
          snapshot->mapOdomEpoch() != config.odom_epoch ||
          snapshot->mapSensor() != core::SensorInstanceId::lidar(calibration.lidar().id()) ||
          snapshot->rows().empty()) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "finalized-map ICP snapshot has inconsistent source or map identity"));
      }
      auto inflation = lidarFinalizedMapOwnerPoseCovarianceInflation(snapshot);
      if (!inflation) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "finalized-map owner covariance inflation is not canonical: " +
                inflation.error().detail));
      }
      owner_pose_covariance_inflation = inflation.value();
      finalized_owners.assign(snapshot->owners().begin(), snapshot->owners().end());
      if (finalized_owners.empty() || finalized_owners.size() != snapshot->uniqueOwnerCount()) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "finalized-map ICP snapshot owner provenance is incomplete or ambiguous"));
      }
    }

    const double admitted_information_cap = raw_information_cap / base_covariance_inflation;
    const double live_information_scale = 1.0 / base_covariance_inflation;
    const double effective_map_covariance_inflation =
        std::max(owner_pose_covariance_inflation,
                 config.finalized_map_correlation_inflation_floor);
    const double finalized_map_covariance_inflation =
        base_covariance_inflation * effective_map_covariance_inflation;
    const double finalized_map_information_scale =
        1.0 / finalized_map_covariance_inflation;
    if (!std::isfinite(admitted_information_cap) || admitted_information_cap <= 0.0 ||
        !std::isfinite(live_information_scale) || live_information_scale <= 0.0 ||
        !std::isfinite(finalized_map_covariance_inflation) ||
        finalized_map_covariance_inflation <= 1.0 ||
        !std::isfinite(finalized_map_information_scale) ||
        finalized_map_information_scale <= 0.0) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
          "LiDAR base/map covariance inflation or admitted information cap is invalid"));
    }

    const auto source_ancestry = std::find_if(
        source->lineage.usage.begin(), source->lineage.usage.end(), [&](const auto& usage) {
          const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
          return measurement != nullptr && *measurement == sweep.id;
        });
    if (source_ancestry == source->lineage.usage.end()) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "sealed LiDAR source cloud has no exact source-sweep ancestry"));
    }

    core::ObservationLineage lineage;
    lineage.id = *lineage_id;
    lineage.usage.push_back(core::ObservationUsage{source_ancestry->slice,
                                                   core::ObservationRole::PrimaryResidual,
                                                   *consumer, *factor_group, std::nullopt});

    using ConditioningRootKey = std::pair<std::uint8_t, std::uint64_t>;
    std::map<ConditioningRootKey, core::ObservationSlice> conditioning_slices;
    const auto conditioning_root_key = [](const core::ObservationSlice& slice) {
      if (const auto* measurement = std::get_if<core::MeasurementId>(&slice.root)) {
        return ConditioningRootKey{0U, measurement->value()};
      }
      return ConditioningRootKey{1U, std::get<core::GnssObservationId>(slice.root).value()};
    };
    const auto add_conditioning_slice =
        [&](const core::ObservationSlice& candidate) -> std::optional<LocalEstimatorError> {
      const auto [existing, inserted] =
          conditioning_slices.emplace(conditioning_root_key(candidate), candidate);
      if (!inserted && !sameObservationSlice(existing->second, candidate)) {
        return estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                              LocalEstimatorStage::InternalInvariant,
                              "LiDAR conditioning ancestry contains one raw root with conflicting "
                              "slice identity");
      }
      return std::nullopt;
    };
    for (const core::ObservationUsage& ancestry : source->lineage.usage) {
      if (sameObservationSlice(ancestry.slice, source_ancestry->slice)) {
        continue;
      }
      if (auto error = add_conditioning_slice(ancestry.slice)) {
        return Result::failure(std::move(*error));
      }
    }
    std::set<core::MeasurementId> expected_conditioning_measurements(
        source->imu_support.begin(), source->imu_support.end());
    for (const LidarRegistrationTarget* target : matched_live_targets) {
      for (const core::ObservationUsage& ancestry : target->cloud->lineage.usage) {
        if (auto error = add_conditioning_slice(ancestry.slice)) {
          return Result::failure(std::move(*error));
        }
      }
      expected_conditioning_measurements.insert(target->cloud->source_sweep);
      expected_conditioning_measurements.insert(target->cloud->imu_support.begin(),
                                                target->cloud->imu_support.end());
    }
    for (const auto& owner : finalized_owners) {
      if (!owner ||
          owner->batch.sensor != core::SensorInstanceId::lidar(calibration.lidar().id()) ||
          owner->finalized_state.odom_epoch != config.odom_epoch ||
          owner->admission.health.sensor != owner->batch.sensor ||
          owner->admission.odom_epoch != owner->finalized_state.odom_epoch ||
          owner->admission.reference_time != owner->finalized_state.exact_time ||
          !owner->admission.map_eligible) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "finalized-map owner does not retain its accepted LiDAR identity"));
      }
      const auto owner_sweep = std::find_if(
          owner->cloud_lineage.usage.begin(), owner->cloud_lineage.usage.end(),
          [&](const core::ObservationUsage& usage) {
            const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
            return measurement != nullptr && *measurement == owner->sweep;
          });
      if (owner_sweep == owner->cloud_lineage.usage.end()) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "finalized-map owner has no exact source-sweep ancestry"));
      }
      if (auto error = add_conditioning_slice(owner_sweep->slice)) {
        return Result::failure(std::move(*error));
      }
      expected_conditioning_measurements.insert(owner->sweep);
    }
    std::set<core::MeasurementId> actual_conditioning_measurements;
    for (const auto& [root, slice] : conditioning_slices) {
      static_cast<void>(root);
      lineage.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::ConditioningOnly,
                                                     *consumer, std::nullopt, *correlation});
      if (const auto* measurement = std::get_if<core::MeasurementId>(&slice.root)) {
        actual_conditioning_measurements.insert(*measurement);
      }
    }
    lineage.correlations.push_back(core::CorrelationDeclaration{
        *correlation, core::CorrelationPolicyRevision{has_finalized_map ? 3U : 1U},
        core::CorrelationTreatment::CovarianceInflationAndInformationCap,
        base_covariance_inflation, admitted_information_cap});

    if (actual_conditioning_measurements != expected_conditioning_measurements) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "matched LiDAR target or deskew ancestry is missing, duplicated, or unexpected"));
    }
    if (core::validateLineage(lineage) != core::LineageValidationError::None) {
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::InternalInvariant,
                                            "constructed LiDAR FactorBatch lineage is invalid"));
    }
    auto checksum = lidarFactorLineageChecksum(lineage);
    if (!checksum) {
      return Result::failure(checksum.error());
    }
    lineage.checksum = checksum.value();

    LidarDirectFactorBatch batch;
    batch.metadata.header = sweep.header;
    batch.metadata.batch_id = batch_id;
    batch.metadata.odom_epoch = config.odom_epoch;
    batch.metadata.sensor = core::SensorInstanceId::lidar(calibration.lidar().id());
    batch.metadata.timing.support =
        core::TimeRange{sweep.acquisition.start,
                        std::max(sweep.acquisition.end, source_time + core::Duration{1LL})};
    batch.metadata.timing.measurement_timestamps = {source_time};
    batch.metadata.timing.reference_time = source_time;
    batch.metadata.health = health;
    batch.metadata.timing.produced_at = health.assessed_at + core::Duration{1LL};
    batch.metadata.header.created_at = health.assessed_at + core::Duration{2LL};
    batch.metadata.map_eligible = health.state == core::SensorHealthState::Active;
    batch.metadata.lineage = std::move(lineage);
    batch.source_state = source_state;
    batch.source_time = source_time;
    batch.registration = config.lidar_registration;
    batch.base_covariance_inflation = base_covariance_inflation;
    batch.registration_report = DirectLidarRegistrationReport{
        registration.termination, registration.initial_robust_cost, registration.final_robust_cost,
        registration.diagnostics, registration.work, registration.T_odom_source,
        registration.source_right_correction};

    struct OrderedPair {
      LidarDirectFactorPairSpec factor;
      core::DirectionalObservability observability;
      DirectLidarPairReport report;
    };
    std::vector<OrderedPair> ordered;
    ordered.reserve(registration.target_snapshots.size());
    const std::size_t total_correspondences = std::accumulate(
        registration.target_snapshots.begin(), registration.target_snapshots.end(), std::size_t{0U},
        [](std::size_t total, const auto& snapshot) { return total + snapshot->rows().size(); });
    const std::size_t finalized_correspondences =
        has_finalized_map ? registration.finalized_map_snapshot->rows().size() : 0U;
    if (total_correspondences + finalized_correspondences == 0U) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "accepted direct point ICP result contains no frozen correspondences"));
    }
    for (const auto& snapshot : registration.target_snapshots) {
      const auto input = std::find_if(
          selected_live_targets.begin(), selected_live_targets.end(),
          [&](const auto& candidate) { return candidate.state == snapshot->targetState(); });
      if (input == selected_live_targets.end() || input->time != snapshot->targetTime() ||
          input->cloud->source_sweep != snapshot->targetSweep()) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "direct point ICP snapshot target is absent from its immutable rolling-target batch"));
      }
      auto information =
          lidarFactorInformation(snapshot, config.lidar_registration, live_information_scale);
      if (!information) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "accepted direct point ICP snapshot has no canonical admission information: " +
                information.error().detail));
      }
      ordered.push_back(OrderedPair{
          LidarDirectFactorPairSpec{snapshot->targetState(), snapshot->targetTime(), snapshot,
                                    live_information_scale},
          lidarDirectionalObservability(information.value(), snapshot->targetState(),
                                        snapshot->targetTime(), source_state, source_time),
          DirectLidarPairReport{
              snapshot->targetState(), snapshot->targetTime(), snapshot->targetSweep(),
              snapshot->sourceSweep(), snapshot->sourcePointCount(), snapshot->rows().size(),
              snapshot->sourceRowsExcludedByOwnership(), snapshot->candidateVoxelLookups(),
              snapshot->candidatePointsExamined(), live_information_scale, information.value(),
              snapshot->checksum()}});
    }
    PreparedLidarFactorBatch output;
    output.batch = std::move(batch);
    output.batch.pairs.reserve(ordered.size());
    output.batch.metadata.directional_observability.reserve(ordered.size());
    output.pairs.reserve(ordered.size());
    for (OrderedPair& pair : ordered) {
      output.batch.pairs.push_back(std::move(pair.factor));
      output.batch.metadata.directional_observability.push_back(std::move(pair.observability));
      output.pairs.push_back(std::move(pair.report));
    }
    if (has_finalized_map) {
      const auto& snapshot = registration.finalized_map_snapshot;
      auto information = lidarFinalizedMapFactorInformation(
          snapshot, config.lidar_registration, finalized_map_information_scale);
      if (!information) {
        return Result::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "accepted finalized-map snapshot has no canonical admission information: " +
                information.error().detail));
      }
      output.batch.finalized_map =
          LidarFinalizedMapFactorSpec{snapshot,
                                      config.finalized_map_correlation_inflation_floor,
                                      finalized_map_information_scale};
      output.batch.metadata.directional_observability.push_back(
          lidarUnaryDirectionalObservability(information.value(), source_state, source_time));
      output.finalized_map = DirectLidarFinalizedMapReport{
          snapshot->sourceSweep(),
          snapshot->sourcePointCount(),
          snapshot->rows().size(),
          snapshot->sourceRowsExcludedByOwnership(),
          snapshot->candidateVoxelLookups(),
          snapshot->candidatePointsExamined(),
          snapshot->uniqueOwnerCount(),
          snapshot->mapOdomEpoch(),
          snapshot->mapSensor(),
          snapshot->mapVersion(),
          snapshot->mapChecksum(),
          owner_pose_covariance_inflation,
          config.finalized_map_correlation_inflation_floor,
          effective_map_covariance_inflation,
          finalized_map_information_scale,
          information.value(),
          snapshot->checksum()};
    }
    return Result::success(std::move(output));
  }

  [[nodiscard]] core::Result<bool, LocalEstimatorError> publishFactorOnlyCommit(
      const LocalGraphCommit& commit) {
    using Result = core::Result<bool, LocalEstimatorError>;
    auto recorded = recordGraphPublishedState(commit);
    if (!recorded) {
      return Result::failure(recorded.error());
    }
    auto synchronized = synchronizeRollingTarget(commit);
    if (!synchronized) {
      return Result::failure(synchronized.error());
    }
    auto finality = reconcileVisualFinality(commit);
    if (!finality) {
      return Result::failure(finality.error());
    }
    if (!commit.finalized_states.empty() &&
        !state_timeline.finalizeThrough(commit.finalized_states.back().exact_time)) {
      lifecycle = LocalEstimatorLifecycle::Faulted;
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::StateTimelineFailed, LocalEstimatorStage::StateTimeline,
          "factor-only graph finality could not advance the shared state timeline"));
    }
    ++statistics.graph_commits;
    pending_graph_transactions.push_back(LocalGraphTransactionSolveReport{
        commit.revision, commit.parent, commit.state, commit.state_time, false, commit.solve});
    return Result::success(true);
  }

  [[nodiscard]] std::optional<core::Pose3d> committedPose(const LocalGraphCommit& commit,
                                                          core::StateId state) const {
    const auto found =
        std::find_if(commit.navigation_poses.begin(), commit.navigation_poses.end(),
                     [&](const LocalGraphPoseSnapshot& pose) { return pose.state == state; });
    if (found == commit.navigation_poses.end()) {
      return std::nullopt;
    }
    return found->T_odom_imu;
  }

  [[nodiscard]] core::Result<std::shared_ptr<const AcceptedLidarMapInput>, LocalEstimatorError>
  makeAcceptedMapInput(const core::LidarSweep& sweep, const core::FactorBatchMetadata& metadata,
                       const LocalGraphCommit& commit, core::StateId source_state,
                       const std::shared_ptr<const LidarRegistrationCloud>& cloud) {
    using Result = core::Result<std::shared_ptr<const AcceptedLidarMapInput>, LocalEstimatorError>;
    const auto pose = committedPose(commit, source_state);
    if (!pose || !cloud || cloud->source_sweep != sweep.id) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "accepted LiDAR map payload lacks its exact committed pose, raw sweep, or "
          "registration artifact"));
    }
    auto input = AcceptedLidarMapInput::create(AcceptedLidarMapInputData{
        sweep,
        config.odom_epoch,
        source_state,
        commit.revision,
        metadata.batch_id,
        metadata.health.recovery_epoch,
        *pose,
        calibration.epoch(),
        cloud->checksum,
        metadata.lineage,
    });
    if (!input) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "accepted LiDAR map input could not be created: " + input.error().detail));
    }
    return Result::success(std::move(input).value());
  }

  [[nodiscard]] core::Result<AdmittedLidarCloud, LocalEstimatorError> addAcceptedFactorBatchCloud(
      const core::FactorBatchMetadata& metadata, const LocalGraphCommit& commit,
      core::StateId source_state, const core::LidarSweep& sweep,
      std::shared_ptr<const LidarRegistrationCloud> cloud) {
    using Result = core::Result<AdmittedLidarCloud, LocalEstimatorError>;
    MapAdmissionContext context;
    context.accepted_batch_revision = commit.revision;
    context.kind = MapAdmissionBatchKind::Regular;
    context.health_before = metadata.health;
    context.health_after = metadata.health;
    context.localization_revision = commit.revision;
    const MapAdmissionDecision admission = map_admission.admit(metadata, context);
    if (!admission.admitted()) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "localization-accepted LiDAR FactorBatch was rejected by the map admission gate with "
          "reason " +
              std::to_string(static_cast<int>(admission.reason))));
    }
    const auto pose = committedPose(commit, source_state);
    if (!pose) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "accepted LiDAR FactorBatch source pose is absent from its graph commit"));
    }
    auto map_input = makeAcceptedMapInput(sweep, metadata, commit, source_state, cloud);
    if (!map_input) {
      return Result::failure(map_input.error());
    }
    auto added = addCommittedCloud(source_state, metadata.batch_id, metadata.health.recovery_epoch,
                                   *pose, cloud);
    if (!added) {
      return Result::failure(added.error());
    }
    auto staged = stageFinalizedTargetSweep(metadata, commit, MapAdmissionBatchKind::Regular,
                                            source_state, std::move(cloud));
    if (!staged) {
      return Result::failure(staged.error());
    }
    return Result::success(
        AdmittedLidarCloud{std::move(added).value(), std::move(map_input).value()});
  }

  [[nodiscard]] core::Result<AdmittedLidarCloud, LocalEstimatorError> addInitializationSeed(
      const core::LidarSweep& sweep, core::FactorBatchId batch_id,
      const core::SensorHealthSnapshot& health, const LocalGraphCommit& commit, core::StateId state,
      std::shared_ptr<const LidarRegistrationCloud> cloud) {
    using Result = core::Result<AdmittedLidarCloud, LocalEstimatorError>;
    if (!sweep.acquisition.valid() || !sweep.id.valid() || !batch_id.valid() || !cloud ||
        cloud->source_sweep != sweep.id || health.state != core::SensorHealthState::Active) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "initialization map seed identity, acquisition, cloud, or health is invalid"));
    }
    // FactorBatch metadata requires a checksum even while raw ingress does
    // not yet provide opaque child checksums. Keep the geometry lineage
    // checksum-absent so this private metadata checksum is never presented as
    // a canonical checksum of the raw source root.
    core::ObservationLineage lineage = cloud->lineage;
    auto lineage_checksum = lidarFactorLineageChecksum(lineage);
    if (!lineage_checksum) {
      return Result::failure(lineage_checksum.error());
    }
    lineage.checksum = lineage_checksum.value();

    core::DirectionalObservability zero_information;
    zero_information.absolute_eigenvalue_threshold = 1.0;
    zero_information.relative_eigenvalue_threshold = 0.0;
    zero_information.supported_variables = {core::DirectionalVariable::PoseTranslation,
                                            core::DirectionalVariable::PoseRotation};
    zero_information.endpoints = {
        {core::DirectionalEndpointRole::Unary, state, cloud->reference_time}};

    core::FactorBatchMetadata metadata;
    metadata.header = sweep.header;
    metadata.batch_id = batch_id;
    metadata.odom_epoch = config.odom_epoch;
    metadata.sensor = core::SensorInstanceId::lidar(calibration.lidar().id());
    metadata.timing.support = core::TimeRange{
        sweep.acquisition.start,
        std::max(sweep.acquisition.end, cloud->reference_time + core::Duration{1LL})};
    metadata.timing.measurement_timestamps = {cloud->reference_time};
    metadata.timing.reference_time = cloud->reference_time;
    metadata.timing.produced_at = health.assessed_at + core::Duration{1LL};
    metadata.header.created_at = health.assessed_at + core::Duration{2LL};
    metadata.health = health;
    metadata.map_eligible = true;
    metadata.directional_observability.push_back(std::move(zero_information));
    metadata.lineage = std::move(lineage);

    MapAdmissionContext context;
    context.accepted_batch_revision = commit.revision;
    context.kind = MapAdmissionBatchKind::InitializationSeed;
    context.health_before = health;
    context.health_after = health;
    context.localization_revision = commit.revision;
    const MapAdmissionDecision admission = map_admission.admit(metadata, context);
    if (!admission.admitted()) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
          "accepted initialization cloud was rejected by the map admission gate with reason " +
              std::to_string(static_cast<int>(admission.reason))));
    }
    auto map_input = makeAcceptedMapInput(sweep, metadata, commit, state, cloud);
    if (!map_input) {
      return Result::failure(map_input.error());
    }
    const core::FusionTime admitted_time = cloud->reference_time;
    auto added = addCommittedCloud(state, batch_id, health.recovery_epoch,
                                   commit.estimate.T_odom_imu, cloud);
    if (!added) {
      return Result::failure(added.error());
    }
    auto staged = stageFinalizedTargetSweep(metadata, commit,
                                            MapAdmissionBatchKind::InitializationSeed, state,
                                            std::move(cloud));
    if (!staged) {
      return Result::failure(staged.error());
    }
    lidar_map_initialized = true;
    last_lidar_keyframe_time = admitted_time;
    statistics.last_lidar_keyframe_time = admitted_time;
    return Result::success(
        AdmittedLidarCloud{std::move(added).value(), std::move(map_input).value()});
  }

  [[nodiscard]] core::Result<RollingLidarTargetAddStats, LocalEstimatorError> addCommittedCloud(
      core::StateId state, core::FactorBatchId admitting_batch_id,
      core::SensorRecoveryEpoch recovery_epoch, const core::Pose3d& T_odom_imu,
      std::shared_ptr<const LidarRegistrationCloud> cloud) {
    using Result = core::Result<RollingLidarTargetAddStats, LocalEstimatorError>;
    core::PipelineWorkIdentity work;
    work.state = state;
    const core::ThreadCpuWallTimer target_update_timer;
    auto added =
        rolling_target.add(RegisteredLidarSweep{config.odom_epoch, state, admitting_batch_id,
                                                recovery_epoch, T_odom_imu, std::move(cloud)});
    detail::observeLocalPipelineTiming(
        pipeline_timing, LocalPipelineTimingStage::TargetBuildUpdate, target_update_timer,
        added ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed, work);
    if (!added) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::RollingTargetFailed, LocalEstimatorStage::RollingTarget,
          "committed cloud could not enter the rolling target: " + added.error().detail));
    }
    return Result::success(std::move(added).value());
  }

  void noteCommit(const LocalGraphCommit& commit) {
    // Graph publication is recorded at the writer boundary before any
    // fallible post-commit bookkeeping. This exact repeat proves that the
    // caller is advancing the same state and makes noteCommit safe for both
    // requested states and request-free initialization states.
    auto recorded = recordGraphPublishedState(commit);
    if (!recorded) {
      return;
    }
    const bool root_commit = statistics.graph_commits == 0U;
    ++statistics.graph_commits;
    pending_graph_transactions.push_back(LocalGraphTransactionSolveReport{
        commit.revision, commit.parent, commit.state, commit.state_time, true, commit.solve});
    next_state = commit.state.value() + 1U;
    // Exact-shared frontend batches can arrive after their navigation state
    // commits. Retain the bounded active-lag IMU history so a late LiDAR scan
    // can reconstruct its acquisition backwards without reinserting an IMU
    // edge. ImuBuffer's configured maximum span remains the hard memory cap.
    core::FusionTime retain_from = commit.state_time - config.graph.target_fixed_lag;
    if (!pending_lidar.empty()) {
      // An independently committed visual/guard state may fall inside the
      // oldest queued scan. Its pre-anchor IMU samples remain required by
      // propagateAround() to deskew early returns. The LiDAR queue is bounded
      // and acquisition starts increase strictly, so its front is the bounded
      // retention frontier.
      retain_from = std::min(retain_from, pending_lidar.front().sweep.acquisition.start);
    }
    imu_buffer.discardBefore(retain_from);

    if (root_commit) {
      // Initialization defines the graph's root: no later factor can create a
      // state before it.  Retire bootstrap-era requests strictly before that
      // root so they do not consume the live timeline capacity forever, while
      // retaining the root resolution itself for a possible bit-exact late
      // frontend share.
      if (commit.state_time.nanoseconds > std::numeric_limits<std::int64_t>::min() &&
          !state_timeline.finalizeThrough(commit.state_time - core::Duration{1LL})) {
        lifecycle = LocalEstimatorLifecycle::Faulted;
      }
    } else if (!commit.finalized_states.empty() &&
               !state_timeline.finalizeThrough(commit.finalized_states.back().exact_time)) {
      // After the root, a successful append is not finality: exact-share
      // factor batches may still arrive anywhere inside the active fixed-lag
      // window. Only graph-published marginalization advances this frontier.
      lifecycle = LocalEstimatorLifecycle::Faulted;
    }
  }
};

core::Result<LocalEstimator, LocalEstimatorError> LocalEstimator::create(
    core::CalibrationBundle calibration, LocalEstimatorConfig config) {
  using Result = core::Result<LocalEstimator, LocalEstimatorError>;
  if (!validConfig(config, calibration)) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
        "local estimator bounds, identities, calibration, or initialization window are invalid"));
  }
  applyCalibration(config, calibration);
  auto pipeline_timing = std::make_shared<LocalPipelineTimingRecorder>(config.pipeline_timing);
  auto target = RollingLidarTargetBuilder::create(config.rolling_target);
  if (!target) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
        "rolling LiDAR target profile is invalid: " + target.error().detail));
  }
  auto finalized_target = FinalizedLidarTargetMap::create(config.finalized_lidar_target);
  if (!finalized_target) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
        "finalized LiDAR target profile is invalid: " + finalized_target.error().detail));
  }
  auto bootstrap = LidarBootstrapOdometry::create(
      calibration.epoch(), calibration.lidar().extrinsics().T_imu_lidar(),
      config.graph.imu.gravity_odom, config.graph.imu.initial_gyroscope_bias_mean_radps,
      config.lidar_bootstrap, pipeline_timing);
  if (!bootstrap) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
        "LiDAR bootstrap profile is invalid: " + bootstrap.error().detail));
  }
  std::vector<core::SensorInstanceId> health_sensors;
  health_sensors.reserve(1U + config.visual_cameras.size());
  health_sensors.push_back(core::SensorInstanceId::lidar(calibration.lidar().id()));
  for (const VisualCameraConfig& visual : config.visual_cameras) {
    health_sensors.push_back(core::SensorInstanceId::camera(visual.camera));
  }
  auto sensor_health = SensorHealthRegistry::create(SensorHealthRegistryConfig{
      std::move(health_sensors), core::FusionTime{std::numeric_limits<std::int64_t>::min()},
      config.sensor_health_policy});
  if (!sensor_health) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InvalidConfiguration, LocalEstimatorStage::Configuration,
        "sensor health policy is invalid: " + sensor_health.error().detail));
  }
  auto implementation = std::make_unique<Impl>(
      std::move(calibration), std::move(config), std::move(target).value(),
      std::move(finalized_target).value(), std::move(bootstrap).value(),
      std::move(sensor_health).value(), std::move(pipeline_timing));
  auto visual = implementation->createVisualLanes();
  if (!visual) {
    return Result::failure(visual.error());
  }
  return Result::success(LocalEstimator(std::move(implementation)));
}

LocalEstimator::LocalEstimator(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

LocalEstimator::~LocalEstimator() = default;
LocalEstimator::LocalEstimator(LocalEstimator&&) noexcept = default;
LocalEstimator& LocalEstimator::operator=(LocalEstimator&&) noexcept = default;

core::Result<LocalEstimatorImuIngestReport, LocalEstimatorError> LocalEstimator::ingestImu(
    core::ImuSample sample) {
  using Result = core::Result<LocalEstimatorImuIngestReport, LocalEstimatorError>;
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::ImuRejected, LocalEstimatorStage::ImuIngress,
                       "faulted local estimator does not admit new IMU samples"));
  }
  if (sample.header.direct_calibration != implementation_->calibration.epoch()) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::ImuRejected, LocalEstimatorStage::ImuIngress,
                       "IMU sample does not name the estimator calibration epoch"));
  }
  const core::MeasurementId measurement = sample.id;
  const core::FusionTime time = sample.stamp.fusion_time;
  const core::RecordHeader header = sample.header;
  auto appended = implementation_->imu_buffer.append(std::move(sample));
  if (!appended) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::ImuRejected, LocalEstimatorStage::ImuIngress,
                       "IMU buffer rejected the sample: " + appended.error().detail));
  }
  if (!implementation_->first_imu_time) {
    implementation_->first_imu_time = time;
  }
  implementation_->latest_imu_time = time;
  implementation_->latest_imu_header = header;
  ++implementation_->statistics.imu_samples_accepted;
  return Result::success(LocalEstimatorImuIngestReport{measurement, std::move(appended).value()});
}

core::Result<LocalEstimatorLidarEnqueueReport, LocalEstimatorError> LocalEstimator::enqueueLidar(
    core::LidarSweep sweep) {
  using Result = core::Result<LocalEstimatorLidarEnqueueReport, LocalEstimatorError>;
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::LidarRejected, LocalEstimatorStage::LidarIngress,
                       "faulted local estimator does not admit new LiDAR sweeps"));
  }
  if (!validSweep(sweep, implementation_->calibration)) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::LidarRejected, LocalEstimatorStage::LidarIngress,
        "LiDAR identity, layout, acquisition support, calibration, or payload is invalid"));
  }
  if (implementation_->last_lidar_measurement &&
      (sweep.id <= *implementation_->last_lidar_measurement ||
       sweep.acquisition.start <= *implementation_->last_lidar_start)) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::LidarRejected, LocalEstimatorStage::LidarIngress,
                       "LiDAR measurements and acquisition starts must increase strictly"));
  }
  if (implementation_->pending_lidar.size() >=
      implementation_->config.maximum_pending_lidar_sweeps) {
    return Result::failure(estimatorError(LocalEstimatorErrorCode::PendingLidarCapacity,
                                          LocalEstimatorStage::LidarIngress,
                                          "bounded pending LiDAR queue is full"));
  }
  const core::MeasurementId measurement = sweep.id;
  auto admission = implementation_->admitLidar(sweep);
  if (!admission) {
    return Result::failure(admission.error());
  }
  implementation_->last_lidar_measurement = sweep.id;
  implementation_->last_lidar_start = sweep.acquisition.start;
  const StateAdmissionDisposition disposition = admission.value().disposition;
  const std::optional<core::FusionTime> suppressing_time = admission.value().suppressing_state_time;
  implementation_->pending_lidar.push_back(
      Impl::PendingLidarSweep{std::move(sweep), std::move(admission).value()});
  ++implementation_->statistics.lidar_sweeps_enqueued;
  if (disposition == StateAdmissionDisposition::SuppressedTooClose) {
    ++implementation_->statistics.lidar_state_requests_suppressed_by_timeline;
  }
  return Result::success(LocalEstimatorLidarEnqueueReport{
      measurement, disposition, suppressing_time, implementation_->pending_lidar.size()});
}

core::Result<LocalEstimatorImuGuardEnqueueReport, LocalEstimatorError>
LocalEstimator::enqueueImuGuard(core::FusionTime exact_time) {
  using Result = core::Result<LocalEstimatorImuGuardEnqueueReport, LocalEstimatorError>;
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::ImuGuardRejected, LocalEstimatorStage::ImuGuardIngress,
        "faulted local estimator does not admit IMU guard requests"));
  }
  if (!implementation_->graph.initialized() || !implementation_->latest_imu_header) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    return Result::failure(estimatorError(LocalEstimatorErrorCode::ImuGuardRejected,
                                          LocalEstimatorStage::ImuGuardIngress,
                                          "IMU guards require an initialized local graph and an "
                                          "accepted IMU header"));
  }
  auto current = implementation_->graph.estimate();
  if (!current) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
        "initialized local graph has no IMU guard scheduling anchor"));
  }
  if (exact_time <= current.value().state_time) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::ImuGuardRejected, LocalEstimatorStage::ImuGuardIngress,
        "IMU guard exact time must follow the latest committed graph state"));
  }
  if (implementation_->pending_imu_guards.size() >=
      implementation_->config.maximum_pending_imu_guards) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    ++implementation_->statistics.imu_guard_capacity_rejections;
    return Result::failure(estimatorError(LocalEstimatorErrorCode::PendingImuGuardCapacity,
                                          LocalEstimatorStage::ImuGuardIngress,
                                          "bounded pending IMU guard queue is full"));
  }

  std::uint64_t candidate_next_knot_request = implementation_->next_knot_request;
  const auto request = implementation_->allocate<core::KnotRequestId>(candidate_next_knot_request);
  if (!request) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                          LocalEstimatorStage::IdentityAllocation,
                                          "IMU guard knot-request identity exhausted"));
  }
  StateRequest guard;
  guard.header = *implementation_->latest_imu_header;
  guard.id = *request;
  guard.purpose = StateRequestPurpose::ImuGuard;
  guard.exact_time = exact_time;
  auto admitted = implementation_->state_timeline.request(std::move(guard));
  if (!admitted) {
    ++implementation_->statistics.imu_guard_enqueue_rejections;
    if (admitted.error().code == StateTimelineErrorCode::StateCapacity ||
        admitted.error().code == StateTimelineErrorCode::RequestCapacity) {
      ++implementation_->statistics.imu_guard_capacity_rejections;
    }
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::StateTimelineFailed, LocalEstimatorStage::StateTimeline,
        "IMU guard state request was rejected: " + admitted.error().detail));
  }

  implementation_->next_knot_request = candidate_next_knot_request;
  const StateAdmission admission = std::move(admitted).value();
  if (admission.disposition != StateAdmissionDisposition::SuppressedTooClose) {
    implementation_->pending_imu_guards.push_back(Impl::PendingImuGuard{admission});
    std::sort(implementation_->pending_imu_guards.begin(),
              implementation_->pending_imu_guards.end(),
              [](const Impl::PendingImuGuard& lhs, const Impl::PendingImuGuard& rhs) {
                if (lhs.admission.request.exact_time != rhs.admission.request.exact_time) {
                  return lhs.admission.request.exact_time < rhs.admission.request.exact_time;
                }
                return lhs.admission.request.id < rhs.admission.request.id;
              });
  } else {
    ++implementation_->statistics.imu_guard_requests_suppressed_by_timeline;
  }
  ++implementation_->statistics.imu_guards_enqueued;
  implementation_->statistics.imu_guard_pending_high_watermark =
      std::max(implementation_->statistics.imu_guard_pending_high_watermark,
               implementation_->pending_imu_guards.size());
  const bool exactly_shared = admission.disposition == StateAdmissionDisposition::ExactShare;
  const std::vector<core::KnotRequestId> shared_requests =
      admission.resolution ? admission.resolution->requests : std::vector<core::KnotRequestId>{};
  return Result::success(LocalEstimatorImuGuardEnqueueReport{
      admission.request.id, admission.request.exact_time, admission.disposition, exactly_shared,
      shared_requests, implementation_->pending_imu_guards.size()});
}

core::Result<LocalEstimatorCameraIngestReport, LocalEstimatorError> LocalEstimator::ingestCamera(
    core::CameraFrame frame, bool request_keyframe) {
  using Result = core::Result<LocalEstimatorCameraIngestReport, LocalEstimatorError>;
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::CameraRejected, LocalEstimatorStage::CameraIngress,
                       "faulted local estimator does not admit new camera frames"));
  }
  const auto lane_index = implementation_->visualLaneIndex(frame.camera);
  if (!lane_index) {
    ++implementation_->statistics.camera_frames_rejected;
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::CameraNotEnabled, LocalEstimatorStage::CameraIngress,
        "camera frame does not belong to an explicitly enabled visual lane"));
  }
  std::uint64_t candidate_next_knot_request = implementation_->next_knot_request;
  const auto request = implementation_->allocate<core::KnotRequestId>(candidate_next_knot_request);
  if (!request) {
    ++implementation_->statistics.camera_frames_rejected;
    return Result::failure(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                          LocalEstimatorStage::IdentityAllocation,
                                          "visual knot-request identity exhausted"));
  }

  Impl::VisualLaneState& lane = implementation_->visual_lanes[*lane_index];
  const core::MeasurementId measurement = frame.id;
  const core::CameraId camera = frame.camera;
  const core::FusionTime exposure = frame.exposure_midpoint;
  std::optional<LocalGraphCommit> current;
  if (implementation_->graph.initialized()) {
    auto estimate = implementation_->graph.estimate();
    if (!estimate) {
      ++implementation_->statistics.camera_frames_rejected;
      return Result::failure(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                            LocalEstimatorStage::InternalInvariant,
                                            "initialized graph has no camera prediction anchor"));
    }
    current = std::move(estimate).value();
  }
  const bool late_for_graph = current && exposure <= current->state_time;
  auto rotation = implementation_->visualRotationPrior(lane, exposure);

  VisualFrameInput input;
  input.frame = std::move(frame);
  input.imu_rotation_seed = rotation;
  input.state_request = *request;
  input.request_keyframe = request_keyframe;
  if (current && !late_for_graph) {
    input.latest_local_state =
        VisualLocalStateSnapshot{current->odom_epoch, current->state, current->state_time,
                                 current->revision, current->estimate};
  }
  auto tracked = lane.lane.processFrame(std::move(input));
  if (!tracked) {
    ++implementation_->statistics.camera_frames_rejected;
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::VisualLaneFailed, LocalEstimatorStage::VisualFrontend,
        "visual lane rejected camera frame: " + tracked.error().detail));
  }
  implementation_->next_knot_request = candidate_next_knot_request;

  lane.previous_exposure = exposure;
  ++implementation_->statistics.camera_frames_accepted;
  if (late_for_graph) {
    ++implementation_->statistics.camera_frames_late_for_graph;
  }
  if (rotation) {
    ++implementation_->statistics.camera_rotation_seeds_provided;
  }
  if (tracked.value().disposition == VisualFrameDisposition::KeyframeStateRequested) {
    if (!tracked.value().state_admission ||
        tracked.value().state_admission->disposition ==
            StateAdmissionDisposition::SuppressedTooClose ||
        !tracked.value().state_admission->resolution) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
          "visual lane reported an admitted keyframe without an actionable state admission"));
    }
    implementation_->pending_visual_knots.push_back(
        Impl::PendingVisualKnot{*lane_index, measurement, *tracked.value().state_admission});
    std::sort(implementation_->pending_visual_knots.begin(),
              implementation_->pending_visual_knots.end(),
              [](const Impl::PendingVisualKnot& lhs, const Impl::PendingVisualKnot& rhs) {
                if (lhs.admission.request.exact_time != rhs.admission.request.exact_time) {
                  return lhs.admission.request.exact_time < rhs.admission.request.exact_time;
                }
                return lhs.admission.request.id < rhs.admission.request.id;
              });
    ++implementation_->statistics.visual_keyframe_requests;
  }

  LocalEstimatorCameraIngestReport report;
  report.measurement = measurement;
  report.camera = camera;
  report.frame = std::move(tracked).value();
  report.late_for_graph = late_for_graph;
  report.imu_rotation_seed_provided = rotation.has_value();
  report.pending_camera_knots = implementation_->pending_visual_knots.size();
  return Result::success(std::move(report));
}

core::Result<VisualResidualIngestReport, LocalEstimatorError> LocalEstimator::applyVisualResiduals(
    core::CameraId camera, core::LocalGraphRevision revision,
    const std::vector<VisualResidualFeedback>& feedback) {
  using Result = core::Result<VisualResidualIngestReport, LocalEstimatorError>;
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::VisualLaneFailed, LocalEstimatorStage::VisualResidualFeedback,
        "faulted local estimator does not admit visual residual feedback"));
  }
  const auto lane_index = implementation_->visualLaneIndex(camera);
  if (!lane_index) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::CameraNotEnabled, LocalEstimatorStage::VisualResidualFeedback,
        "visual residual feedback names a camera that is not enabled"));
  }
  Impl::VisualLaneState& lane = implementation_->visual_lanes[*lane_index];
  std::vector<VisualResidualFeedback> local_feedback;
  local_feedback.reserve(feedback.size());
  std::set<core::FactorId> seen;
  for (const VisualResidualFeedback& item : feedback) {
    if (!item.factor.valid() || !seen.insert(item.factor).second) {
      return Result::failure(
          estimatorError(LocalEstimatorErrorCode::VisualReferenceUnavailable,
                         LocalEstimatorStage::VisualResidualFeedback,
                         "visual residual feedback contains an invalid or duplicate graph factor"));
    }
    const auto local = lane.identities.graph_to_local_factors.find(item.factor);
    if (local == lane.identities.graph_to_local_factors.end() ||
        !lane.identities.accepted_factors.contains(local->second)) {
      return Result::failure(
          estimatorError(LocalEstimatorErrorCode::VisualReferenceUnavailable,
                         LocalEstimatorStage::VisualResidualFeedback,
                         "visual residual feedback refers to a stale, unknown, or different-lane "
                         "graph factor"));
    }
    local_feedback.push_back(VisualResidualFeedback{local->second, item.squared_mahalanobis});
  }
  auto applied = lane.lane.applyAcceptedResiduals(revision, local_feedback);
  if (!applied) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::VisualLaneFailed, LocalEstimatorStage::VisualResidualFeedback,
        "visual lane rejected residual feedback: " + applied.error().detail));
  }
  VisualResidualIngestReport report = std::move(applied).value();
  for (core::FactorId& retired : report.factor_builder.retired_observation_factors) {
    const auto global = lane.identities.factors.find(retired);
    if (global == lane.identities.factors.end()) {
      return Result::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::VisualResidualFeedback,
          "visual residual result lost its graph-visible factor identity"));
    }
    retired = global->second;
  }
  implementation_->statistics.visual_residual_feedback_items += feedback.size();
  return Result::success(std::move(report));
}

core::Result<LocalEstimatorProcessReport, LocalEstimatorError> LocalEstimator::processReady() {
  using Result = core::Result<LocalEstimatorProcessReport, LocalEstimatorError>;
  LocalEstimatorProcessReport report;
  implementation_->finalized_target_process = FinalizedLidarTargetProcessReport{};
  auto fail = [this](LocalEstimatorError error) {
    implementation_->lifecycle = LocalEstimatorLifecycle::Faulted;
    return Result::failure(std::move(error));
  };
  auto complete = [this, &report]() {
    implementation_->refreshFinalizedTargetProcessSnapshot();
    report.finalized_lidar_target = std::move(implementation_->finalized_target_process);
    report.graph_transactions.swap(implementation_->pending_graph_transactions);
    return Result::success(std::move(report));
  };
  if (implementation_->lifecycle == LocalEstimatorLifecycle::Faulted) {
    return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                               LocalEstimatorStage::InternalInvariant,
                               "faulted local estimator cannot process queued work"));
  }
  if (!implementation_->pending_graph_transactions.empty()) {
    return fail(estimatorError(
        LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
        "local estimator retained graph-transaction diagnostics after a completed process cycle"));
  }

  auto resolve_visual_knots = [&](const LocalGraphCommit& commit,
                                  const std::optional<VisualGraphAttachmentReport>& attachment,
                                  const std::optional<VisualGraphDegradationReport>& degradation)
      -> core::Result<bool, LocalEstimatorError> {
    using ResolveResult = core::Result<bool, LocalEstimatorError>;
    if (implementation_->pending_visual_knots.empty() ||
        implementation_->pending_visual_knots.front().admission.request.exact_time !=
            commit.state_time) {
      if (!implementation_->pending_visual_knots.empty() &&
          implementation_->pending_visual_knots.front().admission.request.exact_time <
              commit.state_time) {
        return ResolveResult::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::VisualKnotResolution,
            "graph commit advanced past an admitted visual keyframe"));
      }
      return ResolveResult::success(false);
    }

    CameraKnotCommitReport camera_report;
    camera_report.exact_time = commit.state_time;
    camera_report.commit = commit;
    camera_report.visual_attachment = attachment;
    camera_report.visual_degradation = degradation;
    std::vector<core::KnotRequestId> shared_requests;
    const auto scheduled = std::find_if(implementation_->state_timeline.resolutions().begin(),
                                        implementation_->state_timeline.resolutions().end(),
                                        [&](const StateResolution& resolution) {
                                          return resolution.exact_time == commit.state_time;
                                        });
    if (scheduled != implementation_->state_timeline.resolutions().end()) {
      shared_requests = scheduled->requests;
    }
    if (shared_requests.empty()) {
      return ResolveResult::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
          "committed visual state is absent from the shared state timeline"));
    }

    while (!implementation_->pending_visual_knots.empty() &&
           implementation_->pending_visual_knots.front().admission.request.exact_time ==
               commit.state_time) {
      const Impl::PendingVisualKnot pending = implementation_->pending_visual_knots.front();
      VisualStateResolution resolution;
      resolution.header = pending.admission.request.header;
      resolution.request = pending.admission.request.id;
      resolution.timeline = *scheduled;
      resolution.odom_epoch = commit.odom_epoch;
      resolution.state = commit.state;
      resolution.created_at_revision = commit.revision;
      auto resolved =
          implementation_->visual_lanes[pending.lane_index].lane.resolveCommittedKeyframe(
              std::move(resolution),
              VisualLocalStateSnapshot{commit.odom_epoch, commit.state, commit.state_time,
                                       commit.revision, commit.estimate});
      if (!resolved) {
        return ResolveResult::failure(estimatorError(
            LocalEstimatorErrorCode::VisualLaneFailed, LocalEstimatorStage::VisualKnotResolution,
            "visual lane could not resolve its committed exact-time state: " +
                resolved.error().detail));
      }
      camera_report.resolved_keyframes.push_back(std::move(resolved).value());
      implementation_->pending_visual_knots.pop_front();
      ++implementation_->statistics.visual_keyframes_resolved;
    }
    ++implementation_->statistics.visual_keyframe_knots_committed;
    report.camera_commits.push_back(std::move(camera_report));
    return ResolveResult::success(true);
  };

  auto resolve_imu_guards =
      [&](const LocalGraphCommit& commit) -> core::Result<bool, LocalEstimatorError> {
    using ResolveResult = core::Result<bool, LocalEstimatorError>;
    if (implementation_->pending_imu_guards.empty() ||
        implementation_->pending_imu_guards.front().admission.request.exact_time !=
            commit.state_time) {
      if (!implementation_->pending_imu_guards.empty() &&
          implementation_->pending_imu_guards.front().admission.request.exact_time <
              commit.state_time) {
        return ResolveResult::failure(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
            "graph commit advanced past an admitted IMU guard"));
      }
      return ResolveResult::success(false);
    }

    std::vector<core::KnotRequestId> shared_requests;
    const auto scheduled = std::find_if(implementation_->state_timeline.resolutions().begin(),
                                        implementation_->state_timeline.resolutions().end(),
                                        [&](const StateResolution& resolution) {
                                          return resolution.exact_time == commit.state_time;
                                        });
    if (scheduled != implementation_->state_timeline.resolutions().end()) {
      shared_requests = scheduled->requests;
    }
    if (shared_requests.empty()) {
      return ResolveResult::failure(estimatorError(
          LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
          "committed IMU guard state is absent from the shared state timeline"));
    }

    ImuGuardCommitReport guard_report;
    guard_report.exact_time = commit.state_time;
    guard_report.commit = commit;
    while (!implementation_->pending_imu_guards.empty() &&
           implementation_->pending_imu_guards.front().admission.request.exact_time ==
               commit.state_time) {
      const Impl::PendingImuGuard pending = implementation_->pending_imu_guards.front();
      const auto resolution_id =
          implementation_->allocate<core::KnotResolutionId>(implementation_->next_knot_resolution);
      if (!resolution_id) {
        return ResolveResult::failure(estimatorError(
            LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
            "IMU guard knot-resolution identity exhausted"));
      }
      guard_report.resolutions.push_back(ImuGuardResolution{
          pending.admission.request.header, *resolution_id, pending.admission.request.id,
          commit.odom_epoch, commit.state, commit.state_time, commit.revision, shared_requests});
      implementation_->pending_imu_guards.pop_front();
      ++implementation_->statistics.imu_guard_requests_resolved;
    }
    ++implementation_->statistics.imu_guard_knots_committed;
    report.imu_guard_commits.push_back(std::move(guard_report));
    return ResolveResult::success(true);
  };

  if (!implementation_->graph.initialized() && implementation_->first_imu_time &&
      implementation_->config.initialization.zero_motion_prior &&
      implementation_->latest_imu_time &&
      *implementation_->latest_imu_time - *implementation_->first_imu_time >=
          implementation_->config.stationary_initializer.minimum_support) {
    const bool retry_due =
        !implementation_->last_initialization_attempt ||
        *implementation_->latest_imu_time - *implementation_->last_initialization_attempt >=
            implementation_->config.stationary_retry_period;
    if (retry_due) {
      ++implementation_->statistics.initialization_attempts;
      ++implementation_->statistics.stationary_initialization_attempts;
      implementation_->last_initialization_attempt = *implementation_->latest_imu_time;
      const core::TimeRange support{
          *implementation_->latest_imu_time -
              implementation_->config.stationary_initializer.minimum_support,
          *implementation_->latest_imu_time};
      auto interval =
          implementation_->imu_buffer.interval(support, implementation_->nominal_imu_period);
      if (interval) {
        const core::ThreadCpuWallTimer stationary_probe_timer;
        auto initialization =
            initializeStationary(interval.value(), implementation_->config.stationary_initializer);
        detail::observeLocalPipelineTiming(implementation_->pipeline_timing,
                                           LocalPipelineTimingStage::StationaryProbe,
                                           stationary_probe_timer,
                                           initialization ? core::PipelineDisposition::Completed
                                                          : core::PipelineDisposition::Rejected);
        if (!initialization) {
          ++implementation_->statistics.initialization_rejections;
          ++implementation_->statistics.stationary_initialization_rejections;
          report.initialization_rejection =
              LocalInitializationRejection{LocalInitializationRejectionStage::StationaryTest,
                                           static_cast<int>(initialization.error().code),
                                           std::nullopt, initialization.error().detail};
        } else {
          detail::IdentityTransaction<bool> initialization_identities(
              implementation_->identityCounters(), false);
          auto lineage =
              implementation_->graphLineage(interval.value(), initialization_identities.counters());
          if (!lineage) {
            return fail(lineage.error());
          }
          if (implementation_->next_state == core::StateId::kInvalidValue) {
            return fail(estimatorError(
                LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
                "navigation state identity exhausted before initialization"));
          }
          const core::StateId state{implementation_->next_state};
          auto committed = implementation_->graph.initialize(LocalGraphInitialization{
              implementation_->config.odom_epoch, state, support.end, initialization.value().state,
              initialization.value().covariance, std::move(lineage).value()});
          if (!committed) {
            static_cast<void>(initialization_identities.abort());
            return fail(estimatorError(
                LocalEstimatorErrorCode::InitializationFailed,
                LocalEstimatorStage::StationaryInitialization,
                "local graph rejected stationary initialization: " + committed.error().detail));
          }
          auto recorded = implementation_->recordGraphPublishedState(committed.value());
          if (!recorded) {
            static_cast<void>(initialization_identities.abort());
            return fail(recorded.error());
          }
          auto committed_identities = initialization_identities.commit();
          if (!committed_identities) {
            return fail(estimatorError(
                LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::IdentityAllocation,
                "committed stationary-initialization identities were already consumed"));
          }
          implementation_->publishIdentityCounters(committed_identities.value().counters);
          auto visual_finality = implementation_->reconcileVisualFinality(committed.value());
          if (!visual_finality) {
            return fail(visual_finality.error());
          }
          implementation_->lifecycle = LocalEstimatorLifecycle::Tracking;
          implementation_->noteCommit(committed.value());
          auto reset = implementation_->resetBootstrap();
          if (!reset) {
            return fail(reset.error());
          }
          report.initialization = std::move(committed).value();
          report.initialization_method = LocalInitializationMethod::StationaryImu;
        }
      } else {
        ++implementation_->statistics.initialization_rejections;
        ++implementation_->statistics.stationary_initialization_rejections;
        report.initialization_rejection = LocalInitializationRejection{
            LocalInitializationRejectionStage::StationaryTest, -1, std::nullopt,
            "stationary initialization lacks an exact IMU interval: " + interval.error().detail};
      }
    }
  }

  if (!implementation_->graph.initialized() &&
      implementation_->config.initialization.mode != InitializationMode::StaticOnly) {
    while (!implementation_->pending_lidar.empty() && implementation_->latest_imu_time &&
           implementation_->pending_lidar.front().sweep.acquisition.end <=
               *implementation_->latest_imu_time &&
           !implementation_->graph.initialized()) {
      const core::LidarSweep& sweep = implementation_->pending_lidar.front().sweep;
      LidarBootstrapProcessReport bootstrap_report;
      bootstrap_report.measurement = sweep.id;

      detail::IdentityTransaction<bool> bootstrap_identities(implementation_->identityCounters(),
                                                             false);
      auto& candidate_counters = bootstrap_identities.counters();
      const auto cloud_record =
          implementation_->allocate<core::DerivedRecordId>(candidate_counters.next_derived);
      const auto cloud_lineage =
          implementation_->allocate<core::ObservationLineageId>(candidate_counters.next_lineage);
      if (!cloud_record || !cloud_lineage) {
        return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "LiDAR bootstrap registration-view identity exhausted"));
      }

      std::optional<ImuInterval> acquisition_imu;
      std::optional<LidarBootstrapCommit> bootstrap_commit;
      std::optional<LidarBootstrapError> bootstrap_error;
      {
        LidarBootstrapInput input;
        input.sweep = sweep;
        input.cloud_record = *cloud_record;
        input.cloud_lineage = *cloud_lineage;
        auto acquisition = implementation_->imu_buffer.interval(
            sweep.acquisition, implementation_->nominal_imu_period);
        if (acquisition) {
          acquisition_imu = acquisition.value();
          input.acquisition_imu = acquisition.value();
        } else {
          bootstrap_error = LidarBootstrapError{
              LidarBootstrapErrorCode::InvalidAcquisitionImu,
              "exact IMU support over the complete LiDAR acquisition is unavailable: " +
                  acquisition.error().detail,
              std::nullopt, std::nullopt};
        }
        if (!implementation_->lidar_bootstrap.empty()) {
          if (!implementation_->last_bootstrap_time ||
              !implementation_->last_bootstrap_measurement ||
              !implementation_->last_bootstrap_acquisition_imu) {
            return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                       LocalEstimatorStage::InternalInvariant,
                                       "non-empty LiDAR bootstrap has no committed reference"));
          }
          auto interval = implementation_->imu_buffer.interval(
              core::TimeRange{*implementation_->last_bootstrap_time, sweep.acquisition.end},
              implementation_->nominal_imu_period);
          if (!bootstrap_error && interval && acquisition_imu) {
            auto lidar_lineage = implementation_->bootstrapLidarLineage(
                *implementation_->last_bootstrap_measurement, sweep.id,
                *implementation_->last_bootstrap_acquisition_imu, *acquisition_imu,
                interval.value(), candidate_counters);
            if (!lidar_lineage) {
              return fail(lidar_lineage.error());
            }
            auto imu_lineage = implementation_->graphLineage(interval.value(), candidate_counters);
            if (!imu_lineage) {
              return fail(imu_lineage.error());
            }
            input.between_reference_imu = interval.value();
            input.lidar_factor_lineage = std::move(lidar_lineage).value();
            input.imu_factor_lineage = std::move(imu_lineage).value();
          } else if (!bootstrap_error) {
            bootstrap_error =
                LidarBootstrapError{LidarBootstrapErrorCode::InvalidSegmentSupport,
                                    "exact IMU support between bootstrap scans is unavailable: " +
                                        interval.error().detail,
                                    std::nullopt, std::nullopt};
          }
        }

        if (!bootstrap_error) {
          auto bootstrapped = implementation_->lidar_bootstrap.add(std::move(input));
          if (bootstrapped) {
            bootstrap_commit = std::move(bootstrapped).value();
          } else {
            bootstrap_error = bootstrapped.error();
          }
        }
      }

      if (bootstrap_error) {
        bootstrap_report.rejection = *bootstrap_error;
        ++implementation_->statistics.lidar_bootstrap_rejections;
        report.initialization_rejection = LocalInitializationRejection{
            LocalInitializationRejectionStage::LidarBootstrap,
            static_cast<int>(bootstrap_error->code), std::nullopt, bootstrap_error->detail};

        auto aborted_identities = bootstrap_identities.abort();
        if (!aborted_identities) {
          return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                     LocalEstimatorStage::IdentityAllocation,
                                     "rejected bootstrap identities were already consumed"));
        }

        const LidarBootstrapErrorCode rejection_code = bootstrap_error->code;
        const bool candidate_only_rejection =
            rejection_code == LidarBootstrapErrorCode::InvalidAcquisitionImu ||
            rejection_code == LidarBootstrapErrorCode::RegistrationCloudBuildFailed ||
            rejection_code == LidarBootstrapErrorCode::RegistrationFailed ||
            rejection_code == LidarBootstrapErrorCode::IncrementGateFailed ||
            rejection_code == LidarBootstrapErrorCode::DeskewFailed;
        const bool restart_required =
            rejection_code == LidarBootstrapErrorCode::Capacity ||
            rejection_code == LidarBootstrapErrorCode::InvalidSegmentSupport;
        if (!candidate_only_rejection && !restart_required) {
          return fail(estimatorError(
              LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::LidarBootstrap,
              "LiDAR bootstrap rejected coordinator-validated input: " + bootstrap_error->detail));
        }

        // Candidate-only failures are transactionally inert inside the
        // bootstrap odometry. Keeping the last committed scan preserves the
        // contiguous initialization window; the next sweep may bridge the
        // rejected candidate with exact IMU support. Only a support break or
        // bounded-capacity rollover invalidates that window.
        if (restart_required) {
          auto reset = implementation_->resetBootstrap();
          if (!reset) {
            return fail(reset.error());
          }
          detail::IdentityTransaction<bool> anchor_identities(implementation_->identityCounters(),
                                                              false);
          auto& anchor_counters = anchor_identities.counters();
          const auto anchor_record =
              implementation_->allocate<core::DerivedRecordId>(anchor_counters.next_derived);
          const auto anchor_lineage =
              implementation_->allocate<core::ObservationLineageId>(anchor_counters.next_lineage);
          if (!anchor_record || !anchor_lineage) {
            return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                       LocalEstimatorStage::IdentityAllocation,
                                       "LiDAR bootstrap re-anchor identity exhausted"));
          }
          if (!acquisition_imu) {
            return fail(estimatorError(
                LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::LidarBootstrap,
                "bootstrap re-anchor has no validated acquisition IMU support"));
          }
          std::optional<LidarBootstrapCommit> anchored_commit;
          std::optional<LidarBootstrapError> anchored_error;
          {
            LidarBootstrapInput anchor;
            anchor.sweep = sweep;
            anchor.cloud_record = *anchor_record;
            anchor.cloud_lineage = *anchor_lineage;
            anchor.acquisition_imu = *acquisition_imu;
            auto anchored = implementation_->lidar_bootstrap.add(std::move(anchor));
            if (anchored) {
              anchored_commit = std::move(anchored).value();
            } else {
              anchored_error = anchored.error();
            }
          }
          if (anchored_commit) {
            auto committed_identities = anchor_identities.commit();
            if (!committed_identities) {
              return fail(
                  estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "committed bootstrap anchor identities were already consumed"));
            }
            implementation_->publishIdentityCounters(committed_identities.value().counters);
            bootstrap_report.commit = *anchored_commit;
            implementation_->last_bootstrap_time = anchored_commit->reference_time;
            implementation_->last_bootstrap_measurement = sweep.id;
            implementation_->last_bootstrap_sweep = sweep;
            implementation_->last_bootstrap_acquisition_imu = *acquisition_imu;
            ++implementation_->statistics.lidar_bootstrap_anchors;
          } else {
            static_cast<void>(anchor_identities.abort());
            if (!anchored_error) {
              return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                         LocalEstimatorStage::LidarBootstrap,
                                         "bootstrap re-anchor produced neither commit nor error"));
            }
            bootstrap_report.rejection = *anchored_error;
            report.initialization_rejection = LocalInitializationRejection{
                LocalInitializationRejectionStage::LidarBootstrap,
                static_cast<int>(anchored_error->code), std::nullopt,
                "LiDAR bootstrap re-anchor failed: " + anchored_error->detail};
            auto empty_reset = implementation_->resetBootstrap();
            if (!empty_reset) {
              return fail(empty_reset.error());
            }
          }
        }
      } else {
        if (!bootstrap_commit) {
          return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                     LocalEstimatorStage::LidarBootstrap,
                                     "bootstrap candidate produced neither commit nor error"));
        }
        auto committed_identities = bootstrap_identities.commit();
        if (!committed_identities) {
          return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                     LocalEstimatorStage::IdentityAllocation,
                                     "committed bootstrap identities were already consumed"));
        }
        implementation_->publishIdentityCounters(committed_identities.value().counters);
        bootstrap_report.commit = *bootstrap_commit;
        implementation_->last_bootstrap_time = bootstrap_commit->reference_time;
        implementation_->last_bootstrap_measurement = sweep.id;
        implementation_->last_bootstrap_sweep = sweep;
        implementation_->last_bootstrap_acquisition_imu = *acquisition_imu;
        if (bootstrap_commit->disposition == LidarBootstrapDisposition::AnchorCreated) {
          ++implementation_->statistics.lidar_bootstrap_anchors;
        } else {
          ++implementation_->statistics.lidar_bootstrap_increments;
          if (!bootstrap_commit->segment) {
            return fail(estimatorError(
                LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
                "committed LiDAR bootstrap increment has no motion segment"));
          }
          implementation_->motion_segments.push_back(*bootstrap_commit->segment);
          while (implementation_->motion_segments.size() >
                 implementation_->config.motion_initializer.maximum_segments) {
            implementation_->motion_segments.erase(implementation_->motion_segments.begin());
          }
          while (implementation_->motion_segments.size() > 1U &&
                 implementation_->motion_segments.back().lidar.end_time -
                         implementation_->motion_segments.front().lidar.start_time >
                     implementation_->config.motion_initializer.maximum_support) {
            implementation_->motion_segments.erase(implementation_->motion_segments.begin());
          }
        }
      }

      implementation_->pending_lidar.pop_front();
      report.bootstrap.push_back(std::move(bootstrap_report));

      if (implementation_->motion_segments.size() <
          implementation_->config.motion_initializer.minimum_segments) {
        continue;
      }
      ++implementation_->statistics.initialization_attempts;
      ++implementation_->statistics.motion_initialization_attempts;
      MotionInitializationRequest proposal_request;
      proposal_request.segments = implementation_->motion_segments;
      proposal_request.imu_noise = implementation_->motionImuNoise();
      proposal_request.pass = MotionInitializationRequest::Pass::RotationDeskewProposal;
      auto proposal = implementation_->motion_initializer.initialize(proposal_request);
      if (!proposal) {
        ++implementation_->statistics.initialization_rejections;
        ++implementation_->statistics.motion_initialization_rejections;
        report.initialization_rejection = LocalInitializationRejection{
            LocalInitializationRejectionStage::MotionBatch, static_cast<int>(proposal.error().code),
            proposal.error().segment,
            "rotation-deskew proposal solve failed: " + proposal.error().detail};
        continue;
      }

      auto refinement = implementation_->lidar_bootstrap.refine(proposal.value().reference_states);
      if (!refinement) {
        ++implementation_->statistics.initialization_rejections;
        ++implementation_->statistics.motion_initialization_rejections;
        report.initialization_rejection = LocalInitializationRejection{
            LocalInitializationRejectionStage::MotionBatch,
            static_cast<int>(refinement.error().code), std::nullopt,
            "full discrete bootstrap refinement failed: " + refinement.error().detail};
        continue;
      }

      MotionInitializationRequest final_request;
      final_request.segments = refinement.value().segments;
      final_request.imu_noise = proposal_request.imu_noise;
      final_request.pass = MotionInitializationRequest::Pass::FullDeskewCommitCandidate;
      auto initialization = implementation_->motion_initializer.initialize(final_request);
      if (!initialization) {
        ++implementation_->statistics.initialization_rejections;
        ++implementation_->statistics.motion_initialization_rejections;
        report.initialization_rejection = LocalInitializationRejection{
            LocalInitializationRejectionStage::MotionBatch,
            static_cast<int>(initialization.error().code), initialization.error().segment,
            "full-deskew commit-candidate solve failed: " + initialization.error().detail};
        continue;
      }
      initialization.value().diagnostics.deskew_solve_passes = 2U;
      initialization.value().diagnostics.refined_sweeps = refinement.value().diagnostics.sweeps;
      initialization.value().diagnostics.refined_registrations =
          refinement.value().diagnostics.registrations;
      initialization.value().diagnostics.refined_deskew_pose_interpolations =
          refinement.value().diagnostics.deskew_pose_interpolations;
      initialization.value().diagnostics.refined_total_registration_cost =
          refinement.value().diagnostics.total_registration_cost;
      initialization.value().diagnostics.refined_maximum_registration_cost =
          refinement.value().diagnostics.maximum_registration_cost;

      detail::IdentityTransaction<bool> initialization_identities(
          implementation_->identityCounters(), false);
      auto lineage = implementation_->motionInitializationLineage(
          refinement.value().segments, initialization.value(),
          initialization_identities.counters());
      if (!lineage) {
        return fail(lineage.error());
      }
      if (implementation_->next_state == core::StateId::kInvalidValue) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::IdentityExhausted, LocalEstimatorStage::IdentityAllocation,
            "navigation state identity exhausted before moving initialization"));
      }
      const core::StateId state{implementation_->next_state};
      auto committed = implementation_->graph.initialize(
          LocalGraphInitialization{implementation_->config.odom_epoch, state,
                                   initialization.value().exact_time, initialization.value().state,
                                   initialization.value().covariance, std::move(lineage).value()});
      if (!committed) {
        static_cast<void>(initialization_identities.abort());
        return fail(estimatorError(
            LocalEstimatorErrorCode::InitializationFailed,
            LocalEstimatorStage::MotionInitialization,
            "local graph rejected moving LiDAR--IMU initialization: " + committed.error().detail));
      }
      auto recorded = implementation_->recordGraphPublishedState(committed.value());
      if (!recorded) {
        static_cast<void>(initialization_identities.abort());
        return fail(recorded.error());
      }
      auto committed_identities = initialization_identities.commit();
      if (!committed_identities) {
        return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "committed moving-initialization identities were already "
                                   "consumed"));
      }
      implementation_->publishIdentityCounters(committed_identities.value().counters);
      auto visual_finality = implementation_->reconcileVisualFinality(committed.value());
      if (!visual_finality) {
        return fail(visual_finality.error());
      }
      implementation_->lifecycle = LocalEstimatorLifecycle::Tracking;
      implementation_->noteCommit(committed.value());
      if (!implementation_->last_bootstrap_measurement || !implementation_->last_bootstrap_sweep ||
          refinement.value().final_cloud->source_sweep !=
              *implementation_->last_bootstrap_measurement ||
          implementation_->last_bootstrap_sweep->id !=
              *implementation_->last_bootstrap_measurement) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::InternalInvariant,
            "moving initialization final cloud has no exact retained source record"));
      }
      const auto seed_batch =
          implementation_->allocate<core::FactorBatchId>(implementation_->next_factor_batch);
      if (!seed_batch) {
        return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "moving-initialization map seed batch identity exhausted"));
      }
      auto seed_health = implementation_->processLidarHealth(
          *seed_batch, refinement.value().final_cloud->reference_time,
          implementation_->last_bootstrap_sweep->acquisition.end, SensorBatchHealthResult::Good);
      if (!seed_health ||
          seed_health.value().update.after.state != core::SensorHealthState::Active ||
          seed_health.value().removal.commit) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::SensorHealthFailed, LocalEstimatorStage::SensorHealth,
            seed_health ? "moving initialization produced a quarantined LiDAR map seed"
                        : seed_health.error().detail));
      }
      auto seeded_target = implementation_->addInitializationSeed(
          *implementation_->last_bootstrap_sweep, *seed_batch, seed_health.value().update.after,
          committed.value(), state, refinement.value().final_cloud);
      if (!seeded_target) {
        return fail(seeded_target.error());
      }
      report.initialization_map_input = seeded_target.value().map_input;
      report.motion_initialization_diagnostics = initialization.value().diagnostics;
      implementation_->motion_segments.clear();
      ++implementation_->statistics.motion_initialization_commits;
      report.initialization = std::move(committed).value();
      report.initialization_method = LocalInitializationMethod::MotionLidarImu;
    }
    report.lifecycle = implementation_->lifecycle;
    report.pending_sweeps = implementation_->pending_lidar.size();
    report.pending_imu_guards = implementation_->pending_imu_guards.size();
    report.pending_camera_knots = implementation_->pending_visual_knots.size();
    report.waiting_for_future_imu = !implementation_->pending_lidar.empty() &&
                                    (!implementation_->latest_imu_time ||
                                     implementation_->pending_lidar.front().sweep.acquisition.end >
                                         *implementation_->latest_imu_time);
    if (!implementation_->graph.initialized()) {
      return complete();
    }
  }

  if (!implementation_->graph.initialized()) {
    report.lifecycle = implementation_->lifecycle;
    report.pending_sweeps = implementation_->pending_lidar.size();
    report.pending_imu_guards = implementation_->pending_imu_guards.size();
    report.pending_camera_knots = implementation_->pending_visual_knots.size();
    report.waiting_for_future_imu = false;
    return complete();
  }

  while (!implementation_->pending_lidar.empty() || !implementation_->pending_imu_guards.empty() ||
         !implementation_->pending_visual_knots.empty()) {
    auto current = implementation_->graph.estimate();
    if (!current) {
      return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                 LocalEstimatorStage::InternalInvariant,
                                 "initialized local graph has no current estimate"));
    }

    std::optional<core::FusionTime> visual_time;
    if (!implementation_->pending_visual_knots.empty()) {
      visual_time.emplace(
          implementation_->pending_visual_knots.front().admission.request.exact_time);
    }
    std::optional<core::FusionTime> guard_time;
    if (!implementation_->pending_imu_guards.empty()) {
      guard_time.emplace(implementation_->pending_imu_guards.front().admission.request.exact_time);
    }
    const auto earliest_non_lidar_time = [&]() -> std::optional<core::FusionTime> {
      if (visual_time && guard_time) {
        return std::min(*visual_time, *guard_time);
      }
      return visual_time ? visual_time : guard_time;
    };
    std::optional<core::FusionTime> imu_only_time;
    if (implementation_->pending_lidar.empty()) {
      imu_only_time = earliest_non_lidar_time();
    }
    std::optional<core::FusionTime> lidar_time;
    bool lidar_exact_share = false;
    std::optional<LocalGraphNavigationSnapshot> lidar_reference_state;
    if (!implementation_->pending_lidar.empty()) {
      Impl::PendingLidarSweep& pending = implementation_->pending_lidar.front();
      if (pending.admission.disposition == StateAdmissionDisposition::SuppressedTooClose) {
        report.dropped_sweeps.push_back(
            LidarDropReport{pending.sweep.id, pending.sweep.acquisition,
                            LidarDropReason::StateRequestSuppressedTooClose});
        implementation_->pending_lidar.pop_front();
        ++implementation_->statistics.lidar_sweeps_dropped;
        continue;
      }
      if (!pending.admission.resolution) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
            "non-suppressed LiDAR state request has no timeline resolution"));
      }
      const auto live_resolution =
          std::find_if(implementation_->state_timeline.resolutions().begin(),
                       implementation_->state_timeline.resolutions().end(),
                       [&](const StateResolution& candidate) {
                         return candidate.exact_time == pending.admission.request.exact_time;
                       });
      if (live_resolution == implementation_->state_timeline.resolutions().end()) {
        report.dropped_sweeps.push_back(LidarDropReport{pending.sweep.id, pending.sweep.acquisition,
                                                        LidarDropReason::ReferenceStateNotLive});
        implementation_->pending_lidar.pop_front();
        ++implementation_->statistics.lidar_sweeps_dropped;
        continue;
      }
      const StateResolution& resolution = *live_resolution;
      if (std::find(resolution.requests.begin(), resolution.requests.end(),
                    pending.admission.request.id) == resolution.requests.end()) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
            "live LiDAR timeline resolution lost its admitted request identity"));
      }
      // Admission describes the state at enqueue time. Re-read the live
      // resolution here because another independent frontend may have created
      // the shared state before this queue reached the graph writer.
      const bool committed_exact_share = resolution.committed_state.has_value();
      if (committed_exact_share) {
        if (resolution.exact_time != pending.sweep.acquisition.end ||
            resolution.exact_time > current.value().state_time) {
          return fail(estimatorError(
              LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
              "committed LiDAR exact share disagrees with its acquisition reference time or "
              "lies after the current graph state"));
        }
        auto reference = implementation_->graph.navigationState(*resolution.committed_state);
        if (!reference &&
            reference.error().code == LocalGraphErrorCode::NavigationStateUnavailable) {
          report.dropped_sweeps.push_back(LidarDropReport{
              pending.sweep.id, pending.sweep.acquisition, LidarDropReason::ReferenceStateNotLive});
          implementation_->pending_lidar.pop_front();
          ++implementation_->statistics.lidar_sweeps_dropped;
          continue;
        }
        if (!reference) {
          return fail(estimatorError(
              LocalEstimatorErrorCode::GraphTransactionFailed, LocalEstimatorStage::LocalGraph,
              "live LiDAR exact-share state query failed: " + reference.error().detail));
        }
        if (reference.value().odom_epoch != current.value().odom_epoch ||
            reference.value().state != *resolution.committed_state ||
            reference.value().exact_time != resolution.exact_time) {
          return fail(estimatorError(
              LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
              "live graph X/V/B snapshot disagrees with the committed timeline resolution"));
        }
        lidar_reference_state = std::move(reference).value();
        lidar_time = resolution.exact_time;
        lidar_exact_share = true;
      } else if (pending.sweep.acquisition.end <= current.value().state_time) {
        // An uncommitted state request can no longer be created behind the
        // graph writer. This is distinct from a committed exact share, which
        // remains attachable for as long as its X/V/B state is live.
        report.dropped_sweeps.push_back(LidarDropReport{pending.sweep.id, pending.sweep.acquisition,
                                                        LidarDropReason::ReferenceStateNotLive});
        implementation_->pending_lidar.pop_front();
        ++implementation_->statistics.lidar_sweeps_dropped;
        continue;
      }

      // Every frontend owns an independent queue. The earliest admitted state
      // request advances the graph; no sensor waits for a synchronized bundle.
      if (!lidar_exact_share) {
        const auto earlier_request = earliest_non_lidar_time();
        if (earlier_request && *earlier_request < pending.sweep.acquisition.start) {
          imu_only_time = earlier_request;
        } else {
          lidar_time = pending.admission.request.exact_time;
          if (*lidar_time <= current.value().state_time) {
            return fail(estimatorError(
                LocalEstimatorErrorCode::InternalInvariant, LocalEstimatorStage::StateTimeline,
                "new LiDAR reference does not follow the committed graph state"));
          }
        }
      }
    }

    const auto earlier_than_lidar = earliest_non_lidar_time();
    if (!imu_only_time && earlier_than_lidar && lidar_time && *earlier_than_lidar < *lidar_time) {
      imu_only_time = earlier_than_lidar;
    }

    if (imu_only_time) {
      const core::FusionTime exact_time = *imu_only_time;
      if (exact_time <= current.value().state_time) {
        return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                   LocalEstimatorStage::StateTimeline,
                                   "pending visual/IMU guard knot is not later than the graph "
                                   "state"));
      }
      if (!implementation_->latest_imu_time || *implementation_->latest_imu_time < exact_time) {
        report.waiting_for_future_imu = true;
        break;
      }
      if (implementation_->next_state == core::StateId::kInvalidValue) {
        return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "navigation state identity exhausted"));
      }
      auto interval = implementation_->imu_buffer.interval(
          core::TimeRange{current.value().state_time, exact_time},
          implementation_->nominal_imu_period);
      if (!interval) {
        return fail(estimatorError(
            LocalEstimatorErrorCode::ImuSupportFailed, LocalEstimatorStage::ImuSupport,
            "exact IMU support for visual/IMU guard knot failed: " + interval.error().detail));
      }
      auto outcome = implementation_->commitImuOnly(core::StateId{implementation_->next_state},
                                                    exact_time, interval.value());
      if (!outcome) {
        return fail(outcome.error());
      }
      report.finalized_states.insert(report.finalized_states.end(),
                                     outcome.value().commit.finalized_states.begin(),
                                     outcome.value().commit.finalized_states.end());
      auto resolved =
          resolve_visual_knots(outcome.value().commit, outcome.value().visual_attachment,
                               outcome.value().visual_degradation);
      if (!resolved) {
        return fail(resolved.error());
      }
      auto resolved_guards = resolve_imu_guards(outcome.value().commit);
      if (!resolved_guards) {
        return fail(resolved_guards.error());
      }
      implementation_->noteCommit(outcome.value().commit);
      continue;
    }

    if (!lidar_time) {
      break;
    }
    Impl::PendingLidarSweep& pending = implementation_->pending_lidar.front();
    if (!implementation_->latest_imu_time ||
        *implementation_->latest_imu_time < pending.sweep.acquisition.end) {
      report.waiting_for_future_imu = true;
      break;
    }
    if (!lidar_exact_share && implementation_->next_state == core::StateId::kInvalidValue) {
      return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "navigation state identity exhausted"));
    }

    core::LidarSweep sweep = std::move(pending.sweep);
    implementation_->pending_lidar.pop_front();
    const core::FusionTime reference_time = *lidar_time;
    const core::StateId state = lidar_exact_share ? lidar_reference_state->state
                                                  : core::StateId{implementation_->next_state};
    std::optional<ImuInterval> navigation_interval;
    core::Result<PropagationResult, PropagationError> propagated = [&]() {
      if (lidar_exact_share) {
        auto before_anchor = implementation_->imu_buffer.interval(
            core::TimeRange{sweep.acquisition.start, reference_time},
            implementation_->nominal_imu_period);
        if (!before_anchor) {
          return core::Result<PropagationResult, PropagationError>::failure(PropagationError{
              PropagationErrorCode::EmptySupport,
              "exact pre-anchor IMU support for late LiDAR deskew is unavailable: " +
                  before_anchor.error().detail});
        }
        return implementation_->propagator.propagateBackwards(
            reference_time, lidar_reference_state->estimate, before_anchor.value());
      }

      auto interval = implementation_->imu_buffer.interval(
          core::TimeRange{current.value().state_time, reference_time},
          implementation_->nominal_imu_period);
      if (!interval) {
        return core::Result<PropagationResult, PropagationError>::failure(
            PropagationError{PropagationErrorCode::EmptySupport,
                             "exact IMU support for LiDAR navigation state is unavailable: " +
                                 interval.error().detail});
      }
      navigation_interval = interval.value();
      if (current.value().state_time <= sweep.acquisition.start) {
        auto deskew_interval = implementation_->imu_buffer.interval(
            core::TimeRange{current.value().state_time, sweep.acquisition.end},
            implementation_->nominal_imu_period);
        if (!deskew_interval) {
          return core::Result<PropagationResult, PropagationError>::failure(
              PropagationError{PropagationErrorCode::EmptySupport,
                               "forward IMU support for the LiDAR acquisition is unavailable: " +
                                   deskew_interval.error().detail});
        }
        return implementation_->propagator.propagate(
            current.value().state_time, current.value().estimate, deskew_interval.value());
      }
      auto before_anchor = implementation_->imu_buffer.interval(
          core::TimeRange{sweep.acquisition.start, current.value().state_time},
          implementation_->nominal_imu_period);
      if (!before_anchor) {
        return core::Result<PropagationResult, PropagationError>::failure(
            PropagationError{PropagationErrorCode::EmptySupport,
                             "pre-anchor IMU support for LiDAR deskew is unavailable: " +
                                 before_anchor.error().detail});
      }
      return implementation_->propagator.propagateAround(
          current.value().state_time, current.value().estimate, before_anchor.value(),
          *navigation_interval);
    }();
    if (!propagated) {
      return fail(estimatorError(LocalEstimatorErrorCode::PropagationFailed,
                                 LocalEstimatorStage::Propagation,
                                 "LiDAR deskew propagation failed: " + propagated.error().detail));
    }

    LidarCommitReport lidar_report;
    lidar_report.measurement = sweep.id;
    lidar_report.navigation_state_created = !lidar_exact_share;
    lidar_report.graph_revision_created = !lidar_exact_share;
    if (reference_time.nanoseconds > std::numeric_limits<std::int64_t>::max() - 3LL) {
      return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "LiDAR FactorBatch timing is exhausted"));
    }
    const auto factor_batch_id =
        implementation_->allocate<core::FactorBatchId>(implementation_->next_factor_batch);
    if (!factor_batch_id) {
      return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "LiDAR FactorBatch identity is exhausted"));
    }
    if (lidar_exact_share) {
      // The X/V/B state and its one IMU edge already exist. This report starts
      // from that immutable graph revision and may advance only if the
      // LiDAR-only FactorBatch is accepted.
      lidar_report.commit = current.value();
    } else {
      if (!navigation_interval) {
        return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                   LocalEstimatorStage::InternalInvariant,
                                   "new LiDAR state has no IMU preintegration interval"));
      }
      auto navigation = implementation_->commitImuOnly(state, reference_time, *navigation_interval);
      if (!navigation) {
        return fail(navigation.error());
      }
      Impl::GraphAppendOutcome navigation_outcome = std::move(navigation).value();
      report.finalized_states.insert(report.finalized_states.end(),
                                     navigation_outcome.commit.finalized_states.begin(),
                                     navigation_outcome.commit.finalized_states.end());
      lidar_report.commit = std::move(navigation_outcome.commit);
      lidar_report.visual_attachment = std::move(navigation_outcome.visual_attachment);
      lidar_report.visual_degradation = std::move(navigation_outcome.visual_degradation);
      auto resolved = resolve_visual_knots(lidar_report.commit, lidar_report.visual_attachment,
                                           lidar_report.visual_degradation);
      if (!resolved) {
        return fail(resolved.error());
      }
      auto resolved_guards = resolve_imu_guards(lidar_report.commit);
      if (!resolved_guards) {
        return fail(resolved_guards.error());
      }
      implementation_->noteCommit(lidar_report.commit);
    }

    core::PipelineWorkIdentity lidar_work;
    lidar_work.measurement = sweep.id;
    lidar_work.state = state;
    const core::ThreadCpuWallTimer tracking_deskew_timer;
    auto deskewed = deskewLidarSweep(
        sweep, reference_time, implementation_->calibration.lidar().extrinsics().T_imu_lidar(),
        propagated.value().trajectory, propagated.value().raw_measurements);
    detail::observeLocalPipelineTiming(
        implementation_->pipeline_timing, LocalPipelineTimingStage::TrackingDeskew,
        tracking_deskew_timer,
        deskewed ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
        lidar_work);
    if (!deskewed) {
      auto health = implementation_->processLidarHealth(*factor_batch_id, reference_time,
                                                        sweep.acquisition.end,
                                                        SensorBatchHealthResult::Failure);
      if (!health) {
        return fail(health.error());
      }
      lidar_report.health_update = health.value().update;
      lidar_report.removed_factor_batches = health.value().removal.batches;
      lidar_report.target_removal = health.value().removal.target_removal;
      if (health.value().removal.commit) {
        lidar_report.commit = *health.value().removal.commit;
        lidar_report.graph_revision_created = true;
      }
      lidar_report.disposition = LidarCommitDisposition::ImuOnlyDeskewRejected;
      lidar_report.degradation_detail = deskewed.error().detail;
      ++implementation_->statistics.lidar_degraded_commits;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }
    lidar_report.deskew_pose_interpolations = deskewed.value().pose_interpolations;

    detail::IdentityTransaction<bool> lidar_identities(implementation_->identityCounters(), false);
    auto& lidar_candidate_counters = lidar_identities.counters();
    const auto derived =
        implementation_->allocate<core::DerivedRecordId>(lidar_candidate_counters.next_derived);
    const auto cloud_lineage = implementation_->allocate<core::ObservationLineageId>(
        lidar_candidate_counters.next_lineage);
    if (!derived || !cloud_lineage) {
      return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "LiDAR registration-view identity exhausted"));
    }
    std::shared_ptr<const LidarRegistrationCloud> source;
    core::ObservationLineage source_lineage =
        registrationCloudLineage(sweep.id, deskewed.value().imu_support,
                                 implementation_->calibration.epoch(), *derived, *cloud_lineage);
    if (core::validateLineage(source_lineage) != core::LineageValidationError::None) {
      static_cast<void>(lidar_identities.abort());
      return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                 LocalEstimatorStage::InternalInvariant,
                                 "constructed LiDAR registration-cloud lineage is invalid"));
    }
    deskewed.value().T_odom_imu_reference = lidar_exact_share
                                                ? lidar_reference_state->estimate.T_odom_imu
                                                : lidar_report.commit.estimate.T_odom_imu;
    {
      const core::ThreadCpuWallTimer cloud_build_timer;
      auto cloud = buildLidarRegistrationCloud(
          std::move(deskewed).value(), source_lineage, implementation_->config.lidar_preprocessing,
          LidarRegistrationIndexConfig{
              implementation_->config.lidar_registration.target_voxel_resolution_m});
      detail::observeLocalPipelineTiming(
          implementation_->pipeline_timing, LocalPipelineTimingStage::RegistrationViewBuild,
          cloud_build_timer,
          cloud ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
          lidar_work);
      if (!cloud) {
        static_cast<void>(lidar_identities.abort());
        auto health = implementation_->processLidarHealth(*factor_batch_id, reference_time,
                                                          sweep.acquisition.end,
                                                          SensorBatchHealthResult::Failure);
        if (!health) {
          return fail(health.error());
        }
        lidar_report.health_update = health.value().update;
        lidar_report.removed_factor_batches = health.value().removal.batches;
        lidar_report.target_removal = health.value().removal.target_removal;
        if (health.value().removal.commit) {
          lidar_report.commit = *health.value().removal.commit;
          lidar_report.graph_revision_created = true;
        }
        lidar_report.disposition = LidarCommitDisposition::ImuOnlyPreprocessingRejected;
        lidar_report.degradation_detail = cloud.error().detail;
        ++implementation_->statistics.lidar_degraded_commits;
        report.commits.push_back(std::move(lidar_report));
        continue;
      }
      lidar_report.preprocessing = cloud.value()->stats;
      source = std::move(cloud).value();
    }

    const bool lidar_factor_due = !implementation_->last_lidar_keyframe_time ||
                                  reference_time - *implementation_->last_lidar_keyframe_time >=
                                      implementation_->config.minimum_lidar_factor_interval;
    const bool finalized_target_available = !implementation_->finalized_target.empty();
    // The persistent finalized map is a global geometric anchor, so query it
    // only when the frontend will actually submit a graph factor. Tracking
    // scans continue to use the pose-aware live window. If the live window is
    // empty, the finalized map remains available as the explicit recovery
    // target regardless of the keyframe cadence.
    const bool use_finalized_target =
        finalized_target_available && (lidar_factor_due || implementation_->rolling_target.empty());
    std::optional<FinalizedLidarTargetReadView> finalized_target_view;
    if (use_finalized_target) {
      finalized_target_view.emplace(implementation_->finalized_target.readView());
    }
    if (implementation_->rolling_target.empty() && !finalized_target_available &&
        implementation_->lidar_map_initialized) {
      // Once localization has admitted a LiDAR map payload, loss of every
      // live target owner is not permission to seed a new map from an
      // unlocalized scan. Keep the IMU state, freeze map insertion, and wait
      // for an explicit relocalization/rebootstrap policy.
      static_cast<void>(lidar_identities.abort());
      lidar_report.disposition = LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained;
      lidar_report.degradation_graph_error_code = LocalGraphErrorCode::FactorBatchStateUnavailable;
      lidar_report.degradation_detail =
          "no live target remains and no graph-final LiDAR target is available yet";
      ++implementation_->statistics.lidar_rejections_target_retained;
      ++implementation_->statistics.lidar_target_state_unavailable_freezes;
      ++implementation_->statistics.lidar_degraded_commits;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    if (implementation_->rolling_target.empty() && !finalized_target_available) {
      auto health = implementation_->processLidarHealth(
          *factor_batch_id, reference_time, sweep.acquisition.end, SensorBatchHealthResult::Good);
      if (!health) {
        static_cast<void>(lidar_identities.abort());
        return fail(health.error());
      }
      lidar_report.health_update = health.value().update;
      lidar_report.removed_factor_batches = health.value().removal.batches;
      lidar_report.target_removal = health.value().removal.target_removal;
      if (health.value().removal.commit) {
        lidar_report.commit = *health.value().removal.commit;
        lidar_report.graph_revision_created = true;
      }
      if (health.value().update.after.state != core::SensorHealthState::Active) {
        static_cast<void>(lidar_identities.abort());
        lidar_report.disposition = LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained;
        lidar_report.degradation_detail =
            "LiDAR initialization seed passed preprocessing while sensor health remains "
            "quarantined";
        ++implementation_->statistics.lidar_degraded_commits;
        report.commits.push_back(std::move(lidar_report));
        continue;
      }
      auto committed_identities = lidar_identities.commit();
      if (!committed_identities) {
        return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "LiDAR bootstrap identities were already consumed"));
      }
      implementation_->publishIdentityCounters(committed_identities.value().counters);
      lidar_report.disposition = LidarCommitDisposition::BootstrapTarget;
      auto added = implementation_->addInitializationSeed(
          sweep, *factor_batch_id, health.value().update.after, lidar_report.commit, state, source);
      if (!added) {
        return fail(added.error());
      }
      lidar_report.target_add = added.value().target_add;
      lidar_report.map_input = std::move(added).value().map_input;
      ++implementation_->statistics.lidar_bootstraps;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    std::optional<RollingLidarTargetBatch> live_target;
    const std::size_t live_target_limit =
        implementation_->config.lidar_registration.maximum_targets -
        (use_finalized_target ? 1U : 0U);
    if (!implementation_->rolling_target.empty() && live_target_limit > 0U) {
      const auto target_lineage = implementation_->allocate<core::ObservationLineageId>(
          lidar_candidate_counters.next_lineage);
      if (!target_lineage) {
        static_cast<void>(lidar_identities.abort());
        return fail(estimatorError(LocalEstimatorErrorCode::IdentityExhausted,
                                   LocalEstimatorStage::IdentityAllocation,
                                   "rolling target output identity exhausted"));
      }
      const core::ThreadCpuWallTimer target_build_timer;
      auto built = implementation_->rolling_target.buildBatch(source, *target_lineage,
                                                                live_target_limit);
      detail::observeLocalPipelineTiming(
          implementation_->pipeline_timing, LocalPipelineTimingStage::TargetBuildUpdate,
          target_build_timer,
          built ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
          lidar_work);
      if (!built) {
        static_cast<void>(lidar_identities.abort());
        return fail(estimatorError(
            LocalEstimatorErrorCode::RollingTargetFailed, LocalEstimatorStage::RollingTarget,
            "scan-local LiDAR target-batch construction failed: " + built.error().detail));
      }
      lidar_report.target = built.value().build;
      live_target = std::move(built).value();
    }

    const std::span<const LidarRegistrationTarget> live_targets =
        live_target ? std::span<const LidarRegistrationTarget>{live_target->targets}
                    : std::span<const LidarRegistrationTarget>{};

    const core::ThreadCpuWallTimer registration_timer;
    auto registration = [&]() {
      if (finalized_target_view) {
        return registerLidarScan(state, source, live_targets, *finalized_target_view,
                                 lidar_report.commit.revision,
                                 implementation_->config.lidar_registration);
      }
      return registerLidarScan(state, source, live_targets,
                               implementation_->config.lidar_registration);
    }();
    const LidarRegistrationWorkCounters& registration_work =
        registration ? registration.value().work : registration.error().work;
    if (registration_work.composite_index_builds > 0U) {
      static_cast<void>(implementation_->pipeline_timing->observe(
          LocalPipelineTimingStage::CompositeTargetIndexBuild,
          registration_work.composite_index_build_duration,
          core::PipelineDisposition::Completed,
          lidar_work));
    }
    detail::observeLocalPipelineTiming(
        implementation_->pipeline_timing, LocalPipelineTimingStage::CorrespondenceRegistrationSolve,
        registration_timer,
        registration ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
        lidar_work);
    if (!registration) {
      static_cast<void>(lidar_identities.abort());
      auto health = implementation_->processLidarHealth(*factor_batch_id, reference_time,
                                                        sweep.acquisition.end,
                                                        SensorBatchHealthResult::Failure);
      if (!health) {
        return fail(health.error());
      }
      lidar_report.health_update = health.value().update;
      lidar_report.removed_factor_batches = health.value().removal.batches;
      lidar_report.target_removal = health.value().removal.target_removal;
      if (health.value().removal.commit) {
        lidar_report.commit = *health.value().removal.commit;
        lidar_report.graph_revision_created = true;
      }
      lidar_report.disposition = LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained;
      lidar_report.registration_error = registration.error();
      lidar_report.degradation_detail = registration.error().detail;
      ++implementation_->statistics.lidar_rejections_target_retained;
      ++implementation_->statistics.lidar_degraded_commits;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    auto health = implementation_->processLidarHealth(
        *factor_batch_id, reference_time, sweep.acquisition.end, SensorBatchHealthResult::Good);
    if (!health) {
      static_cast<void>(lidar_identities.abort());
      return fail(health.error());
    }
    lidar_report.health_update = health.value().update;
    lidar_report.removed_factor_batches = health.value().removal.batches;
    lidar_report.target_removal = health.value().removal.target_removal;
    if (health.value().removal.commit) {
      lidar_report.commit = *health.value().removal.commit;
      lidar_report.graph_revision_created = true;
    }
    if (health.value().update.after.state != core::SensorHealthState::Active) {
      static_cast<void>(lidar_identities.abort());
      lidar_report.disposition = LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained;
      lidar_report.registration = DirectLidarRegistrationReport{
          registration.value().termination, registration.value().initial_robust_cost,
          registration.value().final_robust_cost, registration.value().diagnostics,
          registration.value().work, registration.value().T_odom_source,
          registration.value().source_right_correction};
      lidar_report.degradation_detail =
          "direct point ICP succeeded in shadow mode while LiDAR health remains quarantined";
      ++implementation_->statistics.lidar_degraded_commits;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    if (!lidar_factor_due) {
      // Registration is a complete frontend result even when the bounded
      // keyframe policy deliberately withholds graph and map mutation. The
      // candidate-only cloud and target identities are not externally
      // published, so roll them back while preserving the already allocated
      // health/candidate batch identity as a monotonic audit record.
      static_cast<void>(lidar_identities.abort());
      lidar_report.disposition = LidarCommitDisposition::RegisteredTrackingOnly;
      lidar_report.registration = DirectLidarRegistrationReport{
          registration.value().termination, registration.value().initial_robust_cost,
          registration.value().final_robust_cost, registration.value().diagnostics,
          registration.value().work, registration.value().T_odom_source,
          registration.value().source_right_correction};
      ++implementation_->statistics.lidar_tracking_only_registrations;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    auto finalized_capacity = implementation_->requireFinalizedTargetCapacity();
    if (!finalized_capacity) {
      static_cast<void>(lidar_identities.abort());
      return fail(finalized_capacity.error());
    }

    const core::ThreadCpuWallTimer factor_prepare_timer;
    auto prepared = implementation_->prepareLidarFactorBatch(
        sweep, *factor_batch_id, health.value().update.after, state, reference_time, source,
        live_targets, std::move(registration).value(), lidar_candidate_counters);
    detail::observeLocalPipelineTiming(
        implementation_->pipeline_timing, LocalPipelineTimingStage::LidarFactorBatchPrepare,
        factor_prepare_timer,
        prepared ? core::PipelineDisposition::Completed : core::PipelineDisposition::Failed,
        lidar_work);
    if (!prepared) {
      static_cast<void>(lidar_identities.abort());
      return fail(prepared.error());
    }
    lidar_report.registration = prepared.value().batch.registration_report;
    const core::FactorBatchMetadata batch_metadata = prepared.value().batch.metadata;
    const std::vector<DirectLidarPairReport> candidate_pairs = prepared.value().pairs;
    const std::optional<DirectLidarFinalizedMapReport> candidate_finalized_map =
        prepared.value().finalized_map;
    auto inserted = implementation_->graph.insertFactorBatch(std::move(prepared).value().batch);

    // A constructed batch identity is never reused, including when the graph
    // rejects the atomic insertion. This keeps diagnostics and provenance
    // monotonic across sensor failures.
    auto committed_identities = lidar_identities.commit();
    if (!committed_identities) {
      return fail(estimatorError(LocalEstimatorErrorCode::InternalInvariant,
                                 LocalEstimatorStage::IdentityAllocation,
                                 "constructed LiDAR FactorBatch identities were already consumed"));
    }
    implementation_->publishIdentityCounters(committed_identities.value().counters);

    if (!inserted) {
      const bool target_state_unavailable =
          inserted.error().code == LocalGraphErrorCode::FactorBatchStateUnavailable;
      const bool recoverable =
          target_state_unavailable ||
          inserted.error().code == LocalGraphErrorCode::NonlinearCostIncrease ||
          inserted.error().code == LocalGraphErrorCode::PoseCorrectionLimit ||
          inserted.error().code == LocalGraphErrorCode::NonlinearConvergenceFailure;
      if (!recoverable) {
        LocalEstimatorError error = estimatorError(
            LocalEstimatorErrorCode::GraphTransactionFailed, LocalEstimatorStage::LocalGraph,
            "atomic LiDAR FactorBatch insertion failed: " + inserted.error().detail);
        error.graph_error_code = inserted.error().code;
        error.rejected_solve = inserted.error().rejected_solve;
        error.lidar_registration = inserted.error().lidar_registration;
        error.lidar_pairs = inserted.error().lidar_pairs;
        error.lidar_finalized_map = inserted.error().lidar_finalized_map;
        return fail(std::move(error));
      }
      lidar_report.disposition = LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained;
      lidar_report.degradation_graph_error_code = inserted.error().code;
      lidar_report.rejected_solve = inserted.error().rejected_solve;
      lidar_report.rejected_lidar_pairs = candidate_pairs;
      lidar_report.rejected_lidar_finalized_map = candidate_finalized_map;
      lidar_report.degradation_detail = inserted.error().detail;
      ++implementation_->statistics.lidar_rejections_target_retained;
      if (target_state_unavailable) {
        ++implementation_->statistics.lidar_target_state_unavailable_freezes;
      }
      ++implementation_->statistics.lidar_degraded_commits;
      report.commits.push_back(std::move(lidar_report));
      continue;
    }

    auto published = implementation_->publishFactorOnlyCommit(inserted.value());
    if (!published) {
      return fail(published.error());
    }
    lidar_report.commit = std::move(inserted).value();
    lidar_report.graph_revision_created = true;
    lidar_report.disposition = LidarCommitDisposition::Registered;
    auto mapped = implementation_->addAcceptedFactorBatchCloud(batch_metadata, lidar_report.commit,
                                                               state, sweep, source);
    if (!mapped) {
      return fail(mapped.error());
    }
    lidar_report.target_add = mapped.value().target_add;
    lidar_report.map_input = std::move(mapped).value().map_input;
    implementation_->last_lidar_keyframe_time = reference_time;
    implementation_->statistics.last_lidar_keyframe_time = reference_time;
    ++implementation_->statistics.lidar_registrations;
    report.commits.push_back(std::move(lidar_report));
  }

  report.lifecycle = implementation_->lifecycle;
  report.pending_sweeps = implementation_->pending_lidar.size();
  report.pending_imu_guards = implementation_->pending_imu_guards.size();
  report.pending_camera_knots = implementation_->pending_visual_knots.size();
  return complete();
}

core::Result<LocalEstimatorPropagationReport, LocalEstimatorError> LocalEstimator::propagateTo(
    core::FusionTime exact_time) const {
  using Result = core::Result<LocalEstimatorPropagationReport, LocalEstimatorError>;
  auto anchor = implementation_->graph.estimate();
  if (!anchor) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::PropagationFailed, LocalEstimatorStage::Propagation,
                       "high-rate IMU propagation requires an initialized graph anchor"));
  }
  if (exact_time < anchor.value().state_time) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::PropagationFailed, LocalEstimatorStage::Propagation,
        "high-rate IMU propagation cannot precede the latest committed graph state"));
  }

  LocalEstimatorPropagationReport report;
  report.odom_epoch = anchor.value().odom_epoch;
  report.anchor_state = anchor.value().state;
  report.anchor_revision = anchor.value().revision;
  report.anchor_time = anchor.value().state_time;
  report.anchor_estimate = anchor.value().estimate;
  report.exact_time = exact_time;
  report.propagated_state = anchor.value().estimate;
  if (exact_time == anchor.value().state_time) {
    report.gap_status = ImuPropagationGapStatus::AnchorOnly;
    return Result::success(std::move(report));
  }

  auto interval = implementation_->imu_buffer.interval(
      core::TimeRange{anchor.value().state_time, exact_time}, implementation_->nominal_imu_period);
  if (!interval) {
    LocalEstimatorError error = estimatorError(
        LocalEstimatorErrorCode::ImuSupportFailed, LocalEstimatorStage::ImuSupport,
        "exact IMU support for high-rate propagation failed: " + interval.error().detail);
    error.imu_buffer_error_code = interval.error().code;
    return Result::failure(std::move(error));
  }
  auto propagated = implementation_->propagator.propagate(
      anchor.value().state_time, anchor.value().estimate, interval.value());
  if (!propagated) {
    return Result::failure(
        estimatorError(LocalEstimatorErrorCode::PropagationFailed, LocalEstimatorStage::Propagation,
                       "high-rate midpoint IMU propagation failed: " + propagated.error().detail));
  }

  report.propagated_state = propagated.value().final_state;
  report.raw_imu_support = propagated.value().raw_measurements;
  report.maximum_raw_gap = interval.value().maximum_raw_gap;
  report.inferred_missing_ticks = propagated.value().inferred_missing_ticks;
  report.contains_saturation = interval.value().contains_saturation;
  report.gap_status = report.inferred_missing_ticks == 0U
                          ? ImuPropagationGapStatus::Contiguous
                          : ImuPropagationGapStatus::InferredMissingTicks;
  return Result::success(std::move(report));
}

LocalEstimatorLifecycle LocalEstimator::lifecycle() const noexcept {
  return implementation_->lifecycle;
}

const LocalEstimatorStatistics& LocalEstimator::statistics() const noexcept {
  return implementation_->statistics;
}

const FinalizedLidarTargetMapStatistics& LocalEstimator::finalizedLidarTargetStatistics()
    const noexcept {
  return implementation_->finalized_target.statistics();
}

LocalPipelineTimingReport LocalEstimator::pipelineTimingReport() const noexcept {
  return implementation_->pipeline_timing->snapshot();
}

const LocalEstimatorConfig& LocalEstimator::effectiveConfig() const noexcept {
  return implementation_->config;
}

core::Result<LocalGraphCommit, LocalEstimatorError> LocalEstimator::estimate() const {
  using Result = core::Result<LocalGraphCommit, LocalEstimatorError>;
  auto estimate = implementation_->graph.estimate();
  if (!estimate) {
    return Result::failure(estimatorError(
        LocalEstimatorErrorCode::InitializationFailed, LocalEstimatorStage::MotionInitialization,
        "local estimate is unavailable before the initialization state machine commits"));
  }
  return Result::success(std::move(estimate).value());
}

}  // namespace meridian::local
