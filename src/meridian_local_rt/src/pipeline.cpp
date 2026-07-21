#include "meridian/local_rt/pipeline.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace meridian::local_rt {
namespace {

using Clock = std::chrono::steady_clock;

std::int64_t elapsedNs(const Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

double gravityMagnitude(const core::Vec3d& gravity) {
  return std::sqrt(gravity.x * gravity.x + gravity.y * gravity.y + gravity.z * gravity.z);
}

core::Vec3d rotationVector(const core::Quaterniond& quaternion) {
  const Eigen::Quaterniond eigen_quaternion(quaternion.w(), quaternion.x(), quaternion.y(),
                                            quaternion.z());
  const Eigen::AngleAxisd angle_axis(eigen_quaternion);
  const Eigen::Vector3d vector = angle_axis.axis() * angle_axis.angle();
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

std::optional<core::TimeNs> shifted(core::TimeNs time, std::int64_t offset_ns) {
  return core::TimeNs::checkedAdd(time, offset_ns);
}

Sophus::SE3d sophus(const core::Pose3d& pose) {
  const core::Quaterniond& q = pose.rotation();
  const core::Vec3d& p = pose.translation();
  return Sophus::SE3d(Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()),
                      Eigen::Vector3d(p.x, p.y, p.z));
}

core::Pose3d corePose(const Sophus::SE3d& pose) {
  const Eigen::Quaterniond q(pose.unit_quaternion());
  return core::Pose3d(
      {.x = pose.translation().x(), .y = pose.translation().y(), .z = pose.translation().z()},
      core::Quaterniond(q.w(), q.x(), q.y(), q.z()));
}

}  // namespace

LocalRtPipeline::LocalRtPipeline(LocalRtPipelineConfig config, core::DebugSink& debug_sink)
    : config_(std::move(config)),
      debug_sink_(debug_sink),
      imu_buffer_(config_.estimator.imu_buffer),
      preintegrator_(config_.estimator.imu_model),
      propagator_(config_.estimator.imu_model),
      scan_preprocessor_(config_.scan_preprocessor),
      estimator_(config_.fixed_lag, config_.scan_to_map, config_.estimator.extrinsics.T_imu_lidar) {
  const std::vector<ConfigIssue> issues = config_.estimator.validate();
  if (!issues.empty()) {
    throw std::invalid_argument("local_rt estimator configuration is invalid at " +
                                issues.front().field);
  }
  if (config_.imu_queue_capacity == 0U || config_.lidar_queue_capacity == 0U) {
    throw std::invalid_argument("local_rt observation queue capacities must be positive");
  }
  if (config_.estimator.initialization.mode == core::InitializationMode::kStatic) {
    config_.static_initializer.gravity_m_s2 =
        gravityMagnitude(config_.estimator.imu_model.gravity_odom_m_s2);
    config_.static_initializer.calibrated_bias_prior =
        config_.estimator.initialization.calibrated_bias_prior;
    config_.static_initializer.base_from_imu = config_.estimator.extrinsics.T_base_imu;
    static_initializer_ =
        std::make_unique<initialization::StaticInitializer>(config_.static_initializer);
  } else {
    config_.dynamic_initializer.gravity_m_s2 =
        gravityMagnitude(config_.estimator.imu_model.gravity_odom_m_s2);
    config_.dynamic_initializer.calibrated_bias_prior =
        config_.estimator.initialization.calibrated_bias_prior;
    config_.dynamic_initializer.gyroscope_bias_prior_covariance =
        config_.estimator.initialization.calibrated_bias_prior_covariance.topLeftCorner<3, 3>();
    config_.dynamic_initializer.base_from_imu = config_.estimator.extrinsics.T_base_imu;
    config_.dynamic_initializer.imu_from_lidar = config_.estimator.extrinsics.T_imu_lidar;
    config_.dynamic_initializer.lidar_time_offset_to_imu_ns =
        config_.estimator.extrinsics.lidar_time_offset_to_imu.count();
    dynamic_initializer_ =
        std::make_unique<initialization::DynamicInitializer>(config_.dynamic_initializer);
  }

  worker_ = std::thread([this] { workerLoop(); });
}

LocalRtPipeline::~LocalRtPipeline() {
  stopAndDrain();
}

void LocalRtPipeline::submit(core::ImuSample sample) {
  {
    std::lock_guard lock(mutex_);
    if (!accepting_) {
      throw std::logic_error("local_rt pipeline has stopped accepting observations");
    }
    if (imu_queue_.size() >= config_.imu_queue_capacity) {
      throw std::runtime_error("local_rt IMU queue is full");
    }
    imu_queue_.push_back(std::move(sample));
  }
  condition_.notify_one();
}

void LocalRtPipeline::submit(core::LidarSweep sweep) {
  {
    std::lock_guard lock(mutex_);
    if (!accepting_) {
      throw std::logic_error("local_rt pipeline has stopped accepting observations");
    }
    if (lidar_queue_.size() >= config_.lidar_queue_capacity) {
      throw std::runtime_error("local_rt LiDAR queue is full");
    }
    lidar_queue_.push_back(std::move(sweep));
  }
  condition_.notify_one();
}

void LocalRtPipeline::stopAndDrain() noexcept {
  {
    std::lock_guard lock(mutex_);
    if (stopped_) {
      return;
    }
    accepting_ = false;
  }
  condition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  recordTerminalTrajectory();
  {
    std::lock_guard lock(mutex_);
    stopped_ = true;
  }
}

std::optional<core::InitializationResult> LocalRtPipeline::initializationResult() const {
  std::lock_guard lock(mutex_);
  return initialization_result_;
}

void LocalRtPipeline::workerLoop() noexcept {
  try {
    while (true) {
      std::optional<core::ImuSample> imu;
      std::optional<core::LidarSweep> lidar;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(
            lock, [this] { return !accepting_ || !imu_queue_.empty() || !lidar_queue_.empty(); });
        bool lidar_has_end_support = false;
        if (!lidar_queue_.empty()) {
          const auto lidar_end =
              shifted(lidar_queue_.front().acquisitionEnd(),
                      config_.estimator.extrinsics.lidar_time_offset_to_imu.count());
          const auto latest_imu = imu_buffer_.latestTime();
          lidar_has_end_support =
              !lidar_end.has_value() || (latest_imu.has_value() && *latest_imu >= *lidar_end);
        }
        if (!imu_queue_.empty() && (!lidar_has_end_support || lidar_queue_.empty())) {
          imu.emplace(std::move(imu_queue_.front()));
          imu_queue_.pop_front();
        } else if (!lidar_queue_.empty()) {
          lidar.emplace(std::move(lidar_queue_.front()));
          lidar_queue_.pop_front();
        } else if (!accepting_) {
          break;
        }
      }

      if (imu.has_value()) {
        processImu(std::move(*imu));
        continue;
      }
      if (lidar.has_value() && !processLidar(*lidar)) {
        std::unique_lock lock(mutex_);
        if (accepting_ || !imu_queue_.empty()) {
          lidar_queue_.push_front(std::move(*lidar));
          condition_.wait(lock, [this] { return !accepting_ || !imu_queue_.empty(); });
        }
      }
    }
  } catch (const std::exception& error) {
    core::TimeNs event_time(0);
    {
      std::lock_guard lock(mutex_);
      if (!imu_queue_.empty()) {
        event_time = imu_queue_.front().header().measurementTime();
      } else if (!lidar_queue_.empty()) {
        event_time = lidar_queue_.front().header().measurementTime();
      }
      accepting_ = false;
      imu_queue_.clear();
      lidar_queue_.clear();
    }
    recordInitialization(event_time, core::InitializationStatus::kFailed,
                         std::string("local_rt worker exception: ") + error.what(), {});
  } catch (...) {
    {
      std::lock_guard lock(mutex_);
      accepting_ = false;
      imu_queue_.clear();
      lidar_queue_.clear();
    }
    recordInitialization(core::TimeNs(0), core::InitializationStatus::kFailed,
                         "unknown local_rt worker exception", {});
  }
}

void LocalRtPipeline::processImu(core::ImuSample sample) {
  const core::TimeNs time = sample.header().measurementTime();
  const ImuInsertResult inserted = imu_buffer_.insert(sample);
  if (!inserted.ok()) {
    if (!initialization_result_.has_value()) {
      recordInitialization(time, core::InitializationStatus::kCollecting,
                           "IMU sample rejected by the local exact-support buffer", {});
    }
    return;
  }
  if (initialization_result_.has_value() || !static_initializer_) {
    return;
  }

  initialization::StaticInitializationUpdate update = static_initializer_->add(sample);
  if (update.result.has_value()) {
    acceptInitialization(*update.result, update.reason);
  } else {
    recordInitialization(time, update.status, update.reason, update.quality);
  }
}

bool LocalRtPipeline::processLidar(const core::LidarSweep& sweep) {
  const std::int64_t offset = config_.estimator.extrinsics.lidar_time_offset_to_imu.count();
  const auto begin = shifted(sweep.acquisitionBegin(), offset);
  const auto end = shifted(sweep.acquisitionEnd(), offset);
  if (!begin.has_value() || !end.has_value() || *end < *begin) {
    recordInitialization(sweep.header().measurementTime(), core::InitializationStatus::kFailed,
                         "LiDAR time offset produced invalid IMU support", {});
    return true;
  }
  if (*end > *begin) {
    const ImuIntervalResult scan_support = imu_buffer_.interval(*begin, *end);
    if (!scan_support.ok()) {
      const ImuIntervalFailure* failure = scan_support.error();
      if (failure != nullptr && failure->code == ImuIntervalErrorCode::kEndNotBracketed) {
        return false;
      }
      if (!initialization_result_.has_value()) {
        recordInitialization(
            sweep.header().measurementTime(), core::InitializationStatus::kCollecting,
            failure != nullptr ? failure->message : "LiDAR sweep has no exact IMU support", {});
      }
      return true;
    }
  }

  if (!initialization_result_.has_value()) {
    if (!dynamic_initializer_) {
      return true;
    }
    initialization::DynamicInitializationUpdate update =
        dynamic_initializer_->add(sweep, imu_buffer_, preintegrator_);
    if (update.bootstrap_pose.has_value()) {
      recordBootstrap(*update.bootstrap_pose);
    }
    if (update.result.has_value()) {
      acceptInitialization(*update.result, update.reason,
                           dynamic_initializer_->acceptedAnchorPoints());
    } else {
      recordInitialization(sweep.header().measurementTime(), update.status, update.reason,
                           update.quality);
    }
    return true;
  }

  if (!last_preintegration_time_.has_value() || *end <= *last_preintegration_time_) {
    return true;
  }
  const Clock::time_point lidar_total_begin = Clock::now();
  const core::NavigationState propagation_seed = estimator_.latestState();
  const core::StateId next_state(last_state_id_.value() + 1U);
  const Clock::time_point support_begin = Clock::now();
  ImuIntervalResult interval_result = imu_buffer_.interval(*last_preintegration_time_, *end);
  recordTiming(*end, next_state, "imu_support_query", "lidar_total", elapsedNs(support_begin));
  if (!interval_result.ok()) {
    recordTiming(*end, next_state, "lidar_total", "", elapsedNs(lidar_total_begin));
    return true;
  }
  const ImuInterval& interval = *interval_result.value();
  const Clock::time_point preintegration_begin = Clock::now();
  const PreintegrationResult preintegration =
      preintegrator_.integrate(interval, propagation_seed.imuBias());
  recordTiming(*end, next_state, "preintegration", "lidar_total", elapsedNs(preintegration_begin));
  if (!preintegration.ok()) {
    recordTiming(*end, next_state, "lidar_total", "", elapsedNs(lidar_total_begin));
    return true;
  }

  const Clock::time_point propagation_begin = Clock::now();
  const PropagationResult propagation =
      propagator_.propagate(propagation_seed, next_state, interval);
  recordTiming(*end, next_state, "dense_propagation", "lidar_total", elapsedNs(propagation_begin));
  if (!propagation.ok()) {
    recordTiming(*end, next_state, "lidar_total", "", elapsedNs(lidar_total_begin));
    return true;
  }

  std::optional<lidar::PreparedScan> prepared;
  try {
    prepared.emplace(
        scan_preprocessor_.prepare(sweep, propagation_seed, next_state, *propagation.value()));
  } catch (const std::exception&) {
    recordTiming(*end, next_state, "lidar_total", "", elapsedNs(lidar_total_begin));
    return true;
  }
  recordTiming(*end, next_state, "range_filter", "preprocess", prepared->timing.range_filter_ns);
  recordTiming(*end, next_state, "deskew", "preprocess", prepared->timing.deskew_ns);
  recordTiming(*end, next_state, "target_downsample", "preprocess",
               prepared->timing.target_downsample_ns);
  recordTiming(*end, next_state, "source_downsample", "preprocess",
               prepared->timing.source_downsample_ns);
  recordTiming(*end, next_state, "preprocess", "lidar_total", prepared->timing.total_ns);

  recordPreintegration(interval, *preintegration.value(), last_state_id_, next_state);
  estimator::FixedLagUpdate update = estimator_.addSweep(
      std::move(prepared->frame), *preintegration.value(), propagation.value()->endpoint);
  const lidar::ScanToMapTiming& first_timing = update.first_association.timing;
  const lidar::ScanToMapTiming& second_timing = update.registration.timing;
  const bool has_second_association = update.association_passes > 1U;
  const auto timingSum = [has_second_association](std::int64_t first,
                                                  std::int64_t second) noexcept {
    return first + (has_second_association ? second : 0);
  };
  recordTiming(*end, next_state, "registration_owner_selection", "registration_total",
               timingSum(first_timing.owner_selection_ns, second_timing.owner_selection_ns));
  recordTiming(*end, next_state, "registration_live_composite_rebuild", "registration_total",
               timingSum(first_timing.live_composite_rebuild_ns,
                         second_timing.live_composite_rebuild_ns));
  recordTiming(*end, next_state, "registration_active_query", "registration_total",
               timingSum(first_timing.active_query_ns, second_timing.active_query_ns));
  recordTiming(*end, next_state, "registration_finalized_query", "registration_total",
               timingSum(first_timing.finalized_query_ns, second_timing.finalized_query_ns));
  recordTiming(*end, next_state, "registration_robust_scale", "registration_total",
               timingSum(first_timing.robust_scale_ns, second_timing.robust_scale_ns));
  recordTiming(*end, next_state, "registration_linearize", "registration_total",
               timingSum(first_timing.linearization_ns, second_timing.linearization_ns));
  recordTiming(*end, next_state, "registration_solve", "registration_total",
               timingSum(first_timing.solve_ns, second_timing.solve_ns));
  const auto recordAssociationPass = [&](const lidar::ScanToMapTiming& timing,
                                         const std::string& pass) {
    const std::string parent = "association_" + pass;
    recordTiming(*end, next_state, parent + "_owner_selection", parent,
                 timing.owner_selection_ns);
    recordTiming(*end, next_state, parent + "_live_composite_rebuild", parent,
                 timing.live_composite_rebuild_ns);
    recordTiming(*end, next_state, parent + "_active_query", parent, timing.active_query_ns);
    recordTiming(*end, next_state, parent + "_finalized_query", parent,
                 timing.finalized_query_ns);
    recordTiming(*end, next_state, parent + "_robust_scale", parent, timing.robust_scale_ns);
    recordTiming(*end, next_state, parent + "_linearize", parent, timing.linearization_ns);
  };
  recordAssociationPass(first_timing, "first");
  if (has_second_association) {
    recordAssociationPass(second_timing, "second");
  }
  recordTiming(*end, next_state, "registration_total", "estimator_total",
               update.timing.registration_ns);
  recordTiming(*end, next_state, "factor_build", "estimator_total", update.timing.factor_build_ns);
  recordTiming(*end, next_state, "window_problem_build_first", "window_problem_build",
               update.timing.problem_build_first_ns);
  if (has_second_association) {
    recordTiming(*end, next_state, "window_problem_build_second", "window_problem_build",
                 update.timing.problem_build_second_ns);
  }
  recordTiming(*end, next_state, "window_problem_build", "estimator_total",
               update.timing.problem_build_ns);
  recordTiming(*end, next_state, "window_solve_first", "window_solve",
               update.timing.ceres_solve_first_ns);
  if (has_second_association) {
    recordTiming(*end, next_state, "window_solve_second", "window_solve",
                 update.timing.ceres_solve_second_ns);
  }
  recordTiming(*end, next_state, "window_solve", "estimator_total", update.timing.ceres_solve_ns);
  recordTiming(*end, next_state, "window_validation", "estimator_total",
               update.timing.validation_ns);
  recordTiming(*end, next_state, "marginalization_evaluate", "marginalization",
               update.timing.marginalization_evaluate_ns);
  recordTiming(*end, next_state, "marginalization_eliminate", "marginalization",
               update.timing.marginalization_eliminate_ns);
  recordTiming(*end, next_state, "marginalization_prior_build", "marginalization",
               update.timing.marginalization_prior_build_ns);
  recordTiming(*end, next_state, "target_finalize", "estimator_total",
               update.timing.target_finalize_ns);
  recordTiming(*end, next_state, "target_admit", "estimator_total", update.timing.target_admit_ns);
  recordTiming(*end, next_state, "state_commit", "estimator_total", update.timing.commit_ns);
  recordTiming(*end, next_state, "estimator_total", "lidar_total", update.timing.total_ns);
  recordEstimator(update, next_state, *end, prepared->stats);
  if (update.accepted() && update.optimized.has_value()) {
    ++estimator_revision_;
    last_preintegration_time_ = update.optimized->time();
    last_state_id_ = update.optimized->id();
    recordTrajectory(*update.optimized, core::LocalTrajectoryKind::kOnline, estimator_revision_);
    for (const core::NavigationState& finalized : update.newly_finalized) {
      recordTrajectory(finalized, core::LocalTrajectoryKind::kFinalized, estimator_revision_);
    }

    // A registration-map snapshot is an audit observation of each accepted
    // estimator revision. It deliberately follows the native LiDAR update
    // cadence: no independent debug sampling clock may hide a transient map.
    recordRegistrationMap(*end, update.optimized->id());
  }
  recordTiming(*end, next_state, "lidar_total", "", elapsedNs(lidar_total_begin));
  return true;
}

void LocalRtPipeline::acceptInitialization(const core::InitializationResult& result,
                                           const std::string& reason,
                                           std::span<const lidar::Point3d> anchor_points) {
  if (!anchor_points.empty()) {
    lidar::PreparedScan anchor = scan_preprocessor_.prepareSweepEndPoints(
        result.anchorTime(), result.seedState().id(), anchor_points);
    estimator_.initialize(result.seedState(), std::move(anchor.frame));
  } else {
    estimator_.initialize(result.seedState());
  }
  last_preintegration_time_ = result.anchorTime();
  last_state_id_ = result.seedState().id();
  {
    std::lock_guard lock(mutex_);
    initialization_result_ = result;
  }
  recordInitialization(result.anchorTime(), core::InitializationStatus::kAccepted, reason,
                       result.quality(), result.seedState());
  recordTrajectory(result.seedState(), core::LocalTrajectoryKind::kOnline, estimator_revision_);
  if (!anchor_points.empty()) {
    recordRegistrationMap(result.anchorTime(), result.seedState().id());
  }
}

void LocalRtPipeline::recordInitialization(core::TimeNs time, core::InitializationStatus status,
                                           const std::string& reason,
                                           const core::InitializationQuality& quality,
                                           const std::optional<core::NavigationState>& seed) {
  if (status != core::InitializationStatus::kAccepted && status == last_reported_status_ &&
      reason == last_reported_reason_) {
    return;
  }
  last_reported_status_ = status;
  last_reported_reason_ = reason;
  debug_sink_.record(core::InitializationEvent{
      .event_time = time,
      .mode = config_.estimator.initialization.mode,
      .status = status,
      .reason = reason,
      .quality = quality,
      .accepted_seed = seed,
  });
}

void LocalRtPipeline::recordBootstrap(const initialization::BootstrapPoseSummary& pose) {
  debug_sink_.record(core::BootstrapPoseEvent{
      .measurement_id = pose.measurement_id,
      .measurement_time = pose.measurement_time,
      .odom_from_lidar = pose.odom_from_lidar,
      .source_point_count = static_cast<std::uint64_t>(pose.source_point_count),
      .correspondence_count = static_cast<std::uint64_t>(pose.correspondence_count),
      .point_rmse_m = pose.point_rmse_m,
      .hessian_condition_number = pose.hessian_condition_number,
      .accepted = pose.accepted,
  });
}

void LocalRtPipeline::recordPreintegration(const ImuInterval& interval,
                                           const CombinedPreintegration& preintegration,
                                           core::StateId from, core::StateId to) {
  std::int64_t maximum_gap_ns = 0;
  for (const ImuIntegrationSegment& segment : interval.segments()) {
    maximum_gap_ns = std::max(maximum_gap_ns, segment.support().durationNs().value_or(0));
  }
  debug_sink_.record(core::PreintegrationEvent{
      .from_state = from,
      .to_state = to,
      .support = interval.support(),
      .source_sample_count = static_cast<std::uint64_t>(interval.sourceSampleCount()),
      .integration_segment_count = static_cast<std::uint64_t>(interval.segments().size()),
      .maximum_source_gap_ns = maximum_gap_ns,
      .delta_rotation_vector_rad = rotationVector(preintegration.deltaRotation()),
      .delta_velocity_m_s = preintegration.deltaVelocity(),
      .delta_position_m = preintegration.deltaPosition(),
      .backend = std::string(GtsamCombinedPreintegrator::backendName()),
  });
}

void LocalRtPipeline::recordTiming(core::TimeNs time, std::optional<core::StateId> state_id,
                                   std::string stage, std::string parent_stage,
                                   std::int64_t duration_ns) {
  debug_sink_.record(core::StageTimingEvent{.event_time = time,
                                            .state_id = state_id,
                                            .stage = std::move(stage),
                                            .parent_stage = std::move(parent_stage),
                                            .duration_ns = std::max<std::int64_t>(0, duration_ns)});
}

void LocalRtPipeline::recordTrajectory(const core::NavigationState& state,
                                       core::LocalTrajectoryKind kind, std::uint64_t revision) {
  const Sophus::SE3d odom_from_base =
      sophus(state.odomFromImu()) * sophus(config_.estimator.extrinsics.T_base_imu).inverse();
  debug_sink_.record(core::LocalTrajectoryEvent{
      .state = state,
      .odom_from_base = corePose(odom_from_base),
      .estimator_revision = revision,
      .kind = kind,
  });
}

void LocalRtPipeline::recordRegistrationMap(core::TimeNs time, core::StateId state_id) {
  const Clock::time_point map_snapshot_begin = Clock::now();
  const lidar::PointCloud registration_map = estimator_.registrationMapPointCloud();
  auto map_points = std::make_shared<std::vector<core::LidarPreviewPoint>>();
  map_points->reserve(registration_map.size());
  for (const lidar::Point3d& point : registration_map) {
    if (point.allFinite()) {
      map_points->push_back({.x = static_cast<float>(point.x()),
                             .y = static_cast<float>(point.y()),
                             .z = static_cast<float>(point.z())});
    }
  }
  recordTiming(time, state_id, "registration_map_snapshot", "debug_total",
               elapsedNs(map_snapshot_begin));
  debug_sink_.record(core::LocalRegistrationMapEvent{
      .event_time = time,
      .state_id = state_id,
      .estimator_revision = estimator_revision_,
      .points = std::move(map_points),
  });
}

void LocalRtPipeline::recordEstimator(const estimator::FixedLagUpdate& update,
                                      core::StateId candidate_id, core::TimeNs time,
                                      const lidar::ScanPreprocessorStats& preprocessing) {
  debug_sink_.record(core::LocalEstimatorEvent{
      .state_id = candidate_id,
      .event_time = time,
      .estimator_revision = estimator_revision_ + static_cast<std::uint64_t>(update.accepted()),
      .outcome = std::string(estimator::toString(update.status)) + ": " + update.reason,
      .active_state_count = static_cast<std::uint64_t>(update.active_states),
      .imu_factor_count = static_cast<std::uint64_t>(update.imu_factors),
      .lidar_batch_count =
          static_cast<std::uint64_t>(update.active_lidar_groups + update.finalized_lidar_groups),
      .active_lidar_rows = static_cast<std::uint64_t>(update.active_lidar_rows),
      .finalized_lidar_rows = static_cast<std::uint64_t>(update.finalized_lidar_rows),
      .finalized_map_points = static_cast<std::uint64_t>(update.finalized_map_points),
      .selected_active_owners =
          static_cast<std::uint64_t>(update.registration.selected_active_owners),
      .registration_correspondences = static_cast<std::uint64_t>(update.registration.rows.size()),
      .marginal_prior_rank = static_cast<std::uint64_t>(update.prior_rank),
      .registration_rmse_m = update.registration.rmse_m,
      .initial_cost = update.initial_cost,
      .final_cost = update.final_cost,
      .pose_correction_translation_m = update.correction_translation_m,
      .pose_correction_rotation_rad = update.correction_rotation_rad,
      .accepted = update.accepted(),
      .prepared_target_points = static_cast<std::uint64_t>(preprocessing.target_points),
      .prepared_source_points = static_cast<std::uint64_t>(preprocessing.source_points),
      .association_pass_count = static_cast<std::uint64_t>(update.association_passes),
      .association_input_points = static_cast<std::uint64_t>(
          preprocessing.source_points * update.association_passes),
      .association_rows_before_cap = static_cast<std::uint64_t>(
          update.registration.correspondences_before_cap),
      .registration_iterations = static_cast<std::uint64_t>(update.registration.iterations),
      .live_query_voxel_probes = update.first_association.timing.active_voxel_probes +
                                 (update.association_passes > 1U
                                      ? update.registration.timing.active_voxel_probes
                                      : 0U),
      .finalized_query_voxel_probes =
          update.first_association.timing.finalized_voxel_probes +
          (update.association_passes > 1U
               ? update.registration.timing.finalized_voxel_probes
               : 0U),
      .reassociated_rows = static_cast<std::uint64_t>(update.reassociated_rows),
      .rejected_stale_rows = static_cast<std::uint64_t>(update.rejected_stale_rows),
  });
}

void LocalRtPipeline::recordTerminalTrajectory() noexcept {
  if (terminal_trajectory_recorded_ || !estimator_.initialized()) {
    return;
  }
  terminal_trajectory_recorded_ = true;
  try {
    for (const core::NavigationState& state : estimator_.activeStates()) {
      recordTrajectory(state, core::LocalTrajectoryKind::kTerminal, estimator_revision_);
    }
  } catch (...) {
    // Debug finalization remains observer-only and must not escape shutdown.
  }
}

}  // namespace meridian::local_rt
