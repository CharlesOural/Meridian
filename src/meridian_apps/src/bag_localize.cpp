#include "meridian/apps/bag_localize.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "meridian/core/sha256.hpp"
#include "meridian/local/local_estimator.hpp"
#include "meridian/tools/bag_replay.hpp"
#include "process_report_normalizer.hpp"

namespace meridian::apps {
namespace {

template <class... Visitors>
struct Overloaded : Visitors... {
  using Visitors::operator()...;
};
template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

constexpr core::CalibrationEpoch kCalibrationEpoch{1U};
constexpr core::SessionId kSession{1U};
constexpr core::ConfigRevision kConfigRevision{1U};
constexpr core::ClockRevision kClockRevision{1U};
constexpr core::SourceEpoch kFirstSourceEpoch{1U};
constexpr core::ProducerId kFirstProducer{1U};

[[nodiscard]] core::PipelineStage applicationPipelineStage(std::uint64_t id,
                                                           std::string_view name) {
  const auto stage = core::makePipelineStage(core::PipelineStageId{id}, name);
  if (!stage) {
    throw std::logic_error("invalid built-in bag localization pipeline stage");
  }
  return *stage;
}

[[nodiscard]] const core::PipelineStage& imuIngressStage() {
  static const core::PipelineStage stage =
      applicationPipelineStage(20'001U, "app.driver.imu_ingress");
  return stage;
}

[[nodiscard]] const core::PipelineStage& lidarEnqueueStage() {
  static const core::PipelineStage stage =
      applicationPipelineStage(20'002U, "app.driver.lidar_enqueue");
  return stage;
}

[[nodiscard]] const core::PipelineStage& cameraIngressStage() {
  static const core::PipelineStage stage =
      applicationPipelineStage(20'003U, "app.driver.camera_ingress");
  return stage;
}

[[nodiscard]] const core::PipelineStage& processReadyStage() {
  static const core::PipelineStage stage =
      applicationPipelineStage(20'004U, "app.driver.process_ready");
  return stage;
}

[[nodiscard]] const core::PipelineStage& reportOutputStage() {
  static const core::PipelineStage stage =
      applicationPipelineStage(20'005U, "app.driver.report_output");
  return stage;
}

[[nodiscard]] core::PipelineTimingSample driverTimingSample(const core::PipelineStage& stage,
                                                            const core::CpuWallDuration& duration,
                                                            core::PipelineDisposition disposition,
                                                            core::MeasurementId measurement) {
  core::PipelineTimingSample sample;
  sample.stage = stage;
  sample.wall_duration = duration.wall;
  sample.thread_cpu_duration = duration.thread_cpu;
  if (measurement.valid()) {
    sample.work.measurement = measurement;
  }
  sample.disposition = disposition;
  return sample;
}

[[nodiscard]] std::size_t nearestRankIndex(std::size_t count, double quantile) noexcept {
  const double rank = std::ceil(quantile * static_cast<double>(count));
  return static_cast<std::size_t>(std::max(1.0, rank)) - 1U;
}

class BoundedDurationAccumulator {
public:
  explicit BoundedDurationAccumulator(std::size_t capacity)
      : samples_(capacity), scratch_(capacity) {
    if (capacity == 0U) {
      throw std::invalid_argument("duration accumulator capacity must be nonzero");
    }
  }

  void observe(core::Duration duration) noexcept {
    if (duration.nanoseconds <= 0) {
      return;
    }
    if (!maximum_observed_ || duration.nanoseconds > maximum_observed_->nanoseconds) {
      maximum_observed_ = duration;
    }
    samples_[next_index_] = duration.nanoseconds;
    next_index_ = (next_index_ + 1U) % samples_.size();
    window_size_ = std::min(window_size_ + 1U, samples_.size());
    ++total_samples_;
  }

  [[nodiscard]] std::optional<core::Duration> maximumObserved() const noexcept {
    return maximum_observed_;
  }

  [[nodiscard]] core::DurationStatistics snapshot() const noexcept {
    core::DurationStatistics output;
    output.total_samples = total_samples_;
    output.window_samples = window_size_;
    output.window_capacity = samples_.size();
    if (window_size_ == 0U) {
      return output;
    }
    for (std::size_t index = 0U; index < window_size_; ++index) {
      scratch_[index] = samples_[index];
    }
    std::sort(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(window_size_));
    const auto durationAt = [&](std::size_t index) { return core::Duration{scratch_[index]}; };
    output.minimum = durationAt(0U);
    output.p50 = durationAt(nearestRankIndex(window_size_, 0.50));
    output.p95 = durationAt(nearestRankIndex(window_size_, 0.95));
    output.p99 = durationAt(nearestRankIndex(window_size_, 0.99));
    output.maximum = durationAt(window_size_ - 1U);
    long double sum = 0.0L;
    for (std::size_t index = 0U; index < window_size_; ++index) {
      sum += static_cast<long double>(scratch_[index]);
    }
    output.mean_nanoseconds = static_cast<double>(sum / static_cast<long double>(window_size_));
    return output;
  }

private:
  std::vector<std::int64_t> samples_;
  mutable std::vector<std::int64_t> scratch_;
  std::size_t next_index_{};
  std::size_t window_size_{};
  std::uint64_t total_samples_{};
  std::optional<core::Duration> maximum_observed_;
};

struct ImuContinuityReport {
  core::DurationStatistics newest_gap;
  std::optional<core::Duration> newest_gap_maximum_observed;
  std::size_t retained_samples_high_watermark{};
};

struct DriverTimingReport {
  core::PipelineTimingStatisticsSnapshot imu_ingress;
  core::PipelineTimingStatisticsSnapshot lidar_enqueue;
  core::PipelineTimingStatisticsSnapshot camera_ingress;
  core::PipelineTimingStatisticsSnapshot process_ready;
  core::PipelineTimingStatisticsSnapshot report_output;
  ImuContinuityReport imu_continuity;
  local::LocalPipelineTimingReport local_pipeline;
};

// The acceleration density and both random walks are the explicit NCD Kalibr
// inputs. Kalibr only echoes those assumptions; its 0.019 gyro density is far
// above the recorded residual floor. The 0.002 gyro density is the
// AlphaSense-class OKVIS2 value and remains conservatively above the adjacent-
// difference floor measured in both Quad bags. These are continuous-time
// densities and therefore are not rescaled for the bag's 200 Hz topic.
constexpr double kAccelerometerNoiseDensity = 0.019;
constexpr double kGyroscopeNoiseDensity = 0.002;
constexpr double kAccelerometerBiasRandomWalk = 0.0043;
// The NCD Kalibr value (2.66e-4 rad/s^2/sqrt(Hz)) was already shown by the v1
// full-sequence experiments to be unusable: it lets gyro bias absorb LiDAR
// mismatch during brisk motion and produces late-run heading divergence. Keep
// the validated rig value explicit here instead of silently re-importing the
// dataset field during application composition.
constexpr double kGyroscopeBiasRandomWalk = 4.0e-6;
constexpr double kNominalImuRateHz = 200.0;
constexpr double kGravityMps2 = 9.80665;

[[nodiscard]] constexpr bool sensorModeIncludesLidar(BagLocalizationSensorMode mode) noexcept {
  return mode != BagLocalizationSensorMode::Imu;
}

[[nodiscard]] constexpr bool sensorModeIncludesCameras(BagLocalizationSensorMode mode) noexcept {
  return mode == BagLocalizationSensorMode::Full;
}

[[nodiscard]] constexpr std::string_view sensorModeName(BagLocalizationSensorMode mode) noexcept {
  switch (mode) {
    case BagLocalizationSensorMode::Imu:
      return "imu";
    case BagLocalizationSensorMode::LidarImu:
      return "lidar-imu";
    case BagLocalizationSensorMode::Full:
      return "full";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view initializationModeName(
    BagLocalizationInitializationMode mode) noexcept {
  switch (mode) {
    case BagLocalizationInitializationMode::Dynamic:
      return "dynamic";
    case BagLocalizationInitializationMode::Static:
      return "static";
    case BagLocalizationInitializationMode::SupervisedAuto:
      return "supervised-auto";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view localInitializationModeName(
    local::InitializationMode mode) noexcept {
  switch (mode) {
    case local::InitializationMode::StaticOnly:
      return "static_only";
    case local::InitializationMode::DynamicOnly:
      return "dynamic_only";
    case local::InitializationMode::SupervisedAuto:
      return "supervised_auto";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view zeroMotionPriorSourceName(
    local::ZeroMotionPriorSource source) noexcept {
  switch (source) {
    case local::ZeroMotionPriorSource::Operator:
      return "operator";
    case local::ZeroMotionPriorSource::VehicleSupervisor:
      return "vehicle_supervisor";
    case local::ZeroMotionPriorSource::MissionScenario:
      return "mission_scenario";
  }
  return "unknown";
}

[[nodiscard]] bool effectiveVisualGraphEnabled(const BagLocalizationOptions& options) noexcept {
  return sensorModeIncludesCameras(options.sensor_mode) &&
         options.visual_graph_enabled.value_or(true);
}

[[nodiscard]] constexpr std::string_view replayModeName(BagLocalizationReplayMode mode) noexcept {
  switch (mode) {
    case BagLocalizationReplayMode::Scheduled:
      return "scheduled";
    case BagLocalizationReplayMode::Unpaced:
      return "unpaced";
  }
  return "unknown";
}

[[nodiscard]] constexpr bool replayIsUnpaced(BagLocalizationReplayMode mode) noexcept {
  return mode == BagLocalizationReplayMode::Unpaced;
}

struct CameraRunCounters {
  std::size_t events{};
  std::size_t ingest_accepted{};
  std::size_t ingest_rejected{};
  std::size_t late_for_graph{};
  std::size_t imu_rotation_seeded{};
  std::size_t tracking_only{};
  std::size_t tracking_only_graph_submission_disabled{};
  std::size_t tracking_only_without_local_state{};
  std::size_t keyframe_requests{};
  std::size_t keyframes_suppressed_by_capacity{};
  std::size_t keyframes_suppressed_by_timeline{};
  std::size_t keyframes_rejected_by_timeline{};
  std::size_t graph_degradations{};
};

struct SolverGlobalizationRunCounters {
  std::size_t transactions{};
  std::size_t full_steps_rejected{};
  std::size_t backtracking_trials{};
  std::size_t cauchy_directions_attempted{};
  std::size_t cauchy_steps_accepted{};
  std::size_t cauchy_backtracking_trials{};
  std::size_t zero_step_terminations{};
  double minimum_step_scale{1.0};
};

struct FinalizedLidarTargetRunCounters {
  std::size_t process_reports{};
  std::size_t finality_matches{};
  std::size_t rollback_removals{};
  std::size_t insertion_transactions{};
  std::size_t insertion_input_points{};
  std::size_t insertion_voxels{};
  std::size_t insertion_selection_discarded_points{};
  std::size_t minimum_separation_discarded_points{};
  std::size_t query_voxel_capacity_discarded_points{};
  std::size_t inserted_points{};
  std::size_t touched_query_voxels{};
  std::size_t prune_reports{};
  std::size_t prune_examined_query_voxels{};
  std::size_t prune_examined_points{};
  std::size_t pruned_query_voxels{};
  std::size_t pruned_points{};
  std::size_t capacity_recovery_attempts{};
  std::size_t capacity_recovery_successes{};
  std::size_t capacity_skipped_sweeps{};
  std::size_t capacity_retry_suppressed_sweeps{};
  std::optional<local::FinalizedLidarTargetCapacitySkip> last_capacity_skip;
  std::size_t frozen_process_reports{};
  std::size_t freeze_transitions{};
  std::size_t pending_sweeps{};
  std::size_t pending_unfinalized_sweeps{};
  std::size_t finalized_ready_sweeps{};
  bool insertion_frozen{};
  bool capacity_saturated{};
  std::size_t capacity_skips_since_retry{};
  bool observed_freeze_state{};
  core::SensorHealthState lidar_health{core::SensorHealthState::Failed};
  std::size_t retained_points{};
  std::uint64_t map_version{};
  core::ContentHash map_checksum{};
};

void observeSolverGlobalization(const local::LocalSolveReport& solve,
                                SolverGlobalizationRunCounters* counters) {
  ++counters->transactions;
  counters->full_steps_rejected += solve.nonlinear_full_steps_rejected;
  counters->backtracking_trials += solve.nonlinear_backtracking_trials;
  counters->cauchy_directions_attempted += solve.nonlinear_cauchy_directions_attempted;
  counters->cauchy_steps_accepted += solve.nonlinear_cauchy_steps_accepted;
  counters->cauchy_backtracking_trials += solve.nonlinear_cauchy_backtracking_trials;
  counters->zero_step_terminations += solve.nonlinear_zero_step_terminations;
  counters->minimum_step_scale =
      std::min(counters->minimum_step_scale, solve.minimum_nonlinear_step_scale);
}

struct RunCounters {
  std::size_t events_received{};
  std::size_t imu_events{};
  std::size_t lidar_events{};
  std::size_t camera_events{};
  std::size_t unexpected_events{};
  std::size_t imu_ingest_accepted{};
  std::size_t imu_ingest_rejected{};
  std::size_t lidar_enqueue_accepted{};
  std::size_t lidar_enqueue_rejected{};
  std::size_t camera_ingest_accepted{};
  std::size_t camera_ingest_rejected{};
  std::size_t camera_frames_late_for_graph{};
  std::size_t camera_rotation_seeds_provided{};
  std::size_t camera_tracking_only{};
  std::size_t camera_tracking_only_graph_submission_disabled{};
  std::size_t camera_tracking_only_without_local_state{};
  std::size_t camera_keyframe_requests{};
  std::size_t camera_keyframes_suppressed_by_capacity{};
  std::size_t camera_keyframes_suppressed_by_timeline{};
  std::size_t camera_keyframes_rejected_by_timeline{};
  std::size_t process_calls{};
  std::size_t process_errors{};
  std::size_t initialization_rejections_reported{};
  std::size_t stationary_initializations{};
  std::size_t motion_initializations{};
  std::size_t initialization_bootstrap_anchors{};
  std::size_t initialization_bootstrap_increments{};
  std::size_t initialization_bootstrap_rejections{};
  std::size_t lidar_drops_reported{};
  std::size_t bootstrap_commits{};
  std::size_t registered_commits{};
  std::size_t registered_tracking_only_commits{};
  std::size_t deskew_degraded_commits{};
  std::size_t preprocessing_degraded_commits{};
  std::size_t registration_degraded_commits{};
  std::size_t health_quarantined_commits{};
  std::size_t target_state_unavailable_frozen_commits{};
  std::size_t registration_diagnostics{};
  std::size_t active_registration_diagnostics{};
  std::size_t lidar_registrations_attempted{};
  std::size_t lidar_registrations_accepted{};
  std::size_t lidar_registrations_rejected{};
  std::size_t lidar_registration_outer_iterations{};
  std::size_t lidar_registration_accepted_steps{};
  std::size_t lidar_registration_rejected_trials{};
  std::size_t accepted_registration_direct_graph_rejections{};
  std::size_t accepted_registration_health_quarantines{};
  std::size_t lidar_health_updates{};
  std::size_t lidar_health_transitions{};
  std::size_t lidar_factor_batches_removed{};
  SolverGlobalizationRunCounters committed_solver_globalization;
  SolverGlobalizationRunCounters rejected_solver_globalization;
  std::size_t waiting_for_future_imu_reports{};
  std::size_t pending_sweeps_at_end{};
  std::size_t pending_camera_knots_at_end{};
  std::size_t camera_commit_reports{};
  std::size_t visual_keyframes_resolved{};
  std::size_t visual_only_commits{};
  std::size_t shared_camera_lidar_commits{};
  std::size_t commit_references_deduplicated{};
  std::size_t factor_only_lidar_reports{};
  std::size_t factor_only_graph_revisions{};
  std::size_t factor_only_unchanged_graph_references{};
  std::size_t visual_graph_attachments{};
  std::size_t visual_landmarks_attached{};
  std::size_t visual_factors_attached{};
  std::size_t visual_factors_retired{};
  std::size_t visual_graph_degradations{};
  std::size_t visual_factor_batches_discarded{};
  std::size_t visual_landmark_seeds_discarded{};
  std::size_t visual_factor_specs_discarded{};
  std::size_t visual_stale_track_observations_discarded{};
  std::optional<int> last_visual_degradation_graph_error_code;
  std::string last_visual_degradation_detail;
  std::optional<int> last_initialization_rejection_stage;
  std::optional<int> last_initialization_rejection_code;
  std::string last_initialization_rejection_detail;
  std::optional<int> last_stationary_rejection_code;
  std::string last_stationary_rejection_detail;
  std::optional<int> last_bootstrap_rejection_code;
  std::string last_bootstrap_rejection_detail;
  std::optional<int> last_motion_rejection_code;
  std::optional<std::size_t> last_motion_rejection_segment;
  std::string last_motion_rejection_detail;
  std::optional<int> last_registration_rejection_code;
  std::string last_registration_rejection_detail;
  std::map<int, std::size_t> registration_degradation_codes;
  std::map<local::LidarRegistrationTermination, std::size_t> lidar_registration_terminations;
  std::map<local::LidarRegistrationErrorCode, std::size_t> lidar_registration_errors;
  std::optional<local::DirectLidarRegistrationReport> last_lidar_registration;
  std::optional<local::LidarRegistrationError> last_lidar_registration_error;
  std::size_t accepted_finalized_map_reports{};
  std::size_t rejected_finalized_map_reports{};
  std::optional<local::DirectLidarFinalizedMapReport> last_accepted_finalized_map;
  std::optional<local::DirectLidarFinalizedMapReport> last_rejected_finalized_map;
  std::optional<local::SensorHealthUpdate> last_lidar_health_update;
  std::optional<local::MotionInitializationDiagnostics> motion_initialization_diagnostics;
  FinalizedLidarTargetRunCounters finalized_lidar_target;
  std::map<std::uint64_t, CameraRunCounters> cameras;
};

struct SensorModeAudit {
  bool calibration_available{};
  bool profile_available{};
  bool estimator_config_available{};
  std::size_t expected_imu_topics{1U};
  std::size_t expected_lidar_topics{};
  std::size_t expected_camera_topics{};
  std::size_t configured_imu_topics{};
  std::size_t configured_lidar_topics{};
  std::size_t configured_camera_topics{};
  std::size_t configured_gnss_topics{};
  std::size_t configured_visual_lanes{};
  std::size_t camera_topics_with_runtime_stats{};
  std::size_t camera_messages_seen{};
  std::size_t camera_image_decode_errors{};
  std::size_t camera_events_emitted{};
  std::size_t lidar_topics_with_runtime_stats{};
  std::size_t lidar_messages_seen{};
  std::size_t lidar_events_emitted{};
  bool storage_filter_matches{};
  bool configuration_matches{};
  bool runtime_isolation_matches{};
  bool passed{};
};

struct StateCoverage {
  std::size_t states_written{};
  std::optional<core::StateId> first_state;
  std::optional<core::StateId> last_state;
  std::optional<core::FusionTime> first_time;
  std::optional<core::FusionTime> last_time;
};

struct Failure {
  std::string stage;
  std::string detail;
  std::optional<int> code;
  std::optional<int> substage;
};

[[nodiscard]] std::string jsonString(std::string_view input) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : input) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

[[nodiscard]] std::string timestampString(core::FusionTime time) {
  constexpr std::int64_t billion = 1'000'000'000LL;
  std::int64_t seconds = time.nanoseconds / billion;
  std::int64_t remainder = time.nanoseconds % billion;
  if (remainder < 0) {
    --seconds;
    remainder += billion;
  }
  std::ostringstream output;
  output << seconds << '.' << std::setw(9) << std::setfill('0') << remainder;
  return output.str();
}

void writePoseMatrix(std::ostream& output, const core::Pose3d& pose) {
  const Eigen::Matrix4d matrix = pose.matrix();
  output << '[';
  for (Eigen::Index row = 0; row < 4; ++row) {
    output << (row == 0 ? "[" : ",[ ");
    for (Eigen::Index column = 0; column < 4; ++column) {
      output << (column == 0 ? "" : ",") << matrix(row, column);
    }
    output << ']';
  }
  output << ']';
}

[[nodiscard]] bool writeTumState(std::ofstream* stream, core::FusionTime time,
                                 const core::Pose3d& pose) {
  const Eigen::Quaterniond quaternion = pose.unit_quaternion();
  *stream << timestampString(time) << ' ' << std::setprecision(17) << pose.translation().x() << ' '
          << pose.translation().y() << ' ' << pose.translation().z() << ' ' << quaternion.x() << ' '
          << quaternion.y() << ' ' << quaternion.z() << ' ' << quaternion.w() << '\n';
  return stream->good();
}

void writeOptionalCsv(std::ostream& output, const std::optional<double>& value) {
  if (value) {
    output << *value;
  }
}

void writeEmptyCsvFields(std::ostream& output, std::size_t count) {
  for (std::size_t index = 0U; index < count; ++index) {
    output << ',';
  }
}

void writePoseCsvFields(std::ostream& output, const core::Pose3d& pose) {
  const Eigen::Quaterniond quaternion = pose.unit_quaternion();
  output << ',' << pose.translation().x() << ',' << pose.translation().y() << ','
         << pose.translation().z() << ',' << quaternion.x() << ',' << quaternion.y() << ','
         << quaternion.z() << ',' << quaternion.w();
}

void writePoseJson(std::ostream& output, const core::Pose3d& pose) {
  const Eigen::Quaterniond quaternion = pose.unit_quaternion();
  output << std::setprecision(17) << "{\"translation_m\": [" << pose.translation().x() << ", "
         << pose.translation().y() << ", " << pose.translation().z() << "], \"quaternion_xyzw\": ["
         << quaternion.x() << ", " << quaternion.y() << ", " << quaternion.z() << ", "
         << quaternion.w() << "]}";
}

[[nodiscard]] std::string_view motionInitializationPassName(
    local::MotionInitializationRequest::Pass pass) {
  switch (pass) {
    case local::MotionInitializationRequest::Pass::RotationDeskewProposal:
      return "rotation_deskew_proposal";
    case local::MotionInitializationRequest::Pass::FullDeskewCommitCandidate:
      return "full_deskew_commit_candidate";
  }
  return "unknown";
}

[[nodiscard]] std::string_view motionInitializationObservabilityClassName(
    local::MotionInitializationObservabilityClass observability_class) {
  switch (observability_class) {
    case local::MotionInitializationObservabilityClass::SensorObservable:
      return "sensor_observable";
    case local::MotionInitializationObservabilityClass::PriorResolvedAccelerometerTilt:
      return "prior_resolved_accelerometer_tilt";
  }
  return "unknown";
}

[[nodiscard]] std::string_view lidarRegistrationTerminationName(
    local::LidarRegistrationTermination termination) {
  switch (termination) {
    case local::LidarRegistrationTermination::Converged:
      return "converged";
    case local::LidarRegistrationTermination::IterationLimitReached:
      return "iteration_limit_reached";
  }
  return "unknown";
}

[[nodiscard]] std::string_view lidarRegistrationErrorCodeName(
    local::LidarRegistrationErrorCode code) {
  switch (code) {
    case local::LidarRegistrationErrorCode::InvalidConfig:
      return "invalid_config";
    case local::LidarRegistrationErrorCode::InvalidSource:
      return "invalid_source";
    case local::LidarRegistrationErrorCode::InvalidTarget:
      return "invalid_target";
    case local::LidarRegistrationErrorCode::InconsistentSeed:
      return "inconsistent_seed";
    case local::LidarRegistrationErrorCode::InsufficientCorrespondences:
      return "insufficient_correspondences";
    case local::LidarRegistrationErrorCode::InsufficientObservableRank:
      return "insufficient_observable_rank";
    case local::LidarRegistrationErrorCode::NoDecreasingStep:
      return "no_decreasing_step";
    case local::LidarRegistrationErrorCode::NumericalFailure:
      return "numerical_failure";
  }
  return "unknown";
}

[[nodiscard]] std::string_view sensorHealthStateName(core::SensorHealthState state) {
  switch (state) {
    case core::SensorHealthState::Active:
      return "active";
    case core::SensorHealthState::Suspect:
      return "suspect";
    case core::SensorHealthState::Failed:
      return "failed";
    case core::SensorHealthState::Recovering:
      return "recovering";
  }
  return "unknown";
}

[[nodiscard]] std::string_view sensorModalityName(core::SensorModality modality) {
  switch (modality) {
    case core::SensorModality::Lidar:
      return "lidar";
    case core::SensorModality::Visual:
      return "visual";
    case core::SensorModality::Gnss:
      return "gnss";
  }
  return "unknown";
}

[[nodiscard]] std::string_view finalizedLidarTargetCapacitySkipReasonName(
    local::FinalizedLidarTargetCapacitySkipReason reason) {
  switch (reason) {
    case local::FinalizedLidarTargetCapacitySkipReason::RetryAfterPruneStillFull:
      return "retry_after_prune_still_full";
    case local::FinalizedLidarTargetCapacitySkipReason::RetrySuppressedWhileSaturated:
      return "retry_suppressed_while_saturated";
  }
  return "unknown";
}

void writeCsvString(std::ostream& output, std::string_view value) {
  output << '"';
  for (const char character : value) {
    if (character == '"') {
      output << "\"\"";
    } else {
      output << character;
    }
  }
  output << '"';
}

void writeNavigationDiagnosticsHeader(std::ostream& output) {
  output << "timestamp_ns,state_id,graph_revision,parent_revision,commit_kind,"
            "navigation_state_created,"
            "position_x_m,position_y_m,position_z_m,quaternion_x,quaternion_y,quaternion_z,"
            "quaternion_w,velocity_x_mps,velocity_y_mps,velocity_z_mps,gyro_bias_x_radps,"
            "gyro_bias_y_radps,gyro_bias_z_radps,accel_bias_x_mps2,accel_bias_y_mps2,"
            "accel_bias_z_mps2,cov_rotation_x,cov_rotation_y,cov_rotation_z,cov_velocity_x,"
            "cov_velocity_y,cov_velocity_z,cov_position_x,cov_position_y,cov_position_z,"
            "cov_gyro_bias_x,cov_gyro_bias_y,cov_gyro_bias_z,cov_accel_bias_x,"
            "cov_accel_bias_y,cov_accel_bias_z,navigation_states,combined_imu_factors,"
            "active_lidar_direct_batch_factors,active_visual_landmarks,active_visual_factors,"
            "marginalized_navigation_states,nonlinear_iterations,"
            "convergence_interval_duration_s,convergence_sigma_fraction,"
            "effective_translation_convergence_m,effective_rotation_convergence_rad,"
            "effective_velocity_convergence_mps,"
            "effective_accelerometer_bias_convergence_mps2,"
            "effective_gyroscope_bias_convergence_radps,"
            "effective_visual_log_inverse_range_convergence,"
            "maximum_iteration_translation_correction_m,"
            "maximum_iteration_rotation_correction_rad,"
            "maximum_transaction_translation_correction_m,"
            "maximum_transaction_rotation_correction_rad,"
            "marginalization_translation_correction_m,"
            "marginalization_rotation_correction_rad,nonlinear_full_steps_rejected,"
            "nonlinear_backtracking_trials,nonlinear_cauchy_directions_attempted,"
            "nonlinear_cauchy_steps_accepted,nonlinear_cauchy_backtracking_trials,"
            "nonlinear_zero_step_terminations,"
            "minimum_nonlinear_step_scale,last_iteration_objective_change,"
            "variables_relinearized,"
            "variables_reeliminated,factors_recalculated,cliques,error_before,error_after,"
            "lidar_registration_present,lidar_registration_termination,"
            "lidar_registration_initial_robust_cost,lidar_registration_final_robust_cost,"
            "lidar_registration_target_count,lidar_registration_live_target_count,"
            "lidar_registration_finalized_map_target_count,"
            "lidar_registration_correspondences,lidar_registration_live_correspondences,"
            "lidar_registration_finalized_map_correspondences,"
            "lidar_registration_overlap_fraction,"
            "lidar_registration_effective_correspondences,"
            "lidar_registration_maximum_squared_residual_m2,"
            "lidar_registration_huber_delta_m,"
            "lidar_registration_characteristic_length_m,"
            "lidar_registration_normalized_observable_eigenvalue_threshold,"
            "lidar_registration_observable_rank,"
            "lidar_registration_geometric_information_scale,"
            "lidar_registration_source_points_considered,"
            "lidar_registration_source_points_selected,"
            "lidar_registration_source_points_omitted_by_capacity,"
            "lidar_registration_invalid_source_points,"
            "lidar_registration_invalid_target_points,"
            "lidar_registration_target_index_voxels,"
            "lidar_registration_composite_index_builds,"
            "lidar_registration_composite_index_input_owners,"
            "lidar_registration_composite_index_input_points,"
            "lidar_registration_composite_index_retained_owners,"
            "lidar_registration_composite_index_retained_points,"
            "lidar_registration_composite_index_occupied_voxels,"
            "lidar_registration_composite_index_per_voxel_discarded_points,"
            "lidar_registration_composite_index_total_discarded_points,"
            "lidar_registration_composite_index_build_wall_ns,"
            "lidar_registration_composite_index_build_thread_cpu_ns,"
            "lidar_registration_association_builds,"
            "lidar_registration_candidate_voxel_lookups,"
            "lidar_registration_candidate_points_examined,"
            "lidar_registration_finalized_map_candidate_voxel_lookups,"
            "lidar_registration_finalized_map_candidate_occupied_voxels,"
            "lidar_registration_finalized_map_candidate_points_examined,"
            "lidar_registration_finalized_map_stale_fallbacks,"
            "lidar_registration_frozen_objective_evaluations,"
            "lidar_registration_normal_equation_evaluations,"
            "lidar_registration_outer_iterations,lidar_registration_gauss_newton_trials,"
            "lidar_registration_lm_damping_trials,"
            "lidar_registration_rejected_frozen_cost_trials,"
            "lidar_registration_accepted_steps,"
            "lidar_registration_T_odom_source_x_m,lidar_registration_T_odom_source_y_m,"
            "lidar_registration_T_odom_source_z_m,lidar_registration_T_odom_source_qx,"
            "lidar_registration_T_odom_source_qy,lidar_registration_T_odom_source_qz,"
            "lidar_registration_T_odom_source_qw,"
            "lidar_registration_source_right_correction_x_m,"
            "lidar_registration_source_right_correction_y_m,"
            "lidar_registration_source_right_correction_z_m,"
            "lidar_registration_source_right_correction_qx,"
            "lidar_registration_source_right_correction_qy,"
            "lidar_registration_source_right_correction_qz,"
            "lidar_registration_source_right_correction_qw,"
            "lidar_registration_normalized_information_eigenvalue_0,"
            "lidar_registration_normalized_information_eigenvalue_1,"
            "lidar_registration_normalized_information_eigenvalue_2,"
            "lidar_registration_normalized_information_eigenvalue_3,"
            "lidar_registration_normalized_information_eigenvalue_4,"
            "lidar_registration_normalized_information_eigenvalue_5,"
            "lidar_registration_physical_information_eigenvalue_0,"
            "lidar_registration_physical_information_eigenvalue_1,"
            "lidar_registration_physical_information_eigenvalue_2,"
            "lidar_registration_physical_information_eigenvalue_3,"
            "lidar_registration_physical_information_eigenvalue_4,"
            "lidar_registration_physical_information_eigenvalue_5,"
            "lidar_registration_error_code,lidar_registration_error_detail,"
            "lidar_health_update_present,lidar_health_batch_id,lidar_health_sensor_instance,"
            "lidar_health_before_state,lidar_health_before_recovery_epoch,"
            "lidar_health_before_transition_sequence,lidar_health_before_assessed_at_ns,"
            "lidar_health_after_state,lidar_health_after_recovery_epoch,"
            "lidar_health_after_transition_sequence,lidar_health_after_assessed_at_ns,"
            "lidar_health_transitioned,lidar_health_consecutive_failures,"
            "lidar_health_recovery_good_shadow_results,lidar_removed_factor_batches,"
            "lidar_batch_pairs,lidar_batch_finalized_map_factors,lidar_batch_target_factors,"
            "lidar_batch_total_correspondences,lidar_batch_excluded_source_rows,"
            "lidar_batch_live_information_scale,"
            "lidar_batch_finalized_map_information_scale,"
            "lidar_batch_correlation_present,lidar_batch_correlation_policy_revision,"
            "lidar_batch_covariance_inflation,lidar_batch_declared_information_cap,"
            "lidar_batch_conditioning_measurement_roots,"
            "accepted_finalized_map_present,accepted_finalized_map_source_sweep,"
            "accepted_finalized_map_source_point_count,accepted_finalized_map_correspondences,"
            "accepted_finalized_map_excluded_source_rows,"
            "accepted_finalized_map_candidate_voxel_lookups,"
            "accepted_finalized_map_candidate_points_examined,"
            "accepted_finalized_map_unique_owners,accepted_finalized_map_odom_epoch,"
            "accepted_finalized_map_sensor_modality,accepted_finalized_map_sensor_instance,"
            "accepted_finalized_map_version,accepted_finalized_map_checksum,"
            "accepted_finalized_map_owner_pose_covariance_inflation,"
            "accepted_finalized_map_configured_correlation_inflation_floor,"
            "accepted_finalized_map_effective_covariance_inflation,"
            "accepted_finalized_map_information_scale,accepted_finalized_map_information_rank,"
            "accepted_finalized_map_information_eigenvalue_0,"
            "accepted_finalized_map_information_eigenvalue_1,"
            "accepted_finalized_map_information_eigenvalue_2,"
            "accepted_finalized_map_information_eigenvalue_3,"
            "accepted_finalized_map_information_eigenvalue_4,"
            "accepted_finalized_map_information_eigenvalue_5,"
            "accepted_finalized_map_snapshot_checksum,"
            "rejected_finalized_map_present,rejected_finalized_map_source_sweep,"
            "rejected_finalized_map_source_point_count,rejected_finalized_map_correspondences,"
            "rejected_finalized_map_excluded_source_rows,"
            "rejected_finalized_map_candidate_voxel_lookups,"
            "rejected_finalized_map_candidate_points_examined,"
            "rejected_finalized_map_unique_owners,rejected_finalized_map_odom_epoch,"
            "rejected_finalized_map_sensor_modality,rejected_finalized_map_sensor_instance,"
            "rejected_finalized_map_version,rejected_finalized_map_checksum,"
            "rejected_finalized_map_owner_pose_covariance_inflation,"
            "rejected_finalized_map_configured_correlation_inflation_floor,"
            "rejected_finalized_map_effective_covariance_inflation,"
            "rejected_finalized_map_information_scale,rejected_finalized_map_information_rank,"
            "rejected_finalized_map_information_eigenvalue_0,"
            "rejected_finalized_map_information_eigenvalue_1,"
            "rejected_finalized_map_information_eigenvalue_2,"
            "rejected_finalized_map_information_eigenvalue_3,"
            "rejected_finalized_map_information_eigenvalue_4,"
            "rejected_finalized_map_information_eigenvalue_5,"
            "rejected_finalized_map_snapshot_checksum,"
            "rejected_nonlinear_iterations,"
            "rejected_convergence_interval_duration_s,rejected_convergence_sigma_fraction,"
            "rejected_effective_translation_convergence_m,"
            "rejected_effective_rotation_convergence_rad,"
            "rejected_effective_velocity_convergence_mps,"
            "rejected_effective_accelerometer_bias_convergence_mps2,"
            "rejected_effective_gyroscope_bias_convergence_radps,"
            "rejected_effective_visual_log_inverse_range_convergence,"
            "rejected_last_translation_correction_m,"
            "rejected_last_rotation_correction_rad,"
            "rejected_last_velocity_correction_mps,"
            "rejected_last_accelerometer_bias_correction_mps2,"
            "rejected_last_gyroscope_bias_correction_radps,"
            "rejected_last_visual_log_inverse_range_correction,"
            "rejected_maximum_transaction_translation_correction_m,"
            "rejected_maximum_transaction_rotation_correction_rad,"
            "rejected_nonlinear_full_steps,rejected_nonlinear_backtracking_trials,"
            "rejected_nonlinear_cauchy_directions_attempted,"
            "rejected_nonlinear_cauchy_steps_accepted,"
            "rejected_nonlinear_cauchy_backtracking_trials,"
            "rejected_nonlinear_zero_step_terminations,"
            "rejected_minimum_nonlinear_step_scale,rejected_last_iteration_objective_change,"
            "rejected_error_before,rejected_error_after,"
            "degradation_graph_error_code,degradation_detail,"
            "motion_segments,motion_imu_knots,motion_support_ns,motion_rotation_excitation_rad,"
            "motion_acceleration_excitation_mps2,motion_scalar_dimension,"
            "motion_expected_data_rank,motion_data_rank,motion_calibrated_data_rank,"
            "motion_full_rank,motion_prior_resolved_accel_tilt_modes,"
            "motion_observability_class,motion_data_supported_condition,"
            "motion_calibrated_data_supported_condition,motion_full_hessian_condition,"
            "motion_initial_error,motion_final_error,motion_lidar_error,motion_imu_error,"
            "motion_bias_prior_error,motion_gauge_error,motion_lidar_residual_dimension,"
            "motion_imu_residual_dimension,motion_bias_prior_residual_dimension,"
            "motion_gauge_residual_dimension,motion_total_residual_dimension,"
            "motion_effective_degrees_of_freedom,motion_lidar_msw,motion_imu_msw,"
            "motion_reduced_chi_square,motion_holdout_lidar_segments,"
            "motion_holdout_residual_dimension,motion_holdout_error,motion_holdout_msw,"
            "motion_statistically_compatible,motion_conditioned_lidar_imu_approximation,"
            "motion_minimum_imu_conditioning_covariance_inflation,"
            "motion_covariance_residual_inflation,motion_deskew_solve_passes,"
            "motion_refined_sweeps,motion_refined_registrations,"
            "motion_refined_deskew_pose_interpolations,motion_refined_total_registration_cost,"
            "motion_refined_maximum_registration_cost,motion_bias_prior_mahalanobis,"
            "motion_solver_iterations,motion_pass\n";
}

void writeFinalizedMapCsvFields(std::ostream& output,
                                const local::DirectLidarFinalizedMapReport* report) {
  output << ',' << (report != nullptr ? 1 : 0);
  if (report == nullptr) {
    // Twenty-four fields follow the explicit presence bit.
    writeEmptyCsvFields(output, 24U);
    return;
  }
  output << ',' << report->source_sweep.value() << ',' << report->source_point_count << ','
         << report->correspondences << ',' << report->source_rows_excluded_by_ownership << ','
         << report->candidate_voxel_lookups << ',' << report->candidate_points_examined << ','
         << report->unique_finalized_owners << ',' << report->map_odom_epoch.value() << ','
         << static_cast<int>(report->map_sensor.modality) << ',' << report->map_sensor.instance
         << ',' << report->map_version << ',';
  writeCsvString(output, core::sha256Hex(report->map_checksum));
  output << ',' << report->owner_pose_covariance_inflation << ','
         << report->configured_correlation_inflation_floor << ','
         << report->effective_covariance_inflation << ',' << report->information_scale << ','
         << report->physical_information.rank;
  for (Eigen::Index index = 0; index < report->physical_information.eigenvalues.rows(); ++index) {
    output << ',' << report->physical_information.eigenvalues(index);
  }
  output << ',';
  writeCsvString(output, core::sha256Hex(report->snapshot_checksum));
}

[[nodiscard]] bool writeNavigationDiagnostics(
    std::ofstream* output, const local::LocalGraphCommit& commit, std::string_view commit_kind,
    bool navigation_state_created, const local::DirectLidarRegistrationReport* registration,
    const local::LidarRegistrationError* registration_error,
    const local::LocalSolveReport* rejected_solve,
    const std::optional<local::LocalGraphErrorCode>& degradation_graph_error_code,
    std::string_view degradation_detail, const local::MotionInitializationDiagnostics* motion,
    const local::SensorHealthUpdate* health_update, std::size_t removed_factor_batches,
    const local::DirectLidarFinalizedMapReport* rejected_finalized_map) {
  const auto& state = commit.estimate;
  const auto& pose = state.T_odom_imu;
  const Eigen::Quaterniond quaternion = pose.unit_quaternion();
  *output << std::setprecision(17) << commit.state_time.nanoseconds << ',' << commit.state.value()
          << ',' << commit.revision.value() << ',' << commit.parent.value() << ',' << commit_kind
          << ',' << (navigation_state_created ? 1 : 0) << ',' << pose.translation().x() << ','
          << pose.translation().y() << ',' << pose.translation().z() << ',' << quaternion.x() << ','
          << quaternion.y() << ',' << quaternion.z() << ',' << quaternion.w() << ','
          << state.velocity_odom.x() << ',' << state.velocity_odom.y() << ','
          << state.velocity_odom.z() << ',' << state.gyro_bias.x() << ',' << state.gyro_bias.y()
          << ',' << state.gyro_bias.z() << ',' << state.accel_bias.x() << ','
          << state.accel_bias.y() << ',' << state.accel_bias.z();
  for (Eigen::Index index = 0; index < commit.covariance.matrix.rows(); ++index) {
    *output << ',' << commit.covariance.matrix(index, index);
  }
  const auto& solve = commit.solve;
  *output << ',' << solve.navigation_states << ',' << solve.combined_imu_factors << ','
          << solve.active_lidar_direct_batch_factors << ',' << solve.active_visual_landmarks << ','
          << solve.active_visual_factors << ',' << solve.marginalized_navigation_states << ','
          << solve.nonlinear_iterations << ',' << solve.convergence_interval_duration_s << ','
          << solve.convergence_sigma_fraction << ',' << solve.effective_translation_convergence_m
          << ',' << solve.effective_rotation_convergence_rad << ','
          << solve.effective_velocity_convergence_mps << ','
          << solve.effective_accelerometer_bias_convergence_mps2 << ','
          << solve.effective_gyroscope_bias_convergence_radps << ','
          << solve.effective_visual_log_inverse_range_convergence << ','
          << solve.maximum_iteration_translation_correction_m << ','
          << solve.maximum_iteration_rotation_correction_rad << ','
          << solve.maximum_transaction_translation_correction_m << ','
          << solve.maximum_transaction_rotation_correction_rad << ','
          << solve.marginalization_translation_correction_m << ','
          << solve.marginalization_rotation_correction_rad << ','
          << solve.nonlinear_full_steps_rejected << ',' << solve.nonlinear_backtracking_trials
          << ',' << solve.nonlinear_cauchy_directions_attempted << ','
          << solve.nonlinear_cauchy_steps_accepted << ','
          << solve.nonlinear_cauchy_backtracking_trials << ','
          << solve.nonlinear_zero_step_terminations << ',' << solve.minimum_nonlinear_step_scale
          << ',' << solve.last_iteration_objective_change << ',' << solve.variables_relinearized
          << ',' << solve.variables_reeliminated << ',' << solve.factors_recalculated << ','
          << solve.cliques << ',';
  writeOptionalCsv(*output, solve.error_before);
  *output << ',';
  writeOptionalCsv(*output, solve.error_after);
  *output << ',' << (registration != nullptr ? 1 : 0);
  if (registration != nullptr) {
    const local::LidarRegistrationDiagnostics& diagnostics = registration->diagnostics;
    const local::LidarRegistrationWorkCounters& work = registration->work;
    *output << ',' << lidarRegistrationTerminationName(registration->termination) << ','
            << registration->initial_robust_cost << ',' << registration->final_robust_cost << ','
            << diagnostics.target_count << ',' << diagnostics.live_target_count << ','
            << diagnostics.finalized_map_target_count << ',' << diagnostics.correspondences << ','
            << diagnostics.live_correspondences << ',' << diagnostics.finalized_map_correspondences
            << ',' << diagnostics.overlap_fraction << ',' << diagnostics.effective_correspondences
            << ',' << diagnostics.maximum_squared_residual_m2 << ',' << diagnostics.huber_delta_m
            << ',' << diagnostics.characteristic_length_m << ','
            << diagnostics.normalized_observable_eigenvalue_threshold << ','
            << diagnostics.observable_rank << ',' << diagnostics.geometric_information_scale << ','
            << work.source_points_considered << ',' << work.source_points_selected << ','
            << work.source_points_omitted_by_capacity << ',' << work.invalid_source_points << ','
            << work.invalid_target_points << ',' << work.target_index_voxels << ','
            << work.composite_index_builds << ',' << work.composite_index_input_owners << ','
            << work.composite_index_input_points << ',' << work.composite_index_retained_owners
            << ',' << work.composite_index_retained_points << ','
            << work.composite_index_occupied_voxels << ','
            << work.composite_index_per_voxel_discarded_points << ','
            << work.composite_index_total_discarded_points << ','
            << work.composite_index_build_duration.wall.nanoseconds << ',';
    if (work.composite_index_build_duration.thread_cpu) {
      *output << work.composite_index_build_duration.thread_cpu->nanoseconds;
    }
    *output << ',' << work.association_builds << ',' << work.candidate_voxel_lookups << ','
            << work.candidate_points_examined << ',' << work.finalized_map_candidate_voxel_lookups
            << ',' << work.finalized_map_candidate_occupied_voxels << ','
            << work.finalized_map_candidate_points_examined << ','
            << work.finalized_map_stale_fallbacks << ',' << work.frozen_objective_evaluations << ','
            << work.normal_equation_evaluations << ',' << work.outer_iterations << ','
            << work.gauss_newton_trials << ',' << work.lm_damping_trials << ','
            << work.rejected_frozen_cost_trials << ',' << work.accepted_steps;
    writePoseCsvFields(*output, registration->T_odom_source);
    writePoseCsvFields(*output, registration->source_right_correction);
    for (Eigen::Index index = 0;
         index < diagnostics.normalized_directional_information.eigenvalues.rows(); ++index) {
      *output << ',' << diagnostics.normalized_directional_information.eigenvalues(index);
    }
    for (Eigen::Index index = 0; index < diagnostics.physical_information.eigenvalues.rows();
         ++index) {
      *output << ',' << diagnostics.physical_information.eigenvalues(index);
    }
  } else {
    // Seventy-three registration fields after the explicit presence bit.
    writeEmptyCsvFields(*output, 73U);
  }
  *output << ',';
  if (registration_error != nullptr) {
    *output << lidarRegistrationErrorCodeName(registration_error->code);
  }
  *output << ',';
  writeCsvString(*output, registration_error != nullptr ? registration_error->detail : "");
  *output << ',' << (health_update != nullptr ? 1 : 0);
  if (health_update != nullptr) {
    *output << ',' << health_update->batch_id.value() << ',' << health_update->after.sensor.instance
            << ',' << sensorHealthStateName(health_update->before.state) << ','
            << health_update->before.recovery_epoch.value() << ','
            << health_update->before.transition_sequence << ','
            << health_update->before.assessed_at.nanoseconds << ','
            << sensorHealthStateName(health_update->after.state) << ','
            << health_update->after.recovery_epoch.value() << ','
            << health_update->after.transition_sequence << ','
            << health_update->after.assessed_at.nanoseconds << ','
            << (health_update->transitioned ? 1 : 0) << ',' << health_update->consecutive_failures
            << ',' << health_update->recovery_good_shadow_results;
  } else {
    writeEmptyCsvFields(*output, 13U);
  }
  *output << ',' << removed_factor_batches;

  std::size_t batch_correspondences = 0U;
  std::size_t batch_excluded_source_rows = 0U;
  std::optional<double> batch_live_information_scale;
  for (const local::DirectLidarPairReport& pair : commit.lidar_pairs) {
    batch_correspondences += pair.correspondences;
    batch_excluded_source_rows += pair.source_rows_excluded_by_ownership;
    if (!batch_live_information_scale) {
      batch_live_information_scale = pair.information_scale;
    }
  }
  std::optional<double> batch_finalized_map_information_scale;
  if (commit.lidar_finalized_map) {
    batch_correspondences += commit.lidar_finalized_map->correspondences;
    batch_excluded_source_rows += commit.lidar_finalized_map->source_rows_excluded_by_ownership;
    batch_finalized_map_information_scale = commit.lidar_finalized_map->information_scale;
  }
  const std::size_t finalized_map_factor_count = commit.lidar_finalized_map ? 1U : 0U;
  const std::size_t batch_target_factor_count =
      commit.lidar_pairs.size() + finalized_map_factor_count;
  *output << ',' << commit.lidar_pairs.size() << ',' << finalized_map_factor_count << ','
          << batch_target_factor_count << ',' << batch_correspondences << ','
          << batch_excluded_source_rows << ',';
  if (batch_live_information_scale) {
    *output << *batch_live_information_scale;
  }
  *output << ',';
  if (batch_finalized_map_information_scale) {
    *output << *batch_finalized_map_information_scale;
  }
  const core::CorrelationDeclaration* lidar_correlation =
      batch_target_factor_count > 0U && commit.lineage.correlations.size() == 1U
          ? &commit.lineage.correlations.front()
          : nullptr;
  std::set<std::uint64_t> conditioning_measurement_roots;
  if (lidar_correlation != nullptr) {
    for (const core::ObservationUsage& usage : commit.lineage.usage) {
      if (usage.role != core::ObservationRole::ConditioningOnly) {
        continue;
      }
      if (const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root)) {
        conditioning_measurement_roots.insert(measurement->value());
      }
    }
  }
  *output << ',' << (lidar_correlation != nullptr ? 1 : 0) << ',';
  if (lidar_correlation != nullptr) {
    *output << lidar_correlation->policy.value();
  }
  *output << ',';
  if (lidar_correlation != nullptr) {
    *output << lidar_correlation->covariance_inflation;
  }
  *output << ',';
  if (lidar_correlation != nullptr && lidar_correlation->total_information_cap) {
    *output << *lidar_correlation->total_information_cap;
  }
  *output << ',';
  if (lidar_correlation != nullptr) {
    *output << conditioning_measurement_roots.size();
  }
  writeFinalizedMapCsvFields(*output,
                             commit.lidar_finalized_map ? &*commit.lidar_finalized_map : nullptr);
  writeFinalizedMapCsvFields(*output, rejected_finalized_map);
  if (rejected_solve) {
    *output << ',' << rejected_solve->nonlinear_iterations << ','
            << rejected_solve->convergence_interval_duration_s << ','
            << rejected_solve->convergence_sigma_fraction << ','
            << rejected_solve->effective_translation_convergence_m << ','
            << rejected_solve->effective_rotation_convergence_rad << ','
            << rejected_solve->effective_velocity_convergence_mps << ','
            << rejected_solve->effective_accelerometer_bias_convergence_mps2 << ','
            << rejected_solve->effective_gyroscope_bias_convergence_radps << ','
            << rejected_solve->effective_visual_log_inverse_range_convergence << ','
            << rejected_solve->last_iteration_translation_correction_m << ','
            << rejected_solve->last_iteration_rotation_correction_rad << ','
            << rejected_solve->last_iteration_velocity_correction_mps << ','
            << rejected_solve->last_iteration_accelerometer_bias_correction_mps2 << ','
            << rejected_solve->last_iteration_gyroscope_bias_correction_radps << ','
            << rejected_solve->last_iteration_visual_log_inverse_range_correction << ','
            << rejected_solve->maximum_transaction_translation_correction_m << ','
            << rejected_solve->maximum_transaction_rotation_correction_rad << ','
            << rejected_solve->nonlinear_full_steps_rejected << ','
            << rejected_solve->nonlinear_backtracking_trials << ','
            << rejected_solve->nonlinear_cauchy_directions_attempted << ','
            << rejected_solve->nonlinear_cauchy_steps_accepted << ','
            << rejected_solve->nonlinear_cauchy_backtracking_trials << ','
            << rejected_solve->nonlinear_zero_step_terminations << ','
            << rejected_solve->minimum_nonlinear_step_scale << ','
            << rejected_solve->last_iteration_objective_change << ',';
    writeOptionalCsv(*output, rejected_solve->error_before);
    *output << ',';
    writeOptionalCsv(*output, rejected_solve->error_after);
  } else {
    writeEmptyCsvFields(*output, 27U);
  }
  *output << ',';
  if (degradation_graph_error_code) {
    *output << static_cast<int>(*degradation_graph_error_code);
  }
  *output << ',';
  writeCsvString(*output, degradation_detail);
  if (motion) {
    *output << ',' << motion->segments << ',' << motion->imu_knots << ','
            << motion->support.nanoseconds << ',' << motion->rotation_excitation_rad << ','
            << motion->acceleration_excitation_mps2 << ',' << motion->scalar_dimension << ','
            << motion->expected_data_rank << ',' << motion->data_rank << ','
            << motion->calibrated_data_rank << ',' << motion->full_rank << ','
            << motion->prior_resolved_accel_tilt_modes << ','
            << motionInitializationObservabilityClassName(motion->observability_class) << ','
            << motion->data_supported_condition << ','
            << motion->calibrated_data_supported_condition << ',' << motion->full_hessian_condition
            << ',' << motion->initial_error << ',' << motion->final_error << ','
            << motion->lidar_error << ',' << motion->imu_error << ',' << motion->bias_prior_error
            << ',' << motion->gauge_error << ',' << motion->lidar_residual_dimension << ','
            << motion->imu_residual_dimension << ',' << motion->bias_prior_residual_dimension << ','
            << motion->gauge_residual_dimension << ',' << motion->total_residual_dimension << ','
            << motion->effective_degrees_of_freedom << ','
            << motion->lidar_mean_squared_whitened_residual << ','
            << motion->imu_mean_squared_whitened_residual << ',' << motion->reduced_chi_square
            << ',' << motion->holdout_lidar_segments << ',' << motion->holdout_residual_dimension
            << ',' << motion->holdout_error << ',' << motion->holdout_mean_squared_whitened_residual
            << ',' << static_cast<int>(motion->statistically_compatible) << ','
            << static_cast<int>(motion->conditioned_lidar_imu_approximation) << ','
            << motion->minimum_imu_conditioning_covariance_inflation << ','
            << motion->covariance_residual_inflation << ',' << motion->deskew_solve_passes << ','
            << motion->refined_sweeps << ',' << motion->refined_registrations << ','
            << motion->refined_deskew_pose_interpolations << ','
            << motion->refined_total_registration_cost << ','
            << motion->refined_maximum_registration_cost << ',' << motion->bias_prior_mahalanobis
            << ',' << motion->solver_iterations << ','
            << motionInitializationPassName(motion->pass);
  } else {
    // Forty-seven moving-initializer fields. Keep row width identical for
    // non-initialization commits so standard CSV readers remain reliable.
    writeEmptyCsvFields(*output, 47U);
  }
  *output << '\n';
  return output->good();
}

[[nodiscard]] core::ImuCalibration newerCollegeImuCalibration() {
  return core::ImuCalibration(
      "alphasense_imu", "/alphasense_driver_ros/imu", core::ImuSensorModel::BoschBmi085,
      kNominalImuRateHz, kGravityMps2,
      core::ImuNoiseModel(kAccelerometerNoiseDensity, kGyroscopeNoiseDensity,
                          kAccelerometerBiasRandomWalk, kGyroscopeBiasRandomWalk));
}

[[nodiscard]] core::Result<tools::ReplayProfile, tools::ReplayProfileError> makeReplayProfile(
    const core::CalibrationBundle& calibration, const BagLocalizationOptions& options) {
  tools::NewerCollegeReplayOptions replay;
  replay.bounds = tools::ReplayBounds{options.maximum_events, options.maximum_bag_messages};
  replay.session = kSession;
  replay.config = kConfigRevision;
  replay.clock_revision = kClockRevision;
  replay.first_source_epoch = kFirstSourceEpoch;
  replay.first_producer = kFirstProducer;
  replay.include_lidar = sensorModeIncludesLidar(options.sensor_mode);
  replay.include_cameras = sensorModeIncludesCameras(options.sensor_mode);
  // NCD publishes an image time and Kalibr supplies its signed offset to the
  // IMU clock. The archive does not supply exposure duration, so zero remains
  // explicit rather than inventing a rolling/global-shutter interval.
  replay.image_exposure = core::Duration{0};
  return tools::makeNewerCollegeRos2ReplayProfile(calibration, replay);
}

[[nodiscard]] local::LocalEstimatorConfig effectiveEstimatorConfig(
    const core::CalibrationBundle& calibration, const BagLocalizationOptions& options) {
  local::LocalEstimatorConfig config;
  switch (options.initialization_mode) {
    case BagLocalizationInitializationMode::Dynamic:
      config.initialization.mode = local::InitializationMode::DynamicOnly;
      break;
    case BagLocalizationInitializationMode::Static:
      config.initialization.mode = local::InitializationMode::StaticOnly;
      config.initialization.zero_motion_prior =
          local::ZeroMotionPrior{config.odom_epoch, local::ZeroMotionPriorSource::Operator};
      break;
    case BagLocalizationInitializationMode::SupervisedAuto:
      config.initialization.mode = local::InitializationMode::SupervisedAuto;
      config.initialization.zero_motion_prior =
          local::ZeroMotionPrior{config.odom_epoch, local::ZeroMotionPriorSource::Operator};
      break;
  }
  const core::ImuCalibration& imu = calibration.imu();
  const core::ImuNoiseModel& noise = imu.noise();
  config.stationary_initializer.nominal_period =
      core::Duration{static_cast<std::int64_t>(std::llround(1.0e9 / imu.nominalRateHz()))};
  config.stationary_initializer.gravity_mps2 = imu.gravityMagnitude();
  config.pipeline_timing.window_capacity = options.replay.timing_window_capacity;
  config.graph.imu.gravity_odom = Eigen::Vector3d{0.0, 0.0, -imu.gravityMagnitude()};
  config.graph.imu.accelerometer_noise_density_mps2_sqrt_hz = noise.accelerometerNoiseDensity();
  config.graph.imu.gyroscope_noise_density_radps_sqrt_hz = noise.gyroscopeNoiseDensity();
  config.graph.imu.accelerometer_bias_random_walk_mps3_sqrt_hz =
      noise.accelerometerBiasRandomWalk();
  config.graph.imu.gyroscope_bias_random_walk_radps2_sqrt_hz = noise.gyroscopeBiasRandomWalk();
  if (sensorModeIncludesCameras(options.sensor_mode)) {
    config.visual_cameras.reserve(calibration.cameras().size());
    for (const core::CameraCalibration& camera : calibration.cameras()) {
      local::VisualLaneConfig lane;
      lane.graph_submission_enabled = effectiveVisualGraphEnabled(options);
      config.visual_cameras.push_back(local::VisualCameraConfig{camera.id(), std::move(lane)});
    }
  }
  // GLIM's reference local smoother retains five seconds. The longer window
  // is important for nonlinear LiDAR/IMU relinearization before a boundary is
  // frozen into a marginal prior; the count remains a separate hard bound.
  config.graph.maximum_navigation_states = 64U;
  config.graph.target_fixed_lag = core::Duration{5'000'000'000LL};
  config.lidar_preprocessing.voxel_size_m = 0.30;
  config.lidar_preprocessing.maximum_output_points = 20'000U;
  config.lidar_bootstrap.preprocessing = config.lidar_preprocessing;
  // The bounded moving initializer first removes rotational distortion, then
  // re-deskews every retained sweep from its provisional states and resolves the
  // same discrete batch. LiDAR information is conservatively reduced because
  // those registrations and the batch IMU factors reuse raw IMU support.
  config.motion_initializer.maximum_solver_iterations = 1'000U;
  config.motion_initializer.solver_relative_error_tolerance = 1.0e-5;
  config.motion_initializer.solver_absolute_error_tolerance = 1.0e-6;
  // The initializer classifies the eigenvalues of a Jacobi-scaled Hessian.
  // Quad Easy's first independently valid 20-segment batch contains three
  // data-supported modes in [3.5e-11, 4.3e-11].  A 1e-10 numerical cutoff
  // therefore discarded real support and made the gauge priors appear to
  // resolve five modes instead of the physically expected two.  This is a
  // numerical rank-classification tolerance, not an observability-policy
  // relaxation: the required ranks and maximum prior-resolved modes below are
  // unchanged.
  config.motion_initializer.hessian_absolute_rank_tolerance = 1.0e-11;
  config.motion_initializer.hessian_relative_rank_tolerance = 1.0e-11;
  // Retain scan-local geometry for the complete five-second live-state
  // horizon. Registration selects a bounded newest/age-spread subset and each
  // factor remains attached to the target scan's real graph state.
  config.rolling_target.maximum_retained_sweeps = 50U;
  config.rolling_target.maximum_retained_points = 1'000'000U;
  config.rolling_target.odom_epoch = config.odom_epoch;
  // Newer College LiDAR is 10 Hz. Keep every scan in the direct-registration
  // and health lanes while admitting graph/map keyframes at approximately
  // 5 Hz. Full-bag measurements showed that three pose-aware factors at every
  // scan were redundant and dominated the fixed-lag solve.
  config.minimum_lidar_factor_interval = core::Duration{150'000'000LL};
  // GLIM's CPU odometry hands its scan-to-model result to the IMU smoother at
  // precision 1e3. Direct nearest-neighbor ICP still has locally correlated rows,
  // so keep the physical information spectrum bounded at the same deliberate
  // production scale while D005/D006 calibration remains benchmark-driven.
  config.lidar_registration.maximum_translation_information = 2.0e3;
  config.lidar_registration.maximum_targets = 3U;
  if (options.maximum_lidar_targets) {
    config.lidar_registration.maximum_targets = *options.maximum_lidar_targets;
  }
  config.rolling_target.registration = config.lidar_registration;
  config.lidar_bootstrap.registration = config.lidar_registration;
  config.lidar_bootstrap.registration.maximum_targets = 1U;
  // Single-target bootstrap/refinement needs the scan geometry that the
  // tighter multi-owner tracking cap intentionally removes.
  config.lidar_bootstrap.registration.maximum_composite_points_per_voxel = 64U;
  config.lidar_bootstrap.registration.maximum_translation_information = 1.0e3;
  config.lidar_bootstrap.registration.residual_standard_deviation_m = 0.05;
  return config;
}

[[nodiscard]] std::string_view replayCompletionName(tools::ReplayCompletion completion) {
  switch (completion) {
    case tools::ReplayCompletion::EndOfBag:
      return "end_of_bag";
    case tools::ReplayCompletion::EventLimit:
      return "event_limit";
    case tools::ReplayCompletion::BagMessageLimit:
      return "bag_message_limit";
    case tools::ReplayCompletion::VisitorStop:
      return "visitor_stop";
  }
  return "unknown";
}

[[nodiscard]] std::string_view lifecycleName(local::LocalEstimatorLifecycle lifecycle) {
  switch (lifecycle) {
    case local::LocalEstimatorLifecycle::AwaitingInitialization:
      return "awaiting_initialization";
    case local::LocalEstimatorLifecycle::Tracking:
      return "tracking";
    case local::LocalEstimatorLifecycle::Faulted:
      return "faulted";
  }
  return "unknown";
}

[[nodiscard]] SensorModeAudit makeSensorModeAudit(
    const BagLocalizationOptions& options, const core::CalibrationBundle* calibration,
    const tools::ReplayProfile* profile, const local::LocalEstimatorConfig* estimator_config,
    const tools::ReplayStats& replay_stats, const RunCounters& counters) {
  SensorModeAudit audit;
  audit.calibration_available = calibration != nullptr;
  audit.profile_available = profile != nullptr;
  audit.estimator_config_available = estimator_config != nullptr;
  audit.expected_lidar_topics = sensorModeIncludesLidar(options.sensor_mode) ? 1U : 0U;
  audit.expected_camera_topics =
      sensorModeIncludesCameras(options.sensor_mode) && calibration != nullptr
          ? calibration->cameras().size()
          : 0U;

  if (profile != nullptr) {
    for (const tools::ReplaySource& source : profile->sources()) {
      std::visit(Overloaded{
                     [&](const tools::ImuReplaySource&) { ++audit.configured_imu_topics; },
                     [&](const tools::LidarReplaySource& lidar) {
                       ++audit.configured_lidar_topics;
                       const auto found = replay_stats.configured_topics.find(lidar.topic);
                       if (found != replay_stats.configured_topics.end()) {
                         ++audit.lidar_topics_with_runtime_stats;
                         audit.lidar_messages_seen += found->second.messages_seen;
                         audit.lidar_events_emitted += found->second.events_emitted;
                       }
                     },
                     [&](const tools::CameraReplaySource& camera) {
                       ++audit.configured_camera_topics;
                       const auto found = replay_stats.configured_topics.find(camera.topic);
                       if (found != replay_stats.configured_topics.end()) {
                         ++audit.camera_topics_with_runtime_stats;
                         audit.camera_messages_seen += found->second.messages_seen;
                         audit.camera_image_decode_errors += found->second.image_decode_errors;
                         audit.camera_events_emitted += found->second.events_emitted;
                       }
                     },
                     [&](const tools::GnssReplaySource&) { ++audit.configured_gnss_topics; },
                 },
                 source);
    }
  }
  if (estimator_config != nullptr) {
    audit.configured_visual_lanes = estimator_config->visual_cameras.size();
  }

  audit.configuration_matches = audit.calibration_available && audit.profile_available &&
                                audit.estimator_config_available &&
                                audit.configured_imu_topics == audit.expected_imu_topics &&
                                audit.configured_lidar_topics == audit.expected_lidar_topics &&
                                audit.configured_camera_topics == audit.expected_camera_topics &&
                                audit.configured_gnss_topics == 0U &&
                                audit.configured_visual_lanes == audit.expected_camera_topics;

  const bool camera_runtime_isolated =
      sensorModeIncludesCameras(options.sensor_mode) ||
      (audit.configured_camera_topics == 0U && audit.camera_topics_with_runtime_stats == 0U &&
       audit.camera_messages_seen == 0U && audit.camera_image_decode_errors == 0U &&
       audit.camera_events_emitted == 0U && counters.camera_events == 0U &&
       counters.camera_ingest_accepted == 0U && counters.camera_ingest_rejected == 0U &&
       counters.cameras.empty());
  const bool lidar_runtime_isolated =
      sensorModeIncludesLidar(options.sensor_mode) ||
      (audit.configured_lidar_topics == 0U && audit.lidar_topics_with_runtime_stats == 0U &&
       audit.lidar_messages_seen == 0U && audit.lidar_events_emitted == 0U &&
       counters.lidar_events == 0U && counters.lidar_enqueue_accepted == 0U &&
       counters.lidar_enqueue_rejected == 0U);
  audit.storage_filter_matches =
      replay_stats.bag_messages_read == replay_stats.configured_messages_seen &&
      replay_stats.unknown_topics.empty();
  const bool storage_runtime_isolated =
      sensorModeIncludesCameras(options.sensor_mode) || audit.storage_filter_matches;
  audit.runtime_isolation_matches = camera_runtime_isolated && lidar_runtime_isolated &&
                                    storage_runtime_isolated && counters.unexpected_events == 0U;
  audit.passed = audit.configuration_matches && audit.runtime_isolation_matches;
  return audit;
}

class LocalizationDriver {
public:
  LocalizationDriver(local::LocalEstimator* estimator, BagLocalizationSensorMode sensor_mode,
                     const core::Pose3d& T_base_imu,
                     const local::LocalEstimatorConfig& estimator_config,
                     std::size_t timing_window_capacity, std::ofstream* imu_trajectory,
                     std::ofstream* base_trajectory, std::ofstream* fixed_lag_imu_trajectory,
                     std::ofstream* fixed_lag_base_trajectory,
                     std::ofstream* navigation_diagnostics)
      : estimator_(estimator),
        sensor_mode_(sensor_mode),
        T_imu_base_(T_base_imu.inverse()),
        imu_trajectory_(imu_trajectory),
        base_trajectory_(base_trajectory),
        fixed_lag_imu_trajectory_(fixed_lag_imu_trajectory),
        fixed_lag_base_trajectory_(fixed_lag_base_trajectory),
        navigation_diagnostics_(navigation_diagnostics),
        imu_buffer_count_capacity_(estimator_config.imu_buffer.maximum_samples),
        lidar_pending_count_capacity_(estimator_config.maximum_pending_lidar_sweeps),
        imu_ingress_timing_(imuIngressStage(), timing_window_capacity),
        lidar_enqueue_timing_(lidarEnqueueStage(), timing_window_capacity),
        camera_ingress_timing_(cameraIngressStage(), timing_window_capacity),
        process_ready_timing_(processReadyStage(), timing_window_capacity),
        report_output_timing_(reportOutputStage(), timing_window_capacity),
        imu_gap_timing_(timing_window_capacity) {}

  [[nodiscard]] tools::ReplayVisitResult visit(tools::DomainEvent&& event) {
    ++counters_.events_received;
    if (!first_recorded_arrival_) {
      first_recorded_arrival_ = event.metadata().recorded_arrival;
    }
    last_recorded_arrival_ = event.metadata().recorded_arrival;
    const core::MeasurementId measurement = event.metadata().measurement_sequence;

    const bool admitted = std::visit(
        Overloaded{
            [&](const core::ImuSample& observation) {
              const core::ThreadCpuWallTimer timer;
              ++counters_.imu_events;
              core::ImuSample owned = observation;
              auto result = estimator_->ingestImu(std::move(owned));
              if (!result) {
                ++counters_.imu_ingest_rejected;
                observeImuIngress(timer, core::PipelineDisposition::Rejected, measurement);
                setEstimatorFailure("imu_ingest", result.error());
                return false;
              }
              ++counters_.imu_ingest_accepted;
              imu_retained_samples_ = result.value().buffer.retained_samples;
              imu_retained_samples_high_watermark_ =
                  std::max(imu_retained_samples_high_watermark_, imu_retained_samples_);
              imu_gap_timing_.observe(result.value().buffer.newest_gap);
              observeImuIngress(timer, core::PipelineDisposition::Accepted, measurement);
              return true;
            },
            [&](const core::LidarSweep& observation) {
              const core::ThreadCpuWallTimer timer;
              ++counters_.lidar_events;
              if (!sensorModeIncludesLidar(sensor_mode_)) {
                observeLidarEnqueue(timer, core::PipelineDisposition::Rejected, measurement);
                failure_ = Failure{"sensor_mode_violation", "IMU-only replay emitted a LiDAR event",
                                   std::nullopt, std::nullopt};
                return false;
              }
              // DomainEvent deliberately exposes immutable observations. The
              // record copy is shallow for the immutable point payload, then
              // ownership of that record is moved into LocalEstimator.
              core::LidarSweep owned = observation;
              auto result = estimator_->enqueueLidar(std::move(owned));
              if (!result) {
                ++counters_.lidar_enqueue_rejected;
                observeLidarEnqueue(timer, core::PipelineDisposition::Rejected, measurement);
                setEstimatorFailure("lidar_enqueue", result.error());
                return false;
              }
              ++counters_.lidar_enqueue_accepted;
              lidar_pending_sweeps_ = result.value().pending_sweeps;
              observeLidarEnqueue(timer, core::PipelineDisposition::Accepted, measurement);
              return true;
            },
            [&](const core::CameraFrame& observation) {
              const core::ThreadCpuWallTimer timer;
              ++counters_.camera_events;
              if (!sensorModeIncludesCameras(sensor_mode_)) {
                observeDriverStage(&camera_ingress_timing_, cameraIngressStage(), timer,
                                   core::PipelineDisposition::Rejected, measurement);
                failure_ =
                    Failure{"sensor_mode_violation", "isolated replay emitted a camera event",
                            std::nullopt, std::nullopt};
                return false;
              }
              CameraRunCounters& camera = counters_.cameras[observation.camera.value()];
              ++camera.events;
              // CameraFrame also owns an immutable shared payload. Preserve
              // DomainEvent constness and move a shallow record copy into the
              // explicitly enabled visual lane.
              core::CameraFrame owned = observation;
              auto result = estimator_->ingestCamera(std::move(owned), false);
              if (!result) {
                ++counters_.camera_ingest_rejected;
                ++camera.ingest_rejected;
                observeDriverStage(&camera_ingress_timing_, cameraIngressStage(), timer,
                                   core::PipelineDisposition::Rejected, measurement);
                setEstimatorFailure("camera_ingest", result.error());
                return false;
              }
              ++counters_.camera_ingest_accepted;
              ++camera.ingest_accepted;
              noteCameraIngest(result.value());
              observeDriverStage(&camera_ingress_timing_, cameraIngressStage(), timer,
                                 core::PipelineDisposition::Accepted, measurement);
              return true;
            },
            [&](const auto&) {
              ++counters_.unexpected_events;
              failure_ = Failure{"unexpected_event",
                                 "configured " + std::string(sensorModeName(sensor_mode_)) +
                                     " profile emitted an unsupported modality",
                                 std::nullopt, std::nullopt};
              return false;
            },
        },
        event.observation());
    if (!admitted) {
      return tools::ReplayVisitResult::failure(tools::ReplayVisitorError{failure_->detail});
    }

    ++counters_.process_calls;
    const core::ThreadCpuWallTimer process_timer;
    auto processed = estimator_->processReady();
    if (!processed) {
      ++counters_.process_errors;
      observeDriverStage(&process_ready_timing_, processReadyStage(), process_timer,
                         core::PipelineDisposition::Failed, measurement, lidarPendingSnapshot());
      setEstimatorFailure("process_ready", processed.error());
      return tools::ReplayVisitResult::failure(tools::ReplayVisitorError{failure_->detail});
    }
    lidar_pending_sweeps_ = processed.value().pending_sweeps;
    observeDriverStage(&process_ready_timing_, processReadyStage(), process_timer,
                       core::PipelineDisposition::Completed, measurement, lidarPendingSnapshot());
    const core::ThreadCpuWallTimer report_timer;
    if (!consumeProcessReport(processed.value())) {
      observeDriverStage(&report_output_timing_, reportOutputStage(), report_timer,
                         core::PipelineDisposition::Failed, measurement);
      return tools::ReplayVisitResult::failure(tools::ReplayVisitorError{failure_->detail});
    }
    observeDriverStage(&report_output_timing_, reportOutputStage(), report_timer,
                       core::PipelineDisposition::Completed, measurement);
    return tools::ReplayVisitResult::success(tools::ReplayVisitAction::Continue);
  }

  [[nodiscard]] const RunCounters& counters() const noexcept { return counters_; }
  [[nodiscard]] const StateCoverage& coverage() const noexcept { return coverage_; }
  [[nodiscard]] std::size_t fixedLagStatesWritten() const noexcept {
    return fixed_lag_states_written_;
  }
  [[nodiscard]] const std::optional<Failure>& failure() const noexcept { return failure_; }
  [[nodiscard]] std::optional<core::ArrivalTime> firstRecordedArrival() const noexcept {
    return first_recorded_arrival_;
  }
  [[nodiscard]] std::optional<core::ArrivalTime> lastRecordedArrival() const noexcept {
    return last_recorded_arrival_;
  }
  [[nodiscard]] local::LocalEstimatorLifecycle lifecycle() const noexcept { return lifecycle_; }
  [[nodiscard]] bool finalizeFixedLagTrajectories() {
    for (const local::LocalGraphPoseSnapshot& state : latest_navigation_poses_) {
      if ((last_fixed_lag_state_ && state.state <= *last_fixed_lag_state_) ||
          (last_fixed_lag_time_ && state.exact_time <= *last_fixed_lag_time_)) {
        continue;
      }
      if (!writeFixedLagPose(state.state, state.exact_time, state.T_odom_imu)) {
        return false;
      }
    }
    fixed_lag_imu_trajectory_->flush();
    fixed_lag_base_trajectory_->flush();
    if (!fixed_lag_imu_trajectory_->good() || !fixed_lag_base_trajectory_->good()) {
      failure_ = Failure{"trajectory_output", "final fixed-lag trajectory flush failed",
                         std::nullopt, std::nullopt};
      return false;
    }
    return true;
  }

  [[nodiscard]] DriverTimingReport timingReport() const noexcept {
    return DriverTimingReport{
        imu_ingress_timing_.snapshot(),
        lidar_enqueue_timing_.snapshot(),
        camera_ingress_timing_.snapshot(),
        process_ready_timing_.snapshot(),
        report_output_timing_.snapshot(),
        ImuContinuityReport{imu_gap_timing_.snapshot(), imu_gap_timing_.maximumObserved(),
                            imu_retained_samples_high_watermark_},
        estimator_->pipelineTimingReport()};
  }

private:
  void observeDriverStage(
      core::BoundedPipelineTimingAccumulator* accumulator, const core::PipelineStage& stage,
      const core::ThreadCpuWallTimer& timer, core::PipelineDisposition disposition,
      core::MeasurementId measurement,
      std::optional<core::PipelineQueueSnapshot> queue = std::nullopt) noexcept {
    core::PipelineTimingSample sample =
        driverTimingSample(stage, timer.elapsed(), disposition, measurement);
    sample.queue = std::move(queue);
    const auto status = accumulator->observe(sample);
    static_cast<void>(status);
  }

  [[nodiscard]] core::PipelineQueueSnapshot imuBufferSnapshot() const {
    core::PipelineQueueSnapshot queue;
    queue.name = *core::PipelineQueueName::make("local_estimator.imu_buffer");
    queue.count = imu_retained_samples_;
    queue.count_capacity = imu_buffer_count_capacity_;
    queue.accepted = counters_.imu_ingest_accepted;
    queue.rejected = counters_.imu_ingest_rejected;
    return queue;
  }

  void observeImuIngress(const core::ThreadCpuWallTimer& timer,
                         core::PipelineDisposition disposition,
                         core::MeasurementId measurement) noexcept {
    core::PipelineTimingSample sample =
        driverTimingSample(imuIngressStage(), timer.elapsed(), disposition, measurement);
    sample.queue = imuBufferSnapshot();
    const auto status = imu_ingress_timing_.observe(sample);
    static_cast<void>(status);
  }

  [[nodiscard]] core::PipelineQueueSnapshot lidarPendingSnapshot() const {
    core::PipelineQueueSnapshot queue;
    queue.name = *core::PipelineQueueName::make("local_estimator.pending_lidar");
    queue.count = lidar_pending_sweeps_;
    queue.count_capacity = lidar_pending_count_capacity_;
    queue.accepted = counters_.lidar_enqueue_accepted;
    queue.rejected = counters_.lidar_enqueue_rejected;
    return queue;
  }

  void observeLidarEnqueue(const core::ThreadCpuWallTimer& timer,
                           core::PipelineDisposition disposition,
                           core::MeasurementId measurement) noexcept {
    core::PipelineTimingSample sample =
        driverTimingSample(lidarEnqueueStage(), timer.elapsed(), disposition, measurement);
    sample.queue = lidarPendingSnapshot();
    const auto status = lidar_enqueue_timing_.observe(sample);
    static_cast<void>(status);
  }

  void noteCameraIngest(const local::LocalEstimatorCameraIngestReport& report) {
    CameraRunCounters& camera = counters_.cameras[report.camera.value()];
    counters_.pending_camera_knots_at_end = report.pending_camera_knots;
    if (report.late_for_graph) {
      ++counters_.camera_frames_late_for_graph;
      ++camera.late_for_graph;
    }
    if (report.imu_rotation_seed_provided) {
      ++counters_.camera_rotation_seeds_provided;
      ++camera.imu_rotation_seeded;
    }
    switch (report.frame.disposition) {
      case local::VisualFrameDisposition::TrackingOnly:
        ++counters_.camera_tracking_only;
        ++camera.tracking_only;
        break;
      case local::VisualFrameDisposition::TrackingOnlyGraphSubmissionDisabled:
        ++counters_.camera_tracking_only_graph_submission_disabled;
        ++camera.tracking_only_graph_submission_disabled;
        break;
      case local::VisualFrameDisposition::TrackingOnlyNoLocalState:
        ++counters_.camera_tracking_only_without_local_state;
        ++camera.tracking_only_without_local_state;
        break;
      case local::VisualFrameDisposition::KeyframeStateRequested:
        ++counters_.camera_keyframe_requests;
        ++camera.keyframe_requests;
        break;
      case local::VisualFrameDisposition::KeyframeSuppressedByCapacity:
        ++counters_.camera_keyframes_suppressed_by_capacity;
        ++camera.keyframes_suppressed_by_capacity;
        break;
      case local::VisualFrameDisposition::KeyframeSuppressedByTimeline:
        ++counters_.camera_keyframes_suppressed_by_timeline;
        ++camera.keyframes_suppressed_by_timeline;
        break;
      case local::VisualFrameDisposition::KeyframeRejectedByTimeline:
        ++counters_.camera_keyframes_rejected_by_timeline;
        ++camera.keyframes_rejected_by_timeline;
        break;
    }
  }

  void observeFinalizedLidarTarget(
      const local::FinalizedLidarTargetProcessReport& report) noexcept {
    FinalizedLidarTargetRunCounters& target = counters_.finalized_lidar_target;
    ++target.process_reports;
    target.finality_matches += report.finality_matches;
    target.rollback_removals += report.rollback_removals;
    for (const local::FinalizedLidarTargetInsertStats& insertion : report.insertions) {
      ++target.insertion_transactions;
      target.insertion_input_points += insertion.input_points;
      target.insertion_voxels += insertion.insertion_voxels;
      target.insertion_selection_discarded_points += insertion.insertion_selection_discarded_points;
      target.minimum_separation_discarded_points += insertion.minimum_separation_discarded_points;
      target.query_voxel_capacity_discarded_points +=
          insertion.query_voxel_capacity_discarded_points;
      target.inserted_points += insertion.admitted_points;
      target.touched_query_voxels += insertion.touched_query_voxels;
    }
    for (const local::FinalizedLidarTargetPruneStats& prune : report.prunes) {
      ++target.prune_reports;
      target.prune_examined_query_voxels += prune.examined_query_voxels;
      target.prune_examined_points += prune.examined_points;
      target.pruned_query_voxels += prune.removed_query_voxels;
      target.pruned_points += prune.removed_points;
    }
    target.capacity_recovery_attempts += report.capacity_recovery_attempts;
    target.capacity_recovery_successes += report.capacity_recovery_successes;
    target.capacity_skipped_sweeps += report.capacity_skips.size();
    for (const local::FinalizedLidarTargetCapacitySkip& skip : report.capacity_skips) {
      if (skip.reason ==
          local::FinalizedLidarTargetCapacitySkipReason::RetrySuppressedWhileSaturated) {
        ++target.capacity_retry_suppressed_sweeps;
      }
      target.last_capacity_skip = skip;
    }
    if (report.insertion_frozen) {
      ++target.frozen_process_reports;
      if (!target.observed_freeze_state || !target.insertion_frozen) {
        ++target.freeze_transitions;
      }
    }
    target.observed_freeze_state = report.insertion_frozen;
    target.pending_sweeps = report.pending_sweeps;
    target.pending_unfinalized_sweeps = report.pending_unfinalized_sweeps;
    target.finalized_ready_sweeps = report.finalized_ready_sweeps;
    target.insertion_frozen = report.insertion_frozen;
    target.capacity_saturated = report.capacity_saturated;
    target.capacity_skips_since_retry = report.capacity_skips_since_retry;
    target.lidar_health = report.lidar_health;
    target.retained_points = report.retained_points;
    target.map_version = report.map_version;
    target.map_checksum = report.map_checksum;
  }

  void setEstimatorFailure(std::string stage, const local::LocalEstimatorError& error) {
    if (error.lidar_finalized_map) {
      ++counters_.rejected_finalized_map_reports;
      counters_.last_rejected_finalized_map = error.lidar_finalized_map;
    }
    failure_ = Failure{std::move(stage), error.detail, static_cast<int>(error.code),
                       static_cast<int>(error.stage)};
  }

  [[nodiscard]] bool writeFixedLagPose(core::StateId state, core::FusionTime time,
                                       const core::Pose3d& T_odom_imu) {
    if ((last_fixed_lag_state_ && state <= *last_fixed_lag_state_) ||
        (last_fixed_lag_time_ && time <= *last_fixed_lag_time_)) {
      failure_ = Failure{"trajectory_output",
                         "fixed-lag state identity or time was duplicate/non-monotonic",
                         std::nullopt, std::nullopt};
      return false;
    }
    if (!writeTumState(fixed_lag_imu_trajectory_, time, T_odom_imu) ||
        !writeTumState(fixed_lag_base_trajectory_, time, T_odom_imu * T_imu_base_)) {
      failure_ = Failure{"trajectory_output", "failed while writing fixed-lag trajectory",
                         std::nullopt, std::nullopt};
      return false;
    }
    last_fixed_lag_state_ = state;
    last_fixed_lag_time_ = time;
    ++fixed_lag_states_written_;
    return true;
  }

  [[nodiscard]] bool consumeFinalizedStates(
      std::span<const local::LocalGraphFinalizedState> finalized_states) {
    for (const local::LocalGraphFinalizedState& finalized : finalized_states) {
      if (!writeFixedLagPose(finalized.state, finalized.exact_time,
                             finalized.final_estimate.T_odom_imu)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool emitState(
      const local::LocalGraphCommit& commit, std::string_view commit_kind,
      const local::DirectLidarRegistrationReport* registration = nullptr,
      const local::LidarRegistrationError* registration_error = nullptr,
      const local::LocalSolveReport* rejected_solve = nullptr,
      const std::optional<local::LocalGraphErrorCode>& degradation_graph_error_code = std::nullopt,
      std::string_view degradation_detail = {},
      const local::MotionInitializationDiagnostics* motion = nullptr,
      const local::SensorHealthUpdate* health_update = nullptr,
      std::size_t removed_factor_batches = 0U,
      const local::DirectLidarFinalizedMapReport* rejected_finalized_map = nullptr) {
    if (last_observed_graph_revision_ && commit.revision <= *last_observed_graph_revision_) {
      failure_ = Failure{"trajectory_output",
                         "accepted navigation state did not advance the graph revision",
                         std::nullopt, std::nullopt};
      return false;
    }
    if (coverage_.last_state && commit.state.value() <= coverage_.last_state->value()) {
      failure_ =
          Failure{"trajectory_output", "accepted state identity was duplicate or non-monotonic",
                  std::nullopt, std::nullopt};
      return false;
    }
    if (coverage_.last_time && commit.state_time <= *coverage_.last_time) {
      failure_ = Failure{"trajectory_output", "accepted state time was duplicate or non-monotonic",
                         std::nullopt, std::nullopt};
      return false;
    }
    const core::Pose3d T_odom_base = commit.estimate.T_odom_imu * T_imu_base_;
    if (!writeTumState(imu_trajectory_, commit.state_time, commit.estimate.T_odom_imu) ||
        !writeTumState(base_trajectory_, commit.state_time, T_odom_base) ||
        !writeNavigationDiagnostics(
            navigation_diagnostics_, commit, commit_kind, true, registration, registration_error,
            rejected_solve, degradation_graph_error_code, degradation_detail, motion, health_update,
            removed_factor_batches, rejected_finalized_map)) {
      failure_ = Failure{"trajectory_output", "failed while writing localization output",
                         std::nullopt, std::nullopt};
      return false;
    }
    latest_navigation_poses_ = commit.navigation_poses;
    last_observed_graph_revision_ = commit.revision;
    if (rejected_solve != nullptr) {
      observeSolverGlobalization(*rejected_solve, &counters_.rejected_solver_globalization);
    }
    if (!coverage_.first_state) {
      coverage_.first_state = commit.state;
      coverage_.first_time = commit.state_time;
    }
    coverage_.last_state = commit.state;
    coverage_.last_time = commit.state_time;
    ++coverage_.states_written;
    if (coverage_.states_written % 10U == 0U) {
      imu_trajectory_->flush();
      base_trajectory_->flush();
      fixed_lag_imu_trajectory_->flush();
      fixed_lag_base_trajectory_->flush();
      navigation_diagnostics_->flush();
      if (!imu_trajectory_->good() || !base_trajectory_->good() ||
          !fixed_lag_imu_trajectory_->good() || !fixed_lag_base_trajectory_->good() ||
          !navigation_diagnostics_->good()) {
        failure_ = Failure{"trajectory_output", "periodic localization output flush failed",
                           std::nullopt, std::nullopt};
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool observeFactorOnlyLidarReport(const local::LidarCommitReport& lidar,
                                                  std::string_view commit_kind) {
    const local::LocalGraphCommit& commit = lidar.commit;
    if (lidar.navigation_state_created || !coverage_.last_state || !coverage_.last_time ||
        commit.state != *coverage_.last_state || commit.state_time != *coverage_.last_time ||
        !last_observed_graph_revision_) {
      failure_ =
          Failure{"trajectory_output",
                  "factor-only LiDAR report does not reference the latest emitted navigation state",
                  std::nullopt, std::nullopt};
      return false;
    }
    if (lidar.graph_revision_created) {
      if (commit.revision <= *last_observed_graph_revision_) {
        failure_ = Failure{"trajectory_output",
                           "factor-only LiDAR graph transaction did not advance revision",
                           std::nullopt, std::nullopt};
        return false;
      }
    } else if (commit.revision != *last_observed_graph_revision_) {
      failure_ =
          Failure{"trajectory_output",
                  "unchanged factor-only LiDAR report disagrees with the observed graph revision",
                  std::nullopt, std::nullopt};
      return false;
    }

    const local::DirectLidarRegistrationReport* registration =
        lidar.registration ? &*lidar.registration : nullptr;
    const local::LidarRegistrationError* registration_error =
        lidar.registration_error ? &*lidar.registration_error : nullptr;
    const local::LocalSolveReport* rejected_solve =
        lidar.rejected_solve ? &*lidar.rejected_solve : nullptr;
    const local::SensorHealthUpdate* health_update =
        lidar.health_update ? &*lidar.health_update : nullptr;
    const local::DirectLidarFinalizedMapReport* rejected_finalized_map =
        lidar.rejected_lidar_finalized_map ? &*lidar.rejected_lidar_finalized_map : nullptr;
    if (!writeNavigationDiagnostics(
            navigation_diagnostics_, commit, commit_kind, false, registration, registration_error,
            rejected_solve, lidar.degradation_graph_error_code, lidar.degradation_detail, nullptr,
            health_update, lidar.removed_factor_batches.size(), rejected_finalized_map)) {
      failure_ = Failure{"trajectory_output", "failed while writing factor-only diagnostics",
                         std::nullopt, std::nullopt};
      return false;
    }

    ++counters_.factor_only_lidar_reports;
    if (lidar.graph_revision_created) {
      ++counters_.factor_only_graph_revisions;
      latest_navigation_poses_ = commit.navigation_poses;
      last_observed_graph_revision_ = commit.revision;
    } else {
      ++counters_.factor_only_unchanged_graph_references;
    }
    if (rejected_solve != nullptr) {
      observeSolverGlobalization(*rejected_solve, &counters_.rejected_solver_globalization);
    }
    if (counters_.factor_only_lidar_reports % 10U == 0U) {
      navigation_diagnostics_->flush();
      if (!navigation_diagnostics_->good()) {
        failure_ = Failure{"trajectory_output", "periodic factor-only diagnostics flush failed",
                           std::nullopt, std::nullopt};
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool consumeProcessReport(const local::LocalEstimatorProcessReport& report) {
    observeFinalizedLidarTarget(report.finalized_lidar_target);
    auto normalized_result = detail::normalizeProcessReport(report);
    if (!normalized_result) {
      failure_ = Failure{"process_report_normalization", normalized_result.error().detail,
                         static_cast<int>(normalized_result.error().code), std::nullopt};
      return false;
    }
    const detail::NormalizedProcessReport& normalized = normalized_result.value();
    for (const local::LocalGraphTransactionSolveReport& transaction : report.graph_transactions) {
      if (!transaction.revision.valid() || !transaction.state.valid() ||
          (last_solver_transaction_revision_ &&
           transaction.parent != *last_solver_transaction_revision_) ||
          (!last_solver_transaction_revision_ && transaction.parent.valid())) {
        failure_ = Failure{"solver_observability",
                           "local graph transaction solve stream is invalid or non-contiguous",
                           std::nullopt, std::nullopt};
        return false;
      }
      observeSolverGlobalization(transaction.solve, &counters_.committed_solver_globalization);
      last_solver_transaction_revision_ = transaction.revision;
    }
    if (!consumeFinalizedStates(report.finalized_states)) {
      return false;
    }
    lifecycle_ = report.lifecycle;
    counters_.pending_sweeps_at_end = report.pending_sweeps;
    counters_.pending_camera_knots_at_end = report.pending_camera_knots;
    counters_.camera_commit_reports += report.camera_commits.size();
    for (const local::CameraKnotCommitReport& camera : report.camera_commits) {
      counters_.visual_keyframes_resolved += camera.resolved_keyframes.size();
    }
    counters_.visual_only_commits += normalized.camera_only_states;
    counters_.shared_camera_lidar_commits += normalized.shared_camera_lidar_states;
    counters_.commit_references_deduplicated += normalized.duplicate_references_removed;
    if (report.waiting_for_future_imu) {
      ++counters_.waiting_for_future_imu_reports;
    }
    if (report.initialization_rejection) {
      ++counters_.initialization_rejections_reported;
      counters_.last_initialization_rejection_stage =
          static_cast<int>(report.initialization_rejection->stage);
      counters_.last_initialization_rejection_code = report.initialization_rejection->code;
      counters_.last_initialization_rejection_detail = report.initialization_rejection->detail;
      switch (report.initialization_rejection->stage) {
        case local::LocalInitializationRejectionStage::StationaryTest:
          counters_.last_stationary_rejection_code = report.initialization_rejection->code;
          counters_.last_stationary_rejection_detail = report.initialization_rejection->detail;
          break;
        case local::LocalInitializationRejectionStage::LidarBootstrap:
          counters_.last_bootstrap_rejection_code = report.initialization_rejection->code;
          counters_.last_bootstrap_rejection_detail = report.initialization_rejection->detail;
          break;
        case local::LocalInitializationRejectionStage::MotionBatch:
          counters_.last_motion_rejection_code = report.initialization_rejection->code;
          counters_.last_motion_rejection_segment = report.initialization_rejection->segment;
          counters_.last_motion_rejection_detail = report.initialization_rejection->detail;
          break;
      }
    }
    for (const local::LidarBootstrapProcessReport& bootstrap : report.bootstrap) {
      if (bootstrap.rejection) {
        ++counters_.initialization_bootstrap_rejections;
        counters_.last_bootstrap_rejection_code = static_cast<int>(bootstrap.rejection->code);
        counters_.last_bootstrap_rejection_detail = bootstrap.rejection->detail;
      }
      if (!bootstrap.commit) {
        continue;
      }
      if (bootstrap.commit->disposition == local::LidarBootstrapDisposition::AnchorCreated) {
        ++counters_.initialization_bootstrap_anchors;
      } else {
        ++counters_.initialization_bootstrap_increments;
      }
    }
    if (report.initialization_method == local::LocalInitializationMethod::StationaryImu) {
      ++counters_.stationary_initializations;
    } else if (report.initialization_method == local::LocalInitializationMethod::MotionLidarImu) {
      ++counters_.motion_initializations;
    }
    if (report.motion_initialization_diagnostics) {
      counters_.motion_initialization_diagnostics = *report.motion_initialization_diagnostics;
    }
    counters_.lidar_drops_reported += report.dropped_sweeps.size();
    for (const detail::NormalizedProcessCommit& normalized_commit : normalized.commits) {
      const local::VisualGraphAttachmentReport* visual_attachment = nullptr;
      const local::VisualGraphDegradationReport* visual_degradation = nullptr;
      const auto accept_attachment = [&](const auto& optional_attachment) {
        if (!optional_attachment) {
          return true;
        }
        const local::VisualGraphAttachmentReport& candidate = *optional_attachment;
        if (visual_attachment != nullptr &&
            (visual_attachment->camera != candidate.camera ||
             visual_attachment->lane_attachment != candidate.lane_attachment ||
             visual_attachment->transaction_state != candidate.transaction_state ||
             visual_attachment->committed_revision != candidate.committed_revision)) {
          return false;
        }
        visual_attachment = &candidate;
        return true;
      };
      const auto accept_degradation = [&](const auto& optional_degradation) {
        if (!optional_degradation) {
          return true;
        }
        const local::VisualGraphDegradationReport& candidate = *optional_degradation;
        if (visual_degradation != nullptr &&
            (visual_degradation->camera != candidate.camera ||
             visual_degradation->lane_attachment != candidate.lane_attachment ||
             visual_degradation->transaction_state != candidate.transaction_state ||
             visual_degradation->committed_revision != candidate.committed_revision ||
             visual_degradation->rejected_graph_error_code !=
                 candidate.rejected_graph_error_code)) {
          return false;
        }
        visual_degradation = &candidate;
        return true;
      };
      if (normalized_commit.lidar != nullptr &&
          !accept_attachment(normalized_commit.lidar->visual_attachment)) {
        failure_ =
            Failure{"process_report_normalization",
                    "shared commit references disagree on visual attachment metadata",
                    static_cast<int>(
                        detail::ProcessReportNormalizationErrorCode::ConflictingCommitReferences),
                    std::nullopt};
        return false;
      }
      if (normalized_commit.lidar != nullptr &&
          !accept_degradation(normalized_commit.lidar->visual_degradation)) {
        failure_ =
            Failure{"process_report_normalization",
                    "shared commit references disagree on visual degradation metadata",
                    static_cast<int>(
                        detail::ProcessReportNormalizationErrorCode::ConflictingCommitReferences),
                    std::nullopt};
        return false;
      }
      for (const local::CameraKnotCommitReport* camera : normalized_commit.cameras) {
        if (!accept_attachment(camera->visual_attachment)) {
          failure_ =
              Failure{"process_report_normalization",
                      "shared commit references disagree on visual attachment metadata",
                      static_cast<int>(
                          detail::ProcessReportNormalizationErrorCode::ConflictingCommitReferences),
                      std::nullopt};
          return false;
        }
        if (!accept_degradation(camera->visual_degradation)) {
          failure_ =
              Failure{"process_report_normalization",
                      "shared commit references disagree on visual degradation metadata",
                      static_cast<int>(
                          detail::ProcessReportNormalizationErrorCode::ConflictingCommitReferences),
                      std::nullopt};
          return false;
        }
      }
      if (visual_attachment != nullptr) {
        ++counters_.visual_graph_attachments;
        counters_.visual_landmarks_attached += visual_attachment->graph_landmarks.size();
        counters_.visual_factors_attached += visual_attachment->graph_factors.size();
        counters_.visual_factors_retired += visual_attachment->graph_factor_retirements.size();
      }
      if (visual_degradation != nullptr) {
        ++counters_.visual_graph_degradations;
        counters_.visual_factor_batches_discarded += visual_degradation->factor_batches_discarded;
        counters_.visual_landmark_seeds_discarded += visual_degradation->landmark_seeds_discarded;
        counters_.visual_factor_specs_discarded += visual_degradation->factors_discarded;
        counters_.visual_stale_track_observations_discarded +=
            visual_degradation->stale_track_observations_discarded;
        counters_.last_visual_degradation_graph_error_code =
            static_cast<int>(visual_degradation->rejected_graph_error_code);
        counters_.last_visual_degradation_detail = visual_degradation->rejection_detail;
        ++counters_.cameras[visual_degradation->camera.value()].graph_degradations;
      }

      if (normalized_commit.initialization) {
        const bool moving =
            report.initialization_method == local::LocalInitializationMethod::MotionLidarImu;
        const local::MotionInitializationDiagnostics* diagnostics =
            report.motion_initialization_diagnostics ? &*report.motion_initialization_diagnostics
                                                     : nullptr;
        if (!emitState(*normalized_commit.commit,
                       moving ? "motion_initialization" : "stationary_initialization", nullptr,
                       nullptr, nullptr, std::nullopt, {}, diagnostics)) {
          return false;
        }
        continue;
      }

      if (normalized_commit.lidar == nullptr) {
        if (!emitState(*normalized_commit.commit, "visual_keyframe")) {
          return false;
        }
        continue;
      }

      const local::LidarCommitReport& lidar = *normalized_commit.lidar;
      if (lidar.commit.lidar_finalized_map) {
        ++counters_.accepted_finalized_map_reports;
        counters_.last_accepted_finalized_map = lidar.commit.lidar_finalized_map;
      }
      if (lidar.rejected_lidar_finalized_map) {
        ++counters_.rejected_finalized_map_reports;
        counters_.last_rejected_finalized_map = lidar.rejected_lidar_finalized_map;
      }
      std::string_view commit_kind;
      switch (lidar.disposition) {
        case local::LidarCommitDisposition::BootstrapTarget:
          ++counters_.bootstrap_commits;
          commit_kind = "lidar_target_bootstrap";
          break;
        case local::LidarCommitDisposition::Registered:
          ++counters_.registered_commits;
          commit_kind = "lidar_registered";
          break;
        case local::LidarCommitDisposition::RegisteredTrackingOnly:
          ++counters_.registered_tracking_only_commits;
          commit_kind = "lidar_registered_tracking_only";
          break;
        case local::LidarCommitDisposition::ImuOnlyDeskewRejected:
          ++counters_.deskew_degraded_commits;
          commit_kind = "imu_only_deskew_rejected";
          break;
        case local::LidarCommitDisposition::ImuOnlyPreprocessingRejected:
          ++counters_.preprocessing_degraded_commits;
          commit_kind = "imu_only_preprocessing_rejected";
          break;
        case local::LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained:
          ++counters_.registration_degraded_commits;
          if (lidar.degradation_graph_error_code) {
            counters_.last_registration_rejection_code =
                static_cast<int>(*lidar.degradation_graph_error_code);
            ++counters_.registration_degradation_codes[static_cast<int>(
                *lidar.degradation_graph_error_code)];
            if (*lidar.degradation_graph_error_code ==
                local::LocalGraphErrorCode::FactorBatchStateUnavailable) {
              ++counters_.target_state_unavailable_frozen_commits;
            }
          }
          counters_.last_registration_rejection_detail = lidar.degradation_detail;
          commit_kind = "imu_only_registration_rejected";
          break;
        case local::LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained:
          ++counters_.health_quarantined_commits;
          commit_kind = "imu_only_lidar_health_quarantined";
          break;
      }
      if (lidar.registration || lidar.registration_error) {
        ++counters_.lidar_registrations_attempted;
      }
      const local::LidarRegistrationWorkCounters* registration_work = nullptr;
      if (lidar.registration) {
        ++counters_.lidar_registrations_accepted;
        ++counters_.registration_diagnostics;
        if (lidar.registration->diagnostics.correspondences > 0U &&
            lidar.registration->diagnostics.observable_rank > 0U) {
          ++counters_.active_registration_diagnostics;
        }
        ++counters_.lidar_registration_terminations[lidar.registration->termination];
        if (lidar.disposition ==
            local::LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained) {
          ++counters_.accepted_registration_direct_graph_rejections;
        }
        if (lidar.disposition ==
            local::LidarCommitDisposition::ImuOnlyHealthQuarantinedTargetRetained) {
          ++counters_.accepted_registration_health_quarantines;
        }
        registration_work = &lidar.registration->work;
        counters_.last_lidar_registration = lidar.registration;
      }
      if (lidar.registration_error) {
        ++counters_.lidar_registrations_rejected;
        ++counters_.lidar_registration_errors[lidar.registration_error->code];
        registration_work = &lidar.registration_error->work;
        counters_.last_lidar_registration_error = lidar.registration_error;
      }
      if (lidar.health_update) {
        ++counters_.lidar_health_updates;
        if (lidar.health_update->transitioned) {
          ++counters_.lidar_health_transitions;
        }
        counters_.last_lidar_health_update = lidar.health_update;
      }
      counters_.lidar_factor_batches_removed += lidar.removed_factor_batches.size();
      if (registration_work != nullptr) {
        counters_.lidar_registration_outer_iterations += registration_work->outer_iterations;
        counters_.lidar_registration_accepted_steps += registration_work->accepted_steps;
        counters_.lidar_registration_rejected_trials +=
            registration_work->rejected_frozen_cost_trials;
      }
      const local::DirectLidarRegistrationReport* registration =
          lidar.registration ? &*lidar.registration : nullptr;
      const local::LidarRegistrationError* registration_error =
          lidar.registration_error ? &*lidar.registration_error : nullptr;
      const local::LocalSolveReport* rejected_solve =
          lidar.rejected_solve ? &*lidar.rejected_solve : nullptr;
      const local::SensorHealthUpdate* health_update =
          lidar.health_update ? &*lidar.health_update : nullptr;
      const local::DirectLidarFinalizedMapReport* rejected_finalized_map =
          lidar.rejected_lidar_finalized_map ? &*lidar.rejected_lidar_finalized_map : nullptr;
      if (normalized_commit.emits_navigation_state) {
        if (!emitState(*normalized_commit.commit, commit_kind, registration, registration_error,
                       rejected_solve, lidar.degradation_graph_error_code, lidar.degradation_detail,
                       nullptr, health_update, lidar.removed_factor_batches.size(),
                       rejected_finalized_map)) {
          return false;
        }
      } else if (!observeFactorOnlyLidarReport(lidar, commit_kind)) {
        return false;
      }
    }
    return true;
  }

  local::LocalEstimator* estimator_;
  BagLocalizationSensorMode sensor_mode_;
  core::Pose3d T_imu_base_;
  std::ofstream* imu_trajectory_;
  std::ofstream* base_trajectory_;
  std::ofstream* fixed_lag_imu_trajectory_;
  std::ofstream* fixed_lag_base_trajectory_;
  std::ofstream* navigation_diagnostics_;
  std::vector<local::LocalGraphPoseSnapshot> latest_navigation_poses_;
  std::optional<core::LocalGraphRevision> last_observed_graph_revision_;
  std::optional<core::LocalGraphRevision> last_solver_transaction_revision_;
  std::optional<core::StateId> last_fixed_lag_state_;
  std::optional<core::FusionTime> last_fixed_lag_time_;
  std::size_t fixed_lag_states_written_{};
  std::size_t imu_buffer_count_capacity_{};
  std::size_t lidar_pending_count_capacity_{};
  std::size_t imu_retained_samples_{};
  std::size_t imu_retained_samples_high_watermark_{};
  std::size_t lidar_pending_sweeps_{};
  core::BoundedPipelineTimingAccumulator imu_ingress_timing_;
  core::BoundedPipelineTimingAccumulator lidar_enqueue_timing_;
  core::BoundedPipelineTimingAccumulator camera_ingress_timing_;
  core::BoundedPipelineTimingAccumulator process_ready_timing_;
  core::BoundedPipelineTimingAccumulator report_output_timing_;
  BoundedDurationAccumulator imu_gap_timing_;
  RunCounters counters_;
  StateCoverage coverage_;
  std::optional<Failure> failure_;
  std::optional<core::ArrivalTime> first_recorded_arrival_;
  std::optional<core::ArrivalTime> last_recorded_arrival_;
  local::LocalEstimatorLifecycle lifecycle_{local::LocalEstimatorLifecycle::AwaitingInitialization};
};

void writeReplayStats(std::ostream& output, const tools::ReplayStats& stats) {
  output << "    \"bag_messages_read\": " << stats.bag_messages_read << ",\n"
         << "    \"configured_messages_seen\": " << stats.configured_messages_seen << ",\n"
         << "    \"events_emitted\": " << stats.events_emitted << ",\n"
         << "    \"unknown_topics\": {";
  bool first = true;
  for (const auto& [topic, count] : stats.unknown_topics) {
    output << (first ? "\n" : ",\n") << "      " << jsonString(topic) << ": " << count;
    first = false;
  }
  output << (first ? "" : "\n    ") << "},\n"
         << "    \"configured_topics\": {";
  first = true;
  for (const auto& [topic, topic_stats] : stats.configured_topics) {
    output << (first ? "\n" : ",\n") << "      " << jsonString(topic)
           << ": {\"messages_seen\": " << topic_stats.messages_seen
           << ", \"deserialization_errors\": " << topic_stats.deserialization_errors
           << ", \"conversion_errors\": " << topic_stats.conversion_errors
           << ", \"image_decode_errors\": " << topic_stats.image_decode_errors
           << ", \"clock_guard_errors\": " << topic_stats.clock_guard_errors
           << ", \"events_emitted\": " << topic_stats.events_emitted << '}';
    first = false;
  }
  output << (first ? "" : "\n    ") << "}\n";
}

void writeOptionalDurationNanoseconds(std::ostream& output,
                                      const std::optional<core::Duration>& duration) {
  if (duration) {
    output << duration->nanoseconds;
  } else {
    output << "null";
  }
}

void writeDurationStatistics(std::ostream& output, const core::DurationStatistics& statistics) {
  output << "{\"total_samples\": " << statistics.total_samples
         << ", \"window_samples\": " << statistics.window_samples
         << ", \"window_capacity\": " << statistics.window_capacity << ", \"minimum_ns\": ";
  writeOptionalDurationNanoseconds(output, statistics.minimum);
  output << ", \"p50_ns\": ";
  writeOptionalDurationNanoseconds(output, statistics.p50);
  output << ", \"p95_ns\": ";
  writeOptionalDurationNanoseconds(output, statistics.p95);
  output << ", \"p99_ns\": ";
  writeOptionalDurationNanoseconds(output, statistics.p99);
  output << ", \"maximum_ns\": ";
  writeOptionalDurationNanoseconds(output, statistics.maximum);
  output << ", \"mean_ns\": ";
  if (statistics.mean_nanoseconds) {
    output << *statistics.mean_nanoseconds;
  } else {
    output << "null";
  }
  output << '}';
}

void writeDispositionCounts(std::ostream& output, const core::PipelineDispositionCounts& counts) {
  output << "{\"completed\": " << counts.completed << ", \"accepted\": " << counts.accepted
         << ", \"rejected\": " << counts.rejected << ", \"deferred\": " << counts.deferred
         << ", \"dropped\": " << counts.dropped << ", \"skipped\": " << counts.skipped
         << ", \"failed\": " << counts.failed << '}';
}

void writeQueueSnapshot(std::ostream& output, const core::PipelineQueueSnapshot& queue) {
  output << "{\"name\": " << jsonString(queue.name.view()) << ", \"count\": " << queue.count
         << ", \"bytes\": " << queue.bytes << ", \"count_capacity\": ";
  if (queue.count_capacity) {
    output << *queue.count_capacity;
  } else {
    output << "null";
  }
  output << ", \"byte_capacity\": ";
  if (queue.byte_capacity) {
    output << *queue.byte_capacity;
  } else {
    output << "null";
  }
  output << ", \"oldest_age_ns\": ";
  writeOptionalDurationNanoseconds(output, queue.oldest_age);
  output << ", \"accepted\": " << queue.accepted << ", \"rejected\": " << queue.rejected
         << ", \"dropped_oldest\": " << queue.dropped_oldest
         << ", \"dropped_newest\": " << queue.dropped_newest
         << ", \"skipped_stale\": " << queue.skipped_stale
         << ", \"skipped_policy\": " << queue.skipped_policy << '}';
}

void writeTimingSample(std::ostream& output, const core::PipelineTimingSample& sample) {
  output << "{\"schema_version\": " << sample.schema_version
         << ", \"stage\": {\"id\": " << sample.stage.id.value()
         << ", \"name\": " << jsonString(sample.stage.name.view())
         << "}, \"wall_ns\": " << sample.wall_duration.nanoseconds << ", \"thread_cpu_ns\": ";
  writeOptionalDurationNanoseconds(output, sample.thread_cpu_duration);
  output << ", \"disposition\": " << jsonString(core::pipelineDispositionName(sample.disposition))
         << ", \"queue\": ";
  if (sample.queue) {
    writeQueueSnapshot(output, *sample.queue);
  } else {
    output << "null";
  }
  output << '}';
}

void writeTimingStatistics(std::ostream& output,
                           const core::PipelineTimingStatisticsSnapshot& statistics) {
  output << "{\"schema_version\": " << statistics.schema_version
         << ", \"stage\": {\"id\": " << statistics.stage.id.value()
         << ", \"name\": " << jsonString(statistics.stage.name.view()) << "}, \"wall\": ";
  writeDurationStatistics(output, statistics.wall);
  output << ", \"thread_cpu\": ";
  writeDurationStatistics(output, statistics.thread_cpu);
  output << ", \"dispositions\": ";
  writeDispositionCounts(output, statistics.dispositions);
  output << ", \"latest_queue\": ";
  if (statistics.latest_queue) {
    writeQueueSnapshot(output, *statistics.latest_queue);
  } else {
    output << "null";
  }
  output << '}';
}

void writeScheduledReplayRuntime(std::ostream& output,
                                 const tools::ScheduledReplayRuntimeReport& runtime) {
  output << "{\"queue\": {\"terminal\": ";
  writeQueueSnapshot(output, runtime.queue.terminal);
  output << ", \"maximum_count\": " << runtime.queue.maximum_count
         << ", \"maximum_bytes\": " << runtime.queue.maximum_bytes
         << ", \"maximum_oldest_age_ns\": ";
  writeOptionalDurationNanoseconds(output, runtime.queue.maximum_oldest_age);
  output << "}, \"producer_total\": ";
  writeTimingSample(output, runtime.producer_total);
  output << ", \"consumer_total\": ";
  writeTimingSample(output, runtime.consumer_total);
  output << ", \"producer_schedule_enqueue\": ";
  writeTimingStatistics(output, runtime.producer_schedule_enqueue);
  output << ", \"consumer_visit\": ";
  writeTimingStatistics(output, runtime.consumer_visit);
  output << '}';
}

void writeDriverTimingReport(std::ostream& output, const DriverTimingReport& report) {
  output << "{\"imu_ingress\": {\"timing\": ";
  writeTimingStatistics(output, report.imu_ingress);
  output << ", \"continuity\": {\"retained_samples_high_watermark\": "
         << report.imu_continuity.retained_samples_high_watermark
         << ", \"newest_gap_maximum_observed_ns\": ";
  writeOptionalDurationNanoseconds(output, report.imu_continuity.newest_gap_maximum_observed);
  output << ", \"newest_gap\": ";
  writeDurationStatistics(output, report.imu_continuity.newest_gap);
  output << "}}, \"lidar_enqueue\": ";
  writeTimingStatistics(output, report.lidar_enqueue);
  output << ", \"camera_ingress\": ";
  writeTimingStatistics(output, report.camera_ingress);
  output << ", \"process_ready\": ";
  writeTimingStatistics(output, report.process_ready);
  output << ", \"report_output\": ";
  writeTimingStatistics(output, report.report_output);
  output << '}';
}

void writeLocalPipelineTimingReport(std::ostream& output,
                                    const local::LocalPipelineTimingReport& report) {
  output << "{\"schema_version\": " << report.schema_version
         << ", \"window_capacity\": " << report.window_capacity << ", \"span_semantics\": "
         << jsonString(local::localPipelineTimingSpanSemanticsName(report.span_semantics))
         << ", \"stages\": [";
  for (std::size_t index = 0U; index < report.stages.size(); ++index) {
    output << (index == 0U ? "" : ", ");
    writeTimingStatistics(output, report.stages[index]);
  }
  output << "]}";
}

void writeInformationEigenvalues(std::ostream& output,
                                 const core::RankAwareInformation& information) {
  output << '[';
  for (Eigen::Index index = 0; index < information.eigenvalues.rows(); ++index) {
    output << (index == 0 ? "" : ", ") << information.eigenvalues(index);
  }
  output << ']';
}

void writeRankAwareInformation(std::ostream& output,
                               const core::RankAwareInformation& information) {
  output << "{\"rank\": " << information.rank << ", \"eigenvalues\": ";
  writeInformationEigenvalues(output, information);
  output << ", \"basis_columns\": [";
  for (Eigen::Index column = 0; column < information.basis.cols(); ++column) {
    output << (column == 0 ? "" : ", ") << '[';
    for (Eigen::Index row = 0; row < information.basis.rows(); ++row) {
      output << (row == 0 ? "" : ", ") << information.basis(row, column);
    }
    output << ']';
  }
  output << "]}";
}

void writeLidarRegistrationWork(std::ostream& output,
                                const local::LidarRegistrationWorkCounters& work) {
  output << "{\"source_points_considered\": " << work.source_points_considered
         << ", \"source_points_selected\": " << work.source_points_selected
         << ", \"source_points_omitted_by_capacity\": " << work.source_points_omitted_by_capacity
         << ", \"invalid_source_points\": " << work.invalid_source_points
         << ", \"invalid_target_points\": " << work.invalid_target_points
         << ", \"target_index_voxels\": " << work.target_index_voxels
         << ", \"composite_index_builds\": " << work.composite_index_builds
         << ", \"composite_index_input_owners\": " << work.composite_index_input_owners
         << ", \"composite_index_input_points\": " << work.composite_index_input_points
         << ", \"composite_index_retained_owners\": "
         << work.composite_index_retained_owners
         << ", \"composite_index_retained_points\": "
         << work.composite_index_retained_points
         << ", \"composite_index_occupied_voxels\": "
         << work.composite_index_occupied_voxels
         << ", \"composite_index_per_voxel_discarded_points\": "
         << work.composite_index_per_voxel_discarded_points
         << ", \"composite_index_total_discarded_points\": "
         << work.composite_index_total_discarded_points
         << ", \"composite_index_build_wall_ns\": "
         << work.composite_index_build_duration.wall.nanoseconds
         << ", \"composite_index_build_thread_cpu_ns\": ";
  writeOptionalDurationNanoseconds(output, work.composite_index_build_duration.thread_cpu);
  output << ", \"association_builds\": " << work.association_builds
         << ", \"candidate_voxel_lookups\": " << work.candidate_voxel_lookups
         << ", \"candidate_points_examined\": " << work.candidate_points_examined
         << ", \"finalized_map_candidate_voxel_lookups\": "
         << work.finalized_map_candidate_voxel_lookups
         << ", \"finalized_map_candidate_occupied_voxels\": "
         << work.finalized_map_candidate_occupied_voxels
         << ", \"finalized_map_candidate_points_examined\": "
         << work.finalized_map_candidate_points_examined
         << ", \"finalized_map_stale_fallbacks\": " << work.finalized_map_stale_fallbacks
         << ", \"frozen_objective_evaluations\": " << work.frozen_objective_evaluations
         << ", \"normal_equation_evaluations\": " << work.normal_equation_evaluations
         << ", \"outer_iterations\": " << work.outer_iterations
         << ", \"gauss_newton_trials\": " << work.gauss_newton_trials
         << ", \"lm_damping_trials\": " << work.lm_damping_trials
         << ", \"rejected_frozen_cost_trials\": " << work.rejected_frozen_cost_trials
         << ", \"accepted_steps\": " << work.accepted_steps << '}';
}

void writeLidarRegistrationDiagnostics(std::ostream& output,
                                       const local::LidarRegistrationDiagnostics& diagnostics) {
  output << "{\"target_count\": " << diagnostics.target_count
         << ", \"live_target_count\": " << diagnostics.live_target_count
         << ", \"finalized_map_target_count\": " << diagnostics.finalized_map_target_count
         << ", \"correspondences\": " << diagnostics.correspondences
         << ", \"live_correspondences\": " << diagnostics.live_correspondences
         << ", \"finalized_map_correspondences\": " << diagnostics.finalized_map_correspondences
         << ", \"overlap_fraction\": " << diagnostics.overlap_fraction
         << ", \"effective_correspondences\": " << diagnostics.effective_correspondences
         << ", \"maximum_squared_residual_m2\": " << diagnostics.maximum_squared_residual_m2
         << ", \"huber_delta_m\": " << diagnostics.huber_delta_m
         << ", \"characteristic_length_m\": " << diagnostics.characteristic_length_m
         << ", \"normalized_observable_eigenvalue_threshold\": "
         << diagnostics.normalized_observable_eigenvalue_threshold
         << ", \"observable_rank\": " << diagnostics.observable_rank
         << ", \"maximum_supported_normalized_information\": "
         << diagnostics.maximum_supported_normalized_information
         << ", \"normalized_information_cap\": " << diagnostics.normalized_information_cap
         << ", \"geometric_information_scale\": " << diagnostics.geometric_information_scale
         << ", \"raw_normalized_hessian_eigenvalues\": [";
  for (Eigen::Index index = 0; index < diagnostics.raw_normalized_hessian_eigenvalues.rows();
       ++index) {
    output << (index == 0 ? "" : ", ") << diagnostics.raw_normalized_hessian_eigenvalues(index);
  }
  output << "], \"normalized_directional_information\": ";
  writeRankAwareInformation(output, diagnostics.normalized_directional_information);
  output << ", \"physical_information\": ";
  writeRankAwareInformation(output, diagnostics.physical_information);
  output << '}';
}

void writeDirectLidarFinalizedMapReport(std::ostream& output,
                                        const local::DirectLidarFinalizedMapReport& report) {
  output << std::setprecision(17) << "{\"source_sweep\": " << report.source_sweep.value()
         << ", \"source_point_count\": " << report.source_point_count
         << ", \"correspondences\": " << report.correspondences
         << ", \"source_rows_excluded_by_ownership\": " << report.source_rows_excluded_by_ownership
         << ", \"candidate_voxel_lookups\": " << report.candidate_voxel_lookups
         << ", \"candidate_points_examined\": " << report.candidate_points_examined
         << ", \"unique_finalized_owners\": " << report.unique_finalized_owners
         << ", \"map_odom_epoch\": " << report.map_odom_epoch.value()
         << ", \"map_sensor\": {\"modality\": " << static_cast<int>(report.map_sensor.modality)
         << ", \"instance\": " << report.map_sensor.instance << "}"
         << ", \"map_version\": " << report.map_version
         << ", \"map_checksum\": " << jsonString(core::sha256Hex(report.map_checksum))
         << ", \"owner_pose_covariance_inflation\": " << report.owner_pose_covariance_inflation
         << ", \"configured_correlation_inflation_floor\": "
         << report.configured_correlation_inflation_floor
         << ", \"effective_covariance_inflation\": "
         << report.effective_covariance_inflation
         << ", \"information_scale\": " << report.information_scale
         << ", \"physical_information\": ";
  writeRankAwareInformation(output, report.physical_information);
  output << ", \"snapshot_checksum\": " << jsonString(core::sha256Hex(report.snapshot_checksum))
         << '}';
}

void writeSensorHealthSnapshot(std::ostream& output, const core::SensorHealthSnapshot& snapshot) {
  output << "{\"sensor\": {\"modality\": " << static_cast<int>(snapshot.sensor.modality)
         << ", \"instance\": " << snapshot.sensor.instance
         << "}, \"state\": " << jsonString(sensorHealthStateName(snapshot.state))
         << ", \"recovery_epoch\": " << snapshot.recovery_epoch.value()
         << ", \"transition_sequence\": " << snapshot.transition_sequence
         << ", \"assessed_at_ns\": " << snapshot.assessed_at.nanoseconds << '}';
}

void writeSensorHealthUpdate(std::ostream& output, const local::SensorHealthUpdate& update) {
  output << "{\"batch_id\": " << update.batch_id.value() << ", \"before\": ";
  writeSensorHealthSnapshot(output, update.before);
  output << ", \"after\": ";
  writeSensorHealthSnapshot(output, update.after);
  output << ", \"transitioned\": " << (update.transitioned ? "true" : "false")
         << ", \"consecutive_failures\": " << update.consecutive_failures
         << ", \"recovery_good_shadow_results\": " << update.recovery_good_shadow_results << '}';
}

void writeVisualLaneConfig(std::ostream& output, const local::VisualLaneConfig& lane) {
  const auto& frontend = lane.frontend;
  const auto& factor = lane.factor_builder;
  output << std::setprecision(17)
         << "{\"graph_submission_enabled\": " << (lane.graph_submission_enabled ? "true" : "false")
         << ", \"frontend\": {\"grid_columns\": " << frontend.grid_columns
         << ", \"grid_rows\": " << frontend.grid_rows
         << ", \"features_per_cell\": " << frontend.features_per_cell
         << ", \"detector_quality\": " << frontend.detector_quality
         << ", \"minimum_feature_distance_px\": " << frontend.minimum_feature_distance_px
         << ", \"detector_block_size\": " << frontend.detector_block_size
         << ", \"image_border_px\": " << frontend.image_border_px
         << ", \"klt_window_size\": " << frontend.klt_window_size
         << ", \"klt_pyramid_levels\": " << frontend.klt_pyramid_levels
         << ", \"klt_max_iterations\": " << frontend.klt_max_iterations
         << ", \"klt_epsilon\": " << frontend.klt_epsilon
         << ", \"maximum_klt_error\": " << frontend.maximum_klt_error
         << ", \"maximum_forward_backward_error_px\": "
         << frontend.maximum_forward_backward_error_px
         << ", \"recovery_minimum_tracks\": " << frontend.recovery_minimum_tracks
         << ", \"recovery_minimum_retention\": " << frontend.recovery_minimum_retention
         << ", \"keyframe_minimum_tracks\": " << frontend.keyframe_minimum_tracks
         << ", \"keyframe_minimum_spatial_coverage\": "
         << frontend.keyframe_minimum_spatial_coverage
         << ", \"keyframe_maximum_overlap\": " << frontend.keyframe_maximum_overlap
         << ", \"keyframe_minimum_parallax_px\": " << frontend.keyframe_minimum_parallax_px
         << ", \"minimum_keyframe_interval_ns\": " << frontend.minimum_keyframe_interval.nanoseconds
         << ", \"maximum_keyframe_interval_ns\": " << frontend.maximum_keyframe_interval.nanoseconds
         << ", \"maximum_time_uncertainty_ns\": " << frontend.maximum_time_uncertainty.nanoseconds
         << ", \"base_pixel_sigma\": " << frontend.base_pixel_sigma
         << "}, \"factor_builder\": {\"minimum_non_anchor_observations\": "
         << factor.minimum_non_anchor_observations
         << ", \"maximum_pending_observations\": " << factor.maximum_pending_observations
         << ", \"maximum_observations_per_track\": " << factor.maximum_observations_per_track
         << ", \"maximum_active_tracks\": " << factor.maximum_active_tracks
         << ", \"maximum_missed_keyframes\": " << factor.maximum_missed_keyframes
         << ", \"minimum_baseline_m\": " << factor.minimum_baseline_m
         << ", \"minimum_parallax_rad\": " << factor.minimum_parallax_rad
         << ", \"minimum_range_m\": " << factor.minimum_range_m
         << ", \"maximum_range_m\": " << factor.maximum_range_m
         << ", \"triangulation_huber_angle_rad\": " << factor.triangulation_huber_angle_rad
         << ", \"triangulation_iterations\": " << factor.triangulation_iterations
         << ", \"maximum_triangulation_condition\": " << factor.maximum_triangulation_condition
         << ", \"maximum_inlier_reprojection_error_px\": "
         << factor.maximum_inlier_reprojection_error_px
         << ", \"maximum_reprojection_rmse_px\": " << factor.maximum_reprojection_rmse_px
         << ", \"huber_delta_sigma\": " << factor.huber_delta_sigma
         << ", \"outlier_chi_squared_gate\": " << factor.outlier_chi_squared_gate
         << ", \"outlier_commits_before_retirement\": " << factor.outlier_commits_before_retirement
         << "}, \"queues\": {\"maximum_pending_keyframes\": " << lane.maximum_pending_keyframes
         << ", \"maximum_pending_factor_batches\": " << lane.maximum_pending_factor_batches
         << ", \"maximum_pending_factor_retirements\": " << lane.maximum_pending_factor_retirements
         << ", \"maximum_factor_retirements_per_attachment\": "
         << lane.maximum_factor_retirements_per_attachment
         << ", \"maximum_landmarks_per_attachment\": " << lane.maximum_landmarks_per_attachment
         << ", \"maximum_factors_per_attachment\": " << lane.maximum_factors_per_attachment << "}}";
}

void writeLidarPreprocessConfig(std::ostream& output, const local::LidarPreprocessConfig& config) {
  output << "{\"parallel_worker_count\": " << config.parallel_worker_count
         << ", \"minimum_range_m\": " << config.minimum_range_m
         << ", \"maximum_range_m\": " << config.maximum_range_m
         << ", \"voxel_size_m\": " << config.voxel_size_m
         << ", \"maximum_output_points\": " << config.maximum_output_points << '}';
}

void writeLidarRegistrationConfig(std::ostream& output,
                                  const local::LidarRegistrationConfig& config) {
  output << "{\"target_voxel_resolution_m\": " << config.target_voxel_resolution_m
         << ", \"source_voxel_size_m\": " << config.source_voxel_size_m
         << ", \"maximum_correspondence_distance_m\": " << config.maximum_correspondence_distance_m
         << ", \"maximum_voxel_search_radius\": " << config.maximum_voxel_search_radius
         << ", \"maximum_source_points\": " << config.maximum_source_points
         << ", \"maximum_target_points_per_target\": " << config.maximum_target_points_per_target
         << ", \"maximum_targets\": " << config.maximum_targets
         << ", \"maximum_composite_owners\": " << config.maximum_composite_owners
         << ", \"maximum_composite_indexed_points\": "
         << config.maximum_composite_indexed_points
         << ", \"maximum_composite_points_per_voxel\": "
         << config.maximum_composite_points_per_voxel
         << ", \"parallel_worker_count\": " << config.parallel_worker_count
         << ", \"minimum_correspondences\": " << config.minimum_correspondences
         << ", \"residual_standard_deviation_m\": " << config.residual_standard_deviation_m
         << ", \"huber_delta_multiplier\": " << config.huber_delta_multiplier
         << ", \"minimum_huber_delta_m\": " << config.minimum_huber_delta_m
         << ", \"maximum_huber_delta_m\": " << config.maximum_huber_delta_m
         << ", \"maximum_outer_iterations\": " << config.maximum_outer_iterations
         << ", \"maximum_lm_damping_retries\": " << config.maximum_lm_damping_retries
         << ", \"initial_relative_damping\": " << config.initial_relative_damping
         << ", \"damping_increase\": " << config.damping_increase
         << ", \"translation_convergence_m\": " << config.translation_convergence_m
         << ", \"rotation_convergence_rad\": " << config.rotation_convergence_rad
         << ", \"maximum_correction_translation_m\": " << config.maximum_correction_translation_m
         << ", \"maximum_correction_rotation_rad\": " << config.maximum_correction_rotation_rad
         << ", \"minimum_observable_rank\": " << config.minimum_observable_rank
         << ", \"absolute_normalized_observable_eigenvalue\": "
         << config.absolute_normalized_observable_eigenvalue
         << ", \"relative_normalized_observable_eigenvalue\": "
         << config.relative_normalized_observable_eigenvalue
         << ", \"minimum_characteristic_length_m\": " << config.minimum_characteristic_length_m
         << ", \"maximum_characteristic_length_m\": " << config.maximum_characteristic_length_m
         << ", \"maximum_translation_information\": " << config.maximum_translation_information
         << ", \"seed_translation_consistency_tolerance_m\": "
         << config.seed_translation_consistency_tolerance_m
         << ", \"seed_rotation_consistency_tolerance_rad\": "
         << config.seed_rotation_consistency_tolerance_rad << '}';
}

void writeFinalizedLidarTargetMapConfig(std::ostream& output,
                                        const local::FinalizedLidarTargetMapConfig& config) {
  output << std::setprecision(17) << "{\"odom_epoch\": " << config.odom_epoch.value()
         << ", \"sensor\": {\"modality\": " << static_cast<int>(config.sensor.modality)
         << ", \"instance\": " << config.sensor.instance << "}"
         << ", \"query_voxel_size_m\": " << config.query_voxel_size_m
         << ", \"insertion_voxel_size_m\": " << config.insertion_voxel_size_m
         << ", \"maximum_points_per_query_voxel\": " << config.maximum_points_per_query_voxel
         << ", \"minimum_point_separation_m\": " << config.minimum_point_separation_m
         << ", \"maximum_supported_query_distance_m\": "
         << config.maximum_supported_query_distance_m
         << ", \"maximum_radius_m\": " << config.maximum_radius_m
         << ", \"hard_point_capacity\": " << config.hard_point_capacity << '}';
}

void writeEstimatorConfig(std::ostream& output, const local::LocalEstimatorConfig& config) {
  const auto& stationary = config.stationary_initializer;
  const auto& bootstrap = config.lidar_bootstrap;
  const auto& motion = config.motion_initializer;
  const auto& graph = config.graph;
  const auto& preprocessing = config.lidar_preprocessing;
  const auto& target = config.rolling_target;
  const auto& registration = config.lidar_registration;
  output << std::setprecision(17) << "      \"odom_epoch\": " << config.odom_epoch.value() << ",\n"
         << "      \"first_state\": " << config.first_state.value() << ",\n"
         << "      \"initialization\": {\"mode\": "
         << jsonString(localInitializationModeName(config.initialization.mode))
         << ", \"zero_motion_prior\": ";
  if (config.initialization.zero_motion_prior) {
    output << "{\"odom_epoch\": " << config.initialization.zero_motion_prior->odom_epoch.value()
           << ", \"source\": "
           << jsonString(zeroMotionPriorSourceName(config.initialization.zero_motion_prior->source))
           << '}';
  } else {
    output << "null";
  }
  output << "},\n"
         << "      \"maximum_pending_lidar_sweeps\": " << config.maximum_pending_lidar_sweeps
         << ",\n"
         << "      \"maximum_pending_finalized_lidar_sweeps\": "
         << config.maximum_pending_finalized_lidar_sweeps << ",\n"
         << "      \"finalized_lidar_prune_interval_sweeps\": "
         << config.finalized_lidar_prune_interval_sweeps << ",\n"
         << "      \"maximum_pending_imu_guards\": " << config.maximum_pending_imu_guards << ",\n"
         << "      \"minimum_lidar_factor_interval_ns\": "
         << config.minimum_lidar_factor_interval.nanoseconds << ",\n"
         << "      \"state_timeline\": {\"minimum_state_interval_ns\": "
         << config.state_timeline.minimum_state_interval.nanoseconds
         << ", \"maximum_navigation_states\": " << config.state_timeline.maximum_navigation_states
         << ", \"maximum_retained_requests\": " << config.state_timeline.maximum_retained_requests
         << "},\n"
         << "      \"stationary_retry_period_ns\": " << config.stationary_retry_period.nanoseconds
         << ",\n"
         << "      \"pipeline_timing\": {\"window_capacity\": "
         << config.pipeline_timing.window_capacity
         << ", \"maximum_window_capacity\": " << local::kMaximumLocalPipelineTimingWindowCapacity
         << "},\n"
         << "      \"sensor_health_policy\": {\"consecutive_failures_to_suspect\": "
         << config.sensor_health_policy.consecutive_failures_to_suspect
         << ", \"consecutive_failures_to_failed\": "
         << config.sensor_health_policy.consecutive_failures_to_failed
         << ", \"recovery_good_shadow_results\": "
         << config.sensor_health_policy.recovery_good_shadow_results
         << ", \"suspect_after_no_result_ns\": "
         << config.sensor_health_policy.suspect_after_no_result.nanoseconds
         << ", \"failed_after_no_result_ns\": "
         << config.sensor_health_policy.failed_after_no_result.nanoseconds << "},\n"
         << "      \"maximum_recent_faulty_batches_to_remove\": "
         << config.maximum_recent_faulty_batches_to_remove << ",\n"
         << "      \"imu_buffer\": {\"maximum_samples\": " << config.imu_buffer.maximum_samples
         << ", \"maximum_span_ns\": " << config.imu_buffer.maximum_span.nanoseconds << "},\n"
         << "      \"stationary_initializer\": {"
         << "\"minimum_support_ns\": " << stationary.minimum_support.nanoseconds
         << ", \"nominal_period_ns\": " << stationary.nominal_period.nanoseconds
         << ", \"maximum_time_uncertainty_ns\": " << stationary.maximum_time_uncertainty.nanoseconds
         << ", \"gravity_mps2\": " << stationary.gravity_mps2
         << ", \"maximum_mean_angular_rate_radps\": " << stationary.maximum_mean_angular_rate_radps
         << ", \"maximum_gyro_stddev_radps\": " << stationary.maximum_gyro_stddev_radps
         << ", \"maximum_gravity_norm_error_mps2\": " << stationary.maximum_gravity_norm_error_mps2
         << ", \"maximum_accel_stddev_mps2\": " << stationary.maximum_accel_stddev_mps2
         << ", \"accelerometer_bias_prior_sigma_mps2\": "
         << stationary.accelerometer_bias_prior_sigma_mps2 << "},\n"
         << "      \"lidar_bootstrap\": {\"maximum_sweeps\": " << bootstrap.maximum_sweeps
         << ", \"target_reuse_covariance_inflation\": "
         << bootstrap.target_reuse_covariance_inflation
         << ", \"imu_conditioning_covariance_inflation\": "
         << bootstrap.imu_conditioning_covariance_inflation
         << ", \"maximum_increment_translation_m\": " << bootstrap.maximum_increment_translation_m
         << ", \"maximum_increment_rotation_rad\": " << bootstrap.maximum_increment_rotation_rad
         << ", \"maximum_observable_condition\": " << bootstrap.maximum_observable_condition
         << ", \"preprocessing\": ";
  writeLidarPreprocessConfig(output, bootstrap.preprocessing);
  output << ", \"registration\": ";
  writeLidarRegistrationConfig(output, bootstrap.registration);
  output << "},\n"
         << "      \"motion_initializer\": {\"minimum_segments\": " << motion.minimum_segments
         << ", \"maximum_segments\": " << motion.maximum_segments
         << ", \"maximum_imu_knots\": " << motion.maximum_imu_knots
         << ", \"minimum_support_ns\": " << motion.minimum_support.nanoseconds
         << ", \"maximum_support_ns\": " << motion.maximum_support.nanoseconds
         << ", \"maximum_time_uncertainty_ns\": " << motion.maximum_time_uncertainty.nanoseconds
         << ", \"maximum_raw_imu_gap_ns\": " << motion.maximum_raw_imu_gap.nanoseconds
         << ", \"minimum_rotation_excitation_rad\": " << motion.minimum_rotation_excitation_rad
         << ", \"minimum_acceleration_excitation_mps2\": "
         << motion.minimum_acceleration_excitation_mps2
         << ", \"information_basis_orthonormal_tolerance\": "
         << motion.information_basis_orthonormal_tolerance
         << ", \"information_zero_tolerance\": " << motion.information_zero_tolerance
         << ", \"hessian_absolute_rank_tolerance\": " << motion.hessian_absolute_rank_tolerance
         << ", \"hessian_relative_rank_tolerance\": " << motion.hessian_relative_rank_tolerance
         << ", \"maximum_supported_hessian_condition\": "
         << motion.maximum_supported_hessian_condition
         << ", \"covariance_symmetry_relative_tolerance\": "
         << motion.covariance_symmetry_relative_tolerance
         << ", \"minimum_covariance_eigenvalue\": " << motion.minimum_covariance_eigenvalue
         << ", \"position_gauge_sigma_m\": " << motion.position_gauge_sigma_m
         << ", \"yaw_gauge_sigma_rad\": " << motion.yaw_gauge_sigma_rad
         << ", \"maximum_bias_prior_mahalanobis\": " << motion.maximum_bias_prior_mahalanobis
         << ", \"minimum_imu_conditioning_covariance_inflation\": "
         << motion.minimum_imu_conditioning_covariance_inflation
         << ", \"maximum_prior_resolved_accel_tilt_modes\": "
         << motion.maximum_prior_resolved_accel_tilt_modes
         << ", \"holdout_lidar_segments\": " << motion.holdout_lidar_segments
         << ", \"maximum_lidar_msw\": " << motion.maximum_lidar_mean_squared_whitened_residual
         << ", \"maximum_imu_msw\": " << motion.maximum_imu_mean_squared_whitened_residual
         << ", \"maximum_reduced_chi_square\": " << motion.maximum_reduced_chi_square
         << ", \"maximum_holdout_msw\": " << motion.maximum_holdout_mean_squared_whitened_residual
         << ", \"maximum_covariance_residual_inflation\": "
         << motion.maximum_covariance_residual_inflation
         << ", \"minimum_orientation_variance_rad2\": " << motion.minimum_orientation_variance_rad2
         << ", \"minimum_velocity_variance_m2ps2\": " << motion.minimum_velocity_variance_m2ps2
         << ", \"minimum_position_variance_m2\": " << motion.minimum_position_variance_m2
         << ", \"minimum_gyro_bias_variance_rad2ps2\": "
         << motion.minimum_gyro_bias_variance_rad2ps2
         << ", \"minimum_accel_bias_variance_m2ps4\": " << motion.minimum_accel_bias_variance_m2ps4
         << ", \"maximum_solver_iterations\": " << motion.maximum_solver_iterations
         << ", \"solver_relative_error_tolerance\": " << motion.solver_relative_error_tolerance
         << ", \"solver_absolute_error_tolerance\": " << motion.solver_absolute_error_tolerance
         << "},\n"
         << "      \"graph\": {\"maximum_navigation_states\": " << graph.maximum_navigation_states
         << ", \"target_fixed_lag_ns\": " << graph.target_fixed_lag.nanoseconds
         << ", \"pose_rotation_relinearization_rad\": " << graph.pose_rotation_relinearization_rad
         << ", \"pose_translation_relinearization_m\": " << graph.pose_translation_relinearization_m
         << ", \"velocity_relinearization_mps\": " << graph.velocity_relinearization_mps
         << ", \"accelerometer_bias_relinearization_mps2\": "
         << graph.accelerometer_bias_relinearization_mps2
         << ", \"gyroscope_bias_relinearization_radps\": "
         << graph.gyroscope_bias_relinearization_radps
         << ", \"visual_log_inverse_range_relinearization\": "
         << graph.visual_log_inverse_range_relinearization
         << ", \"maximum_nonlinear_iterations\": " << graph.maximum_nonlinear_iterations
         << ", \"nonlinear_convergence_sigma_fraction\": "
         << graph.nonlinear_convergence_sigma_fraction
         << ", \"nonlinear_translation_convergence_m\": "
         << graph.nonlinear_translation_convergence_m
         << ", \"nonlinear_rotation_convergence_rad\": " << graph.nonlinear_rotation_convergence_rad
         << ", \"nonlinear_velocity_convergence_mps\": " << graph.nonlinear_velocity_convergence_mps
         << ", \"nonlinear_accelerometer_bias_convergence_mps2\": "
         << graph.nonlinear_accelerometer_bias_convergence_mps2
         << ", \"nonlinear_gyroscope_bias_convergence_radps\": "
         << graph.nonlinear_gyroscope_bias_convergence_radps
         << ", \"nonlinear_visual_log_inverse_range_convergence\": "
         << graph.nonlinear_visual_log_inverse_range_convergence
         << ", \"maximum_nonlinear_backtracking_steps\": "
         << graph.maximum_nonlinear_backtracking_steps
         << ", \"nonlinear_backtracking_reduction\": " << graph.nonlinear_backtracking_reduction
         << ", \"nonlinear_objective_absolute_convergence\": "
         << graph.nonlinear_objective_absolute_convergence
         << ", \"nonlinear_objective_relative_convergence\": "
         << graph.nonlinear_objective_relative_convergence
         << ", \"maximum_transaction_translation_correction_m\": "
         << graph.maximum_transaction_translation_correction_m
         << ", \"maximum_transaction_rotation_correction_rad\": "
         << graph.maximum_transaction_rotation_correction_rad
         << ", \"complete_objective_nonsmooth_absolute_allowance\": "
         << graph.complete_objective_nonsmooth_absolute_allowance
         << ", \"complete_objective_nonsmooth_relative_allowance\": "
         << graph.complete_objective_nonsmooth_relative_allowance
         << ", \"maximum_visual_landmarks_per_transaction\": "
         << graph.maximum_visual_landmarks_per_transaction
         << ", \"maximum_visual_factors_per_transaction\": "
         << graph.maximum_visual_factors_per_transaction
         << ", \"maximum_visual_factor_retirements_per_transaction\": "
         << graph.maximum_visual_factor_retirements_per_transaction
         << ", \"maximum_direct_lidar_factors_per_transaction\": "
         << graph.maximum_direct_lidar_factors_per_transaction
         << ", \"maximum_finalized_lidar_owners_per_factor\": "
         << graph.maximum_finalized_lidar_owners_per_factor
         << ", \"maximum_active_factor_batches\": " << graph.maximum_active_factor_batches
         << ", \"maximum_removable_factor_batches\": " << graph.maximum_removable_factor_batches
         << ", \"maximum_factor_batches_per_removal_transaction\": "
         << graph.maximum_factor_batches_per_removal_transaction
         << ", \"maximum_terminal_factor_batch_records\": "
         << graph.maximum_terminal_factor_batch_records << ", \"imu\": {\"gravity_odom\": ["
         << graph.imu.gravity_odom.x() << ", " << graph.imu.gravity_odom.y() << ", "
         << graph.imu.gravity_odom.z() << "], \"accelerometer_noise_density_mps2_sqrt_hz\": "
         << graph.imu.accelerometer_noise_density_mps2_sqrt_hz
         << ", \"gyroscope_noise_density_radps_sqrt_hz\": "
         << graph.imu.gyroscope_noise_density_radps_sqrt_hz
         << ", \"accelerometer_bias_random_walk_mps3_sqrt_hz\": "
         << graph.imu.accelerometer_bias_random_walk_mps3_sqrt_hz
         << ", \"gyroscope_bias_random_walk_radps2_sqrt_hz\": "
         << graph.imu.gyroscope_bias_random_walk_radps2_sqrt_hz
         << ", \"integration_noise_density\": " << graph.imu.integration_noise_density
         << ", \"preintegration_accelerometer_bias_variance_m2ps4\": "
         << graph.imu.preintegration_accelerometer_bias_variance_m2ps4
         << ", \"preintegration_gyroscope_bias_variance_rad2ps2\": "
         << graph.imu.preintegration_gyroscope_bias_variance_rad2ps2
         << ", \"initial_accelerometer_bias_mean_mps2\": ["
         << graph.imu.initial_accelerometer_bias_mean_mps2.x() << ", "
         << graph.imu.initial_accelerometer_bias_mean_mps2.y() << ", "
         << graph.imu.initial_accelerometer_bias_mean_mps2.z() << ']'
         << ", \"initial_gyroscope_bias_mean_radps\": ["
         << graph.imu.initial_gyroscope_bias_mean_radps.x() << ", "
         << graph.imu.initial_gyroscope_bias_mean_radps.y() << ", "
         << graph.imu.initial_gyroscope_bias_mean_radps.z() << ']'
         << ", \"initial_accelerometer_bias_sigma_mps2\": "
         << graph.imu.initial_accelerometer_bias_sigma_mps2
         << ", \"initial_gyroscope_bias_sigma_radps\": "
         << graph.imu.initial_gyroscope_bias_sigma_radps << "}},\n"
         << "      \"lidar_preprocessing\": ";
  writeLidarPreprocessConfig(output, preprocessing);
  output << ",\n"
         << "      \"rolling_target\": {\"odom_epoch\": " << target.odom_epoch.value()
         << ", \"maximum_retained_sweeps\": " << target.maximum_retained_sweeps
         << ", \"maximum_retained_points\": " << target.maximum_retained_points
         << ", \"selection_policy\": \"newest_age_spread\", \"registration\": ";
  writeLidarRegistrationConfig(output, target.registration);
  output << "},\n"
         << "      \"finalized_lidar_target\": ";
  writeFinalizedLidarTargetMapConfig(output, config.finalized_lidar_target);
  output << ",\n"
         << "      \"lidar_registration\": ";
  writeLidarRegistrationConfig(output, registration);
  const double lidar_base_covariance_inflation = config.lidar_target_reuse_covariance_inflation *
                                                 config.lidar_imu_conditioning_covariance_inflation;
  output << ",\n"
         << "      \"lidar_factor_correlation\": {\"live_only_policy_revision\": 1, "
         << "\"mixed_or_finalized_map_policy_revision\": 3, "
         << "\"target_reuse_covariance_inflation\": "
         << config.lidar_target_reuse_covariance_inflation
         << ", \"imu_conditioning_covariance_inflation\": "
         << config.lidar_imu_conditioning_covariance_inflation
         << ", \"base_covariance_inflation\": " << lidar_base_covariance_inflation
         << ", \"finalized_map_correlation_inflation_floor\": "
         << config.finalized_map_correlation_inflation_floor
         << ", \"live_information_scale_formula\": " << jsonString("1 / base_covariance_inflation")
         << ", \"live_translation_information_cap\": "
         << registration.maximum_translation_information / lidar_base_covariance_inflation
         << ", \"effective_finalized_map_covariance_inflation_formula\": "
         << jsonString(
                "max(owner_pose_covariance_inflation, "
                "finalized_map_correlation_inflation_floor)")
         << ", \"finalized_map_covariance_inflation_formula\": "
         << jsonString(
                "base_covariance_inflation * "
                "effective_finalized_map_covariance_inflation")
         << ", \"finalized_map_information_scale_formula\": "
         << jsonString(
                "1 / (base_covariance_inflation * "
                "effective_finalized_map_covariance_inflation)")
         << ", \"finalized_map_translation_information_cap_formula\": "
         << jsonString(
                "maximum_translation_information / "
                "finalized_map_covariance_inflation")
         << "},\n"
         << "      \"visual_cameras\": [";
  for (std::size_t index = 0U; index < config.visual_cameras.size(); ++index) {
    const local::VisualCameraConfig& camera = config.visual_cameras[index];
    output << (index == 0U ? "\n" : ",\n") << "        {\"camera_id\": " << camera.camera.value()
           << ", \"lane\": ";
    writeVisualLaneConfig(output, camera.lane);
    output << '}';
  }
  output << (config.visual_cameras.empty() ? "" : "\n      ") << "]\n";
}

void writeReplayProfile(std::ostream& output, const tools::ReplayProfile& profile) {
  output << "      \"name\": " << jsonString(profile.name()) << ",\n"
         << "      \"session_id\": " << profile.session().value() << ",\n"
         << "      \"config_revision\": " << profile.config().value() << ",\n"
         << "      \"calibration_epoch\": " << profile.calibration().value() << ",\n"
         << "      \"max_events\": " << profile.bounds().max_events << ",\n"
         << "      \"max_bag_messages\": " << profile.bounds().max_bag_messages << ",\n"
         << "      \"sources\": [\n";
  for (std::size_t index = 0; index < profile.sources().size(); ++index) {
    const tools::ReplaySource& source = profile.sources()[index];
    std::visit(
        [&](const auto& typed) {
          const core::AffineClockModel& clock = typed.source.clock;
          output << "        {\"topic\": " << jsonString(typed.topic)
                 << ", \"source_name\": " << jsonString(typed.source.name)
                 << ", \"producer_id\": " << typed.source.producer.value()
                 << ", \"required\": " << (typed.required ? "true" : "false")
                 << ", \"clock\": {\"revision\": " << clock.revision.value()
                 << ", \"source_epoch\": " << clock.source_epoch.value()
                 << ", \"raw_reference_ns\": " << clock.raw_reference.nanoseconds
                 << ", \"fusion_reference_ns\": " << clock.fusion_reference.nanoseconds
                 << ", \"rate\": " << std::setprecision(17) << clock.rate
                 << ", \"uncertainty_ns\": " << clock.uncertainty.nanoseconds
                 << ", \"status\": " << static_cast<int>(clock.status) << '}';
          using Source = std::decay_t<decltype(typed)>;
          if constexpr (std::is_same_v<Source, tools::LidarReplaySource>) {
            output << ", \"lidar_id\": " << typed.lidar.value()
                   << ", \"header_to_point_time_origin_ns\": "
                   << typed.header_to_point_time_origin.nanoseconds;
          } else if constexpr (std::is_same_v<Source, tools::CameraReplaySource>) {
            output << ", \"camera_id\": " << typed.camera.value() << ", \"wire_format\": "
                   << jsonString(typed.wire_format == tools::CameraWireFormat::Image
                                     ? "image"
                                     : "compressed_image")
                   << ", \"stamp_to_exposure_midpoint_ns\": "
                   << typed.stamp_to_exposure_midpoint.nanoseconds
                   << ", \"exposure_ns\": " << typed.exposure.nanoseconds;
          }
          output << '}';
        },
        source);
    output << (index + 1U == profile.sources().size() ? "\n" : ",\n");
  }
  output << "      ]\n";
}

[[nodiscard]] std::string_view cameraTimestampReferenceName(
    core::CameraTimestampReference reference) noexcept {
  switch (reference) {
    case core::CameraTimestampReference::SensorReportedImageTime:
      return "sensor_reported_image_time";
    case core::CameraTimestampReference::ExposureMidpoint:
      return "exposure_midpoint";
  }
  return "unknown";
}

void writeCalibrationProfile(std::ostream& output, const core::CalibrationBundle& calibration) {
  const core::ImuCalibration& imu = calibration.imu();
  const core::ImuNoiseModel& noise = imu.noise();
  const core::LidarCalibration& lidar = calibration.lidar();
  output << std::setprecision(17) << "{\"epoch\": " << calibration.epoch().value()
         << ", \"imu\": {\"name\": " << jsonString(imu.name())
         << ", \"topic\": " << jsonString(imu.sourceTopic())
         << ", \"nominal_rate_hz\": " << imu.nominalRateHz()
         << ", \"gravity_mps2\": " << imu.gravityMagnitude()
         << ", \"accelerometer_noise_density\": " << noise.accelerometerNoiseDensity()
         << ", \"gyroscope_noise_density\": " << noise.gyroscopeNoiseDensity()
         << ", \"accelerometer_bias_random_walk\": " << noise.accelerometerBiasRandomWalk()
         << ", \"gyroscope_bias_random_walk\": " << noise.gyroscopeBiasRandomWalk()
         << "}, \"T_base_imu\": ";
  writePoseMatrix(output, calibration.baseFromImu().T_base_imu());
  output << ", \"lidar\": {\"id\": " << lidar.id().value()
         << ", \"name\": " << jsonString(lidar.name())
         << ", \"topic\": " << jsonString(lidar.sourceTopic())
         << ", \"sweep_timestamp_reference\": \"sweep_start\""
         << ", \"point_time_convention\": \"offset_from_sweep_timestamp\""
         << ", \"T_imu_lidar\": ";
  writePoseMatrix(output, lidar.extrinsics().T_imu_lidar());
  output << "}, \"cameras\": [";
  for (std::size_t index = 0U; index < calibration.cameras().size(); ++index) {
    const core::CameraCalibration& camera = calibration.cameras()[index];
    const core::PinholeEquidistantCameraModel& model = camera.model();
    const core::ImageDimensions dimensions = model.imageSize();
    output << (index == 0U ? "\n" : ",\n") << "      {\"id\": " << camera.id().value()
           << ", \"name\": " << jsonString(camera.name())
           << ", \"topic\": " << jsonString(camera.sourceTopic())
           << ", \"timing\": {\"timestamp_reference\": "
           << jsonString(cameraTimestampReferenceName(camera.timing().timestampReference()))
           << ", \"convention\": "
              "\"imu_time_equals_camera_time_plus_offset\""
           << ", \"imu_time_minus_camera_time_ns\": "
           << camera.timing().imuTimeMinusCameraTime().nanoseconds
           << "}, \"model\": {\"projection\": \"pinhole\", "
              "\"distortion_model\": \"equidistant\", \"width\": "
           << dimensions.width << ", \"height\": " << dimensions.height << ", \"intrinsics\": ["
           << model.fx() << ", " << model.fy() << ", " << model.cx() << ", " << model.cy()
           << "], \"distortion\": [";
    for (std::size_t coefficient = 0U; coefficient < model.distortion().size(); ++coefficient) {
      output << (coefficient == 0U ? "" : ", ") << model.distortion()[coefficient];
    }
    output << "]}, \"T_imu_camera\": ";
    writePoseMatrix(output, camera.extrinsics().T_imu_camera());
    output << '}';
  }
  output << (calibration.cameras().empty() ? "" : "\n    ") << "]}";
}

void writeSolverGlobalizationRunCounters(std::ostream& output,
                                         const SolverGlobalizationRunCounters& counters) {
  output << "{\"transactions\": " << counters.transactions
         << ", \"full_steps_rejected\": " << counters.full_steps_rejected
         << ", \"gauss_newton_backtracking_trials\": " << counters.backtracking_trials
         << ", \"cauchy_directions_attempted\": " << counters.cauchy_directions_attempted
         << ", \"cauchy_steps_accepted\": " << counters.cauchy_steps_accepted
         << ", \"cauchy_backtracking_trials\": " << counters.cauchy_backtracking_trials
         << ", \"zero_step_terminations\": " << counters.zero_step_terminations
         << ", \"minimum_step_scale\": ";
  if (counters.transactions > 0U) {
    output << counters.minimum_step_scale;
  } else {
    output << "null";
  }
  output << '}';
}

void writeFinalizedLidarTargetRunCounters(std::ostream& output,
                                          const FinalizedLidarTargetRunCounters& counters) {
  output << "{\"process_reports\": " << counters.process_reports
         << ", \"finality_matches\": " << counters.finality_matches
         << ", \"rollback_removals\": " << counters.rollback_removals
         << ", \"insertions\": {\"transactions\": " << counters.insertion_transactions
         << ", \"input_points\": " << counters.insertion_input_points
         << ", \"insertion_voxels\": " << counters.insertion_voxels
         << ", \"insertion_selection_discarded_points\": "
         << counters.insertion_selection_discarded_points
         << ", \"minimum_separation_discarded_points\": "
         << counters.minimum_separation_discarded_points
         << ", \"query_voxel_capacity_discarded_points\": "
         << counters.query_voxel_capacity_discarded_points
         << ", \"admitted_points\": " << counters.inserted_points
         << ", \"touched_query_voxels\": " << counters.touched_query_voxels << "}"
         << ", \"pruning\": {\"reports\": " << counters.prune_reports
         << ", \"examined_query_voxels\": " << counters.prune_examined_query_voxels
         << ", \"examined_points\": " << counters.prune_examined_points
         << ", \"removed_query_voxels\": " << counters.pruned_query_voxels
         << ", \"removed_points\": " << counters.pruned_points << "}"
         << ", \"capacity\": {\"recovery_attempts\": " << counters.capacity_recovery_attempts
         << ", \"recovery_successes\": " << counters.capacity_recovery_successes
         << ", \"skipped_sweeps\": " << counters.capacity_skipped_sweeps
         << ", \"retry_suppressed_sweeps\": " << counters.capacity_retry_suppressed_sweeps
         << ", \"last_skip\": ";
  if (counters.last_capacity_skip) {
    const local::FinalizedLidarTargetCapacitySkip& skip = *counters.last_capacity_skip;
    output << "{\"reason\": " << jsonString(finalizedLidarTargetCapacitySkipReasonName(skip.reason))
           << ", \"batch_id\": " << skip.batch.batch_id.value() << ", \"sensor_modality\": "
           << jsonString(sensorModalityName(skip.batch.sensor.modality))
           << ", \"sensor_instance\": " << skip.batch.sensor.instance
           << ", \"state_id\": " << skip.state.value()
           << ", \"exact_time_ns\": " << skip.exact_time.nanoseconds
           << ", \"final_revision\": " << skip.final_revision.value()
           << ", \"sweep_id\": " << skip.sweep.value()
           << ", \"cloud_checksum\": " << jsonString(core::sha256Hex(skip.cloud_checksum))
           << ", \"input_points\": " << skip.input_points
           << ", \"retained_points\": " << skip.retained_points
           << ", \"hard_point_capacity\": " << skip.hard_point_capacity
           << ", \"map_version\": " << skip.map_version
           << ", \"map_checksum\": " << jsonString(core::sha256Hex(skip.map_checksum)) << '}';
  } else {
    output << "null";
  }
  output << "}"
         << ", \"freeze\": {\"frozen_process_reports\": " << counters.frozen_process_reports
         << ", \"transitions_into_frozen\": " << counters.freeze_transitions << "}"
         << ", \"final\": {\"pending_sweeps\": " << counters.pending_sweeps
         << ", \"pending_unfinalized_sweeps\": " << counters.pending_unfinalized_sweeps
         << ", \"finalized_ready_sweeps\": " << counters.finalized_ready_sweeps
         << ", \"insertion_frozen\": " << (counters.insertion_frozen ? "true" : "false")
         << ", \"capacity_saturated\": " << (counters.capacity_saturated ? "true" : "false")
         << ", \"capacity_skips_since_retry\": " << counters.capacity_skips_since_retry
         << ", \"lidar_health\": " << jsonString(sensorHealthStateName(counters.lidar_health))
         << ", \"retained_points\": " << counters.retained_points
         << ", \"map_version\": " << counters.map_version
         << ", \"map_checksum\": " << jsonString(core::sha256Hex(counters.map_checksum)) << "}}";
}

[[nodiscard]] bool writeRunReport(
    const BagLocalizationOptions& options, const std::filesystem::path& report_path,
    const core::CalibrationBundle* calibration, const tools::ReplayProfile* profile,
    const local::LocalEstimatorConfig* estimator_config, const tools::ReplayStats& replay_stats,
    std::optional<tools::ReplayCompletion> replay_completion,
    const tools::ScheduledReplayRuntimeReport* scheduled_runtime,
    const DriverTimingReport* driver_timing,
    const local::LocalEstimatorStatistics* estimator_statistics, const RunCounters& counters,
    const StateCoverage& coverage, local::LocalEstimatorLifecycle lifecycle,
    std::optional<core::ArrivalTime> first_recorded_arrival,
    std::optional<core::ArrivalTime> last_recorded_arrival, std::int64_t wall_start_ns,
    std::int64_t wall_end_ns, const std::optional<Failure>& failure, bool execution_succeeded,
    bool localization_available, bool full_bag_consumed) {
  std::ofstream output(report_path, std::ios::out | std::ios::trunc);
  if (!output) {
    return false;
  }
  const double elapsed_seconds = static_cast<double>(wall_end_ns - wall_start_ns) * 1.0e-9;
  const bool has_processed_seconds =
      first_recorded_arrival.has_value() && last_recorded_arrival.has_value();
  const double processed_seconds = has_processed_seconds
                                       ? static_cast<double>(last_recorded_arrival->nanoseconds -
                                                             first_recorded_arrival->nanoseconds) *
                                             1.0e-9
                                       : 0.0;
  const std::optional<double> state_coverage_seconds =
      coverage.first_time && coverage.last_time
          ? std::optional<double>(static_cast<double>(coverage.last_time->nanoseconds -
                                                      coverage.first_time->nanoseconds) *
                                  1.0e-9)
          : std::nullopt;
  const SensorModeAudit sensor_mode_audit =
      makeSensorModeAudit(options, calibration, profile, estimator_config, replay_stats, counters);
  output << std::setprecision(17) << "{\n"
         << "  \"schema_version\": 27,\n"
         << "  \"application\": \"meridian_bag_localize\",\n"
         << "  \"execution_succeeded\": " << (execution_succeeded ? "true" : "false") << ",\n"
         << "  \"localization_available\": " << (localization_available ? "true" : "false") << ",\n"
         << "  \"full_bag_consumed\": " << (full_bag_consumed ? "true" : "false") << ",\n"
         << "  \"partial_coverage\": " << (full_bag_consumed ? "false" : "true") << ",\n"
         << "  \"input\": {\"bag_uri\": " << jsonString(options.bag_uri.string())
         << ", \"calibration_root\": " << jsonString(options.calibration_root.string())
         << ", \"collection\": "
         << (options.collection == ros::NewerCollegeCollection::Collection1 ? 1 : 2)
         << ", \"output_directory\": " << jsonString(options.output_directory.string())
         << ", \"sensor_mode\": " << jsonString(sensorModeName(options.sensor_mode))
         << ", \"initialization\": "
         << jsonString(initializationModeName(options.initialization_mode))
         << ", \"replay_mode\": " << jsonString(replayModeName(options.replay.mode))
         << ", \"playback_rate\": ";
  if (replayIsUnpaced(options.replay.mode)) {
    output << "null";
  } else {
    output << options.replay.playback_rate;
  }
  output << ", \"visual_graph\": ";
  if (sensorModeIncludesCameras(options.sensor_mode)) {
    output << jsonString(effectiveVisualGraphEnabled(options) ? "enabled" : "disabled");
  } else {
    output << "\"not_applicable\"";
  }
  output << "},\n"
         << "  \"pipeline_runtime\": {\"mode\": " << jsonString(replayModeName(options.replay.mode))
         << ", \"schedule\": {\"playback_rate\": ";
  if (replayIsUnpaced(options.replay.mode)) {
    output << "null";
  } else {
    output << options.replay.playback_rate;
  }
  output << ", \"queue_count_capacity\": " << options.replay.queue_count_capacity
         << ", \"queue_byte_capacity\": " << options.replay.queue_byte_capacity
         << ", \"timing_window_capacity\": " << options.replay.timing_window_capacity
         << "}, \"scheduled\": ";
  if (scheduled_runtime) {
    writeScheduledReplayRuntime(output, *scheduled_runtime);
  } else {
    output << "null";
  }
  output << ", \"driver_stages\": ";
  if (driver_timing) {
    writeDriverTimingReport(output, *driver_timing);
  } else {
    output << "null";
  }
  output << ", \"local_stages\": ";
  if (driver_timing) {
    writeLocalPipelineTimingReport(output, driver_timing->local_pipeline);
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"sensor_mode_proof\": {\n"
         << "    \"calibration_available\": "
         << (sensor_mode_audit.calibration_available ? "true" : "false")
         << ", \"profile_available\": " << (sensor_mode_audit.profile_available ? "true" : "false")
         << ", \"estimator_config_available\": "
         << (sensor_mode_audit.estimator_config_available ? "true" : "false") << ",\n"
         << "    \"expected\": {\"imu_topics\": " << sensor_mode_audit.expected_imu_topics
         << ", \"lidar_topics\": " << sensor_mode_audit.expected_lidar_topics
         << ", \"camera_topics\": " << sensor_mode_audit.expected_camera_topics << "},\n"
         << "    \"configured\": {\"imu_topics\": " << sensor_mode_audit.configured_imu_topics
         << ", \"lidar_topics\": " << sensor_mode_audit.configured_lidar_topics
         << ", \"camera_topics\": " << sensor_mode_audit.configured_camera_topics
         << ", \"gnss_topics\": " << sensor_mode_audit.configured_gnss_topics
         << ", \"visual_lanes\": " << sensor_mode_audit.configured_visual_lanes << "},\n"
         << "    \"runtime\": {\"camera_topics_with_stats\": "
         << sensor_mode_audit.camera_topics_with_runtime_stats
         << ", \"camera_messages_seen\": " << sensor_mode_audit.camera_messages_seen
         << ", \"camera_image_decode_errors\": " << sensor_mode_audit.camera_image_decode_errors
         << ", \"camera_events_emitted\": " << sensor_mode_audit.camera_events_emitted
         << ", \"camera_domain_events\": " << counters.camera_events
         << ", \"camera_ingest_accepted\": " << counters.camera_ingest_accepted
         << ", \"camera_ingest_rejected\": " << counters.camera_ingest_rejected
         << ", \"lidar_topics_with_stats\": " << sensor_mode_audit.lidar_topics_with_runtime_stats
         << ", \"lidar_messages_seen\": " << sensor_mode_audit.lidar_messages_seen
         << ", \"lidar_events_emitted\": " << sensor_mode_audit.lidar_events_emitted
         << ", \"lidar_domain_events\": " << counters.lidar_events
         << ", \"lidar_enqueue_accepted\": " << counters.lidar_enqueue_accepted
         << ", \"lidar_enqueue_rejected\": " << counters.lidar_enqueue_rejected
         << ", \"bag_messages_read\": " << replay_stats.bag_messages_read
         << ", \"configured_messages_seen\": " << replay_stats.configured_messages_seen
         << ", \"unknown_topic_count\": " << replay_stats.unknown_topics.size()
         << ", \"storage_filter_matches\": "
         << (sensor_mode_audit.storage_filter_matches ? "true" : "false")
         << ", \"unexpected_events\": " << counters.unexpected_events << "},\n"
         << "    \"configuration_matches\": "
         << (sensor_mode_audit.configuration_matches ? "true" : "false")
         << ", \"runtime_isolation_matches\": "
         << (sensor_mode_audit.runtime_isolation_matches ? "true" : "false")
         << ", \"passed\": " << (sensor_mode_audit.passed ? "true" : "false") << "\n"
         << "  },\n"
         << "  \"trajectory_outputs\": {\n"
         << "    \"imu\": {\"path\": "
         << jsonString((options.output_directory / "trajectory_imu.tum").string())
         << ", \"world_frame\": \"odom\", \"body_frame\": \"alphasense_imu\", "
            "\"pose_convention\": \"T_odom_imu maps IMU coordinates into odom\", "
            "\"timestamp_reference\": \"exact navigation-state time\", "
            "\"calibration_epoch\": ";
  if (calibration) {
    output << calibration->epoch().value();
  } else {
    output << "null";
  }
  output << ", \"applied_fixed_transform\": \"identity\"},\n"
         << "    \"base\": {\"path\": "
         << jsonString((options.output_directory / "trajectory_base.tum").string())
         << ", \"world_frame\": \"odom\", \"body_frame\": \"base_link\", "
            "\"pose_convention\": \"T_odom_base maps base coordinates into odom\", "
            "\"timestamp_reference\": \"exact navigation-state time\", "
            "\"calibration_epoch\": ";
  if (calibration) {
    output << calibration->epoch().value();
  } else {
    output << "null";
  }
  output << ", \"applied_fixed_transform_name\": \"T_imu_base=inverse(T_base_imu)\", "
            "\"T_imu_base_matrix\": ";
  if (calibration) {
    writePoseMatrix(output, calibration->baseFromImu().T_base_imu().inverse());
  } else {
    output << "null";
  }
  output << "},\n"
         << "    \"fixed_lag_imu\": {\"path\": "
         << jsonString((options.output_directory / "trajectory_imu_fixed_lag.tum").string())
         << ", \"world_frame\": \"odom\", \"body_frame\": \"alphasense_imu\", "
            "\"timestamp_reference\": \"finalized fixed-lag state plus active tail at replay "
            "end\"},\n"
         << "    \"fixed_lag_base\": {\"path\": "
         << jsonString((options.output_directory / "trajectory_base_fixed_lag.tum").string())
         << ", \"world_frame\": \"odom\", \"body_frame\": \"base_link\", "
            "\"timestamp_reference\": \"finalized fixed-lag state plus active tail at replay "
            "end\"},\n"
         << "    \"navigation_diagnostics\": {\"path\": "
         << jsonString((options.output_directory / "navigation_diagnostics.csv").string())
         << ", \"schema\": \"meridian_navigation_diagnostics_csv_v19\", "
            "\"pose_frame\": \"odom<-alphasense_imu\", "
            "\"covariance_order\": "
            "\"rotation_velocity_position_gyro_bias_accel_bias\"}\n"
         << "  },\n"
         << "  \"timing\": {\"wall_start_unix_ns\": " << wall_start_ns
         << ", \"wall_end_unix_ns\": " << wall_end_ns
         << ", \"elapsed_seconds\": " << elapsed_seconds << ", \"processed_bag_seconds\": ";
  if (has_processed_seconds) {
    output << processed_seconds;
  } else {
    output << "null";
  }
  output << ", \"real_time_factor\": ";
  if (has_processed_seconds && elapsed_seconds > 0.0) {
    output << processed_seconds / elapsed_seconds;
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"replay_completion\": "
         << (replay_completion ? jsonString(replayCompletionName(*replay_completion)) : "null")
         << ",\n"
         << "  \"lifecycle\": " << jsonString(lifecycleName(lifecycle)) << ",\n"
         << "  \"failure\": ";
  if (failure) {
    output << "{\"stage\": " << jsonString(failure->stage)
           << ", \"detail\": " << jsonString(failure->detail) << ", \"code\": ";
    if (failure->code) {
      output << *failure->code;
    } else {
      output << "null";
    }
    output << ", \"substage\": ";
    if (failure->substage) {
      output << *failure->substage;
    } else {
      output << "null";
    }
    output << '}';
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"coverage\": {\"states_written\": " << coverage.states_written
         << ", \"first_state_id\": ";
  if (coverage.first_state) {
    output << coverage.first_state->value();
  } else {
    output << "null";
  }
  output << ", \"last_state_id\": ";
  if (coverage.last_state) {
    output << coverage.last_state->value();
  } else {
    output << "null";
  }
  output << ", \"first_state_time_ns\": ";
  if (coverage.first_time) {
    output << coverage.first_time->nanoseconds;
  } else {
    output << "null";
  }
  output << ", \"last_state_time_ns\": ";
  if (coverage.last_time) {
    output << coverage.last_time->nanoseconds;
  } else {
    output << "null";
  }
  output << ", \"state_coverage_seconds\": ";
  if (state_coverage_seconds) {
    output << *state_coverage_seconds;
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"event_counts\": {\"received\": " << counters.events_received
         << ", \"imu\": " << counters.imu_events << ", \"lidar\": " << counters.lidar_events
         << ", \"camera\": " << counters.camera_events
         << ", \"unexpected\": " << counters.unexpected_events
         << ", \"imu_ingest_accepted\": " << counters.imu_ingest_accepted
         << ", \"imu_ingest_rejected\": " << counters.imu_ingest_rejected
         << ", \"lidar_enqueue_accepted\": " << counters.lidar_enqueue_accepted
         << ", \"lidar_enqueue_rejected\": " << counters.lidar_enqueue_rejected
         << ", \"camera_ingest_accepted\": " << counters.camera_ingest_accepted
         << ", \"camera_ingest_rejected\": " << counters.camera_ingest_rejected << "},\n"
         << "  \"camera_counts\": {\"late_for_graph\": " << counters.camera_frames_late_for_graph
         << ", \"imu_rotation_seeded\": " << counters.camera_rotation_seeds_provided
         << ", \"tracking_only\": " << counters.camera_tracking_only
         << ", \"tracking_only_graph_submission_disabled\": "
         << counters.camera_tracking_only_graph_submission_disabled
         << ", \"tracking_only_without_local_state\": "
         << counters.camera_tracking_only_without_local_state
         << ", \"keyframe_requests\": " << counters.camera_keyframe_requests
         << ", \"keyframes_suppressed_by_capacity\": "
         << counters.camera_keyframes_suppressed_by_capacity
         << ", \"keyframes_suppressed_by_timeline\": "
         << counters.camera_keyframes_suppressed_by_timeline
         << ", \"keyframes_rejected_by_timeline\": "
         << counters.camera_keyframes_rejected_by_timeline << ", \"per_camera\": {";
  bool first_camera = true;
  for (const auto& [camera_id, camera] : counters.cameras) {
    output << (first_camera ? "\n" : ",\n") << "    " << jsonString(std::to_string(camera_id))
           << ": {\"events\": " << camera.events
           << ", \"ingest_accepted\": " << camera.ingest_accepted
           << ", \"ingest_rejected\": " << camera.ingest_rejected
           << ", \"late_for_graph\": " << camera.late_for_graph
           << ", \"imu_rotation_seeded\": " << camera.imu_rotation_seeded
           << ", \"tracking_only\": " << camera.tracking_only
           << ", \"tracking_only_graph_submission_disabled\": "
           << camera.tracking_only_graph_submission_disabled
           << ", \"tracking_only_without_local_state\": "
           << camera.tracking_only_without_local_state
           << ", \"keyframe_requests\": " << camera.keyframe_requests
           << ", \"keyframes_suppressed_by_capacity\": " << camera.keyframes_suppressed_by_capacity
           << ", \"keyframes_suppressed_by_timeline\": " << camera.keyframes_suppressed_by_timeline
           << ", \"keyframes_rejected_by_timeline\": " << camera.keyframes_rejected_by_timeline
           << ", \"graph_degradations\": " << camera.graph_degradations << '}';
    first_camera = false;
  }
  output << (first_camera ? "" : "\n  ") << "}},\n"
         << "  \"processing_counts\": {\"process_calls\": " << counters.process_calls
         << ", \"process_errors\": " << counters.process_errors
         << ", \"initialization_rejections_reported\": "
         << counters.initialization_rejections_reported
         << ", \"stationary_initializations\": " << counters.stationary_initializations
         << ", \"motion_initializations\": " << counters.motion_initializations
         << ", \"initialization_bootstrap_anchors\": " << counters.initialization_bootstrap_anchors
         << ", \"initialization_bootstrap_increments\": "
         << counters.initialization_bootstrap_increments
         << ", \"initialization_bootstrap_rejections\": "
         << counters.initialization_bootstrap_rejections
         << ", \"last_initialization_rejection_stage\": ";
  if (counters.last_initialization_rejection_stage) {
    output << *counters.last_initialization_rejection_stage;
  } else {
    output << "null";
  }
  output << ", \"last_initialization_rejection_code\": ";
  if (counters.last_initialization_rejection_code) {
    output << *counters.last_initialization_rejection_code;
  } else {
    output << "null";
  }
  output << ", \"last_initialization_rejection_detail\": "
         << (counters.last_initialization_rejection_detail.empty()
                 ? "null"
                 : jsonString(counters.last_initialization_rejection_detail))
         << ", \"last_stationary_rejection\": ";
  if (counters.last_stationary_rejection_code) {
    output << "{\"code\": " << *counters.last_stationary_rejection_code
           << ", \"detail\": " << jsonString(counters.last_stationary_rejection_detail) << '}';
  } else {
    output << "null";
  }
  output << ", \"last_bootstrap_rejection\": ";
  if (counters.last_bootstrap_rejection_code) {
    output << "{\"code\": " << *counters.last_bootstrap_rejection_code
           << ", \"detail\": " << jsonString(counters.last_bootstrap_rejection_detail) << '}';
  } else {
    output << "null";
  }
  output << ", \"last_motion_rejection\": ";
  if (counters.last_motion_rejection_code) {
    output << "{\"code\": " << *counters.last_motion_rejection_code << ", \"segment\": ";
    if (counters.last_motion_rejection_segment) {
      output << *counters.last_motion_rejection_segment;
    } else {
      output << "null";
    }
    output << ", \"detail\": " << jsonString(counters.last_motion_rejection_detail) << '}';
  } else {
    output << "null";
  }
  output << ", \"lidar_drops_reported\": " << counters.lidar_drops_reported
         << ", \"pending_sweeps_at_end\": " << counters.pending_sweeps_at_end
         << ", \"pending_camera_knots_at_end\": " << counters.pending_camera_knots_at_end
         << ", \"waiting_for_future_imu_reports\": " << counters.waiting_for_future_imu_reports
         << ", \"camera_commit_reports\": " << counters.camera_commit_reports
         << ", \"visual_keyframes_resolved\": " << counters.visual_keyframes_resolved
         << ", \"visual_only_commits\": " << counters.visual_only_commits
         << ", \"shared_camera_lidar_commits\": " << counters.shared_camera_lidar_commits
         << ", \"commit_references_deduplicated\": " << counters.commit_references_deduplicated
         << ", \"factor_only_lidar_reports\": " << counters.factor_only_lidar_reports
         << ", \"factor_only_graph_revisions\": " << counters.factor_only_graph_revisions
         << ", \"factor_only_unchanged_graph_references\": "
         << counters.factor_only_unchanged_graph_references
         << ", \"visual_graph_attachments\": " << counters.visual_graph_attachments
         << ", \"visual_landmarks_attached\": " << counters.visual_landmarks_attached
         << ", \"visual_factors_attached\": " << counters.visual_factors_attached
         << ", \"visual_factors_retired\": " << counters.visual_factors_retired
         << ", \"visual_graph_degradations\": " << counters.visual_graph_degradations
         << ", \"visual_factor_batches_discarded\": " << counters.visual_factor_batches_discarded
         << ", \"visual_landmark_seeds_discarded\": " << counters.visual_landmark_seeds_discarded
         << ", \"visual_factor_specs_discarded\": " << counters.visual_factor_specs_discarded
         << ", \"visual_stale_track_observations_discarded\": "
         << counters.visual_stale_track_observations_discarded << "},\n"
         << "  \"local_solver_globalization\": {\"committed\": ";
  writeSolverGlobalizationRunCounters(output, counters.committed_solver_globalization);
  output << ", \"rejected\": ";
  writeSolverGlobalizationRunCounters(output, counters.rejected_solver_globalization);
  output << "},\n"
         << "  \"finalized_lidar_target\": ";
  writeFinalizedLidarTargetRunCounters(output, counters.finalized_lidar_target);
  output << ",\n"
         << "  \"registration_counts\": {\"bootstrap_commits\": " << counters.bootstrap_commits
         << ", \"registered_commits\": " << counters.registered_commits
         << ", \"registered_tracking_only_commits\": " << counters.registered_tracking_only_commits
         << ", \"target_state_unavailable_frozen_commits\": "
         << counters.target_state_unavailable_frozen_commits
         << ", \"registration_diagnostics\": " << counters.registration_diagnostics
         << ", \"active_registration_diagnostics\": " << counters.active_registration_diagnostics
         << ", \"health_quarantined_commits\": " << counters.health_quarantined_commits << "},\n"
         << "  \"lidar_registration_diagnostics\": {\"attempted\": "
         << counters.lidar_registrations_attempted
         << ", \"accepted\": " << counters.lidar_registrations_accepted
         << ", \"rejected\": " << counters.lidar_registrations_rejected
         << ", \"outer_iterations\": " << counters.lidar_registration_outer_iterations
         << ", \"accepted_steps\": " << counters.lidar_registration_accepted_steps
         << ", \"rejected_frozen_cost_trials\": " << counters.lidar_registration_rejected_trials
         << ", \"accepted_registration_direct_graph_rejections\": "
         << counters.accepted_registration_direct_graph_rejections
         << ", \"accepted_registration_health_quarantines\": "
         << counters.accepted_registration_health_quarantines
         << ", \"accepted_finalized_map_reports\": " << counters.accepted_finalized_map_reports
         << ", \"rejected_finalized_map_reports\": " << counters.rejected_finalized_map_reports
         << ", \"terminations\": {"
         << "\"converged\": "
         << (counters.lidar_registration_terminations.contains(
                 local::LidarRegistrationTermination::Converged)
                 ? counters.lidar_registration_terminations.at(
                       local::LidarRegistrationTermination::Converged)
                 : 0U)
         << ", \"iteration_limit_reached\": "
         << (counters.lidar_registration_terminations.contains(
                 local::LidarRegistrationTermination::IterationLimitReached)
                 ? counters.lidar_registration_terminations.at(
                       local::LidarRegistrationTermination::IterationLimitReached)
                 : 0U);
  const auto registration_error_count = [&](local::LidarRegistrationErrorCode code) {
    const auto found = counters.lidar_registration_errors.find(code);
    return found == counters.lidar_registration_errors.end() ? std::size_t{0U} : found->second;
  };
  output << "}, \"errors\": {\"invalid_config\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InvalidConfig)
         << ", \"invalid_source\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InvalidSource)
         << ", \"invalid_target\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InvalidTarget)
         << ", \"inconsistent_seed\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InconsistentSeed)
         << ", \"insufficient_correspondences\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InsufficientCorrespondences)
         << ", \"insufficient_observable_rank\": "
         << registration_error_count(local::LidarRegistrationErrorCode::InsufficientObservableRank)
         << ", \"no_decreasing_step\": "
         << registration_error_count(local::LidarRegistrationErrorCode::NoDecreasingStep)
         << ", \"numerical_failure\": "
         << registration_error_count(local::LidarRegistrationErrorCode::NumericalFailure)
         << "}, \"last_registration\": ";
  if (counters.last_lidar_registration) {
    const local::DirectLidarRegistrationReport& registration = *counters.last_lidar_registration;
    output << "{\"termination\": "
           << jsonString(lidarRegistrationTerminationName(registration.termination))
           << ", \"initial_robust_cost\": " << registration.initial_robust_cost
           << ", \"final_robust_cost\": " << registration.final_robust_cost
           << ", \"diagnostics\": ";
    writeLidarRegistrationDiagnostics(output, registration.diagnostics);
    output << ", \"work\": ";
    writeLidarRegistrationWork(output, registration.work);
    output << ", \"T_odom_source\": ";
    writePoseJson(output, registration.T_odom_source);
    output << ", \"source_right_correction\": ";
    writePoseJson(output, registration.source_right_correction);
    output << '}';
  } else {
    output << "null";
  }
  output << ", \"last_error\": ";
  if (counters.last_lidar_registration_error) {
    const local::LidarRegistrationError& error = *counters.last_lidar_registration_error;
    output << "{\"code\": " << jsonString(lidarRegistrationErrorCodeName(error.code))
           << ", \"detail\": " << jsonString(error.detail) << ", \"work\": ";
    writeLidarRegistrationWork(output, error.work);
    output << '}';
  } else {
    output << "null";
  }
  output << ", \"last_accepted_finalized_map\": ";
  if (counters.last_accepted_finalized_map) {
    writeDirectLidarFinalizedMapReport(output, *counters.last_accepted_finalized_map);
  } else {
    output << "null";
  }
  output << ", \"last_rejected_finalized_map\": ";
  if (counters.last_rejected_finalized_map) {
    writeDirectLidarFinalizedMapReport(output, *counters.last_rejected_finalized_map);
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"lidar_sensor_health\": {\"updates\": " << counters.lidar_health_updates
         << ", \"transitions\": " << counters.lidar_health_transitions
         << ", \"health_quarantined_commits\": " << counters.health_quarantined_commits
         << ", \"factor_batches_removed\": " << counters.lidar_factor_batches_removed
         << ", \"last_update\": ";
  if (counters.last_lidar_health_update) {
    writeSensorHealthUpdate(output, *counters.last_lidar_health_update);
  } else {
    output << "null";
  }
  output << "},\n"
         << "  \"degradation_counts\": {\"deskew\": " << counters.deskew_degraded_commits
         << ", \"preprocessing\": " << counters.preprocessing_degraded_commits
         << ", \"registration_target_retained\": " << counters.registration_degraded_commits
         << ", \"health_quarantined_target_retained\": " << counters.health_quarantined_commits
         << ", \"registration_target_frozen\": " << counters.target_state_unavailable_frozen_commits
         << ", \"visual_graph\": " << counters.visual_graph_degradations
         << ", \"last_visual_graph_error_code\": ";
  if (counters.last_visual_degradation_graph_error_code) {
    output << *counters.last_visual_degradation_graph_error_code;
  } else {
    output << "null";
  }
  output << ", \"last_visual_degradation_detail\": "
         << (counters.last_visual_degradation_detail.empty()
                 ? "null"
                 : jsonString(counters.last_visual_degradation_detail))
         << ", \"last_registration_graph_error_code\": ";
  if (counters.last_registration_rejection_code) {
    output << *counters.last_registration_rejection_code;
  } else {
    output << "null";
  }
  output << ", \"last_registration_rejection_detail\": "
         << (counters.last_registration_rejection_detail.empty()
                 ? "null"
                 : jsonString(counters.last_registration_rejection_detail))
         << ", \"registration_graph_error_codes\": {";
  bool first_registration_degradation = true;
  for (const auto& [code, count] : counters.registration_degradation_codes) {
    output << (first_registration_degradation ? "" : ", ") << jsonString(std::to_string(code))
           << ": " << count;
    first_registration_degradation = false;
  }
  output << "}},\n"
         << "  \"initialization_diagnostics\": ";
  if (counters.motion_initialization_diagnostics) {
    const auto& diagnostics = *counters.motion_initialization_diagnostics;
    output << "{\"method\": \"motion_lidar_imu\", \"pass\": "
           << jsonString(motionInitializationPassName(diagnostics.pass))
           << ", \"segments\": " << diagnostics.segments
           << ", \"imu_knots\": " << diagnostics.imu_knots
           << ", \"support_ns\": " << diagnostics.support.nanoseconds
           << ", \"rotation_excitation_rad\": " << diagnostics.rotation_excitation_rad
           << ", \"acceleration_excitation_mps2\": " << diagnostics.acceleration_excitation_mps2
           << ", \"scalar_dimension\": " << diagnostics.scalar_dimension
           << ", \"expected_data_rank\": " << diagnostics.expected_data_rank
           << ", \"data_rank\": " << diagnostics.data_rank
           << ", \"calibrated_data_rank\": " << diagnostics.calibrated_data_rank
           << ", \"full_rank\": " << diagnostics.full_rank
           << ", \"prior_resolved_accel_tilt_modes\": "
           << diagnostics.prior_resolved_accel_tilt_modes << ", \"observability_class\": "
           << jsonString(
                  motionInitializationObservabilityClassName(diagnostics.observability_class))
           << ", \"data_supported_condition\": " << diagnostics.data_supported_condition
           << ", \"calibrated_data_supported_condition\": "
           << diagnostics.calibrated_data_supported_condition
           << ", \"full_hessian_condition\": " << diagnostics.full_hessian_condition
           << ", \"initial_error\": " << diagnostics.initial_error
           << ", \"final_error\": " << diagnostics.final_error
           << ", \"lidar_error\": " << diagnostics.lidar_error
           << ", \"imu_error\": " << diagnostics.imu_error
           << ", \"bias_prior_error\": " << diagnostics.bias_prior_error
           << ", \"gauge_error\": " << diagnostics.gauge_error
           << ", \"lidar_residual_dimension\": " << diagnostics.lidar_residual_dimension
           << ", \"imu_residual_dimension\": " << diagnostics.imu_residual_dimension
           << ", \"bias_prior_residual_dimension\": " << diagnostics.bias_prior_residual_dimension
           << ", \"gauge_residual_dimension\": " << diagnostics.gauge_residual_dimension
           << ", \"total_residual_dimension\": " << diagnostics.total_residual_dimension
           << ", \"effective_degrees_of_freedom\": " << diagnostics.effective_degrees_of_freedom
           << ", \"lidar_msw\": " << diagnostics.lidar_mean_squared_whitened_residual
           << ", \"imu_msw\": " << diagnostics.imu_mean_squared_whitened_residual
           << ", \"reduced_chi_square\": " << diagnostics.reduced_chi_square
           << ", \"holdout_lidar_segments\": " << diagnostics.holdout_lidar_segments
           << ", \"holdout_residual_dimension\": " << diagnostics.holdout_residual_dimension
           << ", \"holdout_error\": " << diagnostics.holdout_error
           << ", \"holdout_msw\": " << diagnostics.holdout_mean_squared_whitened_residual
           << ", \"statistically_compatible\": "
           << (diagnostics.statistically_compatible ? "true" : "false")
           << ", \"conditioned_lidar_imu_approximation\": "
           << (diagnostics.conditioned_lidar_imu_approximation ? "true" : "false")
           << ", \"minimum_imu_conditioning_covariance_inflation\": "
           << diagnostics.minimum_imu_conditioning_covariance_inflation
           << ", \"covariance_residual_inflation\": " << diagnostics.covariance_residual_inflation
           << ", \"deskew_solve_passes\": " << diagnostics.deskew_solve_passes
           << ", \"refined_sweeps\": " << diagnostics.refined_sweeps
           << ", \"refined_registrations\": " << diagnostics.refined_registrations
           << ", \"refined_deskew_pose_interpolations\": "
           << diagnostics.refined_deskew_pose_interpolations
           << ", \"refined_total_registration_cost\": "
           << diagnostics.refined_total_registration_cost
           << ", \"refined_maximum_registration_cost\": "
           << diagnostics.refined_maximum_registration_cost
           << ", \"bias_prior_mahalanobis\": " << diagnostics.bias_prior_mahalanobis
           << ", \"solver_iterations\": " << diagnostics.solver_iterations << '}';
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"replay_stats\": {\n";
  writeReplayStats(output, replay_stats);
  output << "  },\n"
         << "  \"estimator_statistics\": ";
  if (estimator_statistics) {
    const auto& stats = *estimator_statistics;
    output << "{\"imu_samples_accepted\": " << stats.imu_samples_accepted
           << ", \"lidar_sweeps_enqueued\": " << stats.lidar_sweeps_enqueued
           << ", \"lidar_sweeps_dropped\": " << stats.lidar_sweeps_dropped
           << ", \"camera_frames_accepted\": " << stats.camera_frames_accepted
           << ", \"camera_frames_rejected\": " << stats.camera_frames_rejected
           << ", \"camera_frames_late_for_graph\": " << stats.camera_frames_late_for_graph
           << ", \"camera_rotation_seeds_provided\": " << stats.camera_rotation_seeds_provided
           << ", \"visual_keyframe_requests\": " << stats.visual_keyframe_requests
           << ", \"lidar_state_requests_suppressed_by_timeline\": "
           << stats.lidar_state_requests_suppressed_by_timeline
           << ", \"visual_keyframe_knots_committed\": " << stats.visual_keyframe_knots_committed
           << ", \"visual_keyframes_resolved\": " << stats.visual_keyframes_resolved
           << ", \"visual_graph_attachments\": " << stats.visual_graph_attachments
           << ", \"visual_landmarks_attached\": " << stats.visual_landmarks_attached
           << ", \"visual_factors_attached\": " << stats.visual_factors_attached
           << ", \"visual_factors_retired\": " << stats.visual_factors_retired
           << ", \"visual_graph_degradations\": " << stats.visual_graph_degradations
           << ", \"visual_factor_batches_discarded\": " << stats.visual_factor_batches_discarded
           << ", \"visual_landmark_seeds_discarded\": " << stats.visual_landmark_seeds_discarded
           << ", \"visual_factor_specs_discarded\": " << stats.visual_factor_specs_discarded
           << ", \"visual_stale_track_observations_discarded\": "
           << stats.visual_stale_track_observations_discarded
           << ", \"visual_residual_feedback_items\": " << stats.visual_residual_feedback_items
           << ", \"visual_lane_finality_updates\": " << stats.visual_lane_finality_updates
           << ", \"visual_finalized_landmarks\": " << stats.visual_finalized_landmarks
           << ", \"visual_finalized_factors\": " << stats.visual_finalized_factors
           << ", \"visual_finalized_tracks_pruned\": " << stats.visual_finalized_tracks_pruned
           << ", \"visual_finality_pending_factors_pruned\": "
           << stats.visual_finality_pending_factors_pruned
           << ", \"initialization_attempts\": " << stats.initialization_attempts
           << ", \"initialization_rejections\": " << stats.initialization_rejections
           << ", \"stationary_initialization_attempts\": "
           << stats.stationary_initialization_attempts
           << ", \"stationary_initialization_rejections\": "
           << stats.stationary_initialization_rejections
           << ", \"lidar_bootstrap_anchors\": " << stats.lidar_bootstrap_anchors
           << ", \"lidar_bootstrap_increments\": " << stats.lidar_bootstrap_increments
           << ", \"lidar_bootstrap_rejections\": " << stats.lidar_bootstrap_rejections
           << ", \"motion_initialization_attempts\": " << stats.motion_initialization_attempts
           << ", \"motion_initialization_rejections\": " << stats.motion_initialization_rejections
           << ", \"motion_initialization_commits\": " << stats.motion_initialization_commits
           << ", \"graph_commits\": " << stats.graph_commits
           << ", \"rolling_target_pose_synchronizations\": "
           << stats.rolling_target_pose_synchronizations
           << ", \"rolling_target_sweeps_synchronized\": "
           << stats.rolling_target_sweeps_synchronized
           << ", \"rolling_target_finalized_sweeps_evicted\": "
           << stats.rolling_target_finalized_sweeps_evicted
           << ", \"rolling_target_finalized_points_evicted\": "
           << stats.rolling_target_finalized_points_evicted
           << ", \"lidar_registrations\": " << stats.lidar_registrations
           << ", \"lidar_tracking_only_registrations\": " << stats.lidar_tracking_only_registrations
           << ", \"last_lidar_keyframe_time_ns\": ";
    if (stats.last_lidar_keyframe_time) {
      output << stats.last_lidar_keyframe_time->nanoseconds;
    } else {
      output << "null";
    }
    output
        << ", \"lidar_bootstraps\": " << stats.lidar_bootstraps
        << ", \"lidar_degraded_commits\": " << stats.lidar_degraded_commits
        << ", \"lidar_rejections_target_retained\": " << stats.lidar_rejections_target_retained
        << ", \"lidar_target_state_unavailable_freezes\": "
        << stats.lidar_target_state_unavailable_freezes
        << ", \"lidar_health_transitions\": " << stats.lidar_health_transitions
        << ", \"lidar_shadow_evaluations\": " << stats.lidar_shadow_evaluations
        << ", \"lidar_failure_removal_transactions\": " << stats.lidar_failure_removal_transactions
        << ", \"lidar_faulty_batches_removed\": " << stats.lidar_faulty_batches_removed
        << ", \"lidar_faulty_target_sweeps_removed\": " << stats.lidar_faulty_target_sweeps_removed
        << ", \"lidar_faulty_target_points_removed\": " << stats.lidar_faulty_target_points_removed
        << ", \"finalized_lidar_pending_high_watermark\": "
        << stats.finalized_lidar_pending_high_watermark
        << ", \"finalized_lidar_finality_matches\": " << stats.finalized_lidar_finality_matches
        << ", \"finalized_lidar_rollback_removals\": " << stats.finalized_lidar_rollback_removals
        << ", \"finalized_lidar_freeze_events\": " << stats.finalized_lidar_freeze_events
        << ", \"finalized_lidar_frozen_high_watermark\": "
        << stats.finalized_lidar_frozen_high_watermark
        << ", \"finalized_lidar_insertions\": " << stats.finalized_lidar_insertions
        << ", \"finalized_lidar_inserted_points\": " << stats.finalized_lidar_inserted_points
        << ", \"finalized_lidar_prune_attempts\": " << stats.finalized_lidar_prune_attempts
        << ", \"finalized_lidar_prune_transactions\": " << stats.finalized_lidar_prune_transactions
        << ", \"finalized_lidar_pruned_points\": " << stats.finalized_lidar_pruned_points
        << ", \"finalized_lidar_capacity_recovery_attempts\": "
        << stats.finalized_lidar_capacity_recovery_attempts
        << ", \"finalized_lidar_capacity_recovery_successes\": "
        << stats.finalized_lidar_capacity_recovery_successes
        << ", \"finalized_lidar_capacity_skipped_sweeps\": "
        << stats.finalized_lidar_capacity_skipped_sweeps
        << ", \"finalized_lidar_capacity_retry_suppressions\": "
        << stats.finalized_lidar_capacity_retry_suppressions << '}';
  } else {
    output << "null";
  }
  output << ",\n"
         << "  \"effective_profile\": ";
  if (calibration && profile && estimator_config) {
    output << "{\n"
           << "    \"calibration\": ";
    writeCalibrationProfile(output, *calibration);
    output << ",\n"
           << "    \"replay\": {\n";
    writeReplayProfile(output, *profile);
    output << "    },\n"
           << "    \"offline_execution\": {\"mode\": "
           << jsonString(replayModeName(options.replay.mode)) << ", \"playback_rate\": ";
    if (replayIsUnpaced(options.replay.mode)) {
      output << "null";
    } else {
      output << options.replay.playback_rate;
    }
    output << ", \"queue_count_capacity\": " << options.replay.queue_count_capacity
           << ", \"queue_byte_capacity\": " << options.replay.queue_byte_capacity
           << ", \"timing_window_capacity\": " << options.replay.timing_window_capacity << "},\n"
           << "    \"local_estimator\": {\n";
    writeEstimatorConfig(output, *estimator_config);
    output << "    }\n"
           << "  }\n";
  } else {
    output << "null\n";
  }
  output << "}\n";
  output.close();
  return output.good();
}

[[nodiscard]] std::int64_t wallNowNanoseconds() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

[[nodiscard]] std::optional<std::size_t> parsePositiveSize(std::string_view text) {
  std::size_t value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0U) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<double> parsePositiveDouble(std::string_view text) {
  double value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !std::isfinite(value) || value <= 0.0) {
    return std::nullopt;
  }
  return value;
}

[[nodiscard]] std::optional<std::string> replayOptionsError(
    const BagLocalizationReplayOptions& options) {
  if (options.queue_count_capacity == 0U || options.queue_byte_capacity == 0U ||
      options.timing_window_capacity == 0U) {
    return "replay queue and timing capacities must be non-zero";
  }
  if (replayIsUnpaced(options.mode)) {
    if (options.playback_rate != 0.0) {
      return "unpaced replay cannot be combined with a playback rate";
    }
    return std::nullopt;
  }
  if (!std::isfinite(options.playback_rate) || options.playback_rate <= 0.0) {
    return "scheduled replay requires a finite positive playback rate";
  }
  return std::nullopt;
}

[[nodiscard]] bool scheduledQueueHasLoss(
    const tools::ScheduledReplayRuntimeReport& runtime) noexcept {
  const core::PipelineQueueSnapshot& queue = runtime.queue.terminal;
  return queue.count != 0U || queue.bytes != 0U ||
         queue.accepted != runtime.consumer_visit.wall.total_samples || queue.rejected != 0U ||
         queue.dropped_oldest != 0U || queue.dropped_newest != 0U || queue.skipped_stale != 0U ||
         queue.skipped_policy != 0U;
}

[[nodiscard]] std::string scheduledQueueLossDetail(
    const tools::ScheduledReplayRuntimeReport& runtime) {
  const core::PipelineQueueSnapshot& queue = runtime.queue.terminal;
  std::ostringstream detail;
  detail << "scheduled replay was not loss-free: terminal_count=" << queue.count
         << ", terminal_bytes=" << queue.bytes << ", accepted=" << queue.accepted
         << ", consumed=" << runtime.consumer_visit.wall.total_samples
         << ", rejected=" << queue.rejected << ", dropped_oldest=" << queue.dropped_oldest
         << ", dropped_newest=" << queue.dropped_newest << ", skipped_stale=" << queue.skipped_stale
         << ", skipped_policy=" << queue.skipped_policy;
  return detail.str();
}

[[nodiscard]] std::optional<std::int64_t> durationNanoseconds(
    const std::optional<core::Duration>& duration) noexcept {
  return duration ? std::optional<std::int64_t>(duration->nanoseconds) : std::nullopt;
}

[[nodiscard]] BagLocalizationRuntimeSummary makeRuntimeSummary(
    const BagLocalizationOptions& options,
    const tools::ScheduledReplayRuntimeReport* scheduled_runtime,
    const DriverTimingReport* driver_timing) {
  BagLocalizationRuntimeSummary summary;
  summary.replay_mode = options.replay.mode;
  summary.playback_rate = replayIsUnpaced(options.replay.mode) ? 0.0 : options.replay.playback_rate;
  summary.queue_count_capacity = options.replay.queue_count_capacity;
  summary.queue_byte_capacity = options.replay.queue_byte_capacity;
  summary.timing_window_capacity = options.replay.timing_window_capacity;
  if (scheduled_runtime) {
    const core::PipelineQueueSnapshot& queue = scheduled_runtime->queue.terminal;
    summary.queue_terminal_count = queue.count;
    summary.queue_terminal_bytes = queue.bytes;
    summary.queue_maximum_count = scheduled_runtime->queue.maximum_count;
    summary.queue_maximum_bytes = scheduled_runtime->queue.maximum_bytes;
    summary.queue_maximum_oldest_age_ns =
        durationNanoseconds(scheduled_runtime->queue.maximum_oldest_age);
    summary.queue_rejected = queue.rejected;
    summary.queue_dropped = queue.dropped_oldest + queue.dropped_newest;
    summary.queue_skipped = queue.skipped_stale + queue.skipped_policy;
    summary.producer_total_wall_ns = scheduled_runtime->producer_total.wall_duration.nanoseconds;
    summary.producer_p95_wall_ns =
        durationNanoseconds(scheduled_runtime->producer_schedule_enqueue.wall.p95);
    summary.consumer_total_wall_ns = scheduled_runtime->consumer_total.wall_duration.nanoseconds;
    summary.consumer_p95_wall_ns = durationNanoseconds(scheduled_runtime->consumer_visit.wall.p95);
  }
  if (driver_timing) {
    summary.process_ready_p95_wall_ns = durationNanoseconds(driver_timing->process_ready.wall.p95);
    summary.report_output_p95_wall_ns = durationNanoseconds(driver_timing->report_output.wall.p95);
  }
  return summary;
}

}  // namespace

BagLocalizationCommandLineResult parseBagLocalizationArguments(
    std::span<const std::string_view> arguments) {
  BagLocalizationCommandLine parsed;
  bool has_bag = false;
  bool has_calibration = false;
  bool has_collection = false;
  bool has_sensor_mode = false;
  bool has_output = false;
  bool has_maximum_events = false;
  bool has_maximum_bag_messages = false;
  bool has_replay_mode = false;
  bool has_playback_rate = false;

  const auto fail = [](std::string detail) {
    return BagLocalizationCommandLineResult::failure(
        BagLocalizationCommandLineError{std::move(detail)});
  };

  for (std::size_t index = 0U; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--help" || argument == "-h") {
      parsed.help = true;
      return BagLocalizationCommandLineResult::success(std::move(parsed));
    }
    if (index + 1U >= arguments.size()) {
      return fail("missing value after " + std::string(argument));
    }
    const std::string_view value = arguments[++index];
    if (argument == "--bag") {
      parsed.options.bag_uri = value;
      has_bag = true;
    } else if (argument == "--calib") {
      parsed.options.calibration_root = value;
      has_calibration = true;
    } else if (argument == "--collection") {
      if (value == "1") {
        parsed.options.collection = ros::NewerCollegeCollection::Collection1;
      } else if (value == "2") {
        parsed.options.collection = ros::NewerCollegeCollection::Collection2;
      } else {
        return fail("--collection must be 1 or 2");
      }
      has_collection = true;
    } else if (argument == "--sensor-mode") {
      if (value == "imu") {
        parsed.options.sensor_mode = BagLocalizationSensorMode::Imu;
      } else if (value == "lidar-imu") {
        parsed.options.sensor_mode = BagLocalizationSensorMode::LidarImu;
      } else if (value == "full") {
        parsed.options.sensor_mode = BagLocalizationSensorMode::Full;
      } else {
        return fail("--sensor-mode must be imu, lidar-imu, or full");
      }
      has_sensor_mode = true;
    } else if (argument == "--initialization") {
      if (value == "dynamic") {
        parsed.options.initialization_mode = BagLocalizationInitializationMode::Dynamic;
      } else if (value == "static") {
        parsed.options.initialization_mode = BagLocalizationInitializationMode::Static;
      } else if (value == "supervised-auto") {
        parsed.options.initialization_mode = BagLocalizationInitializationMode::SupervisedAuto;
      } else {
        return fail("--initialization must be dynamic, static, or supervised-auto");
      }
    } else if (argument == "--output") {
      parsed.options.output_directory = value;
      has_output = true;
    } else if (argument == "--max-events") {
      const auto count = parsePositiveSize(value);
      if (!count) {
        return fail("--max-events must be a positive integer");
      }
      parsed.options.maximum_events = *count;
      has_maximum_events = true;
    } else if (argument == "--max-bag-messages") {
      const auto count = parsePositiveSize(value);
      if (!count) {
        return fail("--max-bag-messages must be a positive integer");
      }
      parsed.options.maximum_bag_messages = *count;
      has_maximum_bag_messages = true;
    } else if (argument == "--replay-mode") {
      if (value == "scheduled") {
        parsed.options.replay.mode = BagLocalizationReplayMode::Scheduled;
      } else if (value == "unpaced") {
        parsed.options.replay.mode = BagLocalizationReplayMode::Unpaced;
      } else {
        return fail("--replay-mode must be scheduled or unpaced");
      }
      has_replay_mode = true;
    } else if (argument == "--playback-rate") {
      const auto rate = parsePositiveDouble(value);
      if (!rate) {
        return fail("--playback-rate must be a finite positive number");
      }
      parsed.options.replay.playback_rate = *rate;
      has_playback_rate = true;
    } else if (argument == "--queue-count-capacity") {
      const auto count = parsePositiveSize(value);
      if (!count) {
        return fail("--queue-count-capacity must be a positive integer");
      }
      parsed.options.replay.queue_count_capacity = *count;
    } else if (argument == "--queue-byte-capacity") {
      const auto count = parsePositiveSize(value);
      if (!count) {
        return fail("--queue-byte-capacity must be a positive integer");
      }
      parsed.options.replay.queue_byte_capacity = *count;
    } else if (argument == "--timing-window-capacity") {
      const auto count = parsePositiveSize(value);
      if (!count) {
        return fail("--timing-window-capacity must be a positive integer");
      }
      parsed.options.replay.timing_window_capacity = *count;
    } else if (argument == "--maximum-lidar-targets") {
      const auto count = parsePositiveSize(value);
      if (!count || *count > 2U) {
        return fail("--maximum-lidar-targets must be 1 or 2");
      }
      parsed.options.maximum_lidar_targets = *count;
    } else if (argument == "--visual-graph") {
      if (value == "enabled") {
        parsed.options.visual_graph_enabled = true;
      } else if (value == "disabled") {
        parsed.options.visual_graph_enabled = false;
      } else {
        return fail("--visual-graph must be enabled or disabled");
      }
    } else {
      return fail("unknown argument " + std::string(argument));
    }
  }

  if (!has_bag || !has_calibration || !has_collection || !has_sensor_mode || !has_output ||
      !has_maximum_events || !has_maximum_bag_messages || !has_replay_mode) {
    return fail("all seven input options and --replay-mode are required");
  }
  if (parsed.options.replay.mode == BagLocalizationReplayMode::Scheduled && !has_playback_rate) {
    return fail("scheduled replay requires --playback-rate");
  }
  if (parsed.options.replay.mode == BagLocalizationReplayMode::Unpaced && has_playback_rate) {
    return fail("unpaced replay forbids --playback-rate");
  }
  if (parsed.options.sensor_mode != BagLocalizationSensorMode::Full &&
      parsed.options.visual_graph_enabled.has_value()) {
    return fail("--visual-graph is valid only with --sensor-mode full");
  }
  return BagLocalizationCommandLineResult::success(std::move(parsed));
}

BagLocalizationOutcome runBagLocalization(const BagLocalizationOptions& options) {
  BagLocalizationOutcome outcome;
  outcome.imu_trajectory = options.output_directory / "trajectory_imu.tum";
  outcome.base_trajectory = options.output_directory / "trajectory_base.tum";
  outcome.fixed_lag_imu_trajectory = options.output_directory / "trajectory_imu_fixed_lag.tum";
  outcome.fixed_lag_base_trajectory = options.output_directory / "trajectory_base_fixed_lag.tum";
  outcome.navigation_diagnostics = options.output_directory / "navigation_diagnostics.csv";
  outcome.run_report = options.output_directory / "run_report.json";
  outcome.runtime = makeRuntimeSummary(options, nullptr, nullptr);
  const std::int64_t wall_start_ns = wallNowNanoseconds();

  std::error_code directory_error;
  std::filesystem::create_directories(options.output_directory, directory_error);
  if (directory_error) {
    outcome.suggested_exit_code = 3;
    outcome.error = "cannot create output directory: " + directory_error.message();
    return outcome;
  }
  std::ofstream imu_trajectory(outcome.imu_trajectory, std::ios::out | std::ios::trunc);
  std::ofstream base_trajectory(outcome.base_trajectory, std::ios::out | std::ios::trunc);
  std::ofstream fixed_lag_imu_trajectory(outcome.fixed_lag_imu_trajectory,
                                         std::ios::out | std::ios::trunc);
  std::ofstream fixed_lag_base_trajectory(outcome.fixed_lag_base_trajectory,
                                          std::ios::out | std::ios::trunc);
  std::ofstream navigation_diagnostics(outcome.navigation_diagnostics,
                                       std::ios::out | std::ios::trunc);
  if (!imu_trajectory || !base_trajectory || !fixed_lag_imu_trajectory ||
      !fixed_lag_base_trajectory || !navigation_diagnostics) {
    outcome.suggested_exit_code = 3;
    outcome.error = "cannot open localization outputs";
    return outcome;
  }
  writeNavigationDiagnosticsHeader(navigation_diagnostics);
  if (!navigation_diagnostics.good()) {
    outcome.suggested_exit_code = 3;
    outcome.error = "cannot write navigation diagnostics header";
    return outcome;
  }

  tools::ReplayStats replay_stats;
  RunCounters counters;
  StateCoverage coverage;
  local::LocalEstimatorLifecycle lifecycle = local::LocalEstimatorLifecycle::AwaitingInitialization;
  std::optional<Failure> failure;
  std::optional<tools::ReplayCompletion> replay_completion;
  std::optional<tools::ScheduledReplayRuntimeReport> scheduled_runtime;
  std::optional<DriverTimingReport> driver_timing;
  std::optional<core::ArrivalTime> first_recorded_arrival;
  std::optional<core::ArrivalTime> last_recorded_arrival;
  const core::CalibrationBundle* calibration_pointer = nullptr;
  const tools::ReplayProfile* profile_pointer = nullptr;
  const local::LocalEstimatorConfig* estimator_config_pointer = nullptr;
  const local::LocalEstimatorStatistics* estimator_statistics_pointer = nullptr;
  std::optional<core::CalibrationBundle> calibration_storage;
  std::optional<tools::ReplayProfile> profile_storage;
  std::optional<local::LocalEstimatorConfig> estimator_config_storage;

  if (const auto replay_error = replayOptionsError(options.replay)) {
    failure = Failure{"options", *replay_error, std::nullopt, std::nullopt};
    const std::int64_t wall_end_ns = wallNowNanoseconds();
    const bool report_written = writeRunReport(
        options, outcome.run_report, nullptr, nullptr, nullptr, replay_stats, replay_completion,
        nullptr, nullptr, nullptr, counters, coverage, lifecycle, first_recorded_arrival,
        last_recorded_arrival, wall_start_ns, wall_end_ns, failure, false, false, false);
    outcome.suggested_exit_code = report_written ? 2 : 3;
    outcome.error = failure->detail;
    return outcome;
  }

  if (!sensorModeIncludesCameras(options.sensor_mode) && options.visual_graph_enabled.has_value()) {
    failure = Failure{"options", "visual graph selection is valid only in full sensor mode",
                      std::nullopt, std::nullopt};
    const std::int64_t wall_end_ns = wallNowNanoseconds();
    const bool report_written = writeRunReport(
        options, outcome.run_report, nullptr, nullptr, nullptr, replay_stats, replay_completion,
        nullptr, nullptr, nullptr, counters, coverage, lifecycle, first_recorded_arrival,
        last_recorded_arrival, wall_start_ns, wall_end_ns, failure, false, false, false);
    outcome.suggested_exit_code = report_written ? 2 : 3;
    outcome.error = failure->detail;
    return outcome;
  }

  auto calibration_result =
      ros::loadNewerCollegeCalibration(options.calibration_root, options.collection,
                                       kCalibrationEpoch, newerCollegeImuCalibration());
  if (!calibration_result) {
    failure = Failure{"calibration", calibration_result.error().detail,
                      static_cast<int>(calibration_result.error().code), std::nullopt};
  } else {
    calibration_storage.emplace(std::move(calibration_result).value());
    core::CalibrationBundle& calibration = *calibration_storage;
    calibration_pointer = &calibration;
    estimator_config_storage.emplace(effectiveEstimatorConfig(calibration, options));
    local::LocalEstimatorConfig& estimator_config = *estimator_config_storage;
    estimator_config_pointer = &estimator_config;
    auto profile_result = makeReplayProfile(calibration, options);
    if (!profile_result) {
      failure = Failure{"replay_profile", profile_result.error().detail,
                        static_cast<int>(profile_result.error().code), std::nullopt};
    } else {
      profile_storage.emplace(std::move(profile_result).value());
      tools::ReplayProfile& profile = *profile_storage;
      profile_pointer = &profile;
      auto estimator_result = local::LocalEstimator::create(calibration, estimator_config);
      if (!estimator_result) {
        failure = Failure{"estimator_create", estimator_result.error().detail,
                          static_cast<int>(estimator_result.error().code),
                          static_cast<int>(estimator_result.error().stage)};
      } else {
        local::LocalEstimator estimator = std::move(estimator_result).value();
        estimator_config_pointer = &estimator.effectiveConfig();
        LocalizationDriver driver(
            &estimator, options.sensor_mode, calibration.baseFromImu().T_base_imu(),
            estimator.effectiveConfig(), options.replay.timing_window_capacity, &imu_trajectory,
            &base_trajectory, &fixed_lag_imu_trajectory, &fixed_lag_base_trajectory,
            &navigation_diagnostics);
        if (replayIsUnpaced(options.replay.mode)) {
          auto replay = tools::replayRos2Bag(
              options.bag_uri, profile,
              [&](tools::DomainEvent&& event) { return driver.visit(std::move(event)); });
          if (!replay) {
            replay_stats = replay.error().stats;
            failure = driver.failure().value_or(Failure{"replay", replay.error().detail,
                                                        static_cast<int>(replay.error().code),
                                                        std::nullopt});
          } else {
            replay_stats = replay.value().stats;
            replay_completion = replay.value().completion;
          }
        } else {
          tools::ScheduledReplayOptions schedule;
          schedule.playback_rate = options.replay.playback_rate;
          schedule.queue_count_capacity = options.replay.queue_count_capacity;
          schedule.queue_byte_capacity = options.replay.queue_byte_capacity;
          schedule.timing_window_capacity = options.replay.timing_window_capacity;
          auto replay = tools::replayRos2BagScheduled(
              options.bag_uri, profile, schedule,
              [&](tools::DomainEvent&& event) { return driver.visit(std::move(event)); });
          if (!replay) {
            replay_stats = replay.error().stats;
            if (replay.error().runtime) {
              scheduled_runtime = *replay.error().runtime;
            }
            failure = driver.failure().value_or(Failure{"replay", replay.error().detail,
                                                        static_cast<int>(replay.error().code),
                                                        std::nullopt});
          } else {
            replay_stats = replay.value().stats;
            replay_completion = replay.value().completion;
            scheduled_runtime = replay.value().runtime;
          }
        }
        if (!driver.finalizeFixedLagTrajectories() && !failure) {
          failure = driver.failure().value_or(Failure{"trajectory_output",
                                                      "fixed-lag trajectory finalization failed",
                                                      std::nullopt, std::nullopt});
        }
        counters = driver.counters();
        coverage = driver.coverage();
        lifecycle = driver.lifecycle();
        first_recorded_arrival = driver.firstRecordedArrival();
        last_recorded_arrival = driver.lastRecordedArrival();
        driver_timing = driver.timingReport();
        outcome.runtime =
            makeRuntimeSummary(options, scheduled_runtime ? &*scheduled_runtime : nullptr,
                               driver_timing ? &*driver_timing : nullptr);
        estimator_statistics_pointer = &estimator.statistics();
        if (scheduled_runtime && scheduledQueueHasLoss(*scheduled_runtime) && !failure) {
          failure = Failure{"scheduled_replay_queue", scheduledQueueLossDetail(*scheduled_runtime),
                            std::nullopt, std::nullopt};
        }
        const SensorModeAudit sensor_mode_audit =
            makeSensorModeAudit(options, calibration_pointer, profile_pointer,
                                estimator_config_pointer, replay_stats, counters);
        if (!failure && !sensor_mode_audit.passed) {
          failure =
              Failure{"sensor_mode_audit",
                      "configured sources or runtime activity violated " +
                          std::string(sensorModeName(options.sensor_mode)) + " sensor isolation",
                      std::nullopt, std::nullopt};
        }

        outcome.states_written = coverage.states_written;
        outcome.fixed_lag_states_written = driver.fixedLagStatesWritten();
        outcome.localization_available = coverage.states_written > 0U;
        outcome.execution_succeeded = !failure.has_value();
        outcome.full_bag_consumed = replay_completion == tools::ReplayCompletion::EndOfBag;
        if (!outcome.execution_succeeded) {
          outcome.suggested_exit_code = 4;
        } else if (!outcome.localization_available) {
          outcome.suggested_exit_code = 5;
          failure =
              Failure{"coverage", "replay completed without producing an initialized local state",
                      std::nullopt, std::nullopt};
          outcome.error = failure->detail;
        } else {
          outcome.suggested_exit_code = 0;
        }

        imu_trajectory.flush();
        base_trajectory.flush();
        fixed_lag_imu_trajectory.flush();
        fixed_lag_base_trajectory.flush();
        navigation_diagnostics.flush();
        if ((!imu_trajectory.good() || !base_trajectory.good() ||
             !fixed_lag_imu_trajectory.good() || !fixed_lag_base_trajectory.good() ||
             !navigation_diagnostics.good()) &&
            !failure) {
          failure = Failure{"trajectory_output", "localization output flush failed after replay",
                            std::nullopt, std::nullopt};
          outcome.execution_succeeded = false;
          outcome.suggested_exit_code = 3;
        }
        const std::int64_t wall_end_ns = wallNowNanoseconds();
        const bool report_written = writeRunReport(
            options, outcome.run_report, calibration_pointer, profile_pointer,
            estimator_config_pointer, replay_stats, replay_completion,
            scheduled_runtime ? &*scheduled_runtime : nullptr,
            driver_timing ? &*driver_timing : nullptr, estimator_statistics_pointer, counters,
            coverage, lifecycle, first_recorded_arrival, last_recorded_arrival, wall_start_ns,
            wall_end_ns, failure, outcome.execution_succeeded, outcome.localization_available,
            outcome.full_bag_consumed);
        if (!report_written) {
          outcome.execution_succeeded = false;
          outcome.suggested_exit_code = 3;
          outcome.error = "failed to write machine-readable run report";
        } else if (failure && outcome.error.empty()) {
          outcome.error = failure->detail;
        }
        return outcome;
      }
    }

    const std::int64_t wall_end_ns = wallNowNanoseconds();
    const bool report_written = writeRunReport(
        options, outcome.run_report, calibration_pointer, profile_pointer, estimator_config_pointer,
        replay_stats, replay_completion, scheduled_runtime ? &*scheduled_runtime : nullptr,
        driver_timing ? &*driver_timing : nullptr, estimator_statistics_pointer, counters, coverage,
        lifecycle, first_recorded_arrival, last_recorded_arrival, wall_start_ns, wall_end_ns,
        failure, false, false, false);
    outcome.suggested_exit_code = report_written ? 4 : 3;
    outcome.error = failure ? failure->detail : "localization setup failed";
    return outcome;
  }

  const std::int64_t wall_end_ns = wallNowNanoseconds();
  const bool report_written = writeRunReport(
      options, outcome.run_report, nullptr, nullptr, nullptr, replay_stats, replay_completion,
      nullptr, nullptr, nullptr, counters, coverage, lifecycle, first_recorded_arrival,
      last_recorded_arrival, wall_start_ns, wall_end_ns, failure, false, false, false);
  outcome.suggested_exit_code = report_written ? 4 : 3;
  outcome.error = failure ? failure->detail : "calibration setup failed";
  return outcome;
}

}  // namespace meridian::apps
