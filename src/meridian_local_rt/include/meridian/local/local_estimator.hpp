#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"
#include "meridian/local/graph.hpp"
#include "meridian/local/lidar_bootstrap.hpp"
#include "meridian/local/lidar_map_payload.hpp"
#include "meridian/local/lidar_registration.hpp"
#include "meridian/local/lidar_registration_cloud.hpp"
#include "meridian/local/pipeline_observability.hpp"
#include "meridian/local/rolling_lidar_target.hpp"
#include "meridian/local/sensor_health.hpp"
#include "meridian/local/visual_lane.hpp"

namespace meridian::local {

enum class InitializationMode {
  StaticOnly,
  DynamicOnly,
  SupervisedAuto,
};

enum class ZeroMotionPriorSource {
  Operator,
  VehicleSupervisor,
  MissionScenario,
};

// Explicit authority for applying the zero-velocity/static-world hypothesis
// during the startup window of one odometry epoch. IMU statistics verify this
// prior; they never create it. A new epoch requires a new matching prior.
struct ZeroMotionPrior {
  core::OdomEpoch odom_epoch;
  ZeroMotionPriorSource source{ZeroMotionPriorSource::VehicleSupervisor};

  [[nodiscard]] bool valid() const noexcept { return odom_epoch.valid(); }
};

struct InitializationPolicy {
  // Dynamic LiDAR--IMU initialization is the safe default for an unsupervised
  // startup. StaticOnly and the static branch of SupervisedAuto require the
  // explicit zero-motion prior below.
  InitializationMode mode{InitializationMode::DynamicOnly};
  std::optional<ZeroMotionPrior> zero_motion_prior;
};

struct VisualCameraConfig {
  core::CameraId camera;
  VisualLaneConfig lane;
};

struct LocalEstimatorConfig {
  core::OdomEpoch odom_epoch{1U};
  core::StateId first_state{1U};
  InitializationPolicy initialization;
  std::size_t maximum_pending_lidar_sweeps{32U};
  // Explicit outage/heartbeat requests are bounded independently from sensor
  // queues. Capacity is reported to the caller; guards are never silently
  // dropped or synthesized by the estimator.
  std::size_t maximum_pending_imu_guards{64U};
  // Cameras are enabled only by listing their calibrated IDs here. An empty
  // vector is a deliberate IMU/LiDAR-only configuration; calibrated cameras
  // are never enabled implicitly.
  std::vector<VisualCameraConfig> visual_cameras;
  StateTimelineConfig state_timeline;
  core::Duration stationary_retry_period{50'000'000LL};
  LocalPipelineTimingConfig pipeline_timing;
  ImuBufferConfig imu_buffer;
  StationaryInitializerConfig stationary_initializer;
  LidarBootstrapOdometryConfig lidar_bootstrap;
  MotionInitializerConfig motion_initializer;
  LocalGraphConfig graph;
  SensorHealthPolicyConfig sensor_health_policy;
  std::size_t maximum_recent_faulty_batches_to_remove{3U};
  LidarPreprocessConfig lidar_preprocessing;
  RollingLidarTargetConfig rolling_target;
  // Localization-accepted keyframes remain independently staged until their
  // exact graph state becomes final. This capacity bounds both pre-finality
  // payloads and health-frozen finalized payloads; rolling-target eviction can
  // never consume this storage.
  std::size_t maximum_pending_finalized_lidar_sweeps{128U};
  // Finalized geometry is compacted incrementally into one fixed-odom,
  // owner-preserving spatial index. Graph-factor scans borrow a synchronous
  // read view of this map; live pose-owned targets remain a separate
  // relinearizable overlay and are never relabelled as fixed geometry.
  FinalizedLidarTargetMapConfig finalized_lidar_target;
  // pruneAround() is the only O(map) maintenance operation. Schedule it by
  // successful persistent-map admissions rather than hiding it in every scan.
  std::size_t finalized_lidar_prune_interval_sweeps{128U};
  // Every LiDAR sweep is still deskewed, preprocessed, registered, and
  // health-assessed. Only successful registrations separated by at least
  // this interval may publish a LiDAR FactorBatch and map payload. Zero
  // deliberately admits every successful registration.
  core::Duration minimum_lidar_factor_interval{};
  // One exact-neighbor point registration profile is shared by target
  // selection, the bounded frontend solve, and the stateless graph factors.
  // Normal tracking accepts every physically observable subspace; the direct
  // factor cannot invent precision in projected null directions.
  LidarRegistrationConfig lidar_registration;
  // Tracking LiDAR reuses target geometry and the IMU samples that condition
  // its deskew/prediction. These independent benchmark seeds form one explicit
  // batch-local covariance inflation; they never add an IMU residual to the
  // LiDAR FactorBatch.
  double lidar_target_reuse_covariance_inflation{1.5};
  double lidar_imu_conditioning_covariance_inflation{4.0};
  // Finalized-map targets reuse geometry whose scan-level correlation is not
  // represented by the reconstructed owner-pose marginal alone.  The unary
  // map channel therefore uses max(reconstructed owner-pose inflation, this
  // configured floor).  One disables any additional floor without changing
  // the live LiDAR channel.
  double finalized_map_correlation_inflation_floor{1.0};
};

enum class LocalEstimatorLifecycle {
  AwaitingInitialization,
  Tracking,
  Faulted,
};

enum class LocalEstimatorStage {
  Configuration,
  ImuIngress,
  ImuGuardIngress,
  LidarIngress,
  CameraIngress,
  StateTimeline,
  VisualFrontend,
  VisualKnotResolution,
  VisualGraphAttachment,
  VisualResidualFeedback,
  StationaryInitialization,
  LidarBootstrap,
  MotionInitialization,
  ImuSupport,
  Propagation,
  Deskew,
  LidarPreprocessing,
  RollingTarget,
  FinalizedTarget,
  SensorHealth,
  LocalGraph,
  IdentityAllocation,
  InternalInvariant,
};

enum class LocalEstimatorErrorCode {
  InvalidConfiguration,
  ImuRejected,
  ImuGuardRejected,
  LidarRejected,
  CameraRejected,
  CameraNotEnabled,
  VisualLaneFailed,
  StateTimelineFailed,
  VisualReferenceUnavailable,
  PendingLidarCapacity,
  PendingImuGuardCapacity,
  InitializationFailed,
  ImuSupportFailed,
  PropagationFailed,
  GraphTransactionFailed,
  RollingTargetFailed,
  FinalizedTargetFailed,
  PendingFinalizedTargetCapacity,
  SensorHealthFailed,
  IdentityExhausted,
  InternalInvariant,
};

struct LocalEstimatorError {
  LocalEstimatorErrorCode code{LocalEstimatorErrorCode::InternalInvariant};
  LocalEstimatorStage stage{LocalEstimatorStage::InternalInvariant};
  std::string detail;
  std::optional<LocalGraphErrorCode> graph_error_code;
  std::optional<DirectLidarRegistrationReport> lidar_registration;
  std::optional<LidarRegistrationError> lidar_registration_error;
  // Pair-specific evidence from a rejected LiDAR graph transaction.  The
  // deliberate IMU-only fallback has no LiDAR pairs of its own, so retaining
  // this vector is the only way to diagnose which target failed without
  // confusing rejected candidate state with committed graph state.
  std::vector<DirectLidarPairReport> lidar_pairs;
  std::optional<DirectLidarFinalizedMapReport> lidar_finalized_map;
  std::optional<LocalSolveReport> rejected_solve;
  std::optional<ImuBufferErrorCode> imu_buffer_error_code;
};

struct LocalEstimatorImuIngestReport {
  core::MeasurementId measurement;
  ImuAppendReport buffer;
};

struct LocalEstimatorLidarEnqueueReport {
  core::MeasurementId measurement;
  StateAdmissionDisposition state_admission{StateAdmissionDisposition::NewState};
  std::optional<core::FusionTime> suppressing_state_time;
  std::size_t pending_sweeps{};
};

struct LocalEstimatorImuGuardEnqueueReport {
  core::KnotRequestId request;
  core::FusionTime exact_time;
  StateAdmissionDisposition state_admission{StateAdmissionDisposition::NewState};
  bool exactly_shared{};
  std::vector<core::KnotRequestId> exactly_shared_requests;
  std::size_t pending_guards{};
};

struct LocalEstimatorCameraIngestReport {
  core::MeasurementId measurement;
  core::CameraId camera;
  VisualFrameReport frame;
  // A late image is still tracked by the frontend, but it cannot request a
  // state behind the graph writer's finalized time.
  bool late_for_graph{};
  bool imu_rotation_seed_provided{};
  std::size_t pending_camera_knots{};
};

enum class LidarDropReason {
  BeforeInitialization,
  // The requested exact-share state has already left the live local graph.
  // Committed states still inside the lag window accept LiDAR-only factors.
  ReferenceStateNotLive,
  StateRequestSuppressedTooClose,
};

struct LidarDropReport {
  core::MeasurementId measurement;
  core::TimeRange acquisition;
  LidarDropReason reason{LidarDropReason::BeforeInitialization};
};

enum class LidarCommitDisposition {
  BootstrapTarget,
  Registered,
  // Direct registration and health assessment succeeded, but the deliberate
  // LiDAR keyframe policy emitted neither a FactorBatch nor a map payload.
  // The sensor-neutral IMU navigation state remains committed.
  RegisteredTrackingOnly,
  ImuOnlyDeskewRejected,
  ImuOnlyPreprocessingRejected,
  // The LiDAR batch was not accepted by localization. Its cloud is therefore
  // never inserted into or used to replace the rolling target, including
  // when the requested target state has just left the live lag window.
  ImuOnlyRegistrationRejectedTargetRetained,
  // Registration ran in shadow mode while the sensor was Suspect, Failed, or
  // Recovering. No factor or map payload was admitted.
  ImuOnlyHealthQuarantinedTargetRetained,
};

struct VisualGraphAttachmentReport {
  core::CameraId camera;
  VisualAttachmentId lane_attachment;
  core::StateId transaction_state;
  core::LocalGraphRevision committed_revision;
  std::vector<VisualLandmarkId> graph_landmarks;
  std::vector<core::FactorId> graph_factors;
  std::vector<core::FactorId> graph_factor_retirements;
};

struct VisualGraphDegradationReport {
  core::CameraId camera;
  VisualAttachmentId lane_attachment;
  core::StateId transaction_state;
  core::LocalGraphRevision committed_revision;
  LocalGraphErrorCode rejected_graph_error_code{LocalGraphErrorCode::SolverFailure};
  std::string rejection_detail;
  std::size_t factor_batches_discarded{};
  std::size_t landmark_seeds_discarded{};
  std::size_t factors_discarded{};
  std::size_t stale_track_observations_discarded{};
  std::size_t factor_retirements_preserved{};
};

struct LidarCommitReport {
  core::MeasurementId measurement;
  LidarCommitDisposition disposition{LidarCommitDisposition::BootstrapTarget};
  LocalGraphCommit commit;
  // True only when this LiDAR transaction created the report's navigation
  // state and its single incoming IMU edge. False means `commit` is a
  // factor-only revision of an already-published state; trajectory writers
  // must retain its diagnostics without emitting a duplicate state sample.
  bool navigation_state_created{true};
  // True when processing this sweep committed at least one graph transaction.
  // An exact-share tracking/rejection report may reference the unchanged
  // current commit while still carrying valuable frontend diagnostics.
  bool graph_revision_created{true};
  std::size_t deskew_pose_interpolations{};
  std::optional<LidarPreprocessStats> preprocessing;
  std::optional<RollingLidarTargetBatchBuildStats> target;
  std::optional<RollingLidarTargetAddStats> target_add;
  // Present only after this exact raw sweep's localization transaction and
  // map admission both succeed. Tracking-only and every rejection path keep
  // this null, so dense mapping cannot consume provisional localization.
  std::shared_ptr<const AcceptedLidarMapInput> map_input;
  std::optional<DirectLidarRegistrationReport> registration;
  std::optional<LidarRegistrationError> registration_error;
  std::optional<SensorHealthUpdate> health_update;
  std::vector<SensorFactorBatchRef> removed_factor_batches;
  std::optional<RollingLidarTargetRemovalStats> target_removal;
  // Candidate-only pair reports when `commit` is an IMU-only fallback.  These
  // reports are diagnostic and never describe factors in the fallback commit.
  std::vector<DirectLidarPairReport> rejected_lidar_pairs;
  std::optional<DirectLidarFinalizedMapReport> rejected_lidar_finalized_map;
  std::optional<VisualGraphAttachmentReport> visual_attachment;
  std::optional<VisualGraphDegradationReport> visual_degradation;
  // Present when the LiDAR graph transaction was rejected and this report
  // instead carries the deliberate IMU-only fallback commit.
  std::optional<LocalGraphErrorCode> degradation_graph_error_code;
  std::optional<LocalSolveReport> rejected_solve;
  std::string degradation_detail;
};

// One graph-owned state may resolve several bit-identical camera requests.
// Older queued visual factors, if any, are attached to this same atomic
// SensorKnotAppend; factors from the just-resolved keyframe wait for the next
// sensor transaction.
struct CameraKnotCommitReport {
  core::FusionTime exact_time;
  LocalGraphCommit commit;
  std::optional<VisualGraphAttachmentReport> visual_attachment;
  std::optional<VisualGraphDegradationReport> visual_degradation;
  std::vector<VisualResolvedKeyframeReport> resolved_keyframes;
};

struct ImuGuardResolution {
  core::RecordHeader header;
  core::KnotResolutionId id;
  core::KnotRequestId request;
  core::OdomEpoch odom_epoch;
  core::StateId state;
  core::FusionTime exact_time;
  core::LocalGraphRevision created_at_revision;
  std::vector<core::KnotRequestId> exactly_shared_requests;
};

// The embedded commit contains the complete graph finality publication for
// this transaction. Several bit-identical guards resolve to the same state.
struct ImuGuardCommitReport {
  core::FusionTime exact_time;
  LocalGraphCommit commit;
  std::vector<ImuGuardResolution> resolutions;
};

struct LocalEstimatorStatistics {
  std::size_t imu_samples_accepted{};
  std::size_t imu_guards_enqueued{};
  std::size_t imu_guard_enqueue_rejections{};
  std::size_t imu_guard_capacity_rejections{};
  std::size_t imu_guard_pending_high_watermark{};
  std::size_t imu_guard_knots_committed{};
  std::size_t imu_guard_requests_resolved{};
  std::size_t imu_guard_requests_suppressed_by_timeline{};
  std::size_t lidar_sweeps_enqueued{};
  std::size_t lidar_sweeps_dropped{};
  std::size_t camera_frames_accepted{};
  std::size_t camera_frames_rejected{};
  std::size_t camera_frames_late_for_graph{};
  std::size_t camera_rotation_seeds_provided{};
  std::size_t visual_keyframe_requests{};
  std::size_t lidar_state_requests_suppressed_by_timeline{};
  std::size_t visual_keyframe_knots_committed{};
  std::size_t visual_keyframes_resolved{};
  std::size_t visual_graph_attachments{};
  std::size_t visual_landmarks_attached{};
  std::size_t visual_factors_attached{};
  std::size_t visual_factors_retired{};
  std::size_t visual_graph_degradations{};
  std::size_t visual_factor_batches_discarded{};
  std::size_t visual_landmark_seeds_discarded{};
  std::size_t visual_factor_specs_discarded{};
  std::size_t visual_stale_track_observations_discarded{};
  std::size_t visual_residual_feedback_items{};
  std::size_t visual_lane_finality_updates{};
  std::size_t visual_finalized_landmarks{};
  std::size_t visual_finalized_factors{};
  std::size_t visual_finalized_tracks_pruned{};
  std::size_t visual_finality_pending_factors_pruned{};
  std::size_t initialization_attempts{};
  std::size_t initialization_rejections{};
  std::size_t stationary_initialization_attempts{};
  std::size_t stationary_initialization_rejections{};
  std::size_t lidar_bootstrap_anchors{};
  std::size_t lidar_bootstrap_increments{};
  std::size_t lidar_bootstrap_rejections{};
  std::size_t motion_initialization_attempts{};
  std::size_t motion_initialization_rejections{};
  std::size_t motion_initialization_commits{};
  std::size_t graph_commits{};
  std::size_t rolling_target_pose_synchronizations{};
  std::size_t rolling_target_sweeps_synchronized{};
  std::size_t rolling_target_finalized_sweeps_evicted{};
  std::size_t rolling_target_finalized_points_evicted{};
  std::size_t lidar_registrations{};
  std::size_t lidar_tracking_only_registrations{};
  // Historical cadence anchor for admitted LiDAR factor/map keyframes. It is
  // deliberately not rewound when a recent faulty batch is retracted.
  std::optional<core::FusionTime> last_lidar_keyframe_time;
  std::size_t lidar_bootstraps{};
  std::size_t lidar_degraded_commits{};
  std::size_t lidar_rejections_target_retained{};
  std::size_t lidar_target_state_unavailable_freezes{};
  std::size_t lidar_health_transitions{};
  std::size_t lidar_shadow_evaluations{};
  std::size_t lidar_failure_removal_transactions{};
  std::size_t lidar_faulty_batches_removed{};
  std::size_t lidar_faulty_target_sweeps_removed{};
  std::size_t lidar_faulty_target_points_removed{};
  std::size_t finalized_lidar_pending_high_watermark{};
  std::size_t finalized_lidar_finality_matches{};
  std::size_t finalized_lidar_rollback_removals{};
  std::size_t finalized_lidar_freeze_events{};
  std::size_t finalized_lidar_frozen_high_watermark{};
  std::size_t finalized_lidar_insertions{};
  std::size_t finalized_lidar_inserted_points{};
  // PointCapacity is a persistent-map resource condition, not a LiDAR or
  // localization failure. One exact-origin prune/retry is attempted before a
  // finalized payload is terminally skipped. While capacity remains
  // saturated, retries are bounded by finalized_lidar_prune_interval_sweeps.
  std::size_t finalized_lidar_capacity_recovery_attempts{};
  std::size_t finalized_lidar_capacity_recovery_successes{};
  std::size_t finalized_lidar_capacity_skipped_sweeps{};
  std::size_t finalized_lidar_capacity_retry_suppressions{};
  std::size_t finalized_lidar_prune_attempts{};
  std::size_t finalized_lidar_prune_transactions{};
  std::size_t finalized_lidar_pruned_points{};
};

enum class LocalInitializationMethod {
  StationaryImu,
  MotionLidarImu,
};

enum class LocalInitializationRejectionStage {
  StationaryTest,
  LidarBootstrap,
  MotionBatch,
};

struct LocalInitializationRejection {
  LocalInitializationRejectionStage stage{LocalInitializationRejectionStage::StationaryTest};
  int code{};
  std::optional<std::size_t> segment;
  std::string detail;
};

struct LidarBootstrapProcessReport {
  core::MeasurementId measurement;
  std::optional<LidarBootstrapCommit> commit;
  std::optional<LidarBootstrapError> rejection;
};

// Solver diagnostics for exactly one committed local-graph revision. A
// process cycle may create a navigation state and then attach a sensor-pure
// FactorBatch to that already-live state. Keep every revision here even when
// the later sensor transaction becomes the representative state commit.
struct LocalGraphTransactionSolveReport {
  core::LocalGraphRevision revision;
  core::LocalGraphRevision parent;
  core::StateId state;
  core::FusionTime state_time;
  bool navigation_state_created{};
  LocalSolveReport solve;
};

enum class FinalizedLidarTargetCapacitySkipReason {
  RetryAfterPruneStillFull,
  RetrySuppressedWhileSaturated,
};

// Terminal record for one graph-final, localization-accepted payload that
// could not enter the bounded persistent target. The localization FactorBatch,
// rolling target, and dense-map admission remain accepted; only this optional
// registration-map copy is skipped. The map version/checksum name the exact
// still-readable map state after the decision (and after an optional prune).
struct FinalizedLidarTargetCapacitySkip {
  SensorFactorBatchRef batch;
  core::StateId state;
  core::FusionTime exact_time;
  core::LocalGraphRevision final_revision;
  core::MeasurementId sweep;
  core::ContentHash cloud_checksum{};
  std::size_t input_points{};
  std::size_t retained_points{};
  std::size_t hard_point_capacity{};
  std::uint64_t map_version{};
  core::ContentHash map_checksum{};
  FinalizedLidarTargetCapacitySkipReason reason{
      FinalizedLidarTargetCapacitySkipReason::RetryAfterPruneStillFull};
};

// Bounded persistent-target work performed during exactly one processReady()
// call. Exact transaction records are retained for replay diagnostics, while
// the queue/map snapshot makes freezes and high-watermark pressure visible
// even in cycles without an insertion.
struct FinalizedLidarTargetProcessReport {
  std::size_t finality_matches{};
  std::size_t rollback_removals{};
  std::vector<FinalizedLidarTargetInsertStats> insertions;
  std::vector<FinalizedLidarTargetPruneStats> prunes;
  std::size_t capacity_recovery_attempts{};
  std::size_t capacity_recovery_successes{};
  std::vector<FinalizedLidarTargetCapacitySkip> capacity_skips;
  std::size_t pending_sweeps{};
  std::size_t pending_unfinalized_sweeps{};
  std::size_t finalized_ready_sweeps{};
  bool insertion_frozen{};
  bool capacity_saturated{};
  std::size_t capacity_skips_since_retry{};
  core::SensorHealthState lidar_health{core::SensorHealthState::Failed};
  std::size_t retained_points{};
  std::uint64_t map_version{};
  core::ContentHash map_checksum{};
};

struct LocalEstimatorProcessReport {
  LocalEstimatorLifecycle lifecycle{LocalEstimatorLifecycle::AwaitingInitialization};
  // Strictly revision-ordered and complete for every graph transaction
  // committed during this processReady() call. This is the authoritative
  // solver-work stream; representative state commits must not be used to
  // infer transaction counts.
  std::vector<LocalGraphTransactionSolveReport> graph_transactions;
  // Ordered final estimates for every navigation state marginalized by any
  // graph transaction performed during this processReady() call. This
  // sensor-neutral publication is independent of which later optional
  // frontend transaction becomes the representative state commit.
  std::vector<LocalGraphFinalizedState> finalized_states;
  std::optional<LocalGraphCommit> initialization;
  std::optional<LocalInitializationMethod> initialization_method;
  // Present only when a moving LiDAR--IMU batch produced the initialization
  // committed above. Keeping the numerical observability report at the
  // coordinator API boundary lets replay/ROS applications expose the exact
  // initialization decision without reaching into an optimizer backend.
  std::optional<MotionInitializationDiagnostics> motion_initialization_diagnostics;
  // Moving initialization has no ordinary LidarCommitReport. Its accepted
  // seed payload is therefore published once at this process-report level.
  std::shared_ptr<const AcceptedLidarMapInput> initialization_map_input;
  std::optional<LocalInitializationRejection> initialization_rejection;
  std::vector<LidarBootstrapProcessReport> bootstrap;
  std::vector<LidarDropReport> dropped_sweeps;
  std::vector<LidarCommitReport> commits;
  std::vector<CameraKnotCommitReport> camera_commits;
  std::vector<ImuGuardCommitReport> imu_guard_commits;
  FinalizedLidarTargetProcessReport finalized_lidar_target;
  std::size_t pending_sweeps{};
  std::size_t pending_imu_guards{};
  std::size_t pending_camera_knots{};
  bool waiting_for_future_imu{};
};

enum class ImuPropagationGapStatus {
  AnchorOnly,
  Contiguous,
  InferredMissingTicks,
};

struct LocalEstimatorPropagationReport {
  core::OdomEpoch odom_epoch;
  core::StateId anchor_state;
  core::LocalGraphRevision anchor_revision;
  core::FusionTime anchor_time;
  core::NavStateEstimate anchor_estimate;
  core::FusionTime exact_time;
  core::NavStateEstimate propagated_state;
  std::vector<core::MeasurementId> raw_imu_support;
  core::Duration maximum_raw_gap{};
  std::size_t inferred_missing_ticks{};
  bool contains_saturation{};
  ImuPropagationGapStatus gap_status{ImuPropagationGapStatus::AnchorOnly};
};

// Deterministic ROS-free orchestration of the IMU buffer, discrete deskew,
// bounded direct point registration, pose-aware rolling targets, and one shared
// local graph. Ingress and processing are separate so ROS/bag adapters can
// preserve their own callback model while driving the same deterministic API.
class LocalEstimator {
public:
  [[nodiscard]] static core::Result<LocalEstimator, LocalEstimatorError> create(
      core::CalibrationBundle calibration, LocalEstimatorConfig config = {});

  ~LocalEstimator();
  LocalEstimator(LocalEstimator&&) noexcept;
  LocalEstimator& operator=(LocalEstimator&&) noexcept;
  LocalEstimator(const LocalEstimator&) = delete;
  LocalEstimator& operator=(const LocalEstimator&) = delete;

  [[nodiscard]] core::Result<LocalEstimatorImuIngestReport, LocalEstimatorError> ingestImu(
      core::ImuSample sample);

  [[nodiscard]] core::Result<LocalEstimatorLidarEnqueueReport, LocalEstimatorError> enqueueLidar(
      core::LidarSweep sweep);

  // Explicitly requests an IMU-backed navigation state at a bit-exact time.
  // The estimator allocates all request/state identities. This is a bounded
  // production outage API, not an automatic hidden heartbeat mode.
  [[nodiscard]] core::Result<LocalEstimatorImuGuardEnqueueReport, LocalEstimatorError>
  enqueueImuGuard(core::FusionTime exact_time);

  // Tracks every accepted image immediately. Only selected keyframes request
  // exact graph states; `request_keyframe` is an explicit benchmark/debug
  // override passed to the lane's normal keyframe policy.
  [[nodiscard]] core::Result<LocalEstimatorCameraIngestReport, LocalEstimatorError> ingestCamera(
      core::CameraFrame frame, bool request_keyframe = false);

  // Residual feedback uses graph-visible factor IDs. The coordinator maps
  // them back to the selected lane's private IDs, so retirements later rejoin
  // the same globally unique factor identities in a SensorKnotAppend.
  [[nodiscard]] core::Result<VisualResidualIngestReport, LocalEstimatorError> applyVisualResiduals(
      core::CameraId camera, core::LocalGraphRevision revision,
      const std::vector<VisualResidualFeedback>& feedback);

  // Applies the configured initialization policy. Static initialization is
  // attempted only when an explicit matching ZeroMotionPrior is present;
  // dynamic initialization uses the bounded two-pass LiDAR--IMU batch. It then
  // consumes every queued sweep whose full acquisition interval has exact IMU
  // support. Expected modality rejections become explicit degraded commits;
  // invariant/solver failures return an error and place the estimator in
  // Faulted state.
  [[nodiscard]] core::Result<LocalEstimatorProcessReport, LocalEstimatorError> processReady();

  // Replays the same midpoint propagation used by deskew from the latest
  // committed optimized state. It is read-only: no state/request identity,
  // graph revision, queue, finality, or buffer retention is changed.
  [[nodiscard]] core::Result<LocalEstimatorPropagationReport, LocalEstimatorError> propagateTo(
      core::FusionTime exact_time) const;

  [[nodiscard]] LocalEstimatorLifecycle lifecycle() const noexcept;
  [[nodiscard]] const LocalEstimatorStatistics& statistics() const noexcept;
  [[nodiscard]] const FinalizedLidarTargetMapStatistics& finalizedLidarTargetStatistics()
      const noexcept;
  [[nodiscard]] LocalPipelineTimingReport pipelineTimingReport() const noexcept;
  [[nodiscard]] const LocalEstimatorConfig& effectiveConfig() const noexcept;
  [[nodiscard]] core::Result<LocalGraphCommit, LocalEstimatorError> estimate() const;

private:
  struct Impl;
  explicit LocalEstimator(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace meridian::local
