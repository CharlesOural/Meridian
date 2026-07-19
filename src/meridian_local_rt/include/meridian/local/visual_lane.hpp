#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/event_scheduler.hpp"
#include "meridian/local/graph.hpp"
#include "meridian/local/visual_factor.hpp"
#include "meridian/local/visual_frontend.hpp"

namespace meridian::local {

// One visual lane owns one calibrated camera stream. State IDs remain owned by
// the shared graph writer; this component never allocates a modality pose.
class VisualAttachmentId {
public:
  constexpr VisualAttachmentId() = default;
  explicit constexpr VisualAttachmentId(std::uint64_t value) : value_(value) {}
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalidValue; }
  auto operator<=>(const VisualAttachmentId&) const = default;

  static constexpr std::uint64_t kInvalidValue = std::numeric_limits<std::uint64_t>::max();

private:
  std::uint64_t value_{kInvalidValue};
};

// Immutable view of one graph-owned navigation state. For frame processing it
// is the latest committed state and may precede the exposure. For factor
// construction it must exactly match the resolved camera state.
struct VisualLocalStateSnapshot {
  core::OdomEpoch odom_epoch;
  core::StateId state;
  core::FusionTime exact_time;
  core::LocalGraphRevision observed_at_revision;
  core::NavStateEstimate estimate;
};

// Graph-writer response to one visual request in an admitted timeline state.
// The sensor-neutral timeline resolution remains intact so exact sharing is
// observable without inventing a camera-owned navigation state.
struct VisualStateResolution {
  core::RecordHeader header;
  core::KnotRequestId request;
  StateResolution timeline;
  core::OdomEpoch odom_epoch;
  core::StateId state;
  core::LocalGraphRevision created_at_revision;
};

struct VisualLaneConfig {
  VisualFrontendConfig frontend;
  VisualFactorBuilderConfig factor_builder;

  // Benchmark/fault-isolation switch. Tracking, IMU rotation seeding,
  // frontend keyframe selection and descriptors remain active. When false,
  // the lane never requests graph states or queues visual factors.
  bool graph_submission_enabled{true};

  // All bounds reject new work. Dropping an older accepted keyframe or factor
  // batch would make track and graph ancestry inconsistent.
  std::size_t maximum_pending_keyframes{8U};
  std::size_t maximum_pending_factor_batches{8U};
  std::size_t maximum_pending_factor_retirements{4096U};
  std::size_t maximum_factor_retirements_per_attachment{4096U};

  // These must cover the factor builder's single-keyframe worst case and must
  // also match or undercut the LocalGraph transaction bounds used by the app.
  std::size_t maximum_landmarks_per_attachment{512U};
  std::size_t maximum_factors_per_attachment{4096U};
};

struct VisualFrameInput {
  core::CameraFrame frame;
  std::optional<VisualRotationPrior> imu_rotation_seed;
  std::optional<VisualLocalStateSnapshot> latest_local_state;

  // Allocated by the composition root's shared ID source. It may go unused
  // when this image is not selected as a keyframe.
  core::KnotRequestId state_request;
  bool request_keyframe{};
};

enum class VisualFrameDisposition {
  TrackingOnly,
  TrackingOnlyGraphSubmissionDisabled,
  TrackingOnlyNoLocalState,
  KeyframeStateRequested,
  KeyframeSuppressedByCapacity,
  KeyframeSuppressedByTimeline,
  KeyframeRejectedByTimeline,
};

struct VisualLaneQueueState {
  std::size_t pending_keyframes{};
  std::size_t pending_keyframe_features{};
  std::size_t pending_factor_batches{};
  std::size_t pending_landmark_seeds{};
  std::size_t pending_factor_specs{};
  std::size_t pending_factor_retirements{};
  bool graph_attachment_in_flight{};
};

struct VisualFrameReport {
  VisualFrontendOutput frontend;
  VisualFrameDisposition disposition{VisualFrameDisposition::TrackingOnly};
  std::optional<StateAdmission> state_admission;
  std::optional<StateTimelineError> timeline_rejection;
  std::optional<core::LocalGraphRevision> prediction_anchor_revision;
  VisualLaneQueueState queues;
};

struct VisualResolvedKeyframeReport {
  VisualStateResolution resolution;
  VisualFactorBatchReport factor_builder;
  std::vector<BriskTrackDescriptor> descriptors;
  bool graph_batch_queued{};
  VisualLaneQueueState queues;
};

struct VisualResidualIngestReport {
  VisualResidualFeedbackReport factor_builder;
  std::size_t retirements_queued{};
  VisualLaneQueueState queues;
};

// Copyable graph input prepared from the front of the lane's queues. The
// caller moves these two fields directly into SensorKnotAppend. Accepted and
// rejected graph updates must be reported through the acknowledgement API.
struct VisualGraphInput {
  VisualAttachmentId id;
  core::StateId candidate_state;
  core::FusionTime candidate_time;
  std::optional<VisualFactorBatch> visual;
  std::vector<core::FactorId> visual_factor_retirements;
};

struct VisualGraphInputDegradationReport {
  VisualAttachmentId attachment;
  std::size_t factor_batches_discarded{};
  std::size_t landmark_seeds_discarded{};
  std::size_t active_landmark_initializations_rolled_back{};
  std::size_t stale_track_observations_discarded{};
  std::size_t factors_discarded{};
  std::size_t factor_health_entries_removed{};
  std::size_t factor_retirements_preserved{};
  VisualLaneQueueState queues;
};

struct VisualLaneFinalityReport {
  core::LocalGraphRevision revision;
  VisualFactorFinalityReport factor_builder;
  std::size_t pending_batches_pruned{};
  std::size_t pending_landmark_seeds_pruned{};
  std::size_t pending_factor_specs_pruned{};
  std::size_t pending_track_retirements_pruned{};
  std::size_t pending_factor_retirements_pruned{};
  VisualLaneQueueState queues;
};

struct VisualLaneStatistics {
  std::size_t frames_accepted{};
  std::size_t frames_rejected{};
  std::size_t frames_tracking_only_graph_submission_disabled{};
  std::size_t frontend_keyframes{};
  std::size_t state_requests_admitted{};
  std::size_t keyframes_without_local_state{};
  std::size_t keyframes_suppressed_by_capacity{};
  std::size_t keyframes_suppressed_by_timeline{};
  std::size_t keyframes_rejected_by_timeline{};
  std::size_t keyframes_resolved{};
  std::size_t factor_batches_queued{};
  std::size_t graph_inputs_prepared{};
  std::size_t graph_inputs_accepted{};
  std::size_t graph_inputs_rejected{};
  std::size_t graph_inputs_degraded{};
  std::size_t uncommitted_factor_batches_discarded{};
  std::size_t stale_track_observations_discarded{};
  std::size_t residual_feedback_items{};
  std::size_t factor_retirements_queued{};
  std::size_t graph_finality_updates{};
  std::size_t finalized_tracks_pruned{};
  std::size_t finalized_pending_factor_specs_pruned{};
};

enum class VisualLaneStage {
  Configuration,
  FrameIngress,
  Frontend,
  StateResolution,
  FactorBuilder,
  ResidualFeedback,
  GraphAttachment,
};

enum class VisualLaneErrorCode {
  InvalidConfiguration,
  CameraNotCalibrated,
  InvalidFrameInput,
  InvalidLocalState,
  FrontendRejected,
  UnknownStateRequest,
  OutOfOrderStateResolution,
  InvalidStateResolution,
  FactorBuilderRejected,
  PendingFactorBatchCapacity,
  PendingRetirementCapacity,
  ResidualFeedbackRejected,
  GraphAttachmentAlreadyInFlight,
  NoGraphInputReady,
  InvalidGraphCandidate,
  AttachmentIdentityExhausted,
  UnknownGraphAttachment,
  InvalidGraphRevision,
  InvalidGraphFinality,
};

struct VisualLaneError {
  VisualLaneErrorCode code{VisualLaneErrorCode::InvalidConfiguration};
  VisualLaneStage stage{VisualLaneStage::Configuration};
  std::string detail;
  std::optional<VisualFrontendError> frontend;
  std::optional<VisualFactorBuilderError> factor_builder;
};

// ROS-free visual coordinator based on the OKVIS/OKVIS2-X ordering lesson in
// `okvis_multisensor_processing/src/ThreadedSlam.cpp` (state admission before
// frontend association) and `okvis_frontend/src/Frontend.cpp` (new landmark
// initialization only on keyframes). Meridian retains graph state authority
// while tracking every image and requesting states only for selected frames.
class VisualLane {
public:
  [[nodiscard]] static core::Result<VisualLane, VisualLaneError> create(
      const core::CalibrationBundle& calibration, core::CameraId camera,
      StateTimeline& state_timeline, VisualLaneConfig config = {});

  ~VisualLane();
  VisualLane(VisualLane&&) noexcept;
  VisualLane& operator=(VisualLane&&) noexcept;
  VisualLane(const VisualLane&) = delete;
  VisualLane& operator=(const VisualLane&) = delete;

  // Processes tracking immediately. A selected keyframe is retained only when
  // its exact-time state request is admitted by the shared timeline.
  [[nodiscard]] core::Result<VisualFrameReport, VisualLaneError> processFrame(
      VisualFrameInput input);

  // Called only after the graph writer has committed and published the state
  // resolution. The exact state estimate is used for triangulation; no pose is
  // extrapolated or copied from the frame-time prediction anchor.
  [[nodiscard]] core::Result<VisualResolvedKeyframeReport, VisualLaneError>
  resolveCommittedKeyframe(VisualStateResolution resolution, VisualLocalStateSnapshot exact_state);

  // Feedback must come from an accepted LocalGraph revision. Resulting factor
  // retirements are queued for a later atomic SensorKnotAppend.
  [[nodiscard]] core::Result<VisualResidualIngestReport, VisualLaneError> applyAcceptedResiduals(
      core::LocalGraphRevision revision, const std::vector<VisualResidualFeedback>& feedback);

  // Factors are deliberately attached after their observation state has
  // committed. `candidate_time` is the transaction envelope time; immutable
  // factor endpoints retain their exact camera times and StateIds.
  [[nodiscard]] core::Result<VisualGraphInput, VisualLaneError> prepareGraphInput(
      const ImuKnotAppend& candidate_navigation);

  // Accepted removes precisely the prepared queue prefix. Rejected retains it
  // for deterministic retry at a later navigation transaction.
  [[nodiscard]] core::Result<VisualLaneQueueState, VisualLaneError> acknowledgeGraphInputAccepted(
      VisualAttachmentId attachment, core::LocalGraphRevision committed_revision);
  [[nodiscard]] core::Result<VisualLaneQueueState, VisualLaneError> acknowledgeGraphInputRejected(
      VisualAttachmentId attachment);
  // Called only after the same navigation/LiDAR transaction commits without
  // visual work. All queued, not-yet-accepted batches are discarded together
  // because later batches may depend on a landmark seeded by the rejected
  // front batch. Accepted factor state and queued retirements are retained.
  [[nodiscard]] core::Result<VisualGraphInputDegradationReport, VisualLaneError>
  acknowledgeGraphInputDegraded(VisualAttachmentId attachment);

  // Called after a successful graph commit and after any in-flight attachment
  // acknowledgement. It prunes accepted builder state and all queued work that
  // depends on finalized graph objects without recording a degradation.
  [[nodiscard]] core::Result<VisualLaneFinalityReport, VisualLaneError> reconcileGraphFinality(
      const VisualFinalityUpdate& update);

  [[nodiscard]] const VisualLaneStatistics& statistics() const noexcept;
  [[nodiscard]] VisualLaneQueueState queueState() const noexcept;

private:
  struct Impl;
  explicit VisualLane(std::unique_ptr<Impl> implementation);
  std::unique_ptr<Impl> implementation_;
};

}  // namespace meridian::local
