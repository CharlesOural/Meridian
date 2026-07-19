#include "meridian/local/visual_lane.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <utility>

namespace meridian::local {
namespace {

[[nodiscard]] VisualLaneError laneError(VisualLaneErrorCode code, VisualLaneStage stage,
                                        std::string detail) {
  return VisualLaneError{code, stage, std::move(detail), std::nullopt, std::nullopt};
}

[[nodiscard]] VisualLaneError frontendError(VisualFrontendError error) {
  VisualLaneError result =
      laneError(VisualLaneErrorCode::FrontendRejected, VisualLaneStage::Frontend,
                "visual frontend rejected the frame: " + error.detail);
  result.frontend = std::move(error);
  return result;
}

[[nodiscard]] VisualLaneError factorBuilderError(VisualLaneErrorCode code, VisualLaneStage stage,
                                                 VisualFactorBuilderError error) {
  VisualLaneError result =
      laneError(code, stage, "visual factor builder rejected the update: " + error.detail);
  result.factor_builder = std::move(error);
  return result;
}

[[nodiscard]] bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool validFactorBuilderConfig(const VisualFactorBuilderConfig& config) noexcept {
  return config.minimum_non_anchor_observations >= 1U &&
         config.maximum_pending_observations >= config.minimum_non_anchor_observations + 1U &&
         config.maximum_observations_per_track >= config.maximum_pending_observations &&
         config.maximum_active_tracks > 0U && config.maximum_missed_keyframes > 0U &&
         positiveFinite(config.minimum_baseline_m) && positiveFinite(config.minimum_parallax_rad) &&
         config.minimum_parallax_rad < 3.14159265358979323846 &&
         positiveFinite(config.minimum_range_m) && positiveFinite(config.maximum_range_m) &&
         config.minimum_range_m < config.maximum_range_m &&
         positiveFinite(config.triangulation_huber_angle_rad) &&
         config.triangulation_iterations > 0U &&
         positiveFinite(config.maximum_triangulation_condition) &&
         positiveFinite(config.maximum_inlier_reprojection_error_px) &&
         positiveFinite(config.maximum_reprojection_rmse_px) &&
         positiveFinite(config.huber_delta_sigma) &&
         positiveFinite(config.outlier_chi_squared_gate) &&
         config.outlier_commits_before_retirement > 0U;
}

[[nodiscard]] EquidistantCameraParameters cameraParameters(
    const core::PinholeEquidistantCameraModel& camera) {
  const auto distortion = camera.distortion();
  const core::ImageDimensions image = camera.imageSize();
  return EquidistantCameraParameters{image.width,   image.height, camera.fx(),   camera.fy(),
                                     camera.cx(),   camera.cy(),  distortion[0], distortion[1],
                                     distortion[2], distortion[3]};
}

[[nodiscard]] bool finiteState(const VisualLocalStateSnapshot& state) noexcept {
  return state.odom_epoch.valid() && state.state.valid() && state.observed_at_revision.valid() &&
         state.estimate.T_odom_imu.matrix().allFinite() &&
         state.estimate.velocity_odom.allFinite() && state.estimate.gyro_bias.allFinite() &&
         state.estimate.accel_bias.allFinite();
}

[[nodiscard]] bool containsRequest(const std::vector<core::KnotRequestId>& requests,
                                   core::KnotRequestId requested) {
  return std::find(requests.begin(), requests.end(), requested) != requests.end();
}

[[nodiscard]] bool sortedUniqueValidRequests(const std::vector<core::KnotRequestId>& requests) {
  if (requests.empty()) {
    return false;
  }
  for (std::size_t index = 0U; index < requests.size(); ++index) {
    if (!requests[index].valid() || (index > 0U && requests[index] <= requests[index - 1U])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool hasGraphWork(const VisualFactorBatch& batch) noexcept {
  return !batch.new_landmarks.empty() || !batch.factors.empty();
}

}  // namespace

struct VisualLane::Impl {
  struct PendingKeyframe {
    VisualFrontendOutput frontend;
    StateAdmission admission;
    core::OdomEpoch prediction_odom_epoch;
  };

  struct PreparedAttachment {
    VisualAttachmentId id;
    bool includes_factor_batch{};
    std::size_t retirement_count{};
  };

  Impl(core::CalibrationEpoch calibration_epoch_in, core::CameraId camera_id_in,
       EquidistantCameraParameters camera_model_in, core::ImuFromCameraTransform imu_from_camera_in,
       StateTimeline& state_timeline_in, VisualLaneConfig config_in)
      : calibration_epoch(calibration_epoch_in),
        camera_id(camera_id_in),
        camera_model(camera_model_in),
        imu_from_camera(std::move(imu_from_camera_in)),
        state_timeline(&state_timeline_in),
        config(std::move(config_in)),
        frontend(EquidistantCamera(camera_model), config.frontend),
        factor_builder(config.factor_builder) {}

  [[nodiscard]] VisualLaneQueueState queueState() const noexcept {
    VisualLaneQueueState result;
    result.pending_keyframes = pending_keyframes.size();
    for (const PendingKeyframe& pending : pending_keyframes) {
      result.pending_keyframe_features += pending.frontend.features.size();
    }
    result.pending_factor_batches = pending_factor_batches.size();
    for (const VisualFactorBatch& batch : pending_factor_batches) {
      result.pending_landmark_seeds += batch.new_landmarks.size();
      result.pending_factor_specs += batch.factors.size();
    }
    result.pending_factor_retirements = pending_factor_retirements.size();
    result.graph_attachment_in_flight = prepared_attachment.has_value();
    return result;
  }

  core::CalibrationEpoch calibration_epoch;
  core::CameraId camera_id;
  EquidistantCameraParameters camera_model;
  core::ImuFromCameraTransform imu_from_camera;
  StateTimeline* state_timeline;
  VisualLaneConfig config;
  GridKltVisualFrontend frontend;
  VisualFactorBatchBuilder factor_builder;
  std::deque<PendingKeyframe> pending_keyframes;
  std::deque<VisualFactorBatch> pending_factor_batches;
  std::deque<core::FactorId> pending_factor_retirements;
  std::optional<PreparedAttachment> prepared_attachment;
  std::optional<core::OdomEpoch> active_odom_epoch;
  std::optional<core::LocalGraphRevision> latest_observed_graph_revision;
  std::optional<core::LocalGraphRevision> latest_attachment_revision;
  std::uint64_t next_attachment_id{};
  VisualLaneStatistics statistics;
};

core::Result<VisualLane, VisualLaneError> VisualLane::create(
    const core::CalibrationBundle& calibration, core::CameraId camera,
    StateTimeline& state_timeline, VisualLaneConfig config) {
  using Result = core::Result<VisualLane, VisualLaneError>;
  const core::CameraCalibration* selected = calibration.camera(camera);
  if (selected == nullptr) {
    return Result::failure(
        laneError(VisualLaneErrorCode::CameraNotCalibrated, VisualLaneStage::Configuration,
                  "visual lane camera ID is absent from the calibration bundle"));
  }
  const EquidistantCameraParameters parameters = cameraParameters(selected->model());
  const EquidistantCamera camera_model(parameters);
  const std::size_t pending_observer_count =
      config.factor_builder.maximum_pending_observations > 0U
          ? config.factor_builder.maximum_pending_observations - 1U
          : 0U;
  const std::size_t factors_per_track = std::max<std::size_t>(1U, pending_observer_count);
  const bool factor_bound_representable =
      config.factor_builder.maximum_active_tracks <=
      std::numeric_limits<std::size_t>::max() / factors_per_track;
  const std::size_t worst_factors_per_keyframe =
      factor_bound_representable ? config.factor_builder.maximum_active_tracks * factors_per_track
                                 : std::numeric_limits<std::size_t>::max();
  const bool bounds_valid =
      config.maximum_pending_keyframes > 0U && config.maximum_pending_factor_batches > 0U &&
      config.maximum_pending_factor_retirements > 0U &&
      config.maximum_factor_retirements_per_attachment > 0U &&
      config.maximum_factor_retirements_per_attachment <=
          config.maximum_pending_factor_retirements &&
      config.maximum_landmarks_per_attachment >= config.factor_builder.maximum_active_tracks &&
      factor_bound_representable &&
      config.maximum_factors_per_attachment >= worst_factors_per_keyframe;
  if (!calibration.epoch().valid() || !camera.valid() || selected->id() != camera ||
      !camera_model.valid() || !config.frontend.valid(camera_model) ||
      !validFactorBuilderConfig(config.factor_builder) || !bounds_valid) {
    return Result::failure(
        laneError(VisualLaneErrorCode::InvalidConfiguration, VisualLaneStage::Configuration,
                  "visual camera, frontend, factor builder, or queue/transaction bounds "
                  "are invalid"));
  }

  auto implementation =
      std::make_unique<Impl>(calibration.epoch(), camera, parameters, selected->extrinsics(),
                             state_timeline, std::move(config));
  return Result::success(VisualLane(std::move(implementation)));
}

VisualLane::VisualLane(std::unique_ptr<Impl> implementation)
    : implementation_(std::move(implementation)) {}

VisualLane::~VisualLane() = default;
VisualLane::VisualLane(VisualLane&&) noexcept = default;
VisualLane& VisualLane::operator=(VisualLane&&) noexcept = default;

core::Result<VisualFrameReport, VisualLaneError> VisualLane::processFrame(VisualFrameInput input) {
  using Result = core::Result<VisualFrameReport, VisualLaneError>;
  Impl& state = *implementation_;
  const core::CameraFrame& frame = input.frame;
  const bool frame_valid = frame.id.valid() && frame.camera == state.camera_id &&
                           frame.header.direct_calibration == state.calibration_epoch &&
                           frame.exposure_midpoint == frame.stamp.fusion_time &&
                           input.state_request.valid();
  if (!frame_valid) {
    ++state.statistics.frames_rejected;
    return Result::failure(
        laneError(VisualLaneErrorCode::InvalidFrameInput, VisualLaneStage::FrameIngress,
                  "camera identity, calibration, exposure midpoint, fusion stamp, or "
                  "preallocated state-request ID is invalid"));
  }
  if (input.imu_rotation_seed &&
      input.imu_rotation_seed->imu_calibration != state.calibration_epoch) {
    ++state.statistics.frames_rejected;
    return Result::failure(laneError(VisualLaneErrorCode::InvalidFrameInput,
                                     VisualLaneStage::FrameIngress,
                                     "IMU rotation seed does not name the lane calibration epoch"));
  }
  if (input.latest_local_state) {
    const VisualLocalStateSnapshot& local = *input.latest_local_state;
    if (!finiteState(local) || local.exact_time > frame.exposure_midpoint ||
        (state.active_odom_epoch && local.odom_epoch != *state.active_odom_epoch) ||
        (state.latest_observed_graph_revision &&
         local.observed_at_revision < *state.latest_observed_graph_revision)) {
      ++state.statistics.frames_rejected;
      return Result::failure(
          laneError(VisualLaneErrorCode::InvalidLocalState, VisualLaneStage::FrameIngress,
                    "local prediction anchor is invalid, newer than the exposure, from "
                    "another odom epoch, or from a regressed graph revision"));
    }
  }

  auto frontend_result = state.frontend.process(input.frame, std::move(input.imu_rotation_seed),
                                                input.request_keyframe);
  if (!frontend_result) {
    ++state.statistics.frames_rejected;
    return Result::failure(frontendError(frontend_result.error()));
  }

  VisualFrontendOutput frontend = std::move(frontend_result).value();
  ++state.statistics.frames_accepted;
  if (input.latest_local_state) {
    state.active_odom_epoch = input.latest_local_state->odom_epoch;
    state.latest_observed_graph_revision = input.latest_local_state->observed_at_revision;
  }

  VisualFrameReport report;
  report.frontend = frontend;
  if (input.latest_local_state) {
    report.prediction_anchor_revision = input.latest_local_state->observed_at_revision;
  }
  if (frontend.keyframe) {
    ++state.statistics.frontend_keyframes;
  }
  if (!state.config.graph_submission_enabled) {
    ++state.statistics.frames_tracking_only_graph_submission_disabled;
    report.disposition = VisualFrameDisposition::TrackingOnlyGraphSubmissionDisabled;
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }
  if (!frontend.keyframe) {
    report.disposition = VisualFrameDisposition::TrackingOnly;
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }

  if (!input.latest_local_state) {
    ++state.statistics.keyframes_without_local_state;
    report.disposition = VisualFrameDisposition::TrackingOnlyNoLocalState;
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }
  if (state.pending_keyframes.size() >= state.config.maximum_pending_keyframes) {
    ++state.statistics.keyframes_suppressed_by_capacity;
    report.disposition = VisualFrameDisposition::KeyframeSuppressedByCapacity;
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }

  StateRequest request;
  request.header = frame.header;
  request.id = input.state_request;
  request.sensor = core::SensorInstanceId::camera(state.camera_id);
  request.purpose = StateRequestPurpose::VisualKeyframe;
  request.exact_time = frontend.exposure_midpoint;
  request.lineage = frontend.lineage;
  auto admission = state.state_timeline->request(std::move(request));
  if (!admission) {
    ++state.statistics.keyframes_rejected_by_timeline;
    report.disposition = VisualFrameDisposition::KeyframeRejectedByTimeline;
    report.timeline_rejection = admission.error();
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }

  report.state_admission = admission.value();
  if (admission.value().disposition == StateAdmissionDisposition::SuppressedTooClose) {
    ++state.statistics.keyframes_suppressed_by_timeline;
    report.disposition = VisualFrameDisposition::KeyframeSuppressedByTimeline;
    report.queues = state.queueState();
    return Result::success(std::move(report));
  }

  report.disposition = VisualFrameDisposition::KeyframeStateRequested;
  state.pending_keyframes.push_back(Impl::PendingKeyframe{std::move(frontend), admission.value(),
                                                          input.latest_local_state->odom_epoch});
  ++state.statistics.state_requests_admitted;
  report.queues = state.queueState();
  return Result::success(std::move(report));
}

core::Result<VisualResolvedKeyframeReport, VisualLaneError> VisualLane::resolveCommittedKeyframe(
    VisualStateResolution resolution, VisualLocalStateSnapshot exact_state) {
  using Result = core::Result<VisualResolvedKeyframeReport, VisualLaneError>;
  Impl& state = *implementation_;
  const auto match = std::find_if(state.pending_keyframes.begin(), state.pending_keyframes.end(),
                                  [&resolution](const Impl::PendingKeyframe& pending) {
                                    return pending.admission.request.id == resolution.request;
                                  });
  if (match == state.pending_keyframes.end()) {
    return Result::failure(
        laneError(VisualLaneErrorCode::UnknownStateRequest, VisualLaneStage::StateResolution,
                  "state resolution does not refer to a pending visual keyframe"));
  }
  if (match != state.pending_keyframes.begin()) {
    return Result::failure(
        laneError(VisualLaneErrorCode::OutOfOrderStateResolution, VisualLaneStage::StateResolution,
                  "visual keyframe resolutions must be consumed in exposure-time order"));
  }
  if (state.pending_factor_batches.size() >= state.config.maximum_pending_factor_batches) {
    return Result::failure(laneError(VisualLaneErrorCode::PendingFactorBatchCapacity,
                                     VisualLaneStage::FactorBuilder,
                                     "bounded pending visual-factor queue is full"));
  }

  const Impl::PendingKeyframe& pending = state.pending_keyframes.front();
  const bool resolution_valid =
      resolution.request.valid() && resolution.odom_epoch.valid() && resolution.state.valid() &&
      resolution.created_at_revision.valid() &&
      resolution.request == pending.admission.request.id &&
      resolution.timeline.exact_time == pending.frontend.exposure_midpoint &&
      resolution.timeline.committed_state == resolution.state &&
      resolution.odom_epoch == pending.prediction_odom_epoch &&
      resolution.header.direct_calibration == state.calibration_epoch &&
      sortedUniqueValidRequests(resolution.timeline.requests) &&
      containsRequest(resolution.timeline.requests, resolution.request);
  const bool state_valid = finiteState(exact_state) &&
                           exact_state.odom_epoch == resolution.odom_epoch &&
                           exact_state.state == resolution.state &&
                           exact_state.exact_time == resolution.timeline.exact_time &&
                           exact_state.observed_at_revision >= resolution.created_at_revision;
  if (!resolution_valid || !state_valid) {
    return Result::failure(
        laneError(VisualLaneErrorCode::InvalidStateResolution, VisualLaneStage::StateResolution,
                  "resolution identity, exact time, shared requests, calibration, odom "
                  "epoch, or graph-owned state snapshot is inconsistent"));
  }

  VisualKeyframeContext context;
  context.state = resolution.state;
  context.exact_time = resolution.timeline.exact_time;
  context.camera = state.camera_id;
  context.calibration = state.calibration_epoch;
  context.camera_model = state.camera_model;
  context.imu_from_camera = state.imu_from_camera;
  context.T_odom_imu = exact_state.estimate.T_odom_imu;
  auto built = state.factor_builder.processKeyframe(pending.frontend, context);
  if (!built) {
    return Result::failure(factorBuilderError(VisualLaneErrorCode::FactorBuilderRejected,
                                              VisualLaneStage::FactorBuilder, built.error()));
  }

  VisualFactorBatch batch = std::move(built).value();
  VisualResolvedKeyframeReport report;
  report.resolution = resolution;
  report.factor_builder = batch.report;
  report.descriptors = pending.frontend.descriptors;
  report.graph_batch_queued = hasGraphWork(batch);
  if (report.graph_batch_queued) {
    state.pending_factor_batches.push_back(std::move(batch));
    ++state.statistics.factor_batches_queued;
  }
  state.pending_keyframes.pop_front();
  state.active_odom_epoch = resolution.odom_epoch;
  state.latest_observed_graph_revision = exact_state.observed_at_revision;
  ++state.statistics.keyframes_resolved;
  report.queues = state.queueState();
  return Result::success(std::move(report));
}

core::Result<VisualResidualIngestReport, VisualLaneError> VisualLane::applyAcceptedResiduals(
    core::LocalGraphRevision revision, const std::vector<VisualResidualFeedback>& feedback) {
  using Result = core::Result<VisualResidualIngestReport, VisualLaneError>;
  Impl& state = *implementation_;
  if (feedback.size() >
      state.config.maximum_pending_factor_retirements - state.pending_factor_retirements.size()) {
    return Result::failure(
        laneError(VisualLaneErrorCode::PendingRetirementCapacity, VisualLaneStage::ResidualFeedback,
                  "residual feedback could exceed the bounded retirement queue"));
  }
  auto feedback_result = state.factor_builder.applyAcceptedResiduals(revision, feedback);
  if (!feedback_result) {
    return Result::failure(factorBuilderError(VisualLaneErrorCode::ResidualFeedbackRejected,
                                              VisualLaneStage::ResidualFeedback,
                                              feedback_result.error()));
  }

  VisualResidualIngestReport report;
  report.factor_builder = std::move(feedback_result).value();
  report.retirements_queued = report.factor_builder.retired_observation_factors.size();
  for (const core::FactorId factor : report.factor_builder.retired_observation_factors) {
    state.pending_factor_retirements.push_back(factor);
  }
  state.statistics.residual_feedback_items += feedback.size();
  state.statistics.factor_retirements_queued += report.retirements_queued;
  report.queues = state.queueState();
  return Result::success(std::move(report));
}

core::Result<VisualGraphInput, VisualLaneError> VisualLane::prepareGraphInput(
    const ImuKnotAppend& candidate_navigation) {
  using Result = core::Result<VisualGraphInput, VisualLaneError>;
  Impl& state = *implementation_;
  if (state.prepared_attachment) {
    return Result::failure(laneError(
        VisualLaneErrorCode::GraphAttachmentAlreadyInFlight, VisualLaneStage::GraphAttachment,
        "acknowledge the current visual graph input before preparing another"));
  }
  if (state.pending_factor_batches.empty() && state.pending_factor_retirements.empty()) {
    return Result::failure(laneError(VisualLaneErrorCode::NoGraphInputReady,
                                     VisualLaneStage::GraphAttachment,
                                     "no visual factor batch or factor retirement is queued"));
  }
  if (!candidate_navigation.state.valid()) {
    return Result::failure(laneError(VisualLaneErrorCode::InvalidGraphCandidate,
                                     VisualLaneStage::GraphAttachment,
                                     "candidate navigation state identity is invalid"));
  }
  if (!state.pending_factor_batches.empty()) {
    const VisualFactorBatch& batch = state.pending_factor_batches.front();
    if (candidate_navigation.exact_time <= batch.exact_time ||
        batch.new_landmarks.size() > state.config.maximum_landmarks_per_attachment ||
        batch.factors.size() > state.config.maximum_factors_per_attachment) {
      return Result::failure(
          laneError(VisualLaneErrorCode::InvalidGraphCandidate, VisualLaneStage::GraphAttachment,
                    "candidate transaction must follow the resolved keyframe and the "
                    "queued batch must fit graph transaction bounds"));
    }
  }
  if (state.next_attachment_id == VisualAttachmentId::kInvalidValue) {
    return Result::failure(laneError(VisualLaneErrorCode::AttachmentIdentityExhausted,
                                     VisualLaneStage::GraphAttachment,
                                     "visual graph attachment identity exhausted"));
  }

  VisualGraphInput output;
  output.id = VisualAttachmentId(state.next_attachment_id++);
  output.candidate_state = candidate_navigation.state;
  output.candidate_time = candidate_navigation.exact_time;
  if (!state.pending_factor_batches.empty()) {
    output.visual = state.pending_factor_batches.front();
    // The batch time is the containing graph transaction time. Observation
    // refs inside every factor keep the exact resolved camera timestamps.
    output.visual->exact_time = candidate_navigation.exact_time;
  }
  const std::size_t retirement_count =
      std::min(state.pending_factor_retirements.size(),
               state.config.maximum_factor_retirements_per_attachment);
  output.visual_factor_retirements.reserve(retirement_count);
  for (std::size_t index = 0U; index < retirement_count; ++index) {
    output.visual_factor_retirements.push_back(state.pending_factor_retirements[index]);
  }
  state.prepared_attachment =
      Impl::PreparedAttachment{output.id, output.visual.has_value(), retirement_count};
  ++state.statistics.graph_inputs_prepared;
  return Result::success(std::move(output));
}

core::Result<VisualLaneQueueState, VisualLaneError> VisualLane::acknowledgeGraphInputAccepted(
    VisualAttachmentId attachment, core::LocalGraphRevision committed_revision) {
  using Result = core::Result<VisualLaneQueueState, VisualLaneError>;
  Impl& state = *implementation_;
  if (!state.prepared_attachment || state.prepared_attachment->id != attachment) {
    return Result::failure(
        laneError(VisualLaneErrorCode::UnknownGraphAttachment, VisualLaneStage::GraphAttachment,
                  "accepted graph acknowledgement does not match the in-flight input"));
  }
  if (!committed_revision.valid() || (state.latest_attachment_revision &&
                                      committed_revision <= *state.latest_attachment_revision)) {
    return Result::failure(
        laneError(VisualLaneErrorCode::InvalidGraphRevision, VisualLaneStage::GraphAttachment,
                  "accepted graph revision must be valid and strictly increasing"));
  }

  if (state.prepared_attachment->includes_factor_batch) {
    state.pending_factor_batches.pop_front();
  }
  for (std::size_t index = 0U; index < state.prepared_attachment->retirement_count; ++index) {
    state.pending_factor_retirements.pop_front();
  }
  state.latest_attachment_revision = committed_revision;
  state.prepared_attachment.reset();
  ++state.statistics.graph_inputs_accepted;
  return Result::success(state.queueState());
}

core::Result<VisualLaneQueueState, VisualLaneError> VisualLane::acknowledgeGraphInputRejected(
    VisualAttachmentId attachment) {
  using Result = core::Result<VisualLaneQueueState, VisualLaneError>;
  Impl& state = *implementation_;
  if (!state.prepared_attachment || state.prepared_attachment->id != attachment) {
    return Result::failure(
        laneError(VisualLaneErrorCode::UnknownGraphAttachment, VisualLaneStage::GraphAttachment,
                  "rejected graph acknowledgement does not match the in-flight input"));
  }
  state.prepared_attachment.reset();
  ++state.statistics.graph_inputs_rejected;
  return Result::success(state.queueState());
}

core::Result<VisualGraphInputDegradationReport, VisualLaneError>
VisualLane::acknowledgeGraphInputDegraded(VisualAttachmentId attachment) {
  using Result = core::Result<VisualGraphInputDegradationReport, VisualLaneError>;
  Impl& state = *implementation_;
  if (!state.prepared_attachment || state.prepared_attachment->id != attachment) {
    return Result::failure(
        laneError(VisualLaneErrorCode::UnknownGraphAttachment, VisualLaneStage::GraphAttachment,
                  "degraded graph acknowledgement does not match the in-flight input"));
  }

  VisualGraphInputDegradationReport report;
  report.attachment = attachment;
  report.factor_retirements_preserved = state.pending_factor_retirements.size();
  for (const VisualFactorBatch& batch : state.pending_factor_batches) {
    const VisualUncommittedBatchDiscardReport discarded =
        state.factor_builder.discardUncommittedBatch(batch);
    ++report.factor_batches_discarded;
    report.landmark_seeds_discarded += discarded.landmark_seeds_discarded;
    report.active_landmark_initializations_rolled_back +=
        discarded.active_landmark_initializations_rolled_back;
    report.stale_track_observations_discarded += discarded.stale_track_observations_discarded;
    report.factors_discarded += discarded.factors_discarded;
    report.factor_health_entries_removed += discarded.factor_health_entries_removed;
  }
  state.pending_factor_batches.clear();
  state.prepared_attachment.reset();
  ++state.statistics.graph_inputs_degraded;
  state.statistics.uncommitted_factor_batches_discarded += report.factor_batches_discarded;
  state.statistics.stale_track_observations_discarded += report.stale_track_observations_discarded;
  report.queues = state.queueState();
  return Result::success(std::move(report));
}

core::Result<VisualLaneFinalityReport, VisualLaneError> VisualLane::reconcileGraphFinality(
    const VisualFinalityUpdate& update) {
  using Result = core::Result<VisualLaneFinalityReport, VisualLaneError>;
  Impl& state = *implementation_;
  if (state.prepared_attachment) {
    return Result::failure(laneError(
        VisualLaneErrorCode::GraphAttachmentAlreadyInFlight, VisualLaneStage::GraphAttachment,
        "acknowledge the in-flight graph input before reconciling committed finality"));
  }

  const std::set<core::FactorId> finalized_factors(update.finalized_factors.begin(),
                                                   update.finalized_factors.end());
  std::set<VisualLandmarkId> invalid_landmarks(update.finalized_landmarks.begin(),
                                               update.finalized_landmarks.end());
  std::deque<VisualFactorBatch> candidate_batches = state.pending_factor_batches;
  std::deque<core::FactorId> candidate_retirements = state.pending_factor_retirements;

  VisualLaneFinalityReport report;
  report.revision = update.revision;
  for (auto batch = candidate_batches.begin(); batch != candidate_batches.end();) {
    auto factor = batch->factors.begin();
    while (factor != batch->factors.end()) {
      if (finalized_factors.contains(factor->id) || invalid_landmarks.contains(factor->landmark)) {
        factor = batch->factors.erase(factor);
        ++report.pending_factor_specs_pruned;
      } else {
        ++factor;
      }
    }

    // A queued seed cannot be committed without at least one observation
    // factor in its own batch. If child finality removed its last factor, prune
    // the seed and make that dependency unavailable to every later batch.
    auto seed = batch->new_landmarks.begin();
    while (seed != batch->new_landmarks.end()) {
      const bool has_child = std::any_of(batch->factors.begin(), batch->factors.end(),
                                         [&](const VisualReprojectionFactorSpec& child) {
                                           return child.landmark == seed->landmark;
                                         });
      if (invalid_landmarks.contains(seed->landmark) || !has_child) {
        invalid_landmarks.insert(seed->landmark);
        seed = batch->new_landmarks.erase(seed);
        ++report.pending_landmark_seeds_pruned;
      } else {
        ++seed;
      }
    }

    const auto retired_track_end = std::remove_if(
        batch->retired_tracks.begin(), batch->retired_tracks.end(),
        [&](const VisualTrackRetirement& retirement) {
          return retirement.landmark.valid() && invalid_landmarks.contains(retirement.landmark);
        });
    report.pending_track_retirements_pruned +=
        static_cast<std::size_t>(std::distance(retired_track_end, batch->retired_tracks.end()));
    batch->retired_tracks.erase(retired_track_end, batch->retired_tracks.end());
    batch->report.factors_emitted = batch->factors.size();
    batch->report.tracks_initialized = batch->new_landmarks.size();

    if (!hasGraphWork(*batch)) {
      batch = candidate_batches.erase(batch);
      ++report.pending_batches_pruned;
    } else {
      ++batch;
    }
  }

  // A landmark invalidated by an earlier seed can also be referenced by a
  // later batch. Perform the bounded queue pass again after the dependency set
  // is complete; this is deterministic and avoids leaving a stale child that
  // would be rejected repeatedly by the graph.
  for (auto batch = candidate_batches.begin(); batch != candidate_batches.end();) {
    const auto factor_end = std::remove_if(batch->factors.begin(), batch->factors.end(),
                                           [&](const VisualReprojectionFactorSpec& factor) {
                                             return invalid_landmarks.contains(factor.landmark);
                                           });
    report.pending_factor_specs_pruned +=
        static_cast<std::size_t>(std::distance(factor_end, batch->factors.end()));
    batch->factors.erase(factor_end, batch->factors.end());
    batch->report.factors_emitted = batch->factors.size();
    if (!hasGraphWork(*batch)) {
      batch = candidate_batches.erase(batch);
      ++report.pending_batches_pruned;
    } else {
      ++batch;
    }
  }

  const auto retirement_end =
      std::remove_if(candidate_retirements.begin(), candidate_retirements.end(),
                     [&](core::FactorId factor) { return finalized_factors.contains(factor); });
  report.pending_factor_retirements_pruned +=
      static_cast<std::size_t>(std::distance(retirement_end, candidate_retirements.end()));
  candidate_retirements.erase(retirement_end, candidate_retirements.end());

  auto reconciled = state.factor_builder.reconcileFinality(update);
  if (!reconciled) {
    return Result::failure(factorBuilderError(VisualLaneErrorCode::InvalidGraphFinality,
                                              VisualLaneStage::GraphAttachment,
                                              reconciled.error()));
  }
  report.factor_builder = std::move(reconciled).value();
  state.pending_factor_batches = std::move(candidate_batches);
  state.pending_factor_retirements = std::move(candidate_retirements);
  ++state.statistics.graph_finality_updates;
  state.statistics.finalized_tracks_pruned += report.factor_builder.accepted_tracks_pruned;
  state.statistics.finalized_pending_factor_specs_pruned += report.pending_factor_specs_pruned;
  report.queues = state.queueState();
  return Result::success(std::move(report));
}

const VisualLaneStatistics& VisualLane::statistics() const noexcept {
  return implementation_->statistics;
}

VisualLaneQueueState VisualLane::queueState() const noexcept {
  return implementation_->queueState();
}

}  // namespace meridian::local
