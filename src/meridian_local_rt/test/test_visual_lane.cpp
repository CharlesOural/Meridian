#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <utility>
#include <vector>

#include "meridian/local/visual_lane.hpp"

namespace meridian::local {
namespace {

constexpr std::uint32_t kImageWidth = 320U;
constexpr std::uint32_t kImageHeight = 240U;

[[nodiscard]] core::CalibrationBundle calibration() {
  core::ImuCalibration imu("imu", "/imu", core::ImuSensorModel::Generic, 200.0, 9.80665,
                           core::ImuNoiseModel(1.0e-3, 2.0e-4, 3.0e-5, 4.0e-6));
  core::LidarCalibration lidar(
      core::LidarId(0U), "lidar", "/points", core::ImuFromLidarTransform(core::Pose3d{}),
      core::LidarTimingCalibration(core::LidarSweepTimestampReference::SweepStart,
                                   core::LidarPointTimeConvention::OffsetFromSweepTimestamp));
  core::CameraCalibration camera(
      core::CameraId(0U), "camera", "/camera",
      core::PinholeEquidistantCameraModel(200.0, 200.0, 159.5, 119.5,
                                          std::array<double, 4>{0.0, 0.0, 0.0, 0.0},
                                          core::ImageDimensions{kImageWidth, kImageHeight}),
      core::ImuFromCameraTransform(core::Pose3d{}),
      core::CameraTimingCalibration(core::CameraTimestampReference::ExposureMidpoint,
                                    core::Duration{0LL}));
  auto result = core::CalibrationBundle::create(core::CalibrationEpoch(7U), std::move(imu),
                                                core::BaseFromImuTransform(core::Pose3d{}),
                                                std::move(lidar), {std::move(camera)});
  EXPECT_TRUE(result);
  return std::move(result).value();
}

[[nodiscard]] VisualLaneConfig laneConfig() {
  VisualLaneConfig config;
  config.frontend.features_per_cell = 3U;
  config.frontend.recovery_minimum_tracks = 20U;
  config.frontend.keyframe_minimum_tracks = 40U;
  config.frontend.minimum_keyframe_interval = core::Duration{100'000'000LL};
  config.frontend.maximum_keyframe_interval = core::Duration{500'000'000LL};
  config.factor_builder.maximum_active_tracks = 144U;
  config.factor_builder.maximum_pending_observations = 4U;
  config.factor_builder.maximum_observations_per_track = 12U;
  config.factor_builder.minimum_baseline_m = 0.02;
  config.factor_builder.minimum_parallax_rad = 0.008;
  config.factor_builder.maximum_inlier_reprojection_error_px = 5.0;
  config.factor_builder.maximum_reprojection_rmse_px = 3.0;
  return config;
}

[[nodiscard]] cv::Mat texturedImage() {
  cv::Mat image(static_cast<int>(kImageHeight), static_cast<int>(kImageWidth), CV_8UC1);
  cv::RNG random(0x51A7U);
  random.fill(image, cv::RNG::UNIFORM, 0, 255);
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);
  for (int row = 18; row < image.rows; row += 28) {
    for (int column = 18; column < image.cols; column += 28) {
      const int shade = ((row / 28 + column / 28) % 2 == 0) ? 20 : 235;
      cv::rectangle(image, cv::Rect(column - 4, row - 4, 9, 9), cv::Scalar(shade), cv::FILLED);
    }
  }
  return image;
}

[[nodiscard]] cv::Mat translated(const cv::Mat& image, double x) {
  cv::Mat output;
  const cv::Matx23d transform{1.0, 0.0, x, 0.0, 1.0, 0.0};
  cv::warpAffine(image, output, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
  return output;
}

[[nodiscard]] core::CameraFrame frame(const cv::Mat& image, std::uint64_t id,
                                      std::int64_t time_ns) {
  auto pixels =
      std::make_shared<std::vector<std::byte>>(image.step * static_cast<std::size_t>(image.rows));
  for (int row = 0; row < image.rows; ++row) {
    std::memcpy(pixels->data() + static_cast<std::size_t>(row) * image.step, image.ptr(row),
                image.step);
  }
  core::CameraFrame result;
  result.header.trace = core::TraceId(id);
  result.header.producer = core::ProducerId(1U);
  result.header.session = core::SessionId(1U);
  result.header.config = core::ConfigRevision(1U);
  result.header.direct_calibration = core::CalibrationEpoch(7U);
  result.id = core::MeasurementId(id);
  result.camera = core::CameraId(0U);
  result.stamp.raw_time = core::RawDeviceTime{time_ns};
  result.stamp.fusion_time = core::FusionTime{time_ns};
  result.stamp.host_arrival_time = core::ArrivalTime{time_ns};
  result.stamp.clock_revision = core::ClockRevision(1U);
  result.stamp.source_epoch = core::SourceEpoch(1U);
  result.stamp.ingress_sequence = core::IngressSequence(id);
  result.exposure_midpoint = core::FusionTime{time_ns};
  result.width = kImageWidth;
  result.height = kImageHeight;
  result.stride = static_cast<std::uint32_t>(image.step);
  result.encoding = core::ImageEncoding::Mono8;
  result.pixels = std::move(pixels);
  return result;
}

[[nodiscard]] VisualLocalStateSnapshot state(std::uint64_t state_id, std::int64_t time_ns,
                                             std::uint64_t revision, double translation_x = 0.0) {
  VisualLocalStateSnapshot result;
  result.odom_epoch = core::OdomEpoch(3U);
  result.state = core::StateId(state_id);
  result.exact_time = core::FusionTime{time_ns};
  result.observed_at_revision = core::LocalGraphRevision(revision);
  result.estimate.T_odom_imu =
      core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{translation_x, 0.0, 0.0});
  return result;
}

[[nodiscard]] VisualStateResolution resolution(std::uint64_t request, std::uint64_t state_id,
                                               std::int64_t time_ns, std::uint64_t revision) {
  VisualStateResolution result;
  result.header.direct_calibration = core::CalibrationEpoch(7U);
  result.request = core::KnotRequestId(request);
  result.timeline = StateResolution{
      core::FusionTime{time_ns}, {core::KnotRequestId(request)}, core::StateId(state_id)};
  result.odom_epoch = core::OdomEpoch(3U);
  result.state = core::StateId(state_id);
  result.created_at_revision = core::LocalGraphRevision(revision);
  return result;
}

[[nodiscard]] VisualFrameInput input(const cv::Mat& image, std::uint64_t frame_id,
                                     std::int64_t time_ns, std::uint64_t request_id,
                                     VisualLocalStateSnapshot latest) {
  VisualFrameInput result;
  result.frame = frame(image, frame_id, time_ns);
  result.latest_local_state = std::move(latest);
  result.state_request = core::KnotRequestId(request_id);
  result.request_keyframe = true;
  return result;
}

[[nodiscard]] VisualLane createLane(StateTimeline& timeline,
                                    VisualLaneConfig config = laneConfig()) {
  auto created = VisualLane::create(calibration(), core::CameraId(0U), timeline, std::move(config));
  EXPECT_TRUE(created) << created.error().detail;
  return std::move(created).value();
}

TEST(VisualLane, TracksWithoutStateButDoesNotInventAnotherStateOrPose) {
  StateTimeline timeline;
  VisualLane lane = createLane(timeline);
  VisualFrameInput first;
  first.frame = frame(texturedImage(), 10U, 100'000'000LL);
  first.state_request = core::KnotRequestId(100U);
  first.request_keyframe = true;

  const auto report = lane.processFrame(std::move(first));
  ASSERT_TRUE(report) << report.error().detail;
  EXPECT_TRUE(report.value().frontend.keyframe);
  EXPECT_EQ(report.value().disposition, VisualFrameDisposition::TrackingOnlyNoLocalState);
  EXPECT_TRUE(timeline.resolutions().empty());
  EXPECT_EQ(lane.queueState().pending_keyframes, 0U);
  EXPECT_EQ(lane.statistics().keyframes_without_local_state, 1U);
}

TEST(VisualLane, GraphSubmissionCanBeDisabledWithoutDisablingTracking) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLaneConfig config = laneConfig();
  config.graph_submission_enabled = false;
  VisualLane lane = createLane(timeline, config);
  const cv::Mat image = texturedImage();

  const auto first = lane.processFrame(input(image, 15U, 100'000'000LL, 150U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(first) << first.error().detail;
  EXPECT_TRUE(first.value().frontend.keyframe);
  EXPECT_GT(first.value().frontend.features.size(), 0U);
  EXPECT_GT(first.value().frontend.descriptors.size(), 0U);
  EXPECT_EQ(first.value().frontend.report.descriptor_tracks,
            first.value().frontend.descriptors.size());
  EXPECT_EQ(first.value().disposition, VisualFrameDisposition::TrackingOnlyGraphSubmissionDisabled);
  EXPECT_FALSE(first.value().state_admission);
  EXPECT_TRUE(timeline.resolutions().empty());
  EXPECT_EQ(lane.queueState().pending_keyframes, 0U);
  EXPECT_EQ(lane.queueState().pending_factor_batches, 0U);

  VisualFrameInput second =
      input(translated(image, 2.0), 16U, 300'000'000LL, 151U, state(1U, 0LL, 1U));
  VisualRotationPrior rotation;
  rotation.imu_calibration = core::CalibrationEpoch(7U);
  rotation.previous_exposure_midpoint = core::FusionTime{100'000'000LL};
  rotation.current_exposure_midpoint = core::FusionTime{300'000'000LL};
  rotation.imu_support = {core::MeasurementId(140U), core::MeasurementId(141U)};
  second.imu_rotation_seed = rotation;
  const auto continued = lane.processFrame(std::move(second));
  ASSERT_TRUE(continued) << continued.error().detail;
  EXPECT_GT(continued.value().frontend.report.input_tracks, 0U);
  EXPECT_EQ(continued.value().frontend.report.rotation_prior, RotationPriorStatus::Applied);
  EXPECT_EQ(continued.value().disposition,
            VisualFrameDisposition::TrackingOnlyGraphSubmissionDisabled);
  EXPECT_TRUE(timeline.resolutions().empty());
  EXPECT_EQ(lane.queueState().pending_keyframes, 0U);
  EXPECT_EQ(lane.queueState().pending_factor_batches, 0U);
  EXPECT_EQ(lane.statistics().frames_accepted, 2U);
  EXPECT_EQ(lane.statistics().frames_tracking_only_graph_submission_disabled, 2U);
  EXPECT_GE(lane.statistics().frontend_keyframes, 1U);
  EXPECT_EQ(lane.statistics().state_requests_admitted, 0U);
  EXPECT_EQ(lane.statistics().factor_batches_queued, 0U);
}

TEST(VisualLane, TimelineSuppressionIsAnObservableTrackingDisposition) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat image = texturedImage();
  const auto first = lane.processFrame(input(image, 20U, 100'000'000LL, 200U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_EQ(first.value().disposition, VisualFrameDisposition::KeyframeStateRequested);

  const auto second = lane.processFrame(
      input(translated(image, 2.0), 21U, 150'000'000LL, 201U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(second) << second.error().detail;
  EXPECT_EQ(second.value().disposition, VisualFrameDisposition::KeyframeSuppressedByTimeline);
  ASSERT_TRUE(second.value().state_admission);
  EXPECT_EQ(second.value().state_admission->disposition,
            StateAdmissionDisposition::SuppressedTooClose);
  EXPECT_FALSE(second.value().timeline_rejection);
  EXPECT_EQ(lane.queueState().pending_keyframes, 1U);
  EXPECT_EQ(lane.statistics().keyframes_suppressed_by_timeline, 1U);
}

TEST(VisualLane, TimelineRejectionIsObservableWithoutMutatingTheVisualQueue) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat image = texturedImage();
  const auto first = lane.processFrame(input(image, 22U, 100'000'000LL, 220U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_EQ(first.value().disposition, VisualFrameDisposition::KeyframeStateRequested);

  const auto rejected = lane.processFrame(
      input(translated(image, 2.0), 23U, 300'000'000LL, 219U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(rejected) << rejected.error().detail;
  EXPECT_EQ(rejected.value().disposition, VisualFrameDisposition::KeyframeRejectedByTimeline);
  ASSERT_TRUE(rejected.value().timeline_rejection);
  EXPECT_EQ(rejected.value().timeline_rejection->code, StateTimelineErrorCode::RequestIdOutOfOrder);
  EXPECT_FALSE(rejected.value().state_admission);
  EXPECT_EQ(lane.queueState().pending_keyframes, 1U);
  EXPECT_EQ(lane.statistics().keyframes_rejected_by_timeline, 1U);
}

TEST(VisualLane, IndependentLidarRequestLeavesPendingVisualWorkIntact) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat image = texturedImage();
  const auto first = lane.processFrame(input(image, 25U, 200'000'000LL, 250U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_EQ(first.value().disposition, VisualFrameDisposition::KeyframeStateRequested);
  ASSERT_GT(first.value().frontend.features.size(), 0U);

  StateRequest lidar_request;
  lidar_request.header = frame(image, 251U, 190'000'000LL).header;
  lidar_request.id = core::KnotRequestId{251U};
  lidar_request.sensor = core::SensorInstanceId::lidar(core::LidarId{0U});
  lidar_request.purpose = StateRequestPurpose::LidarReference;
  lidar_request.exact_time = core::FusionTime{190'000'000LL};
  lidar_request.lineage = first.value().frontend.lineage;
  const auto lidar = timeline.request(std::move(lidar_request));
  ASSERT_TRUE(lidar);
  EXPECT_EQ(lidar.value().disposition, StateAdmissionDisposition::SuppressedTooClose);
  EXPECT_EQ(lane.queueState().pending_keyframes, 1U);

  const auto continued = lane.processFrame(
      input(translated(image, 2.0), 26U, 300'000'000LL, 252U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(continued) << continued.error().detail;
  EXPECT_GT(continued.value().frontend.report.input_tracks, 0U);
  EXPECT_EQ(continued.value().disposition, VisualFrameDisposition::KeyframeStateRequested);
  EXPECT_EQ(lane.queueState().pending_keyframes, 2U);
}

TEST(VisualLane, InvalidResolutionCannotSubstituteAStalePose) {
  StateTimeline timeline;
  VisualLane lane = createLane(timeline);
  const auto admitted =
      lane.processFrame(input(texturedImage(), 30U, 100'000'000LL, 300U, state(1U, 0LL, 1U)));
  ASSERT_TRUE(admitted) << admitted.error().detail;

  const auto rejected = lane.resolveCommittedKeyframe(resolution(300U, 2U, 100'000'000LL, 2U),
                                                      state(2U, 99'000'000LL, 2U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualLaneErrorCode::InvalidStateResolution);
  EXPECT_EQ(lane.queueState().pending_keyframes, 1U);
  EXPECT_EQ(lane.statistics().keyframes_resolved, 0U);
}

TEST(VisualLane, ResolvedStatesDriveTriangulationAndRetryableGraphInput) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat base = texturedImage();

  const auto first = lane.processFrame(input(base, 40U, 100'000'000LL, 400U, state(9U, 0LL, 9U)));
  ASSERT_TRUE(first) << first.error().detail;
  const auto first_resolved = lane.resolveCommittedKeyframe(
      resolution(400U, 10U, 100'000'000LL, 10U), state(10U, 100'000'000LL, 10U, 0.0));
  ASSERT_TRUE(first_resolved) << first_resolved.error().detail;
  EXPECT_FALSE(first_resolved.value().graph_batch_queued);

  VisualFrameInput second_input =
      input(translated(base, 8.0), 41U, 300'000'000LL, 401U, state(10U, 100'000'000LL, 10U, 0.0));
  VisualRotationPrior rotation;
  rotation.previous_exposure_midpoint = core::FusionTime{100'000'000LL};
  rotation.current_exposure_midpoint = core::FusionTime{300'000'000LL};
  rotation.imu_calibration = core::CalibrationEpoch(7U);
  rotation.imu_support = {core::MeasurementId(900U), core::MeasurementId(901U)};
  second_input.imu_rotation_seed = rotation;
  const auto second = lane.processFrame(std::move(second_input));
  ASSERT_TRUE(second) << second.error().detail;
  EXPECT_EQ(second.value().frontend.report.rotation_prior, RotationPriorStatus::Applied);
  const auto second_resolved = lane.resolveCommittedKeyframe(
      resolution(401U, 11U, 300'000'000LL, 11U), state(11U, 300'000'000LL, 11U, -0.2));
  ASSERT_TRUE(second_resolved) << second_resolved.error().detail;

  const auto third = lane.processFrame(input(translated(base, 16.0), 42U, 500'000'000LL, 402U,
                                             state(11U, 300'000'000LL, 11U, -0.2)));
  ASSERT_TRUE(third) << third.error().detail;
  const auto third_resolved = lane.resolveCommittedKeyframe(
      resolution(402U, 12U, 500'000'000LL, 12U), state(12U, 500'000'000LL, 12U, -0.4));
  ASSERT_TRUE(third_resolved) << third_resolved.error().detail;
  ASSERT_TRUE(third_resolved.value().graph_batch_queued);
  EXPECT_GT(third_resolved.value().factor_builder.tracks_initialized, 0U);
  EXPECT_GT(third_resolved.value().factor_builder.factors_emitted, 0U);

  ImuKnotAppend candidate;
  candidate.state = core::StateId(13U);
  candidate.exact_time = core::FusionTime{700'000'000LL};
  const auto prepared = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(prepared) << prepared.error().detail;
  ASSERT_TRUE(prepared.value().visual);
  EXPECT_EQ(prepared.value().candidate_state, core::StateId(13U));
  EXPECT_EQ(prepared.value().visual->exact_time, core::FusionTime{700'000'000LL});
  ASSERT_FALSE(prepared.value().visual->factors.empty());
  for (const VisualReprojectionFactorSpec& factor : prepared.value().visual->factors) {
    EXPECT_NE(factor.anchor.state, factor.observer.state);
    EXPECT_LE(factor.observer.exact_time, core::FusionTime{500'000'000LL});
    EXPECT_NE(factor.observer.state, core::StateId(13U));
  }
  const core::FactorId observed_factor = prepared.value().visual->factors.front().id;
  const VisualAttachmentId first_attachment = prepared.value().id;

  const auto rejected = lane.acknowledgeGraphInputRejected(first_attachment);
  ASSERT_TRUE(rejected) << rejected.error().detail;
  EXPECT_EQ(rejected.value().pending_factor_batches, 1U);

  candidate.state = core::StateId(14U);
  candidate.exact_time = core::FusionTime{800'000'000LL};
  const auto retry = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(retry) << retry.error().detail;
  ASSERT_TRUE(retry.value().visual);
  EXPECT_EQ(retry.value().visual->factors.front().id, observed_factor);
  const auto accepted =
      lane.acknowledgeGraphInputAccepted(retry.value().id, core::LocalGraphRevision(14U));
  ASSERT_TRUE(accepted) << accepted.error().detail;
  EXPECT_EQ(accepted.value().pending_factor_batches, 0U);

  const std::vector<VisualResidualFeedback> outlier{VisualResidualFeedback{observed_factor, 100.0}};
  const auto feedback_one = lane.applyAcceptedResiduals(core::LocalGraphRevision(15U), outlier);
  ASSERT_TRUE(feedback_one) << feedback_one.error().detail;
  EXPECT_EQ(feedback_one.value().retirements_queued, 0U);
  const auto feedback_two = lane.applyAcceptedResiduals(core::LocalGraphRevision(16U), outlier);
  ASSERT_TRUE(feedback_two) << feedback_two.error().detail;
  EXPECT_EQ(feedback_two.value().retirements_queued, 1U);

  candidate.state = core::StateId(15U);
  candidate.exact_time = core::FusionTime{900'000'000LL};
  const auto retirement = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(retirement) << retirement.error().detail;
  EXPECT_FALSE(retirement.value().visual);
  ASSERT_EQ(retirement.value().visual_factor_retirements.size(), 1U);
  EXPECT_EQ(retirement.value().visual_factor_retirements.front(), observed_factor);
  const auto retirement_accepted =
      lane.acknowledgeGraphInputAccepted(retirement.value().id, core::LocalGraphRevision(17U));
  ASSERT_TRUE(retirement_accepted) << retirement_accepted.error().detail;
  EXPECT_EQ(retirement_accepted.value().pending_factor_retirements, 0U);
}

TEST(VisualLane, GraphFinalityPrunesQueuedDependenciesWithoutDegradation) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat base = texturedImage();

  ASSERT_TRUE(lane.processFrame(input(base, 45U, 100'000'000LL, 450U, state(9U, 0LL, 9U))));
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(450U, 10U, 100'000'000LL, 10U),
                                            state(10U, 100'000'000LL, 10U, 0.0)));
  ASSERT_TRUE(lane.processFrame(
      input(translated(base, 8.0), 46U, 300'000'000LL, 451U, state(10U, 100'000'000LL, 10U, 0.0))));
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(451U, 11U, 300'000'000LL, 11U),
                                            state(11U, 300'000'000LL, 11U, -0.2)));
  ASSERT_TRUE(lane.processFrame(input(translated(base, 16.0), 47U, 500'000'000LL, 452U,
                                      state(11U, 300'000'000LL, 11U, -0.2))));
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(452U, 12U, 500'000'000LL, 12U),
                                            state(12U, 500'000'000LL, 12U, -0.4)));

  ImuKnotAppend candidate;
  candidate.state = core::StateId(13U);
  candidate.exact_time = core::FusionTime{700'000'000LL};
  const auto initial = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(initial) << initial.error().detail;
  ASSERT_TRUE(initial.value().visual);
  ASSERT_FALSE(initial.value().visual->new_landmarks.empty());
  ASSERT_FALSE(initial.value().visual->factors.empty());
  std::vector<VisualLandmarkId> finalized_landmarks;
  std::vector<core::FactorId> finalized_factors;
  for (const VisualLandmarkSeed& seed : initial.value().visual->new_landmarks) {
    finalized_landmarks.push_back(seed.landmark);
  }
  for (const VisualReprojectionFactorSpec& factor : initial.value().visual->factors) {
    finalized_factors.push_back(factor.id);
  }
  std::sort(finalized_landmarks.begin(), finalized_landmarks.end());
  std::sort(finalized_factors.begin(), finalized_factors.end());
  ASSERT_TRUE(
      lane.acknowledgeGraphInputAccepted(initial.value().id, core::LocalGraphRevision(14U)));

  const core::FactorId retiring = finalized_factors.front();
  const std::vector<VisualResidualFeedback> outlier{{retiring, 100.0}};
  ASSERT_TRUE(lane.applyAcceptedResiduals(core::LocalGraphRevision(15U), outlier));
  ASSERT_TRUE(lane.applyAcceptedResiduals(core::LocalGraphRevision(16U), outlier));
  ASSERT_EQ(lane.queueState().pending_factor_retirements, 1U);

  ASSERT_TRUE(lane.processFrame(input(translated(base, 24.0), 48U, 700'000'000LL, 453U,
                                      state(12U, 500'000'000LL, 14U, -0.4))));
  const auto fourth = lane.resolveCommittedKeyframe(resolution(453U, 13U, 700'000'000LL, 15U),
                                                    state(13U, 700'000'000LL, 15U, -0.6));
  ASSERT_TRUE(fourth) << fourth.error().detail;
  ASSERT_TRUE(fourth.value().graph_batch_queued);
  ASSERT_GT(lane.queueState().pending_factor_specs, 0U);

  VisualFinalityUpdate update;
  update.revision = core::LocalGraphRevision(17U);
  update.finalized_factors = finalized_factors;
  update.finalized_landmarks = finalized_landmarks;
  const auto reconciled = lane.reconcileGraphFinality(update);
  ASSERT_TRUE(reconciled) << reconciled.error().detail;
  EXPECT_GT(reconciled.value().factor_builder.accepted_tracks_pruned, 0U);
  EXPECT_GT(reconciled.value().pending_factor_specs_pruned, 0U);
  EXPECT_EQ(reconciled.value().pending_factor_retirements_pruned, 1U);
  EXPECT_EQ(reconciled.value().queues.pending_factor_retirements, 0U);
  EXPECT_EQ(lane.statistics().graph_inputs_degraded, 0U);
  EXPECT_EQ(lane.statistics().graph_finality_updates, 1U);

  candidate.state = core::StateId(14U);
  candidate.exact_time = core::FusionTime{900'000'000LL};
  const auto remaining = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(remaining) << remaining.error().detail;
  ASSERT_TRUE(remaining.value().visual);
  for (const VisualReprojectionFactorSpec& factor : remaining.value().visual->factors) {
    EXPECT_FALSE(std::binary_search(finalized_factors.begin(), finalized_factors.end(), factor.id));
    EXPECT_FALSE(std::binary_search(finalized_landmarks.begin(), finalized_landmarks.end(),
                                    factor.landmark));
  }
  EXPECT_TRUE(remaining.value().visual_factor_retirements.empty());
  ASSERT_TRUE(lane.acknowledgeGraphInputRejected(remaining.value().id));

  VisualFinalityUpdate empty;
  empty.revision = core::LocalGraphRevision(18U);
  ASSERT_TRUE(lane.reconcileGraphFinality(empty));
  const auto repeated = lane.reconcileGraphFinality(empty);
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, VisualLaneErrorCode::InvalidGraphFinality);
}

TEST(VisualLane, DegradedInputDiscardsDependentBatchesAndRecoversWithFreshIdentities) {
  StateTimeline timeline({core::Duration{100'000'000LL}, 64U, 256U});
  VisualLane lane = createLane(timeline);
  const cv::Mat base = texturedImage();

  const auto first = lane.processFrame(input(base, 50U, 100'000'000LL, 500U, state(9U, 0LL, 9U)));
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(500U, 10U, 100'000'000LL, 10U),
                                            state(10U, 100'000'000LL, 10U, 0.0)));
  const auto second = lane.processFrame(
      input(translated(base, 8.0), 51U, 300'000'000LL, 501U, state(10U, 100'000'000LL, 10U, 0.0)));
  ASSERT_TRUE(second) << second.error().detail;
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(501U, 11U, 300'000'000LL, 11U),
                                            state(11U, 300'000'000LL, 11U, -0.2)));
  const auto third = lane.processFrame(input(translated(base, 16.0), 52U, 500'000'000LL, 502U,
                                             state(11U, 300'000'000LL, 11U, -0.2)));
  ASSERT_TRUE(third) << third.error().detail;
  const auto third_resolved = lane.resolveCommittedKeyframe(
      resolution(502U, 12U, 500'000'000LL, 12U), state(12U, 500'000'000LL, 12U, -0.4));
  ASSERT_TRUE(third_resolved) << third_resolved.error().detail;
  ASSERT_TRUE(third_resolved.value().graph_batch_queued);

  ImuKnotAppend candidate;
  candidate.state = core::StateId(13U);
  candidate.exact_time = core::FusionTime{700'000'000LL};
  const auto poison = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(poison) << poison.error().detail;
  ASSERT_TRUE(poison.value().visual);
  ASSERT_FALSE(poison.value().visual->new_landmarks.empty());
  ASSERT_FALSE(poison.value().visual->factors.empty());
  std::set<VisualLandmarkId> discarded_landmarks;
  std::set<core::FactorId> discarded_factors;
  for (const VisualLandmarkSeed& seed : poison.value().visual->new_landmarks) {
    discarded_landmarks.insert(seed.landmark);
  }
  for (const VisualReprojectionFactorSpec& factor : poison.value().visual->factors) {
    discarded_factors.insert(factor.id);
  }

  const auto degraded = lane.acknowledgeGraphInputDegraded(poison.value().id);
  ASSERT_TRUE(degraded) << degraded.error().detail;
  EXPECT_EQ(degraded.value().factor_batches_discarded, 1U);
  EXPECT_EQ(degraded.value().landmark_seeds_discarded, discarded_landmarks.size());
  EXPECT_EQ(degraded.value().factors_discarded, discarded_factors.size());
  EXPECT_GT(degraded.value().active_landmark_initializations_rolled_back, 0U);
  EXPECT_GT(degraded.value().stale_track_observations_discarded, 0U);
  EXPECT_EQ(degraded.value().queues.pending_factor_batches, 0U);

  // The same frontend tracks continue, but the builder must triangulate fresh
  // local landmark/factor IDs after rolling back graph work that never
  // committed.
  const auto fourth = lane.processFrame(input(translated(base, 24.0), 53U, 900'000'000LL, 503U,
                                              state(13U, 700'000'000LL, 13U, -0.4)));
  ASSERT_TRUE(fourth) << fourth.error().detail;
  const auto fourth_resolved = lane.resolveCommittedKeyframe(
      resolution(503U, 14U, 900'000'000LL, 14U), state(14U, 900'000'000LL, 14U, -0.6));
  ASSERT_TRUE(fourth_resolved) << fourth_resolved.error().detail;

  const auto fifth = lane.processFrame(input(translated(base, 32.0), 54U, 1'100'000'000LL, 504U,
                                             state(14U, 900'000'000LL, 14U, -0.6)));
  ASSERT_TRUE(fifth) << fifth.error().detail;
  ASSERT_TRUE(lane.resolveCommittedKeyframe(resolution(504U, 15U, 1'100'000'000LL, 15U),
                                            state(15U, 1'100'000'000LL, 15U, -0.8)));
  const auto sixth = lane.processFrame(input(translated(base, 40.0), 55U, 1'300'000'000LL, 505U,
                                             state(15U, 1'100'000'000LL, 15U, -0.8)));
  ASSERT_TRUE(sixth) << sixth.error().detail;
  const auto sixth_resolved = lane.resolveCommittedKeyframe(
      resolution(505U, 16U, 1'300'000'000LL, 16U), state(16U, 1'300'000'000LL, 16U, -1.0));
  ASSERT_TRUE(sixth_resolved) << sixth_resolved.error().detail;
  ASSERT_TRUE(sixth_resolved.value().graph_batch_queued);

  candidate.state = core::StateId(17U);
  candidate.exact_time = core::FusionTime{1'500'000'000LL};
  const auto recovered = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(recovered) << recovered.error().detail;
  ASSERT_TRUE(recovered.value().visual);
  ASSERT_FALSE(recovered.value().visual->new_landmarks.empty());
  ASSERT_FALSE(recovered.value().visual->factors.empty());
  std::set<VisualLandmarkId> recovered_landmarks;
  std::set<core::FactorId> recovered_factors;
  for (const VisualLandmarkSeed& seed : recovered.value().visual->new_landmarks) {
    EXPECT_FALSE(discarded_landmarks.contains(seed.landmark));
    recovered_landmarks.insert(seed.landmark);
  }
  for (const VisualReprojectionFactorSpec& factor : recovered.value().visual->factors) {
    EXPECT_FALSE(discarded_factors.contains(factor.id));
    recovered_factors.insert(factor.id);
  }
  EXPECT_EQ(recovered_landmarks.size(), recovered.value().visual->new_landmarks.size());
  EXPECT_EQ(recovered_factors.size(), recovered.value().visual->factors.size());
  const core::FactorId accepted_factor = recovered.value().visual->factors.front().id;
  ASSERT_TRUE(
      lane.acknowledgeGraphInputAccepted(recovered.value().id, core::LocalGraphRevision(17U)));

  // Queue a retirement for an accepted factor, then degrade a later factor
  // batch. The uncommitted batch is dropped while the accepted-factor
  // retirement remains queued for the next graph transaction.
  const std::vector<VisualResidualFeedback> outlier{VisualResidualFeedback{accepted_factor, 100.0}};
  ASSERT_TRUE(lane.applyAcceptedResiduals(core::LocalGraphRevision(18U), outlier));
  const auto second_outlier = lane.applyAcceptedResiduals(core::LocalGraphRevision(19U), outlier);
  ASSERT_TRUE(second_outlier) << second_outlier.error().detail;
  ASSERT_EQ(second_outlier.value().retirements_queued, 1U);

  const auto seventh = lane.processFrame(input(translated(base, 48.0), 56U, 1'700'000'000LL, 506U,
                                               state(17U, 1'500'000'000LL, 17U, -1.0)));
  ASSERT_TRUE(seventh) << seventh.error().detail;
  const auto seventh_resolved = lane.resolveCommittedKeyframe(
      resolution(506U, 18U, 1'700'000'000LL, 20U), state(18U, 1'700'000'000LL, 20U, -1.2));
  ASSERT_TRUE(seventh_resolved) << seventh_resolved.error().detail;
  ASSERT_TRUE(seventh_resolved.value().graph_batch_queued);

  candidate.state = core::StateId(19U);
  candidate.exact_time = core::FusionTime{1'900'000'000LL};
  const auto later_poison = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(later_poison) << later_poison.error().detail;
  ASSERT_TRUE(later_poison.value().visual);
  ASSERT_EQ(later_poison.value().visual_factor_retirements.size(), 1U);
  EXPECT_EQ(later_poison.value().visual_factor_retirements.front(), accepted_factor);
  const auto later_degraded = lane.acknowledgeGraphInputDegraded(later_poison.value().id);
  ASSERT_TRUE(later_degraded) << later_degraded.error().detail;
  EXPECT_EQ(later_degraded.value().factor_retirements_preserved, 1U);
  EXPECT_EQ(later_degraded.value().queues.pending_factor_retirements, 1U);

  candidate.state = core::StateId(20U);
  candidate.exact_time = core::FusionTime{2'000'000'000LL};
  const auto retirement = lane.prepareGraphInput(candidate);
  ASSERT_TRUE(retirement) << retirement.error().detail;
  EXPECT_FALSE(retirement.value().visual);
  ASSERT_EQ(retirement.value().visual_factor_retirements.size(), 1U);
  EXPECT_EQ(retirement.value().visual_factor_retirements.front(), accepted_factor);
  ASSERT_TRUE(
      lane.acknowledgeGraphInputAccepted(retirement.value().id, core::LocalGraphRevision(21U)));
  EXPECT_EQ(lane.queueState().pending_factor_retirements, 0U);
}

}  // namespace
}  // namespace meridian::local
