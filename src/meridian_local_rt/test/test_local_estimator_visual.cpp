#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "meridian/local/local_estimator.hpp"

namespace meridian::local {
namespace {

constexpr std::int64_t kMs = 1'000'000LL;
constexpr std::int64_t kImuPeriod = 5 * kMs;
constexpr std::uint32_t kImageWidth = 320U;
constexpr std::uint32_t kImageHeight = 240U;

[[nodiscard]] core::RecordHeader header(std::uint64_t trace = 1U) {
  core::RecordHeader result;
  result.trace = core::TraceId{trace};
  result.producer = core::ProducerId{1U};
  result.session = core::SessionId{1U};
  result.config = core::ConfigRevision{1U};
  result.direct_calibration = core::CalibrationEpoch{1U};
  return result;
}

[[nodiscard]] core::SourceStamp stamp(std::int64_t time_ns, std::uint64_t sequence) {
  core::SourceStamp result;
  result.raw_time = core::RawDeviceTime{time_ns};
  result.fusion_time = core::FusionTime{time_ns};
  result.host_arrival_time = core::ArrivalTime{time_ns};
  result.clock_revision = core::ClockRevision{1U};
  result.source_epoch = core::SourceEpoch{1U};
  result.ingress_sequence = core::IngressSequence{sequence};
  return result;
}

[[nodiscard]] core::CameraCalibration cameraCalibration(core::CameraId id) {
  return core::CameraCalibration(
      id, "camera_" + std::to_string(id.value()), "/camera_" + std::to_string(id.value()),
      core::PinholeEquidistantCameraModel(200.0, 200.0, 159.5, 119.5,
                                          std::array<double, 4>{0.0, 0.0, 0.0, 0.0},
                                          core::ImageDimensions{kImageWidth, kImageHeight}),
      core::ImuFromCameraTransform{core::Pose3d{}},
      core::CameraTimingCalibration(core::CameraTimestampReference::ExposureMidpoint,
                                    core::Duration{0LL}));
}

[[nodiscard]] core::CalibrationBundle calibration(std::size_t cameras) {
  core::ImuCalibration imu("imu", "/imu", core::ImuSensorModel::Generic, 200.0, 9.80665,
                           core::ImuNoiseModel(1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6));
  core::LidarCalibration lidar(
      core::LidarId{1U}, "lidar", "/points", core::ImuFromLidarTransform{core::Pose3d{}},
      core::LidarTimingCalibration{core::LidarSweepTimestampReference::SweepStart,
                                   core::LidarPointTimeConvention::OffsetFromSweepTimestamp});
  std::vector<core::CameraCalibration> camera_models;
  for (std::size_t index = 0U; index < cameras; ++index) {
    camera_models.push_back(cameraCalibration(core::CameraId{index}));
  }
  auto result = core::CalibrationBundle::create(core::CalibrationEpoch{1U}, std::move(imu),
                                                core::BaseFromImuTransform{core::Pose3d{}},
                                                std::move(lidar), std::move(camera_models));
  EXPECT_TRUE(result);
  return std::move(result).value();
}

[[nodiscard]] VisualLaneConfig visualConfig() {
  VisualLaneConfig config;
  config.frontend.features_per_cell = 3U;
  config.frontend.recovery_minimum_tracks = 20U;
  config.frontend.keyframe_minimum_tracks = 40U;
  config.frontend.minimum_keyframe_interval = core::Duration{100 * kMs};
  config.frontend.maximum_keyframe_interval = core::Duration{500 * kMs};
  config.frontend.base_pixel_sigma = 5.0;
  config.factor_builder.maximum_active_tracks = 144U;
  config.factor_builder.maximum_pending_observations = 3U;
  config.factor_builder.maximum_observations_per_track = 12U;
  config.factor_builder.minimum_baseline_m = 0.01;
  config.factor_builder.minimum_parallax_rad = 0.004;
  config.factor_builder.maximum_inlier_reprojection_error_px = 6.0;
  config.factor_builder.maximum_reprojection_rmse_px = 4.0;
  return config;
}

[[nodiscard]] LocalEstimatorConfig estimatorConfig(std::vector<core::CameraId> cameras) {
  LocalEstimatorConfig config;
  config.odom_epoch = core::OdomEpoch{3U};
  config.first_state = core::StateId{10U};
  config.initialization.mode = InitializationMode::StaticOnly;
  config.initialization.zero_motion_prior =
      ZeroMotionPrior{config.odom_epoch, ZeroMotionPriorSource::MissionScenario};
  config.stationary_initializer.minimum_support = core::Duration{1'000 * kMs};
  config.stationary_retry_period = core::Duration{50 * kMs};
  config.state_timeline.minimum_state_interval = core::Duration{40 * kMs};
  for (const core::CameraId camera : cameras) {
    config.visual_cameras.push_back(VisualCameraConfig{camera, visualConfig()});
  }
  config.lidar_preprocessing.minimum_range_m = 0.5;
  config.lidar_preprocessing.maximum_range_m = 30.0;
  config.lidar_preprocessing.voxel_size_m = 0.1;
  config.lidar_preprocessing.maximum_output_points = 2'000U;
  config.rolling_target.maximum_retained_sweeps = 4U;
  config.rolling_target.maximum_retained_points = 10'000U;
  config.lidar_registration.target_voxel_resolution_m = 0.5;
  config.lidar_registration.source_voxel_size_m = 0.5;
  config.lidar_registration.maximum_target_points_per_target = 10'000U;
  config.lidar_registration.minimum_correspondences = 20U;
  config.lidar_registration.residual_standard_deviation_m = 0.05;
  config.lidar_registration.absolute_normalized_observable_eigenvalue = 1.0e-8;
  config.finalized_lidar_target.query_voxel_size_m =
      config.lidar_registration.target_voxel_resolution_m;
  config.finalized_lidar_target.maximum_supported_query_distance_m =
      config.lidar_registration.maximum_correspondence_distance_m;
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

[[nodiscard]] core::CameraFrame frame(const cv::Mat& image, core::CameraId camera, std::uint64_t id,
                                      std::int64_t time_ns) {
  auto pixels =
      std::make_shared<std::vector<std::byte>>(image.step * static_cast<std::size_t>(image.rows));
  for (int row = 0; row < image.rows; ++row) {
    std::memcpy(pixels->data() + static_cast<std::size_t>(row) * image.step, image.ptr(row),
                image.step);
  }
  core::CameraFrame result;
  result.header = header(id);
  result.id = core::MeasurementId{id};
  result.camera = camera;
  result.stamp = stamp(time_ns, id);
  result.exposure_midpoint = core::FusionTime{time_ns};
  result.width = kImageWidth;
  result.height = kImageHeight;
  result.stride = static_cast<std::uint32_t>(image.step);
  result.encoding = core::ImageEncoding::Mono8;
  result.pixels = std::move(pixels);
  return result;
}

[[nodiscard]] core::ImuSample imu(std::int64_t time_ns, std::uint64_t id,
                                  double acceleration_x = 0.0) {
  core::ImuSample sample;
  sample.header = header(id);
  sample.id = core::MeasurementId{id};
  sample.stamp = stamp(time_ns, id);
  sample.specific_force_mps2 = Eigen::Vector3d{acceleration_x, 0.0, 9.80665};
  return sample;
}

void appendImu(LocalEstimator& estimator, std::int64_t& last_time, std::int64_t through,
               std::uint64_t& next_id, bool short_acceleration = false) {
  for (std::int64_t time = last_time + kImuPeriod; time <= through; time += kImuPeriod) {
    const double acceleration = short_acceleration && time <= 1'105 * kMs ? -4.0 : 0.0;
    auto appended = estimator.ingestImu(imu(time, next_id++, acceleration));
    ASSERT_TRUE(appended) << appended.error().detail;
    last_time = time;
  }
}

void initialize(LocalEstimator& estimator, std::int64_t& last_time, std::uint64_t& next_id) {
  auto first = estimator.ingestImu(imu(0, next_id++));
  ASSERT_TRUE(first) << first.error().detail;
  last_time = 0;
  appendImu(estimator, last_time, 1'000 * kMs, next_id);
  auto initialized = estimator.processReady();
  ASSERT_TRUE(initialized) << initialized.error().detail;
  ASSERT_TRUE(initialized.value().initialization);
  ASSERT_EQ(initialized.value().initialization->state_time, core::FusionTime{1'000 * kMs});
}

[[nodiscard]] core::LidarSweep sweep(std::uint64_t id, std::int64_t start_ns) {
  constexpr std::uint32_t kWidth = 64U;
  constexpr std::uint32_t kRowsPerPlane = 8U;
  constexpr std::uint32_t kHeight = 3U * kRowsPerPlane;
  constexpr std::int64_t kDuration = 90 * kMs;
  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(static_cast<std::size_t>(kWidth) * kHeight);
  for (std::uint32_t row = 0U; row < kHeight; ++row) {
    const std::uint32_t plane = row / kRowsPerPlane;
    const std::uint32_t plane_row = row % kRowsPerPlane;
    const double narrow_axis = -1.38 + 0.09 * static_cast<double>(plane_row);
    for (std::uint32_t column = 0U; column < kWidth; ++column) {
      const double long_axis = 4.13 + 0.09 * static_cast<double>(column);
      Eigen::Vector3d position;
      if (plane == 0U) {
        position = Eigen::Vector3d{long_axis, narrow_axis, -1.73};
      } else if (plane == 1U) {
        position = Eigen::Vector3d{long_axis, 3.37, narrow_axis + 1.02};
      } else {
        position = Eigen::Vector3d{9.41, long_axis - 6.92, narrow_axis + 0.51};
      }
      core::LidarPoint point;
      point.x = static_cast<float>(position.x());
      point.y = static_cast<float>(position.y());
      point.z = static_cast<float>(position.z());
      point.intensity = static_cast<float>(column + row);
      point.ring = static_cast<std::uint16_t>(row);
      point.source_index = row * kWidth + column;
      point.time_offset_ns =
          static_cast<std::int32_t>((static_cast<std::int64_t>(column) * (kDuration - 1)) /
                                    static_cast<std::int64_t>(kWidth - 1U));
      points->push_back(point);
    }
  }
  core::LidarSweep result;
  result.header = header(id);
  result.id = core::MeasurementId{id};
  result.lidar = core::LidarId{1U};
  result.stamp = stamp(start_ns, id);
  result.acquisition =
      core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{start_ns + kDuration}};
  result.layout = core::LidarLayout{kWidth, kHeight, true};
  result.points = std::move(points);
  return result;
}

TEST(LocalEstimatorVisual, CameraAndLidarShareOneExactGraphState) {
  auto created = LocalEstimator::create(calibration(1U), estimatorConfig({core::CameraId{0U}}));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id);

  const cv::Mat image = texturedImage();
  auto keyframe =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 10'000U, 1'190 * kMs), true);
  ASSERT_TRUE(keyframe) << keyframe.error().detail;
  ASSERT_EQ(keyframe.value().frame.disposition, VisualFrameDisposition::KeyframeStateRequested);
  auto watermark =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 10'001U, 1'200 * kMs), false);
  ASSERT_TRUE(watermark) << watermark.error().detail;
  EXPECT_TRUE(watermark.value().imu_rotation_seed_provided);
  EXPECT_EQ(watermark.value().frame.frontend.report.rotation_prior, RotationPriorStatus::Applied);
  ASSERT_TRUE(estimator.enqueueLidar(sweep(20'000U, 1'100 * kMs)));

  auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().commits.size(), 1U);
  ASSERT_EQ(processed.value().camera_commits.size(), 1U);
  const LidarCommitReport& lidar = processed.value().commits.front();
  const CameraKnotCommitReport& camera = processed.value().camera_commits.front();
  EXPECT_EQ(lidar.commit.state, camera.commit.state);
  EXPECT_EQ(lidar.commit.state_time, core::FusionTime{1'190 * kMs});
  EXPECT_EQ(camera.exact_time, core::FusionTime{1'190 * kMs});
  ASSERT_EQ(camera.resolved_keyframes.size(), 1U);
  ASSERT_TRUE(camera.resolved_keyframes.front().resolution.timeline.committed_state);
  EXPECT_EQ(*camera.resolved_keyframes.front().resolution.timeline.committed_state,
            camera.commit.state);
  EXPECT_EQ(estimator.statistics().graph_commits, 2U);
  EXPECT_EQ(estimator.statistics().visual_keyframe_knots_committed, 1U);
}

TEST(LocalEstimatorVisual, VisualStateInsideSweepPreservesCompleteLidarDeskew) {
  auto created = LocalEstimator::create(calibration(1U), estimatorConfig({core::CameraId{0U}}));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id);

  const cv::Mat image = texturedImage();
  auto keyframe =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 10'100U, 1'145 * kMs), true);
  ASSERT_TRUE(keyframe) << keyframe.error().detail;
  ASSERT_EQ(keyframe.value().frame.disposition, VisualFrameDisposition::KeyframeStateRequested);
  const auto lidar = estimator.enqueueLidar(sweep(20'100U, 1'100 * kMs));
  ASSERT_TRUE(lidar) << lidar.error().detail;
  EXPECT_EQ(lidar.value().state_admission, StateAdmissionDisposition::NewState);

  auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().camera_commits.size(), 1U);
  ASSERT_EQ(processed.value().commits.size(), 1U);
  EXPECT_EQ(processed.value().camera_commits.front().exact_time, core::FusionTime{1'145 * kMs});
  EXPECT_EQ(processed.value().commits.front().commit.state_time, core::FusionTime{1'190 * kMs});
  EXPECT_NE(processed.value().camera_commits.front().commit.state,
            processed.value().commits.front().commit.state);
  EXPECT_GT(processed.value().commits.front().deskew_pose_interpolations, 0U);
  EXPECT_TRUE(processed.value().dropped_sweeps.empty());
  EXPECT_EQ(estimator.statistics().graph_commits, 3U);
}

TEST(LocalEstimatorVisual, EarlierVisualRequestSuppressesOnlyCloseLidarRequest) {
  LocalEstimatorConfig config = estimatorConfig({core::CameraId{0U}});
  config.state_timeline.minimum_state_interval = core::Duration{100 * kMs};
  auto created = LocalEstimator::create(calibration(1U), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id);

  const cv::Mat image = texturedImage();
  auto keyframe =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 20'000U, 1'100 * kMs), true);
  ASSERT_TRUE(keyframe) << keyframe.error().detail;
  ASSERT_EQ(keyframe.value().frame.disposition, VisualFrameDisposition::KeyframeStateRequested);
  ASSERT_TRUE(
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 20'001U, 1'200 * kMs), false));

  // The sweep spans [1100, 1190] ms. Its end is only 90 ms after the admitted
  // visual request. The sensor-neutral timeline keeps the visual state and
  // terminates only the later LiDAR request; it never aliases the scan factor
  // to the nearby camera state.
  const auto lidar_enqueue = estimator.enqueueLidar(sweep(21'000U, 1'100 * kMs));
  ASSERT_TRUE(lidar_enqueue);
  EXPECT_EQ(lidar_enqueue.value().state_admission, StateAdmissionDisposition::SuppressedTooClose);
  auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;

  ASSERT_EQ(processed.value().camera_commits.size(), 1U);
  EXPECT_TRUE(processed.value().commits.empty());
  ASSERT_EQ(processed.value().dropped_sweeps.size(), 1U);
  EXPECT_EQ(processed.value().dropped_sweeps.front().reason,
            LidarDropReason::StateRequestSuppressedTooClose);
  const CameraKnotCommitReport& camera = processed.value().camera_commits.front();
  EXPECT_EQ(camera.exact_time, core::FusionTime{1'100 * kMs});
  EXPECT_EQ(estimator.statistics().graph_commits, 2U);
  EXPECT_EQ(estimator.statistics().visual_keyframes_resolved, 1U);
  EXPECT_EQ(estimator.statistics().lidar_state_requests_suppressed_by_timeline, 1U);
}

TEST(LocalEstimatorVisual, EarlierLidarRequestSuppressesOnlyCloseVisualRequest) {
  LocalEstimatorConfig config = estimatorConfig({core::CameraId{0U}});
  config.state_timeline.minimum_state_interval = core::Duration{100 * kMs};
  auto created = LocalEstimator::create(calibration(1U), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  ASSERT_EQ(estimator.estimate().value().state_time, core::FusionTime{1'000 * kMs});
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id);

  // LiDAR and camera ingress are independent. The LiDAR request arrives first
  // at 1190 ms, so the close 1200 ms visual request terminates as suppressed;
  // neither request displaces or cancels the other sensor's accepted work.
  ASSERT_TRUE(estimator.enqueueLidar(sweep(22'000U, 1'100 * kMs)));
  const cv::Mat image = texturedImage();
  auto camera =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 22'001U, 1'200 * kMs), true);
  ASSERT_TRUE(camera) << camera.error().detail;
  ASSERT_EQ(camera.value().frame.disposition, VisualFrameDisposition::KeyframeSuppressedByTimeline);
  ASSERT_TRUE(camera.value().frame.state_admission);
  const core::KnotRequestId suppressed_request = camera.value().frame.state_admission->request.id;
  EXPECT_TRUE(suppressed_request.valid());
  EXPECT_EQ(camera.value().frame.state_admission->suppressing_state_time,
            core::FusionTime{1'190 * kMs});
  EXPECT_EQ(camera.value().pending_camera_knots, 0U);

  auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().commits.size(), 1U);
  EXPECT_TRUE(processed.value().camera_commits.empty());
  EXPECT_EQ(processed.value().commits.front().commit.state_time, core::FusionTime{1'190 * kMs});
  EXPECT_EQ(processed.value().pending_camera_knots, 0U);
  EXPECT_EQ(estimator.statistics().graph_commits, 2U);
  EXPECT_EQ(estimator.statistics().visual_keyframe_requests, 0U);

  // Timeline suppression removes only unresolved graph work. Tracking still
  // advanced on the suppressed image and remains continuous.
  auto continued = estimator.ingestCamera(
      frame(translated(image, 2.0), core::CameraId{0U}, 22'002U, 1'300 * kMs), true);
  ASSERT_TRUE(continued) << continued.error().detail;
  EXPECT_GT(continued.value().frame.frontend.report.input_tracks, 0U);
  EXPECT_EQ(continued.value().frame.disposition, VisualFrameDisposition::KeyframeStateRequested);
}

TEST(LocalEstimatorVisual, ResolvesEveryExactlySharedCameraWithoutPrematureTimelineFinality) {
  LocalEstimatorConfig config = estimatorConfig({core::CameraId{0U}, core::CameraId{1U}});
  // Initialization is itself a graph-published timeline state, followed by
  // the first shared frontend state and one later frontend state.
  config.graph.maximum_navigation_states = 2U;
  config.state_timeline.maximum_navigation_states = 3U;
  config.state_timeline.maximum_retained_requests = 4U;
  auto created = LocalEstimator::create(calibration(2U), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id);

  const cv::Mat image = texturedImage();
  auto camera_zero =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 21'000U, 1'190 * kMs), true);
  ASSERT_TRUE(camera_zero) << camera_zero.error().detail;
  ASSERT_TRUE(camera_zero.value().frame.state_admission);
  const core::KnotRequestId camera_zero_request =
      camera_zero.value().frame.state_admission->request.id;

  auto camera_one =
      estimator.ingestCamera(frame(image, core::CameraId{1U}, 21'001U, 1'190 * kMs), true);
  ASSERT_TRUE(camera_one) << camera_one.error().detail;
  ASSERT_TRUE(camera_one.value().frame.state_admission);
  const core::KnotRequestId camera_one_request =
      camera_one.value().frame.state_admission->request.id;
  EXPECT_EQ(camera_one.value().frame.state_admission->disposition,
            StateAdmissionDisposition::ExactShare);

  ASSERT_TRUE(
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 21'002U, 1'200 * kMs), false));
  ASSERT_TRUE(
      estimator.ingestCamera(frame(image, core::CameraId{1U}, 21'003U, 1'200 * kMs), false));
  ASSERT_TRUE(estimator.enqueueLidar(sweep(22'000U, 1'100 * kMs)));

  auto processed = estimator.processReady();
  ASSERT_TRUE(processed) << processed.error().detail;
  ASSERT_EQ(processed.value().commits.size(), 1U);
  ASSERT_EQ(processed.value().camera_commits.size(), 1U);
  const CameraKnotCommitReport& shared_commit = processed.value().camera_commits.front();
  ASSERT_EQ(shared_commit.resolved_keyframes.size(), 2U);
  for (const VisualResolvedKeyframeReport& resolved : shared_commit.resolved_keyframes) {
    EXPECT_EQ(resolved.resolution.state, processed.value().commits.front().commit.state);
    ASSERT_EQ(resolved.resolution.timeline.requests.size(), 3U);
    EXPECT_TRUE(std::binary_search(resolved.resolution.timeline.requests.begin(),
                                   resolved.resolution.timeline.requests.end(),
                                   camera_zero_request));
    EXPECT_TRUE(std::binary_search(resolved.resolution.timeline.requests.begin(),
                                   resolved.resolution.timeline.requests.end(),
                                   camera_one_request));
  }
  EXPECT_NE(shared_commit.resolved_keyframes[0].resolution.request,
            shared_commit.resolved_keyframes[1].resolution.request);
  EXPECT_EQ(estimator.statistics().visual_keyframes_resolved, 2U);

  // The committed state remains live in the timeline until graph-published
  // marginalization, leaving its exact-share API intact. A bounded second
  // slot admits this later independent state without prematurely pruning the
  // first resolution. Initialization occupies the other retained slot.
  appendImu(estimator, imu_time, 1'270 * kMs, imu_id);
  auto later = estimator.ingestCamera(frame(image, core::CameraId{0U}, 21'004U, 1'250 * kMs), true);
  ASSERT_TRUE(later) << later.error().detail;
  ASSERT_EQ(later.value().frame.disposition, VisualFrameDisposition::KeyframeStateRequested);
  auto later_processed = estimator.processReady();
  ASSERT_TRUE(later_processed) << later_processed.error().detail;
  ASSERT_EQ(later_processed.value().camera_commits.size(), 1U);
  ASSERT_EQ(later_processed.value().camera_commits.front().resolved_keyframes.size(), 1U);
  EXPECT_EQ(estimator.statistics().visual_keyframes_resolved, 3U);
}

TEST(LocalEstimatorVisual, WaitsForExactImuAndTracksLateFramesWithoutAKnot) {
  auto created = LocalEstimator::create(calibration(2U),
                                        estimatorConfig({core::CameraId{0U}, core::CameraId{1U}}));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  appendImu(estimator, imu_time, 1'200 * kMs, imu_id);

  const cv::Mat image = texturedImage();
  auto queued =
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 30'000U, 1'250 * kMs), true);
  ASSERT_TRUE(queued) << queued.error().detail;
  auto waiting = estimator.processReady();
  ASSERT_TRUE(waiting) << waiting.error().detail;
  EXPECT_TRUE(waiting.value().waiting_for_future_imu);
  EXPECT_EQ(waiting.value().pending_camera_knots, 1U);
  EXPECT_TRUE(waiting.value().camera_commits.empty());

  appendImu(estimator, imu_time, 1'260 * kMs, imu_id);
  auto committed = estimator.processReady();
  ASSERT_TRUE(committed) << committed.error().detail;
  ASSERT_EQ(committed.value().camera_commits.size(), 1U);
  EXPECT_EQ(committed.value().camera_commits.front().exact_time, core::FusionTime{1'250 * kMs});

  auto late = estimator.ingestCamera(frame(image, core::CameraId{1U}, 30'001U, 1'220 * kMs), true);
  ASSERT_TRUE(late) << late.error().detail;
  EXPECT_TRUE(late.value().late_for_graph);
  EXPECT_EQ(late.value().frame.disposition, VisualFrameDisposition::TrackingOnlyNoLocalState);
  EXPECT_EQ(late.value().pending_camera_knots, 0U);
  EXPECT_EQ(estimator.statistics().camera_frames_late_for_graph, 1U);
}

TEST(LocalEstimatorVisual, InterleavedLanesRemapIdenticalPrivateIdsToDistinctGraphIds) {
  auto created = LocalEstimator::create(calibration(2U),
                                        estimatorConfig({core::CameraId{0U}, core::CameraId{1U}}));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);
  const cv::Mat base = texturedImage();

  struct FrameStep {
    core::CameraId camera;
    std::int64_t time_ns;
    double image_translation;
    std::uint64_t id;
  };
  const std::vector<FrameStep> steps{
      {core::CameraId{0U}, 1'100 * kMs, 0.0, 40'000U},
      {core::CameraId{1U}, 1'150 * kMs, 0.0, 40'001U},
      {core::CameraId{0U}, 1'300 * kMs, 4.0, 40'002U},
      {core::CameraId{1U}, 1'350 * kMs, 4.0, 40'003U},
      {core::CameraId{0U}, 1'500 * kMs, 8.0, 40'004U},
      {core::CameraId{1U}, 1'550 * kMs, 8.0, 40'005U},
      {core::CameraId{0U}, 1'700 * kMs, 12.0, 40'006U},
  };

  std::optional<VisualGraphAttachmentReport> camera_zero_attachment;
  std::optional<VisualGraphAttachmentReport> camera_one_attachment;
  for (const FrameStep& step : steps) {
    SCOPED_TRACE(step.id);
    appendImu(estimator, imu_time, step.time_ns, imu_id, true);
    auto ingested = estimator.ingestCamera(
        frame(translated(base, step.image_translation), step.camera, step.id, step.time_ns), true);
    ASSERT_TRUE(ingested) << ingested.error().detail;
    auto processed = estimator.processReady();
    ASSERT_TRUE(processed) << processed.error().detail;
    ASSERT_EQ(processed.value().camera_commits.size(), 1U);
    const auto& attachment = processed.value().camera_commits.front().visual_attachment;
    if (attachment && !attachment->graph_factors.empty()) {
      if (attachment->camera == core::CameraId{0U}) {
        camera_zero_attachment = attachment;
      } else if (attachment->camera == core::CameraId{1U}) {
        camera_one_attachment = attachment;
      }
    }
  }

  ASSERT_TRUE(camera_zero_attachment);
  ASSERT_TRUE(camera_one_attachment);
  ASSERT_FALSE(camera_zero_attachment->graph_landmarks.empty());
  ASSERT_FALSE(camera_one_attachment->graph_landmarks.empty());
  ASSERT_FALSE(camera_zero_attachment->graph_factors.empty());
  ASSERT_FALSE(camera_one_attachment->graph_factors.empty());
  EXPECT_NE(camera_zero_attachment->graph_landmarks.front(),
            camera_one_attachment->graph_landmarks.front());
  EXPECT_NE(camera_zero_attachment->graph_factors.front(),
            camera_one_attachment->graph_factors.front());
  EXPECT_EQ(estimator.statistics().visual_graph_attachments, 2U);
  EXPECT_GT(estimator.statistics().visual_factors_attached, 0U);

  const core::FactorId retiring = camera_zero_attachment->graph_factors.front();
  auto wrong_lane =
      estimator.applyVisualResiduals(core::CameraId{1U}, estimator.estimate().value().revision,
                                     {VisualResidualFeedback{retiring, 100.0}});
  ASSERT_FALSE(wrong_lane);
  EXPECT_EQ(wrong_lane.error().code, LocalEstimatorErrorCode::VisualReferenceUnavailable);

  auto feedback_one =
      estimator.applyVisualResiduals(core::CameraId{0U}, estimator.estimate().value().revision,
                                     {VisualResidualFeedback{retiring, 100.0}});
  ASSERT_TRUE(feedback_one) << feedback_one.error().detail;
  EXPECT_EQ(feedback_one.value().retirements_queued, 0U);

  appendImu(estimator, imu_time, 1'750 * kMs, imu_id, true);
  ASSERT_TRUE(estimator.ingestCamera(
      frame(translated(base, 12.0), core::CameraId{1U}, 40'007U, 1'750 * kMs), true));
  auto intervening = estimator.processReady();
  ASSERT_TRUE(intervening) << intervening.error().detail;
  auto feedback_two =
      estimator.applyVisualResiduals(core::CameraId{0U}, estimator.estimate().value().revision,
                                     {VisualResidualFeedback{retiring, 100.0}});
  ASSERT_TRUE(feedback_two) << feedback_two.error().detail;
  EXPECT_EQ(feedback_two.value().retirements_queued, 1U);
  ASSERT_EQ(feedback_two.value().factor_builder.retired_observation_factors.size(), 1U);
  EXPECT_EQ(feedback_two.value().factor_builder.retired_observation_factors.front(), retiring);

  appendImu(estimator, imu_time, 1'900 * kMs, imu_id, true);
  ASSERT_TRUE(estimator.ingestCamera(
      frame(translated(base, 16.0), core::CameraId{0U}, 40'008U, 1'900 * kMs), true));
  auto retired = estimator.processReady();
  ASSERT_TRUE(retired) << retired.error().detail;
  ASSERT_EQ(retired.value().camera_commits.size(), 1U);
  std::optional<VisualGraphAttachmentReport> retirement_attachment =
      retired.value().camera_commits.front().visual_attachment;
  if (!retirement_attachment || retirement_attachment->camera != core::CameraId{0U}) {
    appendImu(estimator, imu_time, 1'950 * kMs, imu_id, true);
    ASSERT_TRUE(estimator.ingestCamera(
        frame(translated(base, 16.0), core::CameraId{1U}, 40'009U, 1'950 * kMs), true));
    auto fair_retry = estimator.processReady();
    ASSERT_TRUE(fair_retry) << fair_retry.error().detail;
    ASSERT_EQ(fair_retry.value().camera_commits.size(), 1U);
    retirement_attachment = fair_retry.value().camera_commits.front().visual_attachment;
  }
  ASSERT_TRUE(retirement_attachment);
  EXPECT_EQ(retirement_attachment->camera, core::CameraId{0U});
  ASSERT_EQ(retirement_attachment->graph_factor_retirements.size(), 1U);
  EXPECT_EQ(retirement_attachment->graph_factor_retirements.front(), retiring);
}

TEST(LocalEstimatorVisual, RejectedLidarCandidateAttachesPendingVisualBatchOnceOnImuFallback) {
  LocalEstimatorConfig config = estimatorConfig({core::CameraId{0U}});
  config.lidar_registration.minimum_correspondences = 10'000U;
  auto created = LocalEstimator::create(calibration(1U), std::move(config));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  std::int64_t imu_time{};
  std::uint64_t imu_id = 1U;
  initialize(estimator, imu_time, imu_id);

  const cv::Mat image = texturedImage();
  appendImu(estimator, imu_time, 1'220 * kMs, imu_id, true);
  ASSERT_TRUE(estimator.enqueueLidar(sweep(60'001U, 1'100 * kMs)));
  ASSERT_TRUE(
      estimator.ingestCamera(frame(image, core::CameraId{0U}, 60'000U, 1'200 * kMs), false));
  auto bootstrap = estimator.processReady();
  ASSERT_TRUE(bootstrap) << bootstrap.error().detail;
  ASSERT_EQ(bootstrap.value().commits.size(), 1U);
  EXPECT_EQ(bootstrap.value().commits.front().disposition, LidarCommitDisposition::BootstrapTarget)
      << bootstrap.value().commits.front().degradation_detail;

  const std::array<std::pair<std::int64_t, double>, 3> keyframes{{
      {1'300 * kMs, 2.0},
      {1'500 * kMs, 6.0},
      {1'700 * kMs, 10.0},
  }};
  std::uint64_t camera_id = 60'002U;
  for (const auto& [time, translation] : keyframes) {
    appendImu(estimator, imu_time, time, imu_id);
    auto ingested = estimator.ingestCamera(
        frame(translated(image, translation), core::CameraId{0U}, camera_id++, time), true);
    ASSERT_TRUE(ingested) << ingested.error().detail;
    auto processed = estimator.processReady();
    ASSERT_TRUE(processed) << processed.error().detail;
    ASSERT_EQ(processed.value().camera_commits.size(), 1U);
  }

  const std::size_t attachments_before = estimator.statistics().visual_graph_attachments;
  const std::size_t factors_before = estimator.statistics().visual_factors_attached;
  appendImu(estimator, imu_time, 1'920 * kMs, imu_id);
  ASSERT_TRUE(estimator.enqueueLidar(sweep(60'100U, 1'800 * kMs)));
  ASSERT_TRUE(estimator.ingestCamera(
      frame(translated(image, 12.0), core::CameraId{0U}, camera_id++, 1'900 * kMs), false));

  auto degraded = estimator.processReady();
  ASSERT_TRUE(degraded) << degraded.error().detail;
  ASSERT_EQ(degraded.value().commits.size(), 1U);
  const LidarCommitReport& commit = degraded.value().commits.front();
  EXPECT_EQ(commit.disposition, LidarCommitDisposition::ImuOnlyRegistrationRejectedTargetRetained)
      << commit.degradation_detail;
  ASSERT_TRUE(commit.visual_attachment);
  ASSERT_FALSE(commit.visual_attachment->graph_factors.empty());
  EXPECT_EQ(estimator.statistics().visual_graph_attachments, attachments_before + 1U);
  EXPECT_EQ(estimator.statistics().visual_factors_attached,
            factors_before + commit.visual_attachment->graph_factors.size());
  const std::set<core::FactorId> unique_factors(commit.visual_attachment->graph_factors.begin(),
                                                commit.visual_attachment->graph_factors.end());
  EXPECT_EQ(unique_factors.size(), commit.visual_attachment->graph_factors.size());

  std::vector<VisualResidualFeedback> feedback;
  feedback.reserve(commit.visual_attachment->graph_factors.size());
  for (const core::FactorId factor : commit.visual_attachment->graph_factors) {
    feedback.push_back(VisualResidualFeedback{factor, 0.0});
  }
  auto accepted = estimator.applyVisualResiduals(core::CameraId{0U},
                                                 estimator.estimate().value().revision, feedback);
  ASSERT_TRUE(accepted) << accepted.error().detail;
}

TEST(LocalEstimatorVisual, CalibratedButDisabledCameraIsRejectedExplicitly) {
  auto created = LocalEstimator::create(calibration(1U), estimatorConfig({}));
  ASSERT_TRUE(created) << created.error().detail;
  auto estimator = std::move(created).value();
  auto rejected =
      estimator.ingestCamera(frame(texturedImage(), core::CameraId{0U}, 50'000U, 100 * kMs), true);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalEstimatorErrorCode::CameraNotEnabled);
}

}  // namespace
}  // namespace meridian::local
