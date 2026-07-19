#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/equidistant_camera.hpp"
#include "meridian/local/visual_frontend.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] EquidistantCameraParameters newerCollegeCamera() {
  return EquidistantCameraParameters{720U,    540U,     352.779,  354.609, 359.035,
                                     260.546, -0.04217, -0.00413, 0.00179, -0.00063};
}

[[nodiscard]] cv::Mat texturedImage() {
  cv::Mat image(540, 720, CV_8UC1);
  cv::RNG random(0x5EEDU);
  random.fill(image, cv::RNG::UNIFORM, 0, 255);
  cv::GaussianBlur(image, image, cv::Size(3, 3), 0.6);
  for (int y = 20; y < image.rows; y += 40) {
    for (int x = 20; x < image.cols; x += 40) {
      const int shade = ((x / 40 + y / 40) % 2 == 0) ? 20 : 235;
      cv::rectangle(image, cv::Rect(x - 4, y - 4, 9, 9), cv::Scalar(shade), cv::FILLED);
    }
  }
  return image;
}

[[nodiscard]] cv::Mat translated(const cv::Mat& image, double x, double y) {
  cv::Mat output;
  const cv::Matx23d transform{1.0, 0.0, x, 0.0, 1.0, y};
  cv::warpAffine(image, output, transform, image.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT101);
  return output;
}

[[nodiscard]] cv::Mat rolled(const cv::Mat& image, const EquidistantCameraParameters& camera,
                             double angle_rad) {
  cv::Mat map_x(image.size(), CV_32FC1);
  cv::Mat map_y(image.size(), CV_32FC1);
  const double cosine = std::cos(angle_rad);
  const double sine = std::sin(angle_rad);
  for (int row = 0; row < image.rows; ++row) {
    for (int column = 0; column < image.cols; ++column) {
      const double current_x = (static_cast<double>(column) - camera.cx) / camera.fx;
      const double current_y = (static_cast<double>(row) - camera.cy) / camera.fy;
      const double previous_x = cosine * current_x + sine * current_y;
      const double previous_y = -sine * current_x + cosine * current_y;
      map_x.at<float>(row, column) = static_cast<float>(camera.fx * previous_x + camera.cx);
      map_y.at<float>(row, column) = static_cast<float>(camera.fy * previous_y + camera.cy);
    }
  }
  cv::Mat output;
  cv::remap(image, output, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REFLECT101);
  return output;
}

[[nodiscard]] core::CameraFrame makeFrame(const cv::Mat& image, std::uint64_t id,
                                          std::int64_t time_ns) {
  auto pixels =
      std::make_shared<std::vector<std::byte>>(image.step * static_cast<std::size_t>(image.rows));
  for (int row = 0; row < image.rows; ++row) {
    std::memcpy(pixels->data() + static_cast<std::size_t>(row) * image.step, image.ptr(row),
                image.step);
  }

  core::CameraFrame frame;
  frame.header.direct_calibration = core::CalibrationEpoch(7U);
  frame.id = core::MeasurementId(id);
  frame.camera = core::CameraId(0U);
  frame.stamp.source_epoch = core::SourceEpoch(3U);
  frame.stamp.fusion_time = core::FusionTime{time_ns};
  frame.exposure_midpoint = core::FusionTime{time_ns};
  frame.width = static_cast<std::uint32_t>(image.cols);
  frame.height = static_cast<std::uint32_t>(image.rows);
  frame.stride = static_cast<std::uint32_t>(image.step);
  frame.encoding = core::ImageEncoding::Mono8;
  frame.pixels = pixels;
  return frame;
}

TEST(EquidistantCamera, ProjectionRoundTripAndAnalyticJacobians) {
  const EquidistantCamera camera(newerCollegeCamera());
  ASSERT_TRUE(camera.valid());

  const std::vector<Eigen::Vector3d> points{
      {0.0, 0.0, 1.0}, {0.3, -0.2, 1.4}, {-0.8, 0.45, 1.2}, {0.95, 0.65, 1.0}};
  constexpr double kStep = 1.0e-6;
  for (const Eigen::Vector3d& point : points) {
    const auto projection = camera.project(point);
    ASSERT_TRUE(projection) << projection.error().detail;
    const auto ray = camera.unproject(projection.value().pixel);
    ASSERT_TRUE(ray) << ray.error().detail;
    EXPECT_LT((ray.value().unit_ray - point.normalized()).norm(), 2.0e-10);

    Eigen::Matrix<double, 2, 3> numeric_projection;
    for (int axis = 0; axis < 3; ++axis) {
      Eigen::Vector3d plus = point;
      Eigen::Vector3d minus = point;
      plus(axis) += kStep;
      minus(axis) -= kStep;
      const auto plus_projection = camera.project(plus);
      const auto minus_projection = camera.project(minus);
      ASSERT_TRUE(plus_projection);
      ASSERT_TRUE(minus_projection);
      numeric_projection.col(axis) =
          (plus_projection.value().pixel - minus_projection.value().pixel) / (2.0 * kStep);
    }
    EXPECT_LT((numeric_projection - projection.value().point_jacobian).norm(), 2.0e-5);

    Eigen::Matrix<double, 3, 2> numeric_unprojection;
    for (int axis = 0; axis < 2; ++axis) {
      Eigen::Vector2d plus = projection.value().pixel;
      Eigen::Vector2d minus = projection.value().pixel;
      plus(axis) += kStep;
      minus(axis) -= kStep;
      const auto plus_ray = camera.unproject(plus);
      const auto minus_ray = camera.unproject(minus);
      ASSERT_TRUE(plus_ray);
      ASSERT_TRUE(minus_ray);
      numeric_unprojection.col(axis) =
          (plus_ray.value().unit_ray - minus_ray.value().unit_ray) / (2.0 * kStep);
    }
    EXPECT_LT((numeric_unprojection - ray.value().pixel_jacobian).norm(), 2.0e-7);
  }
}

TEST(EquidistantCamera, RejectsNonMonotonicCalibrationAndBehindPoint) {
  EquidistantCameraParameters parameters = newerCollegeCamera();
  parameters.k1 = -10.0;
  const EquidistantCamera invalid_camera(parameters);
  EXPECT_FALSE(invalid_camera.valid());

  const EquidistantCamera camera(newerCollegeCamera());
  const auto behind = camera.project(Eigen::Vector3d{0.0, 0.0, -1.0});
  ASSERT_FALSE(behind);
  EXPECT_EQ(behind.error().code, CameraModelErrorCode::PointBehindCamera);
}

TEST(GridKltVisualFrontend, TracksReplenishesAndDescriptorsOnlyKeyframes) {
  const EquidistantCamera camera(newerCollegeCamera());
  VisualFrontendConfig config;
  config.features_per_cell = 3U;
  config.recovery_minimum_tracks = 24U;
  config.keyframe_minimum_tracks = 60U;
  GridKltVisualFrontend frontend(camera, config);

  const cv::Mat first_image = texturedImage();
  const cv::Mat second_image = translated(first_image, 3.0, -2.0);

  const auto first = frontend.process(makeFrame(first_image, 10U, 0LL));
  ASSERT_TRUE(first) << first.error().detail;
  EXPECT_TRUE(first.value().keyframe);
  EXPECT_EQ(first.value().report.keyframe_reason, VisualKeyframeReason::FirstFrame);
  EXPECT_GT(first.value().features.size(), 100U);
  EXPECT_GT(first.value().descriptors.size(), 50U);
  EXPECT_EQ(core::validateLineage(first.value().lineage), core::LineageValidationError::None);

  VisualRotationPrior rotation_prior;
  rotation_prior.previous_exposure_midpoint = core::FusionTime{0LL};
  rotation_prior.current_exposure_midpoint = core::FusionTime{50'000'000LL};
  rotation_prior.imu_calibration = core::CalibrationEpoch(8U);
  rotation_prior.imu_support = {core::MeasurementId(100U), core::MeasurementId(101U)};
  const auto second = frontend.process(makeFrame(second_image, 11U, 50'000'000LL), rotation_prior);
  ASSERT_TRUE(second) << second.error().detail;
  EXPECT_FALSE(second.value().keyframe);
  EXPECT_TRUE(second.value().descriptors.empty());
  EXPECT_EQ(second.value().report.rotation_prior, RotationPriorStatus::Applied);
  EXPECT_GT(second.value().report.rotation_seeded_tracks, 100U);
  EXPECT_GT(second.value().report.retained_tracks, 100U);
  EXPECT_GT(second.value().report.retained_fraction, 0.75);
  EXPECT_EQ(second.value().lineage.usage.size(), 3U);
  EXPECT_EQ(core::validateLineage(second.value().lineage), core::LineageValidationError::None);

  const auto timed_keyframe = frontend.process(makeFrame(second_image, 12U, 1'100'000'000LL));
  ASSERT_TRUE(timed_keyframe) << timed_keyframe.error().detail;
  EXPECT_TRUE(timed_keyframe.value().keyframe);
  EXPECT_EQ(timed_keyframe.value().report.keyframe_reason, VisualKeyframeReason::MaximumInterval);
  EXPECT_FALSE(timed_keyframe.value().descriptors.empty());

  const cv::Mat blank = cv::Mat::zeros(first_image.size(), CV_8UC1);
  const auto recovery = frontend.process(makeFrame(blank, 13U, 1'300'000'000LL));
  ASSERT_TRUE(recovery) << recovery.error().detail;
  EXPECT_TRUE(recovery.value().keyframe);
  EXPECT_EQ(recovery.value().report.keyframe_reason, VisualKeyframeReason::Recovery);
  EXPECT_EQ(recovery.value().report.quality, VisualTrackingQuality::Recovery);
  EXPECT_TRUE(recovery.value().descriptors.empty());
  EXPECT_EQ(recovery.value().report.output_tracks, 0U);
}

TEST(GridKltVisualFrontend, RejectsEpochBreakUntilExplicitReset) {
  const EquidistantCamera camera(newerCollegeCamera());
  GridKltVisualFrontend frontend(camera);
  const cv::Mat image = texturedImage();
  ASSERT_TRUE(frontend.process(makeFrame(image, 20U, 0LL)));

  core::CameraFrame epoch_break = makeFrame(image, 21U, 50'000'000LL);
  epoch_break.stamp.source_epoch = core::SourceEpoch(4U);
  const auto rejected = frontend.process(epoch_break);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualFrontendErrorCode::SourceEpochChanged);

  frontend.reset();
  EXPECT_FALSE(frontend.initialized());
  EXPECT_TRUE(frontend.process(epoch_break));
}

TEST(GridKltVisualFrontend, TimeUncertaintyFailureDoesNotAdvanceState) {
  const EquidistantCamera camera(newerCollegeCamera());
  GridKltVisualFrontend frontend(camera);
  const cv::Mat image = texturedImage();
  ASSERT_TRUE(frontend.process(makeFrame(image, 40U, 0LL)));

  core::CameraFrame uncertain = makeFrame(image, 41U, 50'000'000LL);
  uncertain.stamp.uncertainty = core::Duration{3'000'000LL};
  const auto rejected = frontend.process(uncertain);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualFrontendErrorCode::TimeUncertain);

  // The rejected timestamp was not committed, so the corrected record at the
  // same time remains admissible.
  EXPECT_TRUE(frontend.process(makeFrame(image, 42U, 50'000'000LL)));
}

TEST(GridKltVisualFrontend, PureRotationDoesNotCreateParallaxKeyframe) {
  const EquidistantCameraParameters parameters = newerCollegeCamera();
  const EquidistantCamera camera(parameters);
  GridKltVisualFrontend frontend(camera);
  const cv::Mat first_image = texturedImage();
  constexpr double kRollRadians = 0.08;
  const cv::Mat rolled_image = rolled(first_image, parameters, kRollRadians);

  ASSERT_TRUE(frontend.process(makeFrame(first_image, 30U, 0LL)));
  VisualRotationPrior prior;
  prior.R_current_previous =
      Eigen::AngleAxisd(kRollRadians, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  prior.previous_exposure_midpoint = core::FusionTime{0LL};
  prior.current_exposure_midpoint = core::FusionTime{200'000'000LL};
  prior.imu_calibration = core::CalibrationEpoch(8U);
  prior.imu_support = {core::MeasurementId(200U), core::MeasurementId(201U)};

  const auto rotated = frontend.process(makeFrame(rolled_image, 31U, 200'000'000LL), prior);
  ASSERT_TRUE(rotated) << rotated.error().detail;
  EXPECT_FALSE(rotated.value().keyframe);
  EXPECT_EQ(rotated.value().report.keyframe_reason, VisualKeyframeReason::None);
  EXPECT_TRUE(rotated.value().report.rotation_compensated_parallax_available);
  EXPECT_LT(rotated.value().report.median_keyframe_parallax_px, 1.0);
  EXPECT_GT(rotated.value().report.retained_fraction, 0.7);
}

}  // namespace
}  // namespace meridian::local
