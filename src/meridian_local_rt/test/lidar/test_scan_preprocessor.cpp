#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include "meridian/local_rt/lidar/scan_preprocessor.hpp"

namespace meridian::local_rt::lidar {
namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000;

core::Pose3d yawPose(double x, double yaw_rad) {
  const double half_yaw = 0.5 * yaw_rad;
  return core::Pose3d({.x = x, .y = 0.0, .z = 0.0},
                      core::Quaterniond(std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw)));
}

core::LidarSweep sweepWithPoints(std::vector<core::LidarPoint> points) {
  return core::LidarSweep(
      core::ObservationHeader(core::SensorId("lidar"), core::CalibrationId("calibration"),
                              core::MeasurementId(1U), core::TimeNs(0), "lidar"),
      core::TimeNs(0), core::TimeNs(kSecondNs), std::move(points));
}

DensePropagation rotatingPropagation() {
  const core::NavigationState endpoint(core::StateId(2U), core::TimeNs(kSecondNs),
                                       yawPose(2.0, 0.5 * std::numbers::pi), {}, {});
  return DensePropagation{
      .samples = {{.time = endpoint.time(),
                   .odom_from_imu = endpoint.odomFromImu(),
                   .velocity_odom_m_s = {}}},
      .endpoint = endpoint,
  };
}

TEST(ScanPreprocessorTest, UsesExtrinsicAndInterpolatedPoseForSweepEndDeskew) {
  ScanPreprocessorOptions options;
  options.minimum_range_m = 0.1;
  options.maximum_range_m = 10.0;
  options.target_downsample_voxel_m = 0.1;
  options.source_downsample_voxel_m = 0.1;
  // T_imu_lidar has a one-metre lever arm. Rotation of this lever arm is what
  // makes the expected start and midpoint coordinates extrinsic-sensitive.
  options.T_imu_lidar = core::Pose3d({.x = 1.0, .y = 0.0, .z = 0.0}, {});
  const ScanPreprocessor preprocessor(options);

  std::vector<core::LidarPoint> raw{
      {.x = 2.0F,
       .time_offset_ns = 0,
       .source_index = 0U,
       .intensity = std::nullopt,
       .ring = std::nullopt},
      {.x = 2.0F,
       .time_offset_ns = kSecondNs / 2,
       .source_index = 1U,
       .intensity = std::nullopt,
       .ring = std::nullopt},
      {.x = 2.01F,
       .time_offset_ns = kSecondNs,
       .source_index = 2U,
       .intensity = std::nullopt,
       .ring = std::nullopt},
      {.x = 2.04F,
       .time_offset_ns = kSecondNs,
       .source_index = 3U,
       .intensity = std::nullopt,
       .ring = std::nullopt},
  };
  const core::NavigationState seed(core::StateId(1U), core::TimeNs(0), {}, {}, {});
  const DensePropagation propagation = rotatingPropagation();
  const PreparedScan prepared =
      preprocessor.prepare(sweepWithPoints(raw), seed, core::StateId(2U), propagation);

  ASSERT_EQ(prepared.frame.time, core::TimeNs(kSecondNs));
  ASSERT_EQ(prepared.frame.state_id, core::StateId(2U));
  ASSERT_EQ(prepared.frame.target_points_lidar.size(), 3U);
  ASSERT_EQ(prepared.frame.source_points_lidar.size(), 3U);
  EXPECT_EQ(prepared.stats.input_points, 4U);
  EXPECT_EQ(prepared.stats.range_rejected_points, 0U);
  EXPECT_EQ(prepared.stats.support_rejected_points, 0U);
  EXPECT_EQ(prepared.stats.deskewed_points, 4U);

  const double midpoint = 3.0 / std::sqrt(2.0) - 1.0;
  const std::array<Point3d, 3> expected{
      Point3d{-1.0, -1.0, 0.0},
      Point3d{midpoint, -midpoint, 0.0},
      Point3d{static_cast<double>(2.04F), 0.0, 0.0},
  };
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    EXPECT_TRUE(prepared.frame.target_points_lidar[index].isApprox(expected[index], 1.0e-6));
    EXPECT_TRUE(prepared.frame.source_points_lidar[index].isApprox(expected[index], 1.0e-6));
  }

  // Voxel representatives and output ordering must be independent of incoming
  // packet order; the per-point acquisition offsets remain attached to points.
  std::reverse(raw.begin(), raw.end());
  const PreparedScan reversed =
      preprocessor.prepare(sweepWithPoints(std::move(raw)), seed, core::StateId(2U), propagation);
  ASSERT_EQ(reversed.frame.target_points_lidar.size(), prepared.frame.target_points_lidar.size());
  for (std::size_t index = 0U; index < prepared.frame.target_points_lidar.size(); ++index) {
    EXPECT_TRUE(reversed.frame.target_points_lidar[index].isApprox(
        prepared.frame.target_points_lidar[index], 0.0));
    EXPECT_TRUE(reversed.frame.source_points_lidar[index].isApprox(
        prepared.frame.source_points_lidar[index], 0.0));
  }
}

}  // namespace
}  // namespace meridian::local_rt::lidar
