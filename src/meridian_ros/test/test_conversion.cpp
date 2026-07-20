#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sensor_msgs/msg/point_field.hpp>
#include <string>
#include <vector>

#include "meridian/ros/conversion.hpp"

namespace meridian::ros {
namespace {

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

ImuConversionConfig imuConfig() {
  return ImuConversionConfig{
      .sensor_id = core::SensorId("test_imu"),
      .calibration_id = core::CalibrationId("test_imu_calibration"),
      .expected_frame_id = "imu_frame",
  };
}

LidarConversionConfig lidarConfig() {
  return LidarConversionConfig{
      .sensor_id = core::SensorId("test_lidar"),
      .calibration_id = core::CalibrationId("test_lidar_calibration"),
      .expected_frame_id = "lidar_frame",
      .x_field = "x",
      .y_field = "y",
      .z_field = "z",
      .time_field = "time_offset",
      .intensity_field = "intensity",
      .ring_field = "channel",
      .x_datatype = PointField::FLOAT32,
      .y_datatype = PointField::FLOAT32,
      .z_datatype = PointField::FLOAT32,
      .time_datatype = PointField::UINT32,
      .intensity_datatype = PointField::FLOAT32,
      .ring_datatype = PointField::UINT8,
      .time_scale_to_nanoseconds = 1.0,
      .time_reference = PointTimeReference::kOffsetFromHeader,
  };
}

PointField field(std::string name, std::uint32_t offset, std::uint8_t datatype,
                 std::uint32_t count = 1U) {
  PointField result;
  result.name = std::move(name);
  result.offset = offset;
  result.datatype = datatype;
  result.count = count;
  return result;
}

template <typename T>
void writeScalar(std::vector<std::uint8_t>& data, std::size_t offset, T value,
                 bool destination_big_endian) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  constexpr bool kHostBigEndian = std::endian::native == std::endian::big;
  if (destination_big_endian != kHostBigEndian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(data.data() + offset, bytes.data(), sizeof(T));
}

PointCloud2 makeOrganizedCloud(std::uint32_t width, std::uint32_t height,
                               std::uint32_t row_padding = 0U, bool big_endian = false) {
  PointCloud2 cloud;
  cloud.header.stamp.sec = 1;
  cloud.header.stamp.nanosec = 10U;
  cloud.header.frame_id = "lidar_frame";
  cloud.width = width;
  cloud.height = height;
  cloud.fields = {
      field("x", 0, PointField::FLOAT32),           field("y", 4, PointField::FLOAT32),
      field("z", 8, PointField::FLOAT32),           field("intensity", 16, PointField::FLOAT32),
      field("time_offset", 20, PointField::UINT32), field("reflectivity", 24, PointField::UINT16),
      field("channel", 26, PointField::UINT8),      field("ambient", 28, PointField::UINT16),
      field("range", 32, PointField::UINT32),
  };
  cloud.is_bigendian = big_endian;
  cloud.point_step = 48U;
  cloud.row_step = width * cloud.point_step + row_padding;
  cloud.data.resize(static_cast<std::size_t>(height) * cloud.row_step);
  cloud.is_dense = false;
  return cloud;
}

void setPoint(PointCloud2& cloud, std::uint32_t row, std::uint32_t column, float x, float y,
              float z, float intensity, std::uint32_t time_offset, std::uint8_t ring) {
  const std::size_t offset = static_cast<std::size_t>(row) * cloud.row_step +
                             static_cast<std::size_t>(column) * cloud.point_step;
  writeScalar(cloud.data, offset + 0U, x, cloud.is_bigendian);
  writeScalar(cloud.data, offset + 4U, y, cloud.is_bigendian);
  writeScalar(cloud.data, offset + 8U, z, cloud.is_bigendian);
  writeScalar(cloud.data, offset + 16U, intensity, cloud.is_bigendian);
  writeScalar(cloud.data, offset + 20U, time_offset, cloud.is_bigendian);
  writeScalar(cloud.data, offset + 26U, ring, cloud.is_bigendian);
}

TEST(ImuConversion, ConvertsOnlyMeasuredAngularVelocityAndSpecificForce) {
  sensor_msgs::msg::Imu message;
  message.header.stamp.sec = 2;
  message.header.stamp.nanosec = 3U;
  message.header.frame_id = "imu_frame";
  message.orientation.x = std::numeric_limits<double>::quiet_NaN();
  message.angular_velocity.x = 1.0;
  message.angular_velocity.y = 2.0;
  message.angular_velocity.z = 3.0;
  message.linear_acceleration.x = 4.0;
  message.linear_acceleration.y = 5.0;
  message.linear_acceleration.z = 6.0;

  const auto result = convertImu(message, core::MeasurementId(42), imuConfig());

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result.value().header().measurementTime().count(), 2'000'000'003LL);
  EXPECT_EQ(result.value().header().measurementId().value(), 42U);
  EXPECT_EQ(result.value().angularVelocityRadS(), (core::Vec3d{1.0, 2.0, 3.0}));
  EXPECT_EQ(result.value().specificForceMS2(), (core::Vec3d{4.0, 5.0, 6.0}));
}

TEST(ImuConversion, RejectsNonFiniteMeasurementsAndUnexpectedFrame) {
  sensor_msgs::msg::Imu message;
  message.header.frame_id = "wrong";
  auto result = convertImu(message, core::MeasurementId(0), imuConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kUnexpectedFrameId);

  message.header.frame_id = "imu_frame";
  message.linear_acceleration.z = std::numeric_limits<double>::infinity();
  result = convertImu(message, core::MeasurementId(0), imuConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kNonFiniteImu);
  EXPECT_EQ(result.error().field, "linear_acceleration");
}

TEST(PointCloudConversion, AcceptsConfiguredLayoutPaddingAndRowTimeRewind) {
  PointCloud2 cloud = makeOrganizedCloud(2U, 2U, 8U);
  setPoint(cloud, 0, 0, 0.0F, 0.0F, 0.0F, 10.0F, 0U, 0U);
  setPoint(cloud, 0, 1, 1.0F, 2.0F, 3.0F, 20.0F, 100U, 1U);
  setPoint(cloud, 1, 0, 4.0F, 5.0F, 6.0F, std::numeric_limits<float>::quiet_NaN(), 0U, 2U);
  setPoint(cloud, 1, 1, std::numeric_limits<float>::quiet_NaN(), 8.0F, 9.0F, 40.0F, 100U, 3U);

  const auto result = convertPointCloud2(cloud, core::MeasurementId(7), lidarConfig());

  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  EXPECT_EQ(result.stats().source_points, 4U);
  EXPECT_EQ(result.stats().accepted_points, 3U);
  EXPECT_EQ(result.stats().nonfinite_xyz_points, 1U);
  EXPECT_EQ(result.stats().zero_xyz_points, 1U);
  EXPECT_EQ(result.stats().nonfinite_intensity_points, 1U);
  EXPECT_EQ(result.stats().flattened_time_regressions, 1U);
  EXPECT_EQ(result.stats().row_padding_bytes, 16U);
  EXPECT_EQ(result.stats().minimum_time_offset_ns, 0U);
  EXPECT_EQ(result.stats().maximum_time_offset_ns, 100U);
  EXPECT_TRUE(result.stats().has_intensity);
  EXPECT_TRUE(result.stats().has_ring);

  const core::LidarSweep& sweep = result.value();
  ASSERT_EQ(sweep.size(), 3U);
  EXPECT_EQ(sweep.acquisitionBegin().count(), 1'000'000'010LL);
  EXPECT_EQ(sweep.acquisitionEnd().count(), 1'000'000'110LL);
  EXPECT_EQ(sweep.points()[0].x, 0.0F);  // zero returns are preserved, not invalidated.
  EXPECT_EQ(sweep.points()[2].source_index, 2U);
  EXPECT_FALSE(sweep.points()[2].intensity.has_value());
  ASSERT_TRUE(sweep.points()[2].ring.has_value());
  EXPECT_EQ(*sweep.points()[2].ring, 2U);
}

TEST(PointCloudConversion, DecodesBigEndianFields) {
  PointCloud2 cloud = makeOrganizedCloud(1U, 1U, 0U, true);
  setPoint(cloud, 0, 0, 1.25F, -2.5F, 3.75F, 42.5F, 123'456U, 127U);

  const auto result = convertPointCloud2(cloud, core::MeasurementId(1), lidarConfig());

  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  ASSERT_EQ(result.value().size(), 1U);
  EXPECT_FLOAT_EQ(result.value().points()[0].x, 1.25F);
  EXPECT_FLOAT_EQ(result.value().points()[0].y, -2.5F);
  EXPECT_EQ(result.value().points()[0].time_offset_ns, 123'456);
  ASSERT_TRUE(result.value().points()[0].intensity.has_value());
  EXPECT_FLOAT_EQ(*result.value().points()[0].intensity, 42.5F);
  EXPECT_EQ(*result.value().points()[0].ring, 127U);
}

TEST(PointCloudConversion, DecodesConfiguredNumericTypesAndSecondOffsets) {
  PointCloud2 cloud;
  cloud.header.stamp.sec = 10;
  cloud.header.frame_id = "lidar_frame";
  cloud.width = 1U;
  cloud.height = 1U;
  cloud.fields = {
      field("x", 0, PointField::FLOAT64),         field("y", 8, PointField::INT16),
      field("z", 12, PointField::FLOAT32),        field("time_offset", 16, PointField::FLOAT32),
      field("intensity", 20, PointField::UINT16), field("channel", 22, PointField::UINT16),
  };
  cloud.point_step = 24U;
  cloud.row_step = 24U;
  cloud.data.resize(24U);
  writeScalar(cloud.data, 0U, 1.5, false);
  writeScalar(cloud.data, 8U, std::int16_t{-2}, false);
  writeScalar(cloud.data, 12U, 3.25F, false);
  writeScalar(cloud.data, 16U, 0.125F, false);
  writeScalar(cloud.data, 20U, std::uint16_t{512}, false);
  writeScalar(cloud.data, 22U, std::uint16_t{300}, false);

  auto config = lidarConfig();
  config.x_datatype = PointField::FLOAT64;
  config.y_datatype = PointField::INT16;
  config.time_datatype = PointField::FLOAT32;
  config.intensity_datatype = PointField::UINT16;
  config.ring_datatype = PointField::UINT16;
  config.time_scale_to_nanoseconds = 1.0e9;

  const auto result = convertPointCloud2(cloud, core::MeasurementId(3), config);
  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  ASSERT_EQ(result.value().size(), 1U);
  EXPECT_FLOAT_EQ(result.value().points()[0].x, 1.5F);
  EXPECT_FLOAT_EQ(result.value().points()[0].y, -2.0F);
  EXPECT_FLOAT_EQ(result.value().points()[0].z, 3.25F);
  EXPECT_EQ(result.value().points()[0].time_offset_ns, 125'000'000);
  EXPECT_FLOAT_EQ(*result.value().points()[0].intensity, 512.0F);
  EXPECT_EQ(*result.value().points()[0].ring, 300U);
  EXPECT_EQ(result.value().acquisitionBegin().count(), 10'125'000'000LL);
  EXPECT_EQ(result.value().acquisitionEnd().count(), 10'125'000'000LL);
}

TEST(PointCloudConversion, SupportsAbsolutePointTimeAndCloudsWithoutPointTime) {
  PointCloud2 cloud = makeOrganizedCloud(1U, 1U);
  cloud.header.stamp.sec = 2;
  cloud.header.stamp.nanosec = 0U;
  setPoint(cloud, 0, 0, 1.0F, 2.0F, 3.0F, 4.0F, 0U, 6U);
  for (PointField& value : cloud.fields) {
    if (value.name == "time_offset") {
      value.datatype = PointField::FLOAT64;
      value.offset = 36U;
    }
  }
  writeScalar(cloud.data, 36U, 2.25, false);

  auto config = lidarConfig();
  config.time_datatype = PointField::FLOAT64;
  config.time_scale_to_nanoseconds = 1.0e9;
  config.time_reference = PointTimeReference::kAbsolute;
  auto result = convertPointCloud2(cloud, core::MeasurementId(4), config);
  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  EXPECT_EQ(result.value().points()[0].time_offset_ns, 250'000'000);

  cloud = makeOrganizedCloud(1U, 1U);
  setPoint(cloud, 0, 0, 1.0F, 2.0F, 3.0F, 4.0F, 5U, 6U);
  cloud.fields.erase(
      std::remove_if(cloud.fields.begin(), cloud.fields.end(),
                     [](const PointField& value) { return value.name == "time_offset"; }),
      cloud.fields.end());
  config = lidarConfig();
  config.time_field.clear();
  result = convertPointCloud2(cloud, core::MeasurementId(5), config);
  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  EXPECT_EQ(result.value().points()[0].time_offset_ns, 0);
  EXPECT_EQ(result.value().acquisitionBegin(), result.value().header().measurementTime());
  EXPECT_EQ(result.value().acquisitionEnd(), result.value().header().measurementTime());
}

TEST(PointCloudConversion, AcceptsAbsentOptionalIntensityAndRing) {
  PointCloud2 cloud = makeOrganizedCloud(1U, 1U);
  cloud.fields.erase(std::remove_if(cloud.fields.begin(), cloud.fields.end(),
                                    [](const PointField& value) {
                                      return value.name == "intensity" || value.name == "channel";
                                    }),
                     cloud.fields.end());
  setPoint(cloud, 0, 0, 1.0F, 2.0F, 3.0F, 4.0F, 5U, 6U);

  const auto result = convertPointCloud2(cloud, core::MeasurementId(1), lidarConfig());
  ASSERT_TRUE(result.hasValue()) << result.error().detail;
  EXPECT_FALSE(result.stats().has_intensity);
  EXPECT_FALSE(result.stats().has_ring);
  EXPECT_FALSE(result.value().points()[0].intensity.has_value());
  EXPECT_FALSE(result.value().points()[0].ring.has_value());
}

TEST(PointCloudConversion, RejectsMissingDuplicateAndMistypedRequiredFields) {
  PointCloud2 cloud = makeOrganizedCloud(1U, 1U);
  cloud.fields.erase(
      std::remove_if(cloud.fields.begin(), cloud.fields.end(),
                     [](const PointField& value) { return value.name == "time_offset"; }),
      cloud.fields.end());
  auto result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kMissingRequiredField);
  EXPECT_EQ(result.error().field, "time_offset");

  cloud = makeOrganizedCloud(1U, 1U);
  cloud.fields.push_back(field("x", 36, PointField::FLOAT32));
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kDuplicateField);

  cloud = makeOrganizedCloud(1U, 1U);
  for (PointField& value : cloud.fields) {
    if (value.name == "time_offset") {
      value.datatype = PointField::FLOAT32;
    }
  }
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kUnexpectedFieldDatatype);
}

TEST(PointCloudConversion, RejectsMalformedLayoutBeforeReadingPayload) {
  PointCloud2 cloud = makeOrganizedCloud(2U, 1U);
  cloud.row_step = cloud.point_step;
  cloud.data.resize(cloud.row_step);
  auto result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kInvalidRowStep);

  cloud = makeOrganizedCloud(1U, 1U);
  cloud.data.pop_back();
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kInvalidDataSize);

  cloud = makeOrganizedCloud(1U, 1U);
  cloud.fields.push_back(field("outside", 47, PointField::UINT16));
  cloud.fields.push_back(field("overlap", 2, PointField::FLOAT32));
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_TRUE(result.hasValue()) << result.error().detail;

  cloud = makeOrganizedCloud(1U, 1U);
  for (PointField& value : cloud.fields) {
    if (value.name == "x") {
      value.offset = 47;
    }
  }
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kFieldOutOfBounds);

  cloud = makeOrganizedCloud(1U, 1U);
  for (PointField& value : cloud.fields) {
    if (value.name == "y") {
      value.offset = 2;
    }
  }
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kOverlappingFields);
}

TEST(PointCloudConversion, RejectsCloudWithNoFiniteGeometryButReportsDisposition) {
  PointCloud2 cloud = makeOrganizedCloud(2U, 1U);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  setPoint(cloud, 0, 0, nan, 1.0F, 2.0F, 0.0F, 0U, 0U);
  setPoint(cloud, 0, 1, 1.0F, nan, 2.0F, 0.0F, 10U, 0U);

  const auto result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kNoFinitePoints);
  EXPECT_EQ(result.stats().source_points, 2U);
  EXPECT_EQ(result.stats().nonfinite_xyz_points, 2U);
  EXPECT_EQ(result.stats().accepted_points, 0U);
}

TEST(PointCloudConversion, RejectsInvalidHeaderTimeAndFrame) {
  PointCloud2 cloud = makeOrganizedCloud(1U, 1U);
  cloud.header.stamp.nanosec = 1'000'000'000U;
  auto result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kInvalidTimestamp);

  cloud = makeOrganizedCloud(1U, 1U);
  cloud.header.frame_id.clear();
  result = convertPointCloud2(cloud, core::MeasurementId(0), lidarConfig());
  ASSERT_FALSE(result.hasValue());
  EXPECT_EQ(result.error().code, ConversionErrorCode::kEmptyFrameId);
}

}  // namespace
}  // namespace meridian::ros
