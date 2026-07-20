#pragma once

#include <cstdint>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "meridian/core/ids.hpp"
#include "meridian/core/observations.hpp"
#include "meridian/ros/conversion_result.hpp"

namespace meridian::ros {

struct ImuConversionConfig final {
  core::SensorId sensor_id;
  core::CalibrationId calibration_id;
  std::string expected_frame_id;
};

enum class PointTimeReference : std::uint8_t {
  kOffsetFromHeader,
  kAbsolute,
};

struct LidarConversionConfig final {
  core::SensorId sensor_id;
  core::CalibrationId calibration_id;
  std::string expected_frame_id;
  std::string x_field;
  std::string y_field;
  std::string z_field;
  std::string time_field;
  std::string intensity_field;
  std::string ring_field;
  std::uint8_t x_datatype;
  std::uint8_t y_datatype;
  std::uint8_t z_datatype;
  std::uint8_t time_datatype;
  std::uint8_t intensity_datatype;
  std::uint8_t ring_datatype;
  double time_scale_to_nanoseconds;
  PointTimeReference time_reference;
};

[[nodiscard]] ConversionResult<core::ImuSample> convertImu(const sensor_msgs::msg::Imu& message,
                                                           core::MeasurementId measurement_id,
                                                           const ImuConversionConfig& config);

[[nodiscard]] ConversionResult<core::LidarSweep, PointCloudConversionStats> convertPointCloud2(
    const sensor_msgs::msg::PointCloud2& message, core::MeasurementId measurement_id,
    const LidarConversionConfig& config);

}  // namespace meridian::ros
