#pragma once

#include <builtin_interfaces/msg/time.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>

#include "meridian/core/api.hpp"

namespace meridian::ros {

enum class ConversionErrorCode {
  EmptyMessage,
  InvalidDimensions,
  TruncatedData,
  MissingRequiredField,
  UnsupportedFieldType,
  MissingPerPointTime,
  InvalidPerPointTime,
  UnsupportedImageEncoding,
  ImageDecodeFailed,
  NonFiniteMeasurement,
  UnavailableMeasurement,
  UnknownCovariance,
  InvalidCovariance,
  InvalidClockMapping,
};

struct ConversionError {
  ConversionErrorCode code{};
  std::string detail;
};

struct ObservationContext {
  core::RecordHeader header;
  core::AffineClockModel clock;
  core::ArrivalTime host_arrival_time;
  std::optional<core::SourceSequence> device_sequence;
  core::IngressSequence ingress_sequence;
  core::MeasurementId measurement;
};

struct LidarConversionContext {
  ObservationContext observation;
  core::LidarId lidar;
  // Offset from the ROS header stamp to the origin of the per-point time field.
  // Ouster ROS messages use zero here: `t` is nanoseconds since scan start and
  // the header is the scan-start stamp.
  core::Duration header_to_point_time_origin{};
};

struct CameraConversionContext {
  ObservationContext observation;
  core::CameraId camera;
  // Calibrated signed offset from the clock-mapped wire stamp to exposure
  // midpoint. The adapter preserves raw/arrival metadata and applies this once
  // to both SourceStamp::fusion_time and CameraFrame::exposure_midpoint.
  core::Duration stamp_to_exposure_midpoint{};
  core::Duration exposure{};
};

struct ConversionStats {
  std::size_t input_elements{};
  std::size_t output_elements{};
  std::size_t discarded_non_finite{};
};

template <typename Record>
struct Converted {
  Record record;
  ConversionStats stats;
};

using ImuResult = core::Result<core::ImuSample, ConversionError>;
using LidarResult = core::Result<Converted<core::LidarSweep>, ConversionError>;
using CameraResult = core::Result<core::CameraFrame, ConversionError>;
using GnssResult = core::Result<core::GnssObservation, ConversionError>;

[[nodiscard]] core::FusionTime fromRosTime(const builtin_interfaces::msg::Time& stamp) noexcept;

[[nodiscard]] ImuResult convertImu(const sensor_msgs::msg::Imu& message,
                                   const ObservationContext& context);

// Requires x/y/z and a per-point relative time field. Ouster's `t` (integer
// nanoseconds) and the common `time` (floating-point seconds) conventions are
// accepted. Wire row padding and endianness are honored.
[[nodiscard]] LidarResult convertLidar(const sensor_msgs::msg::PointCloud2& message,
                                       const LidarConversionContext& context);

[[nodiscard]] CameraResult convertImage(const sensor_msgs::msg::Image& message,
                                        const CameraConversionContext& context);

// JPEG/PNG payloads are decoded to tightly packed mono8, which is the visual
// frontend's canonical input representation.
[[nodiscard]] CameraResult convertCompressedImage(const sensor_msgs::msg::CompressedImage& message,
                                                  const CameraConversionContext& context);

[[nodiscard]] GnssResult convertGnss(const sensor_msgs::msg::NavSatFix& message,
                                     const ObservationContext& context);

}  // namespace meridian::ros
