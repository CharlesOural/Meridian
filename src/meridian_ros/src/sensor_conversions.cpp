#include "meridian/ros/sensor_conversions.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <optional>
#include <sensor_msgs/msg/point_field.hpp>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace meridian::ros {
namespace {

using sensor_msgs::msg::PointField;

struct FieldRef {
  std::uint32_t offset{};
  std::uint8_t datatype{};
  std::uint32_t count{};
};

[[nodiscard]] ConversionError error(ConversionErrorCode code, std::string detail) {
  return ConversionError{code, std::move(detail)};
}

[[nodiscard]] std::optional<FieldRef> findField(const sensor_msgs::msg::PointCloud2& message,
                                                std::string_view name) {
  for (const auto& field : message.fields) {
    if (field.name == name) {
      return FieldRef{field.offset, field.datatype, field.count};
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::size_t datatypeSize(std::uint8_t datatype) noexcept {
  switch (datatype) {
    case PointField::INT8:
    case PointField::UINT8:
      return 1U;
    case PointField::INT16:
    case PointField::UINT16:
      return 2U;
    case PointField::INT32:
    case PointField::UINT32:
    case PointField::FLOAT32:
      return 4U;
    case PointField::FLOAT64:
      return 8U;
    default:
      return 0U;
  }
}

[[nodiscard]] bool fieldFits(const FieldRef& field, std::uint32_t point_step) noexcept {
  const std::size_t size = datatypeSize(field.datatype);
  return field.count >= 1U && size != 0U &&
         static_cast<std::size_t>(field.offset) + size <= point_step;
}

template <typename Scalar>
[[nodiscard]] Scalar readScalar(const std::uint8_t* bytes, bool wire_big_endian) noexcept {
  std::array<std::uint8_t, sizeof(Scalar)> storage{};
  std::memcpy(storage.data(), bytes, sizeof(Scalar));
  constexpr bool host_big_endian = std::endian::native == std::endian::big;
  if (wire_big_endian != host_big_endian && sizeof(Scalar) > 1U) {
    std::reverse(storage.begin(), storage.end());
  }
  Scalar value{};
  std::memcpy(&value, storage.data(), sizeof(Scalar));
  return value;
}

[[nodiscard]] double readNumber(const std::uint8_t* bytes, std::uint8_t datatype,
                                bool big_endian) noexcept {
  switch (datatype) {
    case PointField::INT8:
      return static_cast<double>(readScalar<std::int8_t>(bytes, big_endian));
    case PointField::UINT8:
      return static_cast<double>(readScalar<std::uint8_t>(bytes, big_endian));
    case PointField::INT16:
      return static_cast<double>(readScalar<std::int16_t>(bytes, big_endian));
    case PointField::UINT16:
      return static_cast<double>(readScalar<std::uint16_t>(bytes, big_endian));
    case PointField::INT32:
      return static_cast<double>(readScalar<std::int32_t>(bytes, big_endian));
    case PointField::UINT32:
      return static_cast<double>(readScalar<std::uint32_t>(bytes, big_endian));
    case PointField::FLOAT32:
      return static_cast<double>(readScalar<float>(bytes, big_endian));
    case PointField::FLOAT64:
      return readScalar<double>(bytes, big_endian);
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

[[nodiscard]] bool integerDatatype(std::uint8_t datatype) noexcept {
  return datatype == PointField::INT8 || datatype == PointField::UINT8 ||
         datatype == PointField::INT16 || datatype == PointField::UINT16 ||
         datatype == PointField::INT32 || datatype == PointField::UINT32;
}

[[nodiscard]] bool floatingDatatype(std::uint8_t datatype) noexcept {
  return datatype == PointField::FLOAT32 || datatype == PointField::FLOAT64;
}

[[nodiscard]] std::int64_t addChecked(std::int64_t lhs, std::int64_t rhs, bool* valid) noexcept {
  if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
    *valid = false;
    return 0;
  }
  return lhs + rhs;
}

[[nodiscard]] core::Result<core::SourceStamp, ConversionError> mapWireStamp(
    const ObservationContext& context, const builtin_interfaces::msg::Time& stamp) {
  const core::FusionTime wire = fromRosTime(stamp);
  auto mapped = core::mapSourceStamp(context.clock, core::RawDeviceTime{wire.nanoseconds},
                                     context.host_arrival_time, context.device_sequence,
                                     context.ingress_sequence);
  if (!mapped) {
    return core::Result<core::SourceStamp, ConversionError>::failure(
        error(ConversionErrorCode::InvalidClockMapping, mapped.error().detail));
  }
  return core::Result<core::SourceStamp, ConversionError>::success(std::move(mapped).value());
}

[[nodiscard]] bool imageDimensionsFit(const sensor_msgs::msg::Image& message,
                                      std::size_t bytes_per_pixel) noexcept {
  const std::size_t packed_row = static_cast<std::size_t>(message.width) * bytes_per_pixel;
  if (message.width == 0U || message.height == 0U || message.step < packed_row) {
    return false;
  }
  if (message.height > std::numeric_limits<std::size_t>::max() / message.step) {
    return false;
  }
  return message.data.size() >= static_cast<std::size_t>(message.height) * message.step;
}

[[nodiscard]] CameraResult makeCameraFrame(const CameraConversionContext& context,
                                           core::SourceStamp source_stamp, std::uint32_t width,
                                           std::uint32_t height, std::uint32_t stride,
                                           std::shared_ptr<const std::vector<std::byte>> pixels) {
  bool timing_valid = true;
  const std::int64_t exposure_midpoint_ns =
      addChecked(source_stamp.fusion_time.nanoseconds,
                 context.stamp_to_exposure_midpoint.nanoseconds, &timing_valid);
  if (!timing_valid) {
    return CameraResult::failure(
        error(ConversionErrorCode::InvalidClockMapping,
              "calibrated camera stamp-to-exposure-midpoint offset overflows fusion time"));
  }

  // Raw device and arrival times remain the unmodified clock metadata. The
  // fusion stamp is the canonical observation time used by downstream sensor
  // fusion, so apply the calibrated signed offset exactly once here and make
  // the explicit camera midpoint agree with it bit-for-bit.
  source_stamp.fusion_time = core::FusionTime{exposure_midpoint_ns};
  core::CameraFrame frame;
  frame.header = context.observation.header;
  frame.id = context.observation.measurement;
  frame.camera = context.camera;
  frame.exposure_midpoint = source_stamp.fusion_time;
  frame.stamp = std::move(source_stamp);
  frame.exposure = context.exposure;
  frame.width = width;
  frame.height = height;
  frame.stride = stride;
  frame.encoding = core::ImageEncoding::Mono8;
  frame.pixels = std::move(pixels);
  return CameraResult::success(std::move(frame));
}

}  // namespace

core::FusionTime fromRosTime(const builtin_interfaces::msg::Time& stamp) noexcept {
  return core::FusionTime{static_cast<std::int64_t>(stamp.sec) * 1'000'000'000LL +
                          static_cast<std::int64_t>(stamp.nanosec)};
}

ImuResult convertImu(const sensor_msgs::msg::Imu& message, const ObservationContext& context) {
  const Eigen::Vector3d acceleration{message.linear_acceleration.x, message.linear_acceleration.y,
                                     message.linear_acceleration.z};
  const Eigen::Vector3d angular_velocity{message.angular_velocity.x, message.angular_velocity.y,
                                         message.angular_velocity.z};
  if (!acceleration.allFinite() || !angular_velocity.allFinite()) {
    return ImuResult::failure(error(ConversionErrorCode::NonFiniteMeasurement,
                                    "IMU acceleration or angular velocity is non-finite"));
  }
  auto mapped_stamp = mapWireStamp(context, message.header.stamp);
  if (!mapped_stamp) {
    return ImuResult::failure(mapped_stamp.error());
  }

  core::ImuSample sample;
  sample.header = context.header;
  sample.id = context.measurement;
  sample.stamp = std::move(mapped_stamp).value();
  sample.specific_force_mps2 = acceleration;
  sample.angular_velocity_radps = angular_velocity;
  return ImuResult::success(std::move(sample));
}

LidarResult convertLidar(const sensor_msgs::msg::PointCloud2& message,
                         const LidarConversionContext& context) {
  if (message.width == 0U || message.height == 0U) {
    return LidarResult::failure(
        error(ConversionErrorCode::EmptyMessage, "point cloud has no points"));
  }
  auto mapped_stamp = mapWireStamp(context.observation, message.header.stamp);
  if (!mapped_stamp) {
    return LidarResult::failure(mapped_stamp.error());
  }
  if (message.point_step == 0U || message.row_step < message.width * message.point_step ||
      static_cast<std::size_t>(message.row_step) * message.height > message.data.size()) {
    return LidarResult::failure(error(ConversionErrorCode::TruncatedData,
                                      "PointCloud2 dimensions exceed row_step or payload size"));
  }

  const auto x = findField(message, "x");
  const auto y = findField(message, "y");
  const auto z = findField(message, "z");
  if (!x || !y || !z) {
    return LidarResult::failure(error(ConversionErrorCode::MissingRequiredField,
                                      "PointCloud2 requires x, y, and z fields"));
  }
  for (const FieldRef* coordinate : {&*x, &*y, &*z}) {
    if (!fieldFits(*coordinate, message.point_step) || !floatingDatatype(coordinate->datatype)) {
      return LidarResult::failure(error(ConversionErrorCode::UnsupportedFieldType,
                                        "x, y, and z must be bounded FLOAT32 or FLOAT64 fields"));
    }
  }

  const auto time_ns = findField(message, "t");
  const auto time_s = findField(message, "time");
  const FieldRef* time = nullptr;
  double time_scale_ns = 1.0;
  if (time_ns && fieldFits(*time_ns, message.point_step) && integerDatatype(time_ns->datatype)) {
    time = &*time_ns;
  } else if (time_s && fieldFits(*time_s, message.point_step) &&
             floatingDatatype(time_s->datatype)) {
    time = &*time_s;
    time_scale_ns = 1.0e9;
  }
  if (time == nullptr) {
    return LidarResult::failure(
        error(ConversionErrorCode::MissingPerPointTime,
              "PointCloud2 requires integer-nanosecond `t` or floating-second `time`"));
  }

  const auto intensity_a = findField(message, "intensity");
  const auto intensity_b = findField(message, "signal");
  const auto intensity_c = findField(message, "reflectivity");
  const FieldRef* intensity = nullptr;
  for (const auto* candidate : {&intensity_a, &intensity_b, &intensity_c}) {
    if (*candidate && fieldFits(**candidate, message.point_step)) {
      intensity = &**candidate;
      break;
    }
  }
  const auto ring_optional = findField(message, "ring");
  const FieldRef* ring = ring_optional && fieldFits(*ring_optional, message.point_step) &&
                                 integerDatatype(ring_optional->datatype)
                             ? &*ring_optional
                             : nullptr;

  const std::size_t input_count = static_cast<std::size_t>(message.width) * message.height;
  std::vector<std::int64_t> offsets;
  offsets.reserve(input_count);
  std::int64_t minimum_offset = std::numeric_limits<std::int64_t>::max();
  std::int64_t maximum_offset = std::numeric_limits<std::int64_t>::min();

  for (std::size_t row = 0; row < message.height; ++row) {
    for (std::size_t column = 0; column < message.width; ++column) {
      const std::size_t base = row * message.row_step + column * message.point_step;
      const double raw = readNumber(message.data.data() + base + time->offset, time->datatype,
                                    message.is_bigendian);
      const double scaled = raw * time_scale_ns;
      if (!std::isfinite(scaled) ||
          scaled < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
          scaled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        return LidarResult::failure(
            error(ConversionErrorCode::InvalidPerPointTime,
                  "per-point time is non-finite or outside int64 nanoseconds"));
      }
      const auto offset = static_cast<std::int64_t>(std::llround(scaled));
      offsets.push_back(offset);
      minimum_offset = std::min(minimum_offset, offset);
      maximum_offset = std::max(maximum_offset, offset);
    }
  }
  if (maximum_offset - minimum_offset > std::numeric_limits<std::int32_t>::max()) {
    return LidarResult::failure(
        error(ConversionErrorCode::InvalidPerPointTime,
              "sweep duration exceeds the int32 normalized-offset API limit"));
  }

  auto points = std::make_shared<core::LidarPoints>();
  points->reserve(input_count);
  std::size_t discarded_non_finite = 0U;
  std::size_t offset_index = 0U;
  for (std::size_t row = 0; row < message.height; ++row) {
    for (std::size_t column = 0; column < message.width; ++column, ++offset_index) {
      const std::size_t base = row * message.row_step + column * message.point_step;
      const auto at = [&](const FieldRef& field) {
        return message.data.data() + base + field.offset;
      };
      const double px = readNumber(at(*x), x->datatype, message.is_bigendian);
      const double py = readNumber(at(*y), y->datatype, message.is_bigendian);
      const double pz = readNumber(at(*z), z->datatype, message.is_bigendian);
      if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
        ++discarded_non_finite;
        continue;
      }

      core::LidarPoint point;
      point.x = static_cast<float>(px);
      point.y = static_cast<float>(py);
      point.z = static_cast<float>(pz);
      if (intensity != nullptr) {
        const double value = readNumber(at(*intensity), intensity->datatype, message.is_bigendian);
        point.intensity = std::isfinite(value) ? static_cast<float>(value) : 0.0F;
      }
      point.time_offset_ns = static_cast<std::int32_t>(offsets[offset_index] - minimum_offset);
      point.source_index = static_cast<std::uint32_t>(offset_index);
      if (ring != nullptr) {
        const double value = readNumber(at(*ring), ring->datatype, message.is_bigendian);
        point.ring = static_cast<std::uint16_t>(
            std::clamp(value, 0.0, static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
      }
      points->push_back(point);
    }
  }
  if (points->empty()) {
    return LidarResult::failure(error(ConversionErrorCode::NonFiniteMeasurement,
                                      "all PointCloud2 returns have non-finite coordinates"));
  }

  bool valid_time = true;
  const auto header_ns = mapped_stamp.value().fusion_time.nanoseconds;
  const auto origin_ns =
      addChecked(header_ns, context.header_to_point_time_origin.nanoseconds, &valid_time);
  const auto start_ns = addChecked(origin_ns, minimum_offset, &valid_time);
  const auto final_return_ns = addChecked(origin_ns, maximum_offset, &valid_time);
  const auto end_ns = addChecked(final_return_ns, 1, &valid_time);
  if (!valid_time || start_ns >= end_ns) {
    return LidarResult::failure(error(ConversionErrorCode::InvalidPerPointTime,
                                      "sweep acquisition interval overflows or is empty"));
  }

  core::LidarSweep sweep;
  sweep.header = context.observation.header;
  sweep.id = context.observation.measurement;
  sweep.lidar = context.lidar;
  sweep.stamp = std::move(mapped_stamp).value();
  sweep.acquisition = core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{end_ns}};
  // Keep the wire raster dimensions and flattened source_index domain even
  // when invalid returns are removed.  A sparse payload no longer has a
  // one-to-one row-major point for every raster slot, so advertising it as
  // organized would make downstream layout validation reject an otherwise
  // usable scan.
  const bool complete_organized_raster = message.height > 1U && discarded_non_finite == 0U;
  sweep.layout = core::LidarLayout{message.width, message.height, complete_organized_raster};
  sweep.points = std::move(points);

  Converted<core::LidarSweep> converted;
  converted.record = std::move(sweep);
  converted.stats =
      ConversionStats{input_count, converted.record.points->size(), discarded_non_finite};
  return LidarResult::success(std::move(converted));
}

CameraResult convertImage(const sensor_msgs::msg::Image& message,
                          const CameraConversionContext& context) {
  if (message.encoding != "mono8") {
    return CameraResult::failure(error(
        ConversionErrorCode::UnsupportedImageEncoding,
        "the canonical visual input is mono8; color/Bayer conversion belongs in the adapter"));
  }
  if (!imageDimensionsFit(message, 1U)) {
    return CameraResult::failure(
        error(ConversionErrorCode::InvalidDimensions,
              "mono8 dimensions, step, and payload size are inconsistent"));
  }
  auto mapped_stamp = mapWireStamp(context.observation, message.header.stamp);
  if (!mapped_stamp) {
    return CameraResult::failure(mapped_stamp.error());
  }

  const std::size_t packed_stride = message.width;
  auto mutable_pixels = std::make_shared<std::vector<std::byte>>(packed_stride * message.height);
  for (std::size_t row = 0; row < message.height; ++row) {
    std::memcpy(mutable_pixels->data() + row * packed_stride,
                message.data.data() + row * message.step, packed_stride);
  }
  std::shared_ptr<const std::vector<std::byte>> pixels = std::move(mutable_pixels);
  return makeCameraFrame(context, std::move(mapped_stamp).value(), message.width, message.height,
                         static_cast<std::uint32_t>(packed_stride), std::move(pixels));
}

CameraResult convertCompressedImage(const sensor_msgs::msg::CompressedImage& message,
                                    const CameraConversionContext& context) {
  if (message.data.empty() ||
      message.data.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return CameraResult::failure(
        error(ConversionErrorCode::EmptyMessage,
              "compressed image payload is empty or exceeds decoder limits"));
  }
  auto mapped_stamp = mapWireStamp(context.observation, message.header.stamp);
  if (!mapped_stamp) {
    return CameraResult::failure(mapped_stamp.error());
  }
  const cv::Mat encoded(1, static_cast<int>(message.data.size()), CV_8UC1,
                        const_cast<std::uint8_t*>(message.data.data()));
  const cv::Mat gray = cv::imdecode(encoded, cv::IMREAD_GRAYSCALE);
  if (gray.empty() || gray.type() != CV_8UC1) {
    return CameraResult::failure(error(ConversionErrorCode::ImageDecodeFailed,
                                       "compressed payload did not decode to an 8-bit image"));
  }

  const std::size_t stride = static_cast<std::size_t>(gray.cols);
  auto mutable_pixels =
      std::make_shared<std::vector<std::byte>>(stride * static_cast<std::size_t>(gray.rows));
  for (int row = 0; row < gray.rows; ++row) {
    std::memcpy(mutable_pixels->data() + static_cast<std::size_t>(row) * stride, gray.ptr(row),
                stride);
  }
  std::shared_ptr<const std::vector<std::byte>> pixels = std::move(mutable_pixels);
  return makeCameraFrame(
      context, std::move(mapped_stamp).value(), static_cast<std::uint32_t>(gray.cols),
      static_cast<std::uint32_t>(gray.rows), static_cast<std::uint32_t>(stride), std::move(pixels));
}

GnssResult convertGnss(const sensor_msgs::msg::NavSatFix& message,
                       const ObservationContext& context) {
  if (message.status.status < sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX ||
      message.status.status > sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX) {
    return GnssResult::failure(
        error(ConversionErrorCode::UnsupportedFieldType,
              "GNSS status value is outside the NavSatStatus wire enumeration"));
  }
  if (message.status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
    return GnssResult::failure(
        error(ConversionErrorCode::UnavailableMeasurement,
              "GNSS message explicitly reports that no position fix is available"));
  }
  if (message.position_covariance_type == sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
    return GnssResult::failure(
        error(ConversionErrorCode::UnknownCovariance,
              "GNSS covariance is unknown; a zero wire matrix is not zero uncertainty"));
  }
  if (message.position_covariance_type > sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN) {
    return GnssResult::failure(
        error(ConversionErrorCode::UnsupportedFieldType,
              "GNSS covariance type is outside the NavSatFix wire enumeration"));
  }

  core::GnssSolutionType solution = core::GnssSolutionType::Unknown;
  switch (message.status.status) {
    case sensor_msgs::msg::NavSatStatus::STATUS_FIX:
      solution = core::GnssSolutionType::Autonomous;
      break;
    case sensor_msgs::msg::NavSatStatus::STATUS_SBAS_FIX:
      solution = core::GnssSolutionType::SbasAugmented;
      break;
    case sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX:
      solution = core::GnssSolutionType::GbasAugmented;
      break;
    default:
      return GnssResult::failure(
          error(ConversionErrorCode::UnsupportedFieldType,
                "GNSS fix status cannot be represented by the generic NavSatFix adapter"));
  }

  core::PositionCovarianceSource covariance_source =
      core::PositionCovarianceSource::ReceiverApproximation;
  switch (message.position_covariance_type) {
    case sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_APPROXIMATED:
      covariance_source = core::PositionCovarianceSource::ReceiverApproximation;
      break;
    case sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN:
      covariance_source = core::PositionCovarianceSource::ReceiverDiagonal;
      break;
    case sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN:
      covariance_source = core::PositionCovarianceSource::ReceiverFull;
      break;
    default:
      return GnssResult::failure(
          error(ConversionErrorCode::UnsupportedFieldType,
                "GNSS covariance source cannot be represented by the domain API"));
  }

  auto position = core::GeodeticPosition::fromWgs84Degrees(message.latitude, message.longitude,
                                                           message.altitude);
  if (!position) {
    return GnssResult::failure(
        error(ConversionErrorCode::NonFiniteMeasurement, position.error().detail));
  }

  std::optional<core::GnssServiceSet> services;
  if (message.status.service != 0U) {
    auto service_set = core::GnssServiceSet::fromMask(message.status.service);
    if (!service_set) {
      return GnssResult::failure(
          error(ConversionErrorCode::UnsupportedFieldType, service_set.error().detail));
    }
    services.emplace(std::move(service_set).value());
  }

  auto mapped_stamp = mapWireStamp(context, message.header.stamp);
  if (!mapped_stamp) {
    return GnssResult::failure(mapped_stamp.error());
  }

  Eigen::Matrix3d covariance;
  for (Eigen::Index row = 0; row < 3; ++row) {
    for (Eigen::Index column = 0; column < 3; ++column) {
      covariance(row, column) =
          message.position_covariance[static_cast<std::size_t>(row * 3 + column)];
    }
  }
  auto domain_covariance = core::PositionCovarianceEnu::fromMatrix(covariance, covariance_source);
  if (!domain_covariance) {
    return GnssResult::failure(
        error(ConversionErrorCode::InvalidCovariance, domain_covariance.error().detail));
  }

  core::GnssObservation observation{
      context.header,
      core::GnssObservationId{context.measurement.value()},
      std::move(mapped_stamp).value(),
      std::move(position).value(),
      std::move(domain_covariance).value(),
      solution,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      core::GnssStatus{core::GnssFixAvailability::Available, core::GnssIntegrityStatus::Unknown,
                       core::GnssCorrectionStatus::Unknown,
                       core::GnssStatusSource::GenericNavSatFix, std::move(services)}};
  const auto validated = core::validateGnssObservation(observation);
  if (!validated) {
    return GnssResult::failure(
        error(ConversionErrorCode::MissingRequiredField, validated.error().detail));
  }
  return GnssResult::success(std::move(observation));
}

}  // namespace meridian::ros
