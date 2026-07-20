#include "meridian/ros/conversion.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sensor_msgs/msg/point_field.hpp>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "meridian/core/time.hpp"

namespace meridian::ros {
namespace {

using sensor_msgs::msg::PointField;

template <typename T, typename Stats = EmptyConversionStats>
ConversionResult<T, Stats> fail(ConversionErrorCode code, std::string field, std::string detail,
                                Stats stats = {}) {
  return ConversionResult<T, Stats>::failure(
      ConversionError{.code = code, .field = std::move(field), .detail = std::move(detail)},
      std::move(stats));
}

std::optional<core::TimeNs> messageTime(const builtin_interfaces::msg::Time& stamp) {
  return core::TimeNs::fromSecNanosec(stamp.sec, stamp.nanosec);
}

std::optional<std::size_t> datatypeSize(std::uint8_t datatype) noexcept {
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
      return std::nullopt;
  }
}

std::string describeDatatype(std::uint8_t datatype) {
  switch (datatype) {
    case PointField::INT8:
      return "INT8";
    case PointField::UINT8:
      return "UINT8";
    case PointField::INT16:
      return "INT16";
    case PointField::UINT16:
      return "UINT16";
    case PointField::INT32:
      return "INT32";
    case PointField::UINT32:
      return "UINT32";
    case PointField::FLOAT32:
      return "FLOAT32";
    case PointField::FLOAT64:
      return "FLOAT64";
    default:
      return "UNKNOWN(" + std::to_string(datatype) + ")";
  }
}

template <typename T>
T readScalar(std::span<const std::uint8_t> data, std::size_t offset, bool source_big_endian) {
  std::array<std::uint8_t, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), data.data() + offset, sizeof(T));
  constexpr bool kHostBigEndian = std::endian::native == std::endian::big;
  if (source_big_endian != kHostBigEndian) {
    std::reverse(bytes.begin(), bytes.end());
  }
  T value{};
  std::memcpy(&value, bytes.data(), sizeof(T));
  return value;
}

long double readNumericScalar(std::span<const std::uint8_t> data, std::size_t offset,
                              std::uint8_t datatype, bool source_big_endian) {
  switch (datatype) {
    case PointField::INT8:
      return readScalar<std::int8_t>(data, offset, source_big_endian);
    case PointField::UINT8:
      return readScalar<std::uint8_t>(data, offset, source_big_endian);
    case PointField::INT16:
      return readScalar<std::int16_t>(data, offset, source_big_endian);
    case PointField::UINT16:
      return readScalar<std::uint16_t>(data, offset, source_big_endian);
    case PointField::INT32:
      return readScalar<std::int32_t>(data, offset, source_big_endian);
    case PointField::UINT32:
      return readScalar<std::uint32_t>(data, offset, source_big_endian);
    case PointField::FLOAT32:
      return readScalar<float>(data, offset, source_big_endian);
    case PointField::FLOAT64:
      return readScalar<double>(data, offset, source_big_endian);
    default:
      return std::numeric_limits<long double>::quiet_NaN();
  }
}

std::optional<std::int64_t> scaledNanoseconds(long double value, double scale) noexcept {
  const long double scaled = value * static_cast<long double>(scale);
  if (!std::isfinite(scaled)) {
    return std::nullopt;
  }
  const long double rounded = std::round(scaled);
  if (rounded < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
      rounded > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(rounded);
}

std::optional<std::int64_t> pointTimeOffset(long double raw_time,
                                            const LidarConversionConfig& config,
                                            core::TimeNs header_time) noexcept {
  const auto nanoseconds = scaledNanoseconds(raw_time, config.time_scale_to_nanoseconds);
  if (!nanoseconds.has_value()) {
    return std::nullopt;
  }
  if (config.time_reference == PointTimeReference::kOffsetFromHeader) {
    return nanoseconds;
  }
  return core::TimeNs::checkedDifference(core::TimeNs(*nanoseconds), header_time);
}

struct FieldExtent final {
  const PointField* field;
  std::size_t begin;
  std::size_t end;
};

const PointField* findField(const std::unordered_map<std::string, const PointField*>& fields,
                            const std::string& name) {
  const auto found = fields.find(name);
  return found == fields.end() ? nullptr : found->second;
}

std::optional<ConversionError> validateConfiguredField(const PointField* field,
                                                       std::string_view name,
                                                       std::uint8_t expected_datatype,
                                                       bool required) {
  if (field == nullptr) {
    if (required) {
      return ConversionError{.code = ConversionErrorCode::kMissingRequiredField,
                             .field = std::string(name),
                             .detail = "required point field is absent"};
    }
    return std::nullopt;
  }
  if (field->count != 1U) {
    return ConversionError{.code = ConversionErrorCode::kInvalidFieldCount,
                           .field = field->name,
                           .detail = "configured scalar field must have count=1"};
  }
  if (field->datatype != expected_datatype) {
    return ConversionError{
        .code = ConversionErrorCode::kUnexpectedFieldDatatype,
        .field = field->name,
        .detail = "expected " + describeDatatype(expected_datatype) + ", received " +
                  describeDatatype(field->datatype),
    };
  }
  return std::nullopt;
}

}  // namespace

const char* toString(ConversionErrorCode code) noexcept {
  switch (code) {
    case ConversionErrorCode::kInvalidTimestamp:
      return "invalid_timestamp";
    case ConversionErrorCode::kEmptyFrameId:
      return "empty_frame_id";
    case ConversionErrorCode::kUnexpectedFrameId:
      return "unexpected_frame_id";
    case ConversionErrorCode::kNonFiniteImu:
      return "nonfinite_imu";
    case ConversionErrorCode::kEmptyCloud:
      return "empty_cloud";
    case ConversionErrorCode::kDimensionOverflow:
      return "dimension_overflow";
    case ConversionErrorCode::kInvalidPointStep:
      return "invalid_point_step";
    case ConversionErrorCode::kInvalidRowStep:
      return "invalid_row_step";
    case ConversionErrorCode::kInvalidDataSize:
      return "invalid_data_size";
    case ConversionErrorCode::kDuplicateField:
      return "duplicate_field";
    case ConversionErrorCode::kInvalidFieldCount:
      return "invalid_field_count";
    case ConversionErrorCode::kUnsupportedFieldDatatype:
      return "unsupported_field_datatype";
    case ConversionErrorCode::kFieldOutOfBounds:
      return "field_out_of_bounds";
    case ConversionErrorCode::kOverlappingFields:
      return "overlapping_fields";
    case ConversionErrorCode::kMissingRequiredField:
      return "missing_required_field";
    case ConversionErrorCode::kUnexpectedFieldDatatype:
      return "unexpected_field_datatype";
    case ConversionErrorCode::kInvalidPointTime:
      return "invalid_point_time";
    case ConversionErrorCode::kTimestampOverflow:
      return "timestamp_overflow";
    case ConversionErrorCode::kNoFinitePoints:
      return "no_finite_points";
  }
  return "unknown_conversion_error";
}

ConversionResult<core::ImuSample> convertImu(const sensor_msgs::msg::Imu& message,
                                             core::MeasurementId measurement_id,
                                             const ImuConversionConfig& config) {
  const auto timestamp = messageTime(message.header.stamp);
  if (!timestamp.has_value()) {
    return fail<core::ImuSample>(ConversionErrorCode::kInvalidTimestamp, "header.stamp",
                                 "seconds/nanoseconds do not form a signed int64 nanosecond time");
  }
  if (message.header.frame_id.empty()) {
    return fail<core::ImuSample>(ConversionErrorCode::kEmptyFrameId, "header.frame_id",
                                 "IMU frame id is empty");
  }
  if (message.header.frame_id != config.expected_frame_id) {
    return fail<core::ImuSample>(
        ConversionErrorCode::kUnexpectedFrameId, "header.frame_id",
        "expected '" + config.expected_frame_id + "', received '" + message.header.frame_id + "'");
  }
  const core::Vec3d angular_velocity{message.angular_velocity.x, message.angular_velocity.y,
                                     message.angular_velocity.z};
  const core::Vec3d specific_force{message.linear_acceleration.x, message.linear_acceleration.y,
                                   message.linear_acceleration.z};
  if (!angular_velocity.isFinite()) {
    return fail<core::ImuSample>(ConversionErrorCode::kNonFiniteImu, "angular_velocity",
                                 "angular velocity contains NaN or infinity");
  }
  if (!specific_force.isFinite()) {
    return fail<core::ImuSample>(ConversionErrorCode::kNonFiniteImu, "linear_acceleration",
                                 "specific force contains NaN or infinity");
  }

  core::ObservationHeader header(config.sensor_id, config.calibration_id, measurement_id,
                                 *timestamp, message.header.frame_id);
  return ConversionResult<core::ImuSample>::success(
      core::ImuSample(std::move(header), angular_velocity, specific_force));
}

ConversionResult<core::LidarSweep, PointCloudConversionStats> convertPointCloud2(
    const sensor_msgs::msg::PointCloud2& message, core::MeasurementId measurement_id,
    const LidarConversionConfig& config) {
  PointCloudConversionStats stats{};
  const auto timestamp = messageTime(message.header.stamp);
  if (!timestamp.has_value()) {
    return fail<core::LidarSweep>(ConversionErrorCode::kInvalidTimestamp, "header.stamp",
                                  "invalid signed int64 nanosecond time", stats);
  }
  if (message.header.frame_id.empty()) {
    return fail<core::LidarSweep>(ConversionErrorCode::kEmptyFrameId, "header.frame_id",
                                  "LiDAR frame id is empty", stats);
  }
  if (message.header.frame_id != config.expected_frame_id) {
    return fail<core::LidarSweep>(
        ConversionErrorCode::kUnexpectedFrameId, "header.frame_id",
        "expected '" + config.expected_frame_id + "', received '" + message.header.frame_id + "'",
        stats);
  }
  if (!config.time_field.empty() && (!std::isfinite(config.time_scale_to_nanoseconds) ||
                                     config.time_scale_to_nanoseconds <= 0.0)) {
    return fail<core::LidarSweep>(ConversionErrorCode::kInvalidPointTime, config.time_field,
                                  "point-time scale to nanoseconds must be finite and positive",
                                  stats);
  }
  if (message.width == 0U || message.height == 0U) {
    return fail<core::LidarSweep>(ConversionErrorCode::kEmptyCloud, "width/height",
                                  "point cloud dimensions must both be nonzero", stats);
  }

  const std::uint64_t point_count = static_cast<std::uint64_t>(message.width) * message.height;
  if (point_count > std::numeric_limits<std::uint32_t>::max() ||
      point_count > std::numeric_limits<std::size_t>::max()) {
    return fail<core::LidarSweep>(ConversionErrorCode::kDimensionOverflow, "width/height",
                                  "point count exceeds supported source-index range", stats);
  }
  stats.source_points = point_count;

  if (message.point_step == 0U) {
    return fail<core::LidarSweep>(ConversionErrorCode::kInvalidPointStep, "point_step",
                                  "point_step must be nonzero", stats);
  }
  const std::uint64_t packed_row_bytes =
      static_cast<std::uint64_t>(message.width) * message.point_step;
  if (packed_row_bytes > std::numeric_limits<std::uint32_t>::max() ||
      message.row_step < packed_row_bytes) {
    return fail<core::LidarSweep>(ConversionErrorCode::kInvalidRowStep, "row_step",
                                  "row_step is smaller than width * point_step", stats);
  }
  stats.row_padding_bytes =
      static_cast<std::uint64_t>(message.row_step - packed_row_bytes) * message.height;

  const std::uint64_t expected_data_bytes =
      static_cast<std::uint64_t>(message.row_step) * message.height;
  if (expected_data_bytes > std::numeric_limits<std::size_t>::max() ||
      message.data.size() != expected_data_bytes) {
    return fail<core::LidarSweep>(ConversionErrorCode::kInvalidDataSize, "data",
                                  "data size must equal height * row_step (expected " +
                                      std::to_string(expected_data_bytes) + ", received " +
                                      std::to_string(message.data.size()) + ")",
                                  stats);
  }

  std::unordered_map<std::string, const PointField*> fields;
  fields.reserve(6U);
  const auto is_configured = [&config](const std::string& name) {
    return name == config.x_field || name == config.y_field || name == config.z_field ||
           (!config.time_field.empty() && name == config.time_field) ||
           (!config.intensity_field.empty() && name == config.intensity_field) ||
           (!config.ring_field.empty() && name == config.ring_field);
  };
  for (const PointField& field : message.fields) {
    if (!is_configured(field.name)) {
      continue;
    }
    if (!fields.emplace(field.name, &field).second) {
      return fail<core::LidarSweep>(ConversionErrorCode::kDuplicateField, field.name,
                                    "configured point field name occurs more than once", stats);
    }
  }

  const PointField* x_field = findField(fields, config.x_field);
  const PointField* y_field = findField(fields, config.y_field);
  const PointField* z_field = findField(fields, config.z_field);
  const PointField* time_field =
      config.time_field.empty() ? nullptr : findField(fields, config.time_field);
  const PointField* intensity_field =
      config.intensity_field.empty() ? nullptr : findField(fields, config.intensity_field);
  const PointField* ring_field =
      config.ring_field.empty() ? nullptr : findField(fields, config.ring_field);

  for (const auto& [field, name, datatype] :
       std::array{std::tuple{x_field, std::string_view(config.x_field), config.x_datatype},
                  std::tuple{y_field, std::string_view(config.y_field), config.y_datatype},
                  std::tuple{z_field, std::string_view(config.z_field), config.z_datatype}}) {
    if (const auto error = validateConfiguredField(field, name, datatype, true); error) {
      return ConversionResult<core::LidarSweep, PointCloudConversionStats>::failure(*error, stats);
    }
  }
  if (const auto error = validateConfiguredField(time_field, config.time_field,
                                                 config.time_datatype, !config.time_field.empty());
      error) {
    return ConversionResult<core::LidarSweep, PointCloudConversionStats>::failure(*error, stats);
  }
  if (const auto error = validateConfiguredField(intensity_field, config.intensity_field,
                                                 config.intensity_datatype, false);
      error) {
    return ConversionResult<core::LidarSweep, PointCloudConversionStats>::failure(*error, stats);
  }
  if (const auto error =
          validateConfiguredField(ring_field, config.ring_field, config.ring_datatype, false);
      error) {
    return ConversionResult<core::LidarSweep, PointCloudConversionStats>::failure(*error, stats);
  }

  std::vector<FieldExtent> extents;
  extents.reserve(6U);
  for (const PointField* field :
       std::array{x_field, y_field, z_field, time_field, intensity_field, ring_field}) {
    if (field == nullptr) {
      continue;
    }
    const auto scalar_size = datatypeSize(field->datatype);
    if (!scalar_size.has_value()) {
      return fail<core::LidarSweep>(
          ConversionErrorCode::kUnsupportedFieldDatatype, field->name,
          "unknown configured PointField datatype " + std::to_string(field->datatype), stats);
    }
    const std::uint64_t field_end = static_cast<std::uint64_t>(field->offset) + *scalar_size;
    if (field_end > message.point_step) {
      return fail<core::LidarSweep>(ConversionErrorCode::kFieldOutOfBounds, field->name,
                                    "configured field extent exceeds point_step", stats);
    }
    extents.push_back(FieldExtent{
        .field = field, .begin = field->offset, .end = static_cast<std::size_t>(field_end)});
  }
  std::sort(extents.begin(), extents.end(), [](const FieldExtent& lhs, const FieldExtent& rhs) {
    return lhs.begin < rhs.begin || (lhs.begin == rhs.begin && lhs.end < rhs.end);
  });
  for (std::size_t index = 1; index < extents.size(); ++index) {
    if (extents[index].begin < extents[index - 1U].end) {
      return fail<core::LidarSweep>(
          ConversionErrorCode::kOverlappingFields, extents[index].field->name,
          "configured field overlaps '" + extents[index - 1U].field->name + "'", stats);
    }
  }
  stats.has_intensity = intensity_field != nullptr;
  stats.has_ring = ring_field != nullptr;

  const std::span<const std::uint8_t> data(message.data.data(), message.data.size());
  std::vector<core::LidarPoint> points;
  points.reserve(static_cast<std::size_t>(point_count));
  std::optional<std::int64_t> previous_time;
  bool first_time = true;

  for (std::uint32_t row = 0; row < message.height; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * message.row_step;
    for (std::uint32_t column = 0; column < message.width; ++column) {
      const std::size_t point_offset =
          row_offset + static_cast<std::size_t>(column) * message.point_step;
      const long double raw_x = readNumericScalar(data, point_offset + x_field->offset,
                                                  x_field->datatype, message.is_bigendian);
      const long double raw_y = readNumericScalar(data, point_offset + y_field->offset,
                                                  y_field->datatype, message.is_bigendian);
      const long double raw_z = readNumericScalar(data, point_offset + z_field->offset,
                                                  z_field->datatype, message.is_bigendian);
      const float x = static_cast<float>(raw_x);
      const float y = static_cast<float>(raw_y);
      const float z = static_cast<float>(raw_z);

      std::int64_t time_offset = 0;
      if (time_field != nullptr) {
        const long double raw_time = readNumericScalar(data, point_offset + time_field->offset,
                                                       time_field->datatype, message.is_bigendian);
        const auto converted_time = pointTimeOffset(raw_time, config, *timestamp);
        if (!converted_time.has_value()) {
          return fail<core::LidarSweep>(
              ConversionErrorCode::kInvalidPointTime, config.time_field,
              "point time is non-finite or outside the signed nanosecond range", stats);
        }
        time_offset = *converted_time;
      }

      if (previous_time.has_value() && time_offset < *previous_time) {
        ++stats.flattened_time_regressions;
      }
      previous_time = time_offset;
      if (first_time) {
        stats.minimum_time_offset_ns = time_offset;
        stats.maximum_time_offset_ns = time_offset;
        first_time = false;
      } else {
        stats.minimum_time_offset_ns = std::min(stats.minimum_time_offset_ns, time_offset);
        stats.maximum_time_offset_ns = std::max(stats.maximum_time_offset_ns, time_offset);
      }

      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        ++stats.nonfinite_xyz_points;
        continue;
      }
      if (x == 0.0F && y == 0.0F && z == 0.0F) {
        ++stats.zero_xyz_points;
      }

      std::optional<float> intensity;
      if (intensity_field != nullptr) {
        const long double raw_intensity =
            readNumericScalar(data, point_offset + intensity_field->offset,
                              intensity_field->datatype, message.is_bigendian);
        const float converted_intensity = static_cast<float>(raw_intensity);
        if (std::isfinite(raw_intensity) && std::isfinite(converted_intensity)) {
          intensity = converted_intensity;
        } else {
          ++stats.nonfinite_intensity_points;
        }
      }

      std::optional<std::uint16_t> ring;
      if (ring_field != nullptr) {
        const long double raw_ring = readNumericScalar(data, point_offset + ring_field->offset,
                                                       ring_field->datatype, message.is_bigendian);
        if (std::isfinite(raw_ring) && raw_ring >= 0.0L &&
            raw_ring <= static_cast<long double>(std::numeric_limits<std::uint16_t>::max()) &&
            std::trunc(raw_ring) == raw_ring) {
          ring = static_cast<std::uint16_t>(raw_ring);
        }
      }
      const std::uint64_t source_index = static_cast<std::uint64_t>(row) * message.width + column;
      points.push_back(core::LidarPoint{
          .x = x,
          .y = y,
          .z = z,
          .time_offset_ns = time_offset,
          .source_index = static_cast<std::uint32_t>(source_index),
          .intensity = intensity,
          .ring = ring,
      });
    }
  }
  stats.accepted_points = points.size();
  if (points.empty()) {
    return fail<core::LidarSweep>(ConversionErrorCode::kNoFinitePoints, "x/y/z",
                                  "cloud contains no point with finite coordinates", stats);
  }

  const auto acquisition_begin = core::TimeNs::checkedAdd(*timestamp, stats.minimum_time_offset_ns);
  const auto acquisition_end = core::TimeNs::checkedAdd(*timestamp, stats.maximum_time_offset_ns);
  if (!acquisition_begin.has_value() || !acquisition_end.has_value()) {
    return fail<core::LidarSweep>(ConversionErrorCode::kTimestampOverflow, config.time_field,
                                  "header stamp plus maximum point offset overflows TimeNs", stats);
  }

  core::ObservationHeader header(config.sensor_id, config.calibration_id, measurement_id,
                                 *timestamp, message.header.frame_id);
  core::LidarSweep sweep(std::move(header), *acquisition_begin, *acquisition_end,
                         std::move(points));
  return ConversionResult<core::LidarSweep, PointCloudConversionStats>::success(std::move(sweep),
                                                                                stats);
}

}  // namespace meridian::ros
