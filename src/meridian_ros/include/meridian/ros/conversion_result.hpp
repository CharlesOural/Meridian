#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace meridian::ros {

enum class ConversionErrorCode : std::uint8_t {
  kInvalidTimestamp,
  kEmptyFrameId,
  kUnexpectedFrameId,
  kNonFiniteImu,
  kEmptyCloud,
  kDimensionOverflow,
  kInvalidPointStep,
  kInvalidRowStep,
  kInvalidDataSize,
  kDuplicateField,
  kInvalidFieldCount,
  kUnsupportedFieldDatatype,
  kFieldOutOfBounds,
  kOverlappingFields,
  kMissingRequiredField,
  kUnexpectedFieldDatatype,
  kInvalidPointTime,
  kTimestampOverflow,
  kNoFinitePoints,
};

[[nodiscard]] const char* toString(ConversionErrorCode code) noexcept;

struct ConversionError final {
  ConversionErrorCode code;
  std::string field;
  std::string detail;
};

struct EmptyConversionStats final {};

struct PointCloudConversionStats final {
  std::uint64_t source_points{};
  std::uint64_t accepted_points{};
  std::uint64_t nonfinite_xyz_points{};
  std::uint64_t zero_xyz_points{};
  std::uint64_t nonfinite_intensity_points{};
  std::uint64_t flattened_time_regressions{};
  std::uint64_t row_padding_bytes{};
  std::int64_t minimum_time_offset_ns{};
  std::int64_t maximum_time_offset_ns{};
  bool has_intensity{};
  bool has_ring{};
};

template <typename T, typename Stats = EmptyConversionStats>
class ConversionResult final {
public:
  [[nodiscard]] static ConversionResult success(T value, Stats stats = {}) {
    return ConversionResult(std::move(value), std::move(stats));
  }

  [[nodiscard]] static ConversionResult failure(ConversionError error, Stats stats = {}) {
    return ConversionResult(std::move(error), std::move(stats));
  }

  [[nodiscard]] bool hasValue() const noexcept { return std::holds_alternative<T>(result_); }
  [[nodiscard]] explicit operator bool() const noexcept { return hasValue(); }

  [[nodiscard]] T& value() & { return std::get<T>(result_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(result_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(result_)); }

  [[nodiscard]] const ConversionError& error() const { return std::get<ConversionError>(result_); }
  [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
  explicit ConversionResult(T value, Stats stats)
      : result_(std::move(value)), stats_(std::move(stats)) {}
  explicit ConversionResult(ConversionError error, Stats stats)
      : result_(std::move(error)), stats_(std::move(stats)) {}

  std::variant<T, ConversionError> result_;
  Stats stats_;
};

}  // namespace meridian::ros
