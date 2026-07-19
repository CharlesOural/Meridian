#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "meridian/core/blob.hpp"
#include "meridian/core/geometry.hpp"
#include "meridian/core/result.hpp"

namespace meridian::core {

enum class CanonicalEncoderState {
  Ready,
  Finalized,
  Failed,
};

enum class CanonicalEncodingError {
  None,
  InvalidMaximumBytes,
  InvalidDomainTag,
  InvalidSchemaVersion,
  OutputLimitExceeded,
  NonFiniteFloatingPoint,
  InvalidVectorShape,
  DimensionOverflow,
  AlreadyFinalized,
  AllocationFailure,
};

// A finalized canonical byte sequence. Both the byte storage and digest have
// no mutating API; copies share the same immutable byte allocation.
class CanonicalByteSequence {
public:
  [[nodiscard]] const ImmutableBytes& storage() const noexcept { return bytes_; }
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return std::span<const std::byte>(*bytes_);
  }
  [[nodiscard]] const ContentHash& digest() const noexcept { return digest_; }

private:
  CanonicalByteSequence(ImmutableBytes bytes, ContentHash digest)
      : bytes_(std::move(bytes)), digest_(digest) {}

  ImmutableBytes bytes_;
  ContentHash digest_{};

  friend class CanonicalEncoder;
};

// Bounded canonical wire encoder.
//
// Every sequence begins with:
//   u64 domain_tag_bytes | domain_tag | u32 schema_version
// All integers are fixed-width big-endian; signed values use their canonical
// modulo-2^N (two's-complement) representation. Byte, string, and hash fields
// carry a u64 byte length. Doubles use their IEEE-754 binary64 bits after
// -0 -> +0 normalization. Eigen fields carry u64 dimensions and are traversed
// row major; Pose3 is encoded through its finite 4x4 matrix representation.
class CanonicalEncoder {
public:
  [[nodiscard]] static Result<CanonicalEncoder, CanonicalEncodingError> create(
      std::string_view domain_tag, std::uint32_t schema_version,
      std::uint64_t maximum_output_bytes);

  CanonicalEncoder(const CanonicalEncoder&) = delete;
  CanonicalEncoder& operator=(const CanonicalEncoder&) = delete;
  CanonicalEncoder(CanonicalEncoder&&) noexcept = default;
  CanonicalEncoder& operator=(CanonicalEncoder&&) noexcept = default;

  [[nodiscard]] CanonicalEncoderState state() const noexcept { return state_; }
  [[nodiscard]] CanonicalEncodingError error() const noexcept { return error_; }
  [[nodiscard]] std::size_t bytesWritten() const noexcept {
    return finalized_ ? finalized_->bytes().size() : bytes_.size();
  }
  [[nodiscard]] std::size_t maximumOutputBytes() const noexcept { return maximum_output_bytes_; }

  [[nodiscard]] CanonicalEncodingError writeU8(std::uint8_t value);
  [[nodiscard]] CanonicalEncodingError writeU16(std::uint16_t value);
  [[nodiscard]] CanonicalEncodingError writeU32(std::uint32_t value);
  [[nodiscard]] CanonicalEncodingError writeU64(std::uint64_t value);
  [[nodiscard]] CanonicalEncodingError writeI8(std::int8_t value);
  [[nodiscard]] CanonicalEncodingError writeI16(std::int16_t value);
  [[nodiscard]] CanonicalEncodingError writeI32(std::int32_t value);
  [[nodiscard]] CanonicalEncodingError writeI64(std::int64_t value);
  [[nodiscard]] CanonicalEncodingError writeBool(bool value);
  [[nodiscard]] CanonicalEncodingError writeOptionalMarker(bool present);
  [[nodiscard]] CanonicalEncodingError writeBytes(std::span<const std::byte> value);
  [[nodiscard]] CanonicalEncodingError writeString(std::string_view value);
  [[nodiscard]] CanonicalEncodingError writeHash(const ContentHash& value);
  [[nodiscard]] CanonicalEncodingError writeDouble(double value);

  template <typename Derived>
  [[nodiscard]] CanonicalEncodingError writeEigenVector(const Eigen::MatrixBase<Derived>& value) {
    static_assert(std::is_same_v<typename Derived::Scalar, double>,
                  "canonical Eigen vectors must contain doubles");
    if (state_ != CanonicalEncoderState::Ready) {
      return unavailableError();
    }
    if (value.rows() != 1 && value.cols() != 1) {
      return fail(CanonicalEncodingError::InvalidVectorShape);
    }
    const auto count = value.size();
    if (count < 0 || static_cast<std::uintmax_t>(count) >
                         static_cast<std::uintmax_t>(std::numeric_limits<std::uint64_t>::max())) {
      return fail(CanonicalEncodingError::DimensionOverflow);
    }
    for (Eigen::Index index = 0; index < count; ++index) {
      if (!std::isfinite(value(index))) {
        return fail(CanonicalEncodingError::NonFiniteFloatingPoint);
      }
    }
    std::size_t field_bytes{};
    if (!fieldSize(8U, static_cast<std::uint64_t>(count), 8U, &field_bytes)) {
      return fail(CanonicalEncodingError::DimensionOverflow);
    }
    if (const auto result = prepareField(field_bytes); result != CanonicalEncodingError::None) {
      return result;
    }
    appendUnsignedUnchecked(static_cast<std::uint64_t>(count), 8U);
    for (Eigen::Index index = 0; index < count; ++index) {
      appendDoubleUnchecked(value(index));
    }
    return CanonicalEncodingError::None;
  }

  template <typename Derived>
  [[nodiscard]] CanonicalEncodingError writeEigenMatrix(const Eigen::MatrixBase<Derived>& value) {
    static_assert(std::is_same_v<typename Derived::Scalar, double>,
                  "canonical Eigen matrices must contain doubles");
    if (state_ != CanonicalEncoderState::Ready) {
      return unavailableError();
    }
    if (value.rows() < 0 || value.cols() < 0 ||
        static_cast<std::uintmax_t>(value.rows()) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::uint64_t>::max()) ||
        static_cast<std::uintmax_t>(value.cols()) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::uint64_t>::max())) {
      return fail(CanonicalEncodingError::DimensionOverflow);
    }
    for (Eigen::Index row = 0; row < value.rows(); ++row) {
      for (Eigen::Index column = 0; column < value.cols(); ++column) {
        if (!std::isfinite(value(row, column))) {
          return fail(CanonicalEncodingError::NonFiniteFloatingPoint);
        }
      }
    }
    const auto rows = static_cast<std::uint64_t>(value.rows());
    const auto columns = static_cast<std::uint64_t>(value.cols());
    std::uint64_t elements{};
    if (!checkedMultiply(rows, columns, &elements)) {
      return fail(CanonicalEncodingError::DimensionOverflow);
    }
    std::size_t field_bytes{};
    if (!fieldSize(16U, elements, 8U, &field_bytes)) {
      return fail(CanonicalEncodingError::DimensionOverflow);
    }
    if (const auto result = prepareField(field_bytes); result != CanonicalEncodingError::None) {
      return result;
    }
    appendUnsignedUnchecked(rows, 8U);
    appendUnsignedUnchecked(columns, 8U);
    for (Eigen::Index row = 0; row < value.rows(); ++row) {
      for (Eigen::Index column = 0; column < value.cols(); ++column) {
        appendDoubleUnchecked(value(row, column));
      }
    }
    return CanonicalEncodingError::None;
  }

  [[nodiscard]] CanonicalEncodingError writePose3(const Pose3d& value);

  [[nodiscard]] Result<CanonicalByteSequence, CanonicalEncodingError> finish();

private:
  explicit CanonicalEncoder(std::size_t maximum_output_bytes)
      : maximum_output_bytes_(maximum_output_bytes) {}

  [[nodiscard]] CanonicalEncodingError unavailableError() const noexcept;
  [[nodiscard]] CanonicalEncodingError fail(CanonicalEncodingError error) noexcept;
  [[nodiscard]] CanonicalEncodingError prepareField(std::size_t bytes) noexcept;
  [[nodiscard]] static bool checkedMultiply(std::uint64_t left, std::uint64_t right,
                                            std::uint64_t* result) noexcept;
  [[nodiscard]] static bool fieldSize(std::size_t prefix_bytes, std::uint64_t element_count,
                                      std::size_t element_bytes, std::size_t* result) noexcept;
  void appendUnsignedUnchecked(std::uint64_t value, std::size_t width);
  void appendDoubleUnchecked(double value);

  std::vector<std::byte> bytes_;
  std::size_t maximum_output_bytes_{};
  CanonicalEncoderState state_{CanonicalEncoderState::Ready};
  CanonicalEncodingError error_{CanonicalEncodingError::None};
  std::optional<CanonicalByteSequence> finalized_;
};

}  // namespace meridian::core
