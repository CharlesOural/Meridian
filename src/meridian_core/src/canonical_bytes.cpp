#include "meridian/core/canonical_bytes.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

#include "meridian/core/sha256.hpp"

namespace meridian::core {

static_assert(sizeof(double) == sizeof(std::uint64_t));
static_assert(std::numeric_limits<double>::is_iec559);

Result<CanonicalEncoder, CanonicalEncodingError> CanonicalEncoder::create(
    std::string_view domain_tag, std::uint32_t schema_version, std::uint64_t maximum_output_bytes) {
  using EncoderResult = Result<CanonicalEncoder, CanonicalEncodingError>;
  const auto maximum_vector_bytes = std::vector<std::byte>{}.max_size();
  if (maximum_output_bytes == 0U ||
      maximum_output_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      maximum_output_bytes > static_cast<std::uint64_t>(maximum_vector_bytes)) {
    return EncoderResult::failure(CanonicalEncodingError::InvalidMaximumBytes);
  }
  if (domain_tag.empty()) {
    return EncoderResult::failure(CanonicalEncodingError::InvalidDomainTag);
  }
  if (schema_version == 0U) {
    return EncoderResult::failure(CanonicalEncodingError::InvalidSchemaVersion);
  }

  CanonicalEncoder encoder(static_cast<std::size_t>(maximum_output_bytes));
  if (encoder.writeString(domain_tag) != CanonicalEncodingError::None ||
      encoder.writeU32(schema_version) != CanonicalEncodingError::None) {
    return EncoderResult::failure(encoder.error());
  }
  return EncoderResult::success(std::move(encoder));
}

CanonicalEncodingError CanonicalEncoder::unavailableError() const noexcept {
  return state_ == CanonicalEncoderState::Finalized ? CanonicalEncodingError::AlreadyFinalized
                                                    : error_;
}

CanonicalEncodingError CanonicalEncoder::fail(CanonicalEncodingError error) noexcept {
  if (state_ != CanonicalEncoderState::Ready) {
    return unavailableError();
  }
  state_ = CanonicalEncoderState::Failed;
  error_ = error;
  return error_;
}

CanonicalEncodingError CanonicalEncoder::prepareField(std::size_t bytes) noexcept {
  if (state_ != CanonicalEncoderState::Ready) {
    return unavailableError();
  }
  if (bytes > maximum_output_bytes_ - bytes_.size()) {
    return fail(CanonicalEncodingError::OutputLimitExceeded);
  }
  const std::size_t required = bytes_.size() + bytes;
  std::size_t reserve_bytes = required;
  if (required > bytes_.capacity()) {
    const std::size_t doubled = bytes_.capacity() > maximum_output_bytes_ / 2U
                                    ? maximum_output_bytes_
                                    : bytes_.capacity() * 2U;
    reserve_bytes =
        std::max(required, std::min(maximum_output_bytes_, std::max<std::size_t>(64U, doubled)));
  }
  try {
    bytes_.reserve(reserve_bytes);
  } catch (const std::bad_alloc&) {
    return fail(CanonicalEncodingError::AllocationFailure);
  } catch (const std::length_error&) {
    return fail(CanonicalEncodingError::AllocationFailure);
  }
  return CanonicalEncodingError::None;
}

bool CanonicalEncoder::checkedMultiply(std::uint64_t left, std::uint64_t right,
                                       std::uint64_t* result) noexcept {
  if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool CanonicalEncoder::fieldSize(std::size_t prefix_bytes, std::uint64_t element_count,
                                 std::size_t element_bytes, std::size_t* result) noexcept {
  if (element_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      element_count >
          static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max() - prefix_bytes) /
                                     element_bytes)) {
    return false;
  }
  *result = prefix_bytes + static_cast<std::size_t>(element_count) * element_bytes;
  return true;
}

void CanonicalEncoder::appendUnsignedUnchecked(std::uint64_t value, std::size_t width) {
  for (std::size_t index = 0U; index < width; ++index) {
    const std::size_t shift = (width - 1U - index) * 8U;
    bytes_.push_back(static_cast<std::byte>(value >> shift));
  }
}

void CanonicalEncoder::appendDoubleUnchecked(double value) {
  if (value == 0.0) {
    value = 0.0;
  }
  appendUnsignedUnchecked(std::bit_cast<std::uint64_t>(value), 8U);
}

CanonicalEncodingError CanonicalEncoder::writeU8(std::uint8_t value) {
  if (const auto result = prepareField(1U); result != CanonicalEncodingError::None) {
    return result;
  }
  appendUnsignedUnchecked(value, 1U);
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writeU16(std::uint16_t value) {
  if (const auto result = prepareField(2U); result != CanonicalEncodingError::None) {
    return result;
  }
  appendUnsignedUnchecked(value, 2U);
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writeU32(std::uint32_t value) {
  if (const auto result = prepareField(4U); result != CanonicalEncodingError::None) {
    return result;
  }
  appendUnsignedUnchecked(value, 4U);
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writeU64(std::uint64_t value) {
  if (const auto result = prepareField(8U); result != CanonicalEncodingError::None) {
    return result;
  }
  appendUnsignedUnchecked(value, 8U);
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writeI8(std::int8_t value) {
  return writeU8(static_cast<std::uint8_t>(value));
}

CanonicalEncodingError CanonicalEncoder::writeI16(std::int16_t value) {
  return writeU16(static_cast<std::uint16_t>(value));
}

CanonicalEncodingError CanonicalEncoder::writeI32(std::int32_t value) {
  return writeU32(static_cast<std::uint32_t>(value));
}

CanonicalEncodingError CanonicalEncoder::writeI64(std::int64_t value) {
  return writeU64(static_cast<std::uint64_t>(value));
}

CanonicalEncodingError CanonicalEncoder::writeBool(bool value) {
  return writeU8(value ? 1U : 0U);
}

CanonicalEncodingError CanonicalEncoder::writeOptionalMarker(bool present) {
  return writeU8(present ? 1U : 0U);
}

CanonicalEncodingError CanonicalEncoder::writeBytes(std::span<const std::byte> value) {
  std::size_t field_bytes{};
  if (!fieldSize(8U, value.size(), 1U, &field_bytes)) {
    return fail(CanonicalEncodingError::DimensionOverflow);
  }
  if (const auto result = prepareField(field_bytes); result != CanonicalEncodingError::None) {
    return result;
  }
  appendUnsignedUnchecked(value.size(), 8U);
  bytes_.insert(bytes_.end(), value.begin(), value.end());
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writeString(std::string_view value) {
  return writeBytes(std::as_bytes(std::span(value.data(), value.size())));
}

CanonicalEncodingError CanonicalEncoder::writeHash(const ContentHash& value) {
  return writeBytes(std::as_bytes(std::span(value)));
}

CanonicalEncodingError CanonicalEncoder::writeDouble(double value) {
  if (state_ != CanonicalEncoderState::Ready) {
    return unavailableError();
  }
  if (!std::isfinite(value)) {
    return fail(CanonicalEncodingError::NonFiniteFloatingPoint);
  }
  if (const auto result = prepareField(8U); result != CanonicalEncodingError::None) {
    return result;
  }
  appendDoubleUnchecked(value);
  return CanonicalEncodingError::None;
}

CanonicalEncodingError CanonicalEncoder::writePose3(const Pose3d& value) {
  const Eigen::Matrix4d matrix = value.matrix();
  if (!matrix.allFinite()) {
    return fail(CanonicalEncodingError::NonFiniteFloatingPoint);
  }
  return writeEigenMatrix(matrix);
}

Result<CanonicalByteSequence, CanonicalEncodingError> CanonicalEncoder::finish() {
  using SequenceResult = Result<CanonicalByteSequence, CanonicalEncodingError>;
  if (state_ == CanonicalEncoderState::Failed) {
    return SequenceResult::failure(error_);
  }
  if (state_ == CanonicalEncoderState::Finalized) {
    return SequenceResult::success(*finalized_);
  }
  try {
    ImmutableBytes immutable = std::make_shared<const std::vector<std::byte>>(std::move(bytes_));
    finalized_.emplace(CanonicalByteSequence{immutable, sha256Bytes(*immutable)});
  } catch (const std::bad_alloc&) {
    return SequenceResult::failure(fail(CanonicalEncodingError::AllocationFailure));
  }
  state_ = CanonicalEncoderState::Finalized;
  return SequenceResult::success(*finalized_);
}

}  // namespace meridian::core
