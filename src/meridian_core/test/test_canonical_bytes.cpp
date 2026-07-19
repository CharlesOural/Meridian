#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "meridian/core/canonical_bytes.hpp"
#include "meridian/core/sha256.hpp"

namespace meridian::core {
namespace {

std::span<const std::byte> bytesOf(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

std::string bytesHex(std::span<const std::byte> bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(bytes.size() * 2U, '\0');
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    const auto value = static_cast<std::uint8_t>(bytes[index]);
    output[2U * index] = kHex[value >> 4U];
    output[2U * index + 1U] = kHex[value & 0x0fU];
  }
  return output;
}

std::uint64_t readU64(std::span<const std::byte> bytes, std::size_t offset) {
  std::uint64_t result{};
  for (std::size_t index = 0U; index < 8U; ++index) {
    result = (result << 8U) | static_cast<std::uint8_t>(bytes[offset + index]);
  }
  return result;
}

CanonicalEncoder readyEncoder(std::string_view domain = "test", std::uint32_t version = 1U,
                              std::uint64_t maximum_bytes = 4096U) {
  auto result = CanonicalEncoder::create(domain, version, maximum_bytes);
  EXPECT_TRUE(result);
  return std::move(result).value();
}

TEST(Sha256, MatchesStandardKnownAnswers) {
  constexpr std::string_view kAbc = "abc";
  constexpr std::string_view kMulti = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

  EXPECT_EQ(sha256Hex(sha256Bytes({})),
            "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256Hex(sha256Bytes(bytesOf(kAbc))),
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256Hex(sha256Bytes(bytesOf(kMulti))),
            "248d6a61d20638b8e5c026930c3e6039"
            "a33ce45964ff2167f6ecedd419db06c1");

  const std::vector<std::byte> million_a(1'000'000U, std::byte{0x61});
  EXPECT_EQ(sha256Hex(sha256Bytes(million_a)),
            "cdc76e5c9914fb9281a1c7e284d73e67"
            "f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256, StreamsAcrossBlockBoundariesAndExposesState) {
  constexpr std::string_view kText =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "a deliberately longer suffix crossing a sha256 block boundary";
  const auto bytes = bytesOf(kText);
  Sha256 stream;
  EXPECT_EQ(stream.state(), Sha256State::Accepting);
  EXPECT_EQ(stream.update(bytes.first(7U)), Sha256Error::None);
  EXPECT_EQ(stream.update(bytes.subspan(7U, 58U)), Sha256Error::None);
  EXPECT_EQ(stream.update(bytes.subspan(65U)), Sha256Error::None);
  EXPECT_EQ(stream.bytesProcessed(), bytes.size());

  auto first = stream.finish();
  ASSERT_TRUE(first);
  EXPECT_EQ(first.value(), sha256Bytes(bytes));
  EXPECT_EQ(stream.state(), Sha256State::Finalized);

  auto second = stream.finish();
  ASSERT_TRUE(second);
  EXPECT_EQ(second.value(), first.value());
  EXPECT_EQ(stream.update({}), Sha256Error::AlreadyFinalized);
  EXPECT_EQ(stream.state(), Sha256State::Finalized);
}

TEST(Sha256, BulkAndBytewiseUpdatesAreIdenticalAcrossBlockAndPaddingBoundaries) {
  std::vector<std::byte> bytes(257U);
  for (std::size_t index = 0U; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::byte>((index * 131U + 17U) & 0xffU);
  }
  for (std::size_t length = 0U; length <= bytes.size(); ++length) {
    const std::span<const std::byte> message(bytes.data(), length);
    const ContentHash bulk = sha256Bytes(message);

    Sha256 bytewise;
    for (const std::byte& byte : message) {
      ASSERT_EQ(bytewise.update(std::span(&byte, 1U)), Sha256Error::None);
    }
    auto bytewise_digest = bytewise.finish();
    ASSERT_TRUE(bytewise_digest);
    EXPECT_EQ(bytewise_digest.value(), bulk) << "length=" << length;

    Sha256 irregular_chunks;
    std::size_t offset{};
    for (const std::size_t chunk : {1U, 63U, 64U, 3U, 65U, 7U, 54U}) {
      const std::size_t count = std::min(chunk, length - offset);
      ASSERT_EQ(irregular_chunks.update(message.subspan(offset, count)), Sha256Error::None);
      offset += count;
      if (offset == length) {
        break;
      }
    }
    if (offset != length) {
      ASSERT_EQ(irregular_chunks.update(message.subspan(offset)), Sha256Error::None);
    }
    auto irregular_digest = irregular_chunks.finish();
    ASSERT_TRUE(irregular_digest);
    EXPECT_EQ(irregular_digest.value(), bulk) << "length=" << length;
  }
}

TEST(CanonicalEncoder, HasGoldenDomainVersionAndIntegerEndianness) {
  CanonicalEncoder encoder = readyEncoder("d", 0x01020304U);
  EXPECT_EQ(encoder.writeU8(0xabU), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeU16(0x1234U), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeU32(0x89abcdefU), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeU64(0x0123456789abcdefULL), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeI8(-2), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeI16(-2), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeI32(-2), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeI64(-2), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeBool(true), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeOptionalMarker(false), CanonicalEncodingError::None);

  auto result = encoder.finish();
  ASSERT_TRUE(result);
  EXPECT_EQ(bytesHex(result.value().bytes()),
            "00000000000000016401020304"
            "ab123489abcdef0123456789abcdef"
            "fefffefffffffefffffffffffffffe0100");
  EXPECT_EQ(result.value().digest(), sha256Bytes(result.value().bytes()));
}

TEST(CanonicalEncoder, LengthPrefixesBytesStringsAndHashes) {
  CanonicalEncoder encoder = readyEncoder("wire", 7U);
  const std::array<std::byte, 2> raw{std::byte{0x00}, std::byte{0xff}};
  ContentHash hash{};
  for (std::size_t index = 0U; index < hash.size(); ++index) {
    hash[index] = static_cast<std::uint8_t>(index);
  }
  EXPECT_EQ(encoder.writeBytes(raw), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeString("A"), CanonicalEncodingError::None);
  EXPECT_EQ(encoder.writeHash(hash), CanonicalEncodingError::None);

  auto result = encoder.finish();
  ASSERT_TRUE(result);
  EXPECT_EQ(bytesHex(result.value().bytes()),
            "00000000000000047769726500000007"
            "000000000000000200ff"
            "000000000000000141"
            "0000000000000020"
            "000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f");
}

TEST(CanonicalEncoder, NormalizesNegativeZeroAndRejectsNonFiniteValues) {
  CanonicalEncoder positive = readyEncoder("zero");
  CanonicalEncoder negative = readyEncoder("zero");
  EXPECT_EQ(positive.writeDouble(0.0), CanonicalEncodingError::None);
  EXPECT_EQ(negative.writeDouble(-0.0), CanonicalEncodingError::None);
  auto positive_result = positive.finish();
  auto negative_result = negative.finish();
  ASSERT_TRUE(positive_result);
  ASSERT_TRUE(negative_result);
  EXPECT_EQ(*positive_result.value().storage(), *negative_result.value().storage());
  EXPECT_EQ(positive_result.value().digest(), negative_result.value().digest());
  EXPECT_TRUE(bytesHex(positive_result.value().bytes()).ends_with("0000000000000000"));

  CanonicalEncoder nan_encoder = readyEncoder();
  const std::size_t header_bytes = nan_encoder.bytesWritten();
  EXPECT_EQ(nan_encoder.writeDouble(std::numeric_limits<double>::quiet_NaN()),
            CanonicalEncodingError::NonFiniteFloatingPoint);
  EXPECT_EQ(nan_encoder.state(), CanonicalEncoderState::Failed);
  EXPECT_EQ(nan_encoder.bytesWritten(), header_bytes);
  auto nan_result = nan_encoder.finish();
  ASSERT_FALSE(nan_result);
  EXPECT_EQ(nan_result.error(), CanonicalEncodingError::NonFiniteFloatingPoint);

  CanonicalEncoder infinity_encoder = readyEncoder();
  EXPECT_EQ(infinity_encoder.writeDouble(std::numeric_limits<double>::infinity()),
            CanonicalEncodingError::NonFiniteFloatingPoint);
}

TEST(CanonicalEncoder, EncodesEigenValuesInRowMajorOrder) {
  Eigen::Matrix2d column_major;
  column_major << 1.0, 2.0, 3.0, 4.0;
  Eigen::Matrix<double, 2, 2, Eigen::RowMajor> row_major = column_major;
  CanonicalEncoder column_encoder = readyEncoder("matrix");
  CanonicalEncoder row_encoder = readyEncoder("matrix");
  EXPECT_EQ(column_encoder.writeEigenMatrix(column_major), CanonicalEncodingError::None);
  EXPECT_EQ(row_encoder.writeEigenMatrix(row_major), CanonicalEncodingError::None);
  auto column_result = column_encoder.finish();
  auto row_result = row_encoder.finish();
  ASSERT_TRUE(column_result);
  ASSERT_TRUE(row_result);
  EXPECT_EQ(*column_result.value().storage(), *row_result.value().storage());
  EXPECT_TRUE(bytesHex(column_result.value().bytes())
                  .ends_with("00000000000000020000000000000002"
                             "3ff00000000000004000000000000000"
                             "40080000000000004010000000000000"));

  const Eigen::Vector2d vector(1.0, -0.0);
  CanonicalEncoder vector_encoder = readyEncoder("vector");
  EXPECT_EQ(vector_encoder.writeEigenVector(vector), CanonicalEncodingError::None);
  auto vector_result = vector_encoder.finish();
  ASSERT_TRUE(vector_result);
  EXPECT_TRUE(bytesHex(vector_result.value().bytes())
                  .ends_with("00000000000000023ff00000000000000000000000000000"));

  CanonicalEncoder invalid_vector = readyEncoder();
  EXPECT_EQ(invalid_vector.writeEigenVector(column_major),
            CanonicalEncodingError::InvalidVectorShape);
}

TEST(CanonicalEncoder, EncodesPoseAsFiniteFourByFourMatrix) {
  CanonicalEncoder encoder = readyEncoder("pose");
  const std::size_t offset = encoder.bytesWritten();
  EXPECT_EQ(encoder.writePose3(Pose3d{}), CanonicalEncodingError::None);
  auto result = encoder.finish();
  ASSERT_TRUE(result);
  const auto encoded = result.value().bytes();
  ASSERT_EQ(encoded.size(), offset + 16U + 16U * 8U);
  EXPECT_EQ(readU64(encoded, offset), 4U);
  EXPECT_EQ(readU64(encoded, offset + 8U), 4U);
  for (std::size_t row = 0U; row < 4U; ++row) {
    for (std::size_t column = 0U; column < 4U; ++column) {
      const std::uint64_t expected = row == column ? 0x3ff0000000000000ULL : 0U;
      EXPECT_EQ(readU64(encoded, offset + 16U + (row * 4U + column) * 8U), expected);
    }
  }

  Pose3d nonfinite;
  nonfinite.translation().x() = std::numeric_limits<double>::infinity();
  CanonicalEncoder invalid = readyEncoder("pose");
  EXPECT_EQ(invalid.writePose3(nonfinite), CanonicalEncodingError::NonFiniteFloatingPoint);
}

TEST(CanonicalEncoder, RejectsNonFiniteEigenFieldBeforeWritingAnyOfIt) {
  Eigen::Vector3d vector(1.0, std::numeric_limits<double>::infinity(), 3.0);
  CanonicalEncoder vector_encoder = readyEncoder();
  const std::size_t vector_offset = vector_encoder.bytesWritten();
  EXPECT_EQ(vector_encoder.writeEigenVector(vector),
            CanonicalEncodingError::NonFiniteFloatingPoint);
  EXPECT_EQ(vector_encoder.bytesWritten(), vector_offset);

  Eigen::Matrix2d matrix = Eigen::Matrix2d::Identity();
  matrix(1, 0) = std::numeric_limits<double>::quiet_NaN();
  CanonicalEncoder matrix_encoder = readyEncoder();
  const std::size_t matrix_offset = matrix_encoder.bytesWritten();
  EXPECT_EQ(matrix_encoder.writeEigenMatrix(matrix),
            CanonicalEncodingError::NonFiniteFloatingPoint);
  EXPECT_EQ(matrix_encoder.bytesWritten(), matrix_offset);
}

TEST(CanonicalEncoder, EnforcesOutputCapBeforeFieldGrowth) {
  // Header for domain "x" is u64 length + one byte + u32 version = 13 bytes.
  CanonicalEncoder exact_header = readyEncoder("x", 1U, 13U);
  EXPECT_EQ(exact_header.bytesWritten(), 13U);
  EXPECT_EQ(exact_header.writeBytes({}), CanonicalEncodingError::OutputLimitExceeded);
  EXPECT_EQ(exact_header.state(), CanonicalEncoderState::Failed);
  EXPECT_EQ(exact_header.bytesWritten(), 13U);

  auto too_small = CanonicalEncoder::create("x", 1U, 12U);
  ASSERT_FALSE(too_small);
  EXPECT_EQ(too_small.error(), CanonicalEncodingError::OutputLimitExceeded);

  CanonicalEncoder exact_value = readyEncoder("x", 1U, 14U);
  EXPECT_EQ(exact_value.writeBool(false), CanonicalEncodingError::None);
  auto exact_result = exact_value.finish();
  ASSERT_TRUE(exact_result);
  EXPECT_EQ(exact_result.value().bytes().size(), 14U);
}

TEST(CanonicalEncoder, RejectsInvalidConfigurationAndHasFinalState) {
  auto no_limit = CanonicalEncoder::create("domain", 1U, 0U);
  ASSERT_FALSE(no_limit);
  EXPECT_EQ(no_limit.error(), CanonicalEncodingError::InvalidMaximumBytes);
  const auto vector_maximum = std::vector<std::byte>{}.max_size();
  if (vector_maximum < std::numeric_limits<std::uint64_t>::max()) {
    auto impossible_limit =
        CanonicalEncoder::create("domain", 1U, static_cast<std::uint64_t>(vector_maximum) + 1U);
    ASSERT_FALSE(impossible_limit);
    EXPECT_EQ(impossible_limit.error(), CanonicalEncodingError::InvalidMaximumBytes);
  }
  auto no_domain = CanonicalEncoder::create("", 1U, 64U);
  ASSERT_FALSE(no_domain);
  EXPECT_EQ(no_domain.error(), CanonicalEncodingError::InvalidDomainTag);
  auto no_version = CanonicalEncoder::create("domain", 0U, 64U);
  ASSERT_FALSE(no_version);
  EXPECT_EQ(no_version.error(), CanonicalEncodingError::InvalidSchemaVersion);

  CanonicalEncoder encoder = readyEncoder();
  auto first = encoder.finish();
  ASSERT_TRUE(first);
  const std::size_t finalized_bytes = encoder.bytesWritten();
  EXPECT_EQ(encoder.state(), CanonicalEncoderState::Finalized);
  EXPECT_GT(finalized_bytes, 0U);
  EXPECT_EQ(encoder.bytesWritten(), finalized_bytes);
  EXPECT_EQ(encoder.writeU8(1U), CanonicalEncodingError::AlreadyFinalized);
  auto second = encoder.finish();
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value().storage().get(), second.value().storage().get());
  EXPECT_EQ(first.value().digest(), second.value().digest());
}

}  // namespace
}  // namespace meridian::core
