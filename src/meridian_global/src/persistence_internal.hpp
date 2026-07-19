#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/core/blob.hpp"
#include "meridian/core/result.hpp"

namespace meridian::global::persistence_internal {

using Bytes = std::vector<std::byte>;

class Sha256 {
public:
  void update(std::span<const std::byte> bytes) noexcept;
  [[nodiscard]] core::ContentHash finish() noexcept;

private:
  static constexpr std::uint32_t rotate(std::uint32_t value, unsigned count) noexcept {
    return (value >> count) | (value << (32U - count));
  }
  void transform() noexcept;

  std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{};
  std::uint64_t total_bytes_{};
};

[[nodiscard]] core::ContentHash hashBytes(std::span<const std::byte> bytes) noexcept;
// Empty, "abc", and a standard multi-block vector are checked.
[[nodiscard]] bool sha256SelfTest() noexcept;
[[nodiscard]] std::string hashHex(const core::ContentHash& hash);

constexpr std::size_t kFrameOverhead = 8U + 4U + 8U + 32U;

struct DecodedFrame {
  core::ContentHash checksum{};
  std::span<const std::byte> payload;
};

enum class ErrorCode {
  IoFailure,
  InvalidRecord,
  SizeMismatch,
  ChecksumMismatch,
  Conflict,
  AmbiguousIntegrity,
};

struct Error {
  ErrorCode code{ErrorCode::IoFailure};
  std::string detail;
};

[[nodiscard]] Bytes frameBytes(std::span<const std::byte> magic, std::uint32_t format_version,
                               std::span<const std::byte> payload);
[[nodiscard]] core::Result<DecodedFrame, Error> decodeFrame(
    std::span<const std::byte> bytes, std::span<const std::byte> expected_magic,
    std::uint32_t expected_format_version, std::uint64_t maximum_payload_bytes,
    const std::filesystem::path& path);

[[nodiscard]] core::Result<Bytes, Error> readFile(const std::filesystem::path& path,
                                                  std::uint64_t maximum_bytes);
[[nodiscard]] std::optional<Error> syncDirectory(const std::filesystem::path& path);
[[nodiscard]] std::optional<Error> ensureDirectory(const std::filesystem::path& path,
                                                   std::string_view purpose);
[[nodiscard]] std::optional<Error> cleanTemporaryFiles(const std::filesystem::path& directory,
                                                       std::size_t* recovered);

enum class DurableWriteDisposition {
  Written,
  ExistingIdentical,
};

struct DurableWriteResult {
  DurableWriteDisposition disposition{DurableWriteDisposition::Written};
  std::int64_t elapsed_us{};
};

enum class AtomicWriteFailpoint {
  None,
  AfterRenameBeforeDirectorySync,
};

// Private deterministic crash-window injection used only by package tests.
void setAtomicWriteFailpointForTesting(AtomicWriteFailpoint failpoint) noexcept;

// Immutable no-replace write. An existing identical destination is reconciled
// by re-fsyncing its directory. A destination that cannot be read after an
// observed/potential rename is ambiguous, never reported as a content conflict.
[[nodiscard]] core::Result<DurableWriteResult, Error> durableWriteNoReplace(
    const std::filesystem::path& destination, std::span<const std::byte> bytes,
    std::uint64_t* temporary_counter);

}  // namespace meridian::global::persistence_internal
