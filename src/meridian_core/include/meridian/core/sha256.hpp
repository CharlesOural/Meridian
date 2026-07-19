#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include "meridian/core/blob.hpp"
#include "meridian/core/result.hpp"

namespace meridian::core {

enum class Sha256State {
  Accepting,
  Finalized,
  Failed,
};

enum class Sha256Error {
  None,
  AlreadyFinalized,
  InputLengthOverflow,
};

// Allocation-free streaming SHA-256. Finalization is idempotent, while any
// update after finalization is explicitly rejected. A stream whose byte count
// cannot be represented by the SHA-256 length field enters Failed state.
class Sha256 {
public:
  [[nodiscard]] Sha256Error update(std::span<const std::byte> bytes) noexcept;
  [[nodiscard]] Result<ContentHash, Sha256Error> finish() noexcept;

  [[nodiscard]] Sha256State state() const noexcept { return state_kind_; }
  [[nodiscard]] Sha256Error error() const noexcept { return error_; }
  [[nodiscard]] std::uint64_t bytesProcessed() const noexcept { return total_bytes_; }

private:
  static constexpr std::uint32_t rotate(std::uint32_t value, unsigned count) noexcept {
    return (value >> count) | (value << (32U - count));
  }

  void transform() noexcept;

  std::array<std::uint32_t, 8> hash_state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                           0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_{};
  std::uint64_t total_bytes_{};
  ContentHash digest_{};
  Sha256State state_kind_{Sha256State::Accepting};
  Sha256Error error_{Sha256Error::None};
};

// One-shot SHA-256 over an addressable byte span.
[[nodiscard]] ContentHash sha256Bytes(std::span<const std::byte> bytes) noexcept;

// Canonical lowercase hexadecimal representation (exactly 64 characters).
[[nodiscard]] std::string sha256Hex(const ContentHash& digest);

}  // namespace meridian::core
