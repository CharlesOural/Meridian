#include "meridian/core/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>

namespace meridian::core {
namespace {

constexpr std::array<std::uint32_t, 64> kRound{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

}  // namespace

Sha256Error Sha256::update(std::span<const std::byte> bytes) noexcept {
  if (state_kind_ == Sha256State::Finalized) {
    return Sha256Error::AlreadyFinalized;
  }
  if (state_kind_ == Sha256State::Failed) {
    return error_;
  }
  constexpr std::uint64_t kMaximumInputBytes = std::numeric_limits<std::uint64_t>::max() / 8U;
  if (bytes.size() > kMaximumInputBytes - total_bytes_) {
    state_kind_ = Sha256State::Failed;
    error_ = Sha256Error::InputLengthOverflow;
    return error_;
  }
  total_bytes_ += bytes.size();
  if (bytes.empty()) {
    return Sha256Error::None;
  }
  std::size_t offset{};
  if (block_size_ != 0U) {
    const std::size_t copied = std::min(block_.size() - block_size_, bytes.size());
    std::memcpy(block_.data() + block_size_, bytes.data(), copied);
    block_size_ += copied;
    offset += copied;
    if (block_size_ == block_.size()) {
      transform();
      block_size_ = 0U;
    }
  }
  while (bytes.size() - offset >= block_.size()) {
    std::memcpy(block_.data(), bytes.data() + offset, block_.size());
    transform();
    offset += block_.size();
  }
  const std::size_t remaining = bytes.size() - offset;
  if (remaining != 0U) {
    std::memcpy(block_.data(), bytes.data() + offset, remaining);
    block_size_ = remaining;
  }
  return Sha256Error::None;
}

Result<ContentHash, Sha256Error> Sha256::finish() noexcept {
  using HashResult = Result<ContentHash, Sha256Error>;
  if (state_kind_ == Sha256State::Failed) {
    return HashResult::failure(error_);
  }
  if (state_kind_ == Sha256State::Finalized) {
    return HashResult::success(digest_);
  }

  const std::uint64_t total_bits = total_bytes_ * 8U;
  block_[block_size_++] = 0x80U;
  if (block_size_ > 56U) {
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0U);
    transform();
    block_size_ = 0U;
  }
  std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0U);
  for (std::size_t index = 0U; index < 8U; ++index) {
    block_[63U - index] = static_cast<std::uint8_t>(total_bits >> (index * 8U));
  }
  transform();
  for (std::size_t word = 0U; word < hash_state_.size(); ++word) {
    for (std::size_t byte = 0U; byte < 4U; ++byte) {
      digest_[word * 4U + byte] =
          static_cast<std::uint8_t>(hash_state_[word] >> ((3U - byte) * 8U));
    }
  }
  state_kind_ = Sha256State::Finalized;
  return HashResult::success(digest_);
}

void Sha256::transform() noexcept {
  std::array<std::uint32_t, 64> schedule{};
  for (std::size_t index = 0U; index < 16U; ++index) {
    const std::size_t offset = index * 4U;
    schedule[index] = (static_cast<std::uint32_t>(block_[offset]) << 24U) |
                      (static_cast<std::uint32_t>(block_[offset + 1U]) << 16U) |
                      (static_cast<std::uint32_t>(block_[offset + 2U]) << 8U) |
                      static_cast<std::uint32_t>(block_[offset + 3U]);
  }
  for (std::size_t index = 16U; index < schedule.size(); ++index) {
    const std::uint32_t s0 = rotate(schedule[index - 15U], 7U) ^
                             rotate(schedule[index - 15U], 18U) ^ (schedule[index - 15U] >> 3U);
    const std::uint32_t s1 = rotate(schedule[index - 2U], 17U) ^ rotate(schedule[index - 2U], 19U) ^
                             (schedule[index - 2U] >> 10U);
    schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
  }

  std::uint32_t a = hash_state_[0];
  std::uint32_t b = hash_state_[1];
  std::uint32_t c = hash_state_[2];
  std::uint32_t d = hash_state_[3];
  std::uint32_t e = hash_state_[4];
  std::uint32_t f = hash_state_[5];
  std::uint32_t g = hash_state_[6];
  std::uint32_t h = hash_state_[7];
  for (std::size_t index = 0U; index < schedule.size(); ++index) {
    const std::uint32_t sum1 = rotate(e, 6U) ^ rotate(e, 11U) ^ rotate(e, 25U);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temporary1 = h + sum1 + choose + kRound[index] + schedule[index];
    const std::uint32_t sum0 = rotate(a, 2U) ^ rotate(a, 13U) ^ rotate(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }
  hash_state_[0] += a;
  hash_state_[1] += b;
  hash_state_[2] += c;
  hash_state_[3] += d;
  hash_state_[4] += e;
  hash_state_[5] += f;
  hash_state_[6] += g;
  hash_state_[7] += h;
}

ContentHash sha256Bytes(std::span<const std::byte> bytes) noexcept {
  Sha256 hash;
  const Sha256Error update_error = hash.update(bytes);
  if (update_error != Sha256Error::None) {
    return {};
  }
  auto result = hash.finish();
  return result ? std::move(result).value() : ContentHash{};
}

std::string sha256Hex(const ContentHash& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(digest.size() * 2U, '\0');
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    output[index * 2U] = kHex[digest[index] >> 4U];
    output[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return output;
}

}  // namespace meridian::core
