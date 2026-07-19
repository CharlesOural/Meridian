#include "persistence_internal.hpp"

#include <fcntl.h>
#include <linux/fs.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <utility>

namespace meridian::global::persistence_internal {
namespace {

using Clock = std::chrono::steady_clock;
std::atomic<AtomicWriteFailpoint> g_atomic_write_failpoint{AtomicWriteFailpoint::None};

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

[[nodiscard]] Error error(ErrorCode code, std::string detail) {
  return Error{code, std::move(detail)};
}

[[nodiscard]] std::int64_t elapsedMicroseconds(Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count();
}

void appendU32(Bytes& bytes, std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index) {
    bytes.push_back(static_cast<std::byte>(value >> ((3U - index) * 8U)));
  }
}

void appendU64(Bytes& bytes, std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index) {
    bytes.push_back(static_cast<std::byte>(value >> ((7U - index) * 8U)));
  }
}

[[nodiscard]] std::uint32_t readU32(std::span<const std::byte> bytes) noexcept {
  std::uint32_t value = 0U;
  for (const std::byte byte : bytes.first<4>()) {
    value = (value << 8U) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

[[nodiscard]] std::uint64_t readU64(std::span<const std::byte> bytes) noexcept {
  std::uint64_t value = 0U;
  for (const std::byte byte : bytes.first<8>()) {
    value = (value << 8U) | static_cast<std::uint8_t>(byte);
  }
  return value;
}

}  // namespace

void Sha256::update(std::span<const std::byte> bytes) noexcept {
  for (const std::byte byte : bytes) {
    block_[block_size_++] = static_cast<std::uint8_t>(byte);
    ++total_bytes_;
    if (block_size_ == block_.size()) {
      transform();
      block_size_ = 0U;
    }
  }
}

core::ContentHash Sha256::finish() noexcept {
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
  core::ContentHash output{};
  for (std::size_t word = 0U; word < state_.size(); ++word) {
    for (std::size_t byte = 0U; byte < 4U; ++byte) {
      output[word * 4U + byte] =
          static_cast<std::uint8_t>(state_[word] >> ((3U - byte) * 8U));
    }
  }
  return output;
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
    const std::uint32_t s1 = rotate(schedule[index - 2U], 17U) ^
                             rotate(schedule[index - 2U], 19U) ^ (schedule[index - 2U] >> 10U);
    schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
  }
  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];
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
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

core::ContentHash hashBytes(std::span<const std::byte> bytes) noexcept {
  Sha256 hash;
  hash.update(bytes);
  return hash.finish();
}

bool sha256SelfTest() noexcept {
  constexpr core::ContentHash kEmpty{0xe3U, 0xb0U, 0xc4U, 0x42U, 0x98U, 0xfcU, 0x1cU, 0x14U,
                                     0x9aU, 0xfbU, 0xf4U, 0xc8U, 0x99U, 0x6fU, 0xb9U, 0x24U,
                                     0x27U, 0xaeU, 0x41U, 0xe4U, 0x64U, 0x9bU, 0x93U, 0x4cU,
                                     0xa4U, 0x95U, 0x99U, 0x1bU, 0x78U, 0x52U, 0xb8U, 0x55U};
  constexpr core::ContentHash kAbc{0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU,
                                   0x41U, 0x41U, 0x40U, 0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U,
                                   0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U, 0x7aU, 0x9cU,
                                   0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU};
  constexpr core::ContentHash kMulti{0x24U, 0x8dU, 0x6aU, 0x61U, 0xd2U, 0x06U, 0x38U, 0xb8U,
                                     0xe5U, 0xc0U, 0x26U, 0x93U, 0x0cU, 0x3eU, 0x60U, 0x39U,
                                     0xa3U, 0x3cU, 0xe4U, 0x59U, 0x64U, 0xffU, 0x21U, 0x67U,
                                     0xf6U, 0xecU, 0xedU, 0xd4U, 0x19U, 0xdbU, 0x06U, 0xc1U};
  constexpr std::string_view kAbcText = "abc";
  constexpr std::string_view kMultiText =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  return hashBytes({}) == kEmpty && hashBytes(std::as_bytes(std::span(kAbcText))) == kAbc &&
         hashBytes(std::as_bytes(std::span(kMultiText))) == kMulti;
}

std::string hashHex(const core::ContentHash& hash) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string output(hash.size() * 2U, '\0');
  for (std::size_t index = 0U; index < hash.size(); ++index) {
    output[index * 2U] = kHex[hash[index] >> 4U];
    output[index * 2U + 1U] = kHex[hash[index] & 0x0fU];
  }
  return output;
}

Bytes frameBytes(std::span<const std::byte> magic, std::uint32_t format_version,
                 std::span<const std::byte> payload) {
  Bytes bytes;
  bytes.reserve(magic.size() + 4U + 8U + 32U + payload.size());
  bytes.insert(bytes.end(), magic.begin(), magic.end());
  appendU32(bytes, format_version);
  appendU64(bytes, payload.size());
  const auto checksum = hashBytes(payload);
  const auto checksum_bytes = std::as_bytes(std::span(checksum));
  bytes.insert(bytes.end(), checksum_bytes.begin(), checksum_bytes.end());
  bytes.insert(bytes.end(), payload.begin(), payload.end());
  return bytes;
}

core::Result<DecodedFrame, Error> decodeFrame(std::span<const std::byte> bytes,
                                              std::span<const std::byte> expected_magic,
                                              std::uint32_t expected_format_version,
                                              std::uint64_t maximum_payload_bytes,
                                              const std::filesystem::path& path) {
  using Result = core::Result<DecodedFrame, Error>;
  const std::size_t overhead = expected_magic.size() + 4U + 8U + 32U;
  if (bytes.size() < overhead) {
    return Result::failure(error(ErrorCode::InvalidRecord,
                                 "durable record is shorter than its frame: " + path.string()));
  }
  if (!std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin())) {
    return Result::failure(
        error(ErrorCode::InvalidRecord, "durable record magic is invalid: " + path.string()));
  }
  std::size_t offset = expected_magic.size();
  const std::uint32_t version = readU32(bytes.subspan(offset, 4U));
  offset += 4U;
  const std::uint64_t payload_size = readU64(bytes.subspan(offset, 8U));
  offset += 8U;
  core::ContentHash checksum{};
  for (std::size_t index = 0U; index < checksum.size(); ++index) {
    checksum[index] = static_cast<std::uint8_t>(bytes[offset + index]);
  }
  offset += checksum.size();
  if (version != expected_format_version || payload_size > maximum_payload_bytes ||
      payload_size != bytes.size() - offset) {
    return Result::failure(error(ErrorCode::InvalidRecord,
                                 "durable record version or payload length is invalid: " +
                                     path.string()));
  }
  const auto payload = bytes.subspan(offset, static_cast<std::size_t>(payload_size));
  if (hashBytes(payload) != checksum) {
    return Result::failure(error(ErrorCode::ChecksumMismatch,
                                 "durable record checksum mismatch: " + path.string()));
  }
  return Result::success(DecodedFrame{checksum, payload});
}

core::Result<Bytes, Error> readFile(const std::filesystem::path& path,
                                    std::uint64_t maximum_bytes) {
  using Result = core::Result<Bytes, Error>;
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    return Result::failure(error(ErrorCode::IoFailure,
                                 "durable record open failed: " + path.string() + ": " +
                                     std::strerror(errno)));
  }
  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    const int saved_errno = errno;
    (void)::close(descriptor);
    return Result::failure(error(ErrorCode::IoFailure,
                                 "durable record metadata read failed: " + path.string() +
                                     ": " + std::strerror(saved_errno)));
  }
  if (!S_ISREG(status.st_mode) || status.st_size < 0) {
    (void)::close(descriptor);
    return Result::failure(
        error(ErrorCode::InvalidRecord, "durable record is not a regular file: " + path.string()));
  }
  if (static_cast<std::uint64_t>(status.st_size) > maximum_bytes ||
      static_cast<std::uint64_t>(status.st_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    (void)::close(descriptor);
    return Result::failure(error(ErrorCode::SizeMismatch,
                                 "durable record exceeds its byte bound: " + path.string()));
  }
  Bytes bytes(static_cast<std::size_t>(status.st_size));
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t count = ::read(descriptor, bytes.data() + static_cast<std::ptrdiff_t>(offset),
                                 bytes.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      const int saved_errno = errno;
      (void)::close(descriptor);
      return Result::failure(error(ErrorCode::IoFailure,
                                   "durable record read failed: " + path.string() + ": " +
                                       std::strerror(saved_errno)));
    }
    offset += static_cast<std::size_t>(count);
  }
  if (::close(descriptor) != 0) {
    return Result::failure(
        error(ErrorCode::IoFailure, "durable record close failed: " + path.string()));
  }
  return Result::success(std::move(bytes));
}

std::optional<Error> syncDirectory(const std::filesystem::path& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    return error(ErrorCode::IoFailure,
                 "directory open failed for fsync: " + path.string() + ": " +
                     std::strerror(errno));
  }
  if (::fsync(descriptor) != 0) {
    const int saved_errno = errno;
    (void)::close(descriptor);
    return error(ErrorCode::IoFailure,
                 "directory fsync failed for " + path.string() + ": " +
                     std::strerror(saved_errno));
  }
  if (::close(descriptor) != 0) {
    return error(ErrorCode::IoFailure,
                 "directory close failed after fsync: " + path.string());
  }
  return std::nullopt;
}

std::optional<Error> ensureDirectory(const std::filesystem::path& path, std::string_view purpose) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(path, filesystem_error);
  const auto status = std::filesystem::symlink_status(path, filesystem_error);
  if (filesystem_error || !std::filesystem::is_directory(status) ||
      std::filesystem::is_symlink(status)) {
    return error(ErrorCode::IoFailure, "cannot create or validate " + std::string(purpose) + " " +
                                           path.string() + ": " + filesystem_error.message());
  }
  if (::chmod(path.c_str(), S_IRWXU) != 0) {
    return error(ErrorCode::IoFailure, "cannot restrict " + std::string(purpose) +
                                           " permissions: " + path.string() + ": " +
                                           std::strerror(errno));
  }
  return syncDirectory(path);
}

std::optional<Error> cleanTemporaryFiles(const std::filesystem::path& directory,
                                         std::size_t* recovered) {
  std::error_code filesystem_error;
  bool changed = false;
  for (std::filesystem::directory_iterator iterator(directory, filesystem_error), end;
       iterator != end && !filesystem_error; iterator.increment(filesystem_error)) {
    const std::string filename = iterator->path().filename().string();
    if (filename.find(".tmp.") == std::string::npos) {
      continue;
    }
    const auto status = iterator->symlink_status(filesystem_error);
    if (filesystem_error || !std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status)) {
      break;
    }
    std::filesystem::remove(iterator->path(), filesystem_error);
    if (filesystem_error) {
      break;
    }
    ++*recovered;
    changed = true;
  }
  if (filesystem_error) {
    return error(ErrorCode::IoFailure, "temporary recovery scan failed in " + directory.string() +
                                           ": " + filesystem_error.message());
  }
  return changed ? syncDirectory(directory) : std::nullopt;
}

void setAtomicWriteFailpointForTesting(AtomicWriteFailpoint failpoint) noexcept {
  g_atomic_write_failpoint.store(failpoint, std::memory_order_release);
}

core::Result<DurableWriteResult, Error> durableWriteNoReplace(
    const std::filesystem::path& destination, std::span<const std::byte> bytes,
    std::uint64_t* temporary_counter) {
  using Result = core::Result<DurableWriteResult, Error>;
  const auto begin = Clock::now();
  const auto reconcile = [&]() -> Result {
    const auto existing = readFile(destination, bytes.size());
    if (!existing) {
      if (existing.error().code == ErrorCode::SizeMismatch) {
        return Result::failure(error(ErrorCode::Conflict,
                                     "immutable destination has a different byte length: " +
                                         destination.string()));
      }
      return Result::failure(error(ErrorCode::AmbiguousIntegrity,
                                   "immutable destination exists but cannot be verified: " +
                                       destination.string() + ": " + existing.error().detail));
    }
    if (existing.value().size() != bytes.size() ||
        !std::equal(existing.value().begin(), existing.value().end(), bytes.begin())) {
      return Result::failure(error(ErrorCode::Conflict,
                                   "immutable destination contains different bytes: " +
                                       destination.string()));
    }
    if (const auto sync_error = syncDirectory(destination.parent_path())) {
      return Result::failure(error(ErrorCode::AmbiguousIntegrity,
                                   "identical immutable destination could not be made durable: " +
                                       destination.string() + ": " + sync_error->detail));
    }
    return Result::success(DurableWriteResult{DurableWriteDisposition::ExistingIdentical,
                                               elapsedMicroseconds(begin)});
  };

  std::error_code filesystem_error;
  if (std::filesystem::exists(destination, filesystem_error)) {
    if (filesystem_error) {
      return Result::failure(error(ErrorCode::AmbiguousIntegrity,
                                   "cannot inspect immutable destination: " +
                                       destination.string() + ": " + filesystem_error.message()));
    }
    return reconcile();
  }
  if (filesystem_error) {
    return Result::failure(error(ErrorCode::IoFailure,
                                 "cannot inspect durable destination: " + destination.string() +
                                     ": " + filesystem_error.message()));
  }

  const auto directory = destination.parent_path();
  const auto temporary =
      directory / (destination.filename().string() + ".tmp." +
                   std::to_string(static_cast<unsigned long long>(::getpid())) + "." +
                   std::to_string((*temporary_counter)++));
  const int descriptor = ::open(temporary.c_str(),
                                O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    return Result::failure(error(ErrorCode::IoFailure,
                                 "temporary durable record open failed: " + temporary.string() +
                                     ": " + std::strerror(errno)));
  }
  std::size_t offset = 0U;
  while (offset < bytes.size()) {
    const ssize_t count = ::write(descriptor, bytes.data() + static_cast<std::ptrdiff_t>(offset),
                                  bytes.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      const int saved_errno = errno;
      (void)::close(descriptor);
      (void)::unlink(temporary.c_str());
      return Result::failure(error(ErrorCode::IoFailure,
                                   "temporary durable record write failed: " +
                                       temporary.string() + ": " + std::strerror(saved_errno)));
    }
    offset += static_cast<std::size_t>(count);
  }
  if (::fsync(descriptor) != 0) {
    const int saved_errno = errno;
    (void)::close(descriptor);
    (void)::unlink(temporary.c_str());
    return Result::failure(error(ErrorCode::IoFailure,
                                 "temporary durable record fsync failed: " + temporary.string() +
                                     ": " + std::strerror(saved_errno)));
  }
  if (::close(descriptor) != 0) {
    (void)::unlink(temporary.c_str());
    return Result::failure(error(ErrorCode::IoFailure,
                                 "temporary durable record close failed: " + temporary.string()));
  }
  const long rename_result = ::syscall(SYS_renameat2, AT_FDCWD, temporary.c_str(), AT_FDCWD,
                                       destination.c_str(), RENAME_NOREPLACE);
  if (rename_result != 0) {
    const int saved_errno = errno;
    (void)::unlink(temporary.c_str());
    if (saved_errno == EEXIST) {
      return reconcile();
    }
    return Result::failure(error(ErrorCode::IoFailure,
                                 "atomic durable record rename failed: " +
                                     destination.string() + ": " + std::strerror(saved_errno)));
  }
  if (g_atomic_write_failpoint.exchange(AtomicWriteFailpoint::None,
                                        std::memory_order_acq_rel) ==
      AtomicWriteFailpoint::AfterRenameBeforeDirectorySync) {
    return Result::failure(error(
        ErrorCode::AmbiguousIntegrity,
        "injected crash window after rename and before destination directory fsync: " +
            destination.string()));
  }
  if (const auto sync_error = syncDirectory(directory)) {
    return Result::failure(error(ErrorCode::AmbiguousIntegrity,
                                 "rename succeeded but destination directory fsync failed: " +
                                     destination.string() + ": " + sync_error->detail));
  }
  return Result::success(
      DurableWriteResult{DurableWriteDisposition::Written, elapsedMicroseconds(begin)});
}

}  // namespace meridian::global::persistence_internal
