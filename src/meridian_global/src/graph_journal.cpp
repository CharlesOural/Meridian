#include "meridian/global/graph_journal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>

#include "persistence_internal.hpp"

namespace meridian::global {
namespace {

using Bytes = std::vector<std::byte>;
using Clock = std::chrono::steady_clock;

constexpr std::array<std::byte, 8> kPreparedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                  std::byte{'N'}, std::byte{'G'}, std::byte{'P'},
                                                  std::byte{'R'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kCommittedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                   std::byte{'N'}, std::byte{'G'}, std::byte{'C'},
                                                   std::byte{'M'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kLegacyCheckpointMagic{
    std::byte{'M'}, std::byte{'R'}, std::byte{'D'}, std::byte{'N'},
    std::byte{'G'}, std::byte{'C'}, std::byte{'P'}, std::byte{'1'}};
constexpr std::uint32_t kFormatVersion = 5U;
constexpr std::uint32_t kLegacyFormatVersion = 4U;
constexpr std::size_t kFrameOverhead = persistence_internal::kFrameOverhead;
constexpr std::uint64_t kMaximumSmallRecordBytes = 64ULL * 1024ULL * 1024ULL;

[[nodiscard]] GraphJournalError makeError(
    GraphJournalErrorCode code, std::string detail,
    std::optional<GlobalGraphRevision> revision = std::nullopt) {
  return GraphJournalError{code, revision, std::move(detail)};
}

[[nodiscard]] std::int64_t elapsedMicroseconds(Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count();
}

[[nodiscard]] bool zeroHash(const core::ContentHash& hash) noexcept {
  return std::all_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte == 0U; });
}

[[nodiscard]] core::ContentHash hashBytes(std::span<const std::byte> bytes) noexcept {
  return persistence_internal::hashBytes(bytes);
}

[[nodiscard]] bool sha256SelfTest() noexcept {
  return persistence_internal::sha256SelfTest();
}

[[nodiscard]] std::string hashHex(const core::ContentHash& hash) {
  return persistence_internal::hashHex(hash);
}

[[nodiscard]] std::string committedFilename(GlobalGraphRevision revision) {
  std::array<char, 64> buffer{};
  const int count = std::snprintf(buffer.data(), buffer.size(), "%020llu.committed",
                                  static_cast<unsigned long long>(revision.value()));
  if (count <= 0 || static_cast<std::size_t>(count) >= buffer.size()) {
    return {};
  }
  return std::string(buffer.data(), static_cast<std::size_t>(count));
}

class Writer {
public:
  void raw(std::span<const std::byte> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
  void u64(std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
      bytes_.push_back(static_cast<std::byte>(value >> ((7U - index) * 8U)));
    }
  }
  void hash(const core::ContentHash& value) { raw(std::as_bytes(std::span(value))); }
  template <typename Tag>
  void id(core::StrongId<Tag> value) {
    u64(value.value());
  }
  void boolean(bool value) { u8(value ? 1U : 0U); }
  [[nodiscard]] Bytes take() && { return std::move(bytes_); }

private:
  Bytes bytes_;
};

class Reader {
public:
  explicit Reader(std::span<const std::byte> bytes, std::size_t maximum_elements)
      : bytes_(bytes), maximum_elements_(maximum_elements) {}

  [[nodiscard]] bool raw(std::span<std::byte> output) noexcept {
    if (output.size() > remaining()) {
      valid_ = false;
      return false;
    }
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), output.size(),
                output.begin());
    offset_ += output.size();
    return true;
  }
  [[nodiscard]] std::uint8_t u8() noexcept {
    if (remaining() < 1U) {
      valid_ = false;
      return 0U;
    }
    return static_cast<std::uint8_t>(bytes_[offset_++]);
  }
  [[nodiscard]] std::uint64_t u64() noexcept {
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < 8U; ++index) {
      value = (value << 8U) | u8();
    }
    return value;
  }
  [[nodiscard]] core::ContentHash hash() noexcept {
    core::ContentHash value{};
    (void)raw(std::as_writable_bytes(std::span(value)));
    return value;
  }
  template <typename Id>
  [[nodiscard]] Id id() noexcept {
    return Id(u64());
  }
  [[nodiscard]] bool boolean() noexcept {
    const std::uint8_t value = u8();
    if (value > 1U) {
      valid_ = false;
    }
    return value == 1U;
  }
  [[nodiscard]] std::size_t count(std::size_t explicit_maximum,
                                  std::size_t minimum_element_bytes) noexcept {
    const std::uint64_t value = u64();
    const std::size_t maximum = std::min(maximum_elements_, explicit_maximum);
    if (minimum_element_bytes == 0U || value > maximum || value > remaining_element_budget_ ||
        value > remaining() / minimum_element_bytes ||
        value > std::numeric_limits<std::size_t>::max()) {
      valid_ = false;
      return 0U;
    }
    const auto count = static_cast<std::size_t>(value);
    remaining_element_budget_ -= count;
    return count;
  }
  void invalidate() noexcept { valid_ = false; }
  [[nodiscard]] bool complete() const noexcept { return valid_ && offset_ == bytes_.size(); }
  [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

private:
  std::span<const std::byte> bytes_;
  std::size_t maximum_elements_{};
  std::size_t remaining_element_budget_{maximum_elements_};
  std::size_t offset_{};
  bool valid_{true};
};

struct DecodedFrame {
  std::span<const std::byte> payload;
  core::ContentHash checksum{};
};

[[nodiscard]] Bytes frameBytes(std::span<const std::byte> magic,
                               std::span<const std::byte> payload) {
  return persistence_internal::frameBytes(magic, kFormatVersion, payload);
}

[[nodiscard]] std::uint32_t frameVersion(std::span<const std::byte> bytes) noexcept {
  if (bytes.size() < 12U) {
    return 0U;
  }
  std::uint32_t version = 0U;
  for (std::size_t index = 8U; index < 12U; ++index) {
    version = (version << 8U) | static_cast<std::uint8_t>(bytes[index]);
  }
  return version;
}

[[nodiscard]] core::Result<DecodedFrame, GraphJournalError> decodeFrame(
    std::span<const std::byte> bytes, std::span<const std::byte> expected_magic,
    std::uint64_t maximum_payload_bytes, const std::filesystem::path& path) {
  using Result = core::Result<DecodedFrame, GraphJournalError>;
  if (bytes.size() >= expected_magic.size() + sizeof(std::uint32_t) &&
      std::equal(expected_magic.begin(), expected_magic.end(), bytes.begin()) &&
      frameVersion(bytes) != kFormatVersion) {
    const bool legacy = frameVersion(bytes) == kLegacyFormatVersion;
    return Result::failure(makeError(
        legacy ? GraphJournalErrorCode::MigrationRequired
               : GraphJournalErrorCode::UnsupportedFormat,
        legacy ? "legacy graph journal record requires explicit migration: " + path.string()
               : "graph journal record format version is unsupported: " + path.string()));
  }
  const auto decoded = persistence_internal::decodeFrame(bytes, expected_magic, kFormatVersion,
                                                         maximum_payload_bytes, path);
  if (!decoded) {
    const auto code = decoded.error().code == persistence_internal::ErrorCode::ChecksumMismatch
                          ? GraphJournalErrorCode::ChecksumMismatch
                          : GraphJournalErrorCode::RecoveryCorruption;
    return Result::failure(makeError(code, decoded.error().detail));
  }
  return Result::success(DecodedFrame{decoded.value().payload, decoded.value().checksum});
}

[[nodiscard]] bool validInputKind(GraphJournalInputKind kind) noexcept {
  switch (kind) {
    case GraphJournalInputKind::SparseSubmapSeal:
    case GraphJournalInputKind::AdjacentConstraint:
    case GraphJournalInputKind::GnssObservation:
    case GraphJournalInputKind::LoopProposal:
    case GraphJournalInputKind::SurveyedAlignment:
    case GraphJournalInputKind::OperatorAction:
      return true;
  }
  return false;
}

[[nodiscard]] bool validDisposition(GraphJournalDisposition disposition) noexcept {
  switch (disposition) {
    case GraphJournalDisposition::Admitted:
    case GraphJournalDisposition::Rejected:
    case GraphJournalDisposition::Deferred:
    case GraphJournalDisposition::Retained:
    case GraphJournalDisposition::Removed:
      return true;
  }
  return false;
}

[[nodiscard]] bool inputLess(const GraphJournalInput& left,
                             const GraphJournalInput& right) noexcept {
  return std::tie(left.kind, left.source_id) < std::tie(right.kind, right.source_id);
}

[[nodiscard]] std::optional<GraphJournalError> canonicalizeInputs(
    std::vector<GraphJournalInput>* inputs, std::size_t maximum_inputs,
    std::optional<GlobalGraphRevision> revision) {
  if (inputs->empty() || inputs->size() > maximum_inputs) {
    return makeError(GraphJournalErrorCode::InvalidTransaction,
                     "transaction input set is empty or exceeds the configured bound", revision);
  }
  for (const GraphJournalInput& input : *inputs) {
    if (!validInputKind(input.kind) || !validDisposition(input.disposition) ||
        input.source_id == std::numeric_limits<std::uint64_t>::max() ||
        zeroHash(input.content_checksum)) {
      return makeError(GraphJournalErrorCode::InvalidTransaction,
                       "transaction contains an invalid input identity, checksum, or disposition",
                       revision);
    }
  }
  std::sort(inputs->begin(), inputs->end(), inputLess);
  for (std::size_t index = 1U; index < inputs->size(); ++index) {
    if (inputs->at(index - 1U).kind == inputs->at(index).kind &&
        inputs->at(index - 1U).source_id == inputs->at(index).source_id) {
      return makeError(GraphJournalErrorCode::InvalidTransaction,
                       "transaction input identities must be unique", revision);
    }
  }
  return std::nullopt;
}

void writeRevisionAndParent(Writer& writer, GlobalGraphRevision revision,
                            std::optional<GlobalGraphRevision> parent) {
  writer.id(revision);
  writer.boolean(parent.has_value());
  if (parent) {
    writer.id(*parent);
  }
}

[[nodiscard]] std::pair<GlobalGraphRevision, std::optional<GlobalGraphRevision>>
readRevisionAndParent(Reader& reader) {
  const GlobalGraphRevision revision = reader.id<GlobalGraphRevision>();
  std::optional<GlobalGraphRevision> parent;
  if (reader.boolean()) {
    parent = reader.id<GlobalGraphRevision>();
  }
  return {revision, parent};
}

void writeInput(Writer& writer, const GraphJournalInput& input) {
  writer.u8(static_cast<std::uint8_t>(input.kind));
  writer.u64(input.source_id);
  writer.hash(input.content_checksum);
  writer.u8(static_cast<std::uint8_t>(input.disposition));
}

[[nodiscard]] GraphJournalInput readInput(Reader& reader) {
  const std::uint8_t kind = reader.u8();
  const std::uint64_t source_id = reader.u64();
  const core::ContentHash checksum = reader.hash();
  const std::uint8_t disposition = reader.u8();
  if (kind > static_cast<std::uint8_t>(GraphJournalInputKind::OperatorAction) ||
      disposition > static_cast<std::uint8_t>(GraphJournalDisposition::Removed)) {
    reader.invalidate();
  }
  return GraphJournalInput{static_cast<GraphJournalInputKind>(kind), source_id, checksum,
                           static_cast<GraphJournalDisposition>(disposition)};
}

struct PreparedPayload {
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  core::ContentHash checkpoint_checksum{};
  std::uint64_t checkpoint_bytes{};
  std::vector<GraphJournalInput> inputs;
};

void writePreparedPayload(Writer& writer, const PreparedPayload& prepared) {
  writeRevisionAndParent(writer, prepared.revision, prepared.parent);
  writer.hash(prepared.checkpoint_checksum);
  writer.u64(prepared.checkpoint_bytes);
  writer.u64(prepared.inputs.size());
  for (const GraphJournalInput& input : prepared.inputs) {
    writeInput(writer, input);
  }
}

[[nodiscard]] PreparedPayload readPreparedPayload(Reader& reader,
                                                  const GraphJournalConfig& config) {
  const auto [revision, parent] = readRevisionAndParent(reader);
  const core::ContentHash checkpoint_checksum = reader.hash();
  const std::uint64_t checkpoint_bytes = reader.u64();
  const std::size_t input_count = reader.count(config.maximum_inputs_per_transaction, 42U);
  std::vector<GraphJournalInput> inputs;
  inputs.reserve(input_count);
  for (std::size_t index = 0U; index < input_count; ++index) {
    inputs.push_back(readInput(reader));
  }
  return PreparedPayload{revision, parent, checkpoint_checksum, checkpoint_bytes,
                         std::move(inputs)};
}

struct CommittedPayload {
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  core::ContentHash checkpoint_checksum{};
  core::ContentHash prepared_checksum{};
};

void writeCommittedPayload(Writer& writer, const CommittedPayload& committed) {
  writeRevisionAndParent(writer, committed.revision, committed.parent);
  writer.hash(committed.checkpoint_checksum);
  writer.hash(committed.prepared_checksum);
}

[[nodiscard]] CommittedPayload readCommittedPayload(Reader& reader) {
  const auto [revision, parent] = readRevisionAndParent(reader);
  return CommittedPayload{revision, parent, reader.hash(), reader.hash()};
}

[[nodiscard]] bool checkpointSchemaError(GlobalGraphCheckpointErrorCode code) noexcept {
  return code == GlobalGraphCheckpointErrorCode::UnsupportedSchema ||
         code == GlobalGraphCheckpointErrorCode::UnsupportedKeySchema ||
         code == GlobalGraphCheckpointErrorCode::UnsupportedConfigurationSchema;
}

struct EncodedCheckpoint {
  Bytes bytes;
  core::ContentHash checksum{};
};

[[nodiscard]] core::Result<EncodedCheckpoint, GraphJournalError> serializeCheckpoint(
    const GlobalGraphCheckpoint& checkpoint, const GraphJournalConfig& config,
    GraphJournalErrorCode invalid_code = GraphJournalErrorCode::InvalidCheckpoint) {
  using Result = core::Result<EncodedCheckpoint, GraphJournalError>;
  auto encoded = encodeGlobalGraphCheckpoint(checkpoint, config.checkpoint_limits);
  if (!encoded) {
    GraphJournalErrorCode code = invalid_code;
    if (checkpointSchemaError(encoded.error().code)) {
      code = GraphJournalErrorCode::MigrationRequired;
    } else if (encoded.error().code == GlobalGraphCheckpointErrorCode::CapacityExceeded ||
               encoded.error().code == GlobalGraphCheckpointErrorCode::InvalidLimits) {
      code = GraphJournalErrorCode::CapacityExceeded;
    }
    return Result::failure(makeError(
        code, "canonical global graph checkpoint encoding failed: " + encoded.error().detail,
        checkpoint.revision.valid() ? std::optional(checkpoint.revision) : std::nullopt));
  }
  EncodedCheckpoint result;
  result.bytes.assign(encoded.value().bytes().begin(), encoded.value().bytes().end());
  result.checksum = encoded.value().digest();
  return Result::success(std::move(result));
}

[[nodiscard]] core::Result<GlobalGraphCheckpoint, GraphJournalError> deserializeCheckpoint(
    std::span<const std::byte> bytes, const GraphJournalConfig& config,
    const std::filesystem::path& path) {
  using Result = core::Result<GlobalGraphCheckpoint, GraphJournalError>;
  if (bytes.size() >= kLegacyCheckpointMagic.size() &&
      std::equal(kLegacyCheckpointMagic.begin(), kLegacyCheckpointMagic.end(), bytes.begin())) {
    return Result::failure(
        makeError(GraphJournalErrorCode::MigrationRequired,
                  "legacy publication-snapshot checkpoint requires explicit journal migration: " +
                      path.string()));
  }
  auto decoded = decodeGlobalGraphCheckpoint(bytes, config.checkpoint_limits);
  if (!decoded) {
    GraphJournalErrorCode code = GraphJournalErrorCode::RecoveryCorruption;
    if (checkpointSchemaError(decoded.error().code)) {
      code = GraphJournalErrorCode::MigrationRequired;
    } else if (decoded.error().code == GlobalGraphCheckpointErrorCode::CapacityExceeded ||
               decoded.error().code == GlobalGraphCheckpointErrorCode::InvalidLimits) {
      code = GraphJournalErrorCode::CapacityExceeded;
    } else if (decoded.error().code == GlobalGraphCheckpointErrorCode::ChecksumMismatch) {
      code = GraphJournalErrorCode::ChecksumMismatch;
    }
    return Result::failure(makeError(
        code, "canonical global graph checkpoint recovery failed: " + decoded.error().detail));
  }
  return Result::success(std::move(decoded).value());
}

[[nodiscard]] Bytes serializePrepared(const PreparedPayload& prepared,
                                      core::ContentHash* checksum) {
  Writer payload_writer;
  writePreparedPayload(payload_writer, prepared);
  const Bytes payload = std::move(payload_writer).take();
  *checksum = hashBytes(payload);
  return frameBytes(kPreparedMagic, payload);
}

[[nodiscard]] Bytes serializeCommitted(const CommittedPayload& committed) {
  Writer payload_writer;
  writeCommittedPayload(payload_writer, committed);
  const Bytes payload = std::move(payload_writer).take();
  return frameBytes(kCommittedMagic, payload);
}

[[nodiscard]] std::optional<GraphJournalError> syncDirectory(const std::filesystem::path& path) {
  const auto result = persistence_internal::syncDirectory(path);
  return result ? std::optional(makeError(GraphJournalErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> ensureDirectory(const std::filesystem::path& path) {
  const auto result = persistence_internal::ensureDirectory(path, "graph journal directory");
  return result ? std::optional(makeError(GraphJournalErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] core::Result<Bytes, GraphJournalError> readFile(const std::filesystem::path& path,
                                                              std::uint64_t maximum_bytes) {
  using Result = core::Result<Bytes, GraphJournalError>;
  const auto result = persistence_internal::readFile(path, maximum_bytes);
  if (!result) {
    const auto code = (result.error().code == persistence_internal::ErrorCode::InvalidRecord ||
                       result.error().code == persistence_internal::ErrorCode::SizeMismatch)
                          ? GraphJournalErrorCode::RecoveryCorruption
                          : GraphJournalErrorCode::IoFailure;
    return Result::failure(makeError(code, result.error().detail));
  }
  return Result::success(result.value());
}

enum class DurableWriteDisposition {
  Written,
  ExistingIdentical,
};

struct DurableWriteResult {
  DurableWriteDisposition disposition{DurableWriteDisposition::Written};
  std::int64_t elapsed_us{};
};

[[nodiscard]] core::Result<DurableWriteResult, GraphJournalError> durableWriteNoReplace(
    const std::filesystem::path& destination, std::span<const std::byte> bytes,
    std::uint64_t* temporary_counter) {
  using Result = core::Result<DurableWriteResult, GraphJournalError>;
  const auto result =
      persistence_internal::durableWriteNoReplace(destination, bytes, temporary_counter);
  if (!result) {
    GraphJournalErrorCode code = GraphJournalErrorCode::IoFailure;
    if (result.error().code == persistence_internal::ErrorCode::Conflict) {
      code = GraphJournalErrorCode::CommitChecksumConflict;
    } else if (result.error().code == persistence_internal::ErrorCode::AmbiguousIntegrity) {
      code = GraphJournalErrorCode::IntegrityFailure;
    }
    return Result::failure(makeError(code, result.error().detail));
  }
  return Result::success(DurableWriteResult{
      result.value().disposition == persistence_internal::DurableWriteDisposition::Written
          ? DurableWriteDisposition::Written
          : DurableWriteDisposition::ExistingIdentical,
      result.value().elapsed_us});
}

[[nodiscard]] std::optional<GraphJournalError> cleanTemporaryFiles(
    const std::filesystem::path& directory, std::size_t* recovered) {
  const auto result = persistence_internal::cleanTemporaryFiles(directory, recovered);
  return result ? std::optional(makeError(GraphJournalErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] std::vector<std::filesystem::path> filesWithExtension(
    const std::filesystem::path& directory, std::string_view extension, std::size_t* scanned,
    std::optional<GraphJournalError>* output_error) {
  std::vector<std::filesystem::path> files;
  std::error_code filesystem_error;
  for (std::filesystem::directory_iterator iterator(directory, filesystem_error), end;
       iterator != end && !filesystem_error; iterator.increment(filesystem_error)) {
    ++*scanned;
    const auto status = iterator->symlink_status(filesystem_error);
    if (filesystem_error) {
      break;
    }
    if (!std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status) ||
        iterator->path().extension() != extension) {
      *output_error =
          makeError(GraphJournalErrorCode::RecoveryCorruption,
                    "unexpected graph journal directory entry: " + iterator->path().string());
      return {};
    }
    files.push_back(iterator->path());
  }
  if (filesystem_error) {
    *output_error = makeError(
        GraphJournalErrorCode::IoFailure,
        "graph journal scan failed in " + directory.string() + ": " + filesystem_error.message());
    return {};
  }
  std::sort(files.begin(), files.end());
  return files;
}

[[nodiscard]] bool checkedAdd(std::uint64_t left, std::uint64_t right,
                              std::uint64_t* output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

}  // namespace

struct GraphJournal::Impl {
  struct StoredCheckpoint {
    GlobalGraphCheckpoint checkpoint;
    core::ContentHash checksum{};
    std::uint64_t canonical_bytes{};
    std::filesystem::path path;
  };

  struct StoredPrepared {
    PreparedPayload payload;
    PreparedGraphCommit token;
    std::uint64_t framed_bytes{};
    std::uint64_t marker_bytes{};
    std::filesystem::path path;
  };

  explicit Impl(GraphJournalConfig input_config)
      : config(std::move(input_config)),
        checkpoints_directory(config.root_directory / "checkpoints"),
        prepared_directory(config.root_directory / "prepared"),
        committed_directory(config.root_directory / "committed") {}

  [[nodiscard]] std::uint64_t reservedMarkerBytes() const noexcept {
    std::uint64_t result = 0U;
    for (const auto& [checksum, prepared] : prepared_by_checksum) {
      if (!committed_prepared_checksums.contains(checksum)) {
        if (!checkedAdd(result, prepared.marker_bytes, &result)) {
          return std::numeric_limits<std::uint64_t>::max();
        }
      }
    }
    return result;
  }

  [[nodiscard]] GraphJournalStatus currentStatus() const noexcept {
    GraphJournalStatus output;
    if (!commits.empty()) {
      output.oldest_committed_revision = GlobalGraphRevision(commits.begin()->first);
      output.committed_head = GlobalGraphRevision(commits.rbegin()->first);
    }
    output.committed_revisions = commits.size();
    output.prepared_transactions = prepared_by_checksum.size();
    output.uncommitted_prepared_transactions =
        prepared_by_checksum.size() - committed_prepared_checksums.size();
    output.durable_bytes = durable_bytes;
    output.reserved_commit_marker_bytes = reservedMarkerBytes();
    output.maximum_bytes = config.maximum_bytes;
    output.recovered_temporary_files = recovered_temporary_files;
    output.recovery_scanned_files = recovery_scanned_files;
    output.retired_unreachable_files = retired_unreachable_files;
    output.integrity_detail = integrity_detail;
    output.last_timing = last_timing;
    if (integrity_failure) {
      output.state = GraphJournalState::ReadOnlyIntegrityFailure;
    } else if (io_failure) {
      output.state = GraphJournalState::ReadOnlyIoFailure;
    } else if (capacity_failure || commits.size() >= config.maximum_revisions ||
               prepared_by_checksum.size() >= config.maximum_prepared_transactions ||
               output.reserved_commit_marker_bytes == std::numeric_limits<std::uint64_t>::max() ||
               output.durable_bytes > config.maximum_bytes ||
               output.reserved_commit_marker_bytes >
                   config.maximum_bytes - std::min(output.durable_bytes, config.maximum_bytes)) {
      output.state = GraphJournalState::ReadOnlyCapacityDegraded;
    }
    return output;
  }

  GraphJournalConfig config;
  std::filesystem::path checkpoints_directory;
  std::filesystem::path prepared_directory;
  std::filesystem::path committed_directory;
  std::map<core::ContentHash, StoredCheckpoint> checkpoints_by_checksum;
  std::map<core::ContentHash, StoredPrepared> prepared_by_checksum;
  std::map<std::uint64_t, GraphJournalCommitRecord> commits;
  std::set<core::ContentHash> committed_prepared_checksums;
  std::uint64_t durable_bytes{};
  std::uint64_t temporary_counter{};
  std::size_t recovered_temporary_files{};
  std::size_t recovery_scanned_files{};
  std::size_t retired_unreachable_files{};
  std::set<std::string> committed_checkpoint_filenames;
  std::set<std::string> committed_prepared_filenames;
  std::size_t verified_committed_markers{};
  std::optional<std::string> integrity_detail;
  GraphJournalTiming last_timing;
  bool capacity_failure{false};
  bool io_failure{false};
  bool integrity_failure{false};
};

namespace {
[[nodiscard]] std::optional<GraphJournalError> validateConfig(const GraphJournalConfig& config);
[[nodiscard]] std::optional<GraphJournalError> addDurableBytes(GraphJournal::Impl& impl,
                                                               std::uint64_t bytes);
[[nodiscard]] std::optional<GraphJournalError> scanCommittedReferences(GraphJournal::Impl& impl);
[[nodiscard]] std::optional<GraphJournalError> recoverCheckpoints(GraphJournal::Impl& impl);
[[nodiscard]] std::optional<GraphJournalError> recoverPrepared(GraphJournal::Impl& impl);
[[nodiscard]] std::optional<GraphJournalError> recoverCommitted(GraphJournal::Impl& impl);
[[nodiscard]] std::optional<GraphJournalError> retireUnreachable(GraphJournal::Impl& impl);
[[nodiscard]] bool sameInputs(std::span<const GraphJournalInput> left,
                              std::span<const GraphJournalInput> right) noexcept;
[[nodiscard]] std::optional<GraphJournalError> verifyCommitRecord(
    const GraphJournalCommitRecord& record, const GraphJournalConfig& config);
}  // namespace

core::Result<std::unique_ptr<GraphJournal>, GraphJournalError> GraphJournal::open(
    GraphJournalConfig config) {
  using Result = core::Result<std::unique_ptr<GraphJournal>, GraphJournalError>;
  if (const auto config_error = validateConfig(config)) {
    return Result::failure(*config_error);
  }
  const auto recovery_begin = Clock::now();
  auto impl = std::make_unique<Impl>(std::move(config));
  if (const auto root_error = ensureDirectory(impl->config.root_directory)) {
    return Result::failure(*root_error);
  }
  if (const auto checkpoint_error = ensureDirectory(impl->checkpoints_directory)) {
    return Result::failure(*checkpoint_error);
  }
  if (const auto prepared_error = ensureDirectory(impl->prepared_directory)) {
    return Result::failure(*prepared_error);
  }
  if (const auto committed_error = ensureDirectory(impl->committed_directory)) {
    return Result::failure(*committed_error);
  }
  if (const auto root_sync_error = syncDirectory(impl->config.root_directory)) {
    return Result::failure(*root_sync_error);
  }
  for (const std::filesystem::path& directory :
       {impl->checkpoints_directory, impl->prepared_directory, impl->committed_directory}) {
    if (const auto cleanup_error =
            cleanTemporaryFiles(directory, &impl->recovered_temporary_files)) {
      return Result::failure(*cleanup_error);
    }
  }
  if (const auto marker_error = scanCommittedReferences(*impl)) {
    return Result::failure(*marker_error);
  }
  if (const auto checkpoint_error = recoverCheckpoints(*impl)) {
    return Result::failure(*checkpoint_error);
  }
  if (const auto prepared_error = recoverPrepared(*impl)) {
    return Result::failure(*prepared_error);
  }
  if (const auto committed_error = recoverCommitted(*impl)) {
    return Result::failure(*committed_error);
  }
  if (const auto retirement_error = retireUnreachable(*impl)) {
    return Result::failure(*retirement_error);
  }
  impl->last_timing.recovery_scan_us = elapsedMicroseconds(recovery_begin);
  if (impl->commits.size() >= impl->config.maximum_revisions ||
      impl->prepared_by_checksum.size() >= impl->config.maximum_prepared_transactions ||
      impl->durable_bytes > impl->config.maximum_bytes ||
      impl->reservedMarkerBytes() >
          impl->config.maximum_bytes - std::min(impl->durable_bytes, impl->config.maximum_bytes)) {
    impl->capacity_failure = true;
  }
  return Result::success(std::unique_ptr<GraphJournal>(new GraphJournal(std::move(impl))));
}

GraphJournal::GraphJournal(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GraphJournal::~GraphJournal() = default;
GraphJournal::GraphJournal(GraphJournal&&) noexcept = default;
GraphJournal& GraphJournal::operator=(GraphJournal&&) noexcept = default;

core::Result<PreparedGraphCommit, GraphJournalError> GraphJournal::prepare(
    GraphJournalPrepareRequest request) {
  using Result = core::Result<PreparedGraphCommit, GraphJournalError>;
  if (request.expected_parent != request.checkpoint.parent) {
    return Result::failure(makeError(GraphJournalErrorCode::InvalidTransaction,
                                     "prepare expected parent does not equal the checkpoint parent",
                                     request.checkpoint.revision));
  }
  if (const auto input_error =
          canonicalizeInputs(&request.inputs, impl_->config.maximum_inputs_per_transaction,
                             request.checkpoint.revision)) {
    return Result::failure(*input_error);
  }

  auto serialized_checkpoint = serializeCheckpoint(request.checkpoint, impl_->config);
  if (!serialized_checkpoint) {
    if (serialized_checkpoint.error().code == GraphJournalErrorCode::CapacityExceeded) {
      impl_->capacity_failure = true;
    }
    return Result::failure(std::move(serialized_checkpoint).error());
  }
  const Bytes& checkpoint_bytes = serialized_checkpoint.value().bytes;
  const core::ContentHash checkpoint_checksum = serialized_checkpoint.value().checksum;
  PreparedPayload prepared_payload{
      request.checkpoint.revision, request.checkpoint.parent, checkpoint_checksum,
      static_cast<std::uint64_t>(checkpoint_bytes.size()), std::move(request.inputs)};
  core::ContentHash prepared_checksum{};
  const Bytes prepared_bytes = serializePrepared(prepared_payload, &prepared_checksum);
  const Bytes marker_bytes = serializeCommitted(CommittedPayload{
      prepared_payload.revision, prepared_payload.parent, checkpoint_checksum, prepared_checksum});
  if (prepared_bytes.size() > std::min(impl_->config.maximum_bytes, kMaximumSmallRecordBytes)) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "serialized prepared record exceeds its hard byte bound",
                                     request.checkpoint.revision));
  }

  if (const auto existing = impl_->prepared_by_checksum.find(prepared_checksum);
      existing != impl_->prepared_by_checksum.end()) {
    const auto checkpoint = impl_->checkpoints_by_checksum.find(checkpoint_checksum);
    auto stored_checkpoint =
        checkpoint == impl_->checkpoints_by_checksum.end()
            ? core::Result<EncodedCheckpoint, GraphJournalError>::failure(
                  makeError(GraphJournalErrorCode::RecoveryCorruption,
                            "idempotent prepared transaction lost its canonical checkpoint"))
            : serializeCheckpoint(checkpoint->second.checkpoint, impl_->config,
                                  GraphJournalErrorCode::RecoveryCorruption);
    core::ContentHash stored_prepared_checksum{};
    const Bytes stored_prepared_bytes =
        serializePrepared(existing->second.payload, &stored_prepared_checksum);
    if (!stored_checkpoint || stored_checkpoint.value().checksum != checkpoint_checksum ||
        stored_checkpoint.value().bytes != checkpoint_bytes ||
        stored_prepared_checksum != prepared_checksum || stored_prepared_bytes != prepared_bytes ||
        existing->second.payload.revision != prepared_payload.revision ||
        existing->second.payload.parent != prepared_payload.parent ||
        existing->second.payload.checkpoint_checksum != checkpoint_checksum ||
        existing->second.payload.checkpoint_bytes != checkpoint_bytes.size() ||
        !sameInputs(existing->second.payload.inputs, prepared_payload.inputs) ||
        checkpoint->second.checkpoint.revision != request.checkpoint.revision) {
      return Result::failure(
          makeError(GraphJournalErrorCode::PreparedChecksumConflict,
                    "prepared checksum collides with different transaction content",
                    request.checkpoint.revision));
    }
    PreparedGraphCommit token = existing->second.token;
    token.idempotent = true;
    token.timing = {};
    return Result::success(std::move(token));
  }

  const GraphJournalStatus state = impl_->currentStatus();
  if (state.state == GraphJournalState::ReadOnlyIntegrityFailure) {
    return Result::failure(
        makeError(GraphJournalErrorCode::IntegrityFailure,
                  "graph journal is read-only after an integrity-ambiguous write",
                  request.checkpoint.revision));
  }
  if (state.state == GraphJournalState::ReadOnlyIoFailure) {
    return Result::failure(makeError(GraphJournalErrorCode::IoFailure,
                                     "graph journal is read-only after a durable I/O failure",
                                     request.checkpoint.revision));
  }
  if (state.state == GraphJournalState::ReadOnlyCapacityDegraded) {
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "graph journal is read-only at its hard capacity",
                                     request.checkpoint.revision));
  }

  const std::optional<GlobalGraphRevision> durable_parent =
      impl_->commits.empty() ? std::nullopt
                             : std::optional(GlobalGraphRevision(impl_->commits.rbegin()->first));
  const std::uint64_t expected_revision = durable_parent ? durable_parent->value() + 1U : 0U;
  if (request.expected_parent != durable_parent ||
      request.checkpoint.revision.value() != expected_revision) {
    return Result::failure(makeError(GraphJournalErrorCode::StaleParentRevision,
                                     "prepare compare-and-swap parent is stale",
                                     request.checkpoint.revision));
  }
  if (impl_->prepared_by_checksum.size() >= impl_->config.maximum_prepared_transactions ||
      impl_->commits.size() >= impl_->config.maximum_revisions) {
    impl_->capacity_failure = true;
    return Result::failure(
        makeError(GraphJournalErrorCode::CapacityExceeded,
                  "graph journal revision or prepared-transaction capacity is exhausted",
                  request.checkpoint.revision));
  }

  std::uint64_t required_bytes = impl_->durable_bytes;
  if (!impl_->checkpoints_by_checksum.contains(checkpoint_checksum) &&
      !checkedAdd(required_bytes, checkpoint_bytes.size(), &required_bytes)) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "graph journal byte accounting overflowed",
                                     request.checkpoint.revision));
  }
  if (!checkedAdd(required_bytes, prepared_bytes.size(), &required_bytes) ||
      !checkedAdd(required_bytes, impl_->reservedMarkerBytes(), &required_bytes) ||
      !checkedAdd(required_bytes, marker_bytes.size(), &required_bytes) ||
      required_bytes > impl_->config.maximum_bytes) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "graph journal cannot reserve the complete commit marker",
                                     request.checkpoint.revision));
  }

  GraphJournalTiming timing;
  const std::filesystem::path checkpoint_path =
      impl_->checkpoints_directory / (hashHex(checkpoint_checksum) + ".checkpoint");
  if (!impl_->checkpoints_by_checksum.contains(checkpoint_checksum)) {
    const auto write =
        durableWriteNoReplace(checkpoint_path, checkpoint_bytes, &impl_->temporary_counter);
    if (!write) {
      impl_->io_failure = write.error().code == GraphJournalErrorCode::IoFailure;
      impl_->integrity_failure = write.error().code == GraphJournalErrorCode::IntegrityFailure;
      GraphJournalError error = write.error();
      error.revision = request.checkpoint.revision;
      if (error.code == GraphJournalErrorCode::CommitChecksumConflict) {
        error.code = GraphJournalErrorCode::ChecksumMismatch;
      }
      return Result::failure(std::move(error));
    }
    timing.checkpoint_write_and_sync_us = write.value().elapsed_us;
    if (write.value().disposition == DurableWriteDisposition::Written) {
      if (const auto byte_error = addDurableBytes(*impl_, checkpoint_bytes.size())) {
        impl_->io_failure = true;
        return Result::failure(*byte_error);
      }
    }
    impl_->checkpoints_by_checksum.emplace(
        checkpoint_checksum,
        Impl::StoredCheckpoint{request.checkpoint, checkpoint_checksum,
                               static_cast<std::uint64_t>(checkpoint_bytes.size()),
                               checkpoint_path});
  }

  const std::filesystem::path prepared_path =
      impl_->prepared_directory / (hashHex(prepared_checksum) + ".prepared");
  const auto prepared_write =
      durableWriteNoReplace(prepared_path, prepared_bytes, &impl_->temporary_counter);
  if (!prepared_write) {
    impl_->io_failure = prepared_write.error().code == GraphJournalErrorCode::IoFailure;
    impl_->integrity_failure =
        prepared_write.error().code == GraphJournalErrorCode::IntegrityFailure;
    GraphJournalError error = prepared_write.error();
    error.revision = request.checkpoint.revision;
    if (error.code == GraphJournalErrorCode::CommitChecksumConflict) {
      error.code = GraphJournalErrorCode::PreparedChecksumConflict;
    }
    return Result::failure(std::move(error));
  }
  timing.prepared_write_and_sync_us = prepared_write.value().elapsed_us;
  if (prepared_write.value().disposition == DurableWriteDisposition::Written) {
    if (const auto byte_error = addDurableBytes(*impl_, prepared_bytes.size())) {
      impl_->io_failure = true;
      return Result::failure(*byte_error);
    }
  }

  PreparedGraphCommit token{prepared_payload.revision,
                            prepared_payload.parent,
                            checkpoint_checksum,
                            prepared_checksum,
                            prepared_payload.checkpoint_bytes,
                            false,
                            timing};
  impl_->prepared_by_checksum.emplace(
      prepared_checksum,
      Impl::StoredPrepared{std::move(prepared_payload), token,
                           static_cast<std::uint64_t>(prepared_bytes.size()),
                           static_cast<std::uint64_t>(marker_bytes.size()), prepared_path});
  impl_->last_timing = timing;
  return Result::success(std::move(token));
}

core::Result<GraphJournalCommitRecord, GraphJournalError> GraphJournal::commitPrepared(
    const PreparedGraphCommit& prepared) {
  using Result = core::Result<GraphJournalCommitRecord, GraphJournalError>;
  const auto stored = impl_->prepared_by_checksum.find(prepared.prepared_checksum);
  if (stored == impl_->prepared_by_checksum.end()) {
    return Result::failure(
        makeError(GraphJournalErrorCode::PreparedNotFound,
                  "prepared transaction is not durable in this journal",
                  prepared.revision.valid() ? std::optional(prepared.revision) : std::nullopt));
  }
  const PreparedGraphCommit& canonical = stored->second.token;
  if (!prepared.revision.valid() || prepared.revision != canonical.revision ||
      prepared.parent != canonical.parent ||
      prepared.checkpoint_checksum != canonical.checkpoint_checksum ||
      prepared.prepared_checksum != canonical.prepared_checksum ||
      prepared.checkpoint_bytes != canonical.checkpoint_bytes) {
    return Result::failure(makeError(GraphJournalErrorCode::PreparedChecksumConflict,
                                     "prepared capability conflicts with its durable record",
                                     canonical.revision));
  }

  if (const auto committed = impl_->commits.find(canonical.revision.value());
      committed != impl_->commits.end()) {
    if (committed->second.prepared_checksum != canonical.prepared_checksum ||
        committed->second.checkpoint_checksum != canonical.checkpoint_checksum) {
      return Result::failure(makeError(GraphJournalErrorCode::RevisionConflict,
                                       "revision is already committed from another transaction",
                                       canonical.revision));
    }
    GraphJournalCommitRecord response = committed->second;
    response.idempotent = true;
    response.timing = {};
    return Result::success(std::move(response));
  }
  if (impl_->io_failure) {
    return Result::failure(makeError(GraphJournalErrorCode::IoFailure,
                                     "graph journal is read-only after a durable I/O failure",
                                     canonical.revision));
  }
  if (impl_->integrity_failure) {
    return Result::failure(makeError(
        GraphJournalErrorCode::IntegrityFailure,
        "graph journal is read-only after an integrity-ambiguous write", canonical.revision));
  }
  if (impl_->commits.size() >= impl_->config.maximum_revisions) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "graph journal revision capacity is exhausted",
                                     canonical.revision));
  }

  const std::optional<GlobalGraphRevision> durable_parent =
      impl_->commits.empty() ? std::nullopt
                             : std::optional(GlobalGraphRevision(impl_->commits.rbegin()->first));
  const std::uint64_t expected_revision = durable_parent ? durable_parent->value() + 1U : 0U;
  if (canonical.parent != durable_parent || canonical.revision.value() != expected_revision) {
    return Result::failure(makeError(GraphJournalErrorCode::StaleParentRevision,
                                     "commit compare-and-swap parent changed after prepare",
                                     canonical.revision));
  }

  const Bytes marker_bytes = serializeCommitted(
      CommittedPayload{canonical.revision, canonical.parent, canonical.checkpoint_checksum,
                       canonical.prepared_checksum});
  if (marker_bytes.size() != stored->second.marker_bytes ||
      marker_bytes.size() > impl_->config.maximum_bytes -
                                std::min(impl_->durable_bytes, impl_->config.maximum_bytes)) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::CapacityExceeded,
                                     "reserved durable commit marker no longer fits",
                                     canonical.revision));
  }
  const std::filesystem::path marker_path =
      impl_->committed_directory / committedFilename(canonical.revision);
  const auto marker_write =
      durableWriteNoReplace(marker_path, marker_bytes, &impl_->temporary_counter);
  if (!marker_write) {
    impl_->io_failure = marker_write.error().code == GraphJournalErrorCode::IoFailure;
    impl_->integrity_failure = marker_write.error().code == GraphJournalErrorCode::IntegrityFailure;
    GraphJournalError error = marker_write.error();
    error.revision = canonical.revision;
    if (error.code == GraphJournalErrorCode::CommitChecksumConflict) {
      error.code = GraphJournalErrorCode::RevisionConflict;
    }
    return Result::failure(std::move(error));
  }
  GraphJournalTiming timing;
  timing.committed_marker_write_and_sync_us = marker_write.value().elapsed_us;
  if (marker_write.value().disposition == DurableWriteDisposition::Written) {
    if (const auto byte_error = addDurableBytes(*impl_, marker_bytes.size())) {
      impl_->io_failure = true;
      return Result::failure(*byte_error);
    }
  }

  const auto checkpoint = impl_->checkpoints_by_checksum.find(canonical.checkpoint_checksum);
  if (checkpoint == impl_->checkpoints_by_checksum.end()) {
    impl_->io_failure = true;
    return Result::failure(makeError(GraphJournalErrorCode::RecoveryCorruption,
                                     "prepared transaction lost its durable checkpoint",
                                     canonical.revision));
  }
  GraphJournalCommitRecord record{canonical.revision,
                                  canonical.parent,
                                  canonical.checkpoint_checksum,
                                  canonical.prepared_checksum,
                                  checkpoint->second.checkpoint,
                                  stored->second.payload.inputs,
                                  false,
                                  timing};
  impl_->commits.emplace(canonical.revision.value(), record);
  impl_->committed_prepared_checksums.insert(canonical.prepared_checksum);
  if (const auto retirement_error = retireUnreachable(*impl_)) {
    (void)retirement_error;
    // The marker is already durable and therefore remains the authoritative
    // commit. Cleanup failure only prevents further mutation.
    impl_->io_failure = true;
  }
  impl_->last_timing = timing;
  if (impl_->commits.size() >= impl_->config.maximum_revisions) {
    impl_->capacity_failure = true;
  }
  return Result::success(std::move(record));
}

core::Result<std::optional<GraphJournalCommitRecord>, GraphJournalError>
GraphJournal::latestCommit() const {
  using Result = core::Result<std::optional<GraphJournalCommitRecord>, GraphJournalError>;
  if (impl_->commits.empty()) {
    return Result::success(std::nullopt);
  }
  if (const auto verification_error =
          verifyCommitRecord(impl_->commits.rbegin()->second, impl_->config)) {
    return Result::failure(*verification_error);
  }
  return Result::success(impl_->commits.rbegin()->second);
}

core::Result<std::vector<GraphJournalCommitRecord>, GraphJournalError> GraphJournal::commitsSince(
    std::optional<GlobalGraphRevision> after, std::size_t maximum_records) const {
  using Result = core::Result<std::vector<GraphJournalCommitRecord>, GraphJournalError>;
  if ((after && !after->valid()) || maximum_records == 0U ||
      maximum_records > impl_->config.maximum_replay_records) {
    return Result::failure(makeError(GraphJournalErrorCode::ReplayLimitExceeded,
                                     "commit replay request exceeds its configured bound", after));
  }
  std::vector<GraphJournalCommitRecord> output;
  output.reserve(std::min(maximum_records, impl_->commits.size()));
  auto iterator = after ? impl_->commits.upper_bound(after->value()) : impl_->commits.begin();
  while (iterator != impl_->commits.end() && output.size() < maximum_records) {
    if (const auto verification_error = verifyCommitRecord(iterator->second, impl_->config)) {
      return Result::failure(*verification_error);
    }
    output.push_back(iterator->second);
    ++iterator;
  }
  return Result::success(std::move(output));
}

GraphJournalStatus GraphJournal::status() const noexcept {
  return impl_->currentStatus();
}

namespace {

[[nodiscard]] bool validCheckpointLimits(const GlobalGraphCheckpointLimits& limits) noexcept {
  return limits.maximum_wire_bytes > 0U && limits.maximum_boundaries > 0U &&
         limits.maximum_chart_placements > 0U && limits.maximum_adjacent_factors > 0U &&
         limits.maximum_gnss_factors > 0U && limits.maximum_loop_factors > 0U &&
         limits.maximum_factor_rows > 0U && limits.maximum_factor_coefficients > 0U &&
         limits.maximum_nested_collection_entries > 0U;
}

[[nodiscard]] std::optional<GraphJournalError> validateConfig(const GraphJournalConfig& config) {
  if (!sha256SelfTest()) {
    return makeError(GraphJournalErrorCode::SerializationFailure,
                     "SHA-256 implementation failed its startup known-answer test");
  }
  if (config.root_directory.empty() || config.maximum_bytes <= kFrameOverhead ||
      !validCheckpointLimits(config.checkpoint_limits) ||
      config.checkpoint_limits.maximum_wire_bytes > config.maximum_bytes ||
      config.maximum_revisions == 0U || config.maximum_prepared_transactions == 0U ||
      config.maximum_inputs_per_transaction == 0U || config.maximum_replay_records == 0U) {
    return makeError(GraphJournalErrorCode::InvalidConfiguration,
                     "graph journal path or hard capacity bounds are invalid");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> addDurableBytes(GraphJournal::Impl& impl,
                                                               std::uint64_t bytes) {
  std::uint64_t total = 0U;
  if (!checkedAdd(impl.durable_bytes, bytes, &total)) {
    return makeError(GraphJournalErrorCode::RecoveryCorruption,
                     "durable graph journal byte count overflowed");
  }
  impl.durable_bytes = total;
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> scanCommittedReferences(GraphJournal::Impl& impl) {
  std::optional<GraphJournalError> scan_error;
  const auto paths = filesWithExtension(impl.committed_directory, ".committed",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  std::uint64_t expected_revision = 0U;
  for (const auto& path : paths) {
    const auto bytes = readFile(path, 4096U);
    if (!bytes) {
      return bytes.error();
    }
    const auto frame = decodeFrame(bytes.value(), kCommittedMagic, 4096U - kFrameOverhead, path);
    if (!frame) {
      if (frame.error().code == GraphJournalErrorCode::MigrationRequired ||
          frame.error().code == GraphJournalErrorCode::UnsupportedFormat) {
        return frame.error();
      }
      if (expected_revision > 0U && frame.error().code != GraphJournalErrorCode::IoFailure) {
        impl.integrity_failure = true;
        impl.integrity_detail =
            "committed tail was excluded after verification failure: " + frame.error().detail;
        break;
      }
      if (frame.error().code != GraphJournalErrorCode::IoFailure) {
        return makeError(GraphJournalErrorCode::IntegrityFailure,
                         "committed closure marker verification failed: " + frame.error().detail);
      }
      return frame.error();
    }
    Reader reader(frame.value().payload, 1U);
    const CommittedPayload committed = readCommittedPayload(reader);
    if (!reader.complete() || !committed.revision.valid() ||
        path.filename().string() != committedFilename(committed.revision) ||
        committed.revision.value() != expected_revision ||
        (expected_revision == 0U && committed.parent.has_value()) ||
        (expected_revision > 0U &&
         (!committed.parent || committed.parent->value() + 1U != expected_revision)) ||
        zeroHash(committed.checkpoint_checksum) || zeroHash(committed.prepared_checksum) ||
        serializeCommitted(committed) != bytes.value()) {
      GraphJournalError failure =
          makeError(GraphJournalErrorCode::RecoveryCorruption,
                    "committed marker closure is malformed: " + path.string(),
                    committed.revision.valid() ? std::optional(committed.revision) : std::nullopt);
      if (expected_revision > 0U) {
        impl.integrity_failure = true;
        impl.integrity_detail =
            "committed tail was excluded after closure verification failure: " + failure.detail;
        break;
      }
      failure.code = GraphJournalErrorCode::IntegrityFailure;
      return failure;
    }
    impl.committed_checkpoint_filenames.insert(hashHex(committed.checkpoint_checksum) +
                                               ".checkpoint");
    impl.committed_prepared_filenames.insert(hashHex(committed.prepared_checksum) + ".prepared");
    ++expected_revision;
  }
  impl.verified_committed_markers = static_cast<std::size_t>(expected_revision);
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> recoverCheckpoints(GraphJournal::Impl& impl) {
  std::optional<GraphJournalError> scan_error;
  const auto paths = filesWithExtension(impl.checkpoints_directory, ".checkpoint",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  bool discarded = false;
  for (const std::filesystem::path& path : paths) {
    const bool committed_reference =
        impl.committed_checkpoint_filenames.contains(path.filename().string());
    const auto discard_unreachable = [&]() -> std::optional<GraphJournalError> {
      std::error_code remove_error;
      if (!std::filesystem::remove(path, remove_error) || remove_error) {
        return makeError(GraphJournalErrorCode::IoFailure,
                         "cannot discard unreachable checkpoint " + path.string() + ": " +
                             remove_error.message());
      }
      ++impl.retired_unreachable_files;
      discarded = true;
      return std::nullopt;
    };
    const auto bytes = readFile(path, impl.config.checkpoint_limits.maximum_wire_bytes);
    if (!bytes) {
      if (committed_reference && bytes.error().code != GraphJournalErrorCode::IoFailure) {
        return makeError(GraphJournalErrorCode::IntegrityFailure,
                         "committed checkpoint file verification failed: " + bytes.error().detail);
      }
      return bytes.error();
    }
    auto checkpoint = deserializeCheckpoint(bytes.value(), impl.config, path);
    if (!checkpoint) {
      if (checkpoint.error().code == GraphJournalErrorCode::MigrationRequired ||
          checkpoint.error().code == GraphJournalErrorCode::UnsupportedFormat ||
          checkpoint.error().code == GraphJournalErrorCode::CapacityExceeded ||
          checkpoint.error().code == GraphJournalErrorCode::IoFailure) {
        return checkpoint.error();
      }
      if (!committed_reference) {
        if (const auto error = discard_unreachable()) {
          return error;
        }
        continue;
      }
      return makeError(
          GraphJournalErrorCode::IntegrityFailure,
          "committed canonical checkpoint verification failed: " + checkpoint.error().detail);
    }
    const core::ContentHash canonical_checksum = hashBytes(bytes.value());
    if (path.filename().string() != hashHex(canonical_checksum) + ".checkpoint") {
      if (!committed_reference) {
        if (const auto error = discard_unreachable()) {
          return error;
        }
        continue;
      }
      return makeError(
          GraphJournalErrorCode::IntegrityFailure,
          "checkpoint filename does not equal the canonical byte digest: " + path.string());
    }
    if (!impl.checkpoints_by_checksum
             .emplace(canonical_checksum,
                      GraphJournal::Impl::StoredCheckpoint{
                          std::move(checkpoint).value(), canonical_checksum,
                          static_cast<std::uint64_t>(bytes.value().size()), path})
             .second) {
      return makeError(GraphJournalErrorCode::RecoveryCorruption,
                       "duplicate checkpoint checksum during recovery");
    }
    if (const auto byte_error = addDurableBytes(impl, bytes.value().size())) {
      return byte_error;
    }
  }
  if (discarded) {
    if (const auto sync_error = syncDirectory(impl.checkpoints_directory)) {
      return sync_error;
    }
  }
  return std::nullopt;
}

[[nodiscard]] bool sameInput(const GraphJournalInput& left,
                             const GraphJournalInput& right) noexcept {
  return left.kind == right.kind && left.source_id == right.source_id &&
         left.content_checksum == right.content_checksum && left.disposition == right.disposition;
}

[[nodiscard]] bool sameInputs(std::span<const GraphJournalInput> left,
                              std::span<const GraphJournalInput> right) noexcept {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(), sameInput);
}

[[nodiscard]] std::optional<GraphJournalError> recoverPrepared(GraphJournal::Impl& impl) {
  std::optional<GraphJournalError> scan_error;
  const auto paths = filesWithExtension(impl.prepared_directory, ".prepared",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  for (const std::filesystem::path& path : paths) {
    const bool committed_reference =
        impl.committed_prepared_filenames.contains(path.filename().string());
    const GraphJournalErrorCode corruption_code = committed_reference
                                                      ? GraphJournalErrorCode::IntegrityFailure
                                                      : GraphJournalErrorCode::RecoveryCorruption;
    const std::uint64_t maximum_record_bytes =
        std::min(impl.config.maximum_bytes, kMaximumSmallRecordBytes);
    const auto bytes = readFile(path, maximum_record_bytes);
    if (!bytes) {
      if (committed_reference && bytes.error().code != GraphJournalErrorCode::IoFailure) {
        return makeError(GraphJournalErrorCode::IntegrityFailure,
                         "committed prepared file verification failed: " + bytes.error().detail);
      }
      return bytes.error();
    }
    const auto frame =
        decodeFrame(bytes.value(), kPreparedMagic, maximum_record_bytes - kFrameOverhead, path);
    if (!frame) {
      if (frame.error().code == GraphJournalErrorCode::MigrationRequired ||
          frame.error().code == GraphJournalErrorCode::UnsupportedFormat) {
        return frame.error();
      }
      if (committed_reference && frame.error().code != GraphJournalErrorCode::IoFailure) {
        return makeError(GraphJournalErrorCode::IntegrityFailure,
                         "committed prepared frame verification failed: " + frame.error().detail);
      }
      return frame.error();
    }
    if (path.filename().string() != hashHex(frame.value().checksum) + ".prepared") {
      return makeError(corruption_code,
                       "prepared filename does not equal its payload checksum: " + path.string());
    }
    Reader reader(frame.value().payload, impl.config.maximum_inputs_per_transaction);
    PreparedPayload prepared = readPreparedPayload(reader, impl.config);
    const std::vector<GraphJournalInput> recovered_order = prepared.inputs;
    const auto input_error = canonicalizeInputs(
        &prepared.inputs, impl.config.maximum_inputs_per_transaction, prepared.revision);
    if (!reader.complete() || input_error || !sameInputs(recovered_order, prepared.inputs) ||
        !prepared.revision.valid() || zeroHash(prepared.checkpoint_checksum)) {
      return makeError(corruption_code,
                       "prepared record is malformed or not canonically ordered: " + path.string(),
                       prepared.revision.valid() ? std::optional(prepared.revision) : std::nullopt);
    }
    const auto checkpoint = impl.checkpoints_by_checksum.find(prepared.checkpoint_checksum);
    if (checkpoint == impl.checkpoints_by_checksum.end() ||
        checkpoint->second.canonical_bytes != prepared.checkpoint_bytes ||
        checkpoint->second.checkpoint.revision != prepared.revision ||
        checkpoint->second.checkpoint.parent != prepared.parent) {
      return makeError(corruption_code, "prepared record conflicts with its durable checkpoint",
                       prepared.revision);
    }
    core::ContentHash canonical_checksum{};
    const Bytes canonical = serializePrepared(prepared, &canonical_checksum);
    if (canonical_checksum != frame.value().checksum || canonical != bytes.value()) {
      return makeError(
          corruption_code,
          "prepared record does not round-trip through canonical encoding: " + path.string(),
          prepared.revision);
    }
    const Bytes marker = serializeCommitted(CommittedPayload{
        prepared.revision, prepared.parent, prepared.checkpoint_checksum, canonical_checksum});
    PreparedGraphCommit token{prepared.revision,
                              prepared.parent,
                              prepared.checkpoint_checksum,
                              canonical_checksum,
                              prepared.checkpoint_bytes,
                              true,
                              {}};
    if (!impl.prepared_by_checksum
             .emplace(canonical_checksum,
                      GraphJournal::Impl::StoredPrepared{
                          std::move(prepared), std::move(token),
                          static_cast<std::uint64_t>(canonical.size()),
                          static_cast<std::uint64_t>(marker.size()), path})
             .second) {
      return makeError(GraphJournalErrorCode::RecoveryCorruption,
                       "duplicate prepared checksum during recovery");
    }
    if (const auto byte_error = addDurableBytes(impl, bytes.value().size())) {
      return byte_error;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> recoverCommitted(GraphJournal::Impl& impl) {
  std::optional<GraphJournalError> scan_error;
  const auto paths = filesWithExtension(impl.committed_directory, ".committed",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  std::uint64_t expected_revision = 0U;
  for (const std::filesystem::path& path : paths) {
    if (expected_revision >= impl.verified_committed_markers) {
      break;
    }
    const auto bytes = readFile(path, 4096U);
    if (!bytes) {
      return bytes.error();
    }
    const auto frame = decodeFrame(bytes.value(), kCommittedMagic, 4096U - kFrameOverhead, path);
    if (!frame) {
      if (frame.error().code == GraphJournalErrorCode::MigrationRequired ||
          frame.error().code == GraphJournalErrorCode::UnsupportedFormat) {
        return frame.error();
      }
      if (frame.error().code == GraphJournalErrorCode::IoFailure) {
        return frame.error();
      }
      return makeError(
          GraphJournalErrorCode::IntegrityFailure,
          "verified committed marker changed during recovery: " + frame.error().detail);
    }
    Reader reader(frame.value().payload, 1U);
    const CommittedPayload committed = readCommittedPayload(reader);
    if (!reader.complete() || !committed.revision.valid() ||
        path.filename().string() != committedFilename(committed.revision) ||
        committed.revision.value() != expected_revision ||
        (expected_revision == 0U && committed.parent.has_value()) ||
        (expected_revision > 0U &&
         (!committed.parent || committed.parent->value() + 1U != expected_revision)) ||
        zeroHash(committed.checkpoint_checksum) || zeroHash(committed.prepared_checksum)) {
      return makeError(
          GraphJournalErrorCode::IntegrityFailure,
          "committed marker chain, identity, or filename is invalid: " + path.string(),
          committed.revision.valid() ? std::optional(committed.revision) : std::nullopt);
    }
    const auto prepared = impl.prepared_by_checksum.find(committed.prepared_checksum);
    const auto checkpoint = impl.checkpoints_by_checksum.find(committed.checkpoint_checksum);
    if (prepared == impl.prepared_by_checksum.end() ||
        checkpoint == impl.checkpoints_by_checksum.end() ||
        prepared->second.payload.revision != committed.revision ||
        prepared->second.payload.parent != committed.parent ||
        prepared->second.payload.checkpoint_checksum != committed.checkpoint_checksum ||
        checkpoint->second.checkpoint.revision != committed.revision ||
        checkpoint->second.checkpoint.parent != committed.parent) {
      return makeError(GraphJournalErrorCode::IntegrityFailure,
                       "committed marker conflicts with its prepared record or checkpoint",
                       committed.revision);
    }
    const Bytes canonical = serializeCommitted(committed);
    if (canonical != bytes.value()) {
      return makeError(
          GraphJournalErrorCode::IntegrityFailure,
          "committed marker does not round-trip through canonical encoding: " + path.string(),
          committed.revision);
    }
    GraphJournalCommitRecord record{committed.revision,
                                    committed.parent,
                                    committed.checkpoint_checksum,
                                    committed.prepared_checksum,
                                    checkpoint->second.checkpoint,
                                    prepared->second.payload.inputs,
                                    false,
                                    {}};
    if (!impl.commits.emplace(expected_revision, std::move(record)).second ||
        !impl.committed_prepared_checksums.insert(committed.prepared_checksum).second) {
      return makeError(GraphJournalErrorCode::IntegrityFailure,
                       "duplicate committed revision or prepared identity", committed.revision);
    }
    if (const auto byte_error = addDurableBytes(impl, bytes.value().size())) {
      return byte_error;
    }
    ++expected_revision;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> retireUnreachable(GraphJournal::Impl& impl) {
  const std::optional<GlobalGraphRevision> head =
      impl.commits.empty() ? std::nullopt
                           : std::optional(GlobalGraphRevision(impl.commits.rbegin()->first));
  const std::uint64_t next_revision = head ? head->value() + 1U : 0U;
  bool prepared_changed = false;
  for (auto iterator = impl.prepared_by_checksum.begin();
       iterator != impl.prepared_by_checksum.end();) {
    const bool committed = impl.committed_prepared_checksums.contains(iterator->first);
    const bool next_candidate = iterator->second.payload.revision.value() == next_revision &&
                                iterator->second.payload.parent == head;
    if (committed || next_candidate) {
      ++iterator;
      continue;
    }
    std::error_code remove_error;
    if (!std::filesystem::remove(iterator->second.path, remove_error) || remove_error) {
      return makeError(GraphJournalErrorCode::IoFailure,
                       "cannot retire unreachable prepared transaction " +
                           iterator->second.path.string() + ": " + remove_error.message(),
                       iterator->second.payload.revision);
    }
    impl.durable_bytes -= std::min(impl.durable_bytes, iterator->second.framed_bytes);
    ++impl.retired_unreachable_files;
    prepared_changed = true;
    iterator = impl.prepared_by_checksum.erase(iterator);
  }
  // Make reference removal durable before deleting any checkpoint it named.
  if (prepared_changed) {
    if (const auto sync_error = syncDirectory(impl.prepared_directory)) {
      return sync_error;
    }
  }

  std::set<core::ContentHash> retained_checkpoints;
  for (const auto& [unused, prepared] : impl.prepared_by_checksum) {
    (void)unused;
    retained_checkpoints.insert(prepared.payload.checkpoint_checksum);
  }
  for (const auto& [unused, committed] : impl.commits) {
    (void)unused;
    retained_checkpoints.insert(committed.checkpoint_checksum);
  }
  bool checkpoint_changed = false;
  for (auto iterator = impl.checkpoints_by_checksum.begin();
       iterator != impl.checkpoints_by_checksum.end();) {
    if (retained_checkpoints.contains(iterator->first)) {
      ++iterator;
      continue;
    }
    std::error_code remove_error;
    if (!std::filesystem::remove(iterator->second.path, remove_error) || remove_error) {
      return makeError(GraphJournalErrorCode::IoFailure,
                       "cannot retire orphan checkpoint " + iterator->second.path.string() + ": " +
                           remove_error.message(),
                       iterator->second.checkpoint.revision);
    }
    impl.durable_bytes -= std::min(impl.durable_bytes, iterator->second.canonical_bytes);
    ++impl.retired_unreachable_files;
    checkpoint_changed = true;
    iterator = impl.checkpoints_by_checksum.erase(iterator);
  }
  if (checkpoint_changed) {
    if (const auto sync_error = syncDirectory(impl.checkpoints_directory)) {
      return sync_error;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<GraphJournalError> verifyCommitRecord(
    const GraphJournalCommitRecord& record, const GraphJournalConfig& config) {
  if (record.revision != record.checkpoint.revision || record.parent != record.checkpoint.parent ||
      zeroHash(record.checkpoint_checksum) || zeroHash(record.prepared_checksum)) {
    return makeError(GraphJournalErrorCode::RecoveryCorruption,
                     "in-memory committed record identity conflicts with its checkpoint",
                     record.revision);
  }
  auto checkpoint =
      serializeCheckpoint(record.checkpoint, config, GraphJournalErrorCode::RecoveryCorruption);
  if (!checkpoint) {
    return checkpoint.error();
  }
  if (checkpoint.value().checksum != record.checkpoint_checksum) {
    return makeError(GraphJournalErrorCode::ChecksumMismatch,
                     "committed checkpoint checksum changed after durable recovery",
                     record.revision);
  }
  std::vector<GraphJournalInput> canonical_inputs = record.inputs;
  if (const auto input_error = canonicalizeInputs(
          &canonical_inputs, config.maximum_inputs_per_transaction, record.revision)) {
    return makeError(GraphJournalErrorCode::RecoveryCorruption,
                     "committed input set failed replay validation: " + input_error->detail,
                     record.revision);
  }
  if (!sameInputs(canonical_inputs, record.inputs)) {
    return makeError(GraphJournalErrorCode::RecoveryCorruption,
                     "committed input set is not canonically ordered", record.revision);
  }
  PreparedPayload prepared{record.revision, record.parent, record.checkpoint_checksum,
                           static_cast<std::uint64_t>(checkpoint.value().bytes.size()),
                           record.inputs};
  core::ContentHash prepared_checksum{};
  (void)serializePrepared(prepared, &prepared_checksum);
  if (prepared_checksum != record.prepared_checksum) {
    return makeError(GraphJournalErrorCode::ChecksumMismatch,
                     "committed prepared checksum changed after durable recovery", record.revision);
  }
  return std::nullopt;
}

}  // namespace

}  // namespace meridian::global
