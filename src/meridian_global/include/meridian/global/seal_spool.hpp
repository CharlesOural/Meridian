#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/sparse_submap.hpp"

namespace meridian::global {

struct OutboxSequenceTag;
using OutboxSequence = core::StrongId<OutboxSequenceTag>;

struct SealSpoolConfig {
  std::filesystem::path root_directory;
  std::uint64_t maximum_bytes{4ULL * 1024ULL * 1024ULL * 1024ULL};
  std::uint64_t maximum_seal_record_bytes{256ULL * 1024ULL * 1024ULL};
  std::size_t maximum_seals{4096U};
  std::size_t maximum_outbox_entries{4096U};
  std::size_t maximum_replay_records{256U};
  std::size_t maximum_vector_elements{2'000'000U};
};

enum class SealSpoolState {
  Ready,
  DegradedStorageCapacity,
  ReadOnlyIoFailure,
  ReadOnlyIntegrityFailure,
};

enum class SealSpoolErrorCode {
  InvalidConfiguration,
  InvalidSeal,
  UnsupportedInMemoryPayload,
  NonDurablePayloadReference,
  CapacityExceeded,
  IdentityConflict,
  SequenceNotFound,
  AcknowledgementConflict,
  ReplayLimitExceeded,
  IoFailure,
  IntegrityFailure,
  ChecksumMismatch,
  RecoveryCorruption,
  SerializationFailure,
};

struct SealSpoolError {
  SealSpoolErrorCode code{SealSpoolErrorCode::IoFailure};
  std::optional<OutboxSequence> sequence;
  std::optional<core::SubmapId> submap;
  std::string detail;
};

struct SealSpoolTiming {
  std::int64_t seal_write_and_sync_us{};
  std::int64_t outbox_write_and_sync_us{};
  std::int64_t acknowledgement_write_and_sync_us{};
};

struct SealEnqueueReport {
  OutboxSequence sequence;
  core::SparseSubmapSealIdentity identity;
  core::ContentHash serialized_record_checksum{};
  std::uint64_t serialized_record_bytes{};
  bool idempotent{false};
  SealSpoolTiming timing;
};

struct SealAcknowledgement {
  OutboxSequence sequence;
  core::SparseSubmapSealIdentity identity;
  core::ContentHash seal_checksum{};
};

struct SealReplayRecord {
  OutboxSequence sequence;
  core::SparseSubmapSealIdentity identity;
  core::ContentHash seal_checksum{};
  core::ContentHash serialized_record_checksum{};
  std::uint64_t serialized_record_bytes{};
  SparseSubmapSeal seal;
};

struct SealSpoolStatus {
  SealSpoolState state{SealSpoolState::Ready};
  OutboxSequence head{0U};
  std::size_t durable_seals{};
  std::size_t outbox_entries{};
  std::size_t acknowledged_entries{};
  std::size_t unacknowledged_entries{};
  std::uint64_t durable_bytes{};
  std::uint64_t reserved_acknowledgement_bytes{};
  std::uint64_t maximum_bytes{};
  std::size_t recovered_temporary_files{};
  std::size_t recovery_scanned_files{};
  std::optional<OutboxSequence> oldest_unacknowledged;
  std::optional<std::filesystem::path> ambiguous_destination;
};

// Single-writer, ROS-free durable sparse-seal spool and replay outbox. Every
// successful mutation has crossed a file fsync, atomic no-replace rename, and
// directory fsync before this API returns success.
class SealSpool {
public:
  struct Impl;

  [[nodiscard]] static core::Result<std::unique_ptr<SealSpool>, SealSpoolError> open(
      SealSpoolConfig config);

  ~SealSpool();
  SealSpool(SealSpool&&) noexcept;
  SealSpool& operator=(SealSpool&&) noexcept;
  SealSpool(const SealSpool&) = delete;
  SealSpool& operator=(const SealSpool&) = delete;

  [[nodiscard]] core::Result<SealEnqueueReport, SealSpoolError> enqueue(SparseSubmapSeal seal);

  [[nodiscard]] core::Result<std::vector<SealReplayRecord>, SealSpoolError> replaySince(
      OutboxSequence after, std::size_t maximum_records) const;

  [[nodiscard]] core::Result<bool, SealSpoolError> acknowledge(
      const SealAcknowledgement& acknowledgement, SealSpoolTiming* timing = nullptr);

  [[nodiscard]] SealSpoolStatus status() const noexcept;

private:
  explicit SealSpool(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::global
