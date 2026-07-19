#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

struct GraphJournalConfig {
  std::filesystem::path root_directory;
  std::uint64_t maximum_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
  // Canonical graph-checkpoint wire and collection bounds. Checkpoint files
  // contain exactly encodeGlobalGraphCheckpoint() bytes, so this is also the
  // journal's checkpoint-file byte bound.
  GlobalGraphCheckpointLimits checkpoint_limits;
  std::size_t maximum_revisions{4096U};
  std::size_t maximum_prepared_transactions{8192U};
  std::size_t maximum_inputs_per_transaction{32768U};
  std::size_t maximum_replay_records{256U};
};

enum class GraphJournalState {
  Ready,
  ReadOnlyCapacityDegraded,
  ReadOnlyIoFailure,
  ReadOnlyIntegrityFailure,
};

enum class GraphJournalInputKind {
  SparseSubmapSeal,
  AdjacentConstraint,
  GnssObservation,
  LoopProposal,
  SurveyedAlignment,
  OperatorAction,
};

enum class GraphJournalDisposition {
  Admitted,
  Rejected,
  Deferred,
  Retained,
  Removed,
};

// A transaction input is identified independently of its disposition. Inputs
// are canonicalized by (kind, source_id); duplicate identities fail closed.
struct GraphJournalInput {
  GraphJournalInputKind kind{GraphJournalInputKind::SparseSubmapSeal};
  std::uint64_t source_id{};
  core::ContentHash content_checksum{};
  GraphJournalDisposition disposition{GraphJournalDisposition::Admitted};
};

struct GraphJournalPrepareRequest {
  // Compare-and-swap parent observed by the solver. It must exactly match both
  // the checkpoint parent and the journal's durable committed head.
  std::optional<GlobalGraphRevision> expected_parent;
  GlobalGraphCheckpoint checkpoint;
  std::vector<GraphJournalInput> inputs;
};

enum class GraphJournalErrorCode {
  InvalidConfiguration,
  InvalidTransaction,
  InvalidCheckpoint,
  StaleParentRevision,
  RevisionConflict,
  PreparedNotFound,
  PreparedChecksumConflict,
  CommitChecksumConflict,
  CapacityExceeded,
  ReplayLimitExceeded,
  IoFailure,
  IntegrityFailure,
  ChecksumMismatch,
  RecoveryCorruption,
  SerializationFailure,
  MigrationRequired,
  UnsupportedFormat,
};

struct GraphJournalError {
  GraphJournalErrorCode code{GraphJournalErrorCode::IoFailure};
  std::optional<GlobalGraphRevision> revision;
  std::string detail;
};

struct GraphJournalTiming {
  std::int64_t checkpoint_write_and_sync_us{};
  std::int64_t prepared_write_and_sync_us{};
  std::int64_t committed_marker_write_and_sync_us{};
  std::int64_t recovery_scan_us{};
};

// Immutable capability returned only after the PREPARED record is durable.
// Its two hashes bind commitPrepared() to the exact checkpoint and inputs.
struct PreparedGraphCommit {
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  // SHA-256 of the exact encodeGlobalGraphCheckpoint() byte stream. This is
  // the checkpoint-file identity and is distinct from checkpoint.checksum,
  // which authenticates the checkpoint's solver-neutral semantic fields.
  core::ContentHash checkpoint_checksum{};
  core::ContentHash prepared_checksum{};
  std::uint64_t checkpoint_bytes{};
  bool idempotent{false};
  GraphJournalTiming timing;
};

struct GraphJournalCommitRecord {
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  // SHA-256 of the exact canonical checkpoint-file bytes.
  core::ContentHash checkpoint_checksum{};
  core::ContentHash prepared_checksum{};
  GlobalGraphCheckpoint checkpoint;
  std::vector<GraphJournalInput> inputs;
  bool idempotent{false};
  GraphJournalTiming timing;
};

struct GraphJournalStatus {
  GraphJournalState state{GraphJournalState::Ready};
  std::optional<GlobalGraphRevision> committed_head;
  std::optional<GlobalGraphRevision> oldest_committed_revision;
  std::size_t committed_revisions{};
  std::size_t prepared_transactions{};
  std::size_t uncommitted_prepared_transactions{};
  std::uint64_t durable_bytes{};
  std::uint64_t reserved_commit_marker_bytes{};
  std::uint64_t maximum_bytes{};
  std::size_t recovered_temporary_files{};
  std::size_t recovery_scanned_files{};
  std::size_t retired_unreachable_files{};
  std::optional<std::string> integrity_detail;
  GraphJournalTiming last_timing;
};

// ROS-free, single-writer durable graph journal. Every .checkpoint file is
// exactly the canonical GlobalGraphCheckpoint byte stream; PREPARED and
// COMMITTED remain small framed atomic records. The implementation stores one
// complete checkpoint for every revision: it never performs lossy graph
// compaction and never deletes a committed recovery path.
class GraphJournal {
public:
  struct Impl;

  [[nodiscard]] static core::Result<std::unique_ptr<GraphJournal>, GraphJournalError> open(
      GraphJournalConfig config);

  ~GraphJournal();
  GraphJournal(GraphJournal&&) noexcept;
  GraphJournal& operator=(GraphJournal&&) noexcept;
  GraphJournal(const GraphJournal&) = delete;
  GraphJournal& operator=(const GraphJournal&) = delete;

  [[nodiscard]] core::Result<PreparedGraphCommit, GraphJournalError> prepare(
      GraphJournalPrepareRequest request);

  [[nodiscard]] core::Result<GraphJournalCommitRecord, GraphJournalError> commitPrepared(
      const PreparedGraphCommit& prepared);

  [[nodiscard]] core::Result<std::optional<GraphJournalCommitRecord>, GraphJournalError>
  latestCommit() const;

  // Returns committed revisions strictly newer than `after`. std::nullopt
  // starts at the oldest retained revision. This journal retains all committed
  // revisions until it becomes explicitly capacity-degraded.
  [[nodiscard]] core::Result<std::vector<GraphJournalCommitRecord>, GraphJournalError> commitsSince(
      std::optional<GlobalGraphRevision> after, std::size_t maximum_records) const;

  [[nodiscard]] GraphJournalStatus status() const noexcept;

private:
  explicit GraphJournal(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::global
