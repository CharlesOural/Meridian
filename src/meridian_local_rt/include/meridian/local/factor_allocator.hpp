#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"

namespace meridian::local {

// Actual local-factor classes share one FactorId counter inside an odometry
// epoch. This type is allocation/audit metadata only; modality-specific raw
// factor specifications remain owned by their typed producer APIs.
enum class LocalFactorKind : std::uint8_t {
  JointNavigationPrior,
  FrozenMarginalPrior,
  CombinedImu,
  DirectLidar,
  VisualReprojection,
};

struct LocalFactorAllocatorConfig {
  // IDs are allocated from [0, maximum_factors_per_epoch). FactorId's invalid
  // sentinel is never allocated.
  std::uint64_t maximum_factors_per_epoch{1'000'000U};
};

struct LocalFactorAllocation {
  core::LocalFactorRef ref;
  LocalFactorKind kind{LocalFactorKind::CombinedImu};
  // Populated only for an immutable replacement. The previous factor remains
  // a distinct committed identity; the journal transaction will record its
  // disposition when the typed journal layer is added.
  std::optional<core::LocalFactorRef> replaces;

  auto operator<=>(const LocalFactorAllocation&) const = default;
};

struct FactorAllocatorEpochCheckpoint {
  core::OdomEpoch odom_epoch;
  // First unallocated ID. Every ID in [0, next_factor_id) was made visible by
  // a committed allocator transaction in this epoch.
  std::uint64_t next_factor_id{};

  auto operator<=>(const FactorAllocatorEpochCheckpoint&) const = default;
};

struct LocalFactorAllocatorCheckpoint {
  static constexpr std::uint32_t kSchemaVersion = 1U;

  std::uint32_t schema_version{kSchemaVersion};
  // Canonical strict OdomEpoch order. Only epochs with a nonzero committed
  // high-water mark are serialized.
  std::vector<FactorAllocatorEpochCheckpoint> epochs;

  friend bool operator==(const LocalFactorAllocatorCheckpoint&,
                         const LocalFactorAllocatorCheckpoint&) = default;
};

enum class LocalFactorAllocationErrorCode {
  InvalidConfig,
  InvalidEpoch,
  InvalidFactorKind,
  TransactionAlreadyOpen,
  TransactionConsumed,
  StaleTransaction,
  OwnerUnavailable,
  EmptyTransaction,
  CapacityExceeded,
  InvalidReplacement,
  ReplacementEpochMismatch,
  ReplacementNotCommitted,
  InvalidCheckpoint,
  NonCanonicalCheckpoint,
  CheckpointExceedsCapacity,
  TransactionTokenExhausted,
};

struct LocalFactorAllocationError {
  LocalFactorAllocationErrorCode code{LocalFactorAllocationErrorCode::InvalidConfig};
  std::optional<core::OdomEpoch> odom_epoch;
  std::optional<core::LocalFactorRef> factor;
  std::string detail;
};

struct LocalFactorAllocationCommit {
  core::OdomEpoch odom_epoch;
  std::uint64_t next_factor_id{};
  std::vector<LocalFactorAllocation> allocations;
};

struct LocalFactorAllocationAbort {
  core::OdomEpoch odom_epoch;
  std::size_t discarded_allocations{};
};

struct LocalFactorCheckpointRestoreReport {
  std::size_t epochs_advanced{};
  std::size_t epochs_retained{};
};

namespace detail {
struct LocalFactorAllocatorState;
}  // namespace detail

class LocalFactorAllocator;

// A transaction owns a private copy of exactly one epoch's high-water mark.
// Candidate refs are valid only inside the surrounding graph candidate until
// commit succeeds. Destruction without commit is an abort.
class LocalFactorAllocationTransaction {
public:
  ~LocalFactorAllocationTransaction();

  LocalFactorAllocationTransaction(LocalFactorAllocationTransaction&& other) noexcept;
  LocalFactorAllocationTransaction& operator=(LocalFactorAllocationTransaction&& other) noexcept;
  LocalFactorAllocationTransaction(const LocalFactorAllocationTransaction&) = delete;
  LocalFactorAllocationTransaction& operator=(const LocalFactorAllocationTransaction&) = delete;

  [[nodiscard]] core::Result<LocalFactorAllocation, LocalFactorAllocationError> allocate(
      LocalFactorKind kind);

  // Allocates a fresh identity for a previously committed factor in the same
  // epoch. It never repurposes the previous ID.
  [[nodiscard]] core::Result<LocalFactorAllocation, LocalFactorAllocationError> allocateReplacement(
      core::LocalFactorRef previous, LocalFactorKind replacement_kind);

  [[nodiscard]] core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError> commit();
  [[nodiscard]] core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError> abort();

  [[nodiscard]] bool consumed() const noexcept { return consumed_; }
  [[nodiscard]] std::size_t preparedAllocations() const noexcept { return allocations_.size(); }

private:
  friend class LocalFactorAllocator;

  LocalFactorAllocationTransaction(std::weak_ptr<detail::LocalFactorAllocatorState> state,
                                   std::uint64_t token, core::OdomEpoch odom_epoch,
                                   std::uint64_t base_next_factor_id,
                                   std::uint64_t maximum_factors_per_epoch);

  void releaseNoexcept() noexcept;
  [[nodiscard]] std::optional<LocalFactorAllocationError> validateActive() const;

  std::weak_ptr<detail::LocalFactorAllocatorState> state_;
  std::uint64_t token_{};
  core::OdomEpoch odom_epoch_;
  std::uint64_t base_next_factor_id_{};
  std::uint64_t candidate_next_factor_id_{};
  std::uint64_t maximum_factors_per_epoch_{};
  std::vector<LocalFactorAllocation> allocations_;
  bool consumed_{true};
};

// Single-writer, ROS-free factor identity owner. Checkpoints are process-local
// rebuild state, not a promise that an in-progress odometry epoch survives a
// process crash; section 12.3 still starts a new epoch after such a crash.
class LocalFactorAllocator {
public:
  explicit LocalFactorAllocator(LocalFactorAllocatorConfig config = {});
  ~LocalFactorAllocator();

  LocalFactorAllocator(LocalFactorAllocator&&) noexcept;
  LocalFactorAllocator& operator=(LocalFactorAllocator&&) noexcept;
  LocalFactorAllocator(const LocalFactorAllocator&) = delete;
  LocalFactorAllocator& operator=(const LocalFactorAllocator&) = delete;

  [[nodiscard]] core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError> prepare(
      core::OdomEpoch odom_epoch);

  // Returns only committed high-water marks; private allocations in an open
  // transaction are deliberately invisible.
  [[nodiscard]] LocalFactorAllocatorCheckpoint checkpoint() const;

  // Monotonic merge used by deterministic in-process rebuild. Existing marks
  // are never decreased, so restoring a stale checkpoint cannot reuse an ID.
  [[nodiscard]] core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>
  restoreCheckpoint(const LocalFactorAllocatorCheckpoint& checkpoint);

private:
  std::shared_ptr<detail::LocalFactorAllocatorState> state_;
};

}  // namespace meridian::local
