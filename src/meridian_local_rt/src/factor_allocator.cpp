#include "meridian/local/factor_allocator.hpp"

#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace meridian::local {
namespace detail {

struct LocalFactorAllocatorState {
  explicit LocalFactorAllocatorState(LocalFactorAllocatorConfig allocator_config)
      : config(std::move(allocator_config)) {}

  LocalFactorAllocatorConfig config;
  mutable std::mutex mutex;
  std::map<core::OdomEpoch, std::uint64_t> next_factor_id_by_epoch;
  std::optional<std::uint64_t> open_transaction;
  std::uint64_t next_transaction_token{1U};
};

}  // namespace detail
namespace {

[[nodiscard]] LocalFactorAllocationError error(
    LocalFactorAllocationErrorCode code, std::string detail,
    std::optional<core::OdomEpoch> odom_epoch = std::nullopt,
    std::optional<core::LocalFactorRef> factor = std::nullopt) {
  return LocalFactorAllocationError{code, odom_epoch, factor, std::move(detail)};
}

[[nodiscard]] bool validFactorKind(LocalFactorKind kind) noexcept {
  switch (kind) {
    case LocalFactorKind::JointNavigationPrior:
    case LocalFactorKind::FrozenMarginalPrior:
    case LocalFactorKind::CombinedImu:
    case LocalFactorKind::DirectLidar:
    case LocalFactorKind::VisualReprojection:
      return true;
  }
  return false;
}

[[nodiscard]] std::uint64_t committedNextFactorId(const detail::LocalFactorAllocatorState& state,
                                                  core::OdomEpoch odom_epoch) {
  const auto found = state.next_factor_id_by_epoch.find(odom_epoch);
  return found == state.next_factor_id_by_epoch.end() ? 0U : found->second;
}

}  // namespace

LocalFactorAllocationTransaction::LocalFactorAllocationTransaction(
    std::weak_ptr<detail::LocalFactorAllocatorState> state, std::uint64_t token,
    core::OdomEpoch odom_epoch, std::uint64_t base_next_factor_id,
    std::uint64_t maximum_factors_per_epoch)
    : state_(std::move(state)),
      token_(token),
      odom_epoch_(odom_epoch),
      base_next_factor_id_(base_next_factor_id),
      candidate_next_factor_id_(base_next_factor_id),
      maximum_factors_per_epoch_(maximum_factors_per_epoch),
      consumed_(false) {}

LocalFactorAllocationTransaction::~LocalFactorAllocationTransaction() {
  releaseNoexcept();
}

LocalFactorAllocationTransaction::LocalFactorAllocationTransaction(
    LocalFactorAllocationTransaction&& other) noexcept
    : state_(std::move(other.state_)),
      token_(other.token_),
      odom_epoch_(other.odom_epoch_),
      base_next_factor_id_(other.base_next_factor_id_),
      candidate_next_factor_id_(other.candidate_next_factor_id_),
      maximum_factors_per_epoch_(other.maximum_factors_per_epoch_),
      allocations_(std::move(other.allocations_)),
      consumed_(other.consumed_) {
  other.token_ = 0U;
  other.consumed_ = true;
  other.state_.reset();
}

LocalFactorAllocationTransaction& LocalFactorAllocationTransaction::operator=(
    LocalFactorAllocationTransaction&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  releaseNoexcept();
  state_ = std::move(other.state_);
  token_ = other.token_;
  odom_epoch_ = other.odom_epoch_;
  base_next_factor_id_ = other.base_next_factor_id_;
  candidate_next_factor_id_ = other.candidate_next_factor_id_;
  maximum_factors_per_epoch_ = other.maximum_factors_per_epoch_;
  allocations_ = std::move(other.allocations_);
  consumed_ = other.consumed_;

  other.token_ = 0U;
  other.consumed_ = true;
  other.state_.reset();
  return *this;
}

std::optional<LocalFactorAllocationError> LocalFactorAllocationTransaction::validateActive() const {
  if (consumed_) {
    return error(LocalFactorAllocationErrorCode::TransactionConsumed,
                 "factor allocation transaction was already consumed", odom_epoch_);
  }
  const std::shared_ptr<detail::LocalFactorAllocatorState> state = state_.lock();
  if (!state) {
    return error(LocalFactorAllocationErrorCode::OwnerUnavailable,
                 "factor allocator no longer exists", odom_epoch_);
  }
  const std::scoped_lock lock(state->mutex);
  if (!state->open_transaction || *state->open_transaction != token_) {
    return error(LocalFactorAllocationErrorCode::StaleTransaction,
                 "factor allocation transaction is not the allocator's active candidate",
                 odom_epoch_);
  }
  return std::nullopt;
}

core::Result<LocalFactorAllocation, LocalFactorAllocationError>
LocalFactorAllocationTransaction::allocate(LocalFactorKind kind) {
  if (const auto active_error = validateActive()) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(*active_error);
  }
  if (!validFactorKind(kind)) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::InvalidFactorKind,
              "factor kind is outside the declared local factor inventory", odom_epoch_));
  }
  if (candidate_next_factor_id_ >= maximum_factors_per_epoch_) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::CapacityExceeded,
              "per-epoch factor identity capacity is exhausted", odom_epoch_));
  }

  LocalFactorAllocation allocation{
      core::LocalFactorRef{odom_epoch_, core::FactorId{candidate_next_factor_id_}}, kind,
      std::nullopt};
  ++candidate_next_factor_id_;
  allocations_.push_back(allocation);
  return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::success(
      std::move(allocation));
}

core::Result<LocalFactorAllocation, LocalFactorAllocationError>
LocalFactorAllocationTransaction::allocateReplacement(core::LocalFactorRef previous,
                                                      LocalFactorKind replacement_kind) {
  if (const auto active_error = validateActive()) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(*active_error);
  }
  if (!previous.odom_epoch.valid() || !previous.factor.valid()) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(error(
        LocalFactorAllocationErrorCode::InvalidReplacement,
        "replacement source must be a valid committed LocalFactorRef", odom_epoch_, previous));
  }
  if (previous.odom_epoch != odom_epoch_) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::ReplacementEpochMismatch,
              "a replacement cannot cross odometry epochs", odom_epoch_, previous));
  }
  if (previous.factor.value() >= base_next_factor_id_) {
    return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::ReplacementNotCommitted,
              "replacement source was not committed before this candidate", odom_epoch_, previous));
  }

  auto allocated = allocate(replacement_kind);
  if (!allocated) {
    return allocated;
  }
  allocations_.back().replaces = previous;
  LocalFactorAllocation result = std::move(allocated).value();
  result.replaces = previous;
  return core::Result<LocalFactorAllocation, LocalFactorAllocationError>::success(
      std::move(result));
}

core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>
LocalFactorAllocationTransaction::commit() {
  if (consumed_) {
    return core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::TransactionConsumed,
              "factor allocation transaction was already consumed", odom_epoch_));
  }
  if (allocations_.empty()) {
    return core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::EmptyTransaction,
              "an empty factor allocation candidate must be aborted", odom_epoch_));
  }
  const std::shared_ptr<detail::LocalFactorAllocatorState> state = state_.lock();
  if (!state) {
    consumed_ = true;
    return core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::OwnerUnavailable, "factor allocator no longer exists",
              odom_epoch_));
  }

  {
    const std::scoped_lock lock(state->mutex);
    const bool owns_prepared_slot = state->open_transaction && *state->open_transaction == token_;
    if (!owns_prepared_slot || committedNextFactorId(*state, odom_epoch_) != base_next_factor_id_) {
      if (owns_prepared_slot) {
        state->open_transaction.reset();
      }
      consumed_ = true;
      state_.reset();
      return core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>::failure(
          error(LocalFactorAllocationErrorCode::StaleTransaction,
                "allocator state changed since this candidate was prepared", odom_epoch_));
    }
    state->next_factor_id_by_epoch[odom_epoch_] = candidate_next_factor_id_;
    state->open_transaction.reset();
  }

  consumed_ = true;
  state_.reset();
  LocalFactorAllocationCommit committed{odom_epoch_, candidate_next_factor_id_,
                                        std::move(allocations_)};
  return core::Result<LocalFactorAllocationCommit, LocalFactorAllocationError>::success(
      std::move(committed));
}

core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError>
LocalFactorAllocationTransaction::abort() {
  if (consumed_) {
    return core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::TransactionConsumed,
              "factor allocation transaction was already consumed", odom_epoch_));
  }
  const std::shared_ptr<detail::LocalFactorAllocatorState> state = state_.lock();
  if (!state) {
    consumed_ = true;
    return core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::OwnerUnavailable, "factor allocator no longer exists",
              odom_epoch_));
  }

  {
    const std::scoped_lock lock(state->mutex);
    if (!state->open_transaction || *state->open_transaction != token_) {
      consumed_ = true;
      state_.reset();
      return core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError>::failure(error(
          LocalFactorAllocationErrorCode::StaleTransaction,
          "factor allocation transaction is not the allocator's active candidate", odom_epoch_));
    }
    state->open_transaction.reset();
  }

  const std::size_t discarded = allocations_.size();
  allocations_.clear();
  consumed_ = true;
  state_.reset();
  return core::Result<LocalFactorAllocationAbort, LocalFactorAllocationError>::success(
      LocalFactorAllocationAbort{odom_epoch_, discarded});
}

void LocalFactorAllocationTransaction::releaseNoexcept() noexcept {
  if (consumed_) {
    return;
  }
  if (const std::shared_ptr<detail::LocalFactorAllocatorState> state = state_.lock()) {
    const std::scoped_lock lock(state->mutex);
    if (state->open_transaction && *state->open_transaction == token_) {
      state->open_transaction.reset();
    }
  }
  consumed_ = true;
  state_.reset();
  allocations_.clear();
}

LocalFactorAllocator::LocalFactorAllocator(LocalFactorAllocatorConfig config)
    : state_(std::make_shared<detail::LocalFactorAllocatorState>(std::move(config))) {}

LocalFactorAllocator::~LocalFactorAllocator() = default;
LocalFactorAllocator::LocalFactorAllocator(LocalFactorAllocator&&) noexcept = default;
LocalFactorAllocator& LocalFactorAllocator::operator=(LocalFactorAllocator&&) noexcept = default;

core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>
LocalFactorAllocator::prepare(core::OdomEpoch odom_epoch) {
  if (!state_) {
    return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::OwnerUnavailable, "factor allocator was moved from",
              odom_epoch));
  }
  const std::scoped_lock lock(state_->mutex);
  if (state_->config.maximum_factors_per_epoch == 0U) {
    return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::InvalidConfig,
              "maximum_factors_per_epoch must be nonzero", odom_epoch));
  }
  if (!odom_epoch.valid()) {
    return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::InvalidEpoch,
              "factor allocation requires a valid odometry epoch", odom_epoch));
  }
  if (state_->open_transaction) {
    return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::TransactionAlreadyOpen,
              "the single-writer allocator already has a prepared candidate", odom_epoch));
  }
  if (state_->next_transaction_token == std::numeric_limits<std::uint64_t>::max()) {
    return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::TransactionTokenExhausted,
              "factor allocation transaction token space is exhausted", odom_epoch));
  }

  const std::uint64_t token = state_->next_transaction_token++;
  state_->open_transaction = token;
  const std::uint64_t base = committedNextFactorId(*state_, odom_epoch);
  LocalFactorAllocationTransaction transaction{state_, token, odom_epoch, base,
                                               state_->config.maximum_factors_per_epoch};
  return core::Result<LocalFactorAllocationTransaction, LocalFactorAllocationError>::success(
      std::move(transaction));
}

LocalFactorAllocatorCheckpoint LocalFactorAllocator::checkpoint() const {
  LocalFactorAllocatorCheckpoint output;
  if (!state_) {
    return output;
  }
  const std::scoped_lock lock(state_->mutex);
  output.epochs.reserve(state_->next_factor_id_by_epoch.size());
  for (const auto& [epoch, next_factor_id] : state_->next_factor_id_by_epoch) {
    if (next_factor_id != 0U) {
      output.epochs.push_back(FactorAllocatorEpochCheckpoint{epoch, next_factor_id});
    }
  }
  return output;
}

core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>
LocalFactorAllocator::restoreCheckpoint(const LocalFactorAllocatorCheckpoint& checkpoint) {
  if (!state_) {
    return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::OwnerUnavailable, "factor allocator was moved from"));
  }
  const std::scoped_lock lock(state_->mutex);
  if (state_->config.maximum_factors_per_epoch == 0U) {
    return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::InvalidConfig,
              "maximum_factors_per_epoch must be nonzero"));
  }
  if (state_->open_transaction) {
    return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::TransactionAlreadyOpen,
              "cannot restore a checkpoint while a candidate is prepared"));
  }
  if (checkpoint.schema_version != LocalFactorAllocatorCheckpoint::kSchemaVersion) {
    return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
        error(LocalFactorAllocationErrorCode::InvalidCheckpoint,
              "unsupported factor allocator checkpoint schema"));
  }

  std::optional<core::OdomEpoch> previous_epoch;
  for (const FactorAllocatorEpochCheckpoint& epoch : checkpoint.epochs) {
    if (!epoch.odom_epoch.valid() || epoch.next_factor_id == 0U) {
      return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
          error(LocalFactorAllocationErrorCode::InvalidCheckpoint,
                "checkpoint entries require valid epochs and nonzero high-water marks",
                epoch.odom_epoch));
    }
    if (previous_epoch && *previous_epoch >= epoch.odom_epoch) {
      return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
          error(LocalFactorAllocationErrorCode::NonCanonicalCheckpoint,
                "checkpoint epochs must be strictly increasing", epoch.odom_epoch));
    }
    if (epoch.next_factor_id > state_->config.maximum_factors_per_epoch) {
      return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::failure(
          error(LocalFactorAllocationErrorCode::CheckpointExceedsCapacity,
                "checkpoint high-water mark exceeds configured epoch capacity", epoch.odom_epoch));
    }
    previous_epoch = epoch.odom_epoch;
  }

  LocalFactorCheckpointRestoreReport report;
  for (const FactorAllocatorEpochCheckpoint& epoch : checkpoint.epochs) {
    const std::uint64_t current = committedNextFactorId(*state_, epoch.odom_epoch);
    if (epoch.next_factor_id > current) {
      state_->next_factor_id_by_epoch[epoch.odom_epoch] = epoch.next_factor_id;
      ++report.epochs_advanced;
    } else {
      ++report.epochs_retained;
    }
  }
  return core::Result<LocalFactorCheckpointRestoreReport, LocalFactorAllocationError>::success(
      report);
}

}  // namespace meridian::local
