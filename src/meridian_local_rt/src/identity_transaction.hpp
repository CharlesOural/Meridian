#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

#include "meridian/core/result.hpp"

namespace meridian::local::detail {

struct IdentityCounters {
  std::uint64_t next_lineage{};
  std::uint64_t next_derived{};
  std::uint64_t next_factor_group{};
  std::uint64_t next_correlation_group{};
  std::uint64_t next_visual_landmark{};
  std::uint64_t next_visual_factor{};

  friend bool operator==(const IdentityCounters&, const IdentityCounters&) = default;
};

enum class IdentityTransactionErrorCode {
  TransactionConsumed,
};

struct IdentityTransactionError {
  IdentityTransactionErrorCode code{IdentityTransactionErrorCode::TransactionConsumed};
};

template <typename State>
class IdentityTransaction {
public:
  struct CommittedState {
    IdentityCounters counters;
    State state;
  };

  IdentityTransaction(IdentityCounters counters, const State& state)
      : original_counters_(counters), candidate_counters_(std::move(counters)), state_(state) {}

  ~IdentityTransaction() { discardNoexcept(); }

  IdentityTransaction(IdentityTransaction&& other) noexcept
      : original_counters_(other.original_counters_),
        candidate_counters_(other.candidate_counters_),
        state_(std::move(other.state_)),
        consumed_(other.consumed_) {
    other.state_.reset();
    other.consumed_ = true;
  }

  IdentityTransaction& operator=(IdentityTransaction&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    discardNoexcept();
    original_counters_ = other.original_counters_;
    candidate_counters_ = other.candidate_counters_;
    state_ = std::move(other.state_);
    consumed_ = other.consumed_;
    other.state_.reset();
    other.consumed_ = true;
    return *this;
  }

  IdentityTransaction(const IdentityTransaction&) = delete;
  IdentityTransaction& operator=(const IdentityTransaction&) = delete;

  [[nodiscard]] IdentityCounters& counters() noexcept {
    assert(!consumed_);
    return candidate_counters_;
  }
  [[nodiscard]] State& state() noexcept {
    assert(!consumed_ && state_.has_value());
    return *state_;
  }

  [[nodiscard]] bool consumed() const noexcept { return consumed_; }

  [[nodiscard]] core::Result<CommittedState, IdentityTransactionError> commit() {
    using Result = core::Result<CommittedState, IdentityTransactionError>;
    if (consumed_) {
      return Result::failure(
          IdentityTransactionError{IdentityTransactionErrorCode::TransactionConsumed});
    }
    State committed_state = std::move(*state_);
    state_.reset();
    consumed_ = true;
    return Result::success(CommittedState{candidate_counters_, std::move(committed_state)});
  }

  // Destroys candidate state before returning the exact live-counter
  // checkpoint. This prevents a caller from reusing identities while an
  // object carrying the rejected candidate identities remains alive.
  [[nodiscard]] core::Result<IdentityCounters, IdentityTransactionError> abort() {
    using Result = core::Result<IdentityCounters, IdentityTransactionError>;
    if (consumed_) {
      return Result::failure(
          IdentityTransactionError{IdentityTransactionErrorCode::TransactionConsumed});
    }
    state_.reset();
    consumed_ = true;
    return Result::success(original_counters_);
  }

private:
  void discardNoexcept() noexcept {
    if (!consumed_) {
      state_.reset();
      consumed_ = true;
    }
  }

  IdentityCounters original_counters_;
  IdentityCounters candidate_counters_;
  std::optional<State> state_;
  bool consumed_{false};
};

}  // namespace meridian::local::detail
