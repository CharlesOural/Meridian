#pragma once

#include <cmath>

namespace meridian::local::detail {

enum class CandidateGateDecision {
  Accepted,
  NonFinite,
  PoseCorrectionLimit,
  CompleteObjectiveIncrease,
};

enum class CandidateGatePhase {
  NonlinearIteration,
  ConvergedTransaction,
};

struct CandidateGateInput {
  double objective_before{};
  double objective_after{};
  double transaction_translation_correction_m{};
  double transaction_rotation_correction_rad{};
  bool all_state_corrections_finite{true};
  CandidateGatePhase phase{CandidateGatePhase::ConvergedTransaction};
};

struct CandidateGateLimits {
  double maximum_transaction_translation_correction_m{};
  double maximum_transaction_rotation_correction_rad{};
  double complete_objective_nonsmooth_absolute_allowance{};
  double complete_objective_nonsmooth_relative_allowance{};
};

// Every nonlinear iteration is subject to finite-state and physical
// correction bounds. Complete-objective acceptance is intentionally deferred
// until the candidate converges: a Gauss--Newton relinearization step may be
// locally nonmonotone while the isolated transaction remains a large net
// improvement over its immutable seed.
[[nodiscard]] inline CandidateGateDecision evaluateCandidateGate(
    const CandidateGateInput& input, const CandidateGateLimits& limits) noexcept {
  if (!input.all_state_corrections_finite || !std::isfinite(input.objective_before) ||
      !std::isfinite(input.objective_after) ||
      !std::isfinite(input.transaction_translation_correction_m) ||
      !std::isfinite(input.transaction_rotation_correction_rad)) {
    return CandidateGateDecision::NonFinite;
  }
  if (input.transaction_translation_correction_m >
          limits.maximum_transaction_translation_correction_m ||
      input.transaction_rotation_correction_rad >
          limits.maximum_transaction_rotation_correction_rad) {
    return CandidateGateDecision::PoseCorrectionLimit;
  }
  if (input.phase == CandidateGatePhase::ConvergedTransaction) {
    const double allowance =
        limits.complete_objective_nonsmooth_absolute_allowance +
        limits.complete_objective_nonsmooth_relative_allowance *
            std::abs(input.objective_before);
    if (input.objective_after > input.objective_before + allowance) {
      return CandidateGateDecision::CompleteObjectiveIncrease;
    }
  }
  return CandidateGateDecision::Accepted;
}

}  // namespace meridian::local::detail
