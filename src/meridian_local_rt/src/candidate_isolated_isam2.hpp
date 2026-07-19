#pragma once

// Private GTSAM seam. This header is intentionally not installed.

#include <gtsam/nonlinear/ISAM2.h>

#include <cstddef>

#include "candidate_isolation_api.hpp"

namespace meridian::local::gtsam_api {

struct CandidateCacheSetStamp {
  std::size_t stateful_factors{};
  core::ContentHash semantic_hash{};

  bool operator==(const CandidateCacheSetStamp&) const = default;
};

struct CandidateGlobalizationResult {
  gtsam::Values estimate;
  double error{};
  double step_scale{1.0};
  std::size_t backtracking_trials{};
  bool rejected_full_step{};
  bool cauchy_direction_attempted{};
  bool cauchy_step_accepted{};
  std::size_t cauchy_backtracking_trials{};
  bool zero_step{};
};

// An iSAM2 copy shares nonlinear factor pointers by default. That is unsafe for
// factors whose linearize() method publishes a mutable cache. This adapter
// preserves the copied Bayes tree, VariableIndex, and factor slots, then
// replaces every CandidateIsolationApi factor with a cache-isolated clone at
// the identical slot and with identical ordered keys.
class CandidateIsolatedISAM2 final : public gtsam::ISAM2 {
public:
  explicit CandidateIsolatedISAM2(const gtsam::ISAM2Params& params);
  CandidateIsolatedISAM2(const CandidateIsolatedISAM2& other);

  CandidateIsolatedISAM2& operator=(const CandidateIsolatedISAM2&) = delete;
  CandidateIsolatedISAM2(CandidateIsolatedISAM2&&) = delete;
  CandidateIsolatedISAM2& operator=(CandidateIsolatedISAM2&&) = delete;

  [[nodiscard]] CandidateCacheSetStamp candidateCacheSetStamp() const;
  [[nodiscard]] bool cacheHandlesDisjointFrom(const CandidateIsolatedISAM2& other) const noexcept;

  // Replace only the currently published nonlinear estimate while preserving
  // this candidate's factor slots, Bayes tree, and linearization point.  This
  // is the private seam used by bounded actual-objective globalization: the
  // next forceFullSolve update relinearizes every non-fixed variable from the
  // accepted damped delta.  It is deliberately unavailable on the public
  // LocalGraph API.
  void setGlobalizedEstimate(const gtsam::Values& estimate);

  // Accept a complete Gauss--Newton estimate only when it has a finite,
  // non-increasing exact nonlinear objective. Otherwise perform one bounded
  // deterministic backtracking sequence along that same finite manifold
  // direction. If it is invalid or not a descent direction, use the
  // linearized graph's Cauchy direction and the same bounded actual-objective
  // check. Publish the accepted trust-region delta through
  // setGlobalizedEstimate().
  [[nodiscard]] CandidateGlobalizationResult globalizeFullStep(
      const gtsam::Values& previous, double previous_error, gtsam::Values full_step,
      double full_step_error, std::size_t maximum_backtracking_steps, double backtracking_reduction,
      bool full_step_is_physically_converged = false,
      double objective_stabilization_tolerance = 0.0);

  // Return only the existing Bayes-tree descendants whose frontal variables
  // must be re-eliminated to make the requested keys leaves. This mirrors the
  // affected-key traversal used by GTSAM's incremental fixed-lag smoother and
  // avoids forcing an unrelated active window through elimination.
  [[nodiscard]] gtsam::FastList<gtsam::Key> affectedKeysForLeafMarginalization(
      const gtsam::FastList<gtsam::Key>& requested_leaf_keys) const;
};

}  // namespace meridian::local::gtsam_api
