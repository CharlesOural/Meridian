#include "candidate_isolated_isam2.hpp"

#include <gtsam/linear/GaussianFactorGraph.h>

#include <algorithm>
#include <boost/pointer_cast.hpp>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>

namespace meridian::local::gtsam_api {
namespace {

inline constexpr std::string_view kCandidateCacheSetDomain{"meridian.local.candidate_cache_set"};
inline constexpr std::uint32_t kCandidateCacheSetSchemaVersion{1U};

[[nodiscard]] const CandidateIsolationApi* isolationApi(
    const gtsam::NonlinearFactor::shared_ptr& factor) noexcept {
  return factor ? dynamic_cast<const CandidateIsolationApi*>(factor.get()) : nullptr;
}

[[nodiscard]] bool finiteCompleteDirection(const gtsam::Values& values,
                                           const gtsam::VectorValues& direction) {
  const gtsam::VectorValues zero = values.localCoordinates(values);
  if (direction.size() != zero.size()) {
    return false;
  }
  for (const auto& [key, expected] : zero) {
    if (!direction.exists(key)) {
      return false;
    }
    const gtsam::Vector& value = direction.at(key);
    if (value.size() != expected.size() || !expected.allFinite() || !value.allFinite()) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<gtsam::VectorValues> finiteLocalDirection(
    const gtsam::Values& from, const gtsam::Values& to) noexcept {
  try {
    gtsam::VectorValues direction = from.localCoordinates(to);
    if (finiteCompleteDirection(from, direction)) {
      return direction;
    }
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

void markAffectedDescendants(gtsam::Key requested_leaf_key,
                             const gtsam::ISAM2::sharedClique& clique,
                             std::set<gtsam::Key>& affected_keys) {
  if (!clique || !clique->conditional()) {
    throw std::runtime_error(
        "fixed-lag affected-key traversal encountered an empty Bayes-tree clique");
  }
  const auto& conditional = clique->conditional();
  if (std::find(conditional->beginParents(), conditional->endParents(), requested_leaf_key) ==
      conditional->endParents()) {
    // If the requested key is absent from this separator, it cannot occur in
    // any descendant separator.
    return;
  }
  affected_keys.insert(conditional->beginFrontals(), conditional->endFrontals());
  for (const gtsam::ISAM2::sharedClique& child : clique->children) {
    markAffectedDescendants(requested_leaf_key, child, affected_keys);
  }
}

}  // namespace

CandidateIsolatedISAM2::CandidateIsolatedISAM2(const gtsam::ISAM2Params& params)
    : gtsam::ISAM2(params) {}

CandidateIsolatedISAM2::CandidateIsolatedISAM2(const CandidateIsolatedISAM2& other)
    : gtsam::ISAM2(other) {
  for (gtsam::FactorIndex index = 0U; index < nonlinearFactors_.size(); ++index) {
    const gtsam::NonlinearFactor::shared_ptr& original = other.nonlinearFactors_.at(index);
    const CandidateIsolationApi* const original_api = isolationApi(original);
    if (original_api == nullptr) {
      continue;
    }

    gtsam::NonlinearFactor::shared_ptr clone = original_api->cloneForCandidate();
    const CandidateIsolationApi* const clone_api = isolationApi(clone);
    if (!clone || clone.get() == original.get() || clone_api == nullptr) {
      throw std::runtime_error(
          "candidate-isolated iSAM2 factor clone is null, aliased, or missing its isolation API");
    }
    if (clone->keys() != original->keys()) {
      throw std::runtime_error("candidate-isolated iSAM2 factor clone changed its ordered keys");
    }
    if (clone_api->candidateCacheHandleIdentity() == original_api->candidateCacheHandleIdentity()) {
      throw std::runtime_error(
          "candidate-isolated iSAM2 factor clone shares its mutable cache handle");
    }
    if (clone_api->candidateCacheStamp() != original_api->candidateCacheStamp()) {
      throw std::runtime_error(
          "candidate-isolated iSAM2 factor clone changed its initial immutable cache state");
    }
    nonlinearFactors_.replace(index, std::move(clone));
  }

  if (candidateCacheSetStamp() != other.candidateCacheSetStamp() ||
      !cacheHandlesDisjointFrom(other)) {
    throw std::runtime_error(
        "candidate-isolated iSAM2 copy violated the graph-wide cache-set invariant");
  }
}

CandidateCacheSetStamp CandidateIsolatedISAM2::candidateCacheSetStamp() const {
  std::size_t stateful_factors = 0U;
  std::uint64_t ordered_keys = 0U;
  for (const gtsam::NonlinearFactor::shared_ptr& factor : nonlinearFactors_) {
    if (isolationApi(factor) == nullptr) {
      continue;
    }
    ++stateful_factors;
    const auto key_count = static_cast<std::uint64_t>(factor->keys().size());
    if (ordered_keys > std::numeric_limits<std::uint64_t>::max() - key_count) {
      throw std::runtime_error("candidate cache-set key count exceeds canonical bounds");
    }
    ordered_keys += key_count;
  }
  constexpr std::uint64_t kFixedMaximumBytes = 256U;
  constexpr std::uint64_t kMaximumBytesPerFactor = 96U;
  constexpr std::uint64_t kMaximumBytesPerKey = 16U;
  if (stateful_factors >
      (std::numeric_limits<std::uint64_t>::max() - kFixedMaximumBytes) / kMaximumBytesPerFactor) {
    throw std::runtime_error("candidate cache-set is too large for canonical encoding");
  }
  const auto factor_bytes =
      kFixedMaximumBytes + kMaximumBytesPerFactor * static_cast<std::uint64_t>(stateful_factors);
  if (ordered_keys >
      (std::numeric_limits<std::uint64_t>::max() - factor_bytes) / kMaximumBytesPerKey) {
    throw std::runtime_error("candidate cache-set keys are too large for canonical encoding");
  }
  const auto maximum_bytes = factor_bytes + kMaximumBytesPerKey * ordered_keys;
  auto encoder = core::CanonicalEncoder::create(kCandidateCacheSetDomain,
                                                kCandidateCacheSetSchemaVersion, maximum_bytes);
  if (!encoder) {
    throw std::runtime_error("candidate cache-set encoder initialization failed");
  }
  const auto write = [](core::CanonicalEncodingError result) {
    return result == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeU64(static_cast<std::uint64_t>(stateful_factors)))) {
    throw std::runtime_error("candidate cache-set count encoding failed");
  }
  for (gtsam::FactorIndex index = 0U; index < nonlinearFactors_.size(); ++index) {
    const gtsam::NonlinearFactor::shared_ptr& factor = nonlinearFactors_.at(index);
    const CandidateIsolationApi* const api = isolationApi(factor);
    if (api == nullptr) {
      continue;
    }
    const CandidateCacheStamp stamp = api->candidateCacheStamp();
    if (!write(encoder.value().writeU64(static_cast<std::uint64_t>(index))) ||
        !write(encoder.value().writeU64(static_cast<std::uint64_t>(factor->keys().size())))) {
      throw std::runtime_error("candidate cache-set slot encoding failed");
    }
    for (const gtsam::Key key : factor->keys()) {
      if (!write(encoder.value().writeU64(static_cast<std::uint64_t>(key)))) {
        throw std::runtime_error("candidate cache-set key encoding failed");
      }
    }
    if (!write(encoder.value().writeU64(stamp.revision)) ||
        !write(encoder.value().writeHash(stamp.semantic_hash))) {
      throw std::runtime_error("candidate cache-set stamp encoding failed");
    }
  }
  auto finalized = encoder.value().finish();
  if (!finalized) {
    throw std::runtime_error("candidate cache-set finalization failed");
  }
  return CandidateCacheSetStamp{stateful_factors, finalized.value().digest()};
}

bool CandidateIsolatedISAM2::cacheHandlesDisjointFrom(
    const CandidateIsolatedISAM2& other) const noexcept {
  if (nonlinearFactors_.size() != other.nonlinearFactors_.size()) {
    return false;
  }
  for (gtsam::FactorIndex index = 0U; index < nonlinearFactors_.size(); ++index) {
    const auto& factor = nonlinearFactors_.at(index);
    const auto& other_factor = other.nonlinearFactors_.at(index);
    const CandidateIsolationApi* const api = isolationApi(factor);
    const CandidateIsolationApi* const other_api = isolationApi(other_factor);
    if ((api == nullptr) != (other_api == nullptr)) {
      return false;
    }
    if (api == nullptr) {
      continue;
    }
    if (!factor || !other_factor || factor.get() == other_factor.get() ||
        factor->keys() != other_factor->keys() ||
        api->candidateCacheHandleIdentity() == other_api->candidateCacheHandleIdentity()) {
      return false;
    }
  }
  return true;
}

void CandidateIsolatedISAM2::setGlobalizedEstimate(const gtsam::Values& estimate) {
  gtsam::VectorValues replacement_delta = theta_.localCoordinates(estimate);
  if (replacement_delta.size() != delta_.size()) {
    throw std::invalid_argument("globalized iSAM2 estimate does not match the live variable set");
  }
  for (const auto& [key, value] : replacement_delta) {
    if (!value.allFinite() || !delta_.exists(key) || delta_.at(key).size() != value.size()) {
      throw std::invalid_argument(
          "globalized iSAM2 estimate has a non-finite or mismatched tangent");
    }
  }
  delta_ = std::move(replacement_delta);
  // delta_ is now complete by construction.  Leaving any replaced-key mask
  // set would cause calculateEstimate() to overwrite the accepted trust-region
  // step with a fresh unglobalized back-substitution.
  deltaReplacedMask_.clear();
}

CandidateGlobalizationResult CandidateIsolatedISAM2::globalizeFullStep(
    const gtsam::Values& previous, double previous_error, gtsam::Values full_step,
    double full_step_error, std::size_t maximum_backtracking_steps, double backtracking_reduction,
    bool full_step_is_physically_converged, double objective_stabilization_tolerance) {
  if (!std::isfinite(previous_error) || maximum_backtracking_steps == 0U ||
      !std::isfinite(backtracking_reduction) || backtracking_reduction <= 0.0 ||
      backtracking_reduction >= 1.0 || !std::isfinite(objective_stabilization_tolerance) ||
      objective_stabilization_tolerance < 0.0) {
    throw std::invalid_argument("candidate globalization inputs are invalid");
  }

  CandidateGlobalizationResult result;
  result.estimate = std::move(full_step);
  result.error = full_step_error;
  if (std::isfinite(full_step_error) && full_step_error <= previous_error) {
    setGlobalizedEstimate(result.estimate);
    return result;
  }

  // The exact objective is accumulated through many floating-point factor
  // evaluations. At a physically converged point, the next raw Bayes-tree
  // delta can therefore be a few ulps uphill even though every meaningful
  // state component is already below its configured resolution. Retain the
  // previous Values immediately in that one bounded case. This accepts no
  // increasing estimate and leaves both the material-overshoot line search
  // and the final transaction gate unchanged.
  if (full_step_is_physically_converged && std::isfinite(full_step_error) &&
      full_step_error > previous_error &&
      full_step_error - previous_error <= objective_stabilization_tolerance) {
    result.estimate = previous;
    result.error = previous_error;
    result.step_scale = 0.0;
    result.rejected_full_step = true;
    result.zero_step = true;
    setGlobalizedEstimate(result.estimate);
    return result;
  }

  result.rejected_full_step = true;
  double step_scale = 1.0;
  const std::optional<gtsam::VectorValues> direction =
      finiteLocalDirection(previous, result.estimate);
  if (direction) {
    for (std::size_t trial = 0U; trial < maximum_backtracking_steps; ++trial) {
      step_scale *= backtracking_reduction;
      ++result.backtracking_trials;
      gtsam::Values trial_estimate = previous.retract(step_scale * *direction);
      const double trial_error = nonlinearFactors_.error(trial_estimate);
      if (std::isfinite(trial_error) && trial_error <= previous_error) {
        result.estimate = std::move(trial_estimate);
        result.error = trial_error;
        result.step_scale = step_scale;
        setGlobalizedEstimate(result.estimate);
        return result;
      }
    }
  }

  result.cauchy_direction_attempted = true;
  const gtsam::GaussianFactorGraph::shared_ptr linearized = nonlinearFactors_.linearize(previous);
  std::optional<gtsam::VectorValues> cauchy_direction;
  if (linearized) {
    try {
      gtsam::VectorValues proposed = linearized->optimizeGradientSearch();
      const double norm = proposed.norm();
      if (finiteCompleteDirection(previous, proposed) && std::isfinite(norm) && norm > 0.0) {
        cauchy_direction = std::move(proposed);
      }
    } catch (const std::exception&) {
      // A singular/empty linear graph has no usable positive Cauchy direction.
    }
  }
  if (cauchy_direction) {
    step_scale = 1.0;
    for (std::size_t trial = 0U; trial < maximum_backtracking_steps; ++trial) {
      ++result.cauchy_backtracking_trials;
      gtsam::Values trial_estimate = previous.retract(step_scale * *cauchy_direction);
      const double trial_error = nonlinearFactors_.error(trial_estimate);
      if (std::isfinite(trial_error) && trial_error <= previous_error) {
        result.estimate = std::move(trial_estimate);
        result.error = trial_error;
        result.step_scale = step_scale;
        result.cauchy_step_accepted = true;
        setGlobalizedEstimate(result.estimate);
        return result;
      }
      step_scale *= backtracking_reduction;
    }
  }

  // A zero trust-region step retains the last accepted estimate while keeping
  // every newly inserted factor and its exact slot in this isolated solver.
  result.estimate = previous;
  result.error = previous_error;
  result.step_scale = 0.0;
  result.zero_step = true;
  setGlobalizedEstimate(result.estimate);
  return result;
}

gtsam::FastList<gtsam::Key> CandidateIsolatedISAM2::affectedKeysForLeafMarginalization(
    const gtsam::FastList<gtsam::Key>& requested_leaf_keys) const {
  std::set<gtsam::Key> affected_keys;
  for (const gtsam::Key requested_leaf_key : requested_leaf_keys) {
    const gtsam::ISAM2::sharedClique containing_clique = (*this)[requested_leaf_key];
    if (!containing_clique) {
      throw std::runtime_error("fixed-lag requested key has no Bayes-tree clique");
    }
    for (const gtsam::ISAM2::sharedClique& child : containing_clique->children) {
      markAffectedDescendants(requested_leaf_key, child, affected_keys);
    }
  }
  return gtsam::FastList<gtsam::Key>(affected_keys.begin(), affected_keys.end());
}

}  // namespace meridian::local::gtsam_api
