#pragma once
#include <Eigen/Core>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "meridian/common/loop_constraint.hpp"
#include "meridian/common/pose.hpp"
namespace meridian::backend {

// One PCM pass result. Handles are the stable values returned by add(). to_admit: buffered,
// in the consistent max-clique, not yet in the graph. to_evict: in the graph but no longer
// in the clique (a newer, larger clique displaced them). to_reject: buffered loops that are
// testable (all four endpoints + chain cov available) but inconsistent with an established
// consensus clique (clique size >= 2 and the loop shares no consistency edge with any clique
// member) -> drop them so they are not retried forever.
struct PcmDecision {
  std::vector<std::size_t> to_admit;
  std::vector<std::size_t> to_evict;
  std::vector<std::size_t> to_reject;
};

class Pcm {
public:
  // chi2_thresh = chi^2_{6,alpha} (squared-Mahalanobis gate). max_nodes caps the exact clique.
  Pcm(double chi2_thresh, int max_nodes);

  using PoseFn = std::function<std::optional<Pose>(std::uint64_t)>;
  using ChainCovFn =
      std::function<std::optional<Eigen::Matrix<double, 6, 6>>(std::uint64_t, std::uint64_t)>;

  // Buffer a loop; returns its stable handle (index). The caller already passed the fitness gate.
  std::size_t add(const LoopConstraint& lc);

  // Recompute consistency among all buffered, not-yet-rejected loops and return the diff from a
  // bounded max-clique. pose(id) = current map-frame pose (nullopt if absent). chain_cov(a,b),
  // a<b = odometry-chain covariance of the relative pose a->b in b's tangent, translation-first
  // (nullopt if absent). Pure: does not mutate in-graph/rejected state — the caller applies the
  // diff and then calls mark_admitted/mark_evicted/mark_rejected.
  PcmDecision update(const PoseFn& pose, const ChainCovFn& chain_cov);

  const LoopConstraint& at(std::size_t handle) const;
  void mark_admitted(std::size_t handle);
  void mark_evicted(std::size_t handle);
  void mark_rejected(std::size_t handle);
  bool in_graph(std::size_t handle) const;
  std::size_t pending_count() const;  // buffered, not in graph, not rejected

  // Pairwise squared-Mahalanobis consistency distance between two buffered loops. nullopt if
  // any endpoint pose or chain cov is unavailable. Exposed for testing.
  std::optional<double> pair_distance_sq(std::size_t a, std::size_t b, const PoseFn& pose,
                                         const ChainCovFn& chain_cov) const;

private:
  enum class State { Pending, InGraph, Rejected };
  struct Entry {
    LoopConstraint lc;
    State state = State::Pending;
  };

  // Covariance of the relative pose X_a^{-1} X_b in its own right tangent, recovered from the
  // odometry chain covariance of the older->newer relative. When a is newer than b the relative
  // is the inverse and the chain cov is transported by Ad_T, T = X_lo^{-1} X_hi (needs the poses).
  // nullopt if either endpoint pose or the chain cov is unavailable.
  std::optional<Eigen::Matrix<double, 6, 6>> relative_cov(std::uint64_t a_id, std::uint64_t b_id,
                                                          const PoseFn& pose,
                                                          const ChainCovFn& chain_cov) const;

  std::vector<Entry> entries_;
  double chi2_thresh_;
  int max_nodes_;
};
}  // namespace meridian::backend
