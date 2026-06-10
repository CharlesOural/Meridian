#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <optional>
#include <unordered_map>

#include "meridian/common/pose.hpp"

namespace meridian::backend {

// Accumulates the odometry chain's relative-pose uncertainty so the covariance of any
// chain-composed relative pose a->b can be recovered without touching the graph. Every
// covariance is right-perturbation, translation-first [rho; phi]. Keyframe ids are assumed
// monotonic along the chain, so "a older than b" is the numeric order a < b.
class ChainCovariance {
public:
  // Resets the chain and anchors it at `first_id` with identity pose and zero covariance.
  void start(std::uint64_t first_id);

  // Appends one edge: T_chain(to) = T_chain(from) * T_from_to, with `cov` the covariance of
  // T_from_to. A no-op when `from_id` is unknown, so a gap leaves all downstream ids unknown.
  void extend(std::uint64_t from_id, std::uint64_t to_id, const Pose& T_from_to,
              const Eigen::Matrix<double, 6, 6>& cov);

  // Covariance of the chain-composed relative pose a->b (a older), expressed in b's tangent;
  // symmetric PSD. nullopt if either id is unknown or a is not older than b.
  std::optional<Eigen::Matrix<double, 6, 6>> between(std::uint64_t a, std::uint64_t b) const;

  bool known(std::uint64_t id) const;

private:
  struct Node {
    Pose T_chain;                           // anchor -> id
    Eigen::Matrix<double, 6, 6> cov_chain;  // of T_chain, in id's tangent
  };
  std::unordered_map<std::uint64_t, Node> nodes_;
};

}  // namespace meridian::backend
