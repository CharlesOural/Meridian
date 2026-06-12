#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "meridian/common/pose.hpp"

namespace meridian::backend {

// A loop edge to re-judge. cov is the loop covariance in Meridian translation-first order.
struct GncLoop {
  std::size_t handle;
  std::uint64_t from_id, to_id;
  Pose T_from_to;
  Eigen::Matrix<double, 6, 6> cov;
};

// A trusted odometry between-edge (known inlier). cov translation-first.
struct GncOdom {
  std::uint64_t from_id, to_id;
  Pose T_from_to;
  Eigen::Matrix<double, 6, 6> cov;
};

struct GncConsolidationInput {
  std::vector<std::uint64_t> keyframes;              // ids appearing in the sub-problem
  std::unordered_map<std::uint64_t, Pose> estimate;  // current map pose per keyframe
  std::vector<GncOdom> odom;                         // known-inlier chain edges
  std::vector<GncLoop> loops;                        // loops to re-judge
  double barc2;                                      // inlier cost threshold = chi^2_{6,alpha}
  double reject_w;           // loops whose final GNC weight < reject_w are rejected
  double prior_sigma = 1.0;  // loose anchor sigma [rad/m] pinning each keyframe near its estimate
};

// Runs a batch GNC (Truncated Least Squares) on a COPY of the sub-problem: each keyframe gets a
// loose isotropic prior at its current estimate (so the sub-graph is well-constrained), odometry
// edges are added as known-inlier BetweenFactors, loops as robustifiable BetweenFactors. Returns
// the handles of loops GNC drives to weight < reject_w. Deterministic. Never throws on a singular
// sub-problem: on failure returns {} (reject nothing).
std::vector<std::size_t> gnc_consolidate(const GncConsolidationInput& in);

}  // namespace meridian::backend
