#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

// The outcome of Stage-C geometric verification of one candidate loop.
struct VerifiedLoop {
  bool accepted = false;
  Pose T_from_to;     // refined relative transform (maps `to` body into `from` body frame)
  double fitness = 0;  // overlap * exp(-rmse/sigma) in [0, 1]
  double overlap = 0;  // inlier fraction of the source cloud
  double rmse = 0;     // inlier residual proxy [m]
  double cond = 0;     // condition number of the GICP information matrix
  // small_gicp's final information matrix, left in its native ROTATION-FIRST [phi; rho]
  // order; the translation-first permutation happens once in covariance shaping.
  Eigen::Matrix<double, 6, 6> info_rot_first = Eigen::Matrix<double, 6, 6>::Zero();
};

// Stage C: refine a candidate loop by GICP between the candidate ("from", the target) and
// the query ("to", the source) clouds, from an initial relative guess, and score it. When
// constructed deterministic, preprocessing/registration run single-threaded so the result is
// bit-reproducible (small_gicp's parallel downsampling is non-deterministic at >1 thread).
class GicpVerifier {
 public:
  GicpVerifier(const PlaceConfig& cfg, bool deterministic);

  VerifiedLoop verify(const PointCloud& from_target, const PointCloud& to_source,
                      const Pose& init_from_to) const;

 private:
  PlaceConfig cfg_;
  int num_threads_;
};

}  // namespace meridian
