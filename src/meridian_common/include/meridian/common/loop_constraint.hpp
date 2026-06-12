#pragma once

#include <cstdint>

#include "meridian/common/gaussian.hpp"
#include "meridian/common/pose.hpp"

namespace meridian {

// A verified loop closure emitted by L5 to L3. `cov` is translation-first (PoseCov6) and
// is reordered to GTSAM order at the back-end boundary; `fitness` (GICP inlier ratio)
// drives the back-end's robust kernel.
struct LoopConstraint {
  std::uint64_t from_id = 0;
  std::uint64_t to_id = 0;
  Pose T_from_to;      // GICP-refined relative transform
  PoseCov6 cov;        // post-PCM weighting
  double fitness = 0;  // GICP fitness / inlier ratio
};

}  // namespace meridian
