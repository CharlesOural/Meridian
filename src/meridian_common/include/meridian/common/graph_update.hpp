#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "meridian/common/gaussian.hpp"
#include "meridian/common/pose.hpp"

namespace meridian {

// The back-end's correction broadcast (L3 -> L4/L2): which keyframes moved and their new
// map-frame poses. L4 clears-and-reintegrates the affected region; L2 rebases its origin.
struct GraphUpdate {
  struct Moved {
    std::uint64_t id = 0;
    Pose new_T_map_body;
    std::optional<PoseCov6> cov;  // translation-first [rho; phi]
  };
  std::vector<Moved> moved;  // empty if optimize() changed nothing materially
  bool loop_closed = false;  // a loop just snapped (large rigid correction)
};

}  // namespace meridian
