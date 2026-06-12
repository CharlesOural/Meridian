#pragma once

#include <cstdint>

#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// A keyframe pose in the map frame — an element of the back-end's corrected trajectory.
struct StampedPose {
  Timestamp stamp = 0;
  std::uint64_t kf_id = 0;
  Pose T_map_body;
};

}  // namespace meridian
