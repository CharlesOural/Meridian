#pragma once

#include <cstdint>
#include <memory>

#include "meridian/common/cloud.hpp"
#include "meridian/common/frame.hpp"
#include "meridian/common/observability.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

namespace meridian::backend {

// Per-keyframe bookkeeping kept outside the factor graph: the shared-immutable data
// handles to republish, and the last pose actually emitted so a GraphUpdate can carry
// only the keyframes that moved materially since.
struct KfRecord {
  Timestamp stamp = 0;
  PointCloudPtr cloud_body;
  std::shared_ptr<const CameraFrame> image;
  ObservabilityReport observability;
  Frame ref_frame = Frame::Odom;
  Pose T_ref_body;  // odom hint at emission time
  std::uint32_t calib_version = 0;
  Pose last_emitted;  // last pose published in a GraphUpdate
  bool ever_emitted = false;
};

}  // namespace meridian::backend
