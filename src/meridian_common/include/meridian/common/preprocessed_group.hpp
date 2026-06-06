#pragma once

#include <optional>

#include "meridian/common/measure_group.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// The L1->L2 currency: the assembled MeasureGroup plus the pipeline's cold-start
// deskew product. `deskewed` is the IMU-only deskewed sweep (points expressed in the
// body frame at the sweep end), absent until the IMU bootstrap converges.
// `cold_start` stays true until a front-end takes over deskew with its own
// trajectory; a steady-state front-end deskews implicitly at each point's own time
// and ignores `deskewed`.
struct PreprocessedGroup {
  MeasureGroup group;
  std::optional<LidarScan> deskewed;
  bool cold_start = true;
};

}  // namespace meridian
