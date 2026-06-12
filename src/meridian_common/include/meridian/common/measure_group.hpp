#pragma once

#include <optional>
#include <vector>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// The per-sweep measurement bundle: one LiDAR sweep plus the IMU spanning it (and any
// image/GNSS in the interval). Built by the aggregator once per sweep; the pipeline
// wraps it as a PreprocessedGroup before the front-end ingests it.
struct MeasureGroup {
  Timestamp t_begin = 0;             // sweep start (= scan.stamp_start)
  Timestamp t_end = 0;               // sweep end (= stamp_start + sweep_duration)
  LidarScan scan;                    // the LiDAR sweep (shared-immutable points)
  std::vector<ImuSample> imu;        // samples in (lower, t_end] with
                                     // lower = min(prev_t_end, t_begin), plus the one
                                     // sample straddling t_begin; overlapping sweeps
                                     // (stamp jitter) share their common window
  std::optional<CameraFrame> image;  // image whose mid-exposure falls in the interval
  std::vector<GnssFix> gnss;         // fixes in the interval (usually 0 or 1)
};

}  // namespace meridian
