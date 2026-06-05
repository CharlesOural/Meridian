#pragma once

#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// The injectable pose-at-time seam used by the standalone deskew warp. The estimation
// frame F_e at time t, expressed in the deskew reference frame (the scan-end / anchor
// pose's frame). The pipeline injects the producer: ImuOnlyDeskew at cold-start /
// restart, and an L2-spline-backed provider in steady state. This package depends only
// on the seam, never on the front-end.
class IDeskewProvider {
 public:
  virtual ~IDeskewProvider() = default;

  // Pose T_ref_Fe of F_e at time t in the deskew reference frame. Must be answerable
  // for any t in [scan.stamp_start, scan.stamp_start + sweep_duration]. Returns false
  // if t is outside the provider's valid horizon (triggers the window-restart fallback).
  virtual bool poseAt(Timestamp t, Pose* T_ref_Fe) const = 0;

  // The reference time the warp compensates TO (scan-end by default). poseAt(anchor())
  // is identity by construction.
  virtual Timestamp anchor() const = 0;

  // True if poseAt() can answer for every point time of the given sweep span.
  virtual bool validHorizonCovers(Timestamp t_begin, Timestamp t_end) const = 0;

  // For telemetry: which provider is active this scan.
  enum class Kind { Spline, ImuOnly };
  virtual Kind kind() const = 0;
};

}  // namespace meridian
