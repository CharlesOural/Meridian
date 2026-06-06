#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <cstdio>
#include <vector>

#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace basalt {
template <int, typename>
class So3Spline;
template <int, int, typename>
class RdSpline;
}  // namespace basalt

namespace meridian {

// Split cumulative B-spline trajectory: an SO(3) cubic spline for rotation and an
// R^3 cubic spline for translation, both running on a single uniform VIRTUAL-time
// knot grid. Real timestamps map through a per-segment table so knot density can
// vary (an outer real segment of knot_dt_ns may carry 1..n_cp_max control points)
// while the underlying basalt kernels only ever see a uniform grid.
//
// The real->virtual map is monotone and piecewise-linear: each outer real segment
// [t_k, t_k + knot_dt_ns) is stretched onto n_cp_k uniform virtual knot intervals,
// so its local slope dv/dt = n_cp_k * dt_v / knot_dt_ns is constant within the
// segment and changes only at outer-segment boundaries. Real-time derivatives are
// the virtual-time derivatives from basalt scaled by powers of that slope.
//
// Knot storage is the basalt aligned_deque, whose element addresses are stable
// across push_back/pop_front, so the raw pointers handed out by segmentFor() stay
// valid as Ceres parameter blocks while the window grows.
class SplineWindow {
public:
  // knot_dt_ns is the nominal outer-segment cadence; n_cp_max caps how many control
  // points one outer segment may receive. The virtual knot spacing is fixed at
  // knot_dt_ns so that an n_cp==1 segment is the uniform-spline special case.
  SplineWindow(Duration knot_dt_ns, int n_cp_max);
  ~SplineWindow();

  SplineWindow(const SplineWindow&) = delete;
  SplineWindow& operator=(const SplineWindow&) = delete;
  SplineWindow(SplineWindow&&) noexcept;
  SplineWindow& operator=(SplineWindow&&) noexcept;

  // Seeds the first N knots all at T0 so the spline is immediately evaluable at t0
  // and starts exactly at T0. The first outer segment opens at t0.
  void initialize(Timestamp t0, const Pose& T0);

  // Extends knot coverage to at least t. The new outer segment receives n_cp
  // (clamped to [1, n_cp_max]) control points placed at uniform real-time steps
  // across the segment; each new knot is seeded from `seed` evaluated at that
  // knot's real time. Control points are not on-trajectory poses, but seeding them
  // with the predicted pose at their knot time is the standard IMU warm start.
  void extendTo(Timestamp t, const std::function<Pose(Timestamp)>& seed, int n_cp);

  // Overwrites every knot from deque index `from_idx` onward with `seed` evaluated one
  // local knot step back (the same warm-start placement rule extendTo uses). Used to
  // refresh trailing knots that carry no measurement support yet: their stored values
  // are pure extrapolation, so each new sweep's IMU prediction supersedes them.
  void reseedFrom(int from_idx, const std::function<Pose(Timestamp)>& seed);

  // Drops the n oldest knots from the front of the trajectory, advancing minTime to
  // the new front knot's real time. Pointers to surviving knots stay valid (deque
  // pop_front invalidates only the erased element), so Ceres parameter blocks and the
  // marginalization prior that still reference kept knots remain sound. n is clamped
  // so at least four knots (the cubic support) remain; dropping more than that is the
  // caller's responsibility (it must have marginalized the dropped knots first).
  void dropOldest(int n);
  // Lowest deque index whose knot storage backs any of the given parameter-block
  // pointers (max int when none does). Bounds front-trimming so externally held
  // knot pointers stay valid.
  int lowestKnotIndexOf(const std::vector<const double*>& ptrs) const;

  Timestamp minTime() const;
  Timestamp maxTime() const;
  bool covers(Timestamp t) const;

  Pose pose(Timestamp t) const;  // T_W_Fe(t)

  // omega in F_e satisfying R(t+dt) = R(t) * exp(omega * dt) (right/body
  // perturbation), so this is directly comparable to a de-biased gyro reading.
  Eigen::Vector3d angularVelocityBody(Timestamp t) const;

  Eigen::Vector3d linearVelocityWorld(Timestamp t) const;

  // d2p/dt2 of the world translation, no gravity term; the caller adds gravity to
  // compare against a de-biased accelerometer reading.
  Eigen::Vector3d linearAccelWorld(Timestamp t) const;

  // Ceres plumbing: the 4+4 knot parameter blocks influencing time t, the normalized
  // segment position, and the per-knot spacing in seconds. A cost functor that feeds
  // the basalt spline helper inv_dt = 1/dt_s recovers real-time derivatives directly,
  // so the residual never has to reason about the virtual-time remap.
  struct SegmentRef {
    // Each so3 pointer addresses one SO(3) knot's quaternion storage laid out as
    // (x, y, z, w) -- the Eigen::Quaterniond coeffs order, matching Sophus' SO3
    // internal storage. Each r3 pointer addresses one 3-vector knot.
    std::array<double*, 4> so3_knots;
    std::array<double*, 4> r3_knots;
    double u;     // normalized position in [0,1) within the segment
    double dt_s;  // real per-knot spacing of this segment [s]
  };
  SegmentRef segmentFor(Timestamp t);

  // The knot index i such that segmentFor(t) returns knots i..i+3; the leading support
  // knot of the segment covering t. Lets a caller relate marginalized/kept knots back
  // to deque positions when trimming the front.
  int leadingKnotIndex(Timestamp t) const;

  // Mutable access to one resident knot's raw storage: so3_knot[i] is the i-th SO(3)
  // quaternion (x,y,z,w), r3_knot[i] the i-th R^3 vector. Lets a caller (e.g. a rigid
  // odom re-anchor) sweep the live knots directly instead of re-deriving them by
  // sampling segments across the covered timeline.
  double* so3KnotData(int i);
  double* r3KnotData(int i);

  int numKnots() const;
  // Diagnostic: prints the last n knots (grid time, spacing, position) to f.
  void dumpTail(std::FILE* f, int n) const;

private:
  // One outer real segment: its real start time, the virtual time at its first new
  // knot, and how many control points it carries. Kept for diagnostics/placement;
  // the evaluable real->virtual map itself lives entirely in kt_.
  struct Segment {
    Timestamp t_real_start;
    int64_t v_start_ns;
    int n_cp;
  };

  // Resolves a real time to (knot interval index i, normalized position u in [0,1),
  // local slope dv/dt). Knots i..i+N-1 are the support of the evaluation; u and slope
  // are exact in double precision (no integer-nanosecond quantization).
  int locate(Timestamp t, double* u, double* slope) const;

  bool initialized_ = false;
  Duration knot_dt_ns_;
  int64_t dt_v_ns_;  // uniform virtual knot spacing in ns (== knot_dt_ns_)
  int n_cp_max_;

  // Real-time coverage bounds, tracked alongside the virtual grid so callers query
  // the timeline they actually live on.
  Timestamp t_min_ = 0;
  Timestamp t_max_ = 0;

  // Real time of every knot, strictly increasing. Knot i is anchored to virtual time
  // i * dt_v_ns_; linear interpolation between consecutive anchors is the monotone
  // piecewise-linear real->virtual map. This is the single source of truth for time
  // remapping; index i in kt_ matches index i in both basalt knot deques.
  std::vector<Timestamp> kt_;

  std::vector<Segment> segments_;

  // Owned so the header stays free of the heavyweight basalt template includes.
  std::unique_ptr<basalt::So3Spline<4, double>> so3_;
  std::unique_ptr<basalt::RdSpline<3, 4, double>> r3_;
};

}  // namespace meridian
