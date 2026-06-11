#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
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
  //
  // non_uniform selects how unequal real knot spacings are handled, immutably for
  // the lifetime of the window:
  //  - false: the piecewise-linear real->virtual warp. Uniform virtual-grid blending
  //    everywhere; real-time derivatives are virtual-time derivatives scaled by the
  //    per-interval warp slope. C2 in virtual time, but only C0 in real time across
  //    knot-spacing changes (velocity/acceleration jump where the slope jumps).
  //  - true: true non-uniform B-spline evaluation. A knot interval whose real spacing
  //    actually varies carries its own blending matrix derived from the real knot
  //    times, so the trajectory is C2 in real time across knot-density changes;
  //    derivatives scale by the exact per-interval real span. An interval whose whole
  //    cubic support is equally spaced at the base cadence instead routes through the
  //    cardinal kernels (no per-interval matrix), running the exact same floating-point
  //    operations as the warp build, so a uniform stretch evaluates byte-for-byte
  //    identically to it. With the mode off, no per-interval matrix code runs at all.
  SplineWindow(Duration knot_dt_ns, int n_cp_max, bool non_uniform);
  SplineWindow(Duration knot_dt_ns, int n_cp_max);
  ~SplineWindow();

  SplineWindow(const SplineWindow&) = delete;
  SplineWindow& operator=(const SplineWindow&) = delete;
  SplineWindow(SplineWindow&&) noexcept;
  SplineWindow& operator=(SplineWindow&&) noexcept;

  // Deep copy carrying every knot VALUE and the full real->virtual time map, so the
  // copy evaluates bit-identically to the original and owns independent knot storage.
  // The copy's knot pointers (segmentFor / so3KnotData / r3KnotData) are distinct from
  // the original's, so residuals rebuilt against the copy point at copy-owned storage
  // a later mutation of the original cannot disturb.
  std::unique_ptr<SplineWindow> clone() const;

  // Seeds the first N knots all at T0 so the spline is immediately evaluable at t0
  // and starts exactly at T0. The first outer segment opens at t0.
  void initialize(Timestamp t0, const Pose& T0);

  // Extends knot coverage to at least t. Each appended outer segment receives n_cp
  // (clamped to [1, n_cp_max]) control points placed at uniform real-time steps
  // across the segment; each new knot is seeded from `seed` evaluated at that
  // knot's real time. Control points are not on-trajectory poses, but seeding them
  // with the predicted pose at their knot time is the standard IMU warm start.
  void extendTo(Timestamp t, const std::function<Pose(Timestamp)>& seed, int n_cp);

  // Same extension loop, but the control-point count is chosen per appended outer
  // segment: density(seg_start, seg_end) is queried once per segment and clamped to
  // [1, n_cp_max]. The fixed-count overload above delegates here with a constant
  // density, with identical arithmetic and knot placement.
  void extendTo(Timestamp t, const std::function<Pose(Timestamp)>& seed,
                const std::function<int(Timestamp seg_start, Timestamp seg_end)>& density);

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
  // Highest deque index whose knot storage backs any of the given parameter-block
  // pointers (-1 when none does). Bounds tail truncation the same way
  // lowestKnotIndexOf bounds front-trimming.
  int highestKnotIndexOf(const std::vector<const double*>& ptrs) const;

  // Removes every trailing whole outer segment whose record starts at or after
  // from_seg_start, popping its knots from the back of both knot deques and from the
  // knot-time table, and restores coverage to the surviving knots. At least the four
  // oldest knots always survive (a segment whose removal would leave fewer is kept
  // whole). Pointers to surviving knots stay valid (pop_back invalidates only the
  // erased elements). Returns the number of knots removed.
  int truncateTailSegments(Timestamp from_seg_start);

  // Outer-segment boundaries and control-point count ("gear") of the segment record
  // covering t. The per-segment records are authoritative for these queries; when no
  // record covers t (e.g. the bootstrap knots, which predate any record), the answer
  // falls back to the knot interval containing t, which spans exactly one nominal
  // cadence there: [knotTime(locate(t)), knotTime(locate(t)) + knot_dt_ns), gear 1.
  Timestamp segmentStartContaining(Timestamp t) const;
  Timestamp segmentEndContaining(Timestamp t) const;
  int segmentGearContaining(Timestamp t) const;

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
    // Per-interval blending matrices (plain and cumulative), set only for an interval
    // whose real knot spacing genuinely varies; nullptr otherwise -- including every
    // interval of a disabled-mode window and every uniformly-spaced interval of a
    // non-uniform one. They point at window-owned storage that stays valid as long as
    // the interval's knots are resident, exactly like the knot pointers. A consumer
    // that sees nullptr falls back to the uniform cardinal matrices, which are exact
    // for equally spaced knots.
    const Eigen::Matrix4d* blend = nullptr;
    const Eigen::Matrix4d* cum_blend = nullptr;
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
  // Real time of knot i (the kt_ table entry). Valid for i in [0, numKnots()).
  Timestamp knotTime(int i) const;

  // Number of cached per-interval blending matrices: numKnots() - 3 in non-uniform
  // mode (one per evaluable knot interval), 0 in uniform mode. Introspection for
  // invariant checks; evaluation indexes the cache directly.
  int intervalBasisCount() const;

private:
  // One outer real segment: its real start time, the virtual time at its first new
  // knot, and how many control points it carries. Kept for diagnostics/placement;
  // the evaluable real->virtual map itself lives entirely in kt_.
  struct Segment {
    Timestamp t_real_start;
    int64_t v_start_ns;
    int n_cp;
  };

  // Per-interval cubic blending matrices for non-uniform evaluation: coefficients of
  // the four B-spline bases alive on one knot interval, as polynomials in the local
  // parameter u. Row j holds the u-coefficients (constant..cubic) of the basis paired
  // with control point i+j, so basis weights are m * (1, u, u^2, u^3)^T. cm is the
  // cumulative form (row j replaced by the sum of rows j..3). When `uniform` is set the
  // interval's whole six-knot support is equally spaced at the base cadence: the
  // matrices are left zero and never read, because evaluation routes through the
  // cardinal kernels instead (byte-identical to the disabled build), where the runtime
  // matrix would otherwise land ~1 ULP off the hard-coded cardinal constant.
  struct IntervalBasis {
    Eigen::Matrix4d m;
    Eigen::Matrix4d cm;
    bool uniform = false;
  };

  // Resolves a real time to (knot interval index i, normalized position u in [0,1),
  // local slope dv/dt). Knots i..i+N-1 are the support of the evaluation; u and slope
  // are exact in double precision (no integer-nanosecond quantization).
  int locate(Timestamp t, double* u, double* slope) const;

  // The segment record whose outer span [t_real_start, t_real_start + knot_dt_ns)
  // covers t, or nullptr when none does (bootstrap knots, or t outside all records).
  const Segment* segmentRecordContaining(Timestamp t) const;

  // Real time of knot m; negative m extrapolates the front spacing backwards
  // (kt_[0] + m * (kt_[1] - kt_[0])), supplying the phantom left neighbors the first
  // two knot intervals need for a complete six-knot basis neighborhood.
  int64_t knotTimeExtrapolated(int m) const;

  // Blending matrices of knot interval i, computed from the real times of the six
  // knots kt_[i-2 .. i+3] (the only knots the four bases alive on the interval
  // depend on). A pure function of those times, so recomputation after a tail
  // truncation reproduces the original bits. When those six knots are equally spaced
  // at the base cadence the result is flagged uniform and the matrices are left unset:
  // the interval evaluates through the cardinal kernels instead.
  IntervalBasis makeIntervalBasis(int i) const;

  // Appends matrices for every knot interval that has become evaluable, restoring
  // the invariant mats_.size() == max(0, numKnots - 3). No-op in uniform mode. Runs
  // only on the single-threaded mutation paths, never during evaluation.
  void syncIntervalBases();

  bool initialized_ = false;
  Duration knot_dt_ns_;
  int64_t dt_v_ns_;  // uniform virtual knot spacing in ns (== knot_dt_ns_)
  int n_cp_max_;
  bool non_uniform_ = false;

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

  // Non-uniform mode only: one entry per knot interval 0..numKnots-4, index i paired
  // with the interval [kt_[i], kt_[i+1]). Entries whose interval is uniformly spaced
  // are flagged and carry no matrix (evaluation routes to the cardinal kernels); the
  // rest hold the interval's per-interval blending matrices. Deque so push_back/
  // pop_front keep element addresses stable -- SegmentRef::blend/cum_blend point into
  // these entries with the same lifetime contract as the knot pointers. Empty in
  // uniform mode.
  std::deque<IntervalBasis, Eigen::aligned_allocator<IntervalBasis>> mats_;

  // Owned so the header stays free of the heavyweight basalt template includes.
  std::unique_ptr<basalt::So3Spline<4, double>> so3_;
  std::unique_ptr<basalt::RdSpline<3, 4, double>> r3_;
};

}  // namespace meridian
