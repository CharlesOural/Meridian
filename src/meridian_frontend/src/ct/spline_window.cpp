#include "ct/spline_window.hpp"

#include <basalt/spline/ceres_spline_helper.h>
#include <basalt/spline/rd_spline.h>
#include <basalt/spline/so3_spline.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sophus/so3.hpp>
#include <unordered_set>
#include <vector>

namespace meridian {

namespace {

constexpr int kOrder = 4;  // cubic B-spline: order 4, degree 3

Sophus::SO3d toSophus(const Eigen::Quaterniond& q) {
  return Sophus::SO3d(q.normalized());
}

}  // namespace

SplineWindow::SplineWindow(Duration knot_dt_ns, int n_cp_max)
    : knot_dt_ns_(knot_dt_ns), dt_v_ns_(knot_dt_ns), n_cp_max_(std::max(1, n_cp_max)) {
  // The basalt deques hold the knots; evaluation runs on a uniform virtual grid of
  // spacing dt_v_ns_, with the real timeline layered on top via the kt_ table.
  so3_ = std::make_unique<basalt::So3Spline<kOrder, double>>(dt_v_ns_, 0);
  r3_ = std::make_unique<basalt::RdSpline<3, kOrder, double>>(dt_v_ns_, 0);
}

SplineWindow::~SplineWindow() = default;
SplineWindow::SplineWindow(SplineWindow&&) noexcept = default;
SplineWindow& SplineWindow::operator=(SplineWindow&&) noexcept = default;

std::unique_ptr<SplineWindow> SplineWindow::clone() const {
  auto copy = std::make_unique<SplineWindow>(knot_dt_ns_, n_cp_max_);
  copy->initialized_ = initialized_;
  copy->dt_v_ns_ = dt_v_ns_;
  copy->t_min_ = t_min_;
  copy->t_max_ = t_max_;
  copy->kt_ = kt_;
  copy->segments_ = segments_;
  // Rebuild the basalt deques by value: each knot's SO(3) rotation and R^3 vector are
  // pushed in order, so the copy holds identical knot values in independent storage.
  // Index i in the copy's deques matches index i in kt_, exactly as in the original.
  copy->so3_ = std::make_unique<basalt::So3Spline<kOrder, double>>(dt_v_ns_, 0);
  copy->r3_ = std::make_unique<basalt::RdSpline<3, kOrder, double>>(dt_v_ns_, 0);
  const int n = static_cast<int>(kt_.size());
  for (int i = 0; i < n; ++i) {
    copy->so3_->knotsPushBack(so3_->getKnot(i));
    copy->r3_->knotsPushBack(r3_->getKnot(i));
  }
  return copy;
}

void SplineWindow::initialize(Timestamp t0, const Pose& T0) {
  so3_ = std::make_unique<basalt::So3Spline<kOrder, double>>(dt_v_ns_, 0);
  r3_ = std::make_unique<basalt::RdSpline<3, kOrder, double>>(dt_v_ns_, 0);
  segments_.clear();
  kt_.clear();

  // Seed the first N knots at the start pose, spaced one nominal cadence apart in
  // real time. With all N knots equal, the spline evaluates to exactly T0 over the
  // first segment, so the window starts where IMU integration predicts.
  const Sophus::SO3d r0 = toSophus(T0.q);
  for (int i = 0; i < kOrder; ++i) {
    so3_->knotsPushBack(r0);
    r3_->knotsPushBack(T0.t);
    kt_.push_back(t0 + static_cast<int64_t>(i) * knot_dt_ns_);
  }

  t_min_ = t0;
  // Real coverage runs up to (but excluding) the real time of the first knot whose
  // N-knot support is not yet complete: kt_[numKnots - N + 1].
  t_max_ = kt_[kt_.size() - kOrder + 1] - 1;
  initialized_ = true;
}

void SplineWindow::extendTo(Timestamp t, const std::function<Pose(Timestamp)>& seed, int n_cp) {
  const int cp = std::clamp(n_cp, 1, n_cp_max_);

  // Append whole outer segments (each of knot_dt_ns_ real duration, carrying cp
  // uniformly spaced knots) until the real horizon reaches t. The trailing N-1 knots
  // of any segment only warm-start future evaluation, so coverage trails the newest
  // knot by N-1 knot intervals.
  while (t_max_ < t) {
    const Timestamp seg_start = kt_.back();
    const Timestamp seg_end = seg_start + knot_dt_ns_;
    const Timestamp step = knot_dt_ns_ / cp;

    segments_.push_back(Segment{seg_start, static_cast<int64_t>(kt_.size()) * dt_v_ns_, cp});

    for (int j = 1; j <= cp; ++j) {
      // The last knot lands exactly on seg_end; integer stepping keeps placement
      // uniform even when knot_dt_ns_ is not divisible by cp.
      const Timestamp kt_real = (j == cp) ? seg_end : seg_start + j * step;
      // A control point is not on the curve: at the start of interval i the spline
      // evaluates to ~(CP_i + 4 CP_{i+1} + CP_{i+2})/6, so CP_j dominates the curve
      // one knot interval before its grid time. Seeding CP_j = seed(kt_j) would
      // reproduce the seed trajectory time-shifted late by one interval; sampling
      // one local interval back makes the warm start land on the seed trajectory.
      const Pose seed_pose = seed(kt_real - step);
      so3_->knotsPushBack(toSophus(seed_pose.q));
      r3_->knotsPushBack(seed_pose.t);
      kt_.push_back(kt_real);
    }

    t_max_ = kt_[kt_.size() - kOrder + 1] - 1;
  }
}

void SplineWindow::reseedFrom(int from_idx, const std::function<Pose(Timestamp)>& seed) {
  const int n = static_cast<int>(kt_.size());
  for (int j = std::max(from_idx, 1); j < n; ++j) {
    // Same placement rule as extendTo: a control point dominates the curve one local
    // knot step before its grid time, so sample the seed there to land on it.
    const Timestamp step = kt_[static_cast<std::size_t>(j)] - kt_[static_cast<std::size_t>(j - 1)];
    const Pose p = seed(kt_[static_cast<std::size_t>(j)] - step);
    so3_->getKnot(j) = toSophus(p.q);
    r3_->getKnot(j) = p.t;
  }
}

int SplineWindow::lowestKnotIndexOf(const std::vector<const double*>& ptrs) const {
  // Identify the lowest deque index whose SO(3) or R^3 knot storage backs any of the
  // given parameter-block pointers. Non-knot blocks (bias, gravity, exposure) match
  // nothing and are ignored. Callers use this to bound front-trimming so externally
  // held knot pointers (marginalization-prior blocks) are never freed.
  if (ptrs.empty()) {
    return std::numeric_limits<int>::max();
  }
  const std::unordered_set<const double*> wanted(ptrs.begin(), ptrs.end());
  const int n = static_cast<int>(kt_.size());
  for (int i = 0; i < n; ++i) {
    if (wanted.count(so3_->getKnot(i).data()) || wanted.count(r3_->getKnot(i).data())) {
      return i;
    }
  }
  return std::numeric_limits<int>::max();
}

void SplineWindow::dropOldest(int n) {
  // Keep the full cubic support resident; evaluation needs four knots at all times.
  const int max_drop = static_cast<int>(kt_.size()) - kOrder;
  const int drop = std::clamp(n, 0, std::max(0, max_drop));
  if (drop <= 0) {
    return;
  }
  for (int i = 0; i < drop; ++i) {
    so3_->knotsPopFront();
    r3_->knotsPopFront();
  }
  kt_.erase(kt_.begin(), kt_.begin() + drop);

  // The new oldest knot defines the lower coverage bound; the virtual grid is rebased
  // so knot index 0 still anchors to virtual time 0 (the basalt pop_front already
  // advanced each spline's start time, but evaluation here uses kt_ index, not basalt
  // time, so only kt_/t_min_ need to stay coherent).
  t_min_ = kt_.front();

  // segments_ is advisory only (never consulted by evaluation); drop the leading
  // records whose outer span lies entirely behind the new front so it stays bounded.
  std::size_t seg_drop = 0;
  while (seg_drop + 1 < segments_.size() && segments_[seg_drop + 1].t_real_start <= t_min_) {
    ++seg_drop;
  }
  if (seg_drop > 0) {
    segments_.erase(segments_.begin(), segments_.begin() + static_cast<std::ptrdiff_t>(seg_drop));
  }
}

Timestamp SplineWindow::minTime() const {
  return t_min_;
}

Timestamp SplineWindow::maxTime() const {
  return t_max_;
}

bool SplineWindow::covers(Timestamp t) const {
  return initialized_ && t >= t_min_ && t <= t_max_;
}

int SplineWindow::locate(Timestamp t, double* u, double* slope) const {
  // Find the knot interval [kt_[i], kt_[i+1]) containing t and return i, with u the
  // normalized position in [0,1) and slope = dv/dt constant within the interval. The
  // map interpolates anchors (kt_[i], i*dt_v_ns_), so virtual time is exact in double
  // precision -- no integer-nanosecond quantization in the evaluation path.
  const Timestamp clamped = std::clamp(t, t_min_, t_max_);
  const int last = static_cast<int>(kt_.size()) - kOrder;

  // The interval index is the largest k in [0, last] with kt_[k] <= clamped. Because
  // kt_ is strictly increasing, upper_bound finds the first knot strictly greater than
  // clamped; the interval owning clamped starts one knot before it. The result is
  // clamped to [0, last] so the right edge maps into the final evaluable interval.
  const auto it = std::upper_bound(kt_.begin(), kt_.end(), clamped);
  int i = static_cast<int>(it - kt_.begin()) - 1;
  i = std::clamp(i, 0, last);

  const double real_span = static_cast<double>(kt_[i + 1] - kt_[i]);
  *u = static_cast<double>(clamped - kt_[i]) / real_span;
  *slope = static_cast<double>(dt_v_ns_) / real_span;
  return i;
}

Pose SplineWindow::pose(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> so3_ptrs{};
  std::array<const double*, kOrder> r3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    so3_ptrs[j] = so3_->getKnot(i + j).data();
    r3_ptrs[j] = r3_->getKnot(i + j).data();
  }

  const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
  Sophus::SO3d r;
  basalt::CeresSplineHelper<kOrder>::template evaluate_lie<double, Sophus::SO3>(so3_ptrs.data(), u,
                                                                                inv_dt_s, &r);
  Eigen::Vector3d p;
  basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 0>(r3_ptrs.data(), u, inv_dt_s,
                                                                     &p);
  return Pose{r.unit_quaternion(), p};
}

Eigen::Vector3d SplineWindow::angularVelocityBody(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> so3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    so3_ptrs[j] = so3_->getKnot(i + j).data();
  }

  const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
  Sophus::SO3d r;
  Eigen::Vector3d vel_v;
  basalt::CeresSplineHelper<kOrder>::template evaluate_lie<double, Sophus::SO3>(
      so3_ptrs.data(), u, inv_dt_s, &r, &vel_v);
  // vel_v is d/dv in the body frame (R(v+dv) = R(v) exp(vel_v dv)); the real-time
  // body rate is vel_v scaled by dv/dt.
  return vel_v * slope;
}

Eigen::Vector3d SplineWindow::linearVelocityWorld(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> r3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    r3_ptrs[j] = r3_->getKnot(i + j).data();
  }

  const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
  Eigen::Vector3d vel_v;
  basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 1>(r3_ptrs.data(), u, inv_dt_s,
                                                                     &vel_v);
  // dp/dt = (dp/dv)(dv/dt).
  return vel_v * slope;
}

Eigen::Vector3d SplineWindow::linearAccelWorld(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> r3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    r3_ptrs[j] = r3_->getKnot(i + j).data();
  }

  const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
  Eigen::Vector3d accel_v;
  basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 2>(r3_ptrs.data(), u, inv_dt_s,
                                                                     &accel_v);
  // d2p/dt2 = (d2p/dv2)(dv/dt)^2; slope is constant within the interval so the
  // derivative-of-slope term vanishes.
  return accel_v * (slope * slope);
}

SplineWindow::SegmentRef SplineWindow::segmentFor(Timestamp t) {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  SegmentRef ref{};
  for (int j = 0; j < kOrder; ++j) {
    // getKnot returns a reference into the basalt aligned_deque; deque element
    // addresses are stable across push_back/pop_front, so these pointers stay valid
    // as Ceres parameter blocks while the window grows.
    ref.so3_knots[j] = so3_->getKnot(i + j).data();
    ref.r3_knots[j] = r3_->getKnot(i + j).data();
  }
  ref.u = u;
  // Real per-knot spacing of this segment in seconds: a Ceres functor that feeds the
  // basalt helper inv_dt = 1/dt_s gets derivatives already in real time, with no
  // separate virtual-time slope correction. For a single-control-point segment this
  // equals the nominal cadence; denser segments report a proportionally smaller dt_s.
  const double real_span_ns = static_cast<double>(dt_v_ns_) / slope;
  ref.dt_s = real_span_ns * 1e-9;
  return ref;
}

int SplineWindow::leadingKnotIndex(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  return locate(t, &u, &slope);
}

double* SplineWindow::so3KnotData(int i) {
  return so3_->getKnot(i).data();
}

double* SplineWindow::r3KnotData(int i) {
  return r3_->getKnot(i).data();
}

int SplineWindow::numKnots() const {
  return static_cast<int>(kt_.size());
}

Timestamp SplineWindow::knotTime(int i) const {
  return kt_[static_cast<std::size_t>(i)];
}

}  // namespace meridian
