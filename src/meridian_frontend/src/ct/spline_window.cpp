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

#include "ct/spline_analytic.hpp"

namespace meridian {

namespace {

constexpr int kOrder = 4;  // cubic B-spline: order 4, degree 3

Sophus::SO3d toSophus(const Eigen::Quaterniond& q) {
  return Sophus::SO3d(q.normalized());
}

// Multiplies a u-polynomial p (coefficients ordered constant..cubic) by the affine
// factor (a + b*u) / d. Polynomials entering each Cox-de Boor level have degree at
// most 2, so the cubic coefficient never overflows the four-coefficient form.
Eigen::Vector4d mulAffine(const Eigen::Vector4d& p, double a, double b, double d) {
  Eigen::Vector4d out;
  out[0] = a * p[0];
  out[1] = a * p[1] + b * p[0];
  out[2] = a * p[2] + b * p[1];
  out[3] = a * p[3] + b * p[2];
  return out / d;
}

// Blending matrix of one cubic B-spline knot interval over an arbitrary
// (non-uniform) knot vector. tau holds the times of the six consecutive knots the
// interval's four live bases depend on; tau[2] and tau[3] bound the interval itself.
// The Cox-de Boor recurrence
//
//   N_{j,k}(t) = (t - tau_j) / (tau_{j+k} - tau_j)             * N_{j,k-1}(t)
//              + (tau_{j+k+1} - t) / (tau_{j+k+1} - tau_{j+1}) * N_{j+1,k-1}(t)
//
// is run symbolically in the local parameter u, with t = tau[2] + u * (tau[3] -
// tau[2]): restricted to the interval, every basis is a polynomial in u, and each
// recurrence level multiplies by factors affine in u. On the interval the only
// nonzero degree-0 basis is the interval's own indicator, so recurrence terms that
// multiply bases vanishing there are dropped -- that is what confines the
// dependence to the six supplied knots. Knot differences are formed in integer
// nanoseconds and cast to double (exact), and only ratios of differences enter, so
// the result is invariant to the time unit. Row r of the result holds the
// u-coefficients of the basis paired with the r-th control point of the interval's
// support; basis weights are m * (1, u, u^2, u^3)^T, and because the live bases sum
// to one for every u, the column sums are (1, 0, 0, 0).
Eigen::Matrix4d nonUniformBlendingMatrix(const std::array<int64_t, 6>& tau) {
  const double h = static_cast<double>(tau[3] - tau[2]);
  // B[r] holds the u-polynomial of the r-th live basis at the current level k; at
  // level 0 only the interval indicator (constant 1) is alive.
  std::array<Eigen::Vector4d, 4> B{};
  for (auto& b : B) {
    b.setZero();
  }
  B[0] = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  for (int k = 1; k <= 3; ++k) {
    std::array<Eigen::Vector4d, 4> next{};
    for (auto& b : next) {
      b.setZero();
    }
    for (int r = 0; r <= k; ++r) {
      // The r-th live basis at level k starts at the knot with local index l. Its
      // two recurrence terms reuse the previous level's live bases r-1 and r; the
      // first is absent for r == 0 and the second for r == k (those bases vanish on
      // the interval), which keeps every tau index used here within [0, 5].
      const int l = 2 - k + r;
      Eigen::Vector4d acc = Eigen::Vector4d::Zero();
      if (r >= 1) {
        // (t - tau_l) / (tau_{l+k} - tau_l), affine in u with slope h.
        const double a = static_cast<double>(tau[2] - tau[static_cast<std::size_t>(l)]);
        const double d = static_cast<double>(tau[static_cast<std::size_t>(l + k)] -
                                             tau[static_cast<std::size_t>(l)]);
        acc += mulAffine(B[static_cast<std::size_t>(r - 1)], a, h, d);
      }
      if (r <= k - 1) {
        // (tau_{l+k+1} - t) / (tau_{l+k+1} - tau_{l+1}), affine in u with slope -h.
        const double a = static_cast<double>(tau[static_cast<std::size_t>(l + k + 1)] - tau[2]);
        const double d = static_cast<double>(tau[static_cast<std::size_t>(l + k + 1)] -
                                             tau[static_cast<std::size_t>(l + 1)]);
        acc += mulAffine(B[static_cast<std::size_t>(r)], a, -h, d);
      }
      next[static_cast<std::size_t>(r)] = acc;
    }
    B = next;
  }
  Eigen::Matrix4d m;
  for (int r = 0; r < 4; ++r) {
    m.row(r) = B[static_cast<std::size_t>(r)].transpose();
  }
  return m;
}

// Cumulative form: row r becomes the sum of rows r..3. Row 0 then sums the full
// partition of unity, i.e. (1, 0, 0, 0), and each later row weights the relative
// motion between consecutive control points.
Eigen::Matrix4d cumulativeForm(const Eigen::Matrix4d& m) {
  Eigen::Matrix4d cm = m;
  for (int r = 0; r < 4; ++r) {
    for (int q = r + 1; q < 4; ++q) {
      cm.row(r) += m.row(q);
    }
  }
  return cm;
}

}  // namespace

SplineWindow::SplineWindow(Duration knot_dt_ns, int n_cp_max, bool non_uniform)
    : knot_dt_ns_(knot_dt_ns),
      dt_v_ns_(knot_dt_ns),
      n_cp_max_(std::max(1, n_cp_max)),
      non_uniform_(non_uniform) {
  // The basalt deques hold the knots; evaluation runs on a uniform virtual grid of
  // spacing dt_v_ns_, with the real timeline layered on top via the kt_ table.
  so3_ = std::make_unique<basalt::So3Spline<kOrder, double>>(dt_v_ns_, 0);
  r3_ = std::make_unique<basalt::RdSpline<3, kOrder, double>>(dt_v_ns_, 0);
}

SplineWindow::SplineWindow(Duration knot_dt_ns, int n_cp_max)
    : SplineWindow(knot_dt_ns, n_cp_max, false) {}

SplineWindow::~SplineWindow() = default;
SplineWindow::SplineWindow(SplineWindow&&) noexcept = default;
SplineWindow& SplineWindow::operator=(SplineWindow&&) noexcept = default;

std::unique_ptr<SplineWindow> SplineWindow::clone() const {
  auto copy = std::make_unique<SplineWindow>(knot_dt_ns_, n_cp_max_, non_uniform_);
  copy->initialized_ = initialized_;
  copy->dt_v_ns_ = dt_v_ns_;
  copy->t_min_ = t_min_;
  copy->t_max_ = t_max_;
  copy->kt_ = kt_;
  copy->segments_ = segments_;
  // Copying the cached matrices (plain double copies, bitwise-exact) instead of
  // recomputing them guarantees the clone evaluates bit-identically even for front
  // intervals whose six-knot neighborhoods include knots dropped long ago.
  copy->mats_ = mats_;
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
  mats_.clear();

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
  syncIntervalBases();
}

void SplineWindow::extendTo(Timestamp t, const std::function<Pose(Timestamp)>& seed, int n_cp) {
  extendTo(t, seed, [n_cp](Timestamp, Timestamp) { return n_cp; });
}

void SplineWindow::extendTo(
    Timestamp t, const std::function<Pose(Timestamp)>& seed,
    const std::function<int(Timestamp seg_start, Timestamp seg_end)>& density) {
  // Append whole outer segments (each of knot_dt_ns_ real duration, carrying cp
  // uniformly spaced knots) until the real horizon reaches t. The trailing N-1 knots
  // of any segment only warm-start future evaluation, so coverage trails the newest
  // knot by N-1 knot intervals.
  while (t_max_ < t) {
    const Timestamp seg_start = kt_.back();
    const Timestamp seg_end = seg_start + knot_dt_ns_;
    const int cp = std::clamp(density(seg_start, seg_end), 1, n_cp_max_);
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
  syncIntervalBases();
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

int SplineWindow::highestKnotIndexOf(const std::vector<const double*>& ptrs) const {
  // Mirror of lowestKnotIndexOf, scanning from the back: the highest deque index
  // whose SO(3) or R^3 knot storage backs any of the given pointers, -1 when none
  // does. Callers use it to bound tail truncation so externally held knot pointers
  // are never popped.
  if (ptrs.empty()) {
    return -1;
  }
  const std::unordered_set<const double*> wanted(ptrs.begin(), ptrs.end());
  for (int i = static_cast<int>(kt_.size()) - 1; i >= 0; --i) {
    if (wanted.count(so3_->getKnot(i).data()) || wanted.count(r3_->getKnot(i).data())) {
      return i;
    }
  }
  return -1;
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

  // mats_ index i pairs with the interval [kt_[i], kt_[i+1]); dropping front knots
  // shifts interval indexing down by the same count, so the cache pops in lockstep.
  // Surviving entries stay as stored: each is final for its interval even though its
  // six-knot neighborhood may now reach before the new front knot.
  if (non_uniform_) {
    for (int i = 0; i < drop; ++i) {
      mats_.pop_front();
    }
  }

  // The new oldest knot defines the lower coverage bound; the virtual grid is rebased
  // so knot index 0 still anchors to virtual time 0 (the basalt pop_front already
  // advanced each spline's start time, but evaluation here uses kt_ index, not basalt
  // time, so only kt_/t_min_ need to stay coherent).
  t_min_ = kt_.front();

  // segments_ never feeds evaluation, but it answers the outer-boundary queries
  // (segment*Containing); drop only the leading records whose outer span lies
  // entirely behind the new front, so every still-covered time keeps its record.
  std::size_t seg_drop = 0;
  while (seg_drop + 1 < segments_.size() && segments_[seg_drop + 1].t_real_start <= t_min_) {
    ++seg_drop;
  }
  if (seg_drop > 0) {
    segments_.erase(segments_.begin(), segments_.begin() + static_cast<std::ptrdiff_t>(seg_drop));
  }
}

int SplineWindow::truncateTailSegments(Timestamp from_seg_start) {
  int removed = 0;
  while (!segments_.empty() && segments_.back().t_real_start >= from_seg_start) {
    const int cp = segments_.back().n_cp;
    // Never go below the four-knot cubic support: a segment whose removal would
    // leave fewer knots is kept whole, and every earlier segment stays behind it.
    if (static_cast<int>(kt_.size()) - cp < kOrder) {
      break;
    }
    for (int j = 0; j < cp; ++j) {
      so3_->knotsPopBack();
      r3_->knotsPopBack();
      kt_.pop_back();
    }
    segments_.pop_back();
    removed += cp;
  }
  if (removed > 0) {
    // Coverage retreats with the surviving knots, same trailing rule as extendTo;
    // pop_back invalidates only the erased elements, so pointers handed out for
    // surviving knots remain valid.
    t_max_ = kt_[kt_.size() - kOrder + 1] - 1;
    syncIntervalBases();
  }
  return removed;
}

Timestamp SplineWindow::segmentStartContaining(Timestamp t) const {
  if (const Segment* s = segmentRecordContaining(t)) {
    return s->t_real_start;
  }
  double u = 0.0;
  double slope = 1.0;
  return kt_[static_cast<std::size_t>(locate(t, &u, &slope))];
}

Timestamp SplineWindow::segmentEndContaining(Timestamp t) const {
  if (const Segment* s = segmentRecordContaining(t)) {
    return s->t_real_start + knot_dt_ns_;
  }
  double u = 0.0;
  double slope = 1.0;
  return kt_[static_cast<std::size_t>(locate(t, &u, &slope))] + knot_dt_ns_;
}

int SplineWindow::segmentGearContaining(Timestamp t) const {
  if (const Segment* s = segmentRecordContaining(t)) {
    return s->n_cp;
  }
  // No record covers t: bootstrap knots span exactly one nominal cadence each, i.e.
  // a single control point per outer span.
  return 1;
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

const SplineWindow::Segment* SplineWindow::segmentRecordContaining(Timestamp t) const {
  // Records are appended in strictly increasing t_real_start order, so the only
  // candidate is the last record starting at or before t; it covers t iff t falls
  // inside its outer span of one nominal cadence.
  const auto it =
      std::upper_bound(segments_.begin(), segments_.end(), t,
                       [](Timestamp value, const Segment& s) { return value < s.t_real_start; });
  if (it == segments_.begin()) {
    return nullptr;
  }
  const Segment& s = *(it - 1);
  if (t >= s.t_real_start + knot_dt_ns_) {
    return nullptr;
  }
  return &s;
}

int64_t SplineWindow::knotTimeExtrapolated(int m) const {
  if (m >= 0) {
    return kt_[static_cast<std::size_t>(m)];
  }
  // Mirror the front spacing backwards. Negative indices only arise for the first
  // two knot intervals, whose six-knot basis neighborhood reaches before the first
  // stored knot.
  const int64_t front_step = kt_[1] - kt_[0];
  return kt_[0] + static_cast<int64_t>(m) * front_step;
}

SplineWindow::IntervalBasis SplineWindow::makeIntervalBasis(int i) const {
  std::array<int64_t, 6> tau{};
  for (int k = 0; k < 6; ++k) {
    tau[static_cast<std::size_t>(k)] = knotTimeExtrapolated(i - 2 + k);
  }
  IntervalBasis b;
  // The cardinal blending matrices are exact for equally spaced knots, so when this
  // interval's entire six-knot support sits at the base virtual cadence the runtime
  // matrix would only reproduce them to within a rounding error (~1 ULP, from summing
  // the rows at runtime instead of using the hard-coded constant). Flag the interval
  // uniform and leave the matrices unset; evaluation then takes the cardinal kernels,
  // the identical floating-point path as the disabled build, so a uniform stretch is
  // byte-identical to it. Gear-1 knots land at exact integer-nanosecond multiples of
  // the cadence, so equal integer deltas is the exact predicate. Only an interval whose
  // spacing actually varies keeps a per-interval matrix.
  b.uniform = true;
  for (int k = 0; k < 5; ++k) {
    if (tau[static_cast<std::size_t>(k + 1)] - tau[static_cast<std::size_t>(k)] != dt_v_ns_) {
      b.uniform = false;
      break;
    }
  }
  if (b.uniform) {
    b.m.setZero();
    b.cm.setZero();
    return b;
  }
  b.m = nonUniformBlendingMatrix(tau);
  b.cm = cumulativeForm(b.m);
  return b;
}

void SplineWindow::syncIntervalBases() {
  if (!non_uniform_) {
    return;
  }
  const int want = std::max(0, static_cast<int>(kt_.size()) - (kOrder - 1));
  // Shrinking (tail truncation) then re-growing reproduces the original bits:
  // makeIntervalBasis is a pure function of the knot times, and re-extension that
  // recreates identical knot times therefore recreates identical matrices.
  while (static_cast<int>(mats_.size()) > want) {
    mats_.pop_back();
  }
  while (static_cast<int>(mats_.size()) < want) {
    mats_.push_back(makeIntervalBasis(static_cast<int>(mats_.size())));
  }
}

int SplineWindow::intervalBasisCount() const {
  return static_cast<int>(mats_.size());
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

  // A disabled-mode window, and a uniformly-spaced interval of a non-uniform one, run
  // the cardinal basalt kernels on the virtual grid: the cardinal matrix is exact for
  // equal spacing, and these are the exact same floating-point operations as the warp
  // build, so the result is byte-identical to it (slope == 1 on a uniform interval).
  if (!non_uniform_ || mats_[static_cast<std::size_t>(i)].uniform) {
    const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
    Sophus::SO3d r;
    basalt::CeresSplineHelper<kOrder>::template evaluate_lie<double, Sophus::SO3>(so3_ptrs.data(),
                                                                                  u, inv_dt_s, &r);
    Eigen::Vector3d p;
    basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 0>(r3_ptrs.data(), u, inv_dt_s,
                                                                       &p);
    return Pose{r.unit_quaternion(), p};
  }

  // Per-interval matrices already encode the real knot spacing; inv_dt carries this
  // interval's real span, so no virtual-warp slope enters anywhere.
  const IntervalBasis& b = mats_[static_cast<std::size_t>(i)];
  const double inv_dt_s = 1e9 / static_cast<double>(kt_[i + 1] - kt_[i]);
  Sophus::SO3d r;
  ct::evaluateLieWithMatrix<double, Sophus::SO3>(so3_ptrs.data(), u, inv_dt_s, b.cm, &r);
  Eigen::Vector3d p;
  ct::evaluateRdWithMatrix<double, 3, 0>(r3_ptrs.data(), u, inv_dt_s, b.m, &p);
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

  // Cardinal kernels for a disabled-mode window or a uniform interval; the warp slope
  // is 1 on a uniform interval, so this matches the per-interval result and stays
  // byte-identical to the disabled build.
  if (!non_uniform_ || mats_[static_cast<std::size_t>(i)].uniform) {
    const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
    Sophus::SO3d r;
    Eigen::Vector3d vel_v;
    basalt::CeresSplineHelper<kOrder>::template evaluate_lie<double, Sophus::SO3>(
        so3_ptrs.data(), u, inv_dt_s, &r, &vel_v);
    // vel_v is d/dv in the body frame (R(v+dv) = R(v) exp(vel_v dv)); the real-time
    // body rate is vel_v scaled by dv/dt.
    return vel_v * slope;
  }

  // inv_dt carries this interval's real span, so the helper returns the real-time
  // body rate directly; no slope chain rule applies.
  const IntervalBasis& b = mats_[static_cast<std::size_t>(i)];
  const double inv_dt_s = 1e9 / static_cast<double>(kt_[i + 1] - kt_[i]);
  Sophus::SO3d r;
  Eigen::Vector3d vel;
  ct::evaluateLieWithMatrix<double, Sophus::SO3>(so3_ptrs.data(), u, inv_dt_s, b.cm, &r, &vel);
  return vel;
}

Eigen::Vector3d SplineWindow::linearVelocityWorld(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> r3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    r3_ptrs[j] = r3_->getKnot(i + j).data();
  }

  // Cardinal kernels for a disabled-mode window or a uniform interval; the warp slope
  // is 1 on a uniform interval, so this stays byte-identical to the disabled build.
  if (!non_uniform_ || mats_[static_cast<std::size_t>(i)].uniform) {
    const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
    Eigen::Vector3d vel_v;
    basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 1>(r3_ptrs.data(), u, inv_dt_s,
                                                                       &vel_v);
    // dp/dt = (dp/dv)(dv/dt).
    return vel_v * slope;
  }

  const IntervalBasis& b = mats_[static_cast<std::size_t>(i)];
  const double inv_dt_s = 1e9 / static_cast<double>(kt_[i + 1] - kt_[i]);
  Eigen::Vector3d vel;
  ct::evaluateRdWithMatrix<double, 3, 1>(r3_ptrs.data(), u, inv_dt_s, b.m, &vel);
  return vel;
}

Eigen::Vector3d SplineWindow::linearAccelWorld(Timestamp t) const {
  double u = 0.0;
  double slope = 1.0;
  const int i = locate(t, &u, &slope);

  std::array<const double*, kOrder> r3_ptrs{};
  for (int j = 0; j < kOrder; ++j) {
    r3_ptrs[j] = r3_->getKnot(i + j).data();
  }

  // Cardinal kernels for a disabled-mode window or a uniform interval; the warp slope
  // is 1 on a uniform interval, so this stays byte-identical to the disabled build.
  if (!non_uniform_ || mats_[static_cast<std::size_t>(i)].uniform) {
    const double inv_dt_s = 1e9 / static_cast<double>(dt_v_ns_);
    Eigen::Vector3d accel_v;
    basalt::CeresSplineHelper<kOrder>::template evaluate<double, 3, 2>(r3_ptrs.data(), u, inv_dt_s,
                                                                       &accel_v);
    // d2p/dt2 = (d2p/dv2)(dv/dt)^2; slope is constant within the interval so the
    // derivative-of-slope term vanishes.
    return accel_v * (slope * slope);
  }

  const IntervalBasis& b = mats_[static_cast<std::size_t>(i)];
  const double inv_dt_s = 1e9 / static_cast<double>(kt_[i + 1] - kt_[i]);
  Eigen::Vector3d accel;
  ct::evaluateRdWithMatrix<double, 3, 2>(r3_ptrs.data(), u, inv_dt_s, b.m, &accel);
  return accel;
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
  // A disabled-mode window, and a uniformly-spaced interval of a non-uniform one, leave
  // the matrix pointers null: residuals then take the cardinal blending matrices, the
  // exact same floating-point path as the warp build, so a uniform interval contributes
  // byte-identical factors. Only an interval whose spacing genuinely varies carries a
  // per-interval matrix.
  if (!non_uniform_ || mats_[static_cast<std::size_t>(i)].uniform) {
    // Real per-knot spacing of this segment in seconds: a Ceres functor that feeds the
    // basalt helper inv_dt = 1/dt_s gets derivatives already in real time, with no
    // separate virtual-time slope correction. For a single-control-point segment this
    // equals the nominal cadence; denser segments report a proportionally smaller dt_s.
    // On a uniform interval slope == 1, so this equals the real span read off the table.
    const double real_span_ns = static_cast<double>(dt_v_ns_) / slope;
    ref.dt_s = real_span_ns * 1e-9;
    return ref;
  }
  // Real span of this knot interval read off the knot-time table directly. The
  // matrix pointers address window-owned cache entries whose lifetime matches the
  // knot pointers': stable across push_back/pop_front, gone only when the interval
  // itself is removed.
  ref.dt_s = static_cast<double>(kt_[i + 1] - kt_[i]) * 1e-9;
  const IntervalBasis& b = mats_[static_cast<std::size_t>(i)];
  ref.blend = &b.m;
  ref.cum_blend = &b.cm;
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
