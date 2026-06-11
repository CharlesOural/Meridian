#include <basalt/spline/ceres_spline_helper.h>
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sophus/so3.hpp>
#include <vector>

#include "ct/spline_analytic.hpp"
#include "ct/spline_window.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

using meridian::Duration;
using meridian::Pose;
using meridian::SplineWindow;
using meridian::Timestamp;

namespace {

constexpr Duration kKnotDt = 50'000'000;  // 50 ms nominal outer cadence

// A smooth, randomized analytic trajectory used to seed the spline knots. Translation
// and rotation are sums of low-frequency sinusoids, so they are infinitely
// differentiable with bounded derivatives -- the spline fitted through samples is
// itself smooth and central finite differences of pose(t) are accurate.
struct SmoothTrajectory {
  std::array<double, 3> ax, fx, px;  // translation: amplitude, freq, phase per axis
  std::array<double, 3> ar, fr, pr;  // rotation tangent: amplitude, freq, phase

  explicit SmoothTrajectory(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> amp(0.2, 0.8);
    std::uniform_real_distribution<double> frq(0.4, 1.4);  // rad/s, low frequency
    std::uniform_real_distribution<double> phs(0.0, 6.28);
    for (int k = 0; k < 3; ++k) {
      ax[k] = amp(rng);
      fx[k] = frq(rng);
      px[k] = phs(rng);
      ar[k] = 0.5 * amp(rng);
      fr[k] = frq(rng);
      pr[k] = phs(rng);
    }
  }

  Pose at(Timestamp t) const {
    const double s = meridian::to_seconds(t);
    Eigen::Vector3d p, phi;
    for (int k = 0; k < 3; ++k) {
      p[k] = ax[k] * std::sin(fx[k] * s + px[k]);
      phi[k] = ar[k] * std::sin(fr[k] * s + pr[k]);
    }
    return Pose{Sophus::SO3d::exp(phi).unit_quaternion(), p};
  }
};

// Builds a window seeded from a smooth trajectory over [t0, t0 + span] with a fixed
// per-segment control-point count.
SplineWindow makeWindow(const SmoothTrajectory& traj, Timestamp t0, Timestamp span, int n_cp) {
  SplineWindow w(kKnotDt, 4);
  w.initialize(t0, traj.at(t0));
  w.extendTo(
      t0 + span, [&](Timestamp t) { return traj.at(t); }, n_cp);
  return w;
}

// Same, but in non-uniform mode and driven by a per-segment density function.
SplineWindow makeNonUniformWindow(const SmoothTrajectory& traj, Timestamp t0, Timestamp span,
                                  const std::function<int(Timestamp, Timestamp)>& density) {
  SplineWindow w(kKnotDt, 4, /*non_uniform=*/true);
  w.initialize(t0, traj.at(t0));
  w.extendTo(
      t0 + span, [&](Timestamp t) { return traj.at(t); }, density);
  return w;
}

// A density function cycling through a fixed gear pattern, one entry per appended
// outer segment.
std::function<int(Timestamp, Timestamp)> cyclingDensity(std::vector<int> pattern) {
  auto idx = std::make_shared<std::size_t>(0);
  return [pattern = std::move(pattern), idx](Timestamp, Timestamp) {
    return pattern[(*idx)++ % pattern.size()];
  };
}

// The first real time at which a window's support is shaped purely by measurement
// knots (the N-1 seed knots placed at and after t0 have cleared the support).
Timestamp measurementStart(Timestamp t0) {
  return t0 + 5 * kKnotDt;
}

void expectBitEqual(double actual, double expected, const char* what) {
  std::uint64_t a = 0;
  std::uint64_t e = 0;
  std::memcpy(&a, &actual, sizeof(a));
  std::memcpy(&e, &expected, sizeof(e));
  EXPECT_EQ(a, e) << what << ": " << actual << " vs golden " << expected;
}

void expectPoseBitEqual(const Pose& a, const Pose& b, const char* what) {
  expectBitEqual(a.q.x(), b.q.x(), what);
  expectBitEqual(a.q.y(), b.q.y(), what);
  expectBitEqual(a.q.z(), b.q.z(), what);
  expectBitEqual(a.q.w(), b.q.w(), what);
  expectBitEqual(a.t.x(), b.t.x(), what);
  expectBitEqual(a.t.y(), b.t.y(), what);
  expectBitEqual(a.t.z(), b.t.z(), what);
}

// Real knot times of a window plus linear extrapolation of the boundary spacings
// outside the resident range, exposed through the B-spline knot-vector indexing
// tau_m = time of knot (m - 3). The cubic basis paired with control point c is
// N_{c,4}, supported on [tau_c, tau_{c+4}); the span of knot interval i is the
// indicator span of index i+3.
struct KnotVector {
  std::vector<std::int64_t> kt;

  explicit KnotVector(const SplineWindow& w) {
    for (int i = 0; i < w.numKnots(); ++i) {
      kt.push_back(w.knotTime(i));
    }
  }

  double tau(int m) const {
    const int j = m - 3;
    const int n = static_cast<int>(kt.size());
    if (j < 0) {
      return static_cast<double>(kt[0] + static_cast<std::int64_t>(j) * (kt[1] - kt[0]));
    }
    if (j >= n) {
      return static_cast<double>(
          kt[static_cast<std::size_t>(n - 1)] +
          static_cast<std::int64_t>(j - (n - 1)) *
              (kt[static_cast<std::size_t>(n - 1)] - kt[static_cast<std::size_t>(n - 2)]));
    }
    return static_cast<double>(kt[static_cast<std::size_t>(j)]);
  }

  // Order-k Cox-de Boor basis N_{m,k}(t); k == 1 is the span indicator. For t inside
  // one span every out-of-span level-1 basis is exactly zero, so the extrapolated
  // knot times only ever multiply zeros and cannot affect in-span values.
  double basis(int m, int k, double t) const {
    if (k == 1) {
      return (tau(m) <= t && t < tau(m + 1)) ? 1.0 : 0.0;
    }
    double acc = 0.0;
    const double dl = tau(m + k - 1) - tau(m);
    if (dl > 0.0) {
      acc += (t - tau(m)) / dl * basis(m, k - 1, t);
    }
    const double dr = tau(m + k) - tau(m + 1);
    if (dr > 0.0) {
      acc += (tau(m + k) - t) / dr * basis(m + 1, k - 1, t);
    }
    return acc;
  }
};

}  // namespace

// Analytic derivatives match central finite differences of pose(t) on a smooth,
// randomized spline, to ~1e-6 relative.
TEST(SplineWindow, AnalyticDerivativesMatchFiniteDifferences) {
  const SmoothTrajectory traj(12345);
  const Timestamp t0 = 1'000'000'000;
  SplineWindow w = makeWindow(traj, t0, 900'000'000, 2);

  const Timestamp h = 100'000;  // 100 us central-difference step
  const double hs = meridian::to_seconds(h);

  for (Timestamp t = measurementStart(t0); t < t0 + 800'000'000; t += 31'000'000) {
    ASSERT_TRUE(w.covers(t + h));
    const Pose c = w.pose(t);
    const Pose fwd = w.pose(t + h);
    const Pose bwd = w.pose(t - h);

    const Eigen::Vector3d v_fd = (fwd.t - bwd.t) / (2.0 * hs);
    const Eigen::Vector3d v_an = w.linearVelocityWorld(t);
    EXPECT_LT((v_fd - v_an).norm(), 1e-6 * (1.0 + v_an.norm()));

    const Eigen::Vector3d a_fd = (fwd.t - 2.0 * c.t + bwd.t) / (hs * hs);
    const Eigen::Vector3d a_an = w.linearAccelWorld(t);
    EXPECT_LT((a_fd - a_an).norm(), 1e-5 * (1.0 + a_an.norm()));

    // Body rate via the right perturbation: omega = Log(R(t-h)^{-1} R(t+h)) / 2h.
    const Eigen::Vector3d w_fd =
        (Sophus::SO3d(bwd.q).inverse() * Sophus::SO3d(fwd.q)).log() / (2.0 * hs);
    const Eigen::Vector3d w_an = w.angularVelocityBody(t);
    EXPECT_LT((w_fd - w_an).norm(), 1e-6 * (1.0 + w_an.norm()));
  }
}

// The body angular velocity is directly gyro-comparable: integrating it with the
// right perturbation R(t+dt) = R(t) exp(omega dt) reproduces pose() rotation.
TEST(SplineWindow, BodyAngularVelocityIsGyroComparable) {
  const SmoothTrajectory traj(777);
  const Timestamp t0 = 0;
  SplineWindow w = makeWindow(traj, t0, 600'000'000, 2);

  const Timestamp t = measurementStart(t0) + 40'000'000;
  const Timestamp dt = 100'000;  // 100 us
  const double dts = meridian::to_seconds(dt);

  const Sophus::SO3d R0(w.pose(t).q);
  const Eigen::Vector3d omega = w.angularVelocityBody(t);
  const Sophus::SO3d R_pred = R0 * Sophus::SO3d::exp(omega * dts);
  const Sophus::SO3d R_true(w.pose(t + dt).q);

  EXPECT_LT((R_pred.inverse() * R_true).log().norm(), 1e-6);
}

// Non-uniform control-point densities (n_cp 1..3) still give C2-continuous evaluation
// across virtual-knot joins. With a uniform per-segment density the real->virtual
// slope is constant, so value, velocity, and acceleration are continuous across an
// internal knot boundary; finite differences from each side must agree.
TEST(SplineWindow, C2ContinuityAcrossInternalKnotJoin) {
  const SmoothTrajectory traj(2024);
  for (int n_cp : {1, 2, 3}) {
    const Timestamp t0 = 0;
    SplineWindow w = makeWindow(traj, t0, 700'000'000, n_cp);

    // An internal knot boundary sits at a multiple of the per-knot real step; choose
    // one well inside the measurement region so the stencil sees only smooth knots.
    const Timestamp sub = kKnotDt / n_cp;
    const Timestamp join = measurementStart(t0) + sub;  // a knot boundary
    ASSERT_TRUE(w.covers(join));

    const Timestamp h = 50'000;  // 50 us
    const double hs = meridian::to_seconds(h);
    auto pos = [&](Timestamp t) { return w.pose(t).t; };

    EXPECT_LT((pos(join - 1) - pos(join + 1)).norm(), 1e-6);  // C0

    // One-sided velocity estimates approaching the join from each side.
    const Eigen::Vector3d v_l = (pos(join - h) - pos(join - 3 * h)) / (2.0 * hs);
    const Eigen::Vector3d v_r = (pos(join + 3 * h) - pos(join + h)) / (2.0 * hs);
    EXPECT_LT((v_l - v_r).norm(), 1e-3);  // C1

    // One-sided acceleration estimates approaching the join from each side.
    const Eigen::Vector3d a_l =
        (pos(join - h) - 2.0 * pos(join - 2 * h) + pos(join - 3 * h)) / (hs * hs);
    const Eigen::Vector3d a_r =
        (pos(join + 3 * h) - 2.0 * pos(join + 2 * h) + pos(join + h)) / (hs * hs);
    EXPECT_LT((a_l - a_r).norm(), 1e-1);  // C2 (curvature continuity)
  }
}

// covers()/minTime()/maxTime() boundary semantics, just-inside and just-outside.
TEST(SplineWindow, CoverageBoundarySemantics) {
  const SmoothTrajectory traj(9);
  const Timestamp t0 = 2'000'000'000;
  SplineWindow w = makeWindow(traj, t0, 300'000'000, 2);

  EXPECT_FALSE(w.covers(w.minTime() - 1));
  EXPECT_TRUE(w.covers(w.minTime()));
  EXPECT_TRUE(w.covers(w.maxTime()));
  EXPECT_FALSE(w.covers(w.maxTime() + 1));

  EXPECT_EQ(w.minTime(), t0);
  EXPECT_GE(w.maxTime(), t0 + 300'000'000);

  SplineWindow fresh(kKnotDt, 4);
  EXPECT_FALSE(fresh.covers(t0));
}

// Extending the window must not perturb poses already evaluable before the extend.
TEST(SplineWindow, ExtendLeavesEvaluatedPosesUnchanged) {
  const SmoothTrajectory traj(55);
  const Timestamp t0 = 500'000'000;
  SplineWindow w(kKnotDt, 4);
  w.initialize(t0, traj.at(t0));
  w.extendTo(
      t0 + 250'000'000, [&](Timestamp t) { return traj.at(t); }, 2);

  std::vector<Timestamp> samples;
  for (Timestamp t = t0 + 10'000'000; t < w.maxTime() - 10'000'000; t += 17'000'000) {
    samples.push_back(t);
  }
  std::vector<Pose> before;
  before.reserve(samples.size());
  for (Timestamp t : samples) {
    ASSERT_TRUE(w.covers(t));
    before.push_back(w.pose(t));
  }

  // Extend further with a different control-point density; existing knots are never
  // touched, only new ones are appended, so prior evaluations must be bit-stable.
  w.extendTo(
      t0 + 600'000'000, [&](Timestamp t) { return traj.at(t); }, 3);

  for (std::size_t k = 0; k < samples.size(); ++k) {
    const Pose after = w.pose(samples[k]);
    EXPECT_LT((after.t - before[k].t).norm(), 1e-12);
    EXPECT_LT((Sophus::SO3d(after.q).inverse() * Sophus::SO3d(before[k].q)).log().norm(), 1e-12);
  }
}

// A SegmentRef is self-sufficient for a Ceres residual: feeding its raw knot pointers,
// u, and inv_dt = 1/dt_s to the basalt spline helper reproduces pose() and the
// real-time linear velocity, even on a non-uniform (n_cp > 1) segment where the
// virtual and real knot spacings differ.
TEST(SplineWindow, SegmentRefReconstructsEvaluation) {
  const SmoothTrajectory traj(31337);
  const Timestamp t0 = 0;
  SplineWindow w = makeWindow(traj, t0, 600'000'000, 3);  // dense: 3 cp per segment

  const Timestamp t = measurementStart(t0) + 70'000'000;
  ASSERT_TRUE(w.covers(t));
  SplineWindow::SegmentRef ref = w.segmentFor(t);
  const double inv_dt = 1.0 / ref.dt_s;

  std::array<const double*, 4> so3_ptrs{ref.so3_knots[0], ref.so3_knots[1], ref.so3_knots[2],
                                        ref.so3_knots[3]};
  std::array<const double*, 4> r3_ptrs{ref.r3_knots[0], ref.r3_knots[1], ref.r3_knots[2],
                                       ref.r3_knots[3]};

  Sophus::SO3d r;
  basalt::CeresSplineHelper<4>::template evaluate_lie<double, Sophus::SO3>(so3_ptrs.data(), ref.u,
                                                                           inv_dt, &r);
  Eigen::Vector3d p;
  basalt::CeresSplineHelper<4>::template evaluate<double, 3, 0>(r3_ptrs.data(), ref.u, inv_dt, &p);
  Eigen::Vector3d v;
  basalt::CeresSplineHelper<4>::template evaluate<double, 3, 1>(r3_ptrs.data(), ref.u, inv_dt, &v);

  const Pose expect = w.pose(t);
  EXPECT_LT((p - expect.t).norm(), 1e-9);
  EXPECT_LT((Sophus::SO3d(expect.q).inverse() * r).log().norm(), 1e-9);
  // inv_dt = 1/dt_s yields real-time velocity directly, no slope correction.
  EXPECT_LT((v - w.linearVelocityWorld(t)).norm(), 1e-9);
}

// numKnots grows by exactly n_cp per appended outer segment, and segmentFor hands out
// N stable knot pointers with a normalized u in [0,1).
TEST(SplineWindow, SegmentRefShapeAndStability) {
  const SmoothTrajectory traj(101);
  const Timestamp t0 = 0;
  SplineWindow w(kKnotDt, 4);
  w.initialize(t0, traj.at(t0));
  const int k0 = w.numKnots();
  EXPECT_EQ(k0, 4);

  w.extendTo(
      t0 + kKnotDt, [&](Timestamp t) { return traj.at(t); }, 3);
  EXPECT_EQ(w.numKnots(), k0 + 3);

  const Timestamp t = w.minTime() + (w.maxTime() - w.minTime()) / 2;
  ASSERT_TRUE(w.covers(t));
  SplineWindow::SegmentRef ref = w.segmentFor(t);
  EXPECT_GE(ref.u, 0.0);
  EXPECT_LT(ref.u, 1.0);
  EXPECT_GT(ref.dt_s, 0.0);
  for (int j = 0; j < 4; ++j) {
    EXPECT_NE(ref.so3_knots[j], nullptr);
    EXPECT_NE(ref.r3_knots[j], nullptr);
  }

  // The pointers must survive a subsequent extend (deque storage is address-stable).
  std::array<double*, 4> so3_before = ref.so3_knots;
  w.extendTo(
      t0 + 4 * kKnotDt, [&](Timestamp t) { return traj.at(t); }, 2);
  SplineWindow::SegmentRef ref2 = w.segmentFor(t);
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(ref2.so3_knots[j], so3_before[j]);
  }
}

// ---------------------------------------------------------------------------
// Non-uniform mode
// ---------------------------------------------------------------------------

// On a uniformly spaced knot vector every interval routes to the cardinal kernels: the
// per-interval matrix would only reproduce the hard-coded cardinal blending matrices to
// within a rounding error, so the window detects the uniform spacing and leaves both
// SegmentRef matrix pointers null, exactly as the disabled build does. The cache still
// holds one (flagged) entry per interval.
TEST(SplineWindowNonUniform, UniformGridRoutesToCardinal) {
  const SmoothTrajectory traj(42);
  SplineWindow w = makeNonUniformWindow(traj, 0, 500'000'000, cyclingDensity({1}));

  ASSERT_EQ(w.intervalBasisCount(), w.numKnots() - 3);
  for (int i = 0; i + 4 <= w.numKnots(); ++i) {
    SplineWindow::SegmentRef ref = w.segmentFor(w.knotTime(i));
    EXPECT_EQ(ref.blend, nullptr) << "interval " << i;
    EXPECT_EQ(ref.cum_blend, nullptr) << "interval " << i;
  }
}

// With all intervals equal, the non-uniform evaluation path is BYTE-IDENTICAL to the
// disabled (uniform) path: every uniform interval routes through the same cardinal
// kernels, so pose value and every derivative evaluator reproduce the disabled build
// bit-for-bit, and the SegmentRef carries null matrix pointers on both windows.
TEST(SplineWindowNonUniform, UniformDensityMatchesDisabledMode) {
  const SmoothTrajectory traj(8);
  const Timestamp t0 = 0;
  SplineWindow off = makeWindow(traj, t0, 700'000'000, 1);
  SplineWindow on = makeNonUniformWindow(traj, t0, 700'000'000, cyclingDensity({1}));
  ASSERT_EQ(off.numKnots(), on.numKnots());

  for (Timestamp t = t0; t < off.maxTime(); t += 13'000'000) {
    expectPoseBitEqual(on.pose(t), off.pose(t), "uniform-density pose");

    const Eigen::Vector3d wb_off = off.angularVelocityBody(t);
    const Eigen::Vector3d wb_on = on.angularVelocityBody(t);
    expectBitEqual(wb_on.x(), wb_off.x(), "ang vel x");
    expectBitEqual(wb_on.y(), wb_off.y(), "ang vel y");
    expectBitEqual(wb_on.z(), wb_off.z(), "ang vel z");

    const Eigen::Vector3d vw_off = off.linearVelocityWorld(t);
    const Eigen::Vector3d vw_on = on.linearVelocityWorld(t);
    expectBitEqual(vw_on.x(), vw_off.x(), "lin vel x");
    expectBitEqual(vw_on.y(), vw_off.y(), "lin vel y");
    expectBitEqual(vw_on.z(), vw_off.z(), "lin vel z");

    const Eigen::Vector3d aw_off = off.linearAccelWorld(t);
    const Eigen::Vector3d aw_on = on.linearAccelWorld(t);
    expectBitEqual(aw_on.x(), aw_off.x(), "lin acc x");
    expectBitEqual(aw_on.y(), aw_off.y(), "lin acc y");
    expectBitEqual(aw_on.z(), aw_off.z(), "lin acc z");

    SplineWindow::SegmentRef ra = off.segmentFor(t);
    SplineWindow::SegmentRef rb = on.segmentFor(t);
    expectBitEqual(rb.u, ra.u, "u");
    expectBitEqual(rb.dt_s, ra.dt_s, "dt_s");
    EXPECT_EQ(ra.blend, nullptr);
    EXPECT_EQ(ra.cum_blend, nullptr);
    EXPECT_EQ(rb.blend, nullptr);
    EXPECT_EQ(rb.cum_blend, nullptr);
  }
}

// Every interval's blending matrix is a partition of unity (basis weights sum to 1
// for all u; column sums are (1,0,0,0)) and its cumulative form has first row
// (1,0,0,0), on randomly gear-switched (genuinely non-uniform) knot vectors.
TEST(SplineWindowNonUniform, PartitionOfUnityAndCumulativeRowZero) {
  const SmoothTrajectory traj(43);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> gear(1, 4);
  SplineWindow w =
      makeNonUniformWindow(traj, 0, 1'000'000'000, [&](Timestamp, Timestamp) { return gear(rng); });

  ASSERT_EQ(w.intervalBasisCount(), w.numKnots() - 3);
  int checked = 0;
  for (int i = 0; i + 4 <= w.numKnots(); ++i) {
    SplineWindow::SegmentRef ref = w.segmentFor(w.knotTime(i));
    // Uniformly spaced intervals route to the cardinal matrices (null pointers); their
    // partition-of-unity property is the cardinal matrix's and is checked elsewhere.
    // Only the genuinely non-uniform intervals carry a per-interval matrix to verify.
    if (ref.blend == nullptr) {
      continue;
    }
    ASSERT_NE(ref.cum_blend, nullptr);
    ++checked;

    const Eigen::Vector4d col_sums = ref.blend->colwise().sum().transpose();
    EXPECT_NEAR(col_sums[0], 1.0, 1e-12);
    EXPECT_NEAR(col_sums[1], 0.0, 1e-12);
    EXPECT_NEAR(col_sums[2], 0.0, 1e-12);
    EXPECT_NEAR(col_sums[3], 0.0, 1e-12);

    EXPECT_NEAR((*ref.cum_blend)(0, 0), 1.0, 1e-15);
    EXPECT_NEAR((*ref.cum_blend)(0, 1), 0.0, 1e-15);
    EXPECT_NEAR((*ref.cum_blend)(0, 2), 0.0, 1e-15);
    EXPECT_NEAR((*ref.cum_blend)(0, 3), 0.0, 1e-15);

    for (double u : {0.0, 0.17, 0.5, 0.83, 1.0}) {
      const Eigen::Vector4d up(1.0, u, u * u, u * u * u);
      const Eigen::Vector4d wgt = (*ref.blend) * up;
      EXPECT_NEAR(wgt.sum(), 1.0, 1e-12);
      // Cumulative row 0 evaluates to the full partition of unity at every u.
      EXPECT_NEAR(((*ref.cum_blend) * up)[0], 1.0, 1e-12);
    }
  }
  EXPECT_GT(checked, 0);
}

// Each runtime matrix row, evaluated at sampled u, matches an independent numeric
// Cox-de Boor evaluation of the corresponding cubic basis on the same knot vector.
TEST(SplineWindowNonUniform, MatchesNumericCoxDeBoor) {
  const SmoothTrajectory traj(44);
  std::mt19937 rng(17);
  std::uniform_int_distribution<int> gear(1, 4);
  SplineWindow w = makeNonUniformWindow(traj, 250'000'000, 900'000'000,
                                        [&](Timestamp, Timestamp) { return gear(rng); });

  const KnotVector kv(w);
  int checked = 0;
  for (int i = 0; i + 4 <= w.numKnots(); ++i) {
    SplineWindow::SegmentRef ref = w.segmentFor(w.knotTime(i));
    // Uniformly spaced intervals route to the cardinal matrices (null pointers); the
    // per-interval matrix only exists where the spacing varies, which is what this
    // numeric Cox-de Boor reconstruction checks.
    if (ref.blend == nullptr) {
      continue;
    }
    ++checked;
    const double h = static_cast<double>(w.knotTime(i + 1) - w.knotTime(i));
    for (double u : {0.05, 0.3, 0.55, 0.8, 0.95}) {
      const double t = static_cast<double>(w.knotTime(i)) + u * h;
      const Eigen::Vector4d up(1.0, u, u * u, u * u * u);
      for (int j = 0; j < 4; ++j) {
        const double from_matrix = ref.blend->row(j).dot(up);
        const double numeric = kv.basis(i + j, 4, t);
        EXPECT_NEAR(from_matrix, numeric, 1e-10)
            << "interval " << i << " basis " << j << " u " << u;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// The non-uniform spline is C2 in REAL time across every knot join, including gear
// changes (1 -> 3 -> 2 -> 1), with arbitrary (randomly perturbed) control points:
// evaluating each join from the left interval at u=1 and the right interval at u=0
// gives identical value, velocity, and acceleration for both SO(3) and R^3.
TEST(SplineWindowNonUniform, C2ContinuousAcrossGearChanges) {
  const SmoothTrajectory traj(2025);
  SplineWindow w = makeNonUniformWindow(traj, 0, 800'000'000, cyclingDensity({1, 3, 2, 1}));

  // Random control points: continuity must come from the per-interval bases alone,
  // not from smooth seeding.
  std::mt19937 rng(99);
  std::uniform_real_distribution<double> un(-0.15, 0.15);
  for (int i = 0; i < w.numKnots(); ++i) {
    Eigen::Map<Eigen::Quaterniond> q(w.so3KnotData(i));
    const Eigen::Vector3d dphi(un(rng), un(rng), un(rng));
    q = (Sophus::SO3d(Eigen::Quaterniond(q)) * Sophus::SO3d::exp(dphi)).unit_quaternion();
    Eigen::Map<Eigen::Vector3d> p(w.r3KnotData(i));
    p += Eigen::Vector3d(un(rng), un(rng), un(rng));
  }

  // Cardinal matrices used wherever an interval is uniformly spaced (the SegmentRef
  // then reports null pointers); they are the exact non-uniform blending for equal
  // spacing, so continuity across a uniform<->varying join still holds.
  const Eigen::Matrix4d Mcard = basalt::CeresSplineHelper<4>::blending_matrix_;
  const Eigen::Matrix4d Ccard = basalt::CeresSplineHelper<4>::cumulative_blending_matrix_;

  int joins = 0;
  // Join k sits between interval k-1 and interval k; both must be evaluable.
  for (int k = 1; k <= w.numKnots() - 4; ++k) {
    const Timestamp join = w.knotTime(k);
    SplineWindow::SegmentRef l = w.segmentFor(join - 1);
    SplineWindow::SegmentRef r = w.segmentFor(join);
    const Eigen::Matrix4d& bl_l = l.blend ? *l.blend : Mcard;
    const Eigen::Matrix4d& bl_r = r.blend ? *r.blend : Mcard;
    const Eigen::Matrix4d& cm_l = l.cum_blend ? *l.cum_blend : Ccard;
    const Eigen::Matrix4d& cm_r = r.cum_blend ? *r.cum_blend : Ccard;
    const double inv_l = 1.0 / l.dt_s;
    const double inv_r = 1.0 / r.dt_s;

    std::array<const double*, 4> so3_l{l.so3_knots[0], l.so3_knots[1], l.so3_knots[2],
                                       l.so3_knots[3]};
    std::array<const double*, 4> so3_r{r.so3_knots[0], r.so3_knots[1], r.so3_knots[2],
                                       r.so3_knots[3]};
    std::array<const double*, 4> r3_l{l.r3_knots[0], l.r3_knots[1], l.r3_knots[2], l.r3_knots[3]};
    std::array<const double*, 4> r3_r{r.r3_knots[0], r.r3_knots[1], r.r3_knots[2], r.r3_knots[3]};

    Sophus::SO3d R_l, R_r;
    Eigen::Vector3d w_l, w_r, dw_l, dw_r;
    meridian::ct::evaluateLieWithMatrix<double, Sophus::SO3>(so3_l.data(), 1.0, inv_l, cm_l, &R_l,
                                                             &w_l, &dw_l);
    meridian::ct::evaluateLieWithMatrix<double, Sophus::SO3>(so3_r.data(), 0.0, inv_r, cm_r, &R_r,
                                                             &w_r, &dw_r);
    EXPECT_LT((R_l.inverse() * R_r).log().norm(), 1e-9) << "join " << k;
    EXPECT_LT((w_l - w_r).norm(), 1e-9 * (1.0 + w_l.norm())) << "join " << k;
    EXPECT_LT((dw_l - dw_r).norm(), 1e-9 * (1.0 + dw_l.norm())) << "join " << k;

    Eigen::Vector3d p_l, p_r, v_l, v_r, a_l, a_r;
    meridian::ct::evaluateRdWithMatrix<double, 3, 0>(r3_l.data(), 1.0, inv_l, bl_l, &p_l);
    meridian::ct::evaluateRdWithMatrix<double, 3, 0>(r3_r.data(), 0.0, inv_r, bl_r, &p_r);
    meridian::ct::evaluateRdWithMatrix<double, 3, 1>(r3_l.data(), 1.0, inv_l, bl_l, &v_l);
    meridian::ct::evaluateRdWithMatrix<double, 3, 1>(r3_r.data(), 0.0, inv_r, bl_r, &v_r);
    meridian::ct::evaluateRdWithMatrix<double, 3, 2>(r3_l.data(), 1.0, inv_l, bl_l, &a_l);
    meridian::ct::evaluateRdWithMatrix<double, 3, 2>(r3_r.data(), 0.0, inv_r, bl_r, &a_r);
    EXPECT_LT((p_l - p_r).norm(), 1e-9) << "join " << k;
    EXPECT_LT((v_l - v_r).norm(), 1e-9 * (1.0 + v_l.norm())) << "join " << k;
    EXPECT_LT((a_l - a_r).norm(), 1e-9 * (1.0 + a_l.norm())) << "join " << k;

    // The public evaluators agree from both sides of the join as well. The two
    // queries are 1 ns apart, so each output legitimately moves by its own time
    // derivative times 1e-9 s: with random control points the acceleration reaches
    // ~1e3 m/s^2 and the jerk ~1e5 m/s^3, so velocity may shift by ~1e-6 and
    // acceleration by ~1e-4 between the queries. A C0-only warp would instead jump
    // the derivatives by order one at a gear change, far beyond these bounds.
    EXPECT_LT((w.pose(join - 1).t - w.pose(join).t).norm(), 1e-6);
    EXPECT_LT((w.linearVelocityWorld(join - 1) - w.linearVelocityWorld(join)).norm(), 1e-5);
    EXPECT_LT((w.linearAccelWorld(join - 1) - w.linearAccelWorld(join)).norm(), 1e-3);
    EXPECT_LT((w.angularVelocityBody(join - 1) - w.angularVelocityBody(join)).norm(), 1e-5);
    ++joins;
  }
  EXPECT_GT(joins, 10);
}

// Velocity and acceleration evaluators carry the correct per-interval 1/dt scaling
// on mixed-density grids: they match central finite differences of pose(t), which is
// C2 in real time in non-uniform mode so the stencil may cross interval joins.
TEST(SplineWindowNonUniform, DerivativesMatchFiniteDifferences) {
  const SmoothTrajectory traj(31415);
  const Timestamp t0 = 1'000'000'000;
  SplineWindow w = makeNonUniformWindow(traj, t0, 900'000'000, cyclingDensity({1, 3, 2, 1}));

  // 20 us step: the warm-start seeding leaves real jerk of order 1e3 m/s^3 in the
  // intervals adjacent to a gear change, so the central-difference truncation error
  // (jerk * h^2 / 6) needs a small h to stay well under the comparison tolerances,
  // while cancellation noise (eps / h, eps / h^2) remains negligible.
  const Timestamp h = 20'000;
  const double hs = meridian::to_seconds(h);

  int checked = 0;
  for (Timestamp t = measurementStart(t0); t < t0 + 800'000'000; t += 7'300'000) {
    ASSERT_TRUE(w.covers(t + h));
    // Restrict the stencil to a single knot interval: within one interval the spline
    // is a polynomial, so central differences are accurate; across a join the jerk
    // is discontinuous (the spline is C2, not C3) and the FD truncation error is no
    // longer O(h^2). Cross-join behavior is covered by the C2 continuity test.
    if (w.leadingKnotIndex(t - h) != w.leadingKnotIndex(t + h)) {
      continue;
    }
    const Pose c = w.pose(t);
    const Pose fwd = w.pose(t + h);
    const Pose bwd = w.pose(t - h);

    const Eigen::Vector3d v_fd = (fwd.t - bwd.t) / (2.0 * hs);
    const Eigen::Vector3d v_an = w.linearVelocityWorld(t);
    EXPECT_LT((v_fd - v_an).norm(), 1e-6 * (1.0 + v_an.norm()));

    const Eigen::Vector3d a_fd = (fwd.t - 2.0 * c.t + bwd.t) / (hs * hs);
    const Eigen::Vector3d a_an = w.linearAccelWorld(t);
    EXPECT_LT((a_fd - a_an).norm(), 1e-5 * (1.0 + a_an.norm()));

    const Eigen::Vector3d w_fd =
        (Sophus::SO3d(bwd.q).inverse() * Sophus::SO3d(fwd.q)).log() / (2.0 * hs);
    const Eigen::Vector3d w_an = w.angularVelocityBody(t);
    EXPECT_LT((w_fd - w_an).norm(), 1e-6 * (1.0 + w_an.norm()));
    ++checked;
  }
  EXPECT_GT(checked, 50);
}

// ---------------------------------------------------------------------------
// Analytic Jacobians with per-interval matrices
// ---------------------------------------------------------------------------

// A null cum_blend pointer binds the cardinal matrix object itself, so the result is
// bit-identical to passing that matrix explicitly (and to the pre-existing
// signature, which had no matrix parameter at all).
TEST(SplineAnalytic, VelAccelJacNullMatrixBindsCardinal) {
  const SmoothTrajectory traj(606);
  std::array<Sophus::SO3d, 4> knots;
  for (int i = 0; i < 4; ++i) {
    knots[static_cast<std::size_t>(i)] = Sophus::SO3d(traj.at(i * kKnotDt).q);
  }
  const double u = 0.37;
  const double inv_dt = 20.0;

  const meridian::ct::So3DerivJac a = meridian::ct::evaluateSo3VelAccelJac(knots, u, inv_dt, true);
  const meridian::ct::So3DerivJac b = meridian::ct::evaluateSo3VelAccelJac(
      knots, u, inv_dt, true, &basalt::CeresSplineHelper<4>::cumulative_blending_matrix_);

  EXPECT_EQ((a.omega - b.omega).norm(), 0.0);
  EXPECT_EQ((a.alpha - b.alpha).norm(), 0.0);
  for (int k = 0; k < 4; ++k) {
    EXPECT_EQ((a.domega_dR[static_cast<std::size_t>(k)] - b.domega_dR[static_cast<std::size_t>(k)])
                  .norm(),
              0.0);
    EXPECT_EQ((a.dalpha_dR[static_cast<std::size_t>(k)] - b.dalpha_dR[static_cast<std::size_t>(k)])
                  .norm(),
              0.0);
  }
}

// On mixed-density intervals the analytic omega/alpha Jacobians agree with central
// numeric differentiation of the matrix-based forward evaluation, and the forward
// values agree between the two code paths.
TEST(SplineAnalytic, VelAccelJacMatchesNumericOnNonUniformIntervals) {
  const SmoothTrajectory traj(7070);
  const Timestamp t0 = 0;
  SplineWindow w = makeNonUniformWindow(traj, t0, 800'000'000, cyclingDensity({2, 1, 3, 4}));

  for (Timestamp t = measurementStart(t0); t < t0 + 600'000'000; t += 41'000'000) {
    SplineWindow::SegmentRef ref = w.segmentFor(t);
    ASSERT_NE(ref.cum_blend, nullptr);
    const double inv_dt = 1.0 / ref.dt_s;

    std::array<Sophus::SO3d, 4> knots;
    // Local copies of the knot quaternions so each can be perturbed independently.
    std::array<std::array<double, 4>, 4> buf{};
    std::array<const double*, 4> ptrs{};
    for (int k = 0; k < 4; ++k) {
      const std::size_t sk = static_cast<std::size_t>(k);
      std::memcpy(buf[sk].data(), ref.so3_knots[k], 4 * sizeof(double));
      ptrs[sk] = buf[sk].data();
      knots[sk] =
          Sophus::SO3d(Eigen::Quaterniond(Eigen::Map<const Eigen::Quaterniond>(ref.so3_knots[k])));
    }

    const meridian::ct::So3DerivJac jac =
        meridian::ct::evaluateSo3VelAccelJac(knots, ref.u, inv_dt, true, ref.cum_blend);

    // Forward values agree with the generic matrix evaluator.
    Sophus::SO3d R0;
    Eigen::Vector3d w0, a0;
    meridian::ct::evaluateLieWithMatrix<double, Sophus::SO3>(ptrs.data(), ref.u, inv_dt,
                                                             *ref.cum_blend, &R0, &w0, &a0);
    EXPECT_LT((jac.omega - w0).norm(), 1e-12 * (1.0 + w0.norm()));
    EXPECT_LT((jac.alpha - a0).norm(), 1e-12 * (1.0 + a0.norm()));

    // Numeric Jacobians: right-perturb each knot along each tangent axis.
    const double eps = 1e-6;
    for (int k = 0; k < 4; ++k) {
      const std::size_t sk = static_cast<std::size_t>(k);
      Eigen::Matrix3d num_dw, num_da;
      for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d delta = Eigen::Vector3d::Zero();
        delta[d] = eps;

        auto eval = [&](const Sophus::SO3d& knot_k, Eigen::Vector3d* wo, Eigen::Vector3d* ao) {
          std::array<double, 4> saved = buf[sk];
          Eigen::Map<Eigen::Quaterniond>(buf[sk].data()) = knot_k.unit_quaternion();
          Sophus::SO3d R;
          meridian::ct::evaluateLieWithMatrix<double, Sophus::SO3>(ptrs.data(), ref.u, inv_dt,
                                                                   *ref.cum_blend, &R, wo, ao);
          buf[sk] = saved;
        };

        Eigen::Vector3d w_p, a_p, w_m, a_m;
        eval(knots[sk] * Sophus::SO3d::exp(delta), &w_p, &a_p);
        eval(knots[sk] * Sophus::SO3d::exp(-delta), &w_m, &a_m);
        num_dw.col(d) = (w_p - w_m) / (2.0 * eps);
        num_da.col(d) = (a_p - a_m) / (2.0 * eps);
      }
      EXPECT_LT((num_dw - jac.domega_dR[sk]).norm(), 1e-6 * (1.0 + jac.domega_dR[sk].norm()))
          << "t " << t << " knot " << k;
      EXPECT_LT((num_da - jac.dalpha_dR[sk]).norm(), 1e-6 * (1.0 + jac.dalpha_dR[sk].norm()))
          << "t " << t << " knot " << k;
    }
  }
}

// On uniformly-spaced knots the analytic Jacobians the LiDAR/IMU/regularizer factors
// hand to Ceres are BYTE-IDENTICAL between a non-uniform window and a disabled one,
// because the non-uniform window detects the uniform spacing and routes every interval
// to the cardinal matrices (the SegmentRef carries null pointers, so the factors fall
// back to the very same hard-coded blending matrices). Each block below reassembles the
// exact Jacobian the corresponding cost emits, once resolving the non-uniform window's
// (null) pointers to cardinal and once with cardinal directly, on identical knots, and
// requires the worst discrepancy per Jacobian type to be exactly zero.
TEST(SplineWindowNonUniform, ResidualJacobiansMatchUniformOnUniformKnots) {
  using meridian::ct::evaluateSo3CumulativeJac;
  using meridian::ct::evaluateSo3VelAccelJac;
  using meridian::ct::evaluateSo3VelocityJac;
  using meridian::ct::So3DerivJac;
  using meridian::ct::So3SplineJac;
  using meridian::ct::So3VelocityJac;
  using meridian::ct::tangentToQuatJac;
  using SO3 = Sophus::SO3d;

  const SmoothTrajectory traj(20240611);
  const Timestamp t0 = 0;
  const Timestamp span = 800'000'000;
  // Gear forced to 1: uniformly spaced knots, but evaluation routed through the
  // per-interval blending-matrix path.
  SplineWindow nu = makeNonUniformWindow(traj, t0, span, cyclingDensity({1}));
  // Uniform reference on the same grid (cardinal matrices, null SegmentRef matrices).
  SplineWindow uni = makeWindow(traj, t0, span, 1);
  ASSERT_EQ(nu.numKnots(), uni.numKnots());
  // The two windows must carry bit-identical control points, so the only thing the
  // comparison below exercises is the blending-matrix path, never the knot values.
  for (int i = 0; i < nu.numKnots(); ++i) {
    ASSERT_EQ(std::memcmp(nu.so3KnotData(i), uni.so3KnotData(i), 4 * sizeof(double)), 0)
        << "so3 knot " << i;
    ASSERT_EQ(std::memcmp(nu.r3KnotData(i), uni.r3KnotData(i), 3 * sizeof(double)), 0)
        << "r3 knot " << i;
  }

  const Eigen::Matrix4d& Ccard = basalt::CeresSplineHelper<4>::cumulative_blending_matrix_;
  const Eigen::Matrix4d& Mcard = basalt::CeresSplineHelper<4>::blending_matrix_;

  // Fixed factor data, shared by both paths so only the matrices vary.
  const Eigen::Vector3d n_plane = Eigen::Vector3d(0.3, -0.7, 0.5).normalized();
  const Eigen::Vector3d q_fe(0.21, -0.34, 0.52);  // body-frame lidar point (post-extrinsic)
  const double w_lidar = 2.5;
  const Eigen::Vector3d g_dir = Eigen::Vector3d(0.02, -0.01, -1.0).normalized();
  const double g_mag = 9.81;
  const double w_gyro = 40.0;
  const double w_accel = 8.0;
  const double w_reg = 0.5;

  // Relative discrepancy of two equal-shape Jacobian blocks, normalized by the
  // magnitude of the uniform-path block plus one (so a near-zero block stays bounded).
  auto relDiff = [](const auto& a, const auto& b) {
    return (a - b).cwiseAbs().maxCoeff() / (1.0 + b.cwiseAbs().maxCoeff());
  };

  double max_lidar_so3 = 0.0;
  double max_lidar_r3 = 0.0;
  double max_imu_so3 = 0.0;
  double max_imu_r3 = 0.0;
  double max_angaccel_so3 = 0.0;
  double max_rate_so3 = 0.0;
  double max_mat_cum = 0.0;
  double max_mat_blend = 0.0;
  int checked = 0;

  for (Timestamp t = measurementStart(t0) + 5'000'000; t < t0 + 600'000'000; t += 9'000'000) {
    ASSERT_TRUE(nu.covers(t));
    SplineWindow::SegmentRef sn = nu.segmentFor(t);
    SplineWindow::SegmentRef su = uni.segmentFor(t);
    // Uniform spacing: both windows route to cardinal, so neither carries matrices.
    ASSERT_EQ(sn.cum_blend, nullptr);
    ASSERT_EQ(sn.blend, nullptr);
    ASSERT_EQ(su.cum_blend, nullptr);
    ASSERT_EQ(su.blend, nullptr);
    ASSERT_EQ(sn.u, su.u);
    ASSERT_EQ(sn.dt_s, su.dt_s);
    const double u = sn.u;
    const double inv_dt = 1.0 / sn.dt_s;
    const Eigen::Vector4d up(1.0, u, u * u, u * u * u);
    const Eigen::Vector4d p1(0.0, 1.0, 2.0 * u, 3.0 * u * u);
    const Eigen::Vector4d p2(0.0, 0.0, 2.0, 6.0 * u);

    // The "non-uniform" path resolves its null pointers to the cardinal matrices -- the
    // exact fallback the residual factors perform -- so it is the identical computation.
    const Eigen::Matrix4d& Cn = sn.cum_blend ? *sn.cum_blend : Ccard;
    const Eigen::Matrix4d& Mn = sn.blend ? *sn.blend : Mcard;
    max_mat_cum = std::max(max_mat_cum, (Cn - Ccard).cwiseAbs().maxCoeff());
    max_mat_blend = std::max(max_mat_blend, (Mn - Mcard).cwiseAbs().maxCoeff());

    std::array<SO3, 4> R;
    std::array<Eigen::Vector3d, 4> P;
    for (int j = 0; j < 4; ++j) {
      const std::size_t k = static_cast<std::size_t>(j);
      R[k] = SO3(Eigen::Map<const Eigen::Quaterniond>(sn.so3_knots[k]));
      P[k] = Eigen::Map<const Eigen::Vector3d>(sn.r3_knots[k]);
    }

    // Blended coefficients, cardinal (uniform) vs per-interval (non-uniform).
    const Eigen::Vector4d lam_r_u = Ccard * up;
    const Eigen::Vector4d dco_u = inv_dt * (Ccard * p1);
    const Eigen::Vector4d lam_p_u = Mcard * up;
    const Eigen::Vector4d lam_acc_u = (inv_dt * inv_dt) * (Mcard * p2);
    const Eigen::Vector4d lam_r_n = Cn * up;
    const Eigen::Vector4d dco_n = inv_dt * (Cn * p1);
    const Eigen::Vector4d lam_p_n = Mn * up;
    const Eigen::Vector4d lam_acc_n = (inv_dt * inv_dt) * (Mn * p2);

    // Point-to-plane cost: SO(3) knot rows ddis_dRt * dRt_dR mapped to ambient
    // quaternion, and R^3 knot rows w * lambda_p * n^T.
    auto lidarJac = [&](const Eigen::Vector4d& lam_r, const Eigen::Vector4d& lam_p,
                        std::array<Eigen::Matrix<double, 1, 4>, 4>* j_so3,
                        std::array<Eigen::Matrix<double, 1, 3>, 4>* j_r3) {
      const So3SplineJac sj = evaluateSo3CumulativeJac(R, lam_r);
      const Eigen::Matrix<double, 1, 3> ddis =
          -w_lidar * n_plane.transpose() * sj.R.matrix() * SO3::hat(q_fe);
      for (int j = 0; j < 4; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        (*j_so3)[k] = (ddis * sj.dRt_dR[k]) * tangentToQuatJac(R[k].unit_quaternion());
        (*j_r3)[k] = (w_lidar * lam_p[j]) * n_plane.transpose();
      }
    };
    std::array<Eigen::Matrix<double, 1, 4>, 4> Lso3_u, Lso3_n;
    std::array<Eigen::Matrix<double, 1, 3>, 4> Lr3_u, Lr3_n;
    lidarJac(lam_r_u, lam_p_u, &Lso3_u, &Lr3_u);
    lidarJac(lam_r_n, lam_p_n, &Lso3_n, &Lr3_n);
    for (int j = 0; j < 4; ++j) {
      const std::size_t k = static_cast<std::size_t>(j);
      max_lidar_so3 = std::max(max_lidar_so3, relDiff(Lso3_n[k], Lso3_u[k]));
      max_lidar_r3 = std::max(max_lidar_r3, relDiff(Lr3_n[k], Lr3_u[k]));
    }

    // Combined gyro+accel cost (shared-bias layout): SO(3) knot rows stack the
    // angular-velocity and specific-force Jacobians; R^3 knot rows are the accel block.
    auto imuJac = [&](const Eigen::Vector4d& lam_r, const Eigen::Vector4d& dco,
                      const Eigen::Vector4d& lam_acc,
                      std::array<Eigen::Matrix<double, 6, 4>, 4>* j_so3,
                      std::array<Eigen::Matrix<double, 6, 3>, 4>* j_r3) {
      const So3SplineJac sj = evaluateSo3CumulativeJac(R, lam_r);
      const So3VelocityJac vj = evaluateSo3VelocityJac(R, lam_r, dco);
      const Eigen::Matrix3d R_fe_w = sj.R.inverse().matrix();
      Eigen::Vector3d a_world = Eigen::Vector3d::Zero();
      for (int j = 0; j < 4; ++j) a_world += lam_acc[j] * P[static_cast<std::size_t>(j)];
      const Eigen::Vector3d v_b = R_fe_w * (a_world - g_dir * g_mag);
      for (int j = 0; j < 4; ++j) {
        const std::size_t k = static_cast<std::size_t>(j);
        Eigen::Matrix<double, 6, 3> jt;
        jt.topRows<3>() = w_gyro * vj.dOmega_dR[k];
        jt.bottomRows<3>() = w_accel * SO3::hat(v_b) * sj.dRt_dR[k];
        (*j_so3)[k] = jt * tangentToQuatJac(R[k].unit_quaternion());
        Eigen::Matrix<double, 6, 3> jr = Eigen::Matrix<double, 6, 3>::Zero();
        jr.bottomRows<3>() = (w_accel * lam_acc[j]) * R_fe_w;
        (*j_r3)[k] = jr;
      }
    };
    std::array<Eigen::Matrix<double, 6, 4>, 4> Iso3_u, Iso3_n;
    std::array<Eigen::Matrix<double, 6, 3>, 4> Ir3_u, Ir3_n;
    imuJac(lam_r_u, dco_u, lam_acc_u, &Iso3_u, &Ir3_u);
    imuJac(lam_r_n, dco_n, lam_acc_n, &Iso3_n, &Ir3_n);
    for (int j = 0; j < 4; ++j) {
      const std::size_t k = static_cast<std::size_t>(j);
      max_imu_so3 = std::max(max_imu_so3, relDiff(Iso3_n[k], Iso3_u[k]));
      max_imu_r3 = std::max(max_imu_r3, relDiff(Ir3_n[k], Ir3_u[k]));
    }

    // Angular-acceleration and rate-anchor costs go straight through the matrix-taking
    // velocity/acceleration Jacobian helper (want_accel true for alpha, false for omega).
    const So3DerivJac dja_u = evaluateSo3VelAccelJac(R, u, inv_dt, true, &Ccard);
    const So3DerivJac dja_n = evaluateSo3VelAccelJac(R, u, inv_dt, true, sn.cum_blend);
    const So3DerivJac djw_u = evaluateSo3VelAccelJac(R, u, inv_dt, false, &Ccard);
    const So3DerivJac djw_n = evaluateSo3VelAccelJac(R, u, inv_dt, false, sn.cum_blend);
    for (int j = 0; j < 4; ++j) {
      const std::size_t k = static_cast<std::size_t>(j);
      const Eigen::Matrix<double, 3, 4> q = tangentToQuatJac(R[k].unit_quaternion());
      max_angaccel_so3 = std::max(
          max_angaccel_so3,
          relDiff((w_reg * dja_n.dalpha_dR[k]) * q, (w_reg * dja_u.dalpha_dR[k]) * q));
      max_rate_so3 = std::max(
          max_rate_so3, relDiff((w_reg * djw_n.domega_dR[k]) * q, (w_reg * djw_u.domega_dR[k]) * q));
    }
    ++checked;
  }
  EXPECT_GT(checked, 30);

  std::cout << "[jac-diff] max relative discrepancy per Jacobian type:\n"
            << "  lidar_so3   = " << max_lidar_so3 << "\n"
            << "  lidar_r3    = " << max_lidar_r3 << "\n"
            << "  imu_so3     = " << max_imu_so3 << "\n"
            << "  imu_r3      = " << max_imu_r3 << "\n"
            << "  angaccel_so3= " << max_angaccel_so3 << "\n"
            << "  rate_so3    = " << max_rate_so3 << "\n"
            << "[mat-diff] max |per-interval - cardinal| (0 == bitwise identical):\n"
            << "  cum_blend   = " << max_mat_cum << "\n"
            << "  blend       = " << max_mat_blend << std::endl;

  // Both paths run the identical cardinal matrices on identical knots, so every
  // Jacobian type and both matrices agree to the bit: any nonzero discrepancy is a
  // routing regression (a uniform interval that failed to fall back to cardinal).
  EXPECT_EQ(max_mat_cum, 0.0);
  EXPECT_EQ(max_mat_blend, 0.0);
  EXPECT_EQ(max_lidar_so3, 0.0);
  EXPECT_EQ(max_lidar_r3, 0.0);
  EXPECT_EQ(max_imu_so3, 0.0);
  EXPECT_EQ(max_imu_r3, 0.0);
  EXPECT_EQ(max_angaccel_so3, 0.0);
  EXPECT_EQ(max_rate_so3, 0.0);
}

// ---------------------------------------------------------------------------
// Disabled-mode contract
// ---------------------------------------------------------------------------

namespace {

// Closed-form seed used when the golden values below were recorded; must never change.
Pose goldenSeed(Timestamp t) {
  const double s = meridian::to_seconds(t);
  const Eigen::Vector3d p(0.7 * std::sin(0.9 * s), -0.4 * std::sin(1.3 * s + 0.5),
                          0.25 * std::sin(0.7 * s + 1.1));
  const Eigen::Vector3d phi(0.3 * std::sin(1.1 * s + 0.2), 0.25 * std::sin(0.8 * s + 2.0),
                            -0.2 * std::sin(1.4 * s + 0.7));
  return Pose{Sophus::SO3d::exp(phi).unit_quaternion(), p};
}

// Recorded from the pre-change implementation, compiled with the package's
// optimization flags: per sample time the 18 doubles are pose q(x,y,z,w), pose t,
// angularVelocityBody, linearVelocityWorld, linearAccelWorld, segmentFor().u,
// segmentFor().dt_s.
constexpr std::array<double, 108> kGolden = {
    // t0 + 123456789 ns
    0x1.2b163c5db9ba2p-3, 0x1.1d09750b26e1ap-5, -0x1.478c398260304p-4, 0x1.f889f25ce3c81p-1,
    0x1.25e4f23efd4d8p-1, -0x1.82961ee8c77afp-2, 0x1.eb6f4d33939bcp-3, 0x1.41bdcb933f651p-3,
    -0x1.7846996231289p-2, 0x1.c87dd29f0357ap-2, 0x1.a3d1feec382dcp-1, 0x1.8be2acb578f92p-2,
    -0x1.ccd5646aa66b8p-4, 0x1.66c8eefea864p+1, 0x1.a839d6e31098p+0, -0x1.d31dae11c7c8p-2,
    0x1.e0652141ef0dcp-2, 0x1.999999999999ap-5,
    // t0 + 200000000 ns
    0x1.3159d7eae0f1fp-3, 0x1.70217e99b0a17p-6, -0x1.194b43ef6abb6p-4, 0x1.f8ec720927706p-1,
    0x1.3c11277b9415fp-1, -0x1.697e4815750b9p-2, 0x1.dd7979d86826dp-3, 0x1.85a0a4964036p-6,
    -0x1.4c0044c7d47a7p-3, 0x1.d55b83634d016p-3, 0x1.3009a6ae5382p-2, 0x1.f45c792ba8cep-3,
    -0x1.02a9150e877cp-4, -0x1.00066b80504p-1, 0x1.317d1c7ac68p-1, -0x1.d3efdd139dp-4, 0x0p+0,
    0x1.999999999999ap-6,
    // t0 + 333333333 ns
    0x1.3076493771d2fp-3, 0x1.314b8d1b0bd32p-7, -0x1.bb7d605dcd155p-5, 0x1.f987603a497bcp-1,
    0x1.4e03be6ec418ap-1, -0x1.42e2c420985b7p-2, 0x1.ca2d66deccd52p-3, -0x1.8fc255053395ap-6,
    -0x1.4e116267f2522p-3, 0x1.0b2dd49442c43p-2, 0x1.d37d2b1aa343p-3, 0x1.477b25f33f752p-2,
    -0x1.3fd58c098f10cp-4, -0x1.0e8c4b7456cp-1, 0x1.10d4a18c0dep-1, -0x1.c1028f4a348p-4,
    0x1.5555547044b69p-2, 0x1.999999999999ap-6,
    // t0 + 450000001 ns
    0x1.2a59306409351p-3, -0x1.2c68ee150fef5p-9, -0x1.4673ec97a1628p-5, 0x1.fa20256091968p-1,
    0x1.59cd23a498c5dp-1, -0x1.1925a000a79a3p-2, 0x1.b606c67b74686p-3, -0x1.0d2a5058496fbp-4,
    -0x1.4ce903ef3857ap-3, 0x1.212fc13327e3fp-2, 0x1.52e792c6f5e6bp-3, 0x1.832d96cae4b05p-2,
    -0x1.731afe04d0974p-4, -0x1.181ab50f19e79p-1, 0x1.db282f4c974dp-2, -0x1.ad45538728853p-4,
    0x1.01b2b34736c2cp-24, 0x1.11111059d0922p-6,
    // t0 + 587000000 ns
    0x1.1ce2975c04419p-3, -0x1.048e913125e5dp-6, -0x1.63fb2d93b89f5p-6, 0x1.fad5b31702e8ep-1,
    0x1.62c1106e73855p-1, -0x1.bfdf6e36941abp-3, 0x1.9aa1cb7288deep-3, -0x1.cc6c06157a8eap-4,
    -0x1.48236a6ea46b9p-3, 0x1.32744846b437ap-2, 0x1.6e73b251631fcp-4, 0x1.bdcdd8fe8969ep-2,
    -0x1.ac1e9b384f12ep-4, -0x1.1f59d651f3246p-1, 0x1.7a7319097ac77p-2, -0x1.926ab2a060872p-4,
    0x1.c28f648901ebbp-3, 0x1.1111127f920f2p-6,
    // t0 + 700000000 ns
    0x1.0cc3b78869eap-3, -0x1.ba4d64f3e633cp-6, -0x1.91d53299619d4p-8, 0x1.fb5fc348dfec6p-1,
    0x1.65fb1fb610315p-1, -0x1.5671e0a19ce03p-3, 0x1.811d57b46c9b9p-3, -0x1.30bd191e0710bp-3,
    -0x1.417a7ad5f960cp-3, 0x1.391537167d6fap-2, 0x1.a4d66a22a65p-6, 0x1.e34fba7a86f84p-2,
    -0x1.d81f09b4a99bp-4, -0x1.220375289e6p-1, 0x1.2177c06b9dbp-2, -0x1.79736b50044p-4, 0x0p+0,
    0x1.999999999999ap-5};

}  // namespace

// With the non-uniform mode off (the default 2-argument constructor), every output
// is bit-identical to the implementation before the mode existed, and SegmentRef
// carries null matrix pointers.
TEST(SplineWindow, DisabledModeMatchesGoldenBitExact) {
  constexpr Timestamp kT0 = 1'000'000'000;
  SplineWindow w(kKnotDt, 4);
  w.initialize(kT0, goldenSeed(kT0));
  w.extendTo(kT0 + 300'000'000, goldenSeed, 2);
  w.extendTo(kT0 + 600'000'000, goldenSeed, 3);
  w.extendTo(kT0 + 800'000'000, goldenSeed, 1);
  w.dropOldest(2);

  ASSERT_EQ(w.numKnots(), 33);
  EXPECT_EQ(w.minTime(), 1'100'000'000);
  EXPECT_EQ(w.maxTime(), 1'849'999'999);
  EXPECT_EQ(w.intervalBasisCount(), 0);

  const std::array<Timestamp, 6> times = {kT0 + 123'456'789, kT0 + 200'000'000, kT0 + 333'333'333,
                                          kT0 + 450'000'001, kT0 + 587'000'000, kT0 + 700'000'000};
  const std::array<int, 6> leads = {0, 3, 8, 14, 22, 27};

  std::size_t k = 0;
  for (std::size_t s = 0; s < times.size(); ++s) {
    const Timestamp t = times[s];
    const Pose p = w.pose(t);
    expectBitEqual(p.q.x(), kGolden[k++], "q.x");
    expectBitEqual(p.q.y(), kGolden[k++], "q.y");
    expectBitEqual(p.q.z(), kGolden[k++], "q.z");
    expectBitEqual(p.q.w(), kGolden[k++], "q.w");
    expectBitEqual(p.t.x(), kGolden[k++], "t.x");
    expectBitEqual(p.t.y(), kGolden[k++], "t.y");
    expectBitEqual(p.t.z(), kGolden[k++], "t.z");
    const Eigen::Vector3d wb = w.angularVelocityBody(t);
    expectBitEqual(wb.x(), kGolden[k++], "wb.x");
    expectBitEqual(wb.y(), kGolden[k++], "wb.y");
    expectBitEqual(wb.z(), kGolden[k++], "wb.z");
    const Eigen::Vector3d vw = w.linearVelocityWorld(t);
    expectBitEqual(vw.x(), kGolden[k++], "vw.x");
    expectBitEqual(vw.y(), kGolden[k++], "vw.y");
    expectBitEqual(vw.z(), kGolden[k++], "vw.z");
    const Eigen::Vector3d aw = w.linearAccelWorld(t);
    expectBitEqual(aw.x(), kGolden[k++], "aw.x");
    expectBitEqual(aw.y(), kGolden[k++], "aw.y");
    expectBitEqual(aw.z(), kGolden[k++], "aw.z");
    SplineWindow::SegmentRef ref = w.segmentFor(t);
    EXPECT_EQ(ref.blend, nullptr);
    EXPECT_EQ(ref.cum_blend, nullptr);
    expectBitEqual(ref.u, kGolden[k++], "u");
    expectBitEqual(ref.dt_s, kGolden[k++], "dt_s");
    EXPECT_EQ(w.leadingKnotIndex(t), leads[s]);
  }
}

// ---------------------------------------------------------------------------
// Lifecycle: cache invariant, truncation, clone, density extension
// ---------------------------------------------------------------------------

// mats_ holds exactly one entry per evaluable knot interval through every mutation,
// and stays empty in disabled mode.
TEST(SplineWindowNonUniform, IntervalBasisCountInvariant) {
  const SmoothTrajectory traj(311);
  SplineWindow w(kKnotDt, 4, /*non_uniform=*/true);
  EXPECT_EQ(w.intervalBasisCount(), 0);

  w.initialize(0, traj.at(0));
  EXPECT_EQ(w.numKnots(), 4);
  EXPECT_EQ(w.intervalBasisCount(), 1);

  auto seed = [&](Timestamp t) { return traj.at(t); };
  w.extendTo(400'000'000, seed, cyclingDensity({2, 1, 3}));
  EXPECT_EQ(w.intervalBasisCount(), w.numKnots() - 3);

  w.dropOldest(3);
  EXPECT_EQ(w.intervalBasisCount(), w.numKnots() - 3);

  const Timestamp cut = w.segmentStartContaining(w.maxTime());
  const int removed = w.truncateTailSegments(cut);
  EXPECT_GT(removed, 0);
  EXPECT_EQ(w.intervalBasisCount(), w.numKnots() - 3);

  w.extendTo(500'000'000, seed, cyclingDensity({1}));
  EXPECT_EQ(w.intervalBasisCount(), w.numKnots() - 3);

  SplineWindow off(kKnotDt, 4);
  off.initialize(0, traj.at(0));
  off.extendTo(300'000'000, seed, 2);
  EXPECT_EQ(off.intervalBasisCount(), 0);
}

// Truncating trailing segments and re-extending with the same seed and density
// reproduces the never-truncated window bit-for-bit: knot times, knot values, cache
// size, and evaluations.
TEST(SplineWindowNonUniform, TruncateThenReextendIsBitIdentical) {
  const SmoothTrajectory traj(626);
  auto seed = [&](Timestamp t) { return traj.at(t); };
  // Density keyed on the segment start so it replays identically after truncation.
  auto density = [](Timestamp seg_start, Timestamp) {
    return 1 + static_cast<int>((seg_start / kKnotDt) % 3);
  };
  const Timestamp t_end = 900'000'000;

  SplineWindow a(kKnotDt, 4, /*non_uniform=*/true);
  a.initialize(0, traj.at(0));
  a.extendTo(t_end, seed, density);

  SplineWindow b(kKnotDt, 4, /*non_uniform=*/true);
  b.initialize(0, traj.at(0));
  b.extendTo(t_end, seed, density);

  const Timestamp cut = b.segmentStartContaining(500'000'000);
  const int n_before = b.numKnots();
  const int removed = b.truncateTailSegments(cut);
  EXPECT_GT(removed, 0);
  EXPECT_EQ(b.numKnots(), n_before - removed);
  EXPECT_EQ(b.intervalBasisCount(), b.numKnots() - 3);
  // The newest surviving knot sits exactly on the cut boundary, so re-extension
  // rebuilds the same segment schedule.
  EXPECT_EQ(b.knotTime(b.numKnots() - 1), cut);
  EXPECT_EQ(b.maxTime(), b.knotTime(b.numKnots() - 3) - 1);

  b.extendTo(t_end, seed, density);

  ASSERT_EQ(a.numKnots(), b.numKnots());
  EXPECT_EQ(a.maxTime(), b.maxTime());
  EXPECT_EQ(a.intervalBasisCount(), b.intervalBasisCount());
  for (int i = 0; i < a.numKnots(); ++i) {
    EXPECT_EQ(a.knotTime(i), b.knotTime(i)) << "knot " << i;
    EXPECT_EQ(std::memcmp(a.so3KnotData(i), b.so3KnotData(i), 4 * sizeof(double)), 0)
        << "so3 knot " << i;
    EXPECT_EQ(std::memcmp(a.r3KnotData(i), b.r3KnotData(i), 3 * sizeof(double)), 0)
        << "r3 knot " << i;
  }
  for (Timestamp t = 0; t < a.maxTime(); t += 23'000'000) {
    expectPoseBitEqual(a.pose(t), b.pose(t), "truncate+re-extend pose");
    SplineWindow::SegmentRef ra = a.segmentFor(t);
    SplineWindow::SegmentRef rb = b.segmentFor(t);
    // A uniform interval routes to cardinal on both windows (null pointers); only a
    // genuinely non-uniform interval carries matrices to compare byte-for-byte.
    EXPECT_EQ(ra.blend == nullptr, rb.blend == nullptr);
    if (ra.blend != nullptr && rb.blend != nullptr) {
      EXPECT_EQ(std::memcmp(ra.blend->data(), rb.blend->data(), 16 * sizeof(double)), 0);
      EXPECT_EQ(std::memcmp(ra.cum_blend->data(), rb.cum_blend->data(), 16 * sizeof(double)), 0);
    }
  }
}

// Knot and matrix pointers for surviving intervals stay valid and unchanged across
// a tail truncation and a subsequent re-extension, and evaluations at surviving
// times are bit-stable.
TEST(SplineWindowNonUniform, PointersStableAcrossTruncation) {
  const SmoothTrajectory traj(717);
  auto seed = [&](Timestamp t) { return traj.at(t); };
  SplineWindow w(kKnotDt, 4, /*non_uniform=*/true);
  w.initialize(0, traj.at(0));
  w.extendTo(800'000'000, seed, cyclingDensity({2, 3, 1}));

  const Timestamp t_early = 180'000'000;
  SplineWindow::SegmentRef before = w.segmentFor(t_early);
  const Pose pose_before = w.pose(t_early);

  const Timestamp cut = w.segmentStartContaining(550'000'000);
  const int removed = w.truncateTailSegments(cut);
  EXPECT_GT(removed, 0);

  SplineWindow::SegmentRef after = w.segmentFor(t_early);
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(after.so3_knots[j], before.so3_knots[j]);
    EXPECT_EQ(after.r3_knots[j], before.r3_knots[j]);
  }
  EXPECT_EQ(after.blend, before.blend);
  EXPECT_EQ(after.cum_blend, before.cum_blend);
  expectPoseBitEqual(w.pose(t_early), pose_before, "post-truncate pose");

  w.extendTo(800'000'000, seed, cyclingDensity({1, 2}));
  SplineWindow::SegmentRef again = w.segmentFor(t_early);
  EXPECT_EQ(again.blend, before.blend);
  expectPoseBitEqual(w.pose(t_early), pose_before, "post-re-extend pose");
}

// clone() of a non-uniform window evaluates bit-identically, owns an equal-size
// matrix cache in independent storage, and hands out distinct pointers.
TEST(SplineWindowNonUniform, CloneIsBitIdentical) {
  const SmoothTrajectory traj(828);
  SplineWindow w = makeNonUniformWindow(traj, 0, 600'000'000, cyclingDensity({3, 1, 2}));
  w.dropOldest(2);

  const std::unique_ptr<SplineWindow> c = w.clone();
  ASSERT_EQ(c->numKnots(), w.numKnots());
  EXPECT_EQ(c->minTime(), w.minTime());
  EXPECT_EQ(c->maxTime(), w.maxTime());
  ASSERT_EQ(c->intervalBasisCount(), w.intervalBasisCount());

  for (Timestamp t = w.minTime(); t < w.maxTime(); t += 11'000'000) {
    expectPoseBitEqual(w.pose(t), c->pose(t), "clone pose");
    const Eigen::Vector3d dv = w.linearVelocityWorld(t) - c->linearVelocityWorld(t);
    EXPECT_EQ(dv.norm(), 0.0);

    SplineWindow::SegmentRef rw = w.segmentFor(t);
    SplineWindow::SegmentRef rc = c->segmentFor(t);
    expectBitEqual(rw.u, rc.u, "clone u");
    expectBitEqual(rw.dt_s, rc.dt_s, "clone dt_s");
    ASSERT_NE(rc.blend, nullptr);
    EXPECT_NE(rc.blend, rw.blend);  // independent storage
    EXPECT_EQ(std::memcmp(rw.blend->data(), rc.blend->data(), 16 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(rw.cum_blend->data(), rc.cum_blend->data(), 16 * sizeof(double)), 0);
    for (int j = 0; j < 4; ++j) {
      EXPECT_NE(rc.so3_knots[j], rw.so3_knots[j]);
    }
  }
}

// The density-function extendTo overload queries the density once per appended outer
// segment (clamped to [1, n_cp_max]), places knots exactly like the fixed-count
// overload, seeds each knot one local step back, and the segment queries report the
// recorded boundaries and gear -- with the nominal-cadence fallback before the first
// record.
TEST(SplineWindow, DensityFunctionExtensionAndSegmentQueries) {
  const SmoothTrajectory traj(929);
  const Timestamp t0 = 100'000'000;
  const Timestamp t_target = t0 + 460'000'000;
  const std::vector<int> gears = {2, 1, 7, 3, 4, 1, 2, 3, 1, 2, 4, 1};  // 7 clamps to 4

  std::vector<Timestamp> seed_queries;
  auto seed = [&](Timestamp t) {
    seed_queries.push_back(t);
    return traj.at(t);
  };
  std::size_t gi = 0;
  auto density = [&](Timestamp, Timestamp) {
    const int g = gears[std::min(gi, gears.size() - 1)];
    ++gi;
    return g;
  };

  SplineWindow w(kKnotDt, 4, /*non_uniform=*/true);
  w.initialize(t0, traj.at(t0));
  w.extendTo(t_target, seed, density);

  // Reconstruct the expected knot times and seed queries with the same loop
  // semantics: whole segments are appended until coverage (which trails the newest
  // knot by three knot intervals) reaches the target.
  std::vector<Timestamp> kts;
  for (int j = 0; j < 4; ++j) {
    kts.push_back(t0 + j * kKnotDt);
  }
  std::vector<Timestamp> expect_seeds;
  std::vector<int> used_gears;
  std::size_t g2 = 0;
  auto coverage = [&kts] { return kts[kts.size() - 3] - 1; };
  while (coverage() < t_target) {
    const Timestamp seg_start = kts.back();
    const int cp = std::clamp(gears[std::min(g2, gears.size() - 1)], 1, 4);
    ++g2;
    used_gears.push_back(cp);
    const Timestamp step = kKnotDt / cp;
    for (int j = 1; j <= cp; ++j) {
      const Timestamp kr = (j == cp) ? seg_start + kKnotDt : seg_start + j * step;
      kts.push_back(kr);
      expect_seeds.push_back(kr - step);
    }
  }

  ASSERT_EQ(w.numKnots(), static_cast<int>(kts.size()));
  for (int i = 0; i < w.numKnots(); ++i) {
    EXPECT_EQ(w.knotTime(i), kts[static_cast<std::size_t>(i)]) << "knot " << i;
  }
  ASSERT_EQ(seed_queries.size(), expect_seeds.size());
  for (std::size_t i = 0; i < expect_seeds.size(); ++i) {
    EXPECT_EQ(seed_queries[i], expect_seeds[i]) << "seed query " << i;
  }

  // Segment queries: the first appended segment opens at the last bootstrap knot.
  Timestamp s = t0 + 3 * kKnotDt;
  for (const int g : used_gears) {
    EXPECT_EQ(w.segmentStartContaining(s + 1), s);
    EXPECT_EQ(w.segmentEndContaining(s + 1), s + kKnotDt);
    EXPECT_EQ(w.segmentGearContaining(s + 1), g);
    s += kKnotDt;
  }
  // Bootstrap fallback: no record covers the seed knots; the answer is the knot
  // interval, which spans one nominal cadence at gear 1.
  EXPECT_EQ(w.segmentStartContaining(t0 + 1), t0);
  EXPECT_EQ(w.segmentEndContaining(t0 + 1), t0 + kKnotDt);
  EXPECT_EQ(w.segmentGearContaining(t0 + 1), 1);
}

// highestKnotIndexOf mirrors lowestKnotIndexOf: highest deque index backing any of
// the given pointers, -1 for none.
TEST(SplineWindow, HighestKnotIndexOf) {
  const SmoothTrajectory traj(515);
  SplineWindow w = makeWindow(traj, 0, 400'000'000, 2);

  EXPECT_EQ(w.highestKnotIndexOf({}), -1);
  const double foreign = 0.0;
  EXPECT_EQ(w.highestKnotIndexOf({&foreign}), -1);

  std::vector<const double*> ptrs = {w.so3KnotData(2), w.r3KnotData(5)};
  EXPECT_EQ(w.highestKnotIndexOf(ptrs), 5);
  EXPECT_EQ(w.lowestKnotIndexOf(ptrs), 2);

  ptrs.push_back(w.r3KnotData(w.numKnots() - 1));
  EXPECT_EQ(w.highestKnotIndexOf(ptrs), w.numKnots() - 1);
}

// truncateTailSegments never removes the four oldest knots: a cut that would leave
// fewer than the cubic support keeps the offending segment whole.
TEST(SplineWindow, TruncateKeepsCubicSupport) {
  const SmoothTrajectory traj(404);
  SplineWindow w(kKnotDt, 4, /*non_uniform=*/true);
  w.initialize(0, traj.at(0));
  auto seed = [&](Timestamp t) { return traj.at(t); };
  w.extendTo(250'000'000, seed, cyclingDensity({2}));
  const int n = w.numKnots();

  // Cutting from before every record start removes whole segments from the back
  // while at least four knots survive; with 4 bootstrap knots + 2 knots per segment
  // every appended segment is removable, leaving exactly the bootstrap.
  const int removed = w.truncateTailSegments(0);
  EXPECT_EQ(w.numKnots(), 4);
  EXPECT_EQ(w.numKnots() + removed, n);
  EXPECT_EQ(w.intervalBasisCount(), w.numKnots() - 3);
}
