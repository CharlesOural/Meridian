#include <basalt/spline/ceres_spline_helper.h>
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <random>
#include <sophus/so3.hpp>
#include <vector>

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

// The first real time at which a window's support is shaped purely by measurement
// knots (the N-1 seed knots placed at and after t0 have cleared the support).
Timestamp measurementStart(Timestamp t0) {
  return t0 + 5 * kKnotDt;
}

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
