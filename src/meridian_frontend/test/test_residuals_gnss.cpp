#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <sophus/so3.hpp>
#include <vector>

#include "ct/residuals_gnss.hpp"
#include "ct/spline_window.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

using meridian::Duration;
using meridian::FrontendGnss;
using meridian::GnssFix;
using meridian::Pose;
using meridian::SplineWindow;
using meridian::Timestamp;
using meridian::ct::addGnssResidual;
using meridian::ct::EnuAnchor;
using meridian::ct::fixTypeFloorStd;
using meridian::ct::flooredCovEnu;
using meridian::ct::GnssGate;
using meridian::ct::GnssGateResult;

namespace {

double tSec(Timestamp t) {
  return meridian::to_seconds(t);
}

FrontendGnss makeCfg() {
  return FrontendGnss{};
}

GnssFix makeFix(double lat, double lon, double alt, GnssFix::FixType type) {
  GnssFix f;
  f.lat_deg = lat;
  f.lon_deg = lon;
  f.alt_m = alt;
  f.fix = type;
  f.cov_enu = Eigen::Matrix3d::Identity() * 1e-4;  // optimistic 1 cm std, floored upward
  return f;
}

// A smooth analytic body-frame trajectory: a gentle translation + a slow turn, used to
// seed the spline so the GNSS residual evaluates against a non-trivial interpolated pose.
struct GroundTruth {
  Eigen::Vector3d w_world{0.1, -0.05, 0.2};
  double ax = 0.8, ay = 0.6, az = 0.3;
  double fx = 0.7, fy = 0.5, fz = 0.6;

  Eigen::Vector3d position(double t) const {
    return Eigen::Vector3d(ax * std::sin(fx * t), ay * std::cos(fy * t), az * std::sin(fz * t));
  }
  Pose pose(double t) const {
    Pose p;
    p.q = Sophus::SO3d::exp(w_world * t).unit_quaternion();
    p.t = position(t);
    return p;
  }
};

// A uniform spline (n_cp == 1) seeded from the ground truth, covering a few knots past
// t_end so segmentFor() at any in-range fix time has full 4-knot support.
SplineWindow makeSpline(const GroundTruth& gt, Timestamp t0, Timestamp t_end, Duration knot_dt) {
  SplineWindow spline(knot_dt, 1);
  spline.initialize(t0, gt.pose(tSec(t0)));
  const auto seed = [&](Timestamp t) { return gt.pose(tSec(t)); };
  const Timestamp target = t_end + 4 * knot_dt;
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    spline.extendTo(t, seed, 1);
  }
  return spline;
}

}  // namespace

// Fix-type floor selection: each quality maps to the spec's (horizontal, vertical) std
// pair, and a None-quality fix has a zero floor (it is rejected upstream).
TEST(GnssResidual, FixTypeFloorSelection) {
  const FrontendGnss cfg = makeCfg();
  EXPECT_EQ(fixTypeFloorStd(GnssFix::FixType::RTK_Fixed, cfg), Eigen::Vector2d(0.05, 0.10));
  EXPECT_EQ(fixTypeFloorStd(GnssFix::FixType::RTK_Float, cfg), Eigen::Vector2d(0.50, 1.00));
  EXPECT_EQ(fixTypeFloorStd(GnssFix::FixType::DGPS, cfg), Eigen::Vector2d(1.50, 3.00));
  EXPECT_EQ(fixTypeFloorStd(GnssFix::FixType::SPP, cfg), Eigen::Vector2d(3.00, 6.00));
  EXPECT_EQ(fixTypeFloorStd(GnssFix::FixType::None, cfg), Eigen::Vector2d(0.0, 0.0));
}

// The floored covariance raises an optimistic reported covariance to at least the
// per-fix-type floor variance on the diagonal; a reported covariance already above the
// floor passes through.
TEST(GnssResidual, FloorRaisesOptimisticCovariance) {
  const FrontendGnss cfg = makeCfg();
  // SPP floor is (3, 6) m std -> (9, 9, 36) m^2 variance.
  const GnssFix spp = makeFix(0, 0, 0, GnssFix::FixType::SPP);
  const Eigen::Matrix3d cov = flooredCovEnu(spp, cfg);
  EXPECT_DOUBLE_EQ(cov(0, 0), 9.0);
  EXPECT_DOUBLE_EQ(cov(1, 1), 9.0);
  EXPECT_DOUBLE_EQ(cov(2, 2), 36.0);

  // A pessimistic reported covariance above the RTK_Fixed floor is not lowered.
  GnssFix rtk = makeFix(0, 0, 0, GnssFix::FixType::RTK_Fixed);
  rtk.cov_enu = Eigen::Matrix3d::Identity();  // 1 m^2 >> (0.05, 0.10) floor
  const Eigen::Matrix3d cov2 = flooredCovEnu(rtk, cfg);
  EXPECT_DOUBLE_EQ(cov2(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(cov2(1, 1), 1.0);
  EXPECT_DOUBLE_EQ(cov2(2, 2), 1.0);
}

// The ENU anchor maps the first fix to its supplied world position exactly, and a later
// fix to a metric ENU offset: a +1 arc-second longitude/latitude step lands near the
// expected metres-per-degree scale, and an altitude delta maps straight to up.
TEST(GnssResidual, EnuAnchorConversion) {
  EnuAnchor anchor;
  EXPECT_FALSE(anchor.set());

  const double lat0 = 22.30;  // a mid-latitude site (Hong Kong-ish, FusionPortable)
  const double lon0 = 114.18;
  const double alt0 = 30.0;
  const GnssFix first = makeFix(lat0, lon0, alt0, GnssFix::FixType::RTK_Fixed);
  const Eigen::Vector3d world_at_first(5.0, -2.0, 1.0);
  anchor.anchor(first, world_at_first);
  ASSERT_TRUE(anchor.set());

  // The anchor fix maps exactly to its supplied world position (raw ENU == 0).
  EXPECT_LT((anchor.toWorld(first) - world_at_first).norm(), 1e-9);

  // A fix one millidegree east and north, 3 m higher.
  const GnssFix moved = makeFix(lat0 + 1e-3, lon0 + 1e-3, alt0 + 3.0, GnssFix::FixType::RTK_Fixed);
  const Eigen::Vector3d w = anchor.toWorld(moved);
  const Eigen::Vector3d enu = w - world_at_first;  // east, north, up offset from anchor
  // One millidegree of latitude is ~111 m; longitude ~111*cos(lat) ~ 102.8 m at 22.3 deg.
  EXPECT_NEAR(enu.x(), 1e-3 * 111319.49 * std::cos(lat0 * M_PI / 180.0), 5.0);  // east
  EXPECT_NEAR(enu.y(), 1e-3 * 110000.0, 2000.0);                                // north (approx)
  EXPECT_DOUBLE_EQ(enu.z(), 3.0);                                               // up exact
}

// The ENU->W tie is the conservative L2 convention: a translation only, with the ENU
// axes identified with W's (E->x, N->y, U->z) and no heading rotation. A fix displaced
// purely toward true geographic East therefore lands along W's +x axis, and one purely
// North along +y, with the cross-axis component zero. The real ENU->W heading is the L3
// datum's job; pinning the identity heading here makes that future dependence explicit,
// so an L3 datum that introduces a yaw is a deliberate, visible change rather than a
// silent regression masked by E/N already coinciding with x/y.
TEST(GnssResidual, EnuToWorldIsTranslationOnlyNoHeading) {
  EnuAnchor anchor;
  const double lat0 = 22.30;
  const double lon0 = 114.18;
  const double alt0 = 30.0;
  const GnssFix first = makeFix(lat0, lon0, alt0, GnssFix::FixType::RTK_Fixed);
  const Eigen::Vector3d world_at_first(5.0, -2.0, 1.0);
  anchor.anchor(first, world_at_first);
  ASSERT_TRUE(anchor.set());

  // A fix one millidegree due East (longitude only): its world delta from the anchor
  // must lie on W's x-axis -- non-zero x, zero y and z -- if and only if E coincides
  // with W.x (the identity-heading convention).
  const GnssFix east_only = makeFix(lat0, lon0 + 1e-3, alt0, GnssFix::FixType::RTK_Fixed);
  const Eigen::Vector3d d_east = anchor.toWorld(east_only) - world_at_first;
  EXPECT_GT(d_east.x(), 1.0);          // a millidegree of longitude is ~100 m
  EXPECT_NEAR(d_east.y(), 0.0, 1e-9);  // no North component leaks into a due-East fix
  EXPECT_NEAR(d_east.z(), 0.0, 1e-9);

  // A fix one millidegree due North (latitude only): world delta on W's y-axis only.
  const GnssFix north_only = makeFix(lat0 + 1e-3, lon0, alt0, GnssFix::FixType::RTK_Fixed);
  const Eigen::Vector3d d_north = anchor.toWorld(north_only) - world_at_first;
  EXPECT_NEAR(d_north.x(), 0.0, 1e-9);
  EXPECT_GT(d_north.y(), 1.0);
  EXPECT_NEAR(d_north.z(), 0.0, 1e-9);
}

// The innovation gate rejects an outlier fix whose normalised innovation exceeds k, and
// accepts an in-gate fix once the re-acquisition run completes. The accept/reject counts
// prove the gate fired.
TEST(GnssResidual, InnovationGateRejectsOutlier) {
  FrontendGnss cfg = makeCfg();
  cfg.innovation_k = 3.0;
  cfg.reacquire_count = 1;  // admit immediately so a single outlier is the only reject
  GnssGate gate(cfg);

  // RTK_Fixed floor variance ~ (0.05^2, 0.05^2, 0.10^2); no extra pose marginal.
  const Eigen::Matrix3d cov = (Eigen::Vector3d(0.0025, 0.0025, 0.01)).asDiagonal();
  const Eigen::Matrix3d no_pose = Eigen::Matrix3d::Zero();

  // A 1 cm residual on a 5 cm-std fix is ~0.2 sigma -> in gate, admitted.
  const GnssGateResult good = gate.gate(Eigen::Vector3d(0.01, 0.0, 0.0), cov, no_pose);
  EXPECT_TRUE(good.accepted);
  EXPECT_NEAR(good.innovation_m, 0.01, 1e-9);

  // A 1 m residual on a 5 cm-std fix is 20 sigma -> rejected by the gate.
  const GnssGateResult bad = gate.gate(Eigen::Vector3d(1.0, 0.0, 0.0), cov, no_pose);
  EXPECT_FALSE(bad.accepted);
  EXPECT_EQ(bad.reason, GnssGateResult::Reason::Gate);

  EXPECT_EQ(gate.accepted(), 1);
  EXPECT_EQ(gate.rejected(), 1);
}

// Re-acquisition persistence: after a gap (a gated-out fix disarms the gate), GNSS is
// re-admitted only once reacquire_count consecutive in-gate fixes pass. Earlier in-gate
// fixes in the run are held out (counted as reacq_persist rejects), not admitted.
TEST(GnssResidual, ReacquirePersistence) {
  FrontendGnss cfg = makeCfg();
  cfg.innovation_k = 3.0;
  cfg.reacquire_count = 5;
  GnssGate gate(cfg);

  const Eigen::Matrix3d cov = (Eigen::Vector3d(0.0025, 0.0025, 0.01)).asDiagonal();
  const Eigen::Matrix3d no_pose = Eigen::Matrix3d::Zero();
  const Eigen::Vector3d small(0.01, 0.0, 0.0);  // well in gate

  // First four in-gate fixes are held out by persistence; the fifth arms and admits.
  for (int i = 0; i < 4; ++i) {
    const GnssGateResult r = gate.gate(small, cov, no_pose);
    EXPECT_FALSE(r.accepted) << "fix " << i << " should be held by persistence";
    EXPECT_EQ(r.reason, GnssGateResult::Reason::ReacquirePersist);
  }
  EXPECT_TRUE(gate.gate(small, cov, no_pose).accepted) << "fifth in-gate fix arms the gate";
  // Once armed, subsequent in-gate fixes admit immediately.
  EXPECT_TRUE(gate.gate(small, cov, no_pose).accepted);

  // An outlier disarms the gate; the run must restart from scratch.
  EXPECT_FALSE(gate.gate(Eigen::Vector3d(1.0, 0.0, 0.0), cov, no_pose).accepted);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(gate.gate(small, cov, no_pose).accepted) << "re-arming run, fix " << i;
  }
  EXPECT_TRUE(gate.gate(small, cov, no_pose).accepted) << "re-armed after a fresh run";
}

// Lever-arm correctness: with the antenna offset from the body, the residual at the true
// fix is zero ONLY when the lever-arm is applied. A fix placed at the antenna world
// position but evaluated with a zero lever-arm leaves a residual equal to the lever-arm's
// rotated length; the correct lever-arm drives it to zero.
TEST(GnssResidual, LeverArmCorrectness) {
  const GroundTruth gt;
  const Duration knot_dt = 25'000'000;  // 25 ms
  const Timestamp t0 = 0;
  const Timestamp t_end = 200'000'000;  // 200 ms
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);

  const Timestamp t_fix = 100'000'000;  // 100 ms, mid-window
  ASSERT_TRUE(spline.covers(t_fix));
  const Pose T_w_fe = spline.pose(t_fix);

  const Eigen::Vector3d lever(0.30, -0.10, 0.20);  // antenna 30 cm forward of the IMU
  // Place the fix exactly at the antenna world position implied by the lever-arm.
  const Eigen::Vector3d z_with_lever = T_w_fe * lever;
  const Eigen::Vector3d z_no_lever = T_w_fe.t;  // antenna assumed at the body origin

  const Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  // Helper: evaluate the residual for a given (z, lever) by adding it to a problem and
  // reading the cost (Ceres residual stored in the problem's evaluation).
  const auto residualNorm = [&](const Eigen::Vector3d& z, const Eigen::Vector3d& l) {
    ceres::Problem problem;
    const bool added = addGnssResidual(problem, spline, z, l, t_fix, sqrt_info, /*huber=*/1e9);
    EXPECT_TRUE(added);
    // Freeze every knot so Evaluate reports the residual at the seeded pose.
    std::vector<double*> blocks;
    problem.GetParameterBlocks(&blocks);
    for (double* b : blocks) {
      if (problem.ParameterBlockSize(b) == 4 && problem.GetManifold(b) == nullptr) {
        problem.SetManifold(b, new ceres::EigenQuaternionManifold());
      }
    }
    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    return std::sqrt(2.0 * cost);  // cost = 0.5 ||r||^2
  };

  // Correct lever-arm at the lever-placed fix -> residual ~ 0.
  EXPECT_LT(residualNorm(z_with_lever, lever), 1e-6);
  // Same fix with a zero lever-arm -> residual ~ the rotated lever length (non-zero).
  const double r_no_lever = residualNorm(z_with_lever, Eigen::Vector3d::Zero());
  EXPECT_NEAR(r_no_lever, lever.norm(), 1e-6);
  // A body-origin fix with a zero lever-arm -> residual ~ 0 (consistency check).
  EXPECT_LT(residualNorm(z_no_lever, Eigen::Vector3d::Zero()), 1e-6);
}

// The residual is bound to the spline pose INTERPOLATED at the fix time, not a nearby
// knot: shifting the fix time by a fraction of a knot interval changes the residual,
// proving continuous-time evaluation rather than nearest-keyframe snapping.
TEST(GnssResidual, BindsToInterpolatedFixTime) {
  const GroundTruth gt;
  const Duration knot_dt = 25'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 200'000'000;
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);

  const Eigen::Vector3d lever(0.2, 0.0, 0.0);
  const Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  // A fix placed at the antenna world position at t_a; evaluated at t_a it is zero, at a
  // 10 ms-later time the interpolated pose differs and the residual is non-zero.
  const Timestamp t_a = 90'000'000;
  const Timestamp t_b = 100'000'000;
  const Eigen::Vector3d z = spline.pose(t_a) * lever;

  const auto residualNorm = [&](Timestamp t) {
    ceres::Problem problem;
    addGnssResidual(problem, spline, z, lever, t, sqrt_info, 1e9);
    std::vector<double*> blocks;
    problem.GetParameterBlocks(&blocks);
    for (double* b : blocks) {
      if (problem.ParameterBlockSize(b) == 4 && problem.GetManifold(b) == nullptr) {
        problem.SetManifold(b, new ceres::EigenQuaternionManifold());
      }
    }
    double cost = 0.0;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    return std::sqrt(2.0 * cost);
  };

  EXPECT_LT(residualNorm(t_a), 1e-6) << "zero at the fix's true time";
  EXPECT_GT(residualNorm(t_b), 1e-3) << "non-zero at a shifted time -> interpolated, not snapped";
}
