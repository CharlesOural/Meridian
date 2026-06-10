#include "ct/residuals_imu.hpp"

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <basalt/spline/ceres_spline_helper.h>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <ceres/sphere_manifold.h>
#include <gtest/gtest.h>
#include <sophus/so3.hpp>

#include "ct/spline_window.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

using meridian::Duration;
using meridian::ImuSample;
using meridian::Pose;
using meridian::SplineWindow;
using meridian::Timestamp;
using meridian::ct::addBiasRandomWalk;
using meridian::ct::addImuResiduals;
using meridian::ct::addMotionRegularizer;
using meridian::ct::BiasKnots;
using meridian::ct::GravityBlock;
using meridian::ct::ImuExcitation;
using meridian::ct::imuExcitation;
using meridian::ct::ImuWeights;
using meridian::ct::knotDensityFromExcitation;
using meridian::ct::makeAngularAccelCost;
using meridian::ct::makeAngularAccelCostAutodiff;
using meridian::ct::makeBiasTieCost;
using meridian::ct::makeBiasTieCostAutodiff;
using meridian::ct::makeImuCost;
using meridian::ct::makeImuCostAutodiff;
using meridian::ct::makeJerkCost;
using meridian::ct::makeJerkCostAutodiff;
using meridian::ct::makeRateAnchorCost;
using meridian::ct::makeRateAnchorCostAutodiff;
using meridian::ct::makeVelocityAnchorCost;
using meridian::ct::makeVelocityAnchorCostAutodiff;

namespace {

constexpr double kGravityMag = 9.81;
const Eigen::Vector3d kGravityWorld(0.0, 0.0, -kGravityMag);

// A smooth analytic ground-truth trajectory: helical translation plus a steady
// turn. Closed-form pose, body rate, and world acceleration let us synthesise a
// perfect IMU and check the residual vanishes there.
struct GroundTruth {
  // Angular rates about each world axis [rad/s] used to drive the orientation.
  Eigen::Vector3d w_world{0.3, -0.2, 0.5};
  double tx_amp = 0.7, ty_amp = 0.5, tz_rate = 0.4;
  double tx_freq = 1.1, ty_freq = 0.9;

  Sophus::SO3d rotation(double t) const {
    return Sophus::SO3d::exp(w_world * t);
  }

  Eigen::Vector3d position(double t) const {
    return Eigen::Vector3d(tx_amp * std::sin(tx_freq * t),
                           ty_amp * std::cos(ty_freq * t), tz_rate * t);
  }

  Eigen::Vector3d accelWorld(double t) const {
    return Eigen::Vector3d(-tx_amp * tx_freq * tx_freq * std::sin(tx_freq * t),
                           -ty_amp * ty_freq * ty_freq * std::cos(ty_freq * t),
                           0.0);
  }

  Pose pose(double t) const {
    Pose p;
    p.q = rotation(t).unit_quaternion();
    p.t = position(t);
    return p;
  }

  // Body angular velocity. With R(t) = exp(w_world * t) about a fixed axis, the
  // increment R(t)^{-1} R(t+dt) = exp(w_world * dt) commutes, so the body rate that
  // satisfies R(t+dt) = R(t) exp(omega * dt) is the constant w_world itself.
  Eigen::Vector3d omegaBody(double) const { return w_world; }

  // The accelerometer's specific force in the body frame for a perfect, unbiased
  // sensor: R^T (a_world - g_world).
  Eigen::Vector3d accelMeasured(double t) const {
    return rotation(t).inverse() * (accelWorld(t) - kGravityWorld);
  }
};

double tSec(Timestamp t) { return meridian::to_seconds(t); }

// Builds a uniform spline (n_cp == 1 per segment, virtual time == real time) that
// follows the ground truth, seeded exactly from the analytic poses.
SplineWindow makeSpline(const GroundTruth& gt, Timestamp t0, Timestamp t_end,
                        Duration knot_dt) {
  SplineWindow spline(knot_dt, 1);
  spline.initialize(t0, gt.pose(tSec(t0)));
  const auto seed = [&](Timestamp t) { return gt.pose(tSec(t)); };
  // Extend segment by segment to a few knots past t_end so every IMU sample in
  // [t0, t_end] is strictly inside the covered horizon.
  const Timestamp target = t_end + 4 * knot_dt;
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    spline.extendTo(t, seed, 1);
  }
  return spline;
}

// Builds a non-uniform spline whose outer segments each carry n_cp control points
// (n_cp > 1 packs them denser than the nominal cadence, so a real segment maps onto
// n_cp virtual knot intervals and the per-segment slope dv/dt is n_cp). Exercises
// the real->virtual remap the uniform path leaves at slope == 1.
SplineWindow makeSplineDense(const GroundTruth& gt, Timestamp t0, Timestamp t_end,
                             Duration knot_dt, int n_cp) {
  SplineWindow spline(knot_dt, n_cp);
  spline.initialize(t0, gt.pose(tSec(t0)));
  const auto seed = [&](Timestamp t) { return gt.pose(tSec(t)); };
  const Timestamp target = t_end + 4 * knot_dt;
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    spline.extendTo(t, seed, n_cp);
  }
  return spline;
}

std::vector<ImuSample> makeImu(const GroundTruth& gt, Timestamp t0, Timestamp t_end,
                               Duration dt) {
  std::vector<ImuSample> out;
  for (Timestamp t = t0; t <= t_end; t += dt) {
    ImuSample s;
    s.stamp = t;
    s.gyro = gt.omegaBody(tSec(t));
    s.acc = gt.accelMeasured(tSec(t));
    out.push_back(s);
  }
  return out;
}

ImuWeights unitWeights() {
  ImuWeights w;
  w.sigma_gyro = 1.0;
  w.sigma_accel = 1.0;
  w.sigma_bias_gyro = 1.0;
  w.sigma_bias_accel = 1.0;
  return w;
}

// Weak absolute anchor on a 3-vector bias block toward a reference value. The
// bias common-mode is unobservable from IMU residuals alone (a constant gyro bias
// trades against a constant body-rate trend, a constant accel bias against the
// trajectory), so a prior is needed to make it identifiable.
struct BiasAnchor {
  Eigen::Vector3d ref;
  double weight;
  template <class T>
  bool operator()(const T* x, T* r) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> v(x);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> res(r);
    res = T(weight) * (v - ref.cast<T>());
    return true;
  }
};

void addBiasAnchor(ceres::Problem& problem, double* block,
                   const Eigen::Vector3d& ref, double weight) {
  problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<BiasAnchor, 3, 3>(
          new BiasAnchor{ref, weight}),
      nullptr, block);
}

// The SO(3) knots are stored as raw (x,y,z,w) quaternions; without a manifold the
// solver would drift them off the unit sphere. Walk the whole timeline and put an
// EigenQuaternionManifold on every SO(3) knot block the problem holds.
void constrainSo3Knots(ceres::Problem& problem, SplineWindow& spline,
                       Duration knot_dt) {
  for (Timestamp t = spline.minTime();; t += knot_dt) {
    SplineWindow::SegmentRef seg =
        spline.segmentFor(std::min(t, spline.maxTime() - 1));
    for (double* q : seg.so3_knots) {
      if (problem.HasParameterBlock(q) &&
          problem.GetManifold(q) == nullptr) {
        problem.SetManifold(q, new ceres::EigenQuaternionManifold());
      }
    }
    if (t >= spline.maxTime() - 1) {
      break;
    }
  }
}

// Reconstructs the knot times the builders above produce: four nominal-cadence
// knots from initialize(), then cp knots per appended segment with the last landing
// exactly on the segment end. Callers assert the count against numKnots() so any
// drift from the builder logic fails loudly.
std::vector<Timestamp> knotTimes(Timestamp t0, Timestamp t_end, Duration knot_dt,
                                 int cp) {
  std::vector<Timestamp> kts;
  for (int j = 0; j < 4; ++j) {
    kts.push_back(t0 + j * knot_dt);
  }
  const Timestamp target = t_end + 4 * knot_dt;
  // Coverage trails the newest knot by three knot intervals, mirroring extendTo.
  const auto coverage = [&kts] { return kts[kts.size() - 3] - 1; };
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    while (coverage() < t) {
      const Timestamp seg_start = kts.back();
      const Timestamp step = knot_dt / cp;
      for (int j = 1; j <= cp; ++j) {
        kts.push_back(j == cp ? seg_start + knot_dt : seg_start + j * step);
      }
    }
  }
  return kts;
}

// The control point with grid time t_j dominates the curve one knot interval
// earlier: at the start of interval i the spline evaluates to
// (CP_i + 4 CP_{i+1} + CP_{i+2})/6, so reproducing a curve p needs
// CP_j = p(t_j - h) - (h^2/6) p''(t_j - h) to second order. Seeding (and pinning)
// with this value keeps the anchor bias at O(h^4); a raw curve sample at t_j would
// time-shift the whole spline by one interval.
double cpTimeSec(const std::vector<Timestamp>& kts, std::size_t j) {
  const std::size_t k = (j + 1 < kts.size()) ? j : j - 1;
  const double h = tSec(kts[k + 1]) - tSec(kts[k]);
  return tSec(kts[j]) - h;
}

Eigen::Vector3d idealPositionCp(const GroundTruth& gt,
                                const std::vector<Timestamp>& kts, std::size_t j) {
  const std::size_t k = (j + 1 < kts.size()) ? j : j - 1;
  const double h = tSec(kts[k + 1]) - tSec(kts[k]);
  const double t = cpTimeSec(kts, j);
  return gt.position(t) - (h * h / 6.0) * gt.accelWorld(t);
}

// The IMU residuals are invariant under a constant left-rotation of the whole
// trajectory (with the matching remap of position and its gravity compensation) and
// under adding any affine function to the position spline: body rate is unchanged
// by a global rotation, and an affine position term has zero acceleration. An
// IMU-only fit therefore converges to the ground truth only up to that 9-DoF gauge.
// Seeding every knot on the analytic curve and pinning the first rotation knot plus
// the first two position knots inside the measured span removes the freedom. The
// rotation seeds are exact for this constant-rate ground truth; positions carry the
// second-order control-point correction.
void reseedToGroundTruth(SplineWindow& spline, const GroundTruth& gt,
                         const std::vector<Timestamp>& kts) {
  const int last = static_cast<int>(kts.size()) - 4;
  for (int i = 0; i <= last; ++i) {
    SplineWindow::SegmentRef seg = spline.segmentFor(kts[i]);
    for (int j = 0; j < 4; ++j) {
      const auto idx = static_cast<std::size_t>(i + j);
      Eigen::Map<Eigen::Quaterniond>(seg.so3_knots[j]) =
          gt.rotation(cpTimeSec(kts, idx)).unit_quaternion();
      Eigen::Map<Eigen::Vector3d>(seg.r3_knots[j]) = idealPositionCp(gt, kts, idx);
    }
  }
}

void pinGauge(ceres::Problem& problem, SplineWindow& spline, Timestamp t_anchor) {
  SplineWindow::SegmentRef seg = spline.segmentFor(t_anchor);
  for (double* block : {seg.so3_knots[0], seg.r3_knots[0], seg.r3_knots[1]}) {
    if (problem.HasParameterBlock(block)) {
      problem.SetParameterBlockConstant(block);
    }
  }
}

}  // namespace

// (a) A perfect IMU stream (omega from the analytic body rate, acc = R^T(a_w - g),
// zero bias) drives the residuals to ~0 at the trajectory the spline fits. Knots are
// seeded on the analytic curve and the gauge is pinned, so the fitted optimum must
// coincide with the ground truth and the residual there must be negligible.
TEST(ResidualsImu, ZeroResidualAtGroundTruth) {
  GroundTruth gt;
  const Duration knot_dt = 25'000'000;  // 25 ms
  const Timestamp t0 = 0;
  const Timestamp t_end = 300'000'000;  // 0.3 s window
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  BiasKnots bias(t0, knot_dt, 1);  // perfect IMU: zero bias, one constant knot
  GravityBlock gravity(kGravityWorld, kGravityMag);
  const auto imu = makeImu(gt, t0, t_end, 5'000'000);

  ceres::Problem problem;
  const int n = addImuResiduals(problem, spline, bias, gravity, imu, t0, t_end,
                                unitWeights());
  ASSERT_GT(n, 0);
  constrainSo3Knots(problem, spline, knot_dt);
  pinGauge(problem, spline, t0);
  problem.SetParameterBlockConstant(bias.gyroBlock(0));
  problem.SetParameterBlockConstant(bias.accelBlock(0));
  problem.SetParameterBlockConstant(gravity.data());

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 40;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  // Perfect, consistent measurements: the fitted residual must be tiny.
  EXPECT_LT(summary.final_cost / static_cast<double>(n), 1e-4);

  // The fitted spline reproduces the analytic kinematics the IMU was built from.
  for (Timestamp t = 40'000'000; t < t_end - 40'000'000; t += 40'000'000) {
    EXPECT_LT((spline.angularVelocityBody(t) - gt.omegaBody(tSec(t))).norm(), 1e-2)
        << "t=" << t;
    EXPECT_LT((spline.linearAccelWorld(t) - gt.accelWorld(tSec(t))).norm(), 5e-2)
        << "t=" << t;
  }
}

// (b) IMU-only window fit: perturb the knots, solve, and recover the trajectory and
// (zero) biases within tolerance.
TEST(ResidualsImu, RecoversTrajectoryFromPerturbedKnots) {
  GroundTruth gt;
  const Duration knot_dt = 50'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 300'000'000;
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  // Perturb every control point away from the ground-truth seed. Walking query
  // times across the window in half-knot steps touches every overlapping segment.
  std::mt19937 rng(12345);
  std::normal_distribution<double> noise(0.0, 0.05);
  for (Timestamp t = spline.minTime(); t < spline.maxTime(); t += knot_dt / 2) {
    SplineWindow::SegmentRef seg = spline.segmentFor(t);
    for (double* q : seg.so3_knots) {
      Eigen::Map<Eigen::Quaterniond> quat(q);
      const Eigen::Vector3d d(noise(rng), noise(rng), noise(rng));
      quat = (quat * Sophus::SO3d::exp(d).unit_quaternion()).normalized();
    }
    for (double* p : seg.r3_knots) {
      Eigen::Map<Eigen::Vector3d> v(p);
      v += Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
    }
  }

  // The gauge anchors must sit at their true values: the fit can only pull the
  // perturbed knots back relative to where the anchors hold the trajectory.
  {
    SplineWindow::SegmentRef g0 = spline.segmentFor(t0);
    Eigen::Map<Eigen::Quaterniond>(g0.so3_knots[0]) =
        gt.rotation(cpTimeSec(kts, 0)).unit_quaternion();
    Eigen::Map<Eigen::Vector3d>(g0.r3_knots[0]) = idealPositionCp(gt, kts, 0);
    Eigen::Map<Eigen::Vector3d>(g0.r3_knots[1]) = idealPositionCp(gt, kts, 1);
  }

  BiasKnots bias(t0, knot_dt, 2);  // let bias float, tie with a random walk
  GravityBlock gravity(kGravityWorld, kGravityMag);
  const auto imu = makeImu(gt, t0, t_end, 5'000'000);

  ceres::Problem problem;
  ASSERT_GT(addImuResiduals(problem, spline, bias, gravity, imu, t0, t_end,
                            unitWeights()),
            0);
  addBiasRandomWalk(problem, bias, unitWeights());
  constrainSo3Knots(problem, spline, knot_dt);
  pinGauge(problem, spline, t0);
  // Weakly anchor the (true zero) bias so its unobservable common-mode is pinned.
  for (int k = 0; k < bias.numKnots(); ++k) {
    addBiasAnchor(problem, bias.gyroBlock(k), Eigen::Vector3d::Zero(), 1.0);
    addBiasAnchor(problem, bias.accelBlock(k), Eigen::Vector3d::Zero(), 1.0);
  }
  // Hold gravity fixed for this pure-trajectory recovery.
  problem.SetParameterBlockConstant(gravity.data());

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  // After the fit the spline kinematics must again match the analytic IMU.
  for (Timestamp t = 30'000'000; t < t_end - 30'000'000; t += 40'000'000) {
    EXPECT_LT((spline.angularVelocityBody(t) - gt.omegaBody(tSec(t))).norm(), 5e-2)
        << "t=" << t;
    EXPECT_LT((spline.linearAccelWorld(t) - gt.accelWorld(tSec(t))).norm(), 5e-1)
        << "t=" << t;
  }
  EXPECT_LT(bias.gyroBiasAt(t_end / 2).norm(), 5e-2);
  EXPECT_LT(bias.accelBiasAt(t_end / 2).norm(), 5e-1);
}

// (c) Gravity magnitude stays constant through a solve under the SphereManifold.
TEST(ResidualsImu, GravityNormConstantThroughSolve) {
  GroundTruth gt;
  const Duration knot_dt = 50'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 300'000'000;
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);

  BiasKnots bias(t0, knot_dt, 1);
  // Start gravity meaningfully off-axis so the solver has to move the direction.
  GravityBlock gravity(Eigen::Vector3d(0.4, 0.2, -1.0), kGravityMag);
  const double norm_before = gravity.gravityWorld().norm();

  const auto imu = makeImu(gt, t0, t_end, 5'000'000);
  ceres::Problem problem;
  // addImuResiduals registers the SphereManifold on the gravity block itself.
  ASSERT_GT(addImuResiduals(problem, spline, bias, gravity, imu, t0, t_end,
                            unitWeights()),
            0);
  ASSERT_NE(problem.GetManifold(gravity.data()), nullptr);
  // Freeze the trajectory so all the correction flows into gravity + bias.
  for (Timestamp t = spline.minTime();; t += knot_dt) {
    SplineWindow::SegmentRef seg = spline.segmentFor(std::min(t, spline.maxTime() - 1));
    for (double* q : seg.so3_knots) {
      if (problem.HasParameterBlock(q)) problem.SetParameterBlockConstant(q);
    }
    for (double* p : seg.r3_knots) {
      if (problem.HasParameterBlock(p)) problem.SetParameterBlockConstant(p);
    }
    if (t >= spline.maxTime() - 1) break;
  }

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 30;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  const double norm_after = gravity.gravityWorld().norm();
  EXPECT_NEAR(norm_after, kGravityMag, 1e-6);
  EXPECT_NEAR(norm_after, norm_before, 1e-6);
  // The stored direction must remain a unit vector.
  EXPECT_NEAR(gravity.direction_.norm(), 1.0, 1e-9);
}

// (d) The bias random-walk tie pulls two split bias knots toward each other.
TEST(ResidualsImu, BiasRandomWalkTiePullsKnotsTogether) {
  const Duration knot_dt = 50'000'000;
  const Timestamp t0 = 0;
  BiasKnots bias(t0, knot_dt, 2);
  bias.setGyroKnot(0, Eigen::Vector3d(0.10, -0.05, 0.02));
  bias.setGyroKnot(1, Eigen::Vector3d(-0.08, 0.07, -0.03));
  bias.setAccelKnot(0, Eigen::Vector3d(0.20, 0.10, -0.15));
  bias.setAccelKnot(1, Eigen::Vector3d(-0.10, -0.20, 0.25));

  // Anchor each knot weakly to its initial value so the problem is well posed; the
  // strong random-walk tie should then collapse the gap between them.
  ceres::Problem problem;
  const int ties = addBiasRandomWalk(problem, bias, [] {
    ImuWeights w;
    w.sigma_bias_gyro = 0.01;   // strong tie
    w.sigma_bias_accel = 0.01;  // strong tie
    return w;
  }());
  ASSERT_EQ(ties, 1);

  const Eigen::Vector3d bg0_init = bias.gyroKnot(0);
  const Eigen::Vector3d bg1_init = bias.gyroKnot(1);
  const Eigen::Vector3d ba0_init = bias.accelKnot(0);
  const Eigen::Vector3d ba1_init = bias.accelKnot(1);

  // Weak absolute anchors keep both knots near their starting values; the much
  // stronger random-walk tie then collapses the gap between them.
  addBiasAnchor(problem, bias.gyroBlock(0), bg0_init, 0.1);
  addBiasAnchor(problem, bias.gyroBlock(1), bg1_init, 0.1);
  addBiasAnchor(problem, bias.accelBlock(0), ba0_init, 0.1);
  addBiasAnchor(problem, bias.accelBlock(1), ba1_init, 0.1);

  const double gyro_gap_before = (bg1_init - bg0_init).norm();
  const double accel_gap_before = (ba1_init - ba0_init).norm();

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  const double gyro_gap_after = (bias.gyroKnot(1) - bias.gyroKnot(0)).norm();
  const double accel_gap_after = (bias.accelKnot(1) - bias.accelKnot(0)).norm();

  EXPECT_LT(gyro_gap_after, 0.2 * gyro_gap_before);
  EXPECT_LT(accel_gap_after, 0.2 * accel_gap_before);
}

// (e) Non-uniform knot density: with n_cp == 2 every outer segment maps onto two
// virtual knot intervals (slope == 2), so the body rate and world accel the residual
// models carry the slope and slope^2 corrections of the real->virtual remap. Feeding
// a perfect IMU built from the same analytic kinematics, the per-sample residual must
// still vanish, and the spline's own evaluators must reproduce the analytic GT inside
// the dense segments. A missing or doubled slope shows up here as a non-zero residual.
//
// The builder's first four knots sit at the nominal cadence, so the spline carries a
// density transition (slope 1 -> 2) early on. The virtual-time grid is C2 in virtual
// time, which a slope break reduces to C1 in real time, so the constant-rate ground
// truth is only exactly representable where all four supporting knots are uniformly
// dense. The measured span therefore starts past the transition's support.
TEST(ResidualsImu, ZeroResidualOnDenseKnots) {
  GroundTruth gt;
  const Duration knot_dt = 40'000'000;  // 40 ms outer cadence
  const Timestamp t0 = 0;
  const Timestamp t_end = 500'000'000;
  const Timestamp t_meas = 200'000'000;  // uniformly dense from here on
  SplineWindow spline = makeSplineDense(gt, t0, t_end, knot_dt, 2);
  const auto kts = knotTimes(t0, t_end, knot_dt, 2);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  BiasKnots bias(t0, knot_dt, 1);  // perfect IMU: zero bias, one constant knot
  GravityBlock gravity(kGravityWorld, kGravityMag);
  const auto imu = makeImu(gt, t0, t_end, 5'000'000);

  ceres::Problem problem;
  const int n = addImuResiduals(problem, spline, bias, gravity, imu, t_meas, t_end,
                                unitWeights());
  ASSERT_GT(n, 0);
  constrainSo3Knots(problem, spline, knot_dt);
  pinGauge(problem, spline, t_meas);
  problem.SetParameterBlockConstant(bias.gyroBlock(0));
  problem.SetParameterBlockConstant(bias.accelBlock(0));
  problem.SetParameterBlockConstant(gravity.data());

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 40;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  // Perfect, consistent measurements through the slope-corrected residual: the
  // fitted cost per sample must be tiny on the dense path too.
  EXPECT_LT(summary.final_cost / static_cast<double>(n), 1e-4);

  // The dense-knot spline reproduces the analytic kinematics the IMU was built from;
  // this is the direct check that the residual and the spline evaluators agree on the
  // real->virtual slope inside an n_cp == 2 segment.
  for (Timestamp t = t_meas + 40'000'000; t < t_end - 40'000'000; t += 40'000'000) {
    EXPECT_LT((spline.angularVelocityBody(t) - gt.omegaBody(tSec(t))).norm(), 1e-2)
        << "t=" << t;
    EXPECT_LT((spline.linearAccelWorld(t) - gt.accelWorld(tSec(t))).norm(), 5e-2)
        << "t=" << t;
  }
}

// ---- F5: adaptive knot density from peak per-sample IMU excitation ----

// imuExcitation reads peak per-sample magnitudes. A fast pure rotation whose world-
// frame angular velocity vectors sum toward zero still yields a large N_omega because
// each sample's magnitude is taken individually.
TEST(KnotDensity, ExcitationIsPeakPerSampleMagnitude) {
  std::vector<ImuSample> imu;
  // Body angular rate of constant magnitude 2.0 rad/s but sweeping direction, so any
  // vector sum cancels; the per-sample magnitude is a flat 2.0.
  for (int i = 0; i < 16; ++i) {
    ImuSample s;
    const double a = 2.0 * M_PI * i / 16.0;
    s.gyro = Eigen::Vector3d(2.0 * std::cos(a), 2.0 * std::sin(a), 0.0);
    s.acc = kGravityWorld * -1.0;  // a stationary specific force of magnitude |g|
    imu.push_back(s);
  }
  const ImuExcitation exc =
      imuExcitation(imu, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), kGravityMag);
  EXPECT_NEAR(exc.n_omega, 2.0, 1e-9);
  // Specific force magnitude == |g| everywhere, so the gravity-removed accel is ~0.
  EXPECT_NEAR(exc.n_accel, 0.0, 1e-9);
}

// The angular threshold ladder adds one control point per crossed band, the larger of
// the two axis mappings wins, and the result clamps to n_cp_max.
TEST(KnotDensity, BandLadderAddsOneCpPerThreshold) {
  const std::vector<double> omega_thresh = {0.5, 1.0, 2.0};  // 3 rising bands
  const std::vector<double> accel_thresh = {1.0, 3.0};
  const double hys = 0.15;
  const int n_cp_max = 4;

  // Below the first band: a single control point.
  EXPECT_EQ(knotDensityFromExcitation({0.1, 0.1}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            1);
  // Omega in the first band -> +1 cp.
  EXPECT_EQ(knotDensityFromExcitation({0.7, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            2);
  // Omega clears all three bands -> 1 + 3 = 4 (== n_cp_max).
  EXPECT_EQ(knotDensityFromExcitation({3.0, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            4);
  // The accel axis is the binding one here (2 bands -> 3 cp) over a 1-band omega.
  EXPECT_EQ(knotDensityFromExcitation({0.7, 4.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            3);
  // Well past every band still clamps to n_cp_max.
  EXPECT_EQ(knotDensityFromExcitation({100.0, 100.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            4);
}

// Hysteresis: a statistic hovering just under a band edge it already held stays in the
// band (no chatter); it only steps down once it falls below the edge by the hysteresis
// fraction. Same inputs always give the same output (determinism).
TEST(KnotDensity, HysteresisPreventsChatter) {
  const std::vector<double> omega_thresh = {1.0};  // one band at 1.0 rad/s
  const std::vector<double> accel_thresh = {};
  const double hys = 0.15;  // drop edge at 0.85
  const int n_cp_max = 2;

  // Rise into the band: 1 -> 2 cp.
  EXPECT_EQ(knotDensityFromExcitation({1.2, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            2);
  // Dip just below the rising edge but above the drop edge while already holding the
  // band: density holds at 2 (no chatter down).
  EXPECT_EQ(knotDensityFromExcitation({0.9, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 2),
            2);
  // Fall below the drop edge: now step down to 1 cp.
  EXPECT_EQ(knotDensityFromExcitation({0.8, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 2),
            1);
  // From the low state, 0.9 is below the rising edge so density does not re-enter.
  EXPECT_EQ(knotDensityFromExcitation({0.9, 0.0}, omega_thresh, accel_thresh, hys,
                                      n_cp_max, 1),
            1);
  // Determinism: same inputs, repeated, same answer.
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(knotDensityFromExcitation({0.9, 0.0}, omega_thresh, accel_thresh, hys,
                                        n_cp_max, 2),
              2);
  }
}

// A full excitation series mapped to its expected n_cp sequence under hysteresis,
// feeding each step's output back as the next step's prev_n_cp (as the front-end does).
TEST(KnotDensity, ExcitationSeriesProducesExpectedSequence) {
  const std::vector<double> omega_thresh = {0.5, 1.5};  // bands at 0.5 and 1.5
  const std::vector<double> accel_thresh = {};
  const double hys = 0.2;  // drop edges at 0.4 and 1.2
  const int n_cp_max = 3;

  // omega series and the n_cp it should yield with feedback hysteresis.
  const std::vector<double> series = {0.1, 0.6, 1.6, 1.3, 0.45, 0.3, 2.0};
  const std::vector<int> expected = {1, 2, 3, 3, 2, 1, 3};
  //  0.10 -> below band0          -> 1
  //  0.60 -> band0 (>=0.5)        -> 2
  //  1.60 -> band0+band1 (>=1.5)  -> 3
  //  1.30 -> holds band1 (>=1.2)  -> 3   (1.3 < 1.5 rising, but >= 1.2 drop edge)
  //  0.45 -> band1 lost, holds b0 -> 2   (0.45 >= 0.4 drop edge of band0)
  //  0.30 -> below 0.4 drop edge  -> 1
  //  2.00 -> both bands           -> 3
  int prev = 1;
  std::vector<int> got;
  for (double w : series) {
    prev = knotDensityFromExcitation({w, 0.0}, omega_thresh, accel_thresh, hys,
                                     n_cp_max, prev);
    got.push_back(prev);
  }
  EXPECT_EQ(got, expected);
}

// ---- F6: gated motion regularizer ----

// At the ground truth the regularizer residual is the spline's true jerk / angular
// acceleration, which for the smooth analytic trajectory is small; more importantly,
// adding it does not move a GT-seeded spline (its contribution to the cost is bounded
// and the fit stays at GT). Here we confirm the residual stays bounded.
TEST(MotionRegularizer, BoundedResidualAtGroundTruth) {
  GroundTruth gt;
  const Duration knot_dt = 40'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 300'000'000;
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  std::vector<Timestamp> mids;
  for (Timestamp t = t0 + knot_dt + knot_dt / 2; t < t_end - knot_dt; t += knot_dt) {
    mids.push_back(t);
  }
  ASSERT_FALSE(mids.empty());

  ceres::Problem problem;
  const int n = addMotionRegularizer(problem, spline, mids, 1e-3, 1.0, 1.0);
  EXPECT_EQ(n, static_cast<int>(mids.size()));
  constrainSo3Knots(problem, spline, knot_dt);

  double cost = 0.0;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
  // The smooth GT trajectory has modest jerk/angular-accel; the low weight keeps the
  // cost bounded and tiny so the regularizer biases nothing where data constrains.
  EXPECT_LT(cost, 1e-2);
}

// A free (unconstrained) control point with no other residual floats arbitrarily; with
// the regularizer on, the solve pins it to the constant-velocity / constant-rate
// continuation of its neighbours, so its solved position stays bounded.
TEST(MotionRegularizer, UnderExcitedSpanStaysBounded) {
  GroundTruth gt;
  const Duration knot_dt = 40'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 320'000'000;
  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  // Perturb one interior R^3 knot far away to model a knot left floating by an under-
  // excited span with no LiDAR/IMU residual touching it.
  const Timestamp t_pick = 160'000'000;
  SplineWindow::SegmentRef seg = spline.segmentFor(t_pick);
  Eigen::Map<Eigen::Vector3d> floated(seg.r3_knots[1]);
  const Eigen::Vector3d before = floated;
  floated += Eigen::Vector3d(3.0, -2.5, 4.0);  // wild displacement

  std::vector<Timestamp> mids;
  for (Timestamp t = t0 + knot_dt + knot_dt / 2; t < t_end - knot_dt; t += knot_dt) {
    mids.push_back(t);
  }

  ceres::Problem problem;
  ASSERT_GT(addMotionRegularizer(problem, spline, mids, 1.0, 1.0, 1.0), 0);
  constrainSo3Knots(problem, spline, knot_dt);
  // Pin everything except the floated knot so the regularizer alone must rein it in.
  std::vector<double*> all_r3;
  for (Timestamp t = spline.minTime(); t < spline.maxTime(); t += knot_dt / 2) {
    SplineWindow::SegmentRef s = spline.segmentFor(t);
    for (double* p : s.r3_knots) {
      if (p != seg.r3_knots[1] && problem.HasParameterBlock(p)) {
        problem.SetParameterBlockConstant(p);
      }
    }
    for (double* q : s.so3_knots) {
      if (problem.HasParameterBlock(q)) {
        problem.SetParameterBlockConstant(q);
      }
    }
  }

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 50;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  // The regularizer pulls the floated knot back toward the smooth continuation of its
  // fixed neighbours, so its solved value is far closer to the unperturbed seed than
  // to the wild displacement it started at.
  const Eigen::Vector3d after = floated;
  EXPECT_LT((after - before).norm(), 1.0)
      << "floated knot was not reined in: " << after.transpose();
}

// --- Analytic-vs-autodiff parity for the IMU factor family -----------------------
//
// The analytic costs are the exact same functions as the kept autodiff functors;
// their Jacobians must agree at near-machine precision once SO(3) and gravity blocks
// are projected through their manifold's PlusJacobian (the two formulations extend the
// function differently OFF the unit sphere, and only the tangent projection is the
// well-defined object the solver consumes). R^3 and bias blocks are unconstrained, so
// they are compared directly.

namespace {

// A random IMU-factor configuration: four bounded-rotation SO(3) knots, four R^3
// knots, biases, a unit gravity direction, and the per-sample inputs.
struct ImuProbe {
  std::array<Eigen::Quaterniond, 4> q;
  std::array<Eigen::Vector3d, 4> p;
  Eigen::Vector3d bg_l, bg_r, ba_l, ba_r;  // non-shared bias blocks
  Eigen::Vector3d g_dir;                   // unit
  Eigen::Vector3d gyro, accel;
  double u = 0.5;
  double inv_dt = 10.0;
  double alpha = 0.5;
  double gravity_mag = 9.81;
  double w_gyro = 1.0;
  double w_accel = 1.0;
};

ImuProbe randomImuProbe(std::mt19937& rng, double rot_scale) {
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  auto randVec = [&] { return Eigen::Vector3d(uni(rng), uni(rng), uni(rng)); };
  ImuProbe pr;
  Eigen::Quaterniond q0(uni(rng) + 1.5, uni(rng), uni(rng), uni(rng));
  q0.normalize();
  for (int j = 0; j < 4; ++j) {
    pr.q[static_cast<std::size_t>(j)] = q0;
    q0 = (q0 * Sophus::SO3d::exp(rot_scale * randVec()).unit_quaternion()).normalized();
    pr.p[static_cast<std::size_t>(j)] = 2.0 * randVec();
  }
  pr.bg_l = 0.1 * randVec();
  pr.bg_r = 0.1 * randVec();
  pr.ba_l = 0.2 * randVec();
  pr.ba_r = 0.2 * randVec();
  pr.g_dir = randVec().normalized();
  pr.gyro = randVec();
  pr.accel = 3.0 * randVec();
  const double dt = 0.005 + 0.045 * 0.5 * (uni(rng) + 1.0);
  pr.inv_dt = 1.0 / dt;
  pr.alpha = 0.5 * (uni(rng) + 1.0);
  pr.u = 0.5 * (uni(rng) + 1.0) * 0.999;
  pr.gravity_mag = 9.81;
  pr.w_gyro = 0.5 + 0.5 * (uni(rng) + 1.0);
  pr.w_accel = 0.5 + 0.5 * (uni(rng) + 1.0);
  return pr;
}

// Compares two ambient SO(3) (Rows by 4) Jacobian blocks after projecting both through
// the EigenQuaternionManifold PlusJacobian at the knot quaternion. The tolerance is
// scaled by the block's largest entry: an individual projected entry that cancels to
// near zero still carries a floor proportional to the BLOCK's natural magnitude, not its
// own cancelled value, so a fixed absolute floor would be meaningless there. The
// autodiff reference itself loses precision in its log/exp dual-number chain when the
// inter-knot rotation is near identity (verified against central finite differences:
// analytic and autodiff track FD to the SAME ~1e-5 floor there), so the SO(3) blocks are
// the autodiff-reference-limited 1e-5; the R^3/bias blocks, which are exact in autodiff,
// stay at the machine floor.
template <int Rows>
void expectSo3BlockMatch(const double* q, const double* ja, const double* jr,
                         ceres::EigenQuaternionManifold& manifold, double tol,
                         int trial, int knot) {
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> plus_jac;
  manifold.PlusJacobian(q, plus_jac.data());
  const Eigen::Map<const Eigen::Matrix<double, Rows, 4, Eigen::RowMajor>> Ja(ja);
  const Eigen::Map<const Eigen::Matrix<double, Rows, 4, Eigen::RowMajor>> Jr(jr);
  const Eigen::Matrix<double, Rows, 3> la = Ja * plus_jac;
  const Eigen::Matrix<double, Rows, 3> lr = Jr * plus_jac;
  const double scale = std::max(1.0, lr.cwiseAbs().maxCoeff());
  for (int row = 0; row < Rows; ++row)
    for (int c = 0; c < 3; ++c)
      EXPECT_NEAR(la(row, c), lr(row, c), tol * scale)
          << "trial " << trial << " so3 knot " << knot << " row " << row << " col " << c;
}

// Relative+absolute tolerance: the R^3 derivative blocks carry inv_dt^Deriv scaling, so
// a jerk Jacobian can reach 1e5 in magnitude and an absolute-only floor would demand
// sub-ULP agreement. The blocks are analytically identical, so the residual after
// machine rounding is bounded by tol*(1+|value|).
void expectR3BlockMatch(const double* ja, const double* jr, int rows, double tol,
                        int trial, int knot, const char* tag) {
  for (int i = 0; i < rows * 3; ++i)
    EXPECT_NEAR(ja[i], jr[i], tol + tol * std::abs(jr[i]))
        << "trial " << trial << " " << tag << " block " << knot << " entry " << i;
}

// Relative+absolute residual comparison; the accel rows carry inv_dt^2-scaled world
// acceleration and the jerk inv_dt^3, so the same machine-precision argument as the
// Jacobian blocks applies.
void expectResMatch(const double* a, const double* r, int n, double tol, int trial) {
  for (int i = 0; i < n; ++i)
    EXPECT_NEAR(a[i], r[i], tol + tol * std::abs(r[i])) << "trial " << trial << " res " << i;
}

}  // namespace

// Main combined gyro+accel factor, single-bias-knot layout: analytic Jacobians match
// the autodiff functor (tangent-projected on SO(3) and gravity).
TEST(AnalyticImuCost, JacobiansMatchAutodiff_Shared) {
  std::mt19937 rng(101);
  ceres::EigenQuaternionManifold quat_manifold;
  ceres::SphereManifold<3> sphere_manifold;
  for (int trial = 0; trial < 60; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    ImuProbe pr = randomImuProbe(rng, rot_scale);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;

    std::unique_ptr<ceres::CostFunction> analytic(
        makeImuCost<true>(pr.gyro, pr.accel, pr.u, pr.inv_dt, pr.alpha, pr.gravity_mag,
                          pr.w_gyro, pr.w_accel));
    std::unique_ptr<ceres::CostFunction> reference(makeImuCostAutodiff<true>(
        pr.gyro, pr.accel, pr.u, pr.inv_dt, pr.alpha, pr.gravity_mag, pr.w_gyro,
        pr.w_accel));

    Eigen::Vector3d b_g = 0.1 * pr.bg_l;
    Eigen::Vector3d b_a = 0.2 * pr.ba_l;
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());
    params.push_back(b_g.data());
    params.push_back(b_a.data());
    params.push_back(pr.g_dir.data());
    const int g_index = 10;

    double ja_so3[4][24], jr_so3[4][24];
    double ja_r3[4][18], jr_r3[4][18];
    double ja_bg[18], jr_bg[18], ja_ba[18], jr_ba[18], ja_g[18], jr_g[18];
    std::vector<double*> ja, jr;
    for (int j = 0; j < 4; ++j) { ja.push_back(ja_so3[j]); jr.push_back(jr_so3[j]); }
    for (int j = 0; j < 4; ++j) { ja.push_back(ja_r3[j]); jr.push_back(jr_r3[j]); }
    ja.push_back(ja_bg); jr.push_back(jr_bg);
    ja.push_back(ja_ba); jr.push_back(jr_ba);
    ja.push_back(ja_g); jr.push_back(jr_g);

    double r_a[6], r_r[6];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, ja.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jr.data()));
    expectResMatch(r_a, r_r, 6, 1e-12, trial);

    // Well-conditioned (large-rotation, interior) trials must match the autodiff
    // reference tightly; near-identity or segment-edge trials are limited by the
    // reference's own small-angle precision (see expectSo3BlockMatch).
    const bool well_conditioned = (rot_scale > 1e-2) && (trial % 5 != 0) && (trial % 7 != 0);
    const double so3_tol = well_conditioned ? 1e-7 : 1e-5;
    for (int j = 0; j < 4; ++j)
      expectSo3BlockMatch<6>(params[j], ja_so3[j], jr_so3[j], quat_manifold,
                             so3_tol, trial, j);
    for (int j = 0; j < 4; ++j)
      expectR3BlockMatch(ja_r3[j], jr_r3[j], 6, 1e-9, trial, j, "r3");
    expectR3BlockMatch(ja_bg, jr_bg, 6, 1e-9, trial, 0, "bg");
    expectR3BlockMatch(ja_ba, jr_ba, 6, 1e-9, trial, 0, "ba");
    // Gravity: ambient block matches directly, and also after the sphere projection.
    expectR3BlockMatch(ja_g, jr_g, 6, 1e-9, trial, 0, "g_ambient");
    Eigen::Matrix<double, 3, 2, Eigen::RowMajor> sphere_jac;
    sphere_manifold.PlusJacobian(params[g_index], sphere_jac.data());
    const Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Jga(ja_g);
    const Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Jgr(jr_g);
    const Eigen::Matrix<double, 6, 2> lga = Jga * sphere_jac;
    const Eigen::Matrix<double, 6, 2> lgr = Jgr * sphere_jac;
    for (int row = 0; row < 6; ++row)
      for (int c = 0; c < 2; ++c)
        EXPECT_NEAR(lga(row, c), lgr(row, c), 1e-9 + 1e-9 * std::abs(lgr(row, c)))
            << "trial " << trial << " gravity sphere row " << row << " col " << c;
  }
}

// Main factor, two-bias-knot interpolated layout (random alpha): four bias blocks plus
// gravity at index 12.
TEST(AnalyticImuCost, JacobiansMatchAutodiff_NonShared) {
  std::mt19937 rng(202);
  ceres::EigenQuaternionManifold quat_manifold;
  ceres::SphereManifold<3> sphere_manifold;
  for (int trial = 0; trial < 60; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    ImuProbe pr = randomImuProbe(rng, rot_scale);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;

    std::unique_ptr<ceres::CostFunction> analytic(
        makeImuCost<false>(pr.gyro, pr.accel, pr.u, pr.inv_dt, pr.alpha, pr.gravity_mag,
                           pr.w_gyro, pr.w_accel));
    std::unique_ptr<ceres::CostFunction> reference(makeImuCostAutodiff<false>(
        pr.gyro, pr.accel, pr.u, pr.inv_dt, pr.alpha, pr.gravity_mag, pr.w_gyro,
        pr.w_accel));

    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());
    params.push_back(pr.bg_l.data());
    params.push_back(pr.bg_r.data());
    params.push_back(pr.ba_l.data());
    params.push_back(pr.ba_r.data());
    params.push_back(pr.g_dir.data());
    const int g_index = 12;

    double ja_so3[4][24], jr_so3[4][24];
    double ja_r3[4][18], jr_r3[4][18];
    double ja_bias[4][18], jr_bias[4][18];
    double ja_g[18], jr_g[18];
    std::vector<double*> ja, jr;
    for (int j = 0; j < 4; ++j) { ja.push_back(ja_so3[j]); jr.push_back(jr_so3[j]); }
    for (int j = 0; j < 4; ++j) { ja.push_back(ja_r3[j]); jr.push_back(jr_r3[j]); }
    for (int j = 0; j < 4; ++j) { ja.push_back(ja_bias[j]); jr.push_back(jr_bias[j]); }
    ja.push_back(ja_g); jr.push_back(jr_g);

    double r_a[6], r_r[6];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, ja.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jr.data()));
    expectResMatch(r_a, r_r, 6, 1e-12, trial);

    const bool well_conditioned = (rot_scale > 1e-2) && (trial % 5 != 0) && (trial % 7 != 0);
    const double so3_tol = well_conditioned ? 1e-7 : 1e-5;
    for (int j = 0; j < 4; ++j)
      expectSo3BlockMatch<6>(params[j], ja_so3[j], jr_so3[j], quat_manifold,
                             so3_tol, trial, j);
    for (int j = 0; j < 4; ++j)
      expectR3BlockMatch(ja_r3[j], jr_r3[j], 6, 1e-9, trial, j, "r3");
    for (int j = 0; j < 4; ++j)
      expectR3BlockMatch(ja_bias[j], jr_bias[j], 6, 1e-9, trial, j, "bias");
    expectR3BlockMatch(ja_g, jr_g, 6, 1e-9, trial, 0, "g_ambient");
    Eigen::Matrix<double, 3, 2, Eigen::RowMajor> sphere_jac;
    sphere_manifold.PlusJacobian(params[g_index], sphere_jac.data());
    const Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Jga(ja_g);
    const Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> Jgr(jr_g);
    const Eigen::Matrix<double, 6, 2> lga = Jga * sphere_jac;
    const Eigen::Matrix<double, 6, 2> lgr = Jgr * sphere_jac;
    for (int row = 0; row < 6; ++row)
      for (int c = 0; c < 2; ++c)
        EXPECT_NEAR(lga(row, c), lgr(row, c), 1e-9 + 1e-9 * std::abs(lgr(row, c)))
            << "trial " << trial << " gravity sphere row " << row << " col " << c;
  }
}

// Residual-only path of the main factor matches a direct basalt forward evaluation
// (evaluate_lie for omega, evaluate<3,2> for a_world) at machine precision.
TEST(AnalyticImuCost, ResidualMatchesForwardEval) {
  std::mt19937 rng(303);
  for (int trial = 0; trial < 30; ++trial) {
    ImuProbe pr = randomImuProbe(rng, 0.3);
    Eigen::Vector3d b_g = 0.1 * pr.bg_l;
    Eigen::Vector3d b_a = 0.2 * pr.ba_l;

    std::unique_ptr<ceres::CostFunction> analytic(
        makeImuCost<true>(pr.gyro, pr.accel, pr.u, pr.inv_dt, pr.alpha, pr.gravity_mag,
                          pr.w_gyro, pr.w_accel));
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());
    params.push_back(b_g.data());
    params.push_back(b_a.data());
    params.push_back(pr.g_dir.data());

    double r_a[6];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, nullptr));

    Sophus::SO3d R_w_fe;
    Sophus::SO3d::Tangent omega;
    basalt::CeresSplineHelper<4>::evaluate_lie<double, Sophus::SO3>(
        params.data(), pr.u, pr.inv_dt, &R_w_fe, &omega, nullptr);
    Eigen::Vector3d a_world;
    basalt::CeresSplineHelper<4>::evaluate<double, 3, 2>(params.data() + 4, pr.u,
                                                         pr.inv_dt, &a_world);
    const Eigen::Vector3d g_world = pr.g_dir * pr.gravity_mag;
    Eigen::Matrix<double, 6, 1> r_ref;
    r_ref.head<3>() = pr.w_gyro * (omega + b_g - pr.gyro);
    r_ref.tail<3>() =
        pr.w_accel * (R_w_fe.inverse() * (a_world - g_world) + b_a - pr.accel);
    expectResMatch(r_a, r_ref.data(), 6, 1e-12, trial);
  }
}

// Jerk regularizer: purely linear in R^3 knots, so analytic and autodiff agree exactly.
TEST(AnalyticImuCompanions, JerkMatchesAutodiff) {
  std::mt19937 rng(404);
  for (int trial = 0; trial < 60; ++trial) {
    ImuProbe pr = randomImuProbe(rng, 0.3);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;
    const double w = pr.w_accel;
    std::unique_ptr<ceres::CostFunction> analytic(makeJerkCost(pr.u, pr.inv_dt, w));
    std::unique_ptr<ceres::CostFunction> reference(makeJerkCostAutodiff(pr.u, pr.inv_dt, w));
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());
    double ja[4][9], jr[4][9];
    std::vector<double*> jap, jrp;
    for (int j = 0; j < 4; ++j) { jap.push_back(ja[j]); jrp.push_back(jr[j]); }
    double r_a[3], r_r[3];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jrp.data()));
    expectResMatch(r_a, r_r, 3, 1e-12, trial);
    for (int j = 0; j < 4; ++j) expectR3BlockMatch(ja[j], jr[j], 3, 1e-12, trial, j, "jerk");
  }
}

// Velocity anchor: linear in R^3 knots, exact match.
TEST(AnalyticImuCompanions, VelocityAnchorMatchesAutodiff) {
  std::mt19937 rng(505);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  for (int trial = 0; trial < 60; ++trial) {
    ImuProbe pr = randomImuProbe(rng, 0.3);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;
    const Eigen::Vector3d v_pred(uni(rng), uni(rng), uni(rng));
    const double w = pr.w_accel;
    std::unique_ptr<ceres::CostFunction> analytic(
        makeVelocityAnchorCost(v_pred, pr.u, pr.inv_dt, w));
    std::unique_ptr<ceres::CostFunction> reference(
        makeVelocityAnchorCostAutodiff(v_pred, pr.u, pr.inv_dt, w));
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());
    double ja[4][9], jr[4][9];
    std::vector<double*> jap, jrp;
    for (int j = 0; j < 4; ++j) { jap.push_back(ja[j]); jrp.push_back(jr[j]); }
    double r_a[3], r_r[3];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jrp.data()));
    expectResMatch(r_a, r_r, 3, 1e-12, trial);
    for (int j = 0; j < 4; ++j) expectR3BlockMatch(ja[j], jr[j], 3, 1e-12, trial, j, "vanchor");
  }
}

// Bias tie: constant Jacobians, exact match.
TEST(AnalyticImuCompanions, BiasTieMatchesAutodiff) {
  std::mt19937 rng(606);
  for (int trial = 0; trial < 30; ++trial) {
    ImuProbe pr = randomImuProbe(rng, 0.3);
    const double w_g = pr.w_gyro, w_a = pr.w_accel;
    std::unique_ptr<ceres::CostFunction> analytic(makeBiasTieCost(w_g, w_a));
    std::unique_ptr<ceres::CostFunction> reference(makeBiasTieCostAutodiff(w_g, w_a));
    std::vector<const double*> params{pr.bg_l.data(), pr.bg_r.data(), pr.ba_l.data(),
                                      pr.ba_r.data()};
    double ja[4][18], jr[4][18];
    std::vector<double*> jap, jrp;
    for (int j = 0; j < 4; ++j) { jap.push_back(ja[j]); jrp.push_back(jr[j]); }
    double r_a[6], r_r[6];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jrp.data()));
    expectResMatch(r_a, r_r, 6, 1e-12, trial);
    for (int j = 0; j < 4; ++j) expectR3BlockMatch(ja[j], jr[j], 6, 1e-12, trial, j, "biastie");
  }
}

// Rate anchor: body angular velocity, SO(3) tangent-projected.
TEST(AnalyticImuCompanions, RateAnchorMatchesAutodiff) {
  std::mt19937 rng(707);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  ceres::EigenQuaternionManifold quat_manifold;
  for (int trial = 0; trial < 60; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    ImuProbe pr = randomImuProbe(rng, rot_scale);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;
    const Eigen::Vector3d w_pred(uni(rng), uni(rng), uni(rng));
    const double w = pr.w_gyro;
    std::unique_ptr<ceres::CostFunction> analytic(
        makeRateAnchorCost(w_pred, pr.u, pr.inv_dt, w));
    std::unique_ptr<ceres::CostFunction> reference(
        makeRateAnchorCostAutodiff(w_pred, pr.u, pr.inv_dt, w));
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    double ja[4][12], jr[4][12];
    std::vector<double*> jap, jrp;
    for (int j = 0; j < 4; ++j) { jap.push_back(ja[j]); jrp.push_back(jr[j]); }
    double r_a[3], r_r[3];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jrp.data()));
    expectResMatch(r_a, r_r, 3, 1e-12, trial);
    for (int j = 0; j < 4; ++j)
      expectSo3BlockMatch<3>(params[j], ja[j], jr[j], quat_manifold, 1e-9, trial, j);
  }
}

// Angular-acceleration regularizer: second SO(3) derivative, tangent-projected. Looser
// tolerance accommodates the lie-bracket / large-rotation series conditioning.
TEST(AnalyticImuCompanions, AngularAccelMatchesAutodiff) {
  std::mt19937 rng(808);
  ceres::EigenQuaternionManifold quat_manifold;
  for (int trial = 0; trial < 60; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    ImuProbe pr = randomImuProbe(rng, rot_scale);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;
    const double w = pr.w_gyro;
    std::unique_ptr<ceres::CostFunction> analytic(makeAngularAccelCost(pr.u, pr.inv_dt, w));
    std::unique_ptr<ceres::CostFunction> reference(
        makeAngularAccelCostAutodiff(pr.u, pr.inv_dt, w));
    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    double ja[4][12], jr[4][12];
    std::vector<double*> jap, jrp;
    for (int j = 0; j < 4; ++j) { jap.push_back(ja[j]); jrp.push_back(jr[j]); }
    double r_a[3], r_r[3];
    ASSERT_TRUE(analytic->Evaluate(params.data(), r_a, jap.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), r_r, jrp.data()));
    expectResMatch(r_a, r_r, 3, 1e-9, trial);
    for (int j = 0; j < 4; ++j)
      expectSo3BlockMatch<3>(params[j], ja[j], jr[j], quat_manifold, 1e-9, trial, j);
  }
}
