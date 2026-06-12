#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <numbers>
#include <random>
#include <sophus/so3.hpp>
#include <vector>

#include "lio/imu_tracker.hpp"
#include "meridian/config/config.hpp"

using meridian::FrontendLio;
using meridian::ImuSample;
using meridian::NavState;
using meridian::Pose;
using meridian::Timestamp;
using meridian::lio::ImuTracker;
using meridian::lio::MotionPrior;

namespace {

constexpr double kG = ImuTracker::kGravityMagnitude;
constexpr Timestamp kT0 = 1'000'000'000;
constexpr Timestamp kStepNs = 5'000'000;  // 200 Hz
constexpr double kStepS = 5e-3;

FrontendLio immediateCfg() {
  FrontendLio cfg;
  cfg.init_stationary_s = 0.0;
  return cfg;
}

ImuSample makeSample(Timestamp stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
  ImuSample s;
  s.stamp = stamp;
  s.acc = acc;
  s.gyro = gyro;
  return s;
}

void expectBitEqual(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  EXPECT_TRUE((a.array() == b.array()).all()) << a.transpose() << " vs " << b.transpose();
}

// The integrator compensates each sample with the attitude *before* the update, so a
// reading of R_prev^T * (0,0,kG) cancels gravity exactly and isolates the rotation.
Eigen::Vector3d gravityReaction(const Eigen::Quaterniond& q_prev) {
  return q_prev.conjugate() * Eigen::Vector3d(0.0, 0.0, kG);
}

// Mirror of the tracker's attitude update for generating gravity-canceling inputs.
Eigen::Quaterniond stepAttitude(const Eigen::Quaterniond& q, const Eigen::Vector3d& omega,
                                double dt) {
  return (q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
}

}  // namespace

TEST(ImuTracker, ConstructsUninitialized) {
  ImuTracker tracker{FrontendLio{}};
  EXPECT_FALSE(tracker.initialized());
  EXPECT_TRUE(tracker.bias_gyro().isZero());
  EXPECT_TRUE(tracker.bias_accel().isZero());
}

// With no standstill window the very first sample seeds the state: identity attitude
// at the sample stamp, zero biases, and the sample already counts toward the interval.
TEST(ImuTracker, ImmediateInitOnFirstSample) {
  ImuTracker tracker{immediateCfg()};
  tracker.add_sample(makeSample(kT0, Eigen::Vector3d(0, 0, kG), Eigen::Vector3d::Zero()));

  EXPECT_TRUE(tracker.initialized());
  const NavState st = tracker.propagated_state();
  EXPECT_EQ(st.stamp, kT0);
  EXPECT_TRUE(st.T_world_body.q.isApprox(Eigen::Quaterniond::Identity()));
  EXPECT_TRUE(st.T_world_body.t.isZero());
  EXPECT_TRUE(st.v_world.isZero());
  EXPECT_TRUE(tracker.bias_gyro().isZero());
  EXPECT_TRUE(tracker.bias_accel().isZero());

  const MotionPrior prior = tracker.interval_prior(kT0, kT0 + kStepNs);
  EXPECT_EQ(prior.imu_count, 1);
}

// Constant rate omega for T seconds must land on exp(omega * T); the gravity-canceling
// accelerometer input keeps velocity and position at exactly zero.
TEST(ImuTracker, GyroOnlyConstantRateMatchesClosedForm) {
  ImuTracker tracker{immediateCfg()};
  const Eigen::Vector3d omega(0.3, -0.2, 0.5);

  constexpr int kN = 400;
  Eigen::Quaterniond q_sim = Eigen::Quaterniond::Identity();
  tracker.add_sample(makeSample(kT0, gravityReaction(q_sim), omega));
  for (int k = 1; k < kN; ++k) {
    tracker.add_sample(makeSample(kT0 + k * kStepNs, gravityReaction(q_sim), omega));
    q_sim = stepAttitude(q_sim, omega, kStepS);
  }

  const NavState st = tracker.propagated_state();
  const double total_s = (kN - 1) * kStepS;
  const Eigen::Quaterniond q_expected = Sophus::SO3d::exp(omega * total_s).unit_quaternion();
  EXPECT_LT(st.T_world_body.q.angularDistance(q_expected), 1e-9);
  EXPECT_LT(st.v_world.norm(), 1e-9);
  EXPECT_LT(st.T_world_body.t.norm(), 1e-9);
  EXPECT_EQ(st.stamp, kT0 + (kN - 1) * kStepNs);
}

// Constant body acceleration at identity attitude: v = a*T and p = a*T^2/2, which the
// per-step trapezoid p += v*dt + a*dt^2/2 reproduces exactly in exact arithmetic.
TEST(ImuTracker, ConstantBodyAccelIntegratesParabola) {
  ImuTracker tracker{immediateCfg()};
  const Eigen::Vector3d a_des(0.4, -0.1, 0.25);
  const Eigen::Vector3d acc = a_des + Eigen::Vector3d(0, 0, kG);

  constexpr int kN = 201;
  for (int k = 0; k < kN; ++k) {
    tracker.add_sample(makeSample(kT0 + k * kStepNs, acc, Eigen::Vector3d::Zero()));
  }

  const NavState st = tracker.propagated_state();
  const double total_s = (kN - 1) * kStepS;  // 1.0 s
  EXPECT_TRUE(st.T_world_body.q.isApprox(Eigen::Quaterniond::Identity()));
  EXPECT_LT((st.v_world - a_des * total_s).norm(), 1e-9);
  EXPECT_LT((st.T_world_body.t - 0.5 * a_des * total_s * total_s).norm(), 1e-9);
}

// Standstill at a known 5 deg roll with known biases: the window mean must recover the
// gyro bias, the tilt (yaw exactly zero), and the accel bias. The accel-bias component
// orthogonal to gravity is unobservable at rest (it is absorbed into the tilt), so the
// synthetic bias is placed along the body z axis.
TEST(ImuTracker, StaticInitRecoversTiltAndBiases) {
  FrontendLio cfg;
  cfg.init_stationary_s = 1.0;
  ImuTracker tracker{cfg};

  const double roll = 5.0 * std::numbers::pi / 180.0;
  const Eigen::Quaterniond q_true(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
  const Eigen::Vector3d b_g_true(0.01, -0.02, 0.015);
  const Eigen::Vector3d b_a_true(0.0, 0.0, 0.05);
  const Eigen::Vector3d f_clean = gravityReaction(q_true);

  std::mt19937 rng(42);
  std::normal_distribution<double> gyro_noise(0.0, 1e-3);
  std::normal_distribution<double> acc_noise(0.0, 1e-2);
  auto noisy = [&](std::normal_distribution<double>& d) {
    return Eigen::Vector3d(d(rng), d(rng), d(rng));
  };

  // Window completes at the sample whose elapsed time reaches 1.0 s, i.e. index 200.
  for (int k = 0; k < 200; ++k) {
    tracker.add_sample(makeSample(kT0 + k * kStepNs, f_clean + b_a_true + noisy(acc_noise),
                                  b_g_true + noisy(gyro_noise)));
    EXPECT_FALSE(tracker.initialized());
  }
  tracker.add_sample(makeSample(kT0 + 200 * kStepNs, f_clean + b_a_true + noisy(acc_noise),
                                b_g_true + noisy(gyro_noise)));
  EXPECT_TRUE(tracker.initialized());

  const NavState st = tracker.propagated_state();
  EXPECT_EQ(st.stamp, kT0 + 200 * kStepNs);
  EXPECT_TRUE(st.T_world_body.t.isZero());
  EXPECT_TRUE(st.v_world.isZero());

  EXPECT_LT((tracker.bias_gyro() - b_g_true).norm(), 1e-3);
  EXPECT_LT((tracker.bias_accel() - b_a_true).norm(), 1e-2);
  EXPECT_LT(st.T_world_body.q.angularDistance(q_true), 2e-3);
  // The alignment axis is the cross product with z_world, so yaw is exactly zero.
  EXPECT_LT(std::abs(Sophus::SO3d(st.T_world_body.q).log().z()), 1e-12);
  EXPECT_NEAR(st.g_world.z(), -kG, 1e-12);
}

// Re-delivering already-seen stamps (the straddle sample at a group boundary, or a
// stale out-of-order sample) must be a complete no-op: state and interval statistics
// stay bit-identical to a tracker that never saw the duplicates.
TEST(ImuTracker, StraddleSampleDedupIsNoOp) {
  ImuTracker with_dup{immediateCfg()};
  ImuTracker clean{immediateCfg()};

  const Eigen::Vector3d gyro(0.1, -0.05, 0.2);
  const Eigen::Vector3d acc(0.3, 0.1, kG - 0.2);
  std::vector<ImuSample> stream;
  for (int k = 0; k <= 20; ++k) {
    stream.push_back(makeSample(kT0 + k * kStepNs, acc, gyro));
  }

  for (int k = 0; k <= 10; ++k) {
    with_dup.add_sample(stream[static_cast<std::size_t>(k)]);
    clean.add_sample(stream[static_cast<std::size_t>(k)]);
  }
  with_dup.add_sample(stream[10]);  // straddle re-delivery
  with_dup.add_sample(stream[5]);   // stale stamp
  for (int k = 11; k <= 20; ++k) {
    with_dup.add_sample(stream[static_cast<std::size_t>(k)]);
    clean.add_sample(stream[static_cast<std::size_t>(k)]);
  }

  const NavState a = with_dup.propagated_state();
  const NavState b = clean.propagated_state();
  EXPECT_EQ(a.stamp, b.stamp);
  EXPECT_TRUE((a.T_world_body.q.coeffs().array() == b.T_world_body.q.coeffs().array()).all());
  expectBitEqual(a.T_world_body.t, b.T_world_body.t);
  expectBitEqual(a.v_world, b.v_world);

  const MotionPrior pa = with_dup.interval_prior(kT0, kT0 + 20 * kStepNs);
  const MotionPrior pb = clean.interval_prior(kT0, kT0 + 20 * kStepNs);
  EXPECT_EQ(pa.imu_count, pb.imu_count);
  expectBitEqual(pa.omega_body, pb.omega_body);
  expectBitEqual(pa.accel_body, pb.accel_body);
  EXPECT_EQ(pa.accel_mag_variance, pb.accel_mag_variance);
}

// The single-pass magnitude variance must agree with a direct two-pass computation
// over the same specific-force magnitudes.
TEST(ImuTracker, WelfordVarianceMatchesTwoPass) {
  ImuTracker tracker{immediateCfg()};

  constexpr int kN = 50;
  std::vector<double> mags;
  for (int k = 0; k < kN; ++k) {
    const Eigen::Vector3d acc(0.3 * std::sin(0.5 * k), 0.2 * std::cos(0.9 * k),
                              kG + 0.4 * std::sin(1.3 * k));
    mags.push_back(acc.norm());
    tracker.add_sample(makeSample(kT0 + k * kStepNs, acc, Eigen::Vector3d::Zero()));
  }

  double mean = 0.0;
  for (const double m : mags) mean += m;
  mean /= kN;
  double var = 0.0;
  for (const double m : mags) var += (m - mean) * (m - mean);
  var /= (kN - 1);

  const MotionPrior prior = tracker.interval_prior(kT0, kT0 + (kN - 1) * kStepNs);
  EXPECT_EQ(prior.imu_count, kN);
  EXPECT_NEAR(prior.accel_mag_variance, var, 1e-12);
}

// interval_prior averages the accumulated samples, resets, and falls back to the last
// real rate (zero accel, uninformative variance) when an interval holds no samples.
TEST(ImuTracker, IntervalPriorAveragesResetsAndFallsBack) {
  ImuTracker tracker{immediateCfg()};

  constexpr int kN = 20;
  Eigen::Quaterniond q_sim = Eigen::Quaterniond::Identity();
  Eigen::Vector3d omega_sum = Eigen::Vector3d::Zero();
  Eigen::Vector3d omega_last = Eigen::Vector3d::Zero();
  for (int k = 0; k < kN; ++k) {
    const Eigen::Vector3d omega(0.2 * std::sin(0.3 * k), 0.1 * std::cos(0.4 * k),
                                -0.15 * std::sin(0.2 * k));
    tracker.add_sample(makeSample(kT0 + k * kStepNs, gravityReaction(q_sim), omega));
    if (k > 0) q_sim = stepAttitude(q_sim, omega, kStepS);
    omega_sum += omega;
    omega_last = omega;
  }

  const Timestamp t_end = kT0 + (kN - 1) * kStepNs;
  const MotionPrior averaged = tracker.interval_prior(kT0, t_end);
  EXPECT_EQ(averaged.imu_count, kN);
  EXPECT_LT((averaged.omega_body - omega_sum / kN).norm(), 1e-15);
  EXPECT_LT(averaged.accel_body.norm(), 1e-15);  // inputs cancel gravity exactly
  EXPECT_GE(averaged.accel_mag_variance, 0.0);
  EXPECT_LT(averaged.accel_mag_variance, ImuTracker::kUninformativeAccelVariance);

  // Consumed: an immediate second query sees an empty interval.
  const MotionPrior empty = tracker.interval_prior(t_end, t_end + kStepNs);
  EXPECT_EQ(empty.imu_count, 0);
  expectBitEqual(empty.omega_body, omega_last);
  EXPECT_TRUE(empty.accel_body.isZero());
  EXPECT_EQ(empty.accel_mag_variance, ImuTracker::kUninformativeAccelVariance);

  // A single sample averages trivially but carries no spread information.
  const Eigen::Vector3d omega_one(0.05, 0.0, -0.1);
  tracker.add_sample(makeSample(t_end + kStepNs, gravityReaction(q_sim), omega_one));
  const MotionPrior one = tracker.interval_prior(t_end, t_end + kStepNs);
  EXPECT_EQ(one.imu_count, 1);
  expectBitEqual(one.omega_body, omega_one);
  EXPECT_EQ(one.accel_mag_variance, ImuTracker::kUninformativeAccelVariance);
}

// The scalar magnitude filter: a zero-scatter interval is adopted outright (gain 1);
// the next interval blends through the uniform-jerk process noise exactly per the
// predict/update algebra.
TEST(ImuTracker, AccelMagnitudeFilterFollowsPredictUpdate) {
  ImuTracker tracker{immediateCfg()};
  EXPECT_DOUBLE_EQ(tracker.filtered_accel_magnitude(), 0.0);
  EXPECT_DOUBLE_EQ(tracker.filtered_accel_variance(), 1.0);

  // Interval 1: constant body accel (1,0,0) -> magnitude scatter 0, gain 1.
  for (int k = 0; k < 10; ++k) {
    tracker.add_sample(
        makeSample(kT0 + k * kStepNs, Eigen::Vector3d(1.0, 0.0, kG), Eigen::Vector3d::Zero()));
  }
  const Timestamp t1 = kT0 + 50'000'000;
  const MotionPrior p1 = tracker.interval_prior(kT0, t1);
  EXPECT_EQ(p1.accel_mag_variance, 0.0);
  EXPECT_DOUBLE_EQ(tracker.filtered_accel_magnitude(), 1.0);
  EXPECT_DOUBLE_EQ(tracker.filtered_accel_variance(), 0.0);

  // Interval 2: body accel alternating (2,0,0)/(0,0,0); mean magnitude stays 1 but the
  // specific-force magnitudes now scatter.
  std::vector<double> mags;
  for (int k = 0; k < 10; ++k) {
    const Eigen::Vector3d acc(k % 2 == 0 ? 2.0 : 0.0, 0.0, kG);
    mags.push_back(acc.norm());
    tracker.add_sample(makeSample(t1 + (k + 1) * kStepNs, acc, Eigen::Vector3d::Zero()));
  }
  double mean = 0.0;
  for (const double m : mags) mean += m;
  mean /= 10.0;
  double r_meas = 0.0;
  for (const double m : mags) r_meas += (m - mean) * (m - mean);
  r_meas /= 9.0;

  const double dt2 = 0.05;
  const MotionPrior p2 = tracker.interval_prior(t1, t1 + 50'000'000);
  EXPECT_NEAR(p2.accel_mag_variance, r_meas, 1e-12);

  FrontendLio cfg = immediateCfg();
  const double q_proc = (cfg.max_expected_jerk * dt2) * (cfg.max_expected_jerk * dt2) / 3.0;
  const double var_pred = 0.0 + q_proc;
  const double gain = var_pred / (var_pred + r_meas);
  EXPECT_NEAR(tracker.filtered_accel_magnitude(), 1.0, 1e-12);  // innovation is zero
  EXPECT_NEAR(tracker.filtered_accel_variance(), (1.0 - gain) * var_pred, 1e-12);
}

// rebase adopts pose, velocity, and stamp from the solved state, keeps the tracker's
// own biases, and propagation resumes from the solved estimate.
TEST(ImuTracker, RebaseResetsKinematicsKeepsBiases) {
  ImuTracker tracker{immediateCfg()};
  for (int k = 0; k < 5; ++k) {
    tracker.add_sample(makeSample(kT0 + k * kStepNs, Eigen::Vector3d(0.5, 0.0, kG),
                                  Eigen::Vector3d(0.2, 0.0, 0.0)));
  }

  NavState solved;
  solved.stamp = kT0 + 5 * kStepNs;
  solved.T_world_body = Pose{Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())),
                             Eigen::Vector3d(1.0, 2.0, 3.0)};
  solved.v_world = Eigen::Vector3d(0.5, -0.1, 0.2);
  solved.b_g = Eigen::Vector3d(9, 9, 9);  // must NOT be adopted
  solved.b_a = Eigen::Vector3d(8, 8, 8);
  tracker.rebase(solved);

  const NavState st = tracker.propagated_state();
  EXPECT_EQ(st.stamp, solved.stamp);
  EXPECT_TRUE((st.T_world_body.q.coeffs().array() == solved.T_world_body.q.coeffs().array()).all());
  expectBitEqual(st.T_world_body.t, solved.T_world_body.t);
  expectBitEqual(st.v_world, solved.v_world);
  EXPECT_TRUE(tracker.bias_gyro().isZero());
  EXPECT_TRUE(tracker.bias_accel().isZero());

  // One gravity-canceling sample 10 ms later: pure constant-velocity translation.
  const Timestamp t_next = solved.stamp + 10'000'000;
  tracker.add_sample(
      makeSample(t_next, gravityReaction(solved.T_world_body.q), Eigen::Vector3d::Zero()));
  const NavState after = tracker.propagated_state();
  EXPECT_EQ(after.stamp, t_next);
  EXPECT_LT((after.T_world_body.t - (solved.T_world_body.t + solved.v_world * 0.01)).norm(), 1e-12);
  EXPECT_LT((after.v_world - solved.v_world).norm(), 1e-12);
  EXPECT_LT(after.T_world_body.q.angularDistance(solved.T_world_body.q), 1e-15);
}
