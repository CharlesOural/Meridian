#include <gtest/gtest.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/imu_preintegration.hpp"
#include "meridian/config/config.hpp"
#include "pim_from_summary.hpp"

// Regression pinning the [dR | dv | dp] -> [theta | p | v] permutation. The summary is
// produced by an independent closed-form integration of a constant-rate IMU segment; the
// rebuilt PIM must, under predict, reproduce that same segment driven forward with gravity.
// A wrong v/p swap diverges position and velocity immediately.

namespace {

using gtsam::NavState;
using gtsam::Rot3;
using meridian::BackendImu;
using meridian::GaussianBlock;
using meridian::ImuPreintegrationSummary;
using meridian::backend::pim_from_summary;

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr int kN = 40;
constexpr double kDt = 0.005;  // [s] per IMU step -> 0.2 s window
constexpr double kGravity = 9.81;

Mat3 expmap(const Vec3& phi) {
  const double n = phi.norm();
  if (n == 0.0) {
    return Mat3::Identity();
  }
  return Mat3(Eigen::AngleAxisd(n, phi / n));
}

// Result of one closed-form integration: the gravity-free preintegrated increment in the
// summary's [dR | dv | dp] order. omega and accel are the bias-corrected body-frame inputs
// (the true motion seen after subtracting the linearization-point bias).
struct Increment {
  Mat3 dR = Mat3::Identity();
  Vec3 dv = Vec3::Zero();
  Vec3 dp = Vec3::Zero();
};

// Forward-Euler integration of the gravity-free increment on the manifold, matching the
// tangent preintegration recurrence. Independent of the production permutation.
Increment integrate(const Vec3& omega, const Vec3& accel) {
  Increment inc;
  const Mat3 step = expmap(omega * kDt);
  for (int k = 0; k < kN; ++k) {
    const Mat3 dR_k = inc.dR;
    const Vec3 acc_i = dR_k * accel;  // specific force rotated into the i-frame
    inc.dp += inc.dv * kDt + 0.5 * acc_i * kDt * kDt;
    inc.dv += acc_i * kDt;
    inc.dR = dR_k * step;
  }
  return inc;
}

// First-order bias Jacobians by central finite differences of integrate() w.r.t. the
// bias-corrected inputs. A gyro-bias perturbation db_g subtracts from omega; an accel-bias
// perturbation db_a subtracts from accel (measured = true + bias, so corrected = meas - b).
struct BiasJacobians {
  Mat3 dR_dbg = Mat3::Zero();
  Mat3 dv_dbg = Mat3::Zero();
  Mat3 dv_dba = Mat3::Zero();
  Mat3 dp_dbg = Mat3::Zero();
  Mat3 dp_dba = Mat3::Zero();
};

BiasJacobians jacobians(const Vec3& omega, const Vec3& accel) {
  BiasJacobians J;
  const double eps = 1e-6;
  const Increment base = integrate(omega, accel);
  for (int i = 0; i < 3; ++i) {
    Vec3 e = Vec3::Zero();
    e[i] = eps;
    // d/d(b_g): corrected omega moves by -db_g, so perturb omega by -e.
    const Increment gp = integrate(omega - e, accel);
    const Increment gm = integrate(omega + e, accel);
    J.dR_dbg.col(i) = Rot3::Logmap(Rot3(base.dR.transpose() * gp.dR)) -
                      Rot3::Logmap(Rot3(base.dR.transpose() * gm.dR));
    J.dR_dbg.col(i) /= (2 * eps);
    J.dv_dbg.col(i) = (gp.dv - gm.dv) / (2 * eps);
    J.dp_dbg.col(i) = (gp.dp - gm.dp) / (2 * eps);
    // d/d(b_a): corrected accel moves by -db_a, so perturb accel by -e.
    const Increment ap = integrate(omega, accel - e);
    const Increment am = integrate(omega, accel + e);
    J.dv_dba.col(i) = (ap.dv - am.dv) / (2 * eps);
    J.dp_dba.col(i) = (ap.dp - am.dp) / (2 * eps);
  }
  return J;
}

// Builds a self-consistent summary with the chosen per-block covariance diagonal in
// [dR | dv | dp] order. The motion is fixed and non-degenerate so position and velocity
// carry distinct signatures.
ImuPreintegrationSummary makeSummary(double cov_dR = 1e-4, double cov_dv = 1e-3,
                                     double cov_dp = 1e-2) {
  const Vec3 omega(0.08, -0.05, 0.11);  // [rad/s] bias-corrected body rate
  const Vec3 accel(0.4, -0.2, 0.3);     // [m/s^2] bias-corrected specific force

  const Increment inc = integrate(omega, accel);
  const BiasJacobians J = jacobians(omega, accel);

  ImuPreintegrationSummary s;
  s.t_i = 1'000'000'000LL;
  s.t_j = s.t_i + static_cast<meridian::Timestamp>(kN * kDt * 1e9);
  s.gravity_mag = kGravity;
  s.bias_g_lin = Vec3(0.01, -0.02, 0.03);
  s.bias_a_lin = Vec3(-0.05, 0.04, -0.01);
  s.delta_R = Eigen::Quaterniond(inc.dR).normalized();
  s.delta_v = inc.dv;
  s.delta_p = inc.dp;
  s.dR_dbg = J.dR_dbg;
  s.dv_dbg = J.dv_dbg;
  s.dv_dba = J.dv_dba;
  s.dp_dbg = J.dp_dbg;
  s.dp_dba = J.dp_dba;
  s.preint_cov.form = GaussianBlock<9>::Form::Covariance;
  s.preint_cov.M.setZero();
  for (int k = 0; k < 3; ++k) s.preint_cov.M(k, k) = cov_dR;
  for (int k = 3; k < 6; ++k) s.preint_cov.M(k, k) = cov_dv;
  for (int k = 6; k < 9; ++k) s.preint_cov.M(k, k) = cov_dp;
  return s;
}

BackendImu makeImu() {
  BackendImu imu;
  imu.acc_noise = 0.02;
  imu.gyr_noise = 0.001;
  imu.acc_bias_rw = 1e-3;
  imu.gyr_bias_rw = 1e-4;
  return imu;
}

}  // namespace

TEST(PimFromSummary, PredictMatchesClosedForm) {
  const ImuPreintegrationSummary s = makeSummary();
  const BackendImu imu = makeImu();
  const auto pim = pim_from_summary(s, imu);

  const gtsam::imuBias::ConstantBias bias(s.bias_a_lin, s.bias_g_lin);
  const Rot3 R_i(Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Vec3(0.2, -0.4, 0.9).normalized())));
  const Vec3 p_i(2.0, -1.0, 0.5);
  const Vec3 v_i(0.7, 0.2, -0.3);
  const NavState state_i(R_i, p_i, v_i);

  const NavState state_j = pim.predict(state_i, bias);

  // Independent closed-form forward integration with the params' gravity (n_gravity points
  // down). The deltas are taken at biasHat, so no bias correction is applied.
  const double dt = static_cast<double>(s.t_j - s.t_i) * 1e-9;
  const Vec3 g(0.0, 0.0, -kGravity);
  const Mat3 Ri = R_i.matrix();
  const Mat3 dR = s.delta_R.toRotationMatrix();

  const Mat3 Rj_expected = Ri * dR;
  const Vec3 pj_expected = p_i + v_i * dt + 0.5 * g * dt * dt + Ri * s.delta_p;
  const Vec3 vj_expected = v_i + g * dt + Ri * s.delta_v;

  EXPECT_LT((state_j.pose().rotation().matrix() - Rj_expected).norm(), 1e-6);
  EXPECT_LT((state_j.position() - pj_expected).norm(), 1e-6);
  EXPECT_LT((state_j.velocity() - vj_expected).norm(), 1e-6);
}

TEST(PimFromSummary, BiasPerturbationUsesJacobians) {
  const ImuPreintegrationSummary s = makeSummary();
  const BackendImu imu = makeImu();
  const auto pim = pim_from_summary(s, imu);

  const Rot3 R_i(Eigen::Quaterniond(Eigen::AngleAxisd(0.2, Vec3(0.1, 0.3, 0.8).normalized())));
  const Vec3 p_i(1.0, 0.0, -0.5);
  const Vec3 v_i(0.3, -0.2, 0.1);
  const NavState state_i(R_i, p_i, v_i);
  const Mat3 Ri = R_i.matrix();

  const gtsam::imuBias::ConstantBias bias0(s.bias_a_lin, s.bias_g_lin);
  const NavState base = pim.predict(state_i, bias0);

  const Vec3 db_a(2e-3, -1e-3, 5e-4);
  const Vec3 db_g(-1e-3, 5e-4, 2e-3);
  const gtsam::imuBias::ConstantBias bias1(s.bias_a_lin + db_a, s.bias_g_lin + db_g);
  const NavState pert = pim.predict(state_i, bias1);

  // First-order corrected-increment shifts from the summary's Jacobians, rotated into the
  // i-frame the same way predict composes them onto the base pose.
  const Vec3 d_dp = s.dp_dbg * db_g + s.dp_dba * db_a;
  const Vec3 d_dv = s.dv_dbg * db_g + s.dv_dba * db_a;
  const Vec3 d_dphi = s.dR_dbg * db_g;

  const Vec3 dp_expected = Ri * d_dp;
  const Vec3 dv_expected = Ri * d_dv;

  // First-order: residual scales with |db|^2, so a loose absolute bound suffices.
  EXPECT_LT(((pert.position() - base.position()) - dp_expected).norm(), 1e-4);
  EXPECT_LT(((pert.velocity() - base.velocity()) - dv_expected).norm(), 1e-4);

  // Rotation shift: the right-perturbation between the two predicted attitudes equals the
  // summary's gyro-bias rotation Jacobian applied to db_g, to first order. The right-
  // Jacobian of the small base rotation contributes the leading correction, so the bound
  // is looser than the additive position/velocity ones.
  const Vec3 dphi_actual = Rot3::Logmap(
      Rot3(base.pose().rotation().matrix().transpose() * pert.pose().rotation().matrix()));
  EXPECT_LT((dphi_actual - d_dphi).norm(), 5e-4);
}

TEST(PimFromSummary, CovIsSpdAndOrdered) {
  // Distinct per-block diagonals so the v/p swap is observable in the assembled cov.
  constexpr double kVarR = 3e-4;
  constexpr double kVarV = 7e-3;
  constexpr double kVarP = 5e-2;
  const ImuPreintegrationSummary s = makeSummary(kVarR, kVarV, kVarP);
  const BackendImu imu = makeImu();
  const auto pim = pim_from_summary(s, imu);

  const Eigen::Matrix<double, 15, 15> cov = pim.preintMeasCov();

  EXPECT_LT((cov - cov.transpose()).norm(), 1e-12);
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eig(cov);
  EXPECT_GT(eig.eigenvalues().minCoeff(), 0.0);

  // GTSAM order [theta | p | v | ...]: the position block (3:6) carries the summary's dp
  // variance and the velocity block (6:9) carries its dv variance, confirming the swap.
  EXPECT_NEAR(cov(0, 0), kVarR, 1e-12);
  EXPECT_NEAR(cov(3, 3), kVarP, 1e-12);
  EXPECT_NEAR(cov(6, 6), kVarV, 1e-12);

  const double dt = static_cast<double>(s.t_j - s.t_i) * 1e-9;
  EXPECT_NEAR(cov(9, 9), imu.acc_bias_rw * imu.acc_bias_rw * dt, 1e-15);
  EXPECT_NEAR(cov(12, 12), imu.gyr_bias_rw * imu.gyr_bias_rw * dt, 1e-15);
}
