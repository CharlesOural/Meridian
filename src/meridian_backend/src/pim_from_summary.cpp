#include "pim_from_summary.hpp"

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/TangentPreintegration.h>

#include <Eigen/Eigenvalues>
#include <boost/shared_ptr.hpp>

namespace meridian::backend {

namespace {

using Mat9 = Eigen::Matrix<double, 9, 9>;
using Mat15 = Eigen::Matrix<double, 15, 15>;
using Mat93 = Eigen::Matrix<double, 9, 3>;

// 9x9 conjugation P * S * P^T where P swaps the second and third 3-blocks, mapping the
// summary order [dR | dv | dp] to the GTSAM tangent order [theta | p | v]. The rotation
// block stays put; velocity and position rows and columns trade places.
Mat9 swap_vp_blocks(const Mat9& s) {
  Mat9 out;
  // Row/column block index map: 0->0, 1(v)->2, 2(p)->1.
  constexpr int perm[3] = {0, 2, 1};
  for (int bi = 0; bi < 3; ++bi) {
    for (int bj = 0; bj < 3; ++bj) {
      out.block<3, 3>(3 * perm[bi], 3 * perm[bj]) = s.block<3, 3>(3 * bi, 3 * bj);
    }
  }
  return out;
}

// Symmetrize, then raise any eigenvalue below min_eig up to it so the result is strictly
// SPD (the summary may arrive as information and the inverse can be marginally indefinite).
Mat9 to_spd(const Mat9& s, double min_eig) {
  const Mat9 sym = 0.5 * (s + s.transpose());
  const Eigen::SelfAdjointEigenSolver<Mat9> eig(sym);
  if (eig.eigenvalues().minCoeff() >= min_eig) {
    return sym;
  }
  const Eigen::Matrix<double, 9, 1> clamped = eig.eigenvalues().cwiseMax(min_eig);
  const Mat9 rebuilt = eig.eigenvectors() * clamped.asDiagonal() * eig.eigenvectors().transpose();
  return 0.5 * (rebuilt + rebuilt.transpose());
}

// Subclass that exposes write access to the protected preintegrated state. It is only ever
// filled from a completed summary, never incrementally integrated, so the params it holds
// matter solely for predict's gravity/Coriolis terms.
class SummaryPreintegration : public gtsam::TangentPreintegration {
public:
  explicit SummaryPreintegration(const boost::shared_ptr<gtsam::PreintegrationParams>& params)
      : gtsam::TangentPreintegration(params, gtsam::imuBias::ConstantBias()) {}

  void fill(const ImuPreintegrationSummary& s, double dt) {
    biasHat_ = gtsam::imuBias::ConstantBias(s.bias_a_lin, s.bias_g_lin);
    deltaTij_ = dt;

    // preintegrated_ = [theta ; p ; v]  (rotation, position, velocity).
    preintegrated_.head<3>() = gtsam::Rot3::Logmap(gtsam::Rot3(s.delta_R));
    preintegrated_.segment<3>(3) = s.delta_p;
    preintegrated_.tail<3>() = s.delta_v;

    // Bias Jacobians in the same [theta ; p ; v] row order. Rotation has no dependence on
    // the accelerometer bias, so its accel-bias rows are zero.
    preintegrated_H_biasOmega_.block<3, 3>(0, 0) = s.dR_dbg;
    preintegrated_H_biasOmega_.block<3, 3>(3, 0) = s.dp_dbg;
    preintegrated_H_biasOmega_.block<3, 3>(6, 0) = s.dv_dbg;

    preintegrated_H_biasAcc_.block<3, 3>(0, 0).setZero();
    preintegrated_H_biasAcc_.block<3, 3>(3, 0) = s.dp_dba;
    preintegrated_H_biasAcc_.block<3, 3>(6, 0) = s.dv_dba;
  }
};

}  // namespace

gtsam::PreintegratedCombinedMeasurements pim_from_summary(const ImuPreintegrationSummary& s,
                                                          const BackendImu& imu) {
  const double dt = static_cast<double>(s.t_j - s.t_i) * 1e-9;

  // The map is gravity-aligned ENU, so gravity points along -z (the "U" sense). The
  // continuous-time covariances are per-axis isotropic: accelerometer/gyroscope from the
  // noise densities squared, bias random walks from their density squared; integration and
  // bias-init terms are tiny regularizers that keep predict and the params object well posed.
  auto params = gtsam::PreintegrationCombinedParams::MakeSharedU(s.gravity_mag);
  params->accelerometerCovariance = imu.acc_noise * imu.acc_noise * gtsam::Matrix3::Identity();
  params->gyroscopeCovariance = imu.gyr_noise * imu.gyr_noise * gtsam::Matrix3::Identity();
  params->biasAccCovariance = imu.acc_bias_rw * imu.acc_bias_rw * gtsam::Matrix3::Identity();
  params->biasOmegaCovariance = imu.gyr_bias_rw * imu.gyr_bias_rw * gtsam::Matrix3::Identity();
  params->integrationCovariance = 1e-8 * gtsam::Matrix3::Identity();
  params->biasAccOmegaInt = 1e-5 * gtsam::Matrix6::Identity();

  SummaryPreintegration base(params);
  base.fill(s, dt);

  // Measurement covariance in GTSAM order [theta | p | v | bias_acc | bias_omega]. The 9x9
  // preintegrated block is the summary's covariance with v/p swapped into place; the bias
  // blocks accumulate the random walk over the interval; preint<->bias cross blocks are
  // left zero (the conservative drop).
  Mat9 preint9 = s.preint_cov.M;
  if (s.preint_cov.form == GaussianBlock<9>::Form::Information) {
    preint9 = preint9.inverse();
  }
  const Mat9 preint_tpv = to_spd(swap_vp_blocks(preint9), 1e-12);

  Mat15 cov15 = Mat15::Zero();
  cov15.topLeftCorner<9, 9>() = preint_tpv;
  cov15.block<3, 3>(9, 9) = imu.acc_bias_rw * imu.acc_bias_rw * dt * gtsam::Matrix3::Identity();
  cov15.block<3, 3>(12, 12) = imu.gyr_bias_rw * imu.gyr_bias_rw * dt * gtsam::Matrix3::Identity();
  cov15 = 0.5 * (cov15 + cov15.transpose()).eval();

  return gtsam::PreintegratedCombinedMeasurements(
      static_cast<const gtsam::PreintegrationType&>(base), cov15);
}

}  // namespace meridian::backend
