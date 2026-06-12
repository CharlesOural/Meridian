#include "gtsam_adapter.hpp"

#include <Eigen/Eigenvalues>

namespace meridian::backend {

gtsam::Pose3 to_gtsam(const Pose& T) {
  return gtsam::Pose3(gtsam::Rot3(T.q), T.t);
}

Pose from_gtsam(const gtsam::Pose3& T) {
  return Pose(T.rotation().toQuaternion(), T.translation());
}

const Eigen::Matrix<double, 6, 6>& P_meridian_gtsam() {
  static const Eigen::Matrix<double, 6, 6> p = [] {
    Eigen::Matrix<double, 6, 6> m = Eigen::Matrix<double, 6, 6>::Zero();
    m.block<3, 3>(0, 3).setIdentity();  // GTSAM rotation rows <- Meridian rotation cols
    m.block<3, 3>(3, 0).setIdentity();  // GTSAM translation rows <- Meridian translation cols
    return m;
  }();
  return p;
}

Eigen::Matrix<double, 6, 6> reorder_meridian_to_gtsam(const Eigen::Matrix<double, 6, 6>& S) {
  const Eigen::Matrix<double, 6, 6>& P = P_meridian_gtsam();
  return P * S * P.transpose();
}

Eigen::Matrix<double, 6, 6> reorder_gtsam_to_meridian(const Eigen::Matrix<double, 6, 6>& S) {
  const Eigen::Matrix<double, 6, 6>& P = P_meridian_gtsam();
  return P.transpose() * S * P;
}

bool ensure_psd(Eigen::Matrix<double, 6, 6>& S, double min_eig) {
  const Eigen::Matrix<double, 6, 6> sym = 0.5 * (S + S.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(sym);
  if (eig.eigenvalues().minCoeff() >= min_eig) {
    S = sym;
    return false;
  }
  const Eigen::Matrix<double, 6, 1> clamped = eig.eigenvalues().cwiseMax(min_eig);
  const Eigen::Matrix<double, 6, 6> rebuilt =
      eig.eigenvectors() * clamped.asDiagonal() * eig.eigenvectors().transpose();
  // The reconstruction is symmetric only up to roundoff; re-symmetrize exactly.
  S = 0.5 * (rebuilt + rebuilt.transpose());
  return true;
}

}  // namespace meridian::backend
