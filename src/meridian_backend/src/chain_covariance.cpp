#include "chain_covariance.hpp"

#include <Eigen/Eigenvalues>

namespace meridian::backend {
namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d s;
  s << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return s;
}

// SE(3) adjoint in the translation-first [rho; phi] tangent, defined by
// T * Exp(xi) * T^{-1} = Exp(Ad_T * xi):  Ad_T = [[R, skew(t)*R], [0, R]].
Eigen::Matrix<double, 6, 6> adjoint(const Pose& T) {
  const Eigen::Matrix3d r = T.R();
  Eigen::Matrix<double, 6, 6> ad = Eigen::Matrix<double, 6, 6>::Zero();
  ad.topLeftCorner<3, 3>() = r;
  ad.topRightCorner<3, 3>() = skew(T.t) * r;
  ad.bottomRightCorner<3, 3>() = r;
  return ad;
}

// Symmetrize, then clamp negative eigenvalues to zero (the cancellation in `between` can
// leave the difference slightly indefinite).
Eigen::Matrix<double, 6, 6> clampPsd(const Eigen::Matrix<double, 6, 6>& s) {
  const Eigen::Matrix<double, 6, 6> sym = 0.5 * (s + s.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(sym);
  if (eig.eigenvalues().minCoeff() >= 0.0) {
    return sym;
  }
  const Eigen::Matrix<double, 6, 1> clamped = eig.eigenvalues().cwiseMax(0.0);
  const Eigen::Matrix<double, 6, 6> rebuilt =
      eig.eigenvectors() * clamped.asDiagonal() * eig.eigenvectors().transpose();
  return 0.5 * (rebuilt + rebuilt.transpose());
}

}  // namespace

void ChainCovariance::start(std::uint64_t first_id) {
  nodes_.clear();
  nodes_[first_id] = Node{Pose{}, Eigen::Matrix<double, 6, 6>::Zero()};
}

void ChainCovariance::extend(std::uint64_t from_id, std::uint64_t to_id, const Pose& T_from_to,
                             const Eigen::Matrix<double, 6, 6>& cov) {
  const auto it = nodes_.find(from_id);
  if (it == nodes_.end()) {
    return;
  }
  // Right-perturbation compose: the parent's uncertainty is transported into the child's
  // tangent by Ad(T_from_to^{-1}) before the independent edge noise is added.
  const Eigen::Matrix<double, 6, 6> ad = adjoint(T_from_to.inverse());
  Node node;
  node.T_chain = it->second.T_chain * T_from_to;
  node.cov_chain = ad * it->second.cov_chain * ad.transpose() + cov;
  nodes_[to_id] = node;
}

std::optional<Eigen::Matrix<double, 6, 6>> ChainCovariance::between(std::uint64_t a,
                                                                    std::uint64_t b) const {
  if (a >= b) {
    return std::nullopt;
  }
  const auto ia = nodes_.find(a);
  const auto ib = nodes_.find(b);
  if (ia == nodes_.end() || ib == nodes_.end()) {
    return std::nullopt;
  }
  // The chains to a and b share every edge up to a, so transporting cov_chain(a) into b's
  // tangent and subtracting leaves exactly the noise accumulated on the a->b segment.
  const Pose T_ab = ia->second.T_chain.inverse() * ib->second.T_chain;
  const Eigen::Matrix<double, 6, 6> ad = adjoint(T_ab.inverse());
  return clampPsd(ib->second.cov_chain - ad * ia->second.cov_chain * ad.transpose());
}

bool ChainCovariance::known(std::uint64_t id) const {
  return nodes_.count(id) != 0;
}

}  // namespace meridian::backend
