#include "ct/pose_marginal.hpp"

#include <ceres/problem.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <sophus/so3.hpp>
#include <unordered_set>
#include <vector>

#include "ct/marginalization.hpp"

namespace meridian::ct {

namespace {
constexpr int kSplineOrder = 4;
}  // namespace

bool robustInverse(const Eigen::MatrixXd& A, Eigen::MatrixXd* out) {
  const Eigen::MatrixXd sym = 0.5 * (A + A.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(sym);
  const Eigen::VectorXd& lambda = es.eigenvalues();
  const double lam_max = lambda.size() > 0 ? lambda.cwiseAbs().maxCoeff() : 0.0;
  const double floor = std::max(1e-12, 1e-9 * lam_max);
  Eigen::VectorXd inv(lambda.size());
  bool any = false;
  for (int i = 0; i < lambda.size(); ++i) {
    if (lambda(i) > floor) {
      inv(i) = 1.0 / lambda(i);
      any = true;
    } else {
      inv(i) = 0.0;
    }
  }
  if (!any) {
    return false;
  }
  *out = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
  return true;
}

bool poseMarginalFromProblem(ceres::Problem& problem, SplineWindow& spline, Timestamp stamp,
                             Eigen::Matrix<double, 6, 6>* cov) {
  if (!spline.covers(stamp)) {
    return false;
  }
  // The pose at stamp depends only on its segment's 4 SO(3) + 4 R^3 knots. Collect
  // them in a fixed tangent order (so3 then r3, 3 DoF each), and gather every residual
  // in the solved problem that touches any of them -- this is their Markov blanket and
  // carries the LiDAR, IMU, bias, and prior information that constrains the pose.
  const SplineWindow::SegmentRef seg = spline.segmentFor(stamp);
  std::array<double*, kSplineOrder * 2> pose_knots{};
  std::array<bool, kSplineOrder * 2> knot_is_so3{};
  for (int j = 0; j < kSplineOrder; ++j) {
    pose_knots[j] = seg.so3_knots[j];
    knot_is_so3[j] = true;
    pose_knots[kSplineOrder + j] = seg.r3_knots[j];
    knot_is_so3[kSplineOrder + j] = false;
  }

  std::vector<ceres::ResidualBlockId> residuals;
  std::unordered_set<const void*> res_seen;
  for (double* p : pose_knots) {
    if (!problem.HasParameterBlock(p)) {
      return false;  // a pose knot is missing from the problem: cannot trust the block
    }
    std::vector<ceres::ResidualBlockId> rs;
    problem.GetResidualBlocksForParameterBlock(p, &rs);
    for (ceres::ResidualBlockId r : rs) {
      if (res_seen.insert(r).second) {
        residuals.push_back(r);
      }
    }
  }
  if (residuals.empty()) {
    return false;
  }

  // Order the blanket with the 8 pose knots first (free, never gauge-pinned at the
  // window's trailing edge), then the remaining free blocks the residuals reach.
  // Constant gauge pins carry no information and are dropped from the order.
  auto makeBlock = [&](double* p) {
    MarginalizationPrior::Block b;
    b.ptr = p;
    b.global_size = problem.ParameterBlockSize(p);
    b.local_size = problem.ParameterBlockTangentSize(p);
    return b;
  };
  std::unordered_set<double*> pose_set(pose_knots.begin(), pose_knots.end());
  std::vector<MarginalizationPrior::Block> order;
  int pose_dim = 0;
  for (double* p : pose_knots) {
    if (problem.IsParameterBlockConstant(p)) {
      return false;  // a pinned pose knot has no posterior spread; defer to the fallback
    }
    order.push_back(makeBlock(p));
    pose_dim += order.back().local_size;
  }
  std::unordered_set<double*> seen(pose_set);
  for (ceres::ResidualBlockId r : residuals) {
    std::vector<double*> ps;
    problem.GetParameterBlocksForResidualBlock(r, &ps);
    for (double* p : ps) {
      if (seen.count(p) || problem.IsParameterBlockConstant(p)) {
        continue;
      }
      seen.insert(p);
      order.push_back(makeBlock(p));
    }
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  linearizeBlocks(problem, residuals, order, &H, &b);

  // Joint marginal covariance of the blanket; the pose-knot marginal is its leading
  // pose_dim x pose_dim block (a sub-block of the joint covariance is the marginal).
  Eigen::MatrixXd cov_full;
  if (!robustInverse(H, &cov_full)) {
    return false;
  }
  const Eigen::MatrixXd cov_knots = cov_full.topLeftCorner(pose_dim, pose_dim);

  // Spline pose Jacobian J (6 x pose_dim) at stamp, by central finite difference of
  // T_W_Fe(stamp) under each knot's tangent perturbation, in the same knot ordering.
  // The pose tangent is the decoupled translation-first chart [delta t_world; delta
  // phi_body] that the LiDAR Jacobian and the rotation-first reorder both assume.
  const Pose T0 = spline.pose(stamp);
  const Eigen::Quaterniond q0_inv = T0.q.conjugate();
  Eigen::MatrixXd Jpose = Eigen::MatrixXd::Zero(6, pose_dim);
  const double eps = 1e-6;
  int col = 0;
  for (int k = 0; k < kSplineOrder * 2; ++k) {
    double* ptr = pose_knots[k];
    if (knot_is_so3[k]) {
      Eigen::Map<Eigen::Quaterniond> qk(ptr);
      const Eigen::Quaterniond saved = qk;
      for (int a = 0; a < 3; ++a) {
        Eigen::Vector3d d = Eigen::Vector3d::Zero();
        d(a) = eps;
        qk = (saved * Sophus::SO3d::exp(d).unit_quaternion()).normalized();
        const Pose Tp = spline.pose(stamp);
        qk = (saved * Sophus::SO3d::exp(-d).unit_quaternion()).normalized();
        const Pose Tm = spline.pose(stamp);
        qk = saved;
        // Body-frame right-perturbation tangent of each sample relative to T0, then a
        // central difference: phi_+- = Log(q0^{-1} q_+-), J = (phi_+ - phi_-)/(2 eps).
        const Eigen::Vector3d phi_p = Sophus::SO3d((q0_inv * Tp.q).normalized()).log();
        const Eigen::Vector3d phi_m = Sophus::SO3d((q0_inv * Tm.q).normalized()).log();
        Jpose.block<3, 1>(0, col) = (Tp.t - Tm.t) / (2.0 * eps);
        Jpose.block<3, 1>(3, col) = (phi_p - phi_m) / (2.0 * eps);
        ++col;
      }
    } else {
      Eigen::Map<Eigen::Vector3d> pk(ptr);
      const Eigen::Vector3d saved = pk;
      for (int a = 0; a < 3; ++a) {
        pk = saved;
        pk(a) = saved(a) + eps;
        const Pose Tp = spline.pose(stamp);
        pk(a) = saved(a) - eps;
        const Pose Tm = spline.pose(stamp);
        pk = saved;
        const Eigen::Vector3d phi_p = Sophus::SO3d((q0_inv * Tp.q).normalized()).log();
        const Eigen::Vector3d phi_m = Sophus::SO3d((q0_inv * Tm.q).normalized()).log();
        Jpose.block<3, 1>(0, col) = (Tp.t - Tm.t) / (2.0 * eps);
        Jpose.block<3, 1>(3, col) = (phi_p - phi_m) / (2.0 * eps);
        ++col;
      }
    }
  }

  *cov = Jpose * cov_knots * Jpose.transpose();
  *cov = 0.5 * (*cov + cov->transpose()).eval();
  return true;
}

}  // namespace meridian::ct
