#pragma once

#include <array>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <basalt/utils/sophus_utils.hpp>
#include <sophus/so3.hpp>

namespace meridian::ct {

// Closed-form Jacobian support for the cumulative-form cubic SO(3) B-spline, shared
// by the analytic LiDAR and visual factors. The forward evaluation matches
// basalt::CeresSplineHelper::evaluate_lie; the Jacobians follow the cumulative-form
// chain rule (Sommer et al., CVPR 2020, with the sign corrections SLICT applies).

constexpr int kSplineJacOrder = 4;

struct So3SplineJac {
  // Evaluated rotation R(u) = R_0 * prod_{j=1..3} exp(lambda_R[j] * delta_j).
  Sophus::SO3d R;
  // dRt_dR[j] = d(right-tangent of R(u)) / d(right-tangent of knot j): a RIGHT
  // perturbation knot_j <- knot_j * exp(theta^) maps to R(u) <- R(u) * exp((dRt_dR[j]
  // theta)^) to first order.
  std::array<Eigen::Matrix3d, kSplineJacOrder> dRt_dR;
};

// Evaluates the rotation and its per-knot tangent Jacobians. lambda_R is the
// cumulative blending row (cumulative_blending_matrix_ * [1,u,u^2,u^3]^T).
inline So3SplineJac evaluateSo3CumulativeJac(
    const std::array<Sophus::SO3d, kSplineJacOrder>& knot, const Eigen::Vector4d& lambda_R) {
  using SO3 = Sophus::SO3d;
  std::array<Eigen::Vector3d, kSplineJacOrder> delta;
  std::array<SO3, kSplineJacOrder> A;
  delta[0] = knot[0].log();
  A[0] = knot[0];
  for (int j = 1; j < kSplineJacOrder; ++j) {
    delta[static_cast<std::size_t>(j)] =
        (knot[static_cast<std::size_t>(j - 1)].inverse() * knot[static_cast<std::size_t>(j)]).log();
    A[static_cast<std::size_t>(j)] = SO3::exp(lambda_R[j] * delta[static_cast<std::size_t>(j)]);
  }
  std::array<SO3, kSplineJacOrder> P;
  P[kSplineJacOrder - 1] = SO3();
  for (int j = kSplineJacOrder - 1; j >= 1; --j) {
    P[static_cast<std::size_t>(j - 1)] =
        P[static_cast<std::size_t>(j)] * A[static_cast<std::size_t>(j)].inverse();
  }

  So3SplineJac out;
  out.R = (P[0] * knot[0].inverse()).inverse();
  for (int j = 0; j < kSplineJacOrder; ++j) {
    const std::size_t sj = static_cast<std::size_t>(j);
    Eigen::Matrix3d jr_lam_j;
    Eigen::Matrix3d jr_inv_j;
    Sophus::rightJacobianSO3(lambda_R[j] * delta[sj], jr_lam_j);
    Sophus::rightJacobianInvSO3(delta[sj], jr_inv_j);
    Eigen::Matrix3d d = lambda_R[j] * P[sj].matrix() * jr_lam_j * jr_inv_j;
    if (j + 1 < kSplineJacOrder) {
      Eigen::Matrix3d jr_lam_n;
      Eigen::Matrix3d jr_inv_n;
      Sophus::rightJacobianSO3(lambda_R[j + 1] * delta[sj + 1], jr_lam_n);
      Sophus::rightJacobianInvSO3(delta[sj + 1], jr_inv_n);
      d -= lambda_R[j + 1] * P[sj + 1].matrix() * jr_lam_n * jr_inv_n.transpose();
    }
    out.dRt_dR[sj] = d;
  }
  return out;
}

// Maps a right-tangent row Jacobian to ambient quaternion (x, y, z, w) coordinates:
// for theta = Log(q^-1 q~) at q~ = q, d theta / d q~ = 2 [ q_w I - [q_v]_x | -q_v ].
// vec(q^-1 (x) q~) = q_w q~_v - q~_w q_v - q_v x q~_v, so the skew enters negated.
// Multiplying J_theta (rows x 3) by this gives the rows x 4 ambient Jacobian Ceres
// consumes on EigenQuaternionManifold blocks.
inline Eigen::Matrix<double, 3, 4> tangentToQuatJac(const Eigen::Quaterniond& q) {
  Eigen::Matrix<double, 3, 4> m;
  m.leftCols<3>() = q.w() * Eigen::Matrix3d::Identity() - Sophus::SO3d::hat(q.vec());
  m.col(3) = -q.vec();
  m *= 2.0;
  return m;
}

}  // namespace meridian::ct
