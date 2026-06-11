#pragma once

#include <basalt/spline/ceres_spline_helper.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <basalt/utils/sophus_utils.hpp>
#include <cmath>
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

// Body angular velocity of the cumulative SO(3) spline at one time, with its per-knot
// right-tangent Jacobian. The velocity is omega = sum_{j=1..3} M_j (dcoeff[j] delta_j),
// delta_j = Log(R_{j-1}^-1 R_j), A_j = exp(lambda_R[j] delta_j), and M_j the prefix
// adjoint product Adj(A_3^-1) ... Adj(A_{j+1}^-1) (M_3 = I). dOmega_dR[j] is the
// derivative w.r.t. a RIGHT perturbation knot_j <- knot_j exp(theta^). The first
// contribution is through the dcoeff[j] delta_j terms (delta_j depends on knots j and
// j-1); the second is through the A_k adjoints inside the prefix products.
struct So3VelocityJac {
  Eigen::Vector3d omega;
  std::array<Eigen::Matrix3d, kSplineJacOrder> dOmega_dR;
};

inline So3VelocityJac evaluateSo3VelocityJac(
    const std::array<Sophus::SO3d, kSplineJacOrder>& knot,
    const Eigen::Vector4d& lambda_R, const Eigen::Vector4d& dcoeff) {
  using SO3 = Sophus::SO3d;
  std::array<Eigen::Vector3d, kSplineJacOrder> delta;
  std::array<SO3, kSplineJacOrder> A;
  for (int i = 1; i < kSplineJacOrder; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    delta[si] = (knot[si - 1].inverse() * knot[si]).log();
    A[si] = SO3::exp(lambda_R[i] * delta[si]);
  }
  // Prefix adjoints M_i = prod_{m=i+1..3} Adj(A_m^-1), M_3 = I.
  std::array<Eigen::Matrix3d, kSplineJacOrder> M;
  M[kSplineJacOrder - 1].setIdentity();
  for (int i = kSplineJacOrder - 2; i >= 1; --i) {
    const std::size_t si = static_cast<std::size_t>(i);
    M[si] = M[si + 1] * A[si + 1].inverse().Adj();
  }
  So3VelocityJac out;
  out.omega.setZero();
  for (int i = 1; i < kSplineJacOrder; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    out.omega += M[si] * (dcoeff[i] * delta[si]);
  }
  for (auto& m : out.dOmega_dR) m.setZero();
  auto jr_inv = [](const Eigen::Vector3d& d) {
    Eigen::Matrix3d J;
    Sophus::rightJacobianInvSO3(d, J);
    return J;
  };
  // Path A: through dcoeff_i * delta_i. delta_i depends on knot i (right, +Jr^-1) and
  // knot i-1 (left and inverted, -Jr^-1^T).
  for (int i = 1; i < kSplineJacOrder; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    const Eigen::Matrix3d Ji = jr_inv(delta[si]);
    out.dOmega_dR[si] += M[si] * (dcoeff[i] * Ji);
    out.dOmega_dR[si - 1] += M[si] * (dcoeff[i] * (-Ji.transpose()));
  }
  // Path B: through the adjoints A_k inside the prefix products (k = 2..3).
  for (int k = 2; k < kSplineJacOrder; ++k) {
    const std::size_t sk = static_cast<std::size_t>(k);
    Eigen::Vector3d S = Eigen::Vector3d::Zero();
    for (int i = 1; i < k; ++i) {
      const std::size_t si = static_cast<std::size_t>(i);
      Eigen::Matrix3d inner = Eigen::Matrix3d::Identity();
      for (int m = i + 1; m <= k - 1; ++m)
        inner = inner * A[static_cast<std::size_t>(m)].inverse().Adj();
      S += inner * (dcoeff[i] * delta[si]);
    }
    const Eigen::Vector3d phi = lambda_R[k] * delta[sk];
    Eigen::Matrix3d Jr_neg;
    Sophus::rightJacobianSO3(Eigen::Vector3d(-phi), Jr_neg);
    const Eigen::Matrix3d dvec_dphi =
        A[sk].inverse().Adj() * Sophus::SO3d::hat(S) * Jr_neg;
    const Eigen::Matrix3d Jk = jr_inv(delta[sk]);
    out.dOmega_dR[sk] += M[sk] * dvec_dphi * (lambda_R[k] * Jk);
    out.dOmega_dR[sk - 1] += M[sk] * dvec_dphi * (lambda_R[k] * (-Jk.transpose()));
  }
  return out;
}

// Body angular velocity and acceleration of the cumulative SO(3) spline at one time,
// each with per-knot right-tangent Jacobians. Forward recursion matches
// evaluate_lie; the Jacobians are first derived w.r.t. a LEFT perturbation of each
// knot (knot_k <- exp(theta^) knot_k) and converted to the RIGHT convention via
// J_right = J_left * R_knot, since exp(theta_L^) R = R exp((R^-1 theta_L)^). The
// velocity-only path zero-initializes dalpha_dR so the unused acceleration block is
// never returned uninitialized.
struct So3DerivJac {
  Eigen::Vector3d omega;
  Eigen::Vector3d alpha;
  std::array<Eigen::Matrix3d, kSplineJacOrder> domega_dR;
  std::array<Eigen::Matrix3d, kSplineJacOrder> dalpha_dR;
};

inline So3DerivJac evaluateSo3VelAccelJac(const std::array<Sophus::SO3d, kSplineJacOrder>& knot,
                                          double u, double inv_dt, bool want_accel,
                                          const Eigen::Matrix4d* cum_blend = nullptr) {
  using SO3 = Sophus::SO3d;
  using Mat3 = Eigen::Matrix3d;
  using Vec3 = Eigen::Vector3d;
  constexpr int N = kSplineJacOrder;
  constexpr int DEG = N - 1;
  using VecN = Eigen::Matrix<double, N, 1>;
  using Helper = basalt::CeresSplineHelper<N>;

  // A null cum_blend binds the uniform cumulative matrix object itself, so the
  // coefficient expressions below run on the same operands as always; a non-null
  // pointer substitutes a per-interval (non-uniform) matrix into the same
  // expressions unchanged.
  const Eigen::Matrix4d& C = cum_blend ? *cum_blend : Helper::cumulative_blending_matrix_;

  // Zero-init: baseCoeffsWithTime takes its output by const-ref (and const_casts
  // internally), which GCC's flow analysis cannot see through once inlined into a
  // Jet-instantiated caller; the helper overwrites every entry, so this is free.
  VecN p = VecN::Zero();
  VecN coeff, dcoeff, ddcoeff;
  Helper::template baseCoeffsWithTime<0>(p, u);
  coeff = C * p;
  Helper::template baseCoeffsWithTime<1>(p, u);
  dcoeff = inv_dt * C * p;
  if (want_accel) {
    Helper::template baseCoeffsWithTime<2>(p, u);
    ddcoeff = inv_dt * inv_dt * C * p;
  }

  So3DerivJac out;
  for (int i = 0; i < N; ++i) {
    out.domega_dR[static_cast<std::size_t>(i)].setZero();
    out.dalpha_dR[static_cast<std::size_t>(i)].setZero();
  }
  out.omega.setZero();
  out.alpha.setZero();

  if (!want_accel) {
    Vec3 delta_vec[DEG];
    Mat3 R_tmp[DEG];
    SO3 exp_k_delta[DEG];
    Mat3 Jr_delta_inv[DEG];
    Mat3 Jr_kdelta[DEG];
    SO3 accum;
    for (int i = DEG - 1; i >= 0; --i) {
      const SO3 r01 = knot[static_cast<std::size_t>(i)].inverse() *
                      knot[static_cast<std::size_t>(i + 1)];
      delta_vec[i] = r01.log();
      Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
      Jr_delta_inv[i] *= knot[static_cast<std::size_t>(i + 1)].inverse().matrix();
      const Vec3 k_delta = coeff[i + 1] * delta_vec[i];
      Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);
      R_tmp[i] = accum.matrix();
      exp_k_delta[i] = SO3::exp(-k_delta);
      accum *= exp_k_delta[i];
    }
    Mat3 d_vel_d_delta[DEG];
    d_vel_d_delta[0] = dcoeff[1] * R_tmp[0] * Jr_delta_inv[0];
    Vec3 rot_vel = delta_vec[0] * dcoeff[1];
    for (int i = 1; i < DEG; ++i) {
      d_vel_d_delta[i] = R_tmp[i - 1] * SO3::hat(rot_vel) * Jr_kdelta[i] * coeff[i + 1] +
                         R_tmp[i] * dcoeff[i + 1];
      d_vel_d_delta[i] *= Jr_delta_inv[i];
      rot_vel = exp_k_delta[i] * rot_vel + delta_vec[i] * dcoeff[i + 1];
    }
    std::array<Mat3, N> d_left;
    for (int i = 0; i < N; ++i) d_left[static_cast<std::size_t>(i)].setZero();
    for (int i = 0; i < DEG; ++i) {
      d_left[static_cast<std::size_t>(i)] -= d_vel_d_delta[i];
      d_left[static_cast<std::size_t>(i + 1)] += d_vel_d_delta[i];
    }
    out.omega = rot_vel;
    for (int k = 0; k < N; ++k)
      out.domega_dR[static_cast<std::size_t>(k)] =
          d_left[static_cast<std::size_t>(k)] * knot[static_cast<std::size_t>(k)].matrix();
    return out;
  }

  // Acceleration path (also yields the velocity Jacobian).
  Vec3 delta_vec[DEG];
  Mat3 exp_k_delta[DEG];
  Mat3 Jr_delta_inv[DEG];
  Mat3 Jr_kdelta[DEG];
  Vec3 rot_vel = Vec3::Zero();
  Vec3 rot_accel = Vec3::Zero();
  Vec3 rot_vel_arr[DEG];
  Vec3 rot_accel_arr[DEG];
  for (int i = 0; i < DEG; ++i) {
    const SO3 r01 = knot[static_cast<std::size_t>(i)].inverse() *
                    knot[static_cast<std::size_t>(i + 1)];
    delta_vec[i] = r01.log();
    Sophus::rightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
    Jr_delta_inv[i] *= knot[static_cast<std::size_t>(i + 1)].inverse().matrix();
    const Vec3 k_delta = coeff[i + 1] * delta_vec[i];
    Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);
    exp_k_delta[i] = SO3::exp(-k_delta).matrix();
    rot_vel = exp_k_delta[i] * rot_vel;
    const Vec3 vel_current = dcoeff[i + 1] * delta_vec[i];
    rot_vel += vel_current;
    rot_accel = exp_k_delta[i] * rot_accel;
    rot_accel += ddcoeff[i + 1] * delta_vec[i] + rot_vel.cross(vel_current);
    rot_vel_arr[i] = rot_vel;
    rot_accel_arr[i] = rot_accel;
  }
  Mat3 d_accel_d_delta[DEG];
  Mat3 d_vel_d_delta[DEG];
  d_vel_d_delta[DEG - 1] =
      coeff[DEG] * exp_k_delta[DEG - 1] * SO3::hat(rot_vel_arr[DEG - 2]) * Jr_kdelta[DEG - 1] +
      Mat3::Identity() * dcoeff[DEG];
  d_accel_d_delta[DEG - 1] =
      coeff[DEG] * exp_k_delta[DEG - 1] * SO3::hat(rot_accel_arr[DEG - 2]) * Jr_kdelta[DEG - 1] +
      Mat3::Identity() * ddcoeff[DEG] +
      dcoeff[DEG] * (SO3::hat(rot_vel_arr[DEG - 1]) -
                     SO3::hat(delta_vec[DEG - 1]) * d_vel_d_delta[DEG - 1]);
  Mat3 pj = Mat3::Identity();
  Vec3 sj = Vec3::Zero();
  for (int i = DEG - 2; i >= 0; --i) {
    sj += dcoeff[i + 2] * pj * delta_vec[i + 1];
    pj *= exp_k_delta[i + 1];
    d_vel_d_delta[i] = Mat3::Identity() * dcoeff[i + 1];
    if (i >= 1)
      d_vel_d_delta[i] += coeff[i + 1] * exp_k_delta[i] * SO3::hat(rot_vel_arr[i - 1]) * Jr_kdelta[i];
    d_accel_d_delta[i] =
        Mat3::Identity() * ddcoeff[i + 1] +
        dcoeff[i + 1] * (SO3::hat(rot_vel_arr[i]) - SO3::hat(delta_vec[i]) * d_vel_d_delta[i]);
    if (i >= 1)
      d_accel_d_delta[i] += coeff[i + 1] * exp_k_delta[i] * SO3::hat(rot_accel_arr[i - 1]) * Jr_kdelta[i];
    d_vel_d_delta[i] = pj * d_vel_d_delta[i];
    d_accel_d_delta[i] = pj * d_accel_d_delta[i] - SO3::hat(sj) * d_vel_d_delta[i];
  }
  std::array<Mat3, N> dl_a, dl_v;
  for (int i = 0; i < N; ++i) {
    dl_a[static_cast<std::size_t>(i)].setZero();
    dl_v[static_cast<std::size_t>(i)].setZero();
  }
  for (int i = 0; i < DEG; ++i) {
    const Mat3 va = d_accel_d_delta[i] * Jr_delta_inv[i];
    const Mat3 vv = d_vel_d_delta[i] * Jr_delta_inv[i];
    dl_a[static_cast<std::size_t>(i)] -= va;
    dl_a[static_cast<std::size_t>(i + 1)] += va;
    dl_v[static_cast<std::size_t>(i)] -= vv;
    dl_v[static_cast<std::size_t>(i + 1)] += vv;
  }
  out.omega = rot_vel;
  out.alpha = rot_accel;
  for (int k = 0; k < N; ++k) {
    out.domega_dR[static_cast<std::size_t>(k)] =
        dl_v[static_cast<std::size_t>(k)] * knot[static_cast<std::size_t>(k)].matrix();
    out.dalpha_dR[static_cast<std::size_t>(k)] =
        dl_a[static_cast<std::size_t>(k)] * knot[static_cast<std::size_t>(k)].matrix();
  }
  return out;
}

// Cumulative Lie-group cubic B-spline evaluation with a caller-supplied cumulative
// blending matrix (per-interval, for non-uniform knot spacing). Value and body-frame
// derivatives follow the same recursion as the uniform helper:
//   R(u) = R_0 * prod_{j=1..3} exp(c_j(u) * Log(R_{j-1}^-1 R_j)),
// with c(u) = cum * (1, u, u^2, u^3)^T. inv_dt is 1 over this interval's real span
// in seconds, so vel/accel/jerk come out as real-time derivatives. The blending
// coefficients stay double throughout -- u and the matrix never carry autodiff
// scalars; only the knots do.
template <class T, template <class> class GroupT>
inline void evaluateLieWithMatrix(T const* const* knots, double u, double inv_dt,
                                  const Eigen::Matrix4d& cum, GroupT<T>* transform_out = nullptr,
                                  typename GroupT<T>::Tangent* vel_out = nullptr,
                                  typename GroupT<T>::Tangent* accel_out = nullptr,
                                  typename GroupT<T>::Tangent* jerk_out = nullptr) {
  using Group = GroupT<T>;
  using Tangent = typename Group::Tangent;
  using Adjoint = typename Group::Adjoint;
  using Helper = basalt::CeresSplineHelper<kSplineJacOrder>;

  // Zero-init (see evaluateSo3VelAccelJac): the const-ref-output helper overwrites
  // every entry, so this only silences flow analysis in Jet-instantiated callers.
  Eigen::Vector4d p = Eigen::Vector4d::Zero();
  Eigen::Vector4d coeff, dcoeff, ddcoeff, dddcoeff;
  Helper::template baseCoeffsWithTime<0>(p, u);
  coeff = cum * p;
  if (vel_out || accel_out || jerk_out) {
    Helper::template baseCoeffsWithTime<1>(p, u);
    dcoeff = inv_dt * cum * p;
    if (accel_out || jerk_out) {
      Helper::template baseCoeffsWithTime<2>(p, u);
      ddcoeff = inv_dt * inv_dt * cum * p;
      if (jerk_out) {
        Helper::template baseCoeffsWithTime<3>(p, u);
        dddcoeff = inv_dt * inv_dt * inv_dt * cum * p;
      }
    }
  }

  if (transform_out) {
    Eigen::Map<Group const> const p00(knots[0]);
    *transform_out = p00;
  }

  Tangent rot_vel, rot_accel, rot_jerk;
  if (vel_out || accel_out || jerk_out) {
    rot_vel.setZero();
  }
  if (accel_out || jerk_out) {
    rot_accel.setZero();
  }
  if (jerk_out) {
    rot_jerk.setZero();
  }

  for (int i = 0; i < kSplineJacOrder - 1; ++i) {
    Eigen::Map<Group const> const p0(knots[i]);
    Eigen::Map<Group const> const p1(knots[i + 1]);

    const Group r01 = p0.inverse() * p1;
    const Tangent delta = r01.log();
    const Group exp_kdelta = Group::exp(delta * coeff[i + 1]);

    if (transform_out) {
      (*transform_out) *= exp_kdelta;
    }

    if (vel_out || accel_out || jerk_out) {
      const Adjoint A = exp_kdelta.inverse().Adj();

      rot_vel = A * rot_vel;
      const Tangent rot_vel_current = delta * dcoeff[i + 1];
      rot_vel += rot_vel_current;

      if (accel_out || jerk_out) {
        rot_accel = A * rot_accel;
        const Tangent accel_lie_bracket = Group::lieBracket(rot_vel, rot_vel_current);
        rot_accel += ddcoeff[i + 1] * delta + accel_lie_bracket;

        if (jerk_out) {
          rot_jerk = A * rot_jerk;
          rot_jerk += dddcoeff[i + 1] * delta +
                      Group::lieBracket(ddcoeff[i + 1] * rot_vel + 2.0 * dcoeff[i + 1] * rot_accel -
                                            dcoeff[i + 1] * accel_lie_bracket,
                                        delta);
        }
      }
    }
  }

  if (vel_out) {
    *vel_out = rot_vel;
  }
  if (accel_out) {
    *accel_out = rot_accel;
  }
  if (jerk_out) {
    *jerk_out = rot_jerk;
  }
}

// Euclidean cubic B-spline value or time derivative with a caller-supplied blending
// matrix (per-interval, for non-uniform knot spacing). DERIV selects the derivative
// order; the inv_dt^DERIV factor converts the d/du derivative into d/dt for the
// interval whose real span is 1/inv_dt seconds. Coefficients stay double; only the
// knots carry T.
template <class T, int DIM, int DERIV>
inline void evaluateRdWithMatrix(T const* const* knots, double u, double inv_dt,
                                 const Eigen::Matrix4d& blend, Eigen::Matrix<T, DIM, 1>* vec_out) {
  if (!vec_out) {
    return;
  }
  using VecD = Eigen::Matrix<T, DIM, 1>;
  using Helper = basalt::CeresSplineHelper<kSplineJacOrder>;

  // Zero-init (see evaluateSo3VelAccelJac): the const-ref-output helper overwrites
  // every entry, so this only silences flow analysis in Jet-instantiated callers.
  Eigen::Vector4d p = Eigen::Vector4d::Zero();
  Eigen::Vector4d coeff;
  Helper::template baseCoeffsWithTime<DERIV>(p, u);
  coeff = std::pow(inv_dt, DERIV) * blend * p;

  vec_out->setZero();
  for (int i = 0; i < kSplineJacOrder; ++i) {
    Eigen::Map<VecD const> const knot_i(knots[i]);
    (*vec_out) += coeff[i] * knot_i;
  }
}

}  // namespace meridian::ct
