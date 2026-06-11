#include "residuals_imu.hpp"

// The vendored coefficient helpers return through const-ref outputs (const_cast
// inside), which GCC's flow analysis flags as maybe-uninitialized once they inline
// into the Jet-instantiated functors below; every entry is in fact overwritten.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <basalt/spline/ceres_spline_helper.h>
#pragma GCC diagnostic pop

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/sphere_manifold.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sophus/so3.hpp>
#include <vector>

#include "spline_analytic.hpp"

namespace meridian::ct {

namespace {

constexpr int kSplineOrder = 4;

// Combined gyro + accel residual at one IMU sample.
//
//   r_gyro = w_g * ( omega_body(t) + b_g - omega_measured )
//   r_acc  = w_a * ( R_W_Fe(t)^T (a_world(t) - g_W) + b_a - a_measured )
//
// g_W = magnitude * direction, with direction held on the unit sphere by the
// caller's manifold so |g_W| stays fixed through the solve.
//
// Parameter layout is 4 SO(3) knots, 4 R^3 knots, then the bias blocks, then the
// gravity direction. With two bias knots the bias is linearly interpolated between
// the bracketing knots (bg_left, bg_right, ba_left, ba_right) using alpha_; with a
// single bias knot only one gyro and one accel block follow so the same parameter
// block is never registered twice in one residual. kSharedBias selects the layout.
template <bool kSharedBias>
struct ImuResidual {
  ImuResidual(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double u, double inv_dt,
              double alpha, double gravity_mag, double w_gyro, double w_accel,
              const Eigen::Matrix4d* blend = nullptr, const Eigen::Matrix4d* cum_blend = nullptr)
      : gyro_(gyro),
        accel_(accel),
        u_(u),
        inv_dt_(inv_dt),
        alpha_(alpha),
        gravity_mag_(gravity_mag),
        w_gyro_(w_gyro),
        w_accel_(w_accel),
        non_uniform_(blend != nullptr && cum_blend != nullptr) {
    if (non_uniform_) {
      blend_ = *blend;
      cum_blend_ = *cum_blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    SO3T R_w_fe;
    Tangent omega_body;
    Vec3T accel_world;
    if (non_uniform_) {
      evaluateLieWithMatrix<T, Sophus::SO3>(params, u_, inv_dt_, cum_blend_, &R_w_fe, &omega_body);
      evaluateRdWithMatrix<T, 3, 2>(params + kSplineOrder, u_, inv_dt_, blend_, &accel_world);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
          params, u_, inv_dt_, &R_w_fe, &omega_body, nullptr);
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 2>(params + kSplineOrder, u_,
                                                                          inv_dt_, &accel_world);
    }

    Vec3T b_g;
    Vec3T b_a;
    int g_index;
    if constexpr (kSharedBias) {
      Eigen::Map<const Vec3T> bg(params[2 * kSplineOrder]);
      Eigen::Map<const Vec3T> ba(params[2 * kSplineOrder + 1]);
      b_g = bg;
      b_a = ba;
      g_index = 2 * kSplineOrder + 2;
    } else {
      const T a = T(alpha_);
      const T one_minus_a = T(1.0) - a;
      Eigen::Map<const Vec3T> bg_left(params[2 * kSplineOrder]);
      Eigen::Map<const Vec3T> bg_right(params[2 * kSplineOrder + 1]);
      Eigen::Map<const Vec3T> ba_left(params[2 * kSplineOrder + 2]);
      Eigen::Map<const Vec3T> ba_right(params[2 * kSplineOrder + 3]);
      b_g = one_minus_a * bg_left + a * bg_right;
      b_a = one_minus_a * ba_left + a * ba_right;
      g_index = 2 * kSplineOrder + 4;
    }

    Eigen::Map<const Vec3T> g_dir(params[g_index]);
    const Vec3T g_world = g_dir * T(gravity_mag_);

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual.template head<3>() = T(w_gyro_) * (omega_body + b_g - gyro_.cast<T>());
    residual.template tail<3>() =
        T(w_accel_) * (R_w_fe.inverse() * (accel_world - g_world) + b_a - accel_.cast<T>());
    return true;
  }

  Eigen::Vector3d gyro_;
  Eigen::Vector3d accel_;
  double u_;
  double inv_dt_;
  double alpha_;
  double gravity_mag_;
  double w_gyro_;
  double w_accel_;
  // Per-interval blending matrices copied by value (a functor outlives the segment
  // reference it was built from); false selects the uniform cardinal matrices.
  bool non_uniform_ = false;
  Eigen::Matrix4d blend_ = Eigen::Matrix4d::Zero();
  Eigen::Matrix4d cum_blend_ = Eigen::Matrix4d::Zero();
};

// Jerk regularizer: the third real-time derivative of the R^3 (translation) spline at
// one time, weighted low. The basalt helper returns the virtual-time derivative; the
// real-time jerk is that scaled by (1/dt)^3, achieved by feeding inv_dt = 1/dt_s. It
// pulls the segment toward constant acceleration (zero jerk), removing the position
// spline's highest unconstrained derivative without biasing toward any pose.
struct JerkResidual {
  JerkResidual(double u, double inv_dt, double weight, const Eigen::Matrix4d* blend = nullptr)
      : u_(u), inv_dt_(inv_dt), weight_(weight), non_uniform_(blend != nullptr) {
    if (non_uniform_) {
      blend_ = *blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Vec3T jerk;
    if (non_uniform_) {
      evaluateRdWithMatrix<T, 3, 3>(params, u_, inv_dt_, blend_, &jerk);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 3>(params, u_, inv_dt_,
                                                                          &jerk);
    }
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * jerk;
    return true;
  }

  double u_;
  double inv_dt_;
  double weight_;
  bool non_uniform_ = false;
  Eigen::Matrix4d blend_ = Eigen::Matrix4d::Zero();
};

// Angular-acceleration regularizer: the body angular acceleration (second derivative of
// the SO(3) spline) at one time, weighted low. It pulls the segment toward constant
// body rate (zero angular acceleration). Same real-time scaling as the jerk term.
struct AngularAccelResidual {
  AngularAccelResidual(double u, double inv_dt, double weight,
                       const Eigen::Matrix4d* cum_blend = nullptr)
      : u_(u), inv_dt_(inv_dt), weight_(weight), non_uniform_(cum_blend != nullptr) {
    if (non_uniform_) {
      cum_blend_ = *cum_blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;
    SO3T R;
    Tangent vel;
    Tangent accel;
    if (non_uniform_) {
      evaluateLieWithMatrix<T, Sophus::SO3>(params, u_, inv_dt_, cum_blend_, &R, &vel, &accel);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
          params, u_, inv_dt_, &R, &vel, &accel);
    }
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * accel;
    return true;
  }

  double u_;
  double inv_dt_;
  double weight_;
  bool non_uniform_ = false;
  Eigen::Matrix4d cum_blend_ = Eigen::Matrix4d::Zero();
};

// Tail velocity anchor: ties the R^3 spline's first derivative at one time to a
// predicted world velocity. Applied only over the span past the newest measurement,
// where the trailing control points would otherwise sit in the cost's null space: a
// solver-chosen wiggle there is invisible to every measurement residual but corrupts
// the state read-out and the next sweep's evaluation. The weight is an honest
// constant-velocity extrapolation uncertainty, weak enough for real measurements to
// override it as soon as they arrive.
struct VelocityAnchorResidual {
  VelocityAnchorResidual(const Eigen::Vector3d& v_pred, double u, double inv_dt, double weight,
                         const Eigen::Matrix4d* blend = nullptr)
      : v_pred_(v_pred), u_(u), inv_dt_(inv_dt), weight_(weight), non_uniform_(blend != nullptr) {
    if (non_uniform_) {
      blend_ = *blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Vec3T vel;
    if (non_uniform_) {
      evaluateRdWithMatrix<T, 3, 1>(params, u_, inv_dt_, blend_, &vel);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 1>(params, u_, inv_dt_,
                                                                          &vel);
    }
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * (vel - v_pred_.cast<T>());
    return true;
  }

  Eigen::Vector3d v_pred_;
  double u_;
  double inv_dt_;
  double weight_;
  bool non_uniform_ = false;
  Eigen::Matrix4d blend_ = Eigen::Matrix4d::Zero();
};

// Tail angular-rate anchor: same role as the velocity anchor for the SO(3) spline,
// tying the body rate to the constant-rate extrapolation of the last gyro sample.
struct RateAnchorResidual {
  RateAnchorResidual(const Eigen::Vector3d& w_pred, double u, double inv_dt, double weight,
                     const Eigen::Matrix4d* cum_blend = nullptr)
      : w_pred_(w_pred), u_(u), inv_dt_(inv_dt), weight_(weight),
        non_uniform_(cum_blend != nullptr) {
    if (non_uniform_) {
      cum_blend_ = *cum_blend;
    }
  }

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;
    SO3T R;
    Tangent omega;
    if (non_uniform_) {
      evaluateLieWithMatrix<T, Sophus::SO3>(params, u_, inv_dt_, cum_blend_, &R, &omega);
    } else {
      basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
          params, u_, inv_dt_, &R, &omega, nullptr);
    }
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * (omega - w_pred_.cast<T>());
    return true;
  }

  Eigen::Vector3d w_pred_;
  double u_;
  double inv_dt_;
  double weight_;
  bool non_uniform_ = false;
  Eigen::Matrix4d cum_blend_ = Eigen::Matrix4d::Zero();
};

// Random-walk tie between two consecutive bias knots, weighted so the cost equals
// the squared increment divided by the continuous-time variance sigma^2 * dt: the
// sqrt-information weight is 1 / (sigma * sqrt(dt)), shared by the gyro and accel
// halves through their own sigmas. Larger gaps are tied more weakly.
struct BiasTieResidual {
  BiasTieResidual(double w_gyro, double w_accel) : w_gyro_(w_gyro), w_accel_(w_accel) {}

  template <class T>
  bool operator()(const T* bg_prev, const T* bg_next, const T* ba_prev, const T* ba_next,
                  T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Eigen::Map<const Vec3T> bgp(bg_prev);
    Eigen::Map<const Vec3T> bgn(bg_next);
    Eigen::Map<const Vec3T> bap(ba_prev);
    Eigen::Map<const Vec3T> ban(ba_next);
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual.template head<3>() = T(w_gyro_) * (bgn - bgp);
    residual.template tail<3>() = T(w_accel_) * (ban - bap);
    return true;
  }

  double w_gyro_;
  double w_accel_;
};

// Non-cumulative R^3 blending row scaled to the real-time Deriv-th derivative:
// c[j] = inv_dt^Deriv * (M * baseCoeffsWithTime<Deriv>(u))[j]. A pure R^3 spline
// derivative is sum_j c[j] * P_j with these scalar weights. A null `blend` binds the
// uniform cardinal matrix object itself, so the expression runs on the same operands
// as always; a non-null pointer substitutes the per-interval matrix unchanged.
template <int Deriv>
Eigen::Matrix<double, kSplineOrder, 1> r3DerivBlend(double u, double inv_dt,
                                                    const Eigen::Matrix4d* blend = nullptr) {
  Eigen::Matrix<double, kSplineOrder, 1> p;
  basalt::CeresSplineHelper<kSplineOrder>::template baseCoeffsWithTime<Deriv>(p, u);
  double scale = 1.0;
  for (int i = 0; i < Deriv; ++i) scale *= inv_dt;
  const Eigen::Matrix4d& M =
      blend ? *blend : basalt::CeresSplineHelper<kSplineOrder>::blending_matrix_;
  return scale * (M * p);
}

// Combined gyro + accel residual at one IMU sample, analytic Jacobians. Math is
// identical to the autodiff ImuResidual<kSharedBias>:
//   r_gyro = w_g * ( omega_body(u) + b_g - omega_measured )
//   r_acc  = w_a * ( R_W_Fe(u)^-1 (a_world(u) - g_W) + b_a - a_measured )
// where g_W = magnitude * direction. SO(3) knot Jacobians are RIGHT-tangent then
// mapped to ambient quaternion (x,y,z,w); R^3/bias/gravity blocks are native ambient.
// No manifold is installed here: the EigenQuaternionManifold on the SO(3) knots and
// the SphereManifold<3> on the gravity direction are attached by the caller, and Ceres
// applies their PlusJacobian to these ambient blocks. The accel residual SUBTRACTS
// gravity, so the gravity-direction block carries the NEGATIVE sign.
template <bool kSharedBias>
class AnalyticImuCost final : public ceres::CostFunction {
public:
  AnalyticImuCost(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double u,
                  double inv_dt, double alpha, double gravity_mag, double w_gyro, double w_accel,
                  const Eigen::Matrix4d* blend = nullptr,
                  const Eigen::Matrix4d* cum_blend = nullptr)
      : gyro_(gyro),
        accel_(accel),
        alpha_(alpha),
        gravity_mag_(gravity_mag),
        w_gyro_(w_gyro),
        w_accel_(w_accel) {
    set_num_residuals(6);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(4);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(3);
    if constexpr (kSharedBias) {
      mutable_parameter_block_sizes()->push_back(3);  // b_g
      mutable_parameter_block_sizes()->push_back(3);  // b_a
    } else {
      mutable_parameter_block_sizes()->push_back(3);  // bg_left
      mutable_parameter_block_sizes()->push_back(3);  // bg_right
      mutable_parameter_block_sizes()->push_back(3);  // ba_left
      mutable_parameter_block_sizes()->push_back(3);  // ba_right
    }
    mutable_parameter_block_sizes()->push_back(3);  // gravity direction

    using Helper = basalt::CeresSplineHelper<kSplineOrder>;
    const Eigen::Vector4d p0(1.0, u, u * u, u * u * u);
    const Eigen::Vector4d p1(0.0, 1.0, 2.0 * u, 3.0 * u * u);
    const Eigen::Vector4d p2(0.0, 0.0, 2.0, 6.0 * u);
    // Null pointers bind the uniform cardinal matrix objects themselves, so the
    // expressions below run on the same operands as always; non-null substitutes the
    // per-interval (non-uniform) matrices into the identical expressions.
    const Eigen::Matrix4d& CM = cum_blend ? *cum_blend : Helper::cumulative_blending_matrix_;
    const Eigen::Matrix4d& M = blend ? *blend : Helper::blending_matrix_;
    lambda_r_ = CM * p0;
    dcoeff_ = inv_dt * (CM * p1);
    lambda_acc_ = (inv_dt * inv_dt) * (M * p2);  // NON-cumulative
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    using SO3 = Sophus::SO3d;
    std::array<SO3, kSplineOrder> R;
    for (int j = 0; j < kSplineOrder; ++j)
      R[static_cast<std::size_t>(j)] = SO3(Eigen::Map<const Eigen::Quaterniond>(parameters[j]));

    const So3SplineJac sj = evaluateSo3CumulativeJac(R, lambda_r_);
    const So3VelocityJac vj = evaluateSo3VelocityJac(R, lambda_r_, dcoeff_);
    const SO3& R_w_fe = sj.R;
    const Eigen::Matrix3d R_fe_w = R_w_fe.inverse().matrix();

    Eigen::Vector3d a_world = Eigen::Vector3d::Zero();
    for (int j = 0; j < kSplineOrder; ++j)
      a_world += lambda_acc_[j] * Eigen::Map<const Eigen::Vector3d>(parameters[kSplineOrder + j]);

    Eigen::Vector3d b_g, b_a;
    int g_index = 0;
    double wl = 0.0, wr = 0.0;
    if constexpr (kSharedBias) {
      b_g = Eigen::Map<const Eigen::Vector3d>(parameters[2 * kSplineOrder]);
      b_a = Eigen::Map<const Eigen::Vector3d>(parameters[2 * kSplineOrder + 1]);
      g_index = 2 * kSplineOrder + 2;
    } else {
      wl = 1.0 - alpha_;
      wr = alpha_;
      const Eigen::Map<const Eigen::Vector3d> bg_l(parameters[2 * kSplineOrder]);
      const Eigen::Map<const Eigen::Vector3d> bg_r(parameters[2 * kSplineOrder + 1]);
      const Eigen::Map<const Eigen::Vector3d> ba_l(parameters[2 * kSplineOrder + 2]);
      const Eigen::Map<const Eigen::Vector3d> ba_r(parameters[2 * kSplineOrder + 3]);
      b_g = wl * bg_l + wr * bg_r;
      b_a = wl * ba_l + wr * ba_r;
      g_index = 2 * kSplineOrder + 4;
    }

    const Eigen::Vector3d g_world =
        Eigen::Map<const Eigen::Vector3d>(parameters[g_index]) * gravity_mag_;
    const Eigen::Vector3d v = a_world - g_world;  // specific force, world frame
    const Eigen::Vector3d v_b = R_fe_w * v;       // rotated into the body frame

    Eigen::Map<Eigen::Matrix<double, 6, 1>> r(residuals);
    r.head<3>() = w_gyro_ * (vj.omega + b_g - gyro_);
    r.tail<3>() = w_accel_ * (v_b + b_a - accel_);

    if (jacobians == nullptr) return true;

    // SO(3) knots. A right perturbation R_w_fe <- R_w_fe exp(dRt theta) moves the
    // body-frame specific force by R^-1 v <- v_b + [v_b]_x (dRt theta) to first order.
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) continue;
      const std::size_t k = static_cast<std::size_t>(j);
      Eigen::Matrix<double, 6, 3> jt;
      jt.topRows<3>() = w_gyro_ * vj.dOmega_dR[k];
      jt.bottomRows<3>() = w_accel_ * SO3::hat(v_b) * sj.dRt_dR[k];
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobians[j]);
      J = jt * tangentToQuatJac(R[k].unit_quaternion());
    }
    // R^3 knots: accel rows only, second-derivative non-cumulative blend.
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[kSplineOrder + j] == nullptr) continue;
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[kSplineOrder + j]);
      J.setZero();
      J.bottomRows<3>() = (w_accel_ * lambda_acc_[j]) * R_fe_w;
    }
    // Bias blocks.
    if constexpr (kSharedBias) {
      if (jacobians[2 * kSplineOrder] != nullptr) {  // b_g
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2 * kSplineOrder]);
        J.setZero();
        J.topRows<3>() = w_gyro_ * Eigen::Matrix3d::Identity();
      }
      if (jacobians[2 * kSplineOrder + 1] != nullptr) {  // b_a
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2 * kSplineOrder + 1]);
        J.setZero();
        J.bottomRows<3>() = w_accel_ * Eigen::Matrix3d::Identity();
      }
    } else {
      const double bg_w[2] = {w_gyro_ * wl, w_gyro_ * wr};
      const double ba_w[2] = {w_accel_ * wl, w_accel_ * wr};
      for (int s = 0; s < 2; ++s) {  // bg_left, bg_right
        if (jacobians[2 * kSplineOrder + s] == nullptr) continue;
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2 * kSplineOrder + s]);
        J.setZero();
        J.topRows<3>() = bg_w[s] * Eigen::Matrix3d::Identity();
      }
      for (int s = 0; s < 2; ++s) {  // ba_left, ba_right
        if (jacobians[2 * kSplineOrder + 2 + s] == nullptr) continue;
        Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(
            jacobians[2 * kSplineOrder + 2 + s]);
        J.setZero();
        J.bottomRows<3>() = ba_w[s] * Eigen::Matrix3d::Identity();
      }
    }
    // Gravity direction: the residual subtracts g_W = mag * dir, so d r_acc / d dir =
    // -w_accel * mag * R^-1 (ambient; the SphereManifold PlusJacobian is applied by
    // Ceres).
    if (jacobians[g_index] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[g_index]);
      J.setZero();
      J.bottomRows<3>() = (-w_accel_ * gravity_mag_) * R_fe_w;
    }
    return true;
  }

private:
  Eigen::Vector3d gyro_, accel_;
  double alpha_, gravity_mag_, w_gyro_, w_accel_;
  Eigen::Vector4d lambda_r_, dcoeff_, lambda_acc_;
};

// Jerk regularizer, analytic. r = w * sum_j c3[j] * P_j over the four R^3 knots, with
// c3 the third-derivative non-cumulative blend; J_j = w * c3[j] * I3.
class AnalyticJerkCost final : public ceres::CostFunction {
public:
  AnalyticJerkCost(double u, double inv_dt, double weight,
                   const Eigen::Matrix4d* blend = nullptr)
      : weight_(weight) {
    set_num_residuals(3);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(3);
    c3_ = r3DerivBlend<3>(u, inv_dt, blend);
  }
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r.setZero();
    for (int j = 0; j < kSplineOrder; ++j)
      r += (weight_ * c3_[j]) * Eigen::Map<const Eigen::Vector3d>(parameters[j]);
    if (jacobians == nullptr) return true;
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) continue;
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[j]);
      J = (weight_ * c3_[j]) * Eigen::Matrix3d::Identity();
    }
    return true;
  }

private:
  double weight_;
  Eigen::Matrix<double, kSplineOrder, 1> c3_;
};

// Tail velocity anchor, analytic. r = w * (sum_j c1[j] * P_j - v_pred); the constant
// v_pred offset drops from the Jacobian, leaving J_j = w * c1[j] * I3.
class AnalyticVelocityAnchorCost final : public ceres::CostFunction {
public:
  AnalyticVelocityAnchorCost(const Eigen::Vector3d& v_pred, double u, double inv_dt, double weight,
                             const Eigen::Matrix4d* blend = nullptr)
      : v_pred_(v_pred), weight_(weight) {
    set_num_residuals(3);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(3);
    c1_ = r3DerivBlend<1>(u, inv_dt, blend);
  }
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    for (int j = 0; j < kSplineOrder; ++j)
      vel += c1_[j] * Eigen::Map<const Eigen::Vector3d>(parameters[j]);
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r = weight_ * (vel - v_pred_);
    if (jacobians == nullptr) return true;
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) continue;
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[j]);
      J = (weight_ * c1_[j]) * Eigen::Matrix3d::Identity();
    }
    return true;
  }

private:
  Eigen::Vector3d v_pred_;
  double weight_;
  Eigen::Matrix<double, kSplineOrder, 1> c1_;
};

// Angular-acceleration regularizer, analytic. r = w * alpha(u); SO(3) knot Jacobians
// are right-tangent (dalpha_dR) mapped to ambient quaternion.
class AnalyticAngularAccelCost final : public ceres::CostFunction {
public:
  AnalyticAngularAccelCost(double u, double inv_dt, double weight,
                           const Eigen::Matrix4d* cum_blend = nullptr)
      : u_(u), inv_dt_(inv_dt), weight_(weight), cum_blend_(cum_blend) {
    set_num_residuals(3);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(4);
  }
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    using SO3 = Sophus::SO3d;
    std::array<SO3, kSplineOrder> R;
    for (int j = 0; j < kSplineOrder; ++j)
      R[static_cast<std::size_t>(j)] = SO3(Eigen::Map<const Eigen::Quaterniond>(parameters[j]));
    const So3DerivJac dj = evaluateSo3VelAccelJac(R, u_, inv_dt_, /*want_accel=*/true, cum_blend_);
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r = weight_ * dj.alpha;
    if (jacobians == nullptr) return true;
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) continue;
      const std::size_t k = static_cast<std::size_t>(j);
      const Eigen::Matrix3d jt = weight_ * dj.dalpha_dR[k];
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[j]);
      J = jt * tangentToQuatJac(R[k].unit_quaternion());
    }
    return true;
  }

private:
  double u_, inv_dt_, weight_;
  // Per-interval cumulative blending matrix; points at window-owned cache with the
  // knot-pointer lifetime (valid while this interval's knots are resident). Null
  // selects the uniform cardinal matrix.
  const Eigen::Matrix4d* cum_blend_ = nullptr;
};

// Tail angular-rate anchor, analytic. r = w * (omega(u) - w_pred); SO(3) knot
// Jacobians are right-tangent (domega_dR) mapped to ambient quaternion.
class AnalyticRateAnchorCost final : public ceres::CostFunction {
public:
  AnalyticRateAnchorCost(const Eigen::Vector3d& w_pred, double u, double inv_dt, double weight,
                         const Eigen::Matrix4d* cum_blend = nullptr)
      : w_pred_(w_pred), u_(u), inv_dt_(inv_dt), weight_(weight), cum_blend_(cum_blend) {
    set_num_residuals(3);
    for (int j = 0; j < kSplineOrder; ++j) mutable_parameter_block_sizes()->push_back(4);
  }
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    using SO3 = Sophus::SO3d;
    std::array<SO3, kSplineOrder> R;
    for (int j = 0; j < kSplineOrder; ++j)
      R[static_cast<std::size_t>(j)] = SO3(Eigen::Map<const Eigen::Quaterniond>(parameters[j]));
    const So3DerivJac dj =
        evaluateSo3VelAccelJac(R, u_, inv_dt_, /*want_accel=*/false, cum_blend_);
    Eigen::Map<Eigen::Vector3d> r(residuals);
    r = weight_ * (dj.omega - w_pred_);
    if (jacobians == nullptr) return true;
    for (int j = 0; j < kSplineOrder; ++j) {
      if (jacobians[j] == nullptr) continue;
      const std::size_t k = static_cast<std::size_t>(j);
      const Eigen::Matrix3d jt = weight_ * dj.domega_dR[k];
      Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>> J(jacobians[j]);
      J = jt * tangentToQuatJac(R[k].unit_quaternion());
    }
    return true;
  }

private:
  Eigen::Vector3d w_pred_;
  double u_, inv_dt_, weight_;
  const Eigen::Matrix4d* cum_blend_ = nullptr;
};

// Random-walk tie between two consecutive bias knots, analytic. Block order is
// bg_prev, bg_next, ba_prev, ba_next. r_head = w_g (bg_next - bg_prev), r_tail =
// w_a (ba_next - ba_prev); constant Jacobians +-w_g I3 / +-w_a I3.
class AnalyticBiasTieCost final : public ceres::CostFunction {
public:
  AnalyticBiasTieCost(double w_gyro, double w_accel) : w_gyro_(w_gyro), w_accel_(w_accel) {
    set_num_residuals(6);
    for (int b = 0; b < 4; ++b) mutable_parameter_block_sizes()->push_back(3);
  }
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::Vector3d> bgp(parameters[0]);
    const Eigen::Map<const Eigen::Vector3d> bgn(parameters[1]);
    const Eigen::Map<const Eigen::Vector3d> bap(parameters[2]);
    const Eigen::Map<const Eigen::Vector3d> ban(parameters[3]);
    Eigen::Map<Eigen::Matrix<double, 6, 1>> r(residuals);
    r.head<3>() = w_gyro_ * (bgn - bgp);
    r.tail<3>() = w_accel_ * (ban - bap);
    if (jacobians == nullptr) return true;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    if (jacobians[0] != nullptr) {  // bg_prev
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
      J.setZero();
      J.topRows<3>() = -w_gyro_ * I;
    }
    if (jacobians[1] != nullptr) {  // bg_next
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[1]);
      J.setZero();
      J.topRows<3>() = w_gyro_ * I;
    }
    if (jacobians[2] != nullptr) {  // ba_prev
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[2]);
      J.setZero();
      J.bottomRows<3>() = -w_accel_ * I;
    }
    if (jacobians[3] != nullptr) {  // ba_next
      Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[3]);
      J.setZero();
      J.bottomRows<3>() = w_accel_ * I;
    }
    return true;
  }

private:
  double w_gyro_, w_accel_;
};

}  // namespace

template <bool kSharedBias>
ceres::CostFunction* makeImuCost(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel,
                                 double u, double inv_dt, double alpha, double gravity_mag,
                                 double w_gyro, double w_accel, const Eigen::Matrix4d* blend,
                                 const Eigen::Matrix4d* cum_blend) {
  return new AnalyticImuCost<kSharedBias>(gyro, accel, u, inv_dt, alpha, gravity_mag, w_gyro,
                                          w_accel, blend, cum_blend);
}

template <bool kSharedBias>
ceres::CostFunction* makeImuCostAutodiff(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel,
                                         double u, double inv_dt, double alpha, double gravity_mag,
                                         double w_gyro, double w_accel,
                                         const Eigen::Matrix4d* blend,
                                         const Eigen::Matrix4d* cum_blend) {
  auto* c = new ceres::DynamicAutoDiffCostFunction<ImuResidual<kSharedBias>, 4>(
      new ImuResidual<kSharedBias>(gyro, accel, u, inv_dt, alpha, gravity_mag, w_gyro, w_accel,
                                   blend, cum_blend));
  const int n_blocks = 2 * kSplineOrder + (kSharedBias ? 3 : 5);
  for (int i = 0; i < n_blocks; ++i) c->AddParameterBlock(i < kSplineOrder ? 4 : 3);
  c->SetNumResiduals(6);
  return c;
}

template ceres::CostFunction* makeImuCost<true>(const Eigen::Vector3d&, const Eigen::Vector3d&,
                                                double, double, double, double, double, double,
                                                const Eigen::Matrix4d*, const Eigen::Matrix4d*);
template ceres::CostFunction* makeImuCost<false>(const Eigen::Vector3d&, const Eigen::Vector3d&,
                                                 double, double, double, double, double, double,
                                                 const Eigen::Matrix4d*, const Eigen::Matrix4d*);
template ceres::CostFunction* makeImuCostAutodiff<true>(const Eigen::Vector3d&,
                                                        const Eigen::Vector3d&, double, double,
                                                        double, double, double, double,
                                                        const Eigen::Matrix4d*,
                                                        const Eigen::Matrix4d*);
template ceres::CostFunction* makeImuCostAutodiff<false>(const Eigen::Vector3d&,
                                                         const Eigen::Vector3d&, double, double,
                                                         double, double, double, double,
                                                         const Eigen::Matrix4d*,
                                                         const Eigen::Matrix4d*);

ceres::CostFunction* makeJerkCost(double u, double inv_dt, double weight,
                                  const Eigen::Matrix4d* blend) {
  return new AnalyticJerkCost(u, inv_dt, weight, blend);
}

ceres::CostFunction* makeJerkCostAutodiff(double u, double inv_dt, double weight,
                                          const Eigen::Matrix4d* blend) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<JerkResidual, 4>(
      new JerkResidual(u, inv_dt, weight, blend));
  for (int i = 0; i < kSplineOrder; ++i) cost->AddParameterBlock(3);
  cost->SetNumResiduals(3);
  return cost;
}

ceres::CostFunction* makeVelocityAnchorCost(const Eigen::Vector3d& v_pred, double u, double inv_dt,
                                            double weight, const Eigen::Matrix4d* blend) {
  return new AnalyticVelocityAnchorCost(v_pred, u, inv_dt, weight, blend);
}

ceres::CostFunction* makeVelocityAnchorCostAutodiff(const Eigen::Vector3d& v_pred, double u,
                                                    double inv_dt, double weight,
                                                    const Eigen::Matrix4d* blend) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<VelocityAnchorResidual, 4>(
      new VelocityAnchorResidual(v_pred, u, inv_dt, weight, blend));
  for (int i = 0; i < kSplineOrder; ++i) cost->AddParameterBlock(3);
  cost->SetNumResiduals(3);
  return cost;
}

ceres::CostFunction* makeAngularAccelCost(double u, double inv_dt, double weight,
                                          const Eigen::Matrix4d* cum_blend) {
  return new AnalyticAngularAccelCost(u, inv_dt, weight, cum_blend);
}

ceres::CostFunction* makeAngularAccelCostAutodiff(double u, double inv_dt, double weight,
                                                  const Eigen::Matrix4d* cum_blend) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<AngularAccelResidual, 4>(
      new AngularAccelResidual(u, inv_dt, weight, cum_blend));
  for (int i = 0; i < kSplineOrder; ++i) cost->AddParameterBlock(4);
  cost->SetNumResiduals(3);
  return cost;
}

ceres::CostFunction* makeRateAnchorCost(const Eigen::Vector3d& w_pred, double u, double inv_dt,
                                        double weight, const Eigen::Matrix4d* cum_blend) {
  return new AnalyticRateAnchorCost(w_pred, u, inv_dt, weight, cum_blend);
}

ceres::CostFunction* makeRateAnchorCostAutodiff(const Eigen::Vector3d& w_pred, double u,
                                                double inv_dt, double weight,
                                                const Eigen::Matrix4d* cum_blend) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<RateAnchorResidual, 4>(
      new RateAnchorResidual(w_pred, u, inv_dt, weight, cum_blend));
  for (int i = 0; i < kSplineOrder; ++i) cost->AddParameterBlock(4);
  cost->SetNumResiduals(3);
  return cost;
}

ceres::CostFunction* makeBiasTieCost(double w_gyro, double w_accel) {
  return new AnalyticBiasTieCost(w_gyro, w_accel);
}

ceres::CostFunction* makeBiasTieCostAutodiff(double w_gyro, double w_accel) {
  return new ceres::AutoDiffCostFunction<BiasTieResidual, 6, 3, 3, 3, 3>(
      new BiasTieResidual(w_gyro, w_accel));
}

GravityBlock::GravityBlock(const Eigen::Vector3d& g_world, double magnitude)
    : magnitude_(magnitude) {
  const double n = g_world.norm();
  // Fall back to "down" if handed a zero vector so the stored value stays unit.
  direction_ = (n > 0.0) ? (g_world / n) : Eigen::Vector3d(0.0, 0.0, -1.0);
}

Eigen::Vector3d GravityBlock::gravityWorld() const {
  return magnitude_ * direction_;
}

BiasKnots::BiasKnots(Timestamp t0, Duration dt_ns, int n_knots) : t0_(t0), dt_ns_(dt_ns) {
  const int n = n_knots < 1 ? 1 : n_knots;
  gyro_.assign(static_cast<std::size_t>(n), Eigen::Vector3d::Zero());
  accel_.assign(static_cast<std::size_t>(n), Eigen::Vector3d::Zero());
}

BiasKnots BiasKnots::clone() const {
  BiasKnots copy(t0_, dt_ns_, numKnots());
  copy.gyro_ = gyro_;
  copy.accel_ = accel_;
  return copy;
}

void BiasKnots::extendTo(Timestamp t) {
  if (dt_ns_ <= 0) {
    return;
  }
  // Appended knots copy the last value: under the random-walk model the increment
  // mean is zero, so the previous estimate is the correct warm start.
  while (knotTime(numKnots() - 1) < t) {
    gyro_.push_back(gyro_.back());
    accel_.push_back(accel_.back());
  }
}

void BiasKnots::dropOldest(int n) {
  const int max_drop = numKnots() - 2;
  const int drop = std::clamp(n, 0, std::max(0, max_drop));
  for (int i = 0; i < drop; ++i) {
    gyro_.pop_front();
    accel_.pop_front();
  }
  t0_ += static_cast<Timestamp>(drop) * dt_ns_;
}

int BiasKnots::lowestKnotIndexOf(const std::vector<const double*>& ptrs) const {
  if (ptrs.empty()) {
    return std::numeric_limits<int>::max();
  }
  for (int k = 0; k < numKnots(); ++k) {
    const double* g = gyro_[static_cast<std::size_t>(k)].data();
    const double* a = accel_[static_cast<std::size_t>(k)].data();
    for (const double* p : ptrs) {
      if (p == g || p == a) {
        return k;
      }
    }
  }
  return std::numeric_limits<int>::max();
}

int BiasKnots::leftIndex(Timestamp t) const {
  if (numKnots() <= 1 || dt_ns_ <= 0) {
    return 0;
  }
  const Timestamp rel = t - t0_;
  long idx = (rel <= 0) ? 0 : static_cast<long>(rel / dt_ns_);
  if (idx < 0) {
    idx = 0;
  }
  // Clamp so idx+1 is always a valid right bracket.
  if (idx > numKnots() - 2) {
    idx = numKnots() - 2;
  }
  return static_cast<int>(idx);
}

double BiasKnots::alpha(Timestamp t) const {
  if (numKnots() <= 1 || dt_ns_ <= 0) {
    return 0.0;
  }
  const int k = leftIndex(t);
  const double num = static_cast<double>(t - knotTime(k));
  double a = num / static_cast<double>(dt_ns_);
  if (a < 0.0) {
    a = 0.0;
  }
  if (a > 1.0) {
    a = 1.0;
  }
  return a;
}

Eigen::Vector3d BiasKnots::gyroBiasAt(Timestamp t) const {
  const int k = leftIndex(t);
  if (numKnots() <= 1) {
    return gyro_[0];
  }
  const double a = alpha(t);
  return (1.0 - a) * gyro_[k] + a * gyro_[k + 1];
}

Eigen::Vector3d BiasKnots::accelBiasAt(Timestamp t) const {
  const int k = leftIndex(t);
  if (numKnots() <= 1) {
    return accel_[0];
  }
  const double a = alpha(t);
  return (1.0 - a) * accel_[k] + a * accel_[k + 1];
}

ceres::Manifold* makeGravityManifold() {
  return new ceres::SphereManifold<3>();
}

ImuExcitation imuExcitation(const std::vector<ImuSample>& samples, const Eigen::Vector3d& b_g,
                            const Eigen::Vector3d& b_a, double gravity_mag) {
  ImuExcitation exc;
  for (const ImuSample& s : samples) {
    const double w = (s.gyro - b_g).norm();
    // Subtracting the gravity MAGNITUDE from the de-biased specific-force magnitude
    // keeps the statistic exact under any attitude without needing an orientation
    // estimate (it only under-reads when specific force is near-orthogonal to gravity,
    // which the angular term and the IMU residual then cover).
    const double a = std::abs((s.acc - b_a).norm() - gravity_mag);
    if (w > exc.n_omega) {
      exc.n_omega = w;
    }
    if (a > exc.n_accel) {
      exc.n_accel = a;
    }
  }
  return exc;
}

namespace {

// Number of rising band edges a statistic clears, with downward hysteresis relative to
// the previous count. A band counts as crossed if the statistic is at/above its edge;
// to step DOWN out of a band already held last segment, the statistic must fall below
// the edge by a `hysteresis` fraction, so density does not chatter at a threshold.
// `prev_bands` is how many bands this statistic's axis held last segment.
int bandsCrossed(double value, const std::vector<double>& thresh, double hysteresis,
                 int prev_bands) {
  int bands = 0;
  for (int i = 0; i < static_cast<int>(thresh.size()); ++i) {
    const double edge = thresh[static_cast<std::size_t>(i)];
    // The drop edge is the band edge relaxed downward by the hysteresis fraction; it
    // applies only to bands we already held (i < prev_bands), so rising uses the plain
    // edge and only falling out of a held band needs the deeper drop.
    const double drop_edge = (i < prev_bands) ? edge * (1.0 - hysteresis) : edge;
    if (value >= drop_edge) {
      ++bands;
    } else {
      break;  // thresholds are strictly increasing, so the first miss ends the run
    }
  }
  return bands;
}

}  // namespace

ImuMeanExcitation segmentMeanExcitation(const std::vector<ImuSample>& samples, Timestamp t0,
                                        Timestamp t1, double gravity_mag) {
  ImuMeanExcitation stats;
  double sum_omega = 0.0;
  Eigen::Vector3d sum_acc = Eigen::Vector3d::Zero();
  for (const ImuSample& s : samples) {
    if (s.stamp <= t0 || s.stamp > t1) {
      continue;
    }
    // Gyro: mean of per-sample NORMS. Each sample's magnitude is frame-invariant, so
    // a fast rotation that sweeps the vectors through every direction (whose vector
    // mean cancels toward zero) still reads its full rate. Raw gyro, no bias
    // correction — the bias (<= ~0.5 rad/s box bound, typically mrad/s) sits far
    // below the band edges, and gating density on a stale bias estimate would couple
    // the gear to estimator health.
    sum_omega += s.gyro.norm();
    // Accel: VECTOR sum, for the mean-vector deviation below.
    sum_acc += s.acc;
    ++stats.n;
  }
  if (stats.n > 0) {
    const double inv_n = 1.0 / static_cast<double>(stats.n);
    stats.mean_omega = sum_omega * inv_n;
    // Deviation of the mean specific-force VECTOR's norm from gravity. Zero-mean
    // vibration (step impacts while walking) cancels in the vector mean, where a
    // mean of per-sample norms reads it as excitation (E||g+n|| > g for zero-mean n
    // by Jensen); sustained acceleration shifts the mean vector and reads through.
    // At rest the mean vector is exactly the gravity vector, so the deviation reads
    // zero without an orientation estimate. Attitude change within one segment
    // shrinks the mean's norm only by O(theta^2) — negligible over a knot interval,
    // and fast rotation is the omega axis's signal anyway.
    stats.mean_accel_dev = std::abs((sum_acc * inv_n).norm() - gravity_mag);
  }
  return stats;
}

int gearFromMeanExcitation(const ImuMeanExcitation& stats, const std::vector<double>& omega_thresh,
                           const std::vector<double>& accel_thresh, double hysteresis,
                           int n_cp_max, int prev_gear) {
  const int cap = std::max(1, n_cp_max);
  if (stats.n == 0) {
    // No data in the segment yet: hold the previous gear (the segment is provisional
    // and gets re-laid once its samples arrive).
    return std::clamp(prev_gear, 1, cap);
  }
  const double hys = std::clamp(hysteresis, 0.0, 0.99);
  // The previous gear distributes as bands held on each axis; since the gear is the
  // larger of the two band counts + 1, the previous band count is prev_gear - 1 and
  // the same held-band budget applies to both axes (the binding axis held more).
  const int prev_bands = std::max(0, prev_gear - 1);
  const int omega_bands = bandsCrossed(stats.mean_omega, omega_thresh, hys, prev_bands);
  const int accel_bands = bandsCrossed(stats.mean_accel_dev, accel_thresh, hys, prev_bands);
  const int bands = std::max(omega_bands, accel_bands);
  return std::clamp(1 + bands, 1, cap);
}

int addImuResiduals(ceres::Problem& problem, SplineWindow& spline, BiasKnots& bias,
                    GravityBlock& gravity, const std::vector<ImuSample>& samples, Timestamp t_begin,
                    Timestamp t_end, const ImuWeights& weights) {
  // White-noise residuals use sqrt-information = 1 / sigma.
  const double w_gyro = 1.0 / weights.sigma_gyro;
  const double w_accel = 1.0 / weights.sigma_accel;

  const bool shared = bias.numKnots() <= 1;

  int added = 0;
  for (const ImuSample& s : samples) {
    if (s.stamp < t_begin || s.stamp > t_end) {
      continue;
    }
    if (!spline.covers(s.stamp)) {
      continue;
    }

    const SplineWindow::SegmentRef seg = spline.segmentFor(s.stamp);
    const int bk = bias.leftIndex(s.stamp);
    const double alpha = bias.alpha(s.stamp);

    ceres::CostFunction* cost = nullptr;
    std::vector<double*> blocks;
    blocks.reserve(2 * kSplineOrder + 5);
    for (double* p : seg.so3_knots) {
      blocks.push_back(p);
    }
    for (double* p : seg.r3_knots) {
      blocks.push_back(p);
    }

    if (shared) {
      cost = makeImuCost<true>(s.gyro, s.acc, seg.u, 1.0 / seg.dt_s, alpha, gravity.magnitude(),
                               w_gyro, w_accel, seg.blend, seg.cum_blend);
      blocks.push_back(bias.gyroBlock(bk));
      blocks.push_back(bias.accelBlock(bk));
    } else {
      cost = makeImuCost<false>(s.gyro, s.acc, seg.u, 1.0 / seg.dt_s, alpha, gravity.magnitude(),
                                w_gyro, w_accel, seg.blend, seg.cum_blend);
      blocks.push_back(bias.gyroBlock(bk));
      blocks.push_back(bias.gyroBlock(bk + 1));
      blocks.push_back(bias.accelBlock(bk));
      blocks.push_back(bias.accelBlock(bk + 1));
    }
    blocks.push_back(gravity.data());

    problem.AddResidualBlock(cost, nullptr, blocks);
    ++added;
  }

  // Pin the gravity block to the unit sphere so |g_W| stays fixed through solves.
  if (added > 0 && problem.HasParameterBlock(gravity.data()) &&
      problem.GetManifold(gravity.data()) == nullptr) {
    problem.SetManifold(gravity.data(), makeGravityManifold());
  }
  return added;
}

int addTailAnchors(ceres::Problem& problem, SplineWindow& spline,
                   const std::vector<Timestamp>& times, const Eigen::Vector3d& v_pred,
                   const Eigen::Vector3d& w_pred, double sigma_vel, double sigma_rate) {
  const double w_vel = 1.0 / std::max(sigma_vel, 1e-9);
  const double w_rate = 1.0 / std::max(sigma_rate, 1e-9);

  int added = 0;
  for (Timestamp t : times) {
    if (!spline.covers(t)) {
      continue;
    }
    const SplineWindow::SegmentRef seg = spline.segmentFor(t);
    const double inv_dt = 1.0 / seg.dt_s;

    {
      ceres::CostFunction* cost = makeVelocityAnchorCost(v_pred, seg.u, inv_dt, w_vel, seg.blend);
      std::vector<double*> blocks(seg.r3_knots.begin(), seg.r3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    {
      ceres::CostFunction* cost = makeRateAnchorCost(w_pred, seg.u, inv_dt, w_rate, seg.cum_blend);
      std::vector<double*> blocks(seg.so3_knots.begin(), seg.so3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    ++added;
  }
  return added;
}

int addBiasRandomWalk(ceres::Problem& problem, BiasKnots& bias, const ImuWeights& weights) {
  const int n = bias.numKnots();
  if (n <= 1) {
    return 0;
  }
  const double dt_s = to_seconds(bias.knotDt());
  // Continuous-time random walk: increment variance is sigma^2 * dt, so the tie's
  // sqrt-information weight is 1 / (sigma * sqrt(dt)) -- longer gaps tie weaker. The
  // tie is a MODEL constraint, not a data increment: it appears at full weight in
  // every problem whose knots are resident, and it never enters the marginalization
  // prior (it touches no dropped spline knot), so re-stating it per solve counts it
  // exactly once per problem. A departing bias knot takes its tie into the prior via
  // the Schur drop in slideWindow.
  const double sqrt_dt = std::sqrt(dt_s > 0.0 ? dt_s : 1.0);
  const double w_bg = 1.0 / (weights.sigma_bias_gyro * sqrt_dt);
  const double w_ba = 1.0 / (weights.sigma_bias_accel * sqrt_dt);

  int added = 0;
  for (int k = 1; k < n; ++k) {
    ceres::CostFunction* cost = makeBiasTieCost(w_bg, w_ba);
    problem.AddResidualBlock(cost, nullptr, bias.gyroBlock(k - 1), bias.gyroBlock(k),
                             bias.accelBlock(k - 1), bias.accelBlock(k));
    ++added;
  }
  return added;
}

int addMotionRegularizer(ceres::Problem& problem, SplineWindow& spline,
                         const std::vector<Timestamp>& times, double weight, double accel_weight,
                         double gyro_weight) {
  if (weight <= 0.0) {
    return 0;
  }
  // The regularizer weights are `weight` relative to the IMU accel/gyro weights so it
  // is small enough to bias nothing once a real measurement constrains the span.
  const double w_jerk = weight * accel_weight;
  const double w_accel = weight * gyro_weight;

  int added = 0;
  for (Timestamp t : times) {
    if (!spline.covers(t)) {
      continue;
    }
    const SplineWindow::SegmentRef seg = spline.segmentFor(t);
    const double inv_dt = 1.0 / seg.dt_s;

    // Jerk on the R^3 knots.
    {
      ceres::CostFunction* cost = makeJerkCost(seg.u, inv_dt, w_jerk, seg.blend);
      std::vector<double*> blocks(seg.r3_knots.begin(), seg.r3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }

    // Angular acceleration on the SO(3) knots.
    {
      ceres::CostFunction* cost = makeAngularAccelCost(seg.u, inv_dt, w_accel, seg.cum_blend);
      std::vector<double*> blocks(seg.so3_knots.begin(), seg.so3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    ++added;
  }
  return added;
}

}  // namespace meridian::ct
