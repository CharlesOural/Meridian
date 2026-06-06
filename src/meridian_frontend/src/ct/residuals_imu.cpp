#include "residuals_imu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <basalt/spline/ceres_spline_helper.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/manifold.h>
#include <ceres/problem.h>
#include <ceres/sphere_manifold.h>
#include <sophus/so3.hpp>

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
  ImuResidual(const Eigen::Vector3d& gyro, const Eigen::Vector3d& accel, double u,
              double inv_dt, double alpha, double gravity_mag, double w_gyro,
              double w_accel)
      : gyro_(gyro),
        accel_(accel),
        u_(u),
        inv_dt_(inv_dt),
        alpha_(alpha),
        gravity_mag_(gravity_mag),
        w_gyro_(w_gyro),
        w_accel_(w_accel) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;

    SO3T R_w_fe;
    Tangent omega_body;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        params, u_, inv_dt_, &R_w_fe, &omega_body, nullptr);

    Vec3T accel_world;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 2>(
        params + kSplineOrder, u_, inv_dt_, &accel_world);

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
    residual.template head<3>() =
        T(w_gyro_) * (omega_body + b_g - gyro_.cast<T>());
    residual.template tail<3>() =
        T(w_accel_) *
        (R_w_fe.inverse() * (accel_world - g_world) + b_a - accel_.cast<T>());
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
};

// Jerk regularizer: the third real-time derivative of the R^3 (translation) spline at
// one time, weighted low. The basalt helper returns the virtual-time derivative; the
// real-time jerk is that scaled by (1/dt)^3, achieved by feeding inv_dt = 1/dt_s. It
// pulls the segment toward constant acceleration (zero jerk), removing the position
// spline's highest unconstrained derivative without biasing toward any pose.
struct JerkResidual {
  JerkResidual(double u, double inv_dt, double weight)
      : u_(u), inv_dt_(inv_dt), weight_(weight) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Vec3T jerk;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 3>(params, u_, inv_dt_, &jerk);
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * jerk;
    return true;
  }

  double u_;
  double inv_dt_;
  double weight_;
};

// Angular-acceleration regularizer: the body angular acceleration (second derivative of
// the SO(3) spline) at one time, weighted low. It pulls the segment toward constant
// body rate (zero angular acceleration). Same real-time scaling as the jerk term.
struct AngularAccelResidual {
  AngularAccelResidual(double u, double inv_dt, double weight)
      : u_(u), inv_dt_(inv_dt), weight_(weight) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;
    SO3T R;
    Tangent vel;
    Tangent accel;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        params, u_, inv_dt_, &R, &vel, &accel);
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * accel;
    return true;
  }

  double u_;
  double inv_dt_;
  double weight_;
};

// Tail velocity anchor: ties the R^3 spline's first derivative at one time to a
// predicted world velocity. Applied only over the span past the newest measurement,
// where the trailing control points would otherwise sit in the cost's null space: a
// solver-chosen wiggle there is invisible to every measurement residual but corrupts
// the state read-out and the next sweep's evaluation. The weight is an honest
// constant-velocity extrapolation uncertainty, weak enough for real measurements to
// override it as soon as they arrive.
struct VelocityAnchorResidual {
  VelocityAnchorResidual(const Eigen::Vector3d& v_pred, double u, double inv_dt,
                         double weight)
      : v_pred_(v_pred), u_(u), inv_dt_(inv_dt), weight_(weight) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    Vec3T vel;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate<T, 3, 1>(params, u_, inv_dt_,
                                                                        &vel);
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * (vel - v_pred_.cast<T>());
    return true;
  }

  Eigen::Vector3d v_pred_;
  double u_;
  double inv_dt_;
  double weight_;
};

// Tail angular-rate anchor: same role as the velocity anchor for the SO(3) spline,
// tying the body rate to the constant-rate extrapolation of the last gyro sample.
struct RateAnchorResidual {
  RateAnchorResidual(const Eigen::Vector3d& w_pred, double u, double inv_dt, double weight)
      : w_pred_(w_pred), u_(u), inv_dt_(inv_dt), weight_(weight) {}

  template <class T>
  bool operator()(T const* const* params, T* residual_ptr) const {
    using Vec3T = Eigen::Matrix<T, 3, 1>;
    using SO3T = Sophus::SO3<T>;
    using Tangent = typename Sophus::SO3<T>::Tangent;
    SO3T R;
    Tangent omega;
    basalt::CeresSplineHelper<kSplineOrder>::template evaluate_lie<T, Sophus::SO3>(
        params, u_, inv_dt_, &R, &omega, nullptr);
    Eigen::Map<Vec3T> residual(residual_ptr);
    residual = T(weight_) * (omega - w_pred_.cast<T>());
    return true;
  }

  Eigen::Vector3d w_pred_;
  double u_;
  double inv_dt_;
  double weight_;
};

// Random-walk tie between two consecutive bias knots, weighted so the cost equals
// the squared increment divided by the continuous-time variance sigma^2 * dt: the
// sqrt-information weight is 1 / (sigma * sqrt(dt)), shared by the gyro and accel
// halves through their own sigmas. Larger gaps are tied more weakly.
struct BiasTieResidual {
  BiasTieResidual(double w_gyro, double w_accel)
      : w_gyro_(w_gyro), w_accel_(w_accel) {}

  template <class T>
  bool operator()(const T* bg_prev, const T* bg_next, const T* ba_prev,
                  const T* ba_next, T* residual_ptr) const {
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

}  // namespace

GravityBlock::GravityBlock(const Eigen::Vector3d& g_world, double magnitude)
    : magnitude_(magnitude) {
  const double n = g_world.norm();
  // Fall back to "down" if handed a zero vector so the stored value stays unit.
  direction_ = (n > 0.0) ? (g_world / n) : Eigen::Vector3d(0.0, 0.0, -1.0);
}

Eigen::Vector3d GravityBlock::gravityWorld() const {
  return magnitude_ * direction_;
}

BiasKnots::BiasKnots(Timestamp t0, Duration dt_ns, int n_knots)
    : t0_(t0), dt_ns_(dt_ns) {
  const int n = n_knots < 1 ? 1 : n_knots;
  gyro_.assign(static_cast<std::size_t>(n), Eigen::Vector3d::Zero());
  accel_.assign(static_cast<std::size_t>(n), Eigen::Vector3d::Zero());
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

ceres::Manifold* makeGravityManifold() { return new ceres::SphereManifold<3>(); }

ImuExcitation imuExcitation(const std::vector<ImuSample>& samples,
                            const Eigen::Vector3d& b_g, const Eigen::Vector3d& b_a,
                            double gravity_mag) {
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

int knotDensityFromExcitation(const ImuExcitation& exc,
                              const std::vector<double>& omega_thresh,
                              const std::vector<double>& accel_thresh, double hysteresis,
                              int n_cp_max, int prev_n_cp) {
  const int cap = std::max(1, n_cp_max);
  const double hys = std::clamp(hysteresis, 0.0, 0.99);
  // The previous count distributes as bands held on each axis; since n_cp is the larger
  // of the two band counts + 1, the previous band count is prev_n_cp - 1 and we apply
  // the same held-band budget to both axes (the binding axis is whichever held more).
  const int prev_bands = std::max(0, prev_n_cp - 1);
  const int omega_bands = bandsCrossed(exc.n_omega, omega_thresh, hys, prev_bands);
  const int accel_bands = bandsCrossed(exc.n_accel, accel_thresh, hys, prev_bands);
  const int bands = std::max(omega_bands, accel_bands);
  return std::clamp(1 + bands, 1, cap);
}

int addImuResiduals(ceres::Problem& problem, SplineWindow& spline, BiasKnots& bias,
                    GravityBlock& gravity, const std::vector<ImuSample>& samples,
                    Timestamp t_begin, Timestamp t_end, const ImuWeights& weights) {
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
      auto* c = new ceres::DynamicAutoDiffCostFunction<ImuResidual<true>, 4>(
          new ImuResidual<true>(s.gyro, s.acc, seg.u, 1.0 / seg.dt_s, alpha,
                                gravity.magnitude(), w_gyro, w_accel));
      // 4 SO(3) + 4 R^3 + b_g + b_a + gravity.
      for (int i = 0; i < 2 * kSplineOrder + 3; ++i) {
        c->AddParameterBlock(i < kSplineOrder ? 4 : 3);
      }
      c->SetNumResiduals(6);
      cost = c;
      blocks.push_back(bias.gyroBlock(bk));
      blocks.push_back(bias.accelBlock(bk));
    } else {
      auto* c = new ceres::DynamicAutoDiffCostFunction<ImuResidual<false>, 4>(
          new ImuResidual<false>(s.gyro, s.acc, seg.u, 1.0 / seg.dt_s, alpha,
                                 gravity.magnitude(), w_gyro, w_accel));
      // 4 SO(3) + 4 R^3 + bg_left + bg_right + ba_left + ba_right + gravity.
      for (int i = 0; i < 2 * kSplineOrder + 5; ++i) {
        c->AddParameterBlock(i < kSplineOrder ? 4 : 3);
      }
      c->SetNumResiduals(6);
      cost = c;
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
      auto* cost = new ceres::DynamicAutoDiffCostFunction<VelocityAnchorResidual, 4>(
          new VelocityAnchorResidual(v_pred, seg.u, inv_dt, w_vel));
      for (int i = 0; i < kSplineOrder; ++i) {
        cost->AddParameterBlock(3);
      }
      cost->SetNumResiduals(3);
      std::vector<double*> blocks(seg.r3_knots.begin(), seg.r3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    {
      auto* cost = new ceres::DynamicAutoDiffCostFunction<RateAnchorResidual, 4>(
          new RateAnchorResidual(w_pred, seg.u, inv_dt, w_rate));
      for (int i = 0; i < kSplineOrder; ++i) {
        cost->AddParameterBlock(4);
      }
      cost->SetNumResiduals(3);
      std::vector<double*> blocks(seg.so3_knots.begin(), seg.so3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    ++added;
  }
  return added;
}

int addBiasRandomWalk(ceres::Problem& problem, BiasKnots& bias,
                      const ImuWeights& weights) {
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
    auto* cost =
        new ceres::AutoDiffCostFunction<BiasTieResidual, 6, 3, 3, 3, 3>(
            new BiasTieResidual(w_bg, w_ba));
    problem.AddResidualBlock(cost, nullptr, bias.gyroBlock(k - 1),
                             bias.gyroBlock(k), bias.accelBlock(k - 1),
                             bias.accelBlock(k));
    ++added;
  }
  return added;
}

int addMotionRegularizer(ceres::Problem& problem, SplineWindow& spline,
                         const std::vector<Timestamp>& times, double weight,
                         double accel_weight, double gyro_weight) {
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
      auto* cost = new ceres::DynamicAutoDiffCostFunction<JerkResidual, 4>(
          new JerkResidual(seg.u, inv_dt, w_jerk));
      for (int i = 0; i < kSplineOrder; ++i) {
        cost->AddParameterBlock(3);
      }
      cost->SetNumResiduals(3);
      std::vector<double*> blocks(seg.r3_knots.begin(), seg.r3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }

    // Angular acceleration on the SO(3) knots.
    {
      auto* cost = new ceres::DynamicAutoDiffCostFunction<AngularAccelResidual, 4>(
          new AngularAccelResidual(seg.u, inv_dt, w_accel));
      for (int i = 0; i < kSplineOrder; ++i) {
        cost->AddParameterBlock(4);
      }
      cost->SetNumResiduals(3);
      std::vector<double*> blocks(seg.so3_knots.begin(), seg.so3_knots.end());
      problem.AddResidualBlock(cost, nullptr, blocks);
    }
    ++added;
  }
  return added;
}

}  // namespace meridian::ct
