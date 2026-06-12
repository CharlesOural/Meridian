#include "lio/imu_tracker.hpp"

#include <Eigen/Geometry>
#include <sophus/so3.hpp>

namespace meridian::lio {
namespace {
constexpr double square(double x) {
  return x * x;
}
}  // namespace

ImuTracker::ImuTracker(const FrontendLio& cfg) : cfg_(cfg) {}

void ImuTracker::add_sample(const ImuSample& s) {
  // The boundary sample of a measure group is re-delivered at the head of the next
  // group; anything at or before the watermark has already been consumed.
  if (s.stamp <= last_stamp_) {
    return;
  }
  last_stamp_ = s.stamp;

  if (!initialized_) {
    if (cfg_.init_stationary_s > 0.0) {
      accumulateInit(s);
      return;
    }
    // No standstill window: dead-reckon from the first sample with identity attitude
    // and zero biases; the sample itself still enters the interval statistics.
    state_.stamp = s.stamp;
    initialized_ = true;
  }

  propagateAndAccumulate(s);
}

bool ImuTracker::initialized() const {
  return initialized_;
}

void ImuTracker::accumulateInit(const ImuSample& s) {
  if (init_count_ == 0) {
    init_begin_ = s.stamp;
  }
  ++init_count_;
  init_gyro_sum_ += s.gyro;
  init_acc_sum_ += s.acc;
  if (to_seconds(s.stamp - init_begin_) >= cfg_.init_stationary_s) {
    finishStaticInit(s.stamp);
  }
}

void ImuTracker::finishStaticInit(Timestamp stamp) {
  const Eigen::Vector3d mean_gyro = init_gyro_sum_ / init_count_;
  const Eigen::Vector3d mean_acc = init_acc_sum_ / init_count_;

  // At rest the accelerometer reads the gravity reaction f = -R^T g, pointing up in
  // body coordinates. The minimal rotation taking the mean reading onto +z_world
  // recovers roll/pitch; its axis is the cross product with z and so lies in the x-y
  // plane, leaving yaw exactly zero.
  const Eigen::Quaterniond q0 =
      Eigen::Quaterniond::FromTwoVectors(mean_acc, Eigen::Vector3d::UnitZ());

  state_.T_world_body = Pose{q0, Eigen::Vector3d::Zero()};
  state_.v_world.setZero();
  state_.stamp = stamp;
  state_.g_world = Eigen::Vector3d(0.0, 0.0, -kGravityMagnitude);
  state_.b_g = mean_gyro;
  // Whatever the mean reading carries beyond the expected rest reaction is accel bias.
  // Only the along-gravity component is observable at rest; the orthogonal part is
  // absorbed into the recovered tilt.
  state_.b_a = mean_acc - q0.conjugate() * (kGravityMagnitude * Eigen::Vector3d::UnitZ());

  // The window asserted standstill, so the unbiased rate is zero by construction.
  last_gyro_.setZero();
  initialized_ = true;
}

void ImuTracker::propagateAndAccumulate(const ImuSample& s) {
  const Eigen::Quaterniond q = state_.T_world_body.q;
  const Eigen::Vector3d omega = s.gyro - state_.b_g;  // body rate [rad/s]
  const Eigen::Vector3d f_body = s.acc - state_.b_a;  // specific force [m/s^2]
  // f = R^T (a - g)  =>  true body acceleration a_body = f + R^T g.
  const Eigen::Vector3d accel_body = f_body + q.conjugate() * state_.g_world;

  // A sample can be fresh (past the watermark) yet stamped at or before a rebased
  // state; it then contributes statistics but cannot be integrated.
  const double dt = to_seconds(s.stamp - state_.stamp);
  if (dt > 0.0) {
    const Eigen::Vector3d accel_world = q * accel_body;
    state_.T_world_body.t += state_.v_world * dt + 0.5 * dt * dt * accel_world;
    state_.v_world += accel_world * dt;
    state_.T_world_body.q = (q * Sophus::SO3d::exp(omega * dt).unit_quaternion()).normalized();
    state_.stamp = s.stamp;
  }

  ++interval_count_;
  gyro_sum_ += omega;
  accel_sum_ += accel_body;
  // Welford single pass on the specific-force magnitude: numerically stable running
  // mean and sum of squared deviations.
  const double mag = f_body.norm();
  const double prev_mean = mag_mean_;
  mag_mean_ += (mag - prev_mean) / interval_count_;
  mag_m2_ += (mag - prev_mean) * (mag - mag_mean_);

  last_gyro_ = omega;
}

MotionPrior ImuTracker::interval_prior(Timestamp t_begin, Timestamp t_end) {
  MotionPrior out;
  if (interval_count_ == 0) {
    // Nothing arrived: hold the last measured rate, assume no linear acceleration, and
    // report the spread as uninformative.
    out.omega_body = last_gyro_;
    out.accel_mag_variance = kUninformativeAccelVariance;
    return out;
  }

  out.imu_count = interval_count_;
  out.omega_body = gyro_sum_ / interval_count_;
  out.accel_body = accel_sum_ / interval_count_;

  if (interval_count_ >= 2) {
    // Unbiased sample variance of the specific-force magnitude over the interval.
    out.accel_mag_variance = mag_m2_ / (interval_count_ - 1);

    const double dt = to_seconds(t_end - t_begin);
    if (dt > 0.0) {
      // Scalar predict/update on the body-acceleration magnitude. Jerk uniform on
      // [-j, j] has variance (2j)^2 / 12 = j^2 / 3; over dt it perturbs the
      // acceleration by j*dt, giving the process noise below. The interval scatter
      // serves as the measurement noise.
      const double process_noise = square(cfg_.max_expected_jerk * dt) / 3.0;
      const double var_pred = accel_kf_var_ + process_noise;
      const double gain = var_pred / (var_pred + out.accel_mag_variance);
      accel_kf_mean_ += gain * (out.accel_body.norm() - accel_kf_mean_);
      accel_kf_var_ = (1.0 - gain) * var_pred;
    }
  } else {
    // A single sample carries no spread information.
    out.accel_mag_variance = kUninformativeAccelVariance;
  }

  interval_count_ = 0;
  gyro_sum_.setZero();
  accel_sum_.setZero();
  mag_mean_ = 0.0;
  mag_m2_ = 0.0;
  return out;
}

void ImuTracker::rebase(const NavState& solved) {
  state_.stamp = solved.stamp;
  state_.T_world_body = solved.T_world_body;
  state_.v_world = solved.v_world;
}

NavState ImuTracker::propagated_state() const {
  return state_;
}

Eigen::Vector3d ImuTracker::bias_gyro() const {
  return state_.b_g;
}

Eigen::Vector3d ImuTracker::bias_accel() const {
  return state_.b_a;
}

double ImuTracker::filtered_accel_magnitude() const {
  return accel_kf_mean_;
}

double ImuTracker::filtered_accel_variance() const {
  return accel_kf_var_;
}

}  // namespace meridian::lio
