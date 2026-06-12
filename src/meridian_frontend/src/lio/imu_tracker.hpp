#pragma once

#include <Eigen/Core>
#include <limits>

#include "meridian/common/nav_state.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace meridian::lio {

// Interval-averaged, gravity-compensated body motion over one scan interval: the mean
// angular rate and linear acceleration plus the scatter of the acceleration magnitude
// (the excitation signal the gravity-regularizer weight adapts to).
struct MotionPrior {
  Eigen::Vector3d omega_body = Eigen::Vector3d::Zero();  // [rad/s]
  Eigen::Vector3d accel_body = Eigen::Vector3d::Zero();  // [m/s^2], gravity removed
  double accel_mag_variance = 0.0;                       // [(m/s^2)^2]
  int imu_count = 0;                                     // samples averaged
};

// Owns the IMU-rate state: static initialization (gravity alignment + bias at rest),
// dead-reckoning propagation between scans, and the running interval statistics the
// registration prior consumes.
class ImuTracker {
public:
  explicit ImuTracker(const FrontendLio& cfg);

  // Ingest one body-frame sample. A stamp already seen is dropped (the boundary sample
  // straddles two consecutive groups); once initialized each new sample propagates the
  // IMU-rate state.
  void add_sample(const ImuSample& s);

  // True once static initialization has fixed gravity and the at-rest biases.
  bool initialized() const;

  // Consume the accumulated statistics over [t_begin, t_end]. With no samples in the
  // interval the prior falls back to zero mean motion with imu_count == 0, which the
  // caller must treat as uninformative.
  MotionPrior interval_prior(Timestamp t_begin, Timestamp t_end);

  // Reset the IMU-rate state to the registered pose, so propagation resumes from the
  // solved estimate instead of the dead-reckoned one.
  void rebase(const NavState& solved);

  // The IMU-rate state; runs ahead of the registered pose between scans.
  NavState propagated_state() const;

  Eigen::Vector3d bias_gyro() const;
  Eigen::Vector3d bias_accel() const;

  // Cross-interval filtered estimate of the body-acceleration magnitude and its
  // variance, driven by the uniform-jerk process model.
  double filtered_accel_magnitude() const;
  double filtered_accel_variance() const;

  // Gravity magnitude pinned during static init [m/s^2]: the initializer fixes only the
  // direction; the magnitude is held at this constant on the sphere.
  static constexpr double kGravityMagnitude = 9.81;

  // Reported when an interval holds fewer than two samples: large enough that the
  // excitation-scaled regularizer weight downstream suppresses the gravity term to
  // numerical irrelevance.
  static constexpr double kUninformativeAccelVariance = 1e9;

private:
  void accumulateInit(const ImuSample& s);
  void finishStaticInit(Timestamp stamp);
  void propagateAndAccumulate(const ImuSample& s);

  FrontendLio cfg_;
  NavState state_;
  bool initialized_ = false;

  // Dedup watermark: samples at or before this stamp have already been consumed.
  Timestamp last_stamp_ = std::numeric_limits<Timestamp>::min();
  // Last unbiased rate seen; the no-sample fallback of interval_prior() holds it.
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero();

  // Standstill accumulation while static initialization is pending.
  Timestamp init_begin_ = 0;
  int init_count_ = 0;
  Eigen::Vector3d init_gyro_sum_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d init_acc_sum_ = Eigen::Vector3d::Zero();

  // Interval accumulators, reset by interval_prior().
  int interval_count_ = 0;
  Eigen::Vector3d gyro_sum_ = Eigen::Vector3d::Zero();   // unbiased rates
  Eigen::Vector3d accel_sum_ = Eigen::Vector3d::Zero();  // gravity-compensated body accel
  double mag_mean_ = 0.0;  // Welford running mean of the specific-force magnitude
  double mag_m2_ = 0.0;    // Welford sum of squared deviations

  // Scalar filter state on the body-acceleration magnitude; persists across intervals.
  double accel_kf_mean_ = 0.0;
  double accel_kf_var_ = 1.0;
};

}  // namespace meridian::lio
