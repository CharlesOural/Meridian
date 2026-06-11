#pragma once

#include <Eigen/Core>

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

  // Gravity magnitude pinned during static init [m/s^2]: the initializer fixes only the
  // direction; the magnitude is held at this constant on the sphere.
  static constexpr double kGravityMagnitude = 9.81;

private:
  FrontendLio cfg_;
  NavState state_;
};

}  // namespace meridian::lio
