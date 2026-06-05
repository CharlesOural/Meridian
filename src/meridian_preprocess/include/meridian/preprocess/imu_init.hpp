#pragma once

#include <cstdint>

#include <Eigen/Core>

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

// Gravity magnitude pinned during static init [m/s^2]. The initializer fixes only the
// gravity direction; the magnitude is held at this constant on the sphere.
inline constexpr double kGravityMagnitude = 9.81;

// The one-time stationary IMU initialization result. Gravity direction and gyro bias
// are recovered from a static window; the accel bias is NOT initialized (it is
// unobservable while static and is left at its zero prior for L2 to estimate).
struct ImuInitState {
  // Gravity in the IMU/body frame [m/s^2], magnitude pinned to kGravityMagnitude.
  Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
  // Gyro bias [rad/s] = mean angular rate at rest.
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  // Accel bias [m/s^2]: left at zero (not initialized while static).
  Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();

  // Empirical static variances accumulated over the init window, for the sanity check
  // and telemetry (NOT used as the live noise model unless the config asks).
  Eigen::Vector3d acc_var = Eigen::Vector3d::Zero();   // [ (m/s^2)^2 ]
  Eigen::Vector3d gyro_var = Eigen::Vector3d::Zero();  // [ (rad/s)^2 ]

  Eigen::Vector3d mean_acc = Eigen::Vector3d::Zero();   // [m/s^2]
  Eigen::Vector3d mean_gyro = Eigen::Vector3d::Zero();  // [rad/s]

  std::uint32_t count = 0;  // samples folded so far
};

// Accumulates a static window of IMU samples with Welford's online mean/variance and,
// once imu_init_count samples are in, recovers gravity direction + gyro bias, gating on
// motion (init_max_var, init_max_grav_err). The accel bias is deliberately not set.
//
// Thread-confined: driven from a single stage thread.
class ImuInitializer {
 public:
  explicit ImuInitializer(const PreprocImu& cfg, int imu_init_count);

  // Folds one sample into the running statistics. After the count reaches the target
  // it evaluates the motion gate exactly once and sets done()/failed() accordingly.
  // Returns true once initialization has converged (done()).
  bool add(const ImuSample& s);

  bool done() const { return done_; }
  // True when the target count was reached but the motion gate rejected the window;
  // the caller clears() and retries on the next static batch.
  bool failed() const { return failed_; }

  const ImuInitState& state() const { return state_; }

  // Discards all accumulated statistics so a fresh static window can be tried.
  void clear();

 private:
  // Folds one sample with Welford's recurrence: updates the running means and the
  // unnormalized second moments.
  void accumulate(const ImuSample& s);
  // Runs once when count_ reaches target_: applies gravity recovery + gyro bias and
  // the motion gate, populating state_ and the done_/failed_ flags.
  void finalize();

  PreprocImu cfg_;
  std::uint32_t target_ = 0;
  std::uint32_t count_ = 0;

  Eigen::Vector3d mean_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d mean_gyro_ = Eigen::Vector3d::Zero();
  // Welford running sums of squared deviations (M2); variance = M2 / count.
  Eigen::Vector3d m2_acc_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d m2_gyro_ = Eigen::Vector3d::Zero();

  ImuInitState state_{};
  bool done_ = false;
  bool failed_ = false;
};

}  // namespace meridian
