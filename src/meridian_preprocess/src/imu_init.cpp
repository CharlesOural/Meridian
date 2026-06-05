#include "meridian/preprocess/imu_init.hpp"

#include <algorithm>
#include <cmath>

#include "meridian/debug/log.hpp"

namespace meridian {

namespace {
constexpr const char* kLogModule = "preprocess.imu";
}

ImuInitializer::ImuInitializer(const PreprocImu& cfg, int imu_init_count)
    : cfg_(cfg), target_(static_cast<std::uint32_t>(std::max(1, imu_init_count))) {}

void ImuInitializer::clear() {
  count_ = 0;
  mean_acc_.setZero();
  mean_gyro_.setZero();
  m2_acc_.setZero();
  m2_gyro_.setZero();
  state_ = ImuInitState{};
  done_ = false;
  failed_ = false;
}

void ImuInitializer::accumulate(const ImuSample& s) {
  ++count_;
  const double n = static_cast<double>(count_);
  // Welford: update the mean, then fold the deviation against the new mean into M2.
  const Eigen::Vector3d da0 = s.acc - mean_acc_;
  mean_acc_ += da0 / n;
  m2_acc_ += da0.cwiseProduct(s.acc - mean_acc_);

  const Eigen::Vector3d dg0 = s.gyro - mean_gyro_;
  mean_gyro_ += dg0 / n;
  m2_gyro_ += dg0.cwiseProduct(s.gyro - mean_gyro_);
}

void ImuInitializer::finalize() {
  const double n = static_cast<double>(count_);
  state_.count = count_;
  state_.mean_acc = mean_acc_;
  state_.mean_gyro = mean_gyro_;
  state_.acc_var = n > 1.0 ? (m2_acc_ / n).eval() : Eigen::Vector3d::Zero();
  state_.gyro_var = n > 1.0 ? (m2_gyro_ / n).eval() : Eigen::Vector3d::Zero();

  // Gravity: negated, normalized mean acceleration scaled to the pinned magnitude.
  // Direction is recovered from the static mean; the magnitude is fixed at G.
  const double acc_norm = mean_acc_.norm();
  if (acc_norm > 0.0) {
    state_.gravity = -mean_acc_ / acc_norm * kGravityMagnitude;
  }
  // Gyro bias is the mean angular rate at rest. Accel bias stays at the zero prior.
  state_.gyro_bias = mean_gyro_;
  state_.accel_bias.setZero();

  // Motion gate: reject if the static variance is too high or the mean specific-force
  // magnitude departs from G (the rig was disturbed during init).
  const double max_var = cfg_.init_max_var;
  const bool var_ok = state_.acc_var.maxCoeff() <= max_var &&
                      state_.gyro_var.maxCoeff() <= max_var;
  const bool grav_ok =
      std::abs(acc_norm - kGravityMagnitude) <= cfg_.init_max_grav_err;

  if (var_ok && grav_ok) {
    done_ = true;
    failed_ = false;
    MERIDIAN_INFO(kLogModule, "event", "imu/init_done", "g_norm", acc_norm, "n", count_);
  } else {
    done_ = false;
    failed_ = true;
    MERIDIAN_WARN(kLogModule, "event", "imu/init_moving", "acc_var",
                  state_.acc_var.maxCoeff(), "gyr_var", state_.gyro_var.maxCoeff(),
                  "g_norm", acc_norm);
    // Discard the rejected window's statistics so the next add() accumulates fresh.
    // failed_ stays set until that next add() opens a new window.
    count_ = 0;
    mean_acc_.setZero();
    mean_gyro_.setZero();
    m2_acc_.setZero();
    m2_gyro_.setZero();
  }
}

bool ImuInitializer::add(const ImuSample& s) {
  if (done_) {
    return true;
  }
  // The first sample after a rejected window opens a fresh attempt; clear the failure
  // flag now that a new window is starting.
  failed_ = false;
  accumulate(s);
  if (count_ >= target_) {
    finalize();
  }
  return done_;
}

}  // namespace meridian
