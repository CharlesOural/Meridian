#include "lio/imu_tracker.hpp"

namespace meridian::lio {

ImuTracker::ImuTracker(const FrontendLio& cfg) : cfg_(cfg) {}

void ImuTracker::add_sample(const ImuSample& s) {
  // TODO(lio): implemented in a later lane
  (void)s;
}

bool ImuTracker::initialized() const {
  // TODO(lio): implemented in a later lane
  return false;
}

MotionPrior ImuTracker::interval_prior(Timestamp t_begin, Timestamp t_end) {
  // TODO(lio): implemented in a later lane
  (void)t_begin;
  (void)t_end;
  return MotionPrior{};
}

void ImuTracker::rebase(const NavState& solved) {
  // TODO(lio): implemented in a later lane
  (void)solved;
}

NavState ImuTracker::propagated_state() const { return state_; }

Eigen::Vector3d ImuTracker::bias_gyro() const { return state_.b_g; }

Eigen::Vector3d ImuTracker::bias_accel() const { return state_.b_a; }

}  // namespace meridian::lio
