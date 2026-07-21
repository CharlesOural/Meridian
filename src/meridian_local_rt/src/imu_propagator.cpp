#include "meridian/local_rt/imu_propagator.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace meridian::local_rt {
namespace {

Eigen::Vector3d eigen(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

Eigen::Quaterniond eigen(const core::Quaterniond& quaternion) {
  return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

core::Vec3d coreVector(const Eigen::Vector3d& vector) {
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

core::Pose3d corePose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
  const Eigen::Quaterniond normalized = orientation.normalized();
  return core::Pose3d(coreVector(position), core::Quaterniond(normalized.w(), normalized.x(),
                                                              normalized.y(), normalized.z()));
}

Eigen::Quaterniond rotationIncrement(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  if (angle < 1.0e-12) {
    Eigen::Quaterniond result(1.0, 0.5 * rotation_vector.x(), 0.5 * rotation_vector.y(),
                              0.5 * rotation_vector.z());
    return result.normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

bool finite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

}  // namespace

PropagationResult::PropagationResult(DensePropagation propagation)
    : result_(std::move(propagation)) {}

PropagationResult::PropagationResult(PropagationFailure failure) : result_(std::move(failure)) {}

bool PropagationResult::ok() const noexcept {
  return std::holds_alternative<DensePropagation>(result_);
}

const DensePropagation* PropagationResult::value() const noexcept {
  return std::get_if<DensePropagation>(&result_);
}

const PropagationFailure* PropagationResult::error() const noexcept {
  return std::get_if<PropagationFailure>(&result_);
}

ImuPropagator::ImuPropagator(ImuModel model) : model_(std::move(model)) {
  Config validation_config;
  validation_config.imu_model = model_;
  for (const ConfigIssue& issue : validation_config.validate()) {
    if (issue.field.starts_with("imu_model.")) {
      throw std::invalid_argument("the IMU propagator requires a valid IMU model");
    }
  }
}

PropagationResult ImuPropagator::propagate(const core::NavigationState& seed,
                                           core::StateId endpoint_id,
                                           const ImuInterval& interval) const {
  if (interval.segments().empty()) {
    return PropagationResult(
        PropagationFailure{.code = PropagationErrorCode::kEmptyInterval,
                           .message = "dense propagation requires at least one IMU segment"});
  }
  if (seed.time() != interval.support().begin()) {
    return PropagationResult(PropagationFailure{
        .code = PropagationErrorCode::kSeedTimeMismatch,
        .message = "the navigation seed time must equal the IMU interval begin"});
  }

  Eigen::Vector3d position = eigen(seed.odomFromImu().translation());
  Eigen::Quaterniond orientation = eigen(seed.odomFromImu().rotation());
  Eigen::Vector3d velocity = eigen(seed.velocityOdomMS());
  const Eigen::Vector3d gyroscope_bias = eigen(seed.imuBias().gyroscopeRadS());
  const Eigen::Vector3d accelerometer_bias = eigen(seed.imuBias().accelerometerMS2());
  const Eigen::Vector3d gravity = eigen(model_.gravity_odom_m_s2);

  std::vector<DenseImuSample> samples;
  samples.reserve(interval.segments().size());
  for (const ImuIntegrationSegment& segment : interval.segments()) {
    const double dt = segment.durationSeconds();
    const Eigen::Vector3d angular_velocity = segment.angularVelocityRadS() - gyroscope_bias;
    const Eigen::Vector3d specific_force = segment.specificForceMS2() - accelerometer_bias;
    const Eigen::Quaterniond midpoint_orientation =
        orientation * rotationIncrement(0.5 * dt * angular_velocity);
    const Eigen::Vector3d acceleration = midpoint_orientation * specific_force + gravity;

    position += dt * velocity + 0.5 * dt * dt * acceleration;
    velocity += dt * acceleration;
    orientation = (orientation * rotationIncrement(dt * angular_velocity)).normalized();

    if (!finite(position) || !finite(velocity) || !orientation.coeffs().array().isFinite().all()) {
      return PropagationResult(
          PropagationFailure{.code = PropagationErrorCode::kNonFiniteResult,
                             .message = "dense IMU propagation produced a non-finite state"});
    }
    samples.push_back({.time = segment.support().end(),
                       .odom_from_imu = corePose(position, orientation),
                       .velocity_odom_m_s = coreVector(velocity)});
  }

  core::NavigationState endpoint(endpoint_id, interval.support().end(),
                                 corePose(position, orientation), coreVector(velocity),
                                 seed.imuBias());
  return PropagationResult(
      DensePropagation{.samples = std::move(samples), .endpoint = std::move(endpoint)});
}

}  // namespace meridian::local_rt
