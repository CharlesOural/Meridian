#include "meridian/local_rt/combined_preintegration.hpp"

#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/PreintegrationCombinedParams.h>

#include <Eigen/Geometry>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include "gtsam_inertial_internal.hpp"

namespace meridian::local_rt {
namespace {

gtsam::Vector3 gtsamVector(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

core::Vec3d coreVector(const gtsam::Vector3& vector) {
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

gtsam::imuBias::ConstantBias gtsamBias(const core::ImuBias& bias) {
  return {gtsamVector(bias.accelerometerMS2()), gtsamVector(bias.gyroscopeRadS())};
}

template <typename Params>
void zeroLegacyInitialBiasCovariance(Params& params) {
  if constexpr (requires(Params & value) { value.biasAccOmegaInt; }) {
    params.biasAccOmegaInt.setZero();
  }
}

}  // namespace

CombinedPreintegration::CombinedPreintegration(
    double duration_seconds, core::Quaterniond delta_rotation, core::Vec3d delta_position,
    core::Vec3d delta_velocity, Eigen::Matrix<double, 15, 15> covariance,
    core::ImuBias linearization_bias, std::shared_ptr<const Impl> impl)
    : duration_seconds_(duration_seconds),
      delta_rotation_(std::move(delta_rotation)),
      delta_position_(delta_position),
      delta_velocity_(delta_velocity),
      covariance_(std::move(covariance)),
      linearization_bias_(linearization_bias),
      impl_(std::move(impl)) {}

PreintegrationResult::PreintegrationResult(CombinedPreintegration preintegration)
    : result_(std::move(preintegration)) {}

PreintegrationResult::PreintegrationResult(PreintegrationFailure failure)
    : result_(std::move(failure)) {}

bool PreintegrationResult::ok() const noexcept {
  return std::holds_alternative<CombinedPreintegration>(result_);
}

const CombinedPreintegration* PreintegrationResult::value() const noexcept {
  return std::get_if<CombinedPreintegration>(&result_);
}

const PreintegrationFailure* PreintegrationResult::error() const noexcept {
  return std::get_if<PreintegrationFailure>(&result_);
}

GtsamCombinedPreintegrator::GtsamCombinedPreintegrator(ImuModel model) : model_(std::move(model)) {
  Config validation_config;
  validation_config.imu_model = model_;
  const std::vector<ConfigIssue> issues = validation_config.validate();
  for (const ConfigIssue& issue : issues) {
    if (issue.field.starts_with("imu_model.")) {
      throw std::invalid_argument("the GTSAM preintegrator requires a valid IMU model");
    }
  }
}

PreintegrationResult GtsamCombinedPreintegrator::integrate(const ImuInterval& interval,
                                                           const core::ImuBias& bias) const {
  if (interval.segments().empty()) {
    return PreintegrationResult(PreintegrationFailure{
        .code = PreintegrationErrorCode::kEmptyInterval,
        .message = "combined preintegration requires at least one IMU segment"});
  }

  try {
    auto params = std::make_shared<gtsam::PreintegrationCombinedParams>(
        gtsamVector(model_.gravity_odom_m_s2));
    params->setAccelerometerCovariance(model_.accelerometer_covariance_density);
    params->setGyroscopeCovariance(model_.gyroscope_covariance_density);
    params->setIntegrationCovariance(model_.integration_covariance_density);
    params->setBiasAccCovariance(model_.accelerometer_bias_random_walk_covariance);
    params->setBiasOmegaCovariance(model_.gyroscope_bias_random_walk_covariance);
    zeroLegacyInitialBiasCovariance(*params);

    gtsam::PreintegratedCombinedMeasurements pim(params, gtsamBias(bias));
    for (const ImuIntegrationSegment& segment : interval.segments()) {
      const double dt = segment.durationSeconds();
      if (!(dt > 0.0) || !segment.angularVelocityRadS().array().isFinite().all() ||
          !segment.specificForceMS2().array().isFinite().all()) {
        return PreintegrationResult(PreintegrationFailure{
            .code = PreintegrationErrorCode::kInvalidMeasurement,
            .message = "an IMU integration segment is non-finite or has invalid duration"});
      }
      pim.integrateMeasurement(segment.specificForceMS2(), segment.angularVelocityRadS(), dt);
    }

    const Eigen::Quaterniond quaternion(pim.deltaRij().toQuaternion());
    auto impl = std::make_shared<const CombinedPreintegration::Impl>(std::move(pim));
    return PreintegrationResult(CombinedPreintegration(
        impl->pim.deltaTij(),
        core::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()),
        coreVector(impl->pim.deltaPij()), coreVector(impl->pim.deltaVij()),
        impl->pim.preintMeasCov(), bias, impl));
  } catch (const std::exception& error) {
    return PreintegrationResult(PreintegrationFailure{
        .code = PreintegrationErrorCode::kBackendFailure, .message = error.what()});
  }
}

std::string_view GtsamCombinedPreintegrator::backendName() noexcept {
#ifdef GTSAM_TANGENT_PREINTEGRATION
  return "gtsam-resolved-tangent";
#else
  return "gtsam-resolved-manifold";
#endif
}

}  // namespace meridian::local_rt
