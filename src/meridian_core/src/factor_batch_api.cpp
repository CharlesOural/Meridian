#include "meridian/core/factor_batch_api.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "meridian/core/blob.hpp"

namespace meridian::core {
namespace {

[[nodiscard]] bool validModality(SensorModality modality) noexcept {
  switch (modality) {
    case SensorModality::Lidar:
    case SensorModality::Visual:
    case SensorModality::Gnss:
      return true;
  }
  return false;
}

[[nodiscard]] bool validHealthState(SensorHealthState state) noexcept {
  switch (state) {
    case SensorHealthState::Active:
    case SensorHealthState::Suspect:
    case SensorHealthState::Failed:
    case SensorHealthState::Recovering:
      return true;
  }
  return false;
}

[[nodiscard]] bool validDirectionalVariable(DirectionalVariable variable) noexcept {
  switch (variable) {
    case DirectionalVariable::PoseTranslation:
    case DirectionalVariable::PoseRotation:
    case DirectionalVariable::Velocity:
    case DirectionalVariable::AccelerometerBias:
    case DirectionalVariable::GyroscopeBias:
      return true;
  }
  return false;
}

[[nodiscard]] bool validEndpointRole(DirectionalEndpointRole role) noexcept {
  switch (role) {
    case DirectionalEndpointRole::Target:
    case DirectionalEndpointRole::Source:
    case DirectionalEndpointRole::Unary:
      return true;
  }
  return false;
}

[[nodiscard]] bool withinSupport(FusionTime time, const TimeRange& support) noexcept {
  return support.contains(time);
}

[[nodiscard]] bool validHeader(const RecordHeader& header) noexcept {
  return header.schema_version != 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() && header.direct_calibration.has_value() &&
         header.direct_calibration->valid();
}

}  // namespace

bool SensorInstanceId::valid() const noexcept {
  return validModality(modality) && instance != kInvalidInstance;
}

SensorHealthValidationError validateSensorHealthSnapshot(
    const SensorHealthSnapshot& snapshot) noexcept {
  if (!snapshot.sensor.valid()) {
    return SensorHealthValidationError::InvalidSensor;
  }
  if (!validHealthState(snapshot.state)) {
    return SensorHealthValidationError::InvalidState;
  }
  if (!snapshot.recovery_epoch.valid()) {
    return SensorHealthValidationError::InvalidRecoveryEpoch;
  }
  if (snapshot.transition_sequence == kInvalidSensorHealthTransitionSequence) {
    return SensorHealthValidationError::InvalidTransitionSequence;
  }
  return SensorHealthValidationError::None;
}

FactorBatchTimingValidationError validateFactorBatchTiming(
    const FactorBatchTiming& timing) noexcept {
  if (!timing.support.valid()) {
    return FactorBatchTimingValidationError::InvalidSupport;
  }
  if (timing.measurement_timestamps.empty() ||
      timing.measurement_timestamps.size() > kMaximumFactorBatchMeasurementTimestamps) {
    return FactorBatchTimingValidationError::InvalidMeasurementCount;
  }
  if (!std::is_sorted(timing.measurement_timestamps.begin(), timing.measurement_timestamps.end()) ||
      std::adjacent_find(timing.measurement_timestamps.begin(),
                         timing.measurement_timestamps.end()) !=
          timing.measurement_timestamps.end()) {
    return FactorBatchTimingValidationError::NonCanonicalMeasurementTimestamps;
  }
  if (!std::all_of(
          timing.measurement_timestamps.begin(), timing.measurement_timestamps.end(),
          [&](FusionTime timestamp) { return withinSupport(timestamp, timing.support); })) {
    return FactorBatchTimingValidationError::MeasurementOutsideSupport;
  }
  if (!withinSupport(timing.reference_time, timing.support)) {
    return FactorBatchTimingValidationError::ReferenceOutsideSupport;
  }
  if (timing.produced_at < timing.support.end) {
    return FactorBatchTimingValidationError::ProducedBeforeSupportEnd;
  }
  return FactorBatchTimingValidationError::None;
}

DirectionalObservabilityValidationError validateDirectionalObservability(
    const DirectionalObservability& observability) noexcept {
  if (observability.tangent != PoseTangentConvention::RightTranslationFirst) {
    return DirectionalObservabilityValidationError::UnsupportedTangent;
  }
  if (!observability.basis.allFinite()) {
    return DirectionalObservabilityValidationError::NonFiniteBasis;
  }
  const Eigen::Matrix<double, 6, 6> gram = observability.basis.transpose() * observability.basis;
  if (!gram.isApprox(Eigen::Matrix<double, 6, 6>::Identity(), 1.0e-9)) {
    return DirectionalObservabilityValidationError::NonOrthonormalBasis;
  }
  if (!observability.eigenvalues.allFinite()) {
    return DirectionalObservabilityValidationError::InvalidEigenvalues;
  }
  for (Eigen::Index index = 0; index < observability.eigenvalues.size(); ++index) {
    if (observability.eigenvalues(index) < 0.0 ||
        (index > 0 && observability.eigenvalues(index) < observability.eigenvalues(index - 1))) {
      return DirectionalObservabilityValidationError::InvalidEigenvalues;
    }
  }
  if (!std::isfinite(observability.absolute_eigenvalue_threshold) ||
      !std::isfinite(observability.relative_eigenvalue_threshold) ||
      observability.absolute_eigenvalue_threshold <= 0.0 ||
      observability.relative_eigenvalue_threshold < 0.0 ||
      observability.relative_eigenvalue_threshold > 1.0) {
    return DirectionalObservabilityValidationError::InvalidThresholds;
  }
  const double threshold =
      std::max(observability.absolute_eigenvalue_threshold,
               observability.relative_eigenvalue_threshold * observability.eigenvalues(5));
  const std::uint32_t computed_rank =
      static_cast<std::uint32_t>((observability.eigenvalues.array() > threshold).count());
  if (observability.rank > 6U || observability.rank != computed_rank) {
    return DirectionalObservabilityValidationError::RankMismatch;
  }
  if (observability.supported_variables.empty() ||
      observability.supported_variables.size() > kMaximumDirectionalObservabilityVariables ||
      !std::all_of(observability.supported_variables.begin(),
                   observability.supported_variables.end(), validDirectionalVariable) ||
      !std::is_sorted(observability.supported_variables.begin(),
                      observability.supported_variables.end()) ||
      std::adjacent_find(observability.supported_variables.begin(),
                         observability.supported_variables.end()) !=
          observability.supported_variables.end()) {
    return DirectionalObservabilityValidationError::InvalidSupportedVariables;
  }
  if (observability.endpoints.empty() ||
      observability.endpoints.size() > kMaximumDirectionalObservabilityEndpoints) {
    return DirectionalObservabilityValidationError::InvalidEndpoints;
  }
  for (std::size_t index = 0U; index < observability.endpoints.size(); ++index) {
    if (!validEndpointRole(observability.endpoints[index].role) ||
        !observability.endpoints[index].state.valid()) {
      return DirectionalObservabilityValidationError::InvalidEndpoints;
    }
  }
  if (observability.endpoints.size() == 1U) {
    if (observability.endpoints.front().role != DirectionalEndpointRole::Unary) {
      return DirectionalObservabilityValidationError::InvalidEndpoints;
    }
  } else if (observability.endpoints[0].role != DirectionalEndpointRole::Target ||
             observability.endpoints[1].role != DirectionalEndpointRole::Source ||
             !(observability.endpoints[0].state < observability.endpoints[1].state) ||
             !(observability.endpoints[0].exact_time < observability.endpoints[1].exact_time)) {
    return DirectionalObservabilityValidationError::InvalidEndpoints;
  }
  return DirectionalObservabilityValidationError::None;
}

FactorBatchMetadataValidationError validateFactorBatchMetadata(
    const FactorBatchMetadata& metadata) noexcept {
  if (!validHeader(metadata.header) || metadata.header.created_at < metadata.timing.produced_at) {
    return FactorBatchMetadataValidationError::InvalidHeader;
  }
  if (!metadata.batch_id.valid()) {
    return FactorBatchMetadataValidationError::InvalidBatchId;
  }
  if (!metadata.odom_epoch.valid()) {
    return FactorBatchMetadataValidationError::InvalidOdomEpoch;
  }
  if (!metadata.sensor.valid()) {
    return FactorBatchMetadataValidationError::InvalidSensor;
  }
  if (validateFactorBatchTiming(metadata.timing) != FactorBatchTimingValidationError::None) {
    return FactorBatchMetadataValidationError::InvalidTiming;
  }
  if (validateSensorHealthSnapshot(metadata.health) != SensorHealthValidationError::None) {
    return FactorBatchMetadataValidationError::InvalidHealth;
  }
  if (metadata.sensor != metadata.health.sensor) {
    return FactorBatchMetadataValidationError::SensorHealthMismatch;
  }
  if (metadata.health.assessed_at < metadata.timing.support.end ||
      metadata.timing.produced_at < metadata.health.assessed_at) {
    return FactorBatchMetadataValidationError::InvalidHealthAssessmentTime;
  }
  if (metadata.directional_observability.empty() ||
      metadata.directional_observability.size() > kMaximumDirectionalObservabilityRecords) {
    return FactorBatchMetadataValidationError::InvalidObservabilityCount;
  }
  for (const DirectionalObservability& observability : metadata.directional_observability) {
    if (validateDirectionalObservability(observability) !=
        DirectionalObservabilityValidationError::None) {
      return FactorBatchMetadataValidationError::InvalidDirectionalObservability;
    }
  }
  if (metadata.map_eligible && metadata.health.state != SensorHealthState::Active) {
    return FactorBatchMetadataValidationError::MapEligibilityRequiresActiveHealth;
  }
  if (!metadata.lineage.id.valid() || !contentHashPresent(metadata.lineage.checksum) ||
      validateLineage(metadata.lineage) != LineageValidationError::None) {
    return FactorBatchMetadataValidationError::InvalidLineage;
  }
  return FactorBatchMetadataValidationError::None;
}

}  // namespace meridian::core
