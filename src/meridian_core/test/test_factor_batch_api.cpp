#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "meridian/core/factor_batch_api.hpp"

namespace meridian::core {
namespace {

[[nodiscard]] ContentHash presentHash(std::uint8_t value) {
  ContentHash hash{};
  hash.front() = value;
  return hash;
}

[[nodiscard]] ObservationLineage makeLineage() {
  ObservationLineage lineage;
  lineage.id = ObservationLineageId{30U};
  lineage.checksum = presentHash(1U);
  ObservationSlice slice;
  slice.root = MeasurementId{31U};
  slice.calibration = CalibrationEpoch{5U};
  lineage.usage.push_back(ObservationUsage{slice, ObservationRole::PrimaryResidual,
                                           DerivedRecordId{32U}, FactorGroupId{33U}, std::nullopt});
  return lineage;
}

[[nodiscard]] DirectionalObservability makeObservability() {
  DirectionalObservability observability;
  observability.eigenvalues << 0.01, 0.1, 0.2, 1.0, 2.0, 3.0;
  observability.rank = 3U;
  observability.absolute_eigenvalue_threshold = 0.5;
  observability.relative_eigenvalue_threshold = 0.1;
  observability.supported_variables = {
      DirectionalVariable::PoseTranslation, DirectionalVariable::PoseRotation,
      DirectionalVariable::Velocity,        DirectionalVariable::AccelerometerBias,
      DirectionalVariable::GyroscopeBias,
  };
  observability.endpoints = {
      {DirectionalEndpointRole::Target, StateId{10U}, FusionTime{110}},
      {DirectionalEndpointRole::Source, StateId{20U}, FusionTime{190}},
  };
  return observability;
}

[[nodiscard]] FactorBatchMetadata makeMetadata() {
  FactorBatchMetadata metadata;
  metadata.header.schema_version = 1U;
  metadata.header.trace = TraceId{1U};
  metadata.header.producer = ProducerId{2U};
  metadata.header.session = SessionId{3U};
  metadata.header.created_at = FusionTime{230};
  metadata.header.config = ConfigRevision{4U};
  metadata.header.direct_calibration = CalibrationEpoch{5U};
  metadata.batch_id = FactorBatchId{6U};
  metadata.odom_epoch = OdomEpoch{7U};
  metadata.sensor = SensorInstanceId::lidar(LidarId{8U});
  metadata.timing.support = TimeRange{FusionTime{100}, FusionTime{200}};
  metadata.timing.measurement_timestamps = {FusionTime{110}, FusionTime{150}, FusionTime{190}};
  metadata.timing.reference_time = FusionTime{190};
  metadata.timing.produced_at = FusionTime{220};
  metadata.health.sensor = metadata.sensor;
  metadata.health.state = SensorHealthState::Active;
  metadata.health.recovery_epoch = SensorRecoveryEpoch{0U};
  metadata.health.transition_sequence = 0U;
  metadata.health.assessed_at = FusionTime{210};
  metadata.map_eligible = true;
  metadata.directional_observability.push_back(makeObservability());
  metadata.lineage = makeLineage();
  return metadata;
}

TEST(SensorInstanceId, TypedFactoriesPreserveModalityAndIdentityDomains) {
  const SensorInstanceId camera = SensorInstanceId::camera(CameraId{7U});
  const SensorInstanceId lidar = SensorInstanceId::lidar(LidarId{7U});
  const SensorInstanceId gnss = SensorInstanceId::gnss(7U);
  EXPECT_TRUE(camera.valid());
  EXPECT_TRUE(lidar.valid());
  EXPECT_TRUE(gnss.valid());
  EXPECT_EQ(camera.modality, SensorModality::Visual);
  EXPECT_EQ(lidar.modality, SensorModality::Lidar);
  EXPECT_EQ(gnss.modality, SensorModality::Gnss);
  EXPECT_NE(camera, lidar);
  EXPECT_FALSE(SensorInstanceId::camera(CameraId{}).valid());
  EXPECT_FALSE(SensorInstanceId::lidar(LidarId{}).valid());
  EXPECT_FALSE(SensorInstanceId::gnss(SensorInstanceId::kInvalidInstance).valid());

  SensorInstanceId invalid = camera;
  invalid.modality = static_cast<SensorModality>(99);
  EXPECT_FALSE(invalid.valid());
}

TEST(SensorHealthSnapshot, RequiresTypedIdentityEpochStateAndSequence) {
  SensorHealthSnapshot health = makeMetadata().health;
  EXPECT_EQ(validateSensorHealthSnapshot(health), SensorHealthValidationError::None);

  health.sensor = SensorInstanceId{};
  EXPECT_EQ(validateSensorHealthSnapshot(health), SensorHealthValidationError::InvalidSensor);
  health = makeMetadata().health;
  health.state = static_cast<SensorHealthState>(99);
  EXPECT_EQ(validateSensorHealthSnapshot(health), SensorHealthValidationError::InvalidState);
  health = makeMetadata().health;
  health.recovery_epoch = SensorRecoveryEpoch{};
  EXPECT_EQ(validateSensorHealthSnapshot(health),
            SensorHealthValidationError::InvalidRecoveryEpoch);
  health = makeMetadata().health;
  health.transition_sequence = kInvalidSensorHealthTransitionSequence;
  EXPECT_EQ(validateSensorHealthSnapshot(health),
            SensorHealthValidationError::InvalidTransitionSequence);
}

TEST(FactorBatchTiming, RequiresCanonicalBoundedTimestampsAndOrderedProduction) {
  FactorBatchTiming timing = makeMetadata().timing;
  EXPECT_EQ(validateFactorBatchTiming(timing), FactorBatchTimingValidationError::None);

  timing.support.end = timing.support.start;
  EXPECT_EQ(validateFactorBatchTiming(timing), FactorBatchTimingValidationError::InvalidSupport);
  timing = makeMetadata().timing;
  timing.measurement_timestamps.clear();
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::InvalidMeasurementCount);
  timing = makeMetadata().timing;
  std::swap(timing.measurement_timestamps[0], timing.measurement_timestamps[1]);
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::NonCanonicalMeasurementTimestamps);
  timing = makeMetadata().timing;
  timing.measurement_timestamps[1] = timing.measurement_timestamps[0];
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::NonCanonicalMeasurementTimestamps);
  timing = makeMetadata().timing;
  timing.measurement_timestamps.front() = FusionTime{99};
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::MeasurementOutsideSupport);
  timing = makeMetadata().timing;
  timing.reference_time = FusionTime{201};
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::ReferenceOutsideSupport);
  timing = makeMetadata().timing;
  timing.produced_at = FusionTime{199};
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::ProducedBeforeSupportEnd);
  timing = makeMetadata().timing;
  timing.measurement_timestamps.assign(kMaximumFactorBatchMeasurementTimestamps + 1U,
                                       FusionTime{150});
  EXPECT_EQ(validateFactorBatchTiming(timing),
            FactorBatchTimingValidationError::InvalidMeasurementCount);
}

TEST(DirectionalObservability, AcceptsExplicitRightTranslationFirstGeneralVariableRecord) {
  const DirectionalObservability observability = makeObservability();
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::None);
  EXPECT_EQ(observability.supported_variables.size(), 5U);
  EXPECT_EQ(observability.endpoints.front().role, DirectionalEndpointRole::Target);
  EXPECT_EQ(observability.endpoints.back().role, DirectionalEndpointRole::Source);
}

TEST(DirectionalObservability, RejectsInvalidBasisSpectrumRankAndThresholds) {
  DirectionalObservability observability = makeObservability();
  observability.tangent = static_cast<PoseTangentConvention>(99);
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::UnsupportedTangent);
  observability = makeObservability();
  observability.basis(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::NonFiniteBasis);
  observability = makeObservability();
  observability.basis(0, 0) = 2.0;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::NonOrthonormalBasis);
  observability = makeObservability();
  observability.eigenvalues(0) = -0.1;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidEigenvalues);
  observability = makeObservability();
  std::swap(observability.eigenvalues(3), observability.eigenvalues(4));
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidEigenvalues);
  observability = makeObservability();
  observability.rank = 2U;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::RankMismatch);
  observability = makeObservability();
  observability.absolute_eigenvalue_threshold = 0.0;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidThresholds);
  observability = makeObservability();
  observability.relative_eigenvalue_threshold = 1.1;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidThresholds);
}

TEST(DirectionalObservability, RequiresCanonicalVariablesAndExplicitEndpointRoles) {
  DirectionalObservability observability = makeObservability();
  std::swap(observability.supported_variables[0], observability.supported_variables[1]);
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidSupportedVariables);
  observability = makeObservability();
  observability.supported_variables[1] = observability.supported_variables[0];
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidSupportedVariables);
  observability = makeObservability();
  observability.endpoints.front().role = DirectionalEndpointRole::Source;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidEndpoints);
  observability = makeObservability();
  observability.endpoints[1].state = observability.endpoints[0].state;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidEndpoints);
  observability = makeObservability();
  observability.endpoints = {
      {DirectionalEndpointRole::Unary, StateId{20U}, FusionTime{190}},
  };
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::None);
  observability.endpoints.front().role = DirectionalEndpointRole::Target;
  EXPECT_EQ(validateDirectionalObservability(observability),
            DirectionalObservabilityValidationError::InvalidEndpoints);
}

TEST(FactorBatchMetadata, AcceptsCompleteRosFreeMetadataWithoutCommonCovariance) {
  const FactorBatchMetadata metadata = makeMetadata();
  EXPECT_EQ(validateFactorBatchMetadata(metadata), FactorBatchMetadataValidationError::None);
}

TEST(FactorBatchMetadata, RejectsInvalidHeaderIdentityTimingAndHealthAssociation) {
  FactorBatchMetadata metadata = makeMetadata();
  metadata.header.trace = TraceId{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidHeader);
  metadata = makeMetadata();
  metadata.header.created_at = FusionTime{219};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidHeader);
  metadata = makeMetadata();
  metadata.batch_id = FactorBatchId{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidBatchId);
  metadata = makeMetadata();
  metadata.odom_epoch = OdomEpoch{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidOdomEpoch);
  metadata = makeMetadata();
  metadata.sensor = SensorInstanceId{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidSensor);
  metadata = makeMetadata();
  metadata.timing.measurement_timestamps.clear();
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidTiming);
  metadata = makeMetadata();
  metadata.health.recovery_epoch = SensorRecoveryEpoch{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidHealth);
  metadata = makeMetadata();
  metadata.health.sensor = SensorInstanceId::camera(CameraId{7U});
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::SensorHealthMismatch);
  metadata = makeMetadata();
  metadata.health.assessed_at = FusionTime{199};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidHealthAssessmentTime);
  metadata = makeMetadata();
  metadata.health.assessed_at = FusionTime{221};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidHealthAssessmentTime);
}

TEST(FactorBatchMetadata, EnforcesObservabilityMapEligibilityAndLineage) {
  FactorBatchMetadata metadata = makeMetadata();
  metadata.directional_observability.clear();
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidObservabilityCount);
  metadata = makeMetadata();
  metadata.directional_observability.front().rank = 0U;
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidDirectionalObservability);
  metadata = makeMetadata();
  metadata.health.state = SensorHealthState::Suspect;
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::MapEligibilityRequiresActiveHealth);
  metadata.map_eligible = false;
  EXPECT_EQ(validateFactorBatchMetadata(metadata), FactorBatchMetadataValidationError::None);
  metadata = makeMetadata();
  metadata.lineage.id = ObservationLineageId{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidLineage);
  metadata = makeMetadata();
  metadata.lineage.checksum = ContentHash{};
  EXPECT_EQ(validateFactorBatchMetadata(metadata),
            FactorBatchMetadataValidationError::InvalidLineage);
}

}  // namespace
}  // namespace meridian::core
