#include <gtest/gtest.h>

#include <cstdint>
#include <optional>

#include "meridian/local/map_admission_gate.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::ContentHash presentHash(std::uint8_t value) {
  core::ContentHash hash{};
  hash.front() = value;
  return hash;
}

[[nodiscard]] core::ObservationLineage lineage() {
  core::ObservationLineage result;
  result.id = core::ObservationLineageId{30U};
  result.checksum = presentHash(1U);
  core::ObservationSlice slice;
  slice.root = core::MeasurementId{31U};
  slice.calibration = core::CalibrationEpoch{5U};
  result.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::PrimaryResidual,
                                                core::DerivedRecordId{32U},
                                                core::FactorGroupId{33U}, std::nullopt});
  return result;
}

[[nodiscard]] core::DirectionalObservability observability() {
  core::DirectionalObservability result;
  result.eigenvalues << 0.01, 0.1, 0.2, 1.0, 2.0, 3.0;
  result.rank = 3U;
  result.absolute_eigenvalue_threshold = 0.5;
  result.relative_eigenvalue_threshold = 0.1;
  result.supported_variables = {
      core::DirectionalVariable::PoseTranslation, core::DirectionalVariable::PoseRotation,
      core::DirectionalVariable::Velocity,        core::DirectionalVariable::AccelerometerBias,
      core::DirectionalVariable::GyroscopeBias,
  };
  result.endpoints = {
      {core::DirectionalEndpointRole::Target, core::StateId{10U}, core::FusionTime{110LL}},
      {core::DirectionalEndpointRole::Source, core::StateId{20U}, core::FusionTime{190LL}},
  };
  return result;
}

[[nodiscard]] core::FactorBatchMetadata metadata(std::uint64_t batch = 6U) {
  core::FactorBatchMetadata result;
  result.header.schema_version = 1U;
  result.header.trace = core::TraceId{1U};
  result.header.producer = core::ProducerId{2U};
  result.header.session = core::SessionId{3U};
  result.header.created_at = core::FusionTime{230LL};
  result.header.config = core::ConfigRevision{4U};
  result.header.direct_calibration = core::CalibrationEpoch{5U};
  result.batch_id = core::FactorBatchId{batch};
  result.odom_epoch = core::OdomEpoch{7U};
  result.sensor = core::SensorInstanceId::lidar(core::LidarId{8U});
  result.timing.support = core::TimeRange{core::FusionTime{100LL}, core::FusionTime{200LL}};
  result.timing.measurement_timestamps = {core::FusionTime{110LL}, core::FusionTime{150LL},
                                          core::FusionTime{190LL}};
  result.timing.reference_time = core::FusionTime{190LL};
  result.timing.produced_at = core::FusionTime{220LL};
  result.health.sensor = result.sensor;
  result.health.state = core::SensorHealthState::Active;
  result.health.recovery_epoch = core::SensorRecoveryEpoch{0U};
  result.health.transition_sequence = 0U;
  result.health.assessed_at = core::FusionTime{210LL};
  result.map_eligible = true;
  result.directional_observability.push_back(observability());
  result.lineage = lineage();
  return result;
}

[[nodiscard]] MapAdmissionContext context(const core::FactorBatchMetadata& batch) {
  MapAdmissionContext result;
  result.accepted_batch_revision = core::LocalGraphRevision{9U};
  result.health_before = batch.health;
  result.health_after = batch.health;
  result.localization_revision = core::LocalGraphRevision{9U};
  return result;
}

void expectDeniedWithoutMutation(MapAdmissionGate* gate, const core::FactorBatchMetadata& batch,
                                 const MapAdmissionContext& admission_context,
                                 MapAdmissionDisposition disposition, MapAdmissionReason reason) {
  const std::size_t count_before = gate->admittedBatchCount();
  const MapAdmissionGateStatistics statistics_before = gate->statistics();
  const MapAdmissionDecision result = gate->admit(batch, admission_context);
  EXPECT_EQ(result.disposition, disposition);
  EXPECT_EQ(result.reason, reason);
  EXPECT_FALSE(result.admitted());
  EXPECT_EQ(gate->admittedBatchCount(), count_before);
  EXPECT_EQ(gate->statistics(), statistics_before);
  EXPECT_FALSE(
      gate->contains(MapAdmissionBatchRef{batch.odom_epoch, batch.sensor, batch.batch_id}));
}

TEST(MapAdmissionGate, EveryFailedPreconditionHasAnExplicitNonMutatingDisposition) {
  MapAdmissionGate gate;

  core::FactorBatchMetadata batch = metadata(10U);
  MapAdmissionContext admission_context = context(batch);
  admission_context.accepted_batch_revision.reset();
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::BatchRejected);

  batch = metadata(11U);
  admission_context = context(batch);
  batch.batch_id = core::FactorBatchId{};
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::InvalidMetadata);

  batch = metadata(12U);
  admission_context = context(batch);
  admission_context.health_after.recovery_epoch = core::SensorRecoveryEpoch{};
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::InvalidHealthSnapshot);

  batch = metadata(13U);
  admission_context = context(batch);
  admission_context.localization_revision = core::LocalGraphRevision{};
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::InvalidLocalizationRevision);

  batch = metadata(14U);
  admission_context = context(batch);
  admission_context.health_after.sensor = core::SensorInstanceId::camera(core::CameraId{8U});
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::SensorMismatch);

  batch = metadata(141U);
  admission_context = context(batch);
  admission_context.accepted_batch_revision = core::LocalGraphRevision{10U};
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::AcceptedRevisionMismatch);

  batch = metadata(15U);
  admission_context = context(batch);
  admission_context.health_after.recovery_epoch = core::SensorRecoveryEpoch{1U};
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::RecoveryEpochMismatch);

  batch = metadata(16U);
  batch.map_eligible = false;
  admission_context = context(batch);
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Frozen,
                              MapAdmissionReason::MapIneligible);

  batch = metadata(17U);
  admission_context = context(batch);
  admission_context.health_after.state = core::SensorHealthState::Suspect;
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Quarantined,
                              MapAdmissionReason::HealthNotActive);

  batch = metadata(18U);
  admission_context = context(batch);
  ++admission_context.health_before.transition_sequence;
  expectDeniedWithoutMutation(&gate, batch, admission_context, MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::HealthSnapshotMismatch);
}

TEST(MapAdmissionGate, InvalidZeroHistoryCapacityRejectsWithoutMutation) {
  MapAdmissionGate gate{MapAdmissionGateConfig{0U}};
  const core::FactorBatchMetadata batch = metadata(18U);
  expectDeniedWithoutMutation(&gate, batch, context(batch), MapAdmissionDisposition::Rejected,
                              MapAdmissionReason::InvalidGateConfig);
  EXPECT_EQ(gate.historyCapacity(), 0U);
}

TEST(MapAdmissionGate, PureEvaluationDoesNotReserveAnOtherwiseEligibleBatch) {
  MapAdmissionGate gate;
  const core::FactorBatchMetadata batch = metadata(20U);
  const MapAdmissionContext admission_context = context(batch);
  const MapAdmissionDecision first = gate.evaluate(batch, admission_context);
  const MapAdmissionDecision second = gate.evaluate(batch, admission_context);
  EXPECT_EQ(first.disposition, MapAdmissionDisposition::Admitted);
  EXPECT_EQ(second.disposition, MapAdmissionDisposition::Admitted);
  EXPECT_EQ(gate.admittedBatchCount(), 0U);
  EXPECT_FALSE(gate.contains(MapAdmissionBatchRef{batch.odom_epoch, batch.sensor, batch.batch_id}));
}

TEST(MapAdmissionGate, AdmitsRegularAndInitializationBatchesExactlyOnce) {
  MapAdmissionGate gate;
  const core::FactorBatchMetadata regular = metadata(30U);
  const MapAdmissionDecision admitted = gate.admit(regular, context(regular));
  EXPECT_TRUE(admitted.admitted());
  EXPECT_EQ(admitted.disposition, MapAdmissionDisposition::Admitted);
  EXPECT_EQ(admitted.reason, MapAdmissionReason::Eligible);
  EXPECT_EQ(gate.admittedBatchCount(), 1U);
  EXPECT_TRUE(
      gate.contains(MapAdmissionBatchRef{regular.odom_epoch, regular.sensor, regular.batch_id}));

  const MapAdmissionDecision duplicate = gate.admit(regular, context(regular));
  EXPECT_FALSE(duplicate.admitted());
  EXPECT_EQ(duplicate.disposition, MapAdmissionDisposition::Rejected);
  EXPECT_EQ(duplicate.reason, MapAdmissionReason::DuplicateBatch);
  EXPECT_EQ(gate.admittedBatchCount(), 1U);

  const core::FactorBatchMetadata initialization = metadata(31U);
  MapAdmissionContext initialization_context = context(initialization);
  initialization_context.kind = MapAdmissionBatchKind::InitializationSeed;
  const MapAdmissionDecision initialized = gate.admit(initialization, initialization_context);
  EXPECT_TRUE(initialized.admitted());
  EXPECT_EQ(initialized.disposition, MapAdmissionDisposition::AcceptedInitializationSeed);
  EXPECT_EQ(initialized.reason, MapAdmissionReason::InitializationSeed);
  EXPECT_EQ(gate.admittedBatchCount(), 2U);
  EXPECT_TRUE(gate.contains(MapAdmissionBatchRef{initialization.odom_epoch, initialization.sensor,
                                                 initialization.batch_id}));
}

TEST(MapAdmissionGate, BatchIdentityIsScopedByEpochAndSensor) {
  MapAdmissionGate gate;
  const core::FactorBatchMetadata first = metadata(35U);
  core::FactorBatchMetadata next_epoch = metadata(35U);
  next_epoch.odom_epoch = core::OdomEpoch{8U};

  ASSERT_TRUE(gate.admit(first, context(first)).admitted());
  ASSERT_TRUE(gate.admit(next_epoch, context(next_epoch)).admitted());
  EXPECT_EQ(gate.admittedBatchCount(), 2U);
  EXPECT_TRUE(gate.contains(MapAdmissionBatchRef{first.odom_epoch, first.sensor, first.batch_id}));
  EXPECT_TRUE(gate.contains(
      MapAdmissionBatchRef{next_epoch.odom_epoch, next_epoch.sensor, next_epoch.batch_id}));
}

TEST(MapAdmissionGate, BoundsDuplicateHistoryWithDeterministicFifoEvictionAndStatistics) {
  MapAdmissionGate gate{MapAdmissionGateConfig{2U}};
  const core::FactorBatchMetadata first = metadata(40U);
  const core::FactorBatchMetadata second = metadata(41U);
  const core::FactorBatchMetadata third = metadata(42U);
  ASSERT_TRUE(gate.admit(first, context(first)).admitted());
  ASSERT_TRUE(gate.admit(second, context(second)).admitted());
  EXPECT_EQ(gate.statistics(), (MapAdmissionGateStatistics{2U, 2U, 0U, 2U}));

  ASSERT_TRUE(gate.admit(third, context(third)).admitted());
  const auto ref = [](const core::FactorBatchMetadata& batch) {
    return MapAdmissionBatchRef{batch.odom_epoch, batch.sensor, batch.batch_id};
  };
  EXPECT_FALSE(gate.contains(ref(first)));
  EXPECT_TRUE(gate.contains(ref(second)));
  EXPECT_TRUE(gate.contains(ref(third)));
  EXPECT_EQ(gate.statistics(), (MapAdmissionGateStatistics{2U, 3U, 1U, 2U}));

  const MapAdmissionGateStatistics before_duplicate = gate.statistics();
  const MapAdmissionDecision duplicate = gate.admit(third, context(third));
  EXPECT_FALSE(duplicate.admitted());
  EXPECT_EQ(duplicate.reason, MapAdmissionReason::DuplicateBatch);
  EXPECT_EQ(gate.statistics(), before_duplicate);

  // Once an identity leaves the explicitly bounded recent history, the gate
  // can admit it again and deterministically evicts the next FIFO member.
  ASSERT_TRUE(gate.admit(first, context(first)).admitted());
  EXPECT_TRUE(gate.contains(ref(first)));
  EXPECT_FALSE(gate.contains(ref(second)));
  EXPECT_TRUE(gate.contains(ref(third)));
  EXPECT_EQ(gate.statistics(), (MapAdmissionGateStatistics{2U, 4U, 2U, 2U}));
}

}  // namespace
}  // namespace meridian::local
