#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sophus/so3.hpp>
#include <utility>

#include "meridian/core/canonical_verification.hpp"

namespace meridian::core {
namespace {

ContentHash hashValue(std::uint8_t seed) {
  ContentHash hash{};
  for (std::size_t index = 0U; index < hash.size(); ++index) {
    hash[index] = static_cast<std::uint8_t>(seed + index);
  }
  return hash;
}

ContentHash takeHash(CanonicalChecksumResult result) {
  if (!result) {
    ADD_FAILURE() << "checksum construction failed: " << static_cast<int>(result.error().code);
    return hashValue(0xe0U);
  }
  return std::move(result).value();
}

RecordHeader header(SessionId session, CalibrationEpoch calibration, std::uint64_t trace) {
  RecordHeader value;
  value.schema_version = 1U;
  value.trace = TraceId{trace};
  value.producer = ProducerId{2U};
  value.session = session;
  value.created_at = FusionTime{static_cast<std::int64_t>(1000U + trace)};
  value.config = ConfigRevision{3U};
  value.direct_calibration = calibration;
  return value;
}

ObservationLineage lineage(std::uint64_t id) {
  ObservationLineage value;
  value.id = ObservationLineageId{id};
  ObservationSlice slice;
  slice.root = MeasurementId{100U + id};
  slice.source_checksum = hashValue(static_cast<std::uint8_t>(10U + id));
  slice.calibration = CalibrationEpoch{5U};
  value.usage.push_back(ObservationUsage{slice, ObservationRole::PrimaryResidual,
                                         DerivedRecordId{200U + id}, FactorGroupId{300U + id},
                                         std::nullopt});
  value.checksum = takeHash(recomputeObservationLineageChecksum(value));
  return value;
}

BlobRef blob(std::uint64_t id) {
  BlobRef value;
  value.store = BlobStoreId{1U};
  value.id = BlobId{id};
  value.checksum = hashValue(static_cast<std::uint8_t>(30U + id));
  value.layout = LayoutId{40U + id};
  value.bytes = 100U + id;
  value.storage = BlobStorage::DurableSpool;
  return value;
}

SparsePayloadCatalog catalog(std::uint64_t base) {
  SparsePayloadCatalog value;
  value.entries = {
      {SparsePayloadKind::InternalTrajectory, blob(base)},
      {SparsePayloadKind::KeyframeIndex, blob(base + 1U)},
      {SparsePayloadKind::RegistrationProxy, blob(base + 2U)},
      {SparsePayloadKind::DenseInputIndex, blob(base + 3U)},
  };
  value.checksum = takeHash(recomputeSparsePayloadCatalogChecksum(value));
  return value;
}

BoundaryNavigationLinearization boundary(StateId state, FusionTime time,
                                         LocalGraphRevision revision, double x) {
  BoundaryNavigationLinearization value;
  value.state = state;
  value.exact_time = time;
  value.final_revision = revision;
  value.T_odom_imu = Pose3d{Sophus::SO3d{}, Vector3d{x, -0.5 * x, 0.25 * x}};
  value.velocity_odom = Vector3d{x + 1.0, x + 2.0, x + 3.0};
  value.gyro_bias = Vector3d{0.01 * x, 0.02 * x, 0.03 * x};
  value.accel_bias = Vector3d{-0.01 * x, -0.02 * x, -0.03 * x};
  return value;
}

LocalVariableRef stateVariable(LocalVariableKind kind, StateId state) {
  LocalVariableRef value;
  value.kind = kind;
  value.state = state;
  return value;
}

FrozenSquareRootFactor factor(StateId from, StateId to) {
  FrozenSquareRootFactor value;
  value.rows = 2U;
  value.columns = 30U;
  value.layout = {
      {stateVariable(LocalVariableKind::Pose, from), 0U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, from), 6U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, from), 9U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, from), 12U, 3U},
      {stateVariable(LocalVariableKind::Pose, to), 15U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, to), 21U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, to), 24U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, to), 27U, 3U},
  };
  value.row_major_A.assign(60U, 0.0);
  value.row_major_A[0U] = 2.0;
  value.row_major_A[31U] = 3.0;
  value.rhs = {0.25, -0.5};
  value.constant_squared_error = 0.125;
  value.numerical_rank = 2U;
  value.absolute_rank_tolerance = 1.0e-12;
  value.relative_rank_tolerance = 1.0e-9;
  value.cost_statistics.source_residual_dof = 4U;
  value.cost_statistics.eliminated_numerical_rank = 1U;
  value.cost_statistics.effective_dof = 3U;
  value.cost_statistics.calibration_revision = ResidualCalibrationRevision{7U};
  value.cost_statistics.calibrated_total_cost_cutoff = 10.0;
  value.checksum = takeHash(recomputeFrozenSquareRootFactorChecksum(value));
  return value;
}

SparseSubmapSeal firstSeal(std::uint64_t submap, StateId state, std::int64_t start,
                           std::uint64_t trace) {
  SparseSubmapSeal seal;
  const SessionId session{1U};
  const CalibrationEpoch calibration{5U};
  seal.header = header(session, calibration, trace);
  seal.ref = SubmapRef{
      session, OdomEpoch{6U}, SubmapId{submap}, calibration, SubmapContentRevision{1U}, {}};
  seal.final_local_revision = LocalGraphRevision{20U + submap};
  seal.support_time = TimeRange{FusionTime{start}, FusionTime{start + 100}};
  seal.frame.boundary_state = state;
  seal.frame.boundary_time = seal.support_time.start;
  seal.frame.gravity_up_odom = Vector3d::UnitZ();
  seal.frame.boundary_yaw_odom = 0.1;
  seal.T_odom_submap = Pose3d{Sophus::SO3d{}, Vector3d{1.0, 2.0, 3.0}};
  seal.boundary_navigation =
      boundary(state, seal.support_time.start, LocalGraphRevision{10U + submap},
               static_cast<double>(submap));
  seal.payloads = catalog(10U * submap);
  seal.lineage = lineage(submap);
  seal.quality_checksum = hashValue(static_cast<std::uint8_t>(70U + submap));
  seal.ref.local_content_checksum = takeHash(recomputeSubmapLocalContentChecksum(seal));
  seal.seal_checksum = takeHash(recomputeSparseSubmapSealChecksum(seal));
  return seal;
}

struct SealFixture {
  SparseSubmapSeal first;
  SparseSubmapSeal successor;
};

SealFixture seals() {
  SealFixture fixture;
  fixture.first = firstSeal(1U, StateId{10U}, 0, 10U);
  fixture.successor = firstSeal(2U, StateId{20U}, 100, 11U);
  fixture.successor.previous = fixture.first.ref;

  CondensedBoundaryTransition local;
  local.header = header(fixture.first.ref.session, fixture.first.ref.calibration, 12U);
  local.odom_epoch = fixture.first.ref.odom_epoch;
  local.from = fixture.first.boundary_navigation;
  local.to = fixture.successor.boundary_navigation;
  local.boundary_factor = factor(local.from.state, local.to.state);
  local.source_factors = {
      {local.odom_epoch, FactorId{1U}},
      {local.odom_epoch, FactorId{2U}},
  };
  local.lineage = lineage(3U);
  local.final_revision = fixture.successor.final_local_revision;
  local.input_partition_checksum = hashValue(80U);
  local.checksum = takeHash(recomputeCondensedBoundaryTransitionChecksum(local));

  SealedBoundaryTransition adjacent{fixture.first.ref, fixture.successor.ref, local, {}};
  adjacent.checksum = takeHash(recomputeSealedBoundaryTransitionChecksum(adjacent));
  fixture.successor.from_previous = adjacent;
  fixture.successor.seal_checksum = takeHash(recomputeSparseSubmapSealChecksum(fixture.successor));
  return fixture;
}

void expectFailure(const SparseSubmapSeal& seal, CanonicalRecordKind domain,
                   CanonicalVerificationFailure failure) {
  const auto result = verifyCanonicalSparseSubmapSeal(seal);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().domain, domain);
  EXPECT_EQ(result.error().failure, failure);
}

TEST(CanonicalVerification, VerifiesFirstSealWithoutTransition) {
  const SparseSubmapSeal seal = seals().first;
  const auto result = verifyCanonicalSparseSubmapSeal(seal);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().root_domain, CanonicalRecordKind::SparseSubmapSeal);
  EXPECT_EQ(result.value().verified_records, 4U);
  EXPECT_EQ(result.value().root_digest, seal.seal_checksum);
}

TEST(CanonicalVerification, VerifiesSuccessorAndEveryLeaf) {
  const SparseSubmapSeal seal = seals().successor;
  ASSERT_TRUE(seal.from_previous);
  EXPECT_TRUE(verifyCanonicalObservationLineageChecksum(seal.lineage));
  EXPECT_TRUE(
      verifyCanonicalObservationLineageChecksum(seal.from_previous->local_transition.lineage));
  EXPECT_TRUE(verifyCanonicalFrozenSquareRootFactorChecksum(
      seal.from_previous->local_transition.boundary_factor));
  EXPECT_TRUE(
      verifyCanonicalCondensedBoundaryTransitionChecksum(seal.from_previous->local_transition));
  EXPECT_TRUE(verifyCanonicalSparsePayloadCatalogChecksum(seal.payloads));
  EXPECT_TRUE(verifyCanonicalSubmapLocalContentChecksum(seal));
  EXPECT_TRUE(verifyCanonicalSealedBoundaryTransitionChecksum(*seal.from_previous));
  EXPECT_TRUE(verifyCanonicalSparseSubmapSealChecksum(seal));

  const auto recursive = verifyCanonicalSparseSubmapSeal(seal);
  ASSERT_TRUE(recursive);
  EXPECT_EQ(recursive.value().verified_records, 8U);
  EXPECT_EQ(recursive.value().root_digest, seal.seal_checksum);
}

TEST(CanonicalVerification, DetectsOneFieldTamperingAtItsFirstOwningDomain) {
  const SparseSubmapSeal original = seals().successor;
  SparseSubmapSeal tampered = original;
  tampered.lineage.usage.front().consumer = DerivedRecordId{999U};
  expectFailure(tampered, CanonicalRecordKind::ObservationLineage,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  tampered.from_previous->local_transition.boundary_factor.rhs.front() += 0.1;
  expectFailure(tampered, CanonicalRecordKind::FrozenSquareRootFactor,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  tampered.from_previous->local_transition.source_factors.back().factor = FactorId{3U};
  expectFailure(tampered, CanonicalRecordKind::CondensedBoundaryTransition,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  ++tampered.payloads.entries.front().root.bytes;
  expectFailure(tampered, CanonicalRecordKind::SparsePayloadCatalog,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  ++tampered.support_time.end.nanoseconds;
  expectFailure(tampered, CanonicalRecordKind::SubmapLocalContent,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  tampered.from_previous->from.local_content_checksum = hashValue(210U);
  expectFailure(tampered, CanonicalRecordKind::SealedBoundaryTransition,
                CanonicalVerificationFailure::DigestMismatch);

  tampered = original;
  tampered.header.trace = TraceId{999U};
  expectFailure(tampered, CanonicalRecordKind::SparseSubmapSeal,
                CanonicalVerificationFailure::DigestMismatch);
}

TEST(CanonicalVerification, ChildMismatchCannotBeHiddenByMatchingParentHashes) {
  SparseSubmapSeal tampered = seals().successor;
  FrozenSquareRootFactor& factor = tampered.from_previous->local_transition.boundary_factor;
  factor.checksum = hashValue(220U);
  CondensedBoundaryTransition& local = tampered.from_previous->local_transition;
  local.checksum = takeHash(recomputeCondensedBoundaryTransitionChecksum(local));
  tampered.from_previous->checksum =
      takeHash(recomputeSealedBoundaryTransitionChecksum(*tampered.from_previous));
  tampered.seal_checksum = takeHash(recomputeSparseSubmapSealChecksum(tampered));

  EXPECT_TRUE(verifyCanonicalCondensedBoundaryTransitionChecksum(local));
  EXPECT_TRUE(verifyCanonicalSealedBoundaryTransitionChecksum(*tampered.from_previous));
  EXPECT_TRUE(verifyCanonicalSparseSubmapSealChecksum(tampered));
  expectFailure(tampered, CanonicalRecordKind::FrozenSquareRootFactor,
                CanonicalVerificationFailure::DigestMismatch);
}

TEST(CanonicalVerification, ReportsSemanticAndBoundedRecomputationFailures) {
  SparseSubmapSeal malformed = seals().successor;
  malformed.from_previous->local_transition.boundary_factor.cost_statistics
      .calibrated_total_cost_cutoff = 0.0;
  const auto semantic = verifyCanonicalSparseSubmapSeal(malformed);
  ASSERT_FALSE(semantic);
  EXPECT_EQ(semantic.error().domain, CanonicalRecordKind::FrozenSquareRootFactor);
  EXPECT_EQ(semantic.error().failure, CanonicalVerificationFailure::SemanticValidation);
  ASSERT_TRUE(std::holds_alternative<FrozenFactorValidationError>(semantic.error().semantic_error));
  EXPECT_EQ(std::get<FrozenFactorValidationError>(semantic.error().semantic_error),
            FrozenFactorValidationError::InvalidCostStatistics);

  const SparseSubmapSeal valid = seals().successor;
  const auto byte_limit = verifyCanonicalSparseSubmapSeal(valid, CanonicalRecordLimits{64U, 1000U});
  ASSERT_FALSE(byte_limit);
  EXPECT_EQ(byte_limit.error().domain, CanonicalRecordKind::ObservationLineage);
  EXPECT_EQ(byte_limit.error().failure, CanonicalVerificationFailure::Recomputation);
  ASSERT_TRUE(byte_limit.error().recomputation_error);
  EXPECT_EQ(byte_limit.error().recomputation_error->code,
            CanonicalRecordErrorCode::EncodingFailure);
  EXPECT_EQ(byte_limit.error().recomputation_error->encoding_error,
            CanonicalEncodingError::OutputLimitExceeded);

  const auto collection_limit =
      verifyCanonicalSparseSubmapSeal(valid, CanonicalRecordLimits{4096U, 1U});
  ASSERT_FALSE(collection_limit);
  EXPECT_EQ(collection_limit.error().domain, CanonicalRecordKind::FrozenSquareRootFactor);
  EXPECT_EQ(collection_limit.error().failure, CanonicalVerificationFailure::Recomputation);
  ASSERT_TRUE(collection_limit.error().recomputation_error);
  EXPECT_EQ(collection_limit.error().recomputation_error->code,
            CanonicalRecordErrorCode::CollectionLimitExceeded);
}

TEST(CanonicalVerification, OpaqueChildrenArePresenceCheckedNotRecomputed) {
  SparseSubmapSeal seal = seals().first;
  seal.lineage.usage.front().slice.source_checksum = hashValue(220U);
  seal.lineage.checksum = takeHash(recomputeObservationLineageChecksum(seal.lineage));
  seal.payloads.entries.front().root.checksum = hashValue(230U);
  seal.payloads.checksum = takeHash(recomputeSparsePayloadCatalogChecksum(seal.payloads));
  seal.quality_checksum = hashValue(240U);
  seal.ref.local_content_checksum = takeHash(recomputeSubmapLocalContentChecksum(seal));
  seal.seal_checksum = takeHash(recomputeSparseSubmapSealChecksum(seal));
  EXPECT_TRUE(verifyCanonicalSparseSubmapSeal(seal));

  seal.lineage.usage.front().slice.source_checksum = {};
  const auto missing = verifyCanonicalSparseSubmapSeal(seal);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().domain, CanonicalRecordKind::ObservationLineage);
  EXPECT_EQ(missing.error().failure, CanonicalVerificationFailure::SemanticValidation);
  ASSERT_TRUE(std::holds_alternative<CanonicalVerificationPreconditionError>(
      missing.error().semantic_error));
  EXPECT_EQ(std::get<CanonicalVerificationPreconditionError>(missing.error().semantic_error),
            CanonicalVerificationPreconditionError::MissingOpaqueChildChecksum);
}

}  // namespace
}  // namespace meridian::core
