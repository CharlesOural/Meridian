#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "meridian/core/sparse_map_api.hpp"

namespace meridian::core {
namespace {

ContentHash presentHash(std::uint8_t value) {
  ContentHash hash{};
  hash.front() = value;
  return hash;
}

ObservationLineage makeLineage(std::uint64_t id) {
  ObservationLineage lineage;
  lineage.id = ObservationLineageId{id};
  lineage.checksum = presentHash(static_cast<std::uint8_t>(40U + id));
  ObservationSlice slice;
  slice.root = MeasurementId{100U + id};
  slice.calibration = CalibrationEpoch{3U};
  lineage.usage.push_back(ObservationUsage{slice, ObservationRole::PrimaryResidual,
                                           DerivedRecordId{200U + id}, FactorGroupId{300U + id},
                                           std::nullopt});
  return lineage;
}

LocalVariableRef stateVariable(LocalVariableKind kind, StateId state) {
  LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

FrozenSquareRootFactor makeBoundaryFactor(StateId from, StateId to) {
  FrozenSquareRootFactor factor;
  factor.rows = 2U;
  factor.columns = 30U;
  factor.layout = {
      {stateVariable(LocalVariableKind::Pose, from), 0U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, from), 6U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, from), 9U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, from), 12U, 3U},
      {stateVariable(LocalVariableKind::Pose, to), 15U, 6U},
      {stateVariable(LocalVariableKind::NavigationVelocity, to), 21U, 3U},
      {stateVariable(LocalVariableKind::GyroBias, to), 24U, 3U},
      {stateVariable(LocalVariableKind::AccelBias, to), 27U, 3U},
  };
  factor.row_major_A.assign(60U, 0.0);
  factor.row_major_A[0U] = 1.0;
  factor.row_major_A[31U] = 2.0;
  factor.rhs = {0.25, -0.5};
  factor.constant_squared_error = 0.125;
  factor.numerical_rank = 2U;
  factor.absolute_rank_tolerance = 1.0e-12;
  factor.relative_rank_tolerance = 1.0e-9;
  factor.cost_statistics.source_residual_dof = 5U;
  factor.cost_statistics.eliminated_numerical_rank = 1U;
  factor.cost_statistics.effective_dof = 4U;
  factor.cost_statistics.calibration_revision = ResidualCalibrationRevision{1U};
  factor.cost_statistics.calibrated_total_cost_cutoff = 12.0;
  factor.checksum = presentHash(70U);
  return factor;
}

BlobRef durableBlob(std::uint64_t id, SparsePayloadKind kind) {
  BlobRef ref;
  ref.store = BlobStoreId{5U};
  ref.id = BlobId{id};
  ref.checksum = presentHash(static_cast<std::uint8_t>(80U + id));
  ref.layout = LayoutId{static_cast<std::uint64_t>(kind) + 1U};
  ref.bytes = 64U + id;
  ref.storage = BlobStorage::DurableSpool;
  return ref;
}

SparsePayloadCatalog makePayloads(std::uint64_t base) {
  SparsePayloadCatalog catalog;
  catalog.entries = {
      {SparsePayloadKind::InternalTrajectory,
       durableBlob(base + 0U, SparsePayloadKind::InternalTrajectory)},
      {SparsePayloadKind::KeyframeIndex, durableBlob(base + 1U, SparsePayloadKind::KeyframeIndex)},
      {SparsePayloadKind::RegistrationProxy,
       durableBlob(base + 2U, SparsePayloadKind::RegistrationProxy)},
      {SparsePayloadKind::DenseInputIndex,
       durableBlob(base + 3U, SparsePayloadKind::DenseInputIndex)},
  };
  catalog.checksum = presentHash(static_cast<std::uint8_t>(100U + base));
  return catalog;
}

SparseSubmapSeal makeSeal(std::uint64_t submap, std::int64_t start) {
  SparseSubmapSeal seal;
  seal.ref.session = SessionId{1U};
  seal.ref.odom_epoch = OdomEpoch{2U};
  seal.ref.id = SubmapId{submap};
  seal.ref.calibration = CalibrationEpoch{3U};
  seal.ref.content_revision = SubmapContentRevision{1U};
  seal.ref.local_content_checksum = presentHash(static_cast<std::uint8_t>(10U + submap));
  seal.header.schema_version = 1U;
  seal.header.trace = TraceId{10U + submap};
  seal.header.producer = ProducerId{4U};
  seal.header.session = seal.ref.session;
  seal.header.config = ConfigRevision{5U};
  seal.header.direct_calibration = seal.ref.calibration;
  seal.final_local_revision = LocalGraphRevision{20U + submap};
  seal.support_time = TimeRange{FusionTime{start}, FusionTime{start + 100}};
  seal.frame.boundary_state = StateId{100U + submap};
  seal.frame.boundary_time = seal.support_time.start;
  seal.boundary_navigation.state = seal.frame.boundary_state;
  seal.boundary_navigation.exact_time = seal.frame.boundary_time;
  seal.boundary_navigation.final_revision = LocalGraphRevision{10U + submap};
  seal.payloads = makePayloads(10U * submap);
  seal.lineage = makeLineage(submap);
  seal.quality_checksum = presentHash(static_cast<std::uint8_t>(120U + submap));
  seal.seal_checksum = presentHash(static_cast<std::uint8_t>(140U + submap));
  return seal;
}

SealedBoundaryTransition makeTransition(const SparseSubmapSeal& from, const SparseSubmapSeal& to) {
  CondensedBoundaryTransition local;
  local.header.schema_version = 1U;
  local.header.trace = TraceId{50U};
  local.header.producer = ProducerId{4U};
  local.header.session = from.ref.session;
  local.header.config = ConfigRevision{5U};
  local.header.direct_calibration = CalibrationEpoch{3U};
  local.odom_epoch = from.ref.odom_epoch;
  local.from = from.boundary_navigation;
  local.to = to.boundary_navigation;
  local.boundary_factor = makeBoundaryFactor(local.from.state, local.to.state);
  local.source_factors = {{local.odom_epoch, FactorId{1U}}, {local.odom_epoch, FactorId{2U}}};
  local.lineage = makeLineage(30U);
  local.final_revision = LocalGraphRevision{to.final_local_revision.value() - 1U};
  local.input_partition_checksum = presentHash(71U);
  local.checksum = presentHash(72U);
  return SealedBoundaryTransition{from.ref, to.ref, std::move(local), presentHash(73U)};
}

SparseSubmapSeal linkedSuccessor(const SparseSubmapSeal& previous) {
  SparseSubmapSeal current =
      makeSeal(previous.ref.id.value() + 1U, previous.support_time.end.nanoseconds);
  current.previous = previous.ref;
  current.from_previous = makeTransition(previous, current);
  return current;
}

TEST(SubmapRefApi, RequiresCompleteContentAddressedIdentity) {
  const SparseSubmapSeal seal = makeSeal(1U, 0);
  EXPECT_EQ(validateSubmapRef(seal.ref), SubmapRefValidationError::None);

  SubmapRef malformed = seal.ref;
  malformed.calibration = CalibrationEpoch{};
  EXPECT_EQ(validateSubmapRef(malformed), SubmapRefValidationError::InvalidIdentity);
  malformed = seal.ref;
  malformed.local_content_checksum = {};
  EXPECT_EQ(validateSubmapRef(malformed), SubmapRefValidationError::MissingLocalContentChecksum);
}

TEST(SparsePayloadCatalogApi, RequiresCanonicalDurableRootSet) {
  SparsePayloadCatalog catalog = makePayloads(10U);
  EXPECT_EQ(validateSparsePayloadCatalog(catalog), SparsePayloadCatalogValidationError::None);

  std::swap(catalog.entries[0], catalog.entries[1]);
  EXPECT_EQ(validateSparsePayloadCatalog(catalog),
            SparsePayloadCatalogValidationError::NonCanonicalEntries);

  catalog = makePayloads(10U);
  catalog.entries.pop_back();
  EXPECT_EQ(validateSparsePayloadCatalog(catalog),
            SparsePayloadCatalogValidationError::MissingRequiredEntry);

  catalog = makePayloads(10U);
  catalog.entries[0].root.storage = BlobStorage::SharedMemoryLease;
  catalog.entries[0].root.lease_token = LeaseToken{LeaseTokenId{99U}, StoreInstanceEpoch{1U}};
  EXPECT_EQ(validateSparsePayloadCatalog(catalog),
            SparsePayloadCatalogValidationError::InvalidDurableReference);

  catalog = makePayloads(10U);
  catalog.entries.push_back({SparsePayloadKind::VisualPlaceCatalog, catalog.entries.front().root});
  EXPECT_EQ(validateSparsePayloadCatalog(catalog),
            SparsePayloadCatalogValidationError::DuplicatePayload);
}

TEST(SparseSealApi, AcceptsFirstSealAndExactSuccessorLink) {
  const SparseSubmapSeal first = makeSeal(1U, 0);
  const SparseSubmapSeal second = linkedSuccessor(first);
  EXPECT_EQ(validateSparseSubmapSeal(first), SparseSubmapSealValidationError::None);
  EXPECT_EQ(validateSparseSubmapSeal(second), SparseSubmapSealValidationError::None);
  EXPECT_EQ(validateSparseSubmapLink(first, second), SparseSubmapLinkValidationError::None);
}

TEST(SparseSealApi, RejectsBoundaryAndReferenceSubstitution) {
  const SparseSubmapSeal first = makeSeal(1U, 0);
  SparseSubmapSeal second = linkedSuccessor(first);
  second.from_previous->local_transition.from.state = StateId{999U};
  for (std::size_t index = 0U; index < 4U; ++index) {
    second.from_previous->local_transition.boundary_factor.layout[index].variable.state =
        StateId{999U};
  }
  // The successor can validate its own endpoint, but the cached predecessor
  // is required to prove the incoming endpoint.
  EXPECT_EQ(validateSparseSubmapSeal(second), SparseSubmapSealValidationError::None);
  EXPECT_EQ(validateSparseSubmapLink(first, second),
            SparseSubmapLinkValidationError::EndpointIdentityMismatch);

  second = linkedSuccessor(first);
  second.previous->local_content_checksum = presentHash(250U);
  EXPECT_EQ(validateSparseSubmapSeal(second), SparseSubmapSealValidationError::PreviousRefMismatch);
}

TEST(SparseSealApi, MatchesEndpointCentersValuesAndRevisionsExactly) {
  const SparseSubmapSeal first = makeSeal(1U, 0);
  SparseSubmapSeal second = linkedSuccessor(first);
  second.from_previous->local_transition.from.velocity_odom.x() = 1.0;
  EXPECT_EQ(validateSparseSubmapSeal(second), SparseSubmapSealValidationError::None);
  EXPECT_EQ(validateSparseSubmapLink(first, second),
            SparseSubmapLinkValidationError::EndpointIdentityMismatch);

  second = linkedSuccessor(first);
  second.from_previous->local_transition.from.final_revision = LocalGraphRevision{10U};
  EXPECT_EQ(validateSparseSubmapSeal(second), SparseSubmapSealValidationError::None);
  EXPECT_EQ(validateSparseSubmapLink(first, second),
            SparseSubmapLinkValidationError::EndpointIdentityMismatch);

  second = linkedSuccessor(first);
  second.from_previous->local_transition.to.gyro_bias.x() = 1.0;
  EXPECT_EQ(validateSparseSubmapSeal(second),
            SparseSubmapSealValidationError::CurrentBoundaryMismatch);
}

TEST(SparseSealApi, KeepsContentTransitionAndEnvelopeChecksumDomainsSeparate) {
  SparseSubmapSeal seal = makeSeal(1U, 0);
  seal.seal_checksum = seal.ref.local_content_checksum;
  EXPECT_EQ(validateSparseSubmapSeal(seal), SparseSubmapSealValidationError::ChecksumDomainAliased);

  const SparseSubmapSeal first = makeSeal(1U, 0);
  SparseSubmapSeal second = linkedSuccessor(first);
  second.from_previous->checksum = second.from_previous->local_transition.checksum;
  EXPECT_EQ(validateSparseSubmapSeal(second),
            SparseSubmapSealValidationError::InvalidBoundaryTransition);
}

TEST(SparseSealIdentityApi, ClassifiesIdempotencyAndConflictsExactly) {
  const SparseSubmapSeal accepted = makeSeal(1U, 0);
  SparseSubmapSeal incoming = accepted;
  EXPECT_EQ(classifySparseSealRedelivery(sparseSubmapSealIdentity(accepted),
                                         sparseSubmapSealIdentity(incoming)),
            SparseSealRedeliveryRelation::Idempotent);

  incoming.seal_checksum = presentHash(251U);
  EXPECT_EQ(classifySparseSealRedelivery(sparseSubmapSealIdentity(accepted),
                                         sparseSubmapSealIdentity(incoming)),
            SparseSealRedeliveryRelation::IdentityConflict);

  incoming = accepted;
  incoming.ref.local_content_checksum = presentHash(252U);
  EXPECT_EQ(classifySparseSealRedelivery(sparseSubmapSealIdentity(accepted),
                                         sparseSubmapSealIdentity(incoming)),
            SparseSealRedeliveryRelation::IdentityConflict);

  incoming = accepted;
  incoming.ref.content_revision = SubmapContentRevision{2U};
  EXPECT_EQ(classifySparseSealRedelivery(sparseSubmapSealIdentity(accepted),
                                         sparseSubmapSealIdentity(incoming)),
            SparseSealRedeliveryRelation::DistinctIdentity);
}

TEST(SparseSealApi, RequiresBothPreviousRefAndTransition) {
  SparseSubmapSeal second = makeSeal(2U, 100);
  second.previous = makeSeal(1U, 0).ref;
  EXPECT_EQ(validateSparseSubmapSeal(second),
            SparseSubmapSealValidationError::IncompletePreviousLink);
}

}  // namespace
}  // namespace meridian::core
