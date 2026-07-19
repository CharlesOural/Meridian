#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <sophus/so3.hpp>
#include <string>
#include <utility>
#include <vector>

#include "meridian/core/canonical_records.hpp"
#include "meridian/core/sha256.hpp"

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
    ADD_FAILURE() << "canonical checksum failed with code " << static_cast<int>(result.error().code)
                  << " and encoder error " << static_cast<int>(result.error().encoding_error);
    return hashValue(0xe0U);
  }
  return std::move(result).value();
}

RecordHeader recordHeader() {
  RecordHeader header;
  header.schema_version = 3U;
  header.trace = TraceId{10U};
  header.producer = ProducerId{11U};
  header.session = SessionId{12U};
  header.created_at = FusionTime{-1234};
  header.config = ConfigRevision{13U};
  header.direct_calibration = CalibrationEpoch{14U};
  return header;
}

ObservationLineage lineageRecord() {
  ObservationLineage lineage;
  lineage.id = ObservationLineageId{20U};

  ObservationUsage primary;
  primary.slice.root = MeasurementId{21U};
  primary.slice.kind = SliceKind::Whole;
  primary.slice.source_checksum = hashValue(1U);
  primary.slice.calibration = CalibrationEpoch{14U};
  primary.role = ObservationRole::PrimaryResidual;
  primary.consumer = DerivedRecordId{22U};
  primary.factor_group = FactorGroupId{23U};
  primary.correlation_group = CorrelationGroupId{24U};

  ObservationUsage conditioning;
  conditioning.slice.root = GnssObservationId{25U};
  conditioning.slice.kind = SliceKind::IndexRange;
  conditioning.slice.begin = 3U;
  conditioning.slice.end = 9U;
  conditioning.slice.source_checksum = hashValue(2U);
  conditioning.slice.calibration = CalibrationEpoch{14U};
  conditioning.role = ObservationRole::ConditioningOnly;
  conditioning.consumer = DerivedRecordId{26U};
  conditioning.correlation_group = CorrelationGroupId{24U};
  lineage.usage = {primary, conditioning};

  CorrelationDeclaration correlation;
  correlation.group = CorrelationGroupId{24U};
  correlation.policy = CorrelationPolicyRevision{27U};
  correlation.treatment = CorrelationTreatment::CovarianceInflationAndInformationCap;
  correlation.covariance_inflation = 1.75;
  correlation.total_information_cap = 42.5;
  lineage.correlations = {correlation};
  return lineage;
}

LocalVariableRef stateVariable(LocalVariableKind kind, StateId state) {
  LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

FrozenSquareRootFactor frozenFactor() {
  FrozenSquareRootFactor factor;
  factor.rows = 1U;
  factor.columns = 30U;
  std::uint32_t offset = 0U;
  for (const StateId state : {StateId{30U}, StateId{31U}}) {
    for (const auto& [kind, dimension] : std::array<std::pair<LocalVariableKind, std::uint32_t>, 4>{
             std::pair{LocalVariableKind::Pose, 6U},
             std::pair{LocalVariableKind::NavigationVelocity, 3U},
             std::pair{LocalVariableKind::GyroBias, 3U},
             std::pair{LocalVariableKind::AccelBias, 3U}}) {
      factor.layout.push_back(SquareRootColumnBlock{stateVariable(kind, state), offset, dimension});
      offset += dimension;
    }
  }
  factor.row_major_A.assign(30U, 0.0);
  factor.row_major_A.front() = 1.0;
  factor.rhs = {0.25};
  factor.constant_squared_error = 0.125;
  factor.numerical_rank = 1U;
  factor.absolute_rank_tolerance = 1.0e-9;
  factor.relative_rank_tolerance = 1.0e-6;
  factor.cost_statistics.source_residual_dof = 4U;
  factor.cost_statistics.eliminated_numerical_rank = 1U;
  factor.cost_statistics.effective_dof = 3U;
  factor.cost_statistics.calibration_revision = ResidualCalibrationRevision{90U};
  factor.cost_statistics.calibrated_total_cost_cutoff = 8.25;
  return factor;
}

BoundaryNavigationLinearization boundary(StateId state, std::int64_t time,
                                         LocalGraphRevision revision, double x) {
  BoundaryNavigationLinearization value;
  value.state = state;
  value.exact_time = FusionTime{time};
  value.final_revision = revision;
  value.T_odom_imu = Pose3d{Sophus::SO3d{}, Vector3d{x, 0.5 * x, -0.25 * x}};
  value.velocity_odom = Vector3d{x + 1.0, x + 2.0, x + 3.0};
  value.gyro_bias = Vector3d{0.01 * x, 0.02 * x, 0.03 * x};
  value.accel_bias = Vector3d{-0.01 * x, -0.02 * x, -0.03 * x};
  return value;
}

CondensedBoundaryTransition condensedTransition(const ObservationLineage& lineage,
                                                const FrozenSquareRootFactor& factor) {
  CondensedBoundaryTransition transition;
  transition.header = recordHeader();
  transition.odom_epoch = OdomEpoch{40U};
  transition.from = boundary(StateId{30U}, 0, LocalGraphRevision{50U}, 1.0);
  transition.to = boundary(StateId{31U}, 1000, LocalGraphRevision{51U}, 2.0);
  transition.boundary_factor = factor;
  transition.source_factors = {
      LocalFactorRef{transition.odom_epoch, FactorId{60U}},
      LocalFactorRef{transition.odom_epoch, FactorId{61U}},
  };
  transition.lineage = lineage;
  transition.final_revision = LocalGraphRevision{52U};
  transition.input_partition_checksum = hashValue(3U);
  return transition;
}

BlobRef durableBlob(std::uint64_t id, std::uint8_t checksum_seed) {
  BlobRef blob;
  blob.store = BlobStoreId{70U};
  blob.id = BlobId{id};
  blob.checksum = hashValue(checksum_seed);
  blob.layout = LayoutId{100U + id};
  blob.bytes = 1000U + id;
  blob.storage = BlobStorage::DurableSpool;
  return blob;
}

SparsePayloadCatalog payloadCatalog() {
  SparsePayloadCatalog catalog;
  catalog.entries = {
      {SparsePayloadKind::InternalTrajectory, durableBlob(1U, 10U)},
      {SparsePayloadKind::KeyframeIndex, durableBlob(2U, 11U)},
      {SparsePayloadKind::RegistrationProxy, durableBlob(3U, 12U)},
      {SparsePayloadKind::DenseInputIndex, durableBlob(4U, 13U)},
  };
  return catalog;
}

struct CanonicalFixture {
  ObservationLineage lineage;
  FrozenSquareRootFactor factor;
  CondensedBoundaryTransition transition;
  SparsePayloadCatalog catalog;
  SparseSubmapSeal seal;
};

CanonicalFixture canonicalFixture() {
  CanonicalFixture fixture;
  fixture.lineage = lineageRecord();
  fixture.lineage.checksum = takeHash(recomputeObservationLineageChecksum(fixture.lineage));

  fixture.factor = frozenFactor();
  fixture.factor.checksum = takeHash(recomputeFrozenSquareRootFactorChecksum(fixture.factor));

  fixture.transition = condensedTransition(fixture.lineage, fixture.factor);
  fixture.transition.checksum =
      takeHash(recomputeCondensedBoundaryTransitionChecksum(fixture.transition));

  fixture.catalog = payloadCatalog();
  fixture.catalog.checksum = takeHash(recomputeSparsePayloadCatalogChecksum(fixture.catalog));

  SparseSubmapSeal& seal = fixture.seal;
  seal.header = recordHeader();
  seal.ref = SubmapRef{seal.header.session,
                       fixture.transition.odom_epoch,
                       SubmapId{81U},
                       *seal.header.direct_calibration,
                       SubmapContentRevision{82U},
                       {}};
  seal.previous = SubmapRef{seal.header.session,
                            fixture.transition.odom_epoch,
                            SubmapId{80U},
                            *seal.header.direct_calibration,
                            SubmapContentRevision{81U},
                            hashValue(30U)};
  seal.final_local_revision = fixture.transition.final_revision;
  seal.support_time = TimeRange{fixture.transition.to.exact_time, FusionTime{2000}};
  seal.frame.boundary_state = fixture.transition.to.state;
  seal.frame.boundary_time = fixture.transition.to.exact_time;
  seal.frame.gravity_up_odom = Vector3d::UnitZ();
  seal.frame.boundary_yaw_odom = 0.2;
  seal.T_odom_submap = Pose3d{Sophus::SO3d{}, Vector3d{2.0, 1.0, -0.5}};
  seal.boundary_navigation = fixture.transition.to;
  seal.payloads = fixture.catalog;
  seal.lineage = fixture.lineage;
  seal.quality_checksum = hashValue(31U);

  seal.ref.local_content_checksum = takeHash(recomputeSubmapLocalContentChecksum(seal));
  SealedBoundaryTransition sealed{*seal.previous, seal.ref, fixture.transition, {}};
  sealed.checksum = takeHash(recomputeSealedBoundaryTransitionChecksum(sealed));
  seal.from_previous = sealed;
  seal.seal_checksum = takeHash(recomputeSparseSubmapSealChecksum(seal));
  return fixture;
}

TEST(CanonicalRecords, ComputesSevenGoldenDomainDigests) {
  const CanonicalFixture fixture = canonicalFixture();
  EXPECT_EQ(sha256Hex(fixture.lineage.checksum),
            "d35311d24bbd7fb7700b1418af84b589fd2cdc884c9527b7fbf763536a7d78dc");
  EXPECT_EQ(sha256Hex(fixture.factor.checksum),
            "bbb71a290c5fd645dfc87f1d87303adae84fd2afb428f2fb839e92b730f483a9");
  EXPECT_EQ(sha256Hex(fixture.transition.checksum),
            "60e505f5eda28b92d12190462e246078ef083b0d753ac31570d8ba2d83c4979e");
  EXPECT_EQ(sha256Hex(fixture.catalog.checksum),
            "a0b33d74814db990018db25d378338c82bd98d850ae3e8c528ad1bf67c9a57c9");
  EXPECT_EQ(sha256Hex(fixture.seal.ref.local_content_checksum),
            "13733fbf526f7b83d112a595aad075aeab6dc06b58d0c3433f55e1308cf38e1a");
  EXPECT_EQ(sha256Hex(fixture.seal.from_previous->checksum),
            "dad9cc8b53c3bff0946c17581caf24a334eabdd0337160aa303b75ea32307a4e");
  EXPECT_EQ(sha256Hex(fixture.seal.seal_checksum),
            "bc0e4f73f5897c17f08e465cfc223d2d1bc64e39a10e6980164e0a3a416abaf5");

  EXPECT_EQ(validateLineage(fixture.lineage), LineageValidationError::None);
  EXPECT_EQ(validateFrozenSquareRootFactor(fixture.factor), FrozenFactorValidationError::None);
  EXPECT_EQ(validateCondensedBoundaryTransition(fixture.transition),
            CondensedTransitionValidationError::None);
  EXPECT_EQ(validateSparsePayloadCatalog(fixture.catalog),
            SparsePayloadCatalogValidationError::None);
  EXPECT_EQ(validateSealedBoundaryTransition(*fixture.seal.from_previous),
            SealedBoundaryTransitionValidationError::None);
  EXPECT_EQ(validateSparseSubmapSeal(fixture.seal), SparseSubmapSealValidationError::None);
}

TEST(CanonicalRecords, DomainConstantsSeparateIdenticalPayloads) {
  constexpr std::array domains{
      kObservationLineageChecksumDomain,
      kFrozenSquareRootFactorChecksumDomain,
      kCondensedBoundaryTransitionChecksumDomain,
      kSparsePayloadCatalogChecksumDomain,
      kSubmapLocalContentChecksumDomain,
      kSealedBoundaryTransitionChecksumDomain,
      kSparseSubmapSealChecksumDomain,
  };
  std::set<std::string> tags;
  std::set<ContentHash> digests;
  for (const CanonicalRecordDomain& domain : domains) {
    EXPECT_GT(domain.schema_version, 0U);
    EXPECT_TRUE(tags.emplace(domain.tag).second);
    auto encoder_result = CanonicalEncoder::create(domain.tag, domain.schema_version, 256U);
    ASSERT_TRUE(encoder_result);
    CanonicalEncoder encoder = std::move(encoder_result).value();
    EXPECT_EQ(encoder.writeU32(0x12345678U), CanonicalEncodingError::None);
    auto encoded = encoder.finish();
    ASSERT_TRUE(encoded);
    EXPECT_TRUE(digests.emplace(encoded.value().digest()).second);
  }
}

TEST(CanonicalRecords, EveryChildDomainRespondsToOwnedFieldMutation) {
  const CanonicalFixture fixture = canonicalFixture();

  ObservationLineage lineage = fixture.lineage;
  lineage.correlations.front().treatment = CorrelationTreatment::NotIndependent;
  EXPECT_NE(takeHash(recomputeObservationLineageChecksum(lineage)), fixture.lineage.checksum);
  lineage = fixture.lineage;
  lineage.checksum = hashValue(200U);
  EXPECT_EQ(takeHash(recomputeObservationLineageChecksum(lineage)), fixture.lineage.checksum);

  FrozenSquareRootFactor factor = fixture.factor;
  factor.row_major_A[1] = 0.5;
  EXPECT_NE(takeHash(recomputeFrozenSquareRootFactorChecksum(factor)), fixture.factor.checksum);
  factor = fixture.factor;
  factor.checksum = hashValue(201U);
  EXPECT_EQ(takeHash(recomputeFrozenSquareRootFactorChecksum(factor)), fixture.factor.checksum);
  factor = fixture.factor;
  factor.cost_statistics.calibration_revision = ResidualCalibrationRevision{91U};
  EXPECT_NE(takeHash(recomputeFrozenSquareRootFactorChecksum(factor)), fixture.factor.checksum);
  factor = fixture.factor;
  factor.cost_statistics.calibrated_total_cost_cutoff = 9.25;
  EXPECT_NE(takeHash(recomputeFrozenSquareRootFactorChecksum(factor)), fixture.factor.checksum);

  CondensedBoundaryTransition transition = fixture.transition;
  transition.source_factors.back().factor = FactorId{62U};
  EXPECT_NE(takeHash(recomputeCondensedBoundaryTransitionChecksum(transition)),
            fixture.transition.checksum);
  transition = fixture.transition;
  transition.checksum = hashValue(202U);
  EXPECT_EQ(takeHash(recomputeCondensedBoundaryTransitionChecksum(transition)),
            fixture.transition.checksum);

  SparsePayloadCatalog catalog = fixture.catalog;
  ++catalog.entries.front().root.bytes;
  EXPECT_NE(takeHash(recomputeSparsePayloadCatalogChecksum(catalog)), fixture.catalog.checksum);
  catalog = fixture.catalog;
  catalog.checksum = hashValue(203U);
  EXPECT_EQ(takeHash(recomputeSparsePayloadCatalogChecksum(catalog)), fixture.catalog.checksum);
}

TEST(CanonicalRecords, LocalContentExcludesLinksAndSelfHashButSealIncludesEnvelope) {
  const CanonicalFixture fixture = canonicalFixture();
  const ContentHash local = fixture.seal.ref.local_content_checksum;

  SparseSubmapSeal seal = fixture.seal;
  seal.support_time.end.nanoseconds += 1;
  EXPECT_NE(takeHash(recomputeSubmapLocalContentChecksum(seal)), local);

  seal = fixture.seal;
  seal.ref.local_content_checksum = hashValue(210U);
  EXPECT_EQ(takeHash(recomputeSubmapLocalContentChecksum(seal)), local);
  seal = fixture.seal;
  seal.previous->id = SubmapId{999U};
  seal.from_previous->checksum = hashValue(211U);
  EXPECT_EQ(takeHash(recomputeSubmapLocalContentChecksum(seal)), local);

  SealedBoundaryTransition adjacent = *fixture.seal.from_previous;
  const ContentHash adjacent_hash = adjacent.checksum;
  adjacent.to.content_revision = SubmapContentRevision{999U};
  EXPECT_NE(takeHash(recomputeSealedBoundaryTransitionChecksum(adjacent)), adjacent_hash);
  adjacent = *fixture.seal.from_previous;
  adjacent.checksum = hashValue(212U);
  EXPECT_EQ(takeHash(recomputeSealedBoundaryTransitionChecksum(adjacent)), adjacent_hash);

  seal = fixture.seal;
  const ContentHash envelope_hash = seal.seal_checksum;
  seal.header.trace = TraceId{999U};
  EXPECT_NE(takeHash(recomputeSparseSubmapSealChecksum(seal)), envelope_hash);
  seal = fixture.seal;
  seal.from_previous->checksum = hashValue(213U);
  EXPECT_NE(takeHash(recomputeSparseSubmapSealChecksum(seal)), envelope_hash);
  seal = fixture.seal;
  seal.seal_checksum = hashValue(214U);
  EXPECT_EQ(takeHash(recomputeSparseSubmapSealChecksum(seal)), envelope_hash);
}

TEST(CanonicalRecords, ReportsDependencyLimitEnumAndFloatingFailuresPrecisely) {
  const CanonicalFixture fixture = canonicalFixture();

  ObservationLineage missing = fixture.lineage;
  missing.usage.front().slice.source_checksum = {};
  auto missing_result = recomputeObservationLineageChecksum(missing);
  ASSERT_FALSE(missing_result);
  EXPECT_EQ(missing_result.error().code, CanonicalRecordErrorCode::MissingDependencyChecksum);
  EXPECT_EQ(missing_result.error().record, CanonicalRecordKind::ObservationLineage);

  auto invalid_limits =
      recomputeObservationLineageChecksum(fixture.lineage, CanonicalRecordLimits{0U, 1U});
  ASSERT_FALSE(invalid_limits);
  EXPECT_EQ(invalid_limits.error().code, CanonicalRecordErrorCode::InvalidLimits);

  auto collection_limit =
      recomputeObservationLineageChecksum(fixture.lineage, CanonicalRecordLimits{4096U, 1U});
  ASSERT_FALSE(collection_limit);
  EXPECT_EQ(collection_limit.error().code, CanonicalRecordErrorCode::CollectionLimitExceeded);

  auto byte_limit =
      recomputeObservationLineageChecksum(fixture.lineage, CanonicalRecordLimits{64U, 100U});
  ASSERT_FALSE(byte_limit);
  EXPECT_EQ(byte_limit.error().code, CanonicalRecordErrorCode::EncodingFailure);
  EXPECT_EQ(byte_limit.error().encoding_error, CanonicalEncodingError::OutputLimitExceeded);

  ObservationLineage unsupported = fixture.lineage;
  unsupported.usage.front().role = static_cast<ObservationRole>(99);
  auto unsupported_result = recomputeObservationLineageChecksum(unsupported);
  ASSERT_FALSE(unsupported_result);
  EXPECT_EQ(unsupported_result.error().code, CanonicalRecordErrorCode::UnsupportedValue);

  FrozenSquareRootFactor nonfinite = fixture.factor;
  nonfinite.rhs.front() = std::numeric_limits<double>::infinity();
  auto nonfinite_result = recomputeFrozenSquareRootFactorChecksum(nonfinite);
  ASSERT_FALSE(nonfinite_result);
  EXPECT_EQ(nonfinite_result.error().code, CanonicalRecordErrorCode::EncodingFailure);
  EXPECT_EQ(nonfinite_result.error().encoding_error,
            CanonicalEncodingError::NonFiniteFloatingPoint);
}

}  // namespace
}  // namespace meridian::core
