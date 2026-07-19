#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sophus/se3.hpp>
#include <string>

#include "meridian/core/api.hpp"

namespace meridian::core {
namespace {

ObservationSlice whole(MeasurementId id) {
  ObservationSlice slice;
  slice.root = id;
  slice.kind = SliceKind::Whole;
  slice.calibration = CalibrationEpoch{1};
  return slice;
}

TEST(StrongId, DoesNotConfuseIdentityWithValueValidity) {
  EXPECT_FALSE(MeasurementId{}.valid());
  EXPECT_TRUE(MeasurementId{0}.valid());
  EXPECT_EQ(MeasurementId{4}, MeasurementId{4});
  EXPECT_NE(MeasurementId{4}, MeasurementId{5});
}

TEST(TimeRange, UsesHalfOpenOwnership) {
  const TimeRange range{FusionTime{10}, FusionTime{20}};
  EXPECT_TRUE(range.valid());
  EXPECT_TRUE(range.contains(FusionTime{10}));
  EXPECT_TRUE(range.contains(FusionTime{19}));
  EXPECT_FALSE(range.contains(FusionTime{20}));
}

TEST(FusionTime, AddsAndSubtractsTypedDurations) {
  const FusionTime time{100};
  EXPECT_EQ(time + Duration{25}, FusionTime{125});
  EXPECT_EQ(time - Duration{25}, FusionTime{75});
}

TEST(ClockMapping, AppliesExplicitAffineRevision) {
  AffineClockModel model;
  model.revision = ClockRevision{1};
  model.source_epoch = SourceEpoch{2};
  model.raw_reference = RawDeviceTime{1'000};
  model.fusion_reference = FusionTime{10'000};
  model.rate = 1.0001;
  model.uncertainty = Duration{50};
  const auto mapped = mapSourceStamp(model, RawDeviceTime{11'000}, ArrivalTime{20'000},
                                     std::nullopt, IngressSequence{3});
  ASSERT_TRUE(mapped);
  EXPECT_EQ(mapped.value().fusion_time.nanoseconds, 20'001);
  EXPECT_EQ(mapped.value().uncertainty.nanoseconds, 50);
}

TEST(ClockGuard, RejectsTimeRewriteInsteadOfClamping) {
  AffineClockModel model;
  model.revision = ClockRevision{1};
  model.source_epoch = SourceEpoch{2};
  SourceClockGuard guard;
  auto first =
      mapSourceStamp(model, RawDeviceTime{100}, ArrivalTime{100}, std::nullopt, IngressSequence{1});
  auto duplicate =
      mapSourceStamp(model, RawDeviceTime{100}, ArrivalTime{101}, std::nullopt, IngressSequence{2});
  ASSERT_TRUE(first);
  ASSERT_TRUE(duplicate);
  EXPECT_TRUE(guard.admit(first.value()));
  const auto rejected = guard.admit(duplicate.value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ClockGuardErrorCode::NonMonotonicRawTime);
}

TEST(ObservationLineage, RejectsUndeclaredDuplicatePrimaryRows) {
  ObservationLineage lineage;
  lineage.usage = {
      ObservationUsage{whole(MeasurementId{7}), ObservationRole::PrimaryResidual,
                       DerivedRecordId{1}, FactorGroupId{1}, std::nullopt},
      ObservationUsage{whole(MeasurementId{7}), ObservationRole::PrimaryResidual,
                       DerivedRecordId{2}, FactorGroupId{2}, std::nullopt},
  };
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::DuplicatePrimaryObservation);
}

TEST(ObservationLineage, AllowsDeclaredCompositeRows) {
  const CorrelationGroupId correlation{5};
  const FactorGroupId factor_group{3};
  ObservationLineage lineage;
  lineage.correlations.push_back(
      CorrelationDeclaration{correlation, CorrelationPolicyRevision{1},
                             CorrelationTreatment::JointCompositeWhitening, 1.0, std::nullopt});
  lineage.usage = {
      ObservationUsage{whole(MeasurementId{7}), ObservationRole::PrimaryResidual,
                       DerivedRecordId{1}, factor_group, correlation},
      ObservationUsage{whole(MeasurementId{7}), ObservationRole::PrimaryResidual,
                       DerivedRecordId{2}, factor_group, correlation},
  };
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::None);
}

TEST(ObservationLineage, RequiresCanonicalValidSlices) {
  ObservationSlice slice = whole(MeasurementId{7U});
  EXPECT_TRUE(slice.valid());

  slice.begin = 1U;
  EXPECT_FALSE(slice.valid());
  slice = whole(MeasurementId{7U});
  slice.kind = SliceKind::IndexRange;
  slice.begin = 2U;
  slice.end = 2U;
  EXPECT_FALSE(slice.valid());
  slice.end = 3U;
  EXPECT_TRUE(slice.valid());
  slice.root = MeasurementId{};
  EXPECT_FALSE(slice.valid());

  slice = whole(MeasurementId{7U});
  slice.kind = static_cast<SliceKind>(99);
  EXPECT_FALSE(slice.valid());
}

TEST(ObservationLineage, ValidatesRolesConsumersAndOptionalIdentities) {
  ObservationLineage lineage;
  lineage.usage.push_back(ObservationUsage{whole(MeasurementId{7U}),
                                           ObservationRole::PrimaryResidual, DerivedRecordId{1U},
                                           FactorGroupId{2U}, std::nullopt});
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::None);

  lineage.usage.front().role = static_cast<ObservationRole>(99);
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidRole);
  lineage.usage.front().role = ObservationRole::PrimaryResidual;
  lineage.usage.front().consumer = DerivedRecordId{};
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidConsumer);
  lineage.usage.front().consumer = DerivedRecordId{1U};
  lineage.usage.front().factor_group = FactorGroupId{};
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidFactorGroup);
  lineage.usage.front().factor_group = FactorGroupId{2U};
  lineage.usage.front().correlation_group = CorrelationGroupId{};
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidCorrelationGroup);
}

TEST(ObservationLineage, ValidatesCorrelationDeclarationsAndCaps) {
  ObservationLineage lineage;
  lineage.usage.push_back(ObservationUsage{whole(MeasurementId{7U}),
                                           ObservationRole::PrimaryResidual, DerivedRecordId{1U},
                                           FactorGroupId{2U}, CorrelationGroupId{3U}});
  lineage.correlations.push_back(
      CorrelationDeclaration{CorrelationGroupId{3U}, CorrelationPolicyRevision{4U},
                             CorrelationTreatment::CovarianceInflationAndInformationCap, 1.5, 8.0});
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::None);

  lineage.correlations.front().treatment = static_cast<CorrelationTreatment>(99);
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidCorrelationTreatment);
  lineage.correlations.front().treatment =
      CorrelationTreatment::CovarianceInflationAndInformationCap;
  lineage.correlations.front().total_information_cap = 0.0;
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidInformationCap);
  lineage.correlations.front().total_information_cap = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::InvalidInformationCap);
  lineage.correlations.front().total_information_cap = 8.0;
  lineage.correlations.push_back(lineage.correlations.front());
  EXPECT_EQ(validateLineage(lineage), LineageValidationError::DuplicateCorrelationDeclaration);
}

TEST(ObservationLineage, MalformedAncestryNeverProvesIndependence) {
  ObservationLineage left;
  left.usage.push_back(ObservationUsage{whole(MeasurementId{1U}), ObservationRole::ConditioningOnly,
                                        DerivedRecordId{1U}, std::nullopt, std::nullopt});
  ObservationLineage right;
  right.usage.push_back(ObservationUsage{whole(MeasurementId{2U}),
                                         ObservationRole::ConditioningOnly, DerivedRecordId{2U},
                                         std::nullopt, std::nullopt});
  EXPECT_TRUE(lineagesAreIndependent(left, right));

  left.usage.front().slice.kind = static_cast<SliceKind>(99);
  EXPECT_FALSE(lineagesAreIndependent(left, right));
  EXPECT_TRUE(left.usage.front().slice.overlaps(right.usage.front().slice));
}

TEST(BoundedQueue, MakesOverflowObservable) {
  BoundedQueue<std::string> queue(2, 8, QueueOverflowPolicy::DropOldest,
                                  [](const std::string& value) { return value.size(); });
  EXPECT_EQ(queue.push("abc"), QueuePushStatus::Accepted);
  EXPECT_EQ(queue.push("def"), QueuePushStatus::Accepted);
  EXPECT_EQ(queue.push("ghij"), QueuePushStatus::AcceptedAfterDroppingOldest);
  EXPECT_EQ(queue.tryPop(), std::optional<std::string>{"def"});
  EXPECT_EQ(queue.stats().dropped_oldest, 1U);
}

TEST(Geometry, TransformDirectionComposesLeftToRight) {
  const Pose3d T_a_b(Sophus::SO3d{}, Eigen::Vector3d{1.0, 0.0, 0.0});
  const Pose3d T_b_c(Sophus::SO3d{}, Eigen::Vector3d{0.0, 2.0, 0.0});
  const Eigen::Vector3d point_c{0.0, 0.0, 3.0};
  const Eigen::Vector3d point_a = T_a_b * T_b_c * point_c;
  EXPECT_TRUE(point_a.isApprox(Eigen::Vector3d{1.0, 2.0, 3.0}));
}

enum class ParseError { Invalid };

TEST(Result, KeepsExpectedRuntimeFailuresTyped) {
  auto accepted = Result<int, ParseError>::success(42);
  ASSERT_TRUE(accepted.hasValue());
  EXPECT_EQ(accepted.value(), 42);

  const auto rejected = Result<int, ParseError>::failure(ParseError::Invalid);
  ASSERT_FALSE(rejected.hasValue());
  EXPECT_EQ(rejected.error(), ParseError::Invalid);
}

}  // namespace
}  // namespace meridian::core
