#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "meridian/core/sensor_observations.hpp"

namespace meridian::core {
namespace {

[[nodiscard]] GnssObservation validObservation() {
  auto position = GeodeticPosition::fromWgs84Degrees(48.8566, 2.3522, 37.5);
  auto covariance = PositionCovarianceEnu::fromMatrix(
      (Eigen::Vector3d{0.04, 0.09, 0.25}).asDiagonal(),
      PositionCovarianceSource::ReceiverFull);
  EXPECT_TRUE(position);
  EXPECT_TRUE(covariance);

  RecordHeader header;
  header.trace = TraceId{1U};
  header.producer = ProducerId{2U};
  header.session = SessionId{3U};
  header.config = ConfigRevision{4U};
  header.direct_calibration = CalibrationEpoch{5U};

  SourceStamp stamp;
  stamp.clock_revision = ClockRevision{6U};
  stamp.source_epoch = SourceEpoch{7U};
  stamp.ingress_sequence = IngressSequence{8U};

  return GnssObservation{std::move(header),
                         GnssObservationId{9U},
                         std::move(stamp),
                         std::move(position).value(),
                         std::move(covariance).value(),
                         GnssSolutionType::Autonomous,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         std::nullopt,
                         GnssStatus{GnssFixAvailability::Available,
                                    GnssIntegrityStatus::Unknown,
                                    GnssCorrectionStatus::Unknown,
                                    GnssStatusSource::GenericNavSatFix,
                                    std::nullopt}};
}

TEST(GeodeticPosition, EnforcesFiniteWgs84Bounds) {
  auto position = GeodeticPosition::fromWgs84Degrees(48.0, 2.0, 100.0);
  ASSERT_TRUE(position);
  EXPECT_DOUBLE_EQ(position.value().latitudeDeg(), 48.0);
  EXPECT_DOUBLE_EQ(position.value().longitudeDeg(), 2.0);
  EXPECT_DOUBLE_EQ(position.value().altitudeM(), 100.0);

  const auto non_finite = GeodeticPosition::fromWgs84Degrees(
      std::numeric_limits<double>::quiet_NaN(), 2.0, 100.0);
  ASSERT_FALSE(non_finite);
  EXPECT_EQ(non_finite.error().code,
            GnssValidationErrorCode::NonFiniteGeodeticPosition);

  const auto latitude = GeodeticPosition::fromWgs84Degrees(90.1, 2.0, 100.0);
  ASSERT_FALSE(latitude);
  EXPECT_EQ(latitude.error().code, GnssValidationErrorCode::LatitudeOutOfRange);

  const auto longitude = GeodeticPosition::fromWgs84Degrees(48.0, -180.1, 100.0);
  ASSERT_FALSE(longitude);
  EXPECT_EQ(longitude.error().code, GnssValidationErrorCode::LongitudeOutOfRange);
}

TEST(PositionCovarianceEnu, EnforcesSymmetryAndFullPositiveSemidefiniteness) {
  Eigen::Matrix3d valid;
  valid << 1.0, 0.25, 0.0,
           0.25, 2.0, 0.0,
           0.0, 0.0, 0.0;
  auto covariance =
      PositionCovarianceEnu::fromMatrix(valid, PositionCovarianceSource::ReceiverFull);
  ASSERT_TRUE(covariance);
  EXPECT_TRUE(covariance.value().matrix().isApprox(valid, 0.0));

  Eigen::Matrix3d asymmetric = valid;
  asymmetric(0, 1) = 0.5;
  auto rejected =
      PositionCovarianceEnu::fromMatrix(asymmetric, PositionCovarianceSource::ReceiverFull);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GnssValidationErrorCode::NonSymmetricCovariance);

  Eigen::Matrix3d indefinite;
  indefinite << 1.0, 2.0, 0.0,
                2.0, 1.0, 0.0,
                0.0, 0.0, 1.0;
  rejected =
      PositionCovarianceEnu::fromMatrix(indefinite, PositionCovarianceSource::ReceiverFull);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code,
            GnssValidationErrorCode::NonPositiveSemidefiniteCovariance);
}

TEST(PositionCovarianceEnu, EnforcesReceiverDiagonalWireSemantics) {
  Eigen::Matrix3d matrix = Eigen::Matrix3d::Identity();
  matrix(0, 1) = 0.1;
  matrix(1, 0) = 0.1;
  const auto rejected =
      PositionCovarianceEnu::fromMatrix(matrix, PositionCovarianceSource::ReceiverDiagonal);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GnssValidationErrorCode::NonDiagonalCovariance);
}

TEST(PositionCovarianceEnu, StoresToleranceBoundedRoundoffAsPositiveSemidefinite) {
  Eigen::Matrix3d roundoff = Eigen::Matrix3d::Zero();
  roundoff.diagonal() << 1.0, 0.5, -4.0 * std::numeric_limits<double>::epsilon();

  const auto covariance =
      PositionCovarianceEnu::fromMatrix(roundoff, PositionCovarianceSource::ReceiverFull);

  ASSERT_TRUE(covariance) << covariance.error().detail;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance.value().matrix());
  ASSERT_EQ(eigen_solver.info(), Eigen::Success);
  EXPECT_GE(eigen_solver.eigenvalues().minCoeff(), 0.0);
}

TEST(GnssServiceSet, PreservesKnownBitsAndRejectsUnknownOrEmptyMasks) {
  constexpr std::uint16_t kGpsGalileo =
      static_cast<std::uint16_t>(GnssConstellationService::Gps) |
      static_cast<std::uint16_t>(GnssConstellationService::Galileo);
  const auto services = GnssServiceSet::fromMask(kGpsGalileo);
  ASSERT_TRUE(services);
  EXPECT_TRUE(services.value().contains(GnssConstellationService::Gps));
  EXPECT_TRUE(services.value().contains(GnssConstellationService::Galileo));
  EXPECT_FALSE(services.value().contains(GnssConstellationService::Glonass));
  EXPECT_FALSE(GnssServiceSet::fromMask(0U));
  EXPECT_FALSE(GnssServiceSet::fromMask(1U << 8U));
}

TEST(GnssObservationValidation, AcceptsConservativeGenericNavSatRecord) {
  const GnssObservation observation = validObservation();
  const auto validated = validateGnssObservation(observation);
  ASSERT_TRUE(validated) << validated.error().detail;
}

TEST(GnssObservationValidation, GenericSourceCannotClaimReceiverOnlyQuality) {
  GnssObservation observation = validObservation();
  observation.solution = GnssSolutionType::RtkFixed;
  auto validated = validateGnssObservation(observation);
  ASSERT_FALSE(validated);
  EXPECT_EQ(validated.error().code,
            GnssValidationErrorCode::InconsistentSourceSemantics);

  observation = validObservation();
  observation.satellites = 12U;
  validated = validateGnssObservation(observation);
  ASSERT_FALSE(validated);
  EXPECT_EQ(validated.error().code,
            GnssValidationErrorCode::InconsistentSourceSemantics);
}

TEST(GnssObservationValidation, ReceiverMetadataRetainsUnknownAndValidatesKnownValues) {
  GnssObservation observation = validObservation();
  observation.status.source = GnssStatusSource::ReceiverSpecific;
  observation.solution = GnssSolutionType::RtkFloat;
  observation.status.integrity = GnssIntegrityStatus::Valid;
  observation.status.corrections = GnssCorrectionStatus::Current;
  observation.correction_age = Duration{150'000'000};
  observation.hdop = 0.8;
  observation.vdop = 1.2;
  observation.satellites = 18U;
  auto validated = validateGnssObservation(observation);
  ASSERT_TRUE(validated) << validated.error().detail;

  observation.hdop = std::numeric_limits<double>::infinity();
  validated = validateGnssObservation(observation);
  ASSERT_FALSE(validated);
  EXPECT_EQ(validated.error().code,
            GnssValidationErrorCode::InvalidDilutionOfPrecision);
}

}  // namespace
}  // namespace meridian::core
