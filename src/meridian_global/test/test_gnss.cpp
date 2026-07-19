#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <sophus/se3.hpp>
#include <vector>

#include "meridian/global/gnss.hpp"

namespace meridian::global {
namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000LL;

TEST(Wgs84Geodesy, KnownAxesAndRoundTripArePrecise) {
  const auto equator = Wgs84Lla::fromDegrees(0.0, 0.0, 0.0);
  ASSERT_TRUE(equator);
  const auto equator_ecef = wgs84LlaToEcef(equator.value());
  ASSERT_TRUE(equator_ecef);
  EXPECT_NEAR(equator_ecef.value().meters().x(), 6378137.0, 1.0e-8);
  EXPECT_NEAR(equator_ecef.value().meters().y(), 0.0, 1.0e-8);
  EXPECT_NEAR(equator_ecef.value().meters().z(), 0.0, 1.0e-8);

  const auto north_pole = Wgs84Lla::fromDegrees(90.0, 0.0, 0.0);
  ASSERT_TRUE(north_pole);
  const auto pole_ecef = wgs84LlaToEcef(north_pole.value());
  ASSERT_TRUE(pole_ecef);
  EXPECT_NEAR(pole_ecef.value().meters().head<2>().norm(), 0.0, 1.0e-8);
  EXPECT_NEAR(pole_ecef.value().meters().z(), 6356752.314245179, 1.0e-6);
  const auto pole_round_trip = wgs84EcefToLla(pole_ecef.value());
  ASSERT_TRUE(pole_round_trip);
  EXPECT_NEAR(pole_round_trip.value().latitudeDeg(), 90.0, 1.0e-12);
  EXPECT_NEAR(pole_round_trip.value().altitudeM(), 0.0, 1.0e-6);

  const auto paris = Wgs84Lla::fromDegrees(48.8566, 2.3522, 35.4);
  ASSERT_TRUE(paris);
  const auto paris_ecef = wgs84LlaToEcef(paris.value());
  ASSERT_TRUE(paris_ecef);
  const auto paris_round_trip = wgs84EcefToLla(paris_ecef.value());
  ASSERT_TRUE(paris_round_trip);
  EXPECT_NEAR(paris_round_trip.value().latitudeDeg(), 48.8566, 1.0e-10);
  EXPECT_NEAR(paris_round_trip.value().longitudeDeg(), 2.3522, 1.0e-10);
  EXPECT_NEAR(paris_round_trip.value().altitudeM(), 35.4, 1.0e-5);
}

TEST(Wgs84Geodesy, LocalEnuDatumUsesEastNorthUpConvention) {
  const auto origin = Wgs84Lla::fromDegrees(48.8566, 2.3522, 35.4);
  ASSERT_TRUE(origin);
  const auto datum = LocalEnuDatum::create(origin.value());
  ASSERT_TRUE(datum);

  const auto enu = EnuPosition::fromMeters(Eigen::Vector3d{123.0, -47.0, 8.5});
  ASSERT_TRUE(enu);
  const auto ecef = datum.value().toEcef(enu.value());
  ASSERT_TRUE(ecef);
  const auto recovered = datum.value().toEnu(ecef.value());
  ASSERT_TRUE(recovered);
  EXPECT_TRUE(recovered.value().meters().isApprox(enu.value().meters(), 1.0e-8));

  const auto lla = datum.value().toLla(enu.value());
  ASSERT_TRUE(lla);
  const auto recovered_from_lla = datum.value().toEnu(lla.value());
  ASSERT_TRUE(recovered_from_lla);
  EXPECT_TRUE(recovered_from_lla.value().meters().isApprox(enu.value().meters(), 2.0e-6));

  EXPECT_NEAR(datum.value().REnuEcef().row(0).dot(datum.value().REnuEcef().row(1)), 0.0, 1.0e-14);
  EXPECT_NEAR(datum.value().REnuEcef().determinant(), 1.0, 1.0e-14);
}

TEST(AntennaPrediction, ExactLeverArmAndJacobiansFollowDeclaredConventions) {
  const Sophus::SO3d rotation = Sophus::SO3d::exp(Eigen::Vector3d{0.2, -0.1, 0.4});
  const core::Pose3d T_map_imu(rotation, Eigen::Vector3d{2.0, -3.0, 1.0});
  const auto lever = AntennaLeverArm::fromImuToAntenna(Eigen::Vector3d{0.8, -0.2, 0.4});
  ASSERT_TRUE(lever);
  const auto alignment = GravityAlignedTransform::create(0.7, Eigen::Vector3d{100.0, -20.0, 5.0});
  ASSERT_TRUE(alignment);
  const auto prediction = predictAntennaPosition(T_map_imu, lever.value(), alignment.value());
  ASSERT_TRUE(prediction);

  const Eigen::Vector3d expected_map =
      T_map_imu.translation() + T_map_imu.so3().matrix() * lever.value().imuToAntennaInImu();
  EXPECT_TRUE(prediction.value().positionMap().isApprox(expected_map, 1.0e-14));
  EXPECT_TRUE(
      prediction.value().positionEnu().isApprox(alignment.value().apply(expected_map), 1.0e-14));

  constexpr double kEpsilon = 1.0e-7;
  Eigen::Matrix<double, 3, 6> numerical_pose;
  for (Eigen::Index column = 0; column < 6; ++column) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta(column) = kEpsilon;
    const auto plus = predictAntennaPosition(T_map_imu * core::Pose3d::exp(delta), lever.value(),
                                             alignment.value());
    const auto minus = predictAntennaPosition(T_map_imu * core::Pose3d::exp(-delta), lever.value(),
                                              alignment.value());
    ASSERT_TRUE(plus);
    ASSERT_TRUE(minus);
    numerical_pose.col(column) =
        (plus.value().positionEnu() - minus.value().positionEnu()) / (2.0 * kEpsilon);
  }
  EXPECT_TRUE(numerical_pose.isApprox(prediction.value().positionEnuJacobianPose(), 2.0e-7));

  Eigen::Matrix<double, 3, 4> numerical_alignment;
  for (Eigen::Index column = 0; column < 4; ++column) {
    double yaw_plus = alignment.value().yawEnuMapRad();
    double yaw_minus = yaw_plus;
    Eigen::Vector3d translation_plus = alignment.value().translationEnu();
    Eigen::Vector3d translation_minus = translation_plus;
    if (column == 3) {
      yaw_plus += kEpsilon;
      yaw_minus -= kEpsilon;
    } else {
      translation_plus(column) += kEpsilon;
      translation_minus(column) -= kEpsilon;
    }
    const auto plus_alignment = GravityAlignedTransform::create(yaw_plus, translation_plus);
    const auto minus_alignment = GravityAlignedTransform::create(yaw_minus, translation_minus);
    ASSERT_TRUE(plus_alignment);
    ASSERT_TRUE(minus_alignment);
    const auto plus = predictAntennaPosition(T_map_imu, lever.value(), plus_alignment.value());
    const auto minus = predictAntennaPosition(T_map_imu, lever.value(), minus_alignment.value());
    ASSERT_TRUE(plus);
    ASSERT_TRUE(minus);
    numerical_alignment.col(column) =
        (plus.value().positionEnu() - minus.value().positionEnu()) / (2.0 * kEpsilon);
  }
  EXPECT_TRUE(
      numerical_alignment.isApprox(prediction.value().positionEnuJacobianAlignment(), 2.0e-7));
}

TEST(GravityAlignedAlignment, RobustlyRecoversYawAndTranslationWithOutliers) {
  constexpr double kTrueYaw = 0.63;
  const Eigen::Vector3d true_translation{41.0, -17.0, 3.5};
  const auto truth = GravityAlignedTransform::create(kTrueYaw, true_translation);
  ASSERT_TRUE(truth);

  std::vector<AlignmentCorrespondence> correspondences;
  for (std::size_t index = 0; index < 32U; ++index) {
    const double i = static_cast<double>(index);
    const Eigen::Vector3d map_position{0.9 * i, 4.0 * std::sin(0.23 * i), 0.3 * std::cos(0.17 * i)};
    Eigen::Vector3d enu_position = truth.value().apply(map_position);
    enu_position += Eigen::Vector3d{0.008 * std::sin(1.7 * i), 0.008 * std::cos(1.1 * i),
                                    0.012 * std::sin(0.8 * i)};
    if (index == 3U || index == 11U || index == 19U || index == 27U) {
      enu_position += Eigen::Vector3d{8.0, -11.0, 4.0};
    }
    correspondences.push_back(AlignmentCorrespondence{
        core::FusionTime{static_cast<std::int64_t>(index) * 200'000'000LL}, map_position,
        enu_position, 0.03 * 0.03 * Eigen::Matrix3d::Identity()});
  }

  const auto estimate = estimateGravityAlignedTransform(correspondences);
  ASSERT_TRUE(estimate) << static_cast<int>(estimate.error().code);
  EXPECT_NEAR(estimate.value().transform().yawEnuMapRad(), kTrueYaw, 1.5e-3);
  EXPECT_TRUE(estimate.value().transform().translationEnu().isApprox(true_translation, 0.025));
  EXPECT_GE(estimate.value().diagnostics().robustOutlierCount(), 4U);
  EXPECT_EQ(estimate.value().diagnostics().correspondenceCount(), correspondences.size());
  EXPECT_TRUE(estimate.value().covariance().allFinite());
  EXPECT_GT(estimate.value().covariance().diagonal().minCoeff(), 0.0);
}

TEST(GravityAlignedAlignment, RejectsMissingExcitationAndTimestampAmbiguity) {
  std::vector<AlignmentCorrespondence> stationary;
  for (std::size_t index = 0; index < 8U; ++index) {
    stationary.push_back(
        AlignmentCorrespondence{core::FusionTime{static_cast<std::int64_t>(index) * kSecondNs},
                                Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Vector3d{5.0, 7.0, 9.0},
                                Eigen::Matrix3d::Identity()});
  }
  const auto degenerate = estimateGravityAlignedTransform(stationary);
  ASSERT_FALSE(degenerate);
  EXPECT_EQ(degenerate.error().code, AlignmentErrorCode::InsufficientHorizontalExcitation);

  stationary[4].stamp = stationary[3].stamp;
  const auto ambiguous = estimateGravityAlignedTransform(stationary);
  ASSERT_FALSE(ambiguous);
  EXPECT_EQ(ambiguous.error().code, AlignmentErrorCode::NonMonotonicTimestamps);
  EXPECT_EQ(ambiguous.error().correspondence_index, 4U);
}

[[nodiscard]] GnssQuality validAutonomousQuality() {
  return GnssQuality(ReceiverReportedSolution::Autonomous, ReceiverIntegrity::Valid,
                     CorrectionStreamState::NotUsed, 12U, 0.8, std::nullopt);
}

[[nodiscard]] GnssHealthSample healthSample(std::uint64_t id, double time_s,
                                            const Eigen::Vector3d& innovation,
                                            GnssQuality quality = validAutonomousQuality()) {
  return GnssHealthSample{core::GnssObservationId{id},
                          core::FusionTime{static_cast<std::int64_t>(time_s * 1.0e9)},
                          std::move(quality), innovation, 0.2 * 0.2 * Eigen::Matrix3d::Identity()};
}

TEST(GnssHealthMonitor, QuarantinesAndRequiresAConsistentBatchForReadmission) {
  GnssHealthConfig config;
  config.consistent_samples_for_admission = 3U;
  config.outage_timeout_s = 1.0;
  config.maximum_reacquisition_sample_gap_s = 0.5;
  const auto created = GnssHealthMonitor::create(config);
  ASSERT_TRUE(created);
  auto monitor = std::move(created).value();

  auto decision = monitor.observe(healthSample(1U, 0.0, Eigen::Vector3d{0.01, 0.0, 0.0}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::QuarantineCurrent);
  decision = monitor.observe(healthSample(2U, 0.2, Eigen::Vector3d{0.0, 0.01, 0.0}));
  ASSERT_TRUE(decision);
  decision = monitor.observe(healthSample(3U, 0.4, Eigen::Vector3d{-0.01, 0.0, 0.01}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().state(), GnssHealthState::Healthy);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::ReleaseConsistentBatch);
  EXPECT_EQ(decision.value().consistentSampleCount(), 3U);
  ASSERT_EQ(decision.value().releasedObservationIds().size(), 3U);
  EXPECT_EQ(decision.value().releasedObservationIds().front(), core::GnssObservationId{1U});

  decision = monitor.observe(healthSample(4U, 0.6, Eigen::Vector3d{0.02, 0.0, 0.0}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::AdmitCurrent);

  const GnssQuality suspect(ReceiverReportedSolution::RtkFixed, ReceiverIntegrity::Suspect,
                            CorrectionStreamState::Current, 18U, 0.5, 0.1);
  decision = monitor.observe(healthSample(5U, 0.8, Eigen::Vector3d::Zero(), suspect));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().state(), GnssHealthState::Quarantined);
  EXPECT_EQ(decision.value().reason(), GnssHealthReason::ReceiverIntegrityNotValid);

  decision = monitor.observe(healthSample(6U, 1.0, Eigen::Vector3d{20.0, 0.0, 0.0}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::QuarantineCurrent);
  decision = monitor.observe(healthSample(7U, 1.2, Eigen::Vector3d{0.01, 0.0, 0.0}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().reason(), GnssHealthReason::ReacquisitionBatchInconsistent);
  decision = monitor.observe(healthSample(8U, 1.4, Eigen::Vector3d{0.0, -0.01, 0.0}));
  ASSERT_TRUE(decision);
  decision = monitor.observe(healthSample(9U, 1.6, Eigen::Vector3d{0.01, 0.01, 0.0}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::ReleaseConsistentBatch);
  ASSERT_EQ(decision.value().releasedObservationIds().size(), 3U);
  EXPECT_EQ(decision.value().releasedObservationIds().front(), core::GnssObservationId{7U});

  decision = monitor.advanceTime(core::FusionTime{3 * kSecondNs});
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().state(), GnssHealthState::Outage);
  EXPECT_EQ(decision.value().reason(), GnssHealthReason::ObservationGap);

  // A stable offset after an outage is accepted as a batch, allowing the
  // backend to re-estimate alignment instead of permanently rejecting GNSS.
  decision = monitor.observe(healthSample(10U, 3.1, Eigen::Vector3d{3.0, -2.0, 0.2}));
  ASSERT_TRUE(decision);
  decision = monitor.observe(healthSample(11U, 3.3, Eigen::Vector3d{3.02, -1.99, 0.19}));
  ASSERT_TRUE(decision);
  decision = monitor.observe(healthSample(12U, 3.5, Eigen::Vector3d{2.99, -2.01, 0.21}));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().state(), GnssHealthState::Healthy);
  EXPECT_EQ(decision.value().disposition(), GnssDisposition::ReleaseConsistentBatch);
}

TEST(GnssQuality, SolutionClassRemainsExplicitReceiverMetadata) {
  const GnssQuality quality(ReceiverReportedSolution::Autonomous, ReceiverIntegrity::Valid,
                            CorrectionStreamState::Current, 15U, 0.7, 0.2);
  EXPECT_EQ(quality.solution(), ReceiverReportedSolution::Autonomous);
  EXPECT_NE(quality.solution(), ReceiverReportedSolution::RtkFixed);
}

}  // namespace
}  // namespace meridian::global
