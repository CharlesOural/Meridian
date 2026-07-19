#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

#include "meridian/global/gnss_fsm.hpp"

namespace meridian::global {
namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000LL;

[[nodiscard]] core::FusionTime time(double seconds) {
  return core::FusionTime{static_cast<std::int64_t>(seconds * 1.0e9)};
}

[[nodiscard]] GnssQuality goodQuality() {
  return GnssQuality(ReceiverReportedSolution::Autonomous, ReceiverIntegrity::Valid,
                     CorrectionStreamState::NotUsed, 12U, 0.8, std::nullopt);
}

[[nodiscard]] GnssFsmConfig testConfig() {
  GnssFsmConfig config;
  config.alignment.minimum_correspondences = 4U;
  config.alignment.minimum_time_span_s = 0.4;
  config.alignment.minimum_horizontal_rms_m = 0.2;
  config.alignment.minimum_pair_baseline_m = 0.1;
  config.maximum_buffered_observations = 8U;
  config.consecutive_failures_to_suspect = 2U;
  config.consecutive_trusted_accepts_to_clear_failures = 1U;
  config.consecutive_post_solve_accepts = 3U;
  config.outage_timeout = core::Duration{kSecondNs};
  config.maximum_qualified_sample_gap = core::Duration{500'000'000LL};
  return config;
}

[[nodiscard]] Eigen::Vector3d movingPoint(std::size_t index) {
  const double value = static_cast<double>(index);
  return Eigen::Vector3d{0.8 * value, 0.35 * value * value, 0.05 * std::sin(value)};
}

[[nodiscard]] Eigen::Vector3d alignedPoint(const Eigen::Vector3d& map_point) {
  constexpr double kYaw = 0.4;
  const double cosine = std::cos(kYaw);
  const double sine = std::sin(kYaw);
  Eigen::Matrix3d rotation;
  rotation << cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0;
  return rotation * map_point + Eigen::Vector3d{20.0, -7.0, 2.0};
}

[[nodiscard]] GnssFsmObservation sample(
    std::uint64_t id, double seconds, const Eigen::Vector3d& map_point,
    std::optional<double> committed_nis = std::nullopt, GnssQuality quality = goodQuality(),
    Eigen::Matrix3d receiver_covariance = 0.04 * Eigen::Matrix3d::Identity()) {
  const Eigen::Vector3d measured_enu = alignedPoint(map_point);
  core::ContentHash hash{};
  hash[0] = static_cast<std::uint8_t>(id + 1U);
  const core::SubmapRef submap{core::SessionId(1U), core::OdomEpoch(1U),
                               core::SubmapId(id + 100U), core::CalibrationEpoch(1U),
                               core::SubmapContentRevision(1U), hash};
  AlignmentCorrespondence correspondence{time(seconds), map_point, measured_enu,
                                         0.05 * Eigen::Matrix3d::Identity()};
  GnssAntennaConstraint factor{submap, core::GnssObservationId(id), map_point, measured_enu,
                               0.06 * Eigen::Matrix3d::Identity()};
  return GnssFsmObservation(std::move(quality), std::move(receiver_covariance),
                            std::move(correspondence), std::move(factor), committed_nis);
}

[[nodiscard]] GnssShadowValidation acceptedValidation(const GnssShadowRequest& request,
                                                      double completed_seconds, double nis = 0.5) {
  GnssShadowValidation validation;
  validation.attempt = request.attempt;
  validation.completed_at = time(completed_seconds);
  validation.optimizer_converged = true;
  validation.graph_validation_passed = true;
  validation.resource_limits_satisfied = true;
  for (const auto& constraint : request.candidate_batch.constraints) {
    validation.post_solve_nis.push_back(GnssObservationNis{constraint.observation, nis});
  }
  return validation;
}

[[nodiscard]] GnssShadowRequest feedObservableWindow(GnssAdmissionFsm& fsm, std::uint64_t first_id,
                                                     double first_time_s) {
  std::optional<GnssShadowRequest> request;
  for (std::size_t index = 0; index < 4U; ++index) {
    const auto decision = fsm.observe(sample(
        first_id + index, first_time_s + 0.2 * static_cast<double>(index), movingPoint(index)));
    EXPECT_TRUE(decision);
    if (decision && decision.value().shadow_request) {
      request = decision.value().shadow_request;
    }
  }
  EXPECT_TRUE(request.has_value());
  return *request;
}

void establishTrusted(GnssAdmissionFsm& fsm) {
  const GnssShadowRequest request = feedObservableWindow(fsm, 1U, 0.0);
  const auto resolved = fsm.resolveShadow(acceptedValidation(request, 0.7));
  ASSERT_TRUE(resolved);
  ASSERT_EQ(resolved.value().snapshot.state, GnssFsmState::Trusted);
}

TEST(GnssAdmissionFsm, NominalAlignmentRequiresObservableFitAndShadowValidation) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  const GnssShadowRequest request = feedObservableWindow(fsm, 1U, 0.0);
  EXPECT_EQ(fsm.state(), GnssFsmState::Aligning);
  EXPECT_EQ(request.kind, GnssShadowKind::InitialAlignment);
  EXPECT_TRUE(request.source_window.valid());
  EXPECT_TRUE(request.candidate_batch.initial_alignment.has_value());
  EXPECT_EQ(request.candidate_batch.constraints.size(), 4U);
  EXPECT_NEAR(request.alignment_seed.yaw_enu_map_rad, 0.4, 1.0e-8);
  EXPECT_TRUE(
      request.alignment_seed.translation_enu.isApprox(Eigen::Vector3d{20.0, -7.0, 2.0}, 1.0e-8));

  const auto resolved = fsm.resolveShadow(acceptedValidation(request, 0.7));
  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved.value().disposition, GnssFsmDisposition::ReleaseGraphBatch);
  EXPECT_EQ(resolved.value().snapshot.state, GnssFsmState::Trusted);
  EXPECT_TRUE(resolved.value().snapshot.alignment_committed);
  ASSERT_TRUE(resolved.value().released_batch.has_value());
  EXPECT_EQ(resolved.value().released_batch->constraints.size(), 4U);
  ASSERT_FALSE(resolved.value().transitions.empty());
  EXPECT_EQ(resolved.value().transitions.back().to, GnssFsmState::Trusted);

  const auto trusted = fsm.observe(sample(20U, 0.8, movingPoint(5U), 0.3));
  ASSERT_TRUE(trusted);
  EXPECT_EQ(trusted.value().disposition, GnssFsmDisposition::AdmitTrustedCurrent);
  ASSERT_TRUE(trusted.value().released_batch.has_value());
  EXPECT_EQ(trusted.value().released_batch->constraints.size(), 1U);
  const auto& admitted = trusted.value().released_batch->constraints.front();
  EXPECT_EQ(admitted.observation, core::GnssObservationId(20U));
  EXPECT_TRUE(admitted.measured_position_enu.isApprox(alignedPoint(movingPoint(5U)), 1.0e-14));
  EXPECT_TRUE(
      admitted.effective_covariance_enu.isApprox(0.06 * Eigen::Matrix3d::Identity(), 1.0e-14));
}

TEST(GnssAdmissionFsm, InsufficientHorizontalExcitationStaysAligning) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  std::optional<GnssFsmReport> last;
  for (std::size_t index = 0; index < 4U; ++index) {
    const auto decision = fsm.observe(
        sample(30U + index, 0.2 * static_cast<double>(index), Eigen::Vector3d{1.0, 2.0, 0.0}));
    ASSERT_TRUE(decision);
    last = decision.value();
  }
  ASSERT_TRUE(last.has_value());
  EXPECT_EQ(last->snapshot.state, GnssFsmState::Aligning);
  EXPECT_EQ(last->reason, GnssFsmReason::AwaitingAlignmentExcitation);
  ASSERT_TRUE(last->alignment_wait_reason.has_value());
  EXPECT_EQ(*last->alignment_wait_reason, AlignmentErrorCode::InsufficientHorizontalExcitation);
  EXPECT_FALSE(last->shadow_request.has_value());
}

TEST(GnssAdmissionFsm, MultipathNisUsesHysteresisBeforeEnteringSuspect) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();
  establishTrusted(fsm);

  const double gray_nis =
      0.5 * (testConfig().trusted_nis_accept_threshold + testConfig().trusted_nis_reject_threshold);
  auto decision = fsm.observe(sample(40U, 0.8, movingPoint(4U), gray_nis));
  ASSERT_TRUE(decision);
  EXPECT_EQ(decision.value().reason, GnssFsmReason::TrustedNisIndeterminate);
  EXPECT_EQ(decision.value().snapshot.consecutive_failures, 0U);
  EXPECT_EQ(fsm.state(), GnssFsmState::Trusted);

  decision = fsm.observe(sample(41U, 1.0, movingPoint(5U), 100.0));
  ASSERT_TRUE(decision);
  EXPECT_EQ(fsm.state(), GnssFsmState::Trusted);
  EXPECT_EQ(decision.value().snapshot.consecutive_failures, 1U);
  EXPECT_FALSE(decision.value().released_batch.has_value());

  decision = fsm.observe(sample(42U, 1.2, movingPoint(6U), 120.0));
  ASSERT_TRUE(decision);
  EXPECT_EQ(fsm.state(), GnssFsmState::Suspect);
  EXPECT_EQ(decision.value().disposition, GnssFsmDisposition::QuarantinedCurrent);
  EXPECT_TRUE(decision.value().snapshot.quarantine_active);
  EXPECT_FALSE(decision.value().released_batch.has_value());
}

TEST(GnssAdmissionFsm, OutagePreservesAlignmentAndReturnsThroughReacquiring) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();
  establishTrusted(fsm);

  const auto outage = fsm.advanceTime(time(1.7));
  ASSERT_TRUE(outage);
  EXPECT_EQ(fsm.state(), GnssFsmState::Unavailable);
  EXPECT_TRUE(outage.value().snapshot.alignment_committed);
  EXPECT_EQ(outage.value().reason, GnssFsmReason::StreamTimedOut);

  const auto returned = fsm.observe(sample(50U, 1.8, movingPoint(0U)));
  ASSERT_TRUE(returned);
  EXPECT_EQ(fsm.state(), GnssFsmState::Reacquiring);
  EXPECT_EQ(returned.value().disposition, GnssFsmDisposition::QuarantinedCurrent);
  ASSERT_FALSE(returned.value().transitions.empty());
  EXPECT_EQ(returned.value().transitions.front().reason,
            GnssFsmReason::StreamReturnedWithAlignment);
}

TEST(GnssAdmissionFsm, BadFirstSampleAfterOutageReturnsToSuspectWithoutRelease) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();
  establishTrusted(fsm);
  ASSERT_TRUE(fsm.advanceTime(time(1.7)));

  const GnssQuality invalid_integrity(ReceiverReportedSolution::RtkFixed,
                                      ReceiverIntegrity::Suspect, CorrectionStreamState::Current,
                                      18U, 0.5, 0.1);
  const auto returned =
      fsm.observe(sample(60U, 1.8, movingPoint(0U), std::nullopt, invalid_integrity));
  ASSERT_TRUE(returned);
  EXPECT_EQ(fsm.state(), GnssFsmState::Suspect);
  EXPECT_EQ(returned.value().reason, GnssFsmReason::ReceiverIntegrityRejected);
  EXPECT_EQ(returned.value().disposition, GnssFsmDisposition::QuarantinedCurrent);
  EXPECT_EQ(returned.value().snapshot.buffered_observations, 0U);
  EXPECT_FALSE(returned.value().released_batch.has_value());
}

TEST(GnssAdmissionFsm, SuccessfulReacquisitionReleasesOneAtomicBatch) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();
  establishTrusted(fsm);
  ASSERT_TRUE(fsm.advanceTime(time(1.7)));

  const GnssShadowRequest request = feedObservableWindow(fsm, 70U, 1.8);
  EXPECT_EQ(request.kind, GnssShadowKind::Reacquisition);
  EXPECT_FALSE(request.candidate_batch.initial_alignment.has_value());
  EXPECT_EQ(fsm.state(), GnssFsmState::Reacquiring);

  const auto resolved = fsm.resolveShadow(acceptedValidation(request, 2.5));
  ASSERT_TRUE(resolved);
  EXPECT_EQ(fsm.state(), GnssFsmState::Trusted);
  EXPECT_EQ(resolved.value().disposition, GnssFsmDisposition::ReleaseGraphBatch);
  ASSERT_TRUE(resolved.value().released_batch.has_value());
  EXPECT_EQ(resolved.value().released_batch->constraints.size(), 4U);
  EXPECT_FALSE(resolved.value().released_batch->initial_alignment.has_value());
  EXPECT_FALSE(resolved.value().snapshot.quarantine_active);
}

TEST(GnssAdmissionFsm, FailedShadowReacquisitionDoesNotDisableFutureAttempts) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();
  establishTrusted(fsm);
  ASSERT_TRUE(fsm.advanceTime(time(1.7)));

  const GnssShadowRequest first_request = feedObservableWindow(fsm, 80U, 1.8);
  auto rejected_validation = acceptedValidation(first_request, 2.5);
  rejected_validation.post_solve_nis.front().nis = 100.0;
  const auto rejected = fsm.resolveShadow(rejected_validation);
  ASSERT_TRUE(rejected);
  EXPECT_EQ(fsm.state(), GnssFsmState::Suspect);
  EXPECT_EQ(rejected.value().reason, GnssFsmReason::ShadowPostSolveNisRejected);
  EXPECT_FALSE(rejected.value().released_batch.has_value());

  const GnssShadowRequest second_request = feedObservableWindow(fsm, 90U, 2.7);
  const auto recovered = fsm.resolveShadow(acceptedValidation(second_request, 3.4));
  ASSERT_TRUE(recovered);
  EXPECT_EQ(fsm.state(), GnssFsmState::Trusted);
  EXPECT_TRUE(recovered.value().released_batch.has_value());
}

TEST(GnssAdmissionFsm, ReceiverMetadataAndCovarianceAreRealAdmissionGates) {
  GnssFsmConfig config = testConfig();
  const auto created = GnssAdmissionFsm::create(config);
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  const GnssQuality missing_satellites(ReceiverReportedSolution::Autonomous,
                                       ReceiverIntegrity::Valid, CorrectionStreamState::NotUsed,
                                       std::nullopt, 0.8, std::nullopt);
  auto rejected = fsm.observe(sample(100U, 0.0, movingPoint(0U), std::nullopt, missing_satellites));
  ASSERT_TRUE(rejected);
  EXPECT_EQ(rejected.value().reason, GnssFsmReason::ReceiverMetadataUnavailable);
  EXPECT_EQ(rejected.value().snapshot.buffered_observations, 0U);

  Eigen::Matrix3d poor_covariance = Eigen::Matrix3d::Identity();
  poor_covariance(0, 0) = 100.0;
  rejected =
      fsm.observe(sample(101U, 0.2, movingPoint(1U), std::nullopt, goodQuality(), poor_covariance));
  ASSERT_TRUE(rejected);
  EXPECT_EQ(rejected.value().reason, GnssFsmReason::ReceiverCovarianceRejected);

  const GnssQuality stale_rtk(ReceiverReportedSolution::RtkFixed, ReceiverIntegrity::Valid,
                              CorrectionStreamState::Stale, 18U, 0.5, 0.2);
  rejected = fsm.observe(sample(102U, 0.4, movingPoint(2U), std::nullopt, stale_rtk));
  ASSERT_TRUE(rejected);
  EXPECT_EQ(rejected.value().reason, GnssFsmReason::ReceiverCorrectionRejected);
}

TEST(GnssAdmissionFsm, HardCapacityAndTimeOrderViolationsAreObservableErrors) {
  GnssFsmConfig config = testConfig();
  config.maximum_buffered_observations = 4U;
  const auto created = GnssAdmissionFsm::create(config);
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  for (std::size_t index = 0; index < 4U; ++index) {
    ASSERT_TRUE(fsm.observe(
        sample(110U + index, 0.2 * static_cast<double>(index), Eigen::Vector3d{1.0, 2.0, 0.0})));
  }
  const auto overflow = fsm.observe(sample(114U, 0.8, Eigen::Vector3d{1.0, 2.0, 0.0}));
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, GnssFsmErrorCode::CapacityExceeded);
  EXPECT_TRUE(overflow.error().uncommitted_window_cleared);
  EXPECT_EQ(fsm.snapshot().buffered_observations, 0U);
  EXPECT_EQ(fsm.snapshot().capacity_resets, 1U);

  ASSERT_TRUE(fsm.observe(sample(115U, 1.0, movingPoint(0U))));
  const auto repeated = fsm.observe(sample(116U, 1.0, movingPoint(1U)));
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, GnssFsmErrorCode::NonMonotonicTime);
  const auto backwards = fsm.advanceTime(time(0.9));
  ASSERT_FALSE(backwards);
  EXPECT_EQ(backwards.error().code, GnssFsmErrorCode::NonMonotonicTime);
}

TEST(GnssAdmissionFsm, UninitializedStreamTimeoutAndReturnUseSpecifiedPaths) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  const auto timeout = fsm.advanceTime(time(5.0));
  ASSERT_TRUE(timeout);
  EXPECT_EQ(fsm.state(), GnssFsmState::Unavailable);
  EXPECT_FALSE(timeout.value().snapshot.alignment_committed);

  const auto returned = fsm.observe(sample(120U, 5.1, movingPoint(0U)));
  ASSERT_TRUE(returned);
  EXPECT_EQ(fsm.state(), GnssFsmState::Aligning);
  ASSERT_FALSE(returned.value().transitions.empty());
  EXPECT_EQ(returned.value().transitions.front().reason,
            GnssFsmReason::StreamReturnedWithoutAlignment);
}

TEST(GnssAdmissionFsm, RejectsIncompleteCoreSubmapIdentityBeforeBuffering) {
  const auto created = GnssAdmissionFsm::create(testConfig());
  ASSERT_TRUE(created);
  auto fsm = std::move(created).value();

  auto missing_checksum = sample(130U, 0.0, movingPoint(0U));
  missing_checksum.graph_constraint.submap.local_content_checksum = {};
  auto rejected = fsm.observe(missing_checksum);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GnssFsmErrorCode::InvalidObservationIdentity);
  EXPECT_EQ(fsm.snapshot().buffered_observations, 0U);

  auto missing_calibration = sample(131U, 0.2, movingPoint(1U));
  missing_calibration.graph_constraint.submap.calibration = core::CalibrationEpoch{};
  rejected = fsm.observe(missing_calibration);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GnssFsmErrorCode::InvalidObservationIdentity);
  EXPECT_EQ(fsm.snapshot().buffered_observations, 0U);
}

}  // namespace
}  // namespace meridian::global
