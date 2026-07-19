#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "meridian/global/visual_place.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation = Eigen::Vector3d::Zero()) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
}

[[nodiscard]] core::ContentHash hash(std::uint8_t value) {
  core::ContentHash result{};
  result[0] = value;
  result[31] = static_cast<std::uint8_t>(value ^ 0xa5U);
  return result;
}

[[nodiscard]] core::SubmapRef submap(std::uint64_t id, std::uint64_t revision = 1U) {
  return core::SubmapRef{core::SessionId(1U), core::OdomEpoch(3U), core::SubmapId(id),
                         core::CalibrationEpoch(5U), core::SubmapContentRevision(revision),
                         hash(static_cast<std::uint8_t>(id + 1U))};
}

[[nodiscard]] VisualBriskDescriptor descriptor(std::uint64_t seed) {
  VisualBriskDescriptor result{};
  std::uint64_t state = seed + 0x9e3779b97f4a7c15ULL;
  for (std::uint8_t& byte : result) {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    byte = static_cast<std::uint8_t>((state * 0x2545f4914f6cdd1dULL) >> 56U);
  }
  return result;
}

[[nodiscard]] VisualVocabulary vocabulary() {
  VisualVocabulary result;
  result.revision = DescriptorModelRevision(7U);
  result.checksum = hash(42U);
  for (std::uint64_t word = 0U; word < 16U; ++word) {
    result.words.push_back(descriptor(1000U + word));
    result.inverse_document_frequency.push_back(1.0 + 0.05 * static_cast<double>(word));
  }
  return result;
}

[[nodiscard]] VisualRetrievalSubmap retrievalRecord(std::uint64_t id,
                                                    std::uint64_t descriptor_offset,
                                                    std::vector<core::SubmapRef> exclusions = {}) {
  VisualRetrievalSubmap result{submap(id), DescriptorModelRevision(7U),
                               hash(42U),  VisualPlaceConfigRevision(1U),
                               {},         std::move(exclusions)};
  for (std::uint64_t frame = 0U; frame < 3U; ++frame) {
    VisualRetrievalKeyframe keyframe;
    keyframe.id = VisualPlaceKeyframeId(id * 100U + frame);
    keyframe.time = core::FusionTime{static_cast<std::int64_t>(id * 1'000'000U + frame * 1000U)};
    keyframe.covisibility_group = id;
    for (std::uint64_t item = 0U; item < 12U; ++item) {
      // Exact vocabulary words make the expected BoW ranking unambiguous.
      keyframe.descriptors.push_back(
          descriptor(1000U + ((descriptor_offset + frame + item) % 16U)));
    }
    result.keyframes.push_back(std::move(keyframe));
  }
  return result;
}

[[nodiscard]] core::RecordHeader header() {
  core::RecordHeader result;
  result.trace = core::TraceId(8U);
  result.producer = core::ProducerId(2U);
  result.session = core::SessionId(1U);
  result.created_at = core::FusionTime{200};
  result.config = core::ConfigRevision(4U);
  return result;
}

[[nodiscard]] core::ObservationLineage lineage() {
  core::ObservationLineage result;
  result.id = core::ObservationLineageId(9U);
  core::ObservationSlice slice;
  slice.root = core::MeasurementId(77U);
  slice.calibration = core::CalibrationEpoch(5U);
  slice.source_checksum = hash(8U);
  core::ObservationUsage usage;
  usage.slice = slice;
  usage.role = core::ObservationRole::PrimaryResidual;
  usage.consumer = core::DerivedRecordId(3U);
  usage.factor_group = core::FactorGroupId(4U);
  result.usage.push_back(usage);
  result.checksum = hash(9U);
  return result;
}

[[nodiscard]] VisualPlaceSeed seed(const core::SubmapRef& candidate,
                                   const core::SubmapRef& query) {
  VisualPlaceSeed result{candidate,
                         query,
                         DescriptorModelRevision(7U),
                         hash(42U),
                         VisualPlaceConfigRevision(1U),
                         core::FusionTime{100},
                         core::FusionTime{10000},
                         0.0,
                         0U,
                         0U,
                         {},
                         {}};
  result.scheduling_score = 0.91;
  result.query_keyframe_votes = 2U;
  result.candidate_keyframe_votes = 2U;
  result.query_support = {VisualPlaceKeyframeId(1U), VisualPlaceKeyframeId(2U)};
  result.candidate_support = {VisualPlaceKeyframeId(3U), VisualPlaceKeyframeId(4U)};
  return result;
}

[[nodiscard]] VisualVerificationInput verificationInput(bool collapse_pixels = false,
                                                        bool collinear = false) {
  const core::SubmapRef candidate = submap(10U);
  const core::SubmapRef query = submap(80U);
  VisualVerificationInput input{header(),
                                ProposalId(17U),
                                seed(candidate, query),
                                candidate,
                                query,
                                DescriptorModelRevision(7U),
                                DescriptorModelRevision(7U),
                                hash(42U),
                                hash(42U),
                                VisualPlaceConfigRevision(1U),
                                VisualPlaceConfigRevision(1U),
                                {core::CalibrationEpoch(5U)},
                                {},
                                {},
                                lineage()};
  const core::Pose3d T_candidate_query = pose({11.0, -4.0, 1.2}, {0.05, -0.08, 0.72});
  for (std::size_t index = 0U; index < 30U; ++index) {
    const double x = collinear ? 0.02 * static_cast<double>(index)
                               : -0.65 + 0.26 * static_cast<double>(index % 6U);
    const double y = collinear ? 0.0 : -0.42 + 0.21 * static_cast<double>((index / 6U) % 5U);
    const double depth = collinear ? 7.0 : 4.5 + 0.23 * static_cast<double>(index % 7U);
    const Eigen::Vector3d point_camera{x * depth, y * depth, depth};
    const Eigen::Vector3d bearing = point_camera.normalized();
    const Eigen::Vector3d point_candidate = T_candidate_query * point_camera;

    MatureVisualLandmark landmark;
    landmark.id = VisualPlaceLandmarkId(index + 1U);
    landmark.position_candidate_submap = point_candidate;
    landmark.covariance_candidate_submap = Eigen::Matrix3d::Identity() * 1.0e-4;
    landmark.reference_camera_center_candidate_submap =
        T_candidate_query.translation() + Eigen::Vector3d{-1.0, 0.3, 0.0};
    landmark.observation_count = 5U;
    landmark.mature = true;
    landmark.descriptors.push_back(descriptor(50'000U + index));
    input.candidate_landmarks.push_back(landmark);

    QueryVisualFeature feature;
    feature.keyframe = VisualPlaceKeyframeId(900U);
    feature.camera = core::CameraId(0U);
    feature.T_query_submap_camera = core::Pose3d{};
    feature.bearing_camera = bearing;
    feature.image_width = 640U;
    feature.image_height = 480U;
    if (collapse_pixels) {
      feature.pixel = {12.0 + static_cast<double>(index % 2U), 12.0};
    } else {
      feature.pixel = {80.0 + 95.0 * static_cast<double>(index % 6U),
                       50.0 + 90.0 * static_cast<double>((index / 6U) % 5U)};
    }
    feature.descriptor = landmark.descriptors.front();
    input.query_features.push_back(feature);
  }
  return input;
}

[[nodiscard]] double poseError(const core::Pose3d& expected, const core::Pose3d& actual) {
  return (expected.inverse() * actual).log().norm();
}

TEST(VisualPlaceIndex, ExcludesAliasedAdjacentPlaceAndVotesAtSubmapLevelDeterministically) {
  VisualPlaceIndexConfig config;
  config.minimum_frame_score = 0.05;
  config.minimum_query_keyframe_votes = 2U;
  config.minimum_candidate_keyframe_votes = 2U;
  VisualPlaceIndex index(vocabulary(), config);

  // Submap 4 is a perceptual alias but explicitly adjacent to query 5.
  ASSERT_TRUE(index.add(retrievalRecord(4U, 0U, {submap(5U)})));
  ASSERT_TRUE(index.add(retrievalRecord(1U, 0U)));
  const auto query = retrievalRecord(5U, 0U, {submap(4U)});
  const auto first = index.query(query, core::FusionTime{20'000'000});
  const auto second = index.query(query, core::FusionTime{20'000'000});
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  ASSERT_EQ(first.value().candidates.size(), 1U);
  EXPECT_EQ(first.value().candidates.front().candidate.id, core::SubmapId(1U));
  EXPECT_GE(first.value().overlap_or_adjacent_exclusions, 1U);
  EXPECT_GE(first.value().candidates.front().query_keyframe_votes, 2U);
  EXPECT_GE(first.value().candidates.front().candidate_keyframe_votes, 2U);
  EXPECT_EQ(first.value().config_revision, VisualPlaceConfigRevision(1U));
  EXPECT_EQ(first.value().candidates.front().valid_until.nanoseconds, 10'020'000'000LL);
  EXPECT_EQ(first.value().candidates.front().scheduling_score,
            second.value().candidates.front().scheduling_score);
  EXPECT_EQ(first.value().frame_hits.size(), second.value().frame_hits.size());
  EXPECT_EQ(first.value().candidates.front().query_support,
            second.value().candidates.front().query_support);
  EXPECT_EQ(first.value().candidates.front().candidate_support,
            second.value().candidates.front().candidate_support);
}

TEST(VisualPlaceIndex, EnforcesVocabularyChecksumAndCapacityWithoutPartialMutation) {
  VisualPlaceIndexConfig config;
  config.maximum_submaps = 1U;
  VisualPlaceIndex index(vocabulary(), config);
  ASSERT_TRUE(index.add(retrievalRecord(1U, 0U)));
  const auto overflow = index.add(retrievalRecord(2U, 0U));
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, VisualPlaceErrorCode::CapacityExceeded);

  auto mismatched = retrievalRecord(3U, 0U);
  mismatched.model_checksum = hash(99U);
  const auto mismatch = index.query(mismatched, core::FusionTime{1});
  ASSERT_FALSE(mismatch);
  EXPECT_EQ(mismatch.error().code, VisualPlaceErrorCode::ModelMismatch);

  auto stale_config = retrievalRecord(4U, 0U);
  stale_config.config_revision = VisualPlaceConfigRevision(2U);
  const auto config_mismatch = index.query(stale_config, core::FusionTime{1});
  ASSERT_FALSE(config_mismatch);
  EXPECT_EQ(config_mismatch.error().code, VisualPlaceErrorCode::ConfigMismatch);
}

TEST(VisualPlaceIndex, SameSparseIdentityWithChangedChecksumOrCalibrationConflicts) {
  VisualPlaceIndex index(vocabulary());
  const VisualRetrievalSubmap original = retrievalRecord(1U, 0U);
  ASSERT_TRUE(index.add(original));

  auto changed_checksum = original;
  changed_checksum.submap.local_content_checksum = hash(200U);
  auto rejected = index.add(std::move(changed_checksum));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualPlaceErrorCode::DuplicateSubmapConflict);

  auto changed_calibration = original;
  changed_calibration.submap.calibration = core::CalibrationEpoch(6U);
  rejected = index.add(std::move(changed_calibration));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualPlaceErrorCode::DuplicateSubmapConflict);
}

TEST(VisualPlaceIndex, InvalidOfflineVocabularyFailsClosed) {
  auto invalid = vocabulary();
  invalid.checksum = {};
  VisualPlaceIndex index(std::move(invalid));
  const auto result = index.add(retrievalRecord(1U, 0U));
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, VisualPlaceErrorCode::InvalidVocabulary);
}

TEST(VisualGeometricVerifier, AcceptsMetricLoopAcrossLargeDriftWithoutOdometrySeed) {
  auto input = verificationInput();
  VisualGeometricVerifierConfig config;
  config.maximum_information_condition = 1.0e14;
  VisualGeometricVerifier verifier(config);
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  ASSERT_TRUE(result.value().measurement.has_value())
      << static_cast<int>(result.value().report.disposition);
  EXPECT_EQ(result.value().report.disposition, VisualVerificationDisposition::AcceptedMetric3d2d);
  EXPECT_FALSE(result.value().report.used_odometry_seed);
  EXPECT_FALSE(result.value().report.retrieval_score_used_as_information);
  EXPECT_EQ(result.value().measurement->information.tangent,
            core::PoseTangentConvention::RightTranslationFirst);
  EXPECT_EQ(result.value().measurement->information.rank, 6U);
  const core::Pose3d expected = pose({11.0, -4.0, 1.2}, {0.05, -0.08, 0.72});
  EXPECT_LT(poseError(expected, result.value().measurement->T_from_to), 2.0e-2);
  for (int index = 1; index < 6; ++index) {
    EXPECT_GE(result.value().report.hessian_eigenvalues(index - 1),
              result.value().report.hessian_eigenvalues(index));
  }
}

TEST(VisualGeometricVerifier, RejectsPoorImageCoverageAfterIndependentGeometry) {
  auto input = verificationInput(true, false);
  VisualGeometricVerifierConfig config;
  config.maximum_information_condition = 1.0e14;
  VisualGeometricVerifier verifier(config);
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().measurement.has_value());
  EXPECT_EQ(result.value().report.disposition, VisualVerificationDisposition::RejectedGridCoverage);
}

TEST(VisualGeometricVerifier, RejectsPerceptualAliasThatPassesDescriptorMatching) {
  auto input = verificationInput();
  std::vector<VisualBriskDescriptor> shifted;
  for (const auto& landmark : input.candidate_landmarks) {
    shifted.push_back(landmark.descriptors.front());
  }
  std::rotate(shifted.begin(), shifted.begin() + 7, shifted.end());
  for (std::size_t index = 0U; index < input.query_features.size(); ++index) {
    input.query_features[index].descriptor = shifted[index];
  }
  VisualGeometricVerifierConfig config;
  config.maximum_information_condition = 1.0e14;
  VisualGeometricVerifier verifier(config);
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().measurement.has_value());
  EXPECT_NE(result.value().report.disposition, VisualVerificationDisposition::AcceptedMetric3d2d);
  EXPECT_GE(result.value().report.ratio_passes, config.minimum_ransac_inliers);
}

TEST(VisualGeometricVerifier, RejectsDegenerateCollinearLandmarkGeometry) {
  auto input = verificationInput(false, true);
  VisualGeometricVerifierConfig config;
  config.minimum_bearing_spread_rad = 1.0e-4;
  config.minimum_median_parallax_rad = 0.0;
  VisualGeometricVerifier verifier(config);
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  EXPECT_FALSE(result.value().measurement.has_value());
  EXPECT_NE(result.value().report.disposition, VisualVerificationDisposition::AcceptedMetric3d2d);
}

TEST(VisualGeometricVerifier, RejectsStaleContentRevisionEvenWithSameNumericSubmapId) {
  auto input = verificationInput();
  input.candidate_submap = submap(input.seed.candidate.id.value(), 2U);
  VisualGeometricVerifier verifier;
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().report.disposition,
            VisualVerificationDisposition::RejectedEndpointMismatch);
  EXPECT_FALSE(result.value().measurement.has_value());
}

TEST(VisualGeometricVerifier, RejectsChangedImmutableChecksumAtSameRevision) {
  auto input = verificationInput();
  input.candidate_submap.local_content_checksum = hash(254U);
  VisualGeometricVerifier verifier;
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value().report.disposition,
            VisualVerificationDisposition::RejectedEndpointMismatch);
  EXPECT_FALSE(result.value().measurement.has_value());
}

TEST(VisualGeometricVerifier, InvalidRecordHeaderFailsBeforeGeometry) {
  auto input = verificationInput();
  input.header.session = core::SessionId{};
  VisualGeometricVerifier verifier;
  const auto result = verifier.verify(input, core::FusionTime{500});
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, VisualPlaceErrorCode::InvalidVerificationInput);
}

TEST(VisualGeometricVerifier, RejectsEndpointSessionAndCalibrationSetMismatchBeforeGeometry) {
  VisualGeometricVerifier verifier;

  auto wrong_session = verificationInput();
  wrong_session.header.session = core::SessionId(2U);
  auto rejected = verifier.verify(wrong_session, core::FusionTime{500});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualPlaceErrorCode::InvalidVerificationInput);

  auto missing_endpoint_calibration = verificationInput();
  missing_endpoint_calibration.calibration_epochs = {core::CalibrationEpoch(6U)};
  rejected = verifier.verify(missing_endpoint_calibration, core::FusionTime{500});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualPlaceErrorCode::InvalidVerificationInput);
}

TEST(VisualGeometricVerifier, RejectsDifferentRevisionOfSamePhysicalSubmapAsSelfLoop) {
  auto input = verificationInput();
  core::SubmapRef revised = input.candidate_submap;
  revised.content_revision = core::SubmapContentRevision(2U);
  revised.local_content_checksum = hash(201U);
  input.seed.query = revised;
  input.query_submap = revised;

  VisualGeometricVerifier verifier;
  const auto rejected = verifier.verify(input, core::FusionTime{500});
  ASSERT_TRUE(rejected);
  EXPECT_EQ(rejected.value().report.disposition,
            VisualVerificationDisposition::RejectedEndpointMismatch);
  EXPECT_FALSE(rejected.value().measurement.has_value());
}

TEST(VisualGeometricVerifier, RejectsScaleLessMonocularInsteadOfInventingTranslation) {
  VisualGeometricVerifier verifier;
  const auto result = verifier.rejectScaleLessMonocular();
  EXPECT_EQ(result.report.disposition,
            VisualVerificationDisposition::RejectedMonocular2d2dScaleLess);
  EXPECT_FALSE(result.measurement.has_value());
}

}  // namespace
}  // namespace meridian::global
