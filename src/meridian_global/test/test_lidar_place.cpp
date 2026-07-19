#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <vector>

#include "meridian/global/lidar_place.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::RecordHeader header(std::uint64_t id, std::int64_t time_ns) {
  core::RecordHeader result;
  result.trace = core::TraceId(id);
  result.producer = core::ProducerId(4U);
  result.session = core::SessionId(9U);
  result.created_at = core::FusionTime{time_ns};
  result.config = core::ConfigRevision(3U);
  result.direct_calibration = core::CalibrationEpoch(1U);
  return result;
}

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation = Eigen::Vector3d::Zero(),
                                const Eigen::Vector3d& rotation = Eigen::Vector3d::Zero()) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
}

[[nodiscard]] FinalizedSubmapFrame submap(std::uint64_t id, std::int64_t support_end_ns,
                                          const core::Pose3d& T_odom_submap = {}) {
  core::ContentHash hash{};
  hash[0] = static_cast<std::uint8_t>(id + 1U);
  return FinalizedSubmapFrame{
      core::SubmapRef{core::SessionId(9U), core::OdomEpoch(2U), core::SubmapId(id),
                      core::CalibrationEpoch(1U), core::SubmapContentRevision(1U), hash},
      T_odom_submap, core::FusionTime{support_end_ns}};
}

[[nodiscard]] std::array<std::uint8_t, 32> binaryDescriptor(std::uint64_t seed) {
  const auto mix = [](std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  };
  std::array<std::uint8_t, 32> result{};
  for (std::size_t block = 0U; block < 4U; ++block) {
    const std::uint64_t word = mix(seed + block * 101U);
    for (std::size_t byte = 0U; byte < 8U; ++byte) {
      result[block * 8U + byte] =
          static_cast<std::uint8_t>((word >> static_cast<unsigned int>(byte * 8U)) & 0xffU);
    }
  }
  return result;
}

[[nodiscard]] ImmutableLidarPlaceDescriptor manualDescriptor(
    std::uint64_t id, std::int64_t support_end_ns, std::span<const Eigen::Vector2d> positions,
    std::uint64_t descriptor_seed = 100U) {
  auto result = std::make_shared<LidarPlaceDescriptor>(header(id, support_end_ns),
                                                       submap(id, support_end_ns));
  result->model_revision = LidarDescriptorModelRevision(1U);
  result->config_revision = LidarPlaceConfigRevision(1U);
  result->registration_proxy_checksum[0] = static_cast<std::uint8_t>(id + 1U);
  result->ground_height_submap_m = 0.0;
  for (std::size_t index = 0U; index < positions.size(); ++index) {
    LidarBevFeature feature;
    feature.position_submap_m = positions[index];
    feature.binary_descriptor = binaryDescriptor(descriptor_seed + index);
    feature.response = static_cast<float>(positions.size() - index);
    result->features.push_back(feature);
  }
  result->build.model_revision = result->model_revision;
  result->build.config_revision = result->config_revision;
  result->build.retained_features = result->features.size();
  return std::shared_ptr<const LidarPlaceDescriptor>(std::move(result));
}

[[nodiscard]] std::vector<Eigen::Vector2d> asymmetricFeatures() {
  std::vector<Eigen::Vector2d> result;
  for (std::size_t index = 0U; index < 16U; ++index) {
    const double angle = 0.37 * static_cast<double>(index);
    const double radius = 3.0 + 0.17 * static_cast<double>(index);
    result.emplace_back(radius * std::cos(angle) + 0.13 * index,
                        radius * std::sin(angle) - 0.07 * index * index);
  }
  return result;
}

[[nodiscard]] RegistrationProxy cornerProxy(std::uint8_t checksum) {
  RegistrationProxy proxy;
  proxy.voxel_resolution_m = 0.35;
  proxy.checksum[0] = checksum;
  for (int first = -4; first <= 4; ++first) {
    for (int second = -4; second <= 4; ++second) {
      const double a = 0.45 * static_cast<double>(first);
      const double b = 0.45 * static_cast<double>(second);
      proxy.points.push_back(RegistrationProxyPoint{{a, b, 0.0}, Eigen::Vector3d::UnitZ(), 1.0});
      proxy.points.push_back(
          RegistrationProxyPoint{{-2.2, a, b + 2.2}, Eigen::Vector3d::UnitX(), 1.0});
      proxy.points.push_back(
          RegistrationProxyPoint{{a, -2.2, b + 2.2}, Eigen::Vector3d::UnitY(), 1.0});
    }
  }
  return proxy;
}

[[nodiscard]] RegistrationProxy planeProxy(std::uint8_t checksum, const Eigen::Vector3d& normal) {
  RegistrationProxy proxy;
  proxy.voxel_resolution_m = 0.35;
  proxy.checksum[0] = checksum;
  for (int x = -6; x <= 6; ++x) {
    for (int y = -6; y <= 6; ++y) {
      proxy.points.push_back(RegistrationProxyPoint{
          {0.35 * static_cast<double>(x), 0.35 * static_cast<double>(y), 0.0}, normal, 1.0});
    }
  }
  return proxy;
}

[[nodiscard]] RegistrationProxy expressedInTo(const RegistrationProxy& from,
                                              const core::Pose3d& T_from_to,
                                              std::uint8_t checksum) {
  RegistrationProxy to;
  to.voxel_resolution_m = from.voxel_resolution_m;
  to.checksum[0] = checksum;
  const core::Pose3d T_to_from = T_from_to.inverse();
  const Eigen::Matrix3d R_to_from = T_to_from.so3().matrix();
  to.points.reserve(from.points.size());
  for (const RegistrationProxyPoint& point : from.points) {
    to.points.push_back(RegistrationProxyPoint{T_to_from * point.point_submap,
                                               R_to_from * point.normal_submap, point.weight});
  }
  return to;
}

[[nodiscard]] core::ObservationLineage lineage(std::uint64_t id) {
  core::ObservationSlice slice;
  slice.root = core::MeasurementId(id + 1000U);
  slice.kind = core::SliceKind::Whole;
  slice.calibration = core::CalibrationEpoch(1U);
  slice.source_checksum[0] = static_cast<std::uint8_t>(id);
  core::ObservationUsage usage;
  usage.slice = slice;
  usage.role = core::ObservationRole::PrimaryResidual;
  usage.consumer = core::DerivedRecordId(id);
  usage.factor_group = core::FactorGroupId(id);
  core::ObservationLineage result;
  result.id = core::ObservationLineageId(id);
  result.usage.push_back(usage);
  result.checksum[0] = static_cast<std::uint8_t>(id + 1U);
  return result;
}

[[nodiscard]] LidarLoopVerificationRequest verificationRequest(std::uint64_t id,
                                                               const core::SubmapRef& from,
                                                               const core::SubmapRef& to,
                                                               const RegistrationProxy& from_proxy,
                                                               const RegistrationProxy& to_proxy,
                                                               const core::Pose3d& seed_pose) {
  core::RecordHeader seed_header = header(id, 100);
  LidarRetrievalSeed seed(seed_header, from, to);
  seed.model_revision = LidarDescriptorModelRevision(1U);
  seed.config_revision = LidarPlaceConfigRevision(1U);
  seed.from_proxy_checksum = from_proxy.checksum;
  seed.to_proxy_checksum = to_proxy.checksum;
  seed.T_from_to_seed = seed_pose;
  seed.valid_until = core::FusionTime{300};
  seed.descriptor_matches = 30U;
  seed.ransac_inliers = 25U;
  seed.ransac_inlier_ratio = 0.8;
  seed.ransac_rmse_m = 0.1;
  seed.retrieval_score = 24.0;
  LidarLoopVerificationRequest request(header(id + 100U, 150), ProposalId(id), std::move(seed));
  request.evaluated_at = core::FusionTime{150};
  request.calibration_epochs = {core::CalibrationEpoch(1U)};
  request.lineage = lineage(id);
  return request;
}

[[nodiscard]] LidarLoopVerifierConfig testVerifierConfig() {
  LidarLoopVerifierConfig config;
  config.minimum_proxy_points_per_submap = 60U;
  config.minimum_correspondences_each_direction = 40U;
  config.minimum_bidirectional_overlap = 0.65;
  config.maximum_correspondence_distance_m = 0.8;
  config.maximum_residual_median_m = 0.18;
  config.maximum_residual_quantile_m = 0.40;
  config.translation_convergence_m = 5.0e-4;
  config.rotation_convergence_rad = 5.0e-4;
  return config;
}

TEST(LidarPlaceDescriptor, BuildsDensityValuedGravityAlignedBev) {
  RegistrationProxy proxy;
  proxy.voxel_resolution_m = 0.30;
  proxy.checksum[0] = 7U;
  for (int x = 0; x < 80; ++x) {
    for (int y = 0; y < 80; ++y) {
      const std::uint32_t hash =
          static_cast<std::uint32_t>(x * 73856093U) ^ static_cast<std::uint32_t>(y * 19349663U);
      if (hash % 5U >= 2U) {
        continue;
      }
      const int repetitions = 1 + static_cast<int>((hash >> 5U) % 3U);
      for (int repetition = 0; repetition < repetitions; ++repetition) {
        proxy.points.push_back(RegistrationProxyPoint{
            {0.5 * x + 0.02 * repetition, 0.5 * y, -0.2 + 0.01 * static_cast<double>(hash % 11U)},
            Eigen::Vector3d::UnitZ(),
            1.0});
      }
    }
  }
  LidarPlaceDescriptorConfig config;
  config.self_similarity_hamming_threshold = 0;
  const auto built = buildLidarPlaceDescriptor(header(1U, 0), submap(1U, 0), proxy, config);
  ASSERT_TRUE(built) << built.error().detail;
  EXPECT_EQ(built.value()->model_revision, config.model_revision);
  EXPECT_EQ(built.value()->config_revision, config.config_revision);
  EXPECT_EQ(built.value()->registration_proxy_checksum, proxy.checksum);
  EXPECT_GT(built.value()->build.occupied_bev_cells, 500U);
  EXPECT_GT(built.value()->features.size(), 4U);
  EXPECT_TRUE(std::isfinite(built.value()->ground_height_submap_m));
}

TEST(LidarPlaceIndex, RecoversDeterministicMetricSeedAndReportsTtlTopK) {
  const std::vector<Eigen::Vector2d> to_positions = asymmetricFeatures();
  const double yaw = 0.43;
  Eigen::Matrix2d rotation;
  rotation << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
  const Eigen::Vector2d translation(4.2, -1.7);
  std::vector<Eigen::Vector2d> from_positions;
  from_positions.reserve(to_positions.size());
  std::transform(to_positions.begin(), to_positions.end(), std::back_inserter(from_positions),
                 [&](const Eigen::Vector2d& point) { return rotation * point + translation; });
  const auto candidate = manualDescriptor(1U, 0, from_positions);
  const auto query = manualDescriptor(9U, 20'000'000'000LL, to_positions);
  LidarPlaceIndexConfig config;
  config.minimum_descriptor_matches = 8U;
  config.minimum_ransac_inliers = 6U;
  config.ransac_inlier_distance_m = 0.25;
  config.minimum_ransac_baseline_m = 0.5;
  LidarPlaceIndex index(config);
  ASSERT_TRUE(index.insert(candidate));

  LidarPlaceRetrievalRequest request(query, core::FusionTime{21'000'000'000LL});
  request.top_k = 1U;
  const auto first = index.retrieve(request);
  const auto second = index.retrieve(request);
  ASSERT_TRUE(first) << first.error().detail;
  ASSERT_TRUE(second) << second.error().detail;
  ASSERT_EQ(first.value().seeds.size(), 1U);
  ASSERT_EQ(second.value().seeds.size(), 1U);
  EXPECT_EQ(first.value().report.emitted_top_k, 1U);
  EXPECT_EQ(first.value().seeds.front().valid_until.nanoseconds, 51'000'000'000LL);
  EXPECT_EQ(first.value().seeds.front().ransac_inliers,
            second.value().seeds.front().ransac_inliers);
  const core::Pose3d expected(Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, yaw)),
                              Eigen::Vector3d(translation.x(), translation.y(), 0.0));
  EXPECT_LT((expected.inverse() * first.value().seeds.front().T_from_to_seed).log().norm(), 1.0e-9);

  request.excluded = {lidarPlaceEntryKey(*candidate)};
  const auto excluded = index.retrieve(request);
  ASSERT_TRUE(excluded) << excluded.error().detail;
  EXPECT_TRUE(excluded.value().seeds.empty());
  EXPECT_EQ(excluded.value().report.excluded_explicit_policy, 1U);
}

TEST(LidarPlaceIndex, CapacityFailureIsDeterministicAndNeverEvicts) {
  const std::vector<Eigen::Vector2d> positions = asymmetricFeatures();
  LidarPlaceIndexConfig config;
  config.maximum_entries = 2U;
  config.maximum_total_features = 32U;
  config.maximum_features_per_entry = 16U;
  config.maximum_index_references = 512U;
  LidarPlaceIndex index(config);
  const auto first = manualDescriptor(1U, 0, positions, 10U);
  const auto second = manualDescriptor(2U, 20'000'000'000LL, positions, 20U);
  const auto third = manualDescriptor(3U, 40'000'000'000LL, positions, 30U);
  ASSERT_TRUE(index.insert(first));
  ASSERT_TRUE(index.insert(second));
  const auto duplicate = index.insert(first);
  ASSERT_TRUE(duplicate);
  EXPECT_EQ(duplicate.value().disposition, LidarPlaceIndexInsertDisposition::AlreadyPresent);
  const auto conflicting = index.insert(manualDescriptor(1U, 0, positions, 999U));
  ASSERT_FALSE(conflicting);
  EXPECT_EQ(conflicting.error().code, LidarPlaceIndexErrorCode::IdentityConflict);
  const auto exhausted = index.insert(third);
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, LidarPlaceIndexErrorCode::EntryCapacity);
  EXPECT_EQ(index.size(), 2U);
  EXPECT_EQ(index.featureCount(), 32U);
}

TEST(LidarPlaceIndex, SameSparseIdentityWithChangedImmutableChecksumIsAConflict) {
  const std::vector<Eigen::Vector2d> positions = asymmetricFeatures();
  LidarPlaceIndex index;
  const auto original = manualDescriptor(1U, 0, positions, 10U);
  ASSERT_TRUE(index.insert(original));

  auto altered = std::make_shared<LidarPlaceDescriptor>(*original);
  altered->submap.ref.local_content_checksum[0] ^= 0xffU;
  const auto rejected = index.insert(
      std::shared_ptr<const LidarPlaceDescriptor>(std::move(altered)));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LidarPlaceIndexErrorCode::IdentityConflict);
  EXPECT_EQ(index.size(), 1U);
}

TEST(LidarLoopVerifier, AcceptsLargeOdomDriftBecauseGeometrySeedIsIndependent) {
  const RegistrationProxy from_proxy = cornerProxy(11U);
  const core::Pose3d truth = pose({0.35, -0.22, 0.08}, {0.015, -0.01, 0.24});
  const RegistrationProxy to_proxy = expressedInTo(from_proxy, truth, 12U);
  const FinalizedSubmapFrame from = submap(1U, 0, pose({0.0, 0.0, 0.0}));
  // The current corrected/odom XY hypothesis is intentionally catastrophic;
  // it must not be a verifier gate.
  const FinalizedSubmapFrame to =
      submap(20U, 20'000'000'000LL, pose({500.0, -300.0, 0.0}));
  Eigen::Matrix<double, 6, 1> perturbation;
  perturbation << 0.06, -0.04, 0.02, 0.008, -0.006, 0.025;
  const core::Pose3d seed = truth * core::Pose3d::exp(perturbation);
  const auto verified =
      verifyLidarLoop(verificationRequest(1U, from.ref, to.ref, from_proxy, to_proxy, seed),
                      from_proxy,
                      to_proxy, testVerifierConfig());
  ASSERT_TRUE(verified) << verified.error().detail;
  ASSERT_TRUE(verified.value().measurement.has_value());
  EXPECT_EQ(verified.value().report.disposition, LidarLoopVerificationDisposition::Accepted);
  EXPECT_LT((truth.inverse() * verified.value().measurement->T_from_to).log().norm(), 0.03);
  EXPECT_LE(verified.value().report.final_cost, verified.value().report.initial_cost);
  EXPECT_GE(verified.value().measurement->information.rank, 5U);
  EXPECT_EQ(verified.value().measurement->information.tangent,
            core::PoseTangentConvention::RightTranslationFirst);
  EXPECT_FALSE(verified.value().report.dynamic_fraction.has_value());
}

TEST(LidarLoopVerifier, FalseAliasWithInconsistentNormalsCannotBecomeFactor) {
  const RegistrationProxy from_proxy = planeProxy(21U, Eigen::Vector3d::UnitZ());
  const RegistrationProxy to_proxy = planeProxy(22U, Eigen::Vector3d::UnitX());
  LidarLoopVerifierConfig config = testVerifierConfig();
  config.minimum_pair_normal_cosine = 0.8;
  config.minimum_normal_consistency = 0.8;
  const auto verified =
      verifyLidarLoop(verificationRequest(2U, submap(1U, 0).ref,
                                          submap(30U, 20'000'000'000LL).ref,
                                          from_proxy, to_proxy, core::Pose3d{}),
                      from_proxy, to_proxy, config);
  ASSERT_TRUE(verified) << verified.error().detail;
  EXPECT_FALSE(verified.value().measurement.has_value());
  EXPECT_EQ(verified.value().report.normal_consistent_correspondences, 0U);
  EXPECT_DOUBLE_EQ(verified.value().report.normal_consistency, 0.0);
  EXPECT_EQ(verified.value().report.information.rank, 0U);
}

TEST(LidarLoopVerifier, PlanarGeometryExportsOnlySupportedRank) {
  const RegistrationProxy from_proxy = planeProxy(31U, Eigen::Vector3d::UnitZ());
  const RegistrationProxy to_proxy = planeProxy(32U, Eigen::Vector3d::UnitZ());
  const auto verified =
      verifyLidarLoop(verificationRequest(3U, submap(1U, 0).ref,
                                          submap(40U, 20'000'000'000LL).ref,
                                          from_proxy, to_proxy, core::Pose3d{}),
                      from_proxy, to_proxy, testVerifierConfig());
  ASSERT_TRUE(verified) << verified.error().detail;
  ASSERT_TRUE(verified.value().measurement.has_value());
  const core::RankAwareInformation& information = verified.value().measurement->information;
  EXPECT_EQ(information.rank, 3U);
  EXPECT_GT(information.eigenvalues(2), 0.0);
  EXPECT_DOUBLE_EQ(information.eigenvalues(3), 0.0);
  EXPECT_LT(
      (information.basis.transpose() * information.basis - Eigen::Matrix<double, 6, 6>::Identity())
          .cwiseAbs()
          .maxCoeff(),
      1.0e-10);
}

}  // namespace
}  // namespace meridian::global
