#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sophus/so3.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "meridian/local/lidar_composite_target.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::ContentHash checksum(std::uint8_t value) {
  core::ContentHash output{};
  output.fill(value);
  return output;
}

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation) {
  return core::Pose3d{Sophus::SO3d{}, translation};
}

[[nodiscard]] std::shared_ptr<const LidarCompositeTargetPoints> points(
    std::initializer_list<LidarCompositeTargetPoint> rows) {
  return std::make_shared<const LidarCompositeTargetPoints>(rows);
}

[[nodiscard]] LidarCompositeTargetOwnerInput owner(
    LidarCompositeOwnerKind kind, std::uint64_t state, std::uint64_t sweep,
    const Eigen::Vector3d& translation,
    std::shared_ptr<const LidarCompositeTargetPoints> geometry) {
  LidarCompositeTargetOwnerInput output;
  output.kind = kind;
  output.state = core::StateId{state};
  output.time = core::FusionTime{static_cast<std::int64_t>(state * 1'000'000ULL)};
  output.sweep = core::MeasurementId{sweep};
  output.pose_revision = core::LocalGraphRevision{state + 100U};
  output.geometry_checksum = checksum(static_cast<std::uint8_t>((sweep % 250U) + 1U));
  output.T_odom_owner = pose(translation);
  output.points = std::move(geometry);
  return output;
}

struct BruteNeighbor {
  bool found{};
  std::size_t owner_slot{};
  std::size_t local_point_index{};
  std::uint32_t source_index{};
  double distance_squared_m2{std::numeric_limits<double>::infinity()};
};

struct RetainedPointIdentity {
  std::size_t owner_slot{};
  std::uint32_t source_index{};
  std::size_t local_point_index{};

  auto operator<=>(const RetainedPointIdentity&) const = default;
};

using ReferenceVoxelKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;

struct ReferenceCandidate {
  ReferenceVoxelKey key;
  RetainedPointIdentity identity;
  double center_distance_squared_m2{};
  std::uint64_t spatial_rank{};
};

[[nodiscard]] std::uint64_t referenceMix64(std::uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::uint64_t referenceSpatialRank(const ReferenceCandidate& candidate) {
  std::uint64_t rank = 0x636f6d706f736974ULL;
  const auto combine = [&rank](std::uint64_t value) {
    rank = referenceMix64(rank ^ referenceMix64(value + 0x9e3779b97f4a7c15ULL));
  };
  combine(static_cast<std::uint64_t>(std::get<0>(candidate.key)));
  combine(static_cast<std::uint64_t>(std::get<1>(candidate.key)));
  combine(static_cast<std::uint64_t>(std::get<2>(candidate.key)));
  combine(candidate.identity.owner_slot);
  combine(candidate.identity.source_index);
  combine(candidate.identity.local_point_index);
  return rank;
}

[[nodiscard]] std::vector<RetainedPointIdentity> referenceRetainedPoints(
    const LidarCompositeTarget& target) {
  const LidarCompositeTargetConfig& config = target.config();
  std::vector<ReferenceCandidate> candidates;
  for (std::size_t owner_index = 0U; owner_index < target.owners().size(); ++owner_index) {
    const LidarCompositeTargetOwnerInput& target_owner = target.owners()[owner_index];
    for (std::size_t point_index = 0U; point_index < target_owner.points->size(); ++point_index) {
      const LidarCompositeTargetPoint& point = (*target_owner.points)[point_index];
      const Eigen::Vector3d point_odom = target_owner.T_odom_owner * point.point_owner;
      const Eigen::Array3d voxel = (point_odom / config.voxel_size_m).array().floor();
      const ReferenceVoxelKey key{static_cast<std::int64_t>(voxel.x()),
                                  static_cast<std::int64_t>(voxel.y()),
                                  static_cast<std::int64_t>(voxel.z())};
      const Eigen::Vector3d center{
          (static_cast<double>(std::get<0>(key)) + 0.5) * config.voxel_size_m,
          (static_cast<double>(std::get<1>(key)) + 0.5) * config.voxel_size_m,
          (static_cast<double>(std::get<2>(key)) + 0.5) * config.voxel_size_m};
      candidates.push_back(ReferenceCandidate{
          key, RetainedPointIdentity{owner_index, point.source_index, point_index},
          (point_odom - center).squaredNorm(), 0U});
    }
  }
  const auto within_voxel_less = [](const ReferenceCandidate& lhs, const ReferenceCandidate& rhs) {
    return std::tuple{lhs.key, lhs.center_distance_squared_m2, lhs.identity} <
           std::tuple{rhs.key, rhs.center_distance_squared_m2, rhs.identity};
  };
  std::sort(candidates.begin(), candidates.end(), within_voxel_less);
  std::vector<ReferenceCandidate> voxel_bounded;
  for (std::size_t begin = 0U; begin < candidates.size();) {
    std::size_t end = begin + 1U;
    while (end < candidates.size() && candidates[end].key == candidates[begin].key) {
      ++end;
    }
    const std::size_t retained = std::min(config.maximum_points_per_voxel, end - begin);
    voxel_bounded.insert(voxel_bounded.end(),
                         candidates.begin() + static_cast<std::ptrdiff_t>(begin),
                         candidates.begin() + static_cast<std::ptrdiff_t>(begin + retained));
    begin = end;
  }
  if (voxel_bounded.size() > config.maximum_total_points) {
    for (ReferenceCandidate& candidate : voxel_bounded) {
      candidate.spatial_rank = referenceSpatialRank(candidate);
    }
    const auto spatial_less = [](const ReferenceCandidate& lhs, const ReferenceCandidate& rhs) {
      return std::tuple{lhs.spatial_rank, lhs.key, lhs.identity} <
             std::tuple{rhs.spatial_rank, rhs.key, rhs.identity};
    };
    std::nth_element(
        voxel_bounded.begin(),
        voxel_bounded.begin() + static_cast<std::ptrdiff_t>(config.maximum_total_points),
        voxel_bounded.end(), spatial_less);
    voxel_bounded.resize(config.maximum_total_points);
  }
  std::vector<RetainedPointIdentity> identities;
  identities.reserve(voxel_bounded.size());
  for (const ReferenceCandidate& candidate : voxel_bounded) {
    identities.push_back(candidate.identity);
  }
  std::sort(identities.begin(), identities.end());
  return identities;
}

[[nodiscard]] std::vector<RetainedPointIdentity> queriedRetainedPoints(
    const LidarCompositeTarget& target) {
  std::vector<RetainedPointIdentity> identities;
  for (std::size_t owner_index = 0U; owner_index < target.owners().size(); ++owner_index) {
    const LidarCompositeTargetOwnerInput& target_owner = target.owners()[owner_index];
    for (std::size_t point_index = 0U; point_index < target_owner.points->size(); ++point_index) {
      const LidarCompositeTargetPoint& point = (*target_owner.points)[point_index];
      const Eigen::Vector3d point_odom = target_owner.T_odom_owner * point.point_owner;
      const LidarCompositeTargetNeighbor neighbor = target.nearestExact(point_odom, 1.0e-9);
      if (neighbor.found && neighbor.owner_slot == owner_index &&
          neighbor.local_point_index == point_index) {
        identities.push_back(RetainedPointIdentity{owner_index, point.source_index, point_index});
      }
    }
  }
  std::sort(identities.begin(), identities.end());
  return identities;
}

[[nodiscard]] std::uint8_t kindRankForTest(LidarCompositeOwnerKind kind) {
  return kind == LidarCompositeOwnerKind::Live ? 0U : 1U;
}

[[nodiscard]] BruteNeighbor bruteNearest(const LidarCompositeTarget& target,
                                         const Eigen::Vector3d& query, double maximum_distance_m) {
  BruteNeighbor best;
  const double maximum_squared = maximum_distance_m * maximum_distance_m;
  for (std::size_t owner_index = 0U; owner_index < target.owners().size(); ++owner_index) {
    const LidarCompositeTargetOwnerInput& target_owner = target.owners()[owner_index];
    for (std::size_t point_index = 0U; point_index < target_owner.points->size(); ++point_index) {
      const LidarCompositeTargetPoint& point = (*target_owner.points)[point_index];
      const double distance_squared =
          (target_owner.T_odom_owner * point.point_owner - query).squaredNorm();
      if (distance_squared > maximum_squared) {
        continue;
      }
      const bool better_distance = !best.found || distance_squared < best.distance_squared_m2;
      const bool better_tie =
          best.found && distance_squared == best.distance_squared_m2 &&
          std::tuple{kindRankForTest(target_owner.kind), owner_index, point.source_index,
                     point_index} <
              std::tuple{kindRankForTest(target.owners()[best.owner_slot].kind), best.owner_slot,
                         best.source_index, best.local_point_index};
      if (!better_distance && !better_tie) {
        continue;
      }
      best = BruteNeighbor{true, owner_index, point_index, point.source_index, distance_squared};
    }
  }
  return best;
}

TEST(LidarCompositeTarget, ExactQueryMatchesBruteForceAcrossPoseOwnedClouds) {
  const auto live_points =
      points({{{0.0, 0.0, 0.0}, 10U}, {{1.0, 0.0, 0.0}, 11U}, {{0.0, 1.0, 0.0}, 12U}});
  const auto finalized_points = points({{{0.0, 0.0, 0.0}, 20U}, {{0.0, 0.0, 1.0}, 21U}});
  std::vector<LidarCompositeTargetOwnerInput> owners;
  owners.push_back(owner(LidarCompositeOwnerKind::Finalized, 4U, 40U,
                         Eigen::Vector3d{-1.0, 0.0, 0.0}, finalized_points));
  owners.push_back(
      owner(LidarCompositeOwnerKind::Live, 8U, 80U, Eigen::Vector3d{1.0, 0.0, 0.0}, live_points));

  auto created = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, std::move(owners));
  ASSERT_TRUE(created) << created.error().detail;
  const LidarCompositeTarget& target = created.value();
  EXPECT_EQ(target.statistics().input_owners, 2U);
  EXPECT_EQ(target.statistics().retained_points, 5U);

  const std::vector<Eigen::Vector3d> queries{
      {1.08, 0.02, 0.0}, {-1.05, 0.01, 1.0}, {2.0, 0.0, 0.0}, {20.0, 20.0, 20.0}};
  for (const Eigen::Vector3d& query : queries) {
    const BruteNeighbor expected = bruteNearest(target, query, 0.5);
    const LidarCompositeTargetNeighbor actual = target.nearestExact(query, 0.5);
    ASSERT_TRUE(actual.query_valid);
    EXPECT_EQ(actual.voxel_lookups, 27U);
    ASSERT_EQ(actual.found, expected.found);
    if (!actual.found) {
      continue;
    }
    EXPECT_EQ(actual.owner_slot, expected.owner_slot);
    EXPECT_EQ(actual.local_point_index, expected.local_point_index);
    EXPECT_EQ(actual.source_index, expected.source_index);
    EXPECT_DOUBLE_EQ(actual.distance_squared_m2, expected.distance_squared_m2);
    const auto& selected_owner = target.owners()[actual.owner_slot];
    EXPECT_TRUE(
        actual.point_odom.isApprox(selected_owner.T_odom_owner * actual.point_owner, 1.0e-15));
  }
}

TEST(LidarCompositeTarget, QuerySpanningSeveralVoxelsHonorsTheBoundedRadius) {
  LidarCompositeTargetConfig config;
  config.voxel_size_m = 0.25;
  config.maximum_query_distance_m = 0.60;
  config.maximum_voxel_search_radius = 3U;
  auto created = LidarCompositeTarget::create(
      config, {owner(LidarCompositeOwnerKind::Live, 1U, 1U, Eigen::Vector3d::Zero(),
                     points({{{0.0, 0.0, 0.0}, 7U}}))});
  ASSERT_TRUE(created) << created.error().detail;

  const auto neighbor = created.value().nearestExact(Eigen::Vector3d{0.55, 0.0, 0.0}, 0.60);
  ASSERT_TRUE(neighbor.query_valid);
  ASSERT_TRUE(neighbor.found);
  EXPECT_EQ(neighbor.source_index, 7U);
  EXPECT_EQ(neighbor.voxel_lookups, 343U);

  const auto over_limit = created.value().nearestExact(Eigen::Vector3d{0.55, 0.0, 0.0}, 0.61);
  EXPECT_FALSE(over_limit.query_valid);
}

TEST(LidarCompositeTarget, EqualDistanceTiePrefersLiveThenCanonicalOwnerAndPoint) {
  std::vector<LidarCompositeTargetOwnerInput> owners;
  owners.push_back(owner(LidarCompositeOwnerKind::Finalized, 1U, 10U,
                         Eigen::Vector3d{-0.1, 0.0, 0.0}, points({{{0.0, 0.0, 0.0}, 1U}})));
  owners.push_back(owner(LidarCompositeOwnerKind::Live, 9U, 90U, Eigen::Vector3d{0.1, 0.0, 0.0},
                         points({{{0.0, 0.0, 0.0}, 9U}})));
  auto created = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, std::move(owners));
  ASSERT_TRUE(created) << created.error().detail;
  const auto live_tie = created.value().nearestExact(Eigen::Vector3d::Zero(), 0.5);
  ASSERT_TRUE(live_tie.found);
  EXPECT_EQ(created.value().owners()[live_tie.owner_slot].kind, LidarCompositeOwnerKind::Live);

  std::vector<LidarCompositeTargetOwnerInput> live_owners;
  live_owners.push_back(owner(LidarCompositeOwnerKind::Live, 7U, 70U,
                              Eigen::Vector3d{0.1, 0.0, 0.0}, points({{{0.0, 0.0, 0.0}, 70U}})));
  live_owners.push_back(owner(LidarCompositeOwnerKind::Live, 3U, 30U,
                              Eigen::Vector3d{-0.1, 0.0, 0.0}, points({{{0.0, 0.0, 0.0}, 30U}})));
  auto canonical =
      LidarCompositeTarget::create(LidarCompositeTargetConfig{}, std::move(live_owners));
  ASSERT_TRUE(canonical) << canonical.error().detail;
  const auto owner_tie = canonical.value().nearestExact(Eigen::Vector3d::Zero(), 0.5);
  ASSERT_TRUE(owner_tie.found);
  EXPECT_EQ(canonical.value().owners()[owner_tie.owner_slot].state, core::StateId{3U});

  auto same_owner = LidarCompositeTarget::create(
      LidarCompositeTargetConfig{},
      {owner(LidarCompositeOwnerKind::Live, 5U, 50U, Eigen::Vector3d::Zero(),
             points({{{-0.1, 0.0, 0.0}, 100U}, {{0.1, 0.0, 0.0}, 40U}}))});
  ASSERT_TRUE(same_owner) << same_owner.error().detail;
  const auto point_tie = same_owner.value().nearestExact(Eigen::Vector3d::Zero(), 0.5);
  ASSERT_TRUE(point_tie.found);
  EXPECT_EQ(point_tie.source_index, 40U);
  EXPECT_EQ(point_tie.local_point_index, 1U);
}

TEST(LidarCompositeTarget, EnforcesPerVoxelAndTotalPointCapsDeterministically) {
  auto dense = std::make_shared<LidarCompositeTargetPoints>();
  dense->reserve(25U);
  for (std::uint32_t index = 0U; index < 25U; ++index) {
    dense->push_back(LidarCompositeTargetPoint{
        Eigen::Vector3d{0.05 + 0.01 * static_cast<double>(index), 0.1, 0.1}, index});
  }
  LidarCompositeTargetConfig per_voxel_config;
  per_voxel_config.maximum_total_points = 100U;
  auto per_voxel = LidarCompositeTarget::create(
      per_voxel_config,
      {owner(LidarCompositeOwnerKind::Live, 1U, 1U, Eigen::Vector3d::Zero(), dense)});
  ASSERT_TRUE(per_voxel) << per_voxel.error().detail;
  EXPECT_EQ(per_voxel.value().statistics().retained_points, 20U);
  EXPECT_EQ(per_voxel.value().statistics().per_voxel_capacity_discarded_points, 5U);
  const auto dense_query = per_voxel.value().nearestExact(Eigen::Vector3d{0.2, 0.1, 0.1}, 0.5);
  ASSERT_TRUE(dense_query.query_valid);
  EXPECT_EQ(dense_query.points_examined, 20U);

  auto spread = std::make_shared<LidarCompositeTargetPoints>();
  for (std::uint32_t index = 0U; index < 12U; ++index) {
    spread->push_back(LidarCompositeTargetPoint{
        Eigen::Vector3d{2.0 * static_cast<double>(index), 0.0, 0.0}, index + 100U});
  }
  LidarCompositeTargetConfig total_config;
  total_config.maximum_total_points = 7U;
  auto total = LidarCompositeTarget::create(
      total_config,
      {owner(LidarCompositeOwnerKind::Finalized, 2U, 2U, Eigen::Vector3d::Zero(), spread)});
  ASSERT_TRUE(total) << total.error().detail;
  EXPECT_EQ(total.value().statistics().retained_points, 7U);
  EXPECT_EQ(total.value().statistics().total_capacity_discarded_points, 5U);
  EXPECT_LE(total.value().statistics().occupied_voxels, 7U);
}

TEST(LidarCompositeTarget, StreamingCapsRetainTheExactReferencePointIdentities) {
  auto live_geometry = std::make_shared<LidarCompositeTargetPoints>();
  auto finalized_geometry = std::make_shared<LidarCompositeTargetPoints>();
  for (std::uint32_t voxel = 0U; voxel < 12U; ++voxel) {
    for (std::uint32_t row = 0U; row < 5U; ++row) {
      const double base_x = static_cast<double>(voxel) + 0.07 + 0.071 * row;
      live_geometry->push_back(LidarCompositeTargetPoint{
          Eigen::Vector3d{base_x, 0.11 + 0.013 * row, 0.17}, 100U + 10U * voxel + row});
      finalized_geometry->push_back(LidarCompositeTargetPoint{
          Eigen::Vector3d{base_x + 0.009, 0.31 + 0.011 * row, 0.23}, 1'000U + 10U * voxel + row});
    }
  }
  const auto live =
      owner(LidarCompositeOwnerKind::Live, 9U, 90U, Eigen::Vector3d::Zero(), live_geometry);
  const auto finalized = owner(LidarCompositeOwnerKind::Finalized, 3U, 30U, Eigen::Vector3d::Zero(),
                               finalized_geometry);

  LidarCompositeTargetConfig per_voxel_only;
  per_voxel_only.maximum_points_per_voxel = 4U;
  per_voxel_only.maximum_total_points = 100U;
  auto voxel_bounded = LidarCompositeTarget::create(per_voxel_only, {finalized, live});
  ASSERT_TRUE(voxel_bounded) << voxel_bounded.error().detail;
  EXPECT_EQ(voxel_bounded.value().statistics().input_points, 120U);
  EXPECT_EQ(voxel_bounded.value().statistics().retained_points, 48U);
  EXPECT_EQ(voxel_bounded.value().statistics().per_voxel_capacity_discarded_points, 72U);
  EXPECT_EQ(queriedRetainedPoints(voxel_bounded.value()),
            referenceRetainedPoints(voxel_bounded.value()));

  LidarCompositeTargetConfig globally_bounded = per_voxel_only;
  globally_bounded.maximum_total_points = 17U;
  auto forward = LidarCompositeTarget::create(globally_bounded, {finalized, live});
  auto reverse = LidarCompositeTarget::create(globally_bounded, {live, finalized});
  ASSERT_TRUE(forward) << forward.error().detail;
  ASSERT_TRUE(reverse) << reverse.error().detail;
  EXPECT_EQ(forward.value().statistics().retained_points, 17U);
  EXPECT_EQ(forward.value().statistics().total_capacity_discarded_points, 31U);
  EXPECT_EQ(queriedRetainedPoints(forward.value()), referenceRetainedPoints(forward.value()));
  EXPECT_EQ(queriedRetainedPoints(reverse.value()), referenceRetainedPoints(reverse.value()));
  EXPECT_EQ(queriedRetainedPoints(forward.value()), queriedRetainedPoints(reverse.value()));
  EXPECT_EQ(forward.value().checksum(), reverse.value().checksum());
}

TEST(LidarCompositeTarget, ChecksumNamesTheCanonicalOwnerInputRecipe) {
  EXPECT_EQ(kLidarCompositeTargetChecksumSchemaVersion, 3U);
  const auto geometry = points({{{0.1, 0.2, 0.3}, 10U}, {{1.1, 1.2, 1.3}, 11U}});
  auto base_owner =
      owner(LidarCompositeOwnerKind::Live, 5U, 50U, Eigen::Vector3d::Zero(), geometry);
  auto base = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {base_owner});
  ASSERT_TRUE(base) << base.error().detail;

  LidarCompositeTargetConfig revised_config;
  revised_config.maximum_total_points -= 1U;
  auto config_changed = LidarCompositeTarget::create(revised_config, {base_owner});
  ASSERT_TRUE(config_changed) << config_changed.error().detail;
  EXPECT_NE(base.value().checksum(), config_changed.value().checksum());

  auto checksum_owner = base_owner;
  checksum_owner.geometry_checksum = checksum(201U);
  auto geometry_name_changed =
      LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {checksum_owner});
  ASSERT_TRUE(geometry_name_changed) << geometry_name_changed.error().detail;
  EXPECT_NE(base.value().checksum(), geometry_name_changed.value().checksum());

  auto count_owner = base_owner;
  count_owner.points =
      points({{{0.1, 0.2, 0.3}, 10U}, {{1.1, 1.2, 1.3}, 11U}, {{2.1, 2.2, 2.3}, 12U}});
  auto point_count_changed =
      LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {count_owner});
  ASSERT_TRUE(point_count_changed) << point_count_changed.error().detail;
  EXPECT_NE(base.value().checksum(), point_count_changed.value().checksum());

  // Geometry bytes are named transitively. Holding the declared geometry hash
  // and point count fixed therefore preserves the recipe checksum even if a
  // deliberately inconsistent caller supplies different raw bytes.
  auto inconsistent_owner = base_owner;
  inconsistent_owner.points = points({{{10.1, 10.2, 10.3}, 110U}, {{11.1, 11.2, 11.3}, 111U}});
  auto same_declared_recipe =
      LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {inconsistent_owner});
  ASSERT_TRUE(same_declared_recipe) << same_declared_recipe.error().detail;
  EXPECT_EQ(base.value().checksum(), same_declared_recipe.value().checksum());
}

TEST(LidarCompositeTarget, CanonicalBuildIsIndependentOfOwnerInputOrdering) {
  const auto first_points = points({{{0.1, 0.2, 0.3}, 1U}, {{1.1, 0.2, 0.3}, 2U}});
  const auto second_points = points({{{-0.1, 0.2, 0.3}, 3U}, {{-1.1, 0.2, 0.3}, 4U}});
  const auto first =
      owner(LidarCompositeOwnerKind::Live, 4U, 40U, Eigen::Vector3d{1.0, 2.0, 3.0}, first_points);
  const auto second = owner(LidarCompositeOwnerKind::Finalized, 2U, 20U,
                            Eigen::Vector3d{-1.0, -2.0, -3.0}, second_points);
  auto forward = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {first, second});
  auto reverse = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {second, first});
  ASSERT_TRUE(forward) << forward.error().detail;
  ASSERT_TRUE(reverse) << reverse.error().detail;
  EXPECT_EQ(forward.value().checksum(), reverse.value().checksum());
  ASSERT_EQ(forward.value().owners().size(), reverse.value().owners().size());
  for (std::size_t index = 0U; index < forward.value().owners().size(); ++index) {
    EXPECT_EQ(forward.value().owners()[index].state, reverse.value().owners()[index].state);
    EXPECT_EQ(forward.value().owners()[index].sweep, reverse.value().owners()[index].sweep);
  }
  const Eigen::Vector3d query{1.1, 2.2, 3.3};
  const auto forward_neighbor = forward.value().nearestExact(query, 0.5);
  const auto reverse_neighbor = reverse.value().nearestExact(query, 0.5);
  EXPECT_EQ(forward_neighbor.found, reverse_neighbor.found);
  EXPECT_EQ(forward_neighbor.owner_slot, reverse_neighbor.owner_slot);
  EXPECT_EQ(forward_neighbor.local_point_index, reverse_neighbor.local_point_index);
  EXPECT_EQ(forward_neighbor.points_examined, reverse_neighbor.points_examined);
}

TEST(LidarCompositeTarget, RebuildWithRevisedOwnerPoseMovesOnlyTheAccelerationView) {
  const auto geometry = points({{{0.0, 0.0, 0.0}, 7U}});
  auto initial_owner =
      owner(LidarCompositeOwnerKind::Live, 5U, 50U, Eigen::Vector3d::Zero(), geometry);
  auto revised_owner = initial_owner;
  revised_owner.pose_revision = core::LocalGraphRevision{999U};
  revised_owner.T_odom_owner = pose(Eigen::Vector3d{2.0, 0.0, 0.0});

  auto initial = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {initial_owner});
  auto revised = LidarCompositeTarget::create(LidarCompositeTargetConfig{}, {revised_owner});
  ASSERT_TRUE(initial) << initial.error().detail;
  ASSERT_TRUE(revised) << revised.error().detail;
  EXPECT_NE(initial.value().checksum(), revised.value().checksum());
  EXPECT_EQ(initial.value().owners().front().points.get(), geometry.get());
  EXPECT_EQ(revised.value().owners().front().points.get(), geometry.get());
  EXPECT_TRUE(initial.value().owners().front().points->front().point_owner.isZero(0.0));
  EXPECT_TRUE(revised.value().owners().front().points->front().point_owner.isZero(0.0));

  EXPECT_TRUE(initial.value().nearestExact(Eigen::Vector3d::Zero(), 0.5).found);
  EXPECT_FALSE(initial.value().nearestExact(Eigen::Vector3d{2.0, 0.0, 0.0}, 0.5).found);
  EXPECT_FALSE(revised.value().nearestExact(Eigen::Vector3d::Zero(), 0.5).found);
  const auto moved = revised.value().nearestExact(Eigen::Vector3d{2.0, 0.0, 0.0}, 0.5);
  ASSERT_TRUE(moved.found);
  EXPECT_TRUE(moved.point_owner.isZero(0.0));
  EXPECT_TRUE(moved.point_odom.isApprox(Eigen::Vector3d{2.0, 0.0, 0.0}, 0.0));
}

}  // namespace
}  // namespace meridian::local
