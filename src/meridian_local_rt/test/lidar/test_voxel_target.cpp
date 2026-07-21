#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sophus/se3.hpp>
#include <stdexcept>

#include "meridian/local_rt/lidar/voxel_target.hpp"

namespace meridian::local_rt::lidar {
namespace {

VoxelTargetOptions targetOptions() {
  return VoxelTargetOptions{
      .voxel_size_m = 1.0,
      .retention_radius_m = 100.0,
      .max_voxels = 8U,
      .max_points_per_voxel = 4U,
      .minimum_point_spacing_m = 0.0,
      .max_neighbor_voxel_radius = 4U,
  };
}

bool lexicographicallyLess(const Point3d& lhs, const Point3d& rhs) noexcept {
  if (lhs.x() != rhs.x()) {
    return lhs.x() < rhs.x();
  }
  if (lhs.y() != rhs.y()) {
    return lhs.y() < rhs.y();
  }
  return lhs.z() < rhs.z();
}

std::optional<NearestTargetPoint> bruteForceNearest(std::span<const Point3d> points,
                                                    const Point3d& query, double gate) {
  const double maximum_squared_distance = gate * gate;
  std::optional<NearestTargetPoint> nearest;
  for (const Point3d& point : points) {
    const double squared_distance = (point - query).squaredNorm();
    if (squared_distance > maximum_squared_distance) {
      continue;
    }
    if (!nearest.has_value() || squared_distance < nearest->squared_distance_m2 ||
        (squared_distance == nearest->squared_distance_m2 &&
         lexicographicallyLess(point, nearest->point))) {
      nearest = NearestTargetPoint{point, squared_distance};
    }
  }
  return nearest;
}

void expectNearestMatchesBruteForce(const BoundedVoxelTarget& target,
                                    std::span<const Point3d> stored, const Point3d& query,
                                    double gate) {
  const auto actual = target.nearestNeighbor(query, gate);
  const auto expected = bruteForceNearest(stored, query, gate);
  ASSERT_EQ(actual.has_value(), expected.has_value());
  if (actual.has_value()) {
    EXPECT_TRUE(actual->point.isApprox(expected->point, 0.0));
    EXPECT_DOUBLE_EQ(actual->squared_distance_m2, expected->squared_distance_m2);
  }
}

TEST(VoxelTargetTest, ValidatesEveryMemoryAndSearchBound) {
  auto options = targetOptions();
  options.max_voxels = 0U;
  EXPECT_THROW(static_cast<void>(BoundedVoxelTarget{options}), std::invalid_argument);
  options = targetOptions();
  options.max_points_per_voxel = 0U;
  EXPECT_THROW(static_cast<void>(BoundedVoxelTarget{options}), std::invalid_argument);
  options = targetOptions();
  options.max_neighbor_voxel_radius = 0U;
  EXPECT_THROW(static_cast<void>(BoundedVoxelTarget{options}), std::invalid_argument);
  options = targetOptions();
  options.minimum_point_spacing_m = -0.1;
  EXPECT_THROW(static_cast<void>(BoundedVoxelTarget{options}), std::invalid_argument);
}

TEST(VoxelTargetTest, TransformsPointsAndFindsDeterministicNearestNeighbor) {
  BoundedVoxelTarget target(targetOptions());
  const Sophus::SE3d T_target_source(Eigen::Quaterniond::Identity(), Point3d{1.0, 2.0, 3.0});
  const PointCloud source{Point3d{-0.1, 0.0, 0.0}, Point3d{0.1, 0.0, 0.0}};
  const auto stats = target.update(source, T_target_source, Point3d{1.0, 2.0, 3.0});
  EXPECT_EQ(stats.inserted_points, 2U);

  const auto nearest = target.nearestNeighbor(Point3d{1.0, 2.0, 3.0}, 0.5);
  ASSERT_TRUE(nearest.has_value());
  // Equal-distance candidates choose the lexicographically smaller point.
  EXPECT_TRUE(nearest->point.isApprox(Point3d{0.9, 2.0, 3.0}, 1.0e-15));
  EXPECT_NEAR(nearest->squared_distance_m2, 0.01, 1.0e-15);
  EXPECT_THROW(target.nearestNeighbor(Point3d::Zero(), 5.0), std::invalid_argument);
}

TEST(VoxelTargetTest, PrunedNeighborStencilMatchesBruteForceAtVoxelBoundaries) {
  auto options = targetOptions();
  options.max_voxels = 2048U;
  options.max_points_per_voxel = 1U;
  BoundedVoxelTarget target(options);
  PointCloud points;
  for (int x = -5; x <= 5; ++x) {
    for (int y = -5; y <= 5; ++y) {
      for (int z = -5; z <= 5; ++z) {
        points.emplace_back(static_cast<double>(x) + 0.37, static_cast<double>(y) + 0.61,
                            static_cast<double>(z) + 0.19);
      }
    }
  }
  static_cast<void>(target.updateTargetFrame(points, Point3d::Zero()));
  const PointCloud stored = target.pointCloud();
  const std::array<Point3d, 8> queries{
      Point3d{0.0, 0.0, 0.0},
      Point3d{0.999999, -0.000001, 0.5},
      Point3d{-1.000001, 1.999999, -0.999999},
      Point3d{0.43, -0.27, 0.91},
      Point3d{1.0, -2.0, 3.0},
      Point3d{std::nextafter(1.0, 0.0), std::nextafter(-2.0, 0.0), 0.5},
      Point3d{std::nextafter(1.0, 2.0), std::nextafter(-2.0, -3.0), -0.5},
      Point3d{-3.75, 2.125, -1.875},
  };
  for (const Point3d& query : queries) {
    for (const double gate : {0.03125, 0.5, 0.999999, 1.0, 1.000001, 1.5, 2.75, 4.0}) {
      expectNearestMatchesBruteForce(target, stored, query, gate);
    }
  }
}

TEST(VoxelTargetTest, IncludesExactGateAndPreservesLexicographicTiesAcrossVoxels) {
  auto options = targetOptions();
  options.max_voxels = 32U;
  BoundedVoxelTarget target(options);
  // The positive-side point is in the center-distance-zero stencil group and
  // is visited before the lexicographically smaller point in voxel -2.
  const Point3d query{0.125, 0.25, 0.5};
  const PointCloud points{Point3d{1.375, 0.25, 0.5}, Point3d{-1.125, 0.25, 0.5}};
  static_cast<void>(target.updateTargetFrame(points, Point3d::Zero()));

  const auto nearest = target.nearestNeighbor(query, 1.25);
  ASSERT_TRUE(nearest.has_value());
  EXPECT_TRUE(nearest->point.isApprox(points[1], 0.0));
  EXPECT_DOUBLE_EQ(nearest->squared_distance_m2, 1.5625);
  EXPECT_FALSE(target.nearestNeighbor(query, std::nextafter(1.25, 0.0)).has_value());
}

TEST(VoxelTargetTest, ReportsOnlyBranchAndBoundWorkWhenCountersAreRequested) {
  auto options = targetOptions();
  options.max_voxels = 64U;
  BoundedVoxelTarget target(options);
  const Point3d query{0.5, 0.5, 0.5};
  const PointCloud points{query, Point3d{2.5, 0.5, 0.5}, Point3d{-2.5, 0.5, 0.5},
                          Point3d{0.5, 2.5, 0.5}};
  static_cast<void>(target.updateTargetFrame(points, Point3d::Zero()));

  VoxelTargetQueryStats stats{99U, 99U, 99U};
  const auto nearest = target.nearestNeighbor(query, 4.0, &stats);
  ASSERT_TRUE(nearest.has_value());
  EXPECT_TRUE(nearest->point.isApprox(query, 0.0));
  EXPECT_EQ(stats.voxel_hash_probes, 1U);
  EXPECT_EQ(stats.occupied_buckets, 1U);
  EXPECT_EQ(stats.point_candidates, 1U);

  BoundedVoxelTarget empty(options);
  stats = {99U, 99U, 99U};
  EXPECT_FALSE(empty.nearestNeighbor(query, 4.0, &stats).has_value());
  EXPECT_EQ(stats.voxel_hash_probes, 0U);
  EXPECT_EQ(stats.occupied_buckets, 0U);
  EXPECT_EQ(stats.point_candidates, 0U);
}

TEST(VoxelTargetTest, EnforcesSpacingPerVoxelAndPointCapacity) {
  auto options = targetOptions();
  options.max_points_per_voxel = 2U;
  options.minimum_point_spacing_m = 0.2;
  BoundedVoxelTarget target(options);
  const PointCloud points{Point3d{0.10, 0.0, 0.0}, Point3d{0.15, 0.0, 0.0}, Point3d{0.70, 0.0, 0.0},
                          Point3d{0.90, 0.0, 0.0}};
  const auto stats = target.updateTargetFrame(points, Point3d::Zero());

  EXPECT_EQ(stats.inserted_points, 2U);
  EXPECT_EQ(stats.rejected_spacing_points, 1U);
  EXPECT_EQ(stats.rejected_voxel_capacity_points, 1U);
  EXPECT_EQ(target.voxelCount(), 1U);
  EXPECT_EQ(target.pointCount(), 2U);
}

TEST(VoxelTargetTest, CapacityKeepsSpatiallyClosestVoxelsWithExplicitTies) {
  auto options = targetOptions();
  options.max_voxels = 2U;
  options.max_points_per_voxel = 1U;
  BoundedVoxelTarget target(options);

  const PointCloud first{Point3d{0.1, 0.0, 0.0}, Point3d{1.1, 0.0, 0.0}, Point3d{2.1, 0.0, 0.0}};
  const auto first_stats = target.updateTargetFrame(first, Point3d::Zero());
  EXPECT_EQ(first_stats.inserted_points, 2U);
  EXPECT_EQ(first_stats.rejected_target_capacity_points, 1U);
  EXPECT_EQ(target.voxelCount(), 2U);

  const PointCloud replacement{Point3d{-0.1, 0.0, 0.0}};
  const auto replacement_stats = target.updateTargetFrame(replacement, Point3d::Zero());
  EXPECT_EQ(replacement_stats.evicted_voxels, 1U);
  EXPECT_EQ(replacement_stats.evicted_points, 1U);
  EXPECT_EQ(replacement_stats.inserted_points, 1U);
  EXPECT_LE(target.voxelCount(), options.max_voxels);
  EXPECT_LE(target.pointCount(), options.max_voxels * options.max_points_per_voxel);

  const PointCloud stored = target.pointCloud();
  ASSERT_EQ(stored.size(), 2U);
  EXPECT_TRUE(stored[0].isApprox(Point3d{-0.1, 0.0, 0.0}, 0.0));
  EXPECT_TRUE(stored[1].isApprox(Point3d{0.1, 0.0, 0.0}, 0.0));
}

TEST(VoxelTargetTest, PrunesStoredPointsAndRejectsNewPointsOutsideRetentionRadius) {
  auto options = targetOptions();
  options.retention_radius_m = 2.0;
  BoundedVoxelTarget target(options);
  const PointCloud initial{Point3d{0.0, 0.0, 0.0}, Point3d{3.0, 0.0, 0.0}};
  static_cast<void>(target.updateTargetFrame(initial, Point3d{1.5, 0.0, 0.0}));

  const auto stats = target.updateTargetFrame({}, Point3d::Zero());
  EXPECT_EQ(stats.pruned_voxels, 1U);
  EXPECT_EQ(stats.pruned_points, 1U);
  ASSERT_EQ(target.pointCount(), 1U);
  EXPECT_TRUE(target.pointCloud()[0].isApprox(Point3d::Zero(), 0.0));

  const PointCloud outside{Point3d{5.0, 0.0, 0.0}};
  const auto outside_stats = target.updateTargetFrame(outside, Point3d::Zero());
  EXPECT_EQ(outside_stats.rejected_retention_points, 1U);
  EXPECT_EQ(outside_stats.inserted_points, 0U);
  EXPECT_EQ(target.pointCount(), 1U);
}

}  // namespace
}  // namespace meridian::local_rt::lidar
