#include <gtest/gtest.h>

#include <Eigen/Core>
#include <limits>
#include <random>
#include <vector>

#include "lio/voxel_grid_map.hpp"
#include "meridian/config/config.hpp"

using Eigen::Vector3d;
using meridian::FrontendLio;
using meridian::lio::voxelDownsample;
using meridian::lio::VoxelGridMap;

namespace {

bool exactlyEqual(const Vector3d& a, const Vector3d& b) {
  return (a.array() == b.array()).all();
}

std::vector<Vector3d> randomCloud(std::mt19937& rng, std::size_t n, double lo, double hi) {
  std::uniform_real_distribution<double> coord(lo, hi);
  std::vector<Vector3d> pts;
  pts.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    pts.emplace_back(coord(rng), coord(rng), coord(rng));
  }
  return pts;
}

}  // namespace

TEST(VoxelGridMap, ConstructsEmpty) {
  VoxelGridMap map{FrontendLio{}};
  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.voxel_count(), 0u);
  Vector3d neighbor{-1.0, -2.0, -3.0};
  EXPECT_FALSE(map.nearest(Vector3d::Zero(), &neighbor));
  // A miss leaves the output untouched.
  EXPECT_TRUE(exactlyEqual(neighbor, Vector3d{-1.0, -2.0, -3.0}));
}

TEST(VoxelGridMap, InsertRespectsCapAndSpacing) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 2;  // minimum spacing = 1/sqrt(2) ~= 0.707 m
  VoxelGridMap map{cfg};

  map.insert({Vector3d{0.1, 0.5, 0.5}});
  EXPECT_EQ(map.size(), 1u);

  // 0.1 m from the stored point: rejected by the spacing rule.
  map.insert({Vector3d{0.2, 0.5, 0.5}});
  EXPECT_EQ(map.size(), 1u);

  // 0.8 m away: accepted, filling the voxel.
  map.insert({Vector3d{0.9, 0.5, 0.5}});
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.voxel_count(), 1u);

  // ~0.8 m from both stored points (spacing passes), but the voxel is at capacity.
  map.insert({Vector3d{0.5, 0.99, 0.99}});
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.voxel_count(), 1u);

  // A different voxel still accepts points.
  map.insert({Vector3d{1.5, 0.5, 0.5}});
  EXPECT_EQ(map.size(), 3u);
  EXPECT_EQ(map.voxel_count(), 2u);
}

// nearest() must agree with an exhaustive scan of the stored points. Points are
// inserted one at a time so size() deltas reveal which ones the cap/spacing rules
// actually kept; with max_corr_dist_m <= voxel_size_m the 27-cell probe covers every
// point within range, so brute force is the exact oracle.
TEST(VoxelGridMap, NearestMatchesBruteForce) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 20;
  cfg.max_corr_dist_m = 1.0;
  VoxelGridMap map{cfg};

  std::mt19937 rng(42);
  std::vector<Vector3d> stored;
  for (const Vector3d& p : randomCloud(rng, 4000, -10.0, 10.0)) {
    const std::size_t before = map.size();
    map.insert({p});
    if (map.size() > before) {
      stored.push_back(p);
    }
  }
  ASSERT_GT(stored.size(), 500u);

  const double max_sq = cfg.max_corr_dist_m * cfg.max_corr_dist_m;
  int hits = 0;
  int misses = 0;
  auto check = [&](const Vector3d& q) {
    const Vector3d* bf_best = nullptr;
    double bf_sq = std::numeric_limits<double>::infinity();
    for (const Vector3d& p : stored) {
      const double d_sq = (p - q).squaredNorm();
      if (d_sq < bf_sq) {
        bf_sq = d_sq;
        bf_best = &p;
      }
    }
    const bool bf_hit = bf_best != nullptr && bf_sq <= max_sq;
    Vector3d neighbor{1e9, 1e9, 1e9};
    const bool hit = map.nearest(q, &neighbor);
    ASSERT_EQ(hit, bf_hit);
    if (hit) {
      EXPECT_TRUE(exactlyEqual(neighbor, *bf_best));
      ++hits;
    } else {
      EXPECT_TRUE(exactlyEqual(neighbor, Vector3d{1e9, 1e9, 1e9}));
      ++misses;
    }
  };

  // Queries over the cloud's box (mixed hits/misses) plus a far box (all misses).
  for (const Vector3d& q : randomCloud(rng, 500, -10.5, 10.5)) {
    check(q);
  }
  for (const Vector3d& q : randomCloud(rng, 100, 30.0, 40.0)) {
    check(q);
  }
  EXPECT_GT(hits, 200);
  EXPECT_GT(misses, 100);
}

TEST(VoxelGridMap, NegativeCoordinatesUseFloorCells) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 20;
  cfg.max_corr_dist_m = 0.5;
  VoxelGridMap map{cfg};

  // 0.1 m apart across the coordinate planes: distinct cells (-1,-1,-1) and (0,0,0)
  // under floor indexing, so the spacing rule does not apply and both are kept.
  // Truncation toward zero would fold them into one cell and reject the second.
  map.insert({Vector3d{-0.05, -0.05, -0.05}, Vector3d{0.05, 0.05, 0.05}});
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.voxel_count(), 2u);

  // The neighbor probe crosses the planes in both directions.
  Vector3d neighbor;
  ASSERT_TRUE(map.nearest(Vector3d{0.04, 0.04, 0.04}, &neighbor));
  EXPECT_TRUE(exactlyEqual(neighbor, Vector3d{0.05, 0.05, 0.05}));
  ASSERT_TRUE(map.nearest(Vector3d{-0.3, -0.3, -0.3}, &neighbor));
  EXPECT_TRUE(exactlyEqual(neighbor, Vector3d{-0.05, -0.05, -0.05}));
}

TEST(VoxelGridMap, ClipFarFromErasesDistantVoxels) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 20;
  cfg.max_range_m = 10.0;
  cfg.max_corr_dist_m = 0.5;
  VoxelGridMap map{cfg};

  map.insert({Vector3d{0.5, 0.5, 0.5}, Vector3d{2.5, 0.5, 0.5},      // near
              Vector3d{50.5, 0.5, 0.5}, Vector3d{0.5, 30.5, 0.5}});  // far
  ASSERT_EQ(map.size(), 4u);
  ASSERT_EQ(map.voxel_count(), 4u);

  map.clipFarFrom(Vector3d::Zero());
  EXPECT_EQ(map.size(), 2u);
  EXPECT_EQ(map.voxel_count(), 2u);

  Vector3d neighbor;
  EXPECT_TRUE(map.nearest(Vector3d{0.5, 0.5, 0.5}, &neighbor));
  EXPECT_TRUE(map.nearest(Vector3d{2.5, 0.5, 0.5}, &neighbor));
  EXPECT_FALSE(map.nearest(Vector3d{50.5, 0.5, 0.5}, &neighbor));
  EXPECT_FALSE(map.nearest(Vector3d{0.5, 30.5, 0.5}, &neighbor));

  map.clear();
  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.voxel_count(), 0u);
}

TEST(VoxelDownsample, FirstPointPerCellInInputOrder) {
  const Vector3d a{0.2, 0.2, 0.2};
  const Vector3d b{0.8, 0.8, 0.8};  // same cell as a at voxel 1.0
  const Vector3d c{1.2, 0.2, 0.2};  // its own cell
  const Vector3d d{0.4, 0.4, 0.4};  // same cell as a

  const auto out = voxelDownsample({a, b, c, d}, 1.0);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_TRUE(exactlyEqual(out[0], a));
  EXPECT_TRUE(exactlyEqual(out[1], c));

  // Reversing the input changes the survivor: insertion order decides.
  const auto rev = voxelDownsample({b, a}, 1.0);
  ASSERT_EQ(rev.size(), 1u);
  EXPECT_TRUE(exactlyEqual(rev[0], b));

  // A smaller voxel separates a and b.
  EXPECT_EQ(voxelDownsample({a, b}, 0.5).size(), 2u);

  // Points straddling the coordinate planes occupy distinct cells.
  EXPECT_EQ(voxelDownsample({Vector3d{-0.1, -0.1, -0.1}, Vector3d{0.1, 0.1, 0.1}}, 1.0).size(), 2u);
}

TEST(VoxelDownsample, Deterministic) {
  std::mt19937 rng(7);
  const auto cloud = randomCloud(rng, 1500, -8.0, 8.0);

  const auto out1 = voxelDownsample(cloud, 0.7);
  const auto out2 = voxelDownsample(cloud, 0.7);
  ASSERT_EQ(out1.size(), out2.size());
  ASSERT_LT(out1.size(), cloud.size());
  for (std::size_t i = 0; i < out1.size(); ++i) {
    EXPECT_TRUE(exactlyEqual(out1[i], out2[i]));
  }
}

// Two maps fed the same insert/clip/query sequence must produce bit-identical
// results: nothing observable may depend on hash-map iteration order or any other
// run-to-run state.
TEST(VoxelGridMap, IdenticalSequencesProduceIdenticalResults) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 10;
  cfg.max_range_m = 12.0;
  cfg.max_corr_dist_m = 0.5;

  std::mt19937 rng(99);
  std::vector<std::vector<Vector3d>> batches;
  for (int i = 0; i < 5; ++i) {
    batches.push_back(randomCloud(rng, 800, -15.0, 15.0));
  }
  const auto queries = randomCloud(rng, 400, -15.0, 15.0);

  VoxelGridMap map_a{cfg};
  VoxelGridMap map_b{cfg};
  for (const auto& batch : batches) {
    map_a.insert(batch);
    map_b.insert(batch);
    map_a.clipFarFrom(Vector3d::Zero());
    map_b.clipFarFrom(Vector3d::Zero());
  }
  ASSERT_EQ(map_a.size(), map_b.size());
  ASSERT_EQ(map_a.voxel_count(), map_b.voxel_count());

  int hits = 0;
  for (const Vector3d& q : queries) {
    Vector3d na{0.0, 0.0, 0.0};
    Vector3d nb{0.0, 0.0, 0.0};
    const bool hit_a = map_a.nearest(q, &na);
    const bool hit_b = map_b.nearest(q, &nb);
    ASSERT_EQ(hit_a, hit_b);
    EXPECT_TRUE(exactlyEqual(na, nb));
    hits += hit_a ? 1 : 0;
  }
  EXPECT_GT(hits, 0);
}

// neighborsWithin against brute force over the inserted points: same membership for
// radii inside the one-cell probe envelope, and a deterministic output order.
TEST(VoxelGridMap, NeighborsWithinMatchesBruteForceAndIsDeterministic) {
  FrontendLio cfg;
  cfg.voxel_size_m = 1.0;
  cfg.max_points_per_voxel = 20;
  VoxelGridMap map{cfg};

  std::mt19937 rng(7);
  const auto cloud = randomCloud(rng, 600, -4.0, 4.0);
  map.insert(cloud);

  // Replicate the insert filter to know exactly which points the map kept.
  std::vector<Vector3d> kept;
  {
    VoxelGridMap shadow{cfg};
    for (const Vector3d& p : cloud) {
      const std::size_t before = shadow.size();
      shadow.insert({p});
      if (shadow.size() > before) {
        kept.push_back(p);
      }
    }
  }

  const auto queries = randomCloud(rng, 50, -3.0, 3.0);
  const double radius = 0.8;  // inside the 27-cell envelope for voxel 1.0
  std::vector<Vector3d> nbrs;
  std::vector<Vector3d> nbrs_again;
  for (const Vector3d& q : queries) {
    map.neighborsWithin(q, radius, &nbrs);
    std::size_t brute = 0;
    for (const Vector3d& p : kept) {
      if ((p - q).squaredNorm() <= radius * radius) {
        ++brute;
      }
    }
    EXPECT_EQ(nbrs.size(), brute);
    map.neighborsWithin(q, radius, &nbrs_again);
    ASSERT_EQ(nbrs.size(), nbrs_again.size());
    for (std::size_t i = 0; i < nbrs.size(); ++i) {
      EXPECT_TRUE(exactlyEqual(nbrs[i], nbrs_again[i]));
    }
  }
}
