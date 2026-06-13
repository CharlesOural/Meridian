#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "voxel_hash_map.hpp"

using meridian::Aabb;
using meridian::LidarPoint;
using meridian::MapConfig;
using meridian::PlaneHit;
using meridian::PointCloud;
using meridian::Pose;
using meridian::map::VoxelHashMap;

namespace {

MapConfig cfg() {
  MapConfig c;
  c.reg_voxel_m = 1.0;
  c.reg_max_pts = 20;
  c.reg_min_plane_pts = 5;
  c.reg_planarity = 0.1;
  c.reg_neighbor_ring = 1;
  c.reg_seed = 0;
  return c;
}

// A flat (z=0) patch of points filling one voxel around (0.5,0.5,0).
meridian::PointCloudPtr plane_patch() {
  auto cl = std::make_shared<PointCloud>();
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      LidarPoint p;
      p.xyz = Eigen::Vector3f(0.1f + 0.18f * static_cast<float>(i),
                              0.1f + 0.22f * static_cast<float>(j), 0.f);
      cl->push_back(p);
    }
  }
  return cl;
}

}  // namespace

TEST(VoxelHashMap, QueryReturnsCachedPlane) {
  VoxelHashMap m(cfg());
  m.integrate_keyframe(1, *plane_patch(), Pose{});
  auto h = m.query_plane(Eigen::Vector3f(0.5f, 0.5f, 0.f));
  ASSERT_TRUE(h.has_value());
  EXPECT_NEAR(std::abs(h->n.z()), 1.0f, 1e-3);   // normal is vertical
  EXPECT_NEAR(std::abs(h->d), 0.0f, 1e-2);       // plane passes through z=0
  EXPECT_GE(h->n_pts, 5);
}

TEST(VoxelHashMap, EmptyRegionReturnsNullopt) {
  VoxelHashMap m(cfg());
  m.integrate_keyframe(1, *plane_patch(), Pose{});
  EXPECT_FALSE(m.query_plane(Eigen::Vector3f(100.f, 100.f, 100.f)).has_value());
}

TEST(VoxelHashMap, NeighbourRingFindsAdjacentPlane) {
  VoxelHashMap m(cfg());
  m.integrate_keyframe(1, *plane_patch(), Pose{});
  // A query in the empty voxel one cell over in +x still finds the planar neighbour.
  auto h = m.query_plane(Eigen::Vector3f(1.5f, 0.5f, 0.f));
  ASSERT_TRUE(h.has_value());
  EXPECT_NEAR(std::abs(h->n.z()), 1.0f, 1e-3);
}

TEST(VoxelHashMap, ClearKeyframeRemovesVoxels) {
  VoxelHashMap m(cfg());
  m.integrate_keyframe(1, *plane_patch(), Pose{});
  EXPECT_GT(m.voxel_count(), 0u);
  m.clear_keyframes({1});
  EXPECT_EQ(m.voxel_count(), 0u);
  EXPECT_FALSE(m.query_plane(Eigen::Vector3f(0.5f, 0.5f, 0.f)).has_value());
}

TEST(VoxelHashMap, HotWindowPrunesFarVoxels) {
  VoxelHashMap m(cfg());
  m.integrate_keyframe(1, *plane_patch(), Pose{});  // near origin
  // Add a far patch.
  auto far = std::make_shared<PointCloud>(*plane_patch());
  Pose shift{Eigen::Quaterniond::Identity(), Eigen::Vector3d(50, 0, 0)};
  m.integrate_keyframe(2, *far, shift);
  EXPECT_GT(m.voxel_count(), 1u);

  Aabb hot;
  hot.min = Eigen::Vector3f(-5, -5, -5);
  hot.max = Eigen::Vector3f(5, 5, 5);
  m.set_hot_window(hot);
  // Near voxel survives, far one pruned.
  EXPECT_TRUE(m.query_plane(Eigen::Vector3f(0.5f, 0.5f, 0.f)).has_value());
  EXPECT_FALSE(m.query_plane(Eigen::Vector3f(50.5f, 0.5f, 0.f)).has_value());
}

TEST(VoxelHashMap, ReservoirEvictionDeterministic) {
  // Overfill one voxel (>cap) and check two same-seed maps agree exactly.
  auto dense = std::make_shared<PointCloud>();
  for (int i = 0; i < 200; ++i) {
    LidarPoint p;
    const float f = static_cast<float>(i) / 200.f;
    p.xyz = Eigen::Vector3f(0.2f + 0.5f * f, 0.3f, 0.01f * f);  // same voxel @ res 1.0
    dense->push_back(p);
  }
  VoxelHashMap a(cfg());
  VoxelHashMap b(cfg());
  a.integrate_keyframe(1, *dense, Pose{});
  b.integrate_keyframe(1, *dense, Pose{});
  const auto ca = a.centroids();
  const auto cb = b.centroids();
  ASSERT_EQ(ca.size(), cb.size());
  ASSERT_EQ(ca.size(), 1u);
  EXPECT_NEAR((ca[0] - cb[0]).norm(), 0.f, 1e-6);  // identical retained sample
}
