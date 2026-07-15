#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/map/imap_layer.hpp"
#include "meridian/map/keyframe_store.hpp"

using meridian::CalibrationSet;
using meridian::GraphUpdate;
using meridian::IMapLayer;
using meridian::KeyframePacket;
using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::makeMapLayer;
using meridian::MapBackend;
using meridian::MapConfig;
using meridian::PointCloud;
using meridian::Pose;

namespace {

MapConfig cfg() {
  MapConfig c;
  c.backend = MapBackend::Cpu;
  c.reg_voxel_m = 1.0;
  c.reg_min_plane_pts = 5;
  c.tsdf_voxel_m = 0.2;
  c.tsdf_trunc_voxels = 3;
  c.tsdf_max_integration_dist_m = 100.0;
  return c;
}

Pose at(double x, double y, double z) {
  return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, y, z)};
}

// A horizontal floor patch 2 m below the sensor (body z=-2), ~1.4 m square and dense, so
// the rays form a proper TSDF slab and Marching Cubes yields a surface.
constexpr float kFloorZ = -2.0f;
meridian::PointCloudPtr patch() {
  auto cl = std::make_shared<PointCloud>();
  for (int i = 0; i < 14; ++i) {
    for (int j = 0; j < 14; ++j) {
      LidarPoint p;
      p.xyz = Eigen::Vector3f(-0.6f + 0.092f * static_cast<float>(i),
                              -0.6f + 0.092f * static_cast<float>(j), kFloorZ);
      p.intensity = 50.f;
      cl->push_back(p);
    }
  }
  return cl;
}

KeyframePacket kf(std::uint64_t id, const meridian::PointCloudPtr& cloud) {
  KeyframePacket p;
  p.id = id;
  p.cloud_body = cloud;
  return p;
}

IMapLayer::Ptr build(std::shared_ptr<KeyframeStore> store) {
  return makeMapLayer(cfg(), std::shared_ptr<const CalibrationSet>{}, std::move(store), nullptr);
}

}  // namespace

TEST(LayeredMap, IntegrateAndQuery) {
  auto store = std::make_shared<KeyframeStore>();
  auto m = build(store);
  m->integrate(kf(1, patch()), at(0, 0, 0));
  // Tier R serves the cached plane near the floor patch (map z = kFloorZ).
  auto h = m->query_plane(Eigen::Vector3f(0, 0, kFloorZ));
  ASSERT_TRUE(h.has_value());
  EXPECT_NEAR(std::abs(h->n.z()), 1.0f, 1e-3);
  // The surface backend produced a mesh.
  EXPECT_GT(m->extract_mesh().vertices.size(), 0u);
  EXPECT_EQ(store->size(), 1u);
}

// The clear-and-rebuild invariant (spec 06 §7.4): moving a keyframe and rebuilding yields
// the same map as integrating from scratch at the corrected poses.
TEST(LayeredMap, LoopRebuildEqualsFreshIntegrate) {
  // Three patches; we move the middle one. The far two do not overlap its region.
  auto p1 = patch(), p2 = patch(), p3 = patch();

  auto store_a = std::make_shared<KeyframeStore>();
  auto a = build(store_a);
  a->integrate(kf(1, p1), at(0, 0, 0));
  a->integrate(kf(2, p2), at(5, 0, 0));
  a->integrate(kf(3, p3), at(20, 0, 0));

  // Move keyframe 2 up by 2 m and rebuild.
  GraphUpdate up;
  up.loop_closed = true;
  up.moved.push_back({2, at(5, 0, 2), std::nullopt});
  a->apply_graph_update(up);

  // Reference: a fresh map with keyframe 2 already at the corrected pose.
  auto store_b = std::make_shared<KeyframeStore>();
  auto b = build(store_b);
  b->integrate(kf(1, patch()), at(0, 0, 0));
  b->integrate(kf(2, patch()), at(5, 0, 2));
  b->integrate(kf(3, patch()), at(20, 0, 0));

  const auto da = a->diagnostics();
  const auto db = b->diagnostics();
  EXPECT_EQ(da.tsdf_blocks, db.tsdf_blocks);
  EXPECT_EQ(da.reg_voxels, db.reg_voxels);

  // Keyframe 2's floor moved from map z=kFloorZ to z=kFloorZ+2; old location is empty.
  EXPECT_TRUE(a->query_plane(Eigen::Vector3f(5, 0, kFloorZ + 2.f)).has_value());
  EXPECT_FALSE(a->query_plane(Eigen::Vector3f(5, 0, kFloorZ)).has_value());
}

TEST(LayeredMap, NvbloxBackendFailsFastWhenNotCompiled) {
  MapConfig c = cfg();
  c.backend = MapBackend::Nvblox;
  auto store = std::make_shared<KeyframeStore>();
#ifdef MERIDIAN_MAP_HAS_NVBLOX
  EXPECT_NO_THROW(
      makeMapLayer(c, std::shared_ptr<const CalibrationSet>{}, store, nullptr));
#else
  EXPECT_THROW(makeMapLayer(c, std::shared_ptr<const CalibrationSet>{}, store, nullptr),
               std::invalid_argument);
#endif
}
