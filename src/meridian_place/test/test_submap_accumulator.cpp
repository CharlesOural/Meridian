#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/map/keyframe_store.hpp"
#include "submap_accumulator.hpp"

using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::PointCloud;
using meridian::Pose;
using meridian::SubmapAccumulator;

namespace {

meridian::PointCloudPtr one_point(float x, float y, float z) {
  auto c = std::make_shared<PointCloud>();
  LidarPoint p;
  p.xyz = Eigen::Vector3f(x, y, z);
  c->push_back(p);
  return c;
}

Pose at_x(double x) { return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, 0, 0)}; }

// A pose lookup backed by the store's snapshots (stands in for the back-end's corrected poses).
SubmapAccumulator::PoseLookup pose_from(std::shared_ptr<const KeyframeStore> s) {
  return [s](std::uint64_t id) { return s->pose(id); };
}

}  // namespace

TEST(SubmapAccumulator, ComposesIntoAnchorFrame) {
  auto store = std::make_shared<KeyframeStore>();
  // Each keyframe holds one point at its own body origin -> world point == pose translation.
  store->put(0, one_point(0, 0, 0), nullptr, at_x(0));
  store->put(1, one_point(0, 0, 0), nullptr, at_x(1));
  store->put(2, one_point(0, 0, 0), nullptr, at_x(2));

  SubmapAccumulator acc(store, pose_from(store), /*window=*/3, /*voxel=*/0.25, /*cache=*/8);
  const auto sm = acc.submap(2);
  ASSERT_TRUE(sm);
  ASSERT_EQ(sm->size(), 3u);
  // In anchor(2) frame, the three world points (0,1,2) become (-2,-1,0), ordered by cell.
  EXPECT_NEAR((*sm)[0].xyz.x(), -2.f, 1e-4f);
  EXPECT_NEAR((*sm)[1].xyz.x(), -1.f, 1e-4f);
  EXPECT_NEAR((*sm)[2].xyz.x(), 0.f, 1e-4f);
}

TEST(SubmapAccumulator, WindowLimitsToLastN) {
  auto store = std::make_shared<KeyframeStore>();
  for (std::uint64_t i = 0; i < 5; ++i) store->put(i, one_point(0, 0, 0), nullptr, at_x(double(i)));

  SubmapAccumulator acc(store, pose_from(store), /*window=*/2, /*voxel=*/0.25, /*cache=*/8);
  const auto sm = acc.submap(4);
  ASSERT_TRUE(sm);
  EXPECT_EQ(sm->size(), 2u);  // only kf3, kf4
  EXPECT_NEAR((*sm)[0].xyz.x(), -1.f, 1e-4f);
  EXPECT_NEAR((*sm)[1].xyz.x(), 0.f, 1e-4f);
}

TEST(SubmapAccumulator, RotationComposesCorrectly) {
  auto store = std::make_shared<KeyframeStore>();
  // anchor at origin yawed +90 deg; a neighbour 1 m ahead in world.
  const Eigen::Quaterniond yaw90(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));
  store->put(0, one_point(0, 0, 0), nullptr, Pose{Eigen::Quaterniond::Identity(),
                                                  Eigen::Vector3d(1, 0, 0)});
  store->put(1, one_point(0, 0, 0), nullptr, Pose{yaw90, Eigen::Vector3d(0, 0, 0)});

  SubmapAccumulator acc(store, pose_from(store), /*window=*/2, /*voxel=*/0.05, /*cache=*/8);
  const auto sm = acc.submap(1);
  ASSERT_TRUE(sm);
  ASSERT_EQ(sm->size(), 2u);
  // kf0 world point (1,0,0) into anchor frame R=+90 about z, t=0: R^T*(1,0,0) = (0,-1,0).
  // kf1's own point is at the anchor origin (0,0,0). Cell order: (0,-1,0) before (0,0,0).
  EXPECT_NEAR((*sm)[0].xyz.x(), 0.f, 1e-4f);
  EXPECT_NEAR((*sm)[0].xyz.y(), -1.f, 1e-4f);
  EXPECT_NEAR((*sm)[1].xyz.norm(), 0.f, 1e-4f);
}

TEST(SubmapAccumulator, InvalidateDropsCachedWindow) {
  auto store = std::make_shared<KeyframeStore>();
  for (std::uint64_t i = 0; i < 3; ++i) store->put(i, one_point(0, 0, 0), nullptr, at_x(double(i)));

  SubmapAccumulator acc(store, pose_from(store), 3, 0.25, 8);
  const auto first = acc.submap(2);
  ASSERT_TRUE(first);
  EXPECT_EQ(acc.submap(2).get(), first.get());  // cached: same handle

  acc.invalidate({1});  // kf1 is inside anchor-2's window -> drop the cache entry
  const auto rebuilt = acc.submap(2);
  ASSERT_TRUE(rebuilt);
  EXPECT_NE(rebuilt.get(), first.get());  // recomposed
  EXPECT_EQ(rebuilt->size(), first->size());

  acc.invalidate({99});  // unrelated id -> no effect
  EXPECT_EQ(acc.submap(2).get(), rebuilt.get());
}

TEST(SubmapAccumulator, MissingAnchorPoseYieldsNull) {
  auto store = std::make_shared<KeyframeStore>();
  store->put(0, one_point(0, 0, 0), nullptr, at_x(0));
  // pose lookup that knows nothing -> anchor has no corrected pose.
  SubmapAccumulator acc(store, [](std::uint64_t) { return std::optional<Pose>{}; }, 3, 0.25, 8);
  EXPECT_EQ(acc.submap(0), nullptr);
}
