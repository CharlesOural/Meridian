#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/map/keyframe_store.hpp"

using meridian::Aabb;
using meridian::CameraFrame;
using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::PointCloud;
using meridian::Pose;

namespace {

meridian::PointCloudPtr make_cloud(std::size_t n, float x) {
  auto c = std::make_shared<PointCloud>();
  c->resize(n);
  for (auto& p : *c) p.xyz = Eigen::Vector3f(x, 0.f, 0.f);
  return c;
}

Pose at(double x, double y, double z) {
  return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, y, z)};
}

}  // namespace

TEST(KeyframeStore, PutGetRoundTrip) {
  KeyframeStore s;
  EXPECT_EQ(s.size(), 0u);
  EXPECT_FALSE(s.contains(7));

  auto cloud = make_cloud(10, 5.f);
  s.put(7, cloud, nullptr, at(1, 2, 3));

  EXPECT_EQ(s.size(), 1u);
  EXPECT_TRUE(s.contains(7));
  ASSERT_TRUE(s.cloud(7));
  EXPECT_EQ(s.cloud(7)->size(), 10u);
  EXPECT_EQ(s.cloud(7).get(), cloud.get());  // same handle, no copy
  EXPECT_EQ(s.image(7), nullptr);
  ASSERT_TRUE(s.pose(7).has_value());
  EXPECT_DOUBLE_EQ(s.pose(7)->t.x(), 1.0);
  EXPECT_DOUBLE_EQ(s.pose(7)->t.z(), 3.0);
}

TEST(KeyframeStore, MissingIdReturnsEmpty) {
  KeyframeStore s;
  s.put(1, make_cloud(3, 0.f), nullptr, at(0, 0, 0));
  EXPECT_EQ(s.cloud(99), nullptr);
  EXPECT_EQ(s.image(99), nullptr);
  EXPECT_FALSE(s.pose(99).has_value());
  EXPECT_FALSE(s.bounds(99).has_value());
}

TEST(KeyframeStore, WithinRadiusFindsNearbyAscending) {
  KeyframeStore s;
  s.put(3, make_cloud(1, 0.f), nullptr, at(0, 0, 0));
  s.put(1, make_cloud(1, 0.f), nullptr, at(1, 0, 0));
  s.put(2, make_cloud(1, 0.f), nullptr, at(10, 0, 0));  // far

  const auto near = s.within_radius(Eigen::Vector3f(0, 0, 0), 2.0f);
  ASSERT_EQ(near.size(), 2u);
  EXPECT_EQ(near[0], 1u);  // ascending id, both within 2 m
  EXPECT_EQ(near[1], 3u);

  EXPECT_TRUE(s.within_radius(Eigen::Vector3f(100, 0, 0), 1.0f).empty());
}

TEST(KeyframeStore, PutOverwritesSameId) {
  KeyframeStore s;
  s.put(5, make_cloud(2, 0.f), nullptr, at(0, 0, 0));
  s.put(5, make_cloud(4, 0.f), nullptr, at(9, 9, 9));
  EXPECT_EQ(s.size(), 1u);
  EXPECT_EQ(s.cloud(5)->size(), 4u);
  EXPECT_DOUBLE_EQ(s.pose(5)->t.x(), 9.0);
}

TEST(KeyframeStore, BoundsTrackPose) {
  KeyframeStore s;
  // A cloud of points at body x=5; placed at map origin -> bounds around (5,0,0).
  s.put(1, make_cloud(8, 5.f), nullptr, at(0, 0, 0));
  auto b = s.bounds(1);
  ASSERT_TRUE(b.has_value());
  EXPECT_NEAR(b->min.x(), 5.f, 1e-4);
  EXPECT_NEAR(b->max.x(), 5.f, 1e-4);
}

TEST(KeyframeStore, UpdatePoseMovesBounds) {
  KeyframeStore s;
  s.put(1, make_cloud(8, 5.f), nullptr, at(0, 0, 0));
  s.update_pose(1, at(100, 0, 0));
  EXPECT_DOUBLE_EQ(s.pose(1)->t.x(), 100.0);
  auto b = s.bounds(1);
  ASSERT_TRUE(b.has_value());
  EXPECT_NEAR(b->min.x(), 105.f, 1e-3);  // body x=5 shifted by +100
}

TEST(KeyframeStore, InRegionFindsOverlapping) {
  KeyframeStore s;
  s.put(1, make_cloud(4, 0.f), nullptr, at(0, 0, 0));    // bounds ~ origin
  s.put(2, make_cloud(4, 0.f), nullptr, at(50, 0, 0));   // bounds ~ (50,0,0)
  s.put(3, make_cloud(4, 0.f), nullptr, at(2, 0, 0));    // bounds ~ (2,0,0)

  Aabb region;
  region.min = Eigen::Vector3f(-1, -1, -1);
  region.max = Eigen::Vector3f(3, 1, 1);
  const auto hit = s.in_region(region);
  ASSERT_EQ(hit.size(), 2u);
  EXPECT_EQ(hit[0], 1u);  // ascending
  EXPECT_EQ(hit[1], 3u);
}
