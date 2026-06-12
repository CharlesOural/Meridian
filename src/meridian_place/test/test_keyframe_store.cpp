#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/place/keyframe_store.hpp"

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
