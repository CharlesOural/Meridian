#include <gtest/gtest.h>

#include <memory>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/config/config.hpp"
#include "gicp_verify.hpp"

using meridian::GicpVerifier;
using meridian::LidarPoint;
using meridian::PlaceConfig;
using meridian::PointCloud;
using meridian::Pose;
using meridian::VerifiedLoop;

namespace {

// Three orthogonal planes (a room corner) so GICP is fully constrained in all 6 DoF,
// optionally shifted by a known offset.
PointCloud corner(const Eigen::Vector3f& shift) {
  PointCloud c;
  for (int i = 0; i <= 20; ++i) {
    for (int j = 0; j <= 20; ++j) {
      const float u = static_cast<float>(i) * 0.1f;
      const float w = static_cast<float>(j) * 0.1f;
      LidarPoint a, b, d;
      a.xyz = Eigen::Vector3f(u, w, 0.f) + shift;  // z = 0 plane
      b.xyz = Eigen::Vector3f(0.f, u, w) + shift;  // x = 0 plane
      d.xyz = Eigen::Vector3f(u, 0.f, w) + shift;  // y = 0 plane
      c.push_back(a);
      c.push_back(b);
      c.push_back(d);
    }
  }
  return c;
}

PlaceConfig test_cfg() {
  PlaceConfig c;
  c.gicp_downsample = 0.1;
  c.gicp_max_corr_dist = 1.0;
  c.gicp_fitness_min = 0.3;
  c.gicp_overlap_min = 0.3;
  c.gicp_rmse_max = 1.0;
  c.gicp_cond_max = 1e6;
  c.gicp_fit_sigma = 0.3;
  return c;
}

Pose identity() { return Pose{}; }

}  // namespace

TEST(GicpVerify, RecoversKnownTranslation) {
  const Eigen::Vector3f offset(0.15f, 0.10f, 0.05f);
  const PointCloud target = corner(Eigen::Vector3f::Zero());
  const PointCloud source = corner(offset);  // source = target shifted by offset

  GicpVerifier gv(test_cfg(), /*deterministic=*/true);
  const VerifiedLoop v = gv.verify(target, source, identity());

  EXPECT_TRUE(v.accepted);
  EXPECT_GT(v.fitness, 0.3);
  // Mapping source onto target undoes the offset: T_target_source.t ~ -offset.
  EXPECT_NEAR(v.T_from_to.t.x(), -0.15, 0.03);
  EXPECT_NEAR(v.T_from_to.t.y(), -0.10, 0.03);
  EXPECT_NEAR(v.T_from_to.t.z(), -0.05, 0.03);
}

TEST(GicpVerify, NonOverlappingIsRejected) {
  const PointCloud target = corner(Eigen::Vector3f::Zero());
  const PointCloud source = corner(Eigen::Vector3f(50.f, 0.f, 0.f));  // far away

  GicpVerifier gv(test_cfg(), true);
  const VerifiedLoop v = gv.verify(target, source, identity());
  EXPECT_FALSE(v.accepted);
  EXPECT_LT(v.overlap, 0.3);
}

TEST(GicpVerify, DeterministicSingleThread) {
  const PointCloud target = corner(Eigen::Vector3f::Zero());
  const PointCloud source = corner(Eigen::Vector3f(0.12f, 0.08f, 0.04f));

  GicpVerifier gv(test_cfg(), true);
  const VerifiedLoop a = gv.verify(target, source, identity());
  const VerifiedLoop b = gv.verify(target, source, identity());
  EXPECT_DOUBLE_EQ(a.T_from_to.t.x(), b.T_from_to.t.x());
  EXPECT_DOUBLE_EQ(a.T_from_to.t.y(), b.T_from_to.t.y());
  EXPECT_DOUBLE_EQ(a.T_from_to.t.z(), b.T_from_to.t.z());
  EXPECT_DOUBLE_EQ(a.fitness, b.fitness);
  EXPECT_TRUE(a.info_rot_first.isApprox(b.info_rot_first));
}

TEST(GicpVerify, EmptyCloudReturnsUnaccepted) {
  GicpVerifier gv(test_cfg(), true);
  EXPECT_FALSE(gv.verify(PointCloud{}, corner(Eigen::Vector3f::Zero()), identity()).accepted);
  EXPECT_FALSE(gv.verify(corner(Eigen::Vector3f::Zero()), PointCloud{}, identity()).accepted);
}
