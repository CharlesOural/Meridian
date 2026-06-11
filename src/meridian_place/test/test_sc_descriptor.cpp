#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "sc_descriptor.hpp"

using meridian::LidarPoint;
using meridian::PointCloud;
using meridian::ScanContextDb;
using meridian::ScCandidate;

namespace {

// A scene with 12 vertical bars at distinct azimuths/ranges/heights, rigidly yawed about
// +z. Rotating the scene shifts the polar descriptor's columns, so the recovered shift is
// the yaw.
PointCloud featured_scene(double yaw_rad) {
  PointCloud c;
  for (int k = 0; k < 12; ++k) {
    const double az = static_cast<double>(k) * (30.0 * M_PI / 180.0) + yaw_rad;
    const double r = 5.0 + 0.5 * k;
    const double h = 1.0 + 0.4 * k;
    for (double z = 0.0; z <= h; z += 0.2) {
      LidarPoint p;
      p.xyz = Eigen::Vector3f(static_cast<float>(r * std::cos(az)),
                              static_cast<float>(r * std::sin(az)), static_cast<float>(z));
      c.push_back(p);
    }
  }
  return c;
}

ScanContextDb make_db() { return ScanContextDb(20, 60, 80.0, 10, 0.3); }

}  // namespace

TEST(ScanContextDb, IdenticalSceneMatchesTightly) {
  ScanContextDb db = make_db();
  db.add(0, featured_scene(0.0));
  db.add(1, featured_scene(0.0));
  const auto cands = db.retrieve(1, {0}, 5);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands[0].id, 0u);
  EXPECT_LT(cands[0].sc_dist, 0.02);
  EXPECT_NEAR(cands[0].yaw_guess_rad, 0.0, 0.11);  // within ~one sector
}

TEST(ScanContextDb, RecoversKnownYaw) {
  const double yaw = M_PI / 2.0;  // 90 deg
  ScanContextDb db = make_db();
  db.add(0, featured_scene(0.0));
  db.add(1, featured_scene(yaw));
  const auto cands = db.retrieve(1, {0}, 5);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_EQ(cands[0].id, 0u);
  EXPECT_LT(cands[0].sc_dist, 0.1);
  const double unit = 2.0 * M_PI / 60.0;
  EXPECT_NEAR(cands[0].yaw_guess_rad, yaw, 1.5 * unit);
}

TEST(ScanContextDb, DissimilarSceneRankedWorse) {
  ScanContextDb db = make_db();
  db.add(0, featured_scene(0.0));
  db.add(1, featured_scene(M_PI / 6.0));  // a true 30-deg-yawed revisit
  // A degenerate scene (one bar) is structurally unlike the 12-bar scene.
  PointCloud lone;
  for (double z = 0.0; z <= 3.0; z += 0.1) {
    LidarPoint p;
    p.xyz = Eigen::Vector3f(6.f, 0.f, static_cast<float>(z));
    lone.push_back(p);
  }
  db.add(2, lone);

  const auto good = db.retrieve(1, {0}, 5);
  const auto bad = db.retrieve(2, {0}, 5);
  ASSERT_EQ(good.size(), 1u);
  // The genuine revisit is far closer than the degenerate scene (which may be filtered out).
  const double bad_dist = bad.empty() ? 1.0 : bad[0].sc_dist;
  EXPECT_LT(good[0].sc_dist, bad_dist);
}

TEST(ScanContextDb, EmptyCloudDoesNotCrash) {
  ScanContextDb db = make_db();
  db.add(0, PointCloud{});
  db.add(1, featured_scene(0.0));
  EXPECT_NO_THROW((void)db.retrieve(1, {0}, 5));
  EXPECT_EQ(db.size(), 2u);
}

TEST(ScanContextDb, RetrieveIsDeterministic) {
  ScanContextDb db = make_db();
  db.add(0, featured_scene(0.0));
  db.add(1, featured_scene(0.3));
  db.add(2, featured_scene(0.6));
  db.add(3, featured_scene(0.9));  // query
  const auto a = db.retrieve(3, {0, 1, 2}, 3);
  const auto b = db.retrieve(3, {0, 1, 2}, 3);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, b[i].id);
    EXPECT_DOUBLE_EQ(a[i].sc_dist, b[i].sc_dist);
    EXPECT_DOUBLE_EQ(a[i].yaw_guess_rad, b[i].yaw_guess_rad);
  }
}
