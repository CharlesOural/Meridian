#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/config/config.hpp"
#include "meridian/place/iloop_detector.hpp"
#include "meridian/map/keyframe_store.hpp"

using meridian::KeyframePoseSource;
using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::LoopConstraint;
using meridian::makeLoopDetector;
using meridian::PlaceConfig;
using meridian::PointCloud;
using meridian::PointCloudPtr;
using meridian::Pose;
using meridian::Timestamp;

namespace {

PointCloud world_scene() {
  PointCloud c;
  for (int i = 0; i <= 30; ++i) {
    for (int j = 0; j <= 30; ++j) {
      const float u = static_cast<float>(i) * 0.4f;
      const float w = static_cast<float>(j) * 0.4f;
      LidarPoint a, b, d;
      a.xyz = Eigen::Vector3f(u, w, 0.f);
      b.xyz = Eigen::Vector3f(0.f, u, w);
      d.xyz = Eigen::Vector3f(u, 0.f, w);
      c.push_back(a);
      c.push_back(b);
      c.push_back(d);
    }
  }
  const float pillars[6][3] = {{2, 9, 4}, {9, 2, 3}, {10, 10, 5},
                               {3, 4, 2}, {11, 6, 4}, {6, 11, 5}};
  for (const auto& p : pillars) {
    for (float z = 0.f; z <= p[2]; z += 0.2f) {
      LidarPoint q;
      q.xyz = Eigen::Vector3f(p[0], p[1], z);
      c.push_back(q);
    }
  }
  return c;
}

PointCloudPtr observe(const PointCloud& scene, const Pose& T) {
  const Pose inv = T.inverse();
  auto out = std::make_shared<PointCloud>();
  out->reserve(scene.size());
  for (const LidarPoint& p : scene) {
    LidarPoint q = p;
    q.xyz = (inv * p.xyz.cast<double>()).cast<float>();
    out->push_back(q);
  }
  return out;
}

Pose at(double x, double y, double z) {
  return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, y, z)};
}

PlaceConfig cfg() {
  PlaceConfig c;
  c.enable = true;
  c.min_kf_gap = 3;
  c.min_time_gap = 0.0;
  c.cooldown_kf = 0;
  c.sc_max_xy = 100.0;
  c.sc_dist_thresh = 0.05;
  c.submap_window = 1;
  c.submap_voxel = 0.1;
  c.gicp_downsample = 0.2;
  c.gicp_fitness_min = 0.3;
  c.gicp_overlap_min = 0.3;
  c.gicp_rmse_max = 1.0;
  c.gicp_cond_max = 1e8;
  c.gicp_fit_sigma = 0.3;
  return c;
}

KeyframePoseSource pose_source(std::shared_ptr<const KeyframeStore> store) {
  KeyframePoseSource ps;
  ps.pose = [store](std::uint64_t id) { return store->pose(id); };
  ps.stamp = [](std::uint64_t id) {
    return std::optional<Timestamp>(static_cast<Timestamp>(id) * 1'000'000'000LL);
  };
  ps.chain_cov = [](std::uint64_t a, std::uint64_t b) {
    const double n = static_cast<double>(b > a ? b - a : a - b);
    return std::optional<Eigen::Matrix<double, 6, 6>>(
        (0.25 * n + 0.01) * Eigen::Matrix<double, 6, 6>::Identity());
  };
  return ps;
}

// Run the same scene/trajectory through a fresh detector and collect every emitted loop.
std::vector<LoopConstraint> run_once() {
  const PointCloud scene = world_scene();
  auto store = std::make_shared<KeyframeStore>();
  const std::vector<Pose> traj = {at(1, 1, 0), at(4, 1, 0), at(7, 1, 0), at(4, 1, 0), at(1, 1, 0)};
  auto det = makeLoopDetector(cfg(), store, pose_source(store), /*deterministic=*/true);
  std::vector<LoopConstraint> all;
  for (std::uint64_t j = 0; j < traj.size(); ++j) {
    PointCloudPtr cloud = observe(scene, traj[j]);
    store->put(j, cloud, nullptr, traj[j]);
    det->add_keyframe(j, cloud, traj[j]);
    for (const LoopConstraint& lc : det->detect()) all.push_back(lc);
  }
  return all;
}

}  // namespace

TEST(LoopDetectorDeterminism, TwoRunsByteIdentical) {
  const std::vector<LoopConstraint> a = run_once();
  const std::vector<LoopConstraint> b = run_once();
  ASSERT_FALSE(a.empty());
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].from_id, b[i].from_id);
    EXPECT_EQ(a[i].to_id, b[i].to_id);
    EXPECT_DOUBLE_EQ(a[i].fitness, b[i].fitness);
    EXPECT_DOUBLE_EQ(a[i].T_from_to.t.x(), b[i].T_from_to.t.x());
    EXPECT_DOUBLE_EQ(a[i].T_from_to.t.y(), b[i].T_from_to.t.y());
    EXPECT_DOUBLE_EQ(a[i].T_from_to.t.z(), b[i].T_from_to.t.z());
    EXPECT_DOUBLE_EQ(a[i].T_from_to.q.w(), b[i].T_from_to.q.w());
    EXPECT_TRUE(a[i].cov.M.isApprox(b[i].cov.M, 0.0));  // exact
  }
}
