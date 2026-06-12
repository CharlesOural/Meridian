#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/config/config.hpp"
#include "meridian/place/iloop_detector.hpp"
#include "meridian/place/keyframe_store.hpp"

using meridian::KeyframePoseSource;
using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::LoopConstraint;
using meridian::makeLoopDetector;
using meridian::ObservabilityReport;
using meridian::PlaceConfig;
using meridian::PointCloud;
using meridian::PointCloudPtr;
using meridian::Pose;
using meridian::Timestamp;

namespace {

// A fixed world scene in world coordinates: a room corner (three orthogonal planes) gives
// GICP full 6-DoF constraint, while height-varying pillars scattered around make the place
// LOOK different from different viewpoints, so Scan Context can tell co-located keyframes from
// merely co-visible ones (a rigid scene aligns under GICP from any good init, so retrieval is
// what must distinguish a true revisit).
PointCloud world_scene() {
  PointCloud c;
  for (int i = 0; i <= 30; ++i) {
    for (int j = 0; j <= 30; ++j) {
      const float u = static_cast<float>(i) * 0.4f;
      const float w = static_cast<float>(j) * 0.4f;
      LidarPoint a, b, d;
      a.xyz = Eigen::Vector3f(u, w, 0.f);   // floor z=0
      b.xyz = Eigen::Vector3f(0.f, u, w);   // wall x=0
      d.xyz = Eigen::Vector3f(u, 0.f, w);   // wall y=0
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

// The scene as observed from a keyframe pose: world points expressed in the keyframe body.
PointCloudPtr observe(const PointCloud& scene, const Pose& T_map_body) {
  const Pose inv = T_map_body.inverse();
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

PlaceConfig loop_cfg() {
  PlaceConfig c;
  c.enable = true;
  c.min_kf_gap = 3;
  c.min_time_gap = 0.0;
  c.cooldown_kf = 0;
  c.detect_period_kf = 1;
  c.sc_max_xy = 100.0;
  c.sc_dist_thresh = 0.05;  // tight: a true revisit has near-zero Scan Context distance
  c.submap_window = 1;
  c.submap_voxel = 0.1;
  c.gicp_downsample = 0.2;
  c.gicp_max_corr_dist = 1.0;
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
  ps.obs = [](std::uint64_t) { return std::optional<ObservabilityReport>{}; };
  return ps;
}

}  // namespace

TEST(LoopDetectorE2E, FindsRevisitNoFalseLoop) {
  const PointCloud scene = world_scene();
  auto store = std::make_shared<KeyframeStore>();
  // There-and-back: KF0 and KF4 are the same place; KF1/KF3 and KF2 are elsewhere.
  const std::vector<Pose> traj = {at(1, 1, 0), at(4, 1, 0), at(7, 1, 0), at(4, 1, 0), at(1, 1, 0)};

  auto detector = makeLoopDetector(loop_cfg(), store, pose_source(store), /*deterministic=*/true);

  std::vector<std::vector<LoopConstraint>> per_step;
  for (std::uint64_t j = 0; j < traj.size(); ++j) {
    PointCloudPtr cloud = observe(scene, traj[j]);
    store->put(j, cloud, nullptr, traj[j]);
    detector->add_keyframe(j, cloud, traj[j]);
    per_step.push_back(detector->detect());
  }

  for (std::uint64_t j = 0; j < 4; ++j)
    EXPECT_TRUE(per_step[j].empty()) << "false loop at step " << j;

  ASSERT_EQ(per_step[4].size(), 1u);
  const LoopConstraint& lc = per_step[4][0];
  EXPECT_EQ(lc.from_id, 0u);
  EXPECT_EQ(lc.to_id, 4u);
  EXPECT_GT(lc.fitness, 0.3);
  // KF0 and KF4 share a pose, so the relative transform is ~identity.
  EXPECT_LT(lc.T_from_to.t.norm(), 0.1);
  EXPECT_LT(Eigen::AngleAxisd(lc.T_from_to.q).angle(), 0.05);

  const meridian::LoopDiagnostics diag = detector->diagnostics();
  EXPECT_EQ(diag.keyframes, 5u);
  EXPECT_GE(diag.emitted, 1u);
}
