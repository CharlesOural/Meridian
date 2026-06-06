#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

#include "ct/image_pyramid_view.hpp"
#include "ct/visual_map.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/common/pose.hpp"

using meridian::CameraModel;
using meridian::IntrinsicsCamera;
using meridian::Pose;
using meridian::ct::ImagePyramidView;
using meridian::ct::LidarHit;
using meridian::ct::VisualMap;
using meridian::ct::VisualMapConfig;
using meridian::ct::VisualPoint;

namespace {

constexpr int kWidth = 320;
constexpr int kHeight = 320;
constexpr double kFx = 100.0;
constexpr double kCx = 160.0;

IntrinsicsCamera makeIntrinsics() {
  IntrinsicsCamera k;
  k.fx = kFx;
  k.fy = kFx;
  k.cx = kCx;
  k.cy = kCx;
  k.model = IntrinsicsCamera::Distortion::None;
  k.coeffs = {0, 0, 0, 0, 0};
  k.width = kWidth;
  k.height = kHeight;
  return k;
}

// A 3-level pyramid; level 0 is a checkerboard whose square size sets the local
// gradient magnitude, so different image regions carry different Shi-Tomasi scores.
std::vector<cv::Mat> makePyramid(int square_px) {
  std::vector<cv::Mat> pyr;
  cv::Mat l0(kHeight, kWidth, CV_8UC1);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      const int cell = ((x / square_px) + (y / square_px)) & 1;
      l0.at<std::uint8_t>(y, x) = cell ? 220 : 40;
    }
  }
  pyr.push_back(l0);
  cv::Mat prev = l0;
  for (int l = 1; l < 3; ++l) {
    cv::Mat next(prev.rows / 2, prev.cols / 2, CV_8UC1);
    for (int y = 0; y < next.rows; ++y) {
      for (int x = 0; x < next.cols; ++x) {
        next.at<std::uint8_t>(y, x) = prev.at<std::uint8_t>(y * 2, x * 2);
      }
    }
    pyr.push_back(next);
    prev = next;
  }
  return pyr;
}

// A pyramid that is highly textured (fine checker) on the left half and flat on the
// right half, so the per-cell best-gradient winner is predictable.
std::vector<cv::Mat> makeLeftTexturedPyramid() {
  std::vector<cv::Mat> pyr;
  cv::Mat l0(kHeight, kWidth, CV_8UC1);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      std::uint8_t v;
      if (x < kWidth / 2) {
        v = ((x / 2 + y / 2) & 1) ? 230 : 20;  // strong fine texture
      } else {
        v = static_cast<std::uint8_t>(120 + (x % 3));  // nearly flat
      }
      l0.at<std::uint8_t>(y, x) = v;
    }
  }
  pyr.push_back(l0);
  cv::Mat prev = l0;
  for (int l = 1; l < 3; ++l) {
    cv::Mat next(prev.rows / 2, prev.cols / 2, CV_8UC1);
    for (int y = 0; y < next.rows; ++y) {
      for (int x = 0; x < next.cols; ++x) {
        next.at<std::uint8_t>(y, x) = prev.at<std::uint8_t>(y * 2, x * 2);
      }
    }
    pyr.push_back(next);
    prev = next;
  }
  return pyr;
}

// World point at depth z that projects to pixel (u,v) under identity camera pose.
Eigen::Vector3d worldAtPixel(double u, double v, double z) {
  return Eigen::Vector3d((u - kCx) * z / kFx, (v - kCx) * z / kFx, z);
}

LidarHit makeHit(const Eigen::Vector3d& p_world, const Eigen::Vector3d& n, bool normal_valid) {
  LidarHit h;
  h.p_world = p_world;
  h.p_lidar = p_world;  // identity extrinsic for the test
  h.plane.n = n.normalized();
  h.plane.d = -h.plane.n.dot(p_world);
  h.plane.valid = normal_valid;
  return h;
}

}  // namespace

TEST(VisualMap, PromotePicksBestGradientPerCell) {
  VisualMapConfig cfg;  // grid_cell_px = 32
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  ASSERT_TRUE(cam.valid());
  // 16px checker: vertices (corners) at multiples of 16; a square centre is flat
  // over the whole 9x9 Shi-Tomasi window (window 9 < square 16). A vertex carries
  // gradient in both image directions (high Shi-Tomasi min-eigenvalue, a "corner").
  const auto pyr = makePyramid(16);
  ImagePyramidView img(pyr);
  const Pose T_w_c;  // identity
  const Eigen::Vector3d n(0, 0, -1);

  // Two hits in the SAME 32px cell [64,96): one at a checker vertex (corner -> high
  // Shi-Tomasi score), one at a square centre (flat -> low score). Corner must win.
  const double depth = 5.0;
  const Eigen::Vector3d on_corner = worldAtPixel(80.0, 80.0, depth);  // vertex
  const Eigen::Vector3d in_flat = worldAtPixel(72.0, 72.0, depth);    // square centre
  // Flat hit FIRST so a naive "keep first" would pick the wrong one.
  std::vector<LidarHit> hits = {makeHit(in_flat, n, true), makeHit(on_corner, n, true)};
  const int added = map.promote(img, cam, T_w_c, 1.0, hits);
  EXPECT_EQ(added, 1);  // one cell -> one winner
  ASSERT_EQ(map.size(), 1u);

  const auto cands = map.visibleCandidates(cam, T_w_c);
  ASSERT_EQ(cands.size(), 1u);
  Eigen::Vector2d uv;
  ASSERT_TRUE(cam.project(cands[0]->p_world, &uv));
  // The winner's pixel must be the corner (u≈80), not the flat centre (u≈72).
  EXPECT_NEAR(uv.x(), 80.0, 2.0);
}

TEST(VisualMap, NormalLessHitsRejected) {
  VisualMapConfig cfg;
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);
  const Pose T_w_c;

  // A hit with an invalid (uninitialized) normal must not be promoted.
  const Eigen::Vector3d p = worldAtPixel(160, 160, 5.0);
  std::vector<LidarHit> hits = {makeHit(p, Eigen::Vector3d(0, 0, -1), /*normal_valid=*/false)};
  EXPECT_EQ(map.promote(img, cam, T_w_c, 1.0, hits), 0);
  EXPECT_EQ(map.size(), 0u);

  // A zero-normal hit (valid flag but degenerate normal) is also rejected.
  LidarHit z;
  z.p_world = p;
  z.plane.valid = true;
  z.plane.n = Eigen::Vector3d::Zero();
  EXPECT_EQ(map.promote(img, cam, T_w_c, 1.0, {z}), 0);
  EXPECT_EQ(map.size(), 0u);
}

TEST(VisualMap, DisabledWhenCameraInvalid) {
  VisualMapConfig cfg;
  VisualMap map(cfg);
  CameraModel cam;  // default-constructed -> invalid (fx == 0)
  ASSERT_FALSE(cam.valid());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);
  const Pose T_w_c;
  const Eigen::Vector3d p = worldAtPixel(160, 160, 5.0);
  std::vector<LidarHit> hits = {makeHit(p, Eigen::Vector3d(0, 0, -1), true)};
  EXPECT_EQ(map.promote(img, cam, T_w_c, 1.0, hits), 0);
  EXPECT_TRUE(map.visibleCandidates(cam, T_w_c).empty());
}

TEST(VisualMap, EvictionBoundsCountOverLongTrajectory) {
  VisualMapConfig cfg;
  cfg.active_box_m = 10.0;  // small box so the bound is reached quickly
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);

  // March the camera forward along +x, promoting a fresh fan of hits each step. The
  // active box is +/-10 m, so points more than 10 m behind are box-deleted. The
  // active count must stay bounded no matter how long the trajectory runs.
  std::size_t max_seen = 0;
  for (int step = 0; step < 200; ++step) {
    Pose T_w_c;
    T_w_c.t = Eigen::Vector3d(static_cast<double>(step) * 1.0, 0, 0);
    // Promote a grid of hits in front of the camera (camera +z is world +z here).
    std::vector<LidarHit> hits;
    const Eigen::Vector3d n(0, 0, -1);
    for (int gx = 0; gx < 6; ++gx) {
      for (int gy = 0; gy < 6; ++gy) {
        const double u = 40.0 + gx * 48.0;
        const double v = 40.0 + gy * 48.0;
        // Place each point in world relative to the current camera position.
        Eigen::Vector3d pc = worldAtPixel(u, v, 5.0);
        hits.push_back(makeHit(T_w_c.t + pc, n, true));
      }
    }
    map.promote(img, cam, T_w_c, 1.0, hits);
    map.evict(T_w_c.t);
    max_seen = std::max(max_seen, map.size());
  }
  // Bound: points within +/-10 m of x over a 5 m depth fan; comfortably under a few
  // hundred. The key assertion is that it does NOT grow with the 200-step length.
  EXPECT_LE(map.size(), 1500u);
  EXPECT_LE(max_seen, 2000u);
  // And it is genuinely bounded: far fewer than the 200*36 = 7200 promoted points.
  EXPECT_LT(map.size(), 7200u);
}

TEST(VisualMap, ObservationCapIsRespected) {
  VisualMapConfig cfg;
  cfg.ref_obs_cap = 5;
  cfg.ref_add_angle_deg = 0.1;   // accept almost any new view as novel
  cfg.ref_converged_obs = 1000;  // disable the converged latch for this test
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);

  // Promote one point straight ahead.
  Pose T0;
  const Eigen::Vector3d n(0, 0, -1);
  const Eigen::Vector3d p = worldAtPixel(160, 160, 5.0);
  ASSERT_EQ(map.promote(img, cam, T0, 1.0, {makeHit(p, n, true)}), 1);

  // Orbit the camera around the point so each frame is a genuinely new view, then
  // run the lifecycle update. The stored observation count must never exceed the cap.
  const auto cands0 = map.visibleCandidates(cam, T0);
  ASSERT_EQ(cands0.size(), 1u);
  for (int i = 0; i < 40; ++i) {
    const double ang = (static_cast<double>(i) + 1.0) * 0.15;
    Pose T;
    // Move the camera on a small arc keeping the point in view near image centre.
    T.t = Eigen::Vector3d(std::sin(ang) * 0.5, 0.0, 0.0);
    map.updateAfterSolve(img, cam, T, 1.0);
  }
  const auto cands = map.visibleCandidates(cam, T0);
  ASSERT_EQ(cands.size(), 1u);
  EXPECT_LE(cands[0]->obs_count, cfg.ref_obs_cap);
  EXPECT_GE(cands[0]->obs_count, 1);
}

TEST(VisualMap, ConvergedLatchFreezesObservations) {
  VisualMapConfig cfg;
  cfg.ref_obs_cap = 50;
  cfg.ref_add_angle_deg = 1.0;        // each step is a new view
  cfg.ref_converged_obs = 4;          // latch after 4 obs ...
  cfg.ref_converged_angle_deg = 5.0;  // ... spanning >= 5 deg
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);

  Pose T0;
  const Eigen::Vector3d n(0, 0, -1);
  const Eigen::Vector3d p = worldAtPixel(160, 160, 5.0);
  ASSERT_EQ(map.promote(img, cam, T0, 1.0, {makeHit(p, n, true)}), 1);

  // Add several distinct views to trigger the latch.
  int latched_count = -1;
  for (int i = 0; i < 10; ++i) {
    Pose T;
    T.t = Eigen::Vector3d(0.3 * static_cast<double>(i + 1), 0.0, 0.0);
    map.updateAfterSolve(img, cam, T, 1.0);
    const auto cands = map.visibleCandidates(cam, T0);
    ASSERT_EQ(cands.size(), 1u);
    // Once converged, obs_count stops growing.
    const int c = cands[0]->obs_count;
    if (latched_count >= 0) {
      EXPECT_EQ(c, latched_count);
    }
    if (c >= cfg.ref_converged_obs && latched_count < 0) {
      // Give one more step to confirm it stays frozen.
      latched_count = c;
    }
  }
  EXPECT_GT(latched_count, 0);
}

TEST(VisualMap, DeterministicStateAcrossIdenticalRuns) {
  const auto pyr = makeLeftTexturedPyramid();
  ImagePyramidView img(pyr);
  CameraModel cam(makeIntrinsics());

  const auto run = [&]() {
    VisualMapConfig cfg;
    cfg.active_box_m = 12.0;
    VisualMap map(cfg);
    const Eigen::Vector3d n(0, 0, -1);
    for (int step = 0; step < 30; ++step) {
      Pose T;
      T.t = Eigen::Vector3d(static_cast<double>(step) * 0.5, 0, 0);
      std::vector<LidarHit> hits;
      for (int gx = 0; gx < 8; ++gx) {
        for (int gy = 0; gy < 8; ++gy) {
          const double u = 30.0 + gx * 36.0;
          const double v = 30.0 + gy * 36.0;
          Eigen::Vector3d pc = worldAtPixel(u, v, 4.0 + 0.01 * (gx + gy));
          hits.push_back(makeHit(T.t + pc, n, true));
        }
      }
      map.promote(img, cam, T, 1.0, hits);
      map.updateAfterSolve(img, cam, T, 1.0);
      map.evict(T.t);
    }
    return map.summary();
  };

  const VisualMap::StateSummary a = run();
  const VisualMap::StateSummary b = run();
  EXPECT_EQ(a.count, b.count);
  EXPECT_EQ(a.next_id, b.next_id);
  EXPECT_EQ(a.hash, b.hash);
  EXPECT_GT(a.count, 0u);
}

// Depth-continuity gate: a near (foreground) point in front of a far surface, placed so
// the near point's pixel falls inside a far candidate's half-patch footprint, must drop
// that far candidate as occluded, while a far candidate well clear of the near point
// survives. Sweeping depth_continuity_m across the depth gap proves the gate boundary:
// a gate wider than the gap keeps the edge candidate, a gate narrower drops it.
TEST(VisualMap, DepthContinuityGateDropsOccludedAcrossEdge) {
  CameraModel cam(makeIntrinsics());
  ASSERT_TRUE(cam.valid());
  const auto pyr = makePyramid(8);  // dense texture so every promotion has gradient
  ImagePyramidView img(pyr);
  const Pose T_w_c;  // identity
  const Eigen::Vector3d n(0, 0, -1);

  constexpr double kNearZ = 2.0;
  constexpr double kFarZ = 8.0;  // 6 m gap from the near surface
  // half_patch_ is 4 px; place the near point 2 px from the "edge" far point so the near
  // pixel sits inside the edge far point's +/-4 px footprint, in a DIFFERENT 32 px cell
  // (so both survive promotion as separate cell winners).
  const Eigen::Vector3d far_edge = worldAtPixel(160.0, 160.0, kFarZ);  // cell (5,5)
  const Eigen::Vector3d near_pt = worldAtPixel(158.0, 160.0, kNearZ);  // cell (4,5), 2 px away
  // A far point far from the near point (cell (1,1)): no discontinuity in its footprint.
  const Eigen::Vector3d far_clear = worldAtPixel(48.0, 48.0, kFarZ);

  // Run promotion + candidate selection at one gate and return the surviving candidate
  // pixels by value. The VisualMap owns the points the candidate pointers index, so it
  // must outlive the projection; we resolve pixels here while it is still alive.
  const auto survivingPixels = [&](double gate) {
    VisualMapConfig cfg;
    cfg.depth_continuity_m = gate;
    VisualMap map(cfg);
    std::vector<LidarHit> hits = {makeHit(far_edge, n, true), makeHit(near_pt, n, true),
                                  makeHit(far_clear, n, true)};
    const int added = map.promote(img, cam, T_w_c, 1.0, hits);
    EXPECT_EQ(added, 3) << "three points in three distinct cells must all promote";
    std::vector<Eigen::Vector2d> px;
    for (const VisualPoint* p : map.visibleCandidates(cam, T_w_c)) {
      Eigen::Vector2d uv;
      if (cam.project(p->p_world, &uv)) px.push_back(uv);
    }
    return px;
  };
  const auto hasPixelNear = [](const std::vector<Eigen::Vector2d>& px, double u, double v) {
    return std::any_of(px.begin(), px.end(), [&](const Eigen::Vector2d& uv) {
      return std::abs(uv.x() - u) < 2.0 && std::abs(uv.y() - v) < 2.0;
    });
  };

  // Wide gate (> 6 m gap): the gate never bites, so the edge far candidate survives.
  {
    const auto px = survivingPixels(10.0);
    EXPECT_TRUE(hasPixelNear(px, 160.0, 160.0))
        << "wide gate must keep the edge candidate (no discontinuity detected)";
    EXPECT_TRUE(hasPixelNear(px, 48.0, 48.0)) << "clear far candidate must always survive";
  }

  // Narrow gate (< 6 m gap): the near point's depth inside the edge candidate's footprint
  // exceeds the gate, so the edge candidate is dropped as occluded, while the clear far
  // candidate (no discontinuity nearby) survives.
  {
    const auto px = survivingPixels(0.5);
    EXPECT_FALSE(hasPixelNear(px, 160.0, 160.0))
        << "narrow gate must drop the edge candidate straddling the occlusion edge";
    EXPECT_TRUE(hasPixelNear(px, 48.0, 48.0))
        << "the unoccluded far candidate must survive the narrow gate";
  }
}

TEST(VisualMap, VisibleCandidatesAreOnePerCell) {
  VisualMapConfig cfg;
  VisualMap map(cfg);
  CameraModel cam(makeIntrinsics());
  const auto pyr = makePyramid(8);
  ImagePyramidView img(pyr);
  const Pose T_w_c;
  const Eigen::Vector3d n(0, 0, -1);

  // Promote points in distinct cells, then confirm visibleCandidates yields one per
  // occupied cell (no cell appears twice).
  std::vector<LidarHit> hits;
  for (int gx = 0; gx < 5; ++gx) {
    for (int gy = 0; gy < 5; ++gy) {
      const double u = 40.0 + gx * 48.0;
      const double v = 40.0 + gy * 48.0;
      hits.push_back(makeHit(worldAtPixel(u, v, 5.0), n, true));
    }
  }
  map.promote(img, cam, T_w_c, 1.0, hits);
  const auto cands = map.visibleCandidates(cam, T_w_c);
  ASSERT_FALSE(cands.empty());

  std::vector<int> cells;
  const int cell = cfg.grid_cell_px;
  const int gw = (kWidth + cell - 1) / cell;
  for (const VisualPoint* p : cands) {
    Eigen::Vector3d pc = p->p_world;
    Eigen::Vector2d uv;
    ASSERT_TRUE(cam.project(pc, &uv));
    const int idx = (static_cast<int>(uv.y()) / cell) * gw + (static_cast<int>(uv.x()) / cell);
    cells.push_back(idx);
  }
  std::sort(cells.begin(), cells.end());
  EXPECT_EQ(std::adjacent_find(cells.begin(), cells.end()), cells.end());
}
