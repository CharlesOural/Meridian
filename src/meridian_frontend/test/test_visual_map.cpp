#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
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

namespace meridian::ct {

// Test-only friend: reaches the private candidate seams so the frustum cull can be
// driven against an unculled full-map scan through the identical per-point pipeline.
class VisualMapFrustumEquivalenceAccess {
public:
  // Production path: frustum-culled candidate list fed through the visibility pipeline.
  static std::vector<std::int64_t> frustumSelect(const VisualMap& m, const CameraModel& cam,
                                                 const Pose& T_w_c) {
    return m.selectVisibleFromCandidates(cam, T_w_c, m.frustumCandidateIds(cam, T_w_c));
  }
  // Reference path: every live point id (ascending) fed through the SAME pipeline,
  // bypassing the frustum cull entirely.
  static std::vector<std::int64_t> fullScanSelect(const VisualMap& m, const CameraModel& cam,
                                                  const Pose& T_w_c) {
    std::vector<std::int64_t> all;
    all.reserve(m.points_.size());
    for (const auto& [id, uptr] : m.points_) all.push_back(id);
    return m.selectVisibleFromCandidates(cam, T_w_c, all);
  }
  // The raw frustum candidate-id list, for asserting the cull is a strict superset.
  static std::vector<std::int64_t> frustumCandidates(const VisualMap& m, const CameraModel& cam,
                                                     const Pose& T_w_c) {
    return m.frustumCandidateIds(cam, T_w_c);
  }

  // Seed a point at an exact world position with an initialized normal, bypassing
  // promote() so points the camera cannot see (behind it, off the image, outside the
  // frustum) can be placed -- those are precisely the cases the cull must handle. The
  // normal faces the camera centre so it is never the reason a point is excluded.
  static std::int64_t seedPoint(VisualMap& m, const Eigen::Vector3d& p_world,
                                const Eigen::Vector3d& cam_centre) {
    auto pt = std::make_unique<VisualPoint>();
    pt->p_world = p_world;
    Eigen::Vector3d n = (cam_centre - p_world);
    pt->n_world = n.norm() > 1e-9 ? n.normalized() : Eigen::Vector3d::UnitZ();
    pt->normal_initialized = true;
    const std::int64_t id = m.next_id_++;
    pt->id = id;
    const VisualMap::VoxelKey key = m.voxelKey(p_world);
    m.points_.emplace(id, std::move(pt));
    m.addPointAt(key, id);
    return id;
  }
};

}  // namespace meridian::ct

using meridian::ct::VisualMapFrustumEquivalenceAccess;

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

// Intrinsics with plumb-bob radial-tangential distortion (k1,k2,p1,p2,k3). A large
// positive k1 bows the true on-image frustum boundary outward between the sampled
// image corners, which is exactly where a too-tight corner-ray cull would wrongly
// drop a point the per-point projection still admits.
IntrinsicsCamera makeDistortedIntrinsics(double k1, double k2 = 0.0, double p1 = 0.0,
                                         double p2 = 0.0, double k3 = 0.0) {
  IntrinsicsCamera k = makeIntrinsics();
  k.model = IntrinsicsCamera::Distortion::RadTan;
  k.coeffs = {k1, k2, p1, p2, k3};
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
    map.refreshVisibleCache(cam, T);
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
    map.refreshVisibleCache(cam, T);
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
      map.refreshVisibleCache(cam, T);
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

// ---------------------------------------------------------------------------
// Frustum-cull equivalence: the production selectVisibleIds culls candidates to the
// view frustum via frustumCandidateIds before running the per-point projection +
// on-image + depth-continuity pipeline. The claim under test is that this is a strict
// restriction of the iteration: the culled candidate list and the whole-map list must
// yield the IDENTICAL visible set (same ids, same order). A too-tight cull would drop
// a point the full scan keeps, which also perturbs the shared depth buffer and so the
// occlusion decisions of OTHER points -- meaning any divergence shows up as an exact
// vector mismatch here. The battery stresses every boundary where a tight cull bites.
// ---------------------------------------------------------------------------
namespace {

using EqAccess = meridian::ct::VisualMapFrustumEquivalenceAccess;

// Assert the frustum-culled selection equals the full-map-scan selection exactly, and
// that the frustum candidate set is itself a superset of the points the pipeline keeps
// (a necessary condition: a kept point that is not even a candidate is impossible).
::testing::AssertionResult frustumEqualsFullScan(const VisualMap& map, const CameraModel& cam,
                                                 const Pose& T_w_c, const char* where) {
  const std::vector<std::int64_t> frustum = EqAccess::frustumSelect(map, cam, T_w_c);
  const std::vector<std::int64_t> full = EqAccess::fullScanSelect(map, cam, T_w_c);
  if (frustum != full) {
    std::string msg = std::string("frustum != full-scan at ") + where + "\n  frustum: ";
    for (std::int64_t id : frustum) msg += std::to_string(id) + " ";
    msg += "\n  full:    ";
    for (std::int64_t id : full) msg += std::to_string(id) + " ";
    return ::testing::AssertionFailure() << msg;
  }
  // Strict-superset check: every kept id must appear in the raw candidate list.
  std::vector<std::int64_t> cands = EqAccess::frustumCandidates(map, cam, T_w_c);
  std::sort(cands.begin(), cands.end());
  for (std::int64_t id : full) {
    if (!std::binary_search(cands.begin(), cands.end(), id)) {
      return ::testing::AssertionFailure()
             << "kept id " << id << " was culled from the candidate set at " << where;
    }
  }
  return ::testing::AssertionSuccess();
}

// World point at the given camera-frame normalized image coordinates (x,y) and depth,
// expressed in the world frame for camera pose T_w_c. (x,y) are pre-distortion ray
// coordinates; the distortion map then bends where they actually land on the image,
// which is why a dense sweep of these is the right cull stressor.
Eigen::Vector3d worldAtRay(const Pose& T_w_c, double x, double y, double z) {
  const Eigen::Vector3d p_cam(x * z, y * z, z);
  return T_w_c.q * p_cam + T_w_c.t;
}

// World point that lands at distorted pixel (u,v) at camera-frame depth z for pose
// T_w_c, recovered by unprojecting through the lens model. Used to walk a point
// pixel-by-pixel across an image edge regardless of distortion.
Eigen::Vector3d worldAtPixelDepthPose(const CameraModel& cam, const Pose& T_w_c, double u, double v,
                                      double z) {
  const Eigen::Vector3d ray = cam.unproject(Eigen::Vector2d(u, v));  // [x,y,1]
  const Eigen::Vector3d p_cam(ray.x() * z, ray.y() * z, z);
  return T_w_c.q * p_cam + T_w_c.t;
}

// Seed a map with a dense cloud spanning the cull boundary for camera pose T_w_c:
//  - a fine grid of pre-distortion rays from well inside to well outside the image,
//    at several depths (so per-cell occlusion and depth-continuity are exercised),
//  - the four extreme image corners and points a hair inside/outside each edge,
//  - points dead-centre, far down the optical axis, and just in front of / behind the
//    camera (z just above and just below zero).
void seedBoundaryCloud(VisualMap& map, const Pose& T_w_c) {
  const Eigen::Vector3d c = T_w_c.t;
  // Normalized-coordinate extent of the image corners (pinhole), inflated to 1.6x so
  // the grid reaches well past the image into the region the cull must reject. The
  // grid is fine enough that consecutive samples straddle every image edge.
  const double x_corner = (static_cast<double>(kWidth - 1) - kCx) / kFx;
  const double y_corner = (static_cast<double>(kHeight - 1) - kCx) / kFx;
  const double xr = 1.6 * x_corner;
  const double yr = 1.6 * y_corner;
  constexpr int kGrid = 41;  // odd: a sample lands exactly on the optical axis
  const std::array<double, 4> depths = {1.5, 3.0, 7.0, 25.0};
  for (double z : depths) {
    for (int i = 0; i < kGrid; ++i) {
      const double x = -xr + 2.0 * xr * static_cast<double>(i) / static_cast<double>(kGrid - 1);
      for (int j = 0; j < kGrid; ++j) {
        const double y = -yr + 2.0 * yr * static_cast<double>(j) / static_cast<double>(kGrid - 1);
        EqAccess::seedPoint(map, worldAtRay(T_w_c, x, y, z), c);
      }
    }
  }
  // Exact image corners and a sub-pixel step inside / outside each edge, at two depths.
  const double eps = 0.4 / kFx;  // ~0.4 px in normalized units
  const std::array<double, 4> edge_x = {-x_corner, x_corner, -x_corner, x_corner};
  const std::array<double, 4> edge_y = {-y_corner, -y_corner, y_corner, y_corner};
  for (double z : {2.0, 6.0}) {
    for (int k = 0; k < 4; ++k) {
      for (double dx : {-eps, 0.0, eps}) {
        for (double dy : {-eps, 0.0, eps}) {
          EqAccess::seedPoint(map, worldAtRay(T_w_c, edge_x[k] + dx, edge_y[k] + dy, z), c);
        }
      }
    }
    EqAccess::seedPoint(map, worldAtRay(T_w_c, 0.0, 0.0, z), c);  // dead centre
  }
  // Far down the optical axis and just in front of / behind the camera.
  EqAccess::seedPoint(map, worldAtRay(T_w_c, 0.0, 0.0, 80.0), c);
  EqAccess::seedPoint(map, worldAtRay(T_w_c, 0.0, 0.0, 1e-3), c);   // z just > 0
  EqAccess::seedPoint(map, worldAtRay(T_w_c, 0.0, 0.0, -1e-3), c);  // z just < 0
  EqAccess::seedPoint(map, worldAtRay(T_w_c, 0.5, 0.3, -2.0), c);   // well behind
  EqAccess::seedPoint(map, c + Eigen::Vector3d(0.0, 0.0, 0.0), c);  // at the centre
}

// A spread of camera poses: identity, pure translations, and rotations about each
// axis (so the world-frame frustum planes are exercised away from the canonical
// orientation), plus a combined pose.
std::vector<Pose> equivalencePoses() {
  std::vector<Pose> poses;
  poses.emplace_back();  // identity
  {
    Pose t;
    t.t = Eigen::Vector3d(3.0, -2.0, 1.5);
    poses.push_back(t);
  }
  const std::array<Eigen::Vector3d, 3> axes = {
      Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitZ()};
  for (const Eigen::Vector3d& axis : axes) {
    for (double ang : {0.3, -0.7, 1.1}) {
      Pose r;
      r.q = Eigen::Quaterniond(Eigen::AngleAxisd(ang, axis));
      r.t = Eigen::Vector3d(1.0, 0.5, -0.5);
      poses.push_back(r);
    }
  }
  {
    Pose combo;
    combo.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.6, Eigen::Vector3d(1, 1, 1).normalized()));
    combo.t = Eigen::Vector3d(-4.0, 2.0, 3.0);
    poses.push_back(combo);
  }
  return poses;
}

// Build a fresh map seeded for `seed_pose`, then assert the frustum cull is exactly
// equivalent to the full scan when VIEWED FROM `view_pose` (which may differ from the
// seed pose, so the cloud lands partly off-frustum -- the case the cull must trim).
void checkEquivalence(const CameraModel& cam, const Pose& seed_pose, const Pose& view_pose,
                      const char* where) {
  VisualMapConfig cfg;
  VisualMap map(cfg);
  seedBoundaryCloud(map, seed_pose);
  EXPECT_TRUE(frustumEqualsFullScan(map, cam, view_pose, where));
}

}  // namespace

// Undistorted (pinhole) camera: the frustum corner rays are exact, so equivalence
// must hold trivially for every pose. This is the baseline that isolates a distortion
// failure from a plain geometric one.
TEST(VisualMapFrustum, EquivalentPinholeAllPoses) {
  CameraModel cam(makeIntrinsics());
  ASSERT_TRUE(cam.valid());
  const auto poses = equivalencePoses();
  for (std::size_t i = 0; i < poses.size(); ++i) {
    const std::string where = "pinhole pose " + std::to_string(i);
    // Seed and view from the same pose: the dense cloud fills the frustum and beyond.
    checkEquivalence(cam, poses[i], poses[i], where.c_str());
    // Seed from pose 0 but view from pose i: most of the cloud falls outside the
    // frustum, so the cull does real work and must still match the full scan.
    checkEquivalence(cam, poses[0], poses[i], (where + " (cross-view)").c_str());
  }
}

// Moderate barrel distortion (k1 = -0.28, a realistic wide-angle value): the on-image
// boundary bows inward/outward between corners. The x1.25 corner-ray margin must keep
// the cull a strict superset.
TEST(VisualMapFrustum, EquivalentModerateDistortionAllPoses) {
  CameraModel cam(makeDistortedIntrinsics(-0.28, 0.06));
  ASSERT_TRUE(cam.valid());
  const auto poses = equivalencePoses();
  for (std::size_t i = 0; i < poses.size(); ++i) {
    const std::string where = "moderate-distortion pose " + std::to_string(i);
    checkEquivalence(cam, poses[i], poses[i], where.c_str());
    checkEquivalence(cam, poses[0], poses[i], (where + " (cross-view)").c_str());
  }
}

// Heavy distortion: a large positive k1 (pincushion) pushes the true on-image
// boundary OUTWARD from the straight corner-to-corner edges by the largest amount,
// the worst case for a fixed angular margin. This is the test most likely to expose
// an insufficient x1.25 margin; if it fails the margin must be widened until it holds.
TEST(VisualMapFrustum, EquivalentHeavyDistortionAllPoses) {
  for (double k1 : {0.4, 0.8, 1.2}) {
    CameraModel cam(makeDistortedIntrinsics(k1));
    ASSERT_TRUE(cam.valid());
    const auto poses = equivalencePoses();
    for (std::size_t i = 0; i < poses.size(); ++i) {
      const std::string where =
          "heavy-distortion k1=" + std::to_string(k1) + " pose " + std::to_string(i);
      checkEquivalence(cam, poses[i], poses[i], where.c_str());
      checkEquivalence(cam, poses[0], poses[i], (where + " (cross-view)").c_str());
    }
  }
}

// Strong tangential + radial mix: tangential terms skew the boundary asymmetrically,
// so the four side planes are stressed unequally. Combined with a strong k1 this is a
// hard, asymmetric cull boundary.
TEST(VisualMapFrustum, EquivalentTangentialDistortionAllPoses) {
  CameraModel cam(makeDistortedIntrinsics(0.5, 0.1, 0.03, -0.02, 0.0));
  ASSERT_TRUE(cam.valid());
  const auto poses = equivalencePoses();
  for (std::size_t i = 0; i < poses.size(); ++i) {
    const std::string where = "tangential-distortion pose " + std::to_string(i);
    checkEquivalence(cam, poses[i], poses[i], where.c_str());
    checkEquivalence(cam, poses[0], poses[i], (where + " (cross-view)").c_str());
  }
}

// Direct boundary probe, independent of the dense cloud: walk a point pixel-by-pixel
// across each image edge at several depths and confirm the cull keeps it for as long
// as the full scan does. This pins the exact transition rather than sampling near it.
TEST(VisualMapFrustum, EquivalentWalkingEachEdge) {
  for (double k1 : {0.0, -0.28, 0.8}) {
    CameraModel cam(k1 == 0.0 ? CameraModel(makeIntrinsics())
                              : CameraModel(makeDistortedIntrinsics(k1)));
    ASSERT_TRUE(cam.valid());
    const Pose T_w_c;  // identity view
    const Eigen::Vector3d c = T_w_c.t;
    for (double z : {2.0, 5.0, 15.0}) {
      // Sweep a point across the full image plus a generous outside margin along the
      // top and left edges (the other two are symmetric under this square model).
      for (int along = -40; along < kWidth + 40; along += 1) {
        VisualMapConfig cfg;
        VisualMap map(cfg);
        // Top edge: vary u, hold v just inside the top.
        const double u = static_cast<double>(along);
        EqAccess::seedPoint(map, worldAtPixelDepthPose(cam, T_w_c, u, 6.0, z), c);
        // Left edge: vary v, hold u just inside the left.
        EqAccess::seedPoint(map, worldAtPixelDepthPose(cam, T_w_c, 6.0, u, z), c);
        const std::string where =
            "edge-walk k1=" + std::to_string(k1) + " z=" + std::to_string(z) +
            " along=" + std::to_string(along);
        EXPECT_TRUE(frustumEqualsFullScan(map, cam, T_w_c, where.c_str()));
      }
    }
  }
}
