#include "ct/residuals_lidar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <basalt/spline/ceres_spline_helper.h>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>
#include <gtest/gtest.h>
#include <sophus/so3.hpp>

#include "ct/spline_window.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

using meridian::Duration;
using meridian::FrontendLidar;
using meridian::LidarPoint;
using meridian::Pose;
using meridian::SplineWindow;
using meridian::Timestamp;
using meridian::ct::addLidarResiduals;
using meridian::ct::associate;
using meridian::ct::capHitsByNormalStrata;
using meridian::ct::LidarAssocStats;
using meridian::ct::LidarCapStats;
using meridian::ct::LidarHit;
using meridian::ct::LidarLocalMap;
using meridian::ct::PlaneFit;
using meridian::ct::voxelDownsample;
using meridian::ct::weakTranslationAxes;

namespace {

double tSec(Timestamp t) { return meridian::to_seconds(t); }

FrontendLidar makeCfg() {
  FrontendLidar cfg;
  cfg.voxel_map_m = 0.05;       // fine map so gridded walls stay dense
  cfg.num_match_points = 5;
  cfg.max_match_dist_sq = 5.0;
  cfg.plane_thresh = 0.1;
  cfg.point_cov = 1e-3;
  return cfg;
}

// A smooth analytic ground-truth trajectory for the LiDAR/IMU body frame F_e: a
// gentle Lissajous translation plus a slow turn, all kept well inside the room.
struct GroundTruth {
  Eigen::Vector3d w_world{0.15, -0.1, 0.25};
  double ax = 0.6, ay = 0.5, az = 0.3;
  double fx = 0.7, fy = 0.5, fz = 0.6;

  Sophus::SO3d rotation(double t) const { return Sophus::SO3d::exp(w_world * t); }
  Eigen::Vector3d position(double t) const {
    return Eigen::Vector3d(ax * std::sin(fx * t), ay * std::cos(fy * t),
                           az * std::sin(fz * t));
  }
  Eigen::Vector3d accelWorld(double t) const {
    return Eigen::Vector3d(-ax * fx * fx * std::sin(fx * t),
                           -ay * fy * fy * std::cos(fy * t),
                           -az * fz * fz * std::sin(fz * t));
  }
  Pose pose(double t) const {
    Pose p;
    p.q = rotation(t).unit_quaternion();
    p.t = position(t);
    return p;
  }
};

// Builds a uniform spline (n_cp == 1) following the ground truth, seeded from the
// analytic poses, covering a few knots past t_end.
SplineWindow makeSpline(const GroundTruth& gt, Timestamp t0, Timestamp t_end,
                        Duration knot_dt) {
  SplineWindow spline(knot_dt, 1);
  spline.initialize(t0, gt.pose(tSec(t0)));
  const auto seed = [&](Timestamp t) { return gt.pose(tSec(t)); };
  const Timestamp target = t_end + 4 * knot_dt;
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    spline.extendTo(t, seed, 1);
  }
  return spline;
}

// Knot-time reconstruction and control-point seeding helpers, replicated from the
// IMU residual test: a control point's grid time dominates the curve one interval
// earlier, so reproducing a curve p needs CP_j = p(t_j - h) - (h^2/6) p''(t_j - h).
std::vector<Timestamp> knotTimes(Timestamp t0, Timestamp t_end, Duration knot_dt,
                                 int cp) {
  std::vector<Timestamp> kts;
  for (int j = 0; j < 4; ++j) {
    kts.push_back(t0 + j * knot_dt);
  }
  const Timestamp target = t_end + 4 * knot_dt;
  const auto coverage = [&kts] { return kts[kts.size() - 3] - 1; };
  for (Timestamp t = t0 + knot_dt; t <= target; t += knot_dt) {
    while (coverage() < t) {
      const Timestamp seg_start = kts.back();
      const Timestamp step = knot_dt / cp;
      for (int j = 1; j <= cp; ++j) {
        kts.push_back(j == cp ? seg_start + knot_dt : seg_start + j * step);
      }
    }
  }
  return kts;
}

double cpTimeSec(const std::vector<Timestamp>& kts, std::size_t j) {
  const std::size_t k = (j + 1 < kts.size()) ? j : j - 1;
  const double h = tSec(kts[k + 1]) - tSec(kts[k]);
  return tSec(kts[j]) - h;
}

Eigen::Vector3d idealPositionCp(const GroundTruth& gt,
                                const std::vector<Timestamp>& kts, std::size_t j) {
  const std::size_t k = (j + 1 < kts.size()) ? j : j - 1;
  const double h = tSec(kts[k + 1]) - tSec(kts[k]);
  const double t = cpTimeSec(kts, j);
  return gt.position(t) - (h * h / 6.0) * gt.accelWorld(t);
}

void reseedToGroundTruth(SplineWindow& spline, const GroundTruth& gt,
                         const std::vector<Timestamp>& kts) {
  const int last = static_cast<int>(kts.size()) - 4;
  for (int i = 0; i <= last; ++i) {
    SplineWindow::SegmentRef seg = spline.segmentFor(kts[i]);
    for (int j = 0; j < 4; ++j) {
      const auto idx = static_cast<std::size_t>(i + j);
      Eigen::Map<Eigen::Quaterniond>(seg.so3_knots[j]) =
          gt.rotation(cpTimeSec(kts, idx)).unit_quaternion();
      Eigen::Map<Eigen::Vector3d>(seg.r3_knots[j]) = idealPositionCp(gt, kts, idx);
    }
  }
}

void constrainSo3Knots(ceres::Problem& problem, SplineWindow& spline,
                       Duration knot_dt) {
  for (Timestamp t = spline.minTime();; t += knot_dt) {
    SplineWindow::SegmentRef seg =
        spline.segmentFor(std::min(t, spline.maxTime() - 1));
    for (double* q : seg.so3_knots) {
      if (problem.HasParameterBlock(q) && problem.GetManifold(q) == nullptr) {
        problem.SetManifold(q, new ceres::EigenQuaternionManifold());
      }
    }
    if (t >= spline.maxTime() - 1) {
      break;
    }
  }
}

// Pins the gauge: the first SO(3) knot and the first two R^3 knots of the anchor
// segment. LiDAR alone fixes the trajectory only up to a global rigid motion, so a
// minimal gauge must be held.
void pinGauge(ceres::Problem& problem, SplineWindow& spline, Timestamp t_anchor) {
  SplineWindow::SegmentRef seg = spline.segmentFor(t_anchor);
  for (double* block : {seg.so3_knots[0], seg.r3_knots[0], seg.r3_knots[1]}) {
    if (problem.HasParameterBlock(block)) {
      problem.SetParameterBlockConstant(block);
    }
  }
}

// A box room: six axis-aligned walls. Each wall is a plane with an outward-facing
// extent; gridded sample points populate the map so a 5-NN fit recovers the wall.
struct BoxRoom {
  double hx = 4.0, hy = 4.0, hz = 2.5;  // half-extents [m]

  // World points on all six walls on a regular grid, dense enough that any query
  // near a wall finds 5 coplanar neighbours within max_match_dist_sq.
  std::vector<Eigen::Vector3d> wallPoints(double step) const {
    std::vector<Eigen::Vector3d> pts;
    for (double a = -hx; a <= hx + 1e-9; a += step) {
      for (double b = -hz; b <= hz + 1e-9; b += step) {
        pts.emplace_back(a, -hy, b);  // y = -hy wall
        pts.emplace_back(a, hy, b);   // y = +hy wall
      }
    }
    for (double a = -hy; a <= hy + 1e-9; a += step) {
      for (double b = -hz; b <= hz + 1e-9; b += step) {
        pts.emplace_back(-hx, a, b);  // x = -hx wall
        pts.emplace_back(hx, a, b);   // x = +hx wall
      }
    }
    for (double a = -hx; a <= hx + 1e-9; a += step) {
      for (double b = -hy; b <= hy + 1e-9; b += step) {
        pts.emplace_back(a, b, -hz);  // floor
        pts.emplace_back(a, b, hz);   // ceiling
      }
    }
    return pts;
  }

  // Wall samples kept `margin` away from every wall edge. A query at such a point
  // has all five nearest map neighbours on the same wall, so the plane fit is the
  // exact wall and the point-to-plane residual is pure float-quantization noise.
  std::vector<Eigen::Vector3d> interiorWallPoints(double margin,
                                                  double step) const {
    std::vector<Eigen::Vector3d> pts;
    const double ix = hx - margin, iy = hy - margin, iz = hz - margin;
    for (double a = -ix; a <= ix + 1e-9; a += step) {
      for (double b = -iz; b <= iz + 1e-9; b += step) {
        pts.emplace_back(a, -hy, b);
        pts.emplace_back(a, hy, b);
      }
    }
    for (double a = -iy; a <= iy + 1e-9; a += step) {
      for (double b = -iz; b <= iz + 1e-9; b += step) {
        pts.emplace_back(-hx, a, b);
        pts.emplace_back(hx, a, b);
      }
    }
    for (double a = -ix; a <= ix + 1e-9; a += step) {
      for (double b = -iy; b <= iy + 1e-9; b += step) {
        pts.emplace_back(a, b, -hz);
        pts.emplace_back(a, b, hz);
      }
    }
    return pts;
  }
};

// Builds a synthetic scan: for each query world point on a wall, evaluate the GT
// lidar pose at a per-point time and back-project the world point into the lidar
// frame, so that associating it through the GT spline reproduces the world point
// exactly. t_offset is spread across the sweep window so the residual exercises
// many distinct spline segments.
std::vector<LidarPoint> makeScan(const GroundTruth& gt, const Pose& T_fe_lidar,
                                 const BoxRoom& room, Timestamp t0_scan,
                                 Duration sweep) {
  const std::vector<Eigen::Vector3d> walls = room.interiorWallPoints(1.0, 0.5);
  std::vector<LidarPoint> scan;
  scan.reserve(walls.size());
  const int n = static_cast<int>(walls.size());
  for (int i = 0; i < n; ++i) {
    const auto frac = static_cast<double>(i) / static_cast<double>(n);
    const Timestamp t = t0_scan + static_cast<Timestamp>(frac * sweep);
    const Pose T_w_fe = gt.pose(tSec(t));
    const Pose T_w_l = T_w_fe * T_fe_lidar;
    const Eigen::Vector3d p_l = T_w_l.inverse() * walls[i];
    LidarPoint lp;
    lp.xyz = p_l.cast<float>();
    lp.t_offset_ns = static_cast<std::int32_t>(t - t0_scan);
    lp.range = static_cast<float>(p_l.norm());
    scan.push_back(lp);
  }
  return scan;
}

}  // namespace

// (a) Plane fit: a noisy patch sampled on a known plane recovers (n, d) within
// tolerance; a non-planar (volumetric) cluster is rejected.
TEST(ResidualsLidar, PlaneFitRecoversAndRejects) {
  FrontendLidar cfg = makeCfg();
  cfg.voxel_map_m = 1e-3;  // keep every sample so the patch stays dense
  cfg.plane_thresh = 0.05;

  const Eigen::Vector3d n_true = Eigen::Vector3d(0.3, -0.5, 1.0).normalized();
  const double d_true = -0.7;  // plane: n.x + d = 0

  std::mt19937 rng(7);
  std::normal_distribution<double> jitter(0.0, 0.002);

  // An orthonormal in-plane basis to sample a patch around a point on the plane.
  Eigen::Vector3d e1 = n_true.unitOrthogonal();
  Eigen::Vector3d e2 = n_true.cross(e1);
  const Eigen::Vector3d p0 = -d_true * n_true;  // a point on the plane

  // A coarse 0.15 m grid, like real sparse LiDAR returns: the five nearest map
  // points then span ~0.3 m, so the normal is well-conditioned against the small
  // off-plane noise. A dense cluster would make the in-plane normal direction
  // noise-dominated and the fit meaningless.
  std::vector<Eigen::Vector3d> patch;
  for (int i = -8; i <= 8; ++i) {
    for (int j = -8; j <= 8; ++j) {
      Eigen::Vector3d p = p0 + (0.15 * i) * e1 + (0.15 * j) * e2;
      p += jitter(rng) * n_true;  // tiny off-plane noise
      patch.push_back(p);
    }
  }
  LidarLocalMap map(cfg);
  map.insert(patch);
  ASSERT_TRUE(map.initialized());

  PlaneFit fit;
  ASSERT_TRUE(map.fitPlane(p0 + 0.03 * e1, &fit));
  ASSERT_TRUE(fit.valid);
  // Normal recovered up to sign; align before comparing. Single-precision tree
  // storage and the sparse grid limit the achievable accuracy to ~1e-2.
  const double sign = (fit.n.dot(n_true) < 0.0) ? -1.0 : 1.0;
  EXPECT_LT((sign * fit.n - n_true).norm(), 2e-2);
  EXPECT_NEAR(sign * fit.d, d_true, 1e-2);

  // A genuinely volumetric cluster: the eight corners of a 0.4 m cube plus its
  // centre. Any five nearest neighbours of the centre span the cube's depth, so
  // every candidate plane leaves some neighbour well beyond plane_thresh -> reject.
  std::vector<Eigen::Vector3d> blob;
  const Eigen::Vector3d c(5.0, 5.0, 5.0);
  const double h = 0.2;  // half-side -> 0.4 m cube, depth >> plane_thresh
  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      for (int sz : {-1, 1}) {
        blob.emplace_back(c + Eigen::Vector3d(sx * h, sy * h, sz * h));
      }
    }
  }
  blob.push_back(c);
  LidarLocalMap blob_map(cfg);
  blob_map.insert(blob);
  PlaneFit bad;
  EXPECT_FALSE(blob_map.fitPlane(c, &bad));
  EXPECT_FALSE(bad.valid);
}

// voxelDownsample keeps one representative per occupied cell and is the identity at
// non-positive voxel size.
TEST(ResidualsLidar, VoxelDownsampleCollapsesCells) {
  std::vector<LidarPoint> pts;
  for (int i = 0; i < 50; ++i) {
    LidarPoint p;
    // Many points inside one 0.1 m cell around the origin.
    p.xyz = Eigen::Vector3f(0.01f * (i % 5), 0.01f * ((i / 5) % 5), 0.0f);
    pts.push_back(p);
  }
  LidarPoint far;
  far.xyz = Eigen::Vector3f(2.0f, 2.0f, 2.0f);
  pts.push_back(far);

  const auto down = voxelDownsample(pts, 0.1);
  EXPECT_EQ(down.size(), 2u);  // one origin cell + the far point
  EXPECT_EQ(voxelDownsample(pts, 0.0).size(), pts.size());
}

// (b) Zero-residual: a box-room map, a GT-seeded spline, and a scan ray-cast from
// the GT poses at per-point times. associate + addLidarResiduals at the GT spline
// must yield near-zero cost and a high accepted fraction.
TEST(ResidualsLidar, ZeroResidualAtGroundTruth) {
  const GroundTruth gt;
  const FrontendLidar cfg = makeCfg();
  const Duration knot_dt = 25'000'000;  // 25 ms
  const Timestamp t0 = 0;
  const Timestamp t_end = 200'000'000;  // 0.2 s sweep

  // A small extrinsic so the residual exercises a non-identity T_fe_lidar.
  const Pose T_fe_lidar(
      Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.03, 0.05)).unit_quaternion(),
      Eigen::Vector3d(0.1, 0.0, -0.05));

  BoxRoom room;
  LidarLocalMap map(cfg);
  map.insert(room.wallPoints(0.1));
  ASSERT_TRUE(map.initialized());

  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  const auto scan = makeScan(gt, T_fe_lidar, room, t0, t_end);
  std::vector<LidarHit> hits;
  const LidarAssocStats stats =
      associate(spline, T_fe_lidar, scan, t0, map, cfg, &hits);

  ASSERT_GT(stats.attempted, 0);
  // The vast majority of clean, on-plane points must associate and be accepted.
  EXPECT_GT(static_cast<double>(stats.accepted),
            0.8 * static_cast<double>(stats.attempted));
  ASSERT_FALSE(hits.empty());

  ceres::Problem problem;
  const int n = addLidarResiduals(problem, spline, T_fe_lidar, hits, cfg);
  ASSERT_GT(n, 0);
  constrainSo3Knots(problem, spline, knot_dt);

  double cost = 0.0;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr,
                   nullptr);
  // Points lie exactly on the mapped walls and the spline is at GT, so the
  // point-to-plane residual is the plane-fit error of the gridded wall only.
  EXPECT_LT(cost / static_cast<double>(n), 1e-3);
}

// (c) Pulling power: perturb the spline knots, then solve with only LiDAR residuals
// and the gauge pinned. The trajectory must be pulled back to the ground truth.
TEST(ResidualsLidar, RecoversTrajectoryFromPerturbedKnots) {
  const GroundTruth gt;
  const FrontendLidar cfg = makeCfg();
  const Duration knot_dt = 40'000'000;  // 40 ms
  const Timestamp t0 = 0;
  const Timestamp t_end = 240'000'000;

  const Pose T_fe_lidar(Eigen::Quaterniond::Identity(),
                        Eigen::Vector3d(0.05, 0.0, 0.0));

  BoxRoom room;
  LidarLocalMap map(cfg);
  map.insert(room.wallPoints(0.1));

  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  // Associate at the GROUND-TRUTH spline so the correspondences and planes are
  // correct, then perturb the knots and let LiDAR pull them back. The sweep spans the
  // whole spline, including the knots makeSpline carries a few intervals past t_end: a
  // pose near the window end draws on those trailing control points, so they must see
  // data or that end of the fit is under-determined and need not match ground truth.
  const auto scan = makeScan(gt, T_fe_lidar, room, t0, t_end + 4 * knot_dt);
  std::vector<LidarHit> hits;
  const LidarAssocStats stats =
      associate(spline, T_fe_lidar, scan, t0, map, cfg, &hits);
  ASSERT_GT(stats.accepted, 50);

  // Perturb every control point away from the GT seed.
  std::mt19937 rng(2024);
  std::normal_distribution<double> noise(0.0, 0.03);
  for (Timestamp t = spline.minTime(); t < spline.maxTime(); t += knot_dt / 2) {
    SplineWindow::SegmentRef seg = spline.segmentFor(t);
    for (double* q : seg.so3_knots) {
      Eigen::Map<Eigen::Quaterniond> quat(q);
      const Eigen::Vector3d d(noise(rng), noise(rng), noise(rng));
      quat = (quat * Sophus::SO3d::exp(d).unit_quaternion()).normalized();
    }
    for (double* p : seg.r3_knots) {
      Eigen::Map<Eigen::Vector3d> v(p);
      v += Eigen::Vector3d(noise(rng), noise(rng), noise(rng));
    }
  }

  // The gauge anchors must hold their true values so the fit pulls the rest back
  // relative to a fixed reference.
  {
    SplineWindow::SegmentRef g0 = spline.segmentFor(t0);
    Eigen::Map<Eigen::Quaterniond>(g0.so3_knots[0]) =
        gt.rotation(cpTimeSec(kts, 0)).unit_quaternion();
    Eigen::Map<Eigen::Vector3d>(g0.r3_knots[0]) = idealPositionCp(gt, kts, 0);
    Eigen::Map<Eigen::Vector3d>(g0.r3_knots[1]) = idealPositionCp(gt, kts, 1);
  }

  ceres::Problem problem;
  ASSERT_GT(addLidarResiduals(problem, spline, T_fe_lidar, hits, cfg), 0);
  constrainSo3Knots(problem, spline, knot_dt);
  pinGauge(problem, spline, t0);

  ceres::Solver::Options opts;
  opts.linear_solver_type = ceres::DENSE_QR;
  opts.max_num_iterations = 60;
  ceres::Solver::Summary summary;
  ceres::Solve(opts, &problem, &summary);

  // The fitted spline must reproduce the GT trajectory inside the measured span. The
  // first interior sample sits just after the pinned gauge and sees the least diverse
  // set of plane normals in the sweep, so its position is the loosest-constrained (a
  // few cm); the bound leaves margin for that and for cross-architecture rounding.
  for (Timestamp t = t0 + knot_dt; t < t_end - knot_dt; t += knot_dt) {
    const Pose fit = spline.pose(t);
    const Pose truth = gt.pose(tSec(t));
    EXPECT_LT((fit.t - truth.t).norm(), 8e-2) << "t=" << t;
    const double ang = fit.q.angularDistance(truth.q);
    EXPECT_LT(ang, 5e-2) << "t=" << t;
  }
}

// Parallel association is deterministic and order-stable: the schedule(static) split
// gives each thread a contiguous ascending point range, and the per-thread hit buffers
// are merged in thread-index order, so the produced hit set and its order are identical
// across repeated runs. The downstream cap selects survivors by a stride over the
// (t_offset_ns, ring) key, which is only replay-stable if associate()'s output order is
// itself stable -- this asserts that property. A large scan is used so >1 thread engages.
TEST(ResidualsLidar, ParallelAssociateIsDeterministicAndOrderStable) {
  const GroundTruth gt;
  const FrontendLidar cfg = makeCfg();
  const Duration knot_dt = 25'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 200'000'000;

  const Pose T_fe_lidar(Sophus::SO3d::exp(Eigen::Vector3d(0.02, -0.03, 0.05)).unit_quaternion(),
                        Eigen::Vector3d(0.1, 0.0, -0.05));

  BoxRoom room;
  LidarLocalMap map(cfg);
  map.insert(room.wallPoints(0.1));
  ASSERT_TRUE(map.initialized());

  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  reseedToGroundTruth(spline, gt, kts);

  // A dense scan (finer wall grid) so the point count is well above the thread count and
  // the static chunking actually distributes work across threads.
  std::vector<LidarPoint> scan;
  {
    const std::vector<Eigen::Vector3d> walls = room.interiorWallPoints(0.5, 0.15);
    const int n = static_cast<int>(walls.size());
    scan.reserve(walls.size());
    for (int i = 0; i < n; ++i) {
      const auto frac = static_cast<double>(i) / static_cast<double>(n);
      const Timestamp t = t0 + static_cast<Timestamp>(frac * t_end);
      const Pose T_w_l = gt.pose(tSec(t)) * T_fe_lidar;
      const Eigen::Vector3d p_l = T_w_l.inverse() * walls[i];
      LidarPoint lp;
      lp.xyz = p_l.cast<float>();
      lp.t_offset_ns = static_cast<std::int32_t>(t - t0);
      lp.ring = static_cast<std::uint16_t>(i % 64);
      scan.push_back(lp);
    }
  }
  ASSERT_GT(scan.size(), 200u);

  std::vector<LidarHit> a;
  std::vector<LidarHit> b;
  const LidarAssocStats sa = associate(spline, T_fe_lidar, scan, t0, map, cfg, &a);
  const LidarAssocStats sb = associate(spline, T_fe_lidar, scan, t0, map, cfg, &b);

  ASSERT_GT(sa.accepted, 0);
  EXPECT_EQ(sa.attempted, sb.attempted);
  EXPECT_EQ(sa.matched, sb.matched);
  EXPECT_EQ(sa.accepted, sb.accepted);
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(static_cast<int>(a.size()), sa.accepted);

  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].t_offset_ns, b[i].t_offset_ns) << "hit " << i;
    EXPECT_EQ(a[i].ring, b[i].ring) << "hit " << i;
    EXPECT_EQ(a[i].t, b[i].t) << "hit " << i;
    EXPECT_EQ(a[i].p_world, b[i].p_world) << "hit " << i;  // bit-exact, not near
    EXPECT_EQ(a[i].p_lidar, b[i].p_lidar) << "hit " << i;
    EXPECT_EQ(a[i].plane.n, b[i].plane.n) << "hit " << i;
    EXPECT_EQ(a[i].plane.d, b[i].plane.d) << "hit " << i;
    EXPECT_EQ(a[i].weight, b[i].weight) << "hit " << i;
  }

  // Output order is ascending in the per-point time key the cap relies on, since each
  // thread keeps its contiguous range in scan order and the merge preserves thread order
  // (the scan's t_offset_ns is monotonic in index here).
  for (std::size_t i = 1; i < a.size(); ++i) {
    EXPECT_LE(a[i - 1].t_offset_ns, a[i].t_offset_ns) << "order break at " << i;
  }
}

// (d) Gating: a point with no nearby plane and a point whose neighbours are all too
// far are both rejected; the stats reflect the rejections.
TEST(ResidualsLidar, GatingRejectsUnsupportedAndFarPoints) {
  const GroundTruth gt;
  const FrontendLidar cfg = makeCfg();
  const Duration knot_dt = 25'000'000;
  const Timestamp t0 = 0;
  const Timestamp t_end = 100'000'000;

  const Pose T_fe_lidar;  // identity extrinsic

  BoxRoom room;
  LidarLocalMap map(cfg);
  map.insert(room.wallPoints(0.1));

  SplineWindow spline = makeSpline(gt, t0, t_end, knot_dt);
  const auto kts = knotTimes(t0, t_end, knot_dt, 1);
  ASSERT_EQ(static_cast<int>(kts.size()), spline.numKnots());
  reseedToGroundTruth(spline, gt, kts);

  // Build a scan of two bad points, both stamped mid-window so the spline covers
  // them. The world transform is the GT lidar pose; we choose lidar-frame points
  // whose world image is far from any wall.
  const Timestamp t_mid = t_end / 2;
  const Pose T_w_l = gt.pose(tSec(t_mid)) * T_fe_lidar;

  // (1) A point near the room centre: no wall plane within range -> no match.
  const Eigen::Vector3d centre_w(0.0, 0.0, 0.0);
  // (2) A point just outside a wall, far from any mapped neighbour (>> sqrt(5) m
  // from the nearest grid sample once placed beyond the wall extent corner).
  const Eigen::Vector3d far_w(room.hx + 6.0, room.hy + 6.0, 0.0);

  std::vector<LidarPoint> scan;
  for (const Eigen::Vector3d& pw : {centre_w, far_w}) {
    const Eigen::Vector3d p_l = T_w_l.inverse() * pw;
    LidarPoint lp;
    lp.xyz = p_l.cast<float>();
    lp.t_offset_ns = static_cast<std::int32_t>(t_mid - t0);
    scan.push_back(lp);
  }

  std::vector<LidarHit> hits;
  const LidarAssocStats stats =
      associate(spline, T_fe_lidar, scan, t0, map, cfg, &hits);

  EXPECT_EQ(stats.attempted, 2);   // both times are covered
  EXPECT_EQ(stats.matched, 0);     // neither finds a valid supporting plane
  EXPECT_EQ(stats.accepted, 0);
  EXPECT_TRUE(hits.empty());
}

namespace {

// A hit carrying just the fields the cap reads: a unit plane normal, the stable
// (t_offset_ns, ring) subsample key, and a weight. The point/plane geometry is
// irrelevant to the cap logic so it is left at defaults.
LidarHit makeHit(const Eigen::Vector3d& n, std::int32_t t_off, std::uint16_t ring,
                 double weight = 1.0) {
  LidarHit h;
  h.plane.n = n.normalized();
  h.plane.d = 0.0;
  h.plane.valid = true;
  h.t_offset_ns = t_off;
  h.ring = ring;
  h.weight = weight;
  return h;
}

}  // namespace

// Within budget: the cap is a pass-through that keeps every inlier and drops none.
TEST(ResidualsLidarCap, WithinBudgetKeepsAll) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 1000;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 50;

  std::vector<LidarHit> hits;
  for (int i = 0; i < 200; ++i) {
    hits.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
  }
  std::vector<LidarHit> out;
  const LidarCapStats s = capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &out);
  EXPECT_EQ(s.kept, 200);
  EXPECT_EQ(s.dropped, 0);
  EXPECT_EQ(out.size(), 200u);
}

// Over budget: the kept count equals the cap and the dropped count makes up the rest.
TEST(ResidualsLidarCap, RespectsCap) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 300;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 10;

  std::vector<LidarHit> hits;
  // A spread of normals across the six axes so several strata are populated.
  const std::array<Eigen::Vector3d, 6> axes = {
      Eigen::Vector3d(1, 0, 0),  Eigen::Vector3d(-1, 0, 0), Eigen::Vector3d(0, 1, 0),
      Eigen::Vector3d(0, -1, 0), Eigen::Vector3d(0, 0, 1),  Eigen::Vector3d(0, 0, -1)};
  for (int i = 0; i < 1200; ++i) {
    hits.push_back(makeHit(axes[i % 6], i, static_cast<std::uint16_t>(i % 64)));
  }
  std::vector<LidarHit> out;
  const LidarCapStats s = capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &out);
  EXPECT_EQ(s.kept, 300);
  EXPECT_EQ(s.dropped, 900);
  EXPECT_EQ(static_cast<int>(out.size()), 300);
}

// Determinism: identical inputs (same outer-step) yield byte-identical selection, and
// the selection is independent of the association order (a shuffled input picks the
// same set, since the cap sorts by the stable per-point key first).
TEST(ResidualsLidarCap, DeterministicSubsample) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 200;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 10;

  std::vector<LidarHit> hits;
  for (int i = 0; i < 1000; ++i) {
    const Eigen::Vector3d n = (i % 2 == 0) ? Eigen::Vector3d(1, 0, 0)
                                           : Eigen::Vector3d(0, 1, 0);
    hits.push_back(makeHit(n, i, static_cast<std::uint16_t>(i % 16)));
  }

  std::vector<LidarHit> a;
  std::vector<LidarHit> b;
  capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &a);
  capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &b);
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].t_offset_ns, b[i].t_offset_ns);
    EXPECT_EQ(a[i].ring, b[i].ring);
  }

  // Shuffle the input order and re-cap: the selected key set must be identical because
  // the per-point key, not arrival order, decides selection.
  std::vector<LidarHit> shuffled = hits;
  std::mt19937 rng(99);
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  std::vector<LidarHit> c;
  capHitsByNormalStrata(shuffled, cfg, 0, {}, 0.966, &c);
  ASSERT_EQ(a.size(), c.size());
  auto keyset = [](const std::vector<LidarHit>& v) {
    std::vector<std::pair<std::int32_t, std::uint16_t>> k;
    for (const LidarHit& h : v) k.emplace_back(h.t_offset_ns, h.ring);
    std::sort(k.begin(), k.end());
    return k;
  };
  EXPECT_EQ(keyset(a), keyset(c));
}

// The stride phase rotates with the outer-step index, so consecutive steps visit a
// different (moving) subset rather than a frozen one.
TEST(ResidualsLidarCap, OuterStepRotatesPhase) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 100;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 0;

  std::vector<LidarHit> hits;
  for (int i = 0; i < 1000; ++i) {
    hits.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
  }
  std::vector<LidarHit> s0;
  std::vector<LidarHit> s1;
  capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &s0);
  capHitsByNormalStrata(hits, cfg, 1, {}, 0.966, &s1);
  ASSERT_EQ(s0.size(), s1.size());
  // The two phases select overlapping-but-shifted subsets; their leading keys differ.
  EXPECT_NE(s0.front().t_offset_ns, s1.front().t_offset_ns);
}

// Weak-axis exemption: a corridor-like scene where one translation axis is starved of
// normals. The few points whose normal lies along that weak axis must all survive the
// cap regardless of budget, while the abundant orthogonal-wall points are decimated.
TEST(ResidualsLidarCap, WeakAxisPointsExemptFromCap) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 50;  // a tight cap that would otherwise thin everything
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 5;

  // Corridor: two long side walls (normals along +/-x) flood the scene, while the far
  // end-cap wall (normal along z) contributes only a handful of points. z is the weak
  // translation axis — only the scarce end-cap points constrain it.
  std::vector<LidarHit> hits;
  for (int i = 0; i < 2000; ++i) {
    const Eigen::Vector3d n = (i % 2 == 0) ? Eigen::Vector3d(1, 0, 0)
                                           : Eigen::Vector3d(-1, 0, 0);
    hits.push_back(makeHit(n, i, static_cast<std::uint16_t>(i % 64)));
  }
  // The 8 scarce weak-axis (z-normal) points, tagged with distinctive offsets.
  std::vector<std::int32_t> weak_offsets;
  for (int i = 0; i < 8; ++i) {
    const std::int32_t off = 100000 + i;
    weak_offsets.push_back(off);
    hits.push_back(makeHit(Eigen::Vector3d(0, 0, 1), off, 0));
  }

  const std::vector<Eigen::Vector3d> weak_axes = {Eigen::Vector3d(0, 0, 1)};
  std::vector<LidarHit> out;
  const LidarCapStats s = capHitsByNormalStrata(hits, cfg, 0, weak_axes, 0.966, &out);

  // Every weak-axis point survives even though the cap is far below their stratum's
  // proportional share would dictate.
  for (std::int32_t off : weak_offsets) {
    const bool present = std::any_of(out.begin(), out.end(), [off](const LidarHit& h) {
      return h.t_offset_ns == off;
    });
    EXPECT_TRUE(present) << "weak-axis point off=" << off << " was dropped";
  }
  // The exempt points push the kept total at least to the weak count; the abundant
  // x-wall points are still capped, so the total stays bounded near the budget.
  EXPECT_GE(s.kept, 8);
  EXPECT_LE(s.kept, cfg.max_lidar_factors + 8);
  EXPECT_GT(s.dropped, 0);
}

// Per-stratum floor: a sparsely-sampled normal stratum keeps at least
// min_factors_per_normal points (clamped to its population) even when a far more
// populous stratum competes for the same budget.
TEST(ResidualsLidarCap, PerStratumFloorProtectsRareNormal) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 120;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 20;

  std::vector<LidarHit> hits;
  // 1000 points on the +x wall.
  for (int i = 0; i < 1000; ++i) {
    hits.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
  }
  // 30 points on the +y wall: a rare stratum that the proportional split alone would
  // nearly starve, but the floor guarantees it at least min_factors_per_normal.
  for (int i = 0; i < 30; ++i) {
    hits.push_back(makeHit(Eigen::Vector3d(0, 1, 0), 200000 + i, 0));
  }

  std::vector<LidarHit> out;
  capHitsByNormalStrata(hits, cfg, 0, {}, 0.966, &out);
  const int y_kept = static_cast<int>(std::count_if(out.begin(), out.end(),
      [](const LidarHit& h) { return h.plane.n.y() > 0.5; }));
  EXPECT_GE(y_kept, cfg.min_factors_per_normal);
}

// Detection (not exemption): the weak-axis detector itself must flag exactly the
// starved translation axis. Two axes are flooded with normals (well-constrained), the
// third is starved, so the returned set must contain the starved axis and ONLY it.
TEST(WeakTranslationAxes, FlagsOnlyTheStarvedAxis) {
  constexpr double kappa = 10.0;
  std::vector<LidarHit> hits;
  // Flood +/-x and +/-y so lambda_x, lambda_y >> kappa: both are strongly observed.
  for (int i = 0; i < 1000; ++i) {
    hits.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
    hits.push_back(makeHit(Eigen::Vector3d(0, 1, 0), 1000 + i, 0));
  }
  // No z-normal hits at all: lambda_z = 0, the most degenerate possible weak axis.
  const std::vector<Eigen::Vector3d> weak = weakTranslationAxes(hits, kappa);

  ASSERT_EQ(weak.size(), 1u) << "exactly one axis (z) must be weak";
  // The weak eigenvector is (anti)parallel to z; x and y must be near zero.
  EXPECT_GT(std::abs(weak[0].z()), 0.99);
  EXPECT_LT(std::abs(weak[0].x()), 0.05);
  EXPECT_LT(std::abs(weak[0].y()), 0.05);
}

// Gate boundary: for weight-1 axis-aligned hits the per-axis eigenvalue equals the hit
// count, so lambda/(lambda+kappa) < 0.1 flips between 1 hit (lambda=1, 1/11<0.1 -> weak)
// and 2 hits (lambda=2, 2/12>=0.1 -> not weak) at kappa=10. One hit either side of the
// gate must flip the z axis's inclusion, proving the threshold (not just emptiness) bites.
TEST(WeakTranslationAxes, DegeneracyThresholdBoundaryFlipsInclusion) {
  constexpr double kappa = 10.0;
  const auto floodXY = [](std::vector<LidarHit>& h) {
    for (int i = 0; i < 1000; ++i) {
      h.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
      h.push_back(makeHit(Eigen::Vector3d(0, 1, 0), 1000 + i, 0));
    }
  };
  const auto zIsWeak = [&](const std::vector<Eigen::Vector3d>& w) {
    return std::any_of(w.begin(), w.end(),
                       [](const Eigen::Vector3d& a) { return std::abs(a.z()) > 0.99; });
  };

  // One z hit: lambda_z = 1, 1/(1+10) = 0.0909 < 0.1 -> z still weak.
  {
    std::vector<LidarHit> hits;
    floodXY(hits);
    hits.push_back(makeHit(Eigen::Vector3d(0, 0, 1), 999000, 0));
    const std::vector<Eigen::Vector3d> weak = weakTranslationAxes(hits, kappa);
    EXPECT_TRUE(zIsWeak(weak)) << "z below the gate must be flagged weak (lambda=1)";
  }
  // Two z hits: lambda_z = 2, 2/(2+10) = 0.1667 >= 0.1 -> z no longer weak. The single
  // extra hit on the scarce axis crosses the conditioning gate and flips inclusion.
  {
    std::vector<LidarHit> hits;
    floodXY(hits);
    hits.push_back(makeHit(Eigen::Vector3d(0, 0, 1), 999000, 0));
    hits.push_back(makeHit(Eigen::Vector3d(0, 0, 1), 999001, 0));
    const std::vector<Eigen::Vector3d> weak = weakTranslationAxes(hits, kappa);
    EXPECT_FALSE(zIsWeak(weak)) << "z above the gate must not be flagged weak (lambda=2)";
  }
}

// A non-positive degeneracy_thresh falls back to kappa = 10, so the boundary is the
// same as the explicit-10 case: 1 z hit is weak, 2 are not.
TEST(WeakTranslationAxes, NonPositiveThreshFallsBackToDefault) {
  const auto floodXY = [](std::vector<LidarHit>& h) {
    for (int i = 0; i < 1000; ++i) {
      h.push_back(makeHit(Eigen::Vector3d(1, 0, 0), i, 0));
      h.push_back(makeHit(Eigen::Vector3d(0, 1, 0), 1000 + i, 0));
    }
  };
  const auto zIsWeak = [&](const std::vector<Eigen::Vector3d>& w) {
    return std::any_of(w.begin(), w.end(),
                       [](const Eigen::Vector3d& a) { return std::abs(a.z()) > 0.99; });
  };

  std::vector<LidarHit> one;
  floodXY(one);
  one.push_back(makeHit(Eigen::Vector3d(0, 0, 1), 999000, 0));
  EXPECT_TRUE(zIsWeak(weakTranslationAxes(one, 0.0)));
  EXPECT_TRUE(zIsWeak(weakTranslationAxes(one, -5.0)));

  std::vector<LidarHit> two = one;
  two.push_back(makeHit(Eigen::Vector3d(0, 0, 1), 999001, 0));
  EXPECT_FALSE(zIsWeak(weakTranslationAxes(two, 0.0)));
}

// Detection feeds exemption end-to-end: a corridor scene (x,y flooded, z genuinely
// starved) run through the REAL detector and then the cap must keep the scarce z-normal
// evidence. This closes the loop the hand-fed exemption test left open: a mis-detection
// (wrong axis or empty set) would let the cap decimate the corridor-axis evidence. The z
// axis carries a single hit (lambda_z = 1, conditioning 1/11 < 0.1 -> detected weak), the
// one point that constrains the corridor axis and must therefore survive the cap.
TEST(WeakTranslationAxes, DetectedAxisSurvivesCap) {
  FrontendLidar cfg = makeCfg();
  cfg.max_lidar_factors = 50;
  cfg.normal_strata = 7;
  cfg.min_factors_per_normal = 5;

  std::vector<LidarHit> hits;
  for (int i = 0; i < 2000; ++i) {
    const Eigen::Vector3d n = (i % 2 == 0) ? Eigen::Vector3d(1, 0, 0)
                                           : Eigen::Vector3d(0, 1, 0);
    hits.push_back(makeHit(n, i, static_cast<std::uint16_t>(i % 64)));
  }
  // A single scarce z-normal point: the only evidence on the degenerate corridor axis.
  const std::int32_t weak_off = 100000;
  hits.push_back(makeHit(Eigen::Vector3d(0, 0, 1), weak_off, 0));

  // Detect the weak axis from the data itself, then cap with that detected set.
  const std::vector<Eigen::Vector3d> weak = weakTranslationAxes(hits, 10.0);
  ASSERT_EQ(weak.size(), 1u) << "the starved z axis must be detected as weak";
  EXPECT_GT(std::abs(weak[0].z()), 0.99) << "detector must flag the corridor (z) axis";

  std::vector<LidarHit> out;
  capHitsByNormalStrata(hits, cfg, 0, weak, 0.966, &out);
  const bool present = std::any_of(out.begin(), out.end(), [weak_off](const LidarHit& h) {
    return h.t_offset_ns == weak_off;
  });
  EXPECT_TRUE(present) << "the detected weak-axis point was dropped by the cap";

  // The exempt scarce point is not enough to fill the budget alone, so the abundant
  // walls are still decimated: the cap genuinely bit on the non-weak strata.
  EXPECT_GT(static_cast<int>(hits.size()) - static_cast<int>(out.size()), 0)
      << "cap must still drop abundant wall points";
}

// ---------------------------------------------------------------------------
// Analytic point-to-plane cost: derivative correctness and autodiff parity.
// ---------------------------------------------------------------------------

namespace {

// One randomized factor configuration: 4 SO(3) knots (quaternions), 4 R^3 knots,
// plus a random plane, point, extrinsic, and segment position.
struct AnalyticProbe {
  std::array<Eigen::Quaterniond, 4> q;
  std::array<Eigen::Vector3d, 4> p;
  Eigen::Vector3d point;
  Eigen::Vector3d n;
  double d = 0.0;
  Eigen::Quaterniond q_ext;
  Eigen::Vector3d t_ext;
  double u = 0.0;
  double weight = 1.0;
};

AnalyticProbe randomProbe(std::mt19937& rng, double rot_scale) {
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  auto randVec = [&] { return Eigen::Vector3d(uni(rng), uni(rng), uni(rng)); };
  AnalyticProbe pr;
  Eigen::Quaterniond q0 = Eigen::Quaterniond(uni(rng) + 1.5, uni(rng), uni(rng), uni(rng));
  q0.normalize();
  for (int j = 0; j < 4; ++j) {
    // Consecutive knots a bounded rotation apart, like a real warm-started window.
    pr.q[static_cast<std::size_t>(j)] = q0;
    q0 = (q0 * Sophus::SO3d::exp(rot_scale * randVec()).unit_quaternion()).normalized();
    pr.p[static_cast<std::size_t>(j)] = 2.0 * randVec();
  }
  pr.point = 3.0 * randVec();
  pr.n = randVec().normalized();
  pr.d = uni(rng);
  pr.q_ext = Eigen::Quaterniond(uni(rng) + 1.5, uni(rng), uni(rng), uni(rng)).normalized();
  pr.t_ext = 0.2 * randVec();
  pr.u = 0.5 * (uni(rng) + 1.0) * 0.999;
  pr.weight = 0.5 + 0.5 * (uni(rng) + 1.0);
  return pr;
}

}  // namespace

// Reference autodiff functor: the exact math the analytic cost replaces, evaluated
// through the same basalt helper. Autodiff Jacobians are exact (no finite-difference
// noise), so parity is checked at near-machine precision -- but only after projecting
// both ambient quaternion Jacobians through the manifold's PlusJacobian: the two
// formulations extend the function differently OFF the unit sphere, and only the
// tangent projection is the well-defined object the solver consumes.
namespace {

struct RefLidarFunctor {
  RefLidarFunctor(const AnalyticProbe& pr) : pr_(pr) {}
  template <class T>
  bool operator()(T const* const* params, T* residual) const {
    Sophus::SO3<T> R_w_fe;
    basalt::CeresSplineHelper<4>::evaluate_lie<T, Sophus::SO3>(params, pr_.u, 10.0, &R_w_fe);
    Eigen::Matrix<T, 3, 1> p_w_fe;
    basalt::CeresSplineHelper<4>::evaluate<T, 3, 0>(params + 4, pr_.u, 10.0, &p_w_fe);
    const Eigen::Matrix<T, 3, 1> q_fe = (pr_.q_ext * pr_.point + pr_.t_ext).cast<T>();
    residual[0] =
        T(pr_.weight) * (pr_.n.cast<T>().dot(R_w_fe * q_fe + p_w_fe) + T(pr_.d));
    return true;
  }
  AnalyticProbe pr_;
};

}  // namespace

// Analytic Jacobians must match autodiff exactly (tangent-projected), across random
// configurations including near-identity and large inter-knot rotations and the
// segment edges u ~ 0 / u ~ 1.
TEST(AnalyticLidarCost, JacobiansMatchAutodiffTangentProjected) {
  std::mt19937 rng(23);
  ceres::EigenQuaternionManifold quat_manifold;
  for (int trial = 0; trial < 60; ++trial) {
    const double rot_scale = (trial % 2 == 0) ? 1e-4 : 0.4;
    AnalyticProbe pr = randomProbe(rng, rot_scale);
    if (trial % 5 == 0) pr.u = 1e-6;
    if (trial % 7 == 0) pr.u = 1.0 - 1e-6;

    std::unique_ptr<ceres::CostFunction> analytic(meridian::ct::makeLidarPlaneCost(
        pr.point, pr.n, pr.d, pr.q_ext, pr.t_ext, pr.u, pr.weight));
    auto* ref = new ceres::DynamicAutoDiffCostFunction<RefLidarFunctor, 4>(
        new RefLidarFunctor(pr));
    for (int i = 0; i < 8; ++i) ref->AddParameterBlock(i < 4 ? 4 : 3);
    ref->SetNumResiduals(1);
    std::unique_ptr<ceres::CostFunction> reference(ref);

    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());

    double ja_store[4][4];
    double jr_store[4][4];
    double ja_p[4][3];
    double jr_p[4][3];
    std::vector<double*> ja_ptrs;
    std::vector<double*> jr_ptrs;
    for (int j = 0; j < 4; ++j) ja_ptrs.push_back(ja_store[j]);
    for (int j = 0; j < 4; ++j) ja_ptrs.push_back(ja_p[j]);
    for (int j = 0; j < 4; ++j) jr_ptrs.push_back(jr_store[j]);
    for (int j = 0; j < 4; ++j) jr_ptrs.push_back(jr_p[j]);

    double r_a = 0.0;
    double r_r = 0.0;
    ASSERT_TRUE(analytic->Evaluate(params.data(), &r_a, ja_ptrs.data()));
    ASSERT_TRUE(reference->Evaluate(params.data(), &r_r, jr_ptrs.data()));
    EXPECT_NEAR(r_a, r_r, 1e-12);

    for (int j = 0; j < 4; ++j) {
      // Tangent-project the quaternion-block rows through the manifold.
      Eigen::Matrix<double, 4, 3, Eigen::RowMajor> plus_jac;
      quat_manifold.PlusJacobian(params[j], plus_jac.data());
      const Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Ja(ja_store[j]);
      const Eigen::Map<Eigen::Matrix<double, 1, 4, Eigen::RowMajor>> Jr(jr_store[j]);
      const Eigen::Matrix<double, 1, 3> la = Ja * plus_jac;
      const Eigen::Matrix<double, 1, 3> lr = Jr * plus_jac;
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(la(0, c), lr(0, c), 1e-9 + 1e-9 * std::abs(lr(0, c)))
            << "trial " << trial << " so3 knot " << j << " col " << c;
      }
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(ja_p[j][c], jr_p[j][c], 1e-12)
            << "trial " << trial << " r3 knot " << j << " col " << c;
      }
    }
  }
}

// The analytic cost is the same function as the autodiff functor: residuals agree
// to machine precision on random inputs.
TEST(AnalyticLidarCost, ResidualMatchesAutodiffFunctor) {
  std::mt19937 rng(29);
  for (int trial = 0; trial < 20; ++trial) {
    AnalyticProbe pr = randomProbe(rng, 0.3);

    std::unique_ptr<ceres::CostFunction> analytic(meridian::ct::makeLidarPlaneCost(
        pr.point, pr.n, pr.d, pr.q_ext, pr.t_ext, pr.u, pr.weight));

    std::vector<const double*> params;
    for (int j = 0; j < 4; ++j) params.push_back(pr.q[static_cast<std::size_t>(j)].coeffs().data());
    for (int j = 0; j < 4; ++j) params.push_back(pr.p[static_cast<std::size_t>(j)].data());

    double r_analytic = 0.0;
    ASSERT_TRUE(analytic->Evaluate(params.data(), &r_analytic, nullptr));

    // Forward evaluation through the same basalt helper the autodiff path uses.
    Sophus::SO3d R_w_fe;
    basalt::CeresSplineHelper<4>::evaluate_lie<double, Sophus::SO3>(params.data(), pr.u, 10.0,
                                                                    &R_w_fe);
    Eigen::Vector3d p_w_fe;
    basalt::CeresSplineHelper<4>::evaluate<double, 3, 0>(params.data() + 4, pr.u, 10.0, &p_w_fe);
    const Eigen::Vector3d q_fe = pr.q_ext * pr.point + pr.t_ext;
    const double r_ref = pr.weight * (pr.n.dot(R_w_fe * q_fe + p_w_fe) + pr.d);

    EXPECT_NEAR(r_analytic, r_ref, 1e-12) << "trial " << trial;
  }
}

// ---------------------------------------------------------------------------
// iVox / ikd-Tree backend equivalence: the same stored point set must yield the
// same plane fit through both backends.
// ---------------------------------------------------------------------------

namespace {

}  // namespace

