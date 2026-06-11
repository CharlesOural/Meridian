#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <random>
#include <sophus/so3.hpp>
#include <vector>

#include "ct/ct_frontend.hpp"
#include "ct/keyframe_finalizer.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/calib/intrinsics.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::CalibrationSet;
using meridian::CameraFrame;
using meridian::CtFrontEnd;
using meridian::Duration;
using meridian::Extrinsic;
using meridian::Frame;
using meridian::FrontendConfig;
using meridian::ImuSample;
using meridian::IntrinsicsCamera;
using meridian::KeyframeFinalizer;
using meridian::KeyframeJob;
using meridian::KeyframePacket;
using meridian::LidarPoint;
using meridian::LidarScan;
using meridian::PointCloud;
using meridian::Pose;
using meridian::PreprocessedGroup;
using meridian::Timestamp;
using meridian::to_seconds;

namespace {

constexpr double kG = 9.81;
constexpr std::int64_t kNsPerS = 1'000'000'000LL;

FrontendConfig ctCfg() {
  FrontendConfig cfg;
  cfg.kind = meridian::FrontEndKind::CtLivo;
  cfg.solver_max_iterations = 8;
  cfg.solver_epsi = 1e-4;
  cfg.spline.knot_dt_ms = 100.0;
  cfg.spline.window_knots = 6;
  cfg.lidar.voxel_map_m = 0.2;
  cfg.lidar.num_match_points = 5;
  cfg.lidar.max_match_dist_sq = 4.0;
  cfg.lidar.plane_thresh = 0.1;
  cfg.lidar.point_cov = 1e-3;
  cfg.keyframe.dist_m = 1.0;
  cfg.keyframe.rot_deg = 10.0;
  cfg.keyframe.time_s = 1.0;
  return cfg;
}

struct Plane {
  Eigen::Vector3d n;
  double d;
};

std::vector<Plane> boxRoom(double half) {
  return {
      {Eigen::Vector3d(1, 0, 0), half}, {Eigen::Vector3d(-1, 0, 0), half},
      {Eigen::Vector3d(0, 1, 0), half}, {Eigen::Vector3d(0, -1, 0), half},
      {Eigen::Vector3d(0, 0, 1), half}, {Eigen::Vector3d(0, 0, -1), half},
  };
}

bool rayCast(const std::vector<Plane>& walls, const Eigen::Vector3d& org,
             const Eigen::Vector3d& dir, Eigen::Vector3d* hit) {
  double best = std::numeric_limits<double>::infinity();
  bool found = false;
  for (const Plane& w : walls) {
    const double denom = w.n.dot(dir);
    if (std::abs(denom) < 1e-9) continue;
    const double t = -(w.n.dot(org) + w.d) / denom;
    if (t > 0.1 && t < best) {
      const Eigen::Vector3d p = org + t * dir;
      bool inside = true;
      for (int a = 0; a < 3; ++a) {
        if (std::abs(p[a]) > 2.51) {
          inside = false;
          break;
        }
      }
      if (inside) {
        best = t;
        *hit = p;
        found = true;
      }
    }
  }
  return found;
}

using GtFn = std::function<Pose(Timestamp)>;

LidarScan sweepFromGt(const std::vector<Plane>& walls, const GtFn& gt, Timestamp stamp_start) {
  const Duration sweep = kNsPerS / 10;
  auto cloud = std::make_shared<PointCloud>();
  const int n_az = 60;
  const int n_el = 16;
  int idx = 0;
  const int total = n_az * n_el;
  for (int i = 0; i < n_az; ++i) {
    const double az = 2.0 * M_PI * i / n_az;
    for (int j = 0; j < n_el; ++j, ++idx) {
      const std::int32_t off = static_cast<std::int32_t>(static_cast<double>(sweep) * idx / total);
      const Timestamp t = stamp_start + off;
      const Pose T_world_body = gt(t);
      const Pose T_body_world = T_world_body.inverse();
      const double el = -0.6 + 1.2 * j / (n_el - 1);
      const Eigen::Vector3d dir_body(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az),
                                     std::sin(el));
      const Eigen::Vector3d dir_world = T_world_body.q * dir_body;
      Eigen::Vector3d hit_world;
      if (!rayCast(walls, T_world_body.t, dir_world, &hit_world)) continue;
      const Eigen::Vector3d hit_body = T_body_world * hit_world;
      LidarPoint p;
      p.xyz = hit_body.cast<float>();
      p.range = static_cast<float>(hit_body.norm());
      p.t_offset_ns = off;
      cloud->push_back(p);
    }
  }
  LidarScan scan;
  scan.stamp_start = stamp_start;
  scan.sweep_duration = sweep;
  scan.sensor_frame = Frame::OsSensor0;
  scan.points = cloud;
  return scan;
}

PreprocessedGroup makeGroup(const LidarScan& scan, std::vector<ImuSample> imu) {
  PreprocessedGroup g;
  g.group.scan = scan;
  g.group.t_begin = scan.stamp_start;
  g.group.t_end = scan.stamp_start + scan.sweep_duration;
  g.group.imu = std::move(imu);
  g.cold_start = true;
  return g;
}

std::vector<ImuSample> imuFromGt(const GtFn& gt, Timestamp t0, Timestamp t1, std::mt19937& rng,
                                 double acc_sigma, double gyr_sigma) {
  std::normal_distribution<double> na(0.0, acc_sigma);
  std::normal_distribution<double> ng(0.0, gyr_sigma);
  const Eigen::Vector3d g_world(0, 0, -kG);
  const Duration step = kNsPerS / 100;
  const double h = to_seconds(step);
  std::vector<ImuSample> out;
  for (Timestamp t = t0; t <= t1; t += step) {
    const Pose Tm = gt(t - step);
    const Pose T0 = gt(t);
    const Pose Tp = gt(t + step);
    const Eigen::Vector3d omega = Sophus::SO3d(Tm.q.conjugate() * Tp.q).log() / (2.0 * h);
    const Eigen::Vector3d acc_world = (Tp.t - 2.0 * T0.t + Tm.t) / (h * h);
    const Eigen::Vector3d acc_body = T0.q.conjugate() * (acc_world - g_world);
    ImuSample s;
    s.stamp = t;
    s.acc = acc_body + Eigen::Vector3d(na(rng), na(rng), na(rng));
    s.gyro = omega + Eigen::Vector3d(ng(rng), ng(rng), ng(rng));
    out.push_back(s);
  }
  return out;
}

// Camera looking forward (+x of the body).
Pose bodyCamExtrinsic() {
  Eigen::Matrix3d R;
  R.col(0) = Eigen::Vector3d(0, -1, 0);
  R.col(1) = Eigen::Vector3d(0, 0, -1);
  R.col(2) = Eigen::Vector3d(1, 0, 0);
  Pose T;
  T.q = Eigen::Quaterniond(R).normalized();
  T.t = Eigen::Vector3d::Zero();
  return T;
}

constexpr int kImgW = 320;
constexpr int kImgH = 240;
constexpr double kFx = 200.0;
constexpr double kFy = 200.0;

std::shared_ptr<const CalibrationSet> cameraCalib() {
  auto c = std::make_shared<CalibrationSet>();
  c->estimation_frame = Frame::ImuLink;
  c->imu_acc_noise = 5e-2;
  c->imu_gyr_noise = 5e-3;
  c->imu_acc_bias_rw = 1e-4;
  c->imu_gyr_bias_rw = 1e-5;
  c->version = 1;
  Extrinsic el;
  el.child = Frame::OsSensor0;
  el.parent = Frame::ImuLink;
  el.T_parent_child = Pose{};
  c->extrinsics.push_back(el);
  Extrinsic ec;
  ec.child = Frame::CamLink;
  ec.parent = Frame::ImuLink;
  ec.T_parent_child = bodyCamExtrinsic();
  c->extrinsics.push_back(ec);
  IntrinsicsCamera k;
  k.fx = kFx;
  k.fy = kFy;
  k.cx = kImgW / 2.0;
  k.cy = kImgH / 2.0;
  k.model = IntrinsicsCamera::Distortion::None;
  k.width = kImgW;
  k.height = kImgH;
  k.inv_expo_prior = 1.0;
  k.inv_expo_std = 0.0;
  c->cam_intrinsics[0] = k;
  return c;
}

double wallTexture(const Eigen::Vector3d& p) {
  const double s = std::sin(0.6 * p.x()) * std::cos(0.5 * p.y()) +
                   0.5 * std::sin(0.45 * p.z() + 0.2 * p.x()) +
                   0.4 * std::cos(0.35 * p.y() - 0.25 * p.z());
  return 127.5 + 55.0 * s;
}

cv::Mat renderBoxRoom(const std::vector<Plane>& walls, const Pose& T_world_body) {
  cv::Mat img(kImgH, kImgW, CV_8UC1, cv::Scalar(0));
  const Pose T_world_cam = T_world_body * bodyCamExtrinsic();
  const Eigen::Vector3d org = T_world_cam.t;
  for (int v = 0; v < kImgH; ++v) {
    for (int u = 0; u < kImgW; ++u) {
      const Eigen::Vector3d ray_c((u + 0.5 - kImgW / 2.0) / kFx, (v + 0.5 - kImgH / 2.0) / kFy,
                                  1.0);
      const Eigen::Vector3d dir = (T_world_cam.q * ray_c).normalized();
      Eigen::Vector3d hit;
      if (!rayCast(walls, org, dir, &hit)) continue;
      const int q = std::clamp(static_cast<int>(std::lround(wallTexture(hit))), 0, 255);
      img.at<std::uint8_t>(v, u) = static_cast<std::uint8_t>(q);
    }
  }
  return img;
}

CameraFrame frameFromImage(const cv::Mat& img, Timestamp stamp) {
  CameraFrame f;
  f.stamp = stamp;
  f.sensor_id = 0;
  f.sensor_frame = Frame::CamLink;
  f.width = img.cols;
  f.height = img.rows;
  f.encoding = CameraFrame::Encoding::Mono8;
  f.data = std::make_shared<std::vector<std::uint8_t>>(
      img.data, img.data + static_cast<std::size_t>(img.total()));
  f.exposure_s = 0.01f;
  f.gain = 1.0f;
  return f;
}

std::vector<PreprocessedGroup> buildVisualStream(const std::vector<Plane>& walls, int sweeps,
                                                 const GtFn& gt, std::mt19937& rng,
                                                 bool with_image) {
  std::vector<PreprocessedGroup> stream;
  stream.reserve(sweeps);
  for (int k = 0; k < sweeps; ++k) {
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    const Timestamp t_mid = t0 + kNsPerS / 20;
    LidarScan scan = sweepFromGt(walls, gt, t0);
    auto imu = imuFromGt(gt, t0, t1, rng, 0.01, 0.001);
    PreprocessedGroup g = makeGroup(scan, std::move(imu));
    if (with_image) {
      cv::Mat img = renderBoxRoom(walls, gt(t_mid));
      g.group.image = frameFromImage(img, t_mid);
    }
    stream.push_back(std::move(g));
  }
  return stream;
}

GtFn yawCrawlGt() {
  return [](Timestamp t) {
    const double s = 0.5 * to_seconds(t);
    Pose p;
    p.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.5 * s, Eigen::Vector3d::UnitZ())).normalized();
    p.t = Eigen::Vector3d(0.5 * s, 0.3 * s, 0.0);
    return p;
  };
}

double maxAbsDiff(const Eigen::Matrix<double, 6, 6>& a, const Eigen::Matrix<double, 6, 6>& b) {
  return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

// THE MERGE GATE. On every live keyframe sweep the probe computes the pose marginal two
// ways on the identical solved state: (a) the synchronous routine on the live problem,
// (b) the worker routine on the rebuilt snapshot clone. The two 6x6 blocks must be
// bit-identical (max-abs-diff exactly 0.0 -- per-element equality, not a tolerance), for
// both the AbsolutePrior (first KF) and RelativeBetween (later KFs) branches. LiDAR + IMU
// + visual + marginalization prior are all present on the captured sweeps.
TEST(KeyframeFinalizer, PoseMarginalBitExactSyncVsWorker) {
  std::mt19937 rng(7);
  const auto walls = boxRoom(2.5);
  const auto gt = yawCrawlGt();
  const int sweeps = 20;
  auto stream = buildVisualStream(walls, sweeps, gt, rng, /*with_image=*/true);

  // Live front-end (deterministic=false), but the solve pinned to one thread so the
  // solved trajectory is reproducible and the only quantity under test is the marginal
  // path. The parity probe records both computations per keyframe sweep.
  CtFrontEnd fe(ctCfg(), cameraCalib(), nullptr, /*deterministic=*/false);
  fe.set_force_single_thread_for_test(true);
  fe.set_parity_probe_for_test(true);
  std::atomic<int> emitted{0};
  fe.set_keyframe_sink([&](KeyframePacket&&) { ++emitted; });

  for (const auto& g : stream) fe.ingest(g);

  const auto& results = fe.parity_probe_results_for_test();
  ASSERT_GE(results.size(), 3u) << "stream produced too few keyframes to exercise the gate";

  int absolute = 0;
  int relative = 0;
  int compared = 0;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    // Both paths run the same routine on the same state, so they succeed or fall back
    // together (the first keyframe's pose knots are still gauge-pinned -> both fail and
    // fall back to the LiDAR-only marginal identically).
    EXPECT_EQ(r.sync_ok, r.worker_ok) << "sync/worker disagreed on success at keyframe " << i;
    if (r.sync_ok && r.worker_ok) {
      // Bit-exact: per-element equality over the 36 doubles, NOT a tolerance.
      EXPECT_EQ(maxAbsDiff(r.sync_cov, r.worker_cov), 0.0)
          << "marginal diverged at keyframe " << i << " ("
          << (r.absolute_prior ? "AbsolutePrior" : "RelativeBetween") << ")";
      ++compared;
    }
    if (r.absolute_prior) {
      ++absolute;
    } else {
      ++relative;
    }
  }
  EXPECT_GE(compared, 1) << "no keyframe produced a comparable window posterior";
  EXPECT_GE(absolute, 1) << "the AbsolutePrior branch was never exercised";
  EXPECT_GE(relative, 1) << "the RelativeBetween branch was never exercised";
}

// The same bit-exact merge gate with adaptive knot density engaged: a yaw fast
// enough to hold gear >= 2 exercises multi-knot segments, the tail re-knot between
// capture and the next solve, the index-based gauge pins, and the drain slide --
// every pointer the capture remaps must keep matching its clone by deque index, or
// the rebuilt marginal silently diverges. Bit-identical, not a tolerance.
TEST(KeyframeFinalizer, PoseMarginalBitExactSyncVsWorkerAtHighGear) {
  std::mt19937 rng(31);
  const auto walls = boxRoom(2.5);
  // Still through the init window, then a steady 1.2 rad/s yaw: gear 2 from the
  // first finalized segment onward (mean rate above the 0.5 edge, below 1.5).
  const double spin_start_s = 0.1;
  const GtFn gt = [&](Timestamp t) {
    const double ts = to_seconds(t);
    const double ang = ts <= spin_start_s ? 0.0 : 1.2 * (ts - spin_start_s);
    Pose p;
    p.q = Eigen::Quaterniond(Eigen::AngleAxisd(ang, Eigen::Vector3d::UnitZ())).normalized();
    return p;
  };
  const int sweeps = 20;
  auto stream = buildVisualStream(walls, sweeps, gt, rng, /*with_image=*/true);

  FrontendConfig cfg = ctCfg();
  cfg.ncp.enabled = true;
  cfg.ncp.n_cp_max = 3;
  cfg.ncp.omega_thresh = {0.5, 1.5};
  cfg.ncp.accel_thresh = {1.0, 3.0};
  CtFrontEnd fe(cfg, cameraCalib(), nullptr, /*deterministic=*/false);
  fe.set_force_single_thread_for_test(true);
  fe.set_parity_probe_for_test(true);
  std::atomic<int> emitted{0};
  fe.set_keyframe_sink([&](KeyframePacket&&) { ++emitted; });

  for (const auto& g : stream) fe.ingest(g);

  const auto& results = fe.parity_probe_results_for_test();
  ASSERT_GE(results.size(), 2u) << "stream produced too few keyframes to exercise the gate";

  int compared = 0;
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    EXPECT_EQ(r.sync_ok, r.worker_ok) << "sync/worker disagreed on success at keyframe " << i;
    if (r.sync_ok && r.worker_ok) {
      EXPECT_EQ(maxAbsDiff(r.sync_cov, r.worker_cov), 0.0)
          << "marginal diverged at keyframe " << i << " at gear > 1";
      ++compared;
    }
  }
  EXPECT_GE(compared, 1) << "no keyframe produced a comparable window posterior";
}

// The async live path emits every keyframe with a filled covariance, in id order, and
// the chain has no holes: a worker run reproduces the rel_cov chain end to end.
TEST(KeyframeFinalizer, LivePathEmitsAllKeyframesInOrderWithFilledCov) {
  std::mt19937 rng(11);
  const auto walls = boxRoom(2.5);
  const auto gt = yawCrawlGt();
  const int sweeps = 20;
  auto stream = buildVisualStream(walls, sweeps, gt, rng, /*with_image=*/true);

  std::vector<KeyframePacket> packets;
  std::mutex pkt_mtx;
  {
    CtFrontEnd fe(ctCfg(), cameraCalib(), nullptr, /*deterministic=*/false);
    fe.set_keyframe_sink([&](KeyframePacket&& kf) {
      std::lock_guard<std::mutex> lock(pkt_mtx);
      packets.push_back(std::move(kf));
    });
    for (const auto& g : stream) fe.ingest(g);
    // fe destructor stops the worker: drains every in-flight job, joins, no deadlock.
  }

  ASSERT_GE(packets.size(), 3u);
  // Monotonic id order, no holes, every cov filled (finite, symmetric, positive diag).
  for (std::size_t i = 0; i < packets.size(); ++i) {
    EXPECT_EQ(packets[i].id, static_cast<std::uint64_t>(i)) << "id out of order / a hole at " << i;
    const auto& M = packets[i].constraint_cov.M;
    EXPECT_TRUE(M.allFinite()) << "non-finite cov at keyframe " << i;
    // The marginal is explicitly symmetrized; only float round-off remains.
    EXPECT_LT((M - M.transpose()).cwiseAbs().maxCoeff(), 1e-18) << "cov not symmetric at " << i;
    for (int d = 0; d < 6; ++d) {
      EXPECT_GT(M(d, d), 0.0) << "non-positive variance at keyframe " << i << " axis " << d;
    }
    EXPECT_EQ(packets[i].constraint_cov.form, meridian::GaussianBlock<6>::Form::Covariance);
  }
  // The first keyframe is the absolute anchor; later ones chain relatively.
  EXPECT_EQ(packets.front().constraint_kind, KeyframePacket::ConstraintKind::AbsolutePrior);
}

// Teardown: stop() drains every queued job, joins the worker, and emits each in-flight
// packet with a filled cov -- no deadlock, no lost keyframe. Driven directly on the
// standalone finalizer with a trivial empty-window job (the marginal falls back, which
// still produces a valid filled cov).
TEST(KeyframeFinalizer, StopDrainsAndJoins) {
  std::vector<KeyframePacket> out;
  std::mutex mtx;
  KeyframeFinalizer fin(
      [&](KeyframePacket&& kf) {
        std::lock_guard<std::mutex> lock(mtx);
        out.push_back(std::move(kf));
      },
      nullptr, /*capacity=*/4);
  fin.start();

  // Submit several minimal jobs; each rebuilds an (empty) problem, the marginal falls
  // back, and the packet is forwarded with the fallback cov filled.
  for (int i = 0; i < 6; ++i) {
    KeyframeJob job;
    job.spline = nullptr;  // empty: poseMarginal falls back to the job's fallback cov
    job.bias = std::make_unique<meridian::ct::BiasKnots>(0, kNsPerS / 10, 2);
    job.stamp = static_cast<Timestamp>(i + 1) * kNsPerS / 10;
    job.pkt.id = static_cast<std::uint64_t>(i);
    job.pkt.stamp = job.stamp;
    job.absolute_prior = (i == 0);
    job.fallback_pose_cov = Eigen::Matrix<double, 6, 6>::Identity() * (1.0 + i);
    // No spline -> poseMarginal returns false; finalize uses fallback_pose_cov.
    fin.submit(std::move(job));
  }
  fin.stop();  // drains + joins; must not deadlock

  std::lock_guard<std::mutex> lock(mtx);
  EXPECT_EQ(out.size(), 6u) << "a job was lost across teardown";
  for (std::size_t i = 0; i < out.size(); ++i) {
    EXPECT_EQ(out[i].id, static_cast<std::uint64_t>(i));
    EXPECT_TRUE(out[i].constraint_cov.M.allFinite());
  }
}
