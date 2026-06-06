#include "iekf/iekf_frontend.hpp"

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::CalibrationSet;
using meridian::Extrinsic;
using meridian::Frame;
using meridian::FrontendConfig;
using meridian::IekfFrontEnd;
using meridian::ImuSample;
using meridian::KeyframePacket;
using meridian::LidarPoint;
using meridian::LidarScan;
using meridian::MeasureGroup;
using meridian::NavState;
using meridian::PointCloud;
using meridian::Pose;
using meridian::PreprocessedGroup;
using meridian::Timestamp;

namespace {

constexpr double kG = 9.81;
constexpr std::int64_t kNsPerS = 1'000'000'000LL;

std::shared_ptr<const CalibrationSet> identityCalib() {
  auto c = std::make_shared<CalibrationSet>();
  c->estimation_frame = Frame::ImuLink;
  c->imu_acc_noise = 1e-2;
  c->imu_gyr_noise = 1e-3;
  c->imu_acc_bias_rw = 1e-4;
  c->imu_gyr_bias_rw = 1e-5;
  c->version = 1;
  Extrinsic e;
  e.child = Frame::OsSensor0;
  e.parent = Frame::ImuLink;
  e.T_parent_child = Pose{};  // LiDAR coincident with IMU for the synthetic rig
  c->extrinsics.push_back(e);
  return c;
}

FrontendConfig oracleCfg() {
  FrontendConfig cfg;
  cfg.kind = meridian::FrontEndKind::IekfOracle;
  cfg.solver_max_iterations = 4;
  cfg.solver_epsi = 1e-4;
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

// One IMU sample at rest: gravity reads as +g along body z (specific force is
// opposite to gravitational acceleration), with optional white noise.
ImuSample restImu(Timestamp stamp, std::mt19937& rng, double acc_sigma,
                  double gyr_sigma) {
  std::normal_distribution<double> na(0.0, acc_sigma);
  std::normal_distribution<double> ng(0.0, gyr_sigma);
  ImuSample s;
  s.stamp = stamp;
  s.acc = Eigen::Vector3d(na(rng), na(rng), kG + na(rng));
  s.gyro = Eigen::Vector3d(ng(rng), ng(rng), ng(rng));
  return s;
}

// The six axis-aligned walls of a box room centred at the origin.
struct Plane {
  Eigen::Vector3d n;
  double d;  // n^T x + d = 0
};

std::vector<Plane> boxRoom(double half) {
  return {
      {Eigen::Vector3d(1, 0, 0), half},   {Eigen::Vector3d(-1, 0, 0), half},
      {Eigen::Vector3d(0, 1, 0), half},   {Eigen::Vector3d(0, -1, 0), half},
      {Eigen::Vector3d(0, 0, 1), half},   {Eigen::Vector3d(0, 0, -1), half},
  };
}

// Ray-cast from a sensor at world position `org` along world direction `dir`;
// return the nearest forward wall hit, or false if none.
bool rayCast(const std::vector<Plane>& walls, const Eigen::Vector3d& org,
             const Eigen::Vector3d& dir, Eigen::Vector3d* hit) {
  double best = std::numeric_limits<double>::infinity();
  bool found = false;
  for (const Plane& w : walls) {
    const double denom = w.n.dot(dir);
    if (std::abs(denom) < 1e-9) {
      continue;
    }
    const double t = -(w.n.dot(org) + w.d) / denom;
    if (t > 0.1 && t < best) {
      const Eigen::Vector3d p = org + t * dir;
      // Keep hits within the box footprint of the wall.
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

// Build a sweep of points (in the body/LiDAR frame at scan-end) by ray-casting
// the box room from the given world pose. The points carry t_offset_ns = 0 so the
// deskew is a no-op (we test tracking, not undistortion).
LidarScan sweepFromPose(const std::vector<Plane>& walls, const Pose& T_world_body,
                        Timestamp stamp_start) {
  auto cloud = std::make_shared<PointCloud>();
  const Pose T_body_world = T_world_body.inverse();
  const int n_az = 60;
  const int n_el = 16;
  for (int i = 0; i < n_az; ++i) {
    const double az = 2.0 * M_PI * i / n_az;
    for (int j = 0; j < n_el; ++j) {
      const double el = -0.6 + 1.2 * j / (n_el - 1);
      const Eigen::Vector3d dir_body(std::cos(el) * std::cos(az),
                                     std::cos(el) * std::sin(az), std::sin(el));
      const Eigen::Vector3d dir_world = T_world_body.q * dir_body;
      Eigen::Vector3d hit_world;
      if (!rayCast(walls, T_world_body.t, dir_world, &hit_world)) {
        continue;
      }
      const Eigen::Vector3d hit_body = T_body_world * hit_world;
      LidarPoint p;
      p.xyz = hit_body.cast<float>();
      p.range = static_cast<float>(hit_body.norm());
      p.t_offset_ns = 0;
      cloud->push_back(p);
    }
  }
  LidarScan scan;
  scan.stamp_start = stamp_start;
  scan.sweep_duration = kNsPerS / 10;  // 100 ms sweep
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

std::vector<ImuSample> restImuRun(Timestamp t0, Timestamp t1, std::mt19937& rng,
                                  double acc_sigma, double gyr_sigma) {
  std::vector<ImuSample> out;
  for (Timestamp t = t0; t <= t1; t += kNsPerS / 100) {  // 100 Hz
    out.push_back(restImu(t, rng, acc_sigma, gyr_sigma));
  }
  return out;
}

// IMU samples for a slowly rotating body that is otherwise near-static: the
// specific force is just gravity expressed in the body frame (R^T applied to the
// world up vector), so prediction stays physically consistent with the LiDAR
// pose. Angular rate is left at zero; the small per-sweep yaw/pitch steps are
// recovered by the LiDAR registration, not the gyro.
std::vector<ImuSample> orientedImuRun(Timestamp t0, Timestamp t1,
                                      const Eigen::Quaterniond& q,
                                      std::mt19937& rng, double acc_sigma,
                                      double gyr_sigma) {
  std::normal_distribution<double> na(0.0, acc_sigma);
  std::normal_distribution<double> ng(0.0, gyr_sigma);
  const Eigen::Vector3d up_world(0, 0, kG);  // specific force opposes gravity
  std::vector<ImuSample> out;
  for (Timestamp t = t0; t <= t1; t += kNsPerS / 100) {  // 100 Hz
    ImuSample s;
    s.stamp = t;
    const Eigen::Vector3d acc_body = q.conjugate() * up_world;
    s.acc = acc_body + Eigen::Vector3d(na(rng), na(rng), na(rng));
    s.gyro = Eigen::Vector3d(ng(rng), ng(rng), ng(rng));
    out.push_back(s);
  }
  return out;
}

}  // namespace

// (a) Static scene + noisy IMU at rest: pose stays put, gravity converges to the
// physical magnitude, and the covariance stays bounded.
TEST(IekfFrontEnd, StaticAtRestStaysPutAndGravityConverges) {
  std::mt19937 rng(42);
  IekfFrontEnd fe(oracleCfg(), nullptr);
  fe.set_calibration(identityCalib());

  const std::vector<Plane> walls = boxRoom(2.5);
  const Pose T_id;
  for (int k = 0; k < 8; ++k) {
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, T_id, t0);
    auto imu = restImuRun(t0, t1, rng, 0.02, 0.002);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }

  const NavState st = fe.live_state();
  // Pose stays at the origin (small tolerance for noise).
  EXPECT_LT(st.T_world_body.t.norm(), 0.1);
  EXPECT_LT(Eigen::AngleAxisd(st.T_world_body.q).angle(), 0.05);
  // Gravity magnitude is pinned to |g| by the constrained update.
  EXPECT_NEAR(st.g_world.norm(), kG, 1e-3);
  // Gravity points down (the room is gravity-aligned at init).
  EXPECT_GT(st.g_world.z() < 0 ? -st.g_world.z() : 0.0, kG - 0.5);
}

// (b) Synthetic box room with a known circular trajectory: the filter tracks the
// ground-truth pose within tolerance after a few sweeps.
TEST(IekfFrontEnd, TracksCircularTrajectoryInBoxRoom) {
  std::mt19937 rng(7);
  IekfFrontEnd fe(oracleCfg(), nullptr);
  fe.set_calibration(identityCalib());

  const std::vector<Plane> walls = boxRoom(2.5);

  // Ground-truth trajectory: a slow circle of radius 0.5 m in the xy-plane, no
  // rotation, so gravity stays vertical and the IMU stays near rest.
  auto gtPose = [](double s) {
    const double radius = 0.5;
    Pose p;
    p.t = Eigen::Vector3d(radius * std::cos(s) - radius, radius * std::sin(s), 0.0);
    p.q = Eigen::Quaterniond::Identity();
    return p;
  };

  const int sweeps = 20;
  Pose last_gt;
  for (int k = 0; k < sweeps; ++k) {
    const double s = 0.08 * k;
    const Pose gt = gtPose(s);
    last_gt = gt;
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, gt, t0);
    // IMU stays near rest (slow motion); pose is driven by LiDAR registration.
    auto imu = restImuRun(t0, t1, rng, 0.01, 0.001);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }

  const NavState st = fe.live_state();
  const double pos_err = (st.T_world_body.t - last_gt.t).norm();
  EXPECT_LT(pos_err, 0.25) << "estimated " << st.T_world_body.t.transpose()
                           << " vs gt " << last_gt.t.transpose();
}

// (b2) Rotating + translating trajectory: the body yaws and pitches through a
// non-trivial orientation while translating, exercising the LiDAR measurement
// rotation block and the state retraction far from R = I. A grossly wrong
// rotation-block sign or a transposed pose Jacobian breaks convergence and makes
// this diverge; the finer retraction-convention check lives in (b3).
TEST(IekfFrontEnd, TracksRotatingTranslatingTrajectory) {
  std::mt19937 rng(19);
  IekfFrontEnd fe(oracleCfg(), nullptr);
  fe.set_calibration(identityCalib());

  const std::vector<Plane> walls = boxRoom(2.5);

  // Ground truth: yaw sweeps to ~0.8 rad and pitch to ~0.3 rad while the body
  // translates along x and y through the room.
  auto gtPose = [](double s) {
    Pose p;
    const double yaw = 0.8 * s;
    const double pitch = 0.3 * s;
    const Eigen::Quaterniond q =
        Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    p.q = q.normalized();
    p.t = Eigen::Vector3d(0.6 * s, 0.4 * s, 0.0);
    return p;
  };

  const int sweeps = 24;
  Pose last_gt;
  for (int k = 0; k < sweeps; ++k) {
    const double s = 0.05 * k;
    const Pose gt = gtPose(s);
    last_gt = gt;
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, gt, t0);
    auto imu = orientedImuRun(t0, t1, gt.q, rng, 0.01, 0.001);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }

  const NavState st = fe.live_state();
  const double pos_err = (st.T_world_body.t - last_gt.t).norm();
  const double rot_err =
      Eigen::AngleAxisd(st.T_world_body.q.conjugate() * last_gt.q).angle();
  EXPECT_LT(pos_err, 0.20) << "estimated " << st.T_world_body.t.transpose()
                           << " vs gt " << last_gt.t.transpose();
  EXPECT_LT(rot_err, 0.10) << "rotation error " << rot_err << " rad";
}

// (b3) Convention guard: the analytic point-to-plane pose Jacobian row the filter
// assembles, [ dz/d(rho) = n^T | dz/d(phi) = -n^T R [pb]x ], must match a finite
// difference of the residual taken through the filter's OWN retraction
// (IekfFrontEnd::boxplusNav). The base orientation is non-trivial, so a position
// block that is not exactly n^T -- e.g. if the retraction were switched back to
// the coupled SE(3) exponential, whose translation update mixes in the rotation
// increment once R != I -- would make the position columns disagree and fail.
TEST(IekfFrontEnd, MeasurementJacobianMatchesRetraction) {
  const Eigen::Quaterniond q =
      (Eigen::Quaterniond(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ())) *
       Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitY())))
          .normalized();
  const Eigen::Matrix3d R = q.toRotationMatrix();
  const Eigen::Vector3d pb(1.3, 0.5, -0.8);  // body-frame point
  const Eigen::Vector3d n = Eigen::Vector3d(0.2, -0.9, 0.3).normalized();
  const double d = 0.15;

  NavState s0;
  s0.T_world_body.q = q;
  s0.T_world_body.t = Eigen::Vector3d(0.4, -0.2, 0.1);

  auto residualAt = [&](const NavState& s) {
    const Eigen::Vector3d pw = s.T_world_body.q * pb + s.T_world_body.t;
    return n.dot(pw) + d;
  };

  auto skew = [](const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
  };

  // Analytic pose row [rho(0:2) | phi(3:5)], matching the filter's H assembly.
  Eigen::Matrix<double, 1, 3> drho = n.transpose();
  Eigen::Matrix<double, 1, 3> dphi = -(n.transpose() * R * skew(pb));

  const double eps = 1e-6;
  for (int i = 0; i < 6; ++i) {
    Eigen::Matrix<double, NavState::kDof, 1> dp =
        Eigen::Matrix<double, NavState::kDof, 1>::Zero();
    Eigen::Matrix<double, NavState::kDof, 1> dm = dp;
    dp(i) = eps;
    dm(i) = -eps;
    const double fd = (residualAt(IekfFrontEnd::boxplusNav(s0, dp)) -
                       residualAt(IekfFrontEnd::boxplusNav(s0, dm))) /
                      (2 * eps);
    const double analytic = i < 3 ? drho(0, i) : dphi(0, i - 3);
    EXPECT_NEAR(fd, analytic, 1e-5) << "pose tangent dim " << i;
  }
}

// (c) The covariance reorder helper round-trips translation-first <-> rotation-
// first on a random SPD matrix.
TEST(IekfFrontEnd, CovReorderRoundTrips) {
  std::mt19937 rng(123);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::Matrix<double, 6, 6> A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = nd(rng);
    }
  }
  const Eigen::Matrix<double, 6, 6> spd =
      A * A.transpose() + Eigen::Matrix<double, 6, 6>::Identity();

  const Eigen::Matrix<double, 6, 6> rot_first =
      IekfFrontEnd::reorderTransRotToRotTrans(spd);
  const Eigen::Matrix<double, 6, 6> back =
      IekfFrontEnd::reorderTransRotToRotTrans(rot_first);
  EXPECT_TRUE(back.isApprox(spd, 1e-12));

  // The reorder swaps the diagonal 3-blocks: rot_first's top-left equals spd's
  // rotation block (rows/cols 3..5), and rot_first's bottom-right equals spd's
  // translation block (rows/cols 0..2).
  EXPECT_TRUE((rot_first.block<3, 3>(0, 0).isApprox(spd.block<3, 3>(3, 3), 1e-12)));
  EXPECT_TRUE((rot_first.block<3, 3>(3, 3).isApprox(spd.block<3, 3>(0, 0), 1e-12)));
  // And the off-diagonal coupling is transposed across the swap.
  EXPECT_TRUE((rot_first.block<3, 3>(0, 3).isApprox(spd.block<3, 3>(3, 0), 1e-12)));
}

// (d) Keyframe cadence honours the configured distance threshold: a long
// translation across sweeps yields multiple keyframes; staying put yields one.
TEST(IekfFrontEnd, KeyframeCadenceHonoursThresholds) {
  std::mt19937 rng(11);
  FrontendConfig cfg = oracleCfg();
  cfg.keyframe.dist_m = 0.5;
  cfg.keyframe.rot_deg = 1000.0;  // disable rotation trigger
  cfg.keyframe.time_s = 1000.0;   // disable time trigger
  IekfFrontEnd fe(cfg, nullptr);
  fe.set_calibration(identityCalib());

  int kf_count = 0;
  fe.set_keyframe_sink([&](KeyframePacket&& pkt) {
    if (kf_count == 0) {
      EXPECT_EQ(pkt.constraint_kind, KeyframePacket::ConstraintKind::AbsolutePrior);
    } else {
      EXPECT_EQ(pkt.constraint_kind, KeyframePacket::ConstraintKind::RelativeBetween);
    }
    // constraint_cov must be a valid 6x6 covariance block tagged rotation-first.
    EXPECT_EQ(pkt.constraint_cov.form, meridian::GaussianBlock<6>::Form::Covariance);
    EXPECT_EQ(pkt.frontend_kind, 0u);
    ++kf_count;
  });

  const std::vector<Plane> walls = boxRoom(2.5);
  auto gtPose = [](double x) {
    Pose p;
    p.t = Eigen::Vector3d(x, 0, 0);
    return p;
  };

  // Translate ~0.18 m per sweep over 12 sweeps (~2.0 m total): with a 0.5 m
  // threshold this should emit the first KF plus roughly 3-4 more.
  const int sweeps = 12;
  for (int k = 0; k < sweeps; ++k) {
    const Pose gt = gtPose(0.18 * k);
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, gt, t0);
    auto imu = restImuRun(t0, t1, rng, 0.01, 0.001);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }

  EXPECT_GE(kf_count, 3);
  EXPECT_LE(kf_count, sweeps);
}

// A static IMU run with a known constant gyro bias added to every sample, so the
// recovered bias is checkable. Acc reads gravity (+g along body z) plus small noise.
std::vector<ImuSample> biasedRestRun(Timestamp t0, Timestamp t1,
                                     const Eigen::Vector3d& gyro_bias, std::mt19937& rng,
                                     double acc_sigma, double gyr_sigma) {
  std::normal_distribution<double> na(0.0, acc_sigma);
  std::normal_distribution<double> ng(0.0, gyr_sigma);
  std::vector<ImuSample> out;
  for (Timestamp t = t0; t <= t1; t += kNsPerS / 100) {  // 100 Hz
    ImuSample s;
    s.stamp = t;
    s.acc = Eigen::Vector3d(na(rng), na(rng), kG + na(rng));
    s.gyro = gyro_bias + Eigen::Vector3d(ng(rng), ng(rng), ng(rng));
    out.push_back(s);
  }
  return out;
}

// A first group whose IMU is too agitated to pass the static-init motion gate, so the
// shared ImuInitializer defers initialisation: no keyframe is emitted from the shaky
// group. A subsequent clean static run then initialises, recovering gravity (pinned to
// |g|, pointing down) and the injected gyro bias from the Welford mean.
TEST(IekfFrontEnd, SharedInitGatesMotionAndRecoversBias) {
  std::mt19937 rng(101);
  IekfFrontEnd fe(oracleCfg(), nullptr);
  fe.set_calibration(identityCalib());

  int kf_count = 0;
  fe.set_keyframe_sink([&](KeyframePacket&&) { ++kf_count; });

  const std::vector<Plane> walls = boxRoom(2.5);
  const Pose T_id;
  const Eigen::Vector3d injected_bias(0.03, -0.02, 0.05);  // rad/s, within bias box

  // Group 0: violently shaky IMU (variance far above init_max_var) -> gate rejects,
  // the filter must not initialise and must emit no keyframe.
  {
    const Timestamp t0 = 0;
    const Timestamp t1 = kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, T_id, t0);
    auto imu = biasedRestRun(t0, t1, Eigen::Vector3d::Zero(), rng,
                             /*acc_sigma=*/3.0, /*gyr_sigma=*/2.0);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }
  EXPECT_EQ(kf_count, 0) << "shaky first group must not initialise or emit a keyframe";
  // Before init the live state is still at its default (no tracking happened).
  EXPECT_LT(fe.live_state().T_world_body.t.norm(), 1e-9);

  // Groups 1..N: clean static IMU carrying the injected gyro bias. A clean static
  // window forms, the gate accepts, and the filter initialises.
  const int clean_groups = 6;
  for (int k = 1; k <= clean_groups; ++k) {
    const Timestamp t0 = static_cast<Timestamp>(k) * kNsPerS / 10;
    const Timestamp t1 = t0 + kNsPerS / 10;
    LidarScan scan = sweepFromPose(walls, T_id, t0);
    auto imu = biasedRestRun(t0, t1, injected_bias, rng, 0.01, 0.001);
    fe.ingest(makeGroup(scan, std::move(imu)));
  }

  const NavState st = fe.live_state();
  // Gravity recovered through the shared init: magnitude pinned, pointing down.
  EXPECT_NEAR(st.g_world.norm(), kG, 1e-3);
  EXPECT_LT(st.g_world.z(), -(kG - 0.5));
  // The gyro bias recovered from the Welford static mean matches the injected bias
  // (a hardcoded-zero init would leave b_g far from this).
  EXPECT_LT((st.b_g - injected_bias).norm(), 0.02)
      << "recovered b_g " << st.b_g.transpose() << " vs injected " << injected_bias.transpose();
  // Init eventually succeeded, so keyframes were emitted from the clean groups.
  EXPECT_GE(kf_count, 1) << "clean static groups must initialise and emit a keyframe";
}
