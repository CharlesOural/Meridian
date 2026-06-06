#include "meridian/preprocess/imu_only_deskew.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/preprocess/imu_init.hpp"

using meridian::Extrinsic;
using meridian::ImuInitState;
using meridian::ImuOnlyDeskew;
using meridian::ImuSample;
using meridian::LidarPoint;
using meridian::LidarScan;
using meridian::PointCloud;
using meridian::Pose;

namespace {

ImuSample mkImu(const Eigen::Vector3d& gyro, std::int64_t t) {
  ImuSample s;
  s.gyro = gyro;
  s.acc = Eigen::Vector3d::Zero();  // gravity cancelled below so world accel is zero
  s.stamp = t;
  return s;
}

}  // namespace

// Constant-rate yaw rotation, no translation. A single static world point observed in
// the rotating body frame at several sweep times must deskew to one body-at-end point.
TEST(ImuOnlyDeskew, ConstantRotationDeskewsStaticPoint) {
  ImuInitState init;
  // Set gravity so that pushImu's world-accel (R*acc + g) is zero for acc == 0.
  init.gravity = Eigen::Vector3d::Zero();
  init.gyro_bias = Eigen::Vector3d::Zero();

  Extrinsic ext;  // identity T_imu_lidar (LiDAR == body)
  const Eigen::Vector3d wz(0.0, 0.0, 0.5);  // 0.5 rad/s yaw

  ImuOnlyDeskew dk(init, ext, Pose{}, Eigen::Vector3d::Zero());

  // Feed a dense IMU buffer spanning the sweep [0, 100 ms].
  const std::int64_t dt_ns = 5'000'000;  // 5 ms
  for (std::int64_t t = 0; t <= 100'000'000; t += dt_ns) {
    dk.pushImu(mkImu(wz, t));
  }
  dk.setAnchor(100'000'000);  // compensate to scan end

  // The true static world point (in the ref frame).
  const Eigen::Vector3d p_world(2.0, 1.0, 0.5);

  // Build a scan: each point is p_world expressed in the body frame at its own time.
  std::vector<LidarPoint> pts;
  std::vector<std::int64_t> times = {0, 25'000'000, 50'000'000, 75'000'000, 100'000'000};
  for (std::int64_t t : times) {
    Pose Tri;
    ASSERT_TRUE(dk.poseAt(t, &Tri));
    const Eigen::Vector3d p_body = Tri.inverse() * p_world;  // body-at-t coordinate
    LidarPoint lp;
    lp.xyz = p_body.cast<float>();
    lp.t_offset_ns = static_cast<std::int32_t>(t);
    pts.push_back(lp);
  }

  LidarScan in;
  in.stamp_start = 0;
  in.sweep_duration = 100'000'000;
  in.points = std::make_shared<const PointCloud>(std::move(pts));

  LidarScan out;
  ASSERT_TRUE(dk.deskew(in, &out));
  ASSERT_TRUE(out.points);
  ASSERT_EQ(out.points->size(), 5u);

  // Reference: p_world expressed in the body frame at scan end.
  Pose Tend;
  ASSERT_TRUE(dk.poseAt(100'000'000, &Tend));
  const Eigen::Vector3f expected = (Tend.inverse() * p_world).cast<float>();

  for (const LidarPoint& p : *out.points) {
    EXPECT_NEAR(p.xyz.x(), expected.x(), 1e-3f);
    EXPECT_NEAR(p.xyz.y(), expected.y(), 1e-3f);
    EXPECT_NEAR(p.xyz.z(), expected.z(), 1e-3f);
  }
}

TEST(ImuOnlyDeskew, ReturnsRestartWhenHorizonDoesNotCover) {
  ImuInitState init;
  init.gravity = Eigen::Vector3d::Zero();
  ImuOnlyDeskew dk(init, Extrinsic{}, Pose{}, Eigen::Vector3d::Zero());
  dk.pushImu(mkImu(Eigen::Vector3d::Zero(), 0));
  dk.pushImu(mkImu(Eigen::Vector3d::Zero(), 10'000'000));

  LidarScan in;
  in.stamp_start = 0;
  in.sweep_duration = 50'000'000;  // extends past the integrated horizon (10 ms)
  std::vector<LidarPoint> pts(1);
  pts[0].t_offset_ns = 40'000'000;
  in.points = std::make_shared<const PointCloud>(std::move(pts));

  LidarScan out;
  EXPECT_FALSE(dk.deskew(in, &out));  // signals window restart
}

// integrateSweep pads the IMU horizon to the sweep with bounded zero-order holds: a
// sweep whose ends fall just outside the raw IMU grid (head sample after t_begin, last
// sample before t_end, each within one period) must still deskew, with the anchor at
// t_end. A gap beyond one period is a real hole and must keep failing.
TEST(ImuOnlyDeskew, IntegrateSweepHoldsHorizonToSweepEnds) {
  ImuInitState init;
  init.gravity = Eigen::Vector3d::Zero();
  init.gyro_bias = Eigen::Vector3d::Zero();
  ImuOnlyDeskew dk(init, Extrinsic{}, Pose{}, Eigen::Vector3d::Zero());

  // IMU grid 10..90 ms at 10 ms cadence (period 10 ms). Sweep [5 ms, 95 ms]: the head
  // sample (10 ms) is 5 ms after t_begin and the tail (90 ms) is 5 ms before t_end, both
  // within one period, so the holds extend the horizon to cover the sweep.
  std::vector<ImuSample> imu;
  for (std::int64_t t = 10'000'000; t <= 90'000'000; t += 10'000'000) {
    imu.push_back(mkImu(Eigen::Vector3d(0, 0, 0.5), t));
  }
  const std::int64_t t_begin = 5'000'000;
  const std::int64_t t_end = 95'000'000;
  dk.integrateSweep(imu, t_begin, t_end);

  EXPECT_EQ(dk.anchor(), t_end);
  EXPECT_TRUE(dk.validHorizonCovers(t_begin, t_end));
  Pose at_begin;
  Pose at_end;
  EXPECT_TRUE(dk.poseAt(t_begin, &at_begin));  // covered by the head hold
  EXPECT_TRUE(dk.poseAt(t_end, &at_end));      // covered by the tail ZOH

  // A point at the very start of the sweep must deskew (no restart).
  std::vector<LidarPoint> pts(1);
  pts[0].xyz = Eigen::Vector3f(2.f, 0.f, 0.f);
  pts[0].t_offset_ns = 0;  // at t_begin (offset is from stamp_start == t_begin)
  LidarScan in;
  in.stamp_start = t_begin;
  in.sweep_duration = t_end - t_begin;
  in.points = std::make_shared<const PointCloud>(std::move(pts));

  LidarScan out;
  EXPECT_TRUE(dk.deskew(in, &out));
}

// A head gap larger than one period is a genuine IMU hole: the head hold must not engage,
// so the sweep start stays outside the horizon and deskew restarts.
TEST(ImuOnlyDeskew, IntegrateSweepLeavesLargeHeadGapToFail) {
  ImuInitState init;
  init.gravity = Eigen::Vector3d::Zero();
  ImuOnlyDeskew dk(init, Extrinsic{}, Pose{}, Eigen::Vector3d::Zero());

  std::vector<ImuSample> imu;
  for (std::int64_t t = 50'000'000; t <= 90'000'000; t += 10'000'000) {
    imu.push_back(mkImu(Eigen::Vector3d::Zero(), t));  // period 10 ms, starts at 50 ms
  }
  const std::int64_t t_begin = 0;        // first sample is 50 ms late: a real hole
  const std::int64_t t_end = 95'000'000;
  dk.integrateSweep(imu, t_begin, t_end);

  EXPECT_FALSE(dk.validHorizonCovers(t_begin, t_end));
}

TEST(ImuOnlyDeskew, NeverMutatesInputBuffer) {
  ImuInitState init;
  init.gravity = Eigen::Vector3d::Zero();
  ImuOnlyDeskew dk(init, Extrinsic{}, Pose{}, Eigen::Vector3d::Zero());
  for (std::int64_t t = 0; t <= 100'000'000; t += 5'000'000) {
    dk.pushImu(mkImu(Eigen::Vector3d(0, 0, 0.5), t));
  }
  dk.setAnchor(100'000'000);

  std::vector<LidarPoint> pts(1);
  pts[0].xyz = Eigen::Vector3f(2.f, 0.f, 0.f);
  pts[0].t_offset_ns = 0;
  auto shared = std::make_shared<const PointCloud>(std::move(pts));

  LidarScan in;
  in.stamp_start = 0;
  in.sweep_duration = 100'000'000;
  in.points = shared;

  LidarScan out;
  ASSERT_TRUE(dk.deskew(in, &out));
  // Input buffer is untouched; the output is a distinct allocation.
  EXPECT_FLOAT_EQ((*shared)[0].xyz.x(), 2.f);
  EXPECT_NE(out.points.get(), shared.get());
}
