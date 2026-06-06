#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"
#include "meridian/debug/recording_sink.hpp"
#include "meridian/pipeline/meridian_pipeline.hpp"
#include "meridian/sensors/raw_frames.hpp"

using meridian::Config;
using meridian::MeridianPipeline;
using meridian::PipelineMode;
using meridian::PreprocessedGroup;
using meridian::RawImuFrame;
using meridian::RawLidarFrame;
using meridian::RawPoint;
using meridian::RecordingSink;
using meridian::Timestamp;

namespace {

constexpr Timestamp kNs = 1;
constexpr Timestamp kMs = 1'000'000 * kNs;

// Replay-mode pipeline (synchronous, deterministic) writing to a recording sink the
// test holds a borrowed pointer to.
struct Fixture {
  Fixture() {
    cfg.pipeline.mode = PipelineMode::Replay;
    cfg.preprocess.deskew.imu_init_count = 10;
    cfg.preprocess.lidar.point_filter_num = 1;
    // Exercise the L0/L1 stage and its hand-off to the front-end with the stable iEKF
    // estimator; the CT front-end's own behaviour is covered by its package tests.
    cfg.frontend.kind = meridian::FrontEndKind::IekfOracle;
    // Each flushed group carries only its straddling IMU sample, so let the front-end
    // initialise from a single static sample and emit its first keyframe immediately.
    cfg.frontend.init_time_s = 0.0;
    auto sink = std::make_unique<RecordingSink>();
    rec = sink.get();
    pipeline = std::make_unique<MeridianPipeline>(cfg, std::move(sink));
    pipeline->set_group_sink(
        [this](PreprocessedGroup&& g) { groups.push_back(std::move(g)); });
  }

  // One stationary IMU sample: gravity along -z in the body frame, no rotation.
  void push_imu(Timestamp t) {
    RawImuFrame f;
    f.device_ns = t;
    f.has_device_ns = true;
    f.host_arrival = t;
    f.acc = Eigen::Vector3d(0, 0, 9.81);
    f.gyro = Eigen::Vector3d::Zero();
    pipeline->ingest(f);
  }

  // One 100 ms sweep of valid points starting at t0.
  void push_scan(Timestamp t0) {
    std::vector<RawPoint> pts(50);
    for (std::size_t i = 0; i < pts.size(); ++i) {
      pts[i].x = 5.f;
      // Spread points >0.5 m apart in y so each lands in its own surf voxel and all 50
      // survive the downsample.
      pts[i].y = static_cast<float>(i) * 0.6f;
      pts[i].z = 1.f;
      pts[i].intensity = 100.f;
      pts[i].t = static_cast<std::uint32_t>(i * 2 * kMs);  // spans 0..98 ms
      pts[i].ring = static_cast<std::uint16_t>(i % 8);
      pts[i].range_m = 5.2f;
    }
    RawLidarFrame f;
    f.host_arrival = t0;
    f.device_ns_first_column = t0;
    f.has_device_ns = true;
    f.pts = pts;
    pipeline->ingest(f);
  }

  Config cfg;
  RecordingSink* rec = nullptr;
  std::unique_ptr<MeridianPipeline> pipeline;
  std::vector<PreprocessedGroup> groups;
};

}  // namespace

TEST(MeridianPipeline, BuffersSweepsUntilImuInitThenFlushesDeskewed) {
  Fixture fx;
  const Timestamp t0 = 1'000 * kMs;

  // 5 IMU samples (init needs 10), a sweep [t0+50, t0+148], then one IMU sample past
  // the sweep end: the group closes (coverage gate satisfied) but must be HELD because
  // the bootstrap has not converged (only 6 samples folded).
  for (int i = 0; i < 5; ++i) fx.push_imu(t0 + i * 10 * kMs);
  fx.push_scan(t0 + 50 * kMs);
  fx.push_imu(t0 + 150 * kMs);
  EXPECT_TRUE(fx.groups.empty());

  // Four more IMU samples complete the init (10 total). The flush happens on the next
  // group: a second sweep + its coverage sample must release BOTH groups, in order.
  for (int i = 16; i < 20; ++i) fx.push_imu(t0 + i * 10 * kMs);
  EXPECT_TRUE(fx.groups.empty());
  fx.push_scan(t0 + 200 * kMs);
  fx.push_imu(t0 + 300 * kMs);

  ASSERT_EQ(fx.groups.size(), 2u);
  EXPECT_LT(fx.groups[0].group.t_begin, fx.groups[1].group.t_begin);
  for (const auto& g : fx.groups) {
    ASSERT_TRUE(g.group.scan.points);
    EXPECT_EQ(g.group.scan.points->size(), 50u);
    ASSERT_TRUE(g.deskewed.has_value());
    EXPECT_EQ(g.deskewed->points->size(), 50u);
  }
  // The first flushed group precedes any front-end keyframe, so it is cold-start; the
  // front-end emits its first keyframe on that group, so the second group is no longer
  // cold-start (the front-end now owns the trajectory).
  EXPECT_TRUE(fx.groups[0].cold_start);
  EXPECT_FALSE(fx.groups[1].cold_start);

  // The init-converged event fired exactly once.
  int init_events = 0;
  for (const auto& e : fx.rec->events)
    if (e.tag == std::string("preprocess/imu_init_done")) ++init_events;
  EXPECT_EQ(init_events, 1);

  // The deskewed cloud went out on the body/scan telemetry key.
  bool saw_body_scan = false;
  for (const auto& c : fx.rec->clouds)
    if (c.key == std::string("body/scan")) saw_body_scan = true;
  EXPECT_TRUE(saw_body_scan);
}

TEST(MeridianPipeline, StationaryDeskewIsNearIdentity) {
  Fixture fx;
  const Timestamp t0 = 1'000 * kMs;

  // Converge init first, then deliver one covered sweep.
  for (int i = 0; i < 12; ++i) fx.push_imu(t0 + i * 10 * kMs);
  fx.push_scan(t0 + 130 * kMs);
  fx.push_imu(t0 + 240 * kMs);
  fx.push_imu(t0 + 250 * kMs);

  ASSERT_EQ(fx.groups.size(), 1u);
  ASSERT_TRUE(fx.groups[0].deskewed.has_value());
  const auto& in = *fx.groups[0].group.scan.points;
  const auto& out = *fx.groups[0].deskewed->points;
  ASSERT_EQ(in.size(), out.size());
  // A stationary rig must leave the geometry (numerically) unchanged.
  for (std::size_t i = 0; i < in.size(); ++i) {
    EXPECT_NEAR(in[i].xyz.x(), out[i].xyz.x(), 1e-3f);
    EXPECT_NEAR(in[i].xyz.y(), out[i].xyz.y(), 1e-3f);
    EXPECT_NEAR(in[i].xyz.z(), out[i].xyz.z(), 1e-3f);
  }
}
