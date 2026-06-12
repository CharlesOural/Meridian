#include <gtest/gtest.h>

#include <memory>
#include <vector>

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
    cfg.preprocess.lidar.point_filter_num = 1;
    // Exercise the L0/L1 stage and its hand-off to the front-end; the LIO front-end's
    // own estimation behaviour is covered in depth by its package tests.
    cfg.frontend.kind = meridian::FrontEndKind::Lio;
    auto sink = std::make_unique<RecordingSink>();
    rec = sink.get();
    pipeline = std::make_unique<MeridianPipeline>(cfg, std::move(sink));
    pipeline->set_group_sink([this](PreprocessedGroup&& g) { groups.push_back(std::move(g)); });
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

TEST(MeridianPipeline, ForwardsGroupsImmediately) {
  Fixture fx;
  const Timestamp t0 = 1'000 * kMs;

  // A few IMU samples, then a sweep [t0+50, t0+148]: no group yet, because the
  // aggregator's coverage gate needs an IMU sample past the sweep end.
  for (int i = 0; i < 5; ++i) fx.push_imu(t0 + i * 10 * kMs);
  fx.push_scan(t0 + 50 * kMs);
  EXPECT_TRUE(fx.groups.empty());

  // The covering sample closes the group and it must be forwarded immediately:
  // the pipeline holds nothing back ahead of the front-end.
  fx.push_imu(t0 + 150 * kMs);
  ASSERT_EQ(fx.groups.size(), 1u);

  // Two more covered sweeps forward two more groups, in arrival order.
  fx.push_scan(t0 + 200 * kMs);
  fx.push_imu(t0 + 300 * kMs);
  fx.push_scan(t0 + 350 * kMs);
  fx.push_imu(t0 + 460 * kMs);
  ASSERT_EQ(fx.groups.size(), 3u);

  EXPECT_LT(fx.groups[0].group.t_begin, fx.groups[1].group.t_begin);
  EXPECT_LT(fx.groups[1].group.t_begin, fx.groups[2].group.t_begin);
  for (const auto& g : fx.groups) {
    ASSERT_TRUE(g.group.scan.points);
    EXPECT_EQ(g.group.scan.points->size(), 50u);
    EXPECT_FALSE(g.group.imu.empty());
  }

  // Every forwarded group raised its size telemetry.
  int group_points = 0;
  for (const auto& s : fx.rec->scalars)
    if (s.key == std::string("pipeline/group_points")) ++group_points;
  EXPECT_EQ(group_points, 3);
}
