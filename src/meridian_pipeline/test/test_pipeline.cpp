#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include "meridian/config/config.hpp"
#include "meridian/debug/recording_sink.hpp"
#include "meridian/pipeline/meridian_pipeline.hpp"
#include "meridian/sensors/raw_frames.hpp"

using meridian::Config;
using meridian::GnssFix;
using meridian::KeyframePacket;
using meridian::Marker;
using meridian::MeridianPipeline;
using meridian::PipelineMode;
using meridian::PreprocessedGroup;
using meridian::RawGnssFrame;
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
  explicit Fixture(bool enable_place = false) {
    cfg.pipeline.mode = PipelineMode::Replay;
    cfg.preprocess.lidar.point_filter_num = 1;
    // Exercise the L0/L1 stage and its hand-off to the front-end; the LIO front-end's
    // own estimation behaviour is covered in depth by its package tests.
    cfg.frontend.kind = meridian::FrontEndKind::Lio;
    // Drive the front-end to emit keyframes from the minimal stationary synthetic sweeps
    // these tests feed, so the back-end wiring downstream is actually exercised: init from
    // the first static sample, accept the sparse line sweep (~20 surf voxels), and let the
    // per-sweep time gate (no translation/rotation here) trip a keyframe every sweep.
    cfg.frontend.lio.init_stationary_s = 0.0;
    cfg.frontend.lio.min_keypoints = 5;
    cfg.frontend.keyframe.time_s = 0.05;
    cfg.place.enable = enable_place;  // L5 loop detector on the back-end path
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

  // One healthy RTK fix that passes every gate check (strong fix type, enough sats,
  // tight covariance; the velocity spoof check passes with no velocity source).
  void push_gnss(Timestamp t) {
    RawGnssFrame f;
    f.gps_second_ns = t;
    f.has_pps_reference = true;
    f.host_arrival = t;
    f.lat_deg = 48.0;
    f.lon_deg = 2.0;
    f.alt_m = 100.0;
    f.cov_enu = Eigen::Matrix3d::Identity() * 0.01;
    f.fix = GnssFix::FixType::RTK_Fixed;
    f.num_sats = 12;
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

TEST(MeridianPipeline, BackendTapSeesKeyframesThenAnchoredGnss) {
  Fixture fx;
  std::vector<MeridianPipeline::BackendItem> items;
  fx.pipeline->set_backend_tap(
      [&](const MeridianPipeline::BackendItem& item) { items.push_back(item); });
  std::vector<std::uint64_t> sink_ids;
  fx.pipeline->set_keyframe_sink([&](KeyframePacket&& kf) { sink_ids.push_back(kf.id); });

  // The front-end bootstraps on the first solved sweep and emits its first keyframe on the
  // second, so drive several sweeps; one window carries an accepted GNSS fix.
  const Timestamp t0 = 1'000 * kMs;
  for (int i = 0; i < 5; ++i) fx.push_imu(t0 + i * 10 * kMs);
  fx.push_scan(t0 + 50 * kMs);
  fx.push_imu(t0 + 150 * kMs);
  for (int i = 16; i < 20; ++i) fx.push_imu(t0 + i * 10 * kMs);
  fx.push_scan(t0 + 200 * kMs);
  fx.push_imu(t0 + 300 * kMs);
  fx.push_scan(t0 + 350 * kMs);
  fx.push_imu(t0 + 460 * kMs);
  fx.push_scan(t0 + 500 * kMs);
  fx.push_gnss(t0 + 550 * kMs);  // inside scan-500's sweep window, after a keyframe has emitted
  fx.push_imu(t0 + 610 * kMs);
  ASSERT_GE(fx.groups.size(), 3u);

  // The tap saw the same keyframes the wrapper sink did, in the same order, and each
  // keyframe item was fed before the sink's move consumed it.
  std::vector<std::uint64_t> tap_ids;
  for (const auto& item : items) {
    if (const auto* kf = std::get_if<KeyframePacket>(&item)) tap_ids.push_back(kf->id);
  }
  ASSERT_FALSE(tap_ids.empty());
  EXPECT_EQ(tap_ids, sink_ids);

  // Exactly one GNSS item, fed after the keyframe its sweep produced and anchored to
  // the freshest keyframe id at that point.
  std::uint64_t last_kf_seen = 0;
  int gnss_items = 0;
  for (const auto& item : items) {
    if (const auto* kf = std::get_if<KeyframePacket>(&item)) {
      last_kf_seen = kf->id;
    } else if (const auto* g = std::get_if<MeridianPipeline::GnssForBackend>(&item)) {
      ++gnss_items;
      EXPECT_EQ(g->nearest_kf_id, last_kf_seen);
      EXPECT_DOUBLE_EQ(g->fix.lat_deg, 48.0);
      EXPECT_EQ(g->fix.fix, GnssFix::FixType::RTK_Fixed);
    }
  }
  EXPECT_EQ(gnss_items, 1);

  // The back-end is on by default and the inline replay driver folds the graph on
  // every keyframe, so the corrected trajectory is already populated between ingests.
  EXPECT_TRUE(fx.pipeline->backend_enabled());
  EXPECT_FALSE(fx.pipeline->corrected_trajectory().empty());
}

TEST(MeridianPipeline, LoopDetectorWiringRunsAndStaysDeterministic) {
  // With L5 enabled the detector runs on the back-end driver after every keyframe fold; the
  // short fixture produces no revisit, so it emits nothing, but the feed/detect/fold plumbing
  // must run cleanly and remain bit-reproducible in replay.
  const auto run = []() {
    Fixture fx(/*enable_place=*/true);
    const Timestamp t0 = 1'000 * kMs;
    for (int i = 0; i < 5; ++i) fx.push_imu(t0 + i * 10 * kMs);
    fx.push_scan(t0 + 50 * kMs);
    fx.push_imu(t0 + 150 * kMs);
    for (int i = 16; i < 20; ++i) fx.push_imu(t0 + i * 10 * kMs);
    fx.push_scan(t0 + 200 * kMs);
    fx.push_imu(t0 + 300 * kMs);
    fx.push_scan(t0 + 350 * kMs);
    fx.push_imu(t0 + 460 * kMs);
    fx.push_scan(t0 + 500 * kMs);
    fx.push_imu(t0 + 610 * kMs);
    return fx.pipeline->corrected_trajectory();
  };
  const std::vector<meridian::StampedPose> a = run();
  const std::vector<meridian::StampedPose> b = run();
  EXPECT_FALSE(a.empty());
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].stamp, b[i].stamp);
    EXPECT_DOUBLE_EQ(a[i].T_map_body.t.x(), b[i].T_map_body.t.x());
    EXPECT_DOUBLE_EQ(a[i].T_map_body.t.y(), b[i].T_map_body.t.y());
    EXPECT_DOUBLE_EQ(a[i].T_map_body.t.z(), b[i].T_map_body.t.z());
  }
}

TEST(MeridianPipeline, DisabledBackendReportsEmptyTrajectory) {
  Config cfg;
  cfg.pipeline.mode = PipelineMode::Replay;
  cfg.frontend.kind = meridian::FrontEndKind::Lio;
  cfg.backend.enable = false;
  MeridianPipeline pipeline(cfg, nullptr);
  EXPECT_FALSE(pipeline.backend_enabled());
  EXPECT_TRUE(pipeline.corrected_trajectory().empty());
}
