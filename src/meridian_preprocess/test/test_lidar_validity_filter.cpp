#include "meridian/preprocess/ilidar_preprocessor.hpp"

#include <cmath>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/debug/recording_sink.hpp"

using meridian::Extrinsic;
using meridian::LidarPoint;
using meridian::LidarScan;
using meridian::makeLidarPreprocessor;
using meridian::PointCloud;
using meridian::PreprocessConfig;
using meridian::RecordingSink;

namespace {

LidarPoint mk(float x, float y, float z, float t_ns, float intensity = 1.f) {
  LidarPoint p;
  p.xyz = Eigen::Vector3f(x, y, z);
  p.t_offset_ns = static_cast<std::int32_t>(t_ns);
  p.intensity = intensity;
  return p;
}

LidarScan scanOf(std::vector<LidarPoint> pts) {
  LidarScan s;
  s.stamp_start = 1'000;
  s.points = std::make_shared<const PointCloud>(std::move(pts));
  return s;
}

PreprocessConfig defaultCfg() {
  PreprocessConfig c;
  c.lidar.blind = 0.5;
  c.lidar.det_range = 120.0;
  c.lidar.point_filter_num = 1;  // no decimation unless a test opts in
  c.lidar.intensity_gate = false;
  return c;
}

}  // namespace

TEST(LidarValidityFilter, DropsNanBlindFarInOrder) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(std::nanf(""), 0, 0, 10));        // NaN -> drop
  pts.push_back(mk(0.1f, 0.0f, 0.0f, 20));           // blind (r=0.1 < 0.5) -> drop
  pts.push_back(mk(200.f, 0.f, 0.f, 30));            // far (r=200 > 120) -> drop
  pts.push_back(mk(3.f, 0.f, 0.f, 40));              // valid

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  EXPECT_EQ(out.points->size(), 1u);
  EXPECT_FLOAT_EQ((*out.points)[0].xyz.x(), 3.f);
}

TEST(LidarValidityFilter, IntensityGateOnlyWhenEnabled) {
  PreprocessConfig cfg = defaultCfg();
  cfg.lidar.intensity_gate = true;
  cfg.lidar.i_min = 1.0;
  cfg.lidar.i_max = 100.0;
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(3.f, 0.f, 0.f, 10, 0.f));    // below i_min -> drop
  pts.push_back(mk(3.f, 0.f, 0.f, 20, 50.f));   // in range -> keep
  pts.push_back(mk(3.f, 0.f, 0.f, 30, 500.f));  // above i_max -> drop

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  EXPECT_EQ(out.points->size(), 1u);
  EXPECT_FLOAT_EQ((*out.points)[0].intensity, 50.f);
}

TEST(LidarValidityFilter, DecimationOverValidPointsOnly) {
  PreprocessConfig cfg = defaultCfg();
  cfg.lidar.point_filter_num = 2;  // keep the 2nd, 4th, ... VALID point
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  // Interleave invalids so index-based decimation would keep the wrong points.
  std::vector<LidarPoint> pts;
  pts.push_back(mk(0.1f, 0.f, 0.f, 1));   // blind (invalid, not counted)
  pts.push_back(mk(2.f, 0.f, 0.f, 2));    // valid #1 -> dropped (1 % 2 != 0)
  pts.push_back(mk(0.1f, 0.f, 0.f, 3));   // blind
  pts.push_back(mk(3.f, 0.f, 0.f, 4));    // valid #2 -> KEPT
  pts.push_back(mk(4.f, 0.f, 0.f, 5));    // valid #3 -> dropped
  pts.push_back(mk(5.f, 0.f, 0.f, 6));    // valid #4 -> KEPT

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_TRUE(out.points);
  ASSERT_EQ(out.points->size(), 2u);
  EXPECT_FLOAT_EQ((*out.points)[0].xyz.x(), 3.f);
  EXPECT_FLOAT_EQ((*out.points)[1].xyz.x(), 5.f);
}

TEST(LidarValidityFilter, SortsByTimeOffset) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  // Distinct voxels (0.5 m edge) so the surf-voxel stage keeps all three; the test is
  // about the post-stage time-sort, not downsampling.
  std::vector<LidarPoint> pts;
  pts.push_back(mk(7.f, 0.f, 0.f, 30));
  pts.push_back(mk(3.f, 0.f, 0.f, 10));
  pts.push_back(mk(5.f, 0.f, 0.f, 20));

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_EQ(out.points->size(), 3u);
  EXPECT_LT((*out.points)[0].t_offset_ns, (*out.points)[1].t_offset_ns);
  EXPECT_LT((*out.points)[1].t_offset_ns, (*out.points)[2].t_offset_ns);
}

TEST(LidarValidityFilter, ReusesBufferWhenNothingDroppedAndSorted) {
  PreprocessConfig cfg = defaultCfg();  // point_filter_num == 1
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(2.f, 0.f, 0.f, 10));
  pts.push_back(mk(3.f, 0.f, 0.f, 20));
  pts.push_back(mk(4.f, 0.f, 0.f, 30));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_EQ(out.points.get(), before);  // same Shared-immutable buffer, no copy
}

TEST(LidarValidityFilter, AllocatesNewBufferWhenSomethingDropped) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(0.1f, 0.f, 0.f, 10));  // blind -> dropped
  pts.push_back(mk(3.f, 0.f, 0.f, 20));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_NE(out.points.get(), before);  // a fresh buffer was allocated
  EXPECT_EQ(out.points->size(), 1u);
}

TEST(LidarValidityFilter, AllocatesNewBufferWhenUnsortedEvenIfNothingDropped) {
  PreprocessConfig cfg = defaultCfg();
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(3.f, 0.f, 0.f, 30));
  pts.push_back(mk(2.f, 0.f, 0.f, 10));
  LidarScan in = scanOf(pts);
  const PointCloud* before = in.points.get();

  LidarScan out = pp->process(in);
  EXPECT_NE(out.points.get(), before);  // reordering forces a copy
  EXPECT_EQ(out.points->size(), 2u);
}

namespace {

// A config whose validity gate passes everything in the test cells (blind small, no
// decimation), so each test isolates the surf-voxel stage.
PreprocessConfig surfCfg(double voxel_m, int max_pts, std::uint64_t seed = 0) {
  PreprocessConfig c = defaultCfg();
  c.lidar.blind = 0.01;
  c.lidar.voxel_surf_m = voxel_m;
  c.lidar.surf_max_pts = max_pts;
  c.lidar.surf_seed = seed;
  return c;
}

// The set of t_offset tags surviving a process() call, for order-independent comparison.
std::vector<std::int32_t> survivingTags(const LidarScan& out) {
  std::vector<std::int32_t> tags;
  for (const LidarPoint& p : *out.points) tags.push_back(p.t_offset_ns);
  std::sort(tags.begin(), tags.end());
  return tags;
}

double scalarValue(const RecordingSink& rec, const char* key) {
  for (auto it = rec.scalars.rbegin(); it != rec.scalars.rend(); ++it) {
    if (it->key == key) return it->v;
  }
  return -1.0;
}

}  // namespace

TEST(LidarValidityFilter, SurfVoxelKeepsCentreNearest) {
  // One voxel (edge 0.5, cell [5.0,5.5)^3, centre (5.25,5.25,5.25)). The point nearest the
  // centre must be the sole survivor with surf_max_pts == 1.
  PreprocessConfig cfg = surfCfg(0.5, 1);
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.10f, 5.10f, 5.10f, 11));  // far from centre
  pts.push_back(mk(5.25f, 5.25f, 5.25f, 22));  // exactly at centre -> wins
  pts.push_back(mk(5.40f, 5.40f, 5.40f, 33));  // far from centre

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_EQ(out.points->size(), 1u);
  EXPECT_EQ((*out.points)[0].t_offset_ns, 22);
}

TEST(LidarValidityFilter, SurfVoxelIsOrderInvariant) {
  // The deterministic representative must not depend on iteration order: a shuffled input
  // keeps the identical survivor. newest-wins would fail this.
  PreprocessConfig cfg = surfCfg(0.5, 1);
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  std::vector<LidarPoint> base;
  base.push_back(mk(5.10f, 5.10f, 5.10f, 11));
  base.push_back(mk(5.25f, 5.25f, 5.25f, 22));  // nearest centre
  base.push_back(mk(5.40f, 5.40f, 5.40f, 33));
  base.push_back(mk(5.05f, 5.45f, 5.30f, 44));

  const std::vector<std::int32_t> ref = survivingTags(pp->process(scanOf(base)));

  std::mt19937 rng(7);
  for (int trial = 0; trial < 5; ++trial) {
    std::vector<LidarPoint> shuffled = base;
    std::shuffle(shuffled.begin(), shuffled.end(), rng);
    EXPECT_EQ(survivingTags(pp->process(scanOf(shuffled))), ref);
  }
}

TEST(LidarValidityFilter, SurfVoxelReservoirIsSeedDeterministic) {
  // cap > 1: the reservoir keeps exactly the cap, the same seed reproduces the set, and a
  // different seed picks a different subset.
  std::vector<LidarPoint> pts;
  for (int i = 0; i < 20; ++i) {
    // 20 points packed into the single voxel [5.0,5.5)^3.
    const float x = 5.0f + 0.02f * static_cast<float>(i);
    pts.push_back(mk(x, 5.2f, 5.2f, static_cast<float>(100 + i)));
  }

  PreprocessConfig cfg_a = surfCfg(0.5, 5, /*seed=*/0);
  auto pp_a = makeLidarPreprocessor(cfg_a, Extrinsic{}, 10.0, nullptr);
  const std::vector<std::int32_t> set_a1 = survivingTags(pp_a->process(scanOf(pts)));
  const std::vector<std::int32_t> set_a2 = survivingTags(pp_a->process(scanOf(pts)));

  EXPECT_EQ(set_a1.size(), 5u);          // exactly the cap
  EXPECT_EQ(set_a1, set_a2);             // same seed -> identical reservoir

  PreprocessConfig cfg_b = surfCfg(0.5, 5, /*seed=*/99);
  auto pp_b = makeLidarPreprocessor(cfg_b, Extrinsic{}, 10.0, nullptr);
  const std::vector<std::int32_t> set_b = survivingTags(pp_b->process(scanOf(pts)));
  EXPECT_EQ(set_b.size(), 5u);
  EXPECT_NE(set_a1, set_b);              // a different seed picks a different subset
}

TEST(LidarValidityFilter, SurfVoxelOutputIsTimeSorted) {
  PreprocessConfig cfg = surfCfg(0.5, 4, /*seed=*/3);
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);

  // Three distinct voxels, two points each, fed out of time order; survivors must come
  // back ascending in t_offset_ns.
  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.2f, 5.2f, 5.2f, 90));
  pts.push_back(mk(7.2f, 7.2f, 7.2f, 10));
  pts.push_back(mk(9.2f, 9.2f, 9.2f, 50));
  pts.push_back(mk(5.3f, 5.3f, 5.3f, 30));

  LidarScan out = pp->process(scanOf(pts));
  ASSERT_GE(out.points->size(), 2u);
  for (std::size_t i = 1; i < out.points->size(); ++i) {
    EXPECT_LE((*out.points)[i - 1].t_offset_ns, (*out.points)[i].t_offset_ns);
  }
}

TEST(LidarValidityFilter, EmitsVoxelOutTelemetry) {
  RecordingSink rec;
  PreprocessConfig cfg = surfCfg(0.5, 1);
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, &rec);

  // Two points in one voxel (collapse to 1), one in another (kept) -> n_voxel_out == 2.
  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.10f, 5.10f, 5.10f, 11));
  pts.push_back(mk(5.40f, 5.40f, 5.40f, 22));
  pts.push_back(mk(8.20f, 8.20f, 8.20f, 33));

  LidarScan out = pp->process(scanOf(pts));
  EXPECT_EQ(out.points->size(), 2u);
  EXPECT_DOUBLE_EQ(scalarValue(rec, "lidar/n_out"), 3.0);        // after validity gate
  EXPECT_DOUBLE_EQ(scalarValue(rec, "lidar/n_voxel_out"), 2.0);  // after downsample
}

TEST(LidarValidityFilter, SweepDurationFloorEngagesOnTruncatedScan) {
  // 10 Hz nominal -> 100 ms period; floor_frac 0.5 -> 50 ms floor. A scan filtered down to
  // a few early-column survivors (max offset 2 ms) must have its horizon raised to 50 ms.
  PreprocessConfig cfg = surfCfg(0.5, 1);
  cfg.lidar.sweep_floor_frac = 0.5;
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, /*nominal_rate_hz=*/10.0, nullptr);

  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.2f, 5.2f, 5.2f, 1'000'000));  // 1 ms, distinct voxel
  pts.push_back(mk(8.2f, 8.2f, 8.2f, 2'000'000));  // 2 ms, distinct voxel

  LidarScan in = scanOf(pts);
  in.sweep_duration = 100'000'000;  // original (pre-filter) span: 100 ms

  LidarScan out = pp->process(in);
  EXPECT_EQ(out.sweep_duration, 50'000'000);  // floored to 50 ms
  // No surviving point's offset exceeds the floored horizon.
  for (const LidarPoint& p : *out.points) {
    EXPECT_LE(static_cast<std::int64_t>(p.t_offset_ns), out.sweep_duration);
  }
}

TEST(LidarValidityFilter, SweepFloorNeverExceedsOriginalSpan) {
  // When the original (pre-filter) sweep is shorter than the nominal floor, the floor is
  // capped at the original span so the anchor is never invented past the true sweep end.
  PreprocessConfig cfg = surfCfg(0.5, 1);
  cfg.lidar.sweep_floor_frac = 0.5;
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);  // 50 ms floor

  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.2f, 5.2f, 5.2f, 1'000'000));
  pts.push_back(mk(8.2f, 8.2f, 8.2f, 2'000'000));

  LidarScan in = scanOf(pts);
  in.sweep_duration = 30'000'000;  // original span 30 ms < 50 ms floor

  LidarScan out = pp->process(in);
  EXPECT_EQ(out.sweep_duration, 30'000'000);  // clamped to the original span
}

TEST(LidarValidityFilter, SweepDurationUnchangedWhenSurvivorsSpanPastFloor) {
  // Survivors already span past the floor; the recomputed horizon is just their sweep-end.
  PreprocessConfig cfg = surfCfg(0.5, 1);
  cfg.lidar.sweep_floor_frac = 0.5;
  auto pp = makeLidarPreprocessor(cfg, Extrinsic{}, 10.0, nullptr);  // 50 ms floor

  std::vector<LidarPoint> pts;
  pts.push_back(mk(5.2f, 5.2f, 5.2f, 10'000'000));  // 10 ms
  pts.push_back(mk(8.2f, 8.2f, 8.2f, 80'000'000));  // 80 ms, past the floor

  LidarScan in = scanOf(pts);
  in.sweep_duration = 100'000'000;

  LidarScan out = pp->process(in);
  EXPECT_EQ(out.sweep_duration, 80'000'000);  // surviving sweep-end, floor not engaged
}
