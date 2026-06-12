#include "meridian/sensors/aggregator.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"

using namespace meridian;

namespace {

// Captures every code raised so a test can assert the health side effects of grouping.
class RecordingHealth final : public HealthSink {
 public:
  void update(const SensorHealth& h) override { snapshots.push_back(h); }
  void degrade(std::uint8_t id, HealthCode code) override {
    codes.emplace_back(id, code);
  }
  bool raised(HealthCode code) const {
    for (const auto& [id, c] : codes) {
      (void)id;
      if (c == code) return true;
    }
    return false;
  }
  std::vector<std::pair<std::uint8_t, HealthCode>> codes;
  std::vector<SensorHealth> snapshots;
};

ImuSample imu(Timestamp t, std::uint8_t id = 1) {
  ImuSample s;
  s.stamp = t;
  s.sensor_id = id;
  return s;
}

// A sweep [start, start+dur] with one point at the start and one at the end, so
// sweep_duration equals dur.
LidarScan sweep(Timestamp start, Duration dur, std::uint8_t id = 0) {
  LidarScan s;
  s.stamp_start = start;
  s.sweep_duration = dur;
  s.sensor_id = id;
  auto pts = std::make_shared<PointCloud>();
  LidarPoint a;
  a.t_offset_ns = 0;
  LidarPoint b;
  b.t_offset_ns = static_cast<std::int32_t>(dur);
  pts->push_back(a);
  pts->push_back(b);
  s.points = std::move(pts);
  return s;
}

CameraFrame image(Timestamp t, std::uint8_t id = 2) {
  CameraFrame f;
  f.stamp = t;
  f.sensor_id = id;
  return f;
}

GnssFix gfix(Timestamp t, std::uint8_t id = 3) {
  GnssFix f;
  f.stamp = t;
  f.sensor_id = id;
  return f;
}

struct Fixture {
  Fixture() {
    cfg.max_wait_ms = 150.0;   // 1.5 sweep periods at 10 Hz
    cfg.reorder_ms = 20.0;
    agg = std::make_unique<Aggregator>(cfg, sensors, &health, nullptr);
    agg->set_sink([this](MeasureGroup&& g) { emitted.push_back(std::move(g)); });
  }
  AggregationConfig cfg;
  SensorsConfig sensors;
  RecordingHealth health;
  std::unique_ptr<Aggregator> agg;
  std::vector<MeasureGroup> emitted;
};

constexpr Duration kMs = 1'000'000;     // ns per ms
constexpr Duration kSweep = 100 * kMs;  // 100 ms sweep at 10 Hz

}  // namespace

// The group does not close until IMU has been received past t_end.
TEST(Aggregator, ImuGateHoldsGroupUntilCoverage) {
  Fixture f;
  f.agg->on(sweep(0, kSweep));
  // IMU only up to 50 ms: gate not satisfied, no emission.
  for (Timestamp t = 0; t <= 50 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  EXPECT_TRUE(f.emitted.empty());

  // IMU now spans past t_end (100 ms): the gate opens.
  for (Timestamp t = 60 * kMs; t <= 110 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  ASSERT_EQ(f.emitted.size(), 1u);
  EXPECT_EQ(f.emitted[0].t_begin, 0);
  EXPECT_EQ(f.emitted[0].t_end, kSweep);
  EXPECT_FALSE(f.health.raised(HealthCode::ImuLate));
}

// The IMU set is (prev_t_end, t_end] plus the one sample straddling t_begin.
TEST(Aggregator, StraddlingSampleIncludedAcrossGroups) {
  Fixture f;
  // First sweep [0, 100ms]. IMU at -5ms straddles t_begin=0.
  f.agg->on(imu(-5 * kMs));
  for (Timestamp t = 10 * kMs; t <= 110 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(sweep(0, kSweep));
  ASSERT_EQ(f.emitted.size(), 1u);
  // The straddling -5ms sample plus all in (.. , 100ms].
  EXPECT_EQ(f.emitted[0].imu.front().stamp, -5 * kMs);
  EXPECT_EQ(f.emitted[0].imu.back().stamp, 100 * kMs);

  // Second sweep [100ms, 200ms]. The 100ms sample straddles its t_begin and must be
  // present again as the first IMU of the new group.
  for (Timestamp t = 110 * kMs; t <= 210 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(sweep(kSweep, kSweep));
  ASSERT_EQ(f.emitted.size(), 2u);
  EXPECT_EQ(f.emitted[1].imu.front().stamp, 100 * kMs);
  // (prev_t_end=100ms, 200ms]: the straddler (100ms) plus 110..200ms.
  EXPECT_EQ(f.emitted[1].imu.back().stamp, 200 * kMs);
}

// Stamp jitter can begin a sweep before the previous one ended. The overlapped window
// must appear in both groups, and the new group still carries a sample at or before
// its t_begin even though that sample predates the previous group's end.
TEST(Aggregator, OverlappingSweepsShareImuAndKeepStraddler) {
  Fixture f;
  // First sweep [0, 100ms]; IMU every 10 ms plus a sample at 98 ms, inside what will
  // become the overlap window.
  f.agg->on(imu(-5 * kMs));
  for (Timestamp t = 5 * kMs; t <= 95 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(imu(98 * kMs));
  f.agg->on(imu(105 * kMs));
  f.agg->on(sweep(0, kSweep));
  ASSERT_EQ(f.emitted.size(), 1u);
  EXPECT_EQ(f.emitted[0].imu.back().stamp, 98 * kMs);

  // Second sweep starts 4 ms before the first ended: [96ms, 196ms]. The 95ms sample
  // straddles t_begin=96ms and the 98ms overlap sample belongs to both groups.
  for (Timestamp t = 115 * kMs; t <= 205 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(sweep(96 * kMs, kSweep));
  ASSERT_EQ(f.emitted.size(), 2u);
  EXPECT_EQ(f.emitted[1].imu.front().stamp, 95 * kMs);
  EXPECT_EQ(f.emitted[1].imu[1].stamp, 98 * kMs);
  EXPECT_EQ(f.emitted[1].imu.back().stamp, 195 * kMs);
}

// On timeout the group emits with whatever IMU arrived and flags ImuLate. The timeout
// is driven by the IMU watermark (the gating modality): a long sweep whose IMU never
// reaches t_end ages out once the IMU watermark passes received + max_wait, even though
// IMU coverage never spans the sweep.
TEST(Aggregator, TimeoutEmitsAndFlagsImuLate) {
  Fixture f;
  // A 200 ms sweep so the IMU watermark can advance past max_wait (150 ms) while still
  // never covering t_end=200ms.
  f.agg->on(sweep(0, 2 * kSweep));
  f.agg->on(imu(40 * kMs));
  EXPECT_TRUE(f.emitted.empty());

  // IMU streams up to 190 ms, below t_end=200ms, advancing the watermark past 150ms.
  for (Timestamp t = 50 * kMs; t <= 190 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  ASSERT_GE(f.emitted.size(), 1u);
  EXPECT_EQ(f.emitted[0].t_begin, 0);
  EXPECT_TRUE(f.health.raised(HealthCode::ImuLate));
}

// A sample older than the emitted watermark by more than reorder_ms is dropped.
TEST(Aggregator, LateSampleDroppedWithLateDrop) {
  Fixture f;
  f.agg->on(imu(100 * kMs));  // sets the IMU watermark to 100ms
  // 100ms - 20ms = 80ms is the window edge; 70ms is past it.
  f.agg->on(imu(70 * kMs));
  EXPECT_TRUE(f.health.raised(HealthCode::LateDrop));
}

// The image whose mid-exposure falls inside the interval is bucketed into the group;
// one outside is not.
TEST(Aggregator, ImageBucketedByInterval) {
  Fixture f;
  f.agg->on(image(50 * kMs));   // inside [0, 100ms]
  f.agg->on(image(250 * kMs));  // beyond the first interval
  for (Timestamp t = 0; t <= 110 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(sweep(0, kSweep));
  ASSERT_EQ(f.emitted.size(), 1u);
  ASSERT_TRUE(f.emitted[0].image.has_value());
  EXPECT_EQ(f.emitted[0].image->stamp, 50 * kMs);
}

// GNSS fixes inside the interval are collected; usually 0 or 1.
TEST(Aggregator, GnssBucketedByInterval) {
  Fixture f;
  f.agg->on(gfix(30 * kMs));
  f.agg->on(gfix(300 * kMs));  // outside
  for (Timestamp t = 0; t <= 110 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  f.agg->on(sweep(0, kSweep));
  ASSERT_EQ(f.emitted.size(), 1u);
  ASSERT_EQ(f.emitted[0].gnss.size(), 1u);
  EXPECT_EQ(f.emitted[0].gnss[0].stamp, 30 * kMs);
}

// Groups are emitted in stamp order even when their gates open out of order.
TEST(Aggregator, GroupsEmitInStampOrder) {
  Fixture f;
  // Two sweeps arrive; IMU then arrives to open both gates at once.
  f.agg->on(sweep(0, kSweep));
  f.agg->on(sweep(kSweep, kSweep));
  for (Timestamp t = -5 * kMs; t <= 210 * kMs; t += 10 * kMs) f.agg->on(imu(t));
  ASSERT_EQ(f.emitted.size(), 2u);
  EXPECT_LT(f.emitted[0].t_begin, f.emitted[1].t_begin);
  EXPECT_EQ(f.emitted[0].t_begin, 0);
  EXPECT_EQ(f.emitted[1].t_begin, kSweep);
}
