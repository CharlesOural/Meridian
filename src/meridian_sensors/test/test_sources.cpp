#include "meridian/sensors/sources.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/sensors/raw_frames.hpp"
#include "meridian/time/clock_model.hpp"

using namespace meridian;

namespace {

class RecordingHealth final : public HealthSink {
 public:
  void update(const SensorHealth& h) override { last = h; ++updates; }
  void degrade(std::uint8_t, HealthCode code) override { codes.push_back(code); }
  bool raised(HealthCode code) const {
    for (HealthCode c : codes)
      if (c == code) return true;
    return false;
  }
  SensorHealth last{};
  int updates = 0;
  std::vector<HealthCode> codes;
};

SensorInfo imu_info() {
  SensorInfo i;
  i.id = 1;
  i.modality = Modality::Imu;
  i.sensor_frame = Frame::ImuLink;
  i.nominal_rate_hz = 200.0;
  i.configured_stamp_source = StampSource::SwOffset;
  return i;
}

SensorInfo lidar_info() {
  SensorInfo i;
  i.id = 0;
  i.modality = Modality::Lidar;
  i.sensor_frame = Frame::OsSensor0;
  i.nominal_rate_hz = 10.0;
  i.configured_stamp_source = StampSource::HwPtp;
  return i;
}

RawImuFrame raw_imu(Timestamp device_ns, bool has_dev, Timestamp host) {
  RawImuFrame f;
  f.device_ns = device_ns;
  f.has_device_ns = has_dev;
  f.host_arrival = host;
  f.acc = Eigen::Vector3d(0, 0, 9.81);
  f.gyro = Eigen::Vector3d(0.01, 0, 0);
  return f;
}

}  // namespace

// A device-stamped IMU frame stays on the device timeline even before the clock is
// disciplined: SwOffset is identity on an unseeded clock, so the recorded stamp passes
// through and the raw acc/gyro pass straight through.
TEST(ImuSource, DeviceClockStampAndRawPassthrough) {
  ClockModel clock;
  RecordingHealth health;
  ImuSensorConfig cfg;
  cfg.interval_end_shift_ns = 0;
  TimeHealth hc;
  ImuSource src(imu_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<ImuSample> out;
  src.set_callback([&](ImuSample&& s) { out.push_back(std::move(s)); });

  // A fresh ClockModel disciplines no clock, but a frame carrying a device stamp is kept
  // on the device timeline via the SwOffset path (identity on the unseeded clock).
  src.ingest_raw(raw_imu(1'000, /*has_dev=*/true, /*host=*/5'000));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].stamp, 1'000);  // device stamp passthrough on the unseeded clock
  EXPECT_DOUBLE_EQ(out[0].acc.z(), 9.81);
  EXPECT_DOUBLE_EQ(out[0].gyro.x(), 0.01);
  EXPECT_FALSE(health.raised(HealthCode::ImuNoDeviceClock));
}

// The interval-end shift moves the stamp to the end of the integration interval.
TEST(ImuSource, IntervalEndShiftApplied) {
  ClockModel clock;
  RecordingHealth health;
  ImuSensorConfig cfg;
  cfg.interval_end_shift_ns = 2'500'000;  // +2.5 ms
  TimeHealth hc;
  ImuSource src(imu_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<ImuSample> out;
  src.set_callback([&](ImuSample&& s) { out.push_back(std::move(s)); });
  src.ingest_raw(raw_imu(0, /*has_dev=*/false, /*host=*/1'000'000));
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].stamp, 1'000'000 + 2'500'000);
}

// A backwards stamp is clamped to the last emitted stamp and reported, never emitted.
TEST(ImuSource, MonotonicClampOnRegression) {
  ClockModel clock;
  RecordingHealth health;
  ImuSensorConfig cfg;
  TimeHealth hc;
  ImuSource src(imu_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<ImuSample> out;
  src.set_callback([&](ImuSample&& s) { out.push_back(std::move(s)); });

  src.ingest_raw(raw_imu(0, false, 100'000));
  src.ingest_raw(raw_imu(0, false, 90'000));  // regression: host time stepped back
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].stamp, 100'000);
  EXPECT_EQ(out[1].stamp, 100'000);  // clamped up to the last stamp
  EXPECT_TRUE(health.raised(HealthCode::ClockStepDetected));
}

// PTP-locked LiDAR stamps from the device clock (HwPtp); the per-point offset copies
// with no scaling and sweep_duration is last - first.
TEST(OusterLidarSource, PtpStampAndLosslessPoints) {
  ClockModel clock;
  // Mark the LiDAR PTP clock locked so the source takes the hardware path.
  clock.on_ptp_stats(ClockId::Lidar, /*offset_ns=*/0.0, /*path_delay_ns=*/0.0,
                     /*locked=*/true);
  RecordingHealth health;
  LidarSensorConfig cfg;
  TimeHealth hc;
  OusterLidarSource src(lidar_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<LidarScan> out;
  src.set_callback([&](LidarScan&& s) { out.push_back(std::move(s)); });

  RawPoint p0;
  p0.x = 1.f;
  p0.t = 0;
  RawPoint p1;
  p1.x = 2.f;
  p1.t = 99'000'000;  // 99 ms into the sweep
  std::vector<RawPoint> pts{p0, p1};
  RawLidarFrame f;
  f.has_device_ns = true;
  f.device_ns_first_column = 1'000'000'000;
  f.pts = pts;
  src.ingest_raw(f);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].stamp_start, 1'000'000'000);  // identity on a fresh disciplined clock
  EXPECT_EQ(out[0].sweep_duration, 99'000'000);
  ASSERT_EQ(out[0].points->size(), 2u);
  EXPECT_EQ((*out[0].points)[1].t_offset_ns, 99'000'000);  // no scaling
  EXPECT_EQ(health.last.stamp_src, StampSource::HwPtp);
}

// A LiDAR scan whose points carry no per-point time is rejected (not emitted) and
// flagged, because it cannot be deskewed.
TEST(OusterLidarSource, RejectsScanWithoutPointTime) {
  ClockModel clock;
  clock.on_ptp_stats(ClockId::Lidar, 0.0, 0.0, true);
  RecordingHealth health;
  LidarSensorConfig cfg;
  TimeHealth hc;
  OusterLidarSource src(lidar_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<LidarScan> out;
  src.set_callback([&](LidarScan&& s) { out.push_back(std::move(s)); });

  RawPoint p0;  // t defaults to 0
  RawPoint p1;
  std::vector<RawPoint> pts{p0, p1};
  RawLidarFrame f;
  f.has_device_ns = true;
  f.device_ns_first_column = 1'000;
  f.pts = pts;
  src.ingest_raw(f);

  EXPECT_TRUE(out.empty());
  EXPECT_TRUE(health.raised(HealthCode::LidarNoPointTime));
}

// An empty scan is rejected and flagged.
TEST(OusterLidarSource, RejectsEmptyScan) {
  ClockModel clock;
  RecordingHealth health;
  LidarSensorConfig cfg;
  TimeHealth hc;
  OusterLidarSource src(lidar_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<LidarScan> out;
  src.set_callback([&](LidarScan&& s) { out.push_back(std::move(s)); });

  RawLidarFrame f;
  f.has_device_ns = true;
  f.device_ns_first_column = 1'000;
  // f.pts left empty
  src.ingest_raw(f);

  EXPECT_TRUE(out.empty());
  EXPECT_TRUE(health.raised(HealthCode::EmptyScan));
}

// Without PTP lock the LiDAR degrades to the software-offset path.
TEST(OusterLidarSource, DegradesToSwOffsetWhenUnlocked) {
  ClockModel clock;  // no PTP lock asserted
  RecordingHealth health;
  LidarSensorConfig cfg;
  TimeHealth hc;
  OusterLidarSource src(lidar_info(), cfg, &clock, &health, nullptr, hc);

  std::vector<LidarScan> out;
  src.set_callback([&](LidarScan&& s) { out.push_back(std::move(s)); });

  RawPoint p0;
  p0.t = 0;
  RawPoint p1;
  p1.t = 50'000'000;
  std::vector<RawPoint> pts{p0, p1};
  RawLidarFrame f;
  f.has_device_ns = true;
  f.device_ns_first_column = 2'000;
  f.pts = pts;
  src.ingest_raw(f);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(health.last.stamp_src, StampSource::SwOffset);
  // A configured HwPtp sensor running on SwOffset is degraded (sync lost).
  EXPECT_TRUE(health.raised(HealthCode::SyncLost));
}
