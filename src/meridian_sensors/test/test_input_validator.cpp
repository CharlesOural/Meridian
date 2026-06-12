#include "input_validator.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/time/clock_model.hpp"

using namespace meridian;

namespace {

// Records every health update and every raised code so a test can assert the verdict's
// side effects. `degrade` is the edge event the validator emits once per fault entry.
class RecordingHealth final : public HealthSink {
 public:
  void update(const SensorHealth& h) override {
    last = h;
    ++updates;
  }
  void degrade(std::uint8_t, HealthCode code) override { events.push_back(code); }
  int event_count(HealthCode code) const {
    int n = 0;
    for (HealthCode c : events)
      if (c == code) ++n;
    return n;
  }
  bool raised(HealthCode code) const { return event_count(code) > 0; }
  SensorHealth last{};
  int updates = 0;
  std::vector<HealthCode> events;
};

// Captures the latest scalar value emitted per telemetry key, with `enabled` always true
// so the validator's count keys are surfaced.
class RecordingTelemetry final : public TelemetrySink {
 public:
  bool enabled(const char*) const override { return true; }
  void scalar(const char* key, double v, Timestamp) override { scalars[key] = v; }
  void vec(const char*, const Eigen::Ref<const Eigen::VectorXd>&, Timestamp,
           const char*) override {}
  void cloud(const char*, const PointCloudView&, Frame, Timestamp) override {}
  void pose(const char*, const Pose&, Frame, Timestamp) override {}
  void marker(const Marker&, Timestamp) override {}
  void image(const char*, const ImageOverlay&, Timestamp) override {}
  void timing(const char*, double, Timestamp) override {}
  void event(Level, const char*, std::string_view, Timestamp) override {}

  bool has(const std::string& key) const { return scalars.count(key) != 0; }
  double value(const std::string& key) const {
    auto it = scalars.find(key);
    return it == scalars.end() ? -1.0 : it->second;
  }
  std::map<std::string, double> scalars;
};

SensorInfo lidar_info() {
  SensorInfo i;
  i.id = 0;
  i.modality = Modality::Lidar;
  i.sensor_frame = Frame::OsSensor0;
  i.nominal_rate_hz = 10.0;
  i.configured_stamp_source = StampSource::HwPtp;
  return i;
}

SensorInfo imu_info() {
  SensorInfo i;
  i.id = 1;
  i.modality = Modality::Imu;
  i.sensor_frame = Frame::ImuLink;
  i.nominal_rate_hz = 200.0;
  i.configured_stamp_source = StampSource::SwOffset;
  return i;
}

// A sweep with `n` points; the k-th point is non-finite when its index is in `nan_idx`,
// and every point carries a non-zero per-point time unless `with_time` is false.
LidarScan make_scan(Timestamp start, std::size_t n, const std::vector<std::size_t>& nan_idx,
                    bool with_time = true) {
  LidarScan scan;
  scan.stamp_start = start;
  scan.sensor_id = 0;
  auto pts = std::make_shared<PointCloud>();
  for (std::size_t k = 0; k < n; ++k) {
    LidarPoint p;
    p.xyz = Eigen::Vector3f(1.f, 2.f, 3.f);
    for (std::size_t bad : nan_idx) {
      if (bad == k) p.xyz.x() = std::numeric_limits<float>::quiet_NaN();
    }
    p.t_offset_ns = with_time ? static_cast<std::int32_t>(1000 * (k + 1)) : 0;
    pts->push_back(p);
  }
  scan.sweep_duration = n > 0 ? static_cast<Duration>(1000 * n) : 0;
  scan.points = std::move(pts);
  return scan;
}

InputValidator make_validator(SensorInfo info, ClockModel* clock, RecordingHealth* health,
                              RecordingTelemetry* tele, const ValidatorConfig& cfg = {}) {
  return InputValidator(info, cfg, clock, health, tele, /*failed_timeout_ms=*/1000.0,
                        /*rate_tolerance_frac=*/0.20);
}

}  // namespace

// Check 1 (Rewind): a stamp that does not advance is clamped to the last stamp, reported
// once as ClockStepDetected, and never emitted as a regressing value.
TEST(InputValidator, RewindClampsAndReports) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  EXPECT_EQ(v.on_stamp(100, StampSource::SwOffset).kind, InputValidator::Verdict::Accept);
  const auto r = v.on_stamp(90, StampSource::SwOffset);  // regression
  EXPECT_EQ(r.kind, InputValidator::Verdict::Clamped);
  EXPECT_EQ(r.stamp, 100);  // clamped up to the last stamp, never the regressing value
  EXPECT_TRUE(health.raised(HealthCode::ClockStepDetected));
  EXPECT_TRUE(tele.has("sensors/validator/imu1/ClockStepDetected_count"));
}

// Check 2 (Gap): a raw inter-sample gap beyond gap_periods/nominal_rate raises Dropout
// for the post-gap sample; the count surfaces and the event is emitted once on entry.
TEST(InputValidator, GapRaisesDropout) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  // 200 Hz -> 5 ms period; gap_periods 2.5 -> 12.5 ms dropout threshold.
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  v.on_stamp(0, StampSource::SwOffset);
  v.on_stamp(5'000'000, StampSource::SwOffset);  // 5 ms, within band
  EXPECT_FALSE(health.raised(HealthCode::Dropout));
  v.on_stamp(25'000'000, StampSource::SwOffset);  // 20 ms gap, a hole
  EXPECT_TRUE(health.raised(HealthCode::Dropout));
  EXPECT_TRUE(tele.has("sensors/validator/imu1/Dropout_count"));
}

// Check 2b (Gap escalation): a gap within the dropout band holds the sensor Degraded; a
// gap past failed_timeout escalates the post-gap sample to Failed; a clean sample clears.
TEST(InputValidator, GapEscalatesToFailedPastTimeout) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  // 200 Hz -> 12.5 ms dropout band; failed_timeout 1000 ms -> 1 s escalation threshold.
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  v.on_stamp(0, StampSource::SwOffset);
  v.on_stamp(20'000'000, StampSource::SwOffset);  // 20 ms gap: in-band dropout
  EXPECT_TRUE(health.raised(HealthCode::Dropout));
  EXPECT_EQ(health.last.level, HealthLevel::Degraded);

  // A 2 s gap exceeds failed_timeout: this post-gap sample escalates to Failed.
  v.on_stamp(2'020'000'000, StampSource::SwOffset);
  EXPECT_EQ(health.last.level, HealthLevel::Failed);

  // A normal-cadence sample clears the dropout and de-escalates below Failed (the smoothed
  // rate still lags after a 2 s hole, so RateLow may keep the sensor Degraded; the point is
  // the failed-timeout escalation is gone once the gap closes).
  v.on_stamp(2'025'000'000, StampSource::SwOffset);  // 5 ms gap
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::Dropout));
  EXPECT_NE(health.last.level, HealthLevel::Failed);
}

// Check 3 (Skew): clock skew past skew_warn_ppm raises SkewOutOfRange.
TEST(InputValidator, SkewOutOfRangeRaised) {
  ClockModel clock;
  // Seed a free-running clock with two correspondences implying a large rate error so the
  // estimated skew_ppm exceeds the 200 ppm warn band.
  clock.on_correspondence(ClockId::Imu, /*device_ns=*/0, /*host_ns=*/0, /*std=*/100.0);
  clock.on_correspondence(ClockId::Imu, /*device_ns=*/1'000'000'000 + 5'000'000,
                          /*host_ns=*/1'000'000'000, /*std=*/100.0);  // +5 ms / s = 5000 ppm
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  v.on_stamp(2'000'000'000, StampSource::SwOffset);
  EXPECT_TRUE(health.raised(HealthCode::SkewOutOfRange));
  EXPECT_TRUE(tele.has("sensors/validator/imu1/SkewOutOfRange_count"));
}

// Check 3 (Skew), two-sided: an in-band skew raises nothing; pushing the model out of
// band then raises SkewOutOfRange; returning in-band clears it. A stuck-on (always
// enter()) implementation fails the in-band and the cleared assertions.
TEST(InputValidator, SkewInBandThenOutThenClear) {
  ClockModel clock;
  // 100 ppm: device gains 100 us over 1 s -> well under the 200 ppm warn band.
  clock.on_correspondence(ClockId::Imu, /*device_ns=*/0, /*host_ns=*/0, /*std=*/100.0);
  clock.on_correspondence(ClockId::Imu, /*device_ns=*/1'000'000'000 + 100'000,
                          /*host_ns=*/1'000'000'000, /*std=*/100.0);  // +100 us / s = 100 ppm
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  // In band: no skew fault is raised and the active bit stays clear.
  v.on_stamp(2'000'000'000, StampSource::SwOffset);
  EXPECT_FALSE(health.raised(HealthCode::SkewOutOfRange))
      << "in-band skew must not raise SkewOutOfRange";
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::SkewOutOfRange));

  // Drive the model out of band: device now gains 5 ms over the next second (5000 ppm).
  clock.on_correspondence(ClockId::Imu, /*device_ns=*/3'000'000'000 + 5'000'000,
                          /*host_ns=*/3'000'000'000, /*std=*/100.0);
  v.on_stamp(4'000'000'000, StampSource::SwOffset);
  EXPECT_TRUE(health.raised(HealthCode::SkewOutOfRange)) << "out-of-band skew must raise";
  EXPECT_TRUE(v.health_state().code_bits & health_code_bit(HealthCode::SkewOutOfRange));

  // Pull the model back in band: the fault must clear from the active bitset.
  for (int k = 0; k < 20; ++k) {
    const Timestamp host = static_cast<Timestamp>(5'000'000'000LL + k * 1'000'000'000LL);
    // Track host nearly 1:1 (tiny 50 us/s drift, 50 ppm) so the RLS rate decays in band.
    clock.on_correspondence(ClockId::Imu, host + 50'000, host, /*std=*/100.0);
  }
  v.on_stamp(30'000'000'000, StampSource::SwOffset);
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::SkewOutOfRange))
      << "skew back in band must clear SkewOutOfRange";
}

// Rate band (RateLow): a stream slower than nominal*(1-tol) raises RateLow without a
// dropout (the gap stays inside the dropout band), and an in-band stream raises neither.
TEST(InputValidator, RateLowRaisesAndClears) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  // 200 Hz nominal, 20% tolerance -> band [160, 240] Hz. Dropout band is 2.5 periods
  // (12.5 ms); a 10 ms period is 100 Hz (RateLow) yet under the dropout gap.
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  // Seed two samples 10 ms apart: the first gap sets rate_hz_ = 100 Hz directly.
  v.on_stamp(0, StampSource::SwOffset);
  v.on_stamp(10'000'000, StampSource::SwOffset);
  EXPECT_TRUE(health.raised(HealthCode::RateLow)) << "100 Hz < 160 Hz must raise RateLow";
  EXPECT_FALSE(health.raised(HealthCode::Dropout)) << "10 ms gap is inside the dropout band";
  EXPECT_TRUE(tele.has("sensors/validator/imu1/RateLow_count"));
  EXPECT_TRUE(v.health_state().code_bits & health_code_bit(HealthCode::RateLow));

  // Now feed nominal-cadence samples: the smoothed rate climbs back into band and the
  // RateLow code clears.
  Timestamp t = 10'000'000;
  for (int k = 0; k < 60; ++k) {
    t += 5'000'000;  // 5 ms -> 200 Hz instantaneous
    v.on_stamp(t, StampSource::SwOffset);
  }
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::RateLow))
      << "rate back in band must clear RateLow";
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::RateHigh));
}

// Rate band (RateHigh): a stream faster than nominal*(1+tol) raises RateHigh; an
// in-band stream raises neither rate code.
TEST(InputValidator, RateHighRaisesAndInBandIsClean) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  // 2 ms period -> 500 Hz > 240 Hz: the first gap sets the rate directly, raising RateHigh.
  v.on_stamp(0, StampSource::SwOffset);
  v.on_stamp(2'000'000, StampSource::SwOffset);
  EXPECT_TRUE(health.raised(HealthCode::RateHigh)) << "500 Hz > 240 Hz must raise RateHigh";
  EXPECT_FALSE(health.raised(HealthCode::RateLow));
  EXPECT_TRUE(tele.has("sensors/validator/imu1/RateHigh_count"));

  // A fresh validator fed exactly at nominal raises neither rate code.
  RecordingHealth health2;
  RecordingTelemetry tele2;
  InputValidator v2 = make_validator(imu_info(), &clock, &health2, &tele2);
  Timestamp t = 0;
  for (int k = 0; k < 10; ++k) {
    v2.on_stamp(t, StampSource::SwOffset);
    t += 5'000'000;  // 200 Hz, dead centre of the band
  }
  EXPECT_FALSE(health2.raised(HealthCode::RateLow)) << "in-band cadence must not raise RateLow";
  EXPECT_FALSE(health2.raised(HealthCode::RateHigh)) << "in-band cadence must not raise RateHigh";
}

// Check 4 (Per-point time): an all-zero-offset scan cannot be deskewed and is rejected,
// never reconstructed.
TEST(InputValidator, NoPointTimeRejected) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  LidarScan scan = make_scan(1'000, /*n=*/4, /*nan_idx=*/{}, /*with_time=*/false);
  const auto r = v.on_lidar(scan);
  EXPECT_EQ(r.kind, InputValidator::Verdict::Reject);
  EXPECT_TRUE(health.raised(HealthCode::LidarNoPointTime));
  EXPECT_TRUE(tele.has("sensors/validator/lidar0/LidarNoPointTime_count"));
}

// Check 5 (NaN/Inf, S2 policy): a NaN fraction above nan_ratio_warn warns and DROPS the
// offending points, accepting the survivors; the whole scan is not rejected.
TEST(InputValidator, NanRatioWarnsAndDropsPoints) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  // 2 of 10 non-finite = 0.2 > 0.05 warn band.
  LidarScan scan = make_scan(1'000, /*n=*/10, /*nan_idx=*/{2, 7});
  const auto r = v.on_lidar(scan);
  EXPECT_EQ(r.kind, InputValidator::Verdict::Accept);
  EXPECT_TRUE(health.raised(HealthCode::LidarHighNanRatio));
  ASSERT_TRUE(scan.points);
  EXPECT_EQ(scan.points->size(), 8u);  // the two NaN returns dropped, survivors kept
  for (const LidarPoint& p : *scan.points) EXPECT_TRUE(p.xyz.allFinite());
  EXPECT_TRUE(tele.has("sensors/validator/lidar0/LidarHighNanRatio_count"));
}

// A NaN fraction below the warn band still drops the offending points (the drop is
// unconditional) but raises no health code (the band gates only LidarHighNanRatio).
TEST(InputValidator, NanBelowBandDropsPointsWithoutWarning) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  // 1 of 100 non-finite = 0.01 < 0.05 warn band.
  LidarScan scan = make_scan(1'000, /*n=*/100, /*nan_idx=*/{42});
  const auto r = v.on_lidar(scan);
  EXPECT_EQ(r.kind, InputValidator::Verdict::Accept);
  EXPECT_FALSE(health.raised(HealthCode::LidarHighNanRatio));  // below the band: no warning
  ASSERT_TRUE(scan.points);
  EXPECT_EQ(scan.points->size(), 99u);  // the lone NaN return dropped even below the band
  for (const LidarPoint& p : *scan.points) EXPECT_TRUE(p.xyz.allFinite());
}

// An entirely non-finite scan is rejected (it has no usable content), as is an empty one.
TEST(InputValidator, AllNanRejectedAsEmpty) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  LidarScan scan = make_scan(1'000, /*n=*/4, /*nan_idx=*/{0, 1, 2, 3});
  const auto r = v.on_lidar(scan);
  EXPECT_EQ(r.kind, InputValidator::Verdict::Reject);
  EXPECT_TRUE(health.raised(HealthCode::EmptyScan));
  EXPECT_TRUE(tele.has("sensors/validator/lidar0/EmptyScan_count"));
}

// Check 6 (Empty): a scan with no points is rejected.
TEST(InputValidator, EmptyScanRejected) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  LidarScan scan = make_scan(1'000, /*n=*/0, /*nan_idx=*/{});
  const auto r = v.on_lidar(scan);
  EXPECT_EQ(r.kind, InputValidator::Verdict::Reject);
  EXPECT_TRUE(health.raised(HealthCode::EmptyScan));
}

// IMU content: a non-finite acc/gyro component rejects the sample and raises the IMU-
// specific non-finite code (never a LiDAR-named code), and surfaces its count key; a
// finite sample clears it.
TEST(InputValidator, ImuNonFiniteRejected) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  ImuSample s;
  s.stamp = 1'000;
  s.acc = Eigen::Vector3d(0, 0, std::numeric_limits<double>::infinity());
  s.gyro = Eigen::Vector3d::Zero();
  EXPECT_EQ(v.on_imu(s).kind, InputValidator::Verdict::Reject);
  // The fault must surface under an IMU-specific code, not a LiDAR-named one: triage on
  // the live bag must blame the right sensor.
  EXPECT_TRUE(health.raised(HealthCode::ImuNonFinite));
  EXPECT_FALSE(health.raised(HealthCode::LidarHighNanRatio));
  EXPECT_TRUE(tele.has("sensors/validator/imu1/ImuNonFinite_count"));

  ImuSample good;
  good.stamp = 2'000;
  good.acc = Eigen::Vector3d(0, 0, 9.81);
  good.gyro = Eigen::Vector3d(0.01, 0, 0);
  EXPECT_EQ(v.on_imu(good).kind, InputValidator::Verdict::Accept);
  // The good sample clears the fault from the active bitset.
  EXPECT_FALSE(v.health_state().code_bits & health_code_bit(HealthCode::ImuNonFinite));
}

// Edge-throttle / dup-filter: a sustained fault emits exactly one entry event while the
// per-code count increments every sample; exit followed by re-entry emits a fresh event.
TEST(InputValidator, EdgeThrottledEventWithRunningCount) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(imu_info(), &clock, &health, &tele);

  v.on_stamp(100, StampSource::SwOffset);
  // Three regressions in a row: one ClockStepDetected event, count climbs each sample.
  v.on_stamp(90, StampSource::SwOffset);
  v.on_stamp(80, StampSource::SwOffset);
  v.on_stamp(70, StampSource::SwOffset);
  EXPECT_EQ(health.event_count(HealthCode::ClockStepDetected), 1);
  EXPECT_EQ(tele.value("sensors/validator/imu1/ClockStepDetected_count"), 3.0);

  // A clean advance clears the code; a later regression re-enters and emits a new event.
  v.on_stamp(200, StampSource::SwOffset);
  v.on_stamp(150, StampSource::SwOffset);
  EXPECT_EQ(health.event_count(HealthCode::ClockStepDetected), 2);
  EXPECT_EQ(tele.value("sensors/validator/imu1/ClockStepDetected_count"), 1.0);
}

// A rejected sample's verdict carries Reject, the signal a source uses to skip the
// callback so a corrupt sample never reaches Q_sensors.
TEST(InputValidator, RejectVerdictBlocksDownstream) {
  ClockModel clock;
  RecordingHealth health;
  RecordingTelemetry tele;
  InputValidator v = make_validator(lidar_info(), &clock, &health, &tele);

  LidarScan empty = make_scan(1'000, 0, {});
  EXPECT_EQ(v.on_lidar(empty).kind, InputValidator::Verdict::Reject);

  LidarScan good = make_scan(2'000, 5, {});
  EXPECT_NE(v.on_lidar(good).kind, InputValidator::Verdict::Reject);
}
