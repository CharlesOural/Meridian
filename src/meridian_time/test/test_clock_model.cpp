#include "meridian/time/clock_model.hpp"

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "meridian/time/stamp_source.hpp"

using meridian::ClockId;
using meridian::ClockModel;
using meridian::ClockState;
using meridian::StampSource;
using meridian::Timestamp;

namespace {

// Build the device-clock value a host instant would produce under a known
// offset/skew, then check to_meridian inverts it back to the host instant.
Timestamp forward_model(Timestamp host_ns, double offset_ns, double skew_ppm) {
  const double rate = 1.0 + skew_ppm * 1e-6;
  return static_cast<Timestamp>(
      std::llround(rate * static_cast<double>(host_ns) + offset_ns));
}

}  // namespace

TEST(ClockModel, ToMeridianPassThroughOnFreshClock) {
  ClockModel cm;
  // No correction learned yet: offset 0, skew 0 -> identity.
  EXPECT_EQ(cm.to_meridian(1'234'567, ClockId::Lidar), 1'234'567);
}

TEST(ClockModel, ToMeridianRecoversHostTimeWithKnownOffsetAndSkew) {
  ClockModel cm;
  const double offset_ns = 5'000'000.0;  // device runs 5 ms ahead at t_ref
  const double skew_ppm = 120.0;         // and 120 ppm fast

  // Seed the estimator so its state matches the known model exactly, then feed
  // a long correspondence so the offset/skew converge onto the truth.
  const Timestamp host0 = 0;
  cm.on_correspondence(ClockId::Imu, forward_model(host0, offset_ns, skew_ppm),
                       host0, 1.0);

  for (int i = 1; i <= 200; ++i) {
    const Timestamp host = static_cast<Timestamp>(i) * 100'000'000;  // +0.1 s
    const Timestamp dev = forward_model(host, offset_ns, skew_ppm);
    cm.on_correspondence(ClockId::Imu, dev, host, 1.0);
  }

  // After convergence the recovered host time should track the truth closely.
  const Timestamp host_truth = 20'000'000'000;  // 20 s
  const Timestamp dev = forward_model(host_truth, offset_ns, skew_ppm);
  const Timestamp recovered = cm.to_meridian(dev, ClockId::Imu);
  EXPECT_NEAR(static_cast<double>(recovered),
              static_cast<double>(host_truth), 50'000.0);  // within 50 us
}

TEST(ClockModel, ToMeridianAppliesAnalyticInverseExactly) {
  // Drive the state directly via a single correspondence with zero skew, so the
  // inverse relation reduces to a pure offset subtraction we can check exactly.
  ClockModel cm;
  const Timestamp host = 1'000'000'000;
  const double offset_ns = 7'500'000.0;
  const Timestamp dev = forward_model(host, offset_ns, 0.0);
  cm.on_correspondence(ClockId::Cam, dev, host, 1.0);

  const ClockState s = cm.state(ClockId::Cam);
  EXPECT_NEAR(s.offset_ns, offset_ns, 1.0);
  EXPECT_EQ(cm.to_meridian(dev, ClockId::Cam), host);
}

TEST(ClockModel, DisciplinedFlagViaPtpStats) {
  ClockModel cm;
  EXPECT_FALSE(cm.disciplined(ClockId::Lidar));

  cm.on_ptp_stats(ClockId::Lidar, /*offset_ns=*/1'000.0,
                  /*path_delay_ns=*/2'000.0, /*locked=*/true);

  EXPECT_TRUE(cm.disciplined(ClockId::Lidar));
  EXPECT_TRUE(cm.ptp_locked(ClockId::Lidar));
  EXPECT_EQ(cm.stamp_source(ClockId::Lidar), StampSource::HwPtp);

  const ClockState s = cm.state(ClockId::Lidar);
  EXPECT_DOUBLE_EQ(s.offset_ns, 0.0);
  EXPECT_DOUBLE_EQ(s.skew_ppm, 0.0);
  EXPECT_DOUBLE_EQ(s.offset_std_ns, 1'000.0);  // residual kept for health

  // A disciplined clock converts as a pass-through.
  EXPECT_EQ(cm.to_meridian(42'000'000, ClockId::Lidar), 42'000'000);
}

TEST(ClockModel, PtpLockLossDropsDiscipline) {
  ClockModel cm;
  cm.on_ptp_stats(ClockId::Lidar, 500.0, 1'000.0, /*locked=*/true);
  ASSERT_TRUE(cm.disciplined(ClockId::Lidar));

  cm.on_ptp_stats(ClockId::Lidar, 500.0, 1'000.0, /*locked=*/false);
  EXPECT_FALSE(cm.disciplined(ClockId::Lidar));
  EXPECT_FALSE(cm.ptp_locked(ClockId::Lidar));
  EXPECT_EQ(cm.stamp_source(ClockId::Lidar), StampSource::ArrivalOnly);
}

TEST(ClockModel, PtpLargeResidualDegradesSource) {
  ClockModel cm;
  // Locked but with a residual well past the healthy threshold.
  cm.on_ptp_stats(ClockId::Lidar, /*offset_ns=*/5'000'000.0, 1'000.0,
                  /*locked=*/true);
  EXPECT_TRUE(cm.disciplined(ClockId::Lidar));
  EXPECT_EQ(cm.stamp_source(ClockId::Lidar), StampSource::ArrivalOnly);
  EXPECT_FALSE(cm.ptp_locked(ClockId::Lidar));
}

TEST(ClockModel, PpsEdgeDisciplinesGnss) {
  ClockModel cm;
  const Timestamp edge = 3'000'000'000;
  cm.on_pps_edge(edge);

  EXPECT_TRUE(cm.disciplined(ClockId::Gnss));
  EXPECT_EQ(cm.stamp_source(ClockId::Gnss), StampSource::HwPps);
  const ClockState s = cm.state(ClockId::Gnss);
  EXPECT_EQ(s.t_ref, edge);
  EXPECT_EQ(s.last_update, edge);
}

TEST(ClockModel, StampSourceReporting) {
  ClockModel cm;
  // Default before any observation.
  EXPECT_EQ(cm.stamp_source(ClockId::Imu), StampSource::ArrivalOnly);

  // Software correspondence -> SwOffset.
  cm.on_correspondence(ClockId::Imu, 1'000'500, 1'000'000, 10.0);
  EXPECT_EQ(cm.stamp_source(ClockId::Imu), StampSource::SwOffset);

  // Independent clocks report independent sources.
  cm.on_ptp_stats(ClockId::Lidar, 0.0, 1'000.0, /*locked=*/true);
  EXPECT_EQ(cm.stamp_source(ClockId::Lidar), StampSource::HwPtp);
  EXPECT_EQ(cm.stamp_source(ClockId::Imu), StampSource::SwOffset);
}

TEST(ClockModel, DisciplinedClockIgnoresCorrespondence) {
  ClockModel cm;
  cm.on_ptp_stats(ClockId::Cam, 0.0, 1'000.0, /*locked=*/true);
  cm.on_correspondence(ClockId::Cam, 9'999'999, 1'000'000, 10.0);

  const ClockState s = cm.state(ClockId::Cam);
  EXPECT_DOUBLE_EQ(s.offset_ns, 0.0);
  EXPECT_EQ(cm.stamp_source(ClockId::Cam), StampSource::HwPtp);
}
