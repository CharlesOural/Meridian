#include "meridian/preprocess/gnss_gate.hpp"

#include <cmath>

#include <gtest/gtest.h>

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::ecefToEnu;
using meridian::GnssFix;
using meridian::GnssGate;
using meridian::GnssVerdict;
using meridian::IVelocitySource;
using meridian::PreprocGnss;
using meridian::wgs84ToEcef;
using meridian::wgs84ToEnu;

namespace {

GnssFix mkFix(GnssFix::FixType type, int sats, double trace_var) {
  GnssFix f;
  f.fix = type;
  f.num_sats = static_cast<std::uint8_t>(sats);
  f.cov_enu = Eigen::Matrix3d::Identity() * (trace_var / 3.0);
  f.lat_deg = 47.0;
  f.lon_deg = 8.0;
  f.alt_m = 400.0;
  return f;
}

// A velocity source that always reports zero body velocity.
class ZeroVel : public IVelocitySource {
 public:
  bool velocity(meridian::Timestamp, meridian::Duration,
                Eigen::Vector3d* v) const override {
    *v = Eigen::Vector3d::Zero();
    return true;
  }
};

}  // namespace

TEST(GnssGate, AcceptsGoodFix) {
  PreprocGnss cfg;  // min_fix_type DGPS, min_sats 6, max_pos_var 25
  cfg.spoof_check = false;
  GnssGate gate(cfg, nullptr, nullptr);

  GnssVerdict v = gate.evaluate(mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0));
  EXPECT_TRUE(v.accepted);
  EXPECT_EQ(v.reason, GnssVerdict::Reason::Accepted);
}

TEST(GnssGate, RejectMatrix) {
  PreprocGnss cfg;
  cfg.spoof_check = false;

  {
    GnssGate gate(cfg, nullptr, nullptr);
    GnssVerdict v = gate.evaluate(mkFix(GnssFix::FixType::SPP, 12, 1.0));
    EXPECT_FALSE(v.accepted);
    EXPECT_EQ(v.reason, GnssVerdict::Reason::WeakFix);
  }
  {
    GnssGate gate(cfg, nullptr, nullptr);
    GnssVerdict v = gate.evaluate(mkFix(GnssFix::FixType::RTK_Fixed, 3, 1.0));
    EXPECT_FALSE(v.accepted);
    EXPECT_EQ(v.reason, GnssVerdict::Reason::FewSats);
  }
  {
    GnssGate gate(cfg, nullptr, nullptr);
    GnssVerdict v = gate.evaluate(mkFix(GnssFix::FixType::RTK_Fixed, 12, 100.0));
    EXPECT_FALSE(v.accepted);
    EXPECT_EQ(v.reason, GnssVerdict::Reason::HighCov);
  }
}

TEST(GnssGate, DisabledRejectsEverything) {
  PreprocGnss cfg;
  cfg.enable = false;
  GnssGate gate(cfg, nullptr, nullptr);
  GnssVerdict v = gate.evaluate(mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0));
  EXPECT_FALSE(v.accepted);
  EXPECT_EQ(v.reason, GnssVerdict::Reason::Disabled);
}

TEST(GnssGate, SpoofWhenGnssVelocityDisagreesWithImu) {
  PreprocGnss cfg;
  cfg.spoof_check = true;
  cfg.spoof_vel_thresh = 3.0;
  cfg.spoof_persist = 1;       // declare on the first disagreeing window
  cfg.spoof_window_ms = 1000;  // 1 s
  ZeroVel zero_vel;            // IMU says we are stationary
  GnssGate gate(cfg, &zero_vel, nullptr);

  const std::int64_t s = 1'000'000'000;  // 1 s in ns

  // First fix latches the datum at lat/lon/alt; subsequent fixes drift fast in lon so
  // GNSS-derived velocity is large while IMU velocity is zero.
  GnssFix f0 = mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0);
  f0.stamp = 0;
  EXPECT_TRUE(gate.evaluate(f0).accepted);

  GnssFix f1 = mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0);
  f1.stamp = 2 * s;
  f1.lon_deg = 8.001;  // ~75 m east over 2 s -> ~37 m/s, far above threshold
  GnssVerdict v = gate.evaluate(f1);
  EXPECT_FALSE(v.accepted);
  EXPECT_EQ(v.reason, GnssVerdict::Reason::SpoofVelocity);
}

TEST(GnssGate, AcceptsWhenVelocityConsistent) {
  PreprocGnss cfg;
  cfg.spoof_check = true;
  cfg.spoof_vel_thresh = 3.0;
  cfg.spoof_persist = 1;
  cfg.spoof_window_ms = 1000;
  ZeroVel zero_vel;
  GnssGate gate(cfg, &zero_vel, nullptr);

  const std::int64_t s = 1'000'000'000;
  GnssFix f0 = mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0);
  f0.stamp = 0;
  EXPECT_TRUE(gate.evaluate(f0).accepted);

  GnssFix f1 = mkFix(GnssFix::FixType::RTK_Fixed, 12, 1.0);
  f1.stamp = 2 * s;  // same position -> ~0 m/s, matches the zero IMU velocity
  EXPECT_TRUE(gate.evaluate(f1).accepted);
}

TEST(GnssGate, EnuRoundTripAtDatumIsZero) {
  // ENU of the datum point about itself is the origin.
  const Eigen::Vector3d enu = wgs84ToEnu(47.0, 8.0, 400.0, 47.0, 8.0, 400.0);
  EXPECT_NEAR(enu.norm(), 0.0, 1e-6);
}

TEST(GnssGate, EnuEastNorthUpSigns) {
  // A small increase in longitude is east; latitude is north; altitude is up.
  const Eigen::Vector3d east = wgs84ToEnu(47.0, 8.001, 400.0, 47.0, 8.0, 400.0);
  EXPECT_GT(east.x(), 0.0);
  EXPECT_NEAR(east.y(), 0.0, 1.0);
  EXPECT_NEAR(east.z(), 0.0, 1.0);

  const Eigen::Vector3d north = wgs84ToEnu(47.001, 8.0, 400.0, 47.0, 8.0, 400.0);
  EXPECT_GT(north.y(), 0.0);
  EXPECT_NEAR(north.x(), 0.0, 1.0);

  const Eigen::Vector3d up = wgs84ToEnu(47.0, 8.0, 410.0, 47.0, 8.0, 400.0);
  EXPECT_NEAR(up.z(), 10.0, 1e-3);
  EXPECT_NEAR(up.x(), 0.0, 1e-3);
  EXPECT_NEAR(up.y(), 0.0, 1e-3);
}

TEST(GnssGate, EcefRoundTripConsistency) {
  // ecefToEnu(ecef(p)) about p's own datum is the origin.
  const Eigen::Vector3d ecef = wgs84ToEcef(47.0, 8.0, 400.0);
  const Eigen::Vector3d enu = ecefToEnu(ecef, 47.0, 8.0, 400.0);
  EXPECT_NEAR(enu.norm(), 0.0, 1e-6);
}
