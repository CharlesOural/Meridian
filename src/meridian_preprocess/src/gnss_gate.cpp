#include "meridian/preprocess/gnss_gate.hpp"

#include <cmath>

#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/debug/telemetry_keys.hpp"

namespace meridian {

namespace {

constexpr const char* kLogModule = "preprocess.gnss";

// WGS84 ellipsoid.
constexpr double kWgs84A = 6378137.0;             // semi-major axis [m]
constexpr double kWgs84F = 1.0 / 298.257223563;   // flattening
const double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);  // first eccentricity squared

constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

// Capacity for the 1 s accepted-position ring: low-rate GNSS, so a generous fixed cap.
constexpr std::size_t kAcceptedRingCapacity = 64;

// fix-type ordering for the min_fix_type comparison.
int fixRank(GnssFix::FixType f) {
  switch (f) {
    case GnssFix::FixType::None:
      return 0;
    case GnssFix::FixType::SPP:
      return 1;
    case GnssFix::FixType::DGPS:
      return 2;
    case GnssFix::FixType::RTK_Float:
      return 3;
    case GnssFix::FixType::RTK_Fixed:
      return 4;
  }
  return 0;
}

}  // namespace

Eigen::Vector3d wgs84ToEcef(double lat_deg, double lon_deg, double alt_m) {
  const double lat = lat_deg * kDeg2Rad;
  const double lon = lon_deg * kDeg2Rad;
  const double slat = std::sin(lat);
  const double clat = std::cos(lat);
  const double slon = std::sin(lon);
  const double clon = std::cos(lon);
  // Prime vertical radius of curvature.
  const double N = kWgs84A / std::sqrt(1.0 - kWgs84E2 * slat * slat);
  Eigen::Vector3d ecef;
  ecef.x() = (N + alt_m) * clat * clon;
  ecef.y() = (N + alt_m) * clat * slon;
  ecef.z() = (N * (1.0 - kWgs84E2) + alt_m) * slat;
  return ecef;
}

Eigen::Vector3d ecefToEnu(const Eigen::Vector3d& ecef, double lat0_deg, double lon0_deg,
                          double alt0_m) {
  const Eigen::Vector3d ref = wgs84ToEcef(lat0_deg, lon0_deg, alt0_m);
  const Eigen::Vector3d d = ecef - ref;
  const double lat = lat0_deg * kDeg2Rad;
  const double lon = lon0_deg * kDeg2Rad;
  const double slat = std::sin(lat);
  const double clat = std::cos(lat);
  const double slon = std::sin(lon);
  const double clon = std::cos(lon);
  // Rotate the ECEF delta into the local East-North-Up basis at the datum.
  Eigen::Vector3d enu;
  enu.x() = -slon * d.x() + clon * d.y();
  enu.y() = -slat * clon * d.x() - slat * slon * d.y() + clat * d.z();
  enu.z() = clat * clon * d.x() + clat * slon * d.y() + slat * d.z();
  return enu;
}

Eigen::Vector3d wgs84ToEnu(double lat_deg, double lon_deg, double alt_m, double lat0_deg,
                           double lon0_deg, double alt0_m) {
  return ecefToEnu(wgs84ToEcef(lat_deg, lon_deg, alt_m), lat0_deg, lon0_deg, alt0_m);
}

GnssGate::GnssGate(const PreprocGnss& cfg, const IVelocitySource* velocity_source,
                   TelemetrySink* telemetry)
    : cfg_(cfg),
      velocity_source_(velocity_source),
      telemetry_(telemetry),
      accepted_(kAcceptedRingCapacity) {}

GnssVerdict::Reason GnssGate::qualityReason(const GnssFix& fix) const {
  if (fixRank(fix.fix) < fixRank(cfg_.min_fix_type)) {
    return GnssVerdict::Reason::WeakFix;
  }
  // 0 = source did not report it (NavSatFix-fed rigs), so the sat-count gate is skipped.
  if (fix.num_sats != 0 && static_cast<int>(fix.num_sats) < cfg_.min_sats) {
    return GnssVerdict::Reason::FewSats;
  }
  if (fix.cov_enu.trace() > cfg_.max_pos_var) {
    return GnssVerdict::Reason::HighCov;
  }
  // DOP is not carried on GnssFix; treated as satisfied here.
  return GnssVerdict::Reason::Accepted;
}

bool GnssGate::spoofVelocity(const GnssFix& fix, const Eigen::Vector3d& enu) {
  if (velocity_source_ == nullptr) {
    return false;  // no IMU velocity to disagree with; cannot assert spoofing
  }
  const Duration window = static_cast<Duration>(cfg_.spoof_window_ms) * 1'000'000;

  // Find the oldest accepted position at least one window old to difference against. The
  // ring is time-ordered oldest->newest, so the first qualifying element is the oldest.
  bool have_ref = false;
  Eigen::Vector3d p_ref = Eigen::Vector3d::Zero();
  Timestamp t_ref = 0;
  for (std::size_t i = 0; i < accepted_.size(); ++i) {
    const StampedEnu& e = accepted_[i];
    if (fix.stamp - e.stamp >= window) {
      p_ref = e.enu;
      t_ref = e.stamp;
      have_ref = true;
      break;
    }
  }
  if (!have_ref) {
    return false;  // not enough history yet
  }

  const double dt = to_seconds(fix.stamp - t_ref);
  if (dt <= 0.0) {
    return false;
  }
  const Eigen::Vector3d v_gnss = (enu - p_ref) / dt;

  Eigen::Vector3d v_imu;
  if (!velocity_source_->velocity(fix.stamp, window, &v_imu)) {
    return false;  // no IMU velocity estimate yet
  }

  const double disagree = (v_gnss - v_imu).norm();
  if (disagree > cfg_.spoof_vel_thresh) {
    ++spoof_run_;
  } else {
    spoof_run_ = 0;
  }
  return spoof_run_ >= static_cast<std::uint32_t>(cfg_.spoof_persist);
}

GnssVerdict GnssGate::evaluate(const GnssFix& fix) {
  GnssVerdict v;
  if (!cfg_.enable) {
    v.accepted = false;
    v.reason = GnssVerdict::Reason::Disabled;
    return v;
  }

  const GnssVerdict::Reason q = qualityReason(fix);
  if (q != GnssVerdict::Reason::Accepted) {
    v.accepted = false;
    v.reason = q;
    if (telemetry_ != nullptr) {
      telemetry_->scalar(keys::gnss::Rejected, 1.0, fix.stamp);
    }
    MERIDIAN_WARN(kLogModule, "event", "gnss/quality_reject", "reason",
                  static_cast<int>(q), "sats", static_cast<int>(fix.num_sats));
    return v;
  }

  // Latch the provisional datum on the first quality-passing fix; ENU is relative to it.
  if (!datum_.has_value()) {
    datum_ = DatumOrigin{fix.lat_deg, fix.lon_deg, fix.alt_m};
  }
  const Eigen::Vector3d enu =
      wgs84ToEnu(fix.lat_deg, fix.lon_deg, fix.alt_m, datum_->lat_deg, datum_->lon_deg,
                 datum_->alt_m);
  v.enu_provisional = enu;

  if (cfg_.spoof_check && spoofVelocity(fix, enu)) {
    v.accepted = false;
    v.reason = GnssVerdict::Reason::SpoofVelocity;
    if (telemetry_ != nullptr) {
      telemetry_->scalar(keys::gnss::Spoof, 1.0, fix.stamp);
    }
    MERIDIAN_WARN(kLogModule, "event", "gnss/spoof_suspected", "run", spoof_run_);
    return v;
  }

  v.accepted = true;
  v.reason = GnssVerdict::Reason::Accepted;
  accepted_.push(StampedEnu{fix.stamp, enu});
  if (telemetry_ != nullptr) {
    telemetry_->scalar(keys::gnss::Accepted, 1.0, fix.stamp);
  }
  return v;
}

}  // namespace meridian
