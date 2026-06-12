#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/Core>

#include "meridian/common/ring_buffer.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// Velocity feedback seam for the spoof check. The pipeline injects a small adapter over
// L2's live NavState (or integrated IMU pre-trajectory) so this package stays decoupled
// from the front-end.
class IVelocitySource {
 public:
  virtual ~IVelocitySource() = default;
  // Mean body velocity [m/s] in the provisional ENU frame over [t - window, t].
  // Returns false if no velocity estimate exists yet (pre-trajectory).
  virtual bool velocity(Timestamp t, Duration window, Eigen::Vector3d* v_enu) const = 0;
};

// WGS84 geodetic -> ECEF [m].
Eigen::Vector3d wgs84ToEcef(double lat_deg, double lon_deg, double alt_m);

// ECEF [m] -> local ENU [m] about a geodetic datum origin.
Eigen::Vector3d ecefToEnu(const Eigen::Vector3d& ecef, double lat0_deg, double lon0_deg,
                          double alt0_m);

// WGS84 geodetic -> local ENU [m] about a geodetic datum origin (compose of the above).
Eigen::Vector3d wgs84ToEnu(double lat_deg, double lon_deg, double alt_m, double lat0_deg,
                           double lon0_deg, double alt0_m);

// The verdict the gate emits per fix. enu_provisional is centred on the first accepted
// fix and is provisional only; the authoritative datum belongs to the back-end.
struct GnssVerdict {
  enum class Reason {
    Accepted,
    Disabled,
    WeakFix,
    FewSats,
    HighCov,
    HighDop,
    SpoofVelocity,
    SpoofClock,
  };
  bool accepted = false;
  Reason reason = Reason::Accepted;
  Eigen::Vector3d enu_provisional = Eigen::Vector3d::Zero();
};

// GNSS quality gate + velocity-consistency spoof check + LLA->ENU conversion.
// Maintains a provisional ENU datum (first accepted fix) and a 1 s ring of accepted
// positions for the GNSS-vs-IMU velocity comparison.
//
// Thread-confined: driven from a single stage thread.
class GnssGate {
 public:
  // velocity_source may be nullptr (spoof velocity check then degrades to a pass, since
  // no IMU velocity is available to disagree with). telemetry may be nullptr (no-op).
  GnssGate(const PreprocGnss& cfg, const IVelocitySource* velocity_source,
           TelemetrySink* telemetry);

  // Runs the quality gate, then (if the fix passes and spoof_check is on) the velocity
  // spoof check. Updates the datum and the accepted-position ring on accept.
  GnssVerdict evaluate(const GnssFix& fix);

  // True once a datum origin has been latched from the first accepted fix.
  bool hasDatum() const { return datum_.has_value(); }

 private:
  // dop is unavailable in GnssFix; the quality gate runs the fix-type / sats / cov
  // checks and treats DOP as satisfied.
  GnssVerdict::Reason qualityReason(const GnssFix& fix) const;
  // Returns true if the fix looks spoofed under the velocity-consistency window.
  bool spoofVelocity(const GnssFix& fix, const Eigen::Vector3d& enu);

  struct DatumOrigin {
    double lat_deg = 0;
    double lon_deg = 0;
    double alt_m = 0;
  };
  struct StampedEnu {
    Timestamp stamp = 0;
    Eigen::Vector3d enu = Eigen::Vector3d::Zero();
  };

  PreprocGnss cfg_;
  const IVelocitySource* velocity_source_ = nullptr;
  TelemetrySink* telemetry_ = nullptr;

  std::optional<DatumOrigin> datum_;
  RingBuffer<StampedEnu> accepted_;  // 1 s window of accepted positions
  std::uint32_t spoof_run_ = 0;      // consecutive spoof-suspect windows
};

}  // namespace meridian
