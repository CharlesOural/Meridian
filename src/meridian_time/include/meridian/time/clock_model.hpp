#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "meridian/common/time.hpp"
#include "meridian/time/stamp_source.hpp"

namespace meridian {

// One clock per rig device; flat because there is one device per modality.
enum class ClockId : std::uint16_t { Host = 0, Lidar, Imu, Cam, Gnss };

inline constexpr std::size_t kClockCount = 5;

// Linearized device-clock <-> host-timeline relation for one device, held at
// the reference instant t_ref. The forward model is
//   t_dev = (1 + skew_ppm*1e-6) * t_H + offset_ns,
// so the inverse used to place an event on the host timeline is
//   t_H = (t_dev - offset_ns) / (1 + skew_ppm*1e-6).
struct ClockState {
  double      offset_ns     = 0.0;   // device - host at t_ref
  double      skew_ppm      = 0.0;   // fractional rate error, parts-per-million
  Timestamp   t_ref         = 0;     // instant the linearization holds at
  double      offset_std_ns = 0.0;   // 1-sigma of the offset estimate
  bool        disciplined   = false; // hardware (PTP/PPS) drives this clock
  StampSource source        = StampSource::ArrivalOnly;
  Timestamp   last_update   = 0;
};

// Tracks per-device clock offset/skew and converts device-clock values to the
// single monotonic Meridian timeline. It does not implement PTP/PPS; it only
// observes their quality (on_ptp_stats / on_pps_edge) and runs a lightweight
// software estimator (on_correspondence) for clocks no hardware disciplines.
//
// Thread-safe: read from sensor ingest threads, written by the time estimator.
class ClockModel {
 public:
  // Map a device-clock value to the Meridian (host) timeline. Disciplined
  // clocks pass through unchanged (offset~0, skew~0); other clocks apply the
  // offset/skew inverse relation around t_ref.
  Timestamp to_meridian(Timestamp device_ns, ClockId id) const;

  // Hardware quality observers, fed by the platform time daemon shim. A locked
  // PTP/PPS clock is marked disciplined with offset/skew driven toward zero;
  // the residual offset is retained for health.
  void on_ptp_stats(ClockId id, double offset_ns, double path_delay_ns,
                    bool locked);
  void on_pps_edge(Timestamp host_ns_of_edge);

  // Software estimator update: one (device, matched-host) correspondence with
  // its 1-sigma measurement noise. No-op for disciplined clocks.
  void on_correspondence(ClockId id, Timestamp device_ns, Timestamp host_ns,
                         double meas_std_ns);

  bool        ptp_locked(ClockId id) const;
  bool        disciplined(ClockId id) const;
  StampSource stamp_source(ClockId id) const;
  ClockState  state(ClockId id) const;

 private:
  // Recursive-least-squares state for one clock's affine fit
  //   d(tau) = beta0 + beta1*tau,   d = t_dev - t_H,  tau = (t_H - t_ref) / 1e9,
  // so beta0 is the offset (ns) at t_ref and beta1 is the rate error in
  // ns-per-second (= skew_ppm * 1000). p_* are the upper triangle of the 2x2
  // estimate covariance, propagated by the standard RLS recursion.
  struct RlsState {
    double beta0 = 0.0;
    double beta1 = 0.0;
    double p00 = 0.0;
    double p01 = 0.0;
    double p11 = 0.0;
    bool   seeded = false;
  };

  ClockState&       at(ClockId id);
  const ClockState& at(ClockId id) const;

  std::array<ClockState, kClockCount> states_{};
  std::array<RlsState, kClockCount>   rls_{};

  // Last host instant a PPS edge was observed; reported via the Gnss clock.
  Timestamp last_pps_host_ns_ = 0;

  // Guards all state; held by every public method so reads from ingest threads
  // never observe a partial write from the estimator.
  mutable std::mutex mutex_;
};

}  // namespace meridian
