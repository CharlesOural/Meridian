#include "meridian/time/clock_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>

namespace meridian {

namespace {

// Largest |residual offset| (ns) we still treat a PTP/PPS lock as healthy.
// Beyond this the source degrades to ArrivalOnly even while locked.
constexpr double kPtpHealthyResidualNs = 50'000.0;

// Wide priors that seed a fresh affine fit: the first few correspondences then
// define the offset (ns^2) and the rate slope (ns-per-second squared) rather
// than crawling from zero.
constexpr double kPriorOffsetVarNs2 = 1e18;
constexpr double kPriorSlopeVarNs2PerS2 = 1e12;

// RLS forgetting factor (<1) so the fit slowly discounts old correspondences
// and keeps tracking drift instead of freezing once it has converged.
constexpr double kForgetting = 0.995;

// Convert the fitted slope (device-host drift in ns per host-second) to ppm:
//   skew_ppm = (ns/s) / 1e9 * 1e6 = (ns/s) * 1e-3.
constexpr double kNsPerSecToPpm = 1e-3;

}  // namespace

ClockState& ClockModel::at(ClockId id) {
  return states_[static_cast<std::size_t>(id)];
}

const ClockState& ClockModel::at(ClockId id) const {
  return states_[static_cast<std::size_t>(id)];
}

Timestamp ClockModel::to_meridian(Timestamp device_ns, ClockId id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const ClockState& s = at(id);

  // Disciplined clocks are already on the host timeline (offset/skew ~ 0).
  if (s.disciplined) {
    return device_ns;
  }

  // The forward model anchored at t_ref is
  //   t_dev = t_H + offset + skew*1e-6*(t_H - t_ref),
  // where offset is the device-host difference at t_ref. Solving for t_H:
  //   t_H = (t_dev - offset + skew*1e-6*t_ref) / (1 + skew*1e-6).
  const double skew = s.skew_ppm * 1e-6;
  const double rate = 1.0 + skew;
  const double t_h = (static_cast<double>(device_ns) - s.offset_ns +
                      skew * static_cast<double>(s.t_ref)) /
                     rate;
  return static_cast<Timestamp>(std::llround(t_h));
}

void ClockModel::on_ptp_stats(ClockId id, double offset_ns,
                              double /*path_delay_ns*/, bool locked) {
  const std::lock_guard<std::mutex> lock(mutex_);
  ClockState& s = at(id);

  if (locked) {
    // Hardware holds this clock on the host timeline; the conversion is a
    // pass-through. The reported offset is the residual error the daemon could
    // not remove, kept for health rather than applied to to_meridian.
    s.disciplined = true;
    s.offset_ns = 0.0;
    s.skew_ppm = 0.0;
    s.offset_std_ns = std::abs(offset_ns);
    s.source = (std::abs(offset_ns) <= kPtpHealthyResidualNs)
                   ? StampSource::HwPtp
                   : StampSource::ArrivalOnly;
  } else {
    // Lock lost: fall back to whatever software offset we last estimated.
    s.disciplined = false;
    if (s.source == StampSource::HwPtp) {
      s.source = StampSource::ArrivalOnly;
    }
  }
}

void ClockModel::on_pps_edge(Timestamp host_ns_of_edge) {
  const std::lock_guard<std::mutex> lock(mutex_);
  // A PPS edge disciplines the GNSS clock against the host timeline.
  last_pps_host_ns_ = host_ns_of_edge;

  ClockState& s = at(ClockId::Gnss);
  s.disciplined = true;
  s.offset_ns = 0.0;
  s.skew_ppm = 0.0;
  s.source = StampSource::HwPps;
  s.t_ref = host_ns_of_edge;
  s.last_update = host_ns_of_edge;
}

void ClockModel::on_correspondence(ClockId id, Timestamp device_ns,
                                   Timestamp host_ns, double meas_std_ns) {
  const std::lock_guard<std::mutex> lock(mutex_);
  ClockState& s = at(id);
  RlsState& r = rls_[static_cast<std::size_t>(id)];

  // Hardware-disciplined clocks ignore software correspondences.
  if (s.disciplined) {
    return;
  }

  // Anchor the fit at the first correspondence and seed it with wide diagonal
  // priors so the early observations define offset and slope rather than being
  // dragged toward zero.
  if (!r.seeded) {
    s.t_ref = host_ns;
    r.beta0 = static_cast<double>(device_ns) - static_cast<double>(host_ns);
    r.beta1 = 0.0;
    r.p00 = kPriorOffsetVarNs2;
    r.p01 = 0.0;
    r.p11 = kPriorSlopeVarNs2PerS2;
    r.seeded = true;
  }

  // Regressor x = [1, tau], tau in seconds since t_ref; target d = t_dev - t_H.
  const double tau = (static_cast<double>(host_ns) -
                      static_cast<double>(s.t_ref)) /
                     static_cast<double>(kNanosPerSecond);
  const double d = static_cast<double>(device_ns) - static_cast<double>(host_ns);
  const double meas_var = meas_std_ns * meas_std_ns;

  // RLS update with forgetting. Px = P * x:
  const double px0 = r.p00 + r.p01 * tau;
  const double px1 = r.p01 + r.p11 * tau;
  // Innovation variance s_inn = lambda*R + x^T P x.
  const double s_inn = kForgetting * meas_var + (px0 + px1 * tau);
  // Gain k = P x / s_inn.
  const double k0 = px0 / s_inn;
  const double k1 = px1 / s_inn;
  // Residual against the current fit.
  const double resid = d - (r.beta0 + r.beta1 * tau);
  r.beta0 += k0 * resid;
  r.beta1 += k1 * resid;
  // Covariance downdate P <- (P - k (Px)^T) / lambda, keeping the symmetric
  // upper triangle.
  r.p00 = (r.p00 - k0 * px0) / kForgetting;
  r.p01 = (r.p01 - k0 * px1) / kForgetting;
  r.p11 = (r.p11 - k1 * px1) / kForgetting;

  // Publish the fit into the externally visible linearization. beta0 is the
  // offset at t_ref; beta1 (ns per host-second) maps to ppm.
  s.offset_ns = r.beta0;
  s.skew_ppm = r.beta1 * kNsPerSecToPpm;
  s.offset_std_ns = std::sqrt(std::max(0.0, r.p00));
  s.last_update = host_ns;
  s.source = StampSource::SwOffset;
}

bool ClockModel::ptp_locked(ClockId id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const ClockState& s = at(id);
  return s.disciplined && s.source == StampSource::HwPtp;
}

bool ClockModel::disciplined(ClockId id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return at(id).disciplined;
}

StampSource ClockModel::stamp_source(ClockId id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return at(id).source;
}

ClockState ClockModel::state(ClockId id) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return at(id);
}

}  // namespace meridian
