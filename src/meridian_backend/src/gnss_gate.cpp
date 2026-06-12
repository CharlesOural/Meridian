#include "gnss_gate.hpp"

namespace meridian::backend {

GnssGate::Decision GnssGate::evaluate(const GnssFix& fix, double marginal_pos_trace,
                                      double travelled_since_last, const BackendConfig& cfg) {
  const double cov_trace = fix.cov_enu.trace();

  // Quality: a no-fix solution carries no usable position, and a fix looser than the
  // configured covariance cap would only inject noise.
  if (fix.fix == GnssFix::FixType::None || cov_trace > cfg.gnss_max_cov) {
    return Decision::RejectQuality;
  }

  // Skip-if-confident: drop a fix that is no tighter than the back-end's own position
  // marginal scaled by k^2 (variances scale with k^2). A huge marginal trace passed in
  // makes the right-hand side enormous, disabling this gate.
  if (cfg.gnss_skip_if_confident &&
      cov_trace >= cfg.gnss_skip_confidence_k * cfg.gnss_skip_confidence_k * marginal_pos_trace) {
    return Decision::SkipConfident;
  }

  // Spacing decimation: keep at most one admitted fix per gnss_min_spacing of travel.
  if (travelled_since_last < cfg.gnss_min_spacing) {
    return Decision::SkipSpacing;
  }

  return Decision::Accept;
}

}  // namespace meridian::backend
