#pragma once

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

namespace meridian::backend {

// Per-fix admission gate for GNSS measurements. Decides, for one fix, whether it should
// become a graph factor (Accept) or be dropped for one of three reasons: poor quality,
// the back-end already being tighter than the fix (skip-if-confident), or insufficient
// travel since the last admitted fix (spacing decimation). The gate is stateless except
// for the spacing baseline, which the caller advances and resets through note_admitted.
class GnssGate {
public:
  enum class Decision { Accept, RejectQuality, SkipConfident, SkipSpacing };

  // marginal_pos_trace: trace of the back-end's 3x3 position marginal at the anchor (a
  // large value disables the skip-if-confident gate). travelled_since_last: arc length
  // travelled since the last ADMITTED fix (the caller tracks it).
  Decision evaluate(const GnssFix& fix, double marginal_pos_trace, double travelled_since_last,
                    const BackendConfig& cfg);

  // The fix was admitted: the caller's spacing accumulator should restart from here.
  void note_admitted() {}
};

}  // namespace meridian::backend
