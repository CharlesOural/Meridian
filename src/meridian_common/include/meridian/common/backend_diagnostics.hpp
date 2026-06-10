#pragma once

#include <cstdint>

namespace meridian {

// Pull-side introspection snapshot returned by IBackEnd::diagnostics(): cumulative
// graph composition counters and the state of the last incremental optimization.
struct BackEndDiagnostics {
  std::uint64_t num_keyframes = 0;       // keyframe variables in the graph
  std::uint64_t num_loops = 0;           // loop-closure factors accepted into the graph
  std::uint64_t num_loops_rejected = 0;  // loop candidates rejected by the consistency gate
  std::uint64_t num_gnss_factors = 0;    // GNSS position factors in the graph
  double isam_update_ms = 0.0;           // wall time of the last incremental update
  bool last_optimize_diverged = false;   // true when the last update was abandoned/rolled back
  double chi2 = 0.0;                     // total weighted squared error after the last update
  int variables_relinearized = 0;        // variables relinearized by the last update
  int optimize_lag = 0;                  // keyframes queued behind the optimizer
  bool datum_locked = false;             // true once the GNSS datum (origin/yaw) is fixed
  std::uint64_t fallback_count = 0;      // times the optimizer fell back to a safe restart
};

}  // namespace meridian
