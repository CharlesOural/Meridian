#pragma once

#include "meridian/common/observability.hpp"

namespace meridian {

// Pull-side introspection snapshot returned by IFrontEnd::diagnostics(): the last
// sweep's stage timings, residual statistics, and per-axis observability.
struct FrontEndDiagnostics {
  double scan_time_ms = 0.0;    // full ingest() wall time for the last sweep
  double solve_time_ms = 0.0;   // window-solve share of scan_time_ms
  double deskew_time_ms = 0.0;  // deskew/warp share of scan_time_ms
  int effective_points = 0;     // points that found a plane in the last solve
  double mean_residual = 0.0;   // mean absolute residual over effective points
  double final_residual = 0.0;  // cost at convergence
  ObservabilityReport observability;
  int iterations = 0;     // solver iterations used by the last solve
  int outer_iters = 0;    // re-association passes run by the last solve
  bool restarted = false;  // true once after a window restart, until the next solve
  // True when the wall-clock-bounded solve was cut off at the deadline before
  // convergence (best-iterate-so-far returned). Always false on the deterministic
  // replay path, which ignores the time limit.
  bool deadline_hit = false;
  // Running count of control-point knots Schur-marginalized into the sliding-window
  // prior since construction, and the row dimension of the current prior residual
  // (0 while no prior exists). Together they confirm the prior chain has engaged.
  int knots_marginalized = 0;
  int prior_residual_dim = 0;
};

}  // namespace meridian
