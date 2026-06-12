#pragma once

#include <optional>

#include <Eigen/Core>

#include "meridian/common/gaussian.hpp"
#include "meridian/common/observability.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

// Shape the loop-closure covariance the back-end consumes. Input is the GICP information
// matrix in small_gicp's ROTATION-FIRST order; output is a PoseCov6 covariance in Meridian's
// TRANSLATION-FIRST [rho; phi] order. The shaping (Appendix-R rule, §9.2):
//   Sigma = s(fitness) * (H + lambda I)^-1, inflated along degenerate eigen-directions,
//   permuted to translation-first ONCE, then loosened on axes either endpoint reports as
//   under-observed, with a small PSD floor. Lower fitness => larger Sigma => a marginal loop
//   barely tugs the graph.
PoseCov6 shapeLoopCov(const Eigen::Matrix<double, 6, 6>& info_rot_first, double fitness,
                      const PlaceConfig& cfg,
                      const std::optional<ObservabilityReport>& obs_from,
                      const std::optional<ObservabilityReport>& obs_to);

}  // namespace meridian
