#pragma once

#include <array>
#include <optional>

#include <Eigen/Core>

#include "meridian/common/frame.hpp"

namespace meridian {

// Per-axis observability of the front-end's most recent solve, in a named frame.
// score[k] in [0,1]: 1 = fully observable/well-conditioned, 0 = fully degenerate.
// score order is [tx, ty, tz, rx, ry, rz] (translation-first) expressed in `frame`.
// If `eigvecs` is set, the degenerate directions are its columns rather than the
// frame axes, and `score` are the normalized conditioning along those directions.
struct ObservabilityReport {
  Frame frame = Frame::Body;
  std::array<double, 6> score = {1, 1, 1, 1, 1, 1};
  std::optional<Eigen::Matrix<double, 6, 6>> eigvecs;
};

}  // namespace meridian
