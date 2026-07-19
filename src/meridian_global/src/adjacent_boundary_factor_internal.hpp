#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <cstddef>
#include <string>

#include "meridian/core/canonical_verification.hpp"
#include "meridian/core/result.hpp"
#include "meridian/core/sparse_map_api.hpp"

namespace meridian::global::adjacent_internal {

// Non-informative placement of one frozen odometry chart in the map chart.
// The rotation is restricted to gravity-preserving yaw.
struct OdomEpochChartPlacement {
  core::OdomEpoch odom_epoch;
  core::Pose3d H_map_odom;
};

struct AdjacentBoundaryKeys {
  gtsam::Key anchor_from{};
  gtsam::Key velocity_from{};
  gtsam::Key gyro_bias_from{};
  gtsam::Key accel_bias_from{};
  gtsam::Key anchor_to{};
  gtsam::Key velocity_to{};
  gtsam::Key gyro_bias_to{};
  gtsam::Key accel_bias_to{};
};

struct AdjacentBoundaryFactorLimits {
  std::size_t maximum_rows{4096U};
  std::size_t maximum_coefficients{4096U * 30U};
};

enum class AdjacentBoundaryAdapterErrorCode {
  InvalidLimits,
  InvalidTransition,
  InvalidSparseLink,
  OdomEpochMismatch,
  InvalidChartPlacement,
  DuplicateKey,
  CapacityExceeded,
};

struct AdjacentBoundaryAdapterError {
  AdjacentBoundaryAdapterErrorCode code{AdjacentBoundaryAdapterErrorCode::InvalidTransition};
  std::string detail;
};

using AdjacentBoundaryFactorPtr = boost::shared_ptr<gtsam::NoiseModelFactor>;

// Builds the exact nonlinear chart adapter for the frozen canonical rows. The
// returned factor owns copies of all rows, right-hand sides, chart centers,
// frame transforms, and placement values; no caller-owned storage is retained.
[[nodiscard]] core::Result<AdjacentBoundaryFactorPtr, AdjacentBoundaryAdapterError>
makeAdjacentBoundaryFactor(const core::SparseSubmapSeal& predecessor,
                           const core::SparseSubmapSeal& current,
                           const OdomEpochChartPlacement& placement,
                           const AdjacentBoundaryKeys& keys,
                           const AdjacentBoundaryFactorLimits& limits = {});

}  // namespace meridian::global::adjacent_internal
