#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <vector>

#include "geodetic.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian::backend {

// Result of a datum-lock attempt. T_map_enu maps ENU positions into the map frame:
// p_map = T_map_enu * p_enu. yaw_sigma_rad is the 1-sigma uncertainty of the fitted
// heading (the only weakly observable DoF of a planar similarity fit).
struct DatumResult {
  bool locked = false;
  Pose T_map_enu;
  double yaw_sigma_rad = 0;
};

// Buffers GNSS<->map antenna-position correspondences and fits the 4-DoF similarity
// (yaw about +z + 3-translation, scale = 1, roll = pitch = 0) that aligns the ENU track
// to the map track. The fit only locks once the track carries enough baseline,
// horizontal excitation, and moving fixes to make the yaw observable.
class DatumInitializer {
public:
  // Buffer one accepted correspondence: the ENU position of the fix and the current
  // map-frame antenna position, plus the body speed and the fix stamp.
  void add(const Eigen::Vector3d& p_enu, const Eigen::Vector3d& p_map_antenna, double speed_mps,
           Timestamp stamp);

  // Try the gated 4-DoF yaw+translation fit. Returns locked=true with T_map_enu and the
  // yaw sigma once the baseline, excitation, moving-fix, and yaw-sigma gates all pass;
  // otherwise locked=false (keep buffering).
  DatumResult try_lock(double min_baseline_m, double min_excitation_m, double min_speed_mps,
                       int min_moving_fixes, double yaw_sigma_max_rad);

  std::size_t size() const { return buf_.size(); }

private:
  struct Corr {
    Eigen::Vector3d enu;
    Eigen::Vector3d map;
    double speed;
    Timestamp stamp;
  };
  std::vector<Corr> buf_;
};

}  // namespace meridian::backend
