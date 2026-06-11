#pragma once

#include <Eigen/Core>
#include <functional>
#include <vector>

#include "lio/imu_tracker.hpp"
#include "lio/voxel_grid_map.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace meridian::lio {

// One scan-to-map solve. `cov_right` is the pose covariance in the right/body
// translation-first [rho; phi] tangent; `info_body` is the unscaled data-term
// information in the same chart (the per-axis observability source). `chi` is the
// final sum of squared residuals over the n_corr correspondences. When the solve
// fails its gates (zero correspondences, or n_corr below the keypoint floor)
// `converged` is false and `cov_right` keeps its identity placeholder.
struct RegistrationResult {
  Pose pose;  // T_world_body at the sweep end
  Eigen::Matrix<double, 6, 6> cov_right = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 6> info_body = Eigen::Matrix<double, 6, 6>::Zero();
  int n_corr = 0;
  int gn_iters = 0;
  double dx_norm = 0.0;
  double chi = 0.0;
  bool converged = false;
};

// Nearest-map-point lookup: writes the closest stored point to `query_world` and
// returns true when one lies within the correspondence gate. The lookup owns the gate.
using NearestLookup =
    std::function<bool(const Eigen::Vector3d& query_world, Eigen::Vector3d* neighbor)>;

// Stateless per-sweep machinery: constant-screw deskew of the raw sweep and the
// Gauss-Newton point-to-point alignment against the local map.
class ScanRegistration {
public:
  explicit ScanRegistration(const FrontendLio& cfg);

  // Warp every point to its position at the sweep end `t_end` under the constant
  // body screw (omega_body, vel_body), expressing the result in the body frame at
  // t_end. Points outside [min, cfg.max_range_m] are dropped.
  std::vector<Eigen::Vector3d> deskew(const LidarScan& scan, const Pose& T_body_lidar,
                                      const Eigen::Vector3d& omega_body,
                                      const Eigen::Vector3d& vel_body, Timestamp t_end) const;

  // Iterate association + Gauss-Newton from `T_world_body_guess` until the update norm
  // falls below cfg.convergence_eps or cfg.icp_max_iterations is hit. `prior` drives
  // the adaptive gravity-direction regularizer. Accumulation is sequential over the
  // keypoint order, so the solve is deterministic.
  RegistrationResult registerScan(const std::vector<Eigen::Vector3d>& keypoints_body,
                                  const VoxelGridMap& map, const Pose& T_world_body_guess,
                                  const MotionPrior& prior) const;

  // Identical solve driven by a caller-supplied nearest lookup instead of the map.
  RegistrationResult registerScan(const std::vector<Eigen::Vector3d>& keypoints_body,
                                  const NearestLookup& nearest, const Pose& T_world_body_guess,
                                  const MotionPrior& prior) const;

private:
  FrontendLio cfg_;
};

}  // namespace meridian::lio
