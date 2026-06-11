#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#include "ct/marginalization.hpp"
#include "ct/residuals_imu.hpp"
#include "ct/residuals_lidar.hpp"
#include "ct/residuals_visual.hpp"
#include "ct/spline_window.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"

namespace ceres {
class Problem;
}  // namespace ceres

namespace meridian::ct {

// One absolute-position GNSS measurement, decoupled from CtFrontEnd's nested type so
// the shared builder and the keyframe worker can carry it without that header.
struct GnssFactor {
  Eigen::Vector3d z_world = Eigen::Vector3d::Zero();
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
  Timestamp t_fix = 0;
};

// Everything one outer round's window problem is built from. The decisions a round
// makes on the hot thread (which hits survived the factor cap, whether the motion
// regularizer engaged, which trailing knots are pinned) are passed in as data so the
// builder is a pure function of its inputs: the same inputs always assemble the same
// ceres::Problem, residual-for-residual and pin-for-pin. The keyframe worker captures
// these inputs against cloned state and replays the builder to get a bit-exact problem.
//
// All pointers are non-owning and must outlive the build call. The spline/bias/gravity
// may be the live window (on the hot thread) or worker-owned clones; either way the
// builder reads their current knot values as the linearization point.
struct WindowProblemInputs {
  // Trajectory state (mutated in place: gravity gets its manifold, knots get bounds /
  // manifolds / const pins). On the worker these are clones, so no live state moves.
  SplineWindow* spline = nullptr;
  BiasKnots* bias = nullptr;
  GravityBlock* gravity = nullptr;
  MarginalizationPrior* prior = nullptr;  // null when no prior exists yet

  Timestamp t_begin = 0;
  Timestamp t_end = 0;

  // LiDAR point-to-plane residual inputs: the factor-cap survivors of this round.
  const std::vector<LidarHit>* capped_hits = nullptr;
  Pose T_fe_lidar;
  FrontendLidar lidar_cfg;

  // IMU + bias-tie inputs.
  const std::vector<ImuSample>* imu = nullptr;
  ImuWeights weights;

  // Tail anchors over the unsupported span: covered times and the seed extrapolation.
  std::vector<Timestamp> tail_times;
  Eigen::Vector3d seed_vel_end = Eigen::Vector3d::Zero();
  Eigen::Vector3d seed_omega_end = Eigen::Vector3d::Zero();
  double tail_sigma_vel = 0.2;
  double tail_sigma_rate = 0.1;

  // Under-excitation regularizer: engaged decision + evaluation times. When
  // motion_reg_engaged is false the regularizer is not added (matching the round's gate).
  bool motion_reg_engaged = false;
  std::vector<Timestamp> motion_reg_times;
  double motion_reg_weight = 0.0;
  double motion_reg_accel_weight = 0.0;
  double motion_reg_gyro_weight = 0.0;

  // Visual photometric residuals replayed from captured patches. Empty disables the
  // stage. The exposure block is rebuilt from these values so its const-vs-free state
  // matches the live build.
  std::vector<VisualPatchParams> visual_patches;
  Timestamp t_image = 0;
  double inv_expo = 1.0;  // seed for the single in-window exposure frame
  double inv_expo_prior = 1.0;
  double inv_expo_std = 0.0;
  bool exposure_estimate_en = false;
  FrontendVisual visual_cfg;

  // GNSS absolute-position residuals.
  std::vector<GnssFactor> gnss_factors;
  Eigen::Vector3d t_fe_ant = Eigen::Vector3d::Zero();
  double gnss_huber = 3.0;

  // Bias box bounds (<= 0 disables that axis's bound, matching solveWindow).
  double bias_gyr_max = 0.0;
  double bias_acc_max = 0.0;

  // Leading-active-segment t for the gauge pins when no prior exists.
  Timestamp gauge_seg_t = 0;
  // First knot index held constant as an unsupported lever-arm tail (>= numKnots
  // disables the tail pin). Both classes of pin in solveWindow collapse to this index.
  int first_pinned_knot = 0;
};

// The single owner of the locally held exposure block for one rebuilt problem: the
// builder needs a place to store the in-window exposure scalar with a stable address
// for the photometric residuals and the const/bound registration. The hot thread keeps
// its ExposureChain alive across the round; the worker hands the builder one of these.
// Empty when no visual patches are present.
struct WindowExposureScratch {
  ExposureChain chain;
  explicit WindowExposureScratch(const FrontendVisual& cfg) : chain(cfg) {}
};

// Assembles the per-round window problem into `problem` from `in`. `expo`, when
// non-null, supplies stable exposure-block storage for the photometric residuals
// (required iff in.visual_patches is non-empty). Mirrors solveWindow's per-round
// construction exactly; called by both solveWindow and the keyframe worker.
void buildWindowProblem(ceres::Problem& problem, WindowProblemInputs& in,
                        WindowExposureScratch* expo);

}  // namespace meridian::ct
