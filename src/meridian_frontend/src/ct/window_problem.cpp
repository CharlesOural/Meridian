#include "ct/window_problem.hpp"

#include <ceres/manifold.h>
#include <ceres/problem.h>

#include <algorithm>
#include <vector>

#include "ct/residuals_gnss.hpp"

namespace meridian::ct {

void buildWindowProblem(ceres::Problem& problem, WindowProblemInputs& in,
                        WindowExposureScratch* expo) {
  SplineWindow& spline = *in.spline;
  BiasKnots& bias = *in.bias;
  GravityBlock& gravity = *in.gravity;

  // LiDAR point-to-plane residuals at the current spline (capped set).
  if (in.capped_hits != nullptr) {
    addLidarResiduals(problem, spline, in.T_fe_lidar, *in.capped_hits, in.lidar_cfg);
  }

  // IMU derivative residuals over the sweep span + bias random walk.
  if (in.imu != nullptr) {
    addImuResiduals(problem, spline, bias, gravity, *in.imu, in.t_begin, in.t_end, in.weights);
  }
  addBiasRandomWalk(problem, bias, in.weights);

  // Tail anchors over the span past the newest measurement.
  if (!in.tail_times.empty()) {
    addTailAnchors(problem, spline, in.tail_times, in.seed_vel_end, in.seed_omega_end,
                   in.tail_sigma_vel, in.tail_sigma_rate);
  }

  // Under-excitation regularizer (only when the round engaged it).
  if (in.motion_reg_engaged && !in.motion_reg_times.empty()) {
    addMotionRegularizer(problem, spline, in.motion_reg_times, in.motion_reg_weight,
                         in.motion_reg_accel_weight, in.motion_reg_gyro_weight);
  }

  // Photometric residuals replayed from the captured patches into the SAME problem,
  // against a single-frame exposure chain seeded exactly as the live build.
  if (!in.visual_patches.empty() && expo != nullptr) {
    expo->chain.addFrame(in.t_image, in.inv_expo);
    replayVisualResiduals(problem, spline, in.t_image, expo->chain, /*expo_index=*/0,
                          in.visual_patches);
    if (in.exposure_estimate_en && in.inv_expo_std > 0.0) {
      expo->chain.addTo(problem, in.inv_expo_prior, in.inv_expo_std);
    } else if (problem.HasParameterBlock(expo->chain.block(0))) {
      problem.SetParameterBlockConstant(expo->chain.block(0));
    }
  }

  // GNSS absolute-position residuals.
  for (const GnssFactor& g : in.gnss_factors) {
    addGnssResidual(problem, spline, g.z_world, in.t_fe_ant, g.t_fix, g.sqrt_info, in.gnss_huber);
  }

  // Gravity is pinned to the unit sphere then held constant.
  if (problem.HasParameterBlock(gravity.data())) {
    if (problem.GetManifold(gravity.data()) == nullptr) {
      problem.SetManifold(gravity.data(), makeGravityManifold());
    }
    problem.SetParameterBlockConstant(gravity.data());
  }

  // Hold the unsupported tail knots constant from the first pinned index onward.
  for (int j = in.first_pinned_knot; j < spline.numKnots(); ++j) {
    for (double* p : {spline.so3KnotData(j), spline.r3KnotData(j)}) {
      if (problem.HasParameterBlock(p)) {
        problem.SetParameterBlockConstant(p);
      }
    }
  }

  std::vector<double*> params;
  problem.GetParameterBlocks(&params);
  for (double* p : params) {
    if (problem.ParameterBlockSize(p) == 4 && problem.GetManifold(p) == nullptr) {
      problem.SetManifold(p, new ceres::EigenQuaternionManifold());
    }
  }

  // Bias box bounds, with the stored value first projected into the box.
  for (int k = 0; k < bias.numKnots(); ++k) {
    double* gp = bias.gyroBlock(k);
    double* ap = bias.accelBlock(k);
    if (problem.HasParameterBlock(gp) && in.bias_gyr_max > 0.0) {
      for (int a = 0; a < 3; ++a) {
        gp[a] = std::clamp(gp[a], -in.bias_gyr_max, in.bias_gyr_max);
        problem.SetParameterLowerBound(gp, a, -in.bias_gyr_max);
        problem.SetParameterUpperBound(gp, a, in.bias_gyr_max);
      }
    }
    if (problem.HasParameterBlock(ap) && in.bias_acc_max > 0.0) {
      for (int a = 0; a < 3; ++a) {
        ap[a] = std::clamp(ap[a], -in.bias_acc_max, in.bias_acc_max);
        problem.SetParameterLowerBound(ap, a, -in.bias_acc_max);
        problem.SetParameterUpperBound(ap, a, in.bias_acc_max);
      }
    }
  }

  // Marginalization prior (if any) pins the gauge; otherwise pin the leading active
  // segment's interior support knots, leaving the oldest knot free to be marginalized.
  if (in.prior != nullptr) {
    problem.AddResidualBlock(in.prior->makeCost(), nullptr, [&] {
      std::vector<double*> ptrs;
      ptrs.reserve(in.prior->blocks().size());
      for (const auto& b : in.prior->blocks()) {
        ptrs.push_back(b.ptr);
      }
      return ptrs;
    }());
  } else {
    const SplineWindow::SegmentRef seg = spline.segmentFor(in.gauge_seg_t);
    if (problem.HasParameterBlock(seg.so3_knots[1])) {
      problem.SetParameterBlockConstant(seg.so3_knots[1]);
    }
    for (int j = 1; j <= 2; ++j) {
      if (problem.HasParameterBlock(seg.r3_knots[j])) {
        problem.SetParameterBlockConstant(seg.r3_knots[j]);
      }
    }
  }
}

}  // namespace meridian::ct
