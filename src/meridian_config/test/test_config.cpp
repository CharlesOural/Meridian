#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"
#include "meridian/config/config_loader.hpp"

namespace meridian {
namespace {

TEST(Config, DefaultValidatePasses) {
  Config c;
  std::string err;
  EXPECT_TRUE(c.validate(&err)) << err;
  EXPECT_TRUE(err.empty());
}

TEST(Config, CrossFieldFailsWithMessage) {
  // map voxel coarser than the surf filter violates the tsdf_voxel_m <= voxel_surf_m rule.
  Config c;
  c.preprocess.lidar.voxel_surf_m = 0.1;
  c.map.tsdf_voxel_m = 0.5;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("voxel_surf_m"), std::string::npos) << err;
}

TEST(Config, ZeroKnotDtFails) {
  Config c;
  c.frontend.spline.knot_dt_ms = 0.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("knot_dt_ms"), std::string::npos) << err;
}

TEST(Config, BlindBeyondRangeFails) {
  Config c;
  c.preprocess.lidar.blind = 200.0;
  c.preprocess.lidar.det_range = 120.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("blind"), std::string::npos) << err;
}

TEST(ConfigLoader, RoundTripsFields) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_test_config.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  pipeline: { mode: replay, threads: { frontend: 1, backend: 1, map: 1 } }\n"
         "  time: { source: host, max_skew_ms: 7 }\n"
         "  sensors:\n"
         "    lidar: { topic: /lidar/points, nominal_rate_hz: 10, ptp: false }\n"
         "  preprocess: { voxel_surf_m: 0.4 }\n"
         "  frontend:\n"
         "    spline: { order: cubic, knot_dt_ms: 20, window_knots: 6 }\n"
         "    keyframe: { dist_m: 2.0 }\n"
         "  map: { tsdf_voxel_m: 0.05, colour: false }\n"
         "  debug: { level: warn, telemetry_rate_hz: 5 }\n";
  }

  const Config c = load_config_yaml(path);

  EXPECT_EQ(c.pipeline.mode, PipelineMode::Replay);
  EXPECT_EQ(c.time.source, TimeSource::Host);
  EXPECT_DOUBLE_EQ(c.time.max_skew_ms, 7.0);
  EXPECT_EQ(c.sensors.lidar.topic, "/lidar/points");
  EXPECT_FALSE(c.sensors.lidar.ptp);
  EXPECT_DOUBLE_EQ(c.preprocess.lidar.voxel_surf_m, 0.4);
  EXPECT_DOUBLE_EQ(c.frontend.spline.knot_dt_ms, 20.0);
  EXPECT_EQ(c.frontend.spline.window_knots, 6);
  EXPECT_DOUBLE_EQ(c.frontend.keyframe.dist_m, 2.0);
  EXPECT_DOUBLE_EQ(c.map.tsdf_voxel_m, 0.05);
  EXPECT_FALSE(c.map.colour);
  EXPECT_EQ(c.debug.level, LogLevel::Warn);
}

TEST(ConfigLoader, RejectsUnknownEnum) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_bad_enum.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n  map: { backend: octomap }\n";
  }
  EXPECT_THROW(load_config_yaml(path), std::runtime_error);
}

// ---- new key loads (C1-C6) ----

TEST(ConfigLoader, RoundTripsNewKeys) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_new_keys.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  pipeline: { queue: { meas_capacity: 16, kf_capacity: 32, map_capacity: 48 } }\n"
         "  time:\n"
         "    validator: { gap_periods: 3.0, skew_warn_ppm: 150, nan_ratio_warn: 0.1 }\n"
         "  sensors:\n"
         "    camera: { shutter: global, trigger: pps, exposure_from_meta: false }\n"
         "  preprocess:\n"
         "    voxel_surf_m: 0.6\n"
         "    sweep_floor_frac: 0.4\n"
         "    surf_max_pts: 3\n"
         "    surf_seed: 42\n"
         "  frontend:\n"
         "    degeneracy_thresh: 12.5\n"
         "    max_outer_iters: 5\n"
         "    reassoc_steps: 3\n"
         "    assoc_shift_thresh_m: 0.03\n"
         "    assoc_shift_thresh_deg: 0.5\n"
         "    solver: { max_iterations: 6, epsi: 2.0e-3, time_limit_ms: 80, min_iterations: 3 }\n"
         "    solve: { max_step_trans_m: 0.4, max_step_rot_deg: 8.0 }\n"
         "    bias: { gyr_max: 0.6, acc_max: 4.0 }\n"
         "    motion_reg: { enable: false, weight: 5.0e-3, excitation_floor: 0.2 }\n"
         "    spline:\n"
         "      knot_omega_thresh: [0.5, 1.0, 2.0]\n"
         "      knot_accel_thresh: [1.0, 3.0]\n"
         "      knot_density_hysteresis: 0.2\n"
         "    lidar: { max_lidar_factors: 2000, min_factors_per_normal: 40, normal_strata: 6 }\n";
  }

  const Config c = load_config_yaml(path);

  EXPECT_EQ(c.pipeline.queue.meas_capacity, 16);
  EXPECT_EQ(c.pipeline.queue.kf_capacity, 32);
  EXPECT_EQ(c.pipeline.queue.map_capacity, 48);

  EXPECT_DOUBLE_EQ(c.time.validator.gap_periods, 3.0);
  EXPECT_DOUBLE_EQ(c.time.validator.skew_warn_ppm, 150.0);
  EXPECT_DOUBLE_EQ(c.time.validator.nan_ratio_warn, 0.1);

  EXPECT_EQ(c.sensors.camera.shutter, "global");
  EXPECT_EQ(c.sensors.camera.trigger, "pps");
  EXPECT_FALSE(c.sensors.camera.exposure_from_meta);

  EXPECT_DOUBLE_EQ(c.preprocess.lidar.voxel_surf_m, 0.6);
  EXPECT_DOUBLE_EQ(c.preprocess.lidar.sweep_floor_frac, 0.4);
  EXPECT_EQ(c.preprocess.lidar.surf_max_pts, 3);
  EXPECT_EQ(c.preprocess.lidar.surf_seed, 42u);

  EXPECT_DOUBLE_EQ(c.frontend.degeneracy_thresh, 12.5);
  EXPECT_EQ(c.frontend.max_outer_iters, 5);
  EXPECT_EQ(c.frontend.reassoc_steps, 3);
  EXPECT_DOUBLE_EQ(c.frontend.assoc_shift_thresh_m, 0.03);
  EXPECT_DOUBLE_EQ(c.frontend.assoc_shift_thresh_deg, 0.5);
  EXPECT_EQ(c.frontend.solver_max_iterations, 6);
  EXPECT_DOUBLE_EQ(c.frontend.solver_epsi, 2.0e-3);
  EXPECT_DOUBLE_EQ(c.frontend.solver.time_limit_ms, 80.0);
  EXPECT_EQ(c.frontend.solver.min_iterations, 3);
  EXPECT_DOUBLE_EQ(c.frontend.solve.max_step_trans_m, 0.4);
  EXPECT_DOUBLE_EQ(c.frontend.solve.max_step_rot_deg, 8.0);
  EXPECT_DOUBLE_EQ(c.frontend.bias.gyr_max, 0.6);
  EXPECT_DOUBLE_EQ(c.frontend.bias.acc_max, 4.0);
  EXPECT_FALSE(c.frontend.motion_reg.enable);
  EXPECT_DOUBLE_EQ(c.frontend.motion_reg.weight, 5.0e-3);
  EXPECT_DOUBLE_EQ(c.frontend.motion_reg.excitation_floor, 0.2);
  ASSERT_EQ(c.frontend.spline.knot_omega_thresh.size(), 3u);
  EXPECT_DOUBLE_EQ(c.frontend.spline.knot_omega_thresh[2], 2.0);
  ASSERT_EQ(c.frontend.spline.knot_accel_thresh.size(), 2u);
  EXPECT_DOUBLE_EQ(c.frontend.spline.knot_density_hysteresis, 0.2);
  EXPECT_EQ(c.frontend.lidar.max_lidar_factors, 2000);
  EXPECT_EQ(c.frontend.lidar.min_factors_per_normal, 40);
  EXPECT_EQ(c.frontend.lidar.normal_strata, 6);
}

TEST(Config, DefaultDegeneracyThreshIsNonZero) {
  Config c;
  EXPECT_GT(c.frontend.degeneracy_thresh, 0.0);
}

// ---- new validate rules ----

TEST(Config, RollingShutterRejected) {
  Config c;
  c.sensors.camera.shutter = "rolling";
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("shutter"), std::string::npos) << err;
}

TEST(Config, ZeroQueueCapacityFails) {
  Config c;
  c.pipeline.queue.kf_capacity = 0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("capacity"), std::string::npos) << err;
}

// ---- camera distortion keys ----

TEST(ConfigLoader, LoadsCameraDistortionKeys) {
  const std::string path =
      std::string(::testing::TempDir()) + "/meridian_cam_distortion.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  sensors:\n"
         "    camera:\n"
         "      distortion_model: radtan\n"
         "      distortion_coeffs: [-0.28, 0.07, 0.0006, -0.0004, 0.001]\n"
         "      width: 640\n"
         "      height: 480\n";
  }
  const Config c = load_config_yaml(path);
  EXPECT_EQ(c.sensors.camera.distortion_model, "radtan");
  ASSERT_EQ(c.sensors.camera.distortion_coeffs.size(), 5u);
  EXPECT_DOUBLE_EQ(c.sensors.camera.distortion_coeffs[0], -0.28);
  EXPECT_DOUBLE_EQ(c.sensors.camera.distortion_coeffs[4], 0.001);
  EXPECT_EQ(c.sensors.camera.width, 640);
  EXPECT_EQ(c.sensors.camera.height, 480);
}

// Fewer than five coefficients (e.g. radtan without k3) zero-fill the tail.
TEST(ConfigLoader, ShortDistortionCoeffsZeroFill) {
  const std::string path =
      std::string(::testing::TempDir()) + "/meridian_cam_distortion_short.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  sensors:\n"
         "    camera:\n"
         "      distortion_coeffs: [-0.28, 0.07, 0.001, -0.002]\n";
  }
  const Config c = load_config_yaml(path);
  EXPECT_DOUBLE_EQ(c.sensors.camera.distortion_coeffs[3], -0.002);
  EXPECT_DOUBLE_EQ(c.sensors.camera.distortion_coeffs[4], 0.0);
}

TEST(ConfigLoader, TooManyDistortionCoeffsThrows) {
  const std::string path =
      std::string(::testing::TempDir()) + "/meridian_cam_distortion_long.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  sensors:\n"
         "    camera:\n"
         "      distortion_coeffs: [1, 2, 3, 4, 5, 6]\n";
  }
  EXPECT_THROW(load_config_yaml(path), std::runtime_error);
}

TEST(Config, UnknownDistortionModelRejected) {
  Config c;
  c.sensors.camera.distortion_model = "fisheye_wat";
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("distortion_model"), std::string::npos) << err;
}

TEST(Config, KnownDistortionModelsAccepted) {
  for (const char* m : {"none", "radtan", "plumb_bob", "equidistant", "equi"}) {
    Config c;
    c.sensors.camera.distortion_model = m;
    std::string err;
    EXPECT_TRUE(c.validate(&err)) << m << ": " << err;
  }
}

TEST(Config, SweepFloorFracOutOfRangeFails) {
  Config c;
  c.preprocess.lidar.sweep_floor_frac = 1.5;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("sweep_floor_frac"), std::string::npos) << err;
}

TEST(Config, SurfMaxPtsZeroFails) {
  Config c;
  c.preprocess.lidar.surf_max_pts = 0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("surf_max_pts"), std::string::npos) << err;
}

TEST(Config, MinIterationsAboveMaxFails) {
  Config c;
  c.frontend.solver.min_iterations = 10;
  c.frontend.solver_max_iterations = 5;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("min_iterations"), std::string::npos) << err;
}

TEST(Config, ZeroTimeLimitFails) {
  Config c;
  c.frontend.solver.time_limit_ms = 0.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("time_limit_ms"), std::string::npos) << err;
}

TEST(Config, NonPositiveStepClampFails) {
  Config c;
  c.frontend.solve.max_step_trans_m = 0.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("max_step"), std::string::npos) << err;
}

TEST(Config, NonPositiveBiasBoundFails) {
  Config c;
  c.frontend.bias.acc_max = -1.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("bias"), std::string::npos) << err;
}

TEST(Config, StratumFloorExceedingCapFails) {
  Config c;
  c.frontend.lidar.max_lidar_factors = 100;
  c.frontend.lidar.min_factors_per_normal = 50;
  c.frontend.lidar.normal_strata = 7;  // 350 > 100
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("max_lidar_factors"), std::string::npos) << err;
}

TEST(Config, KnotThreshNonMonotoneFails) {
  Config c;
  c.frontend.spline.knot_omega_thresh = {1.0, 0.5};  // decreasing
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("knot_omega_thresh"), std::string::npos) << err;
}

TEST(Config, KnotHysteresisOutOfRangeFails) {
  Config c;
  c.frontend.spline.knot_density_hysteresis = 1.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("hysteresis"), std::string::npos) << err;
}

}  // namespace
}  // namespace meridian
