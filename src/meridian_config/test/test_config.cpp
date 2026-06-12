#include <gtest/gtest.h>

#include <fstream>
#include <string>

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
         "    surf_seed: 42\n";
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
}

// ---- frontend LIO ----

TEST(ConfigLoader, RoundTripsLioKeys) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_lio_keys.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  frontend:\n"
         "    kind: lio\n"
         "    lio:\n"
         "      voxel_size_m: 0.8\n"
         "      max_points_per_voxel: 15\n"
         "      max_range_m: 60.0\n"
         "      keypoint_voxel_factor: 2.0\n"
         "      min_keypoints: 24\n"
         "      icp_max_iterations: 50\n"
         "      convergence_eps: 1.0e-4\n"
         "      max_corr_dist_m: 0.7\n"
         "      min_beta: 150.0\n"
         "      max_expected_jerk: 4.0\n"
         "      init_stationary_s: 0.5\n"
         "      max_gap_s: 0.8\n"
         "      reseed_cov_inflation: 50.0\n"
         "    keyframe: { dist_m: 1.5, rot_deg: 15.0, time_s: 2.0 }\n";
  }

  const Config c = load_config_yaml(path);

  EXPECT_EQ(c.frontend.kind, FrontEndKind::Lio);
  EXPECT_DOUBLE_EQ(c.frontend.lio.voxel_size_m, 0.8);
  EXPECT_EQ(c.frontend.lio.max_points_per_voxel, 15);
  EXPECT_DOUBLE_EQ(c.frontend.lio.max_range_m, 60.0);
  EXPECT_DOUBLE_EQ(c.frontend.lio.keypoint_voxel_factor, 2.0);
  EXPECT_EQ(c.frontend.lio.min_keypoints, 24);
  EXPECT_EQ(c.frontend.lio.icp_max_iterations, 50);
  EXPECT_DOUBLE_EQ(c.frontend.lio.convergence_eps, 1.0e-4);
  EXPECT_DOUBLE_EQ(c.frontend.lio.max_corr_dist_m, 0.7);
  EXPECT_DOUBLE_EQ(c.frontend.lio.min_beta, 150.0);
  EXPECT_DOUBLE_EQ(c.frontend.lio.max_expected_jerk, 4.0);
  EXPECT_DOUBLE_EQ(c.frontend.lio.init_stationary_s, 0.5);
  EXPECT_DOUBLE_EQ(c.frontend.lio.max_gap_s, 0.8);
  EXPECT_DOUBLE_EQ(c.frontend.lio.reseed_cov_inflation, 50.0);
  EXPECT_DOUBLE_EQ(c.frontend.keyframe.dist_m, 1.5);
  EXPECT_DOUBLE_EQ(c.frontend.keyframe.rot_deg, 15.0);
  EXPECT_DOUBLE_EQ(c.frontend.keyframe.time_s, 2.0);
}

TEST(ConfigLoader, RejectsUnknownFrontendKind) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_bad_frontend_kind.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n  frontend: { kind: ct_livo }\n";
  }
  EXPECT_THROW(load_config_yaml(path), std::runtime_error);
}

TEST(Config, LioValidationFails) {
  Config c;
  c.frontend.lio.voxel_size_m = 0.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("voxel_size_m"), std::string::npos) << err;

  c = Config{};
  c.frontend.lio.icp_max_iterations = 0;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("icp_max_iterations"), std::string::npos) << err;

  c = Config{};
  c.frontend.lio.reseed_cov_inflation = 0.5;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("reseed_cov_inflation"), std::string::npos) << err;
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
  const std::string path = std::string(::testing::TempDir()) + "/meridian_cam_distortion.yaml";
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
  const std::string path = std::string(::testing::TempDir()) + "/meridian_cam_distortion_long.yaml";
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

TEST(Config, RectifyBalanceOutOfRangeFails) {
  Config c;
  c.preprocess.camera.rectify_balance = 1.5;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("rectify_balance"), std::string::npos) << err;
}

TEST(ConfigLoader, LoadsRectifyBalance) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_rectify_balance.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  preprocess:\n"
         "    camera: { rectify_balance: 0.75, pyramid_levels: 2 }\n";
  }
  const Config c = load_config_yaml(path);
  EXPECT_DOUBLE_EQ(c.preprocess.camera.rectify_balance, 0.75);
  EXPECT_EQ(c.preprocess.camera.pyramid_levels, 2);
}

TEST(Config, SurfMaxPtsZeroFails) {
  Config c;
  c.preprocess.lidar.surf_max_pts = 0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("surf_max_pts"), std::string::npos) << err;
}

}  // namespace
}  // namespace meridian

// ---- loader round-trips: debug groups, path aggregation ----

namespace meridian {
namespace {

TEST(ConfigLoader, RoundTripsDebugGroupKeys) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_debug_groups.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  debug:\n"
         "    assoc: { enable: true, max_hz: 5.0 }\n"
         "    solver: { enable: true }\n"
         "    lio: { max_hz: 2.0 }\n"
         "    map_health: { enable: false }\n"
         "    publish_path: false\n"
         "    path_sample_hz: 15.0\n"
         "    path_publish_hz: 2.0\n"
         "    path_max_poses: 1000\n";
  }
  const Config c = load_config_yaml(path);
  EXPECT_TRUE(c.debug.assoc.enable);
  EXPECT_DOUBLE_EQ(c.debug.assoc.max_hz, 5.0);
  EXPECT_TRUE(c.debug.solver.enable);
  EXPECT_DOUBLE_EQ(c.debug.solver.max_hz, 0.0);
  EXPECT_FALSE(c.debug.lio.enable);  // enable absent -> default off
  EXPECT_DOUBLE_EQ(c.debug.lio.max_hz, 2.0);
  EXPECT_FALSE(c.debug.map_health.enable);
  EXPECT_FALSE(c.debug.publish_path);
  EXPECT_DOUBLE_EQ(c.debug.path_sample_hz, 15.0);
  EXPECT_DOUBLE_EQ(c.debug.path_publish_hz, 2.0);
  EXPECT_EQ(c.debug.path_max_poses, 1000);
}

TEST(Config, DebugGroupDefaultsAreOffExceptMapHealth) {
  const Config c;
  EXPECT_FALSE(c.debug.assoc.enable);
  EXPECT_FALSE(c.debug.solver.enable);
  EXPECT_FALSE(c.debug.lio.enable);
  EXPECT_TRUE(c.debug.map_health.enable);  // cheap counters keep the legacy default-on
  EXPECT_TRUE(c.debug.publish_path);
  std::string err;
  EXPECT_TRUE(c.validate(&err)) << err;
}

TEST(Config, DebugPathValidation) {
  Config c;
  c.debug.path_sample_hz = 0.0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("path_sample_hz"), std::string::npos) << err;

  c = Config{};
  c.debug.assoc.max_hz = -1.0;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("assoc.max_hz"), std::string::npos) << err;

  c = Config{};
  c.debug.path_max_poses = 0;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("path_max_poses"), std::string::npos) << err;
}

// ---- backend ----

TEST(Config, BackendDefaults) {
  Config c;
  EXPECT_TRUE(c.backend.enable);
  EXPECT_EQ(c.backend.kind, BackEndKind::Isam2);
  EXPECT_DOUBLE_EQ(c.backend.anchor_sigma, 0.1);
  EXPECT_EQ(c.backend.isam2_relinearize_skip, 1);
  EXPECT_DOUBLE_EQ(c.backend.isam2_relinearize_thresh, 0.1);
  EXPECT_EQ(c.backend.extra_iters_normal, 0);
  EXPECT_EQ(c.backend.extra_iters_loop, 4);
  EXPECT_FALSE(c.backend.isam2_use_qr);
  EXPECT_DOUBLE_EQ(c.backend.optimize_interval_ms, 100.0);
  EXPECT_EQ(c.backend.queue_warn_depth, 32);
  EXPECT_DOUBLE_EQ(c.backend.obs_inflation_max, 1e4);
  EXPECT_DOUBLE_EQ(c.backend.obs_inflation_gamma, 2.0);
  EXPECT_DOUBLE_EQ(c.backend.degenerate_thresh, 0.05);
  EXPECT_TRUE(c.backend.degenerate_lock);
  EXPECT_DOUBLE_EQ(c.backend.loop_min_fitness, 0.5);
  EXPECT_DOUBLE_EQ(c.backend.pcm_chi2_alpha, 0.99);
  EXPECT_EQ(c.backend.pcm_max_nodes, 64);
  EXPECT_EQ(c.backend.robust_kernel, RobustKernel::Huber);
  EXPECT_DOUBLE_EQ(c.backend.loop_huber_k, 1.345);
  EXPECT_DOUBLE_EQ(c.backend.gnss_huber_k, 1.345);
  EXPECT_DOUBLE_EQ(c.backend.gnc_reject_w, 0.1);
  EXPECT_FALSE(c.backend.gnc_enabled);
  EXPECT_EQ(c.backend.gnc_anneal_steps, 5);
  EXPECT_EQ(c.backend.gnc_consolidate_interval, 10);
  EXPECT_TRUE(c.backend.gnss_enabled);
  EXPECT_DOUBLE_EQ(c.backend.gnss_max_cov, 25.0);
  EXPECT_FALSE(c.backend.gnss_lock_yaw);
  EXPECT_DOUBLE_EQ(c.backend.gnss_min_baseline, 5.0);
  EXPECT_DOUBLE_EQ(c.backend.gnss_min_excitation, 3.0);
  EXPECT_DOUBLE_EQ(c.backend.gnss_min_speed, 0.5);
  EXPECT_EQ(c.backend.gnss_min_moving_fixes, 5);
  EXPECT_DOUBLE_EQ(c.backend.gnss_datum_yaw_sigma_max, 5.0);
  EXPECT_TRUE(c.backend.gnss_skip_if_confident);
  EXPECT_DOUBLE_EQ(c.backend.gnss_skip_confidence_k, 1.0);
  EXPECT_DOUBLE_EQ(c.backend.gnss_min_spacing, 1.0);
  EXPECT_FALSE(c.backend.gnss_redistribute);
  EXPECT_EQ(c.backend.gnss_reacq_fix, 1);
  EXPECT_EQ(c.backend.gnss_reacq_persist, 5);
  EXPECT_EQ(c.backend.gnss_redistribute_span_max, 200);
  EXPECT_FALSE(c.backend.extrinsic_refine);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_prior_sigma, 1e-3);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_refine_sigma, 1e-2);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_excite_rot, 0.5);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_excite_trans, 2.0);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_freeze_cov, 1e-6);
  EXPECT_DOUBLE_EQ(c.backend.extrinsic_max_dev, 0.1);
  EXPECT_FALSE(c.backend.keep_inertial);
  EXPECT_DOUBLE_EQ(c.backend.reintegrate_thresh, 0.1);
  EXPECT_FALSE(c.backend.emit_moved_cov);
  EXPECT_DOUBLE_EQ(c.backend.loop_gate_k, 3.0);
  EXPECT_DOUBLE_EQ(c.backend.imu.acc_noise, 0.1);
  EXPECT_DOUBLE_EQ(c.backend.imu.gyr_noise, 0.1);
  EXPECT_DOUBLE_EQ(c.backend.imu.acc_bias_rw, 1e-4);
  EXPECT_DOUBLE_EQ(c.backend.imu.gyr_bias_rw, 1e-4);
  EXPECT_FALSE(c.backend.debug_dump_residuals);
  EXPECT_FALSE(c.backend.snapshot_on_request);
  EXPECT_EQ(c.backend.snapshot_dir, "/tmp/meridian");
}

TEST(ConfigLoader, RoundTripsBackendKeys) {
  const std::string path = std::string(::testing::TempDir()) + "/meridian_backend_keys.yaml";
  {
    std::ofstream f(path);
    f << "meridian:\n"
         "  backend:\n"
         "    enable: false\n"
         "    optimize_interval_ms: 50\n"
         "    extra_iters_loop: 6\n"
         "    gnss_datum_yaw_sigma_max: 2.5\n"
         "    robust: huber\n"
         "    imu: { acc_noise: 0.2, gyr_noise: 0.05, acc_bias_rw: 2.0e-4, gyr_bias_rw: 3.0e-4 }\n";
  }

  const Config c = load_config_yaml(path);

  EXPECT_FALSE(c.backend.enable);
  EXPECT_DOUBLE_EQ(c.backend.optimize_interval_ms, 50.0);
  EXPECT_EQ(c.backend.extra_iters_loop, 6);
  EXPECT_DOUBLE_EQ(c.backend.gnss_datum_yaw_sigma_max, 2.5);
  EXPECT_EQ(c.backend.robust_kernel, RobustKernel::Huber);
  EXPECT_DOUBLE_EQ(c.backend.imu.acc_noise, 0.2);
  EXPECT_DOUBLE_EQ(c.backend.imu.gyr_noise, 0.05);
  EXPECT_DOUBLE_EQ(c.backend.imu.acc_bias_rw, 2.0e-4);
  EXPECT_DOUBLE_EQ(c.backend.imu.gyr_bias_rw, 3.0e-4);
}

TEST(Config, PcmChi2AlphaOutOfRangeFails) {
  Config c;
  c.backend.pcm_chi2_alpha = 1.5;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("pcm_chi2_alpha"), std::string::npos) << err;
}

TEST(Config, ZeroQueueWarnDepthFails) {
  Config c;
  c.backend.queue_warn_depth = 0;
  std::string err;
  EXPECT_FALSE(c.validate(&err));
  EXPECT_NE(err.find("queue_warn_depth"), std::string::npos) << err;
}

}  // namespace
}  // namespace meridian
