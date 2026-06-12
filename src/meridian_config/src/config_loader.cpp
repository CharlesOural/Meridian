#include "meridian/config/config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace meridian {
namespace {

// Map a closed string set to its enum, throwing a key-qualified error otherwise.
template <typename Enum>
Enum parse_enum(const YAML::Node& n, const std::string& key,
                std::initializer_list<std::pair<const char*, Enum>> table) {
  const auto s = n.as<std::string>();
  for (const auto& [name, value] : table) {
    if (s == name) return value;
  }
  std::string allowed;
  for (const auto& [name, value] : table) {
    (void)value;
    if (!allowed.empty()) allowed += "|";
    allowed += name;
  }
  throw std::runtime_error("config: key '" + key + "' has invalid value '" + s +
                           "' (expected one of: " + allowed + ")");
}

// Assign scalar `dst` from sub-node `key` of `parent` iff present; defaults survive.
template <typename T>
void get(const YAML::Node& parent, const char* key, T& dst) {
  if (parent && parent[key]) dst = parent[key].as<T>();
}

void load_pipeline(const YAML::Node& root, PipelineConfig& c) {
  const YAML::Node n = root["pipeline"];
  if (!n) return;
  if (n["mode"]) {
    c.mode =
        parse_enum<PipelineMode>(n["mode"], "pipeline.mode",
                                 {{"live", PipelineMode::Live}, {"replay", PipelineMode::Replay}});
  }
  const YAML::Node t = n["threads"];
  get(t, "frontend", c.threads.frontend);
  get(t, "backend", c.threads.backend);
  get(t, "map", c.threads.map);
  const YAML::Node q = n["queue"];
  get(q, "meas_capacity", c.queue.meas_capacity);
  get(q, "kf_capacity", c.queue.kf_capacity);
  get(q, "map_capacity", c.queue.map_capacity);
}

void load_time(const YAML::Node& root, TimeConfig& c) {
  const YAML::Node n = root["time"];
  if (!n) return;
  if (n["source"]) {
    c.source = parse_enum<TimeSource>(
        n["source"], "time.source",
        {{"ptp", TimeSource::Ptp}, {"pps", TimeSource::Pps}, {"host", TimeSource::Host}});
  }
  get(n, "max_skew_ms", c.max_skew_ms);
  const YAML::Node ptp = n["ptp"];
  get(ptp, "domain", c.ptp.domain);
  get(ptp, "iface", c.ptp.iface);
  get(ptp, "require_lock", c.ptp.require_lock);
  const YAML::Node pps = n["pps"];
  get(pps, "device", c.pps.device);
  get(pps, "expect_hz", c.pps.expect_hz);
  const YAML::Node val = n["validator"];
  get(val, "gap_periods", c.validator.gap_periods);
  get(val, "skew_warn_ppm", c.validator.skew_warn_ppm);
  get(val, "nan_ratio_warn", c.validator.nan_ratio_warn);
}

Eigen::Vector3d read_vec3(const YAML::Node& n, const std::string& key) {
  if (!n.IsSequence() || n.size() != 3) {
    throw std::runtime_error("config: key '" + key + "' must be a 3-element sequence");
  }
  return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>());
}

// Row-major 3x3 rotation given as a 9-element sequence.
Eigen::Matrix3d read_mat3(const YAML::Node& n, const std::string& key) {
  if (!n.IsSequence() || n.size() != 9) {
    throw std::runtime_error("config: key '" + key + "' must be a 9-element sequence");
  }
  Eigen::Matrix3d m;
  for (int r = 0; r < 3; ++r)
    for (int col = 0; col < 3; ++col)
      m(r, col) = n[static_cast<std::size_t>(3 * r + col)].as<double>();
  return m;
}

void load_sensors(const YAML::Node& root, SensorsConfig& c) {
  const YAML::Node n = root["sensors"];
  if (!n) return;

  const YAML::Node l = n["lidar"];
  if (l) {
    get(l, "topic", c.lidar.topic);
    get(l, "time_offset_ms", c.lidar.time_offset_ms);
    get(l, "model", c.lidar.model);
    get(l, "nominal_rate_hz", c.lidar.nominal_rate_hz);
    get(l, "qos_reliable", c.lidar.qos_reliable);
    get(l, "ptp", c.lidar.ptp);
    get(l, "timestamp_mode", c.lidar.timestamp_mode);
    if (l["extrinsic_T"])
      c.lidar.extrinsic_T = read_vec3(l["extrinsic_T"], "sensors.lidar.extrinsic_T");
    if (l["extrinsic_R"])
      c.lidar.extrinsic_R = read_mat3(l["extrinsic_R"], "sensors.lidar.extrinsic_R");
    c.lidar.extrinsic_set =
        static_cast<bool>(l["extrinsic_T"]) || static_cast<bool>(l["extrinsic_R"]);
  }

  const YAML::Node i = n["imu"];
  if (i) {
    get(i, "topic", c.imu.topic);
    get(i, "rate_hz", c.imu.rate_hz);
    get(i, "cov_acc", c.imu.cov_acc);
    get(i, "cov_gyr", c.imu.cov_gyr);
    get(i, "b_acc_cov", c.imu.b_acc_cov);
    get(i, "b_gyr_cov", c.imu.b_gyr_cov);
    get(i, "has_device_clock", c.imu.has_device_clock);
  }

  const YAML::Node cam = n["camera"];
  if (cam) {
    get(cam, "topic", c.camera.topic);
    get(cam, "time_offset_ms", c.camera.time_offset_ms);
    get(cam, "compressed", c.camera.compressed);
    get(cam, "model", c.camera.model);
    get(cam, "nominal_rate_hz", c.camera.nominal_rate_hz);
    if (cam["intrinsics"]) {
      const YAML::Node iv = cam["intrinsics"];
      if (!iv.IsSequence() || iv.size() != 4) {
        throw std::runtime_error("config: key 'sensors.camera.intrinsics' must be [fx,fy,cx,cy]");
      }
      c.camera.intrinsics = Eigen::Vector4d(iv[0].as<double>(), iv[1].as<double>(),
                                            iv[2].as<double>(), iv[3].as<double>());
    }
    get(cam, "distortion_model", c.camera.distortion_model);
    if (cam["distortion_coeffs"]) {
      const YAML::Node dc = cam["distortion_coeffs"];
      if (!dc.IsSequence() || dc.size() > 5) {
        throw std::runtime_error(
            "config: key 'sensors.camera.distortion_coeffs' must be a sequence of "
            "at most 5 values (radtan k1,k2,p1,p2,k3 | equidistant k1..k4)");
      }
      c.camera.distortion_coeffs = {0, 0, 0, 0, 0};
      for (std::size_t i = 0; i < dc.size(); ++i) {
        c.camera.distortion_coeffs[i] = dc[i].as<double>();
      }
    }
    get(cam, "width", c.camera.width);
    get(cam, "height", c.camera.height);
    if (cam["extrinsic"]) {
      const YAML::Node ex = cam["extrinsic"];
      if (!ex.IsSequence() || ex.size() != 7) {
        throw std::runtime_error(
            "config: key 'sensors.camera.extrinsic' must be [tx,ty,tz,qx,qy,qz,qw]");
      }
      const Eigen::Vector3d t(ex[0].as<double>(), ex[1].as<double>(), ex[2].as<double>());
      const Eigen::Quaterniond q(ex[6].as<double>(), ex[3].as<double>(), ex[4].as<double>(),
                                 ex[5].as<double>());
      c.camera.extrinsic = Pose(q, t);
      c.camera.extrinsic_set = true;
    }
    get(cam, "trigger", c.camera.trigger);
    get(cam, "exposure_from_meta", c.camera.exposure_from_meta);
    get(cam, "shutter", c.camera.shutter);
    const YAML::Node photo = cam["photometric"];
    get(photo, "exposure_comp", c.camera.photometric.exposure_comp);
  }

  const YAML::Node g = n["gnss"];
  if (g) {
    get(g, "topic", c.gnss.topic);
    get(g, "enable", c.gnss.enable);
    if (g["extrinsic_T"])
      c.gnss.extrinsic_T = read_vec3(g["extrinsic_T"], "sensors.gnss.extrinsic_T");
    if (g["extrinsic_R"])
      c.gnss.extrinsic_R = read_mat3(g["extrinsic_R"], "sensors.gnss.extrinsic_R");
    c.gnss.extrinsic_set =
        static_cast<bool>(g["extrinsic_T"]) || static_cast<bool>(g["extrinsic_R"]);
  }
}

void load_preprocess(const YAML::Node& root, PreprocessConfig& c) {
  const YAML::Node n = root["preprocess"];
  if (!n) return;
  get(n, "voxel_surf_m", c.lidar.voxel_surf_m);
  get(n, "sweep_floor_frac", c.lidar.sweep_floor_frac);
  get(n, "surf_max_pts", c.lidar.surf_max_pts);
  get(n, "surf_seed", c.lidar.surf_seed);
  get(n, "blind", c.lidar.blind);
  get(n, "point_filter_num", c.lidar.point_filter_num);
  get(n, "det_range", c.lidar.det_range);

  const YAML::Node cam = n["camera"];
  if (cam) {
    get(cam, "photometric_calib", c.camera.photometric_calib);
    get(cam, "vignette_map", c.camera.vignette_map);
    get(cam, "crf_lut", c.camera.crf_lut);
    get(cam, "ref_exposure_s", c.camera.ref_exposure_s);
    get(cam, "ref_gain", c.camera.ref_gain);
    get(cam, "pyramid_levels", c.camera.pyramid_levels);
    get(cam, "rectify_balance", c.camera.rectify_balance);
  }

  const YAML::Node g = n["gnss"];
  if (g) {
    get(g, "enable", c.gnss.enable);
    if (g["min_fix_type"]) {
      c.gnss.min_fix_type =
          parse_enum<GnssFix::FixType>(g["min_fix_type"], "preprocess.gnss.min_fix_type",
                                       {{"none", GnssFix::FixType::None},
                                        {"spp", GnssFix::FixType::SPP},
                                        {"dgps", GnssFix::FixType::DGPS},
                                        {"rtk_float", GnssFix::FixType::RTK_Float},
                                        {"rtk_fixed", GnssFix::FixType::RTK_Fixed}});
    }
    get(g, "min_sats", c.gnss.min_sats);
    get(g, "max_pos_var", c.gnss.max_pos_var);
    get(g, "max_dop", c.gnss.max_dop);
    get(g, "spoof_check", c.gnss.spoof_check);
    get(g, "spoof_vel_thresh", c.gnss.spoof_vel_thresh);
    get(g, "spoof_persist", c.gnss.spoof_persist);
    get(g, "spoof_window_ms", c.gnss.spoof_window_ms);
  }
}

void load_frontend(const YAML::Node& root, FrontendConfig& c) {
  const YAML::Node n = root["frontend"];
  if (!n) return;
  if (n["kind"]) {
    c.kind = parse_enum<FrontEndKind>(n["kind"], "frontend.kind", {{"lio", FrontEndKind::Lio}});
  }
  const YAML::Node lio = n["lio"];
  get(lio, "voxel_size_m", c.lio.voxel_size_m);
  get(lio, "max_points_per_voxel", c.lio.max_points_per_voxel);
  get(lio, "max_range_m", c.lio.max_range_m);
  get(lio, "keypoint_voxel_factor", c.lio.keypoint_voxel_factor);
  get(lio, "min_keypoints", c.lio.min_keypoints);
  get(lio, "icp_max_iterations", c.lio.icp_max_iterations);
  get(lio, "convergence_eps", c.lio.convergence_eps);
  get(lio, "max_corr_dist_m", c.lio.max_corr_dist_m);
  get(lio, "min_beta", c.lio.min_beta);
  get(lio, "max_expected_jerk", c.lio.max_expected_jerk);
  get(lio, "init_stationary_s", c.lio.init_stationary_s);
  get(lio, "max_gap_s", c.lio.max_gap_s);
  get(lio, "reseed_cov_inflation", c.lio.reseed_cov_inflation);
  get(lio, "estimate_gyro_bias", c.lio.estimate_gyro_bias);
  get(lio, "gyro_bias_gain", c.lio.gyro_bias_gain);
  get(lio, "gyro_bias_max", c.lio.gyro_bias_max);
  const YAML::Node kf = n["keyframe"];
  get(kf, "dist_m", c.keyframe.dist_m);
  get(kf, "rot_deg", c.keyframe.rot_deg);
  get(kf, "time_s", c.keyframe.time_s);
}

void load_backend(const YAML::Node& root, BackendConfig& c) {
  const YAML::Node n = root["backend"];
  if (!n) return;
  get(n, "enable", c.enable);
  if (n["kind"]) {
    c.kind = parse_enum<BackEndKind>(n["kind"], "backend.kind", {{"isam2", BackEndKind::Isam2}});
  }
  get(n, "anchor_sigma", c.anchor_sigma);
  get(n, "isam2_relinearize_skip", c.isam2_relinearize_skip);
  // Both the short legacy key and the full name are accepted; the full name wins.
  get(n, "relinearize_thresh", c.isam2_relinearize_thresh);
  get(n, "isam2_relinearize_thresh", c.isam2_relinearize_thresh);
  get(n, "extra_iters_normal", c.extra_iters_normal);
  get(n, "extra_iters_loop", c.extra_iters_loop);
  get(n, "isam2_use_qr", c.isam2_use_qr);
  get(n, "optimize_interval_ms", c.optimize_interval_ms);
  get(n, "queue_warn_depth", c.queue_warn_depth);
  get(n, "obs_inflation_max", c.obs_inflation_max);
  get(n, "obs_inflation_gamma", c.obs_inflation_gamma);
  get(n, "degenerate_thresh", c.degenerate_thresh);
  get(n, "degenerate_lock", c.degenerate_lock);
  get(n, "correct_frontend", c.correct_frontend);
  get(n, "loop_min_fitness", c.loop_min_fitness);
  get(n, "pcm_chi2_alpha", c.pcm_chi2_alpha);
  get(n, "pcm_max_nodes", c.pcm_max_nodes);
  if (n["robust"]) {
    c.robust_kernel = parse_enum<RobustKernel>(n["robust"], "backend.robust",
                                               {{"huber", RobustKernel::Huber},
                                                {"gnc_cauchy", RobustKernel::Cauchy},
                                                {"cauchy", RobustKernel::Cauchy},
                                                {"gm", RobustKernel::Gm},
                                                {"tls", RobustKernel::Tls}});
  }
  get(n, "loop_huber_k", c.loop_huber_k);
  get(n, "gnss_huber_k", c.gnss_huber_k);
  get(n, "gnc_reject_w", c.gnc_reject_w);
  get(n, "gnc_enabled", c.gnc_enabled);
  get(n, "gnc_anneal_steps", c.gnc_anneal_steps);
  get(n, "gnc_consolidate_interval", c.gnc_consolidate_interval);
  get(n, "gnss_enabled", c.gnss_enabled);
  get(n, "gnss_max_cov", c.gnss_max_cov);
  get(n, "gnss_lock_yaw", c.gnss_lock_yaw);
  get(n, "gnss_min_baseline", c.gnss_min_baseline);
  get(n, "gnss_min_excitation", c.gnss_min_excitation);
  get(n, "gnss_min_speed", c.gnss_min_speed);
  get(n, "gnss_min_moving_fixes", c.gnss_min_moving_fixes);
  get(n, "gnss_datum_yaw_sigma_max", c.gnss_datum_yaw_sigma_max);
  get(n, "gnss_skip_if_confident", c.gnss_skip_if_confident);
  get(n, "gnss_skip_confidence_k", c.gnss_skip_confidence_k);
  get(n, "gnss_min_spacing", c.gnss_min_spacing);
  get(n, "gnss_redistribute", c.gnss_redistribute);
  get(n, "gnss_reacq_fix", c.gnss_reacq_fix);
  get(n, "gnss_reacq_persist", c.gnss_reacq_persist);
  get(n, "gnss_redistribute_span_max", c.gnss_redistribute_span_max);
  get(n, "extrinsic_refine", c.extrinsic_refine);
  get(n, "extrinsic_prior_sigma", c.extrinsic_prior_sigma);
  get(n, "extrinsic_refine_sigma", c.extrinsic_refine_sigma);
  get(n, "extrinsic_excite_rot", c.extrinsic_excite_rot);
  get(n, "extrinsic_excite_trans", c.extrinsic_excite_trans);
  get(n, "extrinsic_freeze_cov", c.extrinsic_freeze_cov);
  get(n, "extrinsic_max_dev", c.extrinsic_max_dev);
  get(n, "keep_inertial", c.keep_inertial);
  get(n, "reintegrate_thresh", c.reintegrate_thresh);
  get(n, "emit_moved_cov", c.emit_moved_cov);
  get(n, "loop_gate_k", c.loop_gate_k);
  const YAML::Node imu = n["imu"];
  get(imu, "acc_noise", c.imu.acc_noise);
  get(imu, "gyr_noise", c.imu.gyr_noise);
  get(imu, "acc_bias_rw", c.imu.acc_bias_rw);
  get(imu, "gyr_bias_rw", c.imu.gyr_bias_rw);
  get(n, "debug_dump_residuals", c.debug_dump_residuals);
  get(n, "snapshot_on_request", c.snapshot_on_request);
  get(n, "snapshot_dir", c.snapshot_dir);
}

void load_map(const YAML::Node& root, MapConfig& c) {
  const YAML::Node n = root["map"];
  if (!n) return;
  if (n["backend"]) {
    c.backend =
        parse_enum<MapBackend>(n["backend"], "map.backend", {{"nvblox", MapBackend::Nvblox}});
  }
  get(n, "tsdf_voxel_m", c.tsdf_voxel_m);
  get(n, "reg_voxel_m", c.reg_voxel_m);
  if (n["mesh"]) {
    c.mesh =
        parse_enum<MeshKind>(n["mesh"], "map.mesh", {{"marching_cubes", MeshKind::MarchingCubes}});
  }
  get(n, "colour", c.colour);
}

void load_place(const YAML::Node& root, PlaceConfig& c) {
  const YAML::Node n = root["place"];
  if (!n) return;
  if (n["kind"]) {
    c.kind = parse_enum<PlaceKind>(n["kind"], "place.kind",
                                   {{"scan_context_pp", PlaceKind::ScanContextPp}});
  }
  get(n, "enable", c.enable);
  get(n, "pcm", c.pcm);
  // submap accumulator
  get(n, "submap_window", c.submap_window);
  get(n, "submap_voxel", c.submap_voxel);
  get(n, "submap_cache", c.submap_cache);
  get(n, "gicp_source_submap", c.gicp_source_submap);
  get(n, "cov_psd_floor", c.cov_psd_floor);
  // cadence / gating
  get(n, "detect_period_kf", c.detect_period_kf);
  get(n, "min_time_gap", c.min_time_gap);
  get(n, "min_kf_gap", c.min_kf_gap);
  get(n, "cooldown_kf", c.cooldown_kf);
  // Scan Context++
  get(n, "sc_Nr", c.sc_Nr);
  get(n, "sc_Ns", c.sc_Ns);
  get(n, "sc_rmax", c.sc_rmax);
  get(n, "sc_knn", c.sc_knn);
  get(n, "sc_dist_thresh", c.sc_dist_thresh);
  get(n, "sc_max_xy", c.sc_max_xy);
  get(n, "sc_yaw_search_band", c.sc_yaw_search_band);
  get(n, "sc_topK", c.sc_topK);
  // GICP / small_gicp
  get(n, "gicp_downsample", c.gicp_downsample);
  get(n, "gicp_max_corr_dist", c.gicp_max_corr_dist);
  get(n, "gicp_num_threads", c.gicp_num_threads);
  get(n, "gicp_voxel_res", c.gicp_voxel_res);
  get(n, "gicp_fitness_min", c.gicp_fitness_min);
  get(n, "gicp_overlap_min", c.gicp_overlap_min);
  get(n, "gicp_rmse_max", c.gicp_rmse_max);
  get(n, "gicp_cond_max", c.gicp_cond_max);
  get(n, "gicp_fit_sigma", c.gicp_fit_sigma);
  // The top-level YAML key gicp_fitness_max is the loose RMSE accept band.
  get(n, "gicp_fitness_max", c.gicp_rmse_max);
  // PCM
  get(n, "pcm_chi2_conf", c.pcm_chi2_conf);
  get(n, "pcm_maxclique_ms", c.pcm_maxclique_ms);
  // covariance shaping
  get(n, "cov_lambda", c.cov_lambda);
  get(n, "cov_degenerate_eig", c.cov_degenerate_eig);
  get(n, "cov_degenerate_mult", c.cov_degenerate_mult);
}

void load_debug(const YAML::Node& root, DebugConfig& c) {
  const YAML::Node n = root["debug"];
  if (!n) return;
  if (n["level"]) {
    c.level = parse_enum<LogLevel>(n["level"], "debug.level",
                                   {{"trace", LogLevel::Trace},
                                    {"debug", LogLevel::Debug},
                                    {"info", LogLevel::Info},
                                    {"warn", LogLevel::Warn},
                                    {"error", LogLevel::Error}});
  }
  get(n, "publish_clouds", c.publish_clouds);
  get(n, "publish_markers", c.publish_markers);
  get(n, "publish_odom", c.publish_odom);
  get(n, "timing", c.timing);
  get(n, "telemetry_rate_hz", c.telemetry_rate_hz);
  if (n["cloud_color"]) {
    c.cloud_color = parse_enum<CloudColor>(n["cloud_color"], "debug.cloud_color",
                                           {{"intensity", CloudColor::Intensity},
                                            {"height", CloudColor::Height},
                                            {"z", CloudColor::Height}});
  }
  get(n, "cloud_color_min", c.cloud_color_min);
  get(n, "cloud_color_max", c.cloud_color_max);
  // Debug groups: each block seeds one key-prefix wildcard in the sink's gate table.
  const auto group = [&n](const char* key, DebugGroup& g) {
    const YAML::Node b = n[key];
    get(b, "enable", g.enable);
    get(b, "max_hz", g.max_hz);
  };
  group("assoc", c.assoc);
  group("solver", c.solver);
  group("lio", c.lio);
  group("map_health", c.map_health);
  get(n, "publish_path", c.publish_path);
  get(n, "path_sample_hz", c.path_sample_hz);
  get(n, "path_publish_hz", c.path_publish_hz);
  get(n, "path_max_poses", c.path_max_poses);
}

}  // namespace

bool Config::validate(std::string* error_out) const {
  const auto fail = [&](const std::string& msg) {
    if (error_out) *error_out = msg;
    return false;
  };

  // --- map voxel must be no coarser than the surf filter or the registration voxel ---
  if (map.tsdf_voxel_m <= 0.0) {
    return fail("map.tsdf_voxel_m must be > 0");
  }
  if (preprocess.lidar.voxel_surf_m <= 0.0) {
    return fail("preprocess.voxel_surf_m must be > 0");
  }
  if (map.tsdf_voxel_m > preprocess.lidar.voxel_surf_m) {
    return fail("map.tsdf_voxel_m (" + std::to_string(map.tsdf_voxel_m) +
                ") must be <= preprocess.voxel_surf_m (" +
                std::to_string(preprocess.lidar.voxel_surf_m) + ")");
  }
  if (map.reg_voxel_m <= 0.0) {
    return fail("map.reg_voxel_m must be > 0");
  }
  if (map.tsdf_voxel_m > map.reg_voxel_m) {
    return fail("map.tsdf_voxel_m (" + std::to_string(map.tsdf_voxel_m) +
                ") must be <= map.reg_voxel_m (" + std::to_string(map.reg_voxel_m) + ")");
  }

  // --- front-end LIO ---
  if (frontend.lio.voxel_size_m <= 0.0) {
    return fail("frontend.lio.voxel_size_m must be > 0");
  }
  if (frontend.lio.max_points_per_voxel < 1) {
    return fail("frontend.lio.max_points_per_voxel must be >= 1");
  }
  if (frontend.lio.max_range_m <= 0.0) {
    return fail("frontend.lio.max_range_m must be > 0");
  }
  if (frontend.lio.keypoint_voxel_factor <= 0.0) {
    return fail("frontend.lio.keypoint_voxel_factor must be > 0");
  }
  if (frontend.lio.min_keypoints < 1) {
    return fail("frontend.lio.min_keypoints must be >= 1");
  }
  if (frontend.lio.icp_max_iterations < 1) {
    return fail("frontend.lio.icp_max_iterations must be >= 1");
  }
  if (frontend.lio.convergence_eps <= 0.0) {
    return fail("frontend.lio.convergence_eps must be > 0");
  }
  if (frontend.lio.max_corr_dist_m <= 0.0) {
    return fail("frontend.lio.max_corr_dist_m must be > 0");
  }
  if (frontend.lio.min_beta <= 0.0) {
    return fail("frontend.lio.min_beta must be > 0");
  }
  if (frontend.lio.max_expected_jerk <= 0.0) {
    return fail("frontend.lio.max_expected_jerk must be > 0");
  }
  if (frontend.lio.init_stationary_s < 0.0) {
    return fail("frontend.lio.init_stationary_s must be >= 0");
  }
  if (frontend.lio.max_gap_s <= 0.0) {
    return fail("frontend.lio.max_gap_s must be > 0");
  }
  if (frontend.lio.reseed_cov_inflation < 1.0) {
    return fail("frontend.lio.reseed_cov_inflation must be >= 1");
  }
  if (frontend.lio.gyro_bias_gain < 0.0 || frontend.lio.gyro_bias_gain > 1.0) {
    return fail("frontend.lio.gyro_bias_gain must be in [0, 1]");
  }
  if (frontend.lio.gyro_bias_max <= 0.0) {
    return fail("frontend.lio.gyro_bias_max must be > 0");
  }

  // --- back-end ---
  if (backend.anchor_sigma <= 0.0) {
    return fail("backend.anchor_sigma must be > 0");
  }
  if (backend.extra_iters_normal < 0 || backend.extra_iters_loop < 0) {
    return fail("backend.extra_iters_{normal,loop} must be >= 0");
  }
  if (backend.optimize_interval_ms <= 0.0) {
    return fail("backend.optimize_interval_ms must be > 0");
  }
  if (backend.queue_warn_depth < 1) {
    return fail("backend.queue_warn_depth must be >= 1");
  }
  if (backend.obs_inflation_max < 1.0) {
    return fail("backend.obs_inflation_max must be >= 1");
  }
  if (backend.obs_inflation_gamma <= 0.0) {
    return fail("backend.obs_inflation_gamma must be > 0");
  }
  if (backend.degenerate_thresh < 0.0 || backend.degenerate_thresh > 1.0) {
    return fail("backend.degenerate_thresh must be in [0, 1]");
  }
  if (backend.loop_min_fitness < 0.0 || backend.loop_min_fitness > 1.0) {
    return fail("backend.loop_min_fitness must be in [0, 1]");
  }
  if (backend.pcm_chi2_alpha <= 0.0 || backend.pcm_chi2_alpha >= 1.0) {
    return fail("backend.pcm_chi2_alpha must be in (0, 1)");
  }
  if (backend.pcm_max_nodes < 1) {
    return fail("backend.pcm_max_nodes must be >= 1");
  }

  // --- place (L5) ---
  if (place.sc_Nr < 1 || place.sc_Ns < 1) {
    return fail("place.sc_Nr / place.sc_Ns must be >= 1");
  }
  if (place.sc_rmax <= 0.0) {
    return fail("place.sc_rmax must be > 0");
  }
  if (place.submap_window < 1) {
    return fail("place.submap_window must be >= 1");
  }
  if (place.submap_voxel <= 0.0) {
    return fail("place.submap_voxel must be > 0");
  }
  if (place.submap_cache < 1) {
    return fail("place.submap_cache must be >= 1");
  }
  if (place.detect_period_kf < 1) {
    return fail("place.detect_period_kf must be >= 1");
  }
  if (place.gicp_num_threads < 1) {
    return fail("place.gicp_num_threads must be >= 1");
  }
  if (place.gicp_fitness_min <= 0.0 || place.gicp_fitness_min > 1.0) {
    return fail("place.gicp_fitness_min must be in (0, 1]");
  }
  if (place.gicp_overlap_min < 0.0 || place.gicp_overlap_min > 1.0) {
    return fail("place.gicp_overlap_min must be in [0, 1]");
  }
  if (place.pcm_chi2_conf <= 0.0 || place.pcm_chi2_conf >= 1.0) {
    return fail("place.pcm_chi2_conf must be in (0, 1)");
  }
  // The L5 single-loop self-test and the L3 cross-loop clique must judge consistency on
  // the same chi-square scale, or a loop the detector trusts gets re-rejected (or vice
  // versa) at the back-end boundary.
  if (place.pcm_chi2_conf != backend.pcm_chi2_alpha) {
    return fail("place.pcm_chi2_conf must equal backend.pcm_chi2_alpha");
  }
  if (backend.loop_huber_k <= 0.0 || backend.gnss_huber_k <= 0.0) {
    return fail("backend.{loop,gnss}_huber_k must be > 0");
  }
  if (backend.gnc_reject_w <= 0.0 || backend.gnc_reject_w >= 1.0) {
    return fail("backend.gnc_reject_w must be in (0, 1)");
  }
  if (backend.gnc_anneal_steps < 1) {
    return fail("backend.gnc_anneal_steps must be >= 1");
  }
  if (backend.gnc_consolidate_interval < 0) {
    return fail("backend.gnc_consolidate_interval must be >= 0");
  }
  if (backend.gnss_max_cov <= 0.0) {
    return fail("backend.gnss_max_cov must be > 0");
  }
  if (backend.gnss_min_baseline <= 0.0) {
    return fail("backend.gnss_min_baseline must be > 0");
  }
  if (backend.gnss_min_excitation <= 0.0) {
    return fail("backend.gnss_min_excitation must be > 0");
  }
  if (backend.gnss_min_speed <= 0.0) {
    return fail("backend.gnss_min_speed must be > 0");
  }
  if (backend.gnss_min_moving_fixes < 1) {
    return fail("backend.gnss_min_moving_fixes must be >= 1");
  }
  if (backend.gnss_datum_yaw_sigma_max <= 0.0) {
    return fail("backend.gnss_datum_yaw_sigma_max must be > 0");
  }
  if (backend.gnss_skip_confidence_k <= 0.0) {
    return fail("backend.gnss_skip_confidence_k must be > 0");
  }
  if (backend.gnss_min_spacing < 0.0) {
    return fail("backend.gnss_min_spacing must be >= 0");
  }
  if (backend.gnss_reacq_persist < 1) {
    return fail("backend.gnss_reacq_persist must be >= 1");
  }
  if (backend.gnss_redistribute_span_max < 1) {
    return fail("backend.gnss_redistribute_span_max must be >= 1");
  }
  if (backend.extrinsic_prior_sigma <= 0.0 || backend.extrinsic_refine_sigma <= 0.0) {
    return fail("backend.extrinsic_{prior,refine}_sigma must be > 0");
  }
  if (backend.extrinsic_excite_rot <= 0.0 || backend.extrinsic_excite_trans <= 0.0) {
    return fail("backend.extrinsic_excite_{rot,trans} must be > 0");
  }
  if (backend.extrinsic_freeze_cov <= 0.0) {
    return fail("backend.extrinsic_freeze_cov must be > 0");
  }
  if (backend.extrinsic_max_dev <= 0.0) {
    return fail("backend.extrinsic_max_dev must be > 0");
  }
  if (backend.reintegrate_thresh < 0.0) {
    return fail("backend.reintegrate_thresh must be >= 0");
  }
  if (backend.loop_gate_k <= 0.0) {
    return fail("backend.loop_gate_k must be > 0");
  }
  if (backend.imu.acc_noise <= 0.0 || backend.imu.gyr_noise <= 0.0 ||
      backend.imu.acc_bias_rw <= 0.0 || backend.imu.gyr_bias_rw <= 0.0) {
    return fail("backend.imu.{acc,gyr}_noise and {acc,gyr}_bias_rw must be > 0");
  }

  // --- LiDAR range gate ---
  if (preprocess.lidar.blind < 0.0) {
    return fail("preprocess.blind must be >= 0");
  }
  if (preprocess.lidar.det_range <= 0.0) {
    return fail("preprocess.lidar.det_range must be > 0");
  }
  if (preprocess.lidar.blind >= preprocess.lidar.det_range) {
    return fail("preprocess.blind (" + std::to_string(preprocess.lidar.blind) +
                ") must be < preprocess.lidar.det_range (" +
                std::to_string(preprocess.lidar.det_range) + ")");
  }
  if (preprocess.lidar.point_filter_num < 1) {
    return fail("preprocess.point_filter_num must be >= 1");
  }
  // The sweep-duration floor is a fraction of one nominal period, never more.
  if (preprocess.lidar.sweep_floor_frac <= 0.0 || preprocess.lidar.sweep_floor_frac > 1.0) {
    return fail("preprocess.sweep_floor_frac must be in (0, 1]");
  }
  if (preprocess.lidar.surf_max_pts < 1) {
    return fail("preprocess.surf_max_pts must be >= 1");
  }

  // --- camera pyramid ---
  if (preprocess.camera.pyramid_levels < 1) {
    return fail("preprocess.camera.pyramid_levels must be >= 1");
  }
  if (preprocess.camera.rectify_balance < 0.0 || preprocess.camera.rectify_balance > 1.0) {
    return fail("preprocess.camera.rectify_balance must be in [0,1]");
  }

  // --- sensor rates ---
  if (sensors.lidar.nominal_rate_hz <= 0.0) {
    return fail("sensors.lidar.nominal_rate_hz must be > 0");
  }
  if (sensors.imu.rate_hz <= 0.0) {
    return fail("sensors.imu.rate_hz must be > 0");
  }

  // --- aggregation must wait at least one LiDAR period ---
  const double lidar_period_ms = 1000.0 / sensors.lidar.nominal_rate_hz;
  if (aggregation.max_wait_ms < lidar_period_ms) {
    return fail("aggregation.max_wait_ms (" + std::to_string(aggregation.max_wait_ms) +
                ") must be >= one LiDAR period (" + std::to_string(lidar_period_ms) + " ms)");
  }

  // --- PTP requires at least one PTP-disciplined sensor ---
  if (time.source == TimeSource::Ptp && !sensors.lidar.ptp) {
    return fail(
        "time.source == ptp requires at least one sensor with ptp: true "
        "(sensors.lidar.ptp is false)");
  }

  // --- thread counts ---
  if (pipeline.threads.frontend < 1 || pipeline.threads.backend < 1 || pipeline.threads.map < 1) {
    return fail("pipeline.threads.{frontend,backend,map} must each be >= 1");
  }

  // --- queue capacities ---
  if (pipeline.queue.meas_capacity < 1 || pipeline.queue.kf_capacity < 1 ||
      pipeline.queue.map_capacity < 1) {
    return fail("pipeline.queue.{meas,kf,map}_capacity must each be > 0");
  }

  // --- camera shutter (the rolling-shutter path is designed but not built) ---
  if (sensors.camera.shutter != "global") {
    return fail("sensors.camera.shutter must be 'global' (got '" + sensors.camera.shutter +
                "'); the rolling-shutter path is not built");
  }

  // --- debug groups & path aggregation ---
  for (const auto& [name, g] : {std::pair<const char*, const DebugGroup*>{"assoc", &debug.assoc},
                                {"solver", &debug.solver},
                                {"lio", &debug.lio},
                                {"map_health", &debug.map_health}}) {
    if (g->max_hz < 0.0) {
      return fail(std::string("debug.") + name + ".max_hz must be >= 0");
    }
  }
  if (debug.path_sample_hz <= 0.0 || debug.path_sample_hz > 200.0) {
    return fail("debug.path_sample_hz must be in (0, 200]");
  }
  if (debug.path_publish_hz <= 0.0) {
    return fail("debug.path_publish_hz must be > 0");
  }
  if (debug.path_max_poses < 1) {
    return fail("debug.path_max_poses must be >= 1");
  }

  // --- camera distortion model (closed string set; reject unknowns loudly) ---
  {
    const std::string& m = sensors.camera.distortion_model;
    if (m != "none" && m != "radtan" && m != "plumb_bob" && m != "equidistant" && m != "equi") {
      return fail(
          "sensors.camera.distortion_model must be one of "
          "none|radtan|plumb_bob|equidistant|equi (got '" +
          m + "')");
    }
  }

  return true;
}

Config load_config_yaml(const std::string& path) {
  YAML::Node doc;
  try {
    doc = YAML::LoadFile(path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("config: failed to load '" + path + "': " + e.what());
  }

  const YAML::Node root = doc["meridian"] ? doc["meridian"] : doc;

  Config c;
  load_pipeline(root, c.pipeline);
  load_time(root, c.time);
  load_sensors(root, c.sensors);
  load_preprocess(root, c.preprocess);
  load_frontend(root, c.frontend);
  load_backend(root, c.backend);
  load_map(root, c.map);
  load_place(root, c.place);
  load_debug(root, c.debug);

  std::string err;
  if (!c.validate(&err)) {
    throw std::runtime_error("config: validation failed: " + err);
  }
  return c;
}

}  // namespace meridian
