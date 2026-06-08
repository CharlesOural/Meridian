#include "meridian/config/config_loader.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

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
    c.mode = parse_enum<PipelineMode>(n["mode"], "pipeline.mode",
                                      {{"live", PipelineMode::Live},
                                       {"replay", PipelineMode::Replay}});
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

// Read an arbitrary-length numeric sequence; absent/empty leaves dst untouched.
void get_vec(const YAML::Node& parent, const char* key, std::vector<double>& dst) {
  if (!parent || !parent[key]) return;
  const YAML::Node n = parent[key];
  if (!n.IsSequence()) {
    throw std::runtime_error(std::string("config: key '") + key +
                             "' must be a numeric sequence");
  }
  std::vector<double> v;
  v.reserve(n.size());
  for (const auto& e : n) v.push_back(e.as<double>());
  dst = std::move(v);
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
    for (int col = 0; col < 3; ++col) m(r, col) = n[static_cast<std::size_t>(3 * r + col)].as<double>();
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
    get(l, "ptp", c.lidar.ptp);
    get(l, "timestamp_mode", c.lidar.timestamp_mode);
    if (l["extrinsic_T"]) c.lidar.extrinsic_T = read_vec3(l["extrinsic_T"], "sensors.lidar.extrinsic_T");
    if (l["extrinsic_R"]) c.lidar.extrinsic_R = read_mat3(l["extrinsic_R"], "sensors.lidar.extrinsic_R");
    c.lidar.extrinsic_set = static_cast<bool>(l["extrinsic_T"]) || static_cast<bool>(l["extrinsic_R"]);
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
        throw std::runtime_error(
            "config: key 'sensors.camera.intrinsics' must be [fx,fy,cx,cy]");
      }
      c.camera.intrinsics =
          Eigen::Vector4d(iv[0].as<double>(), iv[1].as<double>(), iv[2].as<double>(),
                          iv[3].as<double>());
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

  const YAML::Node g = n["gnss"];
  if (g) {
    get(g, "enable", c.gnss.enable);
    if (g["min_fix_type"]) {
      c.gnss.min_fix_type = parse_enum<GnssFix::FixType>(
          g["min_fix_type"], "preprocess.gnss.min_fix_type",
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
    c.kind = parse_enum<FrontEndKind>(n["kind"], "frontend.kind",
                                      {{"ct_livo", FrontEndKind::CtLivo},
                                       {"iekf_oracle", FrontEndKind::IekfOracle}});
  }
  get(n, "init_time_s", c.init_time_s);
  get(n, "solver_max_iterations", c.solver_max_iterations);
  get(n, "solver_epsi", c.solver_epsi);
  get(n, "degeneracy_thresh", c.degeneracy_thresh);
  get(n, "max_outer_iters", c.max_outer_iters);
  get(n, "reassoc_steps", c.reassoc_steps);
  get(n, "assoc_shift_thresh_m", c.assoc_shift_thresh_m);
  get(n, "assoc_shift_thresh_deg", c.assoc_shift_thresh_deg);
  const YAML::Node solver = n["solver"];
  // max_iterations / epsi keep their legacy flat keys; the solver block carries the
  // deadline bracket. Accept max_iterations under the block too for spec-name parity.
  get(solver, "max_iterations", c.solver_max_iterations);
  get(solver, "epsi", c.solver_epsi);
  get(solver, "time_limit_ms", c.solver.time_limit_ms);
  get(solver, "min_iterations", c.solver.min_iterations);
  const YAML::Node bias = n["bias"];
  get(bias, "gyr_max", c.bias.gyr_max);
  get(bias, "acc_max", c.bias.acc_max);
  get(bias, "knot_dt_ms", c.bias.knot_dt_ms);
  const YAML::Node mreg = n["motion_reg"];
  get(mreg, "enable", c.motion_reg.enable);
  get(mreg, "weight", c.motion_reg.weight);
  get(mreg, "excitation_floor", c.motion_reg.excitation_floor);
  const YAML::Node sp = n["spline"];
  if (sp) {
    if (sp["order"]) {
      c.spline.order =
          parse_enum<SplineOrder>(sp["order"], "frontend.spline.order",
                                  {{"cubic", SplineOrder::Cubic}});
    }
    get(sp, "knot_dt_ms", c.spline.knot_dt_ms);
    get(sp, "window_knots", c.spline.window_knots);
    get(sp, "n_cp_max", c.spline.n_cp_max);
    get(sp, "time_offset_estimate", c.spline.time_offset_estimate);
    get_vec(sp, "knot_omega_thresh", c.spline.knot_omega_thresh);
    get_vec(sp, "knot_accel_thresh", c.spline.knot_accel_thresh);
    get(sp, "knot_density_hysteresis", c.spline.knot_density_hysteresis);
  }
  const YAML::Node lid = n["lidar"];
  get(lid, "voxel_map_m", c.lidar.voxel_map_m);
  get(lid, "num_match_points", c.lidar.num_match_points);
  get(lid, "max_match_dist_sq", c.lidar.max_match_dist_sq);
  get(lid, "plane_thresh", c.lidar.plane_thresh);
  get(lid, "point_cov", c.lidar.point_cov);
  get(lid, "max_lidar_factors", c.lidar.max_lidar_factors);
  get(lid, "min_factors_per_normal", c.lidar.min_factors_per_normal);
  get(lid, "normal_strata", c.lidar.normal_strata);
  get(lid, "local_map_cube_m", c.lidar.local_map_cube_m);
  const YAML::Node vis = n["visual"];
  get(vis, "enable", c.visual.enable);
  get(vis, "patch", c.visual.patch);
  get(vis, "levels", c.visual.levels);
  get(vis, "img_point_cov", c.visual.img_point_cov);
  get(vis, "outlier_threshold", c.visual.outlier_threshold);
  get(vis, "ncc_thre", c.visual.ncc_thre);
  get(vis, "exposure_estimate_en", c.visual.exposure_estimate_en);
  get(vis, "inv_expo_cov", c.visual.inv_expo_cov);
  get(vis, "inv_expo_min", c.visual.inv_expo_min);
  get(vis, "grid_cell_px", c.visual.grid_cell_px);
  get(vis, "warp_det_min", c.visual.warp_det_min);
  get(vis, "warp_det_max", c.visual.warp_det_max);
  get(vis, "warp_cond_max", c.visual.warp_cond_max);
  get(vis, "active_box_m", c.visual.active_box_m);
  const YAML::Node g = n["gnss"];
  get(g, "use", c.gnss.use);
  get(g, "floor_fixed_h", c.gnss.floor_fixed_h);
  get(g, "floor_fixed_v", c.gnss.floor_fixed_v);
  get(g, "floor_float_h", c.gnss.floor_float_h);
  get(g, "floor_float_v", c.gnss.floor_float_v);
  get(g, "floor_dgps_h", c.gnss.floor_dgps_h);
  get(g, "floor_dgps_v", c.gnss.floor_dgps_v);
  get(g, "floor_spp_h", c.gnss.floor_spp_h);
  get(g, "floor_spp_v", c.gnss.floor_spp_v);
  get(g, "innovation_k", c.gnss.innovation_k);
  get(g, "reacquire_count", c.gnss.reacquire_count);
  get(n, "extrinsic_refine", c.extrinsic_refine);
  const YAML::Node kf = n["keyframe"];
  get(kf, "dist_m", c.keyframe.dist_m);
  get(kf, "rot_deg", c.keyframe.rot_deg);
  get(kf, "time_s", c.keyframe.time_s);
}

void load_backend(const YAML::Node& root, BackendConfig& c) {
  const YAML::Node n = root["backend"];
  if (!n) return;
  if (n["kind"]) {
    c.kind = parse_enum<BackEndKind>(n["kind"], "backend.kind",
                                     {{"isam2", BackEndKind::Isam2}});
  }
  get(n, "relinearize_thresh", c.isam2_relinearize_thresh);
  if (n["robust"]) {
    c.robust_kernel = parse_enum<RobustKernel>(n["robust"], "backend.robust",
                                               {{"gnc_cauchy", RobustKernel::Cauchy},
                                                {"cauchy", RobustKernel::Cauchy},
                                                {"gm", RobustKernel::Gm},
                                                {"tls", RobustKernel::Tls}});
  }
}

void load_map(const YAML::Node& root, MapConfig& c) {
  const YAML::Node n = root["map"];
  if (!n) return;
  if (n["backend"]) {
    c.backend = parse_enum<MapBackend>(n["backend"], "map.backend",
                                       {{"nvblox", MapBackend::Nvblox}});
  }
  get(n, "tsdf_voxel_m", c.tsdf_voxel_m);
  get(n, "reg_voxel_m", c.reg_voxel_m);
  if (n["mesh"]) {
    c.mesh = parse_enum<MeshKind>(n["mesh"], "map.mesh",
                                  {{"marching_cubes", MeshKind::MarchingCubes}});
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
  get(n, "pcm", c.pcm);
  // The top-level YAML key gicp_fitness_max is the loose RMSE accept band.
  get(n, "gicp_fitness_max", c.gicp_rmse_max);
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
  get(n, "timing", c.timing);
  get(n, "telemetry_rate_hz", c.telemetry_rate_hz);
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

  // --- CT spline knot spacing ---
  if (frontend.spline.knot_dt_ms <= 0.0) {
    return fail("frontend.spline.knot_dt_ms must be > 0");
  }
  if (frontend.spline.window_knots < 1) {
    return fail("frontend.spline.window_knots must be >= 1");
  }
  // 0 (or 1) disables adaptive density to one control point per segment; a negative
  // cap is invalid.
  if (frontend.spline.n_cp_max < 0) {
    return fail("frontend.spline.n_cp_max must be >= 0 (<=1 disables adaptive density)");
  }

  // --- front-end solver ---
  if (frontend.init_time_s <= 0.0) {
    return fail("frontend.init_time_s must be > 0");
  }
  if (frontend.solver_max_iterations < 1) {
    return fail("frontend.solver_max_iterations must be >= 1");
  }
  if (frontend.solver_epsi <= 0.0) {
    return fail("frontend.solver_epsi must be > 0");
  }
  if (frontend.degeneracy_thresh < 0.0) {
    return fail("frontend.degeneracy_thresh must be >= 0");
  }
  if (frontend.solver.time_limit_ms <= 0.0) {
    return fail("frontend.solver.time_limit_ms must be > 0");
  }
  if (frontend.solver.min_iterations < 1) {
    return fail("frontend.solver.min_iterations must be >= 1");
  }
  if (frontend.solver.min_iterations > frontend.solver_max_iterations) {
    return fail("frontend.solver.min_iterations (" +
                std::to_string(frontend.solver.min_iterations) +
                ") must be <= frontend.solver_max_iterations (" +
                std::to_string(frontend.solver_max_iterations) + ")");
  }
  if (frontend.max_outer_iters < 1) {
    return fail("frontend.max_outer_iters must be >= 1");
  }
  if (frontend.reassoc_steps < 1) {
    return fail("frontend.reassoc_steps must be >= 1");
  }
  if (frontend.assoc_shift_thresh_m < 0.0 || frontend.assoc_shift_thresh_deg < 0.0) {
    return fail("frontend.assoc_shift_thresh_{m,deg} must be >= 0");
  }
  if (frontend.bias.gyr_max <= 0.0 || frontend.bias.acc_max <= 0.0) {
    return fail("frontend.bias.{gyr_max,acc_max} must be > 0");
  }
  if (frontend.bias.knot_dt_ms <= 0.0) {
    return fail("frontend.bias.knot_dt_ms must be > 0");
  }
  if (frontend.motion_reg.weight < 0.0) {
    return fail("frontend.motion_reg.weight must be >= 0");
  }
  if (frontend.motion_reg.excitation_floor < 0.0) {
    return fail("frontend.motion_reg.excitation_floor must be >= 0");
  }

  // --- front-end LiDAR association ---
  if (frontend.lidar.voxel_map_m <= 0.0) {
    return fail("frontend.lidar.voxel_map_m must be > 0");
  }
  if (frontend.lidar.num_match_points < 3) {
    return fail("frontend.lidar.num_match_points must be >= 3 (a plane needs 3 points)");
  }
  if (frontend.lidar.max_match_dist_sq <= 0.0) {
    return fail("frontend.lidar.max_match_dist_sq must be > 0");
  }
  if (frontend.lidar.plane_thresh <= 0.0) {
    return fail("frontend.lidar.plane_thresh must be > 0");
  }
  if (frontend.lidar.point_cov <= 0.0) {
    return fail("frontend.lidar.point_cov must be > 0");
  }
  if (frontend.lidar.max_lidar_factors < 1) {
    return fail("frontend.lidar.max_lidar_factors must be >= 1");
  }
  if (frontend.lidar.min_factors_per_normal < 0) {
    return fail("frontend.lidar.min_factors_per_normal must be >= 0");
  }
  if (frontend.lidar.normal_strata < 1) {
    return fail("frontend.lidar.normal_strata must be >= 1");
  }
  // Each stratum's guaranteed floor must fit inside the global cap.
  if (frontend.lidar.min_factors_per_normal * frontend.lidar.normal_strata >
      frontend.lidar.max_lidar_factors) {
    return fail("frontend.lidar.min_factors_per_normal * normal_strata (" +
                std::to_string(frontend.lidar.min_factors_per_normal *
                               frontend.lidar.normal_strata) +
                ") must be <= max_lidar_factors (" +
                std::to_string(frontend.lidar.max_lidar_factors) + ")");
  }

  // --- adaptive-knot density ---
  if (frontend.spline.knot_density_hysteresis < 0.0 ||
      frontend.spline.knot_density_hysteresis >= 1.0) {
    return fail("frontend.spline.knot_density_hysteresis must be in [0, 1)");
  }
  const auto monotone_nonneg = [](const std::vector<double>& v) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (v[i] < 0.0) return false;
      if (i > 0 && v[i] <= v[i - 1]) return false;
    }
    return true;
  };
  if (!monotone_nonneg(frontend.spline.knot_omega_thresh)) {
    return fail("frontend.spline.knot_omega_thresh must be strictly increasing and >= 0");
  }
  if (!monotone_nonneg(frontend.spline.knot_accel_thresh)) {
    return fail("frontend.spline.knot_accel_thresh must be strictly increasing and >= 0");
  }

  // --- front-end visual ---
  if (frontend.visual.patch < 1) {
    return fail("frontend.visual.patch must be >= 1");
  }
  if (frontend.visual.img_point_cov <= 0.0) {
    return fail("frontend.visual.img_point_cov must be > 0");
  }
  if (frontend.visual.outlier_threshold <= 0.0) {
    return fail("frontend.visual.outlier_threshold must be > 0");
  }
  if (frontend.visual.inv_expo_cov <= 0.0) {
    return fail("frontend.visual.inv_expo_cov must be > 0");
  }
  if (frontend.visual.inv_expo_min <= 0.0) {
    return fail("frontend.visual.inv_expo_min must be > 0");
  }
  if (frontend.visual.grid_cell_px < 1) {
    return fail("frontend.visual.grid_cell_px must be >= 1");
  }
  if (frontend.visual.warp_det_min <= 0.0 ||
      frontend.visual.warp_det_max <= frontend.visual.warp_det_min) {
    return fail("frontend.visual.warp_det_min/max must satisfy 0 < min < max");
  }
  if (frontend.visual.warp_cond_max < 1.0) {
    return fail("frontend.visual.warp_cond_max must be >= 1");
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
  if (preprocess.lidar.sweep_floor_frac <= 0.0 ||
      preprocess.lidar.sweep_floor_frac > 1.0) {
    return fail("preprocess.sweep_floor_frac must be in (0, 1]");
  }
  if (preprocess.lidar.surf_max_pts < 1) {
    return fail("preprocess.surf_max_pts must be >= 1");
  }

  // --- camera pyramid ---
  if (preprocess.camera.pyramid_levels < 1) {
    return fail("preprocess.camera.pyramid_levels must be >= 1");
  }
  if (frontend.visual.levels < 1) {
    return fail("frontend.visual.levels must be >= 1");
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
                ") must be >= one LiDAR period (" + std::to_string(lidar_period_ms) +
                " ms)");
  }

  // --- PTP requires at least one PTP-disciplined sensor ---
  if (time.source == TimeSource::Ptp && !sensors.lidar.ptp) {
    return fail(
        "time.source == ptp requires at least one sensor with ptp: true "
        "(sensors.lidar.ptp is false)");
  }

  // --- thread counts ---
  if (pipeline.threads.frontend < 1 || pipeline.threads.backend < 1 ||
      pipeline.threads.map < 1) {
    return fail("pipeline.threads.{frontend,backend,map} must each be >= 1");
  }

  // --- queue capacities ---
  if (pipeline.queue.meas_capacity < 1 || pipeline.queue.kf_capacity < 1 ||
      pipeline.queue.map_capacity < 1) {
    return fail("pipeline.queue.{meas,kf,map}_capacity must each be > 0");
  }

  // --- camera shutter (the rolling-shutter path is designed but not built) ---
  if (sensors.camera.shutter != "global") {
    return fail("sensors.camera.shutter must be 'global' (got '" +
                sensors.camera.shutter + "'); the rolling-shutter path is not built");
  }

  // --- camera distortion model (closed string set; reject unknowns loudly) ---
  {
    const std::string& m = sensors.camera.distortion_model;
    if (m != "none" && m != "radtan" && m != "plumb_bob" && m != "equidistant" &&
        m != "equi") {
      return fail("sensors.camera.distortion_model must be one of "
                  "none|radtan|plumb_bob|equidistant|equi (got '" + m + "')");
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
