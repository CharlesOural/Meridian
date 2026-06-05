#include "meridian/config/config_loader.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

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
    get(l, "model", c.lidar.model);
    get(l, "nominal_rate_hz", c.lidar.nominal_rate_hz);
    get(l, "ptp", c.lidar.ptp);
    get(l, "timestamp_mode", c.lidar.timestamp_mode);
    if (l["extrinsic_T"]) c.lidar.extrinsic_T = read_vec3(l["extrinsic_T"], "sensors.lidar.extrinsic_T");
    if (l["extrinsic_R"]) c.lidar.extrinsic_R = read_mat3(l["extrinsic_R"], "sensors.lidar.extrinsic_R");
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
    }
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
  get(n, "voxel_surf_m", c.voxel_surf_m);
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
  const YAML::Node sp = n["spline"];
  if (sp) {
    if (sp["order"]) {
      c.spline.order =
          parse_enum<SplineOrder>(sp["order"], "frontend.spline.order",
                                  {{"cubic", SplineOrder::Cubic}});
    }
    get(sp, "knot_dt_ms", c.spline.knot_dt_ms);
    get(sp, "window_knots", c.spline.window_knots);
  }
  const YAML::Node lid = n["lidar"];
  get(lid, "voxel_map_m", c.lidar.voxel_map_m);
  const YAML::Node vis = n["visual"];
  get(vis, "patch", c.visual.patch);
  get(vis, "levels", c.visual.levels);
  const YAML::Node g = n["gnss"];
  get(g, "use", c.gnss.use);
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
  if (map.tsdf_voxel_m > preprocess.voxel_surf_m) {
    return fail("map.tsdf_voxel_m (" + std::to_string(map.tsdf_voxel_m) +
                ") must be <= preprocess.voxel_surf_m (" +
                std::to_string(preprocess.voxel_surf_m) + ")");
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
