#include "debug/ros_telemetry_sink.hpp"

#include <algorithm>
#include <string_view>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include "conversions/core2ros.hpp"

namespace meridian {

namespace {

constexpr double kHeavyDefaultHz = 2.0;
constexpr double kTimingEwmaAlpha = 0.1;

bool is_wildcard(const std::string& key) {
  return key.size() >= 2 && key.compare(key.size() - 2, 2, "/*") == 0;
}

std::uint8_t to_u8(Level level) { return static_cast<std::uint8_t>(level); }

Level from_log_level(LogLevel level) {
  switch (level) {
    case LogLevel::Trace: return Level::Trace;
    case LogLevel::Debug: return Level::Debug;
    case LogLevel::Info: return Level::Info;
    case LogLevel::Warn: return Level::Warn;
    case LogLevel::Error: return Level::Error;
  }
  return Level::Info;
}

}  // namespace

RosTelemetrySink::RosTelemetrySink(rclcpp::Node* node, const DebugConfig& cfg)
    : node_(node), cfg_(cfg), default_rate_hz_(cfg.telemetry_rate_hz) {
  // The multiplexed scalar/vec topic bursts once per sweep: with every debug group
  // on that is ~60 messages in one instant, so the publisher history must hold
  // several full bursts or the oldest (earliest-emitted) keys of each burst are
  // silently dropped toward a slow subscriber. 512 ≈ 8 bursts ≈ <1 MB.
  pub_telemetry_ = node_->create_publisher<meridian_msgs::msg::Telemetry>(
      "/meridian/telemetry", rclcpp::QoS(512).reliable());
  pub_timing_ = node_->create_publisher<meridian_msgs::msg::StageTiming>(
      "/meridian/stage_timing", rclcpp::QoS(256).reliable());
  pub_events_ = node_->create_publisher<meridian_msgs::msg::Event>("/meridian/events",
                                                                   rclcpp::QoS(50).reliable());
  pub_markers_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/meridian/markers", rclcpp::QoS(5).reliable());

  // The known-hot per-sweep cloud is created here so the stage thread never hits the
  // lazy create_publisher path on its first publish.
  cloud_pubs_["body/scan"] = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
      cloud_topic("body/scan"), rclcpp::QoS(5).best_effort());

  // Seed the wildcard gate table from the config debug groups. Each group is exactly
  // one key-prefix wildcard, the same entries /meridian/set_debug_key writes, so the
  // config posture and a live toggle go through one mechanism. A group max_hz of 0
  // keeps the per-key class default (scalars at telemetry_rate_hz, heavy at 2 Hz).
  const auto seed = [this](const char* prefix, const DebugGroup& g) {
    KeyState st;
    st.enabled = g.enable;
    st.max_hz = g.max_hz;  // 0 = class default, resolved per key in pass()
    wildcards_.emplace_back(prefix, st);
  };
  seed("frontend/assoc/", cfg.assoc);
  seed("frontend/solver/", cfg.solver);
  seed("frontend/deskew/", cfg.deskew);
  seed("frontend/spline/", cfg.spline);
  // Adaptive-knot-density keys ride the spline group: one debug posture covers the
  // whole trajectory-representation surface.
  seed("frontend/ncp/", cfg.spline);
  seed("frontend/map/", cfg.map_health);

  // The spline path stream: an exact-key entry (checked before the spline/* wildcard)
  // so /meridian/path works without the kinematics group. Unlimited key rate — the
  // front-end already samples at the configured cadence and the aggregator throttles
  // the republish to path_publish_hz.
  KeyState path_state;
  path_state.enabled = cfg.publish_path;
  path_state.max_hz = -1.0;  // negative = explicitly unlimited (pass() treats <=0 as no limit)
  keys_["frontend/spline/path_sample"] = path_state;
  if (cfg.publish_path) {
    pub_path_ = node_->create_publisher<nav_msgs::msg::Path>("/meridian/path",
                                                             rclcpp::QoS(1).reliable());
    path_.header.frame_id = frame_name(Frame::Odom);
  }
}

bool RosTelemetrySink::enabled(const char* key) const {
  // Cheap advisory hint a caller may use to skip building a heavy payload; the
  // authoritative enable + rate-limit gate runs inside each publish via pass().
  std::lock_guard<std::mutex> lock(m_);
  return flag_enabled(key);
}

bool RosTelemetrySink::flag_enabled(const std::string& key) const {
  const auto it = keys_.find(key);
  if (it != keys_.end()) return it->second.enabled;
  for (const auto& [prefix, state] : wildcards_) {
    if (key.compare(0, prefix.size(), prefix) == 0) return state.enabled;
  }
  // Class defaults: heavy payload keys follow the publish_* config flags.
  if (key.rfind("body/", 0) == 0 || key.rfind("map/", 0) == 0) return cfg_.publish_clouds;
  return true;
}

bool RosTelemetrySink::pass(const std::string& key, bool heavy) {
  std::lock_guard<std::mutex> lock(m_);
  if (!flag_enabled(key)) return false;

  auto [it, inserted] = keys_.try_emplace(key);
  KeyState& st = it->second;
  if (inserted) {
    // First publication of this key: rate from the matching wildcard group when it
    // carries an explicit max_hz, otherwise the key-class default.
    st.max_hz = default_hz(heavy);
    for (const auto& [prefix, wst] : wildcards_) {
      if (key.compare(0, prefix.size(), prefix) == 0) {
        if (wst.max_hz > 0.0) st.max_hz = wst.max_hz;
        break;
      }
    }
  }
  if (st.max_hz <= 0.0) {
    st.last = Clock::now();
    return true;
  }
  const auto now = Clock::now();
  const auto min_interval = std::chrono::duration<double>(1.0 / st.max_hz);
  if (st.last.time_since_epoch().count() != 0 && (now - st.last) < min_interval)
    return false;
  st.last = now;
  return true;
}

double RosTelemetrySink::default_hz(bool heavy) const {
  return heavy ? std::min(kHeavyDefaultHz, default_rate_hz_) : default_rate_hz_;
}

void RosTelemetrySink::scalar(const char* key, double v, Timestamp t) {
  if (!pass(key, /*heavy=*/false)) return;
  meridian_msgs::msg::Telemetry m;
  m.stamp = to_ros(t);
  m.key = key;
  m.values = {v};
  m.axis_order = "";
  m.unit = unit_of(key);
  pub_telemetry_->publish(m);
}

void RosTelemetrySink::vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v,
                           Timestamp t, const char* axis_order) {
  if (!pass(key, /*heavy=*/false)) return;
  meridian_msgs::msg::Telemetry m;
  m.stamp = to_ros(t);
  m.key = key;
  m.values.assign(v.data(), v.data() + v.size());
  m.axis_order = axis_order ? axis_order : "";
  m.unit = unit_of(key);
  pub_telemetry_->publish(m);
}

void RosTelemetrySink::cloud(const char* key, const PointCloudView& view, Frame f,
                             Timestamp t) {
  if (!pass(key, /*heavy=*/true)) return;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub;
  {
    std::lock_guard<std::mutex> lock(m_);
    auto& slot = cloud_pubs_[key];
    if (!slot) {
      slot = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
          cloud_topic(key), rclcpp::QoS(5).best_effort());
    }
    pub = slot;
  }
  pub->publish(to_pointcloud2(view, frame_name(f), t));
}

void RosTelemetrySink::pose(const char* key, const Pose& p, Frame f, Timestamp t) {
  // Solved-spline path samples aggregate into one nav_msgs/Path instead of a per-key
  // Odometry topic.
  if (std::string_view(key) == "frontend/spline/path_sample") {
    append_path(p, t);
    return;
  }
  if (!cfg_.publish_odom && std::string(key) == "odom/body") return;
  if (!pass(key, /*heavy=*/false)) return;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub;
  {
    std::lock_guard<std::mutex> lock(m_);
    auto& slot = pose_pubs_[key];
    if (!slot) {
      slot = node_->create_publisher<nav_msgs::msg::Odometry>(pose_topic(key),
                                                              rclcpp::QoS(10).reliable());
    }
    pub = slot;
  }
  pub->publish(to_odometry(p, frame_name(f), t));
}

void RosTelemetrySink::append_path(const Pose& p, Timestamp t) {
  if (!pub_path_) return;
  nav_msgs::msg::Path snapshot;
  bool publish_now = false;
  {
    std::lock_guard<std::mutex> lock(m_);
    if (!flag_enabled("frontend/spline/path_sample")) return;  // runtime toggle
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = to_ros(t);
    ps.header.frame_id = path_.header.frame_id;
    ps.pose.position.x = p.t.x();
    ps.pose.position.y = p.t.y();
    ps.pose.position.z = p.t.z();
    ps.pose.orientation.w = p.q.w();
    ps.pose.orientation.x = p.q.x();
    ps.pose.orientation.y = p.q.y();
    ps.pose.orientation.z = p.q.z();
    path_.poses.push_back(ps);
    // Ring cap: nav_msgs/Path republishes the whole array, so the cap bounds both
    // memory and per-publish serialisation on a long mission.
    const auto cap = static_cast<std::size_t>(std::max(cfg_.path_max_poses, 1));
    if (path_.poses.size() > cap) {
      path_.poses.erase(path_.poses.begin(),
                        path_.poses.begin() +
                            static_cast<std::ptrdiff_t>(path_.poses.size() - cap));
    }
    path_.header.stamp = ps.header.stamp;
    const auto now = Clock::now();
    const auto min_interval =
        std::chrono::duration<double>(1.0 / std::max(cfg_.path_publish_hz, 1e-3));
    if (path_last_pub_.time_since_epoch().count() == 0 ||
        (now - path_last_pub_) >= min_interval) {
      path_last_pub_ = now;
      snapshot = path_;  // copy under the lock; publish outside it
      publish_now = true;
    }
  }
  if (publish_now) {
    pub_path_->publish(snapshot);
  }
}

void RosTelemetrySink::marker(const Marker& m, Timestamp t) {
  if (!cfg_.publish_markers) return;
  if (!pass("markers/" + m.ns, /*heavy=*/true)) return;
  visualization_msgs::msg::MarkerArray arr;
  arr.markers.push_back(to_marker(m, t));
  pub_markers_->publish(arr);
}

void RosTelemetrySink::image(const char* key, const ImageOverlay& ov, Timestamp t) {
  if (!pass(key, /*heavy=*/true)) return;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub;
  {
    std::lock_guard<std::mutex> lock(m_);
    auto& slot = image_pubs_[key];
    if (!slot) {
      slot = node_->create_publisher<sensor_msgs::msg::Image>(image_topic(key),
                                                              rclcpp::QoS(5).best_effort());
    }
    pub = slot;
  }
  pub->publish(to_image(ov, t));
}

void RosTelemetrySink::timing(const char* stage, double ms, Timestamp t) {
  if (!cfg_.timing) return;
  meridian_msgs::msg::StageTiming m;
  m.stamp = to_ros(t);
  m.stage = stage;
  m.ms = ms;
  {
    std::lock_guard<std::mutex> lock(m_);
    StageStats& s = stages_[stage];
    s.count += 1;
    s.ewma = s.count == 1 ? ms : (1.0 - kTimingEwmaAlpha) * s.ewma + kTimingEwmaAlpha * ms;
    s.max = std::max(s.max, ms);
    m.ms_avg = s.ewma;
    m.ms_max = s.max;
    m.count = s.count;
  }
  pub_timing_->publish(m);
}

void RosTelemetrySink::event(Level level, const char* tag, std::string_view msg,
                             Timestamp t) {
  meridian_msgs::msg::Event m;
  m.stamp = to_ros(t);
  m.level = to_u8(level);
  m.tag = tag;
  m.message.assign(msg.begin(), msg.end());
  pub_events_->publish(m);

  // One call, two surfaces: the event also lands in the text log.
  const auto logger = node_->get_logger();
  switch (level) {
    case Level::Trace:
    case Level::Debug:
      RCLCPP_DEBUG(logger, "[%s] %.*s", tag, static_cast<int>(msg.size()), msg.data());
      break;
    case Level::Info:
      RCLCPP_INFO(logger, "[%s] %.*s", tag, static_cast<int>(msg.size()), msg.data());
      break;
    case Level::Warn:
      RCLCPP_WARN(logger, "[%s] %.*s", tag, static_cast<int>(msg.size()), msg.data());
      break;
    case Level::Error:
      RCLCPP_ERROR(logger, "[%s] %.*s", tag, static_cast<int>(msg.size()), msg.data());
      break;
  }
}

bool RosTelemetrySink::set_key(const std::string& key, bool enable, double max_hz) {
  std::lock_guard<std::mutex> lock(m_);
  KeyState st;
  st.enabled = enable;
  if (is_wildcard(key)) {
    // Wildcard rate semantics: 0 keeps each key's class default; > 0 overrides. The
    // override is also pushed onto already-materialised keys under the prefix so a
    // live group toggle takes effect without waiting for re-creation.
    st.max_hz = max_hz;
    const std::string prefix = key.substr(0, key.size() - 1);  // keep trailing '/'
    if (max_hz > 0.0) {
      for (auto& [k, s] : keys_) {
        if (k.compare(0, prefix.size(), prefix) == 0) s.max_hz = max_hz;
      }
    }
    for (auto& [p, s] : wildcards_) {
      if (p == prefix) {
        s = st;
        return true;
      }
    }
    wildcards_.emplace_back(prefix, st);
    return true;
  }
  st.max_hz = max_hz > 0.0 ? max_hz : default_rate_hz_;
  keys_[key] = st;
  return true;
}

void RosTelemetrySink::set_default_rate(double hz) {
  std::lock_guard<std::mutex> lock(m_);
  default_rate_hz_ = hz;
}

void RosTelemetrySink::reset_timing() {
  std::lock_guard<std::mutex> lock(m_);
  stages_.clear();
}

std::string RosTelemetrySink::sanitize(const std::string& key) {
  std::string out = key;
  std::replace(out.begin(), out.end(), '/', '_');
  return out;
}

const char* RosTelemetrySink::unit_of(const std::string& key) {
  // Exact physical-unit keys first, so they are never shadowed by the generic
  // suffix / count heuristics below.
  static const std::unordered_map<std::string, const char*> kExact = {
      {"frontend/lidar/res_mean", "m"},
      {"frontend/lidar/res_max", "m"},
      {"frontend/assoc/res_p95", "m"},
      {"frontend/assoc/nn_dist_mean", "m"},
      {"frontend/assoc/nn_dist_p95", "m"},
      {"frontend/assoc/plane_rms_mean", "m"},
      {"frontend/assoc/plane_rms_p95", "m"},
      {"frontend/assoc/res_hist", "count"},
      {"frontend/imu/res_acc", "m/s^2"},
      {"frontend/imu/res_gyr", "rad/s"},
      {"frontend/bias_acc", "m/s^2"},
      {"frontend/bias_gyr", "rad/s"},
      {"frontend/grav_norm", "m/s^2"},
      {"frontend/spline/vel", "m/s"},
      {"frontend/spline/acc", "m/s^2"},
      {"frontend/spline/acc_rms_span", "m/s^2"},
      {"frontend/spline/omega", "rad/s"},
      {"frontend/spline/omega_rms_span", "rad/s"},
      {"frontend/spline/jerk_rms", "m/s^3"},
      {"frontend/window_span_s", "s"},
      {"frontend/lidar/inlier_ratio", "ratio"},
      {"frontend/state/vel_norm", "m/s"},
      {"frontend/state/bias_gyr_norm", "rad/s"},
      {"frontend/state/bias_acc_norm", "m/s^2"},
  };
  const auto it = kExact.find(key);
  if (it != kExact.end()) return it->second;
  const auto ends_with = [&key](const char* suffix) {
    const std::size_t n = std::string_view(suffix).size();
    return key.size() >= n && key.compare(key.size() - n, n, suffix) == 0;
  };
  if (key.find("rate_hz") != std::string::npos) return "hz";
  if (key.find("_ms") != std::string::npos) return "ms";
  if (ends_with("_m")) return "m";
  if (ends_with("_deg")) return "deg";
  if (key.find("ratio") != std::string::npos || key.find("frac") != std::string::npos)
    return "ratio";
  if (key.find("n_") != std::string::npos || key.find("count") != std::string::npos ||
      key.find("depth") != std::string::npos || key.find("dropped") != std::string::npos ||
      key.find("points") != std::string::npos || key.find("levels") != std::string::npos ||
      key.find("reject_") != std::string::npos || key.find("strata") != std::string::npos)
    return "count";
  return "";
}

std::string RosTelemetrySink::cloud_topic(const std::string& key) const {
  if (key == "body/scan") return "/meridian/cloud_body";
  if (key == "map/registered") return "/meridian/cloud_registered";
  if (key == "frontend/lidar/inliers") return "/meridian/cloud_effective";
  if (key == "frontend/assoc/outliers") return "/meridian/cloud_outliers";
  if (key == "frontend/deskew/pre") return "/meridian/cloud_deskew_pre";
  return "/meridian/cloud/" + sanitize(key);
}

std::string RosTelemetrySink::pose_topic(const std::string& key) const {
  if (key == "odom/body") return "/meridian/odom";
  if (key.rfind("calib/", 0) == 0) return "/meridian/extrinsic";
  if (key.rfind("frontend/gnss", 0) == 0) return "/meridian/gnss";
  return "/meridian/pose/" + sanitize(key);
}

std::string RosTelemetrySink::image_topic(const std::string& key) const {
  if (key == "frontend/visual/patches") return "/meridian/visual_patches";
  return "/meridian/image/" + sanitize(key);
}

RclcppLogSink::RclcppLogSink(rclcpp::Logger logger, LogLevel min_level)
    : logger_(std::move(logger)), default_level_(from_log_level(min_level)) {}

void RclcppLogSink::log(Level level, const char* module, std::string_view kvline,
                        Timestamp /*t*/) {
  switch (level) {
    case Level::Trace:
    case Level::Debug:
      RCLCPP_DEBUG(logger_, "[%s] %.*s", module, static_cast<int>(kvline.size()),
                   kvline.data());
      break;
    case Level::Info:
      RCLCPP_INFO(logger_, "[%s] %.*s", module, static_cast<int>(kvline.size()),
                  kvline.data());
      break;
    case Level::Warn:
      RCLCPP_WARN(logger_, "[%s] %.*s", module, static_cast<int>(kvline.size()),
                  kvline.data());
      break;
    case Level::Error:
      RCLCPP_ERROR(logger_, "[%s] %.*s", module, static_cast<int>(kvline.size()),
                   kvline.data());
      break;
  }
}

bool RclcppLogSink::enabled(Level level, const char* module) const {
  std::lock_guard<std::mutex> lock(m_);
  const auto it = module_levels_.find(module);
  const Level min = it != module_levels_.end() ? it->second : default_level_;
  return static_cast<int>(level) >= static_cast<int>(min);
}

void RclcppLogSink::set_level(const std::string& module, Level level) {
  std::lock_guard<std::mutex> lock(m_);
  if (module.empty()) {
    default_level_ = level;
    module_levels_.clear();
  } else {
    module_levels_[module] = level;
  }
}

}  // namespace meridian
