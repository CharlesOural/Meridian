#include "debug/ros_telemetry_sink.hpp"

#include <algorithm>

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
  const auto reliable = rclcpp::QoS(50).reliable();
  pub_telemetry_ =
      node_->create_publisher<meridian_msgs::msg::Telemetry>("/meridian/telemetry", reliable);
  pub_timing_ = node_->create_publisher<meridian_msgs::msg::StageTiming>(
      "/meridian/stage_timing", reliable);
  pub_events_ =
      node_->create_publisher<meridian_msgs::msg::Event>("/meridian/events", reliable);
  pub_markers_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/meridian/markers", rclcpp::QoS(5).reliable());

  // The known-hot per-sweep cloud is created here so the stage thread never hits the
  // lazy create_publisher path on its first publish.
  cloud_pubs_["body/scan"] = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
      cloud_topic("body/scan"), rclcpp::QoS(5).best_effort());
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
  if (inserted) st.max_hz = default_hz(heavy);
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
      // The assembled map cloud is published infrequently (every few folds) and carries the
      // whole accumulated geometry, not a per-sweep view. Latch it (reliable +
      // transient_local) so a late-joining or reconnecting subscriber gets the current map
      // at once instead of waiting up to a fold for the next send, and so it persists after
      // the run ends. Per-sweep clouds stay best-effort: cheap to miss, and a stale one held
      // by a latch would be worse than a brief gap.
      const rclcpp::QoS qos = std::string(key) == "map/cloud"
                                  ? rclcpp::QoS(1).reliable().transient_local()
                                  : rclcpp::QoS(5).best_effort();
      slot = node_->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_topic(key), qos);
    }
    pub = slot;
  }
  pub->publish(to_pointcloud2(view, frame_name(f), t));
}

void RosTelemetrySink::pose(const char* key, const Pose& p, Frame f, Timestamp t) {
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
  st.max_hz = max_hz > 0.0 ? max_hz : default_rate_hz_;
  if (is_wildcard(key)) {
    const std::string prefix = key.substr(0, key.size() - 1);  // keep trailing '/'
    for (auto& [p, s] : wildcards_) {
      if (p == prefix) {
        s = st;
        return true;
      }
    }
    wildcards_.emplace_back(prefix, st);
    return true;
  }
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
  if (key.find("rate_hz") != std::string::npos) return "hz";
  if (key.find("_ms") != std::string::npos) return "ms";
  if (key.find("n_") != std::string::npos || key.find("count") != std::string::npos ||
      key.find("depth") != std::string::npos || key.find("dropped") != std::string::npos ||
      key.find("points") != std::string::npos || key.find("levels") != std::string::npos)
    return "count";
  return "";
}

std::string RosTelemetrySink::cloud_topic(const std::string& key) const {
  if (key == "body/scan") return "/meridian/cloud_body";
  if (key == "map/registered") return "/meridian/cloud_registered";
  if (key == "frontend/lidar/inliers") return "/meridian/cloud_effective";
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
