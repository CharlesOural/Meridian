#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <meridian_msgs/msg/event.hpp>
#include <meridian_msgs/msg/stage_timing.hpp>
#include <meridian_msgs/msg/telemetry.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "meridian/config/config.hpp"
#include "meridian/debug/log.hpp"
#include "meridian/debug/telemetry.hpp"

namespace meridian {

// Publishes telemetry to ROS topics: scalars/vectors multiplex onto /meridian/telemetry,
// timing onto /meridian/stage_timing (with sink-side EWMA/max/count), events onto
// /meridian/events (mirrored to the log), and heavy payloads onto lazily-created
// per-key publishers. A per-key gate (enable flag + min-interval rate limit, runtime
// togglable) sits in front of every publication; heavy payloads default to a low rate
// so introspection never starves the estimator.
//
// Thread-safe: called from source (wrapper) threads and the stage thread concurrently.
class RosTelemetrySink final : public TelemetrySink {
 public:
  RosTelemetrySink(rclcpp::Node* node, const DebugConfig& cfg);

  bool enabled(const char* key) const override;

  void scalar(const char* key, double v, Timestamp t) override;
  void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, Timestamp t,
           const char* axis_order) override;
  void cloud(const char* key, const PointCloudView& view, Frame f, Timestamp t) override;
  void pose(const char* key, const Pose& p, Frame f, Timestamp t) override;
  void marker(const Marker& m, Timestamp t) override;
  void image(const char* key, const ImageOverlay& ov, Timestamp t) override;
  void timing(const char* stage, double ms, Timestamp t) override;
  void event(Level level, const char* tag, std::string_view msg, Timestamp t) override;

  // DebugControl surface (wired to ROS services by the node). `key` may end in "/*"
  // to toggle a whole stream.
  bool set_key(const std::string& key, bool enable, double max_hz);
  void set_default_rate(double hz);
  void reset_timing();

 private:
  using Clock = std::chrono::steady_clock;

  struct KeyState {
    bool enabled = true;
    double max_hz = 0.0;  // 0 = unlimited
    Clock::time_point last{};
  };
  struct StageStats {
    double ewma = 0.0;
    double max = 0.0;
    std::uint64_t count = 0;
  };

  // Flag check only (the cheap pre-build gate).
  bool flag_enabled(const std::string& key) const;
  // Flag + rate limit; consumes a publication slot when it passes.
  bool pass(const std::string& key, bool heavy);
  // Default publication rate for a key class.
  double default_hz(bool heavy) const;
  // PathAggregator: appends one solved-spline sample to the /meridian/path ring and
  // republishes the whole path at most cfg.path_publish_hz.
  void append_path(const Pose& p, Timestamp t);

  static std::string sanitize(const std::string& key);  // '/' -> '_' for topic suffixes
  static const char* unit_of(const std::string& key);
  std::string cloud_topic(const std::string& key) const;
  std::string pose_topic(const std::string& key) const;
  std::string image_topic(const std::string& key) const;

  rclcpp::Node* node_;
  DebugConfig cfg_;

  // Created in the constructor and never reassigned afterwards, so their publish()
  // calls are safe to make outside m_.
  rclcpp::Publisher<meridian_msgs::msg::Telemetry>::SharedPtr pub_telemetry_;
  rclcpp::Publisher<meridian_msgs::msg::StageTiming>::SharedPtr pub_timing_;
  rclcpp::Publisher<meridian_msgs::msg::Event>::SharedPtr pub_events_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;  // null when publish_path off

  mutable std::mutex m_;
  std::unordered_map<std::string, KeyState> keys_;
  std::vector<std::pair<std::string, KeyState>> wildcards_;  // prefix (sans '*') -> state
  double default_rate_hz_;
  std::map<std::string, StageStats> stages_;

  // PathAggregator state (guarded by m_): the growing nav_msgs/Path ring plus the
  // steady-clock stamp of the last republish.
  nav_msgs::msg::Path path_;
  Clock::time_point path_last_pub_{};

  std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
      cloud_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr>
      pose_pubs_;
  std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr>
      image_pubs_;
};

// Bridges core logging onto the node's rclcpp logger, with a runtime-settable
// per-module minimum level.
class RclcppLogSink final : public LogSink {
 public:
  RclcppLogSink(rclcpp::Logger logger, LogLevel min_level);

  void log(Level level, const char* module, std::string_view kvline, Timestamp t) override;
  bool enabled(Level level, const char* module) const override;

  // "" sets the default level for every module without an explicit override.
  void set_level(const std::string& module, Level level);

 private:
  rclcpp::Logger logger_;
  mutable std::mutex m_;
  Level default_level_;
  std::unordered_map<std::string, Level> module_levels_;
};

}  // namespace meridian
