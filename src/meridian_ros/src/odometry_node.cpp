#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <meridian_msgs/srv/reset_timing.hpp>
#include <meridian_msgs/srv/set_debug_key.hpp>
#include <meridian_msgs/srv/set_log_level.hpp>
#include <meridian_msgs/srv/set_telemetry_rate.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "conversions/core2ros.hpp"
#include "conversions/ros2core.hpp"
#include "debug/ros_telemetry_sink.hpp"
#include "meridian/config/config_loader.hpp"
#include "meridian/pipeline/meridian_pipeline.hpp"

namespace meridian {

// Thin wrapper node: declare params -> build Config -> construct the pipeline ->
// subscribe (convert msg -> core, push) -> expose the DebugControl services. Every
// algorithm lives below the pipeline boundary.
class OdometryNode : public rclcpp::Node {
 public:
  OdometryNode() : rclcpp::Node("meridian_odometry") {
    const std::string config_file = declare_parameter<std::string>("config_file", "");
    if (config_file.empty()) {
      throw std::runtime_error("parameter 'config_file' is required");
    }
    cfg_ = load_config_yaml(config_file);

    auto sink = std::make_unique<RosTelemetrySink>(this, cfg_.debug);
    sink_ = sink.get();
    log_sink_ = std::make_unique<RclcppLogSink>(get_logger(), cfg_.debug.level);
    set_log_sink(log_sink_.get());

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    pipeline_ = std::make_unique<MeridianPipeline>(cfg_, std::move(sink));
    pipeline_->set_group_sink([this](PreprocessedGroup&& g) { on_group(g); });

    create_subscriptions();
    create_services();

    pipeline_->start();
    RCLCPP_INFO(get_logger(), "meridian pipeline running (lidar: %s, imu: %s)",
                cfg_.sensors.lidar.topic.c_str(), cfg_.sensors.imu.topic.c_str());
  }

  ~OdometryNode() override {
    pipeline_->stop();
    set_log_sink(nullptr);
  }

 private:
  Timestamp now_ns() { return static_cast<Timestamp>(get_clock()->now().nanoseconds()); }

  void create_subscriptions() {
    const auto qos_cloud = rclcpp::SensorDataQoS().keep_last(5);
    const auto qos_imu = rclcpp::SensorDataQoS().keep_last(400);
    const auto qos_small = rclcpp::SensorDataQoS().keep_last(10);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        cfg_.sensors.lidar.topic, qos_cloud,
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
          RawLidarFrame f;
          if (!to_raw_lidar(*msg, now_ns(), &lidar_scratch_, &f)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "dropping scan: missing x/y/z or per-point time field");
            return;
          }
          pipeline_->ingest(f);
        });

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        cfg_.sensors.imu.topic, qos_imu,
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
          pipeline_->ingest(to_raw_imu(*msg, now_ns()));
        });

    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
        cfg_.sensors.camera.topic, qos_small,
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          auto f = to_raw_camera(*msg, now_ns());
          if (!f) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                                 "dropping image: unsupported encoding '%s'",
                                 msg->encoding.c_str());
            return;
          }
          pipeline_->ingest(*f);
        });

    if (cfg_.sensors.gnss.enable) {
      sub_gnss_ = create_subscription<sensor_msgs::msg::NavSatFix>(
          cfg_.sensors.gnss.topic, qos_small,
          [this](sensor_msgs::msg::NavSatFix::ConstSharedPtr msg) {
            pipeline_->ingest(to_raw_gnss(*msg, now_ns()));
          });
    }
  }

  void create_services() {
    srv_set_key_ = create_service<meridian_msgs::srv::SetDebugKey>(
        "/meridian/set_debug_key",
        [this](meridian_msgs::srv::SetDebugKey::Request::ConstSharedPtr req,
               meridian_msgs::srv::SetDebugKey::Response::SharedPtr res) {
          res->ok = sink_->set_key(req->key, req->enable, req->max_hz);
        });
    srv_set_rate_ = create_service<meridian_msgs::srv::SetTelemetryRate>(
        "/meridian/set_telemetry_rate",
        [this](meridian_msgs::srv::SetTelemetryRate::Request::ConstSharedPtr req,
               meridian_msgs::srv::SetTelemetryRate::Response::SharedPtr res) {
          sink_->set_default_rate(req->default_hz);
          res->ok = true;
        });
    srv_reset_timing_ = create_service<meridian_msgs::srv::ResetTiming>(
        "/meridian/reset_timing",
        [this](meridian_msgs::srv::ResetTiming::Request::ConstSharedPtr,
               meridian_msgs::srv::ResetTiming::Response::SharedPtr res) {
          sink_->reset_timing();
          res->ok = true;
        });
    srv_set_log_ = create_service<meridian_msgs::srv::SetLogLevel>(
        "/meridian/set_log_level",
        [this](meridian_msgs::srv::SetLogLevel::Request::ConstSharedPtr req,
               meridian_msgs::srv::SetLogLevel::Response::SharedPtr res) {
          if (req->level > 4) {
            res->ok = false;
            return;
          }
          log_sink_->set_level(req->module, static_cast<Level>(req->level));
          res->ok = true;
        });
  }

  void on_group(const PreprocessedGroup& g) {
    const auto n = ++group_count_;
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "group #%lu: %zu pts, %zu imu, image=%d, deskewed=%d",
        static_cast<unsigned long>(n), g.group.scan.points ? g.group.scan.points->size() : 0,
        g.group.imu.size(), g.group.image.has_value() ? 1 : 0,
        g.deskewed.has_value() ? 1 : 0);

    // This callback runs on the front-end stage thread, where live_state() is valid, so
    // the odom->body TF tracks the estimate at group rate. The front-end also pushes the
    // pose onto the "odom/body" telemetry key for /meridian/odom; TF is what rviz and
    // downstream nodes consume to place body-frame data in odom.
    publish_body_tf(pipeline_->live_state());
  }

  void publish_body_tf(const NavState& s) {
    const Pose& T = s.T_world_body;
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = to_ros(s.stamp);
    tf.header.frame_id = frame_name(s.ref_frame);  // "odom"
    tf.child_frame_id = "body";
    tf.transform.translation.x = T.t.x();
    tf.transform.translation.y = T.t.y();
    tf.transform.translation.z = T.t.z();
    tf.transform.rotation.w = T.q.w();
    tf.transform.rotation.x = T.q.x();
    tf.transform.rotation.y = T.q.y();
    tf.transform.rotation.z = T.q.z();
    tf_broadcaster_->sendTransform(tf);
  }

  Config cfg_;
  // Borrowed; owned by the pipeline. The pipeline is declared after this pointer and the
  // services, so on destruction the services are torn down first and no service callback
  // can dereference sink_ after the owning pipeline is gone.
  RosTelemetrySink* sink_ = nullptr;
  std::unique_ptr<RclcppLogSink> log_sink_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::unique_ptr<MeridianPipeline> pipeline_;

  // Reused conversion buffer for incoming scans; only ever touched by the single executor
  // thread that runs the subscription callbacks.
  std::vector<RawPoint> lidar_scratch_;
  std::atomic<std::uint64_t> group_count_{0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr sub_gnss_;

  rclcpp::Service<meridian_msgs::srv::SetDebugKey>::SharedPtr srv_set_key_;
  rclcpp::Service<meridian_msgs::srv::SetTelemetryRate>::SharedPtr srv_set_rate_;
  rclcpp::Service<meridian_msgs::srv::ResetTiming>::SharedPtr srv_reset_timing_;
  rclcpp::Service<meridian_msgs::srv::SetLogLevel>::SharedPtr srv_set_log_;
};

}  // namespace meridian

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int code = 0;
  try {
    auto node = std::make_shared<meridian::OdometryNode>();
    // A single-threaded executor keeps every subscription callback on one thread, so the
    // msg->core conversions (including the shared lidar_scratch_) run serialized and the
    // pipeline sees exactly one ingest thread.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& e) {
    RCLCPP_FATAL(rclcpp::get_logger("meridian_odometry"), "%s", e.what());
    code = 1;
  }
  rclcpp::shutdown();
  return code;
}
