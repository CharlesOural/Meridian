// Deterministic offline bag replay: reads a rosbag2 directly and drives the
// pipeline in Replay mode (synchronous ingest on this thread, lossless by
// construction, solver deadline disabled), writing the trajectory as TUM.
//
// This removes every live-path source of nondeterminism -- transport loss,
// queue overflow, thread scheduling, and the wall-clock solver budget -- so
// two runs of the same config on the same bag produce identical output, and
// an A/B difference is attributable to the config alone. It also runs at CPU
// speed instead of bag-clock speed.
//
// Usage:
//   replay_runner <config.yaml> <bag_dir> <out.tum> [max_content_secs]
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "conversions/ros2core.hpp"
#include "meridian/config/config_loader.hpp"
#include "meridian/debug/telemetry.hpp"
#include "meridian/pipeline/meridian_pipeline.hpp"

namespace {

template <typename Msg>
Msg deserialize(const rosbag2_storage::SerializedBagMessageSharedPtr& bag_msg) {
  Msg msg;
  rclcpp::SerializedMessage ser(*bag_msg->serialized_data);
  rclcpp::Serialization<Msg> serializer;
  serializer.deserialize_message(&ser, &msg);
  return msg;
}

// File-backed telemetry: events, scalars/vecs, and per-stage timing land in text
// files in the same block layout `ros2 topic echo` produces, so the offline
// analyzers parse a replay exactly like a live capture. Heavy payloads
// (clouds/markers/images) are dropped.
class FileSink final : public meridian::TelemetrySink {
public:
  FileSink(const std::string& events_path, const std::string& telemetry_path,
           std::string stage_path)
      : ev_(events_path), tm_(telemetry_path), stage_path_(std::move(stage_path)) {}

  ~FileSink() override {
    std::ofstream st(stage_path_);
    for (const auto& [stage, s] : stages_) {
      st << "stage: " << stage << "\nms: " << s.last << "\nms_avg: " << (s.sum / s.n)
         << "\nms_max: " << s.maxv << "\ncount: " << s.n << "\n---\n";
    }
  }

  bool enabled(const char*) const override { return true; }

  void scalar(const char* key, double v, meridian::Timestamp t) override {
    stamp(tm_, t);
    tm_ << "key: " << key << "\nvalues:\n- " << v << "\naxis_order: ''\n---\n";
  }

  void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, meridian::Timestamp t,
           const char* axis_order) override {
    stamp(tm_, t);
    tm_ << "key: " << key << "\nvalues:\n";
    for (Eigen::Index i = 0; i < v.size(); ++i) {
      tm_ << "- " << v(i) << '\n';
    }
    tm_ << "axis_order: " << (axis_order ? axis_order : "''") << "\n---\n";
  }

  void cloud(const char*, const meridian::PointCloudView&, meridian::Frame,
             meridian::Timestamp) override {}
  void pose(const char*, const meridian::Pose&, meridian::Frame, meridian::Timestamp) override {}
  void marker(const meridian::Marker&, meridian::Timestamp) override {}
  void image(const char*, const meridian::ImageOverlay&, meridian::Timestamp) override {}

  void timing(const char* stage, double ms, meridian::Timestamp) override {
    auto& s = stages_[stage];
    s.sum += ms;
    s.maxv = std::max(s.maxv, ms);
    s.last = ms;
    ++s.n;
  }

  void event(meridian::Level lvl, const char* tag, std::string_view msg,
             meridian::Timestamp t) override {
    stamp(ev_, t);
    ev_ << "level: " << static_cast<int>(lvl) << "\ntag: " << tag << "\nmessage: " << msg
        << "\n---\n";
  }

private:
  struct Stat {
    double sum = 0, maxv = 0, last = 0;
    std::uint64_t n = 0;
  };
  static void stamp(std::ofstream& o, meridian::Timestamp t) {
    const auto tt = t > 0 ? t : 0;
    o << "stamp:\n  sec: " << tt / 1000000000LL << "\n  nanosec: " << tt % 1000000000LL << '\n';
  }
  std::ofstream ev_, tm_;
  std::string stage_path_;
  std::map<std::string, Stat> stages_;
};

}  // namespace

int main(int argc, char** argv) {
  using namespace meridian;
  if (argc < 4 || argc > 5) {
    std::fprintf(stderr,
                 "usage: replay_runner <config.yaml> <bag_dir> <out.tum> [max_content_secs]\n");
    return 2;
  }
  const std::string config_path = argv[1];
  const std::string bag_path = argv[2];
  const std::string out_path = argv[3];
  const double max_secs = argc == 5 ? std::stod(argv[4]) : 0.0;

  Config cfg = load_config_yaml(config_path);
  cfg.pipeline.mode = PipelineMode::Replay;  // synchronous + deterministic, regardless of file

  std::ofstream out(out_path);
  if (!out) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    return 1;
  }

  // Sidecar telemetry next to the TUM: <out>_events.txt / _telemetry.txt / _stage.txt.
  const std::string stem = out_path.size() > 4 && out_path.substr(out_path.size() - 4) == ".tum"
                               ? out_path.substr(0, out_path.size() - 4)
                               : out_path;
  MeridianPipeline pipeline(
      cfg, std::make_unique<FileSink>(stem + "_events.txt", stem + "_telemetry.txt",
                                      stem + "_stage.txt"));
  std::uint64_t groups = 0;
  pipeline.set_group_sink([&](PreprocessedGroup&&) {
    // Replay runs the group sink on this thread, where live_state() is valid.
    const NavState s = pipeline.live_state();
    if (s.stamp <= 0) {
      return;  // pre-init groups carry a zero state; a t=0 pose would poison analysis
    }
    const Pose& T = s.T_world_body;
    out << std::fixed;
    out.precision(9);
    out << s.stamp * 1e-9 << ' ';
    out.precision(6);
    out << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' ';
    out.precision(9);
    out << T.q.x() << ' ' << T.q.y() << ' ' << T.q.z() << ' ' << T.q.w() << '\n';
    if (++groups % 250 == 0) {
      std::fprintf(stderr, "  %llu groups\n", static_cast<unsigned long long>(groups));
      out.flush();
    }
  });
  pipeline.start();  // no-op in replay; keeps the call sequence identical to live

  rosbag2_cpp::readers::SequentialReader reader;
  reader.open({bag_path, "sqlite3"}, {"cdr", "cdr"});
  rosbag2_storage::StorageFilter filter;
  filter.topics = {cfg.sensors.lidar.topic, cfg.sensors.imu.topic, cfg.sensors.camera.topic};
  if (cfg.sensors.gnss.enable) {
    filter.topics.push_back(cfg.sensors.gnss.topic);
  }
  reader.set_filter(filter);

  std::vector<RawPoint> scratch;
  std::int64_t t_first = -1;
  std::uint64_t n_msgs = 0;
  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    const auto recv_ns = static_cast<Timestamp>(bag_msg->time_stamp);
    if (t_first < 0) {
      t_first = bag_msg->time_stamp;
    }
    if (max_secs > 0.0 && (bag_msg->time_stamp - t_first) * 1e-9 > max_secs) {
      break;
    }
    ++n_msgs;
    const std::string& topic = bag_msg->topic_name;
    if (topic == cfg.sensors.lidar.topic) {
      const auto msg = deserialize<sensor_msgs::msg::PointCloud2>(bag_msg);
      RawLidarFrame f;
      if (to_raw_lidar(msg, recv_ns, &scratch, &f)) {
        pipeline.ingest(f);
      }
    } else if (topic == cfg.sensors.imu.topic) {
      pipeline.ingest(to_raw_imu(deserialize<sensor_msgs::msg::Imu>(bag_msg), recv_ns));
    } else if (topic == cfg.sensors.camera.topic) {
      std::optional<RawCameraFrame> f;
      if (cfg.sensors.camera.compressed) {
        f = to_raw_camera_compressed(deserialize<sensor_msgs::msg::CompressedImage>(bag_msg),
                                     recv_ns);
      } else {
        f = to_raw_camera(deserialize<sensor_msgs::msg::Image>(bag_msg), recv_ns);
      }
      if (f) {
        pipeline.ingest(*f);
      }
    } else if (cfg.sensors.gnss.enable && topic == cfg.sensors.gnss.topic) {
      pipeline.ingest(to_raw_gnss(deserialize<sensor_msgs::msg::NavSatFix>(bag_msg), recv_ns));
    }
  }
  pipeline.stop();
  out.flush();
  std::fprintf(stderr, "replay done: %llu messages, %llu groups -> %s\n",
               static_cast<unsigned long long>(n_msgs), static_cast<unsigned long long>(groups),
               out_path.c_str());
  return 0;
}
