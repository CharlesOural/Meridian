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
// --realtime drives the pipeline in Live mode instead: the front-end, back-end,
// and loop detector run on their own threads, and bag ingest is paced to
// bag-clock time. This is NOT bit-deterministic (thread scheduling), but it is
// what you want for watching loop closure: the detector's work no longer
// serializes onto the front-end thread, so it cannot perturb the front-end's
// solve timing -- the divergence that synchronous replay suffers when the
// detector is enabled. Use sync replay for the bit-exact baseline, --realtime
// to view the live behaviour.
//
// Usage:
//   replay_runner <config.yaml> <bag_dir> <out.tum> [max_content_secs]
//                 [--dump-keyframes <packets.bin>] [--dump-clouds] [--no-backend]
//                 [--viz] [--realtime [speed]]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "conversions/ros2core.hpp"
#include "debug/ros_telemetry_sink.hpp"
#include "meridian/config/config_loader.hpp"
#include "meridian/debug/file_sink.hpp"
#include "meridian/debug/multi_sink.hpp"
#include "meridian/debug/packet_log.hpp"
#include "meridian/debug/tum_writer.hpp"
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

int usage() {
  std::fprintf(stderr,
               "usage: replay_runner <config.yaml> <bag_dir> <out.tum> [max_content_secs]\n"
               "                     [--dump-keyframes <packets.bin>] [--dump-clouds]\n"
               "                     [--no-backend] [--viz] [--realtime [speed]]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace meridian;
  if (argc < 4) {
    return usage();
  }
  const std::string config_path = argv[1];
  const std::string bag_path = argv[2];
  const std::string out_path = argv[3];

  // argv[4] is max_content_secs iff present and not an option; everything after is flags.
  double max_secs = 0.0;
  std::string dump_path;
  bool dump_clouds = false;
  bool no_backend = false;
  bool viz = false;
  bool realtime = false;
  double speed = 1.0;
  int arg = 4;
  if (arg < argc && std::string_view(argv[arg]).substr(0, 2) != "--") {
    max_secs = std::stod(argv[arg++]);
  }
  for (; arg < argc; ++arg) {
    const std::string_view a = argv[arg];
    if (a == "--dump-keyframes" && arg + 1 < argc) {
      dump_path = argv[++arg];
    } else if (a == "--dump-clouds") {
      dump_clouds = true;
    } else if (a == "--no-backend") {
      no_backend = true;
    } else if (a == "--viz") {
      viz = true;
    } else if (a == "--realtime") {
      realtime = true;
      // Optional playback-speed multiplier follows the flag (e.g. "--realtime 3" = 3x).
      if (arg + 1 < argc && std::string_view(argv[arg + 1]).substr(0, 2) != "--") {
        speed = std::stod(argv[++arg]);
      }
    } else {
      return usage();
    }
  }
  if (speed <= 0.0) {
    std::fprintf(stderr, "error: --realtime speed must be > 0\n");
    return usage();
  }
  if (dump_clouds && dump_path.empty()) {
    std::fprintf(stderr, "error: --dump-clouds requires --dump-keyframes\n");
    return usage();
  }

  Config cfg = load_config_yaml(config_path);
  // Replay: one thread, synchronous, bit-deterministic. --realtime: Live mode, so the
  // back-end and loop detector run off the front-end thread (paced to bag time below).
  cfg.pipeline.mode = realtime ? PipelineMode::Live : PipelineMode::Replay;
  if (no_backend) {
    cfg.backend.enable = false;  // A/B switch: same bag, front-end only
  }

  std::ofstream out(out_path);
  if (!out) {
    std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
    return 1;
  }

  // Sidecar telemetry next to the TUM: <out>_events.txt / _telemetry.txt / _stage.txt.
  const std::string stem = out_path.size() > 4 && out_path.substr(out_path.size() - 4) == ".tum"
                               ? out_path.substr(0, out_path.size() - 4)
                               : out_path;
  // --viz wraps the file sink in a MultiSink that also publishes ROS topics, so a replay can
  // drive RViz/Foxglove while still writing the TUM + sidecars. The sink is observe-only, so it
  // never changes the estimate; the borrowed children must outlive the pipeline that holds the
  // MultiSink, so they are declared before it. (A --viz replay builds extra payloads and is for
  // inspection, not the bit-exact baseline — run without --viz for byte-identical output.)
  std::shared_ptr<rclcpp::Node> viz_node;
  std::unique_ptr<meridian::RosTelemetrySink> ros_sink;
  std::unique_ptr<meridian::TelemetrySink> file_sink;
  std::unique_ptr<meridian::TelemetrySink> sink;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_bc;
  if (viz) {
    if (!rclcpp::ok()) rclcpp::init(0, nullptr);
    viz_node = std::make_shared<rclcpp::Node>("meridian_replay");
    ros_sink = std::make_unique<meridian::RosTelemetrySink>(viz_node.get(), cfg.debug);
    file_sink = make_file_sink(stem);
    auto multi = std::make_unique<meridian::MultiSink>();
    multi->add(ros_sink.get());
    multi->add(file_sink.get());
    sink = std::move(multi);
    // TF tree map->odom->body so RViz/Foxglove place the graph and map cloud (map frame),
    // the registered clouds (odom), and the body scan (body) in one view. map->odom is the
    // back-end correction T_map_odom: identity until the first loop folds, then it jumps —
    // so the whole odom subtree (and the live clouds in it) visibly snaps onto the corrected
    // map graph. Both edges are broadcast dynamically from the replay thread (below).
    tf_bc = std::make_unique<tf2_ros::TransformBroadcaster>(*viz_node);
    std::fprintf(stderr, "  --viz: publishing /meridian/* topics + TF from node 'meridian_replay'\n");
  } else {
    sink = make_file_sink(stem);
  }
  MeridianPipeline pipeline(cfg, std::move(sink));
  // T_map_odom is written by the back-end thread; the group sink (front-end thread under
  // --realtime) reads it for the map->odom TF. The graph-update sink runs on the back-end
  // thread right after the fold, so it snapshots the correction under a lock for the reader.
  struct {
    std::mutex mu;
    meridian::Pose pose;
  } map_odom_shared;
  pipeline.set_graph_update_sink([&](const GraphUpdate&) {
    const meridian::Pose mo = pipeline.map_odom();
    const std::lock_guard<std::mutex> lock(map_odom_shared.mu);
    map_odom_shared.pose = mo;
  });
  std::uint64_t groups = 0;
  pipeline.set_group_sink([&](PreprocessedGroup&&) {
    // Sync replay runs this on the caller thread; --realtime runs it on the front-end
    // thread. Either way live_state() is the front-end's own, valid here.
    const NavState s = pipeline.live_state();
    if (s.stamp <= 0) {
      return;  // pre-init groups carry a zero state; a t=0 pose would poison analysis
    }
    write_tum_line(out, s.stamp, s.T_world_body);
    if (tf_bc) {
      const rclcpp::Time stamp(s.stamp);
      auto fill = [](geometry_msgs::msg::TransformStamped& tf, const meridian::Pose& T) {
        tf.transform.translation.x = T.t.x();
        tf.transform.translation.y = T.t.y();
        tf.transform.translation.z = T.t.z();
        tf.transform.rotation.w = T.q.w();
        tf.transform.rotation.x = T.q.x();
        tf.transform.rotation.y = T.q.y();
        tf.transform.rotation.z = T.q.z();
      };
      // map->odom: the back-end correction, jumping when a loop folds.
      meridian::Pose map_odom;
      {
        const std::lock_guard<std::mutex> lock(map_odom_shared.mu);
        map_odom = map_odom_shared.pose;
      }
      geometry_msgs::msg::TransformStamped mo;
      mo.header.stamp = stamp;
      mo.header.frame_id = "map";
      mo.child_frame_id = "odom";
      fill(mo, map_odom);
      // odom->body: the live front-end estimate.
      geometry_msgs::msg::TransformStamped ob;
      ob.header.stamp = stamp;
      ob.header.frame_id = "odom";
      ob.child_frame_id = "body";
      fill(ob, s.T_world_body);
      tf_bc->sendTransform(mo);
      tf_bc->sendTransform(ob);
    }
    if (++groups % 250 == 0) {
      std::fprintf(stderr, "  %llu groups\n", static_cast<unsigned long long>(groups));
      out.flush();
    }
  });
  // Optional back-end input dump: every keyframe / anchored GNSS fix / loop closure
  // the back-end would consume, recorded in feed order for offline replay.
  std::unique_ptr<PacketLogWriter> packet_log;
  std::uint64_t kf_dumped = 0;
  if (!dump_path.empty()) {
    packet_log = std::make_unique<PacketLogWriter>(dump_path, dump_clouds);
    pipeline.set_backend_tap([&](const MeridianPipeline::BackendItem& item) {
      std::visit(
          [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, KeyframePacket>) {
              packet_log->write_keyframe(v);
              ++kf_dumped;
            } else if constexpr (std::is_same_v<T, MeridianPipeline::GnssForBackend>) {
              packet_log->write_gnss(v.fix, v.nearest_kf_id);
            } else {  // LoopConstraint
              packet_log->write_loop(v);
            }
          },
          item);
    });
  }
  if (realtime) {
    std::fprintf(stderr, "  --realtime: Live mode, threaded, paced to bag clock at %.2gx\n", speed);
    if (speed > 1.0) {
      std::fprintf(stderr,
                   "  WARNING: --realtime %.2gx is faster than real time; the front-end's\n"
                   "  wall-clock solver budget can be starved by the back-end/detector/map-cloud\n"
                   "  work on the other threads, which shows up as drift or divergence. Use 1x\n"
                   "  for a faithful run.\n",
                   speed);
    }
  }
  pipeline.start();  // no-op in replay; spawns the front-end/back-end threads in Live mode

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
  std::chrono::steady_clock::time_point wall_first;
  std::uint64_t n_msgs = 0;
  while (reader.has_next()) {
    auto bag_msg = reader.read_next();
    const auto recv_ns = static_cast<Timestamp>(bag_msg->time_stamp);
    if (t_first < 0) {
      t_first = bag_msg->time_stamp;
      wall_first = std::chrono::steady_clock::now();
    }
    if (max_secs > 0.0 && (bag_msg->time_stamp - t_first) * 1e-9 > max_secs) {
      break;
    }
    // --realtime: hold each message until its bag-clock offset (scaled by speed) has
    // elapsed in wall time, so the Live-mode threads see traffic at the real rate and the
    // lossy live queues are not overrun by a CPU-speed firehose.
    if (realtime) {
      const double bag_dt_s = static_cast<double>(bag_msg->time_stamp - t_first) * 1e-9 / speed;
      const auto target = wall_first + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                           std::chrono::duration<double>(bag_dt_s));
      std::this_thread::sleep_until(target);
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
  packet_log.reset();  // flush + close the dump before reporting it
  // The back-end's corrected map-frame trajectory next to the front-end TUM, one line
  // per keyframe, for direct ATE comparison of the two estimates.
  std::uint64_t kf_corrected = 0;
  if (pipeline.backend_enabled()) {
    const std::string backend_path = stem + "_backend.tum";
    std::ofstream backend_out(backend_path);
    if (!backend_out) {
      std::fprintf(stderr, "error: cannot write %s\n", backend_path.c_str());
      return 1;
    }
    for (const StampedPose& sp : pipeline.corrected_trajectory()) {
      write_tum_line(backend_out, sp.stamp, sp.T_map_body);
      ++kf_corrected;
    }
  }
  std::fprintf(stderr, "replay done: %llu messages, %llu groups, %llu corrected keyframes -> %s\n",
               static_cast<unsigned long long>(n_msgs), static_cast<unsigned long long>(groups),
               static_cast<unsigned long long>(kf_corrected), out_path.c_str());
  if (!dump_path.empty()) {
    std::fprintf(stderr, "  dumped %llu keyframes -> %s\n",
                 static_cast<unsigned long long>(kf_dumped), dump_path.c_str());
  }
  if (viz && rclcpp::ok()) rclcpp::shutdown();
  return 0;
}
