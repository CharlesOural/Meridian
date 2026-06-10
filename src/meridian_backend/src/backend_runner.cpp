// Offline back-end iteration harness: streams a recorded packet log (keyframes,
// GNSS fixes, loop closures) straight into IBackEnd with the deterministic replay
// cadence -- one optimize() fold after every keyframe -- with no pipeline and no
// ROS. Two runs of the same config on the same log produce identical output, so
// an A/B difference is attributable to the back-end config alone.
//
// Usage:
//   backend_runner <config.yaml> <packets.bin> <out.tum>
//                  [--inject-loops <loops.yaml>] [--online-tum <file>]
//                  [--max-keyframes N]
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "meridian/backend/ibackend.hpp"
#include "meridian/calib/calibration_from_config.hpp"
#include "meridian/config/config_loader.hpp"
#include "meridian/debug/file_sink.hpp"
#include "meridian/debug/packet_log.hpp"
#include "meridian/debug/tum_writer.hpp"

namespace {

using namespace meridian;

int usage() {
  std::fprintf(stderr,
               "usage: backend_runner <config.yaml> <packets.bin> <out.tum>\n"
               "                      [--inject-loops <loops.yaml>] [--online-tum <file>]\n"
               "                      [--max-keyframes N]\n");
  return 2;
}

// Loop closures to inject at chosen points in the keyframe stream, keyed by the
// keyframe id whose arrival stages them (so both endpoints already exist in the graph).
struct InjectedLoops {
  std::multimap<std::uint64_t, LoopConstraint> by_kf;
  std::uint64_t total = 0;
  std::uint64_t outliers = 0;  // entries flagged for analysis; injected like any other
};

InjectedLoops parse_loops_yaml(const std::string& path) {
  InjectedLoops out;
  const YAML::Node root = YAML::LoadFile(path);
  const YAML::Node loops = root["loops"];
  if (!loops || !loops.IsSequence()) {
    throw std::runtime_error(path + ": expected a top-level 'loops' sequence");
  }
  for (const YAML::Node& n : loops) {
    LoopConstraint lc;
    lc.from_id = n["from_id"].as<std::uint64_t>();
    lc.to_id = n["to_id"].as<std::uint64_t>();
    const auto t = n["t"].as<std::vector<double>>();
    const auto q = n["q"].as<std::vector<double>>();  // [qx qy qz qw]
    if (t.size() != 3 || q.size() != 4) {
      throw std::runtime_error(path + ": 't' must have 3 entries and 'q' 4");
    }
    // Pose's constructor normalizes the quaternion.
    lc.T_from_to =
        Pose{Eigen::Quaterniond(q[3], q[0], q[1], q[2]), Eigen::Vector3d(t[0], t[1], t[2])};
    lc.cov.form = PoseCov6::Form::Covariance;
    if (n["cov"]) {
      const auto m = n["cov"].as<std::vector<double>>();
      if (m.size() != 36) {
        throw std::runtime_error(path + ": 'cov' must have 36 entries (row-major 6x6)");
      }
      lc.cov.M = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(m.data());
    } else if (n["cov_diag"]) {
      const auto d = n["cov_diag"].as<std::vector<double>>();  // translation-first variances
      if (d.size() != 6) {
        throw std::runtime_error(path + ": 'cov_diag' must have 6 entries");
      }
      lc.cov.M = Eigen::Map<const Eigen::Matrix<double, 6, 1>>(d.data()).asDiagonal();
    } else {
      throw std::runtime_error(path + ": each loop needs 'cov_diag' or 'cov'");
    }
    lc.fitness = n["fitness"] ? n["fitness"].as<double>() : 0.8;
    const std::uint64_t at_kf =
        n["at_kf"] ? n["at_kf"].as<std::uint64_t>() : std::max(lc.from_id, lc.to_id);
    ++out.total;
    if (n["outlier"] && n["outlier"].as<bool>()) {
      ++out.outliers;
    }
    out.by_kf.emplace(at_kf, lc);
  }
  return out;
}

unsigned long long ull(std::uint64_t v) {
  return static_cast<unsigned long long>(v);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    return usage();
  }
  const std::string config_path = argv[1];
  const std::string packets_path = argv[2];
  const std::string out_path = argv[3];

  std::string loops_path;
  std::string online_path;
  std::uint64_t max_keyframes = 0;  // 0 = unlimited
  for (int arg = 4; arg < argc; ++arg) {
    const std::string_view a = argv[arg];
    if (a == "--inject-loops" && arg + 1 < argc) {
      loops_path = argv[++arg];
    } else if (a == "--online-tum" && arg + 1 < argc) {
      online_path = argv[++arg];
    } else if (a == "--max-keyframes" && arg + 1 < argc) {
      try {
        max_keyframes = std::stoull(argv[++arg]);
      } catch (const std::exception&) {
        return usage();
      }
    } else {
      return usage();
    }
  }

  try {
    const Config cfg = load_config_yaml(config_path);
    auto calib = calibrationFromConfig(cfg.sensors);

    std::ofstream out(out_path);
    if (!out) {
      std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
      return 1;
    }
    std::ofstream online;
    if (!online_path.empty()) {
      online.open(online_path);
      if (!online) {
        std::fprintf(stderr, "error: cannot write %s\n", online_path.c_str());
        return 1;
      }
    }

    // Sidecar telemetry next to the TUM: <out>_events.txt / _telemetry.txt / _stage.txt.
    const std::string stem = out_path.size() > 4 && out_path.substr(out_path.size() - 4) == ".tum"
                                 ? out_path.substr(0, out_path.size() - 4)
                                 : out_path;
    auto sink = make_file_sink(stem);
    auto backend = makeBackEnd(cfg.backend, calib, sink.get(), /*deterministic=*/true);

    InjectedLoops inject;
    if (!loops_path.empty()) {
      inject = parse_loops_yaml(loops_path);
    }

    PacketLogReader reader(packets_path);
    std::uint64_t n_kf = 0;
    std::uint64_t n_gnss = 0;
    PacketRecord rec;
    while (reader.next(&rec)) {
      switch (rec.kind) {
        case PacketRecord::Kind::Keyframe: {
          const std::uint64_t id = rec.kf.id;
          backend->add_keyframe(std::move(rec.kf));
          backend->optimize();  // fold after every keyframe: the deterministic replay cadence
          if (online.is_open()) {
            const std::vector<StampedPose> traj = backend->corrected_trajectory();
            if (!traj.empty()) {
              write_tum_line(online, traj.back().stamp, traj.back().T_map_body);
            }
          }
          // Loops keyed to this keyframe stage now and fold at the next keyframe
          // (or at the trailing optimize() if the stream ends here).
          const auto [lo, hi] = inject.by_kf.equal_range(id);
          for (auto it = lo; it != hi; ++it) {
            backend->add_loop_constraint(it->second);
          }
          ++n_kf;
          break;
        }
        case PacketRecord::Kind::Gnss:
          backend->add_absolute(rec.fix, rec.nearest_kf_id);
          ++n_gnss;
          break;
        case PacketRecord::Kind::Loop:
          backend->add_loop_constraint(rec.loop);
          break;
      }
      if (max_keyframes > 0 && n_kf >= max_keyframes) {
        break;
      }
    }
    backend->optimize();  // folds anything staged after the last keyframe; no-op if empty

    for (const StampedPose& sp : backend->corrected_trajectory()) {
      write_tum_line(out, sp.stamp, sp.T_map_body);
    }
    out.flush();

    const BackEndDiagnostics d = backend->diagnostics();
    std::fprintf(stderr,
                 "%llu keyframes, %llu gnss, %llu loops injected (%llu marked outlier), "
                 "%llu admitted, %llu rejected, diverged=%s -> %s\n",
                 ull(n_kf), ull(n_gnss), ull(inject.total), ull(inject.outliers), ull(d.num_loops),
                 ull(d.num_loops_rejected), d.last_optimize_diverged ? "true" : "false",
                 out_path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
