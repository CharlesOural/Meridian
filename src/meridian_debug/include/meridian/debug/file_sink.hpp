#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include <Eigen/Core>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

// File-backed telemetry: events, scalars/vecs, and per-stage timing land in text
// files in the same block layout `ros2 topic echo` produces, so the offline
// analyzers parse a replay exactly like a live capture. Heavy payloads
// (clouds/markers/images) are dropped. Per-stage timing is aggregated in memory
// and dumped once, on destruction.
class FileSink final : public TelemetrySink {
 public:
  FileSink(const std::string& events_path, const std::string& telemetry_path,
           std::string stage_path);
  ~FileSink() override;

  bool enabled(const char*) const override;

  void scalar(const char* key, double v, Timestamp t) override;
  void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, Timestamp t,
           const char* axis_order) override;

  void cloud(const char*, const PointCloudView&, Frame, Timestamp) override;
  void pose(const char*, const Pose&, Frame, Timestamp) override;
  void marker(const Marker&, Timestamp) override;
  void image(const char*, const ImageOverlay&, Timestamp) override;

  void timing(const char* stage, double ms, Timestamp) override;

  void event(Level lvl, const char* tag, std::string_view msg, Timestamp t) override;

 private:
  struct Stat {
    double sum = 0, maxv = 0, last = 0;
    std::uint64_t n = 0;
  };
  static void stamp(std::ofstream& o, Timestamp t);
  std::ofstream ev_, tm_;
  std::string stage_path_;
  std::map<std::string, Stat> stages_;
};

// Opens <stem>_events.txt, <stem>_telemetry.txt, <stem>_stage.txt.
std::unique_ptr<FileSink> make_file_sink(const std::string& stem);

}  // namespace meridian
