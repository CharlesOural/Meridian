#include "meridian/debug/file_sink.hpp"

#include <algorithm>
#include <utility>

namespace meridian {

FileSink::FileSink(const std::string& events_path, const std::string& telemetry_path,
                   std::string stage_path)
    : ev_(events_path), tm_(telemetry_path), stage_path_(std::move(stage_path)) {}

FileSink::~FileSink() {
  std::ofstream st(stage_path_);
  for (const auto& [stage, s] : stages_) {
    st << "stage: " << stage << "\nms: " << s.last << "\nms_avg: " << (s.sum / s.n)
       << "\nms_max: " << s.maxv << "\ncount: " << s.n << "\n---\n";
  }
}

bool FileSink::enabled(const char*) const { return true; }

void FileSink::scalar(const char* key, double v, Timestamp t) {
  stamp(tm_, t);
  tm_ << "key: " << key << "\nvalues:\n- " << v << "\naxis_order: ''\n---\n";
}

void FileSink::vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, Timestamp t,
                   const char* axis_order) {
  stamp(tm_, t);
  tm_ << "key: " << key << "\nvalues:\n";
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    tm_ << "- " << v(i) << '\n';
  }
  tm_ << "axis_order: " << (axis_order ? axis_order : "''") << "\n---\n";
}

void FileSink::cloud(const char*, const PointCloudView&, Frame, Timestamp) {}
void FileSink::pose(const char*, const Pose&, Frame, Timestamp) {}
void FileSink::marker(const Marker&, Timestamp) {}
void FileSink::image(const char*, const ImageOverlay&, Timestamp) {}

void FileSink::timing(const char* stage, double ms, Timestamp) {
  auto& s = stages_[stage];
  s.sum += ms;
  s.maxv = std::max(s.maxv, ms);
  s.last = ms;
  ++s.n;
}

void FileSink::event(Level lvl, const char* tag, std::string_view msg, Timestamp t) {
  stamp(ev_, t);
  ev_ << "level: " << static_cast<int>(lvl) << "\ntag: " << tag << "\nmessage: " << msg
      << "\n---\n";
}

void FileSink::stamp(std::ofstream& o, Timestamp t) {
  const auto tt = t > 0 ? t : 0;
  o << "stamp:\n  sec: " << tt / 1000000000LL << "\n  nanosec: " << tt % 1000000000LL << '\n';
}

std::unique_ptr<FileSink> make_file_sink(const std::string& stem) {
  return std::make_unique<FileSink>(stem + "_events.txt", stem + "_telemetry.txt",
                                    stem + "_stage.txt");
}

}  // namespace meridian
