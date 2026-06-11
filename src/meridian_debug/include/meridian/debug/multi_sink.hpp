#pragma once

#include <vector>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

// Fans every telemetry call out to an ordered list of child sinks (e.g. a live ROS
// sink plus a black-box recorder). `enabled` is true if ANY child wants the key, so
// the core builds a payload at most once for all consumers. Children are borrowed,
// non-owning, and must outlive the MultiSink.
class MultiSink final : public TelemetrySink {
 public:
  void add(TelemetrySink* sink) {
    if (sink != nullptr) sinks_.push_back(sink);
  }

  bool enabled(const char* key) const override {
    for (auto* s : sinks_)
      if (s->enabled(key)) return true;
    return false;
  }

  bool wants_clouds() const override {
    for (auto* s : sinks_)
      if (s->wants_clouds()) return true;
    return false;
  }

  void scalar(const char* key, double v, Timestamp t) override {
    for (auto* s : sinks_) s->scalar(key, v, t);
  }
  void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, Timestamp t,
           const char* axis_order) override {
    for (auto* s : sinks_) s->vec(key, v, t, axis_order);
  }
  void cloud(const char* key, const PointCloudView& v, Frame f, Timestamp t) override {
    for (auto* s : sinks_) s->cloud(key, v, f, t);
  }
  void pose(const char* key, const Pose& p, Frame f, Timestamp t) override {
    for (auto* s : sinks_) s->pose(key, p, f, t);
  }
  void marker(const Marker& m, Timestamp t) override {
    for (auto* s : sinks_) s->marker(m, t);
  }
  void image(const char* key, const ImageOverlay& ov, Timestamp t) override {
    for (auto* s : sinks_) s->image(key, ov, t);
  }
  void timing(const char* stage, double ms, Timestamp t) override {
    for (auto* s : sinks_) s->timing(stage, ms, t);
  }
  void event(Level level, const char* tag, std::string_view msg, Timestamp t) override {
    for (auto* s : sinks_) s->event(level, tag, msg, t);
  }

 private:
  std::vector<TelemetrySink*> sinks_;
};

}  // namespace meridian
