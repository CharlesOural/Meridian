#pragma once

#include <cstddef>
#include <deque>
#include <string>

#include <Eigen/Core>

#include "meridian/debug/telemetry.hpp"

namespace meridian {

// Captures every telemetry call into bounded per-channel buffers so tests can assert
// on them and a forensic snapshot can dump the recent history. Heavy payloads (cloud,
// image, marker) are recorded as summaries, not deep copies. Thread-confined: record
// from one thread, read after the producer is quiescent.
class RecordingSink final : public TelemetrySink {
 public:
  // capacity = max records kept per channel; oldest are dropped first. 0 = unbounded.
  explicit RecordingSink(std::size_t capacity = 0) : capacity_(capacity) {}

  struct ScalarRec {
    std::string key;
    double v;
    Timestamp t;
  };
  struct VecRec {
    std::string key;
    Eigen::VectorXd v;
    std::string axis_order;
    Timestamp t;
  };
  struct CloudRec {
    std::string key;
    std::size_t n_points;
    Frame frame;
    Timestamp t;
  };
  struct PoseRec {
    std::string key;
    Pose pose;
    Frame frame;
    Timestamp t;
  };
  struct MarkerRec {
    std::string ns;
    std::int32_t id;
    Timestamp t;
  };
  struct ImageRec {
    std::string key;
    std::size_t n_patches;
    Timestamp t;
  };
  struct TimingRec {
    std::string stage;
    double ms;
    Timestamp t;
  };
  struct EventRec {
    Level level;
    std::string tag;
    std::string message;
    Timestamp t;
  };

  bool enabled(const char*) const override { return true; }

  void scalar(const char* key, double v, Timestamp t) override {
    push(scalars, ScalarRec{key, v, t});
  }
  void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>& v, Timestamp t,
           const char* axis_order) override {
    push(vecs, VecRec{key, v, axis_order ? axis_order : "", t});
  }
  void cloud(const char* key, const PointCloudView& v, Frame f, Timestamp t) override {
    push(clouds, CloudRec{key, v.points.size(), f, t});
  }
  void pose(const char* key, const Pose& p, Frame f, Timestamp t) override {
    push(poses, PoseRec{key, p, f, t});
  }
  void marker(const Marker& m, Timestamp t) override {
    push(markers, MarkerRec{m.ns, m.id, t});
  }
  void image(const char* key, const ImageOverlay& ov, Timestamp t) override {
    push(images, ImageRec{key, ov.patches.size(), t});
  }
  void timing(const char* stage, double ms, Timestamp t) override {
    push(timings, TimingRec{stage, ms, t});
  }
  void event(Level level, const char* tag, std::string_view msg, Timestamp t) override {
    push(events, EventRec{level, tag, std::string(msg), t});
  }

  void clear() {
    scalars.clear();
    vecs.clear();
    clouds.clear();
    poses.clear();
    markers.clear();
    images.clear();
    timings.clear();
    events.clear();
  }

  std::deque<ScalarRec> scalars;
  std::deque<VecRec> vecs;
  std::deque<CloudRec> clouds;
  std::deque<PoseRec> poses;
  std::deque<MarkerRec> markers;
  std::deque<ImageRec> images;
  std::deque<TimingRec> timings;
  std::deque<EventRec> events;

 private:
  template <typename C, typename R>
  void push(C& channel, R&& rec) {
    channel.push_back(std::forward<R>(rec));
    if (capacity_ > 0 && channel.size() > capacity_) channel.pop_front();
  }

  std::size_t capacity_;
};

}  // namespace meridian
