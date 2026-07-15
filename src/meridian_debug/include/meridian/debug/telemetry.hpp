#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/frame.hpp"
#include "meridian/common/point.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

enum class Level { Trace, Debug, Info, Warn, Error };

// A drawable annotation for the operator surface. The meaning of `points` depends
// on `type` (e.g. LineList = pairs of endpoints, Points = vertices). A marker is
// keyed by (ns, id) so a re-published marker updates in place instead of stacking.
struct Marker {
  enum class Type { Points, LineList, LineStrip, Arrow, Cube, Sphere, Text, Hexagon, Triangles };
  Type type = Type::LineList;
  Frame frame = Frame::Map;
  std::string ns;
  std::int32_t id = 0;
  std::vector<Eigen::Vector3f> points;
  std::array<float, 4> color = {1, 1, 1, 1};   // rgba
  std::vector<std::array<float, 4>> colors;     // optional per-point rgba
  float scale = 0.1f;                           // line width / point size / shaft [m]
  std::string text;                             // used by Type::Text
  Duration lifetime_ns = 0;                     // 0 = persists until overwritten
};

// A borrowed image plus the tracked patches drawn on it. `base` is a non-owning
// view of the image bytes; the sink copies only if the key is enabled.
struct ImageOverlay {
  Frame frame = Frame::CamLink;
  int width = 0, height = 0;
  std::span<const std::uint8_t> base;
  enum class Encoding { Mono8, RGB8 } encoding = Encoding::Mono8;
  struct Patch {
    Eigen::Vector2f uv;   // pixel coordinate
    float residual;       // photometric residual
    float depth;          // LiDAR depth [m]
    int level;            // pyramid level
  };
  std::vector<Patch> patches;
};

// A non-owning view over a contiguous run of LiDAR points, handed to a sink so the
// core publishes a view rather than a copy.
struct PointCloudView {
  std::span<const LidarPoint> points;
};

// Sink for all introspection traffic. The core holds one of these; binding a
// NullSink makes every call a no-op. `enabled` is the cheap gate that lets the
// core skip building an expensive payload when a key is currently off.
class TelemetrySink {
 public:
  virtual ~TelemetrySink() = default;

  virtual bool enabled(const char* key) const = 0;

  virtual void scalar(const char* key, double v, Timestamp t) = 0;
  virtual void vec(const char* key, const Eigen::Ref<const Eigen::VectorXd>&, Timestamp t,
                   const char* axis_order = nullptr) = 0;

  virtual void cloud(const char* key, const PointCloudView&, Frame, Timestamp) = 0;
  virtual void pose(const char* key, const Pose&, Frame, Timestamp) = 0;
  virtual void marker(const Marker&, Timestamp) = 0;
  virtual void image(const char* key, const ImageOverlay&, Timestamp) = 0;

  virtual void timing(const char* stage, double ms, Timestamp t) = 0;

  virtual void event(Level, const char* tag, std::string_view msg, Timestamp t) = 0;

  // Whether any consumer will actually render a cloud payload. Lets the core skip building
  // an expensive assembled cloud (the map cloud) for sinks that drop clouds (file/null).
  // Defaults to true; sinks whose cloud() is a no-op override it to false. Appended last so
  // adding it does not shift the existing vtable slots.
  virtual bool wants_clouds() const { return true; }
};

using Clock = std::chrono::steady_clock;

// Elapsed wall time from `start` to now, in milliseconds.
inline double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Times its enclosing scope and reports the duration to a sink on destruction.
// A null sink makes both ctor and dtor effectively free.
class ScopedTimer {
 public:
  ScopedTimer(TelemetrySink* s, const char* stage, Timestamp t)
      : s_(s), stage_(stage), t_(t), start_(Clock::now()) {}
  ~ScopedTimer() {
    if (s_) s_->timing(stage_, ms_since(start_), t_);
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  TelemetrySink* s_;
  const char* stage_;
  Timestamp t_;
  Clock::time_point start_;
};

#define MERIDIAN_CONCAT_(a, b) a##b
#define MERIDIAN_CONCAT(a, b) MERIDIAN_CONCAT_(a, b)
#define MERIDIAN_SCOPED_TIME(sink, stage, t) \
  ::meridian::ScopedTimer MERIDIAN_CONCAT(_ht_, __LINE__)((sink), (stage), (t))

// The default sink: every call is an empty body and `enabled` is always false, so
// the core skips payload construction entirely.
class NullSink final : public TelemetrySink {
 public:
  bool enabled(const char*) const override { return false; }
  bool wants_clouds() const override { return false; }

  void scalar(const char*, double, Timestamp) override {}
  void vec(const char*, const Eigen::Ref<const Eigen::VectorXd>&, Timestamp,
           const char*) override {}

  void cloud(const char*, const PointCloudView&, Frame, Timestamp) override {}
  void pose(const char*, const Pose&, Frame, Timestamp) override {}
  void marker(const Marker&, Timestamp) override {}
  void image(const char*, const ImageOverlay&, Timestamp) override {}

  void timing(const char*, double, Timestamp) override {}

  void event(Level, const char*, std::string_view, Timestamp) override {}
};

}  // namespace meridian
