#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/debug/telemetry.hpp"

namespace meridian {
namespace {

// Captures the scalar/timing/event traffic it receives so a test can assert on it.
// `enabled` returns true so the core would build payloads against it.
class RecordingSink final : public TelemetrySink {
 public:
  struct ScalarRec {
    std::string key;
    double v;
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
    std::string msg;
    Timestamp t;
  };

  std::vector<ScalarRec> scalars;
  std::vector<TimingRec> timings;
  std::vector<EventRec> events;

  bool enabled(const char*) const override { return true; }

  void scalar(const char* key, double v, Timestamp t) override {
    scalars.push_back({key, v, t});
  }
  void vec(const char*, const Eigen::Ref<const Eigen::VectorXd>&, Timestamp,
           const char*) override {}

  void cloud(const char*, const PointCloudView&, Frame, Timestamp) override {}
  void pose(const char*, const Pose&, Frame, Timestamp) override {}
  void marker(const Marker&, Timestamp) override {}
  void image(const char*, const ImageOverlay&, Timestamp) override {}

  void timing(const char* stage, double ms, Timestamp t) override {
    timings.push_back({stage, ms, t});
  }

  void event(Level level, const char* tag, std::string_view msg, Timestamp t) override {
    events.push_back({level, tag, std::string(msg), t});
  }
};

TEST(NullSink, EnabledIsFalse) {
  NullSink sink;
  EXPECT_FALSE(sink.enabled("anything"));
}

TEST(RecordingSink, CapturesScalarTimingEvent) {
  RecordingSink sink;
  sink.scalar("frontend/iters", 7.0, 100);
  sink.timing("frontend/total", 1.5, 100);
  sink.event(Level::Warn, "frontend", "window restart", 100);

  ASSERT_EQ(sink.scalars.size(), 1u);
  EXPECT_EQ(sink.scalars[0].key, "frontend/iters");
  EXPECT_DOUBLE_EQ(sink.scalars[0].v, 7.0);

  ASSERT_EQ(sink.timings.size(), 1u);
  EXPECT_EQ(sink.timings[0].stage, "frontend/total");

  ASSERT_EQ(sink.events.size(), 1u);
  EXPECT_EQ(sink.events[0].level, Level::Warn);
  EXPECT_EQ(sink.events[0].msg, "window restart");
}

TEST(ScopedTimer, EmitsOneTimingOnScopeExit) {
  RecordingSink sink;
  {
    MERIDIAN_SCOPED_TIME(&sink, "scoped/stage", 42);
    EXPECT_TRUE(sink.timings.empty());
  }
  ASSERT_EQ(sink.timings.size(), 1u);
  EXPECT_EQ(sink.timings[0].stage, "scoped/stage");
  EXPECT_EQ(sink.timings[0].t, 42);
  EXPECT_GE(sink.timings[0].ms, 0.0);
}

TEST(ScopedTimer, NullSinkDoesNotEmit) {
  ScopedTimer t(nullptr, "noop", 0);
  // Destruction of a null-sink timer must not dereference anything.
}

}  // namespace
}  // namespace meridian
