#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "meridian/debug/file_sink.hpp"

namespace meridian {
namespace {

std::string read_file(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void expect_contains(const std::string& haystack, const std::string& needle) {
  EXPECT_NE(haystack.find(needle), std::string::npos) << "missing: " << needle;
}

TEST(FileSink, WritesEventsTelemetryAndStageFiles) {
  const auto dir = std::filesystem::temp_directory_path() / "meridian_test_file_sink";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string stem = (dir / "run").string();

  {
    auto sink = make_file_sink(stem);
    EXPECT_TRUE(sink->enabled("anything"));
    sink->scalar("frontend/iters", 7.0, 1500000000);
    Eigen::VectorXd v(3);
    v << 1.5, 2.5, 3.5;
    sink->vec("frontend/bias", v, 2000000000, "x,y,z");
    sink->event(Level::Warn, "frontend", "window restart", 3000000000);
    sink->timing("frontend/total", 1.5, 4000000000);
  }  // dtor writes the per-stage dump

  const auto events_path = dir / "run_events.txt";
  const auto telemetry_path = dir / "run_telemetry.txt";
  const auto stage_path = dir / "run_stage.txt";
  ASSERT_TRUE(std::filesystem::exists(events_path));
  ASSERT_TRUE(std::filesystem::exists(telemetry_path));
  ASSERT_TRUE(std::filesystem::exists(stage_path));

  const std::string events = read_file(events_path);
  expect_contains(events, "level: " + std::to_string(static_cast<int>(Level::Warn)));
  expect_contains(events, "tag: frontend");
  expect_contains(events, "message: window restart");
  expect_contains(events, "sec: 3");
  expect_contains(events, "---");

  const std::string telemetry = read_file(telemetry_path);
  expect_contains(telemetry, "key: frontend/iters");
  expect_contains(telemetry, "- 7");
  expect_contains(telemetry, "axis_order: ''");
  expect_contains(telemetry, "key: frontend/bias");
  expect_contains(telemetry, "- 1.5");
  expect_contains(telemetry, "- 2.5");
  expect_contains(telemetry, "- 3.5");
  expect_contains(telemetry, "axis_order: x,y,z");
  expect_contains(telemetry, "sec: 1");
  expect_contains(telemetry, "nanosec: 500000000");

  const std::string stage = read_file(stage_path);
  expect_contains(stage, "stage: frontend/total");
  expect_contains(stage, "ms: 1.5");
  expect_contains(stage, "ms_avg: 1.5");
  expect_contains(stage, "ms_max: 1.5");
  expect_contains(stage, "count: 1");
}

}  // namespace
}  // namespace meridian
