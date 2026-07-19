#include <gtest/gtest.h>
#include <rmw/types.h>

#include <cstdint>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>

#include "meridian/ros/pipeline_diagnostics.hpp"

namespace meridian::ros {
namespace {

core::PipelineStage stage() {
  const auto result = core::makePipelineStage(core::PipelineStageId{7U}, "lidar.registration");
  if (!result) {
    throw std::runtime_error("test stage is invalid");
  }
  return *result;
}

std::map<std::string, std::string> fields(const diagnostic_msgs::msg::DiagnosticStatus& status) {
  std::map<std::string, std::string> output;
  for (const auto& value : status.values) {
    output.emplace(value.key, value.value);
  }
  return output;
}

TEST(PipelineDiagnosticsQos, IsBoundedReliableAndVolatile) {
  const auto profile = pipelineTimingQos(17U).get_rmw_qos_profile();
  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 17U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
  EXPECT_THROW(static_cast<void>(pipelineTimingQos(0U)), std::invalid_argument);
}

TEST(PipelineDiagnosticsConversion, PublishesTypedDurationsIdentitiesDispositionAndQueueSnapshot) {
  core::PipelineTimingSample sample;
  sample.stage = stage();
  sample.wall_duration = core::Duration{12'345};
  sample.thread_cpu_duration = core::Duration{10'000};
  sample.work.measurement = core::MeasurementId{9U};
  sample.work.state = core::StateId{10U};
  sample.work.local_revision = core::LocalGraphRevision{11U};
  sample.work.global_revision = core::PipelineGlobalRevision{12U};
  sample.disposition = core::PipelineDisposition::Dropped;
  sample.detail = *core::PipelineDispositionDetail::make("deadline_queue_overflow");
  core::PipelineQueueSnapshot queue;
  queue.name = *core::PipelineQueueName::make("lidar.input");
  queue.count = 3U;
  queue.bytes = 4'096U;
  queue.count_capacity = 8U;
  queue.byte_capacity = 16'384U;
  queue.oldest_age = core::Duration{25'000'000};
  queue.accepted = 100U;
  queue.rejected = 2U;
  queue.dropped_oldest = 3U;
  queue.dropped_newest = 4U;
  queue.skipped_stale = 5U;
  queue.skipped_policy = 6U;
  sample.queue = queue;
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 42;
  stamp.nanosec = 123U;

  const auto message = toDiagnosticArray(sample, stamp);

  EXPECT_EQ(message.header.stamp, stamp);
  ASSERT_EQ(message.status.size(), 1U);
  const auto& status = message.status.front();
  EXPECT_EQ(status.name, "meridian/local/pipeline/lidar.registration");
  EXPECT_EQ(status.level, diagnostic_msgs::msg::DiagnosticStatus::WARN);
  EXPECT_EQ(status.message, "dropped");
  const auto values = fields(status);
  EXPECT_EQ(values.at("record_type"), "sample");
  EXPECT_EQ(values.at("stage_id"), "7");
  EXPECT_EQ(values.at("wall_duration_ns"), "12345");
  EXPECT_EQ(values.at("thread_cpu_duration_ns"), "10000");
  EXPECT_EQ(values.at("measurement_id"), "9");
  EXPECT_EQ(values.at("state_id"), "10");
  EXPECT_EQ(values.at("local_graph_revision"), "11");
  EXPECT_EQ(values.at("global_graph_revision"), "12");
  EXPECT_EQ(values.at("disposition_detail"), "deadline_queue_overflow");
  EXPECT_EQ(values.at("queue.count"), "3");
  EXPECT_EQ(values.at("queue.oldest_age_ns"), "25000000");
  EXPECT_EQ(values.at("queue.dropped_newest"), "4");
  EXPECT_EQ(values.at("queue.skipped_stale"), "5");
}

TEST(PipelineDiagnosticsConversion, PublishesRollingWallCpuPercentilesAndDispositionCounts) {
  core::BoundedPipelineTimingAccumulator accumulator(stage(), 4U);
  for (std::int64_t index = 1; index <= 4; ++index) {
    core::PipelineTimingSample sample;
    sample.stage = stage();
    sample.wall_duration = core::Duration{index * 1'000};
    sample.thread_cpu_duration = core::Duration{index * 800};
    sample.disposition =
        index == 4 ? core::PipelineDisposition::Rejected : core::PipelineDisposition::Completed;
    ASSERT_EQ(accumulator.observe(sample), core::PipelineTimingObserveStatus::Accepted);
  }

  const auto message = toDiagnosticArray(accumulator.snapshot(), builtin_interfaces::msg::Time{});

  ASSERT_EQ(message.status.size(), 1U);
  const auto& status = message.status.front();
  EXPECT_EQ(status.level, diagnostic_msgs::msg::DiagnosticStatus::WARN);
  EXPECT_EQ(status.message, "rolling_statistics");
  const auto values = fields(status);
  EXPECT_EQ(values.at("record_type"), "statistics");
  EXPECT_EQ(values.at("wall.p50_ns"), "2000");
  EXPECT_EQ(values.at("wall.p95_ns"), "4000");
  EXPECT_EQ(values.at("wall.p99_ns"), "4000");
  EXPECT_EQ(values.at("thread_cpu.max_ns"), "3200");
  EXPECT_EQ(values.at("disposition.completed"), "3");
  EXPECT_EQ(values.at("disposition.rejected"), "1");
}

TEST(PipelineDiagnosticsConversion, MarksInvalidCoreRecordsAsErrors) {
  core::PipelineTimingSample invalid;
  invalid.stage = stage();
  invalid.wall_duration = core::Duration{-1};
  const auto message = toDiagnosticArray(invalid, builtin_interfaces::msg::Time{});
  ASSERT_EQ(message.status.size(), 1U);
  EXPECT_EQ(message.status.front().level, diagnostic_msgs::msg::DiagnosticStatus::ERROR);
  EXPECT_EQ(message.status.front().message, "invalid_pipeline_timing_sample");
}

TEST(PipelineTimingPublisher, UsesDeclaredDefaultTopicAndAcceptsBothRecords) {
  if (!rclcpp::ok()) {
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }
  {
    rclcpp::Node node("pipeline_timing_publisher_test");
    PipelineTimingPublisher publisher(node);
    EXPECT_EQ(publisher.topicName(), kLocalPipelineTimingTopic);

    core::PipelineTimingSample sample;
    sample.stage = stage();
    sample.wall_duration = core::Duration{1'000};
    publisher.publish(sample);

    core::BoundedPipelineTimingAccumulator accumulator(stage(), 2U);
    ASSERT_EQ(accumulator.observe(sample), core::PipelineTimingObserveStatus::Accepted);
    publisher.publish(accumulator.snapshot());
  }
  rclcpp::shutdown();
}

}  // namespace
}  // namespace meridian::ros
