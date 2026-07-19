#include "meridian/local/pipeline_observability.hpp"

#include <array>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace meridian::local {
namespace {

[[nodiscard]] constexpr std::size_t stageIndex(LocalPipelineTimingStage stage) noexcept {
  return static_cast<std::size_t>(stage);
}

struct StageDefinition {
  std::uint64_t id;
  std::string_view name;
};

constexpr std::array<StageDefinition, kLocalPipelineTimingStageCount> kStageDefinitions{{
    {30'001U, "local.initialization.stationary_probe"},
    {30'002U, "local.initialization.bootstrap_acquisition_deskew"},
    {30'003U, "local.lidar.registration_view_build"},
    {30'004U, "local.lidar.target_build_update"},
    {30'005U, "local.lidar.correspondence_registration_solve"},
    {30'006U, "local.initialization.motion_batch_solve_refinement"},
    {30'007U, "local.tracking.deskew"},
    {30'008U, "local.graph.transaction_update"},
    {30'009U, "local.lidar.finalized_target_update"},
    {30'010U, "local.lidar.composite_target_index_build"},
    {30'011U, "local.lidar.factor_batch_prepare"},
}};

[[nodiscard]] const std::array<core::PipelineStage, kLocalPipelineTimingStageCount>& stages() {
  static const std::array<core::PipelineStage, kLocalPipelineTimingStageCount> records = [] {
    std::array<core::PipelineStage, kLocalPipelineTimingStageCount> output;
    for (std::size_t index = 0U; index < output.size(); ++index) {
      const auto stage = core::makePipelineStage(core::PipelineStageId{kStageDefinitions[index].id},
                                                 kStageDefinitions[index].name);
      if (!stage) {
        throw std::logic_error("invalid built-in local pipeline timing stage");
      }
      output[index] = *stage;
    }
    return output;
  }();
  return records;
}

[[nodiscard]] bool validStage(LocalPipelineTimingStage stage) noexcept {
  return stageIndex(stage) < kLocalPipelineTimingStageCount;
}

}  // namespace

const core::PipelineStage& localPipelineTimingStageRecord(LocalPipelineTimingStage stage) {
  if (!validStage(stage)) {
    throw std::out_of_range("local pipeline timing stage is outside the stable taxonomy");
  }
  return stages()[stageIndex(stage)];
}

std::string_view localPipelineTimingSpanSemanticsName(
    LocalPipelineTimingSpanSemantics semantics) noexcept {
  switch (semantics) {
    case LocalPipelineTimingSpanSemantics::Inclusive:
      return "inclusive";
  }
  return "unknown";
}

bool LocalPipelineTimingReport::valid() const noexcept {
  if (schema_version != kSchemaVersion || window_capacity == 0U ||
      window_capacity > kMaximumLocalPipelineTimingWindowCapacity ||
      span_semantics != LocalPipelineTimingSpanSemantics::Inclusive) {
    return false;
  }
  for (std::size_t index = 0U; index < stages.size(); ++index) {
    const auto expected_stage = static_cast<LocalPipelineTimingStage>(index);
    if (!stages[index].valid() ||
        stages[index].stage != localPipelineTimingStageRecord(expected_stage) ||
        stages[index].wall.window_capacity != window_capacity ||
        stages[index].thread_cpu.window_capacity != window_capacity) {
      return false;
    }
  }
  return true;
}

const core::PipelineTimingStatisticsSnapshot* LocalPipelineTimingReport::find(
    LocalPipelineTimingStage stage) const noexcept {
  if (!validStage(stage)) {
    return nullptr;
  }
  return &stages[stageIndex(stage)];
}

LocalPipelineTimingRecorder::LocalPipelineTimingRecorder(LocalPipelineTimingConfig config)
    : config_(config) {
  if (!config_.valid()) {
    throw std::invalid_argument(
        "local pipeline timing window must be nonzero and within the hard capacity bound");
  }
  for (std::size_t index = 0U; index < accumulators_.size(); ++index) {
    accumulators_[index] = std::make_unique<core::BoundedPipelineTimingAccumulator>(
        localPipelineTimingStageRecord(static_cast<LocalPipelineTimingStage>(index)),
        config_.window_capacity);
  }
}

LocalPipelineTimingRecorder::~LocalPipelineTimingRecorder() = default;

core::PipelineTimingObserveStatus LocalPipelineTimingRecorder::observe(
    LocalPipelineTimingStage stage, const core::CpuWallDuration& duration,
    core::PipelineDisposition disposition, core::PipelineWorkIdentity work) noexcept {
  if (!validStage(stage)) {
    return core::PipelineTimingObserveStatus::StageMismatch;
  }
  core::PipelineTimingSample sample;
  sample.stage = localPipelineTimingStageRecord(stage);
  sample.wall_duration = duration.wall;
  sample.thread_cpu_duration = duration.thread_cpu;
  sample.work = std::move(work);
  sample.disposition = disposition;
  return accumulators_[stageIndex(stage)]->observe(sample);
}

LocalPipelineTimingReport LocalPipelineTimingRecorder::snapshot() const noexcept {
  LocalPipelineTimingReport output;
  output.window_capacity = config_.window_capacity;
  for (std::size_t index = 0U; index < accumulators_.size(); ++index) {
    output.stages[index] = accumulators_[index]->snapshot();
  }
  return output;
}

}  // namespace meridian::local
