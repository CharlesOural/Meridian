#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include "meridian/core/api.hpp"

namespace meridian::local {

// Stable, ROS-free stage taxonomy for the local estimator's expensive work.
// Values are append-only because run reports and benchmark tooling persist the
// corresponding core::PipelineStage records.
enum class LocalPipelineTimingStage : std::uint8_t {
  StationaryProbe = 0U,
  BootstrapAcquisitionDeskew = 1U,
  RegistrationViewBuild = 2U,
  TargetBuildUpdate = 3U,
  CorrespondenceRegistrationSolve = 4U,
  MotionBatchSolveRefinement = 5U,
  TrackingDeskew = 6U,
  GraphTransactionUpdate = 7U,
  FinalizedTargetUpdate = 8U,
  CompositeTargetIndexBuild = 9U,
  LidarFactorBatchPrepare = 10U,
};

inline constexpr std::size_t kLocalPipelineTimingStageCount = 11U;
inline constexpr std::size_t kMaximumLocalPipelineTimingWindowCapacity = 65'536U;

enum class LocalPipelineTimingSpanSemantics : std::uint8_t {
  // Parent stages include nested substage time. Consumers must not sum stage
  // totals to estimate end-to-end latency.
  Inclusive,
};

[[nodiscard]] std::string_view localPipelineTimingSpanSemanticsName(
    LocalPipelineTimingSpanSemantics semantics) noexcept;

struct LocalPipelineTimingConfig {
  std::size_t window_capacity{1'024U};

  [[nodiscard]] bool valid() const noexcept {
    return window_capacity > 0U && window_capacity <= kMaximumLocalPipelineTimingWindowCapacity;
  }
};

[[nodiscard]] const core::PipelineStage& localPipelineTimingStageRecord(
    LocalPipelineTimingStage stage);

struct LocalPipelineTimingReport {
  static constexpr std::uint32_t kSchemaVersion = 1U;

  std::uint32_t schema_version{kSchemaVersion};
  std::size_t window_capacity{};
  LocalPipelineTimingSpanSemantics span_semantics{LocalPipelineTimingSpanSemantics::Inclusive};
  std::array<core::PipelineTimingStatisticsSnapshot, kLocalPipelineTimingStageCount> stages;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const core::PipelineTimingStatisticsSnapshot* find(
      LocalPipelineTimingStage stage) const noexcept;
};

// Shared by the estimator and its nested ROS-free algorithm components.
// Construction performs all rolling-window allocations; observe() never grows
// storage. The collector is thread-safe through the core accumulators.
class LocalPipelineTimingRecorder {
public:
  explicit LocalPipelineTimingRecorder(LocalPipelineTimingConfig config = {});
  ~LocalPipelineTimingRecorder();

  LocalPipelineTimingRecorder(const LocalPipelineTimingRecorder&) = delete;
  LocalPipelineTimingRecorder& operator=(const LocalPipelineTimingRecorder&) = delete;
  LocalPipelineTimingRecorder(LocalPipelineTimingRecorder&&) = delete;
  LocalPipelineTimingRecorder& operator=(LocalPipelineTimingRecorder&&) = delete;

  [[nodiscard]] core::PipelineTimingObserveStatus observe(
      LocalPipelineTimingStage stage, const core::CpuWallDuration& duration,
      core::PipelineDisposition disposition = core::PipelineDisposition::Completed,
      core::PipelineWorkIdentity work = {}) noexcept;

  [[nodiscard]] LocalPipelineTimingReport snapshot() const noexcept;
  [[nodiscard]] const LocalPipelineTimingConfig& config() const noexcept { return config_; }

private:
  LocalPipelineTimingConfig config_;
  std::array<std::unique_ptr<core::BoundedPipelineTimingAccumulator>,
             kLocalPipelineTimingStageCount>
      accumulators_;
};

}  // namespace meridian::local
