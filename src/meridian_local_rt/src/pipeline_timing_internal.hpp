#pragma once

#include <memory>
#include <utility>

#include "meridian/local/pipeline_observability.hpp"

namespace meridian::local::detail {

inline void observeLocalPipelineTiming(const std::shared_ptr<LocalPipelineTimingRecorder>& recorder,
                                       LocalPipelineTimingStage stage,
                                       const core::ThreadCpuWallTimer& timer,
                                       core::PipelineDisposition disposition,
                                       core::PipelineWorkIdentity work = {}) noexcept {
  if (!recorder) {
    return;
  }
  static_cast<void>(recorder->observe(stage, timer.elapsed(), disposition, std::move(work)));
}

// Defaulting to Failed ensures every early-return and exception path remains
// visible. Successful callers explicitly mark the scope complete.
class LocalPipelineTimingScope {
public:
  LocalPipelineTimingScope(std::shared_ptr<LocalPipelineTimingRecorder> recorder,
                           LocalPipelineTimingStage stage,
                           core::PipelineWorkIdentity work = {}) noexcept
      : recorder_(std::move(recorder)), stage_(stage), work_(std::move(work)) {}

  ~LocalPipelineTimingScope() {
    observeLocalPipelineTiming(recorder_, stage_, timer_, disposition_, std::move(work_));
  }

  LocalPipelineTimingScope(const LocalPipelineTimingScope&) = delete;
  LocalPipelineTimingScope& operator=(const LocalPipelineTimingScope&) = delete;
  LocalPipelineTimingScope(LocalPipelineTimingScope&&) = delete;
  LocalPipelineTimingScope& operator=(LocalPipelineTimingScope&&) = delete;

  void finish(
      core::PipelineDisposition disposition = core::PipelineDisposition::Completed) noexcept {
    disposition_ = disposition;
  }

private:
  std::shared_ptr<LocalPipelineTimingRecorder> recorder_;
  LocalPipelineTimingStage stage_;
  core::PipelineWorkIdentity work_;
  core::ThreadCpuWallTimer timer_;
  core::PipelineDisposition disposition_{core::PipelineDisposition::Failed};
};

}  // namespace meridian::local::detail
