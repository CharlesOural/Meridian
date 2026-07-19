#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

// Fixed-size text keeps hot-path timing records self-contained and allocation
// free. Construction rejects empty or oversized values instead of truncating
// them, because a truncated stage or queue name is ambiguous in telemetry.
template <std::size_t Capacity>
class FixedPipelineText {
public:
  [[nodiscard]] static std::optional<FixedPipelineText> make(std::string_view value) noexcept {
    if (value.empty() || value.size() > Capacity) {
      return std::nullopt;
    }
    FixedPipelineText output;
    for (std::size_t index = 0; index < value.size(); ++index) {
      output.bytes_[index] = value[index];
    }
    output.size_ = value.size();
    return output;
  }

  [[nodiscard]] std::string_view view() const noexcept {
    return std::string_view(bytes_.data(), size_);
  }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }
  [[nodiscard]] constexpr std::size_t capacity() const noexcept { return Capacity; }

  bool operator==(const FixedPipelineText&) const = default;

private:
  std::array<char, Capacity> bytes_{};
  std::size_t size_{};
};

using PipelineStageName = FixedPipelineText<63U>;
using PipelineQueueName = FixedPipelineText<63U>;
using PipelineDispositionDetail = FixedPipelineText<127U>;

struct PipelineStageIdTag;
using PipelineStageId = StrongId<PipelineStageIdTag>;

struct PipelineStage {
  PipelineStageId id;
  PipelineStageName name;

  [[nodiscard]] bool valid() const noexcept { return id.valid() && !name.empty(); }

  bool operator==(const PipelineStage&) const = default;
};

[[nodiscard]] std::optional<PipelineStage> makePipelineStage(PipelineStageId id,
                                                             std::string_view name) noexcept;

// Core deliberately does not depend on meridian_global. This typed value is
// the wire-neutral representation of a global graph revision in runtime
// reports; global adapters copy GlobalGraphRevision::value() into it.
struct PipelineGlobalRevision {
  static constexpr std::uint64_t kInvalidValue = std::numeric_limits<std::uint64_t>::max();

  std::uint64_t value{kInvalidValue};

  [[nodiscard]] bool valid() const noexcept { return value != kInvalidValue; }

  auto operator<=>(const PipelineGlobalRevision&) const = default;
};

enum class PipelineDisposition : std::uint8_t {
  Completed,
  Accepted,
  Rejected,
  Deferred,
  Dropped,
  Skipped,
  Failed,
};

[[nodiscard]] std::string_view pipelineDispositionName(PipelineDisposition disposition) noexcept;

struct PipelineWorkIdentity {
  std::optional<MeasurementId> measurement;
  std::optional<StateId> state;
  std::optional<LocalGraphRevision> local_revision;
  std::optional<PipelineGlobalRevision> global_revision;
};

// One instantaneous queue view. Cumulative counters never reset implicitly;
// queue owners decide the reset/revision boundary and publish it explicitly.
struct PipelineQueueSnapshot {
  PipelineQueueName name;
  std::size_t count{};
  std::size_t bytes{};
  std::optional<std::size_t> count_capacity;
  std::optional<std::size_t> byte_capacity;
  std::optional<Duration> oldest_age;
  std::uint64_t accepted{};
  std::uint64_t rejected{};
  std::uint64_t dropped_oldest{};
  std::uint64_t dropped_newest{};
  std::uint64_t skipped_stale{};
  std::uint64_t skipped_policy{};

  [[nodiscard]] bool valid() const noexcept;
};

struct PipelineTimingSample {
  static constexpr std::uint32_t kSchemaVersion = 1U;

  std::uint32_t schema_version{kSchemaVersion};
  PipelineStage stage;
  Duration wall_duration;
  std::optional<Duration> thread_cpu_duration;
  PipelineWorkIdentity work;
  PipelineDisposition disposition{PipelineDisposition::Completed};
  std::optional<PipelineDispositionDetail> detail;
  std::optional<PipelineQueueSnapshot> queue;

  [[nodiscard]] bool valid() const noexcept;
};

struct DurationStatistics {
  std::uint64_t total_samples{};
  std::size_t window_samples{};
  std::size_t window_capacity{};
  std::optional<Duration> minimum;
  std::optional<Duration> p50;
  std::optional<Duration> p95;
  std::optional<Duration> p99;
  std::optional<Duration> maximum;
  std::optional<double> mean_nanoseconds;
};

struct PipelineDispositionCounts {
  std::uint64_t completed{};
  std::uint64_t accepted{};
  std::uint64_t rejected{};
  std::uint64_t deferred{};
  std::uint64_t dropped{};
  std::uint64_t skipped{};
  std::uint64_t failed{};
};

struct PipelineTimingStatisticsSnapshot {
  static constexpr std::uint32_t kSchemaVersion = 1U;

  std::uint32_t schema_version{kSchemaVersion};
  PipelineStage stage;
  DurationStatistics wall;
  DurationStatistics thread_cpu;
  PipelineDispositionCounts dispositions;
  std::optional<PipelineQueueSnapshot> latest_queue;

  [[nodiscard]] bool valid() const noexcept;
};

enum class PipelineTimingObserveStatus {
  Accepted,
  InvalidSample,
  StageMismatch,
};

// A single-stage, rolling-window accumulator. The constructor performs all
// storage allocation. observe() and snapshot() only overwrite/copy/sort those
// fixed buffers, so recording and percentile calculation do not grow the heap.
//
// Percentiles use deterministic nearest-rank selection over the most recent
// window_capacity accepted samples.
class BoundedPipelineTimingAccumulator {
public:
  BoundedPipelineTimingAccumulator(PipelineStage stage, std::size_t window_capacity);

  BoundedPipelineTimingAccumulator(const BoundedPipelineTimingAccumulator&) = delete;
  BoundedPipelineTimingAccumulator& operator=(const BoundedPipelineTimingAccumulator&) = delete;

  [[nodiscard]] PipelineTimingObserveStatus observe(const PipelineTimingSample& sample) noexcept;
  [[nodiscard]] PipelineTimingStatisticsSnapshot snapshot() const noexcept;
  [[nodiscard]] std::size_t windowCapacity() const noexcept { return samples_.size(); }

private:
  struct Durations {
    std::int64_t wall_nanoseconds{};
    std::int64_t thread_cpu_nanoseconds{};
    bool has_thread_cpu{};
  };

  PipelineStage stage_;
  std::vector<Durations> samples_;
  mutable std::vector<std::int64_t> wall_scratch_;
  mutable std::vector<std::int64_t> cpu_scratch_;
  std::size_t next_index_{};
  std::size_t window_size_{};
  std::uint64_t total_samples_{};
  std::uint64_t total_cpu_samples_{};
  PipelineDispositionCounts dispositions_;
  std::optional<PipelineQueueSnapshot> latest_queue_;
  mutable std::mutex mutex_;
};

struct CpuWallDuration {
  Duration wall;
  std::optional<Duration> thread_cpu;
};

// Linux CLOCK_THREAD_CPUTIME_ID excludes sleep and scheduling delay, while the
// steady clock captures the actual deadline cost observed by the pipeline.
[[nodiscard]] std::optional<Duration> currentThreadCpuTime() noexcept;

class ThreadCpuWallTimer {
public:
  ThreadCpuWallTimer() noexcept;

  [[nodiscard]] CpuWallDuration elapsed() const noexcept;

private:
  std::chrono::steady_clock::time_point wall_start_;
  std::optional<Duration> thread_cpu_start_;
};

}  // namespace meridian::core
