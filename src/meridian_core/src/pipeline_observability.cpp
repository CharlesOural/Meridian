#include "meridian/core/pipeline_observability.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <limits>
#include <stdexcept>
#include <utility>

namespace meridian::core {
namespace {

[[nodiscard]] bool nonnegative(const std::optional<Duration>& duration) noexcept {
  return !duration || duration->nanoseconds >= 0;
}

void incrementDisposition(PipelineDisposition disposition,
                          PipelineDispositionCounts* counts) noexcept {
  switch (disposition) {
    case PipelineDisposition::Completed:
      ++counts->completed;
      return;
    case PipelineDisposition::Accepted:
      ++counts->accepted;
      return;
    case PipelineDisposition::Rejected:
      ++counts->rejected;
      return;
    case PipelineDisposition::Deferred:
      ++counts->deferred;
      return;
    case PipelineDisposition::Dropped:
      ++counts->dropped;
      return;
    case PipelineDisposition::Skipped:
      ++counts->skipped;
      return;
    case PipelineDisposition::Failed:
      ++counts->failed;
      return;
  }
}

[[nodiscard]] std::size_t nearestRankIndex(std::size_t count, double quantile) noexcept {
  const double rank = std::ceil(quantile * static_cast<double>(count));
  return static_cast<std::size_t>(std::max(1.0, rank)) - 1U;
}

[[nodiscard]] DurationStatistics statistics(std::vector<std::int64_t>* scratch, std::size_t count,
                                            std::size_t capacity,
                                            std::uint64_t total_samples) noexcept {
  DurationStatistics output;
  output.total_samples = total_samples;
  output.window_samples = count;
  output.window_capacity = capacity;
  if (count == 0U) {
    return output;
  }

  std::sort(scratch->begin(), scratch->begin() + static_cast<std::ptrdiff_t>(count));
  const auto durationAt = [&](std::size_t index) { return Duration{scratch->at(index)}; };
  output.minimum = durationAt(0U);
  output.p50 = durationAt(nearestRankIndex(count, 0.50));
  output.p95 = durationAt(nearestRankIndex(count, 0.95));
  output.p99 = durationAt(nearestRankIndex(count, 0.99));
  output.maximum = durationAt(count - 1U);

  long double sum = 0.0L;
  for (std::size_t index = 0; index < count; ++index) {
    sum += static_cast<long double>(scratch->at(index));
  }
  output.mean_nanoseconds = static_cast<double>(sum / static_cast<long double>(count));
  return output;
}

[[nodiscard]] std::optional<std::int64_t> timespecNanoseconds(const timespec& value) noexcept {
  constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= kNanosecondsPerSecond) {
    return std::nullopt;
  }
  if (static_cast<std::uint64_t>(value.tv_sec) >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() /
                                 kNanosecondsPerSecond)) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value.tv_sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(value.tv_nsec);
}

}  // namespace

std::optional<PipelineStage> makePipelineStage(PipelineStageId id, std::string_view name) noexcept {
  const auto fixed_name = PipelineStageName::make(name);
  if (!id.valid() || !fixed_name) {
    return std::nullopt;
  }
  return PipelineStage{id, *fixed_name};
}

std::string_view pipelineDispositionName(PipelineDisposition disposition) noexcept {
  switch (disposition) {
    case PipelineDisposition::Completed:
      return "completed";
    case PipelineDisposition::Accepted:
      return "accepted";
    case PipelineDisposition::Rejected:
      return "rejected";
    case PipelineDisposition::Deferred:
      return "deferred";
    case PipelineDisposition::Dropped:
      return "dropped";
    case PipelineDisposition::Skipped:
      return "skipped";
    case PipelineDisposition::Failed:
      return "failed";
  }
  return "unknown";
}

bool PipelineQueueSnapshot::valid() const noexcept {
  if (name.empty() || !nonnegative(oldest_age)) {
    return false;
  }
  if (count_capacity && count > *count_capacity) {
    return false;
  }
  return !byte_capacity || bytes <= *byte_capacity;
}

bool PipelineTimingSample::valid() const noexcept {
  if (schema_version != kSchemaVersion || !stage.valid() || wall_duration.nanoseconds < 0 ||
      !nonnegative(thread_cpu_duration)) {
    return false;
  }
  if (work.measurement && !work.measurement->valid()) {
    return false;
  }
  if (work.state && !work.state->valid()) {
    return false;
  }
  if (work.local_revision && !work.local_revision->valid()) {
    return false;
  }
  if (work.global_revision && !work.global_revision->valid()) {
    return false;
  }
  if (detail && detail->empty()) {
    return false;
  }
  return !queue || queue->valid();
}

bool PipelineTimingStatisticsSnapshot::valid() const noexcept {
  if (schema_version != kSchemaVersion || !stage.valid()) {
    return false;
  }
  const auto valid_statistics = [](const DurationStatistics& value) {
    const bool all_absent = !value.minimum && !value.p50 && !value.p95 && !value.p99 &&
                            !value.maximum && !value.mean_nanoseconds;
    if (value.window_samples == 0U) {
      return all_absent;
    }
    if (value.window_capacity == 0U || value.window_samples > value.window_capacity ||
        !value.minimum || !value.p50 || !value.p95 || !value.p99 || !value.maximum ||
        !value.mean_nanoseconds || !std::isfinite(*value.mean_nanoseconds)) {
      return false;
    }
    return value.minimum->nanoseconds >= 0 &&
           value.minimum->nanoseconds <= value.p50->nanoseconds &&
           value.p50->nanoseconds <= value.p95->nanoseconds &&
           value.p95->nanoseconds <= value.p99->nanoseconds &&
           value.p99->nanoseconds <= value.maximum->nanoseconds;
  };
  return valid_statistics(wall) && valid_statistics(thread_cpu) &&
         (!latest_queue || latest_queue->valid());
}

BoundedPipelineTimingAccumulator::BoundedPipelineTimingAccumulator(PipelineStage stage,
                                                                   std::size_t window_capacity)
    : stage_(std::move(stage)),
      samples_(window_capacity),
      wall_scratch_(window_capacity),
      cpu_scratch_(window_capacity) {
  if (!stage_.valid()) {
    throw std::invalid_argument("pipeline timing accumulator requires a valid stage");
  }
  if (window_capacity == 0U) {
    throw std::invalid_argument("pipeline timing accumulator window capacity must be nonzero");
  }
}

PipelineTimingObserveStatus BoundedPipelineTimingAccumulator::observe(
    const PipelineTimingSample& sample) noexcept {
  if (!sample.valid()) {
    return PipelineTimingObserveStatus::InvalidSample;
  }
  if (sample.stage != stage_) {
    return PipelineTimingObserveStatus::StageMismatch;
  }

  std::scoped_lock lock(mutex_);
  samples_[next_index_] =
      Durations{sample.wall_duration.nanoseconds,
                sample.thread_cpu_duration ? sample.thread_cpu_duration->nanoseconds : 0,
                sample.thread_cpu_duration.has_value()};
  next_index_ = (next_index_ + 1U) % samples_.size();
  window_size_ = std::min(window_size_ + 1U, samples_.size());
  ++total_samples_;
  if (sample.thread_cpu_duration) {
    ++total_cpu_samples_;
  }
  incrementDisposition(sample.disposition, &dispositions_);
  if (sample.queue) {
    latest_queue_ = sample.queue;
  }
  return PipelineTimingObserveStatus::Accepted;
}

PipelineTimingStatisticsSnapshot BoundedPipelineTimingAccumulator::snapshot() const noexcept {
  std::scoped_lock lock(mutex_);
  std::size_t cpu_count = 0U;
  for (std::size_t index = 0; index < window_size_; ++index) {
    wall_scratch_[index] = samples_[index].wall_nanoseconds;
    if (samples_[index].has_thread_cpu) {
      cpu_scratch_[cpu_count] = samples_[index].thread_cpu_nanoseconds;
      ++cpu_count;
    }
  }

  PipelineTimingStatisticsSnapshot output;
  output.stage = stage_;
  output.wall = statistics(&wall_scratch_, window_size_, samples_.size(), total_samples_);
  output.thread_cpu = statistics(&cpu_scratch_, cpu_count, samples_.size(), total_cpu_samples_);
  output.dispositions = dispositions_;
  output.latest_queue = latest_queue_;
  return output;
}

std::optional<Duration> currentThreadCpuTime() noexcept {
#if defined(__linux__)
  timespec value{};
  if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) {
    return std::nullopt;
  }
  const auto nanoseconds = timespecNanoseconds(value);
  if (!nanoseconds) {
    return std::nullopt;
  }
  return Duration{*nanoseconds};
#else
  return std::nullopt;
#endif
}

ThreadCpuWallTimer::ThreadCpuWallTimer() noexcept
    : wall_start_(std::chrono::steady_clock::now()), thread_cpu_start_(currentThreadCpuTime()) {}

CpuWallDuration ThreadCpuWallTimer::elapsed() const noexcept {
  const auto wall_now = std::chrono::steady_clock::now();
  const auto wall_nanoseconds =
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_now - wall_start_).count();

  std::optional<Duration> cpu_elapsed;
  const auto cpu_now = currentThreadCpuTime();
  if (thread_cpu_start_ && cpu_now && cpu_now->nanoseconds >= thread_cpu_start_->nanoseconds) {
    cpu_elapsed = Duration{cpu_now->nanoseconds - thread_cpu_start_->nanoseconds};
  }
  return CpuWallDuration{Duration{wall_nanoseconds}, cpu_elapsed};
}

}  // namespace meridian::core
