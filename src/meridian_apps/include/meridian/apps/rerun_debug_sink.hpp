#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

#include "meridian/core/debug_sink.hpp"

namespace rerun {
class RecordingStream;
}

namespace meridian::apps {

// The sole owner of Rerun IO. Producers only copy small records into a bounded
// queue; one worker owns the RecordingStream, its timeline state, flushing, and
// final file closure.
class RerunDebugSink final : public core::DebugSink {
public:
  struct Options final {
    std::filesystem::path output_path;
    std::size_t queue_capacity{8192U};
  };

  explicit RerunDebugSink(Options options);
  ~RerunDebugSink() override;

  RerunDebugSink(const RerunDebugSink&) = delete;
  RerunDebugSink& operator=(const RerunDebugSink&) = delete;

  [[nodiscard]] bool wantsLidarPreview() const noexcept override { return true; }
  void record(const core::ImuAcceptedEvent& event) noexcept override;
  void record(const core::LidarAcceptedEvent& event) noexcept override;
  void record(const core::LidarPreviewEvent& event) noexcept override;
  void record(const core::IngressFailureEvent& event) noexcept override;
  [[nodiscard]] std::uint64_t droppedEvents() const noexcept override;
  [[nodiscard]] std::uint64_t logErrors() const noexcept override;

  // Drains all accepted records, flushes them from the same worker that logged
  // them, and destroys the RecordingStream so its footer is on disk.
  void shutdown() noexcept;

private:
  using Event = std::variant<core::ImuAcceptedEvent, core::LidarAcceptedEvent,
                             core::LidarPreviewEvent, core::IngressFailureEvent>;

  void enqueue(Event event) noexcept;
  void workerLoop(std::unique_ptr<rerun::RecordingStream> recording) noexcept;

  Options options_;
  mutable std::mutex mutex_;
  std::condition_variable queue_condition_;
  std::deque<Event> queue_;
  bool closing_{false};
  std::thread worker_;
  std::atomic<std::uint64_t> dropped_events_{0U};
  std::atomic<std::uint64_t> log_errors_{0U};
};

}  // namespace meridian::apps
