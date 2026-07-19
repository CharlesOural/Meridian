#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>

namespace meridian::core {

enum class QueueOverflowPolicy {
  RejectNewest,
  DropOldest,
};

enum class QueuePushStatus {
  Accepted,
  AcceptedAfterDroppingOldest,
  RejectedCountLimit,
  RejectedByteLimit,
};

struct QueueStats {
  std::size_t count{};
  std::size_t bytes{};
  std::size_t accepted{};
  std::size_t rejected{};
  std::size_t dropped_oldest{};
};

template <typename T>
class BoundedQueue {
 public:
  using SizeFunction = std::function<std::size_t(const T&)>;

  BoundedQueue(std::size_t max_count, std::size_t max_bytes,
               QueueOverflowPolicy overflow_policy, SizeFunction size_function)
      : max_count_(max_count),
        max_bytes_(max_bytes),
        overflow_policy_(overflow_policy),
        size_function_(std::move(size_function)) {}

  QueuePushStatus push(T value) {
    const std::size_t value_bytes = size_function_(value);
    std::scoped_lock lock(mutex_);
    if (value_bytes > max_bytes_) {
      ++stats_.rejected;
      return QueuePushStatus::RejectedByteLimit;
    }

    bool dropped = false;
    while ((!queue_.empty()) &&
           (queue_.size() >= max_count_ || stats_.bytes + value_bytes > max_bytes_)) {
      if (overflow_policy_ == QueueOverflowPolicy::RejectNewest) {
        ++stats_.rejected;
        return queue_.size() >= max_count_ ? QueuePushStatus::RejectedCountLimit
                                           : QueuePushStatus::RejectedByteLimit;
      }
      stats_.bytes -= queue_.front().bytes;
      queue_.pop_front();
      ++stats_.dropped_oldest;
      dropped = true;
    }

    if (max_count_ == 0) {
      ++stats_.rejected;
      return QueuePushStatus::RejectedCountLimit;
    }
    queue_.push_back(Item{std::move(value), value_bytes});
    stats_.bytes += value_bytes;
    ++stats_.accepted;
    stats_.count = queue_.size();
    return dropped ? QueuePushStatus::AcceptedAfterDroppingOldest
                   : QueuePushStatus::Accepted;
  }

  [[nodiscard]] std::optional<T> tryPop() {
    std::scoped_lock lock(mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    Item item = std::move(queue_.front());
    queue_.pop_front();
    stats_.bytes -= item.bytes;
    stats_.count = queue_.size();
    return std::move(item.value);
  }

  [[nodiscard]] QueueStats stats() const {
    std::scoped_lock lock(mutex_);
    QueueStats snapshot = stats_;
    snapshot.count = queue_.size();
    return snapshot;
  }

 private:
  struct Item {
    T value;
    std::size_t bytes;
  };

  std::size_t max_count_;
  std::size_t max_bytes_;
  QueueOverflowPolicy overflow_policy_;
  SizeFunction size_function_;
  mutable std::mutex mutex_;
  std::deque<Item> queue_;
  QueueStats stats_;
};

}  // namespace meridian::core
