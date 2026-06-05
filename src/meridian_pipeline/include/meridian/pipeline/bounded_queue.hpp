#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace meridian {

// Bounded FIFO connecting two pipeline stages. Producers never block: a full queue
// either rejects the push (try_push) or evicts the oldest element
// (push_or_drop_oldest, the lossy-edge policy — keep the freshest data and let the
// caller count the drop). The consumer blocks in pop() until an element arrives or
// the queue is closed. Safe for multiple producers and one consumer.
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

  // Non-blocking push; false if the queue is full or closed.
  bool try_push(T&& v) {
    {
      std::lock_guard<std::mutex> lock(m_);
      if (closed_ || q_.size() >= capacity_) return false;
      q_.push_back(std::move(v));
    }
    cv_.notify_one();
    return true;
  }

  // Never blocks and never rejects (unless closed): evicts the oldest element when
  // full. Returns the number of elements dropped (0 or 1); the caller reports it.
  std::size_t push_or_drop_oldest(T&& v) {
    std::size_t dropped = 0;
    {
      std::lock_guard<std::mutex> lock(m_);
      if (closed_) return 0;
      if (q_.size() >= capacity_) {
        q_.pop_front();
        dropped = 1;
      }
      q_.push_back(std::move(v));
    }
    cv_.notify_one();
    return dropped;
  }

  // Blocks until an element is available or the queue is closed.
  // Returns false only when the queue is closed AND drained.
  bool pop(T& out) {
    std::unique_lock<std::mutex> lock(m_);
    cv_.wait(lock, [this] { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  // Non-blocking pop; false if empty.
  bool try_pop(T& out) {
    std::lock_guard<std::mutex> lock(m_);
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }

  // Wakes all waiters; subsequent pops drain the remaining elements then return false.
  void close() {
    {
      std::lock_guard<std::mutex> lock(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(m_);
    return q_.size();
  }
  bool closed() const {
    std::lock_guard<std::mutex> lock(m_);
    return closed_;
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
};

}  // namespace meridian
