#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
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

  // Outcome of a protected push: the new element is always enqueued (unless the queue
  // is closed); `evicted` is the element dropped to make room, if any. The caller
  // distinguishes a benign eviction from a costly one by inspecting *evicted.
  struct ProtectedPush {
    bool enqueued = false;
    std::optional<T> evicted;
  };

  // Lossy push that prefers to evict an UNprotected element. When full, scans from the
  // oldest end for the first element `protect` returns false for and evicts that one;
  // only if every queued element is protected does it fall back to evicting the oldest
  // (so the queue can never deadlock by refusing to make room). The new element is
  // always enqueued. The caller inspects the returned eviction to report which kind of
  // element was lost.
  ProtectedPush push_protecting(T&& v, const std::function<bool(const T&)>& protect) {
    ProtectedPush out;
    {
      std::lock_guard<std::mutex> lock(m_);
      if (closed_) return out;
      if (q_.size() >= capacity_) {
        auto victim = q_.begin();
        for (; victim != q_.end(); ++victim) {
          if (!protect(*victim)) break;
        }
        if (victim == q_.end()) victim = q_.begin();
        out.evicted = std::move(*victim);
        q_.erase(victim);
      }
      q_.push_back(std::move(v));
      out.enqueued = true;
    }
    cv_.notify_one();
    return out;
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
