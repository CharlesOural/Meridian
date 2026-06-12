#pragma once

#include <cstddef>
#include <vector>

namespace meridian {

// Fixed-capacity ring buffer (not thread-safe). Once full, push() overwrites the oldest
// element. Index 0 is the oldest retained element, size()-1 the newest. Used for bounded
// measurement history (e.g. the recent IMU window); the cross-thread queues are a
// separate concern owned by the pipeline.
template <typename T>
class RingBuffer {
 public:
  explicit RingBuffer(std::size_t capacity) : buf_(capacity == 0 ? 1 : capacity) {}

  bool empty() const { return size_ == 0; }
  bool full() const { return size_ == buf_.size(); }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return buf_.size(); }
  void clear() {
    head_ = 0;
    size_ = 0;
  }

  void push(const T& v) {
    if (full()) {
      buf_[head_] = v;  // the oldest slot becomes the newest, then advance the window
      head_ = (head_ + 1) % buf_.size();
    } else {
      buf_[(head_ + size_) % buf_.size()] = v;
      ++size_;
    }
  }

  const T& operator[](std::size_t i) const { return buf_[(head_ + i) % buf_.size()]; }
  T& operator[](std::size_t i) { return buf_[(head_ + i) % buf_.size()]; }
  const T& front() const { return buf_[head_]; }                                // oldest
  const T& back() const { return buf_[(head_ + size_ - 1) % buf_.size()]; }      // newest

 private:
  std::vector<T> buf_;
  std::size_t head_ = 0;  // index of the oldest element
  std::size_t size_ = 0;
};

}  // namespace meridian
