#include "meridian/calib/icalibration_provider.hpp"

#include <mutex>
#include <utility>

namespace meridian {

CalibrationStore::CalibrationStore(std::shared_ptr<const CalibrationSet> seed_v0)
    : snapshot_(std::move(seed_v0)), version_(0) {}

std::shared_ptr<const CalibrationSet> CalibrationStore::current() const {
  std::lock_guard<std::mutex> lock(m_);
  return snapshot_;
}

std::uint32_t CalibrationStore::version() const {
  return version_.load(std::memory_order_acquire);
}

void CalibrationStore::publish(std::shared_ptr<const CalibrationSet> refined) {
  // Copy-on-write: swap in the new immutable snapshot and bump the monotonic version.
  // A reader always observes some consistent immutable set, and re-caches when the
  // version it last saw changes.
  {
    std::lock_guard<std::mutex> lock(m_);
    snapshot_ = std::move(refined);
  }
  version_.fetch_add(1, std::memory_order_acq_rel);
}

}  // namespace meridian
