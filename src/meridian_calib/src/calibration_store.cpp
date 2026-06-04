#include "meridian/calib/icalibration_provider.hpp"

#include <atomic>
#include <utility>

namespace meridian {

CalibrationStore::CalibrationStore(std::shared_ptr<const CalibrationSet> seed_v0)
    : snapshot_(std::move(seed_v0)), version_(0) {}

std::shared_ptr<const CalibrationSet> CalibrationStore::current() const {
  return snapshot_.load(std::memory_order_acquire);
}

std::uint32_t CalibrationStore::version() const {
  return version_.load(std::memory_order_acquire);
}

void CalibrationStore::publish(std::shared_ptr<const CalibrationSet> refined) {
  // Copy-on-write: swap in the new immutable snapshot and bump the monotonic version.
  // A reader always observes some consistent immutable set, and re-caches when the
  // version it last saw changes.
  snapshot_.store(std::move(refined), std::memory_order_release);
  version_.fetch_add(1, std::memory_order_acq_rel);
}

}  // namespace meridian
