#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

#include "meridian/calib/calibration_set.hpp"

namespace meridian {

// Read side: any layer holds this and pulls an immutable snapshot. The returned
// shared_ptr<const> stays valid as long as the holder keeps it.
class ICalibrationProvider {
 public:
  virtual ~ICalibrationProvider() = default;
  // Latest published snapshot pointer (cheap).
  virtual std::shared_ptr<const CalibrationSet> current() const = 0;
  // Monotonically increasing version of current(); readers compare against a
  // cached value to detect "calibration changed, re-cache."
  virtual std::uint32_t version() const = 0;
};

// Write side: exactly one writer (the back-end thread); many readers. current()
// is a lock-free atomic load of the snapshot pointer; publish() atomically swaps
// in a freshly built set (copy-on-write), so no reader ever sees a half update.
class CalibrationStore final : public ICalibrationProvider {
 public:
  explicit CalibrationStore(std::shared_ptr<const CalibrationSet> seed_v0);

  std::shared_ptr<const CalibrationSet> current() const override;
  std::uint32_t version() const override;

  // Publish a new immutable set and bump the version. Back-end thread only.
  void publish(std::shared_ptr<const CalibrationSet> refined);

 private:
  // Atomic shared_ptr: reads are lock-free and never tear against a concurrent publish().
  std::atomic<std::shared_ptr<const CalibrationSet>> snapshot_;
  std::atomic<std::uint32_t> version_;
};

}  // namespace meridian
