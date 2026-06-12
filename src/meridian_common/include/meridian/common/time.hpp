#pragma once

#include <cstdint>

namespace meridian {

// Monotonic time on the single Meridian timeline, in integer nanoseconds.
using Timestamp = std::int64_t;  // ns since the timeline epoch
using Duration = std::int64_t;   // ns

inline constexpr std::int64_t kNanosPerSecond = 1'000'000'000;

constexpr double to_seconds(Duration ns) {
  return static_cast<double>(ns) / static_cast<double>(kNanosPerSecond);
}

constexpr Duration from_seconds(double s) {
  return static_cast<Duration>(s * static_cast<double>(kNanosPerSecond));
}

}  // namespace meridian
