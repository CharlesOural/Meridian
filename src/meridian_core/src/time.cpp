#include "meridian/core/time.hpp"

#include <limits>
#include <stdexcept>

namespace meridian::core {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000;

}  // namespace

std::optional<TimeNs> TimeNs::fromSecNanosec(std::int64_t seconds,
                                             std::uint32_t nanoseconds) noexcept {
  if (nanoseconds >= static_cast<std::uint32_t>(kNanosecondsPerSecond)) {
    return std::nullopt;
  }

  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kMaximumWholeSeconds = kMax / kNanosecondsPerSecond;
  constexpr std::int64_t kMaximumFraction = kMax % kNanosecondsPerSecond;
  constexpr std::int64_t kMinimumWholeSeconds = kMin / kNanosecondsPerSecond - 1;
  constexpr std::int64_t kMinimumFraction = kNanosecondsPerSecond + kMin % kNanosecondsPerSecond;
  if (seconds > kMaximumWholeSeconds || seconds < kMinimumWholeSeconds) {
    return std::nullopt;
  }

  const std::int64_t fractional = static_cast<std::int64_t>(nanoseconds);
  if (seconds == kMaximumWholeSeconds && fractional > kMaximumFraction) {
    return std::nullopt;
  }
  if (seconds == kMinimumWholeSeconds) {
    if (fractional < kMinimumFraction) {
      return std::nullopt;
    }
    return TimeNs(kMin + (fractional - kMinimumFraction));
  }

  const std::int64_t whole_seconds = seconds * kNanosecondsPerSecond;
  return TimeNs(whole_seconds + fractional);
}

std::optional<TimeNs> TimeNs::checkedAdd(TimeNs time, std::int64_t delta_ns) noexcept {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  const std::int64_t value = time.count();
  if ((delta_ns > 0 && value > kMax - delta_ns) || (delta_ns < 0 && value < kMin - delta_ns)) {
    return std::nullopt;
  }
  return TimeNs(value + delta_ns);
}

std::optional<std::int64_t> TimeNs::checkedDifference(TimeNs lhs, TimeNs rhs) noexcept {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  const std::int64_t left = lhs.count();
  const std::int64_t right = rhs.count();
  if ((right < 0 && left > kMax + right) || (right > 0 && left < kMin + right)) {
    return std::nullopt;
  }
  return left - right;
}

TimeRange::TimeRange(TimeNs begin, TimeNs end) : begin_(begin), end_(end) {
  if (end_ < begin_) {
    throw std::invalid_argument("TimeRange end precedes begin");
  }
}

std::optional<std::int64_t> TimeRange::durationNs() const noexcept {
  return TimeNs::checkedDifference(end_, begin_);
}

}  // namespace meridian::core
