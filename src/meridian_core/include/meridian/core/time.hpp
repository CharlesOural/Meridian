#pragma once

#include <compare>
#include <cstdint>
#include <optional>

#include "meridian/core/strong_id.hpp"

namespace meridian::core {

struct Duration {
  std::int64_t nanoseconds{};
  auto operator<=>(const Duration&) const = default;
};

struct FusionTime {
  std::int64_t nanoseconds{};
  auto operator<=>(const FusionTime&) const = default;
};

struct RawDeviceTime {
  std::int64_t nanoseconds{};
};

struct ArrivalTime {
  std::int64_t nanoseconds{};
};

[[nodiscard]] constexpr Duration operator-(FusionTime lhs, FusionTime rhs) noexcept {
  return Duration{lhs.nanoseconds - rhs.nanoseconds};
}

[[nodiscard]] constexpr FusionTime operator+(FusionTime time, Duration duration) noexcept {
  return FusionTime{time.nanoseconds + duration.nanoseconds};
}

[[nodiscard]] constexpr FusionTime operator-(FusionTime time,
                                             Duration duration) noexcept {
  return FusionTime{time.nanoseconds - duration.nanoseconds};
}

struct TimeRange {
  FusionTime start;
  FusionTime end;

  [[nodiscard]] constexpr bool valid() const noexcept { return start < end; }
  [[nodiscard]] constexpr bool contains(FusionTime time) const noexcept {
    return start <= time && time < end;
  }
  [[nodiscard]] constexpr Duration duration() const noexcept { return end - start; }
};

enum class TimeMappingStatus {
  Valid,
  Estimated,
  Discontinuous,
  Uncertain,
};

struct SourceStamp {
  RawDeviceTime raw_time;
  FusionTime fusion_time;
  ArrivalTime host_arrival_time;
  ClockRevision clock_revision;
  SourceEpoch source_epoch;
  std::optional<SourceSequence> device_sequence;
  IngressSequence ingress_sequence;
  Duration uncertainty;
  TimeMappingStatus status{TimeMappingStatus::Valid};
};

struct RecordHeader {
  std::uint32_t schema_version{1};
  TraceId trace;
  ProducerId producer;
  SessionId session;
  FusionTime created_at;
  ConfigRevision config;
  std::optional<CalibrationEpoch> direct_calibration;
};

}  // namespace meridian::core
