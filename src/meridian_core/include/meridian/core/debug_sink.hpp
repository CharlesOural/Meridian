#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/ids.hpp"
#include "meridian/core/observations.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

struct ImuAcceptedEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  Vec3d angular_velocity_rad_s;
  Vec3d specific_force_m_s2;
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
};

struct LidarAcceptedEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  std::int64_t acquisition_duration_ns{};
  std::uint64_t source_points{};
  std::uint64_t accepted_points{};
  std::uint64_t nonfinite_xyz_points{};
  std::uint64_t zero_xyz_points{};
  std::uint64_t flattened_time_regressions{};
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
  std::uint64_t decode_queue_depth{};
};

struct LidarPreviewPoint final {
  float x{};
  float y{};
  float z{};
};

struct LidarPreviewEvent final {
  MeasurementId measurement_id;
  TimeNs measurement_time;
  std::shared_ptr<const std::vector<LidarPreviewPoint>> points;
};

struct IngressFailureEvent final {
  std::string sensor_id;
  MeasurementId measurement_id;
  std::optional<TimeNs> measurement_time;
  std::int64_t arrival_steady_ns{};
  std::int64_t conversion_duration_ns{};
  std::string error_code;
  std::string field;
  std::string detail;
};

// Debug sinks are observers only: calls must not throw and must not mutate the
// localization path. Heavy preview geometry is built only when requested.
class DebugSink {
public:
  virtual ~DebugSink() = default;

  [[nodiscard]] virtual bool wantsLidarPreview() const noexcept = 0;
  virtual void record(const ImuAcceptedEvent& event) noexcept = 0;
  virtual void record(const LidarAcceptedEvent& event) noexcept = 0;
  virtual void record(const LidarPreviewEvent& event) noexcept = 0;
  virtual void record(const IngressFailureEvent& event) noexcept = 0;
  [[nodiscard]] virtual std::uint64_t droppedEvents() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t logErrors() const noexcept = 0;
};

class NullDebugSink final : public DebugSink {
public:
  [[nodiscard]] bool wantsLidarPreview() const noexcept override { return false; }
  void record(const ImuAcceptedEvent&) noexcept override {}
  void record(const LidarAcceptedEvent&) noexcept override {}
  void record(const LidarPreviewEvent&) noexcept override {}
  void record(const IngressFailureEvent&) noexcept override {}
  [[nodiscard]] std::uint64_t droppedEvents() const noexcept override { return 0U; }
  [[nodiscard]] std::uint64_t logErrors() const noexcept override { return 0U; }
};

}  // namespace meridian::core
