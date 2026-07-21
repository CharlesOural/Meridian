#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/ids.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

class ObservationHeader final {
public:
  ObservationHeader(SensorId sensor_id, CalibrationId calibration_id, MeasurementId measurement_id,
                    TimeNs measurement_time, std::string frame_id);

  [[nodiscard]] const SensorId& sensorId() const noexcept { return sensor_id_; }
  [[nodiscard]] const CalibrationId& calibrationId() const noexcept { return calibration_id_; }
  [[nodiscard]] MeasurementId measurementId() const noexcept { return measurement_id_; }
  [[nodiscard]] TimeNs measurementTime() const noexcept { return measurement_time_; }
  [[nodiscard]] std::string_view frameId() const noexcept { return frame_id_; }

private:
  SensorId sensor_id_;
  CalibrationId calibration_id_;
  MeasurementId measurement_id_;
  TimeNs measurement_time_;
  std::string frame_id_;
};

class ImuSample final {
public:
  ImuSample(ObservationHeader header, Vec3d angular_velocity_rad_s,
            Vec3d specific_force_m_s2) noexcept;

  [[nodiscard]] const ObservationHeader& header() const noexcept { return header_; }
  [[nodiscard]] const Vec3d& angularVelocityRadS() const noexcept {
    return angular_velocity_rad_s_;
  }
  [[nodiscard]] const Vec3d& specificForceMS2() const noexcept { return specific_force_m_s2_; }

private:
  ObservationHeader header_;
  Vec3d angular_velocity_rad_s_;
  Vec3d specific_force_m_s2_;
};

struct LidarPoint final {
  float x{};
  float y{};
  float z{};
  std::int64_t time_offset_ns{};
  std::uint32_t source_index{};
  std::optional<float> intensity;
  std::optional<std::uint16_t> ring;
};

// A sweep owns its point buffer. Once constructed, callers receive only a const
// view; copies of a sweep share the same immutable storage.
class LidarSweep final {
public:
  LidarSweep(ObservationHeader header, TimeNs acquisition_begin, TimeNs acquisition_end,
             std::vector<LidarPoint> points);

  [[nodiscard]] const ObservationHeader& header() const noexcept { return header_; }
  [[nodiscard]] TimeNs acquisitionBegin() const noexcept { return acquisition_begin_; }
  [[nodiscard]] TimeNs acquisitionEnd() const noexcept { return acquisition_end_; }
  [[nodiscard]] std::span<const LidarPoint> points() const noexcept { return *points_; }
  [[nodiscard]] std::size_t size() const noexcept { return points_->size(); }
  [[nodiscard]] std::size_t storageBytes() const noexcept {
    return points_->size() * sizeof(LidarPoint);
  }

private:
  ObservationHeader header_;
  TimeNs acquisition_begin_;
  TimeNs acquisition_end_;
  std::shared_ptr<const std::vector<LidarPoint>> points_;
};

}  // namespace meridian::core
