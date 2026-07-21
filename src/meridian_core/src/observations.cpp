#include "meridian/core/observations.hpp"

#include <stdexcept>
#include <utility>

namespace meridian::core {

ObservationHeader::ObservationHeader(SensorId sensor_id, CalibrationId calibration_id,
                                     MeasurementId measurement_id, TimeNs measurement_time,
                                     std::string frame_id)
    : sensor_id_(std::move(sensor_id)),
      calibration_id_(std::move(calibration_id)),
      measurement_id_(measurement_id),
      measurement_time_(measurement_time),
      frame_id_(std::move(frame_id)) {
  if (frame_id_.empty()) {
    throw std::invalid_argument("ObservationHeader frame_id cannot be empty");
  }
}

ImuSample::ImuSample(ObservationHeader header, Vec3d angular_velocity_rad_s,
                     Vec3d specific_force_m_s2) noexcept
    : header_(std::move(header)),
      angular_velocity_rad_s_(angular_velocity_rad_s),
      specific_force_m_s2_(specific_force_m_s2) {}

LidarSweep::LidarSweep(ObservationHeader header, TimeNs acquisition_begin, TimeNs acquisition_end,
                       std::vector<LidarPoint> points)
    : header_(std::move(header)),
      acquisition_begin_(acquisition_begin),
      acquisition_end_(acquisition_end),
      points_(std::make_shared<const std::vector<LidarPoint>>(std::move(points))) {
  if (acquisition_end_ < acquisition_begin_) {
    throw std::invalid_argument("LidarSweep acquisition interval is reversed");
  }
  if (points_->empty()) {
    throw std::invalid_argument("LidarSweep must contain at least one finite point");
  }
}

}  // namespace meridian::core
