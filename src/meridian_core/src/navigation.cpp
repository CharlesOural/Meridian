#include "meridian/core/navigation.hpp"

#include <stdexcept>

namespace meridian::core {

ImuBias::ImuBias(Vec3d gyroscope_rad_s, Vec3d accelerometer_m_s2)
    : gyroscope_rad_s_(gyroscope_rad_s), accelerometer_m_s2_(accelerometer_m_s2) {
  if (!gyroscope_rad_s_.isFinite() || !accelerometer_m_s2_.isFinite()) {
    throw std::invalid_argument("ImuBias vectors must be finite");
  }
}

NavigationState::NavigationState(StateId id, TimeNs time, Pose3d odom_from_imu,
                                 Vec3d velocity_odom_m_s, ImuBias imu_bias)
    : id_(id),
      time_(time),
      odom_from_imu_(odom_from_imu),
      velocity_odom_m_s_(velocity_odom_m_s),
      imu_bias_(imu_bias) {
  if (!velocity_odom_m_s_.isFinite()) {
    throw std::invalid_argument("NavigationState velocity must be finite");
  }
}

}  // namespace meridian::core
