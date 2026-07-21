#pragma once

#include "meridian/core/geometry.hpp"
#include "meridian/core/ids.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

class ImuBias final {
public:
  constexpr ImuBias() noexcept = default;
  ImuBias(Vec3d gyroscope_rad_s, Vec3d accelerometer_m_s2);

  [[nodiscard]] constexpr const Vec3d& gyroscopeRadS() const noexcept { return gyroscope_rad_s_; }
  [[nodiscard]] constexpr const Vec3d& accelerometerMS2() const noexcept {
    return accelerometer_m_s2_;
  }

  friend bool operator==(const ImuBias&, const ImuBias&) noexcept = default;

private:
  Vec3d gyroscope_rad_s_{};
  Vec3d accelerometer_m_s2_{};
};

// Canonical local state. odomFromImu() is T_odom_imu, velocity is expressed in
// odom, and both bias vectors are expressed in the IMU frame.
class NavigationState final {
public:
  NavigationState(StateId id, TimeNs time, Pose3d odom_from_imu, Vec3d velocity_odom_m_s,
                  ImuBias imu_bias);

  [[nodiscard]] constexpr StateId id() const noexcept { return id_; }
  [[nodiscard]] constexpr TimeNs time() const noexcept { return time_; }
  [[nodiscard]] constexpr const Pose3d& odomFromImu() const noexcept { return odom_from_imu_; }
  [[nodiscard]] constexpr const Vec3d& velocityOdomMS() const noexcept {
    return velocity_odom_m_s_;
  }
  [[nodiscard]] constexpr const ImuBias& imuBias() const noexcept { return imu_bias_; }

  friend bool operator==(const NavigationState&, const NavigationState&) noexcept = default;

private:
  StateId id_;
  TimeNs time_;
  Pose3d odom_from_imu_;
  Vec3d velocity_odom_m_s_;
  ImuBias imu_bias_;
};

}  // namespace meridian::core
