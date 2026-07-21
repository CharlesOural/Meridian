#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "meridian/core/geometry.hpp"
#include "meridian/core/navigation.hpp"
#include "meridian/local_rt/config.hpp"
#include "meridian/local_rt/imu_types.hpp"

namespace meridian::local_rt {

class CombinedImuCost;

// ROS- and GTSAM-free public view of one combined IMU preintegration.
class CombinedPreintegration final {
public:
  [[nodiscard]] double durationSeconds() const noexcept { return duration_seconds_; }
  [[nodiscard]] const core::Quaterniond& deltaRotation() const noexcept { return delta_rotation_; }
  [[nodiscard]] const core::Vec3d& deltaPosition() const noexcept { return delta_position_; }
  [[nodiscard]] const core::Vec3d& deltaVelocity() const noexcept { return delta_velocity_; }
  // GTSAM combined-preintegration order: rotation, position, velocity,
  // accelerometer bias, gyroscope bias.
  [[nodiscard]] const Eigen::Matrix<double, 15, 15>& covariance() const noexcept {
    return covariance_;
  }
  [[nodiscard]] const core::ImuBias& linearizationBias() const noexcept {
    return linearization_bias_;
  }

private:
  struct Impl;

  CombinedPreintegration(double duration_seconds, core::Quaterniond delta_rotation,
                         core::Vec3d delta_position, core::Vec3d delta_velocity,
                         Eigen::Matrix<double, 15, 15> covariance, core::ImuBias linearization_bias,
                         std::shared_ptr<const Impl> impl);

  double duration_seconds_{};
  core::Quaterniond delta_rotation_{};
  core::Vec3d delta_position_{};
  core::Vec3d delta_velocity_{};
  Eigen::Matrix<double, 15, 15> covariance_;
  core::ImuBias linearization_bias_{};
  std::shared_ptr<const Impl> impl_;

  friend class GtsamCombinedPreintegrator;
  friend class CombinedImuCost;
};

enum class PreintegrationErrorCode : std::uint8_t {
  kEmptyInterval,
  kInvalidMeasurement,
  kBackendFailure,
};

struct PreintegrationFailure final {
  PreintegrationErrorCode code;
  std::string message;
};

class PreintegrationResult final {
public:
  explicit PreintegrationResult(CombinedPreintegration preintegration);
  explicit PreintegrationResult(PreintegrationFailure failure);

  [[nodiscard]] bool ok() const noexcept;
  [[nodiscard]] const CombinedPreintegration* value() const noexcept;
  [[nodiscard]] const PreintegrationFailure* error() const noexcept;

private:
  std::variant<CombinedPreintegration, PreintegrationFailure> result_;
};

class GtsamCombinedPreintegrator final {
public:
  explicit GtsamCombinedPreintegrator(ImuModel model);

  [[nodiscard]] PreintegrationResult integrate(const ImuInterval& interval,
                                               const core::ImuBias& bias) const;
  [[nodiscard]] static std::string_view backendName() noexcept;

private:
  ImuModel model_;
};

}  // namespace meridian::local_rt
