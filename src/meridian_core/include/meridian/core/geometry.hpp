#pragma once

#include <cstddef>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "meridian/core/strong_id.hpp"

namespace meridian::core {

using Pose3d = Sophus::SE3d;
using Vector3d = Eigen::Vector3d;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

enum class PoseTangentConvention {
  RightTranslationFirst,
};

struct PoseCovariance {
  Matrix6d matrix{Matrix6d::Zero()};
  PoseTangentConvention tangent{PoseTangentConvention::RightTranslationFirst};

  [[nodiscard]] bool finite() const noexcept { return matrix.allFinite(); }
};

struct RankAwareInformation {
  Eigen::Matrix<double, 6, 6> basis{Eigen::Matrix<double, 6, 6>::Identity()};
  Eigen::Matrix<double, 6, 1> eigenvalues{Eigen::Matrix<double, 6, 1>::Zero()};
  std::size_t rank{};
  PoseTangentConvention tangent{PoseTangentConvention::RightTranslationFirst};

  [[nodiscard]] bool finite() const noexcept {
    return basis.allFinite() && eigenvalues.allFinite() && rank <= 6;
  }
};

struct NavStateEstimate {
  Pose3d T_odom_imu;
  Vector3d velocity_odom{Vector3d::Zero()};
  Vector3d gyro_bias{Vector3d::Zero()};
  Vector3d accel_bias{Vector3d::Zero()};
};

}  // namespace meridian::core
