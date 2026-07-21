#include "meridian/core/geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace meridian::core {
namespace {

constexpr double kUnitQuaternionSquaredNormTolerance = 1.0e-9;

}  // namespace

bool Vec3d::isFinite() const noexcept {
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

Quaterniond::Quaterniond(double w, double x, double y, double z) : w_(w), x_(x), y_(y), z_(z) {
  if (!isFinite()) {
    throw std::invalid_argument("Quaterniond coefficients must be finite");
  }
  if (std::abs(squaredNorm() - 1.0) > kUnitQuaternionSquaredNormTolerance) {
    throw std::invalid_argument("Quaterniond coefficients must have unit norm");
  }
}

bool Quaterniond::isFinite() const noexcept {
  return std::isfinite(w_) && std::isfinite(x_) && std::isfinite(y_) && std::isfinite(z_);
}

double Quaterniond::squaredNorm() const noexcept {
  return w_ * w_ + x_ * x_ + y_ * y_ + z_ * z_;
}

Pose3d::Pose3d(Vec3d translation, Quaterniond rotation)
    : translation_(translation), rotation_(rotation) {
  if (!translation_.isFinite()) {
    throw std::invalid_argument("Pose3d translation must be finite");
  }
}

}  // namespace meridian::core
