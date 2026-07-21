#include "meridian/local_rt/right_se3_manifold.hpp"

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace meridian::local_rt {
namespace {

using RowMajor7x6 = Eigen::Matrix<double, 7, 6, Eigen::RowMajor>;
using RowMajor6x7 = Eigen::Matrix<double, 6, 7, Eigen::RowMajor>;

Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

bool validPose(const double* value) {
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> pose(value);
  if (!pose.array().isFinite().all()) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(value + 3);
  return std::abs(quaternion.squaredNorm() - 1.0) <= 1.0e-8;
}

gtsam::Pose3 pose(const double* value) {
  const Eigen::Map<const Eigen::Quaterniond> quaternion(value + 3);
  return {gtsam::Rot3::Quaternion(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()),
          gtsam::Point3(value[0], value[1], value[2])};
}

void writePose(const gtsam::Pose3& value, double* destination) {
  const gtsam::Point3 translation = value.translation();
  const Eigen::Quaterniond quaternion(value.rotation().toQuaternion());
  destination[0] = translation.x();
  destination[1] = translation.y();
  destination[2] = translation.z();
  destination[3] = quaternion.x();
  destination[4] = quaternion.y();
  destination[5] = quaternion.z();
  destination[6] = quaternion.w();
}

}  // namespace

bool RightSe3Manifold::Plus(const double* x, const double* delta, double* x_plus_delta) const {
  if (!validPose(x)) {
    return false;
  }
  const Eigen::Map<const Eigen::Matrix<double, 6, 1>> meridian_delta(delta);
  if (!meridian_delta.array().isFinite().all()) {
    return false;
  }
  gtsam::Vector6 gtsam_delta;
  gtsam_delta.head<3>() = meridian_delta.tail<3>();
  gtsam_delta.tail<3>() = meridian_delta.head<3>();
  writePose(pose(x).compose(gtsam::Pose3::Expmap(gtsam_delta)), x_plus_delta);
  return true;
}

bool RightSe3Manifold::PlusJacobian(const double* x, double* jacobian) const {
  if (!validPose(x)) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(x + 3);
  Eigen::Map<RowMajor7x6> result(jacobian);
  result.setZero();
  result.block<3, 3>(0, 0) = quaternion.toRotationMatrix();
  result.block<3, 3>(3, 3) =
      0.5 * (quaternion.w() * Eigen::Matrix3d::Identity() + skew(quaternion.vec()));
  result.block<1, 3>(6, 3) = -0.5 * quaternion.vec().transpose();
  return true;
}

bool RightSe3Manifold::Minus(const double* y, const double* x, double* y_minus_x) const {
  if (!validPose(x) || !validPose(y)) {
    return false;
  }
  const gtsam::Vector6 gtsam_delta = gtsam::Pose3::Logmap(pose(x).between(pose(y)));
  Eigen::Map<Eigen::Matrix<double, 6, 1>> meridian_delta(y_minus_x);
  meridian_delta.head<3>() = gtsam_delta.tail<3>();
  meridian_delta.tail<3>() = gtsam_delta.head<3>();
  return meridian_delta.array().isFinite().all();
}

bool RightSe3Manifold::MinusJacobian(const double* x, double* jacobian) const {
  if (!validPose(x)) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(x + 3);
  Eigen::Map<RowMajor6x7> result(jacobian);
  result.setZero();
  result.block<3, 3>(0, 0) = quaternion.toRotationMatrix().transpose();

  Eigen::Matrix<double, 4, 3> quaternion_plus_jacobian;
  quaternion_plus_jacobian.topRows<3>() =
      0.5 * (quaternion.w() * Eigen::Matrix3d::Identity() + skew(quaternion.vec()));
  quaternion_plus_jacobian.bottomRows<1>() = -0.5 * quaternion.vec().transpose();
  result.block<3, 4>(3, 3) = 4.0 * quaternion_plus_jacobian.transpose();
  return true;
}

}  // namespace meridian::local_rt
