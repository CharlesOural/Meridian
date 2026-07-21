#include "meridian/local_rt/estimator/fixed_linearization_marginal_prior.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt::estimator {
namespace {

using RowMajor6x7d = Eigen::Matrix<double, 6, 7, Eigen::RowMajor>;
using DynamicRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

int blockLocalSize(const MarginalPriorLinearizationBlock& block) noexcept {
  return std::holds_alternative<PosePriorLinearization>(block) ? 6 : 9;
}

int blockAmbientSize(const MarginalPriorLinearizationBlock& block) noexcept {
  return std::holds_alternative<PosePriorLinearization>(block) ? 7 : 9;
}

MarginalPriorBlockKind kind(const MarginalPriorLinearizationBlock& block) noexcept {
  return std::holds_alternative<PosePriorLinearization>(block) ? MarginalPriorBlockKind::kPose
                                                               : MarginalPriorBlockKind::kMotion;
}

core::StateId stateId(const MarginalPriorLinearizationBlock& block) noexcept {
  if (const auto* pose = std::get_if<PosePriorLinearization>(&block)) {
    return pose->state_id;
  }
  return std::get<MotionPriorLinearization>(block).state_id;
}

bool validPose(const std::array<double, 7>& parameters) noexcept {
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> vector(parameters.data());
  if (!vector.array().isFinite().all()) {
    return false;
  }
  const Eigen::Map<const Eigen::Quaterniond> quaternion(parameters.data() + 3);
  return std::abs(quaternion.squaredNorm() - 1.0) <= 1.0e-8;
}

bool validMotion(const std::array<double, 9>& parameters) noexcept {
  return Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters.data()).array().isFinite().all();
}

gtsam::Pose3 gtsamPose(const double* parameters) {
  const Eigen::Map<const Eigen::Quaterniond> quaternion(parameters + 3);
  return {gtsam::Rot3::Quaternion(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()),
          gtsam::Point3(parameters[0], parameters[1], parameters[2])};
}

Matrix6d poseChartJacobian(const double* current, const std::array<double, 7>& linearization) {
  const gtsam::Pose3 relative = gtsamPose(linearization.data()).between(gtsamPose(current));
  const gtsam::Matrix6 gtsam_jacobian = gtsam::Pose3::LogmapDerivative(relative);

  // GTSAM orders the right SE(3) tangent as [omega, velocity], while Meridian
  // orders it as [rho, theta]. Permute both the input and output coordinates.
  Matrix6d meridian_jacobian;
  meridian_jacobian.topLeftCorner<3, 3>() = gtsam_jacobian.bottomRightCorner<3, 3>();
  meridian_jacobian.topRightCorner<3, 3>() = gtsam_jacobian.bottomLeftCorner<3, 3>();
  meridian_jacobian.bottomLeftCorner<3, 3>() = gtsam_jacobian.topRightCorner<3, 3>();
  meridian_jacobian.bottomRightCorner<3, 3>() = gtsam_jacobian.topLeftCorner<3, 3>();
  return meridian_jacobian;
}

int validateAndLocalSize(const std::vector<MarginalPriorLinearizationBlock>& blocks) {
  if (blocks.empty()) {
    throw std::invalid_argument("marginal prior requires at least one parameter block");
  }

  std::set<std::pair<std::uint64_t, MarginalPriorBlockKind>> identities;
  std::size_t total_local_size = 0U;
  for (const MarginalPriorLinearizationBlock& block : blocks) {
    const bool valid = std::holds_alternative<PosePriorLinearization>(block)
                           ? validPose(std::get<PosePriorLinearization>(block).parameters)
                           : validMotion(std::get<MotionPriorLinearization>(block).parameters);
    if (!valid) {
      throw std::invalid_argument("marginal prior linearization block is invalid");
    }
    if (!identities.emplace(stateId(block).value(), kind(block)).second) {
      throw std::invalid_argument("marginal prior contains a duplicate state/block identity");
    }
    total_local_size += static_cast<std::size_t>(blockLocalSize(block));
  }
  if (total_local_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("marginal prior local dimension is not representable");
  }
  return static_cast<int>(total_local_size);
}

}  // namespace

FixedLinearizationMarginalPriorCost::FixedLinearizationMarginalPriorCost(
    std::vector<MarginalPriorLinearizationBlock> blocks, Eigen::MatrixXd square_root_matrix,
    Eigen::VectorXd right_hand_side)
    : blocks_(std::move(blocks)),
      square_root_matrix_(std::move(square_root_matrix)),
      right_hand_side_(std::move(right_hand_side)) {
  const int expected_local_size = validateAndLocalSize(blocks_);
  if (square_root_matrix_.rows() <= 0 ||
      square_root_matrix_.rows() > static_cast<Eigen::Index>(std::numeric_limits<int>::max()) ||
      square_root_matrix_.cols() != expected_local_size ||
      right_hand_side_.size() != square_root_matrix_.rows() ||
      !square_root_matrix_.array().isFinite().all() || !right_hand_side_.array().isFinite().all()) {
    throw std::invalid_argument(
        "marginal prior matrix/vector dimensions and coefficients must be valid");
  }

  set_num_residuals(static_cast<int>(square_root_matrix_.rows()));
  for (const MarginalPriorLinearizationBlock& block : blocks_) {
    mutable_parameter_block_sizes()->push_back(blockAmbientSize(block));
  }
}

bool FixedLinearizationMarginalPriorCost::buildDelta(
    double const* const* parameters, Eigen::VectorXd& delta,
    MarginalPriorChartDisplacement* displacement) const {
  if (parameters == nullptr) {
    return false;
  }
  delta.resize(square_root_matrix_.cols());
  if (displacement != nullptr) {
    displacement->blocks.clear();
    displacement->blocks.reserve(blocks_.size());
    displacement->maximum_pose_translation_m = 0.0;
    displacement->maximum_pose_rotation_rad = 0.0;
    displacement->maximum_motion_tangent_norm = 0.0;
  }

  RightSe3Manifold manifold;
  Eigen::Index offset = 0;
  for (std::size_t index = 0U; index < blocks_.size(); ++index) {
    if (parameters[index] == nullptr) {
      return false;
    }

    const MarginalPriorLinearizationBlock& block = blocks_[index];
    if (const auto* pose = std::get_if<PosePriorLinearization>(&block)) {
      std::array<double, 6> tangent{};
      if (!manifold.Minus(parameters[index], pose->parameters.data(), tangent.data())) {
        return false;
      }
      const Eigen::Map<const Eigen::Matrix<double, 6, 1>> tangent_map(tangent.data());
      delta.segment<6>(offset) = tangent_map;
      if (displacement != nullptr) {
        const double translation_m = tangent_map.head<3>().norm();
        const double rotation_rad = tangent_map.tail<3>().norm();
        displacement->blocks.push_back({.state_id = pose->state_id,
                                        .kind = MarginalPriorBlockKind::kPose,
                                        .tangent_norm = tangent_map.norm(),
                                        .translation_m = translation_m,
                                        .rotation_rad = rotation_rad});
        displacement->maximum_pose_translation_m =
            std::max(displacement->maximum_pose_translation_m, translation_m);
        displacement->maximum_pose_rotation_rad =
            std::max(displacement->maximum_pose_rotation_rad, rotation_rad);
      }
      offset += 6;
      continue;
    }

    const auto& motion = std::get<MotionPriorLinearization>(block);
    const Eigen::Map<const Eigen::Matrix<double, 9, 1>> current(parameters[index]);
    if (!current.array().isFinite().all()) {
      return false;
    }
    const Eigen::Map<const Eigen::Matrix<double, 9, 1>> linearization(motion.parameters.data());
    const Eigen::Matrix<double, 9, 1> tangent = current - linearization;
    delta.segment<9>(offset) = tangent;
    if (displacement != nullptr) {
      const double tangent_norm = tangent.norm();
      displacement->blocks.push_back({.state_id = motion.state_id,
                                      .kind = MarginalPriorBlockKind::kMotion,
                                      .tangent_norm = tangent_norm,
                                      .translation_m = 0.0,
                                      .rotation_rad = 0.0});
      displacement->maximum_motion_tangent_norm =
          std::max(displacement->maximum_motion_tangent_norm, tangent_norm);
    }
    offset += 9;
  }
  return delta.array().isFinite().all();
}

std::optional<MarginalPriorChartDisplacement>
FixedLinearizationMarginalPriorCost::chartDisplacement(double const* const* parameters) const {
  Eigen::VectorXd delta;
  MarginalPriorChartDisplacement displacement;
  if (!buildDelta(parameters, delta, &displacement)) {
    return std::nullopt;
  }
  return displacement;
}

bool FixedLinearizationMarginalPriorCost::Evaluate(double const* const* parameters,
                                                   double* residuals, double** jacobians) const {
  if (residuals == nullptr) {
    return false;
  }
  Eigen::VectorXd delta;
  if (!buildDelta(parameters, delta, nullptr)) {
    return false;
  }
  Eigen::Map<Eigen::VectorXd>(residuals, num_residuals()) =
      square_root_matrix_ * delta - right_hand_side_;
  if (jacobians == nullptr) {
    return true;
  }

  RightSe3Manifold manifold;
  Eigen::Index local_offset = 0;
  for (std::size_t index = 0U; index < blocks_.size(); ++index) {
    const int block_local_size = blockLocalSize(blocks_[index]);
    const int block_ambient_size = blockAmbientSize(blocks_[index]);
    if (jacobians[index] != nullptr) {
      Eigen::Map<DynamicRowMajor> ambient(jacobians[index], num_residuals(), block_ambient_size);
      if (const auto* pose = std::get_if<PosePriorLinearization>(&blocks_[index])) {
        std::array<double, 42> lift_storage{};
        if (!manifold.MinusJacobian(parameters[index], lift_storage.data())) {
          return false;
        }
        const Eigen::Map<const RowMajor6x7d> lift(lift_storage.data());
        const Matrix6d chart_jacobian = poseChartJacobian(parameters[index], pose->parameters);
        if (!chart_jacobian.array().isFinite().all()) {
          return false;
        }
        ambient =
            square_root_matrix_.middleCols(local_offset, block_local_size) * chart_jacobian * lift;
      } else {
        ambient = square_root_matrix_.middleCols(local_offset, block_local_size);
      }
    }
    local_offset += block_local_size;
  }
  return true;
}

}  // namespace meridian::local_rt::estimator
