#pragma once

#include <ceres/cost_function.h>

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include "meridian/core/ids.hpp"

namespace meridian::local_rt::estimator {

struct PosePriorLinearization final {
  core::StateId state_id;
  // T_odom_imu, encoded as [px, py, pz, qx, qy, qz, qw].
  std::array<double, 7> parameters;
};

struct MotionPriorLinearization final {
  core::StateId state_id;
  // [velocity_odom, gyroscope_bias_imu, accelerometer_bias_imu].
  std::array<double, 9> parameters;
};

using MarginalPriorLinearizationBlock =
    std::variant<PosePriorLinearization, MotionPriorLinearization>;

enum class MarginalPriorBlockKind : std::uint8_t {
  kPose,
  kMotion,
};

struct MarginalPriorBlockDisplacement final {
  core::StateId state_id;
  MarginalPriorBlockKind kind;
  double tangent_norm{};
  double translation_m{};
  double rotation_rad{};
};

struct MarginalPriorChartDisplacement final {
  std::vector<MarginalPriorBlockDisplacement> blocks;
  double maximum_pose_translation_m{};
  double maximum_pose_rotation_rad{};
  double maximum_motion_tangent_norm{};
};

// A fixed-linearization square-root prior over an arbitrary ordered list of
// pose and motion blocks. A has one column per local degree of freedom (6 for
// pose, 9 for motion), and the residual is
//
//                         A * delta(x, x0) - b.
//
// Pose deltas use RightSe3Manifold::Minus. Returned ambient Jacobians lift the
// frozen local columns through RightSe3Manifold::MinusJacobian, deliberately
// preserving the first-estimate local Jacobian while the prior remains valid
// inside its monitored chart.
class FixedLinearizationMarginalPriorCost final : public ceres::CostFunction {
public:
  FixedLinearizationMarginalPriorCost(std::vector<MarginalPriorLinearizationBlock> blocks,
                                      Eigen::MatrixXd square_root_matrix,
                                      Eigen::VectorXd right_hand_side);

  FixedLinearizationMarginalPriorCost(const FixedLinearizationMarginalPriorCost&) = delete;
  FixedLinearizationMarginalPriorCost& operator=(const FixedLinearizationMarginalPriorCost&) =
      delete;

  [[nodiscard]] const std::vector<MarginalPriorLinearizationBlock>& blocks() const noexcept {
    return blocks_;
  }
  [[nodiscard]] const Eigen::MatrixXd& squareRootMatrix() const noexcept {
    return square_root_matrix_;
  }
  [[nodiscard]] const Eigen::VectorXd& rightHandSide() const noexcept { return right_hand_side_; }
  [[nodiscard]] int localSize() const noexcept {
    return static_cast<int>(square_root_matrix_.cols());
  }

  [[nodiscard]] std::optional<MarginalPriorChartDisplacement> chartDisplacement(
      double const* const* parameters) const;

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

private:
  [[nodiscard]] bool buildDelta(double const* const* parameters, Eigen::VectorXd& delta,
                                MarginalPriorChartDisplacement* displacement) const;

  std::vector<MarginalPriorLinearizationBlock> blocks_;
  Eigen::MatrixXd square_root_matrix_;
  Eigen::VectorXd right_hand_side_;
};

}  // namespace meridian::local_rt::estimator
