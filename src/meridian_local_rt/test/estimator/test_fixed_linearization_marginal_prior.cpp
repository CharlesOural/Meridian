#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

#include "meridian/local_rt/estimator/fixed_linearization_marginal_prior.hpp"
#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt::estimator {
namespace {

std::array<double, 7> poseParameters(const Eigen::Vector3d& translation, double angle,
                                     const Eigen::Vector3d& axis) {
  const Eigen::Quaterniond quaternion(Eigen::AngleAxisd(angle, axis.normalized()));
  return {translation.x(), translation.y(), translation.z(), quaternion.x(),
          quaternion.y(),  quaternion.z(),  quaternion.w()};
}

Eigen::MatrixXd testSquareRootMatrix(Eigen::Index rows, Eigen::Index columns) {
  Eigen::MatrixXd matrix(rows, columns);
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index column = 0; column < columns; ++column) {
      matrix(row, column) = 0.13 * std::sin(0.19 * static_cast<double>((row + 1) * (column + 2))) +
                            0.01 * static_cast<double>(row - column);
    }
  }
  return matrix;
}

TEST(FixedLinearizationMarginalPriorCost, MatchesNumericJacobiansInEveryOrderedBlock) {
  const std::array<double, 7> pose_a = poseParameters({0.8, -1.2, 0.4}, 0.37, {0.2, -0.7, 0.5});
  const std::array<double, 9> motion_a{0.4, -0.3, 0.2, 0.01, -0.02, 0.03, -0.1, 0.08, -0.04};
  const std::array<double, 7> pose_b = poseParameters({1.4, -0.7, 0.6}, -0.29, {-0.6, 0.1, 0.8});
  const std::array<double, 9> motion_b{-0.2, 0.5, 0.1, -0.03, 0.02, 0.01, 0.07, -0.06, 0.09};
  std::vector<MarginalPriorLinearizationBlock> blocks{
      PosePriorLinearization{core::StateId(10U), pose_a},
      MotionPriorLinearization{core::StateId(10U), motion_a},
      PosePriorLinearization{core::StateId(11U), pose_b},
      MotionPriorLinearization{core::StateId(11U), motion_b},
  };
  const Eigen::MatrixXd square_root = testSquareRootMatrix(19, 30);
  const Eigen::VectorXd right_hand_side = Eigen::VectorXd::LinSpaced(19, -0.4, 0.6);
  FixedLinearizationMarginalPriorCost cost(blocks, square_root, right_hand_side);

  std::vector<std::vector<double>> values{
      std::vector<double>(pose_a.begin(), pose_a.end()),
      std::vector<double>(motion_a.begin(), motion_a.end()),
      std::vector<double>(pose_b.begin(), pose_b.end()),
      std::vector<double>(motion_b.begin(), motion_b.end()),
  };
  std::vector<const double*> parameters;
  for (const std::vector<double>& value : values) {
    parameters.push_back(value.data());
  }
  std::vector<double> residual(static_cast<std::size_t>(cost.num_residuals()));
  ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), nullptr));
  EXPECT_TRUE(Eigen::Map<const Eigen::VectorXd>(residual.data(), cost.num_residuals())
                  .isApprox(-right_hand_side, 1.0e-12));

  constexpr double kStep = 1.0e-6;
  RightSe3Manifold manifold;
  Eigen::Index expected_column_offset = 0;
  for (std::size_t block_index = 0U; block_index < blocks.size(); ++block_index) {
    const bool is_pose = std::holds_alternative<PosePriorLinearization>(blocks[block_index]);
    const int ambient_size = is_pose ? 7 : 9;
    const int local_size = is_pose ? 6 : 9;
    std::vector<double> ambient_storage(
        static_cast<std::size_t>(cost.num_residuals() * ambient_size));
    std::vector<double*> jacobians(blocks.size(), nullptr);
    jacobians[block_index] = ambient_storage.data();
    ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), jacobians.data()));
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        ambient(ambient_storage.data(), cost.num_residuals(), ambient_size);

    Eigen::MatrixXd analytic;
    if (is_pose) {
      std::array<double, 42> plus_storage{};
      ASSERT_TRUE(manifold.PlusJacobian(values[block_index].data(), plus_storage.data()));
      analytic = ambient * Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>>(
                               plus_storage.data());
    } else {
      analytic = ambient;
    }

    Eigen::MatrixXd numeric(cost.num_residuals(), local_size);
    for (int column = 0; column < local_size; ++column) {
      std::vector<double> positive = values[block_index];
      std::vector<double> negative = values[block_index];
      if (is_pose) {
        std::array<double, 6> positive_delta{};
        std::array<double, 6> negative_delta{};
        positive_delta[static_cast<std::size_t>(column)] = kStep;
        negative_delta[static_cast<std::size_t>(column)] = -kStep;
        ASSERT_TRUE(
            manifold.Plus(values[block_index].data(), positive_delta.data(), positive.data()));
        ASSERT_TRUE(
            manifold.Plus(values[block_index].data(), negative_delta.data(), negative.data()));
      } else {
        positive[static_cast<std::size_t>(column)] += kStep;
        negative[static_cast<std::size_t>(column)] -= kStep;
      }

      parameters[block_index] = positive.data();
      std::vector<double> positive_residual(static_cast<std::size_t>(cost.num_residuals()));
      ASSERT_TRUE(cost.Evaluate(parameters.data(), positive_residual.data(), nullptr));
      parameters[block_index] = negative.data();
      std::vector<double> negative_residual(static_cast<std::size_t>(cost.num_residuals()));
      ASSERT_TRUE(cost.Evaluate(parameters.data(), negative_residual.data(), nullptr));
      parameters[block_index] = values[block_index].data();
      numeric.col(column) =
          (Eigen::Map<const Eigen::VectorXd>(positive_residual.data(), cost.num_residuals()) -
           Eigen::Map<const Eigen::VectorXd>(negative_residual.data(), cost.num_residuals())) /
          (2.0 * kStep);
    }
    EXPECT_TRUE(analytic.isApprox(numeric, 3.0e-7));
    EXPECT_TRUE(
        analytic.isApprox(square_root.middleCols(expected_column_offset, local_size), 1.0e-12));
    expected_column_offset += local_size;
  }
}

TEST(FixedLinearizationMarginalPriorCost,
     MatchesNumericPoseJacobiansAtNonzeroTranslationAndRotationDisplacement) {
  const std::array<std::array<double, 7>, 2> linearization{
      poseParameters({0.8, -1.2, 0.4}, 0.37, {0.2, -0.7, 0.5}),
      poseParameters({1.4, -0.7, 0.6}, -0.29, {-0.6, 0.1, 0.8}),
  };
  const std::array<std::array<double, 6>, 2> chart_displacement{
      std::array<double, 6>{0.72, -0.41, 0.33, 0.31, -0.22, 0.17},
      std::array<double, 6>{-0.54, 0.38, 0.46, -0.28, 0.24, 0.19},
  };
  const std::vector<MarginalPriorLinearizationBlock> blocks{
      PosePriorLinearization{core::StateId(20U), linearization[0]},
      PosePriorLinearization{core::StateId(21U), linearization[1]},
  };
  const Eigen::Matrix<double, 12, 12> square_root = Eigen::Matrix<double, 12, 12>::Identity();
  const Eigen::VectorXd right_hand_side = Eigen::VectorXd::LinSpaced(12, -0.3, 0.4);
  FixedLinearizationMarginalPriorCost cost(blocks, square_root, right_hand_side);

  RightSe3Manifold manifold;
  std::array<std::array<double, 7>, 2> current{};
  for (std::size_t index = 0U; index < current.size(); ++index) {
    ASSERT_TRUE(manifold.Plus(linearization[index].data(), chart_displacement[index].data(),
                              current[index].data()));
  }
  std::array<const double*, 2> parameters{current[0].data(), current[1].data()};

  constexpr double kStep = 1.0e-7;
  for (std::size_t block_index = 0U; block_index < current.size(); ++block_index) {
    std::vector<double> ambient_storage(static_cast<std::size_t>(cost.num_residuals() * 7));
    std::array<double*, 2> jacobians{nullptr, nullptr};
    jacobians[block_index] = ambient_storage.data();
    std::vector<double> residual(static_cast<std::size_t>(cost.num_residuals()));
    ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), jacobians.data()));
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        ambient(ambient_storage.data(), cost.num_residuals(), 7);
    std::array<double, 42> plus_storage{};
    ASSERT_TRUE(manifold.PlusJacobian(current[block_index].data(), plus_storage.data()));
    const Eigen::MatrixXd analytic =
        ambient *
        Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>>(plus_storage.data());

    Eigen::MatrixXd numeric(cost.num_residuals(), 6);
    for (int column = 0; column < 6; ++column) {
      std::array<double, 6> positive_delta{};
      std::array<double, 6> negative_delta{};
      positive_delta[static_cast<std::size_t>(column)] = kStep;
      negative_delta[static_cast<std::size_t>(column)] = -kStep;
      std::array<double, 7> positive{};
      std::array<double, 7> negative{};
      ASSERT_TRUE(
          manifold.Plus(current[block_index].data(), positive_delta.data(), positive.data()));
      ASSERT_TRUE(
          manifold.Plus(current[block_index].data(), negative_delta.data(), negative.data()));

      parameters[block_index] = positive.data();
      std::vector<double> positive_residual(static_cast<std::size_t>(cost.num_residuals()));
      ASSERT_TRUE(cost.Evaluate(parameters.data(), positive_residual.data(), nullptr));
      parameters[block_index] = negative.data();
      std::vector<double> negative_residual(static_cast<std::size_t>(cost.num_residuals()));
      ASSERT_TRUE(cost.Evaluate(parameters.data(), negative_residual.data(), nullptr));
      parameters[block_index] = current[block_index].data();
      numeric.col(column) =
          (Eigen::Map<const Eigen::VectorXd>(positive_residual.data(), cost.num_residuals()) -
           Eigen::Map<const Eigen::VectorXd>(negative_residual.data(), cost.num_residuals())) /
          (2.0 * kStep);
    }

    EXPECT_TRUE(analytic.isApprox(numeric, 5.0e-7))
        << "maximum Jacobian error: " << (analytic - numeric).cwiseAbs().maxCoeff();
    const Eigen::Index local_offset = static_cast<Eigen::Index>(6U * block_index);
    EXPECT_GT((analytic - square_root.middleCols(local_offset, 6)).norm(), 1.0e-2);
    EXPECT_GT(analytic.block(local_offset, 3, 3, 3).norm(), 1.0e-2);
  }
}

TEST(FixedLinearizationMarginalPriorCost, ReportsChartMotionAndIsQuaternionSignInvariant) {
  const std::array<double, 7> pose = poseParameters({0.2, -0.1, 0.7}, 0.51, {0.4, 0.3, -0.8});
  const std::array<double, 9> motion{};
  std::vector<MarginalPriorLinearizationBlock> blocks{
      PosePriorLinearization{core::StateId(42U), pose},
      MotionPriorLinearization{core::StateId(42U), motion},
  };
  FixedLinearizationMarginalPriorCost cost(blocks, Eigen::Matrix<double, 15, 15>::Identity(),
                                           Eigen::VectorXd::Zero(15));

  std::array<double, 6> pose_delta{0.3, -0.2, 0.1, 0.04, -0.05, 0.02};
  std::array<double, 7> moved_pose{};
  RightSe3Manifold manifold;
  ASSERT_TRUE(manifold.Plus(pose.data(), pose_delta.data(), moved_pose.data()));
  std::array<double, 9> moved_motion = motion;
  moved_motion[0] = 0.6;
  moved_motion[4] = -0.2;
  std::array<const double*, 2> moved_parameters{moved_pose.data(), moved_motion.data()};
  const auto displacement = cost.chartDisplacement(moved_parameters.data());
  ASSERT_TRUE(displacement.has_value());
  ASSERT_EQ(displacement->blocks.size(), 2U);
  EXPECT_NEAR(displacement->maximum_pose_translation_m,
              Eigen::Map<const Eigen::Vector3d>(pose_delta.data()).norm(), 1.0e-12);
  EXPECT_NEAR(displacement->maximum_pose_rotation_rad,
              Eigen::Map<const Eigen::Vector3d>(pose_delta.data() + 3).norm(), 1.0e-12);
  EXPECT_NEAR(displacement->maximum_motion_tangent_norm, std::sqrt(0.4), 1.0e-12);

  std::array<double, 7> negated_pose = pose;
  for (std::size_t index = 3U; index < negated_pose.size(); ++index) {
    negated_pose[index] = -negated_pose[index];
  }
  std::array<const double*, 2> equivalent_parameters{negated_pose.data(), motion.data()};
  std::array<double, 15> residual{};
  ASSERT_TRUE(cost.Evaluate(equivalent_parameters.data(), residual.data(), nullptr));
  EXPECT_LT((Eigen::Map<const Eigen::Matrix<double, 15, 1>>(residual.data()).norm()), 1.0e-12);
}

TEST(FixedLinearizationMarginalPriorCost, RejectsDuplicateBlocksAndWrongMatrixShape) {
  const std::array<double, 7> pose =
      poseParameters(Eigen::Vector3d::Zero(), 0.2, Eigen::Vector3d::UnitZ());
  const std::vector<MarginalPriorLinearizationBlock> duplicate{
      PosePriorLinearization{core::StateId(1U), pose},
      PosePriorLinearization{core::StateId(1U), pose},
  };
  EXPECT_THROW(static_cast<void>(FixedLinearizationMarginalPriorCost(
                   duplicate, Eigen::MatrixXd::Identity(12, 12), Eigen::VectorXd::Zero(12))),
               std::invalid_argument);

  const std::vector<MarginalPriorLinearizationBlock> one{
      PosePriorLinearization{core::StateId(2U), pose}};
  EXPECT_THROW(static_cast<void>(FixedLinearizationMarginalPriorCost(
                   one, Eigen::MatrixXd::Identity(6, 7), Eigen::VectorXd::Zero(6))),
               std::invalid_argument);
}

}  // namespace
}  // namespace meridian::local_rt::estimator
