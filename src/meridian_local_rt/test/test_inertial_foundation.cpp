#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

#include "meridian/local_rt/combined_imu_cost.hpp"
#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/imu_propagator.hpp"
#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt {
namespace {

ImuInterval stationaryInterval() {
  ImuIntegrationSegments segments;
  segments.emplace_back(core::TimeRange(core::TimeNs(0), core::TimeNs(1'000'000'000)),
                        Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.80665));
  return ImuInterval(core::TimeRange(core::TimeNs(0), core::TimeNs(1'000'000'000)),
                     std::move(segments), 2U);
}

TEST(InertialFoundation, UsesResolvedGtsamCombinedPreintegration) {
  const GtsamCombinedPreintegrator preintegrator(ImuModel{});
  const PreintegrationResult result = preintegrator.integrate(stationaryInterval(), {});

  ASSERT_TRUE(result.ok()) << result.error()->message;
  ASSERT_NE(result.value(), nullptr);
  EXPECT_NEAR(result.value()->durationSeconds(), 1.0, 1.0e-12);
  EXPECT_NEAR(result.value()->deltaRotation().w(), 1.0, 1.0e-12);
  EXPECT_NEAR(result.value()->deltaVelocity().z, 9.80665, 1.0e-9);
  EXPECT_NEAR(result.value()->deltaPosition().z, 4.903325, 1.0e-9);
  EXPECT_TRUE(result.value()->covariance().array().isFinite().all());
  EXPECT_GT(result.value()->covariance().diagonal().minCoeff(), 0.0);
  EXPECT_NE(GtsamCombinedPreintegrator::backendName().find("resolved"), std::string_view::npos);
}

TEST(InertialFoundation, DensePropagationCancelsGravityAtRest) {
  const ImuInterval interval = stationaryInterval();
  const core::NavigationState seed(core::StateId(1U), core::TimeNs(0), {}, {}, {});
  const PropagationResult result =
      ImuPropagator(ImuModel{}).propagate(seed, core::StateId(2U), interval);

  ASSERT_TRUE(result.ok()) << result.error()->message;
  ASSERT_EQ(result.value()->samples.size(), 1U);
  EXPECT_NEAR(result.value()->endpoint.odomFromImu().translation().z, 0.0, 1.0e-12);
  EXPECT_NEAR(result.value()->endpoint.velocityOdomMS().z, 0.0, 1.0e-12);
  EXPECT_EQ(result.value()->endpoint.time(), core::TimeNs(1'000'000'000));
  EXPECT_EQ(result.value()->endpoint.id(), core::StateId(2U));
}

TEST(RightSe3Manifold, PlusAndMinusUseTheRightRhoThetaChart) {
  RightSe3Manifold manifold;
  const std::array<double, 7> pose{1.0, -2.0, 0.5, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 6> delta{0.2, -0.1, 0.05, 0.01, -0.02, 0.03};
  std::array<double, 7> moved{};
  std::array<double, 6> recovered{};
  ASSERT_TRUE(manifold.Plus(pose.data(), delta.data(), moved.data()));
  ASSERT_TRUE(manifold.Minus(moved.data(), pose.data(), recovered.data()));
  EXPECT_TRUE(
      (Eigen::Map<const Eigen::Matrix<double, 6, 1>>(recovered.data())
           .isApprox(Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta.data()), 1.0e-10)));

  std::array<double, 42> plus_storage{};
  std::array<double, 42> minus_storage{};
  ASSERT_TRUE(manifold.PlusJacobian(pose.data(), plus_storage.data()));
  ASSERT_TRUE(manifold.MinusJacobian(pose.data(), minus_storage.data()));
  const Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> plus(plus_storage.data());
  const Eigen::Map<const Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> minus(minus_storage.data());
  EXPECT_TRUE((minus * plus).isApprox(Eigen::Matrix<double, 6, 6>::Identity(), 1.0e-12));
}

TEST(CombinedImuCost, IsWhitenedAndItsAmbientPoseJacobianLiftsTheRightChart) {
  const PreintegrationResult preintegration =
      GtsamCombinedPreintegrator(ImuModel{}).integrate(stationaryInterval(), {});
  ASSERT_TRUE(preintegration.ok());
  CombinedImuCost cost(*preintegration.value());
  RightSe3Manifold manifold;

  std::array<double, 7> pose_i{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 9> motion_i{};
  std::array<double, 7> pose_j = pose_i;
  std::array<double, 9> motion_j{};
  std::array<const double*, 4> parameters{pose_i.data(), motion_i.data(), pose_j.data(),
                                          motion_j.data()};
  std::array<double, 15> residual{};
  std::array<double, 105> pose_i_jacobian{};
  std::array<double, 135> motion_i_jacobian{};
  std::array<double, 105> pose_j_jacobian{};
  std::array<double, 135> motion_j_jacobian{};
  std::array<double*, 4> jacobians{pose_i_jacobian.data(), motion_i_jacobian.data(),
                                   pose_j_jacobian.data(), motion_j_jacobian.data()};
  ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), jacobians.data()));
  EXPECT_LT((Eigen::Map<const Eigen::Matrix<double, 15, 1>>(residual.data()).norm()), 1.0e-8);

  std::array<double, 42> plus_storage{};
  ASSERT_TRUE(manifold.PlusJacobian(pose_j.data(), plus_storage.data()));
  const Eigen::Map<const Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> ambient(
      pose_j_jacobian.data());
  const Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> plus(plus_storage.data());
  const Eigen::Matrix<double, 15, 6> analytic = ambient * plus;

  constexpr double kStep = 1.0e-6;
  Eigen::Matrix<double, 15, 6> numeric;
  for (int column = 0; column < 6; ++column) {
    std::array<double, 6> positive_delta{};
    std::array<double, 6> negative_delta{};
    positive_delta[static_cast<std::size_t>(column)] = kStep;
    negative_delta[static_cast<std::size_t>(column)] = -kStep;
    std::array<double, 7> positive_pose{};
    std::array<double, 7> negative_pose{};
    ASSERT_TRUE(manifold.Plus(pose_j.data(), positive_delta.data(), positive_pose.data()));
    ASSERT_TRUE(manifold.Plus(pose_j.data(), negative_delta.data(), negative_pose.data()));
    parameters[2] = positive_pose.data();
    std::array<double, 15> positive_residual{};
    ASSERT_TRUE(cost.Evaluate(parameters.data(), positive_residual.data(), nullptr));
    parameters[2] = negative_pose.data();
    std::array<double, 15> negative_residual{};
    ASSERT_TRUE(cost.Evaluate(parameters.data(), negative_residual.data(), nullptr));
    numeric.col(column) =
        (Eigen::Map<const Eigen::Matrix<double, 15, 1>>(positive_residual.data()) -
         Eigen::Map<const Eigen::Matrix<double, 15, 1>>(negative_residual.data())) /
        (2.0 * kStep);
  }
  parameters[2] = pose_j.data();
  EXPECT_TRUE(analytic.isApprox(numeric, 2.0e-4));
}

TEST(CombinedImuCost, AllLocalJacobiansMatchNumericalDerivativesAtNonidentityState) {
  ImuIntegrationSegments segments;
  segments.emplace_back(core::TimeRange(core::TimeNs(0), core::TimeNs(400'000'000)),
                        Eigen::Vector3d(0.08, -0.04, 0.06), Eigen::Vector3d(0.3, -0.2, 9.7));
  segments.emplace_back(core::TimeRange(core::TimeNs(400'000'000), core::TimeNs(1'000'000'000)),
                        Eigen::Vector3d(-0.03, 0.07, 0.02), Eigen::Vector3d(-0.1, 0.4, 9.9));
  const ImuInterval interval(core::TimeRange(core::TimeNs(0), core::TimeNs(1'000'000'000)),
                             std::move(segments), 3U);
  const core::ImuBias linearization_bias({.x = 0.004, .y = -0.006, .z = 0.003},
                                         {.x = 0.08, .y = -0.05, .z = 0.03});
  const PreintegrationResult preintegration =
      GtsamCombinedPreintegrator(ImuModel{}).integrate(interval, linearization_bias);
  ASSERT_TRUE(preintegration.ok());
  CombinedImuCost cost(*preintegration.value());
  RightSe3Manifold manifold;

  const Eigen::Quaterniond quaternion_i(
      Eigen::AngleAxisd(0.37, Eigen::Vector3d(0.2, -0.7, 0.4).normalized()));
  const Eigen::Quaterniond quaternion_j(
      Eigen::AngleAxisd(-0.29, Eigen::Vector3d(-0.6, 0.1, 0.5).normalized()));
  std::array<double, 7> pose_i{
      0.4, -0.8, 1.2, quaternion_i.x(), quaternion_i.y(), quaternion_i.z(), quaternion_i.w()};
  std::array<double, 9> motion_i{0.7, -0.3, 0.2, 0.012, -0.018, 0.01, 0.11, -0.07, 0.04};
  std::array<double, 7> pose_j{
      -0.2, 0.5, 1.0, quaternion_j.x(), quaternion_j.y(), quaternion_j.z(), quaternion_j.w()};
  std::array<double, 9> motion_j{0.2, 0.6, -0.1, 0.008, -0.004, 0.016, 0.03, -0.02, 0.09};
  std::array<const double*, 4> parameters{pose_i.data(), motion_i.data(), pose_j.data(),
                                          motion_j.data()};

  std::array<double, 15> residual{};
  std::array<std::array<double, 135>, 4> ambient_storage{};
  std::array<double*, 4> ambient_jacobians{ambient_storage[0].data(), ambient_storage[1].data(),
                                           ambient_storage[2].data(), ambient_storage[3].data()};
  ASSERT_TRUE(cost.Evaluate(parameters.data(), residual.data(), ambient_jacobians.data()));

  constexpr double kStep = 1.0e-6;
  for (std::size_t block = 0U; block < parameters.size(); ++block) {
    const bool pose_block = block == 0U || block == 2U;
    const int local_size = pose_block ? 6 : 9;
    Eigen::MatrixXd analytic(15, local_size);
    if (pose_block) {
      std::array<double, 42> plus_storage{};
      ASSERT_TRUE(manifold.PlusJacobian(parameters[block], plus_storage.data()));
      const Eigen::Map<const Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> ambient(
          ambient_storage[block].data());
      const Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> plus(
          plus_storage.data());
      analytic = ambient * plus;
    } else {
      analytic = Eigen::Map<const Eigen::Matrix<double, 15, 9, Eigen::RowMajor>>(
          ambient_storage[block].data());
    }

    Eigen::MatrixXd numeric(15, local_size);
    for (int column = 0; column < local_size; ++column) {
      std::array<const double*, 4> positive_parameters = parameters;
      std::array<const double*, 4> negative_parameters = parameters;
      std::array<double, 7> positive_pose{};
      std::array<double, 7> negative_pose{};
      std::array<double, 9> positive_motion{};
      std::array<double, 9> negative_motion{};
      if (pose_block) {
        std::array<double, 6> positive_delta{};
        std::array<double, 6> negative_delta{};
        positive_delta[static_cast<std::size_t>(column)] = kStep;
        negative_delta[static_cast<std::size_t>(column)] = -kStep;
        ASSERT_TRUE(manifold.Plus(parameters[block], positive_delta.data(), positive_pose.data()));
        ASSERT_TRUE(manifold.Plus(parameters[block], negative_delta.data(), negative_pose.data()));
        positive_parameters[block] = positive_pose.data();
        negative_parameters[block] = negative_pose.data();
      } else {
        std::copy_n(parameters[block], positive_motion.size(), positive_motion.begin());
        negative_motion = positive_motion;
        positive_motion[static_cast<std::size_t>(column)] += kStep;
        negative_motion[static_cast<std::size_t>(column)] -= kStep;
        positive_parameters[block] = positive_motion.data();
        negative_parameters[block] = negative_motion.data();
      }
      std::array<double, 15> positive_residual{};
      std::array<double, 15> negative_residual{};
      ASSERT_TRUE(cost.Evaluate(positive_parameters.data(), positive_residual.data(), nullptr));
      ASSERT_TRUE(cost.Evaluate(negative_parameters.data(), negative_residual.data(), nullptr));
      numeric.col(column) =
          (Eigen::Map<const Eigen::Matrix<double, 15, 1>>(positive_residual.data()) -
           Eigen::Map<const Eigen::Matrix<double, 15, 1>>(negative_residual.data())) /
          (2.0 * kStep);
    }

    EXPECT_TRUE(analytic.isApprox(numeric, 5.0e-4))
        << "block " << block << " maximum error=" << (analytic - numeric).cwiseAbs().maxCoeff();
  }
}

TEST(CombinedImuCost, SolvesTheFreeEndpointInATwoStateCeresProblem) {
  const PreintegrationResult preintegration =
      GtsamCombinedPreintegrator(ImuModel{}).integrate(stationaryInterval(), {});
  ASSERT_TRUE(preintegration.ok());

  std::array<double, 7> pose_i{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 9> motion_i{};
  std::array<double, 7> pose_j{0.15, -0.08, 0.12, 0.01, -0.015, 0.02, 0.9996374343};
  Eigen::Map<Eigen::Quaterniond>(pose_j.data() + 3).normalize();
  std::array<double, 9> motion_j{0.1, -0.05, 0.08, 0.001, -0.002, 0.0015, 0.02, -0.01, 0.03};

  ceres::Problem problem;
  problem.AddParameterBlock(pose_i.data(), 7, new RightSe3Manifold());
  problem.AddParameterBlock(motion_i.data(), 9);
  problem.AddParameterBlock(pose_j.data(), 7, new RightSe3Manifold());
  problem.AddParameterBlock(motion_j.data(), 9);
  problem.AddResidualBlock(new CombinedImuCost(*preintegration.value()), nullptr, pose_i.data(),
                           motion_i.data(), pose_j.data(), motion_j.data());
  problem.SetParameterBlockConstant(pose_i.data());
  problem.SetParameterBlockConstant(motion_i.data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 30;
  options.function_tolerance = 1.0e-12;
  options.gradient_tolerance = 1.0e-12;
  options.parameter_tolerance = 1.0e-12;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  EXPECT_TRUE(summary.IsSolutionUsable()) << summary.BriefReport();
  EXPECT_LT(Eigen::Map<const Eigen::Vector3d>(pose_j.data()).norm(), 1.0e-6);
  EXPECT_NEAR(std::abs(Eigen::Map<const Eigen::Quaterniond>(pose_j.data() + 3).w()), 1.0, 1.0e-8);
  const double motion_norm = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(motion_j.data()).norm();
  EXPECT_LT(motion_norm, 1.0e-6);
}

}  // namespace
}  // namespace meridian::local_rt
