#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/local_rt/estimator/scan_to_map_cost.hpp"
#include "meridian/local_rt/right_se3_manifold.hpp"

namespace meridian::local_rt::estimator {
namespace {

std::array<double, 7> poseParameters(const Eigen::Vector3d& translation,
                                     const Eigen::Quaterniond& quaternion) {
  const Eigen::Quaterniond normalized = quaternion.normalized();
  return {translation.x(), translation.y(), translation.z(), normalized.x(),
          normalized.y(),  normalized.z(),  normalized.w()};
}

core::Pose3d imuFromLidar() {
  const Eigen::Quaterniond quaternion(
      Eigen::AngleAxisd(0.17, Eigen::Vector3d(0.2, -0.4, 0.7).normalized()));
  return core::Pose3d(
      {.x = 0.31, .y = -0.08, .z = 0.14},
      core::Quaterniond(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()));
}

Eigen::MatrixXd localJacobian(ceres::CostFunction& cost, std::vector<std::array<double, 7>>& poses,
                              std::size_t differentiated_block) {
  std::vector<const double*> parameters;
  parameters.reserve(poses.size());
  for (const std::array<double, 7>& pose : poses) {
    parameters.push_back(pose.data());
  }

  std::vector<double> residual(static_cast<std::size_t>(cost.num_residuals()));
  std::vector<std::vector<double>> ambient_storage(
      poses.size(), std::vector<double>(static_cast<std::size_t>(cost.num_residuals()) * 7U));
  std::vector<double*> ambient;
  ambient.reserve(poses.size());
  for (std::vector<double>& storage : ambient_storage) {
    ambient.push_back(storage.data());
  }
  EXPECT_TRUE(cost.Evaluate(parameters.data(), residual.data(), ambient.data()));

  RightSe3Manifold manifold;
  std::array<double, 42> plus_storage{};
  EXPECT_TRUE(manifold.PlusJacobian(poses[differentiated_block].data(), plus_storage.data()));
  const Eigen::Map<const Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> plus(plus_storage.data());
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> ambient_map(
      ambient_storage[differentiated_block].data(), cost.num_residuals(), 7);
  return ambient_map * plus;
}

Eigen::MatrixXd numericLocalJacobian(ceres::CostFunction& cost,
                                     std::vector<std::array<double, 7>>& poses,
                                     std::size_t differentiated_block) {
  constexpr double kStep = 1.0e-6;
  Eigen::MatrixXd numeric(cost.num_residuals(), 6);
  RightSe3Manifold manifold;

  for (int column = 0; column < 6; ++column) {
    std::array<double, 6> positive_delta{};
    std::array<double, 6> negative_delta{};
    positive_delta[static_cast<std::size_t>(column)] = kStep;
    negative_delta[static_cast<std::size_t>(column)] = -kStep;
    std::array<double, 7> positive_pose{};
    std::array<double, 7> negative_pose{};
    EXPECT_TRUE(manifold.Plus(poses[differentiated_block].data(), positive_delta.data(),
                              positive_pose.data()));
    EXPECT_TRUE(manifold.Plus(poses[differentiated_block].data(), negative_delta.data(),
                              negative_pose.data()));

    std::vector<const double*> parameters;
    parameters.reserve(poses.size());
    for (std::size_t index = 0U; index < poses.size(); ++index) {
      parameters.push_back(index == differentiated_block ? positive_pose.data()
                                                         : poses[index].data());
    }
    std::vector<double> positive(static_cast<std::size_t>(cost.num_residuals()));
    EXPECT_TRUE(cost.Evaluate(parameters.data(), positive.data(), nullptr));
    parameters[differentiated_block] = negative_pose.data();
    std::vector<double> negative(static_cast<std::size_t>(cost.num_residuals()));
    EXPECT_TRUE(cost.Evaluate(parameters.data(), negative.data(), nullptr));
    numeric.col(column) =
        (Eigen::Map<const Eigen::VectorXd>(positive.data(), cost.num_residuals()) -
         Eigen::Map<const Eigen::VectorXd>(negative.data(), cost.num_residuals())) /
        (2.0 * kStep);
  }
  return numeric;
}

TEST(FinalizedScanToMapCost, LiftsRightLocalJacobiansAtNonidentityPoseAndExtrinsic) {
  std::vector<FinalizedLidarRow> rows{
      {.source_lidar = {1.2, -0.4, 0.7},
       .target_odom = {2.1, -0.1, 1.4},
       .sqrt_weight_over_sigma = 2.5},
      {.source_lidar = {-0.3, 1.1, 0.2},
       .target_odom = {0.4, 1.7, -0.2},
       .sqrt_weight_over_sigma = 0.8},
  };
  FinalizedScanToMapCost cost(std::move(rows), imuFromLidar());
  std::vector<std::array<double, 7>> poses{poseParameters(
      {0.8, -1.3, 0.6},
      Eigen::Quaterniond(Eigen::AngleAxisd(0.43, Eigen::Vector3d(-0.3, 0.8, 0.4).normalized())))};

  const Eigen::MatrixXd analytic = localJacobian(cost, poses, 0U);
  const Eigen::MatrixXd numeric = numericLocalJacobian(cost, poses, 0U);
  EXPECT_TRUE(analytic.isApprox(numeric, 2.0e-6)) << "analytic:\n"
                                                  << analytic << "\nnumeric:\n"
                                                  << numeric;
}

TEST(ActiveOwnerScanToMapCost, HasCorrectSourceAndOwnerRightJacobians) {
  std::vector<ActiveLidarRow> rows{
      {.source_lidar = {2.0, 0.1, -0.5},
       .target_lidar = {1.8, -0.2, -0.4},
       .sqrt_weight_over_sigma = 1.7},
      {.source_lidar = {-0.7, 1.4, 0.9},
       .target_lidar = {-0.5, 1.1, 0.8},
       .sqrt_weight_over_sigma = 0.6},
  };
  ActiveOwnerScanToMapCost cost(std::move(rows), imuFromLidar());
  std::vector<std::array<double, 7>> poses{
      poseParameters({1.1, -0.9, 0.3}, Eigen::Quaterniond(Eigen::AngleAxisd(
                                           0.31, Eigen::Vector3d(0.5, -0.1, 0.8).normalized()))),
      poseParameters({0.9, -0.7, 0.4}, Eigen::Quaterniond(Eigen::AngleAxisd(
                                           -0.22, Eigen::Vector3d(-0.2, 0.9, 0.3).normalized()))),
  };

  for (std::size_t block = 0U; block < poses.size(); ++block) {
    const Eigen::MatrixXd analytic = localJacobian(cost, poses, block);
    const Eigen::MatrixXd numeric = numericLocalJacobian(cost, poses, block);
    EXPECT_TRUE(analytic.isApprox(numeric, 2.0e-6)) << "block " << block << " analytic:\n"
                                                    << analytic << "\nnumeric:\n"
                                                    << numeric;
  }
}

TEST(ScanToMapCost, RejectsEmptyNonfiniteAndNonpositiveRows) {
  EXPECT_THROW(static_cast<void>(FinalizedScanToMapCost({}, imuFromLidar())),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(FinalizedScanToMapCost({{.source_lidar = {1.0, 2.0, 3.0},
                                                          .target_odom = {0.0, 0.0, 0.0},
                                                          .sqrt_weight_over_sigma = 0.0}},
                                                        imuFromLidar())),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ActiveOwnerScanToMapCost(
                   {{.source_lidar = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN()},
                     .target_lidar = {0.0, 0.0, 0.0},
                     .sqrt_weight_over_sigma = 1.0}},
                   imuFromLidar())),
               std::invalid_argument);
}

}  // namespace
}  // namespace meridian::local_rt::estimator
