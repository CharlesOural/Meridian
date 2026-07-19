#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <utility>

#include "meridian/global/gnss_interpolation.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
}

[[nodiscard]] GnssInterpolationInput nominalInput() {
  GnssInterpolationInput input;
  input.first = FinalTrajectoryKnot{
      core::StateId{10U}, core::FusionTime{1'000'000'000LL},
      pose({1.0, -0.5, 0.2}, {0.15, -0.08, 0.04}),
      Eigen::Vector3d{1.1, -0.2, 0.1}};
  input.second = FinalTrajectoryKnot{
      core::StateId{11U}, core::FusionTime{1'400'000'000LL},
      pose({1.5, -0.62, 0.26}, {0.22, -0.04, 0.11}),
      Eigen::Vector3d{1.3, -0.35, 0.2}};
  input.observation_time = core::FusionTime{1'170'000'000LL};
  input.lever_arm =
      AntennaLeverArm::fromImuToAntenna({0.35, -0.08, 0.12}).value();
  input.endpoint_covariance.matrix.setIdentity();
  input.endpoint_covariance.matrix *= 1.0e-4;
  input.lever_arm_covariance_imu =
      2.5e-5 * Eigen::Matrix3d::Identity();
  input.interpolation_model_covariance_submap =
      4.0e-4 * Eigen::Matrix3d::Identity();
  return input;
}

TEST(GnssInterpolation, HermiteAndGeodesicReachExactEndpoints) {
  auto input = nominalInput();
  input.observation_time = input.first.exact_time;
  auto first = interpolateFinalAntennaState(input);
  ASSERT_TRUE(first) << first.error().detail;
  EXPECT_TRUE(first.value().body_position_submap.isApprox(
      input.first.T_submap_imu.translation(), 1.0e-14));
  EXPECT_TRUE(first.value().R_submap_imu.matrix().isApprox(
      input.first.T_submap_imu.so3().matrix(), 1.0e-14));
  EXPECT_NEAR(first.value().diagnostics.alpha, 0.0, 0.0);

  input.observation_time = input.second.exact_time;
  auto second = interpolateFinalAntennaState(input);
  ASSERT_TRUE(second) << second.error().detail;
  EXPECT_TRUE(second.value().body_position_submap.isApprox(
      input.second.T_submap_imu.translation(), 1.0e-14));
  EXPECT_TRUE(second.value().R_submap_imu.matrix().isApprox(
      input.second.T_submap_imu.so3().matrix(), 1.0e-14));
  EXPECT_NEAR(second.value().diagnostics.alpha, 1.0, 0.0);
}

TEST(GnssInterpolation, AnalyticEndpointJacobianMatchesCentralDifference) {
  const auto input = nominalInput();
  const auto evaluated = interpolateFinalAntennaState(input);
  ASSERT_TRUE(evaluated) << evaluated.error().detail;

  constexpr double kStep = 1.0e-6;
  Eigen::Matrix<double, 3, 18> numerical;
  for (Eigen::Index column = 0; column < 18; ++column) {
    auto plus = input;
    auto minus = input;
    const bool first_endpoint = column < 9;
    const Eigen::Index endpoint_column = first_endpoint ? column : column - 9;
    FinalTrajectoryKnot* plus_knot =
        first_endpoint ? &plus.first : &plus.second;
    FinalTrajectoryKnot* minus_knot =
        first_endpoint ? &minus.first : &minus.second;
    if (endpoint_column < 6) {
      Eigen::Matrix<double, 6, 1> delta =
          Eigen::Matrix<double, 6, 1>::Zero();
      delta(endpoint_column) = kStep;
      plus_knot->T_submap_imu =
          plus_knot->T_submap_imu * core::Pose3d::exp(delta);
      minus_knot->T_submap_imu =
          minus_knot->T_submap_imu * core::Pose3d::exp(-delta);
    } else {
      const Eigen::Index velocity_axis = endpoint_column - 6;
      plus_knot->velocity_submap(velocity_axis) += kStep;
      minus_knot->velocity_submap(velocity_axis) -= kStep;
    }
    const auto plus_result = interpolateFinalAntennaState(plus);
    const auto minus_result = interpolateFinalAntennaState(minus);
    ASSERT_TRUE(plus_result) << plus_result.error().detail;
    ASSERT_TRUE(minus_result) << minus_result.error().detail;
    numerical.col(column) =
        (plus_result.value().antenna_position_submap -
         minus_result.value().antenna_position_submap) /
        (2.0 * kStep);
  }
  EXPECT_TRUE(evaluated.value().antenna_jacobian_endpoints.isApprox(
      numerical, 3.0e-6))
      << evaluated.value().antenna_jacobian_endpoints << "\nvs\n"
      << numerical;
}

TEST(GnssInterpolation, PropagatesFullJointAndLeverCovariance) {
  auto input = nominalInput();
  input.endpoint_covariance.matrix(0, 9) = 3.0e-5;
  input.endpoint_covariance.matrix(9, 0) = 3.0e-5;
  const auto result = interpolateFinalAntennaState(input);
  ASSERT_TRUE(result) << result.error().detail;
  const Eigen::Matrix3d expected =
      result.value().antenna_jacobian_endpoints *
          input.endpoint_covariance.matrix *
          result.value().antenna_jacobian_endpoints.transpose() +
      result.value().antenna_jacobian_lever_arm *
          input.lever_arm_covariance_imu *
          result.value().antenna_jacobian_lever_arm.transpose() +
      input.interpolation_model_covariance_submap;
  EXPECT_TRUE(result.value().antenna_covariance_submap.isApprox(expected,
                                                                1.0e-14));
  EXPECT_GT(result.value().antenna_covariance_submap.diagonal().minCoeff(),
            0.0);
}

TEST(GnssInterpolation, RejectsUnsupportedBracketAndCovariance) {
  auto input = nominalInput();
  GnssInterpolationConfig config;
  config.maximum_segment_duration = core::Duration{100'000'000LL};
  auto too_long = interpolateFinalAntennaState(input, config);
  ASSERT_FALSE(too_long);
  EXPECT_EQ(too_long.error().code,
            GnssInterpolationErrorCode::SegmentTooLong);

  config.maximum_segment_duration = core::Duration{500'000'000LL};
  input.endpoint_covariance.matrix.setZero();
  auto singular = interpolateFinalAntennaState(input, config);
  ASSERT_FALSE(singular);
  EXPECT_EQ(singular.error().code,
            GnssInterpolationErrorCode::InvalidCovariance);

  input = nominalInput();
  input.observation_time = core::FusionTime{999'999'999LL};
  auto outside = interpolateFinalAntennaState(input, config);
  ASSERT_FALSE(outside);
  EXPECT_EQ(outside.error().code,
            GnssInterpolationErrorCode::ObservationOutsideSegment);
}

}  // namespace
}  // namespace meridian::global
