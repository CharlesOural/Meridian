#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sophus/se3.hpp>
#include <vector>

#include "meridian/local/local_rt.hpp"

namespace meridian::local {
namespace {

core::ImuSample sample(std::uint64_t id, std::int64_t time_ns, Eigen::Vector3d acceleration,
                       Eigen::Vector3d angular_rate = Eigen::Vector3d::Zero()) {
  core::ImuSample result;
  result.id = core::MeasurementId{id};
  result.stamp.fusion_time = core::FusionTime{time_ns};
  result.stamp.source_epoch = core::SourceEpoch{1};
  result.stamp.clock_revision = core::ClockRevision{1};
  result.stamp.ingress_sequence = core::IngressSequence{id};
  result.specific_force_mps2 = std::move(acceleration);
  result.angular_velocity_radps = std::move(angular_rate);
  return result;
}

[[nodiscard]] core::Pose3d scalarPoseAt(const std::vector<TimedNavState>& trajectory,
                                        core::FusionTime time) {
  const auto right = std::lower_bound(
      trajectory.begin(), trajectory.end(), time,
      [](const TimedNavState& state, core::FusionTime query) { return state.time < query; });
  if (right == trajectory.end()) {
    return trajectory.back().state.T_odom_imu;
  }
  if (right->time == time || right == trajectory.begin()) {
    return right->state.T_odom_imu;
  }
  const auto left = std::prev(right);
  const double alpha = static_cast<double>((time - left->time).nanoseconds) /
                       static_cast<double>((right->time - left->time).nanoseconds);
  const core::Pose3d relative = left->state.T_odom_imu.inverse() * right->state.T_odom_imu;
  return left->state.T_odom_imu * core::Pose3d::exp(alpha * relative.log());
}

[[nodiscard]] core::LidarPoints scalarDeskew(const core::LidarSweep& sweep,
                                             core::FusionTime reference_time,
                                             const core::Pose3d& T_imu_lidar,
                                             const std::vector<TimedNavState>& trajectory) {
  const core::Pose3d T_reference_odom = scalarPoseAt(trajectory, reference_time).inverse();
  core::LidarPoints result;
  result.reserve(sweep.points->size());
  for (const auto& point : *sweep.points) {
    const core::FusionTime point_time =
        sweep.acquisition.start + core::Duration{point.time_offset_ns};
    const Eigen::Vector3d point_reference_imu = T_reference_odom *
                                                scalarPoseAt(trajectory, point_time) * T_imu_lidar *
                                                Eigen::Vector3d{point.x, point.y, point.z};
    core::LidarPoint transformed = point;
    transformed.x = static_cast<float>(point_reference_imu.x());
    transformed.y = static_cast<float>(point_reference_imu.y());
    transformed.z = static_cast<float>(point_reference_imu.z());
    result.push_back(transformed);
  }
  return result;
}

void expectPointsBitIdentical(const core::LidarPoints& actual, const core::LidarPoints& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    EXPECT_EQ(actual[index].x, expected[index].x) << "point " << index;
    EXPECT_EQ(actual[index].y, expected[index].y) << "point " << index;
    EXPECT_EQ(actual[index].z, expected[index].z) << "point " << index;
    EXPECT_EQ(actual[index].intensity, expected[index].intensity) << "point " << index;
    EXPECT_EQ(actual[index].time_offset_ns, expected[index].time_offset_ns) << "point " << index;
    EXPECT_EQ(actual[index].ring, expected[index].ring) << "point " << index;
    EXPECT_EQ(actual[index].source_index, expected[index].source_index) << "point " << index;
  }
}

TEST(ImuBuffer, InterpolatesExactIntervalBoundaries) {
  ImuBuffer buffer;
  ASSERT_TRUE(buffer.append(sample(1, 0, {0.0, 0.0, 0.0})));
  ASSERT_TRUE(buffer.append(sample(2, 10'000'000, {2.0, 0.0, 0.0})));
  ASSERT_TRUE(buffer.append(sample(3, 20'000'000, {4.0, 0.0, 0.0})));
  const auto support = buffer.interval(core::TimeRange{{5'000'000}, {15'000'000}}, {10'000'000});
  ASSERT_TRUE(support);
  ASSERT_EQ(support.value().knots.size(), 3U);
  EXPECT_DOUBLE_EQ(support.value().knots.front().specific_force_mps2.x(), 1.0);
  EXPECT_DOUBLE_EQ(support.value().knots.back().specific_force_mps2.x(), 3.0);
  EXPECT_EQ(support.value().raw_measurements.size(), 3U);
}

TEST(ImuBuffer, RejectsEpochBreakingGap) {
  ImuBuffer buffer;
  ASSERT_TRUE(buffer.append(sample(1, 0, {0.0, 0.0, 9.80665})));
  ASSERT_TRUE(buffer.append(sample(2, 30'000'000, {0.0, 0.0, 9.80665})));
  const auto support = buffer.interval(core::TimeRange{{0}, {30'000'000}}, {10'000'000});
  ASSERT_FALSE(support);
  EXPECT_EQ(support.error().code, ImuBufferErrorCode::EpochBreakingGap);
}

TEST(StationaryInitializer, AlignsSpecificForceAndEstimatesGyroBias) {
  ImuBuffer buffer;
  for (std::uint64_t index = 0; index <= 400; ++index) {
    ASSERT_TRUE(buffer.append(sample(index, static_cast<std::int64_t>(index) * 5'000'000LL,
                                     {0.0, 9.80665, 0.0}, {0.001, -0.002, 0.003})));
  }
  const auto support = buffer.interval(core::TimeRange{{0}, {2'000'000'000LL}}, {5'000'000});
  ASSERT_TRUE(support);
  const auto initialization = initializeStationary(support.value());
  ASSERT_TRUE(initialization);
  const Eigen::Vector3d aligned =
      initialization.value().state.T_odom_imu.so3() * Eigen::Vector3d::UnitY();
  EXPECT_TRUE(aligned.isApprox(Eigen::Vector3d::UnitZ(), 1.0e-9));
  EXPECT_TRUE(initialization.value().state.gyro_bias.isApprox(Eigen::Vector3d{0.001, -0.002, 0.003},
                                                              1.0e-12));
  EXPECT_GT(initialization.value().covariance.matrix(12, 12), 1.0e-3);
}

TEST(MidpointPropagation, StaticSpecificForceCancelsGravity) {
  ImuBuffer buffer;
  ASSERT_TRUE(buffer.append(sample(1, 0, {0.0, 0.0, 9.80665})));
  ASSERT_TRUE(buffer.append(sample(2, 5'000'000, {0.0, 0.0, 9.80665})));
  ASSERT_TRUE(buffer.append(sample(3, 10'000'000, {0.0, 0.0, 9.80665})));
  const auto support = buffer.interval(core::TimeRange{{0}, {10'000'000}}, {5'000'000});
  ASSERT_TRUE(support);
  core::NavStateEstimate anchor;
  const auto propagated = MidpointImuPropagator{}.propagate({0}, anchor, support.value());
  ASSERT_TRUE(propagated);
  EXPECT_TRUE(propagated.value().final_state.T_odom_imu.translation().isZero(1.0e-12));
  EXPECT_TRUE(propagated.value().final_state.velocity_odom.isZero(1.0e-12));
}

TEST(MidpointPropagation, ReconstructsOneTrajectoryAroundAnInteriorCommittedState) {
  ImuBuffer buffer;
  for (std::uint64_t index = 0U; index <= 4U; ++index) {
    ASSERT_TRUE(buffer.append(sample(index + 1U, static_cast<std::int64_t>(index) * 5'000'000LL,
                                     {0.4, -0.2, 9.90665}, {0.02, -0.01, 0.03})));
  }
  const auto complete = buffer.interval(core::TimeRange{{0}, {20'000'000}}, {5'000'000});
  const auto before = buffer.interval(core::TimeRange{{0}, {10'000'000}}, {5'000'000});
  const auto after = buffer.interval(core::TimeRange{{10'000'000}, {20'000'000}}, {5'000'000});
  ASSERT_TRUE(complete);
  ASSERT_TRUE(before);
  ASSERT_TRUE(after);

  core::NavStateEstimate initial;
  initial.velocity_odom = Eigen::Vector3d{0.3, -0.1, 0.2};
  initial.gyro_bias = Eigen::Vector3d{0.001, -0.002, 0.003};
  initial.accel_bias = Eigen::Vector3d{0.01, -0.02, 0.03};
  const MidpointImuPropagator propagator;
  const auto oracle = propagator.propagate({0}, initial, complete.value());
  ASSERT_TRUE(oracle);
  ASSERT_EQ(oracle.value().trajectory.size(), 5U);
  const core::NavStateEstimate anchor = oracle.value().trajectory[2U].state;

  const auto backward = propagator.propagateBackwards({10'000'000}, anchor, before.value());
  ASSERT_TRUE(backward) << backward.error().detail;
  ASSERT_EQ(backward.value().trajectory.size(), 3U);
  for (std::size_t index = 0U; index < backward.value().trajectory.size(); ++index) {
    EXPECT_EQ(backward.value().trajectory[index].time, oracle.value().trajectory[index].time);
    EXPECT_LT((backward.value().trajectory[index].state.T_odom_imu.inverse() *
               oracle.value().trajectory[index].state.T_odom_imu)
                  .log()
                  .norm(),
              1.0e-12);
    EXPECT_TRUE(backward.value().trajectory[index].state.velocity_odom.isApprox(
        oracle.value().trajectory[index].state.velocity_odom, 1.0e-12));
  }
  EXPECT_LT((backward.value().final_state.T_odom_imu.inverse() * anchor.T_odom_imu).log().norm(),
            1.0e-15);
  EXPECT_EQ(backward.value().final_state.velocity_odom, anchor.velocity_odom);

  const auto around =
      propagator.propagateAround({10'000'000}, anchor, before.value(), after.value());
  ASSERT_TRUE(around) << around.error().detail;
  ASSERT_EQ(around.value().trajectory.size(), oracle.value().trajectory.size());
  for (std::size_t index = 0U; index < oracle.value().trajectory.size(); ++index) {
    EXPECT_EQ(around.value().trajectory[index].time, oracle.value().trajectory[index].time);
    EXPECT_LT((around.value().trajectory[index].state.T_odom_imu.inverse() *
               oracle.value().trajectory[index].state.T_odom_imu)
                  .log()
                  .norm(),
              1.0e-12);
    EXPECT_TRUE(around.value().trajectory[index].state.velocity_odom.isApprox(
        oracle.value().trajectory[index].state.velocity_odom, 1.0e-12));
    EXPECT_EQ(around.value().trajectory[index].state.gyro_bias, initial.gyro_bias);
    EXPECT_EQ(around.value().trajectory[index].state.accel_bias, initial.accel_bias);
  }
  EXPECT_TRUE(around.value().final_state.velocity_odom.isApprox(
      oracle.value().final_state.velocity_odom, 1.0e-12));
  EXPECT_EQ(around.value().raw_measurements, complete.value().raw_measurements);
}

TEST(LidarDeskew, ExpressesEachPointAtReferenceBodyTime) {
  auto points = std::make_shared<core::LidarPoints>();
  points->push_back(core::LidarPoint{2.0F, 0.0F, 0.0F, 0.0F, 0, 0});
  points->push_back(core::LidarPoint{2.0F, 0.0F, 0.0F, 0.0F, 999'999'999, 0});
  core::LidarSweep sweep;
  sweep.id = core::MeasurementId{1};
  sweep.acquisition = core::TimeRange{{0}, {1'000'000'000}};
  sweep.points = points;

  core::NavStateEstimate first;
  core::NavStateEstimate last;
  last.T_odom_imu = core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{1.0, 0.0, 0.0});
  const std::vector<TimedNavState> trajectory{{{0}, first}, {{1'000'000'000}, last}};
  const auto deskewed = deskewLidarSweep(sweep, {1'000'000'000}, core::Pose3d{}, trajectory);
  ASSERT_TRUE(deskewed);
  EXPECT_NEAR(deskewed.value().points_in_reference_imu->at(0).x, 1.0F, 1.0e-6F);
  EXPECT_NEAR(deskewed.value().points_in_reference_imu->at(1).x, 2.0F, 1.0e-6F);
}

TEST(LidarDeskew, ReusesPosesForRepeatedPointOffsets) {
  auto points = std::make_shared<core::LidarPoints>();
  for (std::uint32_t row = 0U; row < 4U; ++row) {
    for (std::uint32_t column = 0U; column < 3U; ++column) {
      core::LidarPoint point;
      point.x = 1.0F + static_cast<float>(row);
      point.y = 0.25F * static_cast<float>(column);
      point.z = -0.1F * static_cast<float>(row + column);
      point.intensity = static_cast<float>(10U * row + column);
      point.time_offset_ns = static_cast<std::int32_t>(column * 250'000'000U);
      point.ring = static_cast<std::uint16_t>(row);
      point.source_index = row * 3U + column;
      points->push_back(point);
    }
  }

  core::LidarSweep sweep;
  sweep.id = core::MeasurementId{2};
  sweep.acquisition = core::TimeRange{{0}, {750'000'001}};
  sweep.layout = core::LidarLayout{3U, 4U, true};
  sweep.points = points;

  core::NavStateEstimate first;
  core::NavStateEstimate middle;
  core::NavStateEstimate last;
  middle.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{0.0, 0.0, 0.2}),
                                   Eigen::Vector3d{0.3, -0.1, 0.05});
  last.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{0.1, -0.2, 0.3}),
                                 Eigen::Vector3d{0.8, 0.2, -0.1});
  const std::vector<TimedNavState> trajectory{
      {{0}, first}, {{400'000'000}, middle}, {{800'000'000}, last}};
  const core::Pose3d T_imu_lidar(Sophus::SO3d::exp(Eigen::Vector3d{0.02, -0.01, 0.03}),
                                 Eigen::Vector3d{0.1, 0.02, -0.04});

  const auto deskewed = deskewLidarSweep(sweep, {600'000'000}, T_imu_lidar, trajectory);
  ASSERT_TRUE(deskewed);
  EXPECT_EQ(deskewed.value().pose_interpolations, 3U);
  expectPointsBitIdentical(*deskewed.value().points_in_reference_imu,
                           scalarDeskew(sweep, {600'000'000}, T_imu_lidar, trajectory));
}

TEST(LidarDeskew, MatchesScalarOracleForArbitraryUnsortedOffsets) {
  auto points = std::make_shared<core::LidarPoints>();
  const std::vector<std::int32_t> offsets{600'000'000, 25'000'000, 799'000'000, 200'000'000,
                                          475'000'000, 25'000'000, 0,           799'000'000};
  for (std::size_t index = 0U; index < offsets.size(); ++index) {
    core::LidarPoint point;
    point.x = static_cast<float>(0.5 + 0.17 * static_cast<double>(index));
    point.y = static_cast<float>(-0.4 + 0.09 * static_cast<double>(index));
    point.z = static_cast<float>(0.2 - 0.03 * static_cast<double>(index));
    point.intensity = static_cast<float>(index);
    point.time_offset_ns = offsets[index];
    point.ring = static_cast<std::uint16_t>(index % 4U);
    point.source_index = static_cast<std::uint32_t>(100U + index);
    points->push_back(point);
  }

  core::LidarSweep sweep;
  sweep.id = core::MeasurementId{3};
  sweep.acquisition = core::TimeRange{{100'000'000}, {900'000'001}};
  sweep.layout = core::LidarLayout{static_cast<std::uint32_t>(points->size()), 1U, false};
  sweep.points = points;

  core::NavStateEstimate state0;
  core::NavStateEstimate state1;
  core::NavStateEstimate state2;
  core::NavStateEstimate state3;
  state0.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{-0.1, 0.03, 0.02}),
                                   Eigen::Vector3d{-0.2, 0.1, 0.4});
  state1.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{0.03, 0.15, -0.08}),
                                   Eigen::Vector3d{0.0, 0.3, 0.2});
  state2.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{0.2, -0.05, 0.1}),
                                   Eigen::Vector3d{0.6, 0.2, -0.1});
  state3.T_odom_imu = core::Pose3d(Sophus::SO3d::exp(Eigen::Vector3d{0.35, 0.08, -0.12}),
                                   Eigen::Vector3d{1.1, -0.1, 0.05});
  const std::vector<TimedNavState> trajectory{
      {{0}, state0}, {{300'000'000}, state1}, {{700'000'000}, state2}, {{1'000'000'000}, state3}};
  const core::Pose3d T_imu_lidar(Sophus::SO3d::exp(Eigen::Vector3d{-0.04, 0.01, 0.02}),
                                 Eigen::Vector3d{0.08, -0.03, 0.06});

  const auto deskewed = deskewLidarSweep(sweep, {650'000'000}, T_imu_lidar, trajectory);
  ASSERT_TRUE(deskewed);
  EXPECT_EQ(deskewed.value().pose_interpolations, 6U);
  expectPointsBitIdentical(*deskewed.value().points_in_reference_imu,
                           scalarDeskew(sweep, {650'000'000}, T_imu_lidar, trajectory));
}

TEST(LidarDeskew, AcceptsTrajectoryBoundariesAndRejectsAcquisitionEnd) {
  auto points = std::make_shared<core::LidarPoints>();
  points->push_back(core::LidarPoint{1.0F, 0.0F, 0.0F, 1.0F, 0, 0, 10U});
  points->push_back(core::LidarPoint{2.0F, 0.0F, 0.0F, 2.0F, 899'999'999, 1, 11U});
  core::LidarSweep sweep;
  sweep.id = core::MeasurementId{4};
  sweep.acquisition = core::TimeRange{{100'000'000}, {1'000'000'000}};
  sweep.points = points;

  core::NavStateEstimate first;
  core::NavStateEstimate last;
  last.T_odom_imu = core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{0.9, 0.0, 0.0});
  const std::vector<TimedNavState> trajectory{{{100'000'000}, first}, {{1'000'000'000}, last}};

  const auto deskewed = deskewLidarSweep(sweep, {1'000'000'000}, core::Pose3d{}, trajectory);
  ASSERT_TRUE(deskewed);
  EXPECT_EQ(deskewed.value().pose_interpolations, 2U);
  expectPointsBitIdentical(*deskewed.value().points_in_reference_imu,
                           scalarDeskew(sweep, {1'000'000'000}, core::Pose3d{}, trajectory));

  points->back().time_offset_ns = 900'000'000;
  const auto rejected = deskewLidarSweep(sweep, {1'000'000'000}, core::Pose3d{}, trajectory);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, DeskewErrorCode::PointOutsideTrajectory);
  EXPECT_EQ(rejected.error().detail,
            "point offset lies outside the immutable sweep acquisition interval");
}

TEST(LidarDeskew, RejectsPointInsideAcquisitionButOutsideTrajectory) {
  auto points = std::make_shared<core::LidarPoints>();
  points->push_back(core::LidarPoint{1.0F, 0.0F, 0.0F, 0.0F, 0, 0, 0U});
  core::LidarSweep sweep;
  sweep.id = core::MeasurementId{5};
  sweep.acquisition = core::TimeRange{{0}, {1'000'000'000}};
  sweep.points = points;

  core::NavStateEstimate first;
  core::NavStateEstimate last;
  const std::vector<TimedNavState> trajectory{{{100'000'000}, first}, {{900'000'000}, last}};
  const auto rejected = deskewLidarSweep(sweep, {500'000'000}, core::Pose3d{}, trajectory);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, DeskewErrorCode::PointOutsideTrajectory);
  EXPECT_EQ(rejected.error().detail, "requested deskew time is outside the discrete trajectory");
}

}  // namespace
}  // namespace meridian::local
