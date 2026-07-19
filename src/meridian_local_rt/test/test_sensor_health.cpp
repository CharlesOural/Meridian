#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "meridian/local/sensor_health.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::SensorInstanceId lidar() {
  return core::SensorInstanceId::lidar(core::LidarId{1U});
}

[[nodiscard]] core::SensorInstanceId camera() {
  return core::SensorInstanceId::camera(core::CameraId{2U});
}

[[nodiscard]] SensorHealthRegistryConfig registryConfig() {
  SensorHealthRegistryConfig config;
  config.sensors = {camera(), lidar()};
  config.initialized_at = core::FusionTime{0LL};
  config.policy.consecutive_failures_to_suspect = 1U;
  config.policy.consecutive_failures_to_failed = 2U;
  config.policy.recovery_good_shadow_results = 2U;
  config.policy.suspect_after_no_result = core::Duration{10LL};
  config.policy.failed_after_no_result = core::Duration{20LL};
  return config;
}

[[nodiscard]] SensorHealthBatchObservation observation(
    core::SensorInstanceId sensor, std::uint64_t batch, std::uint64_t epoch,
    std::int64_t assessed_at, SensorBatchHealthResult result,
    SensorBatchEvaluationMode mode = SensorBatchEvaluationMode::Primary) {
  return SensorHealthBatchObservation{sensor,
                                      core::FactorBatchId{batch},
                                      core::SensorRecoveryEpoch{epoch},
                                      core::FusionTime{assessed_at},
                                      result,
                                      mode};
}

void expectHealthEqual(const core::SensorHealthSnapshot& left,
                       const core::SensorHealthSnapshot& right) {
  EXPECT_EQ(left.sensor, right.sensor);
  EXPECT_EQ(left.state, right.state);
  EXPECT_EQ(left.recovery_epoch, right.recovery_epoch);
  EXPECT_EQ(left.transition_sequence, right.transition_sequence);
  EXPECT_EQ(left.assessed_at, right.assessed_at);
}

TEST(SensorHealthRegistry, RequiresABoundedUniqueConfiguredSensorSet) {
  SensorHealthRegistryConfig config = registryConfig();
  config.sensors.clear();
  auto created = SensorHealthRegistry::create(config);
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, SensorHealthRegistryErrorCode::InvalidConfig);

  config = registryConfig();
  config.sensors.push_back(lidar());
  created = SensorHealthRegistry::create(config);
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, SensorHealthRegistryErrorCode::DuplicateConfiguredSensor);

  config = registryConfig();
  config.sensors.assign(kMaximumConfiguredSensorHealthInstances + 1U, lidar());
  created = SensorHealthRegistry::create(config);
  ASSERT_FALSE(created);
  EXPECT_EQ(created.error().code, SensorHealthRegistryErrorCode::InvalidConfig);
}

TEST(SensorHealthRegistry, FailureAndTimeoutTransitionsAreStrictlyPerSensor) {
  auto created = SensorHealthRegistry::create(registryConfig());
  ASSERT_TRUE(created);
  SensorHealthRegistry registry = std::move(created).value();
  ASSERT_EQ(registry.size(), 2U);
  const auto initial = registry.snapshots();
  ASSERT_EQ(initial.size(), 2U);
  EXPECT_EQ(initial[0].health.sensor, lidar());
  EXPECT_EQ(initial[1].health.sensor, camera());
  EXPECT_EQ(initial[0].health.state, core::SensorHealthState::Active);
  EXPECT_EQ(initial[1].health.state, core::SensorHealthState::Active);

  const auto camera_before = registry.snapshot(camera());
  ASSERT_TRUE(camera_before);
  const auto lidar_failure =
      registry.observe(observation(lidar(), 1U, 0U, 1LL, SensorBatchHealthResult::Failure));
  ASSERT_TRUE(lidar_failure);
  EXPECT_TRUE(lidar_failure.value().transitioned);
  EXPECT_EQ(lidar_failure.value().after.state, core::SensorHealthState::Suspect);
  const auto camera_after_lidar = registry.snapshot(camera());
  ASSERT_TRUE(camera_after_lidar);
  expectHealthEqual(camera_before.value().health, camera_after_lidar.value().health);

  const auto lidar_before_camera_timeout = registry.snapshot(lidar());
  ASSERT_TRUE(lidar_before_camera_timeout);
  const auto camera_suspect =
      registry.assessTimeout(camera(), core::SensorRecoveryEpoch{0U}, core::FusionTime{10LL});
  ASSERT_TRUE(camera_suspect);
  EXPECT_TRUE(camera_suspect.value().timed_out);
  EXPECT_TRUE(camera_suspect.value().transitioned);
  EXPECT_EQ(camera_suspect.value().after.state, core::SensorHealthState::Suspect);
  const auto lidar_after_camera_timeout = registry.snapshot(lidar());
  ASSERT_TRUE(lidar_after_camera_timeout);
  expectHealthEqual(lidar_before_camera_timeout.value().health,
                    lidar_after_camera_timeout.value().health);

  const auto camera_failed =
      registry.assessTimeout(camera(), core::SensorRecoveryEpoch{0U}, core::FusionTime{20LL});
  ASSERT_TRUE(camera_failed);
  EXPECT_TRUE(camera_failed.value().transitioned);
  EXPECT_EQ(camera_failed.value().after.state, core::SensorHealthState::Failed);
}

TEST(SensorHealthRegistry, RecoveryEpochAdvancesExactlyOncePerAttemptAndRequiresKNewResults) {
  auto created = SensorHealthRegistry::create(registryConfig());
  ASSERT_TRUE(created);
  SensorHealthRegistry registry = std::move(created).value();

  ASSERT_TRUE(
      registry.observe(observation(lidar(), 1U, 0U, 1LL, SensorBatchHealthResult::Failure)));
  const auto failed =
      registry.observe(observation(lidar(), 2U, 0U, 2LL, SensorBatchHealthResult::Failure));
  ASSERT_TRUE(failed);
  EXPECT_EQ(failed.value().after.state, core::SensorHealthState::Failed);
  EXPECT_EQ(failed.value().after.recovery_epoch, core::SensorRecoveryEpoch{0U});
  EXPECT_EQ(failed.value().after.transition_sequence, 2U);

  const auto first_shadow = registry.observe(observation(
      lidar(), 3U, 0U, 3LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_TRUE(first_shadow);
  EXPECT_TRUE(first_shadow.value().transitioned);
  EXPECT_EQ(first_shadow.value().before.state, core::SensorHealthState::Failed);
  EXPECT_EQ(first_shadow.value().before.recovery_epoch, core::SensorRecoveryEpoch{0U});
  EXPECT_EQ(first_shadow.value().after.state, core::SensorHealthState::Recovering);
  EXPECT_EQ(first_shadow.value().after.recovery_epoch, core::SensorRecoveryEpoch{1U});
  EXPECT_EQ(first_shadow.value().after.transition_sequence, 3U);
  EXPECT_EQ(first_shadow.value().recovery_good_shadow_results, 0U);

  const auto recovering_before_errors = registry.snapshot(lidar());
  ASSERT_TRUE(recovering_before_errors);
  const auto stale_epoch = registry.observe(observation(
      lidar(), 4U, 0U, 4LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_FALSE(stale_epoch);
  EXPECT_EQ(stale_epoch.error().code, SensorHealthRegistryErrorCode::StaleRecoveryEpoch);
  const auto future_epoch = registry.observe(observation(
      lidar(), 4U, 2U, 4LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_FALSE(future_epoch);
  EXPECT_EQ(future_epoch.error().code, SensorHealthRegistryErrorCode::FutureRecoveryEpoch);
  const auto stale_batch = registry.observe(observation(
      lidar(), 3U, 1U, 4LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_FALSE(stale_batch);
  EXPECT_EQ(stale_batch.error().code, SensorHealthRegistryErrorCode::StaleBatch);
  const auto recovering_after_errors = registry.snapshot(lidar());
  ASSERT_TRUE(recovering_after_errors);
  expectHealthEqual(recovering_before_errors.value().health,
                    recovering_after_errors.value().health);
  EXPECT_EQ(recovering_after_errors.value().last_batch,
            recovering_before_errors.value().last_batch);
  EXPECT_EQ(recovering_after_errors.value().recovery_good_shadow_results, 0U);

  const auto one_good = registry.observe(observation(
      lidar(), 4U, 1U, 4LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_TRUE(one_good);
  EXPECT_FALSE(one_good.value().transitioned);
  EXPECT_EQ(one_good.value().after.state, core::SensorHealthState::Recovering);
  EXPECT_EQ(one_good.value().recovery_good_shadow_results, 1U);

  const auto recovery_failure = registry.observe(observation(
      lidar(), 5U, 1U, 5LL, SensorBatchHealthResult::Failure, SensorBatchEvaluationMode::Shadow));
  ASSERT_TRUE(recovery_failure);
  EXPECT_EQ(recovery_failure.value().after.state, core::SensorHealthState::Failed);
  EXPECT_EQ(recovery_failure.value().after.recovery_epoch, core::SensorRecoveryEpoch{1U});

  const auto second_attempt = registry.observe(observation(
      lidar(), 6U, 1U, 6LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_TRUE(second_attempt);
  EXPECT_EQ(second_attempt.value().after.state, core::SensorHealthState::Recovering);
  EXPECT_EQ(second_attempt.value().after.recovery_epoch, core::SensorRecoveryEpoch{2U});
  EXPECT_EQ(second_attempt.value().recovery_good_shadow_results, 0U);

  ASSERT_TRUE(registry.observe(observation(lidar(), 7U, 2U, 7LL, SensorBatchHealthResult::Good,
                                           SensorBatchEvaluationMode::Shadow)));
  const auto recovered = registry.observe(observation(
      lidar(), 8U, 2U, 8LL, SensorBatchHealthResult::Good, SensorBatchEvaluationMode::Shadow));
  ASSERT_TRUE(recovered);
  EXPECT_TRUE(recovered.value().transitioned);
  EXPECT_EQ(recovered.value().after.state, core::SensorHealthState::Active);
  EXPECT_EQ(recovered.value().after.recovery_epoch, core::SensorRecoveryEpoch{2U});
  EXPECT_EQ(recovered.value().after.transition_sequence, 6U);
}

}  // namespace
}  // namespace meridian::local
