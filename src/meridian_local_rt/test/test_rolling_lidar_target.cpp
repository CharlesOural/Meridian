#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/rolling_lidar_target.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Matrix3d& rotation = Eigen::Matrix3d::Identity()) {
  return core::Pose3d(Sophus::SO3d(rotation), translation);
}

[[nodiscard]] core::ObservationLineage lineage(std::uint64_t source,
                                               std::uint64_t lineage_id = 0U) {
  core::ObservationUsage usage;
  usage.slice.root = core::MeasurementId(source);
  usage.slice.calibration = core::CalibrationEpoch(1U);
  usage.slice.source_checksum.fill(static_cast<std::uint8_t>((source % 251U) + 1U));
  usage.role = core::ObservationRole::DerivedSummary;
  usage.consumer = core::DerivedRecordId(source);
  core::ObservationLineage result;
  result.id = core::ObservationLineageId(lineage_id == 0U ? source : lineage_id);
  result.usage.push_back(usage);
  return result;
}

[[nodiscard]] std::shared_ptr<const LidarRegistrationCloud> cloud(
    std::uint64_t source, std::int64_t time, const std::vector<Eigen::Vector3d>& positions,
    const core::Pose3d& T_odom_imu_seed = core::Pose3d(),
    double exact_index_resolution_m = 0.25,
    core::ObservationLineage ancestry = {}) {
  return test::sealedLidarRegistrationCloud(
      positions, core::MeasurementId(source), core::FusionTime(time), T_odom_imu_seed,
      exact_index_resolution_m,
      ancestry.id.valid() ? std::move(ancestry) : lineage(source));
}

[[nodiscard]] RollingLidarTargetConfig config(std::size_t maximum_sweeps = 5U,
                                              std::size_t maximum_points = 100U,
                                              std::size_t maximum_targets = 2U) {
  RollingLidarTargetConfig result;
  result.odom_epoch = core::OdomEpoch(7U);
  result.maximum_retained_sweeps = maximum_sweeps;
  result.maximum_retained_points = maximum_points;
  result.registration.target_voxel_resolution_m = 0.25;
  result.registration.source_voxel_size_m = 0.10;
  result.registration.maximum_correspondence_distance_m = 0.20;
  result.registration.maximum_voxel_search_radius = 1U;
  result.registration.maximum_source_points = maximum_points;
  result.registration.maximum_target_points_per_target = maximum_points;
  result.registration.maximum_targets = maximum_targets;
  result.registration.minimum_correspondences = 1U;
  result.registration.minimum_huber_delta_m = 0.05;
  result.registration.maximum_huber_delta_m = 0.20;
  return result;
}

[[nodiscard]] RegisteredLidarSweep registration(
    std::uint64_t state, const core::Pose3d& T_odom_imu,
    std::shared_ptr<const LidarRegistrationCloud> source,
    core::OdomEpoch epoch = core::OdomEpoch(7U),
    core::FactorBatchId admitting_batch_id = core::FactorBatchId{},
    core::SensorRecoveryEpoch recovery_epoch = core::SensorRecoveryEpoch(0U)) {
  if (!admitting_batch_id.valid()) {
    admitting_batch_id = core::FactorBatchId(state);
  }
  return RegisteredLidarSweep{epoch, core::StateId(state), admitting_batch_id, recovery_epoch,
                                 T_odom_imu, std::move(source)};
}

[[nodiscard]] std::vector<Eigen::Vector3d> planarGrid(std::size_t x_count, std::size_t y_count) {
  std::vector<Eigen::Vector3d> output;
  output.reserve(x_count * y_count);
  for (std::size_t x = 0U; x < x_count; ++x) {
    for (std::size_t y = 0U; y < y_count; ++y) {
      output.emplace_back(0.12 * static_cast<double>(x), 0.12 * static_cast<double>(y),
                          0.01 * static_cast<double>((x + y) % 3U));
    }
  }
  return output;
}

TEST(RollingLidarTarget, RejectsInvalidConfiguration) {
  auto invalid = config();
  invalid.maximum_retained_points = 0U;
  auto result = RollingLidarTargetBuilder::create(invalid);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, RollingLidarTargetErrorCode::InvalidConfig);

  invalid = config();
  invalid.registration.maximum_targets = 0U;
  result = RollingLidarTargetBuilder::create(invalid);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, RollingLidarTargetErrorCode::InvalidConfig);
}

TEST(RollingLidarTarget, SealedAdmissionRejectsOversizeAndEvictsWholeSweeps) {
  auto builder_result = RollingLidarTargetBuilder::create(config(2U, 3U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();

  const auto oversize =
      cloud(1U, 10,
            {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}});
  auto rejected = builder.add(registration(1U, core::Pose3d(), oversize));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, RollingLidarTargetErrorCode::TargetCapacity);
  EXPECT_TRUE(builder.empty());

  const auto input = cloud(
      1U, 10, {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}});
  const LidarRegistrationCloud* const cloud_identity = input.get();
  auto added = builder.add(registration(1U, core::Pose3d(), input));
  ASSERT_TRUE(added) << added.error().detail;
  EXPECT_EQ(added.value().input_points, 3U);
  EXPECT_EQ(added.value().retained_points_from_input, 3U);
  EXPECT_EQ(added.value().copied_raw_points, 0U);
  EXPECT_EQ(added.value().deterministic_cap_discarded_points, 0U);

  const auto retained = builder.retainedSweeps();
  ASSERT_EQ(retained.size(), 1U);
  ASSERT_EQ(retained.front().cloud->points.size(), 3U);
  EXPECT_EQ(retained.front().cloud.get(), cloud_identity);
  EXPECT_EQ(retained.front().checksum, input->checksum);
  EXPECT_TRUE(core::contentHashPresent(retained.front().checksum));

  auto next = cloud(2U, 20, {{10.0, 0.0, 0.0}, {11.0, 0.0, 0.0}});
  auto replacement = builder.add(registration(2U, core::Pose3d(), next));
  ASSERT_TRUE(replacement);
  EXPECT_EQ(replacement.value().eviction.sweeps, 1U);
  EXPECT_EQ(replacement.value().eviction.points, 3U);
  EXPECT_EQ(replacement.value().retained_sweeps, 1U);
  EXPECT_EQ(replacement.value().retained_points, 2U);
}

TEST(RollingLidarTarget, SoleAdmissionSharesSealedRegistrationCloud) {
  auto builder_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto immutable_cloud = cloud(1U, 10, {{0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}});
  const LidarRegistrationCloud* const cloud_identity = immutable_cloud.get();

  auto admitted = builder.add(registration(1U, pose({5.0, 0.0, 0.0}), immutable_cloud));
  ASSERT_TRUE(admitted) << admitted.error().detail;
  EXPECT_EQ(admitted.value().copied_raw_points, 0U);
  const auto retained = builder.retainedSweeps();
  ASSERT_EQ(retained.size(), 1U);
  EXPECT_EQ(retained.front().cloud.get(), cloud_identity);
  EXPECT_DOUBLE_EQ(retained.front().T_odom_imu.translation().x(), 5.0);
}

TEST(RollingLidarTarget, PublishesNewestFirstPoseAwareRecordsWithoutMergingScans) {
  auto builder_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  const auto second = cloud(2U, 20, {{1.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, pose({3.0, 0.0, 0.0}), first)));
  ASSERT_TRUE(builder.add(registration(2U, pose({4.0, 0.0, 0.0}), second)));
  const auto source = cloud(3U, 30, {{0.0, 0.0, 0.0}}, pose({5.0, 0.0, 0.0}));

  auto batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(batch) << batch.error().detail;
  ASSERT_EQ(batch.value().targets.size(), 2U);
  EXPECT_EQ(batch.value().targets[0U].state, core::StateId(2U));
  EXPECT_EQ(batch.value().targets[1U].state, core::StateId(1U));
  EXPECT_NE(batch.value().targets[0U].cloud.get(), batch.value().targets[1U].cloud.get());
  EXPECT_EQ(batch.value().targets[0U].cloud->source_sweep, core::MeasurementId(2U));
  EXPECT_EQ(batch.value().targets[1U].cloud->source_sweep, core::MeasurementId(1U));
  EXPECT_EQ(batch.value().targets[0U].cloud->checksum, second->checksum);
  EXPECT_EQ(batch.value().targets[1U].cloud->checksum, first->checksum);
  EXPECT_DOUBLE_EQ(batch.value().targets[0U].T_odom_imu_target_seed.translation().x(), 4.0);
  EXPECT_DOUBLE_EQ(batch.value().targets[1U].T_odom_imu_target_seed.translation().x(), 3.0);
  EXPECT_DOUBLE_EQ(batch.value().targets[0U].T_target_source_seed.translation().x(), 1.0);
  EXPECT_DOUBLE_EQ(batch.value().targets[1U].T_target_source_seed.translation().x(), 2.0);
  EXPECT_EQ(batch.value().build.retained_candidates, 2U);
  EXPECT_EQ(batch.value().build.selected_targets, 2U);
  EXPECT_EQ(batch.value().build.selected_state_span, 1U);
  EXPECT_EQ(batch.value().build.selected_time_span_ns, 10);
  EXPECT_TRUE(core::contentHashPresent(batch.value().checksum));
  EXPECT_EQ(batch.value().source_cloud_checksum, source->checksum);
  EXPECT_TRUE(core::contentHashPresent(batch.value().lineage.checksum));
  ASSERT_EQ(batch.value().lineage.usage.size(), 2U);
  ASSERT_TRUE(std::holds_alternative<core::MeasurementId>(
      batch.value().lineage.usage[0U].slice.root));
  ASSERT_TRUE(std::holds_alternative<core::MeasurementId>(
      batch.value().lineage.usage[1U].slice.root));
  EXPECT_EQ(std::get<core::MeasurementId>(batch.value().lineage.usage[0U].slice.root),
            core::MeasurementId(1U));
  EXPECT_EQ(std::get<core::MeasurementId>(batch.value().lineage.usage[1U].slice.root),
            core::MeasurementId(2U));
  EXPECT_TRUE(std::none_of(batch.value().lineage.usage.begin(), batch.value().lineage.usage.end(),
                           [](const core::ObservationUsage& usage) {
                             const auto* source_id =
                                 std::get_if<core::MeasurementId>(&usage.slice.root);
                             return source_id != nullptr && *source_id == core::MeasurementId(3U);
                           }));
}

TEST(RollingLidarTarget, ScanLocalLimitReservesARegistrationChannelAndItsLineage) {
  auto builder_result = RollingLidarTargetBuilder::create(config(5U, 100U, 3U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), cloud(1U, 10, {{0.0, 0.0, 0.0}}))));
  ASSERT_TRUE(builder.add(registration(2U, core::Pose3d(), cloud(2U, 20, {{1.0, 0.0, 0.0}}))));
  ASSERT_TRUE(builder.add(registration(3U, core::Pose3d(), cloud(3U, 30, {{2.0, 0.0, 0.0}}))));
  const auto source = cloud(4U, 40, {{3.0, 0.0, 0.0}});

  auto limited =
      builder.buildBatch(source, core::ObservationLineageId(100U), std::size_t{2U});
  ASSERT_TRUE(limited) << limited.error().detail;
  ASSERT_EQ(limited.value().targets.size(), 2U);
  EXPECT_EQ(limited.value().targets[0U].state, core::StateId(3U));
  EXPECT_EQ(limited.value().targets[1U].state, core::StateId(1U));
  EXPECT_EQ(limited.value().build.retained_candidates, 3U);
  EXPECT_EQ(limited.value().build.selected_targets, 2U);
  EXPECT_EQ(limited.value().lineage.usage.size(), 2U);

  auto zero = builder.buildBatch(source, core::ObservationLineageId(101U), std::size_t{0U});
  ASSERT_FALSE(zero);
  EXPECT_EQ(zero.error().code, RollingLidarTargetErrorCode::TargetCapacity);
  auto excessive =
      builder.buildBatch(source, core::ObservationLineageId(102U), std::size_t{4U});
  ASSERT_FALSE(excessive);
  EXPECT_EQ(excessive.error().code, RollingLidarTargetErrorCode::TargetCapacity);
}

TEST(RollingLidarTarget, SelectsPoseOwnedTargetsAcrossCompleteRetainedHistoryWithoutPresearch) {
  auto selection_config = config(5U, 100U, 3U);
  auto builder_result = RollingLidarTargetBuilder::create(selection_config);
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  for (std::uint64_t state = 1U; state <= 5U; ++state) {
    ASSERT_TRUE(builder.add(registration(
        state, pose({static_cast<double>(state), 0.0, 0.0}),
        cloud(state, static_cast<std::int64_t>(state * 10U),
              {{100.0 * static_cast<double>(state), 0.0, 0.0}}))));
  }
  // Geometry deliberately has no overlap with any target. buildBatch owns no
  // speculative correspondence pass; registration alone decides support.
  const auto source = cloud(6U, 60, {{-1000.0, 0.0, 0.0}}, pose({6.0, 0.0, 0.0}));

  auto batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(batch) << batch.error().detail;
  ASSERT_EQ(batch.value().targets.size(), 3U);
  EXPECT_EQ(batch.value().targets[0U].state, core::StateId(5U));
  EXPECT_EQ(batch.value().targets[1U].state, core::StateId(3U));
  EXPECT_EQ(batch.value().targets[2U].state, core::StateId(1U));
  EXPECT_EQ(batch.value().build.retained_candidates, 5U);
  EXPECT_EQ(batch.value().build.selected_targets, 3U);
  EXPECT_EQ(batch.value().build.selected_state_span, 4U);
  EXPECT_EQ(batch.value().build.selected_time_span_ns, 40);
}

TEST(RollingLidarTarget, SelectionAndRelativeSeedsAreInvariantToCommonOdomTransform) {
  auto first_result = RollingLidarTargetBuilder::create(config());
  auto transformed_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(first_result);
  ASSERT_TRUE(transformed_result);
  auto first = std::move(first_result).value();
  auto transformed = std::move(transformed_result).value();
  const core::Pose3d G_new_odom =
      pose({-3.0, 2.0, 1.0}, Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitZ()).toRotationMatrix());
  const core::Pose3d T_odom_first = pose({1.0, 0.0, 0.0});
  const core::Pose3d T_odom_second = pose({2.0, 0.0, 0.0});
  const core::Pose3d T_odom_source = pose({3.0, 0.0, 0.0});
  const auto first_cloud = cloud(1U, 10, {{2.0, 0.0, 0.0}});
  const auto second_cloud = cloud(2U, 20, {{1.0, 0.0, 0.0}});
  ASSERT_TRUE(first.add(registration(1U, T_odom_first, first_cloud)));
  ASSERT_TRUE(first.add(registration(2U, T_odom_second, second_cloud)));
  ASSERT_TRUE(transformed.add(registration(1U, G_new_odom * T_odom_first, first_cloud)));
  ASSERT_TRUE(transformed.add(registration(2U, G_new_odom * T_odom_second, second_cloud)));

  auto first_batch = first.buildBatch(cloud(3U, 30, {{0.0, 0.0, 0.0}}, T_odom_source),
                                      core::ObservationLineageId(100U));
  auto transformed_batch =
      transformed.buildBatch(cloud(3U, 30, {{0.0, 0.0, 0.0}}, G_new_odom * T_odom_source),
                             core::ObservationLineageId(101U));
  ASSERT_TRUE(first_batch);
  ASSERT_TRUE(transformed_batch);
  ASSERT_EQ(first_batch.value().targets.size(), transformed_batch.value().targets.size());
  for (std::size_t index = 0U; index < first_batch.value().targets.size(); ++index) {
    EXPECT_EQ(first_batch.value().targets[index].state,
              transformed_batch.value().targets[index].state);
    EXPECT_TRUE(first_batch.value().targets[index].T_target_source_seed.matrix().isApprox(
        transformed_batch.value().targets[index].T_target_source_seed.matrix(), 1.0e-12));
  }
}

TEST(RollingLidarTarget, PoseSynchronizationIsAtomicAndNeverCopiesPointGeometry) {
  auto builder_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  const auto second = cloud(2U, 20, {{0.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, pose({0.0, 0.0, 0.0}), first)));
  ASSERT_TRUE(builder.add(registration(2U, pose({1.0, 0.0, 0.0}), second)));
  const auto before_retained = builder.retainedSweeps();
  const auto source = cloud(3U, 30, {{0.0, 0.0, 0.0}}, pose({12.0, 0.0, 0.0}));
  auto before_batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(before_batch);

  const std::vector<RollingLidarTargetPose> duplicate{{core::StateId(1U), pose({5.0, 0.0, 0.0})},
                                                      {core::StateId(1U), pose({6.0, 0.0, 0.0})}};
  auto rejected = builder.synchronizeCommittedPoses(core::OdomEpoch(7U), duplicate);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, RollingLidarTargetErrorCode::DuplicateState);
  const auto after_rejection = builder.retainedSweeps();
  EXPECT_DOUBLE_EQ(after_rejection[0U].T_odom_imu.translation().x(), 0.0);
  EXPECT_DOUBLE_EQ(after_rejection[1U].T_odom_imu.translation().x(), 1.0);

  const std::vector<RollingLidarTargetPose> revisions{{core::StateId(1U), pose({10.0, 0.0, 0.0})},
                                                      {core::StateId(2U), pose({11.0, 0.0, 0.0})}};
  auto synchronized = builder.synchronizeCommittedPoses(core::OdomEpoch(7U), revisions);
  ASSERT_TRUE(synchronized) << synchronized.error().detail;
  EXPECT_EQ(synchronized.value().matched_retained_sweeps, 2U);
  const auto after = builder.retainedSweeps();
  EXPECT_EQ(after[0U].cloud.get(), before_retained[0U].cloud.get());
  EXPECT_EQ(after[1U].cloud.get(), before_retained[1U].cloud.get());
  EXPECT_EQ(after[0U].checksum, before_retained[0U].checksum);
  EXPECT_EQ(after[1U].checksum, before_retained[1U].checksum);
  EXPECT_DOUBLE_EQ(after[0U].T_odom_imu.translation().x(), 10.0);
  EXPECT_DOUBLE_EQ(after[1U].T_odom_imu.translation().x(), 11.0);

  auto after_batch = builder.buildBatch(source, core::ObservationLineageId(101U));
  ASSERT_TRUE(after_batch);
  EXPECT_DOUBLE_EQ(after_batch.value().targets[0U].T_odom_imu_target_seed.translation().x(), 11.0);
  EXPECT_DOUBLE_EQ(after_batch.value().targets[0U].T_target_source_seed.translation().x(), 1.0);
  EXPECT_DOUBLE_EQ(after_batch.value().targets[1U].T_target_source_seed.translation().x(), 2.0);
  EXPECT_DOUBLE_EQ(before_batch.value().targets[0U].T_odom_imu_target_seed.translation().x(), 1.0);
  EXPECT_EQ(after_batch.value().targets[0U].cloud.get(),
            before_batch.value().targets[0U].cloud.get());
}

TEST(RollingLidarTarget, PoseSynchronizationEvictsOnlyExplicitlyFinalizedOwners) {
  auto builder_result = RollingLidarTargetBuilder::create(config(5U, 100U, 3U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  for (std::uint64_t state = 1U; state <= 5U; ++state) {
    ASSERT_TRUE(builder.add(registration(
        state, pose({static_cast<double>(state), 0.0, 0.0}),
        cloud(state, static_cast<std::int64_t>(state * 10U),
              {{static_cast<double>(state), 0.0, 0.0}}))));
  }

  // Graph commit pose snapshots include states finalized by the same
  // transaction. Exact explicit finality must win over those matching poses.
  const std::vector<RollingLidarTargetPose> active{
      {core::StateId(1U), pose({10.0, 0.0, 0.0})},
      {core::StateId(2U), pose({20.0, 0.0, 0.0})},
      {core::StateId(3U), pose({30.0, 0.0, 0.0})},
      {core::StateId(4U), pose({40.0, 0.0, 0.0})},
      {core::StateId(5U), pose({50.0, 0.0, 0.0})}};
  const std::vector<core::StateId> finalized{core::StateId(1U), core::StateId(2U)};
  auto synchronized =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), active, finalized);
  ASSERT_TRUE(synchronized) << synchronized.error().detail;
  EXPECT_EQ(synchronized.value().matched_retained_sweeps, 3U);
  EXPECT_EQ(synchronized.value().finalized_sweeps_evicted, 2U);
  EXPECT_EQ(synchronized.value().finalized_points_evicted, 2U);
  EXPECT_EQ(synchronized.value().retained_sweeps, 3U);
  EXPECT_EQ(synchronized.value().retained_points, 3U);

  const auto retained = builder.retainedSweeps();
  ASSERT_EQ(retained.size(), 3U);
  EXPECT_EQ(retained.front().state, core::StateId(3U));
  EXPECT_EQ(retained.back().state, core::StateId(5U));
  EXPECT_DOUBLE_EQ(retained.front().T_odom_imu.translation().x(), 30.0);
  EXPECT_EQ(builder.statistics().finalized_sweeps_evicted, 2U);
  EXPECT_EQ(builder.statistics().finalized_points_evicted, 2U);
}

TEST(RollingLidarTarget, MissingNonFinalCommittedPoseRejectsWithoutMutation) {
  auto builder_result = RollingLidarTargetBuilder::create(config(3U, 100U, 2U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  const auto second = cloud(2U, 20, {{1.0, 0.0, 0.0}});
  const auto third = cloud(3U, 30, {{2.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, pose({1.0, 0.0, 0.0}), first)));
  ASSERT_TRUE(builder.add(registration(2U, pose({2.0, 0.0, 0.0}), second)));
  ASSERT_TRUE(builder.add(registration(3U, pose({3.0, 0.0, 0.0}), third)));
  const auto before = builder.retainedSweeps();
  const RollingLidarTargetStatistics statistics_before = builder.statistics();

  // State 1 is explicitly final and would be evicted, but state 2 is neither
  // final nor present in the pose snapshot. The whole synchronization must
  // reject, including the staged state-1 eviction and state-3 pose revision.
  const std::vector<RollingLidarTargetPose> incomplete{
      {core::StateId(1U), pose({10.0, 0.0, 0.0})},
      {core::StateId(3U), pose({30.0, 0.0, 0.0})}};
  const std::vector<core::StateId> finalized{core::StateId(1U)};
  auto rejected =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), incomplete, finalized);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, RollingLidarTargetErrorCode::MissingCommittedPose);

  const auto after = builder.retainedSweeps();
  ASSERT_EQ(after.size(), before.size());
  for (std::size_t index = 0U; index < before.size(); ++index) {
    EXPECT_EQ(after[index].state, before[index].state);
    EXPECT_EQ(after[index].cloud.get(), before[index].cloud.get());
    EXPECT_TRUE(after[index].T_odom_imu.matrix().isApprox(before[index].T_odom_imu.matrix()));
  }
  EXPECT_EQ(builder.statistics().finalized_sweeps_evicted,
            statistics_before.finalized_sweeps_evicted);
  EXPECT_EQ(builder.statistics().finalized_points_evicted,
            statistics_before.finalized_points_evicted);
  EXPECT_EQ(builder.statistics().rejected_pose_synchronizations,
            statistics_before.rejected_pose_synchronizations + 1U);
}

TEST(RollingLidarTarget, RejectsInvalidDuplicateAndNonCanonicalFinalityAtomically) {
  auto builder_result = RollingLidarTargetBuilder::create(config(2U, 100U, 2U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  ASSERT_TRUE(builder.add(registration(
      1U, pose({1.0, 0.0, 0.0}), cloud(1U, 10, {{0.0, 0.0, 0.0}}))));
  ASSERT_TRUE(builder.add(registration(
      2U, pose({2.0, 0.0, 0.0}), cloud(2U, 20, {{1.0, 0.0, 0.0}}))));
  const auto before = builder.retainedSweeps();
  const std::vector<RollingLidarTargetPose> poses{
      {core::StateId(1U), pose({10.0, 0.0, 0.0})},
      {core::StateId(2U), pose({20.0, 0.0, 0.0})}};

  const std::vector<core::StateId> invalid{core::StateId{}};
  auto invalid_result =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), poses, invalid);
  ASSERT_FALSE(invalid_result);
  EXPECT_EQ(invalid_result.error().code, RollingLidarTargetErrorCode::InvalidIdentity);

  const std::vector<core::StateId> duplicate{core::StateId(1U), core::StateId(1U)};
  auto duplicate_result =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), poses, duplicate);
  ASSERT_FALSE(duplicate_result);
  EXPECT_EQ(duplicate_result.error().code, RollingLidarTargetErrorCode::DuplicateState);

  const std::vector<core::StateId> descending{core::StateId(2U), core::StateId(1U)};
  auto descending_result =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), poses, descending);
  ASSERT_FALSE(descending_result);
  EXPECT_EQ(descending_result.error().code, RollingLidarTargetErrorCode::NonMonotonicState);

  const auto after = builder.retainedSweeps();
  ASSERT_EQ(after.size(), before.size());
  for (std::size_t index = 0U; index < before.size(); ++index) {
    EXPECT_EQ(after[index].state, before[index].state);
    EXPECT_EQ(after[index].cloud.get(), before[index].cloud.get());
    EXPECT_TRUE(after[index].T_odom_imu.matrix().isApprox(before[index].T_odom_imu.matrix()));
  }
}

TEST(RollingLidarTarget, MixedNonContiguousFinalityRetainsAndRevisesEveryOtherOwner) {
  auto builder_result = RollingLidarTargetBuilder::create(config(5U, 100U, 3U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  for (std::uint64_t state = 1U; state <= 5U; ++state) {
    ASSERT_TRUE(builder.add(registration(
        state, pose({static_cast<double>(state), 0.0, 0.0}),
        cloud(state, static_cast<std::int64_t>(10U * state),
              {{static_cast<double>(state), 0.0, 0.0}}))));
  }

  std::vector<RollingLidarTargetPose> poses;
  for (std::uint64_t state = 1U; state <= 5U; ++state) {
    poses.push_back(RollingLidarTargetPose{
        core::StateId(state), pose({10.0 * static_cast<double>(state), 0.0, 0.0})});
  }
  const std::vector<core::StateId> finalized{core::StateId(2U), core::StateId(4U)};
  auto synchronized =
      builder.synchronizeCommittedPoses(core::OdomEpoch(7U), poses, finalized);
  ASSERT_TRUE(synchronized) << synchronized.error().detail;
  EXPECT_EQ(synchronized.value().finalized_sweeps_evicted, 2U);
  EXPECT_EQ(synchronized.value().matched_retained_sweeps, 3U);

  const auto retained = builder.retainedSweeps();
  ASSERT_EQ(retained.size(), 3U);
  EXPECT_EQ(retained[0U].state, core::StateId(1U));
  EXPECT_EQ(retained[1U].state, core::StateId(3U));
  EXPECT_EQ(retained[2U].state, core::StateId(5U));
  EXPECT_DOUBLE_EQ(retained[0U].T_odom_imu.translation().x(), 10.0);
  EXPECT_DOUBLE_EQ(retained[1U].T_odom_imu.translation().x(), 30.0);
  EXPECT_DOUBLE_EQ(retained[2U].T_odom_imu.translation().x(), 50.0);
}

TEST(RollingLidarTarget, RemovesExactlyRequestedBatchPayloadAndPreservesProvenance) {
  auto builder_result = RollingLidarTargetBuilder::create(config(4U, 100U, 2U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  const auto second = cloud(2U, 20, {{1.0, 0.0, 0.0}, {1.1, 0.0, 0.0}});
  const auto third = cloud(3U, 30,
                           {{2.0, 0.0, 0.0}, {2.1, 0.0, 0.0}, {2.2, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), first, core::OdomEpoch(7U),
                                       core::FactorBatchId(101U),
                                       core::SensorRecoveryEpoch(0U))));
  ASSERT_TRUE(builder.add(registration(2U, core::Pose3d(), second, core::OdomEpoch(7U),
                                       core::FactorBatchId(102U),
                                       core::SensorRecoveryEpoch(0U))));
  ASSERT_TRUE(builder.add(registration(3U, core::Pose3d(), third, core::OdomEpoch(7U),
                                       core::FactorBatchId(103U),
                                       core::SensorRecoveryEpoch(1U))));

  const auto before = builder.retainedSweeps();
  ASSERT_EQ(before.size(), 3U);
  EXPECT_EQ(before[0U].admitting_batch_id, core::FactorBatchId(101U));
  EXPECT_EQ(before[1U].admitting_batch_id, core::FactorBatchId(102U));
  EXPECT_EQ(before[2U].admitting_batch_id, core::FactorBatchId(103U));
  EXPECT_EQ(before[2U].recovery_epoch, core::SensorRecoveryEpoch(1U));

  const std::vector<core::FactorBatchId> removed_ids{core::FactorBatchId(102U)};
  auto removed = builder.removeFactorBatches(core::OdomEpoch(7U), removed_ids);
  ASSERT_TRUE(removed) << removed.error().detail;
  EXPECT_EQ(removed.value().requested_batches, 1U);
  EXPECT_EQ(removed.value().matched_batches, 1U);
  EXPECT_EQ(removed.value().absent_batches, 0U);
  EXPECT_EQ(removed.value().removed_sweeps, 1U);
  EXPECT_EQ(removed.value().removed_points, 2U);
  EXPECT_EQ(removed.value().retained_sweeps, 2U);
  EXPECT_EQ(removed.value().retained_points, 4U);

  const auto after = builder.retainedSweeps();
  ASSERT_EQ(after.size(), 2U);
  EXPECT_EQ(after[0U].cloud.get(), first.get());
  EXPECT_EQ(after[0U].admitting_batch_id, core::FactorBatchId(101U));
  EXPECT_EQ(after[0U].recovery_epoch, core::SensorRecoveryEpoch(0U));
  EXPECT_EQ(after[1U].cloud.get(), third.get());
  EXPECT_EQ(after[1U].admitting_batch_id, core::FactorBatchId(103U));
  EXPECT_EQ(after[1U].recovery_epoch, core::SensorRecoveryEpoch(1U));
  EXPECT_EQ(builder.statistics().removed_sweeps, 1U);
  EXPECT_EQ(builder.statistics().removed_points, 2U);
}

TEST(RollingLidarTarget, AbsentBatchRemovalIsInert) {
  auto builder_result = RollingLidarTargetBuilder::create(config(2U, 100U, 1U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto retained_cloud = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), retained_cloud,
                                       core::OdomEpoch(7U), core::FactorBatchId(101U))));
  const auto before = builder.retainedSweeps();

  const std::vector<core::FactorBatchId> absent_ids{core::FactorBatchId(999U)};
  auto removed = builder.removeFactorBatches(core::OdomEpoch(7U), absent_ids);
  ASSERT_TRUE(removed) << removed.error().detail;
  EXPECT_EQ(removed.value().matched_batches, 0U);
  EXPECT_EQ(removed.value().absent_batches, 1U);
  EXPECT_EQ(removed.value().removed_sweeps, 0U);
  EXPECT_EQ(removed.value().removed_points, 0U);
  EXPECT_EQ(removed.value().retained_sweeps, 1U);

  const auto after = builder.retainedSweeps();
  ASSERT_EQ(after.size(), 1U);
  EXPECT_EQ(after.front().cloud.get(), before.front().cloud.get());
  EXPECT_EQ(after.front().admitting_batch_id, before.front().admitting_batch_id);
  EXPECT_EQ(after.front().recovery_epoch, before.front().recovery_epoch);
  EXPECT_TRUE(after.front().T_odom_imu.matrix().isApprox(before.front().T_odom_imu.matrix()));
}

TEST(RollingLidarTarget, RejectedBatchRemovalIsAtomic) {
  auto builder_result = RollingLidarTargetBuilder::create(config(2U, 100U, 1U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  const auto second = cloud(2U, 20, {{1.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), first, core::OdomEpoch(7U),
                                       core::FactorBatchId(101U))));
  ASSERT_TRUE(builder.add(registration(2U, core::Pose3d(), second, core::OdomEpoch(7U),
                                       core::FactorBatchId(102U))));
  const auto assert_unchanged = [&builder, &first, &second]() {
    const auto retained = builder.retainedSweeps();
    ASSERT_EQ(retained.size(), 2U);
    EXPECT_EQ(retained[0U].cloud.get(), first.get());
    EXPECT_EQ(retained[1U].cloud.get(), second.get());
    EXPECT_EQ(retained[0U].admitting_batch_id, core::FactorBatchId(101U));
    EXPECT_EQ(retained[1U].admitting_batch_id, core::FactorBatchId(102U));
  };

  const std::vector<core::FactorBatchId> one_id{core::FactorBatchId(101U)};
  auto wrong_epoch = builder.removeFactorBatches(core::OdomEpoch(8U), one_id);
  ASSERT_FALSE(wrong_epoch);
  EXPECT_EQ(wrong_epoch.error().code, RollingLidarTargetErrorCode::EpochMismatch);
  assert_unchanged();

  const std::vector<core::FactorBatchId> duplicate_ids{core::FactorBatchId(101U),
                                                       core::FactorBatchId(101U)};
  auto duplicate = builder.removeFactorBatches(core::OdomEpoch(7U), duplicate_ids);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, RollingLidarTargetErrorCode::DuplicateFactorBatch);
  assert_unchanged();

  const std::vector<core::FactorBatchId> invalid_ids{core::FactorBatchId{}};
  auto invalid = builder.removeFactorBatches(core::OdomEpoch(7U), invalid_ids);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, RollingLidarTargetErrorCode::InvalidIdentity);
  assert_unchanged();

  const std::vector<core::FactorBatchId> over_capacity{
      core::FactorBatchId(101U), core::FactorBatchId(102U), core::FactorBatchId(103U)};
  auto unbounded = builder.removeFactorBatches(core::OdomEpoch(7U), over_capacity);
  ASSERT_FALSE(unbounded);
  EXPECT_EQ(unbounded.error().code, RollingLidarTargetErrorCode::RemovalCapacity);
  assert_unchanged();
}

TEST(RollingLidarTarget, RemovalDoesNotRewindMonotonicAdmissionFrontier) {
  auto builder_result = RollingLidarTargetBuilder::create(config(3U, 100U, 1U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), cloud(1U, 10, {{0.0, 0.0, 0.0}}),
                                       core::OdomEpoch(7U), core::FactorBatchId(101U),
                                       core::SensorRecoveryEpoch(0U))));
  ASSERT_TRUE(builder.add(registration(2U, core::Pose3d(), cloud(2U, 20, {{1.0, 0.0, 0.0}}),
                                       core::OdomEpoch(7U), core::FactorBatchId(102U),
                                       core::SensorRecoveryEpoch(1U))));
  const std::vector<core::FactorBatchId> newest{core::FactorBatchId(102U)};
  ASSERT_TRUE(builder.removeFactorBatches(core::OdomEpoch(7U), newest));

  auto replayed_removed_sweep =
      builder.buildBatch(cloud(2U, 30, {{2.0, 0.0, 0.0}}), core::ObservationLineageId(90U));
  ASSERT_FALSE(replayed_removed_sweep);
  EXPECT_EQ(replayed_removed_sweep.error().code, RollingLidarTargetErrorCode::DuplicateSweep);

  auto stale_state = builder.add(registration(
      2U, core::Pose3d(), cloud(3U, 30, {{2.0, 0.0, 0.0}}), core::OdomEpoch(7U),
      core::FactorBatchId(103U), core::SensorRecoveryEpoch(1U)));
  ASSERT_FALSE(stale_state);
  EXPECT_EQ(stale_state.error().code, RollingLidarTargetErrorCode::DuplicateState);

  auto stale_batch = builder.add(registration(
      3U, core::Pose3d(), cloud(3U, 30, {{2.0, 0.0, 0.0}}), core::OdomEpoch(7U),
      core::FactorBatchId(101U), core::SensorRecoveryEpoch(1U)));
  ASSERT_FALSE(stale_batch);
  EXPECT_EQ(stale_batch.error().code, RollingLidarTargetErrorCode::NonMonotonicFactorBatch);

  auto stale_epoch = builder.add(registration(
      3U, core::Pose3d(), cloud(3U, 30, {{2.0, 0.0, 0.0}}), core::OdomEpoch(7U),
      core::FactorBatchId(103U), core::SensorRecoveryEpoch(0U)));
  ASSERT_FALSE(stale_epoch);
  EXPECT_EQ(stale_epoch.error().code, RollingLidarTargetErrorCode::NonMonotonicRecoveryEpoch);

  auto fresh = builder.add(registration(
      3U, core::Pose3d(), cloud(3U, 30, {{2.0, 0.0, 0.0}}), core::OdomEpoch(7U),
      core::FactorBatchId(103U), core::SensorRecoveryEpoch(1U)));
  ASSERT_TRUE(fresh) << fresh.error().detail;
  EXPECT_EQ(fresh.value().retained_sweeps, 2U);
}

TEST(RollingLidarTarget, RejectsRetainedSourceAndIndexMismatchWithoutMutation) {
  auto builder_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto first = cloud(1U, 10, {{0.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), first)));

  auto retained_source =
      builder.buildBatch(builder.retainedSweeps().front().cloud, core::ObservationLineageId(10U));
  ASSERT_FALSE(retained_source);
  EXPECT_EQ(retained_source.error().code, RollingLidarTargetErrorCode::SourceAlreadyRetained);

  const auto mismatched_index =
      cloud(2U, 20, {{1.0, 0.0, 0.0}}, core::Pose3d{}, 0.30);
  auto invalid_cloud = builder.add(registration(2U, core::Pose3d(), mismatched_index));
  ASSERT_FALSE(invalid_cloud);
  EXPECT_EQ(invalid_cloud.error().code, RollingLidarTargetErrorCode::InvalidCloud);
  EXPECT_EQ(builder.retainedSweeps().size(), 1U);
}

TEST(RollingLidarTarget, RejectsSourceThatPredatesAcceptedTargetWithoutMutation) {
  auto builder_result = RollingLidarTargetBuilder::create(config());
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto target = cloud(10U, 20, {{0.0, 0.0, 0.0}});
  ASSERT_TRUE(builder.add(registration(10U, core::Pose3d(), target)));

  const auto older_sweep = cloud(9U, 30, {{0.0, 0.0, 0.0}});
  auto nonmonotonic_sweep =
      builder.buildBatch(older_sweep, core::ObservationLineageId(100U));
  ASSERT_FALSE(nonmonotonic_sweep);
  EXPECT_EQ(nonmonotonic_sweep.error().code,
            RollingLidarTargetErrorCode::NonMonotonicSweep);

  const auto older_time = cloud(11U, 10, {{0.0, 0.0, 0.0}});
  auto nonmonotonic_time =
      builder.buildBatch(older_time, core::ObservationLineageId(101U));
  ASSERT_FALSE(nonmonotonic_time);
  EXPECT_EQ(nonmonotonic_time.error().code, RollingLidarTargetErrorCode::NonMonotonicTime);

  const auto retained = builder.retainedSweeps();
  ASSERT_EQ(retained.size(), 1U);
  EXPECT_EQ(retained.front().cloud.get(), target.get());
}

TEST(RollingLidarTarget, PreservesLegacyAbsentLineageChecksumsWithoutWeakeningBatchIntegrity) {
  auto builder_result = RollingLidarTargetBuilder::create(config(2U, 100U, 1U));
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  auto target_lineage = lineage(1U);
  target_lineage.usage.front().slice.source_checksum = {};
  target_lineage.checksum.fill(7U);
  const auto target = cloud(1U, 10, {{0.0, 0.0, 0.0}}, core::Pose3d{}, 0.25,
                            std::move(target_lineage));
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), target)));
  auto source_lineage = lineage(2U);
  source_lineage.usage.front().slice.source_checksum = {};
  source_lineage.checksum.fill(9U);
  const auto source = cloud(2U, 20, {{0.0, 0.0, 0.0}}, core::Pose3d{}, 0.25,
                            std::move(source_lineage));

  auto batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(batch) << batch.error().detail;
  EXPECT_FALSE(core::contentHashPresent(batch.value().lineage.checksum));
  EXPECT_TRUE(core::contentHashPresent(batch.value().checksum));

  const auto recomputed = recomputeRollingLidarTargetBatchChecksum(batch.value());
  ASSERT_TRUE(recomputed) << recomputed.error().detail;
  EXPECT_EQ(recomputed.value(), batch.value().checksum);

  RollingLidarTargetBatch mutated = batch.value();
  mutated.lineage.id = core::ObservationLineageId(101U);
  const auto mutated_checksum = recomputeRollingLidarTargetBatchChecksum(mutated);
  ASSERT_TRUE(mutated_checksum) << mutated_checksum.error().detail;
  EXPECT_NE(mutated_checksum.value(), batch.value().checksum);

  mutated = batch.value();
  mutated.lineage.usage.front().consumer = core::DerivedRecordId(999U);
  const auto usage_checksum = recomputeRollingLidarTargetBatchChecksum(mutated);
  ASSERT_TRUE(usage_checksum) << usage_checksum.error().detail;
  EXPECT_NE(usage_checksum.value(), batch.value().checksum);

  mutated = batch.value();
  mutated.lineage.usage.front().correlation_group = core::CorrelationGroupId(7U);
  mutated.lineage.correlations.push_back(core::CorrelationDeclaration{
      core::CorrelationGroupId(7U), core::CorrelationPolicyRevision(1U),
      core::CorrelationTreatment::NotIndependent, 1.0, std::nullopt});
  const auto correlation_checksum = recomputeRollingLidarTargetBatchChecksum(mutated);
  ASSERT_TRUE(correlation_checksum) << correlation_checksum.error().detail;
  EXPECT_NE(correlation_checksum.value(), batch.value().checksum);
}

TEST(RollingLidarTarget, LeavesSparseSourceAdmissionToTypedPointRegistration) {
  auto integration_config = config(2U, 100U, 1U);
  integration_config.registration.minimum_correspondences = 3U;
  auto builder_result = RollingLidarTargetBuilder::create(integration_config);
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  ASSERT_TRUE(builder.add(registration(1U, core::Pose3d(), cloud(1U, 10, {{0.0, 0.0, 0.0}}))));
  const auto source = cloud(2U, 20, {{0.0, 0.0, 0.0}});

  auto batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(batch) << batch.error().detail;
  ASSERT_EQ(batch.value().targets.size(), 1U);
  EXPECT_EQ(batch.value().build.selected_targets, 1U);

  auto registered = registerLidarScan(core::StateId(2U), source, batch.value().targets,
                                          integration_config.registration);
  ASSERT_FALSE(registered);
  EXPECT_EQ(registered.error().code, LidarRegistrationErrorCode::InvalidSource);
}

TEST(RollingLidarTarget, BatchFeedsPoseAwarePointRegistrationDirectly) {
  auto integration_config = config(2U, 100U, 1U);
  integration_config.registration.target_voxel_resolution_m = 0.30;
  integration_config.registration.maximum_correspondence_distance_m = 0.30;
  integration_config.registration.maximum_source_points = 100U;
  integration_config.registration.maximum_target_points_per_target = 100U;
  integration_config.registration.minimum_correspondences = 20U;
  integration_config.registration.minimum_observable_rank = 1U;
  auto builder_result = RollingLidarTargetBuilder::create(integration_config);
  ASSERT_TRUE(builder_result);
  auto builder = std::move(builder_result).value();
  const auto geometry = planarGrid(8U, 6U);
  ASSERT_TRUE(
      builder.add(registration(1U, core::Pose3d(), cloud(1U, 10, geometry, core::Pose3d{}, 0.30))));
  const auto source = cloud(2U, 20, geometry, pose({0.05, 0.0, 0.0}), 0.30);
  auto batch = builder.buildBatch(source, core::ObservationLineageId(100U));
  ASSERT_TRUE(batch) << batch.error().detail;

  auto registered = registerLidarScan(core::StateId(2U), source, batch.value().targets,
                                          integration_config.registration);
  ASSERT_TRUE(registered) << registered.error().detail;
  ASSERT_FALSE(registered.value().target_snapshots.empty());
  EXPECT_EQ(registered.value().target_snapshots.front()->targetState(), core::StateId(1U));
  EXPECT_EQ(registered.value().target_snapshots.front()->sourcePointCount(), geometry.size());
  EXPECT_TRUE(core::contentHashPresent(registered.value().target_snapshots.front()->checksum()));
}

}  // namespace
}  // namespace meridian::local
