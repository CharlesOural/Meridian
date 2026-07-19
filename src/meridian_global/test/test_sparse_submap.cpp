#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string_view>
#include <vector>

#include "meridian/global/sparse_submap.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::RecordHeader header(std::uint64_t trace, std::int64_t time_ns = 0) {
  core::RecordHeader output;
  output.trace = core::TraceId(trace);
  output.producer = core::ProducerId(5U);
  output.session = core::SessionId(7U);
  output.created_at = core::FusionTime{time_ns};
  output.config = core::ConfigRevision(3U);
  output.direct_calibration = core::CalibrationEpoch(11U);
  return output;
}

[[nodiscard]] core::Pose3d pose(double x, double y = 0.0, double z = 0.0, double roll = 0.0,
                                double pitch = 0.0, double yaw = 0.0) {
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
  return core::Pose3d(Sophus::SO3d(rotation), Eigen::Vector3d{x, y, z});
}

[[nodiscard]] LocalStateContributionInput stagedState(std::uint64_t id, std::int64_t time_ns,
                                                      const core::Pose3d& T_odom_imu,
                                                      std::uint64_t retained_bytes = 64U) {
  LocalStateContributionInput output;
  output.header = header(100U + id, time_ns);
  output.state = core::StateId(id);
  output.exact_time = core::FusionTime{time_ns};
  output.odom_epoch = core::OdomEpoch(13U);
  output.calibration = core::CalibrationEpoch(11U);
  output.admitted_revision = core::LocalGraphRevision(id);
  output.provisional_estimate.T_odom_imu = T_odom_imu;
  output.retained_bytes = retained_bytes;
  return output;
}

[[nodiscard]] FinalizedLocalStateInput finalizedState(std::uint64_t id, std::int64_t time_ns,
                                                      const core::Pose3d& T_odom_imu,
                                                      double variance = 0.01) {
  FinalizedLocalStateInput output;
  output.header = header(200U + id, time_ns);
  output.state = core::StateId(id);
  output.exact_time = core::FusionTime{time_ns};
  output.odom_epoch = core::OdomEpoch(13U);
  output.calibration = core::CalibrationEpoch(11U);
  output.final_revision = core::LocalGraphRevision(100U + id);
  output.final_estimate.T_odom_imu = T_odom_imu;
  output.pose_covariance.matrix = core::Matrix6d::Identity() * variance;
  return output;
}

[[nodiscard]] core::ObservationLineage lineage(
    std::uint64_t lineage_id, std::uint64_t measurement_id,
    core::ObservationRole role = core::ObservationRole::PrimaryResidual) {
  core::ObservationSlice slice;
  slice.root = core::MeasurementId(measurement_id);
  slice.kind = core::SliceKind::Whole;
  slice.calibration = core::CalibrationEpoch(11U);
  slice.source_checksum[0] = static_cast<std::uint8_t>(measurement_id & 0xffU);
  core::ObservationUsage usage;
  usage.slice = slice;
  usage.role = role;
  usage.consumer = core::DerivedRecordId(lineage_id);
  if (role == core::ObservationRole::PrimaryResidual) {
    usage.factor_group = core::FactorGroupId(lineage_id);
  }
  core::ObservationLineage output;
  output.id = core::ObservationLineageId(lineage_id);
  output.usage.push_back(usage);
  return output;
}

[[nodiscard]] FinalizedFactorInput factor(
    std::uint64_t id, std::uint64_t state_id, std::int64_t time_ns, std::uint64_t measurement_id,
    FinalizedFactorKind kind = FinalizedFactorKind::VisualReprojection,
    std::uint64_t retained_bytes = 48U) {
  FinalizedFactorInput output;
  output.header = header(300U + id, time_ns);
  output.factor = core::FactorId(id);
  output.kind = kind;
  output.support_states.push_back(core::StateId(state_id));
  output.terminal_state = core::StateId(state_id);
  output.terminal_time = core::FusionTime{time_ns};
  output.final_revision = core::LocalGraphRevision(200U + id);
  output.lineage = lineage(1000U + id, measurement_id,
                           kind == FinalizedFactorKind::IncomingMarginalPrior
                               ? core::ObservationRole::DerivedSummary
                               : core::ObservationRole::PrimaryResidual);
  output.retained_bytes = retained_bytes;
  return output;
}

[[nodiscard]] PlacePayloadInput payload(std::string_view text) {
  auto mutable_bytes = std::make_shared<std::vector<std::byte>>();
  mutable_bytes->reserve(text.size());
  for (const char character : text) {
    mutable_bytes->push_back(static_cast<std::byte>(character));
  }
  PlacePayloadInput output;
  output.in_memory = mutable_bytes;
  return output;
}

[[nodiscard]] FinalizedBoundaryCondensationInput condensation(
    std::uint64_t from, std::uint64_t to, std::vector<core::FactorId> factors = {},
    std::size_t rank = 6U, double endpoint_variance = 0.01) {
  FinalizedBoundaryCondensationInput output;
  output.header = header(400U + to);
  output.from_boundary_state = core::StateId(from);
  output.to_boundary_state = core::StateId(to);
  output.final_revision = core::LocalGraphRevision(500U + to);
  output.joint_pose_covariance.setIdentity();
  output.joint_pose_covariance *= endpoint_variance;
  output.supported_relative_rank = rank;
  output.eliminated_factor_ids = std::move(factors);
  return output;
}

[[nodiscard]] LocalFinalityBarrierInput barrier(std::int64_t time_ns,
                                                std::uint64_t revision = 1000U) {
  LocalFinalityBarrierInput output;
  output.header = header(5000U + revision, time_ns);
  output.finalized_through = core::FusionTime{time_ns};
  output.final_revision = core::LocalGraphRevision(revision);
  return output;
}

void expectSuccess(const core::Result<bool, SparseSubmapError>& result) {
  if (!result) {
    ADD_FAILURE() << result.error().detail;
  }
  ASSERT_TRUE(result);
}

[[nodiscard]] std::array<std::uint8_t, 32> sha256Abc() {
  return {0xbaU, 0x78U, 0x16U, 0xbfU, 0x8fU, 0x01U, 0xcfU, 0xeaU, 0x41U, 0x41U, 0x40U,
          0xdeU, 0x5dU, 0xaeU, 0x22U, 0x23U, 0xb0U, 0x03U, 0x61U, 0xa3U, 0x96U, 0x17U,
          0x7aU, 0x9cU, 0xb4U, 0x10U, 0xffU, 0x61U, 0xf2U, 0x00U, 0x15U, 0xadU};
}

TEST(SparseSubmapCoordinator, DefaultConfigurationConstructsAndZeroTypedIdsAreRejected) {
  SparseSubmapCoordinator coordinator;
  EXPECT_FALSE(coordinator.status().started);

  for (std::size_t which = 0; which < 3U; ++which) {
    SparseSubmapConfig config;
    if (which == 0U) {
      config.split.revision = SparseSubmapPolicyRevision(0U);
    } else if (which == 1U) {
      config.first_submap_id = core::SubmapId(0U);
    } else {
      config.content_revision = core::SubmapContentRevision(0U);
    }
    SparseSubmapCoordinator invalid(config);
    const auto result = invalid.stageState(stagedState(1U, 0, pose(0.0)));
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, SparseSubmapErrorCode::InvalidConfiguration);
  }
}

TEST(SparseSubmapCoordinator, SplitPolicyReportsEveryTriggeredCauseInStableOrder) {
  SparseSubmapConfig config;
  config.split.maximum_travel_m = 0.5;
  config.split.maximum_rotation_rad = 0.2;
  config.split.maximum_duration = core::Duration{5};
  config.split.maximum_keyframes = 1U;
  config.split.maximum_payload_bytes = 1U;
  SparseSubmapCoordinator coordinator(config);
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));

  FinalizedVisualKeyframeInput visual;
  visual.header = header(30U);
  visual.frame = core::CameraFrameId(1U);
  visual.camera = core::CameraId(2U);
  visual.state = core::StateId(1U);
  visual.terminal_time = core::FusionTime{0};
  visual.final_revision = core::LocalGraphRevision(4U);
  visual.place_payload = payload("abc");
  visual.lineage = lineage(20U, 20U, core::ObservationRole::RetrievalSeedOnly);
  visual.retained_bytes = 8U;
  expectSuccess(coordinator.stageVisualKeyframe(std::move(visual)));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0, 0.0, 0.0, 0.0, 0.0, 0.4))));

  const auto split = coordinator.considerSplit(SparseSubmapSplitRequest{core::StateId(2U), true});
  ASSERT_TRUE(split);
  EXPECT_TRUE(split.value().split_requested);
  const std::vector<SparseSubmapSplitReason> expected{
      SparseSubmapSplitReason::LocalReset,    SparseSubmapSplitReason::Travel,
      SparseSubmapSplitReason::Rotation,      SparseSubmapSplitReason::Duration,
      SparseSubmapSplitReason::KeyframeCount, SparseSubmapSplitReason::PayloadBytes};
  EXPECT_EQ(split.value().reasons, expected);
  EXPECT_EQ(coordinator.status().finalizing_submap, core::SubmapId(1U));
  EXPECT_EQ(coordinator.status().active_submap, core::SubmapId(2U));
}

TEST(SparseSubmapCoordinator, SealWaitsForEveryStateAndTheExplicitFinalityBarrier) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageState(stagedState(2U, 5, pose(0.5))));
  expectSuccess(coordinator.stageState(stagedState(3U, 10, pose(1.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(3U), true}));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 3U)));
  expectSuccess(coordinator.advanceFinality(barrier(10)));
  EXPECT_TRUE(coordinator.takeSealed().empty());

  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(3U, 10, pose(1.0))));
  EXPECT_TRUE(coordinator.takeSealed().empty());
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 5, pose(0.5))));
  const auto seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  EXPECT_EQ(seals.front()->core_interval.start, core::FusionTime{0});
  EXPECT_EQ(seals.front()->core_interval.end, core::FusionTime{10});
  EXPECT_EQ(seals.front()->core_state_ids,
            (std::vector<core::StateId>{core::StateId(1U), core::StateId(2U)}));
  EXPECT_EQ(seals.front()->storage_semantics, SparseSealStorageSemantics::VolatileInProcessOnly);
  EXPECT_EQ(coordinator.sealIdentity(seals.front()->submap.ref), seals.front()->identity);
  core::SubmapRef altered = seals.front()->submap.ref;
  altered.local_content_checksum[0] ^= 0xffU;
  EXPECT_FALSE(coordinator.sealIdentity(altered).has_value());
}

TEST(SparseSubmapCoordinator, BoundaryTimestampBelongsOnlyToTheSuccessorPartition) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  expectSuccess(coordinator.stageFactor(factor(1U, 1U, 0, 100U)));
  expectSuccess(coordinator.stageFactor(factor(2U, 2U, 10, 200U)));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 10, pose(1.0))));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 2U, {core::FactorId(1U)})));
  expectSuccess(coordinator.advanceFinality(barrier(10)));
  auto seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  EXPECT_EQ(seals.front()->condensed_factor_ids, (std::vector<core::FactorId>{core::FactorId(1U)}));

  expectSuccess(coordinator.stageState(stagedState(3U, 20, pose(2.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(3U, 20, pose(2.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(3U), true}));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(2U, 3U, {core::FactorId(2U)})));
  expectSuccess(coordinator.advanceFinality(barrier(20, 2000U)));
  seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  EXPECT_EQ(seals.front()->core_interval.start, core::FusionTime{10});
  EXPECT_EQ(seals.front()->condensed_factor_ids, (std::vector<core::FactorId>{core::FactorId(2U)}));
}

TEST(SparseSubmapCoordinator, AbortReassignsTheBoundedSuccessorWithoutIdGap) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(coordinator.stageState(stagedState(3U, 20, pose(2.0))));
  expectSuccess(coordinator.stageFactor(factor(3U, 3U, 20, 300U)));
  expectSuccess(coordinator.abortProvisionalSplit());
  EXPECT_FALSE(coordinator.status().finalizing_submap.has_value());
  EXPECT_EQ(coordinator.status().active_submap, core::SubmapId(1U));

  expectSuccess(coordinator.stageState(stagedState(4U, 30, pose(3.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(4U), true}));
  EXPECT_EQ(coordinator.status().active_submap, core::SubmapId(2U));
  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 10, pose(1.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(3U, 20, pose(2.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(4U, 30, pose(3.0))));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 4U, {core::FactorId(3U)})));
  expectSuccess(coordinator.advanceFinality(barrier(30)));
  ASSERT_EQ(coordinator.takeSealed().size(), 1U);
}

TEST(SparseSubmapCoordinator, BuildsGravityAlignedProxyAndBoundedPlaceIndexes) {
  SparseSubmapConfig config;
  config.registration_voxel_resolution_m = 1.0;
  SparseSubmapCoordinator coordinator(config);
  const core::Pose3d first_pose = pose(3.0, -2.0, 0.5, 0.25, -0.15, 0.7);
  const core::Pose3d second_pose = pose(4.0, -1.0, 0.6, -0.1, 0.2, 0.9);
  expectSuccess(coordinator.stageState(stagedState(1U, 0, first_pose)));

  FinalizedLidarKeyframeInput lidar;
  lidar.header = header(600U);
  lidar.sweep = core::SweepId(9U);
  lidar.state = core::StateId(1U);
  lidar.terminal_time = core::FusionTime{0};
  lidar.final_revision = core::LocalGraphRevision(4U);
  lidar.registration_samples = {
      LidarProxySample{Eigen::Vector3d{0.20, 0.0, 0.0}, Eigen::Vector3d::UnitZ(), 2.0},
      LidarProxySample{Eigen::Vector3d{0.21, 0.0, 0.0}, Eigen::Vector3d::UnitZ(), 1.0}};
  lidar.place_payload = payload("abc");
  lidar.lineage = lineage(60U, 60U, core::ObservationRole::RetrievalSeedOnly);
  lidar.retained_bytes = 32U;
  expectSuccess(coordinator.stageLidarKeyframe(std::move(lidar)));

  FinalizedVisualKeyframeInput visual;
  visual.header = header(601U);
  visual.frame = core::CameraFrameId(10U);
  visual.camera = core::CameraId(2U);
  visual.state = core::StateId(1U);
  visual.terminal_time = core::FusionTime{0};
  visual.final_revision = core::LocalGraphRevision(4U);
  visual.place_payload = payload("abc");
  visual.lineage = lineage(61U, 61U, core::ObservationRole::RetrievalSeedOnly);
  visual.retained_bytes = 32U;
  expectSuccess(coordinator.stageVisualKeyframe(std::move(visual)));

  expectSuccess(coordinator.stageState(stagedState(2U, 10, second_pose)));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, first_pose)));
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 10, second_pose)));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 2U)));
  expectSuccess(coordinator.advanceFinality(barrier(10)));
  const auto seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  const auto& seal = *seals.front();

  EXPECT_LT(std::abs(seal.submap.T_odom_submap.so3().matrix()(2, 0)), 1.0e-12);
  EXPECT_LT(std::abs(seal.submap.T_odom_submap.so3().matrix()(2, 1)), 1.0e-12);
  EXPECT_NEAR(seal.submap.T_odom_submap.so3().matrix()(2, 2), 1.0, 1.0e-12);
  EXPECT_LT((seal.submap.T_odom_submap.translation() - first_pose.translation()).norm(), 1.0e-12);
  EXPECT_EQ(seal.registration_proxy.points.size(), 1U);
  ASSERT_EQ(seal.finalized_trajectory.size(), 1U);
  EXPECT_EQ(seal.finalized_trajectory.front().state, core::StateId(1U));
  EXPECT_LT(seal.finalized_trajectory.front().T_submap_imu.translation().norm(), 1.0e-12);
  EXPECT_NEAR(seal.finalized_trajectory.front().velocity_submap.norm(), 0.0, 1.0e-12);
  ASSERT_EQ(seal.lidar_place_index.size(), 1U);
  ASSERT_EQ(seal.visual_place_index.size(), 1U);
  // Public indirect known-answer test for the SHA-256 implementation.
  EXPECT_EQ(seal.visual_place_index.front().payload.checksum, sha256Abc());
  EXPECT_EQ(seal.lidar_place_index.front().payload.checksum, sha256Abc());
}

TEST(SparseSubmapCoordinator, RejectsOverlappingPrimaryLineageAcrossCorePartitions) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageFactor(factor(1U, 1U, 0, 42U)));
  const auto duplicate = coordinator.stageFactor(factor(2U, 1U, 0, 42U));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, SparseSubmapErrorCode::NonDisjointLineage);
}

TEST(SparseSubmapCoordinator, CondensationRequiresExactPartitionAndExcludesIncomingPrior) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageFactor(factor(1U, 1U, 0, 1U)));
  expectSuccess(
      coordinator.stageFactor(factor(2U, 1U, 0, 2U, FinalizedFactorKind::IncomingMarginalPrior)));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));

  const auto with_prior = coordinator.submitBoundaryCondensation(
      condensation(1U, 2U, {core::FactorId(1U), core::FactorId(2U)}));
  ASSERT_FALSE(with_prior);
  EXPECT_EQ(with_prior.error().code, SparseSubmapErrorCode::InvalidCondensation);
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 2U, {core::FactorId(1U)})));
}

TEST(SparseSubmapCoordinator, ConsecutiveLegacySealsRetainTheirAdjacentPayload) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(2.0, -1.0, 0.2, 0.1, -0.1, 0.3))));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(3.0, -0.5, 0.2, -0.1, 0.1, 0.5))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(
      coordinator.finalizeState(finalizedState(1U, 0, pose(2.0, -1.0, 0.2, 0.1, -0.1, 0.3))));
  expectSuccess(
      coordinator.finalizeState(finalizedState(2U, 10, pose(3.0, -0.5, 0.2, -0.1, 0.1, 0.5))));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 2U)));
  expectSuccess(coordinator.advanceFinality(barrier(10)));
  auto seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  const SparseSubmapSeal first = seals.front();
  EXPECT_FALSE(first->incoming_adjacent.has_value());

  expectSuccess(coordinator.stageState(stagedState(3U, 20, pose(4.0, 0.0, 0.3, 0.0, 0.0, 0.7))));
  expectSuccess(
      coordinator.finalizeState(finalizedState(3U, 20, pose(4.0, 0.0, 0.3, 0.0, 0.0, 0.7))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(3U), true}));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(2U, 3U)));
  expectSuccess(coordinator.advanceFinality(barrier(20, 2000U)));
  seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  const SparseSubmapSeal second = seals.front();
  ASSERT_TRUE(second->incoming_adjacent.has_value());
  EXPECT_EQ(second->incoming_adjacent->global_append.constraint.information.rank, 6U);
  EXPECT_TRUE(second->incoming_adjacent->relative_covariance.finite());
}

TEST(SparseSubmapCoordinator, PreservesTheCondensationSupportedSubspace) {
  SparseSubmapCoordinator coordinator;
  expectSuccess(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, pose(0.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 10, pose(1.0))));
  auto partial = condensation(1U, 2U, {}, 3U);
  // Supported directions are x, y, and yaw rather than an inferred prefix of
  // the covariance eigensystem.
  partial.relative_information_basis.col(0) = core::Matrix6d::Identity().col(0);
  partial.relative_information_basis.col(1) = core::Matrix6d::Identity().col(1);
  partial.relative_information_basis.col(2) = core::Matrix6d::Identity().col(5);
  partial.relative_information_basis.col(3) = core::Matrix6d::Identity().col(2);
  partial.relative_information_basis.col(4) = core::Matrix6d::Identity().col(3);
  partial.relative_information_basis.col(5) = core::Matrix6d::Identity().col(4);
  expectSuccess(coordinator.submitBoundaryCondensation(std::move(partial)));
  expectSuccess(coordinator.advanceFinality(barrier(10)));
  ASSERT_EQ(coordinator.takeSealed().size(), 1U);

  expectSuccess(coordinator.stageState(stagedState(3U, 20, pose(2.0))));
  expectSuccess(coordinator.finalizeState(finalizedState(3U, 20, pose(2.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(3U), true}));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(2U, 3U)));
  expectSuccess(coordinator.advanceFinality(barrier(20, 2000U)));
  const auto seals = coordinator.takeSealed();
  ASSERT_EQ(seals.size(), 1U);
  ASSERT_TRUE(seals.front()->incoming_adjacent.has_value());
  const auto& information = seals.front()->incoming_adjacent->global_append.constraint.information;
  EXPECT_EQ(information.rank, 3U);
  EXPECT_GT(information.eigenvalues.head<3>().minCoeff(), 0.0);
  EXPECT_NEAR(information.eigenvalues.tail<3>().norm(), 0.0, 1.0e-15);
  EXPECT_LT((information.basis.transpose() * information.basis - core::Matrix6d::Identity())
                .cwiseAbs()
                .maxCoeff(),
            1.0e-10);
  const core::Matrix6d projector =
      information.basis.leftCols<3>() * information.basis.leftCols<3>().transpose();
  core::Matrix6d expected_projector = core::Matrix6d::Zero();
  expected_projector(0, 0) = 1.0;
  expected_projector(1, 1) = 1.0;
  expected_projector(5, 5) = 1.0;
  EXPECT_LT((projector - expected_projector).cwiseAbs().maxCoeff(), 1.0e-10);
}

[[nodiscard]] SparseSubmapSeal buildDeterministicSeal(bool reverse_factor_order) {
  SparseSubmapCoordinator coordinator;
  EXPECT_TRUE(coordinator.stageState(stagedState(1U, 0, pose(0.0))));
  EXPECT_TRUE(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  std::array<FinalizedFactorInput, 2> factors{factor(1U, 1U, 0, 101U), factor(2U, 1U, 0, 102U)};
  if (reverse_factor_order) {
    std::reverse(factors.begin(), factors.end());
  }
  for (auto& input : factors) {
    EXPECT_TRUE(coordinator.stageFactor(std::move(input)));
  }
  EXPECT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  EXPECT_TRUE(coordinator.finalizeState(finalizedState(1U, 0, pose(0.0))));
  EXPECT_TRUE(coordinator.finalizeState(finalizedState(2U, 10, pose(1.0))));
  EXPECT_TRUE(coordinator.submitBoundaryCondensation(
      condensation(1U, 2U, {core::FactorId(2U), core::FactorId(1U)})));
  EXPECT_TRUE(coordinator.advanceFinality(barrier(10)));
  auto seals = coordinator.takeSealed();
  EXPECT_EQ(seals.size(), 1U);
  return seals.empty() ? SparseSubmapSeal{} : seals.front();
}

TEST(SparseSubmapCoordinator, SealChecksumIsStableUnderCanonicalFactorReordering) {
  const SparseSubmapSeal forward = buildDeterministicSeal(false);
  const SparseSubmapSeal reverse = buildDeterministicSeal(true);
  ASSERT_TRUE(forward);
  ASSERT_TRUE(reverse);
  EXPECT_EQ(forward->identity, reverse->identity);
  EXPECT_FALSE(std::all_of(forward->identity.seal_checksum.begin(),
                           forward->identity.seal_checksum.end(),
                           [](std::uint8_t value) { return value == 0U; }));
}

TEST(SparseSubmapCoordinator, HardCapsFailWithoutSilentlyTruncatingAcceptedArtifacts) {
  SparseSubmapConfig state_config;
  state_config.maximum_staged_states = 2U;
  SparseSubmapCoordinator state_limited(state_config);
  expectSuccess(state_limited.stageState(stagedState(1U, 0, pose(0.0))));
  expectSuccess(state_limited.stageState(stagedState(2U, 10, pose(1.0))));
  const auto third = state_limited.stageState(stagedState(3U, 20, pose(2.0)));
  ASSERT_FALSE(third);
  EXPECT_EQ(third.error().code, SparseSubmapErrorCode::CapacityExceeded);
  EXPECT_EQ(state_limited.status().staged_states, 2U);

  SparseSubmapConfig proxy_config;
  proxy_config.maximum_registration_proxy_points = 1U;
  proxy_config.registration_voxel_resolution_m = 0.1;
  SparseSubmapCoordinator proxy_limited(proxy_config);
  expectSuccess(proxy_limited.stageState(stagedState(10U, 0, pose(0.0))));
  FinalizedLidarKeyframeInput lidar;
  lidar.header = header(900U);
  lidar.sweep = core::SweepId(90U);
  lidar.state = core::StateId(10U);
  lidar.terminal_time = core::FusionTime{0};
  lidar.final_revision = core::LocalGraphRevision(90U);
  lidar.T_imu_lidar = core::Pose3d{};
  lidar.registration_samples = {
      LidarProxySample{Eigen::Vector3d{0.0, 0.0, 0.0}, Eigen::Vector3d::UnitZ(), 1.0},
      LidarProxySample{Eigen::Vector3d{1.0, 0.0, 0.0}, Eigen::Vector3d::UnitZ(), 1.0}};
  lidar.lineage = lineage(90U, 90U, core::ObservationRole::RetrievalSeedOnly);
  lidar.retained_bytes = 8U;
  expectSuccess(proxy_limited.stageLidarKeyframe(std::move(lidar)));
  expectSuccess(proxy_limited.stageState(stagedState(11U, 10, pose(1.0))));
  ASSERT_TRUE(proxy_limited.considerSplit({core::StateId(11U), true}));
  expectSuccess(proxy_limited.finalizeState(finalizedState(10U, 0, pose(0.0))));
  expectSuccess(proxy_limited.finalizeState(finalizedState(11U, 10, pose(1.0))));
  expectSuccess(proxy_limited.submitBoundaryCondensation(condensation(10U, 11U)));
  const auto failed_seal = proxy_limited.advanceFinality(barrier(10));
  ASSERT_FALSE(failed_seal);
  EXPECT_EQ(failed_seal.error().code, SparseSubmapErrorCode::CapacityExceeded);
  EXPECT_TRUE(proxy_limited.takeSealed().empty());
}

TEST(SparseSubmapCoordinator, RejectsUnobservableBoundaryYawAndInvalidRank) {
  SparseSubmapCoordinator coordinator;
  const core::Pose3d vertical_x = pose(0.0, 0.0, 0.0, 0.0, -std::numbers::pi / 2.0, 0.0);
  expectSuccess(coordinator.stageState(stagedState(1U, 0, vertical_x)));
  expectSuccess(coordinator.stageState(stagedState(2U, 10, pose(1.0))));
  ASSERT_TRUE(coordinator.considerSplit({core::StateId(2U), true}));
  expectSuccess(coordinator.finalizeState(finalizedState(1U, 0, vertical_x)));
  expectSuccess(coordinator.finalizeState(finalizedState(2U, 10, pose(1.0))));
  expectSuccess(coordinator.submitBoundaryCondensation(condensation(1U, 2U)));
  const auto yaw_failure = coordinator.advanceFinality(barrier(10));
  ASSERT_FALSE(yaw_failure);
  EXPECT_EQ(yaw_failure.error().code, SparseSubmapErrorCode::BoundaryYawUnobservable);

  SparseSubmapCoordinator invalid_rank;
  expectSuccess(invalid_rank.stageState(stagedState(10U, 0, pose(0.0))));
  expectSuccess(invalid_rank.stageState(stagedState(11U, 10, pose(1.0))));
  ASSERT_TRUE(invalid_rank.considerSplit({core::StateId(11U), true}));
  auto bad = condensation(10U, 11U);
  bad.supported_relative_rank = 0U;
  const auto rank_failure = invalid_rank.submitBoundaryCondensation(std::move(bad));
  ASSERT_FALSE(rank_failure);
  EXPECT_EQ(rank_failure.error().code, SparseSubmapErrorCode::InvalidCondensation);
}

}  // namespace
}  // namespace meridian::global
