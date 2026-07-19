#include <gtest/gtest.h>

#include <cstddef>
#include <utility>

#include "meridian/global/global_coordinator.hpp"
#include "sparse_seal_test_utils.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::SparseSubmapSeal inEpoch(core::SparseSubmapSeal seal, core::OdomEpoch epoch) {
  seal.ref.odom_epoch = epoch;
  test_support::finalizeLocalContent(&seal);
  test_support::finalizeSealEnvelope(&seal);
  return seal;
}

TEST(GlobalCoordinator, CanonicalSealChainInitializesAndAppendsExactBoundaryGraph) {
  GlobalCoordinator coordinator;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  const core::SparseSubmapSeal second = test_support::successor(first);

  const auto initialized = coordinator.ingestSeal(first);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  EXPECT_EQ(initialized.value().disposition, GlobalSealDisposition::MissionInitialized);
  ASSERT_TRUE(initialized.value().graph_commit.has_value());
  EXPECT_EQ(initialized.value().graph_commit->solve.scalar_dimension, 6U);
  EXPECT_EQ(initialized.value().graph_commit->solve.materialized_navigation_boundaries, 0U);

  const auto appended = coordinator.ingestSeal(second);
  ASSERT_TRUE(appended) << appended.error().detail;
  EXPECT_EQ(appended.value().disposition, GlobalSealDisposition::AdjacentCommitted);
  ASSERT_TRUE(appended.value().graph_commit.has_value());
  EXPECT_EQ(appended.value().graph_commit->solve.scalar_dimension, 30U);
  EXPECT_EQ(appended.value().graph_commit->solve.materialized_navigation_boundaries, 2U);
  EXPECT_EQ(appended.value().graph_commit->solve.adjacent_factors, 1U);

  const GlobalCoordinatorStatus status = coordinator.status();
  EXPECT_TRUE(status.graph_initialized);
  EXPECT_EQ(status.accepted_seals, 2U);
  EXPECT_EQ(status.connected_seals, 2U);
  EXPECT_EQ(status.committed_graph_revision, GlobalGraphRevision(1U));
  const auto map_odom = coordinator.mapOdom(first.ref.odom_epoch);
  ASSERT_TRUE(map_odom.has_value());
  EXPECT_EQ(map_odom->reference_submap, second.ref);
}

TEST(GlobalCoordinator, IdempotentRedeliveryDoesNotMutateGraphOrBookkeeping) {
  GlobalCoordinator coordinator;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  const auto initialized = coordinator.ingestSeal(first);
  ASSERT_TRUE(initialized);
  const GlobalCoordinatorStatus before = coordinator.status();

  const auto duplicate = coordinator.ingestSeal(first);
  ASSERT_TRUE(duplicate);
  EXPECT_EQ(duplicate.value().disposition, GlobalSealDisposition::DuplicateIdempotent);
  EXPECT_FALSE(duplicate.value().graph_mutated);
  EXPECT_FALSE(duplicate.value().graph_commit.has_value());
  const GlobalCoordinatorStatus after = coordinator.status();
  EXPECT_EQ(after.accepted_seals, before.accepted_seals);
  EXPECT_EQ(after.committed_graph_revision, before.committed_graph_revision);
}

TEST(GlobalCoordinator, ExactPredecessorMismatchIsRejectedWithoutAllocatorOrRevisionMutation) {
  GlobalCoordinator coordinator;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  core::SparseSubmapSeal second = test_support::successor(first);
  ASSERT_TRUE(coordinator.ingestSeal(first));
  const GlobalCoordinatorStatus before = coordinator.status();

  core::SubmapRef altered = first.ref;
  altered.local_content_checksum.front() ^= 0x3cU;
  test_support::replacePredecessorRef(&second, altered);
  ASSERT_TRUE(core::verifyCanonicalSparseSubmapSeal(second));
  const auto rejected = coordinator.ingestSeal(second);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalCoordinatorErrorCode::StaleOrOutOfOrderSeal);
  const GlobalCoordinatorStatus after = coordinator.status();
  EXPECT_EQ(after.accepted_seals, before.accepted_seals);
  EXPECT_EQ(after.committed_graph_revision, before.committed_graph_revision);
  EXPECT_EQ(after.connected_seals, before.connected_seals);
}

TEST(GlobalCoordinator, LaterOdomEpochIsRetainedExplicitlyUntilAtomicConnectorExists) {
  GlobalCoordinator coordinator;
  const core::SparseSubmapSeal connected = test_support::firstSeal();
  ASSERT_TRUE(coordinator.ingestSeal(connected));
  const core::SparseSubmapSeal pending_first =
      inEpoch(test_support::firstSeal(20U, 1000), core::OdomEpoch(9U));
  const core::SparseSubmapSeal pending_second = test_support::successor(pending_first);

  const auto first_report = coordinator.ingestSeal(pending_first);
  ASSERT_TRUE(first_report);
  EXPECT_EQ(first_report.value().disposition, GlobalSealDisposition::PendingUnconnectedStored);
  const auto second_report = coordinator.ingestSeal(pending_second);
  ASSERT_TRUE(second_report);
  EXPECT_EQ(second_report.value().disposition, GlobalSealDisposition::PendingUnconnectedStored);
  EXPECT_EQ(coordinator.status().pending_epochs, 1U);
  EXPECT_EQ(coordinator.status().pending_seals, 2U);
  const auto retained = coordinator.pendingSeals(core::OdomEpoch(9U));
  ASSERT_EQ(retained.size(), 2U);
  EXPECT_EQ(retained.front().ref, pending_first.ref);
  EXPECT_EQ(retained.back().ref, pending_second.ref);
  EXPECT_FALSE(coordinator.mapOdom(core::OdomEpoch(9U)).has_value());
}

TEST(GlobalCoordinator, CanonicalAdmissionFailureKeepsTypedCoreEvidence) {
  GlobalCoordinator coordinator;
  core::SparseSubmapSeal corrupted = test_support::firstSeal();
  corrupted.boundary_navigation.velocity_odom.x() += 0.25;

  const auto rejected = coordinator.ingestSeal(corrupted);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GlobalCoordinatorErrorCode::InvalidSeal);
  ASSERT_TRUE(rejected.error().canonical_verification_error.has_value());
  EXPECT_EQ(rejected.error().canonical_verification_error->failure,
            core::CanonicalVerificationFailure::DigestMismatch);
  EXPECT_FALSE(coordinator.status().graph_initialized);
  EXPECT_EQ(coordinator.status().accepted_seals, 0U);
}

}  // namespace
}  // namespace meridian::global
