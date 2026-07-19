#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "meridian/local/factor_allocator.hpp"

namespace meridian::local {
namespace {

TEST(LocalFactorAllocator, AbortedCandidateIdsRemainInvisibleAndReusable) {
  LocalFactorAllocator allocator{LocalFactorAllocatorConfig{8U}};
  const core::OdomEpoch epoch{7U};

  auto prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto transaction = std::move(prepared).value();
  const auto candidate = transaction.allocate(LocalFactorKind::CombinedImu);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate.value().ref, (core::LocalFactorRef{epoch, core::FactorId{0U}}));
  EXPECT_TRUE(allocator.checkpoint().epochs.empty());

  const auto aborted = transaction.abort();
  ASSERT_TRUE(aborted);
  EXPECT_EQ(aborted.value().discarded_allocations, 1U);
  EXPECT_TRUE(allocator.checkpoint().epochs.empty());

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto retried = std::move(prepared).value();
  const auto reused_private_id = retried.allocate(LocalFactorKind::DirectLidar);
  ASSERT_TRUE(reused_private_id);
  EXPECT_EQ(reused_private_id.value().ref, (core::LocalFactorRef{epoch, core::FactorId{0U}}));
  ASSERT_TRUE(retried.commit());

  {
    auto destructor_prepared = allocator.prepare(epoch);
    ASSERT_TRUE(destructor_prepared);
    auto destructor_aborted = std::move(destructor_prepared).value();
    const auto private_id = destructor_aborted.allocate(LocalFactorKind::VisualReprojection);
    ASSERT_TRUE(private_id);
    EXPECT_EQ(private_id.value().ref.factor, core::FactorId{1U});
  }

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto after_destructor_abort = std::move(prepared).value();
  const auto id_after_abort = after_destructor_abort.allocate(LocalFactorKind::CombinedImu);
  ASSERT_TRUE(id_after_abort);
  EXPECT_EQ(id_after_abort.value().ref.factor, core::FactorId{1U});
  ASSERT_TRUE(after_destructor_abort.commit());
}

TEST(LocalFactorAllocator, TransactionIsMoveOnlyAndCanBeConsumedOnlyOnce) {
  static_assert(!std::is_copy_constructible_v<LocalFactorAllocationTransaction>);
  static_assert(!std::is_copy_assignable_v<LocalFactorAllocationTransaction>);
  static_assert(std::is_nothrow_move_constructible_v<LocalFactorAllocationTransaction>);
  static_assert(std::is_nothrow_move_assignable_v<LocalFactorAllocationTransaction>);

  LocalFactorAllocator allocator{LocalFactorAllocatorConfig{8U}};
  auto prepared = allocator.prepare(core::OdomEpoch{3U});
  ASSERT_TRUE(prepared);
  auto original = std::move(prepared).value();
  LocalFactorAllocationTransaction transaction{std::move(original)};
  EXPECT_TRUE(original.consumed());

  const auto moved_from_use = original.allocate(LocalFactorKind::CombinedImu);
  ASSERT_FALSE(moved_from_use);
  EXPECT_EQ(moved_from_use.error().code, LocalFactorAllocationErrorCode::TransactionConsumed);

  const std::vector<LocalFactorKind> kinds{
      LocalFactorKind::JointNavigationPrior, LocalFactorKind::FrozenMarginalPrior,
      LocalFactorKind::CombinedImu, LocalFactorKind::DirectLidar,
      LocalFactorKind::VisualReprojection};
  for (std::uint64_t index = 0U; index < kinds.size(); ++index) {
    const auto allocation = transaction.allocate(kinds[index]);
    ASSERT_TRUE(allocation);
    EXPECT_EQ(allocation.value().ref.factor, core::FactorId{index});
    EXPECT_EQ(allocation.value().kind, kinds[index]);
  }

  const auto committed = transaction.commit();
  ASSERT_TRUE(committed);
  ASSERT_EQ(committed.value().allocations.size(), kinds.size());
  EXPECT_EQ(committed.value().next_factor_id, kinds.size());
  EXPECT_TRUE(transaction.consumed());

  const auto second_commit = transaction.commit();
  ASSERT_FALSE(second_commit);
  EXPECT_EQ(second_commit.error().code, LocalFactorAllocationErrorCode::TransactionConsumed);
  const auto commit_then_abort = transaction.abort();
  ASSERT_FALSE(commit_then_abort);
  EXPECT_EQ(commit_then_abort.error().code, LocalFactorAllocationErrorCode::TransactionConsumed);
}

TEST(LocalFactorAllocator, ReplacementAlwaysReceivesANewCommittedIdentity) {
  LocalFactorAllocator allocator{LocalFactorAllocatorConfig{8U}};
  const core::OdomEpoch epoch{4U};

  auto prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto initial_transaction = std::move(prepared).value();
  const auto initial = initial_transaction.allocate(LocalFactorKind::DirectLidar);
  ASSERT_TRUE(initial);
  ASSERT_TRUE(initial_transaction.commit());

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto replacement_transaction = std::move(prepared).value();
  const auto replacement = replacement_transaction.allocateReplacement(
      initial.value().ref, LocalFactorKind::DirectLidar);
  ASSERT_TRUE(replacement);
  EXPECT_NE(replacement.value().ref, initial.value().ref);
  EXPECT_EQ(replacement.value().ref.factor, core::FactorId{1U});
  EXPECT_EQ(replacement.value().replaces, initial.value().ref);

  const auto replace_uncommitted = replacement_transaction.allocateReplacement(
      replacement.value().ref, LocalFactorKind::DirectLidar);
  ASSERT_FALSE(replace_uncommitted);
  EXPECT_EQ(replace_uncommitted.error().code,
            LocalFactorAllocationErrorCode::ReplacementNotCommitted);

  const auto committed = replacement_transaction.commit();
  ASSERT_TRUE(committed);
  ASSERT_EQ(committed.value().allocations.size(), 1U);
  EXPECT_EQ(committed.value().allocations.front().replaces, initial.value().ref);

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto following = std::move(prepared).value();
  const auto next = following.allocate(LocalFactorKind::VisualReprojection);
  ASSERT_TRUE(next);
  EXPECT_EQ(next.value().ref.factor, core::FactorId{2U});
  ASSERT_TRUE(following.commit());
}

TEST(LocalFactorAllocator, CheckpointOrderIsCanonicalRegardlessOfEpochArrivalOrder) {
  LocalFactorAllocator forward{LocalFactorAllocatorConfig{8U}};
  LocalFactorAllocator delayed{LocalFactorAllocatorConfig{8U}};

  const auto allocate_one = [](LocalFactorAllocator& allocator, core::OdomEpoch epoch) {
    auto prepared = allocator.prepare(epoch);
    if (!prepared) {
      return false;
    }
    auto transaction = std::move(prepared).value();
    if (!transaction.allocate(LocalFactorKind::CombinedImu)) {
      return false;
    }
    return static_cast<bool>(transaction.commit());
  };

  ASSERT_TRUE(allocate_one(forward, core::OdomEpoch{2U}));
  ASSERT_TRUE(allocate_one(forward, core::OdomEpoch{5U}));
  ASSERT_TRUE(allocate_one(forward, core::OdomEpoch{9U}));
  ASSERT_TRUE(allocate_one(delayed, core::OdomEpoch{9U}));
  ASSERT_TRUE(allocate_one(delayed, core::OdomEpoch{2U}));
  ASSERT_TRUE(allocate_one(delayed, core::OdomEpoch{5U}));

  const LocalFactorAllocatorCheckpoint canonical = forward.checkpoint();
  EXPECT_EQ(canonical, delayed.checkpoint());
  ASSERT_EQ(canonical.epochs.size(), 3U);
  EXPECT_EQ(canonical.epochs[0U].odom_epoch, core::OdomEpoch{2U});
  EXPECT_EQ(canonical.epochs[1U].odom_epoch, core::OdomEpoch{5U});
  EXPECT_EQ(canonical.epochs[2U].odom_epoch, core::OdomEpoch{9U});
}

TEST(LocalFactorAllocator, CapacityFailureRollsBackWithTheCandidate) {
  LocalFactorAllocator allocator{LocalFactorAllocatorConfig{2U}};
  const core::OdomEpoch epoch{12U};

  auto prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto transaction = std::move(prepared).value();
  ASSERT_TRUE(transaction.allocate(LocalFactorKind::JointNavigationPrior));
  ASSERT_TRUE(transaction.allocate(LocalFactorKind::CombinedImu));
  const auto exhausted = transaction.allocate(LocalFactorKind::DirectLidar);
  ASSERT_FALSE(exhausted);
  EXPECT_EQ(exhausted.error().code, LocalFactorAllocationErrorCode::CapacityExceeded);
  EXPECT_TRUE(allocator.checkpoint().epochs.empty());
  ASSERT_TRUE(transaction.abort());

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto retried = std::move(prepared).value();
  const auto first = retried.allocate(LocalFactorKind::CombinedImu);
  const auto second = retried.allocate(LocalFactorKind::VisualReprojection);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value().ref.factor, core::FactorId{0U});
  EXPECT_EQ(second.value().ref.factor, core::FactorId{1U});
  ASSERT_TRUE(retried.commit());

  prepared = allocator.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto full = std::move(prepared).value();
  const auto no_more = full.allocate(LocalFactorKind::CombinedImu);
  ASSERT_FALSE(no_more);
  EXPECT_EQ(no_more.error().code, LocalFactorAllocationErrorCode::CapacityExceeded);
  ASSERT_TRUE(full.abort());
}

TEST(LocalFactorAllocator, RestoreAdvancesHighWaterAndNeverReusesCommittedIds) {
  const core::OdomEpoch epoch{15U};
  LocalFactorAllocator original{LocalFactorAllocatorConfig{10U}};
  auto prepared = original.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto transaction = std::move(prepared).value();
  ASSERT_TRUE(transaction.allocate(LocalFactorKind::JointNavigationPrior));
  ASSERT_TRUE(transaction.allocate(LocalFactorKind::CombinedImu));
  ASSERT_TRUE(transaction.allocate(LocalFactorKind::DirectLidar));
  ASSERT_TRUE(transaction.commit());
  const LocalFactorAllocatorCheckpoint checkpoint = original.checkpoint();

  LocalFactorAllocator restored{LocalFactorAllocatorConfig{10U}};
  const auto restore = restored.restoreCheckpoint(checkpoint);
  ASSERT_TRUE(restore);
  EXPECT_EQ(restore.value().epochs_advanced, 1U);
  EXPECT_EQ(restore.value().epochs_retained, 0U);

  prepared = restored.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto after_restore = std::move(prepared).value();
  const auto next = after_restore.allocate(LocalFactorKind::VisualReprojection);
  ASSERT_TRUE(next);
  EXPECT_EQ(next.value().ref.factor, core::FactorId{3U});
  ASSERT_TRUE(after_restore.commit());

  const auto stale_restore = restored.restoreCheckpoint(checkpoint);
  ASSERT_TRUE(stale_restore);
  EXPECT_EQ(stale_restore.value().epochs_advanced, 0U);
  EXPECT_EQ(stale_restore.value().epochs_retained, 1U);

  prepared = restored.prepare(epoch);
  ASSERT_TRUE(prepared);
  auto after_stale_restore = std::move(prepared).value();
  const auto still_monotonic = after_stale_restore.allocate(LocalFactorKind::CombinedImu);
  ASSERT_TRUE(still_monotonic);
  EXPECT_EQ(still_monotonic.value().ref.factor, core::FactorId{4U});
  ASSERT_TRUE(after_stale_restore.commit());

  LocalFactorAllocatorCheckpoint noncanonical = checkpoint;
  noncanonical.epochs.insert(noncanonical.epochs.begin(),
                             FactorAllocatorEpochCheckpoint{core::OdomEpoch{20U}, 1U});
  const LocalFactorAllocatorCheckpoint before_rejection = restored.checkpoint();
  const auto rejected = restored.restoreCheckpoint(noncanonical);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalFactorAllocationErrorCode::NonCanonicalCheckpoint);
  EXPECT_EQ(restored.checkpoint(), before_rejection);
}

}  // namespace
}  // namespace meridian::local
