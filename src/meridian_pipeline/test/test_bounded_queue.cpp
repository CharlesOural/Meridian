#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "meridian/pipeline/bounded_queue.hpp"

using meridian::BoundedQueue;

TEST(BoundedQueue, FifoOrder) {
  BoundedQueue<int> q(4);
  EXPECT_TRUE(q.try_push(1));
  EXPECT_TRUE(q.try_push(2));
  int out = 0;
  EXPECT_TRUE(q.try_pop(out));
  EXPECT_EQ(out, 1);
  EXPECT_TRUE(q.try_pop(out));
  EXPECT_EQ(out, 2);
  EXPECT_FALSE(q.try_pop(out));
}

TEST(BoundedQueue, TryPushRejectsWhenFull) {
  BoundedQueue<int> q(2);
  EXPECT_TRUE(q.try_push(1));
  EXPECT_TRUE(q.try_push(2));
  EXPECT_FALSE(q.try_push(3));
  EXPECT_EQ(q.size(), 2u);
}

TEST(BoundedQueue, DropOldestKeepsFreshest) {
  BoundedQueue<int> q(2);
  EXPECT_EQ(q.push_or_drop_oldest(1), 0u);
  EXPECT_EQ(q.push_or_drop_oldest(2), 0u);
  EXPECT_EQ(q.push_or_drop_oldest(3), 1u);  // evicts 1
  int out = 0;
  EXPECT_TRUE(q.try_pop(out));
  EXPECT_EQ(out, 2);
  EXPECT_TRUE(q.try_pop(out));
  EXPECT_EQ(out, 3);
}

namespace {
struct Tagged {
  int value = 0;
  bool protect = false;
};
bool isProtected(const Tagged& t) { return t.protect; }
}  // namespace

TEST(BoundedQueue, ProtectingEvictsUnprotectedNotProtected) {
  BoundedQueue<Tagged> q(2);
  // A protected element followed by an unprotected one; the next push must evict the
  // unprotected element and leave the protected one in place.
  EXPECT_FALSE(q.push_protecting(Tagged{1, true}, isProtected).evicted.has_value());
  EXPECT_FALSE(q.push_protecting(Tagged{2, false}, isProtected).evicted.has_value());
  auto out = q.push_protecting(Tagged{3, false}, isProtected);
  ASSERT_TRUE(out.enqueued);
  ASSERT_TRUE(out.evicted.has_value());
  EXPECT_EQ(out.evicted->value, 2);
  EXPECT_FALSE(out.evicted->protect);

  Tagged v;
  EXPECT_TRUE(q.try_pop(v));
  EXPECT_EQ(v.value, 1);  // protected element survived
  EXPECT_TRUE(q.try_pop(v));
  EXPECT_EQ(v.value, 3);
}

TEST(BoundedQueue, ProtectingEvictsOldestUnprotectedWhenProtectedIsNewer) {
  BoundedQueue<Tagged> q(2);
  // Order: oldest unprotected, then protected. A full push must skip the protected
  // element and evict the older unprotected one even though it is at the front.
  q.push_protecting(Tagged{1, false}, isProtected);
  q.push_protecting(Tagged{2, true}, isProtected);
  auto out = q.push_protecting(Tagged{3, true}, isProtected);
  ASSERT_TRUE(out.evicted.has_value());
  EXPECT_EQ(out.evicted->value, 1);

  Tagged v;
  EXPECT_TRUE(q.try_pop(v));
  EXPECT_EQ(v.value, 2);
  EXPECT_TRUE(q.try_pop(v));
  EXPECT_EQ(v.value, 3);
}

TEST(BoundedQueue, ProtectingFallsBackToOldestWhenAllProtected) {
  BoundedQueue<Tagged> q(2);
  q.push_protecting(Tagged{1, true}, isProtected);
  q.push_protecting(Tagged{2, true}, isProtected);
  // No unprotected victim exists, so the queue must still make room by evicting the
  // oldest rather than refusing the push.
  auto out = q.push_protecting(Tagged{3, true}, isProtected);
  ASSERT_TRUE(out.enqueued);
  ASSERT_TRUE(out.evicted.has_value());
  EXPECT_EQ(out.evicted->value, 1);
}

TEST(BoundedQueue, ProtectingRejectsWhenClosed) {
  BoundedQueue<Tagged> q(2);
  q.close();
  auto out = q.push_protecting(Tagged{1, false}, isProtected);
  EXPECT_FALSE(out.enqueued);
  EXPECT_FALSE(out.evicted.has_value());
}

TEST(BoundedQueue, CloseWakesConsumerAndDrains) {
  BoundedQueue<int> q(4);
  q.try_push(7);
  std::vector<int> got;
  std::thread consumer([&] {
    int v = 0;
    while (q.pop(v)) got.push_back(v);
  });
  q.close();
  consumer.join();
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0], 7);
  EXPECT_FALSE(q.try_push(8));  // closed queues reject producers
}
