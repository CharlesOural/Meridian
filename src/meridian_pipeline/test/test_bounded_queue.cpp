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
