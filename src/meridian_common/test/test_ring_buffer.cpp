#include <gtest/gtest.h>

#include "meridian/common/ring_buffer.hpp"

using meridian::RingBuffer;

TEST(RingBuffer, FillsThenOverwritesOldest) {
  RingBuffer<int> rb(3);
  EXPECT_TRUE(rb.empty());
  rb.push(1);
  rb.push(2);
  rb.push(3);
  EXPECT_TRUE(rb.full());
  EXPECT_EQ(rb.size(), 3u);
  EXPECT_EQ(rb.front(), 1);
  EXPECT_EQ(rb.back(), 3);

  rb.push(4);  // overwrites the oldest (1)
  EXPECT_EQ(rb.size(), 3u);
  EXPECT_EQ(rb.front(), 2);
  EXPECT_EQ(rb.back(), 4);
  EXPECT_EQ(rb[0], 2);
  EXPECT_EQ(rb[1], 3);
  EXPECT_EQ(rb[2], 4);
}

TEST(RingBuffer, Clear) {
  RingBuffer<int> rb(2);
  rb.push(5);
  rb.clear();
  EXPECT_TRUE(rb.empty());
  EXPECT_EQ(rb.size(), 0u);
}

TEST(RingBuffer, ZeroCapacityClampedToOne) {
  RingBuffer<int> rb(0);
  EXPECT_EQ(rb.capacity(), 1u);
  rb.push(7);
  rb.push(8);
  EXPECT_EQ(rb.back(), 8);
  EXPECT_EQ(rb.size(), 1u);
}
