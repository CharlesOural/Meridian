#include <gtest/gtest.h>

#include "meridian/debug/multi_sink.hpp"
#include "meridian/debug/recording_sink.hpp"

using meridian::Level;
using meridian::MultiSink;
using meridian::NullSink;
using meridian::RecordingSink;

TEST(RecordingSink, CapturesAllChannels) {
  RecordingSink rec;
  rec.scalar("a/b", 1.5, 10);
  Eigen::VectorXd v(2);
  v << 1, 2;
  rec.vec("a/v", v, 11, "tx,ty");
  rec.timing("stage", 3.0, 12);
  rec.event(Level::Warn, "tag", "msg", 13);

  ASSERT_EQ(rec.scalars.size(), 1u);
  EXPECT_EQ(rec.scalars[0].key, "a/b");
  EXPECT_DOUBLE_EQ(rec.scalars[0].v, 1.5);
  ASSERT_EQ(rec.vecs.size(), 1u);
  EXPECT_EQ(rec.vecs[0].axis_order, "tx,ty");
  ASSERT_EQ(rec.timings.size(), 1u);
  EXPECT_EQ(rec.timings[0].stage, "stage");
  ASSERT_EQ(rec.events.size(), 1u);
  EXPECT_EQ(rec.events[0].tag, "tag");
  EXPECT_TRUE(rec.enabled("anything"));
}

TEST(RecordingSink, BoundedCapacityDropsOldest) {
  RecordingSink rec(2);
  rec.scalar("k", 1, 1);
  rec.scalar("k", 2, 2);
  rec.scalar("k", 3, 3);
  ASSERT_EQ(rec.scalars.size(), 2u);
  EXPECT_DOUBLE_EQ(rec.scalars[0].v, 2);
  EXPECT_DOUBLE_EQ(rec.scalars[1].v, 3);
}

TEST(MultiSink, FansOutAndOrsEnabled) {
  RecordingSink a;
  RecordingSink b;
  NullSink null;
  MultiSink multi;
  multi.add(&null);
  multi.add(&a);
  multi.add(&b);

  EXPECT_TRUE(multi.enabled("x"));  // RecordingSink children accept everything
  multi.scalar("x", 7.0, 1);
  EXPECT_EQ(a.scalars.size(), 1u);
  EXPECT_EQ(b.scalars.size(), 1u);

  MultiSink only_null;
  only_null.add(&null);
  EXPECT_FALSE(only_null.enabled("x"));
}
