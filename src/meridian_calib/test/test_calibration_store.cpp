#include <memory>

#include <gtest/gtest.h>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/calib/icalibration_provider.hpp"

using meridian::CalibrationSet;
using meridian::CalibrationStore;
using meridian::Frame;

namespace {

std::shared_ptr<const CalibrationSet> make_set(double acc_noise, Frame est_frame) {
  auto s = std::make_shared<CalibrationSet>();
  s->estimation_frame = est_frame;
  s->imu_acc_noise = acc_noise;
  return s;
}

}  // namespace

TEST(CalibrationStore, SeedIsCurrentAtVersionZero) {
  auto seed = make_set(1.0, Frame::ImuLink);
  CalibrationStore store(seed);

  EXPECT_EQ(store.version(), 0u);
  EXPECT_EQ(store.current(), seed);
  EXPECT_EQ(store.current()->imu_acc_noise, 1.0);
}

TEST(CalibrationStore, PublishBumpsVersionAndSwapsSnapshot) {
  auto seed = make_set(1.0, Frame::ImuLink);
  CalibrationStore store(seed);

  auto refined = make_set(2.0, Frame::CamLink);
  store.publish(refined);

  EXPECT_EQ(store.version(), 1u);
  EXPECT_EQ(store.current(), refined);
  EXPECT_EQ(store.current()->imu_acc_noise, 2.0);
  EXPECT_EQ(store.current()->estimation_frame, Frame::CamLink);
}

TEST(CalibrationStore, ReaderHeldSnapshotIsUnchangedAfterPublish) {
  auto seed = make_set(1.0, Frame::ImuLink);
  CalibrationStore store(seed);

  // A reader grabs the current snapshot before any refinement.
  std::shared_ptr<const CalibrationSet> held = store.current();

  store.publish(make_set(2.0, Frame::CamLink));

  // The pointer the reader holds still observes the original immutable set.
  EXPECT_EQ(held, seed);
  EXPECT_EQ(held->imu_acc_noise, 1.0);
  EXPECT_EQ(held->estimation_frame, Frame::ImuLink);
  EXPECT_NE(held, store.current());
}

TEST(CalibrationStore, RepeatedPublishIncrementsMonotonically) {
  CalibrationStore store(make_set(0.0, Frame::ImuLink));

  store.publish(make_set(1.0, Frame::ImuLink));
  store.publish(make_set(2.0, Frame::ImuLink));
  store.publish(make_set(3.0, Frame::ImuLink));

  EXPECT_EQ(store.version(), 3u);
  EXPECT_EQ(store.current()->imu_acc_noise, 3.0);
}
