#include "meridian/preprocess/imu_init.hpp"

#include <gtest/gtest.h>

#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::ImuInitializer;
using meridian::ImuSample;
using meridian::kGravityMagnitude;
using meridian::PreprocImu;

namespace {

ImuSample mk(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, std::int64_t t) {
  ImuSample s;
  s.acc = acc;
  s.gyro = gyro;
  s.stamp = t;
  return s;
}

}  // namespace

TEST(ImuInitializer, RecoversGravityDirectionAndGyroBias) {
  PreprocImu cfg;
  cfg.init_max_var = 0.5;
  cfg.init_max_grav_err = 0.5;
  ImuInitializer init(cfg, 10);

  // Stationary: specific force is +G along z (reaction to gravity); constant gyro bias.
  const Eigen::Vector3d acc(0.0, 0.0, kGravityMagnitude);
  const Eigen::Vector3d bias(0.01, -0.02, 0.03);

  bool done = false;
  for (int i = 0; i < 10; ++i) {
    done = init.add(mk(acc, bias, i * 5'000'000));
  }
  ASSERT_TRUE(done);
  ASSERT_TRUE(init.done());
  EXPECT_FALSE(init.failed());

  // Gravity points opposite the measured specific force, magnitude pinned to G.
  EXPECT_NEAR(init.state().gravity.z(), -kGravityMagnitude, 1e-6);
  EXPECT_NEAR(init.state().gravity.norm(), kGravityMagnitude, 1e-6);

  EXPECT_NEAR(init.state().gyro_bias.x(), bias.x(), 1e-9);
  EXPECT_NEAR(init.state().gyro_bias.y(), bias.y(), 1e-9);
  EXPECT_NEAR(init.state().gyro_bias.z(), bias.z(), 1e-9);

  // Accel bias is deliberately NOT initialized.
  EXPECT_EQ(init.state().accel_bias, Eigen::Vector3d::Zero());
}

TEST(ImuInitializer, RejectsMotionDuringInit) {
  PreprocImu cfg;
  cfg.init_max_var = 0.01;  // tight: any wobble fails
  cfg.init_max_grav_err = 0.5;
  ImuInitializer init(cfg, 10);

  const Eigen::Vector3d base(0.0, 0.0, kGravityMagnitude);
  for (int i = 0; i < 10; ++i) {
    // Inject alternating perturbations so the static variance exceeds the gate.
    const double w = (i % 2 == 0) ? 0.5 : -0.5;
    init.add(mk(base + Eigen::Vector3d(w, 0, 0), Eigen::Vector3d::Zero(), i * 5'000'000));
  }
  EXPECT_FALSE(init.done());
  EXPECT_TRUE(init.failed());

  // After clear(), a clean static window initializes.
  init.clear();
  bool done = false;
  for (int i = 0; i < 10; ++i) {
    done = init.add(mk(base, Eigen::Vector3d::Zero(), i * 5'000'000));
  }
  EXPECT_TRUE(done);
}

TEST(ImuInitializer, RejectsWhenGravityMagnitudeWrong) {
  PreprocImu cfg;
  cfg.init_max_var = 1.0;
  cfg.init_max_grav_err = 0.5;
  ImuInitializer init(cfg, 5);

  // Specific-force magnitude far from G (e.g. free-fall-ish) -> grav gate fails.
  const Eigen::Vector3d acc(0.0, 0.0, 5.0);
  for (int i = 0; i < 5; ++i) {
    init.add(mk(acc, Eigen::Vector3d::Zero(), i * 5'000'000));
  }
  EXPECT_FALSE(init.done());
  EXPECT_TRUE(init.failed());
}
