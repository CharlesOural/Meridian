#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "meridian/core/navigation.hpp"

namespace meridian::core {
namespace {

TEST(StateId, IsASeparateOrderedStrongIdentity) {
  EXPECT_LT(StateId(4), StateId(5));
  EXPECT_EQ(StateId(4).value(), 4U);
}

TEST(ImuBias, StoresNamedFiniteBiasVectors) {
  const ImuBias bias(Vec3d{0.1, 0.2, 0.3}, Vec3d{-0.1, -0.2, -0.3});
  EXPECT_EQ(bias.gyroscopeRadS(), (Vec3d{0.1, 0.2, 0.3}));
  EXPECT_EQ(bias.accelerometerMS2(), (Vec3d{-0.1, -0.2, -0.3}));

  EXPECT_THROW(static_cast<void>(
                   ImuBias(Vec3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}, Vec3d{})),
               std::invalid_argument);
}

TEST(NavigationState, PreservesCanonicalFramesAndRejectsNonFiniteVelocity) {
  const Pose3d odom_from_imu(Vec3d{1.0, 2.0, 3.0}, Quaterniond());
  const ImuBias bias(Vec3d{0.01, 0.02, 0.03}, Vec3d{0.1, 0.2, 0.3});
  const NavigationState state(StateId(8), TimeNs(42), odom_from_imu, Vec3d{4.0, 5.0, 6.0}, bias);

  EXPECT_EQ(state.id(), StateId(8));
  EXPECT_EQ(state.time(), TimeNs(42));
  EXPECT_EQ(state.odomFromImu(), odom_from_imu);
  EXPECT_EQ(state.velocityOdomMS(), (Vec3d{4.0, 5.0, 6.0}));
  EXPECT_EQ(state.imuBias(), bias);

  EXPECT_THROW(static_cast<void>(NavigationState(
                   StateId(9), TimeNs(43), Pose3d(),
                   Vec3d{0.0, std::numeric_limits<double>::infinity(), 0.0}, ImuBias())),
               std::invalid_argument);
}

}  // namespace
}  // namespace meridian::core
