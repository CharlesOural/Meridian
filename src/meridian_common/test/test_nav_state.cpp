#include <Eigen/Core>
#include <gtest/gtest.h>

#include "meridian/common/nav_state.hpp"

using meridian::NavState;

TEST(NavState, Dof18) { EXPECT_EQ(NavState::kDof, 18); }

TEST(NavState, GravityDefault) {
  const NavState s;
  EXPECT_NEAR(s.g_world.z(), -9.81, 1e-12);
  EXPECT_EQ(s.ref_frame, meridian::Frame::Odom);
}

// 18-DoF error state ordered [ p | R | v | b_g | b_a | g ]; round-trip the box ops.
TEST(NavState, BoxplusBoxminusRoundTrip) {
  NavState s;
  s.v_world = Eigen::Vector3d(1, 2, 3);
  s.b_g = Eigen::Vector3d(0.01, 0.0, -0.02);
  Eigen::Matrix<double, 18, 1> dx;
  for (int i = 0; i < 18; ++i) dx[i] = 0.01 * (i + 1) * ((i % 2) ? -1.0 : 1.0);
  const NavState t = s.boxplus(dx);
  EXPECT_TRUE(t.boxminus(s).isApprox(dx, 1e-9));
}

// The velocity block is dims 6..8: perturbing only it adds straight to v_world.
TEST(NavState, VelocityBlockIsAdditive) {
  NavState s;
  Eigen::Matrix<double, 18, 1> dx = Eigen::Matrix<double, 18, 1>::Zero();
  dx.segment<3>(6) = Eigen::Vector3d(0.5, -0.25, 0.1);
  const NavState t = s.boxplus(dx);
  EXPECT_TRUE(t.v_world.isApprox(s.v_world + dx.segment<3>(6), 1e-12));
}
