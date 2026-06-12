#include <gtest/gtest.h>

#include <Eigen/Core>
#include <random>

#include "meridian/common/cov_reorder.hpp"

using meridian::reorderTransRotToRotTrans;

// The covariance reorder helper round-trips translation-first <-> rotation-first on
// a random SPD matrix, and swaps the diagonal 3-blocks while transposing the
// off-diagonal coupling.
TEST(CovReorder, RoundTrips) {
  std::mt19937 rng(123);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::Matrix<double, 6, 6> A;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      A(i, j) = nd(rng);
    }
  }
  const Eigen::Matrix<double, 6, 6> spd =
      A * A.transpose() + Eigen::Matrix<double, 6, 6>::Identity();

  const Eigen::Matrix<double, 6, 6> rot_first = reorderTransRotToRotTrans(spd);
  const Eigen::Matrix<double, 6, 6> back = reorderTransRotToRotTrans(rot_first);
  EXPECT_TRUE(back.isApprox(spd, 1e-12));

  // The reorder swaps the diagonal 3-blocks: rot_first's top-left equals spd's
  // rotation block (rows/cols 3..5), and rot_first's bottom-right equals spd's
  // translation block (rows/cols 0..2).
  EXPECT_TRUE((rot_first.block<3, 3>(0, 0).isApprox(spd.block<3, 3>(3, 3), 1e-12)));
  EXPECT_TRUE((rot_first.block<3, 3>(3, 3).isApprox(spd.block<3, 3>(0, 0), 1e-12)));
  // And the off-diagonal coupling is transposed across the swap.
  EXPECT_TRUE((rot_first.block<3, 3>(0, 3).isApprox(spd.block<3, 3>(3, 0), 1e-12)));
}
