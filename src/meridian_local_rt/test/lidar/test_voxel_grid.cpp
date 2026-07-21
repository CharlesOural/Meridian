#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "meridian/local_rt/lidar/voxel_grid.hpp"

namespace meridian::local_rt::lidar {
namespace {

TEST(VoxelGridTest, RejectsInvalidConfigurationAndNonfiniteKeyInput) {
  EXPECT_THROW(deterministicVoxelDownsample({}, 0.0), std::invalid_argument);
  EXPECT_THROW(deterministicVoxelDownsample({}, -1.0), std::invalid_argument);
  EXPECT_THROW(deterministicVoxelDownsample({}, std::numeric_limits<double>::infinity()),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(pointToVoxelKey(
                   Point3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}, 1.0)),
               std::invalid_argument);
}

TEST(VoxelGridTest, UsesFloorIndexingAcrossNegativeCoordinates) {
  EXPECT_EQ(pointToVoxelKey(Point3d{-0.01, 0.99, -1.0}, 1.0), (VoxelKey{.x = -1, .y = 0, .z = -1}));
  EXPECT_EQ(pointToVoxelKey(Point3d{1.0, -1.01, 2.99}, 1.0), (VoxelKey{.x = 1, .y = -2, .z = 2}));
}

TEST(VoxelGridTest, SelectionAndOutputArePermutationIndependent) {
  PointCloud input{
      Point3d{0.1, 0.5, 0.5},  Point3d{0.6, 0.5, 0.5},
      Point3d{-0.1, 0.0, 0.0}, Point3d{0.4, 0.5, 0.5},
      Point3d{1.2, 0.0, 0.0},  Point3d{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
  };
  const auto forward = deterministicVoxelDownsample(input, 1.0);
  std::reverse(input.begin(), input.end());
  const auto reversed = deterministicVoxelDownsample(input, 1.0);

  ASSERT_EQ(forward.points.size(), 3U);
  ASSERT_EQ(reversed.points.size(), forward.points.size());
  for (std::size_t index = 0U; index < forward.points.size(); ++index) {
    EXPECT_TRUE(forward.points[index].isApprox(reversed.points[index], 0.0));
  }

  // Voxel keys are ordered (-1,0,0), (0,0,0), (1,0,0). The two points
  // equidistant from the first positive voxel centre select the lexicographically
  // smaller observed coordinate.
  EXPECT_TRUE(forward.points[0].isApprox(Point3d{-0.1, 0.0, 0.0}, 0.0));
  EXPECT_TRUE(forward.points[1].isApprox(Point3d{0.4, 0.5, 0.5}, 0.0));
  EXPECT_TRUE(forward.points[2].isApprox(Point3d{1.2, 0.0, 0.0}, 0.0));
  EXPECT_EQ(forward.stats.input_points, 6U);
  EXPECT_EQ(forward.stats.finite_points, 5U);
  EXPECT_EQ(forward.stats.rejected_nonfinite_points, 1U);
  EXPECT_EQ(forward.stats.output_points, 3U);
}

}  // namespace
}  // namespace meridian::local_rt::lidar
