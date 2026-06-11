#include "lio/voxel_grid_map.hpp"

#include <gtest/gtest.h>

#include "meridian/config/config.hpp"

TEST(VoxelGridMap, ConstructsEmpty) {
  meridian::lio::VoxelGridMap map{meridian::FrontendLio{}};
  EXPECT_EQ(map.size(), 0u);
  EXPECT_EQ(map.voxel_count(), 0u);
}
