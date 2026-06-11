#include "lio/voxel_grid_map.hpp"

namespace meridian::lio {

std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& pts,
                                             double voxel) {
  // TODO(lio): implemented in a later lane
  (void)voxel;
  return pts;
}

VoxelGridMap::VoxelGridMap(const FrontendLio& cfg) : cfg_(cfg) {}

void VoxelGridMap::insert(const std::vector<Eigen::Vector3d>& points_world) {
  // TODO(lio): implemented in a later lane
  (void)points_world;
}

bool VoxelGridMap::nearest(const Eigen::Vector3d& query_world, Eigen::Vector3d* neighbor) const {
  // TODO(lio): implemented in a later lane
  (void)query_world;
  (void)neighbor;
  return false;
}

void VoxelGridMap::clipFarFrom(const Eigen::Vector3d& center_world) {
  // TODO(lio): implemented in a later lane
  (void)center_world;
}

void VoxelGridMap::clear() { voxels_.clear(); }

std::size_t VoxelGridMap::size() const {
  // TODO(lio): implemented in a later lane
  return 0;
}

std::size_t VoxelGridMap::voxel_count() const { return voxels_.size(); }

}  // namespace meridian::lio
