#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "meridian/config/config.hpp"

namespace meridian::lio {

// Voxel-grid downsample keeping original coordinates: one point per occupied cell of
// edge `voxel` [m], the FIRST point of `pts` that lands in the cell. Input order alone
// decides the survivor, so the output is deterministic for a given input sequence.
std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& pts,
                                             double voxel);

// The local map: a hash grid of cfg.voxel_size_m cells over world-frame points, each
// cell holding at most cfg.max_points_per_voxel points. Nearest-neighbor queries probe
// the query's cell and its 26 face/edge/corner neighbors in a fixed order, so lookups
// never depend on hash-map iteration order.
class VoxelGridMap {
public:
  explicit VoxelGridMap(const FrontendLio& cfg);

  // Insert world-frame points, respecting the per-voxel cap and minimum spacing.
  void insert(const std::vector<Eigen::Vector3d>& points_world);

  // Closest map point to `query_world` within cfg.max_corr_dist_m, searched over the
  // 27-cell neighborhood. Returns false (neighbor untouched) when none qualifies.
  bool nearest(const Eigen::Vector3d& query_world, Eigen::Vector3d* neighbor) const;

  // Drop every voxel farther than cfg.max_range_m from `center_world`.
  void clipFarFrom(const Eigen::Vector3d& center_world);

  void clear();

  // Total stored points / occupied voxels.
  std::size_t size() const;
  std::size_t voxel_count() const;

private:
  FrontendLio cfg_;
  std::unordered_map<std::uint64_t, std::vector<Eigen::Vector3d>> voxels_;
};

}  // namespace meridian::lio
