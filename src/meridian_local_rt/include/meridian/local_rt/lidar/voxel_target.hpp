#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sophus/se3.hpp>
#include <span>
#include <unordered_map>
#include <vector>

#include "meridian/local_rt/lidar/voxel_grid.hpp"

namespace meridian::local_rt::lidar {

struct VoxelTargetOptions final {
  double voxel_size_m{};
  double retention_radius_m{};
  std::size_t max_voxels{};
  std::size_t max_points_per_voxel{};
  double minimum_point_spacing_m{};
  std::size_t max_neighbor_voxel_radius{};
};

struct VoxelTargetUpdateStats final {
  std::size_t input_points{};
  std::size_t finite_points{};
  std::size_t inserted_points{};
  std::size_t rejected_nonfinite_points{};
  std::size_t rejected_retention_points{};
  std::size_t rejected_spacing_points{};
  std::size_t rejected_voxel_capacity_points{};
  std::size_t rejected_target_capacity_points{};
  std::size_t pruned_voxels{};
  std::size_t pruned_points{};
  std::size_t evicted_voxels{};
  std::size_t evicted_points{};
};

struct NearestTargetPoint final {
  Point3d point{Point3d::Zero()};
  double squared_distance_m2{};
};

// Optional per-query work counters. A voxel hash probe is counted only when a
// cell survives branch-and-bound and reaches the sparse-map lookup. Point
// candidates count stored points whose exact distance is evaluated.
struct VoxelTargetQueryStats final {
  std::size_t voxel_hash_probes{};
  std::size_t occupied_buckets{};
  std::size_t point_candidates{};
};

// A memory-bounded local target. Each update is deterministic for the same
// target state and point set: transformed candidates are ordered before
// admission, voxel capacity is fixed, and spatial eviction has explicit ties.
class BoundedVoxelTarget final {
public:
  explicit BoundedVoxelTarget(VoxelTargetOptions options);

  [[nodiscard]] const VoxelTargetOptions& options() const noexcept { return options_; }
  [[nodiscard]] bool empty() const noexcept { return voxels_.empty(); }
  [[nodiscard]] std::size_t voxelCount() const noexcept { return voxels_.size(); }
  [[nodiscard]] std::size_t pointCount() const noexcept { return point_count_; }

  void clear() noexcept;

  // Points are transformed into the target frame before admission. The
  // retention centre is expressed in that same target frame.
  [[nodiscard]] VoxelTargetUpdateStats update(std::span<const Point3d> points_source,
                                              const Sophus::SE3d& T_target_source,
                                              const Point3d& retention_center_target);

  [[nodiscard]] VoxelTargetUpdateStats updateTargetFrame(std::span<const Point3d> points_target,
                                                         const Point3d& retention_center_target);

  [[nodiscard]] std::optional<NearestTargetPoint> nearestNeighbor(
      const Point3d& query_target, double max_distance_m,
      VoxelTargetQueryStats* query_stats = nullptr) const;

  [[nodiscard]] bool supportsQueryDistance(double max_distance_m) const noexcept;
  [[nodiscard]] PointCloud pointCloud() const;

private:
  struct NeighborOffset final {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    // Query-independent lower bound from any point in the centre voxel to any
    // point in this voxel, expressed in squared voxel units. The stencil is
    // ordered by this value for allocation-free branch-and-bound traversal.
    double minimum_squared_distance_voxel_units{};
  };

  using VoxelMap = std::unordered_map<VoxelKey, PointCloud, VoxelKeyHash>;

  void prune(const Point3d& retention_center_target, VoxelTargetUpdateStats& stats);
  [[nodiscard]] bool makeRoomFor(const VoxelKey& candidate_key, const Point3d& candidate_point,
                                 const Point3d& retention_center_target,
                                 VoxelTargetUpdateStats& stats);

  VoxelTargetOptions options_;
  // Constructed once with the bounded target. Queries reuse this center-first
  // stencil and reject cells whose AABB cannot beat the incumbent or gate.
  std::vector<NeighborOffset> neighbor_offsets_;
  VoxelMap voxels_;
  std::size_t point_count_{};
};

}  // namespace meridian::local_rt::lidar
