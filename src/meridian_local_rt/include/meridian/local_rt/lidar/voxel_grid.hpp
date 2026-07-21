#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace meridian::local_rt::lidar {

using Point3d = Eigen::Vector3d;
using PointCloud = std::vector<Point3d, Eigen::aligned_allocator<Point3d>>;

struct VoxelKey final {
  std::int64_t x{};
  std::int64_t y{};
  std::int64_t z{};

  friend bool operator==(const VoxelKey&, const VoxelKey&) noexcept = default;
  friend auto operator<=>(const VoxelKey&, const VoxelKey&) noexcept = default;
};

struct VoxelKeyHash final {
  [[nodiscard]] std::size_t operator()(const VoxelKey& key) const noexcept;
};

struct VoxelDownsampleStats final {
  std::size_t input_points{};
  std::size_t finite_points{};
  std::size_t rejected_nonfinite_points{};
  std::size_t output_points{};
};

struct VoxelDownsampleResult final {
  PointCloud points;
  VoxelDownsampleStats stats;
};

// Converts a finite point to a floor-indexed voxel. Invalid voxel sizes and
// coordinates outside the signed 64-bit voxel domain are reported by exception.
[[nodiscard]] VoxelKey pointToVoxelKey(const Point3d& point, double voxel_size);

// Produces exactly one observed point per occupied voxel. The representative is
// the point nearest the voxel centre, with lexicographic coordinate tie-breaking.
// Output is lexicographically ordered by voxel key and is independent of hash-map
// iteration order and input permutation (apart from indistinguishable duplicates).
[[nodiscard]] VoxelDownsampleResult deterministicVoxelDownsample(std::span<const Point3d> points,
                                                                 double voxel_size);

}  // namespace meridian::local_rt::lidar
