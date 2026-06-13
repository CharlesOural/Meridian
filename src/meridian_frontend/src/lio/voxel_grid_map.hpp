#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "meridian/config/config.hpp"

namespace meridian::lio {

// Integer grid cell of a point: floor(coord / edge) per axis. floor (not integer
// truncation) keeps negative coordinates in the cell below zero, so (-edge, 0) does
// not fold onto [0, edge).
struct VoxelCell {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;
  bool operator==(const VoxelCell&) const = default;
};

// Multiply-xor mix of the three indices. Every operand is cast to uint64 before any
// arithmetic, so wraparound is well-defined; cell identity is decided by key equality,
// never by the hash, so a mix collision only costs bucket sharing.
struct VoxelCellHash {
  std::size_t operator()(const VoxelCell& c) const {
    return static_cast<std::size_t>((static_cast<std::uint64_t>(c.x) * 73856093ULL) ^
                                    (static_cast<std::uint64_t>(c.y) * 19349669ULL) ^
                                    (static_cast<std::uint64_t>(c.z) * 83492791ULL));
  }
};

// Voxel-grid downsample keeping original coordinates: one point per occupied cell of
// edge `voxel` [m], the FIRST point of `pts` that lands in the cell. Survivors appear
// in input order, so the output is deterministic for a given input sequence.
std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& pts, double voxel);

// The local map: a hash grid of cfg.voxel_size_m cells over world-frame points, each
// cell holding at most cfg.max_points_per_voxel points. Nearest-neighbor queries probe
// the query's cell and its 26 face/edge/corner neighbors in a fixed order, so lookups
// never depend on hash-map iteration order.
//
// The cell index is an open-addressing (linear-probe) table rather than a node-based
// std::unordered_map: the 27-cell probe then walks a contiguous slot array with no
// per-bucket allocation or pointer chase on the miss path, which dominates the frontend
// cost. The stored points and their order are unchanged, so nearest()/neighborsWithin()
// return identical results to a node-based map.
class VoxelGridMap {
public:
  explicit VoxelGridMap(const FrontendLio& cfg);

  // Insert world-frame points, respecting the per-voxel cap and minimum spacing.
  void insert(const std::vector<Eigen::Vector3d>& points_world);

  // Closest map point to `query_world` within cfg.max_corr_dist_m, searched over the
  // 27-cell neighborhood. Returns false (neighbor untouched) when none qualifies.
  bool nearest(const Eigen::Vector3d& query_world, Eigen::Vector3d* neighbor) const;

  // Append every stored point within `radius` of `query_world` to `out` (cleared
  // first), scanning the 27-cell neighborhood in the same fixed probe order as
  // nearest(), so the output order is deterministic. The probe only reaches one cell
  // out, so radii beyond cfg.voxel_size_m are silently truncated to that envelope.
  void neighborsWithin(const Eigen::Vector3d& query_world, double radius,
                       std::vector<Eigen::Vector3d>* out) const;

  // Drop every voxel whose cell center is farther than cfg.max_range_m from
  // `center_world`.
  void clipFarFrom(const Eigen::Vector3d& center_world);

  void clear();

  // Total stored points / occupied voxels.
  std::size_t size() const;
  std::size_t voxel_count() const;

private:
  // One open-addressing slot. `used` distinguishes a live cell from an empty slot; the
  // points keep their insertion order, exactly as the previous per-voxel vector did.
  struct Slot {
    VoxelCell key;
    std::vector<Eigen::Vector3d> pts;
    bool used = false;
  };

  // Live points of `cell`, or nullptr when the cell is absent. The lookup is a linear
  // probe from the cell's hash slot until the key matches or an empty slot is hit.
  const std::vector<Eigen::Vector3d>* cellPoints(const VoxelCell& cell) const;
  // The slot a key occupies or should occupy (its insertion point on a miss).
  std::size_t probe(const VoxelCell& cell) const;
  // Grow and re-probe all live slots when the load factor crosses the threshold.
  void growIfNeeded();

  FrontendLio cfg_;
  std::vector<Slot> table_;  // size is a power of two; mask_ = size - 1
  std::size_t mask_ = 0;
  std::size_t used_ = 0;   // occupied slots (for the load factor)
  std::size_t size_ = 0;   // total stored points
};

}  // namespace meridian::lio
