#include "lio/voxel_grid_map.hpp"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace meridian::lio {

namespace {

VoxelCell cellOf(const Eigen::Vector3d& p, const double voxel) {
  return VoxelCell{static_cast<std::int64_t>(std::floor(p.x() / voxel)),
                   static_cast<std::int64_t>(std::floor(p.y() / voxel)),
                   static_cast<std::int64_t>(std::floor(p.z() / voxel))};
}

}  // namespace

std::vector<Eigen::Vector3d> voxelDownsample(const std::vector<Eigen::Vector3d>& pts,
                                             double voxel) {
  std::unordered_set<VoxelCell, VoxelCellHash> occupied;
  occupied.reserve(pts.size());
  std::vector<Eigen::Vector3d> out;
  out.reserve(pts.size());
  // Survivors are appended while scanning the input, never by walking the set, so the
  // output order is the input's first-occupancy order.
  for (const Eigen::Vector3d& p : pts) {
    if (occupied.insert(cellOf(p, voxel)).second) {
      out.push_back(p);
    }
  }
  return out;
}

VoxelGridMap::VoxelGridMap(const FrontendLio& cfg) : cfg_(cfg) {}

void VoxelGridMap::insert(const std::vector<Eigen::Vector3d>& points_world) {
  const auto cap = static_cast<std::size_t>(cfg_.max_points_per_voxel);
  // Minimum squared spacing v^2 / cap: rejecting points closer than v / sqrt(cap) to
  // any stored point spreads a full voxel's points out instead of letting one surface
  // patch saturate the cap.
  const double min_spacing_sq =
      cfg_.voxel_size_m * cfg_.voxel_size_m / static_cast<double>(cfg_.max_points_per_voxel);
  for (const Eigen::Vector3d& p : points_world) {
    std::vector<Eigen::Vector3d>& pts = voxels_[cellOf(p, cfg_.voxel_size_m)];
    if (pts.size() >= cap) {
      continue;
    }
    bool too_close = false;
    for (const Eigen::Vector3d& q : pts) {
      if ((q - p).squaredNorm() < min_spacing_sq) {
        too_close = true;
        break;
      }
    }
    if (too_close) {
      continue;
    }
    if (pts.empty()) {
      pts.reserve(cap);
    }
    pts.push_back(p);
    ++size_;
  }
}

bool VoxelGridMap::nearest(const Eigen::Vector3d& query_world, Eigen::Vector3d* neighbor) const {
  const VoxelCell base = cellOf(query_world, cfg_.voxel_size_m);
  double best_sq = std::numeric_limits<double>::infinity();
  const Eigen::Vector3d* best = nullptr;
  // Fixed probe order (x, then y, then z, each -1..1) and in-order scans within each
  // voxel; ties resolve to the first candidate encountered, independent of hash-map
  // iteration order.
  for (std::int64_t dx = -1; dx <= 1; ++dx) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dz = -1; dz <= 1; ++dz) {
        const auto it = voxels_.find(VoxelCell{base.x + dx, base.y + dy, base.z + dz});
        if (it == voxels_.end()) {
          continue;
        }
        for (const Eigen::Vector3d& p : it->second) {
          const double d_sq = (p - query_world).squaredNorm();
          if (d_sq < best_sq) {
            best_sq = d_sq;
            best = &p;
          }
        }
      }
    }
  }
  if (best == nullptr || best_sq > cfg_.max_corr_dist_m * cfg_.max_corr_dist_m) {
    return false;
  }
  *neighbor = *best;
  return true;
}

void VoxelGridMap::neighborsWithin(const Eigen::Vector3d& query_world, double radius,
                                   std::vector<Eigen::Vector3d>* out) const {
  out->clear();
  const double radius_sq = radius * radius;
  const VoxelCell base = cellOf(query_world, cfg_.voxel_size_m);
  for (std::int64_t dx = -1; dx <= 1; ++dx) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dz = -1; dz <= 1; ++dz) {
        const auto it = voxels_.find(VoxelCell{base.x + dx, base.y + dy, base.z + dz});
        if (it == voxels_.end()) {
          continue;
        }
        for (const Eigen::Vector3d& p : it->second) {
          if ((p - query_world).squaredNorm() <= radius_sq) {
            out->push_back(p);
          }
        }
      }
    }
  }
}

void VoxelGridMap::clipFarFrom(const Eigen::Vector3d& center_world) {
  const double max_range_sq = cfg_.max_range_m * cfg_.max_range_m;
  const double v = cfg_.voxel_size_m;
  // Erasure depends only on each voxel's own cell index, so walking the map here is
  // order-independent.
  for (auto it = voxels_.begin(); it != voxels_.end();) {
    const VoxelCell& c = it->first;
    const Eigen::Vector3d cell_center{(static_cast<double>(c.x) + 0.5) * v,
                                      (static_cast<double>(c.y) + 0.5) * v,
                                      (static_cast<double>(c.z) + 0.5) * v};
    if ((cell_center - center_world).squaredNorm() > max_range_sq) {
      size_ -= it->second.size();
      it = voxels_.erase(it);
    } else {
      ++it;
    }
  }
}

void VoxelGridMap::clear() {
  voxels_.clear();
  size_ = 0;
}

std::size_t VoxelGridMap::size() const {
  return size_;
}

std::size_t VoxelGridMap::voxel_count() const {
  return voxels_.size();
}

}  // namespace meridian::lio
