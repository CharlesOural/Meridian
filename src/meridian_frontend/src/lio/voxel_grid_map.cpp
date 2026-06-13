#include "lio/voxel_grid_map.hpp"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace meridian::lio {

namespace {

constexpr std::size_t kInitialSlots = 1024;  // power of two

VoxelCell cellOf(const Eigen::Vector3d& p, const double voxel) {
  return VoxelCell{static_cast<std::int64_t>(std::floor(p.x() / voxel)),
                   static_cast<std::int64_t>(std::floor(p.y() / voxel)),
                   static_cast<std::int64_t>(std::floor(p.z() / voxel))};
}

// Linear-probe slot for `cell` in `table` (mask = size-1): the slot holding the key, or
// the first empty slot on its probe chain (the insertion point on a miss).
template <typename Slot>
std::size_t probeSlot(const std::vector<Slot>& table, std::size_t mask, const VoxelCell& cell) {
  std::size_t idx = VoxelCellHash{}(cell) & mask;
  while (table[idx].used && !(table[idx].key == cell)) {
    idx = (idx + 1) & mask;
  }
  return idx;
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

VoxelGridMap::VoxelGridMap(const FrontendLio& cfg)
    : cfg_(cfg), table_(kInitialSlots), mask_(kInitialSlots - 1) {}

const std::vector<Eigen::Vector3d>* VoxelGridMap::cellPoints(const VoxelCell& cell) const {
  const std::size_t idx = probeSlot(table_, mask_, cell);
  return table_[idx].used ? &table_[idx].pts : nullptr;
}

std::size_t VoxelGridMap::probe(const VoxelCell& cell) const {
  return probeSlot(table_, mask_, cell);
}

void VoxelGridMap::growIfNeeded() {
  // Keep the load factor below 0.7 so probe chains stay short.
  if ((used_ + 1) * 10 < table_.size() * 7) {
    return;
  }
  std::vector<Slot> next(table_.size() * 2);
  const std::size_t next_mask = next.size() - 1;
  for (Slot& s : table_) {
    if (!s.used) {
      continue;
    }
    const std::size_t idx = probeSlot(next, next_mask, s.key);
    next[idx] = std::move(s);
  }
  table_ = std::move(next);
  mask_ = next_mask;
}

void VoxelGridMap::insert(const std::vector<Eigen::Vector3d>& points_world) {
  const auto cap = static_cast<std::size_t>(cfg_.max_points_per_voxel);
  // Minimum squared spacing v^2 / cap: rejecting points closer than v / sqrt(cap) to
  // any stored point spreads a full voxel's points out instead of letting one surface
  // patch saturate the cap.
  const double min_spacing_sq =
      cfg_.voxel_size_m * cfg_.voxel_size_m / static_cast<double>(cfg_.max_points_per_voxel);
  for (const Eigen::Vector3d& p : points_world) {
    growIfNeeded();
    const VoxelCell cell = cellOf(p, cfg_.voxel_size_m);
    const std::size_t idx = probe(cell);
    Slot& slot = table_[idx];
    if (!slot.used) {
      slot.key = cell;
      slot.used = true;
      ++used_;
    }
    std::vector<Eigen::Vector3d>& pts = slot.pts;
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
  // voxel; ties resolve to the first candidate encountered, independent of slot layout.
  for (std::int64_t dx = -1; dx <= 1; ++dx) {
    for (std::int64_t dy = -1; dy <= 1; ++dy) {
      for (std::int64_t dz = -1; dz <= 1; ++dz) {
        const std::vector<Eigen::Vector3d>* pts =
            cellPoints(VoxelCell{base.x + dx, base.y + dy, base.z + dz});
        if (pts == nullptr) {
          continue;
        }
        for (const Eigen::Vector3d& p : *pts) {
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
        const std::vector<Eigen::Vector3d>* pts =
            cellPoints(VoxelCell{base.x + dx, base.y + dy, base.z + dz});
        if (pts == nullptr) {
          continue;
        }
        for (const Eigen::Vector3d& p : *pts) {
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
  // Walk every slot; far cells are erased in place by backward-shift so the probe chains
  // of the survivors stay intact without a tombstone or a full rebuild. After a deletion
  // the slot is re-examined, since the shift may pull another (possibly far) entry into it.
  std::size_t i = 0;
  while (i < table_.size()) {
    if (!table_[i].used) {
      ++i;
      continue;
    }
    const VoxelCell& c = table_[i].key;
    const Eigen::Vector3d cell_center{(static_cast<double>(c.x) + 0.5) * v,
                                      (static_cast<double>(c.y) + 0.5) * v,
                                      (static_cast<double>(c.z) + 0.5) * v};
    if ((cell_center - center_world).squaredNorm() <= max_range_sq) {
      ++i;
      continue;
    }
    // Erase slot i and close the gap by pulling forward any later entry whose probe chain
    // runs through i.
    size_ -= table_[i].pts.size();
    table_[i].pts.clear();
    table_[i].used = false;
    --used_;
    std::size_t hole = i;
    std::size_t j = i;
    while (true) {
      j = (j + 1) & mask_;
      if (!table_[j].used) {
        break;
      }
      const std::size_t h = VoxelCellHash{}(table_[j].key) & mask_;
      // Move j into the hole unless its ideal slot h lies cyclically in (hole, j], which
      // would mean j is already correctly placed relative to the hole.
      const bool in_interval =
          (hole <= j) ? (h > hole && h <= j) : (h > hole || h <= j);
      if (!in_interval) {
        table_[hole] = std::move(table_[j]);
        table_[hole].used = true;
        table_[j].used = false;
        table_[j].pts.clear();
        hole = j;
      }
    }
    // Re-examine slot i: it now holds whatever was shifted in (or is empty).
  }
}

void VoxelGridMap::clear() {
  table_.assign(kInitialSlots, Slot{});
  mask_ = kInitialSlots - 1;
  used_ = 0;
  size_ = 0;
}

std::size_t VoxelGridMap::size() const {
  return size_;
}

std::size_t VoxelGridMap::voxel_count() const {
  return used_;
}

}  // namespace meridian::lio
