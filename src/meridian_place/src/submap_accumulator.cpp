#include "submap_accumulator.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace meridian {

SubmapAccumulator::SubmapAccumulator(std::shared_ptr<const KeyframeStore> store,
                                     PoseLookup pose_of, int window, double voxel, int cache)
    : store_(std::move(store)),
      pose_of_(std::move(pose_of)),
      window_(window),
      voxel_(voxel),
      capacity_(static_cast<std::size_t>(cache)) {}

void SubmapAccumulator::touch(std::uint64_t anchor) const {
  lru_.remove(anchor);
  lru_.push_front(anchor);
  while (lru_.size() > capacity_) {
    const std::uint64_t evict = lru_.back();
    lru_.pop_back();
    cache_.erase(evict);
  }
}

std::shared_ptr<const PointCloud> SubmapAccumulator::compose(
    std::uint64_t anchor, std::vector<std::uint64_t>& window_ids) const {
  window_ids.clear();
  const std::optional<Pose> T_anchor = pose_of_(anchor);
  if (!T_anchor) return nullptr;

  // The last `window_` keyframe ids <= anchor that the store holds (ascending).
  const std::vector<std::uint64_t> all = store_->ids();
  for (const std::uint64_t id : all) {
    if (id > anchor) break;
    window_ids.push_back(id);
  }
  if (static_cast<int>(window_ids.size()) > window_) {
    window_ids.erase(window_ids.begin(), window_ids.end() - window_);
  }

  const Pose T_anchor_inv = T_anchor->inverse();
  // First-point-wins voxel grid keyed by integer cell, iterated in cell order so the output
  // is independent of insertion order (deterministic) regardless of the source clouds.
  std::map<std::array<long long, 3>, LidarPoint> grid;
  const double inv = voxel_ > 0.0 ? 1.0 / voxel_ : 0.0;
  for (const std::uint64_t j : window_ids) {
    const std::optional<Pose> T_j = pose_of_(j);
    const PointCloudPtr cloud_j = store_->cloud(j);
    if (!T_j || !cloud_j) continue;
    const Pose T_a_j = T_anchor_inv * (*T_j);
    for (const LidarPoint& p : *cloud_j) {
      const Eigen::Vector3d pj = p.xyz.cast<double>();
      const Eigen::Vector3d pa = T_a_j * pj;
      LidarPoint q = p;
      q.xyz = pa.cast<float>();
      const std::array<long long, 3> key{
          static_cast<long long>(std::floor(pa.x() * inv)),
          static_cast<long long>(std::floor(pa.y() * inv)),
          static_cast<long long>(std::floor(pa.z() * inv))};
      grid.emplace(key, q);  // first point in a cell wins
    }
  }
  auto out = std::make_shared<PointCloud>();
  out->reserve(grid.size());
  for (const auto& [key, pt] : grid) out->push_back(pt);
  return out;
}

std::shared_ptr<const PointCloud> SubmapAccumulator::submap(std::uint64_t anchor) const {
  const auto it = cache_.find(anchor);
  if (it != cache_.end()) {
    touch(anchor);
    return it->second.cloud;
  }
  std::vector<std::uint64_t> window_ids;
  std::shared_ptr<const PointCloud> cloud = compose(anchor, window_ids);
  if (!cloud) return nullptr;
  cache_[anchor] = CacheEntry{cloud, std::move(window_ids)};
  touch(anchor);
  return cloud;
}

void SubmapAccumulator::invalidate(const std::vector<std::uint64_t>& moved_ids) {
  if (moved_ids.empty()) return;
  for (auto it = cache_.begin(); it != cache_.end();) {
    const auto& w = it->second.window_ids;
    const bool hit = std::any_of(moved_ids.begin(), moved_ids.end(), [&](std::uint64_t m) {
      return std::find(w.begin(), w.end(), m) != w.end();
    });
    if (hit) {
      lru_.remove(it->first);
      it = cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void SubmapAccumulator::clear() {
  cache_.clear();
  lru_.clear();
}

}  // namespace meridian
