#include "voxel_hash_map.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <shared_mutex>
#include <tuple>

#include <Eigen/Eigenvalues>

namespace meridian::map {

VoxelHashMap::VoxelHashMap(const MapConfig& cfg)
    : base_res_(static_cast<float>(cfg.reg_voxel_m)),
      inv_res_(base_res_ > 0.f ? 1.f / base_res_ : 1.f),
      cap_(std::min<std::size_t>(kMaxPtsPerVoxel,
                                 cfg.reg_max_pts > 0 ? static_cast<std::size_t>(cfg.reg_max_pts)
                                                     : kMaxPtsPerVoxel)),
      planarity_thresh_(static_cast<float>(cfg.reg_planarity)),
      min_plane_pts_(cfg.reg_min_plane_pts),
      neighbor_ring_(cfg.reg_neighbor_ring),
      rng_(static_cast<std::mt19937::result_type>(cfg.reg_seed)) {}

VoxelKey VoxelHashMap::key_of(const Eigen::Vector3f& p) const {
  return VoxelKey{static_cast<std::int32_t>(std::floor(p.x() * inv_res_)),
                  static_cast<std::int32_t>(std::floor(p.y() * inv_res_)),
                  static_cast<std::int32_t>(std::floor(p.z() * inv_res_))};
}

Eigen::Vector3f VoxelHashMap::voxel_center(const VoxelKey& k) const {
  return Eigen::Vector3f((static_cast<float>(k.x) + 0.5f) * base_res_,
                         (static_cast<float>(k.y) + 0.5f) * base_res_,
                         (static_cast<float>(k.z) + 0.5f) * base_res_);
}

void VoxelHashMap::refit_plane(RegVoxel& v) const {
  if (v.n < min_plane_pts_) {
    v.is_plane = false;
    v.plane_rms = std::numeric_limits<float>::infinity();
    return;
  }
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (std::uint8_t i = 0; i < v.n; ++i) mean += v.pts[i];
  mean /= static_cast<float>(v.n);
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (std::uint8_t i = 0; i < v.n; ++i) {
    const Eigen::Vector3f d = v.pts[i] - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(v.n);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
  if (es.info() != Eigen::Success) {
    v.is_plane = false;
    return;
  }
  // Ascending eigenvalues; smallest -> plane normal.
  const float l0 = std::max(es.eigenvalues()(0), 0.f);
  const float l1 = std::max(es.eigenvalues()(1), 1e-12f);
  Eigen::Vector3f n = es.eigenvectors().col(0).normalized();
  v.plane_n = n;
  v.plane_d = -n.dot(mean);
  v.plane_rms = std::sqrt(l0);
  v.is_plane = (l0 / l1) < planarity_thresh_;
}

std::optional<PlaneHit> VoxelHashMap::hit_from(const RegVoxel& v) const {
  if (!v.is_plane || v.n == 0) return std::nullopt;
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (std::uint8_t i = 0; i < v.n; ++i) mean += v.pts[i];
  mean /= static_cast<float>(v.n);
  PlaneHit h;
  h.n = v.plane_n;
  h.d = v.plane_d;
  h.rms = v.plane_rms;
  h.centroid = mean;
  h.n_pts = v.n;
  return h;
}

void VoxelHashMap::insert_point(const Eigen::Vector3f& p, Timestamp t,
                                std::vector<VoxelKey>* touched) {
  const VoxelKey k = key_of(p);
  Shard& s = shards_[shard_of(k)];
  {
    std::unique_lock lock(s.mtx);
    RegVoxel& v = s.voxels[k];
    v.last_touch = t;
    ++v.seen;
    if (v.n < cap_) {
      v.pts[v.n++] = p;
    } else {
      // Reservoir: replace a uniformly-chosen slot with probability cap/seen.
      std::uint32_t j;
      {
        std::lock_guard<std::mutex> rl(rng_mtx_);
        j = std::uniform_int_distribution<std::uint32_t>(0, v.seen - 1)(rng_);
      }
      if (j < cap_) v.pts[j] = p;
    }
    refit_plane(v);  // eager refit so query_plane needs only a read lock
  }
  if (touched) touched->push_back(k);
}

void VoxelHashMap::integrate_live(const PointCloudMap& pts_map, Timestamp t) {
  for (const Eigen::Vector3f& p : pts_map) insert_point(p, t, nullptr);
}

void VoxelHashMap::integrate_keyframe(std::uint64_t id, const PointCloud& cloud_body,
                                      const Pose& T_map_body) {
  std::vector<VoxelKey> touched;
  touched.reserve(cloud_body.size());
  for (const LidarPoint& lp : cloud_body) {
    const Eigen::Vector3f p_map = (T_map_body * lp.xyz.cast<double>()).cast<float>();
    insert_point(p_map, 0, &touched);
  }
  std::sort(touched.begin(), touched.end(), [](const VoxelKey& a, const VoxelKey& b) {
    return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
  });
  touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
  std::lock_guard<std::mutex> pl(prov_mtx_);
  kf_voxels_[id] = std::move(touched);
}

std::optional<PlaneHit> VoxelHashMap::query_plane(const Eigen::Vector3f& p_map) const {
  const VoxelKey k0 = key_of(p_map);
  {
    const Shard& s = shards_[shard_of(k0)];
    std::shared_lock lock(s.mtx);
    const auto it = s.voxels.find(k0);
    if (it != s.voxels.end()) {
      if (auto h = hit_from(it->second)) return h;
    }
  }
  if (neighbor_ring_ <= 0) return std::nullopt;
  // Probe the one-ring neighbours; return the lowest-rms planar hit.
  std::optional<PlaneHit> best;
  for (int dx = -neighbor_ring_; dx <= neighbor_ring_; ++dx) {
    for (int dy = -neighbor_ring_; dy <= neighbor_ring_; ++dy) {
      for (int dz = -neighbor_ring_; dz <= neighbor_ring_; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        const VoxelKey k{k0.x + dx, k0.y + dy, k0.z + dz};
        const Shard& s = shards_[shard_of(k)];
        std::shared_lock lock(s.mtx);
        const auto it = s.voxels.find(k);
        if (it == s.voxels.end()) continue;
        auto h = hit_from(it->second);
        if (h && (!best || h->rms < best->rms)) best = h;
      }
    }
  }
  return best;
}

void VoxelHashMap::clear_keyframes(const std::vector<std::uint64_t>& ids) {
  std::vector<VoxelKey> keys;
  {
    std::lock_guard<std::mutex> pl(prov_mtx_);
    for (std::uint64_t id : ids) {
      const auto it = kf_voxels_.find(id);
      if (it == kf_voxels_.end()) continue;
      keys.insert(keys.end(), it->second.begin(), it->second.end());
      kf_voxels_.erase(it);
    }
  }
  for (const VoxelKey& k : keys) {
    Shard& s = shards_[shard_of(k)];
    std::unique_lock lock(s.mtx);
    s.voxels.erase(k);
  }
}

void VoxelHashMap::set_hot_window(const Aabb& hot) {
  for (Shard& s : shards_) {
    std::unique_lock lock(s.mtx);
    for (auto it = s.voxels.begin(); it != s.voxels.end();) {
      if (!hot.contains(voxel_center(it->first))) {
        it = s.voxels.erase(it);
      } else {
        ++it;
      }
    }
  }
}

std::vector<Eigen::Vector3f> VoxelHashMap::centroids() const {
  std::vector<Eigen::Vector3f> out;
  for (const Shard& s : shards_) {
    std::shared_lock lock(s.mtx);
    for (const auto& [k, v] : s.voxels) {
      if (v.n == 0) continue;
      Eigen::Vector3f mean = Eigen::Vector3f::Zero();
      for (std::uint8_t i = 0; i < v.n; ++i) mean += v.pts[i];
      out.push_back(mean / static_cast<float>(v.n));
    }
  }
  return out;
}

std::size_t VoxelHashMap::voxel_count() const {
  std::size_t n = 0;
  for (const Shard& s : shards_) {
    std::shared_lock lock(s.mtx);
    n += s.voxels.size();
  }
  return n;
}

MapDiagnostics VoxelHashMap::diagnostics() const {
  MapDiagnostics d;
  d.reg_voxels = voxel_count();
  d.hot_window_voxels = d.reg_voxels;
  return d;
}

}  // namespace meridian::map
