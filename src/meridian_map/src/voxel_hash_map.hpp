#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "meridian/config/config.hpp"
#include "meridian/map/iregistration_map.hpp"

namespace meridian::map {

// Integer voxel coordinate at the base resolution.
struct VoxelKey {
  std::int32_t x = 0, y = 0, z = 0;
  bool operator==(const VoxelKey& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& k) const noexcept {
    return (static_cast<std::size_t>(static_cast<std::uint32_t>(k.x)) * 73856093u) ^
           (static_cast<std::size_t>(static_cast<std::uint32_t>(k.y)) * 19349663u) ^
           (static_cast<std::size_t>(static_cast<std::uint32_t>(k.z)) * 83492791u);
  }
};

inline constexpr std::size_t kMaxPtsPerVoxel = 20;  // fixed cap (spec default); cfg caps below this
inline constexpr std::size_t kNShards = 64;

// Per-voxel payload: a small bounded representative point set + a lazily-fit cached plane.
struct RegVoxel {
  std::array<Eigen::Vector3f, kMaxPtsPerVoxel> pts;
  std::uint8_t n = 0;       // valid points
  std::uint32_t seen = 0;   // total offered (reservoir counter)
  Eigen::Vector3f plane_n = Eigen::Vector3f::Zero();
  float plane_d = 0.f;
  float plane_rms = std::numeric_limits<float>::infinity();
  bool is_plane = false;
  Timestamp last_touch = 0;
};

// Tier R — the CPU registration voxel-hash (spec 06 §3). Bounded reservoir-sampled voxels
// with cached PCA planes, sharded for concurrent front-end read while the map thread
// writes. Per-keyframe -> voxel provenance lets a loop rebuild clear the exact region.
class VoxelHashMap final : public IRegistrationMap {
 public:
  explicit VoxelHashMap(const MapConfig& cfg);

  void integrate_live(const PointCloudMap& pts_map, Timestamp t) override;
  void integrate_keyframe(std::uint64_t id, const PointCloud& cloud_body,
                          const Pose& T_map_body) override;
  std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const override;
  void clear_keyframes(const std::vector<std::uint64_t>& ids) override;
  void set_hot_window(const Aabb& hot) override;
  MapDiagnostics diagnostics() const override;

  // Map-frame centroids of all occupied voxels (the registered-cloud debug view).
  std::vector<Eigen::Vector3f> centroids() const;
  std::size_t voxel_count() const;

 private:
  struct Shard {
    std::unordered_map<VoxelKey, RegVoxel, VoxelKeyHash> voxels;
    mutable std::shared_mutex mtx;
  };

  VoxelKey key_of(const Eigen::Vector3f& p_map) const;
  static std::size_t shard_of(const VoxelKey& k) { return VoxelKeyHash{}(k) & (kNShards - 1); }
  Eigen::Vector3f voxel_center(const VoxelKey& k) const;

  // Insert one map-frame point into its voxel (write lock held by caller path). Records
  // the touched key into `touched` when non-null (canonical keyframe path).
  void insert_point(const Eigen::Vector3f& p_map, Timestamp t, std::vector<VoxelKey>* touched);
  void refit_plane(RegVoxel& v) const;
  std::optional<PlaneHit> hit_from(const RegVoxel& v) const;

  float base_res_ = 0.2f;
  float inv_res_ = 5.f;
  std::size_t cap_ = kMaxPtsPerVoxel;
  float planarity_thresh_ = 0.1f;
  int min_plane_pts_ = 5;
  int neighbor_ring_ = 1;

  std::array<Shard, kNShards> shards_;
  mutable std::mutex rng_mtx_;
  mutable std::mt19937 rng_;  // seeded reservoir eviction (deterministic)

  mutable std::mutex prov_mtx_;
  std::unordered_map<std::uint64_t, std::vector<VoxelKey>> kf_voxels_;
};

}  // namespace meridian::map
