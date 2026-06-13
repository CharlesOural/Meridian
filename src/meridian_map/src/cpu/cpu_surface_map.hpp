#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "meridian/config/config.hpp"
#include "meridian/map/isurface_map.hpp"

namespace meridian::map {

// Integer voxel coordinate in the TSDF grid.
struct TsdfKey {
  std::int32_t x = 0, y = 0, z = 0;
  bool operator==(const TsdfKey& o) const noexcept { return x == o.x && y == o.y && z == o.z; }
};
struct TsdfKeyHash {
  std::size_t operator()(const TsdfKey& k) const noexcept {
    return (static_cast<std::size_t>(static_cast<std::uint32_t>(k.x)) * 73856093u) ^
           (static_cast<std::size_t>(static_cast<std::uint32_t>(k.y)) * 19349663u) ^
           (static_cast<std::size_t>(static_cast<std::uint32_t>(k.z)) * 83492791u);
  }
};

// One TSDF voxel: running-average signed distance + weight, and an EWMA-fused colour
// scalar (the contributing LiDAR intensity) with its own weight.
struct TsdfVoxel {
  float d = 0.f;
  float w = 0.f;
  float c = 0.f;   // fused colour scalar (intensity)
  float cw = 0.f;  // colour weight (0 = unset)
};

// The portable host surface backend (spec 06 §0, §4, §5). A projective TSDF fused along
// each LiDAR ray, an EWMA colour scalar, and a Marching-Cubes mesh extracted on demand.
// Same fusion semantics as the nvblox backend; lower throughput, no GPU. The mesh is
// fully re-extracted per call (throttled by the pipeline) rather than block-streamed.
class CpuSurfaceMap final : public ISurfaceMap {
 public:
  explicit CpuSurfaceMap(const MapConfig& cfg);

  void integrate(std::uint64_t id, const PointCloud& cloud_body,
                 const std::shared_ptr<const CameraFrame>& image, const Pose& T_map_body,
                 const Pose& T_body_cam) override;
  void clear_keyframes(const std::vector<std::uint64_t>& ids) override;
  const ColorMesh& extract_mesh() override;
  MapDiagnostics diagnostics() const override;

  std::size_t voxel_count() const { return voxels_.size(); }

 private:
  void fuse(const TsdfKey& k, float sdf, float color, bool near_surface,
            std::vector<TsdfKey>* touched);
  const TsdfVoxel* find(const TsdfKey& k) const;

  float voxel_m_ = 0.05f;
  float inv_voxel_ = 20.f;
  float trunc_ = 0.2f;
  float w_max_ = 8.f;
  float color_alpha_ = 0.8f;
  bool color_enable_ = true;
  float max_dist_ = 50.f;
  float conf_w_ = 8.f;

  std::unordered_map<TsdfKey, TsdfVoxel, TsdfKeyHash> voxels_;
  std::unordered_map<std::uint64_t, std::vector<TsdfKey>> kf_voxels_;
  ColorMesh mesh_;
  double last_integrate_ms_ = 0.0;
  double last_mesh_ms_ = 0.0;
};

}  // namespace meridian::map
