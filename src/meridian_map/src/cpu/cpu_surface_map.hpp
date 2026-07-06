#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "meridian/calib/calibration_set.hpp"
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

// One TSDF voxel: running-average signed distance + weight, and an EWMA-fused RGB colour
// stored in linear-light space with its own weight (cw == 0 means never painted).
struct TsdfVoxel {
  float d = 0.f;
  float w = 0.f;
  float cr = 0.f;  // linear-light red
  float cg = 0.f;  // linear-light green
  float cb = 0.f;  // linear-light blue
  float cw = 0.f;  // colour weight (0 = unpainted)
};

// The portable host surface backend. A projective TSDF fused along each LiDAR ray,
// camera-projected RGB colour, and a Marching-Cubes mesh extracted on demand. Same
// fusion semantics as the nvblox backend; lower throughput, no GPU. The mesh is fully
// re-extracted per call (throttled by the pipeline) rather than block-streamed.
class CpuSurfaceMap final : public ISurfaceMap {
public:
  explicit CpuSurfaceMap(const MapConfig& cfg,
                         std::shared_ptr<const CalibrationSet> calib = nullptr);

  void integrate(std::uint64_t id, const PointCloud& cloud_body,
                 const std::shared_ptr<const CameraFrame>& image, const Pose& T_map_body,
                 const Pose& T_body_cam) override;
  void clear_keyframes(const std::vector<std::uint64_t>& ids) override;
  Aabb dirty_bounds(const std::vector<std::uint64_t>& ids) const override;
  const ColorMesh& extract_mesh() override;
  MapDiagnostics diagnostics() const override;

  std::size_t voxel_count() const { return voxels_.size(); }

private:
  void fuse(const TsdfKey& k, float sdf, std::vector<TsdfKey>* touched);
  const TsdfVoxel* find(const TsdfKey& k) const;

  // Project each near-surface voxel this keyframe touched into the camera and, when it is
  // in-bounds and not occluded, blend the sampled pixel colour into the voxel.
  void colour_voxels(const CameraFrame& image, const IntrinsicsCamera& intr, const Pose& T_map_cam,
                     const std::vector<TsdfKey>& near_keys);
  // Discrete sphere trace: true if a fused surface lies between the camera and x_map.
  bool occluded(const Eigen::Vector3f& cam_o, const Eigen::Vector3f& x_map) const;

  float voxel_m_ = 0.05f;
  float inv_voxel_ = 20.f;
  float trunc_ = 0.2f;
  float w_max_ = 8.f;
  float color_alpha_ = 0.8f;
  bool color_enable_ = true;
  bool color_occlusion_ = true;
  float max_dist_ = 50.f;
  float conf_w_ = 8.f;

  std::shared_ptr<const CalibrationSet> calib_;
  std::unordered_map<TsdfKey, TsdfVoxel, TsdfKeyHash> voxels_;
  std::unordered_map<std::uint64_t, std::vector<TsdfKey>> kf_voxels_;
  ColorMesh mesh_;
  bool any_color_ = false;
  double last_integrate_ms_ = 0.0;
  double last_mesh_ms_ = 0.0;
};

}  // namespace meridian::map
