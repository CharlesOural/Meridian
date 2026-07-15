#pragma once

// stdgpu's limits header (reached through the nvblox includes) consumes CHAR_BIT and
// friends without including <climits> itself; provide the macros first.
#include <nvblox/core/cuda_stream.h>
#include <nvblox/core/hash.h>
#include <nvblox/integrators/projective_color_integrator.h>
#include <nvblox/integrators/projective_tsdf_integrator.h>
#include <nvblox/map/common_names.h>
#include <nvblox/mesh/mesh_integrator.h>
#include <nvblox/sensors/image.h>

#include <Eigen/Core>
#include <array>
#include <climits>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/config/config.hpp"
#include "meridian/map/isurface_map.hpp"

namespace meridian::map {

// The GPU surface backend: nvblox TSDF + colour layers fused by its projective
// integrators, meshed by its GPU Marching Cubes. The one piece of CUDA Meridian
// writes is the cloud->range-image rasterizer and the confidence sampler
// (nvblox_integrate.cu); all fusion and meshing kernels are nvblox's own.
//
// Fusion semantics match CpuSurfaceMap: constant per-observation weight 1 with the
// running-average TSDF capped at tsdf_w_max, and colour blended toward the newest
// observation. nvblox's colour fusion is a weight-capped running average
// (W*c + c_obs)/(W + 1), so capping W at (1-a)/a makes it exactly the EWMA
// c <- (1-a)*c + a*c_obs once saturated, with the first observation kept verbatim.
class NvbloxSurfaceMap final : public ISurfaceMap {
public:
  NvbloxSurfaceMap(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib);

  void integrate(std::uint64_t id, const PointCloud& cloud_body,
                 const std::shared_ptr<const CameraFrame>& image, const Pose& T_map_body,
                 const Pose& T_body_cam) override;
  void clear_keyframes(const std::vector<std::uint64_t>& ids) override;
  Aabb dirty_bounds(const std::vector<std::uint64_t>& ids) const override;
  const ColorMesh& extract_mesh() override;
  MapDiagnostics diagnostics() const override;

  std::size_t block_count() const;

private:
  // Host copy of one mesh block, kept so unchanged blocks are never re-read.
  struct HostBlockMesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<std::array<std::uint8_t, 3>> colors;  // empty until colour is fused
    std::vector<std::uint32_t> indices;               // block-local triangle list
  };

  void integrate_color(const CameraFrame& image, const Pose& T_map_cam);
  void refresh_block_cache(const std::vector<nvblox::Index3D>& blocks);
  void assemble_mesh();

  float voxel_m_ = 0.05f;
  int trunc_vox_ = 4;
  float w_max_ = 8.f;
  float color_alpha_ = 0.8f;
  bool color_enable_ = true;
  float max_dist_ = 50.f;
  float conf_w_ = 8.f;
  bool mesh_incremental_ = true;

  std::shared_ptr<const CalibrationSet> calib_;
  std::shared_ptr<nvblox::CudaStream> stream_;
  nvblox::TsdfLayer tsdf_;
  nvblox::ColorLayer color_;
  nvblox::MeshLayer mesh_blocks_;
  nvblox::ProjectiveTsdfIntegrator tsdf_integ_;
  nvblox::ProjectiveColorIntegrator color_integ_;
  nvblox::MeshIntegrator mesh_integ_;

  nvblox::DepthImage depth_;
  nvblox::ColorImage color_img_;

  std::unordered_map<std::uint64_t, std::vector<nvblox::Index3D>> kf_blocks_;
  nvblox::Index3DSet pending_mesh_;
  bool any_color_ = false;

  nvblox::Index3DHashMapType<HostBlockMesh>::type block_cache_;
  ColorMesh mesh_;
  double last_integrate_ms_ = 0.0;
  double last_mesh_ms_ = 0.0;
};

}  // namespace meridian::map
