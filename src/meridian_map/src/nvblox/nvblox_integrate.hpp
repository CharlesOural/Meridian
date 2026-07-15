#pragma once

#include <nvblox/core/cuda_stream.h>
#include <nvblox/map/common_names.h>
#include <nvblox/sensors/image.h>
#include <nvblox/sensors/lidar.h>

#include <Eigen/Core>
#include <vector>

namespace meridian::map::nvcu {

// Rasterize a sensor-frame point cloud into the spherical range image `depth`
// (rows = lidar.num_elevation_divisions(), cols = lidar.num_azimuth_divisions()).
// Pixel values are euclidean range (not z-depth); each pixel keeps the nearest
// return that projects into it (min is order-independent, so the result is
// deterministic regardless of point order). Pixels with no return are 0, which
// the projective integrator treats as invalid. `depth` must already be sized
// rows x cols in device or unified memory.
void rasterize_cloud(const std::vector<Eigen::Vector3f>& points_sensor, const nvblox::Lidar& lidar,
                     nvblox::DepthImage* depth, const nvblox::CudaStream& stream);

// Per-vertex mesh confidence: clamp(W(p) / w_conf, 0, 1) where W(p) is the fused
// TSDF weight of the voxel containing p (layer frame). Positions in unobserved
// space get confidence 0.
void sample_confidence(const std::vector<Eigen::Vector3f>& positions_L,
                       const nvblox::TsdfLayer& tsdf, float w_conf, std::vector<float>* confidence,
                       const nvblox::CudaStream& stream);

}  // namespace meridian::map::nvcu
