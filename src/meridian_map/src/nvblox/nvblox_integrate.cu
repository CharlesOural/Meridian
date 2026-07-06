#include <cuda_runtime.h>
#include <limits.h>
#include <nvblox/core/types.h>

#include <cfloat>
#include <climits>
#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>
#include <stdexcept>
#include <string>

#include "nvblox/nvblox_integrate.hpp"

namespace meridian::map::nvcu {

namespace {

// GPU failures here are unrecoverable for the map: fail fast with the CUDA reason
// instead of silently fusing from garbage buffers.
void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("nvblox_integrate: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

constexpr int kThreads = 256;

int blocks_for(int n) {
  return (n + kThreads - 1) / kThreads;
}

// atomicMin for positive floats: IEEE-754 non-negative values order identically to
// their unsigned bit patterns, so an unsigned atomicMin is a float min. All ranges
// written here are > 0.
__device__ void atomic_min_positive(float* addr, float value) {
  atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(value));
}

__global__ void fill_kernel(float* img, int n, float value) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) img[i] = value;
}

__global__ void rasterize_kernel(const nvblox::Vector3f* pts, int n, nvblox::Lidar lidar,
                                 float* img, int cols) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const nvblox::Vector3f p = pts[i];
  nvblox::Index2D u;
  if (!lidar.project(p, &u)) return;
  atomic_min_positive(&img[u.y() * cols + u.x()], p.norm());
}

// Pixels no return reached keep the +inf sentinel; the projective integrator treats
// 0 as invalid depth, so map them to 0.
__global__ void finalize_kernel(float* img, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && img[i] == FLT_MAX) img[i] = 0.f;
}

__global__ void confidence_kernel(const nvblox::Vector3f* pos, int n,
                                  nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock> block_hash,
                                  float block_size, float inv_w_conf, float* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  nvblox::TsdfVoxel* voxel = nullptr;
  float c = 0.f;
  if (nvblox::getVoxelAtPosition<nvblox::TsdfVoxel>(block_hash, pos[i], block_size, &voxel)) {
    c = fminf(fmaxf(voxel->weight * inv_w_conf, 0.f), 1.f);
  }
  out[i] = c;
}

}  // namespace

void rasterize_cloud(const std::vector<Eigen::Vector3f>& points_sensor, const nvblox::Lidar& lidar,
                     nvblox::DepthImage* depth, const nvblox::CudaStream& stream) {
  const int rows = lidar.num_elevation_divisions();
  const int cols = lidar.num_azimuth_divisions();
  const int numel = rows * cols;
  if (numel <= 0) return;
  cudaStream_t s = stream.get();

  fill_kernel<<<blocks_for(numel), kThreads, 0, s>>>(depth->dataPtr(), numel, FLT_MAX);

  const int n = static_cast<int>(points_sensor.size());
  if (n > 0) {
    nvblox::Vector3f* d_pts = nullptr;
    check_cuda(cudaMallocAsync(&d_pts, sizeof(nvblox::Vector3f) * n, s), "point buffer alloc");
    check_cuda(cudaMemcpyAsync(d_pts, points_sensor.data(), sizeof(nvblox::Vector3f) * n,
                               cudaMemcpyHostToDevice, s),
               "point upload");
    rasterize_kernel<<<blocks_for(n), kThreads, 0, s>>>(d_pts, n, lidar, depth->dataPtr(), cols);
    check_cuda(cudaFreeAsync(d_pts, s), "point buffer free");
  }
  finalize_kernel<<<blocks_for(numel), kThreads, 0, s>>>(depth->dataPtr(), numel);
  check_cuda(cudaGetLastError(), "rasterize kernel launch");
  check_cuda(cudaStreamSynchronize(s), "rasterize sync");
}

void sample_confidence(const std::vector<Eigen::Vector3f>& positions_L,
                       const nvblox::TsdfLayer& tsdf, float w_conf, std::vector<float>* confidence,
                       const nvblox::CudaStream& stream) {
  const int n = static_cast<int>(positions_L.size());
  confidence->assign(positions_L.size(), 0.f);
  if (n == 0 || tsdf.numAllocatedBlocks() == 0) return;
  cudaStream_t s = stream.get();

  auto& gpu_view = tsdf.getGpuLayerView(stream);

  nvblox::Vector3f* d_pos = nullptr;
  float* d_out = nullptr;
  check_cuda(cudaMallocAsync(&d_pos, sizeof(nvblox::Vector3f) * n, s), "position buffer alloc");
  check_cuda(cudaMallocAsync(&d_out, sizeof(float) * n, s), "confidence buffer alloc");
  check_cuda(cudaMemcpyAsync(d_pos, positions_L.data(), sizeof(nvblox::Vector3f) * n,
                             cudaMemcpyHostToDevice, s),
             "position upload");

  const float inv_w_conf = w_conf > 0.f ? 1.f / w_conf : 0.f;
  confidence_kernel<<<blocks_for(n), kThreads, 0, s>>>(d_pos, n, gpu_view.getHash().impl_,
                                                       tsdf.block_size(), inv_w_conf, d_out);
  check_cuda(cudaGetLastError(), "confidence kernel launch");

  check_cuda(
      cudaMemcpyAsync(confidence->data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost, s),
      "confidence download");
  check_cuda(cudaFreeAsync(d_pos, s), "position buffer free");
  check_cuda(cudaFreeAsync(d_out, s), "confidence buffer free");
  check_cuda(cudaStreamSynchronize(s), "confidence sync");
}

}  // namespace meridian::map::nvcu
