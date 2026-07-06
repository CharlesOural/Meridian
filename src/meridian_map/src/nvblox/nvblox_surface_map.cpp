#include "nvblox/nvblox_surface_map.hpp"

#include <nvblox/sensors/camera.h>
#include <nvblox/sensors/lidar.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include "nvblox/nvblox_integrate.hpp"

namespace meridian::map {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Turbo colour map (degree-5 polynomial), x in [0,1] -> RGB bytes.
std::array<std::uint8_t, 3> turbo(float x) {
  x = std::clamp(x, 0.f, 1.f);
  const float x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x4 * x;
  const auto ch = [&](float c0, float c1, float c2, float c3, float c4, float c5) {
    const float v = c0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4 + c5 * x5;
    return static_cast<std::uint8_t>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
  };
  return {ch(0.13572138f, 4.61539260f, -42.66032258f, 132.13108234f, -152.94239396f, 59.28637943f),
          ch(0.09140261f, 2.19418839f, 4.84296658f, -14.18503333f, 4.27729857f, 2.82956604f),
          ch(0.10667330f, 12.64194608f, -60.58204836f, 110.36276771f, -89.90310912f, 27.34824973f)};
}

nvblox::Transform to_isometry(const Pose& p) {
  nvblox::Transform T = nvblox::Transform::Identity();
  T.linear() = p.q.toRotationMatrix().cast<float>();
  T.translation() = p.t.cast<float>();
  return T;
}

// Weight cap that turns the running average (W*c + c_obs)/(W+1) into the EWMA
// (1-alpha)*c + alpha*c_obs once W saturates. The cap must stay > 0, so alpha ~ 1
// degenerates to an epsilon cap (newest observation wins outright).
float color_weight_cap(float alpha) {
  constexpr float kMinCap = 1e-4f;
  if (alpha >= 1.f) return kMinCap;
  if (alpha <= 0.f) return std::numeric_limits<float>::max();
  return std::max(kMinCap, (1.f - alpha) / alpha);
}

// Range-image geometry for one keyframe's rasterization.
struct LidarGrid {
  int rows = 0;
  int cols = 0;
  float below = 0.f;  // magnitude of elevation below horizon [rad]
  float above = 0.f;  // elevation above horizon [rad]
};

constexpr int kAzimuthDivisions = 1024;  // even, ~0.35 deg azimuth bins
constexpr int kDefaultRows = 64;
constexpr float kMinRange = 0.1f;

// Sort key for deterministic block traversal.
bool index_less(const nvblox::Index3D& a, const nvblox::Index3D& b) {
  if (a.x() != b.x()) return a.x() < b.x();
  if (a.y() != b.y()) return a.y() < b.y();
  return a.z() < b.z();
}

std::array<std::uint8_t, 3> to_rgb(const nvblox::Color& c) {
  return {c.r, c.g, c.b};
}

}  // namespace

NvbloxSurfaceMap::NvbloxSurfaceMap(const MapConfig& cfg,
                                   std::shared_ptr<const CalibrationSet> calib)
    : voxel_m_(static_cast<float>(cfg.tsdf_voxel_m)),
      trunc_vox_(cfg.tsdf_trunc_voxels),
      w_max_(static_cast<float>(cfg.tsdf_w_max)),
      color_alpha_(static_cast<float>(cfg.color_ewma_alpha)),
      color_enable_(cfg.colour),
      max_dist_(static_cast<float>(cfg.tsdf_max_integration_dist_m)),
      conf_w_(static_cast<float>(cfg.mesh_conf_w)),
      mesh_incremental_(cfg.mesh_incremental),
      calib_(std::move(calib)),
      stream_(std::make_shared<nvblox::CudaStreamOwning>()),
      tsdf_(voxel_m_, nvblox::MemoryType::kDevice),
      color_(voxel_m_, nvblox::MemoryType::kDevice),
      // Unified memory: mesh kernels write the blocks, the host cache reads them back.
      mesh_blocks_(tsdf_.block_size(), nvblox::MemoryType::kUnified),
      tsdf_integ_(stream_),
      color_integ_(stream_),
      mesh_integ_(stream_),
      depth_(nvblox::MemoryType::kDevice),
      color_img_(nvblox::MemoryType::kDevice) {
  tsdf_integ_.weighting_function_type(nvblox::WeightingFunctionType::kConstantWeight);
  tsdf_integ_.max_weight(w_max_);
  tsdf_integ_.truncation_distance_vox(static_cast<float>(trunc_vox_));
  tsdf_integ_.max_integration_distance_m(max_dist_);

  color_integ_.weighting_function_type(nvblox::WeightingFunctionType::kConstantWeight);
  color_integ_.max_weight(color_weight_cap(color_alpha_));
  color_integ_.truncation_distance_vox(static_cast<float>(trunc_vox_));
  color_integ_.max_integration_distance_m(max_dist_);
  // The occlusion sphere tracer keeps its own ray-length limit, snapshotted at
  // construction; without matching it to the cutoff, no surface beyond that default
  // (7 m) would ever receive colour.
  color_integ_.sphere_tracer().maximum_ray_length_m(max_dist_);

  mesh_integ_.weld_vertices(true);
}

std::size_t NvbloxSurfaceMap::block_count() const {
  return static_cast<std::size_t>(tsdf_.numAllocatedBlocks());
}

void NvbloxSurfaceMap::integrate(std::uint64_t id, const PointCloud& cloud_body,
                                 const std::shared_ptr<const CameraFrame>& image,
                                 const Pose& T_map_body, const Pose& T_body_cam) {
  const auto t0 = Clock::now();
  auto& owned = kf_blocks_[id];  // provenance entry exists even for an empty sweep

  // Host prepass: keep finite in-range returns and measure the elevation extents the
  // rasterization grid must cover.
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(cloud_body.size());
  float el_min = std::numeric_limits<float>::max();
  float el_max = std::numeric_limits<float>::lowest();
  int max_ring = -1;
  for (const LidarPoint& lp : cloud_body) {
    const Eigen::Vector3f& p = lp.xyz;
    if (!p.allFinite()) continue;
    const float r = p.norm();
    if (r < kMinRange || r > max_dist_) continue;
    const float el = std::asin(std::clamp(p.z() / r, -1.f, 1.f));
    el_min = std::min(el_min, el);
    el_max = std::max(el_max, el);
    max_ring = std::max(max_ring, static_cast<int>(lp.ring));
    pts.push_back(p);
  }
  if (pts.empty()) {
    last_integrate_ms_ = ms_since(t0);
    return;
  }

  // The grid rows follow the sensor's beam count when rings are present; the elevation
  // span hugs the data with a half-beam margin so extreme beams land inside the image.
  LidarGrid grid;
  grid.rows = std::clamp(max_ring > 0 ? max_ring + 1 : kDefaultRows, 2, 512);
  grid.cols = kAzimuthDivisions;
  float span = el_max - el_min;
  if (span < 1e-3f) {  // planar sweep: widen so the single beam sits mid-image
    el_min -= 0.05f;
    el_max += 0.05f;
    span = el_max - el_min;
  }
  const float pad = 0.5f * span / static_cast<float>(grid.rows - 1);
  // The lidar model requires a strictly positive angle on each side of the horizon;
  // clamp so fully-above or fully-below clouds still get a valid (slightly wider) grid.
  grid.above = std::max(el_max + pad, 0.01f);
  grid.below = std::max(-(el_min - pad), 0.01f);
  // When the clamp stretched the grid past the data span (one-sided clouds), scale the
  // row count to keep the per-row spacing at the data's density, else distinct beams
  // collide into one row and returns are silently dropped.
  const float span_padded = span + 2.f * pad;
  const float span_grid = grid.above + grid.below;
  if (span_grid > span_padded * 1.01f) {
    const float scale = span_grid / span_padded;
    grid.rows = std::clamp(static_cast<int>(std::ceil(static_cast<float>(grid.rows) * scale)),
                           grid.rows, 512);
  }

  const nvblox::Lidar lidar(grid.cols, grid.rows, kMinRange, max_dist_ + voxel_m_, grid.below,
                            grid.above);

  if (depth_.rows() != grid.rows || depth_.cols() != grid.cols) {
    depth_.resizeAsync(grid.rows, grid.cols, *stream_);
  }
  nvcu::rasterize_cloud(pts, lidar, &depth_, *stream_);

  std::vector<nvblox::Index3D> updated;
  tsdf_integ_.integrateFrame(nvblox::MaskedDepthImageConstView(depth_), to_isometry(T_map_body),
                             lidar, &tsdf_, &updated);
  owned.insert(owned.end(), updated.begin(), updated.end());
  pending_mesh_.insert(updated.begin(), updated.end());

  if (color_enable_ && image && image->data && calib_) {
    integrate_color(*image, T_map_body * T_body_cam);
  }
  last_integrate_ms_ = ms_since(t0);
}

void NvbloxSurfaceMap::integrate_color(const CameraFrame& image, const Pose& T_map_cam) {
  const auto it = calib_->cam_intrinsics.find(image.sensor_id);
  if (it == calib_->cam_intrinsics.end()) return;
  const IntrinsicsCamera& K = it->second;
  const int w = image.width;
  const int h = image.height;
  if (w <= 0 || h <= 0 ||
      image.data->size() < static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                               (image.encoding == CameraFrame::Encoding::RGB8 ? 3u : 1u)) {
    return;
  }

  // Expand to the RGBA layout nvblox fuses. Bayer uses the 2x2-quad averages: full
  // demosaicing quality is irrelevant at voxel resolution.
  std::vector<nvblox::Color> rgba(static_cast<std::size_t>(w) * h);
  const std::uint8_t* src = image.data->data();
  switch (image.encoding) {
    case CameraFrame::Encoding::Mono8:
      for (std::size_t i = 0; i < rgba.size(); ++i) {
        const std::uint8_t v = src[i];
        rgba[i] = nvblox::Color(v, v, v);
      }
      break;
    case CameraFrame::Encoding::RGB8:
      for (std::size_t i = 0; i < rgba.size(); ++i) {
        rgba[i] = nvblox::Color(src[3 * i], src[3 * i + 1], src[3 * i + 2]);
      }
      break;
    case CameraFrame::Encoding::Bayer_RGGB8:
      for (int y = 0; y < h; ++y) {
        const int qy = y & ~1;
        for (int x = 0; x < w; ++x) {
          const int qx = x & ~1;
          const int y1 = std::min(qy + 1, h - 1);
          const int x1 = std::min(qx + 1, w - 1);
          const std::uint8_t r = src[qy * w + qx];
          const std::uint8_t g = static_cast<std::uint8_t>(
              (static_cast<int>(src[qy * w + x1]) + src[y1 * w + qx]) / 2);
          const std::uint8_t b = src[y1 * w + x1];
          rgba[static_cast<std::size_t>(y) * w + x] = nvblox::Color(r, g, b);
        }
      }
      break;
  }
  color_img_.copyFromAsync(static_cast<std::size_t>(h), static_cast<std::size_t>(w), rgba.data(),
                           *stream_);
  stream_->synchronize();

  const nvblox::Camera cam(static_cast<float>(K.fx), static_cast<float>(K.fy),
                           static_cast<float>(K.cx), static_cast<float>(K.cy), w, h);
  std::vector<nvblox::Index3D> updated;
  color_integ_.integrateFrame(color_img_, to_isometry(T_map_cam), cam, tsdf_, &color_, &updated);
  if (updated.empty()) return;

  // The colour pass reports every in-band block in the camera frustum, including blocks
  // whose geometry other keyframes own. Recording those as this keyframe's provenance
  // would let a later clear_keyframes delete geometry that no rebuild re-integrates, so
  // provenance takes nothing from the colour pass: this keyframe's own blocks are
  // already recorded by the geometry pass, and repainting another keyframe's block does
  // not transfer ownership. The mesh recolour set still takes the full list (existing
  // surface may have been repainted).
  pending_mesh_.insert(updated.begin(), updated.end());
  any_color_ = true;
}

void NvbloxSurfaceMap::clear_keyframes(const std::vector<std::uint64_t>& ids) {
  nvblox::Index3DSet to_clear;
  for (const std::uint64_t id : ids) {
    const auto it = kf_blocks_.find(id);
    if (it == kf_blocks_.end()) continue;
    to_clear.insert(it->second.begin(), it->second.end());
    kf_blocks_.erase(it);
  }
  if (to_clear.empty()) return;

  std::vector<nvblox::Index3D> blocks(to_clear.begin(), to_clear.end());
  std::sort(blocks.begin(), blocks.end(), index_less);

  // The three layers clear together so a surviving TSDF block never re-meshes against
  // stale colour or a stale mesh block.
  tsdf_.clearBlocks(blocks);
  color_.clearBlocks(blocks);
  mesh_blocks_.clearBlocks(blocks);
  for (const nvblox::Index3D& b : blocks) {
    block_cache_.erase(b);
    pending_mesh_.erase(b);
  }
  // Marching Cubes reads across block faces, so surviving neighbours of a cleared block
  // hold triangles built from the cleared geometry and must be re-extracted.
  for (const nvblox::Index3D& b : blocks) {
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const nvblox::Index3D n(b.x() + dx, b.y() + dy, b.z() + dz);
          if (tsdf_.isBlockAllocated(n)) pending_mesh_.insert(n);
        }
      }
    }
  }
}

Aabb NvbloxSurfaceMap::dirty_bounds(const std::vector<std::uint64_t>& ids) const {
  Aabb box;
  const float block_m = tsdf_.block_size();
  for (const std::uint64_t id : ids) {
    const auto it = kf_blocks_.find(id);
    if (it == kf_blocks_.end()) continue;
    for (const nvblox::Index3D& b : it->second) {
      const Eigen::Vector3f lo = b.cast<float>() * block_m;
      box.expand(lo);
      box.expand(lo + Eigen::Vector3f::Constant(block_m));
    }
  }
  return box;
}

const ColorMesh& NvbloxSurfaceMap::extract_mesh() {
  const auto t0 = Clock::now();

  if (!mesh_incremental_) {
    // Full re-extraction: drop everything and mesh the whole layer. Simple oracle path;
    // cost scales with map size.
    mesh_blocks_.clear();
    block_cache_.clear();
    mesh_integ_.integrateMeshFromDistanceField(tsdf_, &mesh_blocks_, nvblox::DeviceType::kGPU);
    if (any_color_) mesh_integ_.colorMeshGPU(color_, &mesh_blocks_);
    refresh_block_cache(mesh_blocks_.getAllBlockIndices());
  } else if (!pending_mesh_.empty()) {
    // Incremental: re-mesh only blocks whose geometry or colour changed since the last
    // extraction; unchanged blocks stay cached on the host.
    std::vector<nvblox::Index3D> dirty(pending_mesh_.begin(), pending_mesh_.end());
    std::sort(dirty.begin(), dirty.end(), index_less);
    mesh_integ_.integrateBlocksGPU(tsdf_, dirty, &mesh_blocks_);
    if (any_color_) mesh_integ_.colorMeshGPU(color_, dirty, &mesh_blocks_);
    refresh_block_cache(dirty);
  }
  pending_mesh_.clear();

  assemble_mesh();
  last_mesh_ms_ = ms_since(t0);
  return mesh_;
}

void NvbloxSurfaceMap::refresh_block_cache(const std::vector<nvblox::Index3D>& blocks) {
  for (const nvblox::Index3D& idx : blocks) {
    const auto block = mesh_blocks_.getBlockAtIndex(idx);
    if (!block || block->vertices.empty()) {
      block_cache_.erase(idx);
      continue;
    }
    HostBlockMesh hm;
    const std::size_t nv = block->vertices.size();
    hm.vertices.assign(block->vertices.begin(), block->vertices.end());
    hm.normals.assign(block->normals.begin(), block->normals.end());
    if (block->colors.size() == nv) {
      hm.colors.reserve(nv);
      for (const nvblox::Color& c : block->colors) hm.colors.push_back(to_rgb(c));
    }
    hm.indices.reserve(block->triangles.size());
    for (const int t : block->triangles) hm.indices.push_back(static_cast<std::uint32_t>(t));
    block_cache_[idx] = std::move(hm);
  }
}

void NvbloxSurfaceMap::assemble_mesh() {
  mesh_.clear();

  std::vector<nvblox::Index3D> keys;
  keys.reserve(block_cache_.size());
  for (const auto& [k, v] : block_cache_) keys.push_back(k);
  std::sort(keys.begin(), keys.end(), index_less);

  std::size_t total_v = 0, total_i = 0;
  for (const nvblox::Index3D& k : keys) {
    const HostBlockMesh& hm = block_cache_.at(k);
    total_v += hm.vertices.size();
    total_i += hm.indices.size();
  }
  mesh_.vertices.reserve(total_v);
  mesh_.normals.reserve(total_v);
  mesh_.colors.reserve(total_v);
  mesh_.indices.reserve(total_i);

  for (const nvblox::Index3D& k : keys) {
    const HostBlockMesh& hm = block_cache_.at(k);
    const std::uint32_t base = static_cast<std::uint32_t>(mesh_.vertices.size());
    mesh_.vertices.insert(mesh_.vertices.end(), hm.vertices.begin(), hm.vertices.end());
    mesh_.normals.insert(mesh_.normals.end(), hm.normals.begin(), hm.normals.end());
    if (hm.colors.size() == hm.vertices.size()) {
      mesh_.colors.insert(mesh_.colors.end(), hm.colors.begin(), hm.colors.end());
    } else {
      // Block meshed before any colour was fused: neutral grey placeholder.
      mesh_.colors.insert(mesh_.colors.end(), hm.vertices.size(), {128, 128, 128});
    }
    for (const std::uint32_t i : hm.indices) mesh_.indices.push_back(base + i);
  }

  nvcu::sample_confidence(mesh_.vertices, tsdf_, conf_w_, &mesh_.confidence, *stream_);

  // Without a camera the fused colour layer never fills: colour by map-frame height,
  // matching the cpu backend's fallback so the two backends look alike in viz.
  if (!any_color_ && !mesh_.vertices.empty()) {
    float zmin = std::numeric_limits<float>::max();
    float zmax = std::numeric_limits<float>::lowest();
    for (const Eigen::Vector3f& v : mesh_.vertices) {
      zmin = std::min(zmin, v.z());
      zmax = std::max(zmax, v.z());
    }
    const float span = zmax - zmin;
    for (std::size_t i = 0; i < mesh_.vertices.size(); ++i) {
      const float norm = span > 1e-6f ? (mesh_.vertices[i].z() - zmin) / span : 0.5f;
      mesh_.colors[i] = turbo(norm);
    }
  }
}

MapDiagnostics NvbloxSurfaceMap::diagnostics() const {
  MapDiagnostics d;
  d.tsdf_blocks = block_count();
  d.last_integrate_ms = last_integrate_ms_;
  d.last_mesh_ms = last_mesh_ms_;
  return d;
}

}  // namespace meridian::map
