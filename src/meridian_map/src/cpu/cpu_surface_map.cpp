#include "cpu/cpu_surface_map.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>

#include "cpu/marching_cubes_tables.hpp"

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

}  // namespace

CpuSurfaceMap::CpuSurfaceMap(const MapConfig& cfg)
    : voxel_m_(static_cast<float>(cfg.tsdf_voxel_m)),
      inv_voxel_(cfg.tsdf_voxel_m > 0.0 ? 1.f / static_cast<float>(cfg.tsdf_voxel_m) : 20.f),
      trunc_(static_cast<float>(cfg.tsdf_voxel_m) * static_cast<float>(cfg.tsdf_trunc_voxels)),
      w_max_(static_cast<float>(cfg.tsdf_w_max)),
      color_alpha_(static_cast<float>(cfg.color_ewma_alpha)),
      color_enable_(cfg.colour),
      max_dist_(static_cast<float>(cfg.tsdf_max_integration_dist_m)),
      conf_w_(static_cast<float>(cfg.mesh_conf_w)) {}

const TsdfVoxel* CpuSurfaceMap::find(const TsdfKey& k) const {
  const auto it = voxels_.find(k);
  return it == voxels_.end() ? nullptr : &it->second;
}

void CpuSurfaceMap::fuse(const TsdfKey& k, float sdf, float color, bool near_surface,
                         std::vector<TsdfKey>* touched) {
  const bool fresh = voxels_.find(k) == voxels_.end();
  TsdfVoxel& v = voxels_[k];
  v.d = (v.w * v.d + sdf) / (v.w + 1.f);
  v.w = std::min(v.w + 1.f, w_max_);
  if (color_enable_ && near_surface) {
    if (v.cw <= 0.f) {
      v.c = color;
    } else {
      v.c = (1.f - color_alpha_) * v.c + color_alpha_ * color;
    }
    v.cw = std::min(v.cw + 1.f, w_max_);
  }
  if (fresh && touched) touched->push_back(k);
}

void CpuSurfaceMap::integrate(std::uint64_t id, const PointCloud& cloud_body,
                              const std::shared_ptr<const CameraFrame>& /*image*/,
                              const Pose& T_map_body, const Pose& /*T_body_cam*/) {
  const auto t0 = Clock::now();
  const Eigen::Vector3f o = T_map_body.t.cast<float>();
  std::vector<TsdfKey> touched;
  const int radius = std::max(1, static_cast<int>(std::ceil(trunc_ * inv_voxel_)));

  // Downsample the cloud to one representative per TSDF voxel (map frame) so fusion cost
  // scales with the occupied surface, not the raw return count. Without this a dense sweep
  // fuses tens of points into the same voxel shell, which is both redundant and far too
  // slow. The representative is the voxel-mean position + mean intensity.
  struct Acc {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    float isum = 0.f;
    int n = 0;
  };
  std::unordered_map<TsdfKey, Acc, TsdfKeyHash> ds;
  ds.reserve(cloud_body.size());
  for (const LidarPoint& lp : cloud_body) {
    const Eigen::Vector3f p = (T_map_body * lp.xyz.cast<double>()).cast<float>();
    if ((p - o).norm() > max_dist_) continue;
    const TsdfKey k{static_cast<std::int32_t>(std::floor(p.x() * inv_voxel_)),
                    static_cast<std::int32_t>(std::floor(p.y() * inv_voxel_)),
                    static_cast<std::int32_t>(std::floor(p.z() * inv_voxel_))};
    Acc& a = ds[k];
    a.sum += p;
    a.isum += lp.intensity;
    ++a.n;
  }

  for (const auto& [kp, a] : ds) {
    const Eigen::Vector3f p = a.sum / static_cast<float>(a.n);
    const float intensity = a.isum / static_cast<float>(a.n);
    const Eigen::Vector3f to_sensor = o - p;
    const float rho = to_sensor.norm();
    if (rho < 1e-3f) continue;

    // Fuse a truncation-radius shell of voxels around the surface point. The signed
    // distance is the voxel-to-point distance, signed by which side of the surface the
    // voxel is on (positive toward the sensor = free space, negative behind). This
    // thickens the surface to ~2*trunc so Marching Cubes finds complete cubes, while the
    // lateral extent stays bounded by trunc (no projective long-range smear).
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        for (int dz = -radius; dz <= radius; ++dz) {
          const TsdfKey k{kp.x + dx, kp.y + dy, kp.z + dz};
          const Eigen::Vector3f x((static_cast<float>(k.x) + 0.5f) * voxel_m_,
                                  (static_cast<float>(k.y) + 0.5f) * voxel_m_,
                                  (static_cast<float>(k.z) + 0.5f) * voxel_m_);
          const Eigen::Vector3f d = x - p;
          const float dist = d.norm();
          if (dist > trunc_) continue;
          const float sdf = d.dot(to_sensor) >= 0.f ? dist : -dist;
          const bool near = dist < voxel_m_;
          fuse(k, sdf, intensity, near, &touched);
        }
      }
    }
  }
  // Provenance: append (a keyframe may be re-integrated, so merge rather than overwrite).
  auto& owned = kf_voxels_[id];
  owned.insert(owned.end(), touched.begin(), touched.end());
  last_integrate_ms_ = ms_since(t0);
}

void CpuSurfaceMap::clear_keyframes(const std::vector<std::uint64_t>& ids) {
  for (std::uint64_t id : ids) {
    const auto it = kf_voxels_.find(id);
    if (it == kf_voxels_.end()) continue;
    for (const TsdfKey& k : it->second) voxels_.erase(k);
    kf_voxels_.erase(it);
  }
}

const ColorMesh& CpuSurfaceMap::extract_mesh() {
  const auto t0 = Clock::now();
  mesh_.clear();
  std::vector<float> cscalar;
  float cmin = std::numeric_limits<float>::max();
  float cmax = std::numeric_limits<float>::lowest();

  for (const auto& [k, v] : voxels_) {
    if (v.w <= 0.f) continue;
    // Gather the 8 cube corners (voxel centres of k + offset). Skip if any unobserved.
    std::array<Eigen::Vector3f, 8> cp;
    std::array<float, 8> cd{};
    std::array<float, 8> cc{};
    std::array<float, 8> cw{};
    bool full = true;
    for (int i = 0; i < 8; ++i) {
      const TsdfKey ck{k.x + mc::corner_offset[i][0], k.y + mc::corner_offset[i][1],
                       k.z + mc::corner_offset[i][2]};
      const TsdfVoxel* cv = find(ck);
      if (cv == nullptr || cv->w <= 0.f) {
        full = false;
        break;
      }
      cp[i] = Eigen::Vector3f((static_cast<float>(ck.x) + 0.5f) * voxel_m_,
                              (static_cast<float>(ck.y) + 0.5f) * voxel_m_,
                              (static_cast<float>(ck.z) + 0.5f) * voxel_m_);
      cd[i] = cv->d;
      cc[i] = cv->c;
      cw[i] = cv->w;
    }
    if (!full) continue;

    int mask = 0;
    for (int i = 0; i < 8; ++i)
      if (cd[i] < 0.f) mask |= (1 << i);
    if (mc::edge_table[mask] == 0) continue;

    // Interpolate one vertex per crossed edge.
    std::array<Eigen::Vector3f, 12> ev;
    std::array<float, 12> ec{};
    std::array<float, 12> ew{};
    for (int e = 0; e < 12; ++e) {
      if ((mc::edge_table[mask] & (1 << e)) == 0) continue;
      const int a = mc::edge_corners[e][0];
      const int b = mc::edge_corners[e][1];
      const float denom = cd[b] - cd[a];
      const float ti = std::abs(denom) < 1e-9f ? 0.5f : (-cd[a] / denom);
      ev[e] = cp[a] + ti * (cp[b] - cp[a]);
      ec[e] = cc[a] + ti * (cc[b] - cc[a]);
      ew[e] = cw[a] + ti * (cw[b] - cw[a]);
    }

    const auto& tri = mc::tri_table[mask];
    for (int i = 0; tri[i] >= 0; i += 3) {
      const int e0 = tri[i], e1 = tri[i + 1], e2 = tri[i + 2];
      const Eigen::Vector3f& p0 = ev[e0];
      const Eigen::Vector3f& p1 = ev[e1];
      const Eigen::Vector3f& p2 = ev[e2];
      Eigen::Vector3f nrm = (p1 - p0).cross(p2 - p0);
      const float nl = nrm.norm();
      nrm = nl > 1e-12f ? (nrm / nl) : Eigen::Vector3f(0, 0, 1);
      const std::array<int, 3> es{e0, e1, e2};
      for (int j = 0; j < 3; ++j) {
        const std::uint32_t idx = static_cast<std::uint32_t>(mesh_.vertices.size());
        mesh_.vertices.push_back(ev[es[j]]);
        mesh_.normals.push_back(nrm);
        mesh_.confidence.push_back(std::clamp(ew[es[j]] / conf_w_, 0.f, 1.f));
        // Colour by map-frame height: a gradient that is always meaningful regardless of
        // whether the LiDAR returns carry usable intensity. (Fused intensity ec[] is kept
        // for a future camera/intensity colour mode.)
        const float cval = ev[es[j]].z();
        cscalar.push_back(cval);
        cmin = std::min(cmin, cval);
        cmax = std::max(cmax, cval);
        mesh_.indices.push_back(idx);
      }
    }
  }

  // Colour pass: normalise the fused scalar over the mesh's own range -> turbo RGB.
  const float span = cmax - cmin;
  mesh_.colors.resize(cscalar.size());
  for (std::size_t i = 0; i < cscalar.size(); ++i) {
    const float norm = span > 1e-6f ? (cscalar[i] - cmin) / span : 0.5f;
    mesh_.colors[i] = turbo(norm);
  }
  last_mesh_ms_ = ms_since(t0);
  return mesh_;
}

MapDiagnostics CpuSurfaceMap::diagnostics() const {
  MapDiagnostics d;
  d.tsdf_blocks = voxels_.size();
  d.last_integrate_ms = last_integrate_ms_;
  d.last_mesh_ms = last_mesh_ms_;
  return d;
}

}  // namespace meridian::map
