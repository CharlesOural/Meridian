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

// Exact piecewise sRGB transfer function. Colour is fused in linear-light space, so the
// 8-bit sampled pixel is linearised before blending and the blended value re-encoded.
float srgb_to_linear(std::uint8_t u) {
  const float c = static_cast<float>(u) / 255.f;
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
std::uint8_t linear_to_srgb(float lin) {
  lin = std::clamp(lin, 0.f, 1.f);
  const float s = lin <= 0.0031308f ? 12.92f * lin : 1.055f * std::pow(lin, 1.f / 2.4f) - 0.055f;
  return static_cast<std::uint8_t>(std::clamp(s, 0.f, 1.f) * 255.f + 0.5f);
}

// Forward lens distortion on normalized camera coordinates. The raw frame's pixels sit
// at the distorted projection, so a voxel must go through the same model before fx/cx
// are applied or its colour is sampled from the wrong pixel.
Eigen::Vector2f distort(const IntrinsicsCamera& intr, float xn, float yn) {
  const double x = xn;
  const double y = yn;
  const double r2 = x * x + y * y;
  switch (intr.model) {
    case IntrinsicsCamera::Distortion::None:
      break;
    case IntrinsicsCamera::Distortion::RadTan: {
      const double k1 = intr.coeffs[0], k2 = intr.coeffs[1], p1 = intr.coeffs[2],
                   p2 = intr.coeffs[3], k3 = intr.coeffs[4];
      const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
      const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
      const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
      return {static_cast<float>(xd), static_cast<float>(yd)};
    }
    case IntrinsicsCamera::Distortion::Equidistant: {
      const double r = std::sqrt(r2);
      const double theta = std::atan(r);
      const double t2 = theta * theta;
      const double theta_d =
          theta *
          (1.0 + t2 * (intr.coeffs[0] +
                       t2 * (intr.coeffs[1] + t2 * (intr.coeffs[2] + t2 * intr.coeffs[3]))));
      // r -> 0: theta_d/r -> 1, so the undistorted point passes through unchanged.
      const double scale = r > 1e-8 ? theta_d / r : 1.0;
      return {static_cast<float>(x * scale), static_cast<float>(y * scale)};
    }
  }
  return {xn, yn};
}

// Sampled sRGB pixel at integer (px, py); caller guarantees the coordinates are in-bounds.
// Bayer uses the 2x2-quad averages: full demosaicing quality is irrelevant at voxel scale.
std::array<std::uint8_t, 3> sample_pixel(const CameraFrame& img, int px, int py) {
  const std::uint8_t* src = img.data->data();
  const int w = img.width;
  const int h = img.height;
  switch (img.encoding) {
    case CameraFrame::Encoding::Mono8: {
      const std::uint8_t v = src[static_cast<std::size_t>(py) * w + px];
      return {v, v, v};
    }
    case CameraFrame::Encoding::RGB8: {
      const std::size_t i = static_cast<std::size_t>(py) * w + px;
      return {src[3 * i], src[3 * i + 1], src[3 * i + 2]};
    }
    case CameraFrame::Encoding::Bayer_RGGB8: {
      const int qx = px & ~1;
      const int qy = py & ~1;
      const int x1 = std::min(qx + 1, w - 1);
      const int y1 = std::min(qy + 1, h - 1);
      const std::uint8_t r = src[static_cast<std::size_t>(qy) * w + qx];
      const std::uint8_t g =
          static_cast<std::uint8_t>((static_cast<int>(src[static_cast<std::size_t>(qy) * w + x1]) +
                                     src[static_cast<std::size_t>(y1) * w + qx]) /
                                    2);
      const std::uint8_t b = src[static_cast<std::size_t>(y1) * w + x1];
      return {r, g, b};
    }
  }
  return {0, 0, 0};
}

}  // namespace

CpuSurfaceMap::CpuSurfaceMap(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib)
    : voxel_m_(static_cast<float>(cfg.tsdf_voxel_m)),
      inv_voxel_(cfg.tsdf_voxel_m > 0.0 ? 1.f / static_cast<float>(cfg.tsdf_voxel_m) : 20.f),
      trunc_(static_cast<float>(cfg.tsdf_voxel_m) * static_cast<float>(cfg.tsdf_trunc_voxels)),
      w_max_(static_cast<float>(cfg.tsdf_w_max)),
      color_alpha_(static_cast<float>(cfg.color_ewma_alpha)),
      color_enable_(cfg.colour),
      color_occlusion_(cfg.color_occlusion_check),
      max_dist_(static_cast<float>(cfg.tsdf_max_integration_dist_m)),
      conf_w_(static_cast<float>(cfg.mesh_conf_w)),
      calib_(std::move(calib)) {}

const TsdfVoxel* CpuSurfaceMap::find(const TsdfKey& k) const {
  const auto it = voxels_.find(k);
  return it == voxels_.end() ? nullptr : &it->second;
}

void CpuSurfaceMap::fuse(const TsdfKey& k, float sdf, std::vector<TsdfKey>* touched) {
  const bool fresh = voxels_.find(k) == voxels_.end();
  TsdfVoxel& v = voxels_[k];
  v.d = (v.w * v.d + sdf) / (v.w + 1.f);
  v.w = std::min(v.w + 1.f, w_max_);
  if (fresh && touched) touched->push_back(k);
}

bool CpuSurfaceMap::occluded(const Eigen::Vector3f& cam_o, const Eigen::Vector3f& x_map) const {
  const Eigen::Vector3f ray = x_map - cam_o;
  const float len = ray.norm();
  if (len < 1e-6f) return false;
  const Eigen::Vector3f dir = ray / len;
  const float step = 0.5f * voxel_m_;
  // March from one voxel out to ~1.5 voxels short of the target so the target's own
  // near-surface shell never counts as its occluder.
  const float stop = len - 1.5f * voxel_m_;
  for (float s = voxel_m_; s < stop; s += step) {
    const Eigen::Vector3f q = cam_o + dir * s;
    const TsdfKey qk{static_cast<std::int32_t>(std::floor(q.x() * inv_voxel_)),
                     static_cast<std::int32_t>(std::floor(q.y() * inv_voxel_)),
                     static_cast<std::int32_t>(std::floor(q.z() * inv_voxel_))};
    const TsdfVoxel* qv = find(qk);
    if (qv != nullptr && qv->w > 0.f && qv->d < 0.f) return true;
  }
  return false;
}

void CpuSurfaceMap::colour_voxels(const CameraFrame& image, const IntrinsicsCamera& intr,
                                  const Pose& T_map_cam, const std::vector<TsdfKey>& near_keys) {
  const int w = image.width;
  const int h = image.height;
  if (w <= 0 || h <= 0) return;
  const std::size_t need = static_cast<std::size_t>(w) * static_cast<std::size_t>(h) *
                           (image.encoding == CameraFrame::Encoding::RGB8 ? 3u : 1u);
  if (image.data->size() < need) return;

  const Pose T_cam_map = T_map_cam.inverse();
  const Eigen::Vector3f cam_o = T_map_cam.t.cast<float>();
  const float fx = static_cast<float>(intr.fx);
  const float fy = static_cast<float>(intr.fy);
  const float cx = static_cast<float>(intr.cx);
  const float cy = static_cast<float>(intr.cy);

  for (const TsdfKey& k : near_keys) {
    const auto it = voxels_.find(k);
    if (it == voxels_.end()) continue;
    const Eigen::Vector3f x_map((static_cast<float>(k.x) + 0.5f) * voxel_m_,
                                (static_cast<float>(k.y) + 0.5f) * voxel_m_,
                                (static_cast<float>(k.z) + 0.5f) * voxel_m_);
    const Eigen::Vector3f p_cam = (T_cam_map * x_map.cast<double>()).cast<float>();
    if (p_cam.z() <= 0.f) continue;
    const Eigen::Vector2f nd = distort(intr, p_cam.x() / p_cam.z(), p_cam.y() / p_cam.z());
    const float u = fx * nd.x() + cx;
    const float v = fy * nd.y() + cy;
    if (u < 0.f || u >= static_cast<float>(w) || v < 0.f || v >= static_cast<float>(h)) continue;
    if (color_occlusion_ && occluded(cam_o, x_map)) continue;

    const int px = std::clamp(static_cast<int>(std::lround(u)), 0, w - 1);
    const int py = std::clamp(static_cast<int>(std::lround(v)), 0, h - 1);
    const std::array<std::uint8_t, 3> rgb = sample_pixel(image, px, py);
    const float lr = srgb_to_linear(rgb[0]);
    const float lg = srgb_to_linear(rgb[1]);
    const float lb = srgb_to_linear(rgb[2]);

    TsdfVoxel& vx = it->second;
    if (vx.cw <= 0.f) {
      vx.cr = lr;
      vx.cg = lg;
      vx.cb = lb;
    } else {
      vx.cr = (1.f - color_alpha_) * vx.cr + color_alpha_ * lr;
      vx.cg = (1.f - color_alpha_) * vx.cg + color_alpha_ * lg;
      vx.cb = (1.f - color_alpha_) * vx.cb + color_alpha_ * lb;
    }
    vx.cw = std::min(vx.cw + 1.f, w_max_);
    any_color_ = true;
  }
}

void CpuSurfaceMap::integrate(std::uint64_t id, const PointCloud& cloud_body,
                              const std::shared_ptr<const CameraFrame>& image,
                              const Pose& T_map_body, const Pose& T_body_cam) {
  const auto t0 = Clock::now();
  const Eigen::Vector3f o = T_map_body.t.cast<float>();
  std::vector<TsdfKey> touched;
  std::unordered_set<TsdfKey, TsdfKeyHash> near_set;
  const int radius = std::max(1, static_cast<int>(std::ceil(trunc_ * inv_voxel_)));

  // Downsample the cloud to one representative per TSDF voxel (map frame) so fusion cost
  // scales with the occupied surface, not the raw return count. Without this a dense sweep
  // fuses tens of points into the same voxel shell, which is both redundant and far too
  // slow. The representative is the voxel-mean position.
  struct Acc {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
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
    ++a.n;
  }

  for (const auto& [kp, a] : ds) {
    const Eigen::Vector3f p = a.sum / static_cast<float>(a.n);
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
          fuse(k, sdf, &touched);
          if (dist < voxel_m_) near_set.insert(k);
        }
      }
    }
  }
  // Provenance: append (a keyframe may be re-integrated, so merge rather than overwrite).
  auto& owned = kf_voxels_[id];
  owned.insert(owned.end(), touched.begin(), touched.end());

  // Paint the near-surface voxels this keyframe touched from the camera image, once the
  // whole geometry pass has settled (the occlusion trace reads the fused surface).
  if (color_enable_ && image && image->data && calib_ != nullptr) {
    const auto ki = calib_->cam_intrinsics.find(image->sensor_id);
    if (ki != calib_->cam_intrinsics.end()) {
      const std::vector<TsdfKey> near_keys(near_set.begin(), near_set.end());
      colour_voxels(*image, ki->second, T_map_body * T_body_cam, near_keys);
    }
  }
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

Aabb CpuSurfaceMap::dirty_bounds(const std::vector<std::uint64_t>& ids) const {
  Aabb box;
  for (const std::uint64_t id : ids) {
    const auto it = kf_voxels_.find(id);
    if (it == kf_voxels_.end()) continue;
    for (const TsdfKey& k : it->second) {
      const Eigen::Vector3f lo(static_cast<float>(k.x) * voxel_m_,
                               static_cast<float>(k.y) * voxel_m_,
                               static_cast<float>(k.z) * voxel_m_);
      box.expand(lo);
      box.expand(lo + Eigen::Vector3f::Constant(voxel_m_));
    }
  }
  return box;
}

const ColorMesh& CpuSurfaceMap::extract_mesh() {
  const auto t0 = Clock::now();
  mesh_.clear();
  // Per-vertex height (fallback colour), edge-interpolated linear RGB, and colour weight.
  std::vector<float> vheight;
  std::vector<std::array<float, 3>> vrgb;
  std::vector<float> vcw;
  float cmin = std::numeric_limits<float>::max();
  float cmax = std::numeric_limits<float>::lowest();

  for (const auto& [k, v] : voxels_) {
    if (v.w <= 0.f) continue;
    // Gather the 8 cube corners (voxel centres of k + offset). Skip if any unobserved.
    std::array<Eigen::Vector3f, 8> cp;
    std::array<float, 8> cd{};
    std::array<float, 8> cr{};
    std::array<float, 8> cg{};
    std::array<float, 8> cb{};
    std::array<float, 8> ccw{};
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
      cr[i] = cv->cr;
      cg[i] = cv->cg;
      cb[i] = cv->cb;
      ccw[i] = cv->cw;
      cw[i] = cv->w;
    }
    if (!full) continue;

    int mask = 0;
    for (int i = 0; i < 8; ++i)
      if (cd[i] < 0.f) mask |= (1 << i);
    if (mc::edge_table[mask] == 0) continue;

    // Interpolate one vertex per crossed edge.
    std::array<Eigen::Vector3f, 12> ev;
    std::array<std::array<float, 3>, 12> ergb{};
    std::array<float, 12> ecw{};
    std::array<float, 12> ew{};
    for (int e = 0; e < 12; ++e) {
      if ((mc::edge_table[mask] & (1 << e)) == 0) continue;
      const int a = mc::edge_corners[e][0];
      const int b = mc::edge_corners[e][1];
      const float denom = cd[b] - cd[a];
      const float ti = std::abs(denom) < 1e-9f ? 0.5f : (-cd[a] / denom);
      ev[e] = cp[a] + ti * (cp[b] - cp[a]);
      // An unpainted corner stores zero RGB; blending against it would drag the edge
      // vertex toward black. With exactly one painted endpoint its colour is taken
      // verbatim; with neither, the RGB is never read (weight 0 -> neutral grey).
      const bool pa = ccw[a] > 0.f;
      const bool pb = ccw[b] > 0.f;
      if (pa && pb) {
        ergb[e] = {cr[a] + ti * (cr[b] - cr[a]), cg[a] + ti * (cg[b] - cg[a]),
                   cb[a] + ti * (cb[b] - cb[a])};
      } else if (pa) {
        ergb[e] = {cr[a], cg[a], cb[a]};
      } else if (pb) {
        ergb[e] = {cr[b], cg[b], cb[b]};
      }
      ecw[e] = ccw[a] + ti * (ccw[b] - ccw[a]);
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
        const int ei = es[j];
        vrgb.push_back(ergb[ei]);
        vcw.push_back(ecw[ei]);
        const float cval = ev[ei].z();
        vheight.push_back(cval);
        cmin = std::min(cmin, cval);
        cmax = std::max(cmax, cval);
        mesh_.indices.push_back(idx);
      }
    }
  }

  mesh_.colors.resize(vheight.size());
  if (!any_color_) {
    // No camera colour was ever fused: colour the whole mesh by map-frame height, a
    // gradient that is always meaningful regardless of what the LiDAR returns carry.
    const float span = cmax - cmin;
    for (std::size_t i = 0; i < vheight.size(); ++i) {
      const float norm = span > 1e-6f ? (vheight[i] - cmin) / span : 0.5f;
      mesh_.colors[i] = turbo(norm);
    }
  } else {
    // Camera colour exists somewhere: painted vertices take their fused linear RGB, and
    // vertices that never received colour stay a neutral grey rather than a stray hue.
    for (std::size_t i = 0; i < vheight.size(); ++i) {
      if (vcw[i] > 0.f) {
        mesh_.colors[i] = {linear_to_srgb(vrgb[i][0]), linear_to_srgb(vrgb[i][1]),
                           linear_to_srgb(vrgb[i][2])};
      } else {
        mesh_.colors[i] = {128, 128, 128};
      }
    }
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
