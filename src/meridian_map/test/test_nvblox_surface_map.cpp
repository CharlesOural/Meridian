#include <gtest/gtest.h>

#ifdef MERIDIAN_MAP_HAS_NVBLOX

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/cpu_surface_map.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/cloud.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/map/imap_layer.hpp"
#include "meridian/map/keyframe_store.hpp"
#include "nvblox/nvblox_surface_map.hpp"

using meridian::CalibrationSet;
using meridian::CameraFrame;
using meridian::ColorMesh;
using meridian::GraphUpdate;
using meridian::IMapLayer;
using meridian::IntrinsicsCamera;
using meridian::KeyframePacket;
using meridian::KeyframeStore;
using meridian::LidarPoint;
using meridian::makeMapLayer;
using meridian::MapBackend;
using meridian::MapConfig;
using meridian::PointCloud;
using meridian::Pose;
using meridian::map::CpuSurfaceMap;
using meridian::map::NvbloxSurfaceMap;

namespace {

// A dense unit sphere of LiDAR returns (Fibonacci directions), observed from the centre.
// Denser than the cpu test: the projective backend rasterizes into a 64 x 1024 range
// image, so the sweep must fill most bins for gap-free interpolation.
meridian::PointCloudPtr unit_sphere(int n, float r) {
  auto cl = std::make_shared<PointCloud>();
  const float golden = static_cast<float>(M_PI) * (3.f - std::sqrt(5.f));
  for (int i = 0; i < n; ++i) {
    const float z = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
    const float rad = std::sqrt(std::max(0.f, 1.f - z * z));
    const float theta = golden * static_cast<float>(i);
    LidarPoint p;
    p.xyz = Eigen::Vector3f(r * rad * std::cos(theta), r * rad * std::sin(theta), r * z);
    p.intensity = 0.5f * (z + 1.f) * 200.f;
    cl->push_back(p);
  }
  return cl;
}

constexpr int kSphereN = 200000;

// The whole suite needs a CUDA device; on a toolkit-only host skip rather than abort.
bool gpu_present() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
#define REQUIRE_GPU() \
  if (!gpu_present()) GTEST_SKIP() << "no CUDA device visible"

MapConfig surface_cfg() {
  MapConfig c;
  c.tsdf_voxel_m = 0.1;
  c.tsdf_trunc_voxels = 4;
  c.tsdf_max_integration_dist_m = 50.0;
  return c;
}

struct RadiusStats {
  double mean = 0.0;
  float maxdev = 0.f;
};
RadiusStats radius_stats(const ColorMesh& mesh, float R) {
  RadiusStats s;
  double sum = 0.0;
  for (const auto& v : mesh.vertices) {
    const float rr = v.norm();
    sum += rr;
    s.maxdev = std::max(s.maxdev, std::abs(rr - R));
  }
  s.mean = sum / static_cast<double>(mesh.vertices.size());
  return s;
}

// A vertical wall: a grid of returns on the plane x = wall_x, seen from `origin` (the
// integration pose translates the cloud, so points are given in the sensor frame).
meridian::PointCloudPtr wall_cloud(float wall_x, float half_extent, float step,
                                   const Eigen::Vector3f& origin) {
  auto cl = std::make_shared<PointCloud>();
  for (float y = -half_extent; y <= half_extent; y += step) {
    for (float z = -half_extent; z <= half_extent; z += step) {
      LidarPoint p;
      p.xyz = Eigen::Vector3f(wall_x, y, z) - origin;
      p.intensity = 100.f;
      cl->push_back(p);
    }
  }
  return cl;
}

Pose pose_at(const Eigen::Vector3f& t) {
  return Pose{Eigen::Quaterniond::Identity(), t.cast<double>()};
}

// Camera looking along body +x: camera z (optical axis) -> body x, camera x -> body -y,
// camera y -> body -z.
Pose camera_facing_x() {
  Eigen::Matrix3d R;
  R.col(0) = Eigen::Vector3d(0, -1, 0);  // cam x in body
  R.col(1) = Eigen::Vector3d(0, 0, -1);  // cam y in body
  R.col(2) = Eigen::Vector3d(1, 0, 0);   // cam z in body
  return Pose{Eigen::Quaterniond(R), Eigen::Vector3d::Zero()};
}

std::shared_ptr<const CalibrationSet> camera_calib(int w, int h, double f) {
  auto cs = std::make_shared<CalibrationSet>();
  IntrinsicsCamera K;
  K.fx = f;
  K.fy = f;
  K.cx = w / 2.0;
  K.cy = h / 2.0;
  K.width = w;
  K.height = h;
  K.model = IntrinsicsCamera::Distortion::None;
  cs->cam_intrinsics[0] = K;
  return cs;
}

std::shared_ptr<const CameraFrame> uniform_rgb_frame(int w, int h, std::uint8_t r, std::uint8_t g,
                                                     std::uint8_t b) {
  auto data = std::make_shared<std::vector<std::uint8_t>>();
  data->reserve(static_cast<std::size_t>(w) * h * 3);
  for (int i = 0; i < w * h; ++i) {
    data->push_back(r);
    data->push_back(g);
    data->push_back(b);
  }
  auto f = std::make_shared<CameraFrame>();
  f->sensor_id = 0;
  f->width = w;
  f->height = h;
  f->encoding = CameraFrame::Encoding::RGB8;
  f->data = data;
  return f;
}

}  // namespace

TEST(NvbloxSurfaceMap, MeshesSphereSurface) {
  REQUIRE_GPU();
  NvbloxSurfaceMap m(surface_cfg(), nullptr);
  const float R = 1.0f;
  // Several sweeps: per-sweep observation weight is 1, so fused weight (and with it
  // vertex confidence) accumulates once per keyframe.
  const auto cloud = unit_sphere(kSphereN, R);
  for (std::uint64_t id = 1; id <= 4; ++id) {
    m.integrate(id, *cloud, nullptr, Pose{}, Pose{});
  }

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);
  ASSERT_EQ(mesh.indices.size() % 3, 0u);
  ASSERT_EQ(mesh.colors.size(), mesh.vertices.size());
  ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());
  ASSERT_EQ(mesh.confidence.size(), mesh.vertices.size());

  const RadiusStats s = radius_stats(mesh, R);
  EXPECT_NEAR(s.mean, R, 0.12);
  EXPECT_LT(s.maxdev, 0.5f);

  // Four sweeps at unit weight against W_conf = 8: confidence reaches 4/8 somewhere.
  float cmax = 0.f;
  for (const float c : mesh.confidence) cmax = std::max(cmax, c);
  EXPECT_NEAR(cmax, 0.5f, 0.01f);
}

TEST(NvbloxSurfaceMap, ToleranceEquivalentToCpuBackend) {
  REQUIRE_GPU();
  const float R = 1.0f;
  const auto cloud = unit_sphere(kSphereN, R);

  CpuSurfaceMap cpu(surface_cfg());
  cpu.integrate(1, *cloud, nullptr, Pose{}, Pose{});
  NvbloxSurfaceMap gpu(surface_cfg(), nullptr);
  gpu.integrate(1, *cloud, nullptr, Pose{}, Pose{});

  const RadiusStats sc = radius_stats(cpu.extract_mesh(), R);
  const RadiusStats sg = radius_stats(gpu.extract_mesh(), R);
  // Backends are tolerance-equivalent against the analytic surface, never bit-compared.
  EXPECT_NEAR(sc.mean, R, 0.12);
  EXPECT_NEAR(sg.mean, R, 0.12);
  EXPECT_NEAR(sc.mean, sg.mean, 0.06);
  EXPECT_LT(sg.maxdev, 0.5f);
}

TEST(NvbloxSurfaceMap, ColoursVaryWithHeightFallback) {
  REQUIRE_GPU();
  NvbloxSurfaceMap m(surface_cfg(), nullptr);
  m.integrate(1, *unit_sphere(kSphereN, 1.0f), nullptr, Pose{}, Pose{});
  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.colors.size(), 0u);
  bool varies = false;
  for (const auto& col : mesh.colors) {
    if (col != mesh.colors.front()) {
      varies = true;
      break;
    }
  }
  EXPECT_TRUE(varies);
}

TEST(NvbloxSurfaceMap, CameraColoursSurfaceAndOcclusionBlocksBleed) {
  REQUIRE_GPU();
  const int w = 640, h = 480;
  NvbloxSurfaceMap m(surface_cfg(), camera_calib(w, h, 300.0));

  // Far wall first (seen from between the walls, so it has geometry), then the near
  // wall from the origin with a uniformly red camera frame. The camera at the origin
  // sees only the near wall; the far wall sits in its frustum but behind the near
  // surface, so the sphere-traced colour pass must leave it unpainted.
  // Dense grids: the walls must fill the 64 x 1024 range-image bins or the projective
  // interpolation gates leave holes.
  const Eigen::Vector3f between(3.f, 0.f, 0.f);
  m.integrate(1, *wall_cloud(4.f, 2.0f, 0.01f, between), nullptr, pose_at(between), Pose{});
  m.integrate(2, *wall_cloud(2.f, 1.0f, 0.006f, Eigen::Vector3f::Zero()),
              uniform_rgb_frame(w, h, 220, 30, 30), Pose{}, camera_facing_x());

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);

  int near_red = 0, near_total = 0, far_red = 0, far_total = 0;
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    const auto& c = mesh.colors[i];
    const bool red = c[0] > 150 && c[1] < 100 && c[2] < 100;
    // Central patches only: wall borders straddle voxels and camera frustum edges.
    if (std::abs(v.x() - 2.f) < 0.15f && std::abs(v.y()) < 0.5f && std::abs(v.z()) < 0.5f) {
      ++near_total;
      near_red += red;
    } else if (std::abs(v.x() - 4.f) < 0.15f && std::abs(v.y()) < 0.5f && std::abs(v.z()) < 0.5f) {
      ++far_total;
      far_red += red;
    }
  }
  ASSERT_GT(near_total, 20);
  ASSERT_GT(far_total, 20);
  // The visible wall takes the camera colour; the occluded wall must not.
  EXPECT_GT(static_cast<double>(near_red) / near_total, 0.8);
  EXPECT_LT(static_cast<double>(far_red) / far_total, 0.05);
}

TEST(NvbloxSurfaceMap, CameraColoursSurfaceBeyondSphereTracerDefaultRange) {
  REQUIRE_GPU();
  const int w = 640, h = 480;
  NvbloxSurfaceMap m(surface_cfg(), camera_calib(w, h, 300.0));

  // A wall 12 m out — beyond the occlusion tracer's 7 m construction default, within
  // the configured 50 m cutoff. It must still take the camera colour.
  m.integrate(1, *wall_cloud(12.f, 3.0f, 0.05f, Eigen::Vector3f::Zero()),
              uniform_rgb_frame(w, h, 220, 30, 30), Pose{}, camera_facing_x());

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);
  int red = 0, total = 0;
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    if (std::abs(v.x() - 12.f) < 0.3f && std::abs(v.y()) < 1.5f && std::abs(v.z()) < 1.5f) {
      ++total;
      const auto& c = mesh.colors[i];
      red += (c[0] > 150 && c[1] < 100 && c[2] < 100);
    }
  }
  ASSERT_GT(total, 20);
  EXPECT_GT(static_cast<double>(red) / total, 0.8);
}

TEST(NvbloxSurfaceMap, ClearKeyframeEmptiesGrid) {
  REQUIRE_GPU();
  NvbloxSurfaceMap m(surface_cfg(), nullptr);
  m.integrate(7, *unit_sphere(kSphereN / 4, 1.0f), nullptr, Pose{}, Pose{});
  EXPECT_GT(m.block_count(), 0u);
  m.clear_keyframes({7});
  EXPECT_EQ(m.block_count(), 0u);
  EXPECT_TRUE(m.extract_mesh().vertices.empty());
}

TEST(NvbloxSurfaceMap, DeterministicAcrossReplays) {
  REQUIRE_GPU();
  const auto cloud_a = unit_sphere(kSphereN / 2, 1.0f);
  const auto cloud_b = wall_cloud(3.f, 1.5f, 0.03f, Eigen::Vector3f::Zero());

  auto run = [&](ColorMesh* out) {
    NvbloxSurfaceMap m(surface_cfg(), nullptr);
    m.integrate(1, *cloud_a, nullptr, Pose{}, Pose{});
    m.integrate(2, *cloud_b, nullptr, pose_at(Eigen::Vector3f(0.5f, 0.f, 0.f)), Pose{});
    *out = m.extract_mesh();  // copy
  };
  ColorMesh m1, m2;
  run(&m1);
  run(&m2);

  // Geometric determinism: same counts and same vertex positions every replay. Index
  // arrays are NOT bit-stable — vertex welding collapses coincident vertices and the
  // survivor choice varies with GPU reduction ULPs — so connectivity is compared by
  // triangle count only.
  ASSERT_EQ(m1.vertices.size(), m2.vertices.size());
  ASSERT_EQ(m1.indices.size(), m2.indices.size());
  float max_d = 0.f;
  for (std::size_t i = 0; i < m1.vertices.size(); ++i) {
    max_d = std::max(max_d, (m1.vertices[i] - m2.vertices[i]).norm());
  }
  EXPECT_LT(max_d, 1e-5f);
}

namespace {

// A dense floor patch 2 m below the sensor, mirroring the LayeredMap cpu test but with
// enough returns to fill the projective backend's range-image bins.
constexpr float kFloorZ = -2.0f;
meridian::PointCloudPtr floor_patch() {
  auto cl = std::make_shared<PointCloud>();
  for (int i = 0; i < 140; ++i) {
    for (int j = 0; j < 140; ++j) {
      LidarPoint p;
      p.xyz = Eigen::Vector3f(-0.7f + 0.01f * static_cast<float>(i),
                              -0.7f + 0.01f * static_cast<float>(j), kFloorZ);
      p.intensity = 50.f;
      cl->push_back(p);
    }
  }
  return cl;
}

KeyframePacket make_kf(std::uint64_t id, const meridian::PointCloudPtr& cloud) {
  KeyframePacket p;
  p.id = id;
  p.cloud_body = cloud;
  return p;
}

IMapLayer::Ptr build_nvblox_layer(std::shared_ptr<KeyframeStore> store) {
  MapConfig c;
  c.backend = MapBackend::Nvblox;
  c.reg_voxel_m = 1.0;
  c.reg_min_plane_pts = 5;
  c.tsdf_voxel_m = 0.2;
  c.tsdf_trunc_voxels = 3;
  c.tsdf_max_integration_dist_m = 100.0;
  return makeMapLayer(c, std::shared_ptr<const CalibrationSet>{}, std::move(store), nullptr);
}

Pose at_xyz(double x, double y, double z) {
  return Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(x, y, z)};
}

}  // namespace

// The clear-and-rebuild invariant (spec 06 §7.4) through the façade with the GPU backend:
// moving a keyframe and rebuilding equals integrating the corrected poses from scratch.
TEST(NvbloxLayeredMap, LoopRebuildEqualsFreshIntegrate) {
  REQUIRE_GPU();
  auto store_a = std::make_shared<KeyframeStore>();
  auto a = build_nvblox_layer(store_a);
  a->integrate(make_kf(1, floor_patch()), at_xyz(0, 0, 0));
  a->integrate(make_kf(2, floor_patch()), at_xyz(5, 0, 0));
  a->integrate(make_kf(3, floor_patch()), at_xyz(20, 0, 0));

  GraphUpdate up;
  up.loop_closed = true;
  up.moved.push_back({2, at_xyz(5, 0, 2), std::nullopt});
  a->apply_graph_update(up);

  auto store_b = std::make_shared<KeyframeStore>();
  auto b = build_nvblox_layer(store_b);
  b->integrate(make_kf(1, floor_patch()), at_xyz(0, 0, 0));
  b->integrate(make_kf(2, floor_patch()), at_xyz(5, 0, 2));
  b->integrate(make_kf(3, floor_patch()), at_xyz(20, 0, 0));

  const auto da = a->diagnostics();
  const auto db = b->diagnostics();
  ASSERT_GT(da.tsdf_blocks, 0u);
  EXPECT_EQ(da.tsdf_blocks, db.tsdf_blocks);
  EXPECT_EQ(da.reg_voxels, db.reg_voxels);

  EXPECT_TRUE(a->query_plane(Eigen::Vector3f(5, 0, kFloorZ + 2.f)).has_value());
  EXPECT_FALSE(a->query_plane(Eigen::Vector3f(5, 0, kFloorZ)).has_value());
  EXPECT_GT(a->extract_mesh().vertices.size(), 0u);
}

#else
#include <gtest/gtest.h>
TEST(NvbloxSurfaceMap, SkippedWithoutNvblox) {
  REQUIRE_GPU();
  GTEST_SKIP();
}
#endif  // MERIDIAN_MAP_HAS_NVBLOX
