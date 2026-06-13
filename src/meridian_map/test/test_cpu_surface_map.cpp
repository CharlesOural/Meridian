#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "cpu/cpu_surface_map.hpp"
#include "meridian/common/cloud.hpp"

using meridian::LidarPoint;
using meridian::MapConfig;
using meridian::PointCloud;
using meridian::Pose;
using meridian::map::CpuSurfaceMap;

namespace {

// A dense unit sphere of LiDAR returns (Fibonacci directions), observed from the centre.
meridian::PointCloudPtr unit_sphere(int n, float r) {
  auto cl = std::make_shared<PointCloud>();
  const float golden = static_cast<float>(M_PI) * (3.f - std::sqrt(5.f));
  for (int i = 0; i < n; ++i) {
    const float z = 1.f - 2.f * (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
    const float rad = std::sqrt(std::max(0.f, 1.f - z * z));
    const float theta = golden * static_cast<float>(i);
    LidarPoint p;
    p.xyz = Eigen::Vector3f(r * rad * std::cos(theta), r * rad * std::sin(theta), r * z);
    p.intensity = 0.5f * (z + 1.f) * 200.f;  // height-graded intensity
    cl->push_back(p);
  }
  return cl;
}

}  // namespace

TEST(CpuSurfaceMap, MeshesSphereSurface) {
  MapConfig c;
  c.tsdf_voxel_m = 0.1;
  c.tsdf_trunc_voxels = 4;
  c.tsdf_max_integration_dist_m = 50.0;
  CpuSurfaceMap m(c);
  const float R = 1.0f;
  m.integrate(1, *unit_sphere(20000, R), nullptr, Pose{}, Pose{});

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);
  ASSERT_EQ(mesh.indices.size() % 3, 0u);
  ASSERT_EQ(mesh.colors.size(), mesh.vertices.size());
  ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());

  // Every vertex lies near the sphere; the mean radius matches R within a voxel.
  double sum = 0.0;
  float maxdev = 0.f;
  for (const auto& v : mesh.vertices) {
    const float rr = v.norm();
    sum += rr;
    maxdev = std::max(maxdev, std::abs(rr - R));
  }
  const double mean_r = sum / static_cast<double>(mesh.vertices.size());
  EXPECT_NEAR(mean_r, R, 0.12);  // coarse-voxel sphere; small outward fusion bias
  EXPECT_LT(maxdev, 0.5f);       // within ~one truncation band of the surface
}

TEST(CpuSurfaceMap, ColoursVary) {
  MapConfig c;
  c.tsdf_voxel_m = 0.1;
  c.tsdf_trunc_voxels = 4;
  c.tsdf_max_integration_dist_m = 50.0;
  CpuSurfaceMap m(c);
  m.integrate(1, *unit_sphere(20000, 1.0f), nullptr, Pose{}, Pose{});
  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.colors.size(), 0u);
  // Height-graded intensity -> the turbo colours must not be uniform.
  bool varies = false;
  for (const auto& col : mesh.colors) {
    if (col != mesh.colors.front()) {
      varies = true;
      break;
    }
  }
  EXPECT_TRUE(varies);
}

TEST(CpuSurfaceMap, ClearKeyframeEmptiesGrid) {
  MapConfig c;
  c.tsdf_voxel_m = 0.1;
  c.tsdf_trunc_voxels = 4;
  c.tsdf_max_integration_dist_m = 50.0;
  CpuSurfaceMap m(c);
  m.integrate(7, *unit_sphere(5000, 1.0f), nullptr, Pose{}, Pose{});
  EXPECT_GT(m.voxel_count(), 0u);
  m.clear_keyframes({7});
  EXPECT_EQ(m.voxel_count(), 0u);
  EXPECT_TRUE(m.extract_mesh().vertices.empty());
}
