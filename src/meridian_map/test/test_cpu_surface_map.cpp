#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/cpu_surface_map.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/cloud.hpp"

using meridian::CalibrationSet;
using meridian::CameraFrame;
using meridian::IntrinsicsCamera;
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

namespace {

MapConfig surface_cfg() {
  MapConfig c;
  c.tsdf_voxel_m = 0.1;
  c.tsdf_trunc_voxels = 4;
  c.tsdf_max_integration_dist_m = 50.0;
  return c;
}

Pose pose_at(const Eigen::Vector3f& t) {
  return Pose{Eigen::Quaterniond::Identity(), t.cast<double>()};
}

// Camera looking along body +x: camera z (optical axis) -> body x, camera x -> body -y,
// camera y -> body -z.
Pose camera_facing_x() {
  Eigen::Matrix3d R;
  R.col(0) = Eigen::Vector3d(0, -1, 0);
  R.col(1) = Eigen::Vector3d(0, 0, -1);
  R.col(2) = Eigen::Vector3d(1, 0, 0);
  return Pose{Eigen::Quaterniond(R), Eigen::Vector3d::Zero()};
}

std::shared_ptr<const CalibrationSet> camera_calib(
    int w, int h, double f, IntrinsicsCamera::Distortion model = IntrinsicsCamera::Distortion::None,
    const std::array<double, 5>& coeffs = {0, 0, 0, 0, 0}) {
  auto cs = std::make_shared<CalibrationSet>();
  IntrinsicsCamera K;
  K.fx = f;
  K.fy = f;
  K.cx = w / 2.0;
  K.cy = h / 2.0;
  K.width = w;
  K.height = h;
  K.model = model;
  K.coeffs = coeffs;
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

// A vertical wall: a grid of returns on the plane x = wall_x, given in the sensor frame
// (the integration pose translates the cloud into the map).
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

}  // namespace

TEST(CpuSurfaceMap, CameraColoursSurface) {
  const int w = 640, h = 480;
  CpuSurfaceMap m(surface_cfg(), camera_calib(w, h, 300.0));
  // A wall 2 m ahead, painted by a uniformly red frame from a camera at the origin.
  m.integrate(1, *wall_cloud(2.f, 1.0f, 0.02f, Eigen::Vector3f::Zero()),
              uniform_rgb_frame(w, h, 220, 30, 30), Pose{}, camera_facing_x());

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);
  ASSERT_EQ(mesh.colors.size(), mesh.vertices.size());

  int red = 0, total = 0;
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    if (std::abs(v.y()) < 0.5f && std::abs(v.z()) < 0.5f) {
      ++total;
      const auto& c = mesh.colors[i];
      red += (c[0] > 150 && c[1] < 100 && c[2] < 100);
    }
  }
  ASSERT_GT(total, 20);
  EXPECT_GT(static_cast<double>(red) / total, 0.8);
}

TEST(CpuSurfaceMap, OcclusionBlocksBleed) {
  const int w = 640, h = 480;
  CpuSurfaceMap m(surface_cfg(), camera_calib(w, h, 300.0));

  // Far wall first (seen from between the walls, so it has geometry) without an image,
  // then the near wall from the origin with a uniformly red frame. Only the near wall's
  // own voxels are candidates for this keyframe's colour pass, and the sphere trace keeps
  // the near surface from bleeding onto anything behind it.
  const Eigen::Vector3f between(3.f, 0.f, 0.f);
  m.integrate(1, *wall_cloud(4.f, 2.0f, 0.02f, between), nullptr, pose_at(between), Pose{});
  m.integrate(2, *wall_cloud(2.f, 1.0f, 0.02f, Eigen::Vector3f::Zero()),
              uniform_rgb_frame(w, h, 220, 30, 30), Pose{}, camera_facing_x());

  const auto& mesh = m.extract_mesh();
  ASSERT_GT(mesh.vertices.size(), 100u);

  int near_red = 0, near_total = 0, far_red = 0, far_total = 0;
  for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto& v = mesh.vertices[i];
    const auto& c = mesh.colors[i];
    const bool red = c[0] > 150 && c[1] < 100 && c[2] < 100;
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
  EXPECT_GT(static_cast<double>(near_red) / near_total, 0.8);
  EXPECT_LT(static_cast<double>(far_red) / far_total, 0.05);
}

TEST(CpuSurfaceMap, DistortionMovesColourProjection) {
  const int w = 640, h = 480;
  // With fx = 300 the pinhole frame edge sits at |xn| ~ 1.067; a strong barrel term
  // (k1 = 0.5) pushes |xn| ~ 0.8 out to that edge, so wall points with |y| in roughly
  // (1.7, 2.0) at x = 2 project in-bounds pinhole but out of bounds once distorted.
  // The distorted camera must leave that outer band unpainted while a pinhole control
  // paints it.
  const auto cloud = wall_cloud(2.f, 2.0f, 0.02f, Eigen::Vector3f::Zero());
  const auto image = uniform_rgb_frame(w, h, 220, 30, 30);

  CpuSurfaceMap distorted(
      surface_cfg(),
      camera_calib(w, h, 300.0, IntrinsicsCamera::Distortion::RadTan, {0.5, 0, 0, 0, 0}));
  distorted.integrate(1, *cloud, image, Pose{}, camera_facing_x());
  CpuSurfaceMap pinhole(surface_cfg(), camera_calib(w, h, 300.0));
  pinhole.integrate(1, *cloud, image, Pose{}, camera_facing_x());

  const auto count_band = [](const meridian::ColorMesh& mesh, int* centre_red, int* centre_total,
                             int* band_red, int* band_total) {
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      const auto& v = mesh.vertices[i];
      if (std::abs(v.z()) > 0.3f) continue;
      const auto& c = mesh.colors[i];
      const bool red = c[0] > 150 && c[1] < 100 && c[2] < 100;
      if (std::abs(v.y()) < 0.5f) {
        ++*centre_total;
        *centre_red += red;
      } else if (std::abs(v.y()) > 1.7f && std::abs(v.y()) < 2.0f) {
        ++*band_total;
        *band_red += red;
      }
    }
  };

  int dc_red = 0, dc_total = 0, db_red = 0, db_total = 0;
  count_band(distorted.extract_mesh(), &dc_red, &dc_total, &db_red, &db_total);
  ASSERT_GT(dc_total, 20);
  ASSERT_GT(db_total, 20);
  // The centre still paints (small r2, negligible distortion); the outer band must not.
  EXPECT_GT(static_cast<double>(dc_red) / dc_total, 0.8);
  EXPECT_LT(static_cast<double>(db_red) / db_total, 0.05);

  int pc_red = 0, pc_total = 0, pb_red = 0, pb_total = 0;
  count_band(pinhole.extract_mesh(), &pc_red, &pc_total, &pb_red, &pb_total);
  ASSERT_GT(pb_total, 20);
  // Control: pinhole projection reaches the same band (imperfectly — the signed-by-side
  // fusion blurs at this incidence, so oblique voxels paint patchily even undistorted),
  // while the distorted camera leaves it essentially empty. The distortion model, not
  // the frustum, is what empties the band.
  EXPECT_GT(static_cast<double>(pb_red) / pb_total, static_cast<double>(db_red) / db_total + 0.3);
}

TEST(CpuSurfaceMap, ColourFallbackUnchanged) {
  // With no image the mesh keeps the height-turbo fallback: colours still vary.
  CpuSurfaceMap m(surface_cfg());
  m.integrate(1, *unit_sphere(20000, 1.0f), nullptr, Pose{}, Pose{});
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
