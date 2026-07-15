#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

#include "meridian/common/time.hpp"

namespace meridian {

// Axis-aligned bounding box in the map frame. Default-constructed to the empty box
// (min = +inf, max = -inf) so the first expand() seeds it correctly.
struct Aabb {
  Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f max = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());

  bool empty() const { return (min.array() > max.array()).any(); }

  void expand(const Eigen::Vector3f& p) {
    min = min.cwiseMin(p);
    max = max.cwiseMax(p);
  }
  void expand(const Aabb& o) {
    if (o.empty()) return;
    min = min.cwiseMin(o.min);
    max = max.cwiseMax(o.max);
  }

  bool contains(const Eigen::Vector3f& p) const {
    return (p.array() >= min.array()).all() && (p.array() <= max.array()).all();
  }

  bool intersects(const Aabb& o) const {
    if (empty() || o.empty()) return false;
    return (min.array() <= o.max.array()).all() && (max.array() >= o.min.array()).all();
  }
};

// Nearest cached plane to a query point, returned by IRegistrationMap::query_plane.
// Plane is n.x + d = 0 in the map frame; rms is the fit residual (confidence).
struct PlaneHit {
  Eigen::Vector3f n = Eigen::Vector3f::Zero();         // unit normal, map frame
  float d = 0.f;                                       // plane offset
  float rms = 0.f;                                     // fit residual
  Eigen::Vector3f centroid = Eigen::Vector3f::Zero();  // voxel centroid
  std::uint8_t n_pts = 0;                              // support
};

// The colourised triangle mesh handed L4 -> L6. Host mirror of the surface backend's
// mesh; indices are a flat triangle list (3 per triangle). All per-vertex arrays are
// parallel (vertices/normals/colors/confidence share the vertex index).
struct ColorMesh {
  std::vector<Eigen::Vector3f> vertices;            // map frame
  std::vector<Eigen::Vector3f> normals;             // per-vertex
  std::vector<std::array<std::uint8_t, 3>> colors;  // per-vertex RGB
  std::vector<std::uint32_t> indices;               // triangle list (3 per tri)
  std::vector<float> confidence;                    // per-vertex [0,1]
  Timestamp stamp = 0;                              // extraction time

  void clear() {
    vertices.clear();
    normals.clear();
    colors.clear();
    indices.clear();
    confidence.clear();
  }
};

// L4 health/timing counters, sampled to the debug bus each cycle.
struct MapDiagnostics {
  std::size_t reg_voxels = 0;
  std::size_t tsdf_blocks = 0;
  std::size_t kf_count = 0;
  double last_integrate_ms = 0.0;
  double last_rebuild_ms = 0.0;
  double last_mesh_ms = 0.0;
  std::size_t last_rebuild_kf = 0;
  std::size_t last_rebuild_voxels = 0;
  std::size_t hot_window_voxels = 0;
};

}  // namespace meridian
