#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/map_types.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// Tier S: the dense TSDF + colour surface and its Marching-Cubes mesh. This is the one
// backend-pluggable seam in L4 — concrete backends are CpuSurfaceMap (host),
// NvbloxSurfaceMap (GPU, deferred build), VulkanSurfaceMap (deferred). The fusion
// semantics are identical across backends; only performance and platform differ.
class ISurfaceMap {
public:
  virtual ~ISurfaceMap() = default;

  // Lay this keyframe's TSDF geometry then its colour at the given map pose.
  virtual void integrate(std::uint64_t id, const PointCloud& cloud_body,
                         const std::shared_ptr<const CameraFrame>& image, const Pose& T_map_body,
                         const Pose& T_body_cam) = 0;

  // Clear everything the given keyframes contributed (geometry + colour + mesh together),
  // so a loop region can be rebuilt from corrected poses.
  virtual void clear_keyframes(const std::vector<std::uint64_t>& ids) = 0;

  // Map-frame bounds of the surface volume these keyframes fused (their provenance),
  // which can spill past the retained clouds' own bounds by the truncation band and
  // storage granularity. The rebuild region derives from this, so clearing never
  // outruns the set of keyframes selected for re-integration.
  virtual Aabb dirty_bounds(const std::vector<std::uint64_t>& ids) const = 0;

  // Update and return the standing host mesh (extraction is incremental over changed
  // geometry; the returned reference is valid until the next call).
  virtual const ColorMesh& extract_mesh() = 0;

  virtual MapDiagnostics diagnostics() const = 0;
};

}  // namespace meridian
