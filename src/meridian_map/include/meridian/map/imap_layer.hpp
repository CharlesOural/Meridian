#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <optional>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/graph_update.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/map_types.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/config/config.hpp"
#include "meridian/map/keyframe_store.hpp"

namespace meridian {

class TelemetrySink;

// The L4 façade the pipeline holds. It owns Tier R (registration) + Tier S (the selected
// surface backend) and shares the canonical KeyframeStore. The "store first, derived
// tiers are caches" discipline lives here: integrate() puts into the store then both
// tiers; apply_graph_update() updates the store poses then clears-and-rebuilds the
// affected region from the store at corrected poses.
class IMapLayer {
 public:
  using Ptr = std::unique_ptr<IMapLayer>;
  virtual ~IMapLayer() = default;

  // Integrate a keyframe (cloud + optional RGB) at its CURRENT corrected map pose.
  virtual void integrate(const KeyframePacket& kf, const Pose& T_map_body) = 0;

  // Loop correction: clear the moved keyframes' region and re-integrate from the store.
  virtual void apply_graph_update(const GraphUpdate& update) = 0;

  // Registration query for the front-end (delegates to Tier R).
  virtual std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const = 0;

  // The standing colour mesh for the operator (delegates to the surface backend).
  virtual const ColorMesh& extract_mesh() = 0;

  virtual MapDiagnostics diagnostics() const = 0;
};

// Constructs Tier R + the surface backend named by cfg.backend (cpu always available;
// nvblox only under MERIDIAN_MAP_NVBLOX) + shares the canonical store. Selecting a
// backend not compiled into this build is a fail-fast error.
IMapLayer::Ptr makeMapLayer(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                            std::shared_ptr<KeyframeStore> store,
                            TelemetrySink* telemetry = nullptr);

}  // namespace meridian
