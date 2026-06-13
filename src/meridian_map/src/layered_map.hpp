#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/map/imap_layer.hpp"
#include "meridian/map/iregistration_map.hpp"
#include "meridian/map/isurface_map.hpp"
#include "meridian/map/keyframe_store.hpp"

namespace meridian {
class TelemetrySink;
}

namespace meridian::map {

// The L4 façade: composes Tier R (registration) + the selected ISurfaceMap backend +
// the shared canonical KeyframeStore, and enforces "store first, derived tiers are
// caches". Loop correction is clear-and-rebuild: update the store poses, then clear the
// affected keyframes from both tiers and re-integrate them from the store at corrected
// poses, in ascending-id order (deterministic).
class LayeredMap final : public IMapLayer {
 public:
  LayeredMap(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
             std::shared_ptr<KeyframeStore> store, std::unique_ptr<IRegistrationMap> reg,
             std::unique_ptr<ISurfaceMap> surf, TelemetrySink* telemetry);

  void integrate(const KeyframePacket& kf, const Pose& T_map_body) override;
  void apply_graph_update(const GraphUpdate& update) override;
  std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const override;
  const ColorMesh& extract_mesh() override;
  MapDiagnostics diagnostics() const override;

 private:
  void integrate_id(std::uint64_t id, const Pose& T_map_body);

  MapConfig cfg_;
  std::shared_ptr<const CalibrationSet> calib_;
  std::shared_ptr<KeyframeStore> store_;
  std::unique_ptr<IRegistrationMap> reg_;
  std::unique_ptr<ISurfaceMap> surf_;
  TelemetrySink* sink_ = nullptr;
  std::size_t last_rebuild_kf_ = 0;
};

}  // namespace meridian::map
