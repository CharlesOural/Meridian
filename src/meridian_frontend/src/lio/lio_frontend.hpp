#pragma once

#include <memory>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/frontend_diagnostics.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"

namespace meridian {
class TelemetrySink;
}  // namespace meridian

namespace meridian::lio {

// Discrete LiDAR-inertial front-end. Each sweep is deskewed under a constant body
// screw, voxel-downsampled, and registered against the local voxel map by Gauss-Newton
// point-to-point alignment with an interval-averaged IMU motion prior; the solved pose
// updates the map and drives keyframe emission. Emitted keyframes carry frontend_kind 2.
//
// The estimation frame F_e is imu_link, tagged Frame::Body on emitted quantities; the
// world/odom frame is Frame::Odom. Internal 6-DoF tangents are translation-first
// [rho; phi]; the one rotation-first block is KeyframePacket::constraint_cov, reordered
// exactly once at pack time. Every path is synchronous on the caller's thread, so the
// `deterministic` flag changes nothing: two identical runs are bit-identical either way.
class LioFrontEnd final : public IFrontEnd {
public:
  LioFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
              TelemetrySink* telemetry, bool deterministic = false);

  void set_calibration(std::shared_ptr<const CalibrationSet> calib) override;
  void ingest(const PreprocessedGroup& group) override;
  void ingest_imu_live(const ImuSample& imu) override;
  void apply_correction(const GraphUpdate& update) override;
  NavState live_state() const override;
  void set_keyframe_sink(KeyframeSink sink) override;
  FrontEndDiagnostics diagnostics() const override;

private:
  FrontendConfig cfg_;
  std::shared_ptr<const CalibrationSet> calib_;
  TelemetrySink* telemetry_ = nullptr;
  bool deterministic_ = false;
  KeyframeSink keyframe_sink_;
};

}  // namespace meridian::lio
