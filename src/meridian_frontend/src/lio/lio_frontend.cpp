#include "lio/lio_frontend.hpp"

#include <utility>

namespace meridian::lio {

LioFrontEnd::LioFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                         TelemetrySink* telemetry, bool deterministic)
    : cfg_(cfg), calib_(std::move(calib)), telemetry_(telemetry), deterministic_(deterministic) {}

void LioFrontEnd::set_calibration(std::shared_ptr<const CalibrationSet> calib) {
  calib_ = std::move(calib);
}

void LioFrontEnd::ingest(const PreprocessedGroup& group) {
  // TODO(lio): implemented in a later lane
  (void)group;
}

void LioFrontEnd::ingest_imu_live(const ImuSample& imu) {
  // TODO(lio): implemented in a later lane
  (void)imu;
}

void LioFrontEnd::apply_correction(const GraphUpdate& update) {
  // TODO(lio): implemented in a later lane
  (void)update;
}

NavState LioFrontEnd::live_state() const {
  // TODO(lio): implemented in a later lane
  return NavState{};
}

void LioFrontEnd::set_keyframe_sink(KeyframeSink sink) { keyframe_sink_ = std::move(sink); }

FrontEndDiagnostics LioFrontEnd::diagnostics() const {
  // TODO(lio): implemented in a later lane
  return FrontEndDiagnostics{};
}

}  // namespace meridian::lio
