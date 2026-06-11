#pragma once

#include <functional>
#include <memory>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/frontend_diagnostics.hpp"
#include "meridian/common/graph_update.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// The L2 estimator boundary. Implementations own the trajectory representation;
// only value types cross this interface. ingest/live_state/diagnostics are
// thread-confined to the front-end stage thread; the keyframe sink fires on that
// thread and must only enqueue. Ingested samples are borrowed (the front-end
// copies what it keeps); emitted packets are moved.
class IFrontEnd {
 public:
  virtual ~IFrontEnd() = default;

  // Calibration snapshot: once at start, and again whenever a refined snapshot is
  // published. The front-end copies what it needs and records the snapshot version
  // for packet provenance.
  virtual void set_calibration(std::shared_ptr<const CalibrationSet> calib) = 0;

  // The primary input: one preprocessed sweep bundle. Triggers the registration
  // against the local map; deskew is performed internally at each point's own time.
  virtual void ingest(const PreprocessedGroup& group) = 0;

  // High-rate IMU between sweeps: advances live_state() only, never a solve. The
  // same samples arrive again inside the next group, which is what the window
  // actually optimises against.
  virtual void ingest_imu_live(const ImuSample& imu) = 0;

  // Loop-closure / global-optimisation feedback, applied between ingests on the
  // front-end thread: re-anchors the odom frame and shifts the trajectory by the
  // correction without restarting the window.
  virtual void apply_correction(const GraphUpdate& update) = 0;

  // The smooth odom-frame estimate at the latest valid time.
  virtual NavState live_state() const = 0;

  // Keyframe handoff: exactly one packet per keyframe, moved into the sink.
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void set_keyframe_sink(KeyframeSink sink) = 0;

  // Snapshot of the last solve's timings, residuals, and observability.
  virtual FrontEndDiagnostics diagnostics() const = 0;
};

// Builds the configured implementation; telemetry may be null (no-op sink).
// `deterministic` selects the bit-reproducible replay path: the window solve then
// ignores the wall-clock deadline and runs the fixed iteration schedule.
std::unique_ptr<IFrontEnd> makeFrontEnd(const FrontendConfig& cfg,
                                        std::shared_ptr<const CalibrationSet> calib,
                                        TelemetrySink* telemetry, bool deterministic = false);

}  // namespace meridian
