#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "meridian/calib/calibration_set.hpp"
#include "meridian/common/backend_diagnostics.hpp"
#include "meridian/common/gaussian.hpp"
#include "meridian/common/graph_update.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/loop_constraint.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/stamped_pose.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// L3 global optimizer boundary. All methods are confined to one serial driver thread (the
// pipeline's back-end thread live; the caller's thread in replay). add_* only stage graph
// mutations; optimize() folds everything staged since the last call in one incremental update.
class IBackEnd {
public:
  virtual ~IBackEnd() = default;

  virtual void add_keyframe(KeyframePacket&& kf) = 0;
  virtual void add_loop_constraint(const LoopConstraint& lc) = 0;
  virtual void add_absolute(const GnssFix& fix, std::uint64_t nearest_kf_id) = 0;

  // Folds the staged batch; returns which keyframes moved materially.
  virtual GraphUpdate optimize() = 0;

  virtual std::vector<StampedPose> corrected_trajectory() const = 0;
  virtual std::shared_ptr<const CalibrationSet> refined_calibration() const = 0;
  virtual BackEndDiagnostics diagnostics() const = 0;

  // Scheduling hint: the staged batch carries a correction that should not wait for the
  // cadence timer (an admitted loop or a just-locked GNSS datum).
  virtual bool wants_immediate_optimize() const = 0;

  // Marginal covariance of the latest keyframe pose, translation-first [rho; phi];
  // nullopt before the first optimize().
  virtual std::optional<PoseCov6> latest_pose_marginal() const = 0;
};

std::unique_ptr<IBackEnd> makeBackEnd(const BackendConfig& cfg,
                                      std::shared_ptr<const CalibrationSet> calib,
                                      TelemetrySink* telemetry, bool deterministic = false);

}  // namespace meridian
