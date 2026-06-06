#pragma once

#include <memory>

#include "meridian/calib/extrinsic.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

namespace meridian {

class TelemetrySink;

// Per-sensor LiDAR preprocessing: validity filtering, decimation, point-time
// validation. Input is a raw (Shared-immutable) scan; output is a filtered scan in
// the sensor frame, still undeskewed (deskew needs the trajectory, which lives in L2).
//
// Thread-confined: every instance is called from a single stage thread; it is not
// internally synchronized.
class ILidarPreprocessor {
 public:
  virtual ~ILidarPreprocessor() = default;

  // Returns a new LidarScan whose points satisfy the validity gate, are decimated,
  // and are sorted by t_offset_ns. Reuses raw.points (Shared-immutable) when nothing
  // was dropped; otherwise allocates a fresh Shared-immutable buffer. Never mutates
  // the input buffer bytes.
  virtual LidarScan process(const LidarScan& raw) const = 0;
};

// Builds the concrete validity-filter preprocessor. The self-hit mask referenced by
// cfg.lidar.selfhit_mask is resolved against T_imu_lidar / the calibration upstream.
// nominal_rate_hz is the LiDAR's nominal sweep rate; it sets the sweep_duration floor so
// an over-filtered sweep cannot collapse the deskew horizon. telemetry is non-owning and
// may be nullptr (no-op).
std::unique_ptr<ILidarPreprocessor> makeLidarPreprocessor(const PreprocessConfig& cfg,
                                                          const Extrinsic& T_imu_lidar,
                                                          double nominal_rate_hz,
                                                          TelemetrySink* telemetry);

}  // namespace meridian
