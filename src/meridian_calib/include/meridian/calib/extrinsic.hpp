#pragma once

#include <cstdint>

#include "meridian/common/frame.hpp"
#include "meridian/common/gaussian.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// Provenance of an extrinsic prior, ordered loosest-trust last.
enum class CalibSource : std::uint8_t {
  Unknown = 0,
  Vendor,         // shipped by the sensor vendor
  Kalibr,         // imported from a Kalibr YAML
  TargetBoard,    // target-board app (checkerboard / Apriltag)
  HandEye,        // motion-based hand-eye, no target
  OnlineRefined,  // posterior written back by the back-end; a seed for next mission
  Manual,         // hand-entered (CAD / tape measure), loosest prior
};

// A sensor-to-estimation-frame transform with its prior uncertainty, online-
// refinement regime, and the temporal-offset twin of the geometric extrinsic.
struct Extrinsic {
  Frame child = Frame::Unknown;   // the sensor frame (OsSensor0 / CamLink / GnssLink)
  Frame parent = Frame::ImuLink;  // the estimation frame F_e
  Pose T_parent_child;            // offline prior mean T_<F_e>_<sensor>
  // 6-DoF prior covariance, tangent order [rho; phi] (translation-first).
  PoseCov6 prior_cov;
  bool refine_online = false;  // FIXED vs REFINED regime flag
  Timestamp calibrated_at = 0;  // when this prior was established [ns]

  CalibSource source = CalibSource::Unknown;
  std::uint32_t version = 0;  // monotonically increasing across refinements

  // Hard gate around the prior mean; a refinement leaving this box is rejected.
  double max_drift_trans_m = 0.10;  // ‖Δp‖ cap [m]
  double max_drift_rot_deg = 2.0;   // ‖Δφ‖ cap [deg]

  // Per-sensor temporal offset: t_sensor = t_Fe + time_offset_ns.
  double time_offset_ns = 0.0;      // prior mean [ns]
  double time_offset_std_ns = 0.0;  // prior 1σ [ns]; 0 ⇒ effectively fixed
  bool refine_time_online = false;  // refine the offset as a graph variable?
};

}  // namespace meridian
