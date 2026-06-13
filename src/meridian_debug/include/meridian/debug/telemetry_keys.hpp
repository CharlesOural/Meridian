#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace meridian::keys {

// The single source of truth for every telemetry channel key. A call site references
// a named constant (keys::frontend::SolveMs) instead of a bare string literal, so a
// typo is a compile error and the full set of channels is discoverable in one place.
//
// Naming convention, by transport:
//   '/'-separated  -> a data channel (scalar/vec/cloud/pose/marker/image/event). The
//                     prefix is the owning layer: frontend/, backend/, lidar/, gnss/,
//                     place/, pipeline/, preprocess/, sensors/, map/, odom/, calib/,
//                     wrapper/.
//   '.'-separated  -> a timing stage name passed to timing()/MERIDIAN_SCOPED_TIME,
//                     grouped under keys::stage. The dot mirrors the call-tree nesting
//                     (frontend.ct.solve.linear_solver) the way a profiler reads it.
//
// A key string is the on-the-wire identity (ROS topic suffix, file-log key, replay
// hash input); these literals are the contract and must not change casually. Some keys
// are used in more than one role (an enabled() gate plus an event tag, or a vec plus a
// marker ns sharing a namespace) — one entry covers the string regardless of role, so
// the descriptor carries no single "kind".

enum class Unit : std::uint8_t {
  None,           // dimensionless / not applicable
  Meters,         // m
  Millimeters,    // ms is taken by time; lengths in mm are unused so far
  Ms,             // milliseconds (wall time)
  Hz,             // 1/s
  Count,          // integer tally
  Ratio,          // dimensionless [0,1] or fraction
  Degrees,        // deg
  MetersPerSec,   // m/s
  MetersPerSec2,  // m/s^2
  RadPerSec,      // rad/s
};

// A telemetry key: the wire string plus its unit (for scalar/vec; None otherwise).
// Converts implicitly to const char* so it drops into the TelemetrySink methods and
// enabled() unchanged.
struct Key {
  const char* id = nullptr;
  Unit unit = Unit::None;
  constexpr Key(const char* id_, Unit unit_ = Unit::None) noexcept : id(id_), unit(unit_) {}
  constexpr operator const char*() const noexcept { return id; }
  [[nodiscard]] constexpr std::string_view view() const noexcept { return id; }
};

// ---- L1 LiDAR validity filter -------------------------------------------------------
namespace lidar {
inline constexpr Key NIn{"lidar/n_in", Unit::Count};
inline constexpr Key NOut{"lidar/n_out", Unit::Count};
inline constexpr Key NVoxelOut{"lidar/n_voxel_out", Unit::Count};
inline constexpr Key NNan{"lidar/n_nan", Unit::Count};
inline constexpr Key NBlind{"lidar/n_blind", Unit::Count};
inline constexpr Key NFar{"lidar/n_far", Unit::Count};
inline constexpr Key NSelfhit{"lidar/n_selfhit", Unit::Count};
inline constexpr Key NIntensity{"lidar/n_intensity", Unit::Count};
inline constexpr Key SelfhitFrac{"lidar/selfhit_frac", Unit::Ratio};
}  // namespace lidar

// ---- L1 GNSS gate -------------------------------------------------------------------
namespace gnss {
inline constexpr Key Accepted{"gnss/accepted", Unit::Count};
inline constexpr Key Rejected{"gnss/rejected", Unit::Count};
inline constexpr Key Spoof{"gnss/spoof", Unit::Count};
}  // namespace gnss

// ---- L1 camera preprocessing --------------------------------------------------------
namespace preprocess {
inline constexpr Key CameraPyramid{"preprocess/camera_pyramid"};  // enable gate
inline constexpr Key CameraPyramidLevels{"preprocess/camera_pyramid_levels", Unit::Count};
inline constexpr Key CameraRaw{"preprocess/camera_raw"};        // image
inline constexpr Key CameraIntensity{"preprocess/camera_intensity"};  // image
}  // namespace preprocess

// ---- L2 front-end -------------------------------------------------------------------
namespace frontend {
// solver / trajectory
inline constexpr Key OuterIters{"frontend/outer_iters", Unit::Count};
inline constexpr Key SolveMs{"frontend/solve_ms", Unit::Ms};
inline constexpr Key KeyframeCount{"frontend/keyframe_count", Unit::Count};
// per-axis observability (vec, translation-first) + the worst-axis scalar
inline constexpr Key Observability{"frontend/observability", Unit::Ratio};
inline constexpr Key ObsMin{"frontend/obs_min", Unit::Ratio};
// state
inline constexpr Key BiasGyrNorm{"frontend/state/bias_gyr_norm", Unit::RadPerSec};
inline constexpr Key BiasAccNorm{"frontend/state/bias_acc_norm", Unit::MetersPerSec2};
inline constexpr Key VelNorm{"frontend/state/vel_norm", Unit::MetersPerSec};
// scan-to-map association
inline constexpr Key AssocNAttempted{"frontend/assoc/n_attempted", Unit::Count};
inline constexpr Key AssocNMatched{"frontend/assoc/n_matched", Unit::Count};
// Gauss-Newton solver
inline constexpr Key SolverGnIters{"frontend/solver/gn_iters", Unit::Count};
inline constexpr Key SolverDxNorm{"frontend/solver/dx_norm"};
inline constexpr Key SolverChi{"frontend/solver/chi"};
// local map
inline constexpr Key MapVoxels{"frontend/map/voxels", Unit::Count};
inline constexpr Key MapPoints{"frontend/map/points", Unit::Count};
inline constexpr Key MapSize{"frontend/map/size", Unit::Count};
// lidar stream
inline constexpr Key LidarInliers{"frontend/lidar/inliers"};  // cloud
// LIO scalars
inline constexpr Key LioBeta{"frontend/lio/beta"};
inline constexpr Key LioAccelVar{"frontend/lio/accel_var"};
inline constexpr Key LioNCorr{"frontend/lio/n_corr", Unit::Count};
inline constexpr Key LioDeskewSpanTMs{"frontend/lio/deskew_span_t_ms", Unit::Ms};
// odometry path sample (pose)
inline constexpr Key PathSample{"frontend/path_sample"};
// event tags
inline constexpr Key Keyframe{"frontend/keyframe"};
inline constexpr Key LioInitBacklog{"frontend/lio/init_backlog"};
inline constexpr Key LioInitDone{"frontend/lio/init_done"};
inline constexpr Key LioError{"frontend/lio/error"};
inline constexpr Key LioGap{"frontend/lio/gap"};
inline constexpr Key LioReject{"frontend/lio/reject"};
inline constexpr Key LioReseed{"frontend/lio/reseed"};
}  // namespace frontend

// ---- L3 back-end --------------------------------------------------------------------
namespace backend {
// scalars
inline constexpr Key Chi2{"backend/chi2"};
inline constexpr Key NFactors{"backend/n_factors", Unit::Count};
inline constexpr Key NLoops{"backend/n_loops", Unit::Count};
inline constexpr Key NGnss{"backend/n_gnss", Unit::Count};
inline constexpr Key UpdateMs{"backend/update_ms", Unit::Ms};
inline constexpr Key RelinCount{"backend/relin_count", Unit::Count};
inline constexpr Key QueueDepth{"backend/queue_depth", Unit::Count};
inline constexpr Key OptimizeLag{"backend/optimize_lag", Unit::Count};
inline constexpr Key FallbackCount{"backend/fallback_count", Unit::Count};
inline constexpr Key ObsMin{"backend/obs_min", Unit::Ratio};
inline constexpr Key NKeyframes{"backend/n_keyframes", Unit::Count};
// vec
inline constexpr Key GnssResidual{"backend/gnss/residual", Unit::Meters};
// per-keyframe observability vec; the keyframe id is appended to this prefix
inline constexpr Key ObservabilityPrefix{"backend/observability/", Unit::Ratio};
// markers
inline constexpr Key LoopEdge{"backend/loop_edge"};
inline constexpr Key GraphNodes{"backend/graph_nodes"};
inline constexpr Key GraphEdges{"backend/graph_edges"};
// event tags
inline constexpr Key Contiguity{"backend/contiguity"};
inline constexpr Key PsdClamp{"backend/psd_clamp"};
inline constexpr Key Indeterminate{"backend/indeterminate"};
inline constexpr Key Degenerate{"backend/degenerate"};
inline constexpr Key InfoForm{"backend/info_form"};
inline constexpr Key ObsFrame{"backend/obs_frame"};
inline constexpr Key Relinearize{"backend/relinearize"};
inline constexpr Key RestartBridge{"backend/window_restart_bridge"};
inline constexpr Key MarginalizeSkip{"backend/marginalize_skip"};
inline constexpr Key DatumLocked{"backend/gnss/datum_locked"};
inline constexpr Key GnssDisabled{"backend/gnss_disabled"};
inline constexpr Key GnssSkip{"backend/gnss_skip"};
inline constexpr Key AbsoluteIgnored{"backend/absolute_ignored"};
inline constexpr Key LoopAccepted{"backend/loop_accepted"};
inline constexpr Key LoopRejectedPcm{"backend/loop_rejected_pcm"};
inline constexpr Key LoopRejectedGnc{"backend/loop_rejected_gnc"};
inline constexpr Key ExtrinsicExcited{"backend/extrinsic_excited"};
inline constexpr Key ExtrinsicClamped{"backend/extrinsic_clamped"};
inline constexpr Key ExtrinsicFrozen{"backend/extrinsic_frozen"};
inline constexpr Key G2oSnapshot{"backend/g2o_snapshot"};
inline constexpr Key QueueOverload{"backend/queue_overload"};
}  // namespace backend

// ---- L5 place / loop closure --------------------------------------------------------
namespace place {
inline constexpr Key BestScDist{"place/best_sc_dist", Unit::Ratio};
inline constexpr Key NEligible{"place/n_eligible", Unit::Count};
inline constexpr Key NCandidates{"place/n_candidates", Unit::Count};
inline constexpr Key BestFitness{"place/best_fitness", Unit::Ratio};
inline constexpr Key Verified{"place/verified", Unit::Count};
inline constexpr Key SelfTestRejected{"place/self_test_rejected", Unit::Count};
inline constexpr Key Emitted{"place/emitted", Unit::Count};
}  // namespace place

// ---- Cross-cutting pipeline / sensors / map / odom ----------------------------------
namespace pipeline {
inline constexpr Key QSensorsDepth{"pipeline/q_sensors_depth", Unit::Count};
inline constexpr Key QSensorsDropped{"pipeline/q_sensors_dropped", Unit::Count};
inline constexpr Key QMeasDropped{"pipeline/q_meas_dropped", Unit::Count};
inline constexpr Key QMeasDroppedImu{"pipeline/q_meas_dropped_imu", Unit::Count};
inline constexpr Key QMeasDroppedSweep{"pipeline/q_meas_dropped_sweep", Unit::Count};
inline constexpr Key GroupImuCount{"pipeline/group_imu_count", Unit::Count};
inline constexpr Key GroupPoints{"pipeline/group_points", Unit::Count};
}  // namespace pipeline

namespace sensors {
inline constexpr Key ImuInGroup{"sensors/imu_in_group", Unit::Count};
inline constexpr Key LidarSweepDurationMs{"sensors/lidar/sweep_duration_ms", Unit::Ms};
inline constexpr Key LidarExtrinsicDefault{"sensors/lidar/extrinsic_default"};  // event tag
// Dynamic-suffix prefixes: the sensor id / error code is appended at runtime.
inline constexpr Key ValidatorPrefix{"sensors/validator/", Unit::Count};
inline constexpr Key RatePrefix{"sensors/rate/", Unit::Hz};
}  // namespace sensors

namespace map {
inline constexpr Key Cloud{"map/cloud"};            // cloud
inline constexpr Key Registered{"map/registered"};  // cloud
inline constexpr Key Keyframe{"map/keyframe"};       // pose
inline constexpr Key Mesh{"map/mesh"};               // marker (TRIANGLE_LIST, L4 surface)
}  // namespace map

namespace body {
inline constexpr Key Scan{"body/scan"};  // cloud
}  // namespace body

namespace odom {
inline constexpr Key Body{"odom/body"};  // pose
}  // namespace odom

namespace wrapper {
inline constexpr Key LidarCbN{"wrapper/lidar/cb_n", Unit::Count};
inline constexpr Key LidarLostUpstreamN{"wrapper/lidar/lost_upstream_n", Unit::Count};
inline constexpr Key LidarConvertRejectedN{"wrapper/lidar/convert_rejected_n", Unit::Count};
}  // namespace wrapper

// ---- Sensor health bridge -----------------------------------------------------------
namespace health {
// per-modality sample-rate scalars
inline constexpr Key ImuRateHz{"health/imu/rate_hz", Unit::Hz};
inline constexpr Key LidarRateHz{"health/lidar/rate_hz", Unit::Hz};
inline constexpr Key CameraRateHz{"health/camera/rate_hz", Unit::Hz};
inline constexpr Key GnssRateHz{"health/gnss/rate_hz", Unit::Hz};
inline constexpr Key RateHz{"health/rate_hz", Unit::Hz};
// per-modality level-transition event tags
inline constexpr Key Imu{"health/imu"};
inline constexpr Key Lidar{"health/lidar"};
inline constexpr Key Camera{"health/camera"};
inline constexpr Key Gnss{"health/gnss"};
inline constexpr Key Root{"health"};
inline constexpr Key Degrade{"health/degrade"};
}  // namespace health

// ---- Timing stages (the '.'-separated names handed to timing()) ----------------------
namespace stage {
inline constexpr Key Preprocess{"preprocess"};
inline constexpr Key PreprocessDeskew{"preprocess.deskew"};
inline constexpr Key PreprocessCamera{"preprocess.camera"};
inline constexpr Key PreprocessLidarValidity{"preprocess.lidar.validity"};
inline constexpr Key BackendOptimize{"backend.optimize"};
inline constexpr Key LioIngest{"frontend.lio.ingest", Unit::Ms};
}  // namespace stage

// The whole catalog as a flat range, for tooling: the unit table, the no-duplicate-id
// and no-orphan-literal coverage checks, and anyone enumerating channels.
std::span<const Key> catalog();

// Unit string for a key ("hz","ms","count","ratio","m","deg","m/s","m/s^2","rad/s"),
// or "" for None / unknown. Exact match first, then the registered dynamic-suffix
// prefixes (e.g. "sensors/rate/imu" resolves through sensors/rate/). Replaces the old
// substring heuristic with a catalog lookup.
const char* unit_string(std::string_view key);

}  // namespace meridian::keys
