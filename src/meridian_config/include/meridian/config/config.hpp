#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// Closed string sets. Each maps 1:1 to an implementation selected through a factory;
// the YAML loader rejects any value outside the set.
enum class PipelineMode { Live, Replay };
enum class TimeSource { Ptp, Pps, Host };
enum class FrontEndKind { CtLivo };
enum class BackEndKind { Isam2 };
enum class MapBackend { Nvblox };
enum class MeshKind { MarchingCubes };
enum class PlaceKind { ScanContextPp };
enum class RobustKernel { Huber, Cauchy, Gm, Tls };
enum class SplineOrder { Cubic };
enum class LogLevel { Trace, Debug, Info, Warn, Error };
enum class StoreBackend { Ram, Mmap };
enum class SensorModel { OusterOS1_128, Pinhole, Vn100, BlackflyS, Generic };
enum class NoiseSource { Allan, Measured };

// ---- pipeline ----
struct ThreadCounts {
  int frontend = 1;
  int backend = 1;
  int map = 1;
};
// BoundedQueue capacities for the inter-stage edges (Q_meas lossy, Q_kf/Q_map lossless).
struct QueueConfig {
  int meas_capacity = 8;
  int kf_capacity = 64;
  int map_capacity = 64;
};
struct PipelineConfig {
  PipelineMode mode = PipelineMode::Live;
  ThreadCounts threads{};
  QueueConfig queue{};
};

// ---- time ----
struct PtpConfig {
  int domain = 0;
  std::string iface = "eth0";  // NIC running ptp4l
  bool require_lock = false;   // refuse start until the PTP servo is locked
};
struct PpsConfig {
  std::string device = "/dev/pps0";
  int expect_hz = 1;
};
struct SwOffsetXcorr {
  double window_s = 4.0;
  double grid_ms = 0.5;
  double motion_gate_radps2 = 0.05;
};
struct SwOffsetKf {
  double skew_window_s = 30.0;
  double Q_offset_ns2_per_s = 1e4;
  double R_default_ns = 500.0;
};
struct SwOffsetConfig {
  bool enable = true;
  SwOffsetXcorr xcorr{};
  SwOffsetKf kf{};
  double verify_residual_warn_us = 200.0;
  double reject_jump_ms = 5.0;
};
struct TimeHealth {
  double failed_timeout_ms = 1000.0;
  double rate_tolerance_frac = 0.20;
};
// Standing live stamp-integrity gate; one instance per sensor in L0.
struct ValidatorConfig {
  double gap_periods = 2.5;      // dropout when a raw gap exceeds this many periods
  double skew_warn_ppm = 200.0;  // |skew_ppm| above this raises SkewOutOfRange
  double nan_ratio_warn = 0.05;  // LiDAR NaN/Inf fraction above this raises a warning
};
struct TimeConfig {
  TimeSource source = TimeSource::Ptp;
  double max_skew_ms = 5.0;
  PtpConfig ptp{};
  PpsConfig pps{};
  SwOffsetConfig swoffset{};
  TimeHealth health{};
  ValidatorConfig validator{};
};

// ---- sensors ----
struct LidarSensorConfig {
  int id = 0;
  std::string name = "main";
  std::string frame = "os_sensor0";
  std::string model = "ouster_os1_128";
  std::string topic = "/os/points";
  double nominal_rate_hz = 10.0;
  bool ptp = true;
  std::string timestamp_mode = "TIME_FROM_PTP_1588";
  // T_body_lidar: translation [m] and rotation of the LiDAR sensor frame in body.
  Eigen::Vector3d extrinsic_T = Eigen::Vector3d::Zero();
  Eigen::Matrix3d extrinsic_R = Eigen::Matrix3d::Identity();
  // Whether the config supplied the extrinsic: an identity default is a valid-looking
  // value, so absence can only be recorded at parse time. Consumers warn on default.
  bool extrinsic_set = false;
  // Constant stamp correction onto the body-IMU timeline, applied once at ingest
  // (t_corrected = t_sensor + offset) BEFORE validation and aggregation, so every
  // stamp-driven decision sees corrected time. Calibration-session timeshifts do not
  // automatically transfer to a recording (hardware sync paths differ); set this only
  // after an empirical check on the actual data.
  double time_offset_ms = 0.0;
  // Subscribe RELIABLE instead of best-effort. Best-effort delivery of the large
  // fragmented scan messages silently loses a double-digit percentage in flight when
  // the host is under compute load, and a reliable reader pairs only with a reliable
  // writer -- so this must match the publisher: true for bag replay (with the player's
  // QoS override), false for a sensor driver publishing best-effort.
  bool qos_reliable = false;
};
struct ImuSensorConfig {
  int id = 0;
  std::string name = "main";
  std::string frame = "imu_link";
  std::string model = "vn100";
  std::string topic = "/os/imu";
  double rate_hz = 200.0;
  double cov_acc = 0.1;     // accel white-noise variance [ (m/s^2)^2 ]
  double cov_gyr = 0.1;     // gyro white-noise variance [ (rad/s)^2 ]
  double b_acc_cov = 1e-4;  // accel bias random-walk variance
  double b_gyr_cov = 1e-4;  // gyro bias random-walk variance
  bool has_device_clock = true;
  std::int64_t interval_end_shift_ns = 0;
};
struct CameraPhotometric {
  bool exposure_comp = true;
};
struct CameraSensorConfig {
  int id = 0;
  std::string name = "front";
  std::string frame = "cam_link";
  std::string model = "pinhole";
  std::string topic = "/cam0";
  double nominal_rate_hz = 20.0;
  // Pinhole intrinsics [fx, fy, cx, cy] in pixels.
  Eigen::Vector4d intrinsics = Eigen::Vector4d::Zero();
  // Lens distortion. distortion_model selects the IntrinsicsCamera::Distortion
  // mapping (none | radtan | equidistant; plumb_bob is accepted as radtan). The
  // coefficient layout follows the model: radtan k1,k2,p1,p2,k3; equidistant
  // k1..k4 with the last entry unused. width/height carry the image size geometry
  // and the on-image projection gate need.
  std::string distortion_model = "none";
  std::array<double, 5> distortion_coeffs = {0, 0, 0, 0, 0};
  int width = 0, height = 0;
  // T_body_cam: camera frame expressed in body. extrinsic_set records whether the
  // config supplied it: an identity default is indistinguishable from a real
  // calibration by value, and consuming it silently aims the camera frustum along
  // the wrong axis (the visual stage then runs but never promotes a point).
  Pose extrinsic{};
  bool extrinsic_set = false;
  // Constant stamp correction onto the body-IMU timeline (same semantics and caveat
  // as the LiDAR key: t_corrected = t_sensor + offset, applied once at ingest).
  double time_offset_ms = 0.0;
  // The stream delivers JPEG/PNG CompressedImage payloads that the ingest layer must
  // decode before conversion (the high-resolution frame cameras ship compressed).
  bool compressed = false;
  CameraPhotometric photometric{};
  std::string trigger = "gpio";
  bool exposure_from_meta = true;
  std::string shutter = "global";
};
struct GnssSensorConfig {
  int id = 0;
  std::string name = "rover";
  std::string frame = "gnss_link";
  std::string topic = "/gnss/fix";
  bool enable = true;
  bool pps_disciplines_clock = true;
  // Antenna lever arm: the GNSS phase-centre position in the body (IMU) frame. The back-end
  // needs it to relate an absolute antenna fix to a body pose; without it GNSS is unusable.
  Eigen::Vector3d extrinsic_T = Eigen::Vector3d::Zero();
  Eigen::Matrix3d extrinsic_R = Eigen::Matrix3d::Identity();
  bool extrinsic_set = false;
};
struct SensorsConfig {
  LidarSensorConfig lidar{};
  ImuSensorConfig imu{};
  CameraSensorConfig camera{};
  GnssSensorConfig gnss{};
};

// Time-sync aggregation block, sits beside sensors in the tree.
struct AggregationConfig {
  double max_wait_ms = 150.0;
  double reorder_ms = 20.0;
};

// ---- preprocess (L1) ----
struct PreprocLidar {
  SensorModel model = SensorModel::OusterOS1_128;
  double blind = 0.5;        // [m] reject returns closer than this
  double det_range = 120.0;  // [m] max usable range
  int point_filter_num = 3;  // keep every Nth point
  // Surf-voxel downsample: bucket survivors into a grid of edge voxel_surf_m and keep
  // at most surf_max_pts per cell. The sweep-duration floor never drops below this
  // fraction of one nominal period so deskew always has a horizon to cover.
  double voxel_surf_m = 0.5;      // [m] downsample voxel edge
  double sweep_floor_frac = 0.5;  // sweep-duration floor as a fraction of one period
  int surf_max_pts = 1;           // points kept per surf voxel
  std::uint64_t surf_seed = 0;    // reservoir RNG seed when surf_max_pts > 1
  bool intensity_gate = false;
  double i_min = 0.0;
  double i_max = 1e9;
  bool organize = false;
  std::string selfhit_mask = "";  // mask-set name resolved in the calibration set
};
struct PreprocDeskew {
  int imu_init_count = 10;
  int bootstrap_max_scans = 5;
  int restart_window_scans = 5;
};
struct PreprocCamera {
  bool photometric_calib = true;
  std::string vignette_map = "";
  std::string crf_lut = "";
  double ref_exposure_s = 0.01;
  double ref_gain = 1.0;
  int pyramid_levels = 3;
};
struct PreprocImu {
  NoiseSource noise_source = NoiseSource::Allan;
  double init_max_var = 0.1;
  double init_max_grav_err = 0.5;  // [m/s^2]
  double noise_sanity_ratio = 10.0;
  double imu_acc_fs = 156.0;  // [m/s^2] accel full scale
  double imu_gyr_fs = 34.9;   // [rad/s] gyro full scale
  bool drop_saturated = false;
};
struct PreprocGnss {
  bool enable = true;
  GnssFix::FixType min_fix_type = GnssFix::FixType::DGPS;
  int min_sats = 6;
  double max_pos_var = 25.0;  // [m^2]
  double max_dop = 5.0;
  bool spoof_check = true;
  double spoof_vel_thresh = 3.0;  // [m/s]
  int spoof_persist = 3;
  int spoof_window_ms = 1000;
  bool spoof_clock_check = false;
  double spoof_clock_thresh = 0.05;  // [s]
};
struct PreprocessConfig {
  PreprocLidar lidar{};
  PreprocDeskew deskew{};
  PreprocCamera camera{};
  PreprocImu imu{};
  PreprocGnss gnss{};
};

// ---- frontend (L2) ----
struct FrontendSpline {
  SplineOrder order = SplineOrder::Cubic;
  double knot_dt_ms = 25.0;  // outer-knot spacing [ms]
  int window_knots = 8;
  int n_cp_max = 0;  // adaptive control-point cap; <=1 disables adaptive density
  bool time_offset_estimate = false;
  // Adaptive-knot band edges: each crossed threshold adds one control point. The
  // rising edges are matched against peak per-sample IMU excitation in the segment;
  // a band-down requires falling below the edge by knot_density_hysteresis (fraction).
  std::vector<double> knot_omega_thresh{};  // [rad/s] angular-rate band edges
  std::vector<double> knot_accel_thresh{};  // [m/s^2] specific-force band edges
  double knot_density_hysteresis = 0.15;
};
struct FrontendLidar {
  double voxel_map_m = 0.5;
  int num_match_points = 5;
  double max_match_dist_sq = 5.0;  // [m^2]
  double plane_thresh = 0.1;       // [m]
  double point_cov = 1e-3;
  // Factor-count budget: cap residuals at max_lidar_factors, allocated across
  // normal_strata stratified bins with a per-stratum floor of min_factors_per_normal.
  int max_lidar_factors = 1500;
  int min_factors_per_normal = 50;
  int normal_strata = 7;
  // Side length of the cubic local map kept around the current body position [m]. The
  // ikd-Tree is segmented so points farther than half this from the body are deleted,
  // bounding map RAM and nearest-neighbour search depth. Non-positive disables trimming
  // (the map grows for the whole trajectory).
  double local_map_cube_m = 500.0;
};
struct FrontendVisual {
  // Master gate for the sparse-direct photometric stage. When off (or when the
  // camera intrinsics are zero/invalid) the front-end runs LIO only and the visual
  // code path is skipped entirely.
  bool enable = true;
  int patch = 8;
  int levels = 3;
  double img_point_cov = 100.0;
  double outlier_threshold = 1000.0;
  double ncc_thre = 0.0;
  bool exposure_estimate_en = true;
  // Exposure random-walk tie + positivity clamp: consecutive in-window inverse
  // exposures are tied with weight 1/(inv_expo_cov * dt), and each tau is lower-
  // bounded at inv_expo_min so it can never reach a non-positive value.
  double inv_expo_cov = 1e-2;  // [1/s] random-walk density sigma_tau^2
  double inv_expo_min = 1e-3;  // strict positivity lower bound on tau
  // One-best-per-cell selection grid and the homography degenerate-warp guards.
  int grid_cell_px = 32;
  double warp_det_min = 0.1;
  double warp_det_max = 10.0;
  double warp_cond_max = 50.0;
  // Half-extent [m] of the active visual-map box: points farther than this from the
  // current pose are evicted. Bounds the per-sweep map-maintenance cost to O(local
  // points), independent of trajectory length. Must be near the camera's
  // re-observability range, NOT the LiDAR map scale -- the visual map re-scores its
  // resident points every sweep, so an over-large box makes per-sweep cost grow with
  // distance travelled and starves the real-time budget.
  double active_box_m = 20.0;
};
// Conservative absolute-position GNSS residual. `use` is the master gate: with it off
// (or with no fixes present) the GNSS code path is never entered and the trajectory is
// bit-identical to a build without it. The floors are per-fix-type position-std pairs
// (horizontal, vertical) in metres; the per-fix ENU covariance is floored element-wise
// against them before weighting, since a receiver's reported covariance is optimistic.
// innovation_k is the Mahalanobis k-sigma acceptance gate; reacquire_count is the run
// of consecutive in-gate fixes required to re-admit GNSS after a gap or a gated-out run.
struct FrontendGnss {
  bool use = true;
  double floor_fixed_h = 0.05;  // RTK_Fixed horizontal std [m]
  double floor_fixed_v = 0.10;  // RTK_Fixed vertical std [m]
  double floor_float_h = 0.50;  // RTK_Float horizontal std [m]
  double floor_float_v = 1.00;  // RTK_Float vertical std [m]
  double floor_dgps_h = 1.50;   // DGPS horizontal std [m]
  double floor_dgps_v = 3.00;   // DGPS vertical std [m]
  double floor_spp_h = 3.00;    // SPP horizontal std [m]
  double floor_spp_v = 6.00;    // SPP vertical std [m]
  double innovation_k = 3.0;    // Mahalanobis innovation gate
  int reacquire_count = 5;      // consecutive in-gate fixes to re-admit
};
struct FrontendKeyframe {
  double dist_m = 1.0;
  double rot_deg = 10.0;
  double time_s = 1.0;
};
// Deadline bracket for the wall-clock solve path. min_iterations always run before a
// cutoff so a starved step never publishes a barely-improved pose; the replay path
// ignores time_limit_ms and runs the fixed schedule.
struct FrontendSolver {
  double time_limit_ms = 90.0;
  int min_iterations = 2;
};
// Physical box bounds on the IMU biases so an outlier burst cannot drive a bias
// absurd, plus the cadence of the piecewise-linear bias timeline. Both biases are
// free states tied by a continuous-time random walk between consecutive knots.
struct FrontendBias {
  double gyr_max = 0.5;     // [rad/s]
  double acc_max = 5.0;     // [m/s^2]
  double knot_dt_ms = 500;  // bias knot cadence [ms]
};
// Off-by-default low-weight jerk / angular-acceleration regularizer; engaged only on
// under-excited spans (excitation below the floor and a degenerate eigenaxis).
struct FrontendMotionReg {
  bool enable = true;
  double weight = 1e-3;           // relative to the IMU accel weight
  double excitation_floor = 0.0;  // engage below this excitation level
};
struct FrontendConfig {
  FrontEndKind kind = FrontEndKind::CtLivo;
  double init_time_s = 0.1;
  int solver_max_iterations = 5;
  double solver_epsi = 1e-3;
  // Eigenvalue floor for the per-axis degeneracy score lambda/(lambda+thresh); a zero
  // floor makes every axis score 1 and the degeneracy gate never fires, so keep > 0.
  double degeneracy_thresh = 10.0;
  // Fixed-cadence re-association: at most max_outer_iters association passes, each with
  // reassoc_steps inner LM iterations, stopping early once the keyframe-time pose moves
  // less than assoc_shift_thresh_m AND assoc_shift_thresh_deg since the prior pass.
  int max_outer_iters = 4;
  int reassoc_steps = 2;
  double assoc_shift_thresh_m = 0.02;
  double assoc_shift_thresh_deg = 0.2;
  // Multiplier in (0, 1] on the marginalization prior's information at build time
  // (sqrt(scale) on its sqrt-information and residual, so the prior's Gauss-Newton
  // step direction is preserved while its confidence is deflated). 1.0 leaves the
  // prior untouched and is bit-identical to a build without the knob.
  double marg_prior_scale = 1.0;
  FrontendSolver solver{};
  FrontendBias bias{};
  FrontendMotionReg motion_reg{};
  FrontendSpline spline{};
  FrontendLidar lidar{};
  FrontendVisual visual{};
  FrontendGnss gnss{};
  FrontendKeyframe keyframe{};
};

// ---- backend (L3) ----
struct BackendImu {
  double acc_noise = 0.1;     // [ (m/s^2)/sqrt(Hz) ]
  double gyr_noise = 0.1;     // [ (rad/s)/sqrt(Hz) ]
  double acc_bias_rw = 1e-4;  // [ (m/s^3)/sqrt(Hz) ]
  double gyr_bias_rw = 1e-4;  // [ (rad/s^2)/sqrt(Hz) ]
};
struct BackendConfig {
  bool enable = true;  // bring-up switch: off runs the pipeline without the back-end
  BackEndKind kind = BackEndKind::Isam2;
  // Gauge damping on the first pose, lambda = 1/sigma^2. Two-sided: every pose marginal
  // floors at sigma^2 (keep sigma small), while a rigid correction propagates through the
  // anchor at ~chain_info/(chain_info+lambda) per solver iteration (keep lambda moderate;
  // measured: lambda >= 1e4 visibly freezes the rigid mode even in a batch solve).
  double anchor_sigma = 0.1;
  // iSAM2
  int isam2_relinearize_skip = 1;
  double isam2_relinearize_thresh = 0.1;
  int extra_iters_normal = 0;  // extra Dogleg passes on a normal batch
  int extra_iters_loop = 4;    // extra Dogleg passes when a loop was admitted
  bool isam2_use_qr = false;
  // optimise cadence, decoupled from keyframe insert
  double optimize_interval_ms = 100.0;  // min wall-clock between batched update calls
  int queue_warn_depth = 32;            // input queue depth that raises a warning
  // observability -> noise inflation
  double obs_inflation_max = 1e4;
  double obs_inflation_gamma = 2.0;
  double degenerate_thresh = 0.05;  // score below = degenerate
  bool degenerate_lock = true;      // hard-lock worst axis if degenerate
  // loops + PCM
  double loop_min_fitness = 0.5;  // reject low GICP fitness
  double pcm_chi2_alpha = 0.99;
  int pcm_max_nodes = 64;  // exact max-clique cap
  // robust kernels: huber is the committed incremental loop kernel
  RobustKernel robust_kernel = RobustKernel::Huber;
  double loop_huber_k = 1.345;
  double gnss_huber_k = 1.345;
  double gnc_reject_w = 0.1;  // weight below which a loop is removed
  bool gnc_enabled = false;   // experimental amortised GNC inside iSAM2
  int gnc_anneal_steps = 5;
  int gnc_consolidate_interval = 10;  // admitted loops between batch consolidations (0 = off)
  // GNSS
  bool gnss_enabled = true;
  double gnss_max_cov = 25.0;  // [m^2] drop fix if trace(cov_enu) above
  bool gnss_lock_yaw = false;  // external heading available
  // datum init
  double gnss_min_baseline = 5.0;         // [m] min travelled baseline before datum fit
  double gnss_min_excitation = 3.0;       // [m] min dominant-axis span of buffered ENU track
  double gnss_min_speed = 0.5;            // [m/s] speed a fix must exceed to count as moving
  int gnss_min_moving_fixes = 5;          // moving fixes required before datum fit
  double gnss_datum_yaw_sigma_max = 5.0;  // [deg] reject datum fit above this yaw uncertainty
  // gating + decimation
  bool gnss_skip_if_confident = true;   // skip fix no tighter than the back-end marginal
  double gnss_skip_confidence_k = 1.0;  // kappa in the skip-if-confident test
  double gnss_min_spacing = 1.0;        // [m] min travelled baseline between admitted fixes
  // drift redistribution across an outage span
  bool gnss_redistribute = false;
  int gnss_reacq_fix = 1;                // min fix-quality enum to count as re-acquired
  int gnss_reacq_persist = 5;            // consecutive accepted fixes to declare re-acquisition
  int gnss_redistribute_span_max = 200;  // max keyframes back redistribution reaches
  // online extrinsics, off by default; enable per-platform
  bool extrinsic_refine = false;
  double extrinsic_prior_sigma = 1e-3;
  double extrinsic_refine_sigma = 1e-2;
  double extrinsic_excite_rot = 0.5;    // [rad] cumulative before refine
  double extrinsic_excite_trans = 2.0;  // [m] cumulative before refine
  double extrinsic_freeze_cov = 1e-6;   // freeze when marginal below
  double extrinsic_max_dev = 0.1;       // sanity clamp
  // marginalization
  bool keep_inertial = false;  // keep restart V/B variables live
  // L4 re-integration trigger
  double reintegrate_thresh = 0.1;  // [m] pose-move threshold
  bool emit_moved_cov = false;      // fill moved-set covariances (costly)
  // loop pre-filter marginal
  double loop_gate_k = 3.0;  // k_gate * sigma_pos search radius
  BackendImu imu{};
  // debug
  bool debug_dump_residuals = false;
  bool snapshot_on_request = false;
  std::string snapshot_dir = "/tmp/meridian";
};

// ---- map (L4) ----
struct MapStore {
  StoreBackend backend = StoreBackend::Ram;
};
struct MapConfig {
  MapBackend backend = MapBackend::Nvblox;
  double reg_voxel_m = 0.2;
  int reg_max_pts = 20;
  int reg_max_level = 3;
  double reg_planarity = 0.1;
  int reg_min_plane_pts = 5;
  int reg_neighbor_ring = 1;
  double reg_hot_radius_m = 100.0;
  double tsdf_voxel_m = 0.05;  // must be <= reg_voxel_m and <= preprocess.voxel_surf_m
  int tsdf_trunc_voxels = 4;
  double tsdf_w_max = 100.0;
  bool colour = true;
  MeshKind mesh = MeshKind::MarchingCubes;
  double mesh_max_rate_hz = 2.0;
  double mesh_conf_w = 10.0;
  MapStore store{};
};

// ---- place (L5) ----
struct PlaceConfig {
  PlaceKind kind = PlaceKind::ScanContextPp;
  bool enable = false;  // L5 master switch; off => the pipeline builds no detector
  bool pcm = true;
  // submap accumulator (geometry granularity for retrieval + GICP)
  int submap_window = 5;       // last-N keyframes composed into the anchor frame
  double submap_voxel = 0.25;  // [m] downsample edge after composition
  int submap_cache = 32;       // anchor-keyed LRU size
  bool gicp_source_submap = false;  // GICP source is the anchor submap (else its single cloud)
  double cov_psd_floor = 1e-9;      // floor added to the emitted loop covariance diagonal
  // cadence / gating
  int detect_period_kf = 1;
  double min_time_gap = 30.0;  // [s]
  int min_kf_gap = 30;
  int cooldown_kf = 5;
  // Scan Context++
  int sc_Nr = 20;
  int sc_Ns = 60;
  double sc_rmax = 80.0;  // [m]
  int sc_knn = 15;
  double sc_dist_thresh = 0.13;
  double sc_max_xy = 50.0;  // [m]
  int sc_yaw_search_band = 10;
  int sc_topK = 5;
  // STD/BTC
  int std_max_keypoints = 500;
  int std_knn_kp = 10;
  double std_side_min = 0.5;   // [m]
  double std_side_max = 50.0;  // [m]
  double std_side_tol = 0.2;   // [m]
  int std_min_matches = 4;
  double std_ransac_eps = 0.5;  // [m]
  int std_ransac_iters = 200;
  int std_min_inliers = 5;
  double std_yaw_gate = 30.0;  // [deg]
  std::array<double, 3> scoreB_w = {0.2, 0.5, 0.3};
  int std_topK = 2;
  // GICP / small_gicp
  double gicp_downsample = 0.25;    // [m]
  double gicp_max_corr_dist = 1.0;  // [m]
  int gicp_num_threads = 4;
  double gicp_voxel_res = 1.0;  // [m]
  double gicp_fitness_min = 0.6;
  double gicp_overlap_min = 0.4;
  double gicp_rmse_max = 0.3;  // [m]
  double gicp_cond_max = 1e4;
  double gicp_fit_sigma = 0.1;  // [m]
  bool use_photo_tiebreak = false;
  // PCM
  double pcm_chi2_conf = 0.99;
  double pcm_maxclique_ms = 20.0;
  int pcm_quarantine = 50;
  // covariance shaping to L3
  double cov_lambda = 1e-3;
  double cov_degenerate_eig = 1.0;
  double cov_degenerate_mult = 100.0;
};

// ---- debug ----
struct DebugConfig {
  LogLevel level = LogLevel::Info;
  bool publish_clouds = true;
  bool publish_markers = true;
  bool publish_odom = true;  // /meridian/odom (the rviz pose arrow); TF is published regardless
  bool timing = true;
  double telemetry_rate_hz = 10.0;
};

// The root configuration tree, loaded once and held const for the pipeline's life.
struct Config {
  PipelineConfig pipeline{};
  TimeConfig time{};
  SensorsConfig sensors{};
  AggregationConfig aggregation{};
  PreprocessConfig preprocess{};
  FrontendConfig frontend{};
  BackendConfig backend{};
  MapConfig map{};
  PlaceConfig place{};
  DebugConfig debug{};

  // Runs unit/range/cross-field checks. Returns true on success; on failure returns
  // false and (if error_out != nullptr) writes a precise human-readable reason.
  bool validate(std::string* error_out) const;
};

}  // namespace meridian
