#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "meridian/common/pose.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// Closed string sets. Each maps 1:1 to an implementation selected through a factory;
// the YAML loader rejects any value outside the set.
enum class PipelineMode { Live, Replay };
enum class TimeSource { Ptp, Pps, Host };
enum class FrontEndKind { CtLivo, IekfOracle };
enum class BackEndKind { Isam2 };
enum class MapBackend { Nvblox };
enum class MeshKind { MarchingCubes };
enum class PlaceKind { ScanContextPp };
enum class RobustKernel { Cauchy, Gm, Tls };
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
struct PipelineConfig {
  PipelineMode mode = PipelineMode::Live;
  ThreadCounts threads{};
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
struct TimeConfig {
  TimeSource source = TimeSource::Ptp;
  double max_skew_ms = 5.0;
  PtpConfig ptp{};
  PpsConfig pps{};
  SwOffsetConfig swoffset{};
  TimeHealth health{};
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
  // T_body_cam: camera frame expressed in body.
  Pose extrinsic{};
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
  double blind = 0.5;       // [m] reject returns closer than this
  double det_range = 120.0; // [m] max usable range
  int point_filter_num = 3; // keep every Nth point
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
  double voxel_surf_m = 0.5;  // surf-point voxel-grid leaf size [m]
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
  int n_cp_max = 0;  // adaptive control-point cap; 0 = auto
  bool time_offset_estimate = false;
};
struct FrontendLidar {
  double voxel_map_m = 0.5;
  int num_match_points = 5;
  double max_match_dist_sq = 5.0;  // [m^2]
  double plane_thresh = 0.1;       // [m]
  double point_cov = 1e-3;
};
struct FrontendVisual {
  int patch = 8;
  int levels = 3;
  double img_point_cov = 100.0;
  double outlier_threshold = 1000.0;
  double ncc_thre = 0.0;
  bool exposure_estimate_en = true;
};
struct FrontendGnss {
  bool use = true;
};
struct FrontendKeyframe {
  double dist_m = 1.0;
  double rot_deg = 10.0;
  double time_s = 1.0;
};
struct FrontendConfig {
  FrontEndKind kind = FrontEndKind::CtLivo;
  double init_time_s = 0.1;
  int solver_max_iterations = 5;
  double solver_epsi = 1e-3;
  double degeneracy_thresh = 0.0;
  FrontendSpline spline{};
  FrontendLidar lidar{};
  FrontendVisual visual{};
  FrontendGnss gnss{};
  bool extrinsic_refine = true;
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
  BackEndKind kind = BackEndKind::Isam2;
  double anchor_sigma = 1e-4;
  int isam2_relinearize_skip = 1;
  double isam2_relinearize_thresh = 0.1;
  int isam2_extra_iters = 1;
  bool isam2_use_qr = false;
  double obs_inflation_max = 1e4;
  double obs_inflation_gamma = 2.0;
  double degenerate_thresh = 0.05;
  bool degenerate_lock = true;
  double loop_min_fitness = 0.5;
  double pcm_chi2_alpha = 0.99;
  int pcm_max_nodes = 64;
  RobustKernel robust_kernel = RobustKernel::Cauchy;
  bool gnc_enabled = true;
  int gnc_anneal_steps = 5;
  double gnc_reject_w = 0.1;
  double gnss_huber_k = 1.345;
  bool gnss_enabled = true;
  double gnss_max_cov = 25.0;  // [m^2]
  double gnss_min_baseline = 5.0;
  bool gnss_lock_yaw = false;
  bool extrinsic_refine = true;
  double extrinsic_prior_sigma = 1e-3;
  double extrinsic_refine_sigma = 1e-2;
  double extrinsic_excite_rot = 0.5;    // [rad]
  double extrinsic_excite_trans = 2.0;  // [m]
  double extrinsic_freeze_cov = 1e-6;
  double extrinsic_max_dev = 0.1;
  bool keep_inertial = false;
  double reintegrate_thresh = 0.1;  // [m]
  bool emit_moved_cov = false;
  double loop_gate_k = 3.0;
  BackendImu imu{};
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
  bool pcm = true;
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
  double std_side_min = 0.5;  // [m]
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
  double gicp_downsample = 0.25;   // [m]
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
