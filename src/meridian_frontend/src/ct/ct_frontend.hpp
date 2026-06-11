#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

#include "ct/keyframe_finalizer.hpp"
#include "ct/marginalization.hpp"
#include "ct/residuals_gnss.hpp"
#include "ct/residuals_imu.hpp"
#include "ct/residuals_lidar.hpp"
#include "ct/residuals_visual.hpp"
#include "ct/spline_window.hpp"
#include "ct/visual_map.hpp"
#include "ct/window_problem.hpp"
#include "meridian/calib/calibration_set.hpp"
#include "meridian/calib/camera_model.hpp"
#include "meridian/common/frontend_diagnostics.hpp"
#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/nav_state.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"
#include "meridian/frontend/ifrontend.hpp"
#include "meridian/preprocess/imu_init.hpp"

namespace ceres {
class Problem;
}  // namespace ceres

namespace meridian {

class TelemetrySink;

// Continuous-time LiDAR-inertial front-end. The trajectory is a split SO(3)xR^3
// cubic B-spline solved over a sliding window of control points; IMU samples,
// LiDAR point-to-plane associations, bias random-walk ties, and a marginalization
// prior form a single Ceres problem per sweep. Emitted keyframes carry
// frontend_kind 1.
//
// The estimation frame F_e is imu_link, tagged Frame::Body on emitted quantities;
// the world/odom frame is Frame::Odom. Internal 6-DoF tangents are translation-
// first [rho; phi]; the one rotation-first block is KeyframePacket::constraint_cov,
// reordered exactly once at pack time via the oracle's shared static helper.
class CtFrontEnd final : public IFrontEnd {
public:
  CtFrontEnd(const FrontendConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
             TelemetrySink* telemetry, bool deterministic = false);
  ~CtFrontEnd() override;

  void set_calibration(std::shared_ptr<const CalibrationSet> calib) override;
  void ingest(const PreprocessedGroup& group) override;
  void ingest_imu_live(const ImuSample& imu) override;
  void apply_correction(const GraphUpdate& update) override;
  NavState live_state() const override;
  void set_keyframe_sink(KeyframeSink sink) override;
  FrontEndDiagnostics diagnostics() const override;

  // Test-only: force the window solve to a single Ceres thread even on the live
  // (deterministic_==false) path. Lets a test drive the async keyframe worker while
  // keeping the solved trajectory bit-reproducible against a synchronous reference, so
  // the only variable under comparison is the covariance code path. No effect in
  // production (never called there).
  void set_force_single_thread_for_test(bool on) { force_single_thread_ = on; }

  // Test-only bit-exact parity probe. When enabled, every live keyframe sweep records
  // the pose marginal computed TWO ways on the identical solved state: (a) the
  // synchronous routine on the live ceres::Problem, and (b) the worker routine on the
  // rebuilt snapshot clone. The test asserts the two 6x6 blocks are bit-identical.
  struct ParityProbeResult {
    Eigen::Matrix<double, 6, 6> sync_cov;    // path (a): live problem
    Eigen::Matrix<double, 6, 6> worker_cov;  // path (b): rebuilt snapshot
    bool absolute_prior = false;
    bool sync_ok = false;
    bool worker_ok = false;
  };
  void set_parity_probe_for_test(bool on) { parity_probe_ = on; }
  const std::vector<ParityProbeResult>& parity_probe_results_for_test() const {
    return parity_probe_results_;
  }

  // Test-only: block until the async keyframe worker has emitted every keyframe
  // submitted so far, so a test can assert on the emitted packets without racing the
  // worker. No-op on the deterministic (synchronous) path.
  void drain_keyframes_for_test() {
    if (finalizer_) finalizer_->drain_for_test();
  }

  // Test-only: the carried marginalization prior (null until the first window slide)
  // plus a bias-block classifier over parameter-block pointers, so a test can measure
  // the prior's information restricted to the bias-knot subspace.
  const ct::MarginalizationPrior* prior_for_test() const { return prior_.get(); }
  bool is_bias_block_for_test(const double* p) const {
    for (int i = 0; bias_ && i < bias_->numKnots(); ++i) {
      if (p == bias_->gyroBlock(i) || p == bias_->accelBlock(i)) return true;
    }
    return false;
  }

  // Test-only: the spline and bias knot timeline laid by one steady-state extend.
  struct ExtendKnots {
    std::vector<Timestamp> knot_t;
    std::vector<std::array<double, 4>> so3;  // quaternion storage (x, y, z, w)
    std::vector<std::array<double, 3>> r3;
    std::vector<Timestamp> bias_t;
    std::vector<std::array<double, 3>> bias_gyro;
    std::vector<std::array<double, 3>> bias_accel;
  };

  // Test-only: run one steady-state extend on a clone of the live window, returning the
  // resulting knot timeline. With use_production the production dispatch (steadyState-
  // Extend) runs; otherwise the uniform legacyExtend reference runs. The live window and
  // all extend bookkeeping are left untouched, so a test can invoke both from one
  // identical state and assert the dispatched path equals the uniform path knot-for-knot
  // whenever the control-point cap is one.
  ExtendKnots probe_extend_for_test(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end,
                                    bool use_production);
  Timestamp last_solved_t_for_test() const { return last_solved_t_; }

  // Test-only: the spline-knot drop set each slide computed, paired with the
  // single-segment reference [leadingKnotIndex(t_begin), knotTime(lead)+knot_dt) it
  // must reduce to when exactly one outer segment departs. Both are captured at the
  // identical pre-slide spline state, so a test can assert one-segment byte-identity
  // (unified == legacy) and multi-segment coverage (unified spans every departed
  // segment) directly. Recording is off in production.
  struct SlideDropRecord {
    std::vector<double*> unified;        // [lower, D) so3/r3 knot pointers, in drop order
    std::vector<double*> legacy_oracle;  // [lead, seg_end) so3/r3 knot pointers, in drop order
  };
  void set_record_slide_drops_for_test(bool on) { record_slide_drops_ = on; }
  const std::vector<SlideDropRecord>& slide_drop_records_for_test() const {
    return slide_drop_records_;
  }

private:
  // Integrate the group's IMU forward from the spline's last solved pose/velocity
  // using the current bias and gravity estimates (constant body rate, constant
  // world acceleration per interval). Returns a pose-sampling closure for the
  // spline extension seed and advances the IMU-integration anchor to t_end.
  std::function<Pose(Timestamp)> buildSeed(const std::vector<ImuSample>& imu, Timestamp t_begin,
                                           Timestamp t_end);

  // Rebuilds the window state at `t_begin`: spline and bias knots reopened at the
  // current anchor, the stale marginalization prior (whose blocks point into the
  // discarded knots) dropped, bookkeeping reset. Gravity and the local map are
  // preserved. The anchor itself is the caller's responsibility.
  void restartWindow(Timestamp t_begin);

  // Unbridgeable-gap recovery: predicts the anchor across the data hole on constant
  // velocity, rebuilds the window there, and re-seeds spline + map from this sweep
  // WITHOUT solving it — directly after a reseed the gauge pins hold most of the only
  // evaluable segment, so a solve would warp the sweep; the next sweep solves on a
  // fully extendable window instead.
  void reseedAfterGap(const PreprocessedGroup& group, Timestamp t_begin, Timestamp t_end);

  // Per-sweep visual context handed to solveWindow so the photometric residuals enter
  // the same Ceres problem as LiDAR/IMU. Null fields mean the visual stage is off this
  // sweep. The exposure chain and pyramid view are owned by the caller (ingest) so
  // their parameter-block storage and backing cv::Mat outlive the solve.
  struct VisualContext {
    const ct::ImagePyramidView* img = nullptr;
    ct::ExposureChain* expo = nullptr;
    Timestamp t_image = 0;
    ct::VisualAssocStats stats;             // peak-accepted outer round
    std::vector<ct::VisualUsedPoint> used;  // tracked patches from that round (overlay)
  };

  // One gated GNSS fix ready to enter the cost: the fix's PTP time, its position in the
  // world frame (ENU-converted via the anchor), the antenna lever-arm at solve time, and
  // the 3x3 whitening matrix from the floored covariance. Carried from the pre-solve
  // gating into solveWindow so the residual is rebuilt each outer round at no re-gate.
  struct GnssMeasurement {
    Timestamp t_fix = 0;
    Eigen::Vector3d z_world = Eigen::Vector3d::Zero();
    Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();
  };

  // Per-sweep GNSS context handed to solveWindow. Empty `fixes` means the GNSS stage is
  // off this sweep (config disabled or no fixes present); the antenna lever-arm is read
  // once at gating time. Only populated on the active GNSS path.
  struct GnssContext {
    std::vector<GnssMeasurement> fixes;
    Eigen::Vector3d t_fe_ant = Eigen::Vector3d::Zero();
  };

  // Build, solve, re-associate, and re-solve the window problem for one sweep.
  // Returns the number of accepted LiDAR hits in the final solve. `scan_ds` holds
  // the downsampled raw scan (per-point times are relative to the scan start). When
  // `vis` is non-null and active, the photometric residuals are added each outer round.
  // When `gnss` is non-null and carries fixes, the absolute-position residuals are added
  // each outer round.
  int solveWindow(const PreprocessedGroup& group, Timestamp t_begin, Timestamp t_end,
                  std::vector<LidarPoint>* scan_ds, std::vector<ct::LidarHit>* hits,
                  VisualContext* vis, const GnssContext* gnss);

  // Gate the group's GNSS fixes (config use AND fixes present) against the current
  // seeded spline pose: convert each fix LLA -> world via the ENU anchor (anchoring on
  // the first accepted fix), floor its covariance by fix type, apply the Mahalanobis
  // innovation gate and re-acquisition persistence, and collect the survivors into
  // *ctx. Lazily allocates the anchor/gate the first time the active path runs, so a
  // disabled or fix-free build never touches GNSS state. Emits the GNSS gating telemetry.
  void gateGnss(const PreprocessedGroup& group, Timestamp t_end, GnssContext* ctx);

  // Bias knot cadence from config, floored at one millisecond.
  Duration biasKnotDt() const;

  // Per-segment control-point cap: cfg_.ncp.n_cp_max (floored at 1) when adaptive
  // density is enabled, 1 otherwise (the uniform single-knot spline).
  int maxControlPoints() const;

  // Adaptive-density (ncp) window extension for one steady-state sweep: IMU warm
  // start, tail re-knot of the provisional segments, reseed of surviving pinned
  // knots, gear-driven extension, bias extension, and the IMU-only warm-start
  // mini-solve over the freshly laid tail. Replaces the uniform extend block when
  // cfg_.ncp.enabled.
  void ncpExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end);

  // One steady-state sweep's window extension. Dispatches to ncpExtend when adaptive
  // density is enabled AND the cap permits more than one control point; otherwise the
  // segment count is pinned at one and this routes to the uniform legacyExtend, which
  // is bit-identical to a disabled build.
  void steadyStateExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end);

  // Uniform single-knot window extension for one steady-state sweep: IMU warm start,
  // one outer segment per nominal cadence, and a reseed of the unsupported trailing
  // knots. The extend the disabled and capped-at-one builds use.
  void legacyExtend(const MeasureGroup& mg, Timestamp t_begin, Timestamp t_end);

  // First resident knot index at or past the end of the segment containing t_begin.
  // Every knot strictly before it leaves the residual support once the window steps
  // past t_begin: the slide marginalizes the departing group up to (not including)
  // this index, and the no-prior gauge pins anchor on it. Enumerates indices; never
  // time-samples.
  int firstKeptKnotIndex(Timestamp t_begin) const;

  // Warp the downsampled scan to world at each point's solved spline time and
  // insert it into the local map. `t0_scan` is the scan-start time the per-point
  // offsets are relative to.
  void updateMap(const std::vector<LidarPoint>& scan_ds, Timestamp t0_scan);

  // Marginalize every control-point knot whose cubic support has fully left the
  // window this sweep -- from the oldest knot the carried prior still anchors up to
  // (not including) firstKeptKnotIndex(t_begin) -- folding their LiDAR/IMU/prior
  // information into the new prior via linearizeBlocks + fromSchur. One departing
  // outer segment collapses the bound to that segment's knots (the fixed-cadence
  // drop); a sweep crossing several outer segments retires them all in one Schur
  // build, so the kept set cannot grow without bound. Fires only once the trajectory
  // has grown past the window length.
  void slideWindow(ceres::Problem& problem, Timestamp t_begin, Timestamp t_end);

  // Per-axis observability of the just-solved keyframe pose from the LiDAR hits'
  // point-to-plane Jacobian rows, kept translation-first [tx,ty,tz,rx,ry,rz].
  ObservabilityReport computeObservability(const Pose& T_world_body,
                                           const std::vector<ct::LidarHit>& hits) const;

  // Pose marginal (translation-first [rho; phi]) of the body pose at `stamp` from
  // the LiDAR hit information block; the fallback when the full window posterior is
  // unavailable (no hits / pre-solve / non-invertible blanket).
  Eigen::Matrix<double, 6, 6> poseMarginal(const Pose& T_world_body,
                                           const std::vector<ct::LidarHit>& hits) const;

  // Pose-block marginal covariance (translation-first [rho; phi]) of T_W_Fe(stamp)
  // from the solved window posterior: inverts the joint information of the pose
  // segment's knots over every residual touching them (LiDAR + IMU + prior), then
  // chains through the spline pose Jacobian at stamp. Returns false when the joint
  // information is rank-deficient so the caller can fall back to poseMarginal.
  bool poseMarginalFromProblem(ceres::Problem& problem, Timestamp stamp,
                               Eigen::Matrix<double, 6, 6>* cov);

  // Capture the final outer round's window-problem build inputs (value copies, so the
  // record is self-contained once the live problem and spline have moved on). Filled on
  // the live path's keyframe sweeps so emitKeyframe can package an async finalize job;
  // unused on the deterministic path. `capped_hits`/`imu`/`patches` are the final round's
  // data; the decisions (motion-reg engagement, tail/gauge/pin indices) mirror that round.
  void captureFinalRound(Timestamp t_begin, Timestamp t_end, const std::vector<ImuSample>& imu,
                         const ct::ImuWeights& weights,
                         const std::vector<ct::LidarHit>& capped_hits,
                         const std::vector<ct::VisualPatchParams>& patches,
                         const VisualContext* vis, const GnssContext* gnss, bool motion_reg_engaged,
                         const std::vector<Timestamp>& motion_reg_times,
                         const std::vector<Timestamp>& tail_times, int first_pinned_knot,
                         int gauge_knot_idx);

  // Package the captured final round + partial packet into a KeyframeJob and submit it to
  // the finalizer; the worker fills constraint_cov. Live path only.
  void submitKeyframeJob(KeyframePacket&& pkt, const Pose& T_world_body, Timestamp stamp,
                         const std::vector<ct::LidarHit>& hits, bool absolute_prior);

  bool keyframeDue(const Pose& T_world_body, Timestamp stamp) const;
  void emitKeyframe(const Pose& T_world_body, Timestamp stamp,
                    const std::vector<LidarPoint>& scan_ds, Timestamp t0_scan,
                    const std::vector<ct::LidarHit>& hits);

  // Cache T_fe_lidar (= T_imu_lidar) from the calibration snapshot.
  void refreshExtrinsic();

  // Rebuild the camera model + camera extrinsic from the calibration snapshot and
  // (re)create the visual map. Leaves the model invalid when no usable camera
  // intrinsics are present, which gates the whole visual stage off.
  void refreshCamera();

  // Decode the raw CameraFrame to a single-channel intensity image and build the
  // pyramid the photometric residual samples. Exposure normalization is NOT applied
  // here: the inverse exposure is an optimised state, so the raw intensity pyramid is
  // what the residual's exposure model multiplies. Returns an empty vector when the
  // frame carries no decodable image.
  std::vector<cv::Mat> buildImagePyramid(const CameraFrame& frame) const;

  // True when the visual stage may run this sweep: the config gate is on, the camera
  // model is valid, the camera extrinsic is known, and the group carries an image.
  bool visualEnabled(const PreprocessedGroup& group) const;

  // Publish the tracked-patch overlay: the patches that contributed a residual this
  // sweep, drawn on the current intensity image (box scaled by pyramid level, coloured
  // by photometric residual), on the reserved telemetry image key.
  void publishPatchOverlay(const cv::Mat& intensity, Timestamp t_image,
                           const std::vector<ct::VisualUsedPoint>& used);

  // IMU sqrt-information weights from the calibration noise densities, converted to
  // per-sample stds using the IMU sample period (seconds).
  ct::ImuWeights imuWeights(double imu_dt_s) const;

  // Weak translation axes (unit world vectors) of the current LiDAR association set:
  // the eigenvectors of the point-to-plane translation information block whose
  // eigenvalue is below the degeneracy floor. A LiDAR factor constrains translation
  // along its plane normal, so a hit whose normal aligns with one of these axes is the
  // scarce evidence on a weak axis and is exempted from the factor cap.
  std::vector<Eigen::Vector3d> weakTranslationAxes(const std::vector<ct::LidarHit>& hits) const;

  // Knot-midpoint times of the segments spanning [t_begin, t_end], the evaluation
  // points for the under-excitation regularizer.
  std::vector<Timestamp> segmentMidpointTimes(Timestamp t_begin, Timestamp t_end) const;

  FrontendConfig cfg_;
  TelemetrySink* telemetry_ = nullptr;
  // Replay/sync determinism signal: when set, solveWindow ignores the wall-clock
  // deadline and runs the fixed iteration schedule so a recorded run is reproducible.
  bool deterministic_ = false;
  // Compute the keyframe covariance inline on the sweep thread instead of on the
  // async finalizer. Always true under deterministic_; additionally settable via the
  // MERIDIAN_SYNC_KEYFRAME_COV env var as a live A/B diagnostic knob (same binary,
  // sync-vs-async covariance path).
  bool sync_cov_ = false;
  // First stamp seen by the bootstrap path; drives the moving-start init fallback.
  Timestamp bootstrap_first_t_ = 0;
  // Test seam: pins the window solve to one Ceres thread on the live path (see
  // set_force_single_thread_for_test). Always false in production.
  bool force_single_thread_ = false;

  std::shared_ptr<const CalibrationSet> calib_;
  std::uint32_t calib_version_ = 0;
  Pose T_fe_lidar_;  // LiDAR sensor -> F_e (imu_link)
  bool have_extrinsic_ = false;

  // ---- GNSS stage (conservative absolute position). The anchor and gate are direct
  // value members (no heap allocation) so the inactive path — config disabled, no fix
  // topic, or zero accepted fixes — perturbs neither the heap layout nor the solver, and
  // the trajectory is bit-identical to a build without this stage. The gate is
  // constructed lazily into place from cfg the first time the active path runs.
  ct::EnuAnchor gnss_anchor_;
  std::optional<ct::GnssGate> gnss_gate_;
  // One-time informational "gnss inactive" state, emitted once on the inactive path so
  // the operator can tell a disabled/fix-free build from a fusing one without any other
  // behaviour change.
  bool gnss_inactive_logged_ = false;

  // ---- camera / visual stage ----
  CameraModel cam_model_;  // invalid until valid intrinsics are configured
  Pose T_fe_cam_;          // camera optical frame -> F_e (imu_link)
  bool have_cam_extrinsic_ = false;
  double inv_expo_prior_ = 1.0;  // exposure prior from the camera intrinsics
  double inv_expo_std_ = 0.0;    // exposure prior 1-sigma (0 => fixed)
  // Carried inverse exposure across sweeps: each sweep's exposure block is seeded from
  // and prior-anchored toward this value, realising the exposure random walk across the
  // window through the prior; updated to the solved tau after each sweep.
  double inv_expo_ = 1.0;
  std::unique_ptr<ct::VisualMap> vmap_;
  // One-time loud telemetry/log when the visual stage is requested but the intrinsics
  // are invalid, so a misconfigured camera is visible rather than silently LIO-only.
  bool visual_disabled_logged_ = false;

  // ---- trajectory + auxiliary state ----
  std::unique_ptr<SplineWindow> spline_;
  std::unique_ptr<ct::BiasKnots> bias_;
  std::unique_ptr<ct::GravityBlock> gravity_;
  std::unique_ptr<ct::LidarLocalMap> map_;
  std::unique_ptr<ct::MarginalizationPrior> prior_;

  // Static-window IMU initializer (Welford mean + motion gate). Bootstrap defers
  // until a window of length init_time_s passes the gate, so a cold start on a moving
  // platform never seeds a tilted gravity from a non-static mean. Built lazily on the
  // first group with IMU so its target count can use the measured sample period.
  std::unique_ptr<ImuInitializer> imu_init_;

  bool bootstrapped_ = false;    // gravity + bias initialised, spline + map seeded
  bool have_solved_ = false;     // at least one full window solve completed
  Timestamp last_solved_t_ = 0;  // spline end of the last solve (= prev t_end)
  Timestamp window_dt_ns_ = 0;   // outer knot cadence in ns

  // The t_begin of the most recent sweep whose slide advanced the window. Each slide
  // marginalizes the leading knot of the segment at that t_begin into prior_, so a
  // later sweep must have t_begin past this before it drops the next knot. Zero until
  // the first slide.
  Timestamp window_floor_t_ = 0;

  // The t_begin of the most recent sweep that actually REBUILT the prior (slides can
  // advance window_floor_t_ without building one). Structured prior forgetting
  // integrates its process noise over the span since this stamp, so skipped builds
  // accumulate into the next one instead of being forgotten. Zero until the first
  // build and after a window restart.
  Timestamp last_prior_build_t_ = 0;

  // Real time of the spline's first knot at bootstrap, kept fixed for the whole mission
  // (front-trimming advances minTime() but not this). The marginalization warm-up gates
  // on it so the first knot is dropped only after the trajectory has grown longer than
  // the information window, exactly as before trimming was added.
  Timestamp traj_start_t_ = 0;

  // Bias-integration anchor: the last solved pose/velocity the IMU seed integrates
  // forward from (decoupled from the spline so a failed solve cannot corrupt it).
  Pose anchor_pose_;
  Eigen::Vector3d anchor_vel_ = Eigen::Vector3d::Zero();
  // Terminal velocity / body rate of the latest IMU-integrated seed, captured by
  // buildSeed; they parameterize the tail anchors (constant-extrapolation prediction
  // for the not-yet-measured span).
  Eigen::Vector3d seed_vel_end_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d seed_omega_end_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_bg_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d anchor_ba_ = Eigen::Vector3d::Zero();

  // Live-output state advanced by ingest_imu_live without touching solver state.
  NavState live_state_;
  Timestamp live_stamp_ = 0;

  KeyframeSink keyframe_sink_;
  std::uint64_t next_kf_id_ = 0;
  bool have_prev_kf_ = false;
  std::uint64_t prev_kf_id_ = 0;
  Pose prev_kf_pose_;  // T_world_body of the previous keyframe
  Timestamp prev_kf_stamp_ = 0;
  Eigen::Matrix<double, 6, 6> prev_kf_pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity();

  ObservabilityReport last_obs_;
  FrontEndDiagnostics diag_;

  // Adaptive-density hysteresis state: the gear of the most recently FINALIZED outer
  // segment (a segment whose IMU has arrived), updated in lay order inside ncpExtend.
  // Seeds the next gear decision so a band-down needs the deeper hysteresis drop, and
  // provisional (data-free) segments hold it. One until the first finalized segment,
  // and always one with adaptive density disabled.
  int last_n_cp_ = 1;

  // Max finalized gear laid this sweep (frontend/ct/n_cp telemetry). Equals
  // last_n_cp_ when no segment finalized this sweep; one with ncp disabled.
  int sweep_max_gear_ = 1;

  // Deque index of the first unsupported tail knot the previous solve held constant
  // (the pinning rule in solveWindow), adjusted by the slide's front-trim so it keeps
  // addressing the same knots. ncpExtend reseeds from it and the warm-start
  // mini-solve frees exactly the knots at/after it. -1 (invalid) until the first
  // solve and after a window restart.
  int last_first_pinned_ = -1;

  // LiDAR factor-cap accounting from the final outer step of the last solve (kept and
  // dropped residual counts), surfaced as telemetry.
  ct::LidarCapStats last_cap_stats_;

  // Sweep counter gating the per-family residual-stats probe: the extra per-family
  // Problem::Evaluate sweep runs only every 5th sweep or on a keyframe sweep (every
  // sweep when the solver debug group is on), so its cost stays bounded. Incremented
  // once per solveWindow.
  std::uint64_t resid_probe_counter_ = 0;

  // Stamp of the last frontend/spline/path_sample pose emitted, so the path sampler
  // walks forward at the configured cadence across sweeps without duplicates. Zero
  // until the first sample.
  Timestamp last_path_t_ = 0;

  // Pose-block marginal covariance (translation-first [rho; phi]) of the body pose
  // at the last solved sweep end, read from the window posterior in solveWindow
  // while the Ceres problem is alive. false until the first successful extraction;
  // emitKeyframe prefers it and falls back to the LiDAR-only poseMarginal otherwise.
  Eigen::Matrix<double, 6, 6> last_pose_cov_ = Eigen::Matrix<double, 6, 6>::Identity();
  bool have_pose_cov_ = false;

  // ---- async keyframe-cov worker (live path only) ----
  // On the live path the per-keyframe constraint covariance is computed off T2 by this
  // worker: solveWindow captures the final round's build inputs into capture_, emitKeyframe
  // packages them into a KeyframeJob, and the worker rebuilds the (bit-identical) problem
  // and recovers the marginal a full keyframe interval before L3 needs it. The
  // deterministic path leaves finalizer_ null and computes the marginal inline, preserving
  // replay==live bit-exactness.
  std::unique_ptr<KeyframeFinalizer> finalizer_;

  // The final outer round's window-problem build inputs, captured by value on a live
  // keyframe sweep. Held as a member so emitKeyframe can move it into the job; the
  // contained pointers are rewired to the job's clones inside submitKeyframeJob.
  struct WindowBuildCapture {
    bool valid = false;
    ct::WindowProblemInputs inputs;
    std::unique_ptr<SplineWindow> spline_clone;
    std::unique_ptr<ct::BiasKnots> bias_clone;
    ct::GravityBlock gravity_clone{Eigen::Vector3d(0, 0, -1), 9.81};
    std::unique_ptr<ct::MarginalizationPrior> prior_clone;
    std::vector<ct::LidarHit> capped_hits;
    std::vector<ImuSample> imu;
  };
  WindowBuildCapture capture_;

  // Test-only parity probe state (see set_parity_probe_for_test).
  bool parity_probe_ = false;
  std::vector<ParityProbeResult> parity_probe_results_;

  // Test-only slide-drop recording (see set_record_slide_drops_for_test).
  bool record_slide_drops_ = false;
  std::vector<SlideDropRecord> slide_drop_records_;

  // A hard window reseed carried the pose across a data hole on a constant-velocity
  // prediction whose error no covariance accounts for, so no relative keyframe edge
  // may span it: the next keyframe is emitted as an AbsolutePrior chain break instead.
  bool kf_chain_broken_ = false;
};

}  // namespace meridian
