#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/imu.hpp"
#include "meridian/local/lidar_registration.hpp"
#include "meridian/local/pipeline_observability.hpp"
#include "meridian/local/visual_factor.hpp"

namespace meridian::local {

// Continuous-time IMU noise densities.  Squaring these values gives the
// covariance densities consumed by the private preintegration adapter.
struct ImuPreintegrationConfig {
  Eigen::Vector3d gravity_odom{0.0, 0.0, -9.80665};
  double accelerometer_noise_density_mps2_sqrt_hz{0.02};
  double gyroscope_noise_density_radps_sqrt_hz{0.0015};
  double accelerometer_bias_random_walk_mps3_sqrt_hz{0.0002};
  double gyroscope_bias_random_walk_radps2_sqrt_hz{0.00002};
  double integration_noise_density{1.0e-8};
  // Numerical/linearization covariance inside each CombinedImuFactor. This is
  // deliberately separate from the physical startup-bias calibration prior;
  // the latter must never be copied onto every preintegration edge.
  double preintegration_accelerometer_bias_variance_m2ps4{1.0e-5};
  double preintegration_gyroscope_bias_variance_rad2ps2{1.0e-5};
  // Sensor-calibration mean and uncertainty for the first bias in an odom
  // epoch. The moving initializer consumes the same values; the tracking
  // graph is initialized at its accepted posterior rather than adding a
  // second copy of this prior.
  Eigen::Vector3d initial_accelerometer_bias_mean_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d initial_gyroscope_bias_mean_radps{Eigen::Vector3d::Zero()};
  double initial_accelerometer_bias_sigma_mps2{0.1};
  double initial_gyroscope_bias_sigma_radps{0.01};
};

struct LocalGraphConfig {
  ImuPreintegrationConfig imu;

  // The time lag is the normal retention policy. The count is a hard safety
  // bound; when it is reached the oldest complete navigation state is
  // marginalized even if the time lag has not elapsed.
  std::size_t maximum_navigation_states{64};
  core::Duration target_fixed_lag{5'000'000'000LL};

  double pose_rotation_relinearization_rad{0.025};
  double pose_translation_relinearization_m{0.05};
  double velocity_relinearization_mps{0.05};
  double accelerometer_bias_relinearization_mps2{0.005};
  double gyroscope_bias_relinearization_radps{0.0005};
  double visual_log_inverse_range_relinearization{0.025};

  // Every sensor transaction follows one bounded nonlinear solve path. These
  // values are numerical floors, not claims about physically resolvable state
  // changes. For an IMU interval dt, the solver uses the larger of each floor
  // and convergence_sigma_fraction times the calibrated one-axis process
  // sigma. Continuous white acceleration is propagated as
  // sigma_p = n_a * dt^(3/2) / sqrt(3) and sigma_v = n_a * sqrt(dt);
  // small-angle rotation and both bias random walks use n * sqrt(dt). Visual
  // log-inverse-range has no IMU process model and therefore keeps its floor.
  std::size_t maximum_nonlinear_iterations{8U};
  double nonlinear_convergence_sigma_fraction{0.25};
  double nonlinear_translation_convergence_m{1.0e-4};
  double nonlinear_rotation_convergence_rad{1.0e-4};
  double nonlinear_velocity_convergence_mps{1.0e-4};
  double nonlinear_accelerometer_bias_convergence_mps2{1.0e-5};
  double nonlinear_gyroscope_bias_convergence_radps{1.0e-6};
  double nonlinear_visual_log_inverse_range_convergence{1.0e-5};

  // A Gauss--Newton update is accepted only when the complete nonlinear
  // objective does not increase. Otherwise deterministic manifold
  // backtracking first tests that direction, then the same linearized graph's
  // Cauchy direction. This remains one atomic graph solve rather than an
  // alternate runtime implementation. The bound applies independently to
  // each positive direction.
  std::size_t maximum_nonlinear_backtracking_steps{8U};
  double nonlinear_backtracking_reduction{0.5};
  double nonlinear_objective_absolute_convergence{1.0e-12};
  double nonlinear_objective_relative_convergence{1.0e-9};

  // Cumulative corrections are measured from the IMU-predicted transaction
  // seed over every active navigation pose, not from the previous accepted
  // graph state.  These gates therefore bound optimizer corrections without
  // imposing an artificial bound on platform motion.
  double maximum_transaction_translation_correction_m{1.0};
  double maximum_transaction_rotation_correction_rad{0.35};

  // Explicit complete-objective allowance for solver roundoff and declared
  // bounded model nonsmoothness. Finite-state and physical correction gates
  // remain independently binding; this is not a convergence tolerance.
  double complete_objective_nonsmooth_absolute_allowance{1.0e-9};
  double complete_objective_nonsmooth_relative_allowance{1.0e-3};

  // Bounds transaction-owned visual work independently of frontend sizing.
  // These are admission limits, not alternate processing paths.
  std::size_t maximum_visual_landmarks_per_transaction{512U};
  std::size_t maximum_visual_factors_per_transaction{4096U};
  std::size_t maximum_visual_factor_retirements_per_transaction{4096U};
  // A LiDAR sweep may connect to several distinct live target states, but the
  // complete set is one bounded, atomic graph transaction.
  std::size_t maximum_direct_lidar_factors_per_transaction{15U};
  // A finalized-map factor may retain points from many immutable sweeps.  The
  // unique shared owners carried into one unary factor are independently
  // bounded so provenance validation and archival cannot grow without limit.
  std::size_t maximum_finalized_lidar_owners_per_factor{512U};

  // Asynchronous frontends may attach a completed factor batch to navigation
  // states that are already live in the lag window. The graph retains exact
  // typed payloads for active batches and a bounded terminal journal for
  // downstream provenance archival. Only the newest bounded set remains
  // retractable; older active batches are sealed until state finality.
  std::size_t maximum_active_factor_batches{256U};
  std::size_t maximum_removable_factor_batches{32U};
  std::size_t maximum_factor_batches_per_removal_transaction{16U};
  std::size_t maximum_terminal_factor_batch_records{256U};
};

enum class LocalGraphMarginalizationStatus {
  InactiveWithinLag,
  AppliedNominalLag,
  AppliedWindowCap,
};

enum class LocalGraphCapacityStatus {
  WithinHardCap,
  AtHardCap,
};

// A navigation state becomes final exactly when the successful graph
// transaction removes it from the fixed-lag Bayes tree.  This ROS-free value
// is an immutable snapshot seam for downstream staging; it deliberately does
// not depend on meridian_global or imply that factor journaling/condensation
// has happened.
struct LocalGraphFinalizedState {
  core::StateId state;
  core::FusionTime exact_time;
  core::OdomEpoch odom_epoch;
  core::LocalGraphRevision final_revision;
  core::NavStateEstimate final_estimate;
  core::PoseCovariance pose_covariance;
};

// Graph-visible identity paired with the solver-neutral landmark segment that
// became immutable in this transaction.  The graph identity is needed by the
// estimator to route finality back to the owning camera lane; downstream
// global consumers retain the segment value.
struct LocalGraphFinalizedVisualLandmark {
  VisualLandmarkId landmark;
  core::FinalizedLandmarkSegment segment;
};

// One settled pose from a successful graph revision.  The commit publishes a
// complete chronological snapshot of every navigation state that participated
// in the transaction: states removed by this revision carry their immutable
// pre-marginalization pose, while retained states carry their exact
// post-marginalization pose.  Scan-local caches can therefore refresh their
// pose metadata without depending on GTSAM or querying mutable solver state.
struct LocalGraphPoseSnapshot {
  core::StateId state;
  core::FusionTime exact_time;
  core::Pose3d T_odom_imu;
};

enum class LocalGraphFinalityStatus {
  InactiveWithinWindow,
  PublishedNominalLag,
  PublishedWindowCap,
};

struct LocalGraphFinalityDiagnostics {
  LocalGraphFinalityStatus status{LocalGraphFinalityStatus::InactiveWithinWindow};
  // Reservation and covariance extraction happen against the settled
  // pre-marginalization candidate.  On every successful commit these counts
  // must equal finalized_states.size().
  std::size_t records_reserved{};
  std::size_t pose_covariances_computed{};
  std::optional<core::FusionTime> nominal_cutoff;
  std::optional<core::StateId> oldest_finalized_state;
  std::optional<core::StateId> newest_finalized_state;
  std::optional<core::FusionTime> oldest_finalized_time;
  std::optional<core::FusionTime> newest_finalized_time;
};

struct LocalSolveReport {
  std::size_t navigation_states{};
  std::size_t joint_initial_priors{};
  std::size_t combined_imu_factors{};
  std::size_t active_factor_batches{};
  std::size_t active_lidar_direct_batch_factors{};
  std::size_t factor_batches_added{};
  std::size_t factor_batches_removed{};
  std::size_t factor_batches_sealed{};
  std::size_t factor_batches_finalized{};
  std::size_t lidar_direct_batch_factors_added{};
  std::size_t lidar_direct_batch_factors_removed{};
  std::size_t lidar_direct_batch_factors_sealed{};
  std::size_t lidar_direct_batch_factors_finalized{};
  std::size_t active_visual_landmarks{};
  std::size_t active_visual_factors{};
  std::size_t visual_landmarks_added{};
  std::size_t visual_factors_added{};
  std::size_t visual_factors_retired{};
  std::size_t visual_landmarks_marginalized{};
  std::size_t marginalized_navigation_states{};
  std::size_t marginal_factors_added{};
  std::size_t factors_deleted_by_marginalization{};
  std::size_t variables_relinearized{};
  std::size_t variables_reeliminated{};
  std::size_t factors_recalculated{};
  std::size_t cliques{};
  std::size_t nonlinear_iterations{};
  double convergence_interval_duration_s{};
  double convergence_sigma_fraction{};
  double effective_translation_convergence_m{};
  double effective_rotation_convergence_rad{};
  double effective_velocity_convergence_mps{};
  double effective_accelerometer_bias_convergence_mps2{};
  double effective_gyroscope_bias_convergence_radps{};
  double effective_visual_log_inverse_range_convergence{};
  double last_iteration_translation_correction_m{};
  double last_iteration_rotation_correction_rad{};
  double last_iteration_velocity_correction_mps{};
  double last_iteration_accelerometer_bias_correction_mps2{};
  double last_iteration_gyroscope_bias_correction_radps{};
  double last_iteration_visual_log_inverse_range_correction{};
  double maximum_iteration_translation_correction_m{};
  double maximum_iteration_rotation_correction_rad{};
  double maximum_transaction_translation_correction_m{};
  double maximum_transaction_rotation_correction_rad{};
  double marginalization_translation_correction_m{};
  double marginalization_rotation_correction_rad{};
  std::size_t nonlinear_full_steps_rejected{};
  std::size_t nonlinear_backtracking_trials{};
  std::size_t nonlinear_cauchy_directions_attempted{};
  std::size_t nonlinear_cauchy_steps_accepted{};
  std::size_t nonlinear_cauchy_backtracking_trials{};
  std::size_t nonlinear_zero_step_terminations{};
  double minimum_nonlinear_step_scale{1.0};
  double last_iteration_objective_change{};
  std::optional<double> error_before;
  std::optional<double> error_after;
  bool qr_factorization{true};
  std::size_t relinearize_skip{1};
  LocalGraphMarginalizationStatus marginalization{
      LocalGraphMarginalizationStatus::InactiveWithinLag};
  LocalGraphCapacityStatus capacity{LocalGraphCapacityStatus::WithinHardCap};
};

// One immutable direct point ICP pair. The snapshot owns only this target's
// rows from the common source sweep. information_scale contains only the
// batch-local target-reuse/IMU-conditioning treatment. Finalized-map owner
// uncertainty must never leak into a live binary factor.
struct LidarDirectFactorPairSpec {
  core::StateId target_state;
  core::FusionTime target_time;
  std::shared_ptr<const LidarFactorSnapshot> snapshot;
  double information_scale{};
};

// One immutable unary direct point ICP term against the graph-final odom map.
// The snapshot retains exact finalized-sweep ownership for every frozen row;
// only source_state remains an optimizer endpoint. information_scale contains
// both the batch-local correlation treatment and this map channel's effective
// inflation: max(canonical finalized-owner pose-covariance inflation,
// configured correlation floor).
struct LidarFinalizedMapFactorSpec {
  std::shared_ptr<const FinalizedMapLidarFactorSnapshot> snapshot;
  // Explicit estimator policy needed for atomic graph-side reconstruction of
  // this factor's scale.  It applies only to the unary finalized-map channel.
  double configured_correlation_inflation_floor{1.0};
  double information_scale{};
};

// Exact accepted frontend result. Registration and association remain outside
// the graph; this report is immutable provenance, not an instruction to rerun
// registration from optimizer state.
struct DirectLidarRegistrationReport {
  LidarRegistrationTermination termination{LidarRegistrationTermination::Converged};
  double initial_robust_cost{};
  double final_robust_cost{};
  LidarRegistrationDiagnostics diagnostics;
  LidarRegistrationWorkCounters work;
  // Exact accepted frontend output before graph insertion. Keeping both
  // values separates registration correction from the later graph response
  // in runtime diagnostics without synthesizing a pose measurement factor.
  core::Pose3d T_odom_source;
  core::Pose3d source_right_correction;
};

// Asynchronous, sensor-pure direct LiDAR work against already-live poses. The
// The direct point ICP config seals one geometry-preserving source-sweep
// information ceiling; every retained pair receives the same correlation
// inflation because pair row ownership is disjoint.
// No pose summary, navigation state, or IMU factor is part of this payload.
struct LidarDirectFactorBatch {
  core::FactorBatchMetadata metadata;
  core::StateId source_state;
  core::FusionTime source_time;
  LidarRegistrationConfig registration;
  // Correlation inflation arising from live-target reuse and IMU-assisted
  // deskew/tracking. Revision 3 keeps this batch-wide value independent of
  // finalized-owner pose uncertainty; that uncertainty applies only to the
  // unary finalized-map factor.
  double base_covariance_inflation{6.0};
  DirectLidarRegistrationReport registration_report;
  std::vector<LidarDirectFactorPairSpec> pairs;
  // Canonical solver order is every live pair above, followed by this one
  // unary factor. A map-only batch is valid and remains removable until its
  // source state becomes final.
  std::optional<LidarFinalizedMapFactorSpec> finalized_map;
};

struct SensorFactorBatchRef {
  core::SensorInstanceId sensor;
  core::FactorBatchId batch_id;

  auto operator<=>(const SensorFactorBatchRef&) const = default;
};

enum class FactorBatchRemovalReason {
  SensorFailure,
  FrontendInvalidation,
};

struct FactorBatchRemovalRequest {
  std::vector<SensorFactorBatchRef> batches;
  FactorBatchRemovalReason reason{FactorBatchRemovalReason::SensorFailure};
};

enum class FactorBatchJournalStatus {
  Active,
  SealedByMarginalization,
  Removed,
  FinalizedByMarginalization,
};

// Exact provenance retained by the live graph. Terminal records are also
// emitted on LocalGraphCommit so a downstream durable journal can archive them
// before this bounded local history evicts them.
struct FactorBatchProvenance {
  LidarDirectFactorBatch batch;
  core::LocalGraphRevision inserted_revision;
  std::optional<core::LocalGraphRevision> sealed_revision;
  std::optional<core::LocalGraphRevision> terminal_revision;
  FactorBatchJournalStatus status{FactorBatchJournalStatus::Active};
  bool removable{};
  std::optional<FactorBatchRemovalReason> removal_reason;
};

struct FactorBatchJournalStats {
  std::size_t active_batches{};
  std::size_t active_lidar_direct_factors{};
  std::size_t removable_batches{};
  std::size_t sealed_batches{};
  std::size_t terminal_records{};
  std::size_t terminal_records_evicted{};
};

// Diagnostics for one binary target/source pair owned by a single LiDAR
// sweep transaction. Every field is either sealed by the immutable direct point ICP
// snapshot or read from the admitted private DirectLidarFactor; no registration
// diagnostics are synthesized inside the graph.
struct DirectLidarPairReport {
  core::StateId target_state;
  core::FusionTime target_reference_time;
  core::MeasurementId target_sweep;
  core::MeasurementId source_sweep;
  std::size_t source_point_count{};
  std::size_t correspondences{};
  std::size_t source_rows_excluded_by_ownership{};
  std::size_t candidate_voxel_lookups{};
  std::size_t candidate_points_examined{};
  double information_scale{};
  core::RankAwareInformation physical_information;
  core::ContentHash snapshot_checksum{};
};

struct DirectLidarFinalizedMapReport {
  core::MeasurementId source_sweep;
  std::size_t source_point_count{};
  std::size_t correspondences{};
  std::size_t source_rows_excluded_by_ownership{};
  std::size_t candidate_voxel_lookups{};
  std::size_t candidate_points_examined{};
  std::size_t unique_finalized_owners{};
  core::OdomEpoch map_odom_epoch;
  core::SensorInstanceId map_sensor;
  std::uint64_t map_version{};
  core::ContentHash map_checksum{};
  // Raw value reconstructed solely from the finalized owners and their pose
  // covariances.  Keep it separate from the configured correlation policy.
  double owner_pose_covariance_inflation{1.0};
  double configured_correlation_inflation_floor{1.0};
  double effective_covariance_inflation{1.0};
  double information_scale{};
  core::RankAwareInformation physical_information;
  core::ContentHash snapshot_checksum{};
};

struct LocalGraphCommit {
  core::OdomEpoch odom_epoch;
  core::LocalGraphRevision revision;
  core::LocalGraphRevision parent;
  core::StateId state;
  core::FusionTime state_time;
  core::NavStateEstimate estimate;
  NavigationCovariance covariance;
  std::vector<LocalGraphPoseSnapshot> navigation_poses;
  // Values are settled pre-marginalization snapshots for exactly the states
  // removed by this transaction.  Active/in-lag states never appear here.
  std::vector<LocalGraphFinalizedState> finalized_states;
  // Child factors precede their parent landmarks at the API boundary.  These
  // lists contain only graph-automatic removals; caller-requested visual
  // factor retirements are already acknowledged by the input transaction.
  std::vector<core::FactorId> finalized_visual_factors;
  std::vector<LocalGraphFinalizedVisualLandmark> finalized_visual_landmarks;
  LocalGraphFinalityDiagnostics finality;
  LocalSolveReport solve;
  std::optional<DirectLidarRegistrationReport> lidar_registration;
  std::vector<DirectLidarPairReport> lidar_pairs;
  std::optional<DirectLidarFinalizedMapReport> lidar_finalized_map;
  core::ObservationLineage lineage;
  // Insertions, removals, and finality transitions committed in this exact
  // graph revision. Every record carries the complete typed payload and common
  // provenance metadata.
  std::vector<FactorBatchProvenance> factor_batch_transitions;
};

// Full X/V/B estimate for one state that is still explicit in the bounded
// local graph. Frontends use this read-only snapshot to deskew or track a
// measurement that exactly shares an already-created timeline state. The
// observed revision identifies the graph estimate from which the snapshot was
// read; requesting it never creates a state or an IMU edge.
struct LocalGraphNavigationSnapshot {
  core::OdomEpoch odom_epoch;
  core::LocalGraphRevision observed_revision;
  core::StateId state;
  core::FusionTime exact_time;
  core::NavStateEstimate estimate;
};

struct LocalGraphInitialization {
  core::OdomEpoch odom_epoch;
  core::StateId state;
  core::FusionTime exact_time;
  core::NavStateEstimate estimate;
  NavigationCovariance covariance;
  core::ObservationLineage lineage;
};

struct ImuKnotAppend {
  core::StateId state;
  core::FusionTime exact_time;
  ImuInterval interval;
  core::ObservationLineage lineage;
};

// One estimator-time transaction. The IMU transition always owns the new
// navigation state. Optional sensor-pure LiDAR batches attach asynchronously
// through insertFactorBatch() only; graph transactions never run registration.
struct SensorKnotAppend {
  ImuKnotAppend navigation;
  std::optional<VisualFactorBatch> visual;
  std::vector<core::FactorId> visual_factor_retirements;
};

enum class LocalGraphErrorCode {
  InvalidConfig,
  AlreadyInitialized,
  NotInitialized,
  InvalidInitialization,
  InvalidCovariance,
  InvalidLineage,
  DuplicateState,
  NonMonotonicState,
  InvalidImuInterval,
  InexactImuBoundary,
  SaturatedImu,
  GapBridgeNotImplemented,
  TimestampUncertaintyNotImplemented,
  NavigationStateCapacity,
  SolverFailure,
  NonFiniteEstimate,
  NonlinearCostIncrease,
  PoseCorrectionLimit,
  NonlinearConvergenceFailure,
  MarginalCovarianceFailure,
  FinalityValidationFailure,
  VisualTransactionLimit,
  InvalidVisualBatch,
  DuplicateVisualLandmark,
  DuplicateVisualFactor,
  VisualStateUnavailable,
  VisualLandmarkUnavailable,
  VisualReferenceMismatch,
  VisualFactorRetirementUnavailable,
  VisualFactorEvaluationFailure,
  InvalidFactorBatch,
  FactorBatchEpochMismatch,
  DuplicateFactorBatch,
  StaleFactorBatch,
  FactorBatchLineageConflict,
  FactorBatchHealthUnavailable,
  FactorBatchCapacity,
  FactorBatchStateUnavailable,
  FactorBatchReferenceMismatch,
  FactorBatchRemovalUnavailable,
  NavigationStateUnavailable,
  GraphRevisionExhausted,
};

struct LocalGraphError {
  LocalGraphErrorCode code{};
  std::string detail;
  std::optional<DirectLidarRegistrationReport> lidar_registration;
  std::vector<DirectLidarPairReport> lidar_pairs;
  std::optional<DirectLidarFinalizedMapReport> lidar_finalized_map;
  // Candidate-only solve summary captured before a rejected transaction is
  // discarded. It must never be confused with the later IMU-only fallback
  // commit that may preserve estimator continuity.
  std::optional<LocalSolveReport> rejected_solve;
  std::optional<SensorFactorBatchRef> factor_batch;
};

// ROS-free owner of the local navigation graph.  GTSAM is intentionally hidden
// behind the pImpl so factors and solver choice cannot leak into framework APIs.
class LocalGraph {
public:
  explicit LocalGraph(LocalGraphConfig config = {});
  LocalGraph(LocalGraphConfig config, std::shared_ptr<LocalPipelineTimingRecorder> timing);
  ~LocalGraph();

  LocalGraph(LocalGraph&&) noexcept;
  LocalGraph& operator=(LocalGraph&&) noexcept;
  LocalGraph(const LocalGraph&) = delete;
  LocalGraph& operator=(const LocalGraph&) = delete;

  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> initialize(
      LocalGraphInitialization initialization);

  // Adds one state exactly at `exact_time`.  Every adjacent sample pair in the
  // supplied interval is integrated once with its midpoint measurement and
  // actual clipped duration into one CombinedImuFactor.
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> appendImuKnot(ImuKnotAppend append);

  // Generic atomic sensor transaction used by the estimator. Compatibility
  // entry points above delegate here.
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> appendSensorKnot(
      SensorKnotAppend append);

  // Attaches a sensor-pure typed batch to states that already exist in the
  // live lag window. No navigation state is created and no all-sensor bundle
  // is required. Candidate solve, provenance insertion, and sequence advance
  // commit atomically.
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> insertFactorBatch(
      LidarDirectFactorBatch batch);

  // Atomically retracts a bounded set of still-live recent batches. Factors
  // already condensed by state marginalization are intentionally immutable.
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> removeFactorBatches(
      FactorBatchRemovalRequest request);

  [[nodiscard]] std::optional<FactorBatchProvenance> factorBatchProvenance(
      SensorFactorBatchRef batch) const;
  [[nodiscard]] std::vector<FactorBatchProvenance> factorBatchJournal() const;
  [[nodiscard]] FactorBatchJournalStats factorBatchJournalStats() const noexcept;

  // Returns the current optimized X/V/B value for an explicit state in the
  // live lag window. Finalized/marginalized states are intentionally
  // unavailable because new frontend factors may no longer attach to them.
  [[nodiscard]] core::Result<LocalGraphNavigationSnapshot, LocalGraphError> navigationState(
      core::StateId state) const;

  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> estimate() const;
  [[nodiscard]] bool initialized() const noexcept;

private:
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> appendNavigationKnot(
      SensorKnotAppend append);
  [[nodiscard]] core::Result<LocalGraphCommit, LocalGraphError> applyFactorBatchTransaction(
      std::optional<LidarDirectFactorBatch> insertion,
      std::optional<FactorBatchRemovalRequest> removal);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::local
