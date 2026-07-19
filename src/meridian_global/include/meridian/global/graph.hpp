#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/gnc_tls.hpp"
#include "meridian/global/ids.hpp"

namespace meridian::global {

struct LoopConsensusReport;

enum class AlignmentTangentConvention {
  TranslationEnuThenYaw,
};

struct YawTranslation4 {
  Eigen::Vector3d translation_enu{Eigen::Vector3d::Zero()};
  double yaw_enu_map_rad{};

  [[nodiscard]] Eigen::Matrix3d REnuMap() const noexcept;
  [[nodiscard]] Eigen::Vector3d apply(const Eigen::Vector3d& position_map) const noexcept;
};

struct AlignmentCovariance {
  Eigen::Matrix4d matrix{Eigen::Matrix4d::Zero()};
  AlignmentTangentConvention tangent{AlignmentTangentConvention::TranslationEnuThenYaw};
};

enum class LoopModality {
  Visual,
  Lidar,
};

// Immutable geometric-verifier output. T_from_to maps coordinates in `to`
// into `from`. Retrieval scores schedule verification but never enter this
// measurement or its precision.
struct LoopMeasurement {
  core::RecordHeader header;
  ProposalId proposal;
  LoopModality modality{LoopModality::Visual};
  core::SubmapRef from;
  core::SubmapRef to;
  std::vector<core::CalibrationEpoch> calibration_epochs;
  core::Pose3d T_from_to;
  core::RankAwareInformation information;
  core::ObservationLineage lineage;
};

// A PCM-admitted geometric measurement plus the factor-model-specific TLS
// gate. The declared DoF must equal the supported information rank.
struct RobustLoopCandidate {
  LoopMeasurement measurement;
  GncTlsFactorScale scale;
};

struct RobustLoopBatchAppend {
  // Compare-and-swap parent. A result produced against another graph revision
  // is never silently rebased.
  GlobalGraphRevision expected_parent;
  std::vector<RobustLoopCandidate> candidates;
};

struct GnssAntennaConstraint {
  core::SubmapRef submap;
  core::GnssObservationId observation;
  // Exact-time antenna phase-center position already interpolated in the
  // immutable submap frame, including the calibrated body-to-antenna lever arm.
  Eigen::Vector3d antenna_position_submap{Eigen::Vector3d::Zero()};
  Eigen::Vector3d measured_position_enu{Eigen::Vector3d::Zero()};
  // Receiver, interpolation, lever-arm, and configured model uncertainty.
  Eigen::Matrix3d effective_covariance_enu{Eigen::Matrix3d::Identity()};
};

struct GnssAntennaPrediction {
  Eigen::Vector3d position_enu{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 3, 6> position_jacobian_anchor{Eigen::Matrix<double, 3, 6>::Zero()};
  Eigen::Matrix<double, 3, 4> position_jacobian_alignment{Eigen::Matrix<double, 3, 4>::Zero()};
  core::PoseTangentConvention anchor_tangent{core::PoseTangentConvention::RightTranslationFirst};
  AlignmentTangentConvention alignment_tangent{AlignmentTangentConvention::TranslationEnuThenYaw};
};

// Shared mathematical seam used by proposal diagnostics and by the private
// factor adapter. The antenna point is already exact-time/interpolated in S.
[[nodiscard]] GnssAntennaPrediction predictGnssAntenna(
    const core::Pose3d& T_map_submap, const Eigen::Vector3d& antenna_position_submap,
    const YawTranslation4& T_enu_map) noexcept;

struct GnssBatchAppend {
  // Required only by the first admitted GNSS batch. It is an initializer, not
  // a prior: information enters the graph only through the individual fixes.
  std::optional<YawTranslation4> initial_alignment;
  std::vector<GnssAntennaConstraint> constraints;
};

struct GlobalGraphConfig {
  std::size_t maximum_anchors{4096};
  std::size_t maximum_scalar_dimension{61'444};
  std::size_t maximum_adjacent_factors{4095};
  std::size_t maximum_adjacent_seals_per_transaction{128};
  std::size_t maximum_adjacent_factor_rows{4096};
  std::size_t maximum_adjacent_factor_coefficients{4096U * 30U};
  std::size_t maximum_total_adjacent_factor_rows{4095U * 64U};
  std::size_t maximum_total_adjacent_factor_coefficients{4095U * 64U * 30U};
  std::size_t maximum_gnss_factors{16384};
  std::size_t maximum_loop_factors{512};
  std::size_t maximum_loop_candidates_per_transaction{128};
  std::size_t maximum_loop_calibration_epochs{8};
  std::size_t maximum_loop_lineage_usages{4096};
  std::size_t maximum_loop_lineage_correlations{256};
  std::size_t maximum_solver_iterations{100};

  GncTlsConfig loop_gnc = [] {
    GncTlsConfig config;
    config.maximum_known_inliers = 32768;
    config.maximum_robust_candidates = 512;
    return config;
  }();

  // Numerically equivalent gauge constraint on the first anchor. These are
  // deliberately not sensor uncertainties.
  double mission_gauge_translation_sigma_m{1.0e-4};
  double mission_gauge_rotation_sigma_rad{1.0e-4};

  double covariance_symmetry_relative_tolerance{1.0e-10};
  double minimum_covariance_eigenvalue{1.0e-12};
  double information_basis_orthonormal_tolerance{1.0e-8};
  double information_zero_tolerance{1.0e-12};
  double hessian_absolute_rank_tolerance{1.0e-10};
  double hessian_relative_rank_tolerance{1.0e-13};
  double maximum_hessian_condition{1.0e15};
  double solver_relative_error_tolerance{1.0e-7};
  double solver_absolute_error_tolerance{1.0e-9};
};

enum class GlobalTransactionKind {
  MissionInitialization,
  AdjacentInsertion,
  GnssInsertion,
  RobustLoopInsertion,
};

struct GlobalSolveReport {
  GlobalTransactionKind transaction{GlobalTransactionKind::MissionInitialization};
  std::size_t anchors{};
  std::size_t materialized_navigation_boundaries{};
  std::size_t adjacent_seals_in_transaction{};
  std::size_t adjacent_factors{};
  std::size_t gnss_factors{};
  std::size_t loop_factors{};
  std::size_t scalar_dimension{};
  std::size_t numerical_rank{};
  std::size_t solver_iterations{};
  double initial_error{};
  double final_error{};
  double hessian_condition{};
  bool finite{false};
  bool converged{false};
  bool connected{false};
};

struct SubmapAnchorEstimate {
  core::SubmapRef submap;
  core::Pose3d T_odom_submap;
  core::FusionTime support_end;
  core::Pose3d T_map_submap;
  core::PoseCovariance covariance;
};

enum class MapOdomCovarianceSemantics {
  ConditionalOnSealedLocalFrame,
};

struct MapOdomEstimate {
  GlobalGraphRevision graph_revision;
  core::SubmapRef reference_submap;
  core::Pose3d T_map_odom;
  core::PoseCovariance covariance;
  MapOdomCovarianceSemantics covariance_semantics{
      MapOdomCovarianceSemantics::ConditionalOnSealedLocalFrame};
};

struct GlobalGraphCommit {
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  std::vector<SubmapAnchorEstimate> anchors;
  std::optional<YawTranslation4> alignment;
  std::optional<AlignmentCovariance> alignment_covariance;
  MapOdomEstimate map_odom;
  GlobalSolveReport solve;
};

// Canonical solver-neutral graph recovery state. GTSAM keys, Values, factor
// pointers, and solver archives never cross this API. The wire schema is an
// explicit migration boundary: unsupported schemas fail closed rather than
// being guessed from object layout.
inline constexpr std::uint32_t kGlobalGraphCheckpointSchemaVersion = 1U;
inline constexpr std::uint32_t kGlobalGraphKeySchemaVersion = 1U;
inline constexpr std::uint32_t kGlobalGraphConfigurationSchemaVersion = 1U;

struct GlobalGraphConfigurationIdentity {
  std::uint32_t schema_version{kGlobalGraphConfigurationSchemaVersion};
  core::ContentHash checksum{};
};

struct MissionGaugeCheckpoint {
  GlobalFactorId factor;
  std::uint64_t boundary_slot{};
  core::Pose3d T_map_submap;
  double translation_sigma_m{};
  double rotation_sigma_rad{};
};

struct BoundaryNavigationCheckpoint {
  std::uint64_t slot{};
  core::SparseSubmapSeal seal;
  core::Pose3d T_map_submap;
  std::optional<core::Vector3d> velocity_map;
  std::optional<core::Vector3d> gyro_bias;
  std::optional<core::Vector3d> accel_bias;
};

struct OdomEpochChartPlacementCheckpoint {
  core::OdomEpoch odom_epoch;
  core::Pose3d H_map_odom;
};

struct AdjacentBoundaryCheckpoint {
  GlobalFactorId factor;
  std::uint64_t from_slot{};
  std::uint64_t to_slot{};
  core::SealedBoundaryTransition transition;
};

struct GnssFactorCheckpoint {
  GlobalFactorId factor;
  GnssAntennaConstraint constraint;
};

struct LoopFactorCheckpoint {
  GlobalFactorId factor;
  LoopMeasurement measurement;
  CandidateId candidate;
  GncTlsFactorScale scale;
};

struct GlobalFactorObjectiveCheckpoint {
  GlobalFactorId factor;
  // r^T r, without the one-half convention used by GTSAM's error().
  double whitened_squared_cost{};
};

struct BoundaryMarginalCheckpoint {
  std::uint64_t boundary_slot{};
  core::PoseCovariance covariance;
};

// Tolerances are persisted and checksum-covered, but are not caller policy.
// restoreCheckpoint derives the canonical values from its configured graph and
// requires bit-identical equality before using them.
struct GlobalGraphRecoveryTolerances {
  double objective_absolute{};
  double objective_relative{};
  double estimate_tangent_absolute{};
  double gradient_infinity_absolute{};
  double covariance_absolute{};
  double covariance_relative{};
  double condition_relative{};
};

struct GlobalGraphRecoveryAudit {
  double whitened_squared_objective{};
  double gradient_infinity_norm{};
  std::size_t scalar_dimension{};
  std::size_t numerical_rank{};
  double hessian_condition{};
  std::vector<GlobalFactorObjectiveCheckpoint> factor_objectives;
  std::vector<BoundaryMarginalCheckpoint> boundary_marginals;
  std::optional<AlignmentCovariance> alignment_covariance;
  MapOdomEstimate map_odom;
  GlobalSolveReport committed_solve;
  GlobalGraphRecoveryTolerances tolerances;
};

struct GlobalGraphCheckpoint {
  std::uint32_t schema_version{kGlobalGraphCheckpointSchemaVersion};
  std::uint32_t key_schema_version{kGlobalGraphKeySchemaVersion};
  GlobalGraphConfigurationIdentity configuration;
  core::SessionId mission_session;
  GlobalGraphRevision revision;
  std::optional<GlobalGraphRevision> parent;
  MissionGaugeCheckpoint mission_gauge;
  std::vector<OdomEpochChartPlacementCheckpoint> chart_placements;
  std::vector<BoundaryNavigationCheckpoint> boundaries;
  std::vector<AdjacentBoundaryCheckpoint> adjacent_factors;
  std::vector<GnssFactorCheckpoint> gnss_factors;
  std::vector<LoopFactorCheckpoint> loop_factors;
  // Exact active nonlinear-factor order. Every typed factor identity occurs
  // once; no factor may exist only in this list.
  std::vector<GlobalFactorId> factor_order;
  std::optional<YawTranslation4> alignment;
  std::uint64_t next_boundary_slot{};
  std::uint64_t next_factor_id{};
  std::uint64_t next_candidate_id{};
  GlobalGraphRecoveryAudit recovery;
  core::ContentHash checksum{};
};

struct GlobalGraphCheckpointLimits {
  std::uint64_t maximum_wire_bytes{256ULL * 1024ULL * 1024ULL};
  std::size_t maximum_boundaries{4096U};
  std::size_t maximum_chart_placements{4096U};
  std::size_t maximum_adjacent_factors{4095U};
  std::size_t maximum_gnss_factors{16384U};
  std::size_t maximum_loop_factors{512U};
  std::size_t maximum_factor_rows{4096U};
  std::size_t maximum_factor_coefficients{4096U * 30U};
  std::size_t maximum_nested_collection_entries{16U * 1024U * 1024U};
};

enum class GlobalGraphCheckpointErrorCode {
  InvalidLimits,
  EncodingFailure,
  TruncatedRecord,
  TrailingBytes,
  UnsupportedSchema,
  UnsupportedKeySchema,
  UnsupportedConfigurationSchema,
  CapacityExceeded,
  InvalidBoolean,
  InvalidEnum,
  InvalidFloatingPoint,
  InvalidPose,
  InvalidCheckpoint,
  InvalidSparseSeal,
  ChecksumMismatch,
  NonCanonicalEncoding,
  ConfigurationMismatch,
  AlreadyInitialized,
  ReconstructionFailure,
  RankMismatch,
  ObjectiveMismatch,
  EstimateMismatch,
  MarginalMismatch,
};

struct GlobalGraphCheckpointError {
  GlobalGraphCheckpointErrorCode code{GlobalGraphCheckpointErrorCode::InvalidCheckpoint};
  std::string detail;
  std::optional<core::CanonicalEncodingError> encoding_error;
  std::optional<core::CanonicalVerificationError> canonical_verification_error;
};

// Encodes the complete checkpoint and its internally-derived checksum. Decode
// is bounded, recursively verifies every sparse seal, recomputes the checksum,
// and rejects encodings that do not round-trip byte-identically.
[[nodiscard]] core::Result<core::CanonicalByteSequence, GlobalGraphCheckpointError>
encodeGlobalGraphCheckpoint(const GlobalGraphCheckpoint& checkpoint,
                            GlobalGraphCheckpointLimits limits = {});
[[nodiscard]] core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError>
decodeGlobalGraphCheckpoint(std::span<const std::byte> bytes,
                            GlobalGraphCheckpointLimits limits = {});

enum class RobustLoopApplicationDisposition {
  CommittedNewTlsInlier,
  RetainedCommittedTlsInlier,
  RejectedNewTlsOutlier,
  RemovedCommittedTlsOutlier,
  TlsInlierNotCommitted,
  RetainedCommittedDespiteRejectedTransaction,
};

struct RobustLoopDecision {
  ProposalId proposal;
  CandidateId candidate;
  bool previously_committed{false};
  GncTlsFactorCost cost;
  double final_weight{};
  GncTlsCandidateDisposition tls_disposition{GncTlsCandidateDisposition::TlsOutlier};
  RobustLoopApplicationDisposition application{
      RobustLoopApplicationDisposition::RejectedNewTlsOutlier};
};

enum class RobustLoopTransactionOutcome {
  Committed,
  NoNewTlsInlier,
  RejectedFinalGraphValidation,
};

struct RobustLoopShadowValidationReport {
  std::size_t known_inlier_factors{};
  std::size_t mission_gauge_known_inliers{};
  std::size_t adjacent_known_inliers{};
  std::size_t gnss_known_inliers{};
  std::size_t active_loop_factors_before{};
  std::size_t evaluated_loop_candidates{};
  std::size_t active_loop_factors_after{};
  std::size_t new_tls_inliers{};
  std::size_t new_tls_outliers{};
  std::size_t removed_committed_outliers{};
  double maximum_known_inlier_normalized_squared_cost{};
  bool complete_known_inlier_set{false};
  bool known_inlier_weights_fixed_to_one{false};
  bool final_complete_graph_validation_passed{false};
};

struct RobustLoopTransactionReport {
  GlobalGraphRevision evaluated_parent;
  core::FusionTime pcm_evaluated_at;
  std::size_t pcm_admitted_candidates{};
  RobustLoopTransactionOutcome outcome{RobustLoopTransactionOutcome::NoNewTlsInlier};
  GncTlsReport gnc;
  std::vector<RobustLoopDecision> decisions;
  RobustLoopShadowValidationReport validation;
};

struct RobustLoopTransactionResult {
  RobustLoopTransactionReport report;
  std::optional<GlobalGraphCommit> commit;
};

enum class GlobalGraphErrorCode {
  InvalidConfig,
  AlreadyInitialized,
  NotInitialized,
  InvalidSubmapReference,
  InvalidSparseSeal,
  InvalidSparseLink,
  EmptyAdjacentBatch,
  NonConsecutiveAdjacentBatch,
  DuplicateSubmap,
  StaleSubmapReference,
  InvalidRelativeConstraint,
  InvalidRankAwareInformation,
  InvalidGnssConstraint,
  InvalidCovariance,
  DuplicateGnssObservation,
  EmptyGnssBatch,
  AlignmentInitializerRequired,
  AlignmentAlreadyInitialized,
  DisconnectedProposal,
  AnchorCapacity,
  AdjacentFactorCapacity,
  AdjacentBatchCapacity,
  ScalarDimensionCapacity,
  GnssFactorCapacity,
  SolverFailure,
  SolverDidNotConverge,
  NonFiniteSolution,
  RankDeficientCandidate,
  IllConditionedCandidate,
  MarginalCovarianceFailure,
  RevisionOverflow,
  FactorIdOverflow,
};

struct GlobalGraphError {
  GlobalGraphErrorCode code{};
  std::string detail;
  std::optional<core::CanonicalVerificationError> canonical_verification_error;
};

enum class RobustLoopTransactionErrorCode {
  NotInitialized,
  StaleParentRevision,
  EmptyCandidateBatch,
  CandidateCapacity,
  LoopFactorCapacity,
  InvalidCandidate,
  MissingPcmAdmission,
  NonIndependentLineage,
  DuplicateProposal,
  UnknownSubmap,
  StaleSubmapReference,
  GncTlsFailure,
  IdentityOverflow,
  FinalGraphValidationFailure,
};

struct RobustLoopTransactionError {
  RobustLoopTransactionErrorCode code{RobustLoopTransactionErrorCode::InvalidCandidate};
  std::optional<ProposalId> proposal;
  std::optional<GncTlsError> gnc_error;
  std::optional<GlobalGraphError> graph_error;
  // Present when GNC completed but the final all-inlier graph failed the
  // ordinary graph transaction validation. It preserves forensic context.
  std::optional<RobustLoopTransactionReport> shadow_report;
  std::string detail;
};

// ROS-free single-writer global graph. GTSAM and all tangent permutations are
// private implementation details. Every mutation optimizes and validates a
// complete shadow copy before swapping it into the committed state.
class GlobalGraph {
public:
  explicit GlobalGraph(GlobalGraphConfig config = {});
  ~GlobalGraph();

  GlobalGraph(GlobalGraph&&) noexcept;
  GlobalGraph& operator=(GlobalGraph&&) noexcept;
  GlobalGraph(const GlobalGraph&) = delete;
  GlobalGraph& operator=(const GlobalGraph&) = delete;

  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> initializeMission(
      core::SparseSubmapSeal first_seal);
  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> appendAdjacent(
      core::SparseSubmapSeal current_seal);
  // Stages one exact consecutive chain from the current committed head and
  // optimizes it once. This permits jointly observable chains whose individual
  // transitions are rank-deficient; rejection consumes no revision or ID.
  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> appendAdjacentBatch(
      std::vector<core::SparseSubmapSeal> consecutive_seals);
  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> appendGnssBatch(
      GnssBatchAppend append);
  [[nodiscard]] core::Result<RobustLoopTransactionResult, RobustLoopTransactionError>
  appendRobustLoopBatch(RobustLoopBatchAppend append, const LoopConsensusReport& consensus);

  // Restores only into an empty graph. Reconstruction, optimization, complete
  // objective/rank/estimate/marginal verification, and publication happen on
  // a disposable shadow. Failure consumes no revision, factor ID, or candidate
  // ID and leaves the graph uninitialized.
  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphCheckpointError> restoreCheckpoint(
      const GlobalGraphCheckpoint& checkpoint);

  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> snapshot() const;
  [[nodiscard]] core::Result<GlobalGraphCheckpoint, GlobalGraphError> checkpoint() const;
  [[nodiscard]] bool initialized() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::global
