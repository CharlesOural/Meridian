#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

// Provisional legacy sparse-builder carriers. They are intentionally not
// accepted by GlobalGraph; the graph-facing API consumes core sparse seals.
// These disappear when sparse finality moves to meridian_local_rt.
struct FinalizedSubmapFrame {
  core::SubmapRef ref;
  core::Pose3d T_odom_submap;
  core::FusionTime support_end;
};

struct RelativeAnchorConstraint {
  core::SubmapRef from;
  core::SubmapRef to;
  core::Pose3d T_from_to;
  core::RankAwareInformation information;
};

struct AdjacentSubmapAppend {
  FinalizedSubmapFrame submap;
  RelativeAnchorConstraint constraint;
};

struct SparseSubmapPolicyRevisionTag;
using SparseSubmapPolicyRevision = core::StrongId<SparseSubmapPolicyRevisionTag>;

enum class SparseSubmapLifecycle {
  Active,
  Finalizing,
  Sealed,
};

enum class LocalStateFinality {
  FinalizedAndOutOfLag,
};

enum class FinalizedFactorKind {
  ImuPreintegration,
  BiasRandomWalk,
  VisualReprojection,
  LidarRegistration,
  IncomingMarginalPrior,
};

// A state is staged while it may still move in the fixed lag. Its identity,
// support time, and odom epoch are immutable; FinalizedLocalStateInput later
// supplies the only pose that may enter a sparse seal.
struct LocalStateContributionInput {
  core::RecordHeader header;
  core::StateId state;
  core::FusionTime exact_time;
  core::OdomEpoch odom_epoch;
  core::CalibrationEpoch calibration;
  core::LocalGraphRevision admitted_revision;
  core::NavStateEstimate provisional_estimate;
  std::uint64_t retained_bytes{};
};

struct FinalizedLocalStateInput {
  core::RecordHeader header;
  core::StateId state;
  core::FusionTime exact_time;
  core::OdomEpoch odom_epoch;
  core::CalibrationEpoch calibration;
  core::LocalGraphRevision final_revision;
  core::NavStateEstimate final_estimate;
  core::PoseCovariance pose_covariance;
  LocalStateFinality finality{LocalStateFinality::FinalizedAndOutOfLag};
};

// Factor ownership is determined only by terminal_time. The support list is
// explicit so sealing can prove that every contributing state is out of lag.
// IncomingMarginalPrior records are retained for audit but are never admitted
// to eliminated_factor_ids or the adjacent condensed constraint.
struct FinalizedFactorInput {
  core::RecordHeader header;
  core::FactorId factor;
  FinalizedFactorKind kind{FinalizedFactorKind::ImuPreintegration};
  std::vector<core::StateId> support_states;
  core::StateId terminal_state;
  core::FusionTime terminal_time;
  core::LocalGraphRevision final_revision;
  core::ObservationLineage lineage;
  std::uint64_t retained_bytes{};
};

// Exactly one storage member is set. A BlobRef is merely an immutable record
// handle; this package does not persist it. in_memory bytes remain bounded and
// immutable through shared ownership.
struct PlacePayloadInput {
  core::ContentHash checksum{};
  std::optional<core::BlobRef> record;
  core::ImmutableBytes in_memory;
};

struct LidarProxySample {
  core::Vector3d point_lidar{core::Vector3d::Zero()};
  core::Vector3d normal_lidar{core::Vector3d::UnitZ()};
  double weight{1.0};
};

struct FinalizedLidarKeyframeInput {
  core::RecordHeader header;
  core::SweepId sweep;
  core::StateId state;
  core::FusionTime terminal_time;
  core::LocalGraphRevision final_revision;
  core::Pose3d T_imu_lidar;
  std::vector<LidarProxySample> registration_samples;
  std::optional<PlacePayloadInput> place_payload;
  core::ObservationLineage lineage;
  std::uint64_t retained_bytes{};
};

struct FinalizedVisualKeyframeInput {
  core::RecordHeader header;
  core::CameraFrameId frame;
  core::CameraId camera;
  core::StateId state;
  core::FusionTime terminal_time;
  core::LocalGraphRevision final_revision;
  core::Pose3d T_imu_camera;
  PlacePayloadInput place_payload;
  core::ObservationLineage lineage;
  std::uint64_t retained_bytes{};
};

enum class BoundaryJointCovarianceOrder {
  FromThenToRightTranslationFirst,
};

// This is the exact joint pose marginal computed by the local condensation
// pass without moving either finalized endpoint. The supported rank comes
// from that pass; Meridian derives the relative covariance and its rank-aware
// information in the public translation-first tangent.
struct FinalizedBoundaryCondensationInput {
  core::RecordHeader header;
  core::StateId from_boundary_state;
  core::StateId to_boundary_state;
  core::LocalGraphRevision final_revision;
  Eigen::Matrix<double, 12, 12> joint_pose_covariance{Eigen::Matrix<double, 12, 12>::Zero()};
  BoundaryJointCovarianceOrder covariance_order{
      BoundaryJointCovarianceOrder::FromThenToRightTranslationFirst};
  // Orthonormal translation-first directions. The first
  // supported_relative_rank columns are the observable relative-pose
  // subspace reported by the condensation pass.
  Eigen::Matrix<double, 6, 6> relative_information_basis{Eigen::Matrix<double, 6, 6>::Identity()};
  std::size_t supported_relative_rank{6U};
  std::vector<core::FactorId> eliminated_factor_ids;
};

struct LocalFinalityBarrierInput {
  core::RecordHeader header;
  // All factor/keyframe records with terminal timestamp strictly less than
  // finalized_through have been delivered to this coordinator.
  core::FusionTime finalized_through;
  core::LocalGraphRevision final_revision;
};

struct SparseSubmapSplitPolicy {
  SparseSubmapPolicyRevision revision{1U};
  double maximum_travel_m{15.0};
  double maximum_rotation_rad{1.2};
  core::Duration maximum_duration{15'000'000'000LL};
  std::size_t maximum_keyframes{30U};
  std::uint64_t maximum_payload_bytes{64U * 1024U * 1024U};
};

struct SparseSubmapConfig {
  SparseSubmapSplitPolicy split;
  core::SubmapId first_submap_id{1U};
  core::SubmapContentRevision content_revision{1U};

  std::size_t maximum_staged_states{4096U};
  std::size_t maximum_factors_per_submap{16384U};
  std::size_t maximum_factor_support_states{8U};
  std::size_t maximum_keyframes_per_submap{256U};
  std::size_t maximum_lineage_usages_per_submap{32768U};
  std::size_t maximum_lineage_correlations_per_submap{2048U};
  std::uint64_t maximum_builder_bytes{128U * 1024U * 1024U};
  std::uint64_t maximum_place_payload_bytes{8U * 1024U * 1024U};
  std::size_t maximum_proxy_input_samples_per_submap{1'000'000U};
  std::size_t maximum_registration_proxy_points{100'000U};
  std::size_t maximum_pending_seals{8U};
  std::size_t maximum_seal_identities{4096U};
  std::size_t maximum_seen_input_ids{1'000'000U};
  std::size_t maximum_consumed_primary_slices{1'000'000U};

  double registration_voxel_resolution_m{0.40};
  double minimum_heading_projection_norm{1.0e-6};
  double covariance_symmetry_relative_tolerance{1.0e-10};
  double minimum_supported_covariance_eigenvalue{1.0e-12};
  double maximum_supported_covariance_condition{1.0e14};
};

enum class SparseSubmapSplitReason {
  LocalReset,
  Travel,
  Rotation,
  Duration,
  KeyframeCount,
  PayloadBytes,
};

struct SparseSubmapSplitRequest {
  core::StateId boundary_state;
  bool local_reset{false};
};

struct SparseSubmapSplitReport {
  bool split_requested{false};
  std::vector<SparseSubmapSplitReason> reasons;
  double travel_m{};
  double rotation_rad{};
  core::Duration duration;
  std::size_t keyframes{};
  std::uint64_t payload_bytes{};
  core::FusionTime boundary_time;
};

struct RegistrationProxyPoint {
  core::Vector3d point_submap{core::Vector3d::Zero()};
  core::Vector3d normal_submap{core::Vector3d::UnitZ()};
  double weight{1.0};
};

struct RegistrationProxy {
  double voxel_resolution_m{};
  std::vector<RegistrationProxyPoint> points;
  core::ContentHash checksum{};
};

struct FinalizedSubmapStateRecord {
  core::StateId state;
  core::FusionTime exact_time;
  core::LocalGraphRevision final_local_revision;
  core::Pose3d T_submap_imu;
  core::Vector3d velocity_submap{core::Vector3d::Zero()};
  core::Vector3d gyro_bias{core::Vector3d::Zero()};
  core::Vector3d accel_bias{core::Vector3d::Zero()};
  core::PoseCovariance pose_covariance;
};

struct LidarPlacePayloadIndexEntry {
  core::SweepId sweep;
  core::StateId state;
  core::FusionTime terminal_time;
  core::Pose3d T_submap_lidar;
  PlacePayloadInput payload;
  core::ObservationLineage lineage;
};

struct VisualPlacePayloadIndexEntry {
  core::CameraFrameId frame;
  core::CameraId camera;
  core::StateId state;
  core::FusionTime terminal_time;
  core::Pose3d T_submap_camera;
  PlacePayloadInput payload;
  core::ObservationLineage lineage;
};

enum class SparseSealStorageSemantics {
  VolatileInProcessOnly,
};

struct AdjacentConstraintRecord {
  AdjacentSubmapAppend global_append;
  core::PoseCovariance relative_covariance;
  core::ObservationLineage lineage;
  std::vector<core::FactorId> eliminated_factor_ids;
  core::LocalGraphRevision final_local_revision;
};

// Returned as shared_ptr<const SparseSubmapSealRecord>. The record and every
// child payload are immutable, but volatile: this lifecycle core performs no
// disk write, durability acknowledgement, or dense integration.
struct SparseSubmapSealRecord {
  explicit SparseSubmapSealRecord(FinalizedSubmapFrame immutable_submap)
      : submap(std::move(immutable_submap)) {}

  core::RecordHeader header;
  core::SparseSubmapSealIdentity identity;
  FinalizedSubmapFrame submap;
  core::TimeRange core_interval;
  core::StateId start_boundary_state;
  core::StateId end_boundary_state;
  core::LocalGraphRevision final_local_revision;
  std::vector<core::CalibrationEpoch> calibration_epochs;
  std::vector<core::StateId> core_state_ids;
  std::vector<FinalizedSubmapStateRecord> finalized_trajectory;
  std::vector<core::FactorId> condensed_factor_ids;
  core::ObservationLineage factor_lineage;
  RegistrationProxy registration_proxy;
  std::vector<LidarPlacePayloadIndexEntry> lidar_place_index;
  std::vector<VisualPlacePayloadIndexEntry> visual_place_index;
  std::optional<AdjacentConstraintRecord> incoming_adjacent;
  SparseSealStorageSemantics storage_semantics{SparseSealStorageSemantics::VolatileInProcessOnly};
};

using SparseSubmapSeal = std::shared_ptr<const SparseSubmapSealRecord>;

enum class SparseSubmapErrorCode {
  InvalidConfiguration,
  NotStarted,
  InvalidRecord,
  SessionMismatch,
  OdomEpochMismatch,
  NonMonotonicStateTime,
  DuplicateIdentity,
  UnknownState,
  RecordOutsideOpenInterval,
  InvalidLineage,
  NonDisjointLineage,
  CapacityExceeded,
  SplitAlreadyFinalizing,
  InvalidSplitBoundary,
  NoFinalizingSubmap,
  InvalidCondensation,
  FinalityRegression,
  BoundaryNotFinalized,
  BoundaryYawUnobservable,
  NumericalFailure,
  SealIdentityConflict,
  SubmapIdOverflow,
};

struct SparseSubmapError {
  SparseSubmapErrorCode code{SparseSubmapErrorCode::InvalidRecord};
  std::string detail;
};

struct SparseSubmapStatus {
  bool started{false};
  SparseSubmapLifecycle active_lifecycle{SparseSubmapLifecycle::Active};
  core::SubmapId active_submap;
  std::optional<core::SubmapId> finalizing_submap;
  std::size_t staged_states{};
  std::size_t pending_seals{};
  std::size_t sealed_identities{};
  std::optional<core::FusionTime> finalized_through;
};

// Single-writer, ROS-free lifecycle coordinator. It builds only sparse,
// bounded in-memory artifacts. A later SealSpool adapter may durably serialize
// the immutable output without changing this API.
class SparseSubmapCoordinator {
public:
  explicit SparseSubmapCoordinator(SparseSubmapConfig config = {});
  ~SparseSubmapCoordinator();

  SparseSubmapCoordinator(SparseSubmapCoordinator&&) noexcept;
  SparseSubmapCoordinator& operator=(SparseSubmapCoordinator&&) noexcept;
  SparseSubmapCoordinator(const SparseSubmapCoordinator&) = delete;
  SparseSubmapCoordinator& operator=(const SparseSubmapCoordinator&) = delete;

  [[nodiscard]] core::Result<bool, SparseSubmapError> stageState(LocalStateContributionInput input);
  [[nodiscard]] core::Result<bool, SparseSubmapError> finalizeState(FinalizedLocalStateInput input);
  [[nodiscard]] core::Result<bool, SparseSubmapError> stageFactor(FinalizedFactorInput input);
  [[nodiscard]] core::Result<bool, SparseSubmapError> stageLidarKeyframe(
      FinalizedLidarKeyframeInput input);
  [[nodiscard]] core::Result<bool, SparseSubmapError> stageVisualKeyframe(
      FinalizedVisualKeyframeInput input);

  [[nodiscard]] core::Result<SparseSubmapSplitReport, SparseSubmapError> considerSplit(
      SparseSubmapSplitRequest request);
  [[nodiscard]] core::Result<bool, SparseSubmapError> abortProvisionalSplit();
  [[nodiscard]] core::Result<bool, SparseSubmapError> submitBoundaryCondensation(
      FinalizedBoundaryCondensationInput input);
  [[nodiscard]] core::Result<bool, SparseSubmapError> advanceFinality(
      LocalFinalityBarrierInput input);

  [[nodiscard]] std::vector<SparseSubmapSeal> takeSealed();
  [[nodiscard]] SparseSubmapStatus status() const noexcept;
  [[nodiscard]] std::optional<core::SparseSubmapSealIdentity> sealIdentity(
      const core::SubmapRef& submap) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::global
