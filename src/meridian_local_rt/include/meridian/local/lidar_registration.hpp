#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/lidar_registration_cloud.hpp"

namespace meridian::local {

inline constexpr std::string_view kLidarFactorSnapshotChecksumDomain{
    "meridian.local.direct_point_icp.factor_snapshot"};
inline constexpr std::uint32_t kLidarFactorSnapshotChecksumSchemaVersion{3U};
inline constexpr std::string_view kFinalizedMapLidarFactorSnapshotChecksumDomain{
    "meridian.local.direct_point_icp.finalized_map_factor_snapshot"};
inline constexpr std::uint32_t kFinalizedMapLidarFactorSnapshotChecksumSchemaVersion{3U};

class FinalizedLidarTargetReadView;
struct FinalizedLidarTargetOwner;

namespace detail {
class LidarFactorSnapshotBuilder;
class FinalizedMapLidarFactorSnapshotBuilder;
}  // namespace detail

// One target pose, one target cloud, and one relative seed remain inseparable.
// T_target_source_seed must equal
// target.T_odom_imu_target_seed.inverse() * source.T_odom_imu_seed. Every
// optimizer update is a shared right perturbation:
//   T_target_source(delta) = T_target_source_seed * Exp(delta).
// Consequently two target records constrain one source pose without merging
// either their geometry or their state ownership.
struct LidarRegistrationTarget {
  core::StateId state;
  core::FusionTime time;
  std::shared_ptr<const LidarRegistrationCloud> cloud;
  // Optimized pose metadata is intentionally separate from immutable scan
  // geometry. Fixed-lag graph revisions update this seed without cloning the
  // direct-registration cloud.
  core::Pose3d T_odom_imu_target_seed;
  core::Pose3d T_target_source_seed;
};

struct LidarRegistrationConfig {
  // Target points retain the denser scan artifact while source points are
  // selected at this coarser keypoint spacing. This is direct registration,
  // not feature extraction: every retained row remains a raw 3-D point.
  double target_voxel_resolution_m{1.0};
  double source_voxel_size_m{1.5};
  double maximum_correspondence_distance_m{0.5};
  // Composite correspondence search visits a bounded
  // ceil(maximum_correspondence_distance / target_voxel_resolution) halo.
  // The production profile uses radius one; larger framework/test profiles
  // retain their existing semantics without an unbounded search.
  std::size_t maximum_voxel_search_radius{8U};
  std::size_t maximum_source_points{20'000U};
  std::size_t maximum_target_points_per_target{100'000U};
  std::size_t maximum_targets{5U};
  // Live owners share one odometry-frame search index. The persistent
  // finalized-map base remains independently indexed and is merged with the
  // live overlay by one logical nearest-neighbour query.
  std::size_t maximum_composite_owners{15U};
  std::size_t maximum_composite_indexed_points{150'000U};
  std::size_t maximum_composite_points_per_voxel{20U};
  // Association and normal equations use fixed contiguous worker partitions,
  // followed by a worker-index ordered fold. The result is deterministic for
  // one configured worker count and bounded against executor oversubscription.
  std::size_t parallel_worker_count{4U};
  std::size_t minimum_correspondences{30U};
  // Scalar row uncertainty is deliberately conservative: neighboring map
  // points are locally correlated and target poses are reused. A later batch
  // information cap remains the final authority seen by the smoother.
  double residual_standard_deviation_m{0.10};
  // Huber scale is recomputed from the final correspondence population using
  // median absolute deviation, then clamped. The generous lower bound keeps
  // vegetation from being starved as happened with v1's fixed small kernel.
  double huber_delta_multiplier{2.5};
  double minimum_huber_delta_m{0.15};
  double maximum_huber_delta_m{0.50};

  // v1 needed natural convergence rather than a low hard iteration count.
  // One bounded GN/LM implementation is used; the 100-iteration limit is a
  // fault ceiling, not a second "fast" registration path.
  std::size_t maximum_outer_iterations{100U};
  std::size_t maximum_lm_damping_retries{6U};
  double initial_relative_damping{1.0e-6};
  double damping_increase{10.0};
  double translation_convergence_m{1.0e-3};
  double rotation_convergence_rad{1.0e-3};
  double maximum_correction_translation_m{3.0};
  double maximum_correction_rotation_rad{0.8};

  std::size_t minimum_observable_rank{1U};
  double absolute_normalized_observable_eigenvalue{1.0e-6};
  double relative_normalized_observable_eigenvalue{1.0e-2};
  double minimum_characteristic_length_m{1.0};
  double maximum_characteristic_length_m{30.0};
  // Raw translation-equivalent information ceiling [m^-2]. The complete
  // supported normalized Hessian receives one scalar cap at
  // maximum_translation_information * characteristic_length^2. Unlike
  // independent physical-axis clipping, this preserves geometry, coupling,
  // and rotational lever arm.
  double maximum_translation_information{1.0e3};

  double seed_translation_consistency_tolerance_m{1.0e-6};
  double seed_rotation_consistency_tolerance_rad{1.0e-8};

  bool operator==(const LidarRegistrationConfig&) const = default;
};

[[nodiscard]] bool isValidLidarRegistrationConfig(const LidarRegistrationConfig& config) noexcept;

struct FrozenPointCorrespondence {
  std::size_t source_point_storage_index{};
  std::uint32_t source_index{};
  Eigen::Vector3d source_point{Eigen::Vector3d::Zero()};
  std::size_t target_point_storage_index{};
  std::uint32_t target_source_index{};
  Eigen::Vector3d target_point{Eigen::Vector3d::Zero()};
  // Correspondence identity and robust IRLS weight are frozen together at
  // admission. The graph factor re-evaluates only this sensor-pure point
  // residual and never mutates association state inside iSAM2.
  double association_distance_squared_m2{};
  double association_huber_weight{1.0};
};

// One immutable row owned by the finalized-map channel of the same direct
// registration objective. owner_index resolves through the snapshot's stable
// canonical owner table; no read view or mutable map pointer survives
// registration.
struct FrozenFinalizedMapPointCorrespondence {
  std::size_t source_point_storage_index{};
  std::uint32_t source_index{};
  Eigen::Vector3d source_point{Eigen::Vector3d::Zero()};
  std::uint32_t target_source_index{};
  Eigen::Vector3d target_point_odom{Eigen::Vector3d::Zero()};
  // Index into FinalizedMapLidarFactorSnapshot::owners().  Ownership is
  // canonicalized once per immutable snapshot instead of retaining one
  // shared_ptr in every residual row.
  std::uint32_t owner_index{};
  double association_distance_squared_m2{};
  double association_huber_weight{1.0};
};

class LidarFactorSnapshot {
public:
  [[nodiscard]] core::StateId targetState() const noexcept { return target_state_; }
  [[nodiscard]] core::FusionTime targetTime() const noexcept { return target_time_; }
  [[nodiscard]] core::MeasurementId targetSweep() const noexcept { return target_sweep_; }
  [[nodiscard]] core::StateId sourceState() const noexcept { return source_state_; }
  [[nodiscard]] core::FusionTime sourceTime() const noexcept { return source_time_; }
  [[nodiscard]] core::MeasurementId sourceSweep() const noexcept { return source_sweep_; }
  [[nodiscard]] const core::ContentHash& targetCloudChecksum() const noexcept {
    return target_cloud_checksum_;
  }
  [[nodiscard]] const core::ContentHash& sourceCloudChecksum() const noexcept {
    return source_cloud_checksum_;
  }
  // Number of bounded source rows offered to the multi-target association.
  // Every snapshot produced by one registration reports the same population;
  // rows() is the disjoint subset owned by this particular target.
  [[nodiscard]] std::size_t sourcePointCount() const noexcept { return source_point_count_; }
  [[nodiscard]] const core::Pose3d& associationPose() const noexcept {
    return T_target_source_association_;
  }
  [[nodiscard]] std::span<const FrozenPointCorrespondence> rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t candidateVoxelLookups() const noexcept {
    return candidate_voxel_lookups_;
  }
  [[nodiscard]] std::size_t candidatePointsExamined() const noexcept {
    return candidate_points_examined_;
  }
  [[nodiscard]] std::size_t sourceRowsExcludedByOwnership() const noexcept {
    return source_rows_excluded_by_ownership_;
  }
  [[nodiscard]] double residualStandardDeviationM() const noexcept {
    return residual_standard_deviation_m_;
  }
  [[nodiscard]] double huberDeltaM() const noexcept { return huber_delta_m_; }
  [[nodiscard]] double characteristicLengthM() const noexcept { return characteristic_length_m_; }
  [[nodiscard]] double geometricInformationScale() const noexcept {
    return geometric_information_scale_;
  }
  [[nodiscard]] const core::ContentHash& checksum() const noexcept { return checksum_; }

private:
  core::StateId target_state_;
  core::FusionTime target_time_;
  core::MeasurementId target_sweep_;
  core::StateId source_state_;
  core::FusionTime source_time_;
  core::MeasurementId source_sweep_;
  core::ContentHash target_cloud_checksum_{};
  core::ContentHash source_cloud_checksum_{};
  std::size_t source_point_count_{};
  core::Pose3d T_target_source_association_;
  std::vector<FrozenPointCorrespondence> rows_;
  std::size_t candidate_voxel_lookups_{};
  std::size_t candidate_points_examined_{};
  std::size_t source_rows_excluded_by_ownership_{};
  double residual_standard_deviation_m_{};
  double huber_delta_m_{};
  double characteristic_length_m_{};
  double geometric_information_scale_{};
  core::ContentHash checksum_{};

  friend class detail::LidarFactorSnapshotBuilder;
};

// A finalized map is one fixed odometry-frame target channel, even though its
// rows preserve ownership by many historical poses. The graph therefore sees
// at most one unary factor from this snapshot and never relabels accumulated
// geometry as belonging to the newest live state.
class FinalizedMapLidarFactorSnapshot {
public:
  [[nodiscard]] core::StateId sourceState() const noexcept { return source_state_; }
  [[nodiscard]] core::FusionTime sourceTime() const noexcept { return source_time_; }
  [[nodiscard]] core::MeasurementId sourceSweep() const noexcept { return source_sweep_; }
  [[nodiscard]] const core::ContentHash& sourceCloudChecksum() const noexcept {
    return source_cloud_checksum_;
  }
  [[nodiscard]] std::size_t sourcePointCount() const noexcept { return source_point_count_; }
  [[nodiscard]] core::OdomEpoch mapOdomEpoch() const noexcept { return map_odom_epoch_; }
  [[nodiscard]] core::SensorInstanceId mapSensor() const noexcept { return map_sensor_; }
  [[nodiscard]] std::uint64_t mapVersion() const noexcept { return map_version_; }
  [[nodiscard]] const core::ContentHash& mapChecksum() const noexcept { return map_checksum_; }
  [[nodiscard]] const core::Pose3d& associationPose() const noexcept {
    return T_odom_source_association_;
  }
  [[nodiscard]] std::span<const FrozenFinalizedMapPointCorrespondence> rows() const noexcept {
    return rows_;
  }
  [[nodiscard]] std::span<const std::shared_ptr<const FinalizedLidarTargetOwner>> owners()
      const noexcept {
    return owners_;
  }
  [[nodiscard]] std::size_t uniqueOwnerCount() const noexcept { return owners_.size(); }
  [[nodiscard]] std::size_t candidateVoxelLookups() const noexcept {
    return candidate_voxel_lookups_;
  }
  [[nodiscard]] std::size_t candidateOccupiedVoxels() const noexcept {
    return candidate_occupied_voxels_;
  }
  [[nodiscard]] std::size_t candidatePointsExamined() const noexcept {
    return candidate_points_examined_;
  }
  [[nodiscard]] std::size_t sourceRowsExcludedByOwnership() const noexcept {
    return source_rows_excluded_by_ownership_;
  }
  [[nodiscard]] double residualStandardDeviationM() const noexcept {
    return residual_standard_deviation_m_;
  }
  [[nodiscard]] double huberDeltaM() const noexcept { return huber_delta_m_; }
  [[nodiscard]] double characteristicLengthM() const noexcept { return characteristic_length_m_; }
  [[nodiscard]] double geometricInformationScale() const noexcept {
    return geometric_information_scale_;
  }
  [[nodiscard]] double ownerPoseCovarianceInflation() const noexcept {
    return owner_pose_covariance_inflation_;
  }
  [[nodiscard]] const core::ContentHash& checksum() const noexcept { return checksum_; }

private:
  core::StateId source_state_;
  core::FusionTime source_time_;
  core::MeasurementId source_sweep_;
  core::ContentHash source_cloud_checksum_{};
  std::size_t source_point_count_{};
  core::OdomEpoch map_odom_epoch_;
  core::SensorInstanceId map_sensor_;
  std::uint64_t map_version_{};
  core::ContentHash map_checksum_{};
  core::Pose3d T_odom_source_association_;
  std::vector<std::shared_ptr<const FinalizedLidarTargetOwner>> owners_;
  std::vector<FrozenFinalizedMapPointCorrespondence> rows_;
  std::size_t candidate_voxel_lookups_{};
  std::size_t candidate_occupied_voxels_{};
  std::size_t candidate_points_examined_{};
  std::size_t source_rows_excluded_by_ownership_{};
  double residual_standard_deviation_m_{};
  double huber_delta_m_{};
  double characteristic_length_m_{};
  double geometric_information_scale_{};
  double owner_pose_covariance_inflation_{1.0};
  core::ContentHash checksum_{};

  friend class detail::FinalizedMapLidarFactorSnapshotBuilder;
};

struct LidarRegistrationWorkCounters {
  std::size_t source_points_considered{};
  std::size_t source_points_selected{};
  std::size_t source_points_omitted_by_capacity{};
  std::size_t invalid_source_points{};
  std::size_t invalid_target_points{};
  std::size_t target_index_voxels{};
  std::size_t composite_index_builds{};
  std::size_t composite_index_input_owners{};
  std::size_t composite_index_input_points{};
  std::size_t composite_index_retained_owners{};
  std::size_t composite_index_retained_points{};
  std::size_t composite_index_occupied_voxels{};
  std::size_t composite_index_per_voxel_discarded_points{};
  std::size_t composite_index_total_discarded_points{};
  core::CpuWallDuration composite_index_build_duration;
  std::size_t association_builds{};
  std::size_t candidate_voxel_lookups{};
  std::size_t candidate_occupied_voxels{};
  std::size_t candidate_points_examined{};
  std::size_t finalized_map_candidate_voxel_lookups{};
  std::size_t finalized_map_candidate_occupied_voxels{};
  std::size_t finalized_map_candidate_points_examined{};
  std::size_t finalized_map_stale_fallbacks{};
  std::size_t frozen_objective_evaluations{};
  std::size_t normal_equation_evaluations{};
  std::size_t outer_iterations{};
  std::size_t gauss_newton_trials{};
  std::size_t lm_damping_trials{};
  std::size_t rejected_frozen_cost_trials{};
  std::size_t accepted_steps{};
};

struct LidarRegistrationDiagnostics {
  std::size_t target_count{};
  std::size_t live_target_count{};
  std::size_t finalized_map_target_count{};
  std::size_t correspondences{};
  std::size_t live_correspondences{};
  std::size_t finalized_map_correspondences{};
  double overlap_fraction{};
  double effective_correspondences{};
  double maximum_squared_residual_m2{};
  double huber_delta_m{};
  double characteristic_length_m{};
  double normalized_observable_eigenvalue_threshold{};
  std::size_t observable_rank{};
  double maximum_supported_normalized_information{};
  double normalized_information_cap{};
  double geometric_information_scale{1.0};
  Eigen::Matrix<double, 6, 1> raw_normalized_hessian_eigenvalues{
      Eigen::Matrix<double, 6, 1>::Zero()};
  core::RankAwareInformation normalized_directional_information;
  core::RankAwareInformation physical_information;
};

enum class LidarRegistrationTermination {
  Converged,
  IterationLimitReached,
};

struct LidarRegistrationResult {
  core::StateId source_state;
  core::FusionTime source_time;
  core::Pose3d T_odom_source;
  core::Pose3d source_right_correction;
  std::vector<std::shared_ptr<const LidarFactorSnapshot>> target_snapshots;
  std::shared_ptr<const FinalizedMapLidarFactorSnapshot> finalized_map_snapshot;

  // These matrices all use the right, translation-first physical tangent.
  Eigen::Matrix<double, 6, 6> raw_physical_hessian{Eigen::Matrix<double, 6, 6>::Zero()};
  Eigen::Matrix<double, 6, 6> projected_physical_information{Eigen::Matrix<double, 6, 6>::Zero()};
  // Moore-Penrose inverse on the supported subspace. Zero nullspace entries
  // are not finite confidence; consume them together with the projector.
  Eigen::Matrix<double, 6, 6> supported_physical_covariance{Eigen::Matrix<double, 6, 6>::Zero()};
  Eigen::Matrix<double, 6, 6> normalized_observability_projector{
      Eigen::Matrix<double, 6, 6>::Zero()};

  double initial_robust_cost{};
  double final_robust_cost{};
  LidarRegistrationTermination termination{LidarRegistrationTermination::Converged};
  LidarRegistrationDiagnostics diagnostics;
  LidarRegistrationWorkCounters work;
};

// Canonical aggregate reconstructed only from immutable admission snapshots.
// The initial registration cost is deliberately absent: reassociation changes
// the correspondence objective, so only the final sealed rows can be verified
// after the frontend returns.
struct LidarRegistrationSnapshotAggregate {
  std::size_t target_count{};
  std::size_t live_target_count{};
  std::size_t finalized_map_target_count{};
  std::size_t source_point_count{};
  std::size_t correspondences{};
  double overlap_fraction{};
  double effective_correspondences{};
  double maximum_squared_residual_m2{};
  double huber_delta_m{};
  double final_robust_cost{};
};

enum class LidarRegistrationErrorCode {
  InvalidConfig,
  InvalidSource,
  InvalidTarget,
  InconsistentSeed,
  InsufficientCorrespondences,
  InsufficientObservableRank,
  NoDecreasingStep,
  NumericalFailure,
};

struct LidarRegistrationError {
  LidarRegistrationErrorCode code{};
  std::string detail;
  LidarRegistrationWorkCounters work;
};

// Canonical direct-ICP source view shared by target selection and registration.
// Keeping this as one API prevents a rolling target from estimating overlap on
// a different point population than the solver will actually use.
struct LidarRegistrationSourceSelection {
  std::vector<std::size_t> point_storage_indices;
  std::size_t points_considered{};
  std::size_t points_omitted{};
};

[[nodiscard]] core::Result<LidarRegistrationSourceSelection, LidarRegistrationError>
selectLidarRegistrationSourcePoints(const std::shared_ptr<const LidarRegistrationCloud>& source,
                                    const LidarRegistrationConfig& config = {});

// Reconstructs the multi-target final objective and support diagnostics from
// checksum-bearing, exclusively owned frozen rows. All snapshots must name
// one bit-identical source population. Per-target rows must retain their
// canonical identity order, and overlapping stable source_index ownership is
// rejected independently of process-local point storage order.
[[nodiscard]] core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>
summarizeLidarFactorSnapshots(
    std::span<const std::shared_ptr<const LidarFactorSnapshot>> snapshots);

// Reconstructs the complete mixed objective. Source-row ownership must be
// disjoint across every live snapshot and the optional finalized-map
// snapshot, and all channels must share one frozen Huber population.
[[nodiscard]] core::Result<LidarRegistrationSnapshotAggregate, LidarRegistrationError>
summarizeLidarFactorSnapshots(
    std::span<const std::shared_ptr<const LidarFactorSnapshot>> snapshots,
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& finalized_map_snapshot);

// Canonical ROS/GTSAM-free admission information for one immutable binary
// snapshot. The returned basis and eigenvalues use Meridian's right,
// translation-first physical tangent and exactly the same normalization,
// directional projection, physical cap, and scale as the graph factor.
[[nodiscard]] core::Result<core::RankAwareInformation, LidarRegistrationError>
lidarFactorInformation(const std::shared_ptr<const LidarFactorSnapshot>& snapshot,
                       const LidarRegistrationConfig& config = {}, double information_scale = 1.0);

[[nodiscard]] core::Result<core::RankAwareInformation, LidarRegistrationError>
lidarFinalizedMapFactorInformation(
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& snapshot,
    const LidarRegistrationConfig& config = {}, double information_scale = 1.0);

// Independently recomputes the conservative scalar uncertainty induced by
// every finalized owner pose. The result is 1 + max(lambda_max(J C J^T)) /
// residual_variance and must exactly match the scalar sealed in the snapshot.
// lidarFinalizedMapFactorInformation additionally verifies the full checksum.
[[nodiscard]] core::Result<double, LidarRegistrationError>
lidarFinalizedMapOwnerPoseCovarianceInflation(
    const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& snapshot);

[[nodiscard]] core::Result<LidarRegistrationResult, LidarRegistrationError> registerLidarScan(
    core::StateId source_state, const std::shared_ptr<const LidarRegistrationCloud>& source,
    std::span<const LidarRegistrationTarget> targets, const LidarRegistrationConfig& config = {});

// Runs one association population, one MAD/Huber scale, and one GN/LM solve
// over a rebuilt live pose-aware overlay plus an immutable persistent
// finalized-map read view. Each source row has exactly one owner across both
// physical indexes; an exact distance tie prefers live geometry. Returned
// live rows remain binary owner-local factors and finalized rows remain one
// unary channel with compact owner provenance. An empty live span is valid for
// finalized-map-only recovery.
[[nodiscard]] core::Result<LidarRegistrationResult, LidarRegistrationError> registerLidarScan(
    core::StateId source_state, const std::shared_ptr<const LidarRegistrationCloud>& source,
    std::span<const LidarRegistrationTarget> targets,
    const FinalizedLidarTargetReadView& finalized_map,
    core::LocalGraphRevision live_pose_revision, const LidarRegistrationConfig& config = {});

}  // namespace meridian::local
