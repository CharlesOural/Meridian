#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/graph.hpp"
#include "meridian/local/lidar_registration_cloud.hpp"
#include "meridian/local/map_admission_gate.hpp"

namespace meridian::local {

inline constexpr std::string_view kAcceptedLidarLineageChecksumDomain{
    "meridian.local.lidar_factor_lineage"};
inline constexpr std::uint32_t kAcceptedLidarLineageChecksumSchemaVersion{1U};

enum class AcceptedLidarLineageChecksumErrorCode {
  InvalidLineage,
  Capacity,
  Encoding,
};

struct AcceptedLidarLineageChecksumError {
  AcceptedLidarLineageChecksumErrorCode code{AcceptedLidarLineageChecksumErrorCode::Encoding};
  std::string detail;
};

// Canonical checksum for accepted direct-LiDAR provenance. Unlike the core
// sealed-lineage domain, this domain explicitly records whether each raw root
// checksum is present: current ROS ingress does not yet provide opaque source
// checksums, and absence must remain authenticated rather than fabricated.
[[nodiscard]] core::Result<core::ContentHash, AcceptedLidarLineageChecksumError>
recomputeAcceptedLidarLineageChecksum(const core::ObservationLineage& lineage);

inline constexpr double kDefaultFinalizedLidarMinimumSeparationM{
    0.22360679774997896964};  // sqrt(1 / 20) m, following the KISS-ICP/RKO policy.

struct FinalizedLidarTargetMapConfig {
  core::OdomEpoch odom_epoch;
  core::SensorInstanceId sensor;
  // Registration queries use this spatial partition. A one-cell halo is
  // sufficient because maximum_supported_query_distance_m must not exceed
  // this size.
  double query_voxel_size_m{1.0};
  // One deterministic representative is selected per insertion voxel before
  // the persistent query blocks are considered.
  double insertion_voxel_size_m{0.5};
  std::size_t maximum_points_per_query_voxel{20U};
  double minimum_point_separation_m{kDefaultFinalizedLidarMinimumSeparationM};
  double maximum_supported_query_distance_m{0.5};
  // pruneAround() applies this radius explicitly. Insertion never performs a
  // hidden whole-map scan.
  double maximum_radius_m{100.0};
  std::size_t hard_point_capacity{1'000'000U};
};

// A scan may enter this map only after its owner state and localization batch
// are final. No removal-by-batch API exists intentionally: pre-finality
// rollback remains the live rolling target's responsibility.
struct FinalizedLidarSweep {
  SensorFactorBatchRef batch;
  core::FactorBatchMetadata accepted_batch_metadata;
  core::LocalGraphRevision admission_revision;
  MapAdmissionBatchKind admission_kind{MapAdmissionBatchKind::Regular};
  LocalGraphFinalizedState finalized_state;
  core::CalibrationEpoch calibration;
  std::shared_ptr<const LidarRegistrationCloud> cloud;
};

// Constant-size proof that one FactorBatch was accepted as map-eligible
// localization.  The complete batch is validated at final-map admission, but
// is deliberately not retained here: its lineage may itself reference many
// older map owners.  The receipt authenticates that record by identity and
// checksum while the owner's cloud_lineage below retains the exact raw
// LiDAR/deskew support that produced this geometry.
struct FinalizedLidarAdmissionReceipt {
  core::RecordHeader header;
  core::OdomEpoch odom_epoch;
  core::FusionTime reference_time;
  core::SensorHealthSnapshot health;
  bool map_eligible{};
  core::ObservationLineageId accepted_lineage;
  core::ContentHash accepted_lineage_checksum{};
  core::ContentHash accepted_batch_metadata_checksum{};
};

// One immutable owner shared by every selected point from a finalized sweep.
// Besides avoiding per-point provenance duplication, this preserves the
// numerical pose uncertainty needed by later fixed-map factor weighting.
struct FinalizedLidarTargetOwner {
  SensorFactorBatchRef batch;
  FinalizedLidarAdmissionReceipt admission;
  core::LocalGraphRevision admission_revision;
  MapAdmissionBatchKind admission_kind{MapAdmissionBatchKind::Regular};
  LocalGraphFinalizedState finalized_state;
  core::MeasurementId sweep;
  core::ContentHash cloud_checksum{};
  core::CalibrationEpoch calibration;
  core::ObservationLineage cloud_lineage;
  std::vector<core::MeasurementId> imu_support;
  core::ContentHash final_pose_covariance_checksum{};
};

// Exact point ownership remains available to the later unary/submap-anchor
// factor construction. Geometry from multiple finalized poses is never
// relabelled as geometry owned by the newest pose.
struct FinalizedLidarTargetPoint {
  Eigen::Vector3d point_odom{Eigen::Vector3d::Zero()};
  std::uint32_t source_index{};
  std::shared_ptr<const FinalizedLidarTargetOwner> owner;
};

struct FinalizedLidarTargetInsertStats {
  std::size_t input_points{};
  std::size_t insertion_voxels{};
  std::size_t insertion_selection_discarded_points{};
  std::size_t minimum_separation_discarded_points{};
  std::size_t query_voxel_capacity_discarded_points{};
  std::size_t admitted_points{};
  std::size_t touched_query_voxels{};
  std::size_t retained_query_voxels{};
  std::size_t retained_points{};
  std::uint64_t version{};
  core::ContentHash checksum{};
  // The exact owner sealed by this successful transaction.  Consumers may
  // populate another bounded registration cache without copying the accepted
  // FactorBatch metadata or rediscovering provenance through stored points.
  std::shared_ptr<const FinalizedLidarTargetOwner> owner;
};

struct FinalizedLidarTargetPruneStats {
  std::size_t examined_query_voxels{};
  std::size_t examined_points{};
  std::size_t removed_query_voxels{};
  std::size_t removed_points{};
  std::size_t retained_query_voxels{};
  std::size_t retained_points{};
  std::uint64_t version{};
  core::ContentHash checksum{};
};

struct FinalizedLidarTargetNeighbor {
  bool view_current{};
  bool query_valid{};
  bool found{};
  FinalizedLidarTargetPoint point;
  double distance_squared_m2{};
  // A valid query always visits the canonical 27-cell halo, including absent
  // cells. These counters expose the actual correspondence-search work.
  std::size_t voxel_lookups{};
  std::size_t occupied_voxels{};
  std::size_t points_examined{};
};

struct FinalizedLidarTargetMapStatistics {
  std::size_t insert_attempts{};
  std::size_t rejected_insertions{};
  std::size_t admitted_sweeps{};
  std::size_t input_points{};
  std::size_t insertion_voxels{};
  std::size_t insertion_selection_discarded_points{};
  std::size_t minimum_separation_discarded_points{};
  std::size_t query_voxel_capacity_discarded_points{};
  std::size_t admitted_points{};
  std::size_t prune_attempts{};
  std::size_t prune_transactions{};
  std::size_t pruned_points{};
  std::size_t retained_query_voxels{};
  std::size_t retained_points{};
  std::uint64_t version{};
  core::ContentHash checksum{};
};

enum class FinalizedLidarTargetMapErrorCode {
  InvalidConfig,
  EpochMismatch,
  InvalidIdentity,
  DuplicateSweep,
  DuplicateFactorBatch,
  InvalidRevision,
  InvalidPose,
  InvalidCovariance,
  InvalidCalibration,
  InvalidCloud,
  InvalidMetadata,
  InvalidAdmission,
  MapIneligible,
  StaleFrontier,
  InvalidQuery,
  SpatialIndexFailure,
  PointCapacity,
  ChecksumFailure,
};

struct FinalizedLidarTargetMapError {
  FinalizedLidarTargetMapErrorCode code{};
  std::string detail;
};

namespace detail {
struct FinalizedLidarTargetMapImpl;
}

// Non-owning span-like synchronous snapshot. The map's heap implementation is
// stable across map moves, so a move preserves a view; destroying the owning
// map still invalidates every outstanding view. Mutation makes the next query
// report view_current=false instead of mixing map versions.
class FinalizedLidarTargetReadView final {
public:
  [[nodiscard]] core::OdomEpoch odomEpoch() const noexcept { return odom_epoch_; }
  [[nodiscard]] core::SensorInstanceId sensor() const noexcept { return sensor_; }
  [[nodiscard]] std::uint64_t version() const noexcept { return version_; }
  [[nodiscard]] const core::ContentHash& checksum() const noexcept { return checksum_; }
  [[nodiscard]] FinalizedLidarTargetNeighbor nearestExact(
      const Eigen::Vector3d& query_odom, double requested_maximum_distance_m) const noexcept;

private:
  FinalizedLidarTargetReadView(const detail::FinalizedLidarTargetMapImpl* map,
                               core::OdomEpoch odom_epoch, core::SensorInstanceId sensor,
                               std::uint64_t version, core::ContentHash checksum)
      : map_(map),
        odom_epoch_(odom_epoch),
        sensor_(sensor),
        version_(version),
        checksum_(checksum) {}

  const detail::FinalizedLidarTargetMapImpl* map_{};
  core::OdomEpoch odom_epoch_;
  core::SensorInstanceId sensor_;
  std::uint64_t version_{};
  core::ContentHash checksum_{};

  friend class FinalizedLidarTargetMap;
};

// ROS/GTSAM-free persistent direct-registration target. It is deliberately a
// single-thread synchronous owner: callers serialize insertion/pruning and may
// perform allocation-free const queries between mutations. Only touched query
// blocks are staged on admission; the whole map is never copied.
//
// statistics().checksum is a rolling canonical digest. Creation hashes the
// complete configuration; each successful insert/prune hashes the previous
// digest and a canonical transaction record containing all admitted/removed
// coordinates and provenance. This is O(transaction), not O(map), and
// therefore encodes both deterministic content and its provenance-preserving
// history.
class FinalizedLidarTargetMap final {
public:
  [[nodiscard]] static core::Result<FinalizedLidarTargetMap, FinalizedLidarTargetMapError> create(
      FinalizedLidarTargetMapConfig config);

  ~FinalizedLidarTargetMap();
  FinalizedLidarTargetMap(FinalizedLidarTargetMap&&) noexcept;
  FinalizedLidarTargetMap& operator=(FinalizedLidarTargetMap&&) noexcept;
  FinalizedLidarTargetMap(const FinalizedLidarTargetMap&) = delete;
  FinalizedLidarTargetMap& operator=(const FinalizedLidarTargetMap&) = delete;

  [[nodiscard]] core::Result<FinalizedLidarTargetInsertStats, FinalizedLidarTargetMapError>
  insertFinalizedSweep(FinalizedLidarSweep sweep);

  // Removes points farther than config.maximum_radius_m from origin. It is a
  // separately scheduled O(map) operation; no equivalent scan is hidden in
  // insertFinalizedSweep(). A no-op prune leaves version/checksum unchanged.
  [[nodiscard]] core::Result<FinalizedLidarTargetPruneStats, FinalizedLidarTargetMapError>
  pruneAround(const Eigen::Vector3d& origin_odom);

  // Exact Euclidean <= requested_maximum_distance_m lookup over 27 neighboring
  // query cells. Equal distances resolve by owner state, owner sweep, then
  // source index, independently of unordered_map layout. The requested gate
  // must be positive and no larger than the configured supported maximum.
  [[nodiscard]] FinalizedLidarTargetNeighbor nearestExact(
      const Eigen::Vector3d& query_odom, double requested_maximum_distance_m) const noexcept;

  [[nodiscard]] FinalizedLidarTargetReadView readView() const noexcept;

  [[nodiscard]] const FinalizedLidarTargetMapConfig& config() const noexcept;
  [[nodiscard]] const FinalizedLidarTargetMapStatistics& statistics() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

private:
  explicit FinalizedLidarTargetMap(
      std::unique_ptr<detail::FinalizedLidarTargetMapImpl> implementation);
  std::unique_ptr<detail::FinalizedLidarTargetMapImpl> implementation_;
};

}  // namespace meridian::local
