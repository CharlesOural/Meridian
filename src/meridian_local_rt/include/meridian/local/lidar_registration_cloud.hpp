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
#include "meridian/local/lidar_composite_target.hpp"
#include "meridian/local/lidar_deskew.hpp"

namespace meridian::local {

// One direct-registration return. `source_index` binds this row exactly to the
// owned tracking-deskewed payload; no estimated surface model or covariance is
// stored in this artifact.
struct LidarRegistrationPoint {
  Eigen::Vector3d point{Eigen::Vector3d::Zero()};
  std::uint32_t source_index{};
  float intensity{};
  std::uint16_t ring{};
};

struct LidarPreprocessStats {
  std::size_t input_points{};
  std::size_t valid_range_points{};
  std::size_t deterministic_voxel_points{};
};

struct LidarPreprocessConfig {
  // Execution-only bound reserved for deterministic row-parallel preprocessing.
  // It is deliberately excluded from sealed artifact checksum semantics.
  std::size_t parallel_worker_count{4U};
  double minimum_range_m{1.0};
  double maximum_range_m{80.0};
  double voxel_size_m{0.30};
  // If the voxel view exceeds this bound, a stable spatial hash selects the
  // retained voxels before rows are restored to canonical source-index order.
  std::size_t maximum_output_points{60'000U};
};

struct LidarRegistrationIndexConfig {
  // Direct-registration lookup cell size. It is independent of preprocessing
  // voxel size and is sealed into the artifact checksum.
  double voxel_resolution_m{1.0};
};

enum class LidarPreprocessErrorCode {
  InvalidConfig,
  InvalidLayout,
  InvalidLineage,
  EmptyInput,
  NoUsablePoints,
  ChecksumFailure,
  SpatialIndexFailure,
};

struct LidarPreprocessError {
  LidarPreprocessErrorCode code{};
  std::string detail;
};

inline constexpr std::string_view kLidarRegistrationCloudChecksumDomain{
    "meridian.local.lidar_registration_cloud.artifact"};
inline constexpr std::uint32_t kLidarRegistrationCloudChecksumSchemaVersion{4U};

// Mutable construction input only. create() validates and canonicalizes all
// source identities, verifies selected rows against the temporary full deskew
// payload, seals the registration-only lineage/checksum, and builds the sole
// exact registration index. The full deskew payload is discarded afterward;
// accepted dense mapping re-deskews the immutable raw sweep independently.
struct LidarRegistrationCloudData {
  core::MeasurementId source_sweep;
  core::FusionTime reference_time;
  core::Pose3d T_odom_imu_seed;
  // Temporary provisional tracking-deskewed returns used only to validate the
  // selected registration rows during construction.
  core::LidarLayout layout;
  std::unique_ptr<core::LidarPoints> points_in_reference_imu;
  std::vector<LidarRegistrationPoint> points;
  // IMU IDs used only for deskew/tracking conditioning. They are provenance,
  // never LiDAR-factor information.
  std::vector<core::MeasurementId> imu_support;
  LidarPreprocessStats stats;
  core::ObservationLineage lineage;
};

struct ExactLidarNeighbor {
  bool found{};
  std::size_t point_storage_index{};
  std::uint32_t source_index{};
  double distance_squared_m2{};
  std::size_t voxel_lookups{};
  std::size_t points_examined{};
};

// Immutable scan-local direct-registration artifact. It deliberately owns no
// dense mapping payload. Share only as std::shared_ptr<const
// LidarRegistrationCloud>. nearestExact() performs no allocation or cache
// mutation and is safe for concurrent readers.
class LidarRegistrationCloud final {
public:
  [[nodiscard]] static core::Result<std::shared_ptr<const LidarRegistrationCloud>,
                                    LidarPreprocessError>
  create(LidarRegistrationCloudData data, LidarRegistrationIndexConfig index_config = {});

  ~LidarRegistrationCloud();
  LidarRegistrationCloud(const LidarRegistrationCloud&) = delete;
  LidarRegistrationCloud& operator=(const LidarRegistrationCloud&) = delete;
  LidarRegistrationCloud(LidarRegistrationCloud&&) = delete;
  LidarRegistrationCloud& operator=(LidarRegistrationCloud&&) = delete;

  [[nodiscard]] double exactIndexVoxelResolutionM() const noexcept;
  [[nodiscard]] std::size_t exactIndexVoxelCount() const noexcept;
  [[nodiscard]] std::span<const std::size_t> canonicalPointStorageIndices() const noexcept;
  // Shared owner-local geometry for the composite multi-owner registration
  // index. It is materialized once with the immutable scan artifact so normal
  // tracking never copies every target point again.
  [[nodiscard]] const std::shared_ptr<const LidarCompositeTargetPoints>&
  compositeTargetPoints() const noexcept;
  // Exact Euclidean radius query. Equal-distance candidates are ordered by
  // (source_index, storage_index), independently of hash-table layout.
  [[nodiscard]] ExactLidarNeighbor nearestExact(const Eigen::Vector3d& query,
                                                double maximum_distance_m) const noexcept;

  const core::MeasurementId source_sweep;
  const core::FusionTime reference_time;
  const core::Pose3d T_odom_imu_seed;
  const core::LidarLayout layout;
  const std::vector<LidarRegistrationPoint> points;
  const std::vector<core::MeasurementId> imu_support;
  const LidarPreprocessStats stats;
  const core::ObservationLineage lineage;
  const core::ContentHash checksum;

private:
  struct ExactIndex;
  LidarRegistrationCloud(LidarRegistrationCloudData data, core::ObservationLineage sealed_lineage,
                         core::ContentHash cloud_checksum,
                         std::shared_ptr<const LidarCompositeTargetPoints> composite_points,
                         std::unique_ptr<const ExactIndex> exact_index);

  const std::shared_ptr<const LidarCompositeTargetPoints> composite_points_;
  const std::unique_ptr<const ExactIndex> exact_index_;
};

[[nodiscard]] core::Result<std::shared_ptr<const LidarRegistrationCloud>, LidarPreprocessError>
buildLidarRegistrationCloud(DeskewedSweep deskewed, core::ObservationLineage lineage,
                            const LidarPreprocessConfig& config = {},
                            LidarRegistrationIndexConfig index_config = {});

}  // namespace meridian::local
