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

namespace meridian::local {

inline constexpr std::string_view kLidarCompositeTargetChecksumDomain{
    "meridian.local.lidar_composite_target"};
inline constexpr std::uint32_t kLidarCompositeTargetChecksumSchemaVersion{3U};

enum class LidarCompositeOwnerKind : std::uint8_t {
  Live,
  Finalized,
};

// Immutable owner-frame geometry used by the registration acceleration view.
// local_point_index is the stable vector index; source_index preserves the
// original LiDAR return identity used by frozen factor rows.
struct LidarCompositeTargetPoint {
  Eigen::Vector3d point_owner{Eigen::Vector3d::Zero()};
  std::uint32_t source_index{};
};

using LidarCompositeTargetPoints = std::vector<LidarCompositeTargetPoint>;

// Construction input for one pose-owned target. Geometry remains authoritative
// in the owner frame. The composite index stores only an acceleration copy in
// odom and never transfers ownership to another pose.
struct LidarCompositeTargetOwnerInput {
  LidarCompositeOwnerKind kind{LidarCompositeOwnerKind::Live};
  core::StateId state;
  core::FusionTime time;
  core::MeasurementId sweep;
  core::LocalGraphRevision pose_revision;
  core::ContentHash geometry_checksum{};
  core::Pose3d T_odom_owner;
  std::shared_ptr<const LidarCompositeTargetPoints> points;
};

struct LidarCompositeTargetConfig {
  double voxel_size_m{1.0};
  // This cap is configurable for scan density and compute budget. The
  // implementation enforces a conservative hard ceiling independently of
  // caller input.
  std::size_t maximum_points_per_voxel{20U};
  std::size_t maximum_total_points{150'000U};
  std::size_t maximum_owners{15U};
  // Queries visit a deterministic cube of ceil(distance / voxel_size) cells
  // around the source voxel. The explicit radius cap preserves bounded work
  // for framework users whose correspondence gate spans several voxels;
  // production profiles normally use radius one (a fixed 3x3x3 lookup).
  double maximum_query_distance_m{0.5};
  std::size_t maximum_voxel_search_radius{64U};

  bool operator==(const LidarCompositeTargetConfig&) const = default;
};

struct LidarCompositeTargetBuildStats {
  std::size_t input_owners{};
  std::size_t input_points{};
  std::size_t retained_owners{};
  std::size_t occupied_voxels{};
  std::size_t retained_points{};
  std::size_t per_voxel_capacity_discarded_points{};
  std::size_t total_capacity_discarded_points{};
};

struct LidarCompositeTargetNeighbor {
  bool query_valid{};
  bool found{};
  std::size_t owner_slot{};
  std::size_t local_point_index{};
  std::uint32_t source_index{};
  Eigen::Vector3d point_owner{Eigen::Vector3d::Zero()};
  Eigen::Vector3d point_odom{Eigen::Vector3d::Zero()};
  double distance_squared_m2{};
  // Occupied cells and examined retained points expose the data-dependent
  // part of the bounded voxel work.
  std::size_t voxel_lookups{};
  std::size_t occupied_voxels{};
  std::size_t points_examined{};
};

enum class LidarCompositeTargetErrorCode {
  InvalidConfig,
  EmptyOwners,
  InvalidOwner,
  DuplicateOwner,
  InvalidGeometry,
  Capacity,
  SpatialIndexFailure,
  ChecksumFailure,
};

struct LidarCompositeTargetError {
  LidarCompositeTargetErrorCode code{LidarCompositeTargetErrorCode::InvalidConfig};
  std::string detail;
};

// One immutable, ROS/GTSAM-free acceleration view over several pose-owned
// clouds. Owner records are canonicalized independently of input order. A
// nearest-neighbour result always names the original owner slot and local point
// index; the merged odom cloud is never the mathematical map representation.
class LidarCompositeTarget final {
public:
  [[nodiscard]] static core::Result<LidarCompositeTarget, LidarCompositeTargetError> create(
      LidarCompositeTargetConfig config, std::vector<LidarCompositeTargetOwnerInput> owners);

  ~LidarCompositeTarget();
  LidarCompositeTarget(LidarCompositeTarget&&) noexcept;
  LidarCompositeTarget& operator=(LidarCompositeTarget&&) noexcept;
  LidarCompositeTarget(const LidarCompositeTarget&) = delete;
  LidarCompositeTarget& operator=(const LidarCompositeTarget&) = delete;

  // Exact double-precision reranking uses authoritative owner-local geometry
  // and its pose. Cached odom coordinates decide only which voxel stores the
  // candidate. Equal distances prefer live geometry, then canonical owner and
  // point identity.
  [[nodiscard]] LidarCompositeTargetNeighbor nearestExact(const Eigen::Vector3d& query_odom,
                                                          double maximum_distance_m) const noexcept;

  [[nodiscard]] const LidarCompositeTargetConfig& config() const noexcept;
  [[nodiscard]] const LidarCompositeTargetBuildStats& statistics() const noexcept;
  [[nodiscard]] std::span<const LidarCompositeTargetOwnerInput> owners() const noexcept;
  // O(owners) input-recipe checksum. Owner geometry is named transitively by
  // geometry_checksum; raw points and the derived spatial index are not
  // re-encoded here.
  [[nodiscard]] const core::ContentHash& checksum() const noexcept;

private:
  struct Impl;
  [[nodiscard]] static core::Result<core::ContentHash, LidarCompositeTargetError> computeChecksum(
      const Impl& implementation);
  explicit LidarCompositeTarget(std::unique_ptr<const Impl> implementation);
  std::unique_ptr<const Impl> implementation_;
};

}  // namespace meridian::local
