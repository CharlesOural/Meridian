#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sophus/se3.hpp>
#include <span>
#include <string>
#include <vector>

#include "meridian/core/ids.hpp"
#include "meridian/core/time.hpp"
#include "meridian/local_rt/lidar/voxel_target.hpp"

namespace meridian::local_rt::lidar {

struct ScanToMapOptions final {
  double target_downsample_voxel_m{0.4};
  double source_downsample_voxel_m{1.0};
  std::size_t maximum_active_owners{4U};
  std::size_t maximum_factor_rows{600U};
  std::size_t minimum_correspondences{100U};
  std::size_t maximum_icp_iterations{8U};
  std::size_t maximum_backtracking_steps{6U};
  double maximum_correspondence_distance_m{1.5};
  double point_sigma_m{0.1};
  double huber_mad_multiplier{2.0};
  double minimum_huber_scale_m{0.1};
  double maximum_huber_scale_m{1.0};
  double translation_convergence_m{0.002};
  double rotation_convergence_rad{0.001};
  double maximum_translation_step_m{0.5};
  double maximum_rotation_step_rad{0.15};
  double maximum_prediction_correction_m{3.0};
  double maximum_prediction_correction_rad{0.6};
  double relative_rank_tolerance{1.0e-6};
  double lm_damping{1.0e-4};
  VoxelTargetOptions active_owner_index{.voxel_size_m = 0.5,
                                        .retention_radius_m = 80.0,
                                        .max_voxels = 30'000U,
                                        .max_points_per_voxel = 3U,
                                        .minimum_point_spacing_m = 0.05,
                                        .max_neighbor_voxel_radius = 3U};
  VoxelTargetOptions finalized_base{.voxel_size_m = 0.5,
                                    .retention_radius_m = 100.0,
                                    .max_voxels = 100'000U,
                                    .max_points_per_voxel = 3U,
                                    .minimum_point_spacing_m = 0.1,
                                    .max_neighbor_voxel_radius = 3U};
};

struct ScanFrame final {
  core::StateId state_id;
  core::TimeNs time;
  PointCloud target_points_lidar;
  PointCloud source_points_lidar;
};

struct ScanToMapRow final {
  Point3d source_lidar{Point3d::Zero()};
  std::optional<core::StateId> active_target_state;
  // Active rows store a target point in the target owner's LiDAR frame.
  // Finalized rows store a target point in odom.
  Point3d target{Point3d::Zero()};
  double sqrt_weight_over_sigma{};
  double association_distance_m{};
};

struct ScanToMapTiming final {
  std::int64_t owner_selection_ns{};
  std::int64_t live_composite_rebuild_ns{};
  std::int64_t active_query_ns{};
  std::int64_t finalized_query_ns{};
  std::int64_t robust_scale_ns{};
  std::int64_t linearization_ns{};
  std::int64_t solve_ns{};
  std::int64_t total_ns{};
  std::uint64_t active_queries{};
  std::uint64_t active_voxel_probes{};
  std::uint64_t active_candidate_points{};
  std::uint64_t finalized_queries{};
  std::uint64_t finalized_voxel_probes{};
  std::uint64_t finalized_candidate_points{};
};

enum class ScanToMapStatus : std::uint8_t {
  kAccepted,
  kEmptyTarget,
  kInsufficientCorrespondences,
  kDegenerate,
  kNoDecreasingStep,
  kPredictionCorrectionExceeded,
  kNumericalFailure,
};

struct ScanToMapResult final {
  ScanToMapStatus status{ScanToMapStatus::kNumericalFailure};
  Sophus::SE3d odom_from_imu{};
  std::vector<ScanToMapRow> rows;
  std::size_t selected_active_owners{};
  std::size_t active_rows{};
  std::size_t finalized_rows{};
  std::size_t correspondences_before_cap{};
  std::size_t iterations{};
  std::size_t observable_rank{};
  double huber_scale_m{};
  double rmse_m{};
  double robust_cost{};
  double condition_number{};
  double correction_translation_m{};
  double correction_rotation_rad{};
  ScanToMapTiming timing;

  [[nodiscard]] bool accepted() const noexcept { return status == ScanToMapStatus::kAccepted; }
};

struct FinalizedTargetStats final {
  std::size_t points{};
  std::size_t voxels{};
  VoxelTargetUpdateStats update;
};

// Two-layer registration target. Active geometry remains authoritative in its
// owner-local frame and therefore relinearizable through its owner pose. A
// lazily rebuilt flat odom-frame composite is only the live query accelerator;
// geometry is copied permanently into the finalized base when its owner leaves
// the lag.
class ScanToMapTarget final {
public:
  ScanToMapTarget(ScanToMapOptions options, Sophus::SE3d imu_from_lidar);
  ~ScanToMapTarget();

  ScanToMapTarget(const ScanToMapTarget& other);
  ScanToMapTarget& operator=(const ScanToMapTarget& other);
  ScanToMapTarget(ScanToMapTarget&&) noexcept;
  ScanToMapTarget& operator=(ScanToMapTarget&&) noexcept;

  [[nodiscard]] const ScanToMapOptions& options() const noexcept { return options_; }
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] bool hasActiveOwner(core::StateId state_id) const noexcept;
  [[nodiscard]] std::size_t activeOwnerCount() const noexcept;
  [[nodiscard]] std::size_t finalizedPointCount() const noexcept;
  [[nodiscard]] std::size_t finalizedVoxelCount() const noexcept;
  // Observer-only snapshot in odom: persistent finalized points followed by
  // the current-pose projection of every active owner's indexed points. The
  // returned cloud is lexicographically ordered and does not mutate targets.
  [[nodiscard]] PointCloud registrationMapPointCloud() const;

  void admit(ScanFrame frame, const Sophus::SE3d& odom_from_imu);
  void updateOwnerPose(core::StateId state_id, const Sophus::SE3d& odom_from_imu);
  [[nodiscard]] FinalizedTargetStats finalize(core::StateId state_id,
                                              const Sophus::SE3d& odom_from_imu,
                                              const Point3d& retention_center_odom);

  [[nodiscard]] ScanToMapResult registerScan(std::span<const Point3d> source_points_lidar,
                                             const Sophus::SE3d& predicted_odom_from_imu) const;

  // Associates a scan at the supplied pose exactly once. Unlike registerScan,
  // this does not optimize the pose: it computes correspondences, robust IRLS
  // weights, quality metrics and the deterministic final factor-row cap. This
  // is the normal fixed-lag tracking primitive; iterative registration remains
  // available for initialization and recovery.
  [[nodiscard]] ScanToMapResult associateScan(std::span<const Point3d> source_points_lidar,
                                              const Sophus::SE3d& odom_from_imu) const;

private:
  struct Impl;
  ScanToMapOptions options_;
  Sophus::SE3d imu_from_lidar_;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* toString(ScanToMapStatus status) noexcept;

}  // namespace meridian::local_rt::lidar
