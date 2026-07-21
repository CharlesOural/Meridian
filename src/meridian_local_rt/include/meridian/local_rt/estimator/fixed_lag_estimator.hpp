#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/navigation.hpp"
#include "meridian/local_rt/combined_preintegration.hpp"
#include "meridian/local_rt/estimator/square_root_marginalization.hpp"
#include "meridian/local_rt/imu_types.hpp"
#include "meridian/local_rt/lidar/scan_to_map.hpp"

namespace meridian::local_rt::estimator {

struct FixedLagEstimatorOptions final {
  std::size_t maximum_states{8U};
  std::int64_t maximum_lag_ns{1'000'000'000};
  std::size_t maximum_solver_iterations{6U};
  std::size_t solver_threads{1U};
  double maximum_solver_time_s{0.08};
  double function_tolerance{1.0e-6};
  double gradient_tolerance{1.0e-10};
  double parameter_tolerance{1.0e-8};
  double maximum_translation_correction_m{3.0};
  double maximum_rotation_correction_rad{0.6};
  double maximum_prior_translation_m{2.0};
  double maximum_prior_rotation_rad{0.5};
  double maximum_prior_motion_norm{5.0};

  double initial_translation_sigma_m{0.01};
  double initial_rotation_sigma_rad{0.01};
  double initial_velocity_sigma_m_s{0.5};
  double initial_gyro_bias_sigma_rad_s{0.02};
  double initial_accel_bias_sigma_m_s2{0.3};
  SquareRootMarginalizationOptions marginalization{};
};

struct FixedLagTiming final {
  std::int64_t association_first_ns{};
  std::int64_t association_second_ns{};
  std::int64_t registration_ns{};
  std::int64_t factor_build_ns{};
  std::int64_t problem_build_first_ns{};
  std::int64_t problem_build_second_ns{};
  std::int64_t problem_build_ns{};
  std::int64_t ceres_solve_first_ns{};
  std::int64_t ceres_solve_second_ns{};
  std::int64_t ceres_solve_ns{};
  std::int64_t validation_ns{};
  std::int64_t marginalization_evaluate_ns{};
  std::int64_t marginalization_eliminate_ns{};
  std::int64_t marginalization_prior_build_ns{};
  std::int64_t target_finalize_ns{};
  std::int64_t target_admit_ns{};
  std::int64_t commit_ns{};
  std::int64_t total_ns{};
};

enum class FixedLagStatus : std::uint8_t {
  kAccepted,
  kRegistrationRejected,
  kOptimizationRejected,
  kMarginalizationRejected,
  kInvalidInput,
};

struct FixedLagUpdate final {
  FixedLagStatus status{FixedLagStatus::kInvalidInput};
  std::string reason;
  std::optional<core::NavigationState> predicted;
  std::optional<core::NavigationState> optimized;
  std::vector<core::NavigationState> newly_finalized;
  lidar::ScanToMapResult first_association;
  lidar::ScanToMapResult registration;
  std::size_t active_states{};
  std::size_t imu_factors{};
  std::size_t active_lidar_groups{};
  std::size_t finalized_lidar_groups{};
  std::size_t active_lidar_rows{};
  std::size_t finalized_lidar_rows{};
  std::size_t lidar_rows{};
  std::size_t finalized_map_points{};
  std::size_t prior_rank{};
  std::size_t marginalizations{};
  std::size_t association_passes{};
  std::size_t solver_iterations{};
  std::size_t reassociated_rows{};
  std::size_t rejected_stale_rows{};
  double initial_cost{};
  double final_cost{};
  double correction_translation_m{};
  double correction_rotation_rad{};
  FixedLagTiming timing;

  [[nodiscard]] bool accepted() const noexcept { return status == FixedLagStatus::kAccepted; }
};

// Small, bounded, ROS-free Ceres fixed-lag smoother. Every accepted sweep owns
// immutable scan-local target geometry. Active direct-LiDAR groups remain
// binary between source and owner poses; future rows against geometry moved to
// the finalized base are unary. Adjacent states are coupled by one Combined IMU
// factor. Factors touching the oldest state are absorbed once into a fixed-
// linearization square-root prior before their explicit removal.
class FixedLagEstimator final {
public:
  FixedLagEstimator(FixedLagEstimatorOptions options, lidar::ScanToMapOptions scan_options,
                    core::Pose3d T_imu_lidar);
  ~FixedLagEstimator();

  FixedLagEstimator(const FixedLagEstimator&) = delete;
  FixedLagEstimator& operator=(const FixedLagEstimator&) = delete;

  void initialize(const core::NavigationState& seed,
                  std::optional<lidar::ScanFrame> anchor = std::nullopt);

  [[nodiscard]] bool initialized() const noexcept;
  [[nodiscard]] core::NavigationState latestState() const;
  [[nodiscard]] std::vector<core::NavigationState> activeStates() const;
  // Observer-only odom-frame snapshot of the exact live + finalized geometry
  // available to normal scan-to-map association.
  [[nodiscard]] lidar::PointCloud registrationMapPointCloud() const;

  [[nodiscard]] FixedLagUpdate addSweep(lidar::ScanFrame frame,
                                        const CombinedPreintegration& preintegration,
                                        const core::NavigationState& predicted);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] const char* toString(FixedLagStatus status) noexcept;

}  // namespace meridian::local_rt::estimator
