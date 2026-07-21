#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sophus/se3.hpp>
#include <span>

#include "meridian/local_rt/lidar/voxel_target.hpp"

namespace meridian::local_rt::lidar {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

struct PointToPointRegistrationOptions final {
  std::size_t max_iterations{};
  std::size_t max_source_points{};
  std::size_t minimum_correspondences{};
  double max_correspondence_distance_m{};
  double geman_mcclure_scale_m{};
  double translation_convergence_m{};
  double rotation_convergence_rad{};
  double max_translation_step_m{};
  double max_rotation_step_rad{};
  double relative_rank_tolerance{};
};

enum class RegistrationStatus : std::uint8_t {
  kConverged,
  kIterationLimit,
  kEmptySource,
  kSourcePointLimitExceeded,
  kEmptyTarget,
  kIncompatibleTarget,
  kInvalidInitialGuess,
  kInsufficientCorrespondences,
  kDegenerateGeometry,
  kNumericalFailure,
};

struct RegistrationQuality final {
  std::size_t input_source_points{};
  std::size_t finite_source_points{};
  std::size_t correspondences{};
  std::size_t iterations{};
  std::size_t observable_rank{};
  double inlier_ratio{};
  double robust_cost{};
  double point_rmse_m{};
  double weighted_point_rmse_m{};
  double residual_variance_m2{};
  double hessian_condition_number{};
  double final_translation_step_m{};
  double final_rotation_step_rad{};
  bool step_was_limited{};
  Vector6d hessian_eigenvalues{Vector6d::Zero()};
};

struct PointToPointRegistrationResult final {
  RegistrationStatus status{RegistrationStatus::kNumericalFailure};
  Sophus::SE3d T_target_source{};
  // Both matrices use the final estimate's translation-first left tangent
  // [delta_translation_m, delta_rotation_rad]. Covariance is the robust
  // residual-variance-scaled inverse Hessian and is absent for deficient data.
  Matrix6d hessian{Matrix6d::Zero()};
  std::optional<Matrix6d> covariance;
  RegistrationQuality quality;

  // Reaching the iteration bound still returns a finite candidate for explicit
  // caller-side quality gating. Every other failure leaves the last safe pose.
  [[nodiscard]] bool hasUsableEstimate() const noexcept {
    return status == RegistrationStatus::kConverged ||
           status == RegistrationStatus::kIterationLimit;
  }
};

// Direct point-to-point registration with a translation-first, left SE(3)
// perturbation: T <- Exp([delta_translation, delta_rotation]) * T. Each
// iteration rebuilds nearest-neighbour pairs, applies Geman-McClure IRLS
// weights, bounds both step components, and performs one Gauss-Newton update.
class DirectPointToPointRegistration final {
public:
  explicit DirectPointToPointRegistration(PointToPointRegistrationOptions options);

  [[nodiscard]] const PointToPointRegistrationOptions& options() const noexcept { return options_; }

  [[nodiscard]] PointToPointRegistrationResult align(
      std::span<const Point3d> source_points, const BoundedVoxelTarget& target,
      const Sophus::SE3d& initial_T_target_source) const;

private:
  PointToPointRegistrationOptions options_;
};

}  // namespace meridian::local_rt::lidar
