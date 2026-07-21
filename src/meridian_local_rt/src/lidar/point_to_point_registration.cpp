#include "meridian/local_rt/lidar/point_to_point_registration.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sophus/so3.hpp>
#include <stdexcept>

namespace meridian::local_rt::lidar {
namespace {

using Matrix3x6d = Eigen::Matrix<double, 3, 6>;

bool optionsAreValid(const PointToPointRegistrationOptions& options) noexcept {
  return options.max_iterations > 0U && options.max_source_points > 0U &&
         options.minimum_correspondences >= 3U &&
         options.minimum_correspondences <= options.max_source_points &&
         std::isfinite(options.max_correspondence_distance_m) &&
         options.max_correspondence_distance_m > 0.0 &&
         std::isfinite(options.geman_mcclure_scale_m) && options.geman_mcclure_scale_m > 0.0 &&
         std::isfinite(options.translation_convergence_m) &&
         options.translation_convergence_m > 0.0 &&
         std::isfinite(options.rotation_convergence_rad) &&
         options.rotation_convergence_rad > 0.0 && std::isfinite(options.max_translation_step_m) &&
         options.max_translation_step_m > 0.0 && std::isfinite(options.max_rotation_step_rad) &&
         options.max_rotation_step_rad > 0.0 && std::isfinite(options.relative_rank_tolerance) &&
         options.relative_rank_tolerance > 0.0 && options.relative_rank_tolerance < 1.0;
}

struct Linearization final {
  Matrix6d hessian{Matrix6d::Zero()};
  Vector6d gradient{Vector6d::Zero()};
  std::size_t correspondences{};
  double robust_cost{};
  double unweighted_squared_error{};
  double weighted_squared_error{};
  double weight_sum{};
};

Linearization linearize(std::span<const Point3d> source_points, const BoundedVoxelTarget& target,
                        const Sophus::SE3d& T_target_source,
                        const PointToPointRegistrationOptions& options) {
  Linearization result;
  const double scale_squared = options.geman_mcclure_scale_m * options.geman_mcclure_scale_m;
  const double scale_fourth = scale_squared * scale_squared;

  for (const Point3d& source_point : source_points) {
    const Point3d transformed = T_target_source * source_point;
    if (!transformed.allFinite()) {
      continue;
    }
    const auto neighbor =
        target.nearestNeighbor(transformed, options.max_correspondence_distance_m);
    if (!neighbor.has_value()) {
      continue;
    }

    const Point3d residual = transformed - neighbor->point;
    const double squared_error = residual.squaredNorm();
    const double denominator = scale_squared + squared_error;
    const double weight = scale_fourth / (denominator * denominator);

    Matrix3x6d jacobian;
    jacobian.template leftCols<3>().setIdentity();
    jacobian.template rightCols<3>() = -Sophus::SO3d::hat(transformed);
    result.hessian.noalias() += weight * jacobian.transpose() * jacobian;
    result.gradient.noalias() += weight * jacobian.transpose() * residual;
    result.robust_cost += scale_squared * squared_error / denominator;
    result.unweighted_squared_error += squared_error;
    result.weighted_squared_error += weight * squared_error;
    result.weight_sum += weight;
    ++result.correspondences;
  }
  result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
  return result;
}

struct HessianAnalysis final {
  Vector6d eigenvalues{Vector6d::Zero()};
  std::size_t rank{};
  double condition_number{std::numeric_limits<double>::infinity()};
  std::optional<Matrix6d> inverse;
};

std::optional<HessianAnalysis> analyzeHessian(const Matrix6d& hessian,
                                              double relative_rank_tolerance) {
  if (!hessian.allFinite()) {
    return std::nullopt;
  }
  const Eigen::SelfAdjointEigenSolver<Matrix6d> eigensolver(hessian);
  if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite()) {
    return std::nullopt;
  }

  HessianAnalysis result;
  result.eigenvalues = eigensolver.eigenvalues();
  const double maximum = result.eigenvalues.maxCoeff();
  const double threshold = std::max(0.0, maximum * relative_rank_tolerance);
  for (const double eigenvalue : result.eigenvalues) {
    if (eigenvalue > threshold) {
      ++result.rank;
    }
  }
  if (result.rank != 6U || result.eigenvalues[0] <= 0.0) {
    return result;
  }
  result.condition_number = maximum / result.eigenvalues[0];

  Eigen::LDLT<Matrix6d> decomposition(hessian);
  if (decomposition.info() != Eigen::Success || !decomposition.isPositive()) {
    return result;
  }
  Matrix6d inverse = decomposition.solve(Matrix6d::Identity());
  if (decomposition.info() == Eigen::Success && inverse.allFinite()) {
    result.inverse = 0.5 * (inverse + inverse.transpose());
  }
  return result;
}

void populateFinalLinearization(const Linearization& linearization,
                                std::size_t finite_source_points, double relative_rank_tolerance,
                                PointToPointRegistrationResult& result) {
  result.hessian = linearization.hessian;
  result.quality.correspondences = linearization.correspondences;
  result.quality.inlier_ratio = finite_source_points == 0U
                                    ? 0.0
                                    : static_cast<double>(linearization.correspondences) /
                                          static_cast<double>(finite_source_points);
  result.quality.robust_cost = linearization.robust_cost;
  if (linearization.correspondences > 0U) {
    result.quality.point_rmse_m = std::sqrt(linearization.unweighted_squared_error /
                                            static_cast<double>(linearization.correspondences));
  }
  if (linearization.weight_sum > 0.0) {
    result.quality.weighted_point_rmse_m =
        std::sqrt(linearization.weighted_squared_error / linearization.weight_sum);
  }

  const std::size_t residual_dimensions = 3U * linearization.correspondences;
  const std::size_t degrees_of_freedom = residual_dimensions > 6U ? residual_dimensions - 6U : 1U;
  result.quality.residual_variance_m2 =
      linearization.weighted_squared_error / static_cast<double>(degrees_of_freedom);

  const auto analysis = analyzeHessian(linearization.hessian, relative_rank_tolerance);
  if (!analysis.has_value()) {
    result.quality.observable_rank = 0U;
    result.quality.hessian_condition_number = std::numeric_limits<double>::infinity();
    result.covariance.reset();
    return;
  }
  result.quality.hessian_eigenvalues = analysis->eigenvalues;
  result.quality.observable_rank = analysis->rank;
  result.quality.hessian_condition_number = analysis->condition_number;
  if (analysis->inverse.has_value()) {
    result.covariance =
        result.quality.residual_variance_m2 * static_cast<const Matrix6d&>(*analysis->inverse);
  } else {
    result.covariance.reset();
  }
}

bool limitStep(Eigen::Ref<Eigen::Vector3d> step, double maximum_norm) noexcept {
  const double norm = step.norm();
  if (norm <= maximum_norm) {
    return false;
  }
  step *= maximum_norm / norm;
  return true;
}

}  // namespace

DirectPointToPointRegistration::DirectPointToPointRegistration(
    PointToPointRegistrationOptions options)
    : options_(options) {
  if (!optionsAreValid(options_)) {
    throw std::invalid_argument(
        "PointToPointRegistrationOptions require positive finite iteration, source, "
        "correspondence, distance, kernel, convergence, step, and rank bounds");
  }
}

PointToPointRegistrationResult DirectPointToPointRegistration::align(
    std::span<const Point3d> source_points, const BoundedVoxelTarget& target,
    const Sophus::SE3d& initial_T_target_source) const {
  PointToPointRegistrationResult result;
  result.T_target_source = initial_T_target_source;
  result.quality.input_source_points = source_points.size();

  if (source_points.size() > options_.max_source_points) {
    result.status = RegistrationStatus::kSourcePointLimitExceeded;
    return result;
  }
  PointCloud finite_source;
  finite_source.reserve(source_points.size());
  for (const Point3d& point : source_points) {
    if (point.allFinite()) {
      finite_source.push_back(point);
    }
  }
  result.quality.finite_source_points = finite_source.size();
  if (finite_source.empty()) {
    result.status = RegistrationStatus::kEmptySource;
    return result;
  }
  if (target.empty()) {
    result.status = RegistrationStatus::kEmptyTarget;
    return result;
  }
  if (!target.supportsQueryDistance(options_.max_correspondence_distance_m)) {
    result.status = RegistrationStatus::kIncompatibleTarget;
    return result;
  }
  if (!initial_T_target_source.matrix().allFinite()) {
    result.status = RegistrationStatus::kInvalidInitialGuess;
    return result;
  }

  bool converged = false;
  for (std::size_t iteration = 0U; iteration < options_.max_iterations; ++iteration) {
    const Linearization current =
        linearize(finite_source, target, result.T_target_source, options_);
    result.quality.iterations = iteration + 1U;
    if (current.correspondences < options_.minimum_correspondences) {
      populateFinalLinearization(current, finite_source.size(), options_.relative_rank_tolerance,
                                 result);
      result.status = RegistrationStatus::kInsufficientCorrespondences;
      return result;
    }

    const auto analysis = analyzeHessian(current.hessian, options_.relative_rank_tolerance);
    if (!analysis.has_value()) {
      populateFinalLinearization(current, finite_source.size(), options_.relative_rank_tolerance,
                                 result);
      result.status = RegistrationStatus::kNumericalFailure;
      return result;
    }
    if (analysis->rank != 6U || !analysis->inverse.has_value()) {
      populateFinalLinearization(current, finite_source.size(), options_.relative_rank_tolerance,
                                 result);
      result.status = RegistrationStatus::kDegenerateGeometry;
      return result;
    }

    Eigen::LDLT<Matrix6d> decomposition(current.hessian);
    Vector6d step = decomposition.solve(-current.gradient);
    if (decomposition.info() != Eigen::Success || !step.allFinite()) {
      populateFinalLinearization(current, finite_source.size(), options_.relative_rank_tolerance,
                                 result);
      result.status = RegistrationStatus::kNumericalFailure;
      return result;
    }

    bool step_was_limited = false;
    step_was_limited |= limitStep(step.template head<3>(), options_.max_translation_step_m);
    step_was_limited |= limitStep(step.template tail<3>(), options_.max_rotation_step_rad);
    result.quality.step_was_limited = result.quality.step_was_limited || step_was_limited;
    result.quality.final_translation_step_m = step.template head<3>().norm();
    result.quality.final_rotation_step_rad = step.template tail<3>().norm();

    const Sophus::SE3d candidate = Sophus::SE3d::exp(step) * result.T_target_source;
    if (!candidate.matrix().allFinite()) {
      populateFinalLinearization(current, finite_source.size(), options_.relative_rank_tolerance,
                                 result);
      result.status = RegistrationStatus::kNumericalFailure;
      return result;
    }
    result.T_target_source = candidate;
    if (result.quality.final_translation_step_m <= options_.translation_convergence_m &&
        result.quality.final_rotation_step_rad <= options_.rotation_convergence_rad) {
      converged = true;
      break;
    }
  }

  const Linearization final = linearize(finite_source, target, result.T_target_source, options_);
  populateFinalLinearization(final, finite_source.size(), options_.relative_rank_tolerance, result);
  if (final.correspondences < options_.minimum_correspondences) {
    result.status = RegistrationStatus::kInsufficientCorrespondences;
    return result;
  }
  if (result.quality.observable_rank != 6U || !result.covariance.has_value()) {
    result.status = RegistrationStatus::kDegenerateGeometry;
    return result;
  }
  result.status = converged ? RegistrationStatus::kConverged : RegistrationStatus::kIterationLimit;
  return result;
}

}  // namespace meridian::local_rt::lidar
