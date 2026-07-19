#include "meridian/local/visual_factor.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "visual_gtsam_factor.hpp"

namespace meridian::local {
namespace {

using Matrix26 = Eigen::Matrix<double, 2, 6>;
using Matrix36 = Eigen::Matrix<double, 3, 6>;

[[nodiscard]] VisualReprojectionError reprojectionError(VisualReprojectionErrorCode code,
                                                        std::string detail) {
  return VisualReprojectionError{code, std::move(detail)};
}

[[nodiscard]] VisualFactorBuilderError builderError(VisualFactorBuilderErrorCode code,
                                                    std::string detail) {
  return VisualFactorBuilderError{code, std::move(detail)};
}

[[nodiscard]] bool positiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool validCovariance(const Eigen::Matrix2d& covariance) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-10 * scale) {
    return false;
  }
  return Eigen::LLT<Eigen::Matrix2d>(covariance).info() == Eigen::Success;
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

[[nodiscard]] Matrix26 toMeridianPoseJacobian(const Matrix26& gtsam_order_jacobian) {
  Matrix26 result;
  result.leftCols<3>() = gtsam_order_jacobian.rightCols<3>();
  result.rightCols<3>() = gtsam_order_jacobian.leftCols<3>();
  return result;
}

[[nodiscard]] Matrix26 toGtsamPoseJacobian(const Matrix26& meridian_order_jacobian) {
  Matrix26 result;
  result.leftCols<3>() = meridian_order_jacobian.rightCols<3>();
  result.rightCols<3>() = meridian_order_jacobian.leftCols<3>();
  return result;
}

[[nodiscard]] core::Result<bool, VisualReprojectionError> validateObservation(
    const VisualObservationRef& observation, const char* label) {
  using Result = core::Result<bool, VisualReprojectionError>;
  if (!observation.frame.valid() || !observation.state.valid() || !observation.camera.valid() ||
      !observation.calibration.valid()) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidIdentity,
                          std::string(label) + " observation has an invalid identity"));
  }
  const EquidistantCamera camera(observation.camera_model);
  if (!camera.valid()) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::InvalidCameraModel,
                                             std::string(label) + " camera model is invalid"));
  }
  if (!observation.pixel.allFinite() || !camera.isInsideImage(observation.pixel)) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidPixel,
                          std::string(label) + " pixel is non-finite or outside the image"));
  }
  if (!validCovariance(observation.pixel_covariance)) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidCovariance,
                          std::string(label) + " pixel covariance is not finite SPD"));
  }
  if (!observation.imu_from_camera.T_imu_camera().matrix().allFinite()) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::InvalidExtrinsic,
                                             std::string(label) + " T_imu_camera is non-finite"));
  }
  return Result::success(true);
}

[[nodiscard]] core::Result<bool, VisualReprojectionError> validateSpec(
    const VisualReprojectionFactorSpec& spec) {
  using Result = core::Result<bool, VisualReprojectionError>;
  if (!spec.id.valid() || !spec.landmark.valid() || !spec.track.valid()) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidIdentity,
                          "visual factor, landmark, and track identities must be valid"));
  }
  const auto anchor = validateObservation(spec.anchor, "anchor");
  if (!anchor) {
    return Result::failure(anchor.error());
  }
  const auto observer = validateObservation(spec.observer, "observer");
  if (!observer) {
    return Result::failure(observer.error());
  }
  if (spec.anchor.state == spec.observer.state ||
      spec.anchor.exact_time >= spec.observer.exact_time) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidTimeOrder,
                          "visual observer must be a later, distinct navigation state"));
  }
  if (!positiveFinite(spec.minimum_range_m) || !positiveFinite(spec.maximum_range_m) ||
      spec.minimum_range_m >= spec.maximum_range_m || !positiveFinite(spec.huber_delta_sigma)) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::InvalidRangeBounds,
                          "visual range bounds and Huber width must be finite and ordered"));
  }
  if (!spec.lineage.id.valid() ||
      core::validateLineage(spec.lineage) != core::LineageValidationError::None) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::InvalidLineage,
                                             "visual factor lineage is invalid"));
  }
  return Result::success(true);
}

[[nodiscard]] bool validBuilderConfig(const VisualFactorBuilderConfig& config) {
  return config.minimum_non_anchor_observations >= 1U &&
         config.maximum_pending_observations >= config.minimum_non_anchor_observations + 1U &&
         config.maximum_observations_per_track >= config.maximum_pending_observations &&
         config.maximum_active_tracks > 0U && config.maximum_missed_keyframes > 0U &&
         positiveFinite(config.minimum_baseline_m) && positiveFinite(config.minimum_parallax_rad) &&
         config.minimum_parallax_rad < 3.14159265358979323846 &&
         positiveFinite(config.minimum_range_m) && positiveFinite(config.maximum_range_m) &&
         config.minimum_range_m < config.maximum_range_m &&
         positiveFinite(config.triangulation_huber_angle_rad) &&
         config.triangulation_iterations > 0U &&
         positiveFinite(config.maximum_triangulation_condition) &&
         positiveFinite(config.maximum_inlier_reprojection_error_px) &&
         positiveFinite(config.maximum_reprojection_rmse_px) &&
         positiveFinite(config.huber_delta_sigma) &&
         positiveFinite(config.outlier_chi_squared_gate) &&
         config.outlier_commits_before_retirement > 0U;
}

[[nodiscard]] bool validBuilderFeature(const VisualFeatureObservation& feature) {
  return feature.track.valid() && feature.distorted_pixel.allFinite() &&
         feature.unit_bearing.allFinite() && std::abs(feature.unit_bearing.norm() - 1.0) < 1.0e-3 &&
         validCovariance(feature.pixel_covariance);
}

template <typename Id>
[[nodiscard]] bool canonicalFinalityIds(const std::vector<Id>& ids) {
  for (std::size_t index = 0U; index < ids.size(); ++index) {
    if (!ids[index].valid() || (index > 0U && ids[index] <= ids[index - 1U])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double clampedAngle(const Eigen::Vector3d& lhs, const Eigen::Vector3d& rhs) {
  return std::acos(std::clamp(lhs.dot(rhs), -1.0, 1.0));
}

}  // namespace

core::Result<VisualReprojectionEvaluation, VisualReprojectionError> evaluateVisualReprojection(
    const VisualReprojectionFactorSpec& spec, const core::Pose3d& T_odom_imu_anchor,
    const core::Pose3d& T_odom_imu_observer, double eta) {
  using Result = core::Result<VisualReprojectionEvaluation, VisualReprojectionError>;
  const auto validation = validateSpec(spec);
  if (!validation) {
    return Result::failure(validation.error());
  }
  if (!T_odom_imu_anchor.matrix().allFinite() || !T_odom_imu_observer.matrix().allFinite()) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::NonFinitePose,
                                             "visual anchor and observer poses must be finite"));
  }
  if (!std::isfinite(eta)) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::NonFiniteEta,
                                             "visual log inverse range is non-finite"));
  }

  const double range = std::exp(-eta);
  if (!std::isfinite(range) || range < spec.minimum_range_m || range > spec.maximum_range_m) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::RangeOutsideBounds,
                          "visual anchor range exp(-eta) is outside configured bounds"));
  }

  const EquidistantCamera anchor_camera(spec.anchor.camera_model);
  const EquidistantCamera observer_camera(spec.observer.camera_model);
  const auto anchor_ray = anchor_camera.unproject(spec.anchor.pixel);
  if (!anchor_ray) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::AnchorUnprojectionFailure,
                                             anchor_ray.error().detail));
  }

  const core::Pose3d& T_imu_camera_anchor = spec.anchor.imu_from_camera.T_imu_camera();
  const core::Pose3d& T_imu_camera_observer = spec.observer.imu_from_camera.T_imu_camera();
  const Eigen::Vector3d point_camera_anchor = range * anchor_ray.value().unit_ray;
  const Eigen::Vector3d point_imu_anchor = T_imu_camera_anchor * point_camera_anchor;
  const Eigen::Vector3d point_odom = T_odom_imu_anchor * point_imu_anchor;
  const Eigen::Vector3d point_imu_observer = T_odom_imu_observer.inverse() * point_odom;
  const Eigen::Vector3d point_camera_observer =
      T_imu_camera_observer.inverse() * point_imu_observer;
  if (!point_camera_observer.allFinite()) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::NumericalFailure,
                          "visual frame chain produced a non-finite observer point"));
  }
  if (point_camera_observer.z() <= 1.0e-9) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::ObserverCheirality,
                                             "visual landmark lies behind the observer camera"));
  }

  const auto projection = observer_camera.project(point_camera_observer);
  if (!projection) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::ObserverProjectionFailure,
                                             projection.error().detail));
  }
  if (projection.value().domain != ProjectionDomain::InsideImage) {
    return Result::failure(
        reprojectionError(VisualReprojectionErrorCode::ObserverOutsideImage,
                          "visual landmark projection is outside the observer image"));
  }

  const Eigen::Matrix3d R_odom_imu_anchor = T_odom_imu_anchor.so3().matrix();
  const Eigen::Matrix3d R_odom_imu_observer = T_odom_imu_observer.so3().matrix();
  const Eigen::Matrix3d R_imu_camera_anchor = T_imu_camera_anchor.so3().matrix();
  const Eigen::Matrix3d R_imu_camera_observer = T_imu_camera_observer.so3().matrix();

  // GTSAM Pose3 retract uses a right [rotation, translation] tangent. For the
  // anchor transformFrom derivative this is [-R[p]x, R]. For the observer
  // inverse transform it is [[p]x, -I]. The public matrices are explicitly
  // permuted below to Meridian [translation, rotation].
  Matrix36 point_from_anchor_gtsam;
  point_from_anchor_gtsam.leftCols<3>() = R_imu_camera_observer.transpose() *
                                          R_odom_imu_observer.transpose() *
                                          (-R_odom_imu_anchor * skew(point_imu_anchor));
  point_from_anchor_gtsam.rightCols<3>() =
      R_imu_camera_observer.transpose() * R_odom_imu_observer.transpose() * R_odom_imu_anchor;

  Matrix36 point_from_observer_gtsam;
  point_from_observer_gtsam.leftCols<3>() =
      R_imu_camera_observer.transpose() * skew(point_imu_observer);
  point_from_observer_gtsam.rightCols<3>() = -R_imu_camera_observer.transpose();

  const Eigen::Vector3d point_from_eta = R_imu_camera_observer.transpose() *
                                         R_odom_imu_observer.transpose() * R_odom_imu_anchor *
                                         R_imu_camera_anchor * (-point_camera_anchor);
  const Matrix26 anchor_gtsam = -projection.value().point_jacobian * point_from_anchor_gtsam;
  const Matrix26 observer_gtsam = -projection.value().point_jacobian * point_from_observer_gtsam;

  VisualReprojectionEvaluation result;
  result.predicted_pixel = projection.value().pixel;
  result.residual = spec.observer.pixel - result.predicted_pixel;
  result.anchor_pose_jacobian = toMeridianPoseJacobian(anchor_gtsam);
  result.observer_pose_jacobian = toMeridianPoseJacobian(observer_gtsam);
  result.eta_jacobian = -projection.value().point_jacobian * point_from_eta;
  result.anchor_range_m = range;
  result.observer_depth_m = point_camera_observer.z();
  const Eigen::LLT<Eigen::Matrix2d> covariance_cholesky(spec.observer.pixel_covariance);
  result.squared_mahalanobis = result.residual.dot(covariance_cholesky.solve(result.residual));
  if (!result.predicted_pixel.allFinite() || !result.residual.allFinite() ||
      !result.anchor_pose_jacobian.allFinite() || !result.observer_pose_jacobian.allFinite() ||
      !result.eta_jacobian.allFinite() || !std::isfinite(result.squared_mahalanobis)) {
    return Result::failure(reprojectionError(VisualReprojectionErrorCode::NumericalFailure,
                                             "visual reprojection produced a non-finite result"));
  }
  return Result::success(std::move(result));
}

namespace {

struct StoredVisualObservation {
  VisualObservationRef ref;
  core::Pose3d T_odom_imu;
  Eigen::Vector3d unit_bearing{Eigen::Vector3d::UnitZ()};
  core::ObservationLineage frontend_lineage;
};

struct StoredVisualTrack {
  VisualTrackId id;
  std::optional<VisualLandmarkId> landmark;
  std::vector<StoredVisualObservation> observations;
  double initial_eta{};
  std::size_t missed_keyframes{};
};

struct FactorHealth {
  VisualTrackId track;
  std::size_t consecutive_outliers{};
};

struct TriangulationResult {
  VisualTriangulationDecision decision;
  Eigen::Vector3d point_odom{Eigen::Vector3d::Zero()};
  double eta{};
  std::size_t anchor_index{};
  std::vector<bool> inliers;
};

[[nodiscard]] core::Pose3d cameraPose(const StoredVisualObservation& observation) {
  return observation.T_odom_imu * observation.ref.imu_from_camera.T_imu_camera();
}

[[nodiscard]] TriangulationResult triangulateTrack(const StoredVisualTrack& track,
                                                   const VisualFactorBuilderConfig& config) {
  TriangulationResult result;
  result.decision.track = track.id;
  result.decision.observations = track.observations.size();
  result.inliers.assign(track.observations.size(), false);
  const std::size_t required = config.minimum_non_anchor_observations + 1U;
  if (track.observations.size() < required) {
    result.decision.status = VisualTriangulationStatus::InsufficientObservations;
    return result;
  }

  std::vector<Eigen::Vector3d> centers;
  std::vector<Eigen::Vector3d> rays;
  centers.reserve(track.observations.size());
  rays.reserve(track.observations.size());
  for (const StoredVisualObservation& observation : track.observations) {
    const core::Pose3d T_odom_camera = cameraPose(observation);
    centers.push_back(T_odom_camera.translation());
    rays.push_back(T_odom_camera.so3() * observation.unit_bearing);
  }
  for (std::size_t first = 0U; first < centers.size(); ++first) {
    for (std::size_t second = first + 1U; second < centers.size(); ++second) {
      result.decision.maximum_baseline_m =
          std::max(result.decision.maximum_baseline_m, (centers[second] - centers[first]).norm());
      result.decision.maximum_parallax_rad =
          std::max(result.decision.maximum_parallax_rad, clampedAngle(rays[second], rays[first]));
    }
  }
  if (result.decision.maximum_baseline_m < config.minimum_baseline_m) {
    result.decision.status = VisualTriangulationStatus::InsufficientBaseline;
    return result;
  }
  if (result.decision.maximum_parallax_rad < config.minimum_parallax_rad) {
    result.decision.status = VisualTriangulationStatus::InsufficientParallax;
    return result;
  }

  std::vector<double> weights(track.observations.size(), 1.0);
  Eigen::Vector3d point = Eigen::Vector3d::Zero();

  // An unweighted all-ray solve is not a robust initializer: one mistracked
  // bearing can move the first estimate far enough that Huber IRLS assigns
  // comparable weights to good and bad observations. Seed IRLS from the
  // deterministic two-ray hypothesis with the largest exact reprojection
  // consensus. Consensus observations then participate in the subsequent
  // Huber refinement; grossly inconsistent hypotheses stay trimmed.
  std::size_t best_hypothesis_inliers = 0U;
  double best_hypothesis_score = std::numeric_limits<double>::infinity();
  std::optional<Eigen::Vector3d> best_hypothesis;
  std::vector<bool> best_hypothesis_consensus(track.observations.size(), true);
  const double truncated_error_squared =
      config.maximum_inlier_reprojection_error_px * config.maximum_inlier_reprojection_error_px;
  for (std::size_t first = 0U; first < centers.size(); ++first) {
    for (std::size_t second = first + 1U; second < centers.size(); ++second) {
      Eigen::Matrix3d pair_normal = Eigen::Matrix3d::Zero();
      Eigen::Vector3d pair_right_hand_side = Eigen::Vector3d::Zero();
      for (const std::size_t index : {first, second}) {
        const Eigen::Matrix3d projector =
            Eigen::Matrix3d::Identity() - rays[index] * rays[index].transpose();
        pair_normal.noalias() += projector;
        pair_right_hand_side.noalias() += projector * centers[index];
      }
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> pair_eigen_solver(pair_normal);
      if (pair_eigen_solver.info() != Eigen::Success ||
          !pair_eigen_solver.eigenvalues().allFinite() ||
          pair_eigen_solver.eigenvalues().minCoeff() <= 1.0e-12 ||
          pair_eigen_solver.eigenvalues().maxCoeff() / pair_eigen_solver.eigenvalues().minCoeff() >
              config.maximum_triangulation_condition) {
        continue;
      }
      const Eigen::Vector3d hypothesis = pair_normal.ldlt().solve(pair_right_hand_side);
      if (!hypothesis.allFinite()) {
        continue;
      }

      std::size_t inliers = 0U;
      double score = 0.0;
      std::vector<bool> consensus(track.observations.size(), false);
      for (std::size_t index = 0U; index < track.observations.size(); ++index) {
        const StoredVisualObservation& observation = track.observations[index];
        const Eigen::Vector3d point_camera = cameraPose(observation).inverse() * hypothesis;
        double squared_error = truncated_error_squared;
        if (point_camera.allFinite() && point_camera.z() > 1.0e-9) {
          const EquidistantCamera camera(observation.ref.camera_model);
          const auto projection = camera.project(point_camera);
          if (projection && projection.value().domain == ProjectionDomain::InsideImage) {
            const double error = (projection.value().pixel - observation.ref.pixel).norm();
            if (std::isfinite(error) && error <= config.maximum_inlier_reprojection_error_px) {
              ++inliers;
              consensus[index] = true;
            }
            if (std::isfinite(error)) {
              squared_error = std::min(error * error, truncated_error_squared);
            }
          }
        }
        score += squared_error;
      }
      if (inliers > best_hypothesis_inliers ||
          (inliers == best_hypothesis_inliers && score < best_hypothesis_score)) {
        best_hypothesis_inliers = inliers;
        best_hypothesis_score = score;
        best_hypothesis = hypothesis;
        best_hypothesis_consensus = std::move(consensus);
      }
    }
  }
  if (best_hypothesis) {
    point = *best_hypothesis;
    for (std::size_t index = 0U; index < centers.size(); ++index) {
      if (!best_hypothesis_consensus[index]) {
        weights[index] = 0.0;
        continue;
      }
      const Eigen::Vector3d delta = point - centers[index];
      if (delta.norm() <= 1.0e-12) {
        weights[index] = 0.0;
        continue;
      }
      const double angle = clampedAngle(delta.normalized(), rays[index]);
      weights[index] = angle <= config.triangulation_huber_angle_rad
                           ? 1.0
                           : config.triangulation_huber_angle_rad / angle;
    }
  }
  for (std::size_t iteration = 0U; iteration < config.triangulation_iterations; ++iteration) {
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
    for (std::size_t index = 0U; index < centers.size(); ++index) {
      const Eigen::Matrix3d projector =
          Eigen::Matrix3d::Identity() - rays[index] * rays[index].transpose();
      normal.noalias() += weights[index] * projector;
      right_hand_side.noalias() += weights[index] * projector * centers[index];
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(normal);
    if (eigen_solver.info() != Eigen::Success || !eigen_solver.eigenvalues().allFinite() ||
        eigen_solver.eigenvalues().minCoeff() <= 1.0e-12) {
      result.decision.status = VisualTriangulationStatus::IllConditioned;
      return result;
    }
    result.decision.condition_number =
        eigen_solver.eigenvalues().maxCoeff() / eigen_solver.eigenvalues().minCoeff();
    if (result.decision.condition_number > config.maximum_triangulation_condition) {
      result.decision.status = VisualTriangulationStatus::IllConditioned;
      return result;
    }
    point = normal.ldlt().solve(right_hand_side);
    if (!point.allFinite()) {
      result.decision.status = VisualTriangulationStatus::NumericalFailure;
      return result;
    }
    for (std::size_t index = 0U; index < centers.size(); ++index) {
      if (best_hypothesis && !best_hypothesis_consensus[index]) {
        weights[index] = 0.0;
        continue;
      }
      const Eigen::Vector3d delta = point - centers[index];
      if (delta.norm() <= 1.0e-12) {
        weights[index] = 0.0;
        continue;
      }
      const double angle = clampedAngle(delta.normalized(), rays[index]);
      weights[index] = angle <= config.triangulation_huber_angle_rad
                           ? 1.0
                           : config.triangulation_huber_angle_rad / angle;
    }
  }

  double squared_error_sum = 0.0;
  for (std::size_t index = 0U; index < track.observations.size(); ++index) {
    const StoredVisualObservation& observation = track.observations[index];
    const core::Pose3d T_camera_odom = cameraPose(observation).inverse();
    const Eigen::Vector3d point_camera = T_camera_odom * point;
    if (!point_camera.allFinite() || point_camera.z() <= 1.0e-9) {
      if (index == 0U) {
        result.decision.status = VisualTriangulationStatus::BehindCamera;
        return result;
      }
      continue;
    }
    const double range = point_camera.norm();
    if (range < config.minimum_range_m || range > config.maximum_range_m) {
      if (index == 0U) {
        result.decision.status = VisualTriangulationStatus::RangeOutsideBounds;
        return result;
      }
      continue;
    }
    const EquidistantCamera camera(observation.ref.camera_model);
    const auto projection = camera.project(point_camera);
    if (!projection || projection.value().domain != ProjectionDomain::InsideImage) {
      continue;
    }
    const double error = (projection.value().pixel - observation.ref.pixel).norm();
    if (error > config.maximum_inlier_reprojection_error_px) {
      continue;
    }
    result.inliers[index] = true;
    ++result.decision.inliers;
    squared_error_sum += error * error;
  }
  if (result.decision.inliers < config.minimum_non_anchor_observations + 1U) {
    result.decision.status = VisualTriangulationStatus::TooFewInliers;
    return result;
  }
  result.decision.reprojection_rmse_px =
      std::sqrt(squared_error_sum / static_cast<double>(result.decision.inliers));
  if (result.decision.reprojection_rmse_px > config.maximum_reprojection_rmse_px) {
    result.decision.status = VisualTriangulationStatus::ReprojectionRmseTooLarge;
    return result;
  }
  const auto anchor = std::find(result.inliers.begin(), result.inliers.end(), true);
  if (anchor == result.inliers.end()) {
    result.decision.status = VisualTriangulationStatus::TooFewInliers;
    return result;
  }
  result.anchor_index = static_cast<std::size_t>(std::distance(result.inliers.begin(), anchor));
  const double anchor_range = (point - centers[result.anchor_index]).norm();
  if (!positiveFinite(anchor_range)) {
    result.decision.status = VisualTriangulationStatus::NumericalFailure;
    return result;
  }
  result.point_odom = point;
  result.eta = -std::log(anchor_range);
  result.decision.status = VisualTriangulationStatus::Seeded;
  return result;
}

}  // namespace

struct VisualFactorBatchBuilder::Implementation {
  explicit Implementation(VisualFactorBuilderConfig config_in) : config(std::move(config_in)) {}

  [[nodiscard]] core::ObservationLineage makeFactorLineage(const StoredVisualObservation& anchor,
                                                           const StoredVisualObservation& observer,
                                                           core::FactorId factor_id) {
    const core::DerivedRecordId consumer(next_derived_record_id++);
    core::ObservationLineage lineage;
    lineage.id = core::ObservationLineageId(next_lineage_id++);

    core::ObservationUsage anchor_usage;
    anchor_usage.slice.root = anchor.ref.frame;
    anchor_usage.slice.kind = core::SliceKind::Whole;
    anchor_usage.slice.calibration = anchor.ref.calibration;
    anchor_usage.role = core::ObservationRole::ConditioningOnly;
    anchor_usage.consumer = consumer;
    lineage.usage.push_back(anchor_usage);

    core::ObservationUsage observer_usage;
    observer_usage.slice.root = observer.ref.frame;
    observer_usage.slice.kind = core::SliceKind::Whole;
    observer_usage.slice.calibration = observer.ref.calibration;
    observer_usage.role = core::ObservationRole::PrimaryResidual;
    observer_usage.consumer = consumer;
    observer_usage.factor_group = core::FactorGroupId(factor_id.value());
    lineage.usage.push_back(observer_usage);

    const auto append_conditioning = [&lineage, consumer](const core::ObservationLineage& source,
                                                          core::MeasurementId frame) {
      for (const core::ObservationUsage& usage : source.usage) {
        if (std::holds_alternative<core::MeasurementId>(usage.slice.root) &&
            std::get<core::MeasurementId>(usage.slice.root) == frame) {
          continue;
        }
        core::ObservationUsage copy = usage;
        copy.role = core::ObservationRole::ConditioningOnly;
        copy.consumer = consumer;
        copy.factor_group.reset();
        copy.correlation_group.reset();
        lineage.usage.push_back(std::move(copy));
      }
    };
    append_conditioning(anchor.frontend_lineage, anchor.ref.frame);
    append_conditioning(observer.frontend_lineage, observer.ref.frame);
    return lineage;
  }

  [[nodiscard]] VisualReprojectionFactorSpec makeFactor(const StoredVisualTrack& track,
                                                        const StoredVisualObservation& observer) {
    const core::FactorId factor_id(next_factor_id++);
    VisualReprojectionFactorSpec spec;
    spec.id = factor_id;
    spec.landmark = *track.landmark;
    spec.track = track.id;
    spec.anchor = track.observations.front().ref;
    spec.observer = observer.ref;
    spec.minimum_range_m = config.minimum_range_m;
    spec.maximum_range_m = config.maximum_range_m;
    spec.huber_delta_sigma = config.huber_delta_sigma;
    spec.lineage = makeFactorLineage(track.observations.front(), observer, factor_id);
    return spec;
  }

  void eraseFactorHealthForTrack(VisualTrackId track) {
    for (auto factor = factor_health.begin(); factor != factor_health.end();) {
      if (factor->second.track == track) {
        factor = factor_health.erase(factor);
      } else {
        ++factor;
      }
    }
  }

  VisualFactorBuilderConfig config;
  std::map<VisualTrackId, StoredVisualTrack> tracks;
  std::map<core::FactorId, FactorHealth> factor_health;
  std::optional<core::FusionTime> last_keyframe_time;
  std::optional<core::LocalGraphRevision> last_feedback_revision;
  std::optional<core::LocalGraphRevision> last_finality_revision;
  std::uint64_t next_landmark_id{};
  std::uint64_t next_factor_id{};
  std::uint64_t next_lineage_id{};
  std::uint64_t next_derived_record_id{};
};

VisualFactorBatchBuilder::VisualFactorBatchBuilder(VisualFactorBuilderConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

VisualFactorBatchBuilder::~VisualFactorBatchBuilder() = default;
VisualFactorBatchBuilder::VisualFactorBatchBuilder(VisualFactorBatchBuilder&&) noexcept = default;
VisualFactorBatchBuilder& VisualFactorBatchBuilder::operator=(VisualFactorBatchBuilder&&) noexcept =
    default;

core::Result<VisualFactorBatch, VisualFactorBuilderError> VisualFactorBatchBuilder::processKeyframe(
    const VisualFrontendOutput& frontend, const VisualKeyframeContext& context) {
  using Result = core::Result<VisualFactorBatch, VisualFactorBuilderError>;
  Implementation working = *implementation_;
  if (!validBuilderConfig(working.config)) {
    return Result::failure(builderError(VisualFactorBuilderErrorCode::InvalidConfig,
                                        "visual factor builder configuration is invalid"));
  }
  if (!frontend.keyframe) {
    return Result::failure(
        builderError(VisualFactorBuilderErrorCode::NonKeyframeInput,
                     "visual factors may be built only from accepted keyframes"));
  }
  if (!frontend.source_frame.valid() || !frontend.camera.valid() || !context.state.valid() ||
      !context.camera.valid() || !context.calibration.valid() ||
      context.exact_time != frontend.exposure_midpoint ||
      !context.T_odom_imu.matrix().allFinite() ||
      !context.imu_from_camera.T_imu_camera().matrix().allFinite() ||
      !EquidistantCamera(context.camera_model).valid()) {
    return Result::failure(
        builderError(VisualFactorBuilderErrorCode::InvalidContext,
                     "visual keyframe context is invalid or not at the exact exposure time"));
  }
  if (context.camera != frontend.camera) {
    return Result::failure(
        builderError(VisualFactorBuilderErrorCode::StateOrCameraMismatch,
                     "visual keyframe context camera does not match the frontend output"));
  }
  if (!frontend.lineage.id.valid() ||
      core::validateLineage(frontend.lineage) != core::LineageValidationError::None) {
    return Result::failure(builderError(VisualFactorBuilderErrorCode::InvalidLineage,
                                        "visual frontend keyframe lineage is invalid"));
  }
  const bool source_frame_in_lineage = std::any_of(
      frontend.lineage.usage.begin(), frontend.lineage.usage.end(),
      [&frontend, &context](const core::ObservationUsage& usage) {
        return std::holds_alternative<core::MeasurementId>(usage.slice.root) &&
               std::get<core::MeasurementId>(usage.slice.root) == frontend.source_frame &&
               usage.slice.calibration == context.calibration;
      });
  if (!source_frame_in_lineage) {
    return Result::failure(
        builderError(VisualFactorBuilderErrorCode::InvalidLineage,
                     "visual frontend lineage does not contain the calibrated source frame"));
  }
  if (working.last_keyframe_time && context.exact_time <= *working.last_keyframe_time) {
    return Result::failure(builderError(VisualFactorBuilderErrorCode::NonMonotonicKeyframe,
                                        "visual keyframe times must be strictly increasing"));
  }

  std::vector<const VisualFeatureObservation*> ordered_features;
  ordered_features.reserve(frontend.features.size());
  for (const VisualFeatureObservation& feature : frontend.features) {
    ordered_features.push_back(&feature);
  }
  std::sort(ordered_features.begin(), ordered_features.end(),
            [](const VisualFeatureObservation* lhs, const VisualFeatureObservation* rhs) {
              return lhs->track < rhs->track;
            });
  for (std::size_t index = 0U; index < ordered_features.size(); ++index) {
    if (!validBuilderFeature(*ordered_features[index])) {
      return Result::failure(
          builderError(VisualFactorBuilderErrorCode::InvalidFeature,
                       "visual keyframe contains an invalid feature observation"));
    }
    if (index > 0U && ordered_features[index - 1U]->track == ordered_features[index]->track) {
      return Result::failure(builderError(VisualFactorBuilderErrorCode::DuplicateTrack,
                                          "visual keyframe contains a duplicate track identity"));
    }
  }

  VisualFactorBatch batch;
  batch.exact_time = context.exact_time;
  batch.report.input_features = ordered_features.size();
  std::set<VisualTrackId> seen;
  const EquidistantCamera context_camera(context.camera_model);

  for (const VisualFeatureObservation* feature : ordered_features) {
    seen.insert(feature->track);
    const auto recomputed_ray = context_camera.unproject(feature->distorted_pixel);
    if (!recomputed_ray ||
        (recomputed_ray.value().unit_ray - feature->unit_bearing).norm() > 1.0e-3) {
      return Result::failure(
          builderError(VisualFactorBuilderErrorCode::InvalidFeature,
                       "visual feature bearing does not match its calibrated pixel"));
    }

    StoredVisualObservation observation;
    observation.ref.frame = frontend.source_frame;
    observation.ref.state = context.state;
    observation.ref.exact_time = context.exact_time;
    observation.ref.camera = context.camera;
    observation.ref.calibration = context.calibration;
    observation.ref.pixel = feature->distorted_pixel;
    observation.ref.pixel_covariance = feature->pixel_covariance;
    observation.ref.camera_model = context.camera_model;
    observation.ref.imu_from_camera = context.imu_from_camera;
    observation.T_odom_imu = context.T_odom_imu;
    observation.unit_bearing = recomputed_ray.value().unit_ray;
    observation.frontend_lineage = frontend.lineage;

    auto match = working.tracks.find(feature->track);
    if (match == working.tracks.end()) {
      if (working.tracks.size() >= working.config.maximum_active_tracks) {
        ++batch.report.tracks_rejected_capacity;
        continue;
      }
      StoredVisualTrack track;
      track.id = feature->track;
      track.observations.push_back(std::move(observation));
      working.tracks.emplace(track.id, std::move(track));
      ++batch.report.tracks_created;
      continue;
    }

    StoredVisualTrack& track = match->second;
    track.missed_keyframes = 0U;
    if (track.observations.size() >= working.config.maximum_observations_per_track) {
      batch.retired_tracks.push_back({track.id, track.landmark.value_or(VisualLandmarkId{}),
                                      VisualTrackRetirementReason::MaximumLength});
      working.eraseFactorHealthForTrack(track.id);
      working.tracks.erase(match);
      continue;
    }

    if (track.landmark) {
      VisualReprojectionFactorSpec spec = working.makeFactor(track, observation);
      const auto admission = evaluateVisualReprojection(spec, track.observations.front().T_odom_imu,
                                                        observation.T_odom_imu, track.initial_eta);
      if (!admission || admission.value().residual.norm() >
                            2.0 * working.config.maximum_inlier_reprojection_error_px) {
        ++batch.report.observations_rejected;
        continue;
      }
      working.factor_health.emplace(spec.id, FactorHealth{track.id, 0U});
      batch.factors.push_back(std::move(spec));
      track.observations.push_back(std::move(observation));
      continue;
    }

    track.observations.push_back(std::move(observation));
    if (track.observations.size() > working.config.maximum_pending_observations) {
      track.observations.erase(track.observations.begin() + 1);
    }
    const TriangulationResult triangulation = triangulateTrack(track, working.config);
    batch.report.triangulation.push_back(triangulation.decision);
    if (triangulation.decision.status != VisualTriangulationStatus::Seeded) {
      continue;
    }

    track.landmark = VisualLandmarkId(working.next_landmark_id++);
    track.initial_eta = triangulation.eta;
    std::vector<StoredVisualObservation> inlier_observations;
    inlier_observations.reserve(triangulation.decision.inliers);
    inlier_observations.push_back(track.observations[triangulation.anchor_index]);
    for (std::size_t index = 0U; index < track.observations.size(); ++index) {
      if (index != triangulation.anchor_index && triangulation.inliers[index]) {
        inlier_observations.push_back(track.observations[index]);
      }
    }
    track.observations = std::move(inlier_observations);

    VisualLandmarkSeed seed;
    seed.landmark = *track.landmark;
    seed.track = track.id;
    seed.anchor = track.observations.front().ref;
    seed.eta = track.initial_eta;
    seed.initial_range_m = std::exp(-track.initial_eta);
    seed.triangulation = triangulation.decision;
    batch.new_landmarks.push_back(seed);

    for (std::size_t index = 1U; index < track.observations.size(); ++index) {
      VisualReprojectionFactorSpec spec = working.makeFactor(track, track.observations[index]);
      working.factor_health.emplace(spec.id, FactorHealth{track.id, 0U});
      batch.factors.push_back(std::move(spec));
    }
    ++batch.report.tracks_initialized;
  }

  std::vector<VisualTrackId> lost;
  for (auto& [track_id, track] : working.tracks) {
    if (seen.contains(track_id)) {
      continue;
    }
    ++track.missed_keyframes;
    if (track.missed_keyframes > working.config.maximum_missed_keyframes) {
      lost.push_back(track_id);
    }
  }
  for (const VisualTrackId track_id : lost) {
    const StoredVisualTrack& track = working.tracks.at(track_id);
    batch.retired_tracks.push_back(
        {track.id, track.landmark.value_or(VisualLandmarkId{}), VisualTrackRetirementReason::Lost});
    working.eraseFactorHealthForTrack(track_id);
    working.tracks.erase(track_id);
  }

  batch.report.factors_emitted = batch.factors.size();
  batch.report.active_tracks = working.tracks.size();
  batch.report.tracks_pending = static_cast<std::size_t>(
      std::count_if(working.tracks.begin(), working.tracks.end(),
                    [](const auto& item) { return !item.second.landmark.has_value(); }));
  working.last_keyframe_time = context.exact_time;
  *implementation_ = std::move(working);
  return Result::success(std::move(batch));
}

core::Result<VisualResidualFeedbackReport, VisualFactorBuilderError>
VisualFactorBatchBuilder::applyAcceptedResiduals(
    core::LocalGraphRevision revision, const std::vector<VisualResidualFeedback>& feedback) {
  using Result = core::Result<VisualResidualFeedbackReport, VisualFactorBuilderError>;
  Implementation working = *implementation_;
  if (!revision.valid() ||
      (working.last_feedback_revision && revision <= *working.last_feedback_revision)) {
    return Result::failure(
        builderError(VisualFactorBuilderErrorCode::InvalidFeedbackRevision,
                     "visual residual feedback revision must be valid and increasing"));
  }
  std::set<core::FactorId> seen;
  for (const VisualResidualFeedback& item : feedback) {
    if (!item.factor.valid() || !std::isfinite(item.squared_mahalanobis) ||
        item.squared_mahalanobis < 0.0) {
      return Result::failure(
          builderError(VisualFactorBuilderErrorCode::InvalidFeature,
                       "visual residual feedback must be finite, non-negative, and identified"));
    }
    if (!seen.insert(item.factor).second) {
      return Result::failure(builderError(VisualFactorBuilderErrorCode::DuplicateFeedback,
                                          "visual residual feedback contains a duplicate factor"));
    }
  }

  VisualResidualFeedbackReport report;
  report.revision = revision;
  std::vector<core::FactorId> retired_factors;
  for (const VisualResidualFeedback& item : feedback) {
    auto factor = working.factor_health.find(item.factor);
    if (factor == working.factor_health.end()) {
      ++report.unknown_or_inactive;
      continue;
    }
    ++report.evaluated;
    if (item.squared_mahalanobis > working.config.outlier_chi_squared_gate) {
      ++factor->second.consecutive_outliers;
    } else {
      factor->second.consecutive_outliers = 0U;
    }
    if (factor->second.consecutive_outliers >= working.config.outlier_commits_before_retirement) {
      report.retired_observation_factors.push_back(item.factor);
      retired_factors.push_back(item.factor);
    }
  }
  for (const core::FactorId factor : retired_factors) {
    working.factor_health.erase(factor);
  }
  working.last_feedback_revision = revision;
  *implementation_ = std::move(working);
  return Result::success(std::move(report));
}

VisualUncommittedBatchDiscardReport VisualFactorBatchBuilder::discardUncommittedBatch(
    const VisualFactorBatch& batch) noexcept {
  VisualUncommittedBatchDiscardReport report;
  report.landmark_seeds_discarded = batch.new_landmarks.size();
  report.factors_discarded = batch.factors.size();
  for (const VisualReprojectionFactorSpec& factor : batch.factors) {
    report.factor_health_entries_removed += implementation_->factor_health.erase(factor.id);
  }
  for (const VisualLandmarkSeed& seed : batch.new_landmarks) {
    const auto track = implementation_->tracks.find(seed.track);
    if (track != implementation_->tracks.end() && track->second.landmark == seed.landmark) {
      track->second.landmark.reset();
      if (track->second.observations.size() > 1U) {
        report.stale_track_observations_discarded += track->second.observations.size() - 1U;
        StoredVisualObservation newest = std::move(track->second.observations.back());
        track->second.observations.clear();
        track->second.observations.push_back(std::move(newest));
      }
      ++report.active_landmark_initializations_rolled_back;
    }
  }
  return report;
}

core::Result<VisualFactorFinalityReport, VisualFactorBuilderError>
VisualFactorBatchBuilder::reconcileFinality(const VisualFinalityUpdate& update) {
  using Result = core::Result<VisualFactorFinalityReport, VisualFactorBuilderError>;
  Implementation working = *implementation_;
  if (!update.revision.valid() ||
      (working.last_finality_revision && update.revision <= *working.last_finality_revision) ||
      !canonicalFinalityIds(update.finalized_factors) ||
      !canonicalFinalityIds(update.finalized_landmarks)) {
    return Result::failure(builderError(
        VisualFactorBuilderErrorCode::InvalidFinality,
        "visual finality revision must increase and identities must be valid, sorted, and unique"));
  }

  VisualFactorFinalityReport report;
  report.revision = update.revision;
  report.finalized_factors_observed = update.finalized_factors.size();
  report.finalized_landmarks_observed = update.finalized_landmarks.size();

  // Children first: a factor can disappear independently, while removal of a
  // landmark retires its complete accepted track and any remaining health.
  for (const core::FactorId factor : update.finalized_factors) {
    report.factor_health_entries_pruned += working.factor_health.erase(factor);
  }
  for (const VisualLandmarkId landmark : update.finalized_landmarks) {
    for (auto track = working.tracks.begin(); track != working.tracks.end();) {
      if (track->second.landmark != landmark) {
        ++track;
        continue;
      }
      const std::size_t health_before = working.factor_health.size();
      working.eraseFactorHealthForTrack(track->first);
      report.factor_health_entries_pruned += health_before - working.factor_health.size();
      track = working.tracks.erase(track);
      ++report.accepted_tracks_pruned;
    }
  }

  working.last_finality_revision = update.revision;
  *implementation_ = std::move(working);
  return Result::success(std::move(report));
}

void VisualFactorBatchBuilder::reset() {
  implementation_->tracks.clear();
  implementation_->factor_health.clear();
  implementation_->last_keyframe_time.reset();
  implementation_->last_feedback_revision.reset();
  implementation_->last_finality_revision.reset();
}

std::size_t VisualFactorBatchBuilder::activeTracks() const noexcept {
  return implementation_->tracks.size();
}

namespace gtsam_api {
namespace {

constexpr double kBoundedEtaInteriorScale =
    1.0 - 128.0 * std::numeric_limits<double>::epsilon();

[[nodiscard]] gtsam::SharedNoiseModel robustNoiseModel(const VisualReprojectionFactorSpec& spec) {
  const auto validation = validateSpec(spec);
  if (!validation) {
    throw std::invalid_argument(validation.error().detail);
  }
  const auto gaussian = gtsam::noiseModel::Gaussian::Covariance(spec.observer.pixel_covariance);
  return gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Huber::Create(spec.huber_delta_sigma), gaussian);
}

[[nodiscard]] core::Pose3d toCorePose(const gtsam::Pose3& pose) {
  return core::Pose3d(Sophus::SO3d(pose.rotation().matrix()), pose.translation());
}

}  // namespace

BoundedEtaValue decodeBoundedEta(double latent, double minimum_range_m,
                                 double maximum_range_m) noexcept {
  if (!std::isfinite(latent) || !positiveFinite(minimum_range_m) ||
      !positiveFinite(maximum_range_m) || minimum_range_m >= maximum_range_m) {
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    return BoundedEtaValue{invalid, invalid};
  }
  const double lower_eta = -std::log(maximum_range_m);
  const double upper_eta = -std::log(minimum_range_m);
  const double center = 0.5 * (lower_eta + upper_eta);
  const double half_span = 0.5 * (upper_eta - lower_eta);
  const double normalized_latent = latent / half_span;
  const double bounded = std::tanh(normalized_latent);
  return BoundedEtaValue{
      center + half_span * kBoundedEtaInteriorScale * bounded,
      kBoundedEtaInteriorScale * (1.0 - bounded * bounded)};
}

double encodeBoundedEta(double eta, double minimum_range_m, double maximum_range_m) {
  if (!std::isfinite(eta) || !positiveFinite(minimum_range_m) ||
      !positiveFinite(maximum_range_m) || minimum_range_m >= maximum_range_m) {
    throw std::invalid_argument("bounded eta encoding requires finite eta and valid range bounds");
  }
  const double lower_eta = -std::log(maximum_range_m);
  const double upper_eta = -std::log(minimum_range_m);
  if (eta < lower_eta || eta > upper_eta) {
    throw std::invalid_argument("bounded eta encoding requires eta inside configured bounds");
  }
  const double center = 0.5 * (lower_eta + upper_eta);
  const double half_span = 0.5 * (upper_eta - lower_eta);
  double normalized = (eta - center) / (half_span * kBoundedEtaInteriorScale);
  const double open_limit = std::nextafter(1.0, 0.0);
  normalized = std::clamp(normalized, -open_limit, open_limit);
  return half_span * std::atanh(normalized);
}

VisualFactorEvaluationException::VisualFactorEvaluationException(VisualReprojectionError error)
    : std::runtime_error(std::move(error.detail)), code_(error.code) {}

AnchoredInverseRangeFactor::AnchoredInverseRangeFactor(gtsam::Key anchor_pose_key,
                                                       gtsam::Key observer_pose_key,
                                                       gtsam::Key eta_key,
                                                       VisualReprojectionFactorSpec spec)
    : Base(robustNoiseModel(spec), anchor_pose_key, observer_pose_key, eta_key),
      spec_(std::move(spec)) {
  if (anchor_pose_key == observer_pose_key || anchor_pose_key == eta_key ||
      observer_pose_key == eta_key) {
    throw std::invalid_argument("visual anchor, observer, and eta keys must be distinct");
  }
}

gtsam::Vector AnchoredInverseRangeFactor::evaluateError(
    const gtsam::Pose3& T_odom_imu_anchor, const gtsam::Pose3& T_odom_imu_observer,
    const double& bounded_eta_latent, boost::optional<gtsam::Matrix&> anchor_jacobian,
    boost::optional<gtsam::Matrix&> observer_jacobian,
    boost::optional<gtsam::Matrix&> eta_jacobian) const {
  const BoundedEtaValue bounded_eta =
      decodeBoundedEta(bounded_eta_latent, spec_.minimum_range_m, spec_.maximum_range_m);
  if (!std::isfinite(bounded_eta.eta) || !std::isfinite(bounded_eta.derivative_wrt_latent)) {
    throw VisualFactorEvaluationException(
        VisualReprojectionError{VisualReprojectionErrorCode::NonFiniteEta,
                                "visual bounded inverse-range latent is non-finite"});
  }
  const auto evaluation = evaluateVisualReprojection(spec_, toCorePose(T_odom_imu_anchor),
                                                     toCorePose(T_odom_imu_observer),
                                                     bounded_eta.eta);
  if (!evaluation) {
    throw VisualFactorEvaluationException(evaluation.error());
  }
  if (anchor_jacobian) {
    *anchor_jacobian = toGtsamPoseJacobian(evaluation.value().anchor_pose_jacobian);
  }
  if (observer_jacobian) {
    *observer_jacobian = toGtsamPoseJacobian(evaluation.value().observer_pose_jacobian);
  }
  if (eta_jacobian) {
    *eta_jacobian =
        evaluation.value().eta_jacobian * bounded_eta.derivative_wrt_latent;
  }
  return evaluation.value().residual;
}

}  // namespace gtsam_api
}  // namespace meridian::local
