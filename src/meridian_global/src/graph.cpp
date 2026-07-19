#include "meridian/global/graph.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <boost/make_shared.hpp>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <numbers>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "adjacent_boundary_factor_internal.hpp"
#include "graph_checkpoint_internal.hpp"
#include "meridian/global/loop_consensus.hpp"

namespace meridian::global {
namespace {

using AnchorKey = gtsam::Key;
using VelocityKey = gtsam::Key;
using GyroBiasKey = gtsam::Key;
using AccelBiasKey = gtsam::Key;
using AlignmentKey = gtsam::Key;
using AnchorIdentityKey = core::SparseSubmapIdentityKey;

constexpr double kTwoPi = 2.0 * std::numbers::pi;

[[nodiscard]] double wrapYaw(double yaw) noexcept {
  return std::remainder(yaw, kTwoPi);
}

[[nodiscard]] Eigen::Matrix3d rotationZ(double yaw) noexcept {
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  Eigen::Matrix3d rotation;
  rotation << cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0;
  return rotation;
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) noexcept {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return matrix;
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] GlobalGraphError graphError(GlobalGraphErrorCode code, std::string detail) {
  return GlobalGraphError{code, std::move(detail), std::nullopt};
}

[[nodiscard]] GlobalGraphCheckpointError recoveryError(GlobalGraphCheckpointErrorCode code,
                                                       std::string detail) {
  return GlobalGraphCheckpointError{code, std::move(detail), std::nullopt, std::nullopt};
}

[[nodiscard]] bool withinTolerance(double actual, double expected, double absolute,
                                   double relative) noexcept {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             absolute + relative * std::max({1.0, std::abs(actual), std::abs(expected)});
}

template <typename DerivedLeft, typename DerivedRight>
[[nodiscard]] bool matrixWithinTolerance(const Eigen::MatrixBase<DerivedLeft>& actual,
                                         const Eigen::MatrixBase<DerivedRight>& expected,
                                         double absolute, double relative) noexcept {
  const double scale =
      std::max({1.0, actual.cwiseAbs().maxCoeff(), expected.cwiseAbs().maxCoeff()});
  return actual.allFinite() && expected.allFinite() &&
         (actual - expected).cwiseAbs().maxCoeff() <= absolute + relative * scale;
}

[[nodiscard]] bool poseWithinTolerance(const core::Pose3d& actual, const core::Pose3d& expected,
                                       double tolerance) noexcept {
  return actual.matrix().allFinite() && expected.matrix().allFinite() &&
         (expected.inverse() * actual).log().norm() <= tolerance;
}

[[nodiscard]] bool exactRecoveryTolerances(const GlobalGraphRecoveryTolerances& left,
                                           const GlobalGraphRecoveryTolerances& right) noexcept {
  return left.objective_absolute == right.objective_absolute &&
         left.objective_relative == right.objective_relative &&
         left.estimate_tangent_absolute == right.estimate_tangent_absolute &&
         left.gradient_infinity_absolute == right.gradient_infinity_absolute &&
         left.covariance_absolute == right.covariance_absolute &&
         left.covariance_relative == right.covariance_relative &&
         left.condition_relative == right.condition_relative;
}

[[nodiscard]] RobustLoopTransactionError robustLoopError(
    RobustLoopTransactionErrorCode code, std::string detail,
    std::optional<ProposalId> proposal = std::nullopt) {
  RobustLoopTransactionError result;
  result.code = code;
  result.proposal = proposal;
  result.detail = std::move(detail);
  return result;
}

[[nodiscard]] bool validLoopGncConfig(const GncTlsConfig& config) noexcept {
  return config.maximum_known_inliers <= GncTlsController::kHardMaximumKnownInliers &&
         config.maximum_robust_candidates > 0U &&
         config.maximum_robust_candidates <= GncTlsController::kHardMaximumRobustCandidates &&
         config.maximum_iterations > 0U &&
         config.maximum_iterations <= GncTlsController::kHardMaximumIterations &&
         config.maximum_inner_solver_iterations > 0U &&
         config.maximum_inner_solver_iterations <=
             GncTlsController::kHardMaximumInnerSolverIterations &&
         config.maximum_degrees_of_freedom > 0U &&
         config.maximum_degrees_of_freedom <= GncTlsController::kHardMaximumDegreesOfFreedom &&
         std::isfinite(config.mu_step) && config.mu_step > 1.0 &&
         config.mu_step <= GncTlsController::kHardMaximumMuStep &&
         finitePositive(config.minimum_mu) && finitePositive(config.maximum_mu) &&
         config.minimum_mu <= config.maximum_mu &&
         config.maximum_mu <= GncTlsController::kHardMaximumMu &&
         finitePositive(config.binary_weight_tolerance) && config.binary_weight_tolerance < 0.5 &&
         finitePositive(config.stable_weight_tolerance) &&
         config.stable_weight_tolerance <= config.binary_weight_tolerance &&
         finitePositive(config.minimum_chi_squared_cutoff) &&
         finitePositive(config.maximum_chi_squared_cutoff) &&
         config.minimum_chi_squared_cutoff <= config.maximum_chi_squared_cutoff &&
         config.maximum_chi_squared_cutoff <= GncTlsController::kHardMaximumChiSquaredCutoff &&
         finitePositive(config.maximum_whitened_squared_cost) &&
         config.maximum_whitened_squared_cost <=
             GncTlsController::kHardMaximumWhitenedSquaredCost &&
         finitePositive(config.maximum_normalized_squared_cost) &&
         config.maximum_normalized_squared_cost <=
             GncTlsController::kHardMaximumNormalizedSquaredCost;
}

[[nodiscard]] bool validConfig(const GlobalGraphConfig& config) noexcept {
  const bool known_factor_capacity_covers_graph =
      config.loop_gnc.maximum_known_inliers > config.maximum_adjacent_factors &&
      config.loop_gnc.maximum_known_inliers - config.maximum_adjacent_factors >
          config.maximum_gnss_factors;
  return validLoopGncConfig(config.loop_gnc) && known_factor_capacity_covers_graph &&
         config.maximum_anchors > 0U && config.maximum_scalar_dimension >= 6U &&
         config.maximum_adjacent_factors > 0U &&
         config.maximum_adjacent_seals_per_transaction > 0U &&
         config.maximum_adjacent_seals_per_transaction <= config.maximum_adjacent_factors &&
         config.maximum_adjacent_factor_rows > 0U &&
         config.maximum_adjacent_factor_coefficients >= 30U &&
         config.maximum_total_adjacent_factor_rows >= config.maximum_adjacent_factor_rows &&
         config.maximum_total_adjacent_factor_coefficients >=
             config.maximum_adjacent_factor_coefficients &&
         config.maximum_loop_factors > 0U &&
         config.maximum_loop_factors <= config.loop_gnc.maximum_robust_candidates &&
         config.maximum_loop_candidates_per_transaction > 0U &&
         config.maximum_loop_candidates_per_transaction <= config.maximum_loop_factors &&
         config.maximum_loop_candidates_per_transaction <=
             config.loop_gnc.maximum_robust_candidates &&
         config.maximum_loop_calibration_epochs > 0U && config.maximum_loop_lineage_usages > 0U &&
         config.maximum_loop_lineage_correlations > 0U && config.maximum_solver_iterations >= 2U &&
         finitePositive(config.mission_gauge_translation_sigma_m) &&
         finitePositive(config.mission_gauge_rotation_sigma_rad) &&
         finitePositive(config.covariance_symmetry_relative_tolerance) &&
         finitePositive(config.minimum_covariance_eigenvalue) &&
         finitePositive(config.information_basis_orthonormal_tolerance) &&
         finitePositive(config.information_zero_tolerance) &&
         finitePositive(config.hessian_absolute_rank_tolerance) &&
         finitePositive(config.hessian_relative_rank_tolerance) &&
         config.hessian_relative_rank_tolerance < 1.0 &&
         finitePositive(config.maximum_hessian_condition) &&
         config.maximum_hessian_condition > 1.0 &&
         finitePositive(config.solver_relative_error_tolerance) &&
         finitePositive(config.solver_absolute_error_tolerance);
}

[[nodiscard]] bool validPose(const core::Pose3d& pose) noexcept {
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = pose.so3().matrix();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <
             1.0e-10 &&
         std::abs(rotation.determinant() - 1.0) < 1.0e-10;
}

[[nodiscard]] bool validSubmapRef(const core::SubmapRef& submap) noexcept {
  return core::validateSubmapRef(submap) == core::SubmapRefValidationError::None;
}

[[nodiscard]] bool sameSubmapObject(const core::SubmapRef& lhs,
                                    const core::SubmapRef& rhs) noexcept {
  return lhs.session == rhs.session && lhs.odom_epoch == rhs.odom_epoch && lhs.id == rhs.id;
}

[[nodiscard]] bool validRankAwareInformation(const core::RankAwareInformation& information,
                                             const GlobalGraphConfig& config);

[[nodiscard]] bool validRecordHeader(const core::RecordHeader& header) noexcept {
  return header.schema_version > 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() &&
         (!header.direct_calibration || header.direct_calibration->valid());
}

[[nodiscard]] bool containsCalibration(std::span<const core::CalibrationEpoch> epochs,
                                       core::CalibrationEpoch calibration) noexcept {
  return std::binary_search(epochs.begin(), epochs.end(), calibration);
}

[[nodiscard]] bool validLoopCandidate(const RobustLoopCandidate& candidate,
                                      const GlobalGraphConfig& config, std::string* detail) {
  const LoopMeasurement& measurement = candidate.measurement;
  if (!validRecordHeader(measurement.header) || !measurement.proposal.valid()) {
    *detail = "loop header or proposal identity is invalid";
    return false;
  }
  switch (measurement.modality) {
    case LoopModality::Visual:
    case LoopModality::Lidar:
      break;
    default:
      *detail = "loop modality is invalid";
      return false;
  }
  if (!validSubmapRef(measurement.from) || !validSubmapRef(measurement.to) ||
      sameSubmapObject(measurement.from, measurement.to) ||
      measurement.header.session != measurement.from.session ||
      measurement.header.session != measurement.to.session) {
    *detail = "loop endpoints must be valid and distinct";
    return false;
  }
  if (!validPose(measurement.T_from_to) ||
      !validRankAwareInformation(measurement.information, config)) {
    *detail = "loop pose or rank-aware information is invalid";
    return false;
  }
  if (candidate.scale.degrees_of_freedom != measurement.information.rank ||
      candidate.scale.degrees_of_freedom > config.loop_gnc.maximum_degrees_of_freedom ||
      !finitePositive(candidate.scale.calibrated_chi_squared_cutoff) ||
      candidate.scale.calibrated_chi_squared_cutoff < config.loop_gnc.minimum_chi_squared_cutoff ||
      candidate.scale.calibrated_chi_squared_cutoff > config.loop_gnc.maximum_chi_squared_cutoff) {
    *detail = "loop TLS scale must use the supported information rank and a positive cutoff";
    return false;
  }
  if (measurement.calibration_epochs.empty() ||
      measurement.calibration_epochs.size() > config.maximum_loop_calibration_epochs ||
      !std::is_sorted(measurement.calibration_epochs.begin(),
                      measurement.calibration_epochs.end()) ||
      std::adjacent_find(measurement.calibration_epochs.begin(),
                         measurement.calibration_epochs.end()) !=
          measurement.calibration_epochs.end() ||
      std::any_of(measurement.calibration_epochs.begin(), measurement.calibration_epochs.end(),
                  [](core::CalibrationEpoch epoch) { return !epoch.valid(); })) {
    *detail = "loop calibration epoch set is empty, invalid, non-canonical, or over capacity";
    return false;
  }
  if (measurement.header.direct_calibration &&
      !containsCalibration(measurement.calibration_epochs,
                           *measurement.header.direct_calibration)) {
    *detail = "loop direct calibration is absent from its calibration epoch set";
    return false;
  }
  if (!containsCalibration(measurement.calibration_epochs, measurement.from.calibration) ||
      !containsCalibration(measurement.calibration_epochs, measurement.to.calibration)) {
    *detail = "loop endpoint calibration is absent from its calibration epoch set";
    return false;
  }
  if (!measurement.lineage.id.valid() || measurement.lineage.usage.empty() ||
      measurement.lineage.usage.size() > config.maximum_loop_lineage_usages ||
      measurement.lineage.correlations.size() > config.maximum_loop_lineage_correlations) {
    *detail = "loop lineage identity, roots, or capacity is invalid";
    return false;
  }
  bool has_primary = false;
  for (const core::ObservationUsage& usage : measurement.lineage.usage) {
    if (!containsCalibration(measurement.calibration_epochs, usage.slice.calibration)) {
      *detail = "loop lineage calibration is absent from its calibration epoch set";
      return false;
    }
    has_primary = has_primary || usage.role == core::ObservationRole::PrimaryResidual;
  }
  if (!has_primary ||
      core::validateLineage(measurement.lineage) != core::LineageValidationError::None) {
    *detail = "loop lineage has no primary residual or failed canonical validation";
    return false;
  }
  return true;
}

[[nodiscard]] bool validCovariance(const Eigen::Matrix3d& covariance,
                                   const GlobalGraphConfig& config) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() >
      config.covariance_symmetry_relative_tolerance * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(
      0.5 * (covariance + covariance.transpose()), Eigen::EigenvaluesOnly);
  return eigensolver.info() == Eigen::Success && eigensolver.eigenvalues().allFinite() &&
         eigensolver.eigenvalues().minCoeff() >= config.minimum_covariance_eigenvalue;
}

[[nodiscard]] bool validRankAwareInformation(const core::RankAwareInformation& information,
                                             const GlobalGraphConfig& config) {
  if (!information.finite() || information.rank == 0U || information.rank > 6U ||
      information.tangent != core::PoseTangentConvention::RightTranslationFirst) {
    return false;
  }
  if ((information.basis.transpose() * information.basis - Eigen::Matrix<double, 6, 6>::Identity())
          .cwiseAbs()
          .maxCoeff() > config.information_basis_orthonormal_tolerance) {
    return false;
  }
  for (std::size_t index = 0; index < 6U; ++index) {
    const double eigenvalue = information.eigenvalues(static_cast<Eigen::Index>(index));
    if (index < information.rank) {
      if (!finitePositive(eigenvalue)) {
        return false;
      }
    } else if (std::abs(eigenvalue) > config.information_zero_tolerance) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] gtsam::Pose3 toGtsamPose(const core::Pose3d& pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.so3().matrix()), pose.translation());
}

[[nodiscard]] core::Pose3d fromGtsamPose(const gtsam::Pose3& pose) {
  return core::Pose3d(Sophus::SO3d(pose.rotation().matrix()), pose.translation());
}

[[nodiscard]] Eigen::Matrix<double, 6, 1> meridianPoseError(const gtsam::Pose3& measured_from_to,
                                                            const gtsam::Pose3& T_map_from,
                                                            const gtsam::Pose3& T_map_to) {
  const core::Pose3d measured = fromGtsamPose(measured_from_to);
  const core::Pose3d from = fromGtsamPose(T_map_from);
  const core::Pose3d to = fromGtsamPose(T_map_to);
  return (measured.inverse() * from.inverse() * to).log();
}

[[nodiscard]] core::PoseCovariance fromGtsamPoseCovariance(const gtsam::Matrix6& covariance) {
  core::PoseCovariance converted;
  converted.matrix.topLeftCorner<3, 3>() = covariance.bottomRightCorner<3, 3>();
  converted.matrix.topRightCorner<3, 3>() = covariance.bottomLeftCorner<3, 3>();
  converted.matrix.bottomLeftCorner<3, 3>() = covariance.topRightCorner<3, 3>();
  converted.matrix.bottomRightCorner<3, 3>() = covariance.topLeftCorner<3, 3>();
  return converted;
}

class RankAwareRelativeFactor final : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

  RankAwareRelativeFactor(AnchorKey from_key, AnchorKey to_key, core::Pose3d T_from_to,
                          const core::RankAwareInformation& information,
                          double information_weight = 1.0)
      : Base(gtsam::noiseModel::Unit::Create(information.rank), from_key, to_key),
        measured_from_to_(toGtsamPose(T_from_to)),
        square_root_projection_(static_cast<Eigen::Index>(information.rank), 6) {
    const double square_root_weight = std::sqrt(std::clamp(information_weight, 0.0, 1.0));
    for (std::size_t row = 0; row < information.rank; ++row) {
      square_root_projection_.row(static_cast<Eigen::Index>(row)) =
          square_root_weight * std::sqrt(information.eigenvalues(static_cast<Eigen::Index>(row))) *
          information.basis.col(static_cast<Eigen::Index>(row)).transpose();
    }
  }

  gtsam::Vector evaluateError(
      const gtsam::Pose3& T_map_from, const gtsam::Pose3& T_map_to,
      boost::optional<gtsam::Matrix&> from_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> to_jacobian = boost::none) const override {
    const auto residual = [&](const gtsam::Pose3& from, const gtsam::Pose3& to) -> gtsam::Vector {
      const Eigen::Matrix<double, 6, 1> error = meridianPoseError(measured_from_to_, from, to);
      return square_root_projection_ * error;
    };
    const gtsam::Vector result = residual(T_map_from, T_map_to);
    constexpr double kDerivativeStep = 1.0e-6;
    if (from_jacobian) {
      *from_jacobian = gtsam::Matrix::Zero(result.size(), 6);
      for (Eigen::Index column = 0; column < 6; ++column) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero();
        delta(column) = kDerivativeStep;
        const gtsam::Pose3 plus = T_map_from.compose(gtsam::Pose3::Expmap(delta));
        const gtsam::Pose3 minus = T_map_from.compose(gtsam::Pose3::Expmap(-delta));
        from_jacobian->col(column) =
            (residual(plus, T_map_to) - residual(minus, T_map_to)) / (2.0 * kDerivativeStep);
      }
    }
    if (to_jacobian) {
      *to_jacobian = gtsam::Matrix::Zero(result.size(), 6);
      for (Eigen::Index column = 0; column < 6; ++column) {
        gtsam::Vector6 delta = gtsam::Vector6::Zero();
        delta(column) = kDerivativeStep;
        const gtsam::Pose3 plus = T_map_to.compose(gtsam::Pose3::Expmap(delta));
        const gtsam::Pose3 minus = T_map_to.compose(gtsam::Pose3::Expmap(-delta));
        to_jacobian->col(column) =
            (residual(T_map_from, plus) - residual(T_map_from, minus)) / (2.0 * kDerivativeStep);
      }
    }
    return result;
  }

private:
  gtsam::Pose3 measured_from_to_;
  Eigen::Matrix<double, Eigen::Dynamic, 6> square_root_projection_;
};

class GnssAntennaFactor final : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector4> {
public:
  using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Vector4>;

  GnssAntennaFactor(AnchorKey anchor_key, AlignmentKey alignment_key,
                    Eigen::Vector3d antenna_position_submap, Eigen::Vector3d measured_position_enu,
                    const Eigen::Matrix3d& covariance)
      : Base(gtsam::noiseModel::Gaussian::Covariance(covariance), anchor_key, alignment_key),
        antenna_position_submap_(std::move(antenna_position_submap)),
        measured_position_enu_(std::move(measured_position_enu)) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& T_map_submap, const gtsam::Vector4& alignment,
      boost::optional<gtsam::Matrix&> anchor_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> alignment_jacobian = boost::none) const override {
    const GnssAntennaPrediction prediction =
        predictGnssAntenna(fromGtsamPose(T_map_submap), antenna_position_submap_,
                           YawTranslation4{alignment.head<3>(), alignment(3)});

    if (anchor_jacobian) {
      // GTSAM Pose3 right tangent is [rotation, translation]. Meridian public
      // pose tangents are [translation, rotation], so this permutation is
      // explicit at the private adapter boundary.
      anchor_jacobian->resize(3, 6);
      anchor_jacobian->leftCols<3>() = prediction.position_jacobian_anchor.rightCols<3>();
      anchor_jacobian->rightCols<3>() = prediction.position_jacobian_anchor.leftCols<3>();
    }
    if (alignment_jacobian) {
      *alignment_jacobian = prediction.position_jacobian_alignment;
    }
    return prediction.position_enu - measured_position_enu_;
  }

private:
  Eigen::Vector3d antenna_position_submap_;
  Eigen::Vector3d measured_position_enu_;
};

[[nodiscard]] gtsam::LevenbergMarquardtParams optimizerParams(const GlobalGraphConfig& config) {
  gtsam::LevenbergMarquardtParams params;
  params.maxIterations = config.maximum_solver_iterations;
  params.relativeErrorTol = config.solver_relative_error_tolerance;
  params.absoluteErrorTol = config.solver_absolute_error_tolerance;
  params.linearSolverType = gtsam::NonlinearOptimizerParams::MULTIFRONTAL_QR;
  params.verbosity = gtsam::NonlinearOptimizerParams::SILENT;
  params.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  return params;
}

[[nodiscard]] bool finitePoseCovariance(const core::PoseCovariance& covariance) {
  if (covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !covariance.matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  if ((covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff() > 1.0e-9 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<core::Matrix6d> eigensolver(
      0.5 * (covariance.matrix + covariance.matrix.transpose()), Eigen::EigenvaluesOnly);
  return eigensolver.info() == Eigen::Success && eigensolver.eigenvalues().allFinite() &&
         eigensolver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

}  // namespace

Eigen::Matrix3d YawTranslation4::REnuMap() const noexcept {
  return rotationZ(yaw_enu_map_rad);
}

Eigen::Vector3d YawTranslation4::apply(const Eigen::Vector3d& position_map) const noexcept {
  return REnuMap() * position_map + translation_enu;
}

GnssAntennaPrediction predictGnssAntenna(const core::Pose3d& T_map_submap,
                                         const Eigen::Vector3d& antenna_position_submap,
                                         const YawTranslation4& T_enu_map) noexcept {
  GnssAntennaPrediction prediction;
  const Eigen::Matrix3d R_map_submap = T_map_submap.so3().matrix();
  const Eigen::Vector3d position_map = T_map_submap * antenna_position_submap;
  const Eigen::Matrix3d R_enu_map = T_enu_map.REnuMap();
  const Eigen::Vector3d rotated_position = R_enu_map * position_map;
  prediction.position_enu = rotated_position + T_enu_map.translation_enu;
  prediction.position_jacobian_anchor.leftCols<3>() = R_enu_map * R_map_submap;
  prediction.position_jacobian_anchor.rightCols<3>() =
      -R_enu_map * R_map_submap * skew(antenna_position_submap);
  prediction.position_jacobian_alignment.leftCols<3>().setIdentity();
  prediction.position_jacobian_alignment.col(3) =
      Eigen::Vector3d{-rotated_position.y(), rotated_position.x(), 0.0};
  return prediction;
}

struct GlobalGraph::Impl {
  enum class KnownFactorKind {
    MissionGauge,
    Adjacent,
    Gnss,
  };

  struct BoundaryKeys {
    AnchorKey anchor{};
    VelocityKey velocity{};
    GyroBiasKey gyro_bias{};
    AccelBiasKey accel_bias{};
  };

  struct AnchorRecord {
    core::SparseSubmapSeal seal;
    core::Pose3d C_submap_imu;
    BoundaryKeys keys;
    bool navigation_materialized{false};
    std::size_t insertion_index{};
  };

  struct KnownFactorRecord {
    GlobalFactorId id;
    KnownFactorKind kind{KnownFactorKind::Adjacent};
    std::optional<GncTlsFactorScale> scale;
    gtsam::NonlinearFactor::shared_ptr factor;
    std::optional<AdjacentBoundaryCheckpoint> adjacent;
    std::optional<GnssAntennaConstraint> gnss;
  };

  struct LoopFactorRecord {
    GlobalFactorId factor_id;
    LoopMeasurement measurement;
    CandidateId candidate;
    GncTlsFactorScale scale;
    AnchorKey from_key;
    AnchorKey to_key;
    gtsam::NonlinearFactor::shared_ptr unweighted_factor;
  };

  struct State {
    gtsam::NonlinearFactorGraph factors;
    gtsam::Values values;
    std::vector<KnownFactorRecord> known_factors;
    std::vector<LoopFactorRecord> loop_factors;
    std::map<AnchorIdentityKey, AnchorRecord> anchors;
    std::vector<AnchorIdentityKey> anchor_order;
    std::vector<std::pair<gtsam::Key, gtsam::Key>> connectivity_edges;
    std::unordered_set<std::uint64_t> gnss_observations;
    std::map<core::OdomEpoch, core::Pose3d> chart_placements;
    std::size_t adjacent_factors{};
    std::size_t materialized_navigation_boundaries{};
    std::size_t total_adjacent_rows{};
    std::size_t total_adjacent_coefficients{};
    std::size_t gnss_factors{};
    std::size_t next_anchor_index{};
    std::uint64_t next_factor_id{};
    std::uint64_t next_candidate_id{};
    bool alignment_exists{false};
  };

  explicit Impl(GlobalGraphConfig graph_config) : config(std::move(graph_config)) {}

  [[nodiscard]] BoundaryKeys nextBoundaryKeys(const State& candidate) const {
    const std::size_t index = candidate.next_anchor_index;
    return BoundaryKeys{gtsam::Symbol('x', index), gtsam::Symbol('v', index),
                        gtsam::Symbol('g', index), gtsam::Symbol('b', index)};
  }

  [[nodiscard]] static AlignmentKey alignmentKey() { return gtsam::Symbol('e', 0U); }

  [[nodiscard]] core::Result<GlobalFactorId, GlobalGraphError> allocateFactorId(
      State& candidate) const {
    if (candidate.next_factor_id >= GlobalFactorId::kInvalidValue) {
      return core::Result<GlobalFactorId, GlobalGraphError>::failure(graphError(
          GlobalGraphErrorCode::FactorIdOverflow, "global factor identity space is exhausted"));
    }
    const GlobalFactorId result(candidate.next_factor_id);
    ++candidate.next_factor_id;
    return core::Result<GlobalFactorId, GlobalGraphError>::success(result);
  }

  [[nodiscard]] static std::optional<CandidateId> allocateCandidateId(State& candidate) {
    if (candidate.next_candidate_id >= CandidateId::kInvalidValue) {
      return std::nullopt;
    }
    const CandidateId result(candidate.next_candidate_id);
    ++candidate.next_candidate_id;
    return result;
  }

  [[nodiscard]] static GncTlsFactorScale gnssScale() { return GncTlsFactorScale{3U, 11.34486673}; }

  [[nodiscard]] std::optional<GncTlsFactorScale> adjacentMonitoringScale(
      const core::FrozenSquareRootFactor& factor) const {
    const auto& statistics = factor.cost_statistics;
    if (!statistics.calibrated_total_cost_cutoff ||
        statistics.effective_dof > config.loop_gnc.maximum_degrees_of_freedom ||
        *statistics.calibrated_total_cost_cutoff < config.loop_gnc.minimum_chi_squared_cutoff ||
        *statistics.calibrated_total_cost_cutoff > config.loop_gnc.maximum_chi_squared_cutoff) {
      return std::nullopt;
    }
    return GncTlsFactorScale{static_cast<std::uint32_t>(statistics.effective_dof),
                             *statistics.calibrated_total_cost_cutoff};
  }

  void rebuildFactors(State& candidate) const {
    gtsam::NonlinearFactorGraph rebuilt;
    rebuilt.reserve(candidate.known_factors.size() + candidate.loop_factors.size());
    for (const KnownFactorRecord& factor : candidate.known_factors) {
      rebuilt.push_back(factor.factor);
    }
    for (const LoopFactorRecord& factor : candidate.loop_factors) {
      rebuilt.push_back(factor.unweighted_factor);
    }
    candidate.factors = std::move(rebuilt);
  }

  [[nodiscard]] bool connected(const State& candidate) const {
    const std::size_t variable_count = candidate.values.size();
    if (variable_count == 0U) {
      return false;
    }
    std::unordered_map<gtsam::Key, std::vector<gtsam::Key>> adjacency;
    for (const auto& [from, to] : candidate.connectivity_edges) {
      adjacency[from].push_back(to);
      adjacency[to].push_back(from);
    }
    for (const LoopFactorRecord& loop : candidate.loop_factors) {
      adjacency[loop.from_key].push_back(loop.to_key);
      adjacency[loop.to_key].push_back(loop.from_key);
    }
    const AnchorKey root = candidate.anchors.at(candidate.anchor_order.front()).keys.anchor;
    std::queue<gtsam::Key> pending;
    std::unordered_set<gtsam::Key> visited;
    pending.push(root);
    visited.insert(root);
    while (!pending.empty()) {
      const gtsam::Key current = pending.front();
      pending.pop();
      const auto neighbours = adjacency.find(current);
      if (neighbours == adjacency.end()) {
        continue;
      }
      for (const gtsam::Key neighbour : neighbours->second) {
        if (visited.insert(neighbour).second) {
          pending.push(neighbour);
        }
      }
    }
    return visited.size() == variable_count;
  }

  [[nodiscard]] gtsam::Ordering ordering(const State& candidate) const {
    gtsam::Ordering result;
    for (const AnchorIdentityKey& id : candidate.anchor_order) {
      const AnchorRecord& boundary = candidate.anchors.at(id);
      result.push_back(boundary.keys.anchor);
      if (boundary.navigation_materialized) {
        result.push_back(boundary.keys.velocity);
        result.push_back(boundary.keys.gyro_bias);
        result.push_back(boundary.keys.accel_bias);
      }
    }
    if (candidate.alignment_exists) {
      result.push_back(alignmentKey());
    }
    return result;
  }

  [[nodiscard]] const AnchorRecord& referenceAnchor(const State& candidate) const {
    const AnchorRecord* reference = &candidate.anchors.at(candidate.anchor_order.front());
    for (const AnchorIdentityKey& id : candidate.anchor_order) {
      const AnchorRecord& proposed = candidate.anchors.at(id);
      const auto proposed_key =
          std::tuple{proposed.seal.support_time.end.nanoseconds, proposed.seal.ref.session.value(),
                     proposed.seal.ref.odom_epoch.value(), proposed.seal.ref.id.value(),
                     proposed.seal.ref.content_revision.value()};
      const auto reference_key =
          std::tuple{reference->seal.support_time.end.nanoseconds,
                     reference->seal.ref.session.value(), reference->seal.ref.odom_epoch.value(),
                     reference->seal.ref.id.value(), reference->seal.ref.content_revision.value()};
      if (proposed_key > reference_key) {
        reference = &proposed;
      }
    }
    return *reference;
  }

  class LoopShadowSolver final : public GncTlsShadowSolverApi {
  public:
    LoopShadowSolver(const GlobalGraphConfig& graph_config, const State& shadow)
        : config_(graph_config), shadow_(shadow), working_values_(shadow.values) {}

    [[nodiscard]] const gtsam::Values& workingValues() const noexcept { return working_values_; }

    [[nodiscard]] core::Result<GncTlsShadowSolveResult, ShadowSolveFailure> solve(
        const GncTlsShadowSolveRequest& request) override {
      using Result = core::Result<GncTlsShadowSolveResult, ShadowSolveFailure>;
      if (request.known_inliers.size() != shadow_.known_factors.size() ||
          request.robust_candidates.size() != shadow_.loop_factors.size()) {
        return Result::failure(
            ShadowSolveFailure{ShadowSolveFailureCode::ResourceLimit,
                               "shadow request does not identify the complete active graph"});
      }

      gtsam::NonlinearFactorGraph weighted_graph;
      weighted_graph.reserve(shadow_.known_factors.size() + shadow_.loop_factors.size());
      for (std::size_t index = 0; index < shadow_.known_factors.size(); ++index) {
        const KnownFactorRecord& factor = shadow_.known_factors[index];
        const ShadowKnownInlierWeight& requested = request.known_inliers[index];
        if (requested.factor_id != factor.id || requested.weight != 1.0) {
          return Result::failure(
              ShadowSolveFailure{ShadowSolveFailureCode::ResourceLimit,
                                 "known-inlier IDs or fixed unit weights are incomplete"});
        }
        weighted_graph.push_back(factor.factor);
      }
      for (std::size_t index = 0; index < shadow_.loop_factors.size(); ++index) {
        const LoopFactorRecord& factor = shadow_.loop_factors[index];
        const ShadowCandidateWeight& requested = request.robust_candidates[index];
        if (requested.candidate_id != factor.candidate || !std::isfinite(requested.weight) ||
            requested.weight < 0.0 || requested.weight > 1.0) {
          return Result::failure(ShadowSolveFailure{ShadowSolveFailureCode::ResourceLimit,
                                                    "robust-candidate IDs or weights are invalid"});
        }
        weighted_graph.push_back(boost::make_shared<RankAwareRelativeFactor>(
            factor.from_key, factor.to_key, factor.measurement.T_from_to,
            factor.measurement.information, requested.weight));
      }

      try {
        const double initial_error = weighted_graph.error(working_values_);
        gtsam::LevenbergMarquardtParams params = optimizerParams(config_);
        params.maxIterations = std::min(config_.maximum_solver_iterations,
                                        config_.loop_gnc.maximum_inner_solver_iterations);
        gtsam::LevenbergMarquardtOptimizer optimizer(weighted_graph, working_values_, params);
        gtsam::Values optimized = optimizer.optimize();
        const double final_error = weighted_graph.error(optimized);
        const double allowed_error_increase = std::max(
            config_.solver_absolute_error_tolerance,
            config_.solver_relative_error_tolerance * std::max(1.0, std::abs(initial_error)));
        if (!std::isfinite(initial_error) || !std::isfinite(final_error)) {
          return Result::failure(ShadowSolveFailure{ShadowSolveFailureCode::NumericalFailure,
                                                    "weighted objective is non-finite"});
        }
        if ((optimizer.iterations() >= params.maxIterations &&
             final_error > config_.solver_absolute_error_tolerance) ||
            final_error > initial_error + allowed_error_increase) {
          return Result::failure(ShadowSolveFailure{
              ShadowSolveFailureCode::DidNotConverge,
              "complete weighted graph did not converge within the inner bound"});
        }
        for (const auto& [id, anchor] : shadow_.anchors) {
          (void)id;
          if (!validPose(fromGtsamPose(optimized.at<gtsam::Pose3>(anchor.keys.anchor)))) {
            return Result::failure(ShadowSolveFailure{ShadowSolveFailureCode::NumericalFailure,
                                                      "weighted anchor estimate is invalid"});
          }
        }
        if (shadow_.alignment_exists && !optimized.at<gtsam::Vector4>(alignmentKey()).allFinite()) {
          return Result::failure(ShadowSolveFailure{ShadowSolveFailureCode::NumericalFailure,
                                                    "weighted alignment estimate is invalid"});
        }

        GncTlsShadowSolveResult response;
        response.solver_iterations = optimizer.iterations();
        response.finite_solution = true;
        response.known_inliers.reserve(shadow_.known_factors.size());
        response.robust_candidates.reserve(shadow_.loop_factors.size());
        for (const KnownFactorRecord& factor : shadow_.known_factors) {
          response.known_inliers.push_back(
              ShadowKnownInlierCost{factor.id, 2.0 * factor.factor->error(optimized)});
        }
        for (const LoopFactorRecord& factor : shadow_.loop_factors) {
          response.robust_candidates.push_back(ShadowCandidateCost{
              factor.candidate, 2.0 * factor.unweighted_factor->error(optimized)});
        }
        working_values_ = std::move(optimized);
        return Result::success(std::move(response));
      } catch (const gtsam::IndeterminantLinearSystemException& exception) {
        return Result::failure(
            ShadowSolveFailure{ShadowSolveFailureCode::NumericalFailure,
                               std::string("indeterminate weighted graph: ") + exception.what()});
      } catch (const std::exception& exception) {
        return Result::failure(
            ShadowSolveFailure{ShadowSolveFailureCode::NumericalFailure,
                               std::string("weighted graph solve failed: ") + exception.what()});
      }
    }

  private:
    const GlobalGraphConfig& config_;
    const State& shadow_;
    gtsam::Values working_values_;
  };

  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalGraphError> optimizeAndCommit(
      State candidate, GlobalTransactionKind transaction,
      std::size_t adjacent_seals_in_transaction = 0U) {
    using Result = core::Result<GlobalGraphCommit, GlobalGraphError>;
    GlobalSolveReport report;
    // One canonical active-factor order is used for ordinary commits and for
    // recovery reconstruction. This removes insertion-history-dependent
    // floating-point summation from the persisted objective audit.
    rebuildFactors(candidate);
    report.transaction = transaction;
    report.anchors = candidate.anchors.size();
    report.materialized_navigation_boundaries = candidate.materialized_navigation_boundaries;
    report.adjacent_seals_in_transaction = adjacent_seals_in_transaction;
    report.adjacent_factors = candidate.adjacent_factors;
    report.gnss_factors = candidate.gnss_factors;
    report.loop_factors = candidate.loop_factors.size();
    report.connected = connected(candidate);
    if (!report.connected) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::DisconnectedProposal,
                     "candidate variables are not all connected to the single mission gauge"));
    }

    try {
      report.initial_error = candidate.factors.error(candidate.values);
      gtsam::LevenbergMarquardtOptimizer optimizer(candidate.factors, candidate.values,
                                                   optimizerParams(config));
      gtsam::Values optimized = optimizer.optimize();
      report.solver_iterations = optimizer.iterations();
      report.final_error = candidate.factors.error(optimized);
      report.finite = std::isfinite(report.initial_error) && std::isfinite(report.final_error);
      const double allowed_error_increase = std::max(
          config.solver_absolute_error_tolerance,
          config.solver_relative_error_tolerance * std::max(1.0, std::abs(report.initial_error)));
      report.converged = (report.solver_iterations < config.maximum_solver_iterations ||
                          report.final_error <= config.solver_absolute_error_tolerance) &&
                         report.final_error <= report.initial_error + allowed_error_increase;
      if (!report.finite) {
        return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                          "global candidate objective is non-finite"));
      }
      if (!report.converged) {
        return Result::failure(
            graphError(GlobalGraphErrorCode::SolverDidNotConverge,
                       "shadow optimization reached its iteration limit without convergence"));
      }

      for (const auto& [id, anchor] : candidate.anchors) {
        (void)id;
        if (!validPose(fromGtsamPose(optimized.at<gtsam::Pose3>(anchor.keys.anchor)))) {
          return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                            "optimized submap anchor is invalid"));
        }
        if (anchor.navigation_materialized &&
            (!optimized.at<gtsam::Vector3>(anchor.keys.velocity).allFinite() ||
             !optimized.at<gtsam::Vector3>(anchor.keys.gyro_bias).allFinite() ||
             !optimized.at<gtsam::Vector3>(anchor.keys.accel_bias).allFinite())) {
          return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                            "optimized boundary navigation state is invalid"));
        }
      }
      if (candidate.alignment_exists) {
        gtsam::Vector4 alignment = optimized.at<gtsam::Vector4>(alignmentKey());
        if (!alignment.allFinite()) {
          return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                            "optimized ENU alignment is non-finite"));
        }
        alignment(3) = wrapYaw(alignment(3));
        optimized.update(alignmentKey(), alignment);
      }

      const gtsam::Ordering variable_order = ordering(candidate);
      const auto gaussian = candidate.factors.linearize(optimized);
      const auto hessian_and_gradient = gaussian->hessian(variable_order);
      const gtsam::Matrix& hessian = hessian_and_gradient.first;
      report.scalar_dimension = static_cast<std::size_t>(hessian.rows());
      if (report.scalar_dimension > config.maximum_scalar_dimension) {
        return Result::failure(graphError(GlobalGraphErrorCode::ScalarDimensionCapacity,
                                          "candidate scalar dimension exceeds its hard cap"));
      }
      if (hessian.rows() == 0 || hessian.rows() != hessian.cols() || !hessian.allFinite()) {
        return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                          "candidate Hessian is empty, non-square, or non-finite"));
      }
      Eigen::SelfAdjointEigenSolver<gtsam::Matrix> eigensolver(
          0.5 * (hessian + hessian.transpose()), Eigen::EigenvaluesOnly);
      if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite()) {
        return Result::failure(graphError(GlobalGraphErrorCode::NonFiniteSolution,
                                          "candidate Hessian eigendecomposition failed"));
      }
      const double largest = eigensolver.eigenvalues().maxCoeff();
      const double rank_threshold =
          std::max(config.hessian_absolute_rank_tolerance,
                   config.hessian_relative_rank_tolerance * std::max(0.0, largest));
      double smallest_supported = std::numeric_limits<double>::infinity();
      for (Eigen::Index index = 0; index < eigensolver.eigenvalues().size(); ++index) {
        const double eigenvalue = eigensolver.eigenvalues()(index);
        if (eigenvalue > rank_threshold) {
          ++report.numerical_rank;
          smallest_supported = std::min(smallest_supported, eigenvalue);
        }
      }
      if (report.numerical_rank != report.scalar_dimension) {
        return Result::failure(
            graphError(GlobalGraphErrorCode::RankDeficientCandidate,
                       "candidate graph numerical rank " + std::to_string(report.numerical_rank) +
                           " does not constrain all " + std::to_string(report.scalar_dimension) +
                           " scalar variables at threshold " + std::to_string(rank_threshold) +
                           "; smallest eigenvalue is " +
                           std::to_string(eigensolver.eigenvalues().minCoeff())));
      }
      report.hessian_condition = largest / smallest_supported;
      if (!std::isfinite(report.hessian_condition) ||
          report.hessian_condition > config.maximum_hessian_condition) {
        return Result::failure(
            graphError(GlobalGraphErrorCode::IllConditionedCandidate,
                       "candidate Hessian exceeds the configured supported-space condition limit"));
      }

      gtsam::Marginals marginals(candidate.factors, optimized, gtsam::Marginals::QR);
      std::vector<SubmapAnchorEstimate> anchors;
      anchors.reserve(candidate.anchor_order.size());
      for (const AnchorIdentityKey& id : candidate.anchor_order) {
        const AnchorRecord& anchor = candidate.anchors.at(id);
        const core::PoseCovariance covariance =
            fromGtsamPoseCovariance(marginals.marginalCovariance(anchor.keys.anchor));
        if (!finitePoseCovariance(covariance)) {
          return Result::failure(
              graphError(GlobalGraphErrorCode::MarginalCovarianceFailure,
                         "anchor marginal is non-finite or has the wrong tangent convention"));
        }
        anchors.push_back(SubmapAnchorEstimate{
            anchor.seal.ref, anchor.seal.T_odom_submap, anchor.seal.support_time.end,
            fromGtsamPose(optimized.at<gtsam::Pose3>(anchor.keys.anchor)), covariance});
      }

      std::optional<YawTranslation4> alignment;
      std::optional<AlignmentCovariance> alignment_covariance;
      if (candidate.alignment_exists) {
        const gtsam::Vector4 estimate = optimized.at<gtsam::Vector4>(alignmentKey());
        alignment = YawTranslation4{estimate.head<3>(), wrapYaw(estimate(3))};
        const gtsam::Matrix covariance = marginals.marginalCovariance(alignmentKey());
        if (covariance.rows() != 4 || covariance.cols() != 4 || !covariance.allFinite()) {
          return Result::failure(
              graphError(GlobalGraphErrorCode::MarginalCovarianceFailure,
                         "alignment marginal is not a finite 4-by-4 [translation,yaw] covariance"));
        }
        const Eigen::Matrix4d covariance4 = covariance;
        const double covariance_scale = std::max(1.0, covariance4.cwiseAbs().maxCoeff());
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> covariance_eigensolver(
            0.5 * (covariance4 + covariance4.transpose()), Eigen::EigenvaluesOnly);
        if ((covariance4 - covariance4.transpose()).cwiseAbs().maxCoeff() >
                1.0e-9 * covariance_scale ||
            covariance_eigensolver.info() != Eigen::Success ||
            !covariance_eigensolver.eigenvalues().allFinite() ||
            covariance_eigensolver.eigenvalues().minCoeff() < -1.0e-10 * covariance_scale) {
          return Result::failure(
              graphError(GlobalGraphErrorCode::MarginalCovarianceFailure,
                         "alignment marginal is not symmetric positive semidefinite"));
        }
        alignment_covariance = AlignmentCovariance{covariance4};
      }

      std::optional<GlobalGraphRevision> parent;
      GlobalGraphRevision revision(0U);
      if (latest_commit) {
        parent = latest_commit->revision;
        if (parent->value() >= GlobalGraphRevision::kInvalidValue - 1U) {
          return Result::failure(graphError(GlobalGraphErrorCode::RevisionOverflow,
                                            "global graph revision space is exhausted"));
        }
        revision = GlobalGraphRevision(parent->value() + 1U);
      }

      const AnchorRecord& reference = referenceAnchor(candidate);
      const auto anchor_estimate =
          std::find_if(anchors.begin(), anchors.end(), [&](const SubmapAnchorEstimate& estimate) {
            return estimate.submap == reference.seal.ref;
          });
      if (anchor_estimate == anchors.end()) {
        return Result::failure(graphError(GlobalGraphErrorCode::SolverFailure,
                                          "selected map-to-odom anchor is missing"));
      }
      const core::Pose3d T_map_odom =
          anchor_estimate->T_map_submap * reference.seal.T_odom_submap.inverse();
      const core::Matrix6d adjoint = reference.seal.T_odom_submap.Adj();
      core::PoseCovariance map_odom_covariance;
      map_odom_covariance.matrix =
          adjoint * anchor_estimate->covariance.matrix * adjoint.transpose();
      MapOdomEstimate map_odom{revision, reference.seal.ref, T_map_odom, map_odom_covariance};

      GlobalGraphCheckpoint checkpoint;
      checkpoint.schema_version = kGlobalGraphCheckpointSchemaVersion;
      checkpoint.key_schema_version = kGlobalGraphKeySchemaVersion;
      const GlobalGraphCheckpointLimits checkpoint_limits =
          checkpoint_internal::limitsForConfig(config);
      auto configuration_identity =
          checkpoint_internal::configurationIdentity(config, checkpoint_limits);
      if (!configuration_identity) {
        return Result::failure(graphError(GlobalGraphErrorCode::SolverFailure,
                                          "global configuration identity could not be encoded: " +
                                              configuration_identity.error().detail));
      }
      checkpoint.configuration = std::move(configuration_identity).value();
      checkpoint.mission_session =
          candidate.anchors.at(candidate.anchor_order.front()).seal.ref.session;
      checkpoint.revision = revision;
      checkpoint.parent = parent;
      const auto mission_gauge =
          std::find_if(candidate.known_factors.begin(), candidate.known_factors.end(),
                       [](const KnownFactorRecord& factor) {
                         return factor.kind == KnownFactorKind::MissionGauge;
                       });
      if (mission_gauge == candidate.known_factors.end()) {
        return Result::failure(graphError(GlobalGraphErrorCode::SolverFailure,
                                          "committed graph has no mission gauge factor"));
      }
      const AnchorRecord& first_boundary = candidate.anchors.at(candidate.anchor_order.front());
      checkpoint.mission_gauge = MissionGaugeCheckpoint{
          mission_gauge->id, first_boundary.insertion_index, first_boundary.seal.T_odom_submap,
          config.mission_gauge_translation_sigma_m, config.mission_gauge_rotation_sigma_rad};
      checkpoint.chart_placements.reserve(candidate.chart_placements.size());
      for (const auto& [epoch, placement] : candidate.chart_placements) {
        checkpoint.chart_placements.push_back(OdomEpochChartPlacementCheckpoint{epoch, placement});
      }
      checkpoint.boundaries.reserve(candidate.anchor_order.size());
      for (const AnchorIdentityKey& identity : candidate.anchor_order) {
        const AnchorRecord& boundary = candidate.anchors.at(identity);
        BoundaryNavigationCheckpoint saved;
        saved.slot = boundary.insertion_index;
        saved.seal = boundary.seal;
        saved.T_map_submap = fromGtsamPose(optimized.at<gtsam::Pose3>(boundary.keys.anchor));
        if (boundary.navigation_materialized) {
          saved.velocity_map = optimized.at<gtsam::Vector3>(boundary.keys.velocity);
          saved.gyro_bias = optimized.at<gtsam::Vector3>(boundary.keys.gyro_bias);
          saved.accel_bias = optimized.at<gtsam::Vector3>(boundary.keys.accel_bias);
        }
        checkpoint.boundaries.push_back(std::move(saved));
      }
      checkpoint.adjacent_factors.reserve(candidate.adjacent_factors);
      checkpoint.gnss_factors.reserve(candidate.gnss_factors);
      for (const KnownFactorRecord& factor : candidate.known_factors) {
        if (factor.adjacent) {
          checkpoint.adjacent_factors.push_back(*factor.adjacent);
        }
        if (factor.gnss) {
          checkpoint.gnss_factors.push_back(GnssFactorCheckpoint{factor.id, *factor.gnss});
        }
      }
      checkpoint.loop_factors.reserve(candidate.loop_factors.size());
      for (const LoopFactorRecord& loop : candidate.loop_factors) {
        checkpoint.loop_factors.push_back(
            LoopFactorCheckpoint{loop.factor_id, loop.measurement, loop.candidate, loop.scale});
      }
      checkpoint.factor_order.reserve(candidate.known_factors.size() +
                                      candidate.loop_factors.size());
      for (const KnownFactorRecord& factor : candidate.known_factors) {
        checkpoint.factor_order.push_back(factor.id);
      }
      for (const LoopFactorRecord& loop : candidate.loop_factors) {
        checkpoint.factor_order.push_back(loop.factor_id);
      }
      checkpoint.alignment = alignment;
      checkpoint.next_boundary_slot = candidate.next_anchor_index;
      checkpoint.next_factor_id = candidate.next_factor_id;
      checkpoint.next_candidate_id = candidate.next_candidate_id;

      checkpoint.recovery.whitened_squared_objective = 2.0 * candidate.factors.error(optimized);
      checkpoint.recovery.gradient_infinity_norm =
          hessian_and_gradient.second.size() == 0
              ? 0.0
              : hessian_and_gradient.second.cwiseAbs().maxCoeff();
      checkpoint.recovery.scalar_dimension = report.scalar_dimension;
      checkpoint.recovery.numerical_rank = report.numerical_rank;
      checkpoint.recovery.hessian_condition = report.hessian_condition;
      checkpoint.recovery.factor_objectives.reserve(checkpoint.factor_order.size());
      for (std::size_t index = 0U; index < checkpoint.factor_order.size(); ++index) {
        checkpoint.recovery.factor_objectives.push_back(GlobalFactorObjectiveCheckpoint{
            checkpoint.factor_order[index], 2.0 * candidate.factors.at(index)->error(optimized)});
      }
      checkpoint.recovery.boundary_marginals.reserve(anchors.size());
      for (std::size_t index = 0U; index < anchors.size(); ++index) {
        checkpoint.recovery.boundary_marginals.push_back(
            BoundaryMarginalCheckpoint{index, anchors[index].covariance});
      }
      checkpoint.recovery.alignment_covariance = alignment_covariance;
      checkpoint.recovery.map_odom = map_odom;
      checkpoint.recovery.committed_solve = report;
      checkpoint.recovery.tolerances = checkpoint_internal::recoveryTolerances(config);
      auto finalized_checkpoint =
          checkpoint_internal::finalize(std::move(checkpoint), checkpoint_limits);
      if (!finalized_checkpoint) {
        return Result::failure(graphError(GlobalGraphErrorCode::SolverFailure,
                                          "canonical recovery checkpoint validation failed: " +
                                              finalized_checkpoint.error().detail));
      }
      checkpoint = std::move(finalized_checkpoint).value();

      GlobalGraphCommit commit{revision,
                               parent,
                               std::move(anchors),
                               std::move(alignment),
                               std::move(alignment_covariance),
                               std::move(map_odom),
                               report};
      // Complete every potentially allocating output copy while the committed
      // state is still untouched. The following state/optional moves only
      // install the already validated shadow.
      GlobalGraphCommit returned_commit = commit;
      candidate.values = std::move(optimized);
      state = std::move(candidate);
      latest_commit = std::move(commit);
      latest_checkpoint = std::move(checkpoint);
      return Result::success(std::move(returned_commit));
    } catch (const gtsam::IndeterminantLinearSystemException& exception) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::RankDeficientCandidate,
                     std::string("indeterminate shadow graph: ") + exception.what()));
    } catch (const std::exception& exception) {
      return Result::failure(graphError(GlobalGraphErrorCode::SolverFailure,
                                        std::string("shadow solve failed: ") + exception.what()));
    }
  }

  GlobalGraphConfig config;
  State state;
  std::optional<GlobalGraphCommit> latest_commit;
  std::optional<GlobalGraphCheckpoint> latest_checkpoint;
};

GlobalGraph::GlobalGraph(GlobalGraphConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

GlobalGraph::~GlobalGraph() = default;
GlobalGraph::GlobalGraph(GlobalGraph&&) noexcept = default;
GlobalGraph& GlobalGraph::operator=(GlobalGraph&&) noexcept = default;

core::Result<GlobalGraphCommit, GlobalGraphError> GlobalGraph::initializeMission(
    core::SparseSubmapSeal first_seal) {
  using Result = core::Result<GlobalGraphCommit, GlobalGraphError>;
  if (impl_->latest_commit) {
    return Result::failure(graphError(GlobalGraphErrorCode::AlreadyInitialized,
                                      "the mission gauge can be created only once"));
  }
  if (!validConfig(impl_->config)) {
    return Result::failure(graphError(
        GlobalGraphErrorCode::InvalidConfig,
        "global graph capacities, solver limits, gauge sigmas, and validation tolerances "
        "must be finite and positive"));
  }
  auto verified_seal = core::verifyCanonicalSparseSubmapSeal(first_seal);
  if (!verified_seal) {
    GlobalGraphError failure =
        graphError(GlobalGraphErrorCode::InvalidSparseSeal,
                   "first mission seal failed recursive canonical admission");
    failure.canonical_verification_error = std::move(verified_seal).error();
    return Result::failure(std::move(failure));
  }
  if (!validPose(first_seal.T_odom_submap) || first_seal.previous || first_seal.from_previous) {
    return Result::failure(
        graphError(GlobalGraphErrorCode::InvalidSparseSeal,
                   "first mission seal is invalid or incorrectly carries an incoming transition"));
  }

  Impl::State candidate = impl_->state;
  const Impl::BoundaryKeys keys = impl_->nextBoundaryKeys(candidate);
  const gtsam::Pose3 gauge_pose = toGtsamPose(first_seal.T_odom_submap);
  const gtsam::Vector6 gauge_sigmas =
      (gtsam::Vector6() << impl_->config.mission_gauge_rotation_sigma_rad,
       impl_->config.mission_gauge_rotation_sigma_rad,
       impl_->config.mission_gauge_rotation_sigma_rad,
       impl_->config.mission_gauge_translation_sigma_m,
       impl_->config.mission_gauge_translation_sigma_m,
       impl_->config.mission_gauge_translation_sigma_m)
          .finished();
  auto gauge_id = impl_->allocateFactorId(candidate);
  if (!gauge_id) {
    return Result::failure(std::move(gauge_id).error());
  }
  const auto gauge_factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      keys.anchor, gauge_pose, gtsam::noiseModel::Diagonal::Sigmas(gauge_sigmas));
  candidate.factors.push_back(gauge_factor);
  candidate.known_factors.push_back(
      Impl::KnownFactorRecord{std::move(gauge_id).value(), Impl::KnownFactorKind::MissionGauge,
                              std::nullopt, gauge_factor, std::nullopt, std::nullopt});
  candidate.values.insert(keys.anchor, gauge_pose);
  candidate.chart_placements.emplace(first_seal.ref.odom_epoch, core::Pose3d{});
  const AnchorIdentityKey first_identity = core::sparseSubmapIdentityKey(first_seal.ref);
  candidate.anchor_order.push_back(first_identity);
  const core::Pose3d C_submap_imu =
      first_seal.T_odom_submap.inverse() * first_seal.boundary_navigation.T_odom_imu;
  candidate.anchors.emplace(first_identity,
                            Impl::AnchorRecord{std::move(first_seal), C_submap_imu, keys, false,
                                               candidate.next_anchor_index});
  ++candidate.next_anchor_index;
  return impl_->optimizeAndCommit(std::move(candidate),
                                  GlobalTransactionKind::MissionInitialization);
}

core::Result<GlobalGraphCommit, GlobalGraphError> GlobalGraph::appendAdjacent(
    core::SparseSubmapSeal current_seal) {
  std::vector<core::SparseSubmapSeal> batch;
  batch.reserve(1U);
  batch.push_back(std::move(current_seal));
  return appendAdjacentBatch(std::move(batch));
}

core::Result<GlobalGraphCommit, GlobalGraphError> GlobalGraph::appendAdjacentBatch(
    std::vector<core::SparseSubmapSeal> consecutive_seals) {
  using Result = core::Result<GlobalGraphCommit, GlobalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(graphError(GlobalGraphErrorCode::NotInitialized,
                                      "initialize the mission gauge before adding anchors"));
  }
  if (consecutive_seals.empty()) {
    return Result::failure(graphError(GlobalGraphErrorCode::EmptyAdjacentBatch,
                                      "adjacent transaction requires at least one seal"));
  }
  if (consecutive_seals.size() > impl_->config.maximum_adjacent_seals_per_transaction) {
    return Result::failure(graphError(GlobalGraphErrorCode::AdjacentBatchCapacity,
                                      "adjacent seal batch exceeds its transaction bound"));
  }
  Impl::State candidate = impl_->state;
  for (core::SparseSubmapSeal& current_seal : consecutive_seals) {
    auto verified_seal = core::verifyCanonicalSparseSubmapSeal(current_seal);
    if (!verified_seal) {
      GlobalGraphError failure = graphError(GlobalGraphErrorCode::InvalidSparseSeal,
                                            "adjacent seal failed recursive canonical admission");
      failure.canonical_verification_error = std::move(verified_seal).error();
      return Result::failure(std::move(failure));
    }
    if (!validPose(current_seal.T_odom_submap) || !current_seal.previous ||
        !current_seal.from_previous) {
      return Result::failure(graphError(
          GlobalGraphErrorCode::InvalidSparseSeal,
          "adjacent insertion requires a valid core seal with an exact incoming transition"));
    }
    const AnchorIdentityKey appended_identity = core::sparseSubmapIdentityKey(current_seal.ref);
    if (candidate.anchors.contains(appended_identity)) {
      return Result::failure(graphError(GlobalGraphErrorCode::DuplicateSubmap,
                                        "submap identity already exists in the graph shadow"));
    }
    const AnchorIdentityKey from_identity = core::sparseSubmapIdentityKey(*current_seal.previous);
    const auto from = candidate.anchors.find(from_identity);
    if (from == candidate.anchors.end()) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::DisconnectedProposal,
                     "adjacent chain does not start at a mission-connected predecessor"));
    }
    if (candidate.anchor_order.empty() || candidate.anchor_order.back() != from_identity ||
        current_seal.ref.id.value() <= from->second.seal.ref.id.value()) {
      return Result::failure(graphError(
          GlobalGraphErrorCode::NonConsecutiveAdjacentBatch,
          "adjacent batch is not ordered as one strict extension of the current chain head"));
    }
    if (from->second.seal.ref != *current_seal.previous) {
      return Result::failure(graphError(GlobalGraphErrorCode::StaleSubmapReference,
                                        "incoming seal pins a stale or altered predecessor ref"));
    }
    if (core::validateSparseSubmapLink(from->second.seal, current_seal) !=
        core::SparseSubmapLinkValidationError::None) {
      return Result::failure(graphError(GlobalGraphErrorCode::InvalidSparseLink,
                                        "incoming seal does not exactly extend its predecessor"));
    }
    if (candidate.anchors.size() >= impl_->config.maximum_anchors) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::AnchorCapacity, "global anchor hard cap reached"));
    }
    if (candidate.adjacent_factors >= impl_->config.maximum_adjacent_factors) {
      return Result::failure(graphError(GlobalGraphErrorCode::AdjacentFactorCapacity,
                                        "adjacent-factor hard cap reached"));
    }

    const core::FrozenSquareRootFactor& frozen =
        current_seal.from_previous->local_transition.boundary_factor;
    const std::size_t frozen_rows = frozen.rows;
    const std::size_t frozen_coefficients = frozen.row_major_A.size();
    if (frozen_rows > impl_->config.maximum_adjacent_factor_rows ||
        frozen_coefficients > impl_->config.maximum_adjacent_factor_coefficients ||
        frozen.rhs.size() > impl_->config.maximum_adjacent_factor_rows ||
        frozen_rows > impl_->config.maximum_total_adjacent_factor_rows ||
        frozen_coefficients > impl_->config.maximum_total_adjacent_factor_coefficients ||
        candidate.total_adjacent_rows >
            impl_->config.maximum_total_adjacent_factor_rows - frozen_rows ||
        candidate.total_adjacent_coefficients >
            impl_->config.maximum_total_adjacent_factor_coefficients - frozen_coefficients) {
      return Result::failure(graphError(GlobalGraphErrorCode::AdjacentFactorCapacity,
                                        "adjacent frozen rows exceed a per-factor or graph cap"));
    }
    const std::size_t additional_scalar_dimension =
        15U + (from->second.navigation_materialized ? 0U : 9U);
    const std::size_t current_scalar_dimension = 6U * candidate.anchors.size() +
                                                 9U * candidate.materialized_navigation_boundaries +
                                                 (candidate.alignment_exists ? 4U : 0U);
    if (current_scalar_dimension > impl_->config.maximum_scalar_dimension ||
        additional_scalar_dimension >
            impl_->config.maximum_scalar_dimension - current_scalar_dimension) {
      return Result::failure(graphError(GlobalGraphErrorCode::ScalarDimensionCapacity,
                                        "adjacent insertion exceeds the scalar-dimension cap"));
    }

    Impl::AnchorRecord& from_boundary = candidate.anchors.at(from_identity);
    const auto placement_it = candidate.chart_placements.find(current_seal.ref.odom_epoch);
    if (placement_it == candidate.chart_placements.end()) {
      return Result::failure(graphError(GlobalGraphErrorCode::InvalidSparseLink,
                                        "predecessor odom epoch has no immutable chart placement"));
    }
    const core::Pose3d& H_map_odom = placement_it->second;
    if (!from_boundary.navigation_materialized) {
      from_boundary.navigation_materialized = true;
      candidate.values.insert(
          from_boundary.keys.velocity,
          (H_map_odom.so3().matrix() * from_boundary.seal.boundary_navigation.velocity_odom)
              .eval());
      candidate.values.insert(from_boundary.keys.gyro_bias,
                              from_boundary.seal.boundary_navigation.gyro_bias);
      candidate.values.insert(from_boundary.keys.accel_bias,
                              from_boundary.seal.boundary_navigation.accel_bias);
      ++candidate.materialized_navigation_boundaries;
    }

    const Impl::BoundaryKeys to_keys = impl_->nextBoundaryKeys(candidate);
    const core::Pose3d to_C =
        current_seal.T_odom_submap.inverse() * current_seal.boundary_navigation.T_odom_imu;
    const core::Pose3d from_anchor =
        fromGtsamPose(candidate.values.at<gtsam::Pose3>(from_boundary.keys.anchor));
    const core::Pose3d from_center = H_map_odom * from_boundary.seal.boundary_navigation.T_odom_imu;
    const Eigen::Matrix<double, 6, 1> pose_delta =
        (from_center.inverse() * from_anchor * from_boundary.C_submap_imu).log();
    const core::Pose3d to_anchor = H_map_odom * current_seal.boundary_navigation.T_odom_imu *
                                   core::Pose3d::exp(pose_delta) * to_C.inverse();
    const Eigen::Vector3d velocity_delta =
        H_map_odom.so3().matrix().transpose() *
            candidate.values.at<gtsam::Vector3>(from_boundary.keys.velocity) -
        from_boundary.seal.boundary_navigation.velocity_odom;
    const Eigen::Vector3d gyro_delta =
        candidate.values.at<gtsam::Vector3>(from_boundary.keys.gyro_bias) -
        from_boundary.seal.boundary_navigation.gyro_bias;
    const Eigen::Vector3d accel_delta =
        candidate.values.at<gtsam::Vector3>(from_boundary.keys.accel_bias) -
        from_boundary.seal.boundary_navigation.accel_bias;
    candidate.values.insert(to_keys.anchor, toGtsamPose(to_anchor));
    candidate.values.insert(to_keys.velocity,
                            (H_map_odom.so3().matrix() *
                             (current_seal.boundary_navigation.velocity_odom + velocity_delta))
                                .eval());
    candidate.values.insert(to_keys.gyro_bias,
                            (current_seal.boundary_navigation.gyro_bias + gyro_delta).eval());
    candidate.values.insert(to_keys.accel_bias,
                            (current_seal.boundary_navigation.accel_bias + accel_delta).eval());

    adjacent_internal::AdjacentBoundaryKeys factor_keys{from_boundary.keys.anchor,
                                                        from_boundary.keys.velocity,
                                                        from_boundary.keys.gyro_bias,
                                                        from_boundary.keys.accel_bias,
                                                        to_keys.anchor,
                                                        to_keys.velocity,
                                                        to_keys.gyro_bias,
                                                        to_keys.accel_bias};
    adjacent_internal::AdjacentBoundaryFactorLimits factor_limits{
        impl_->config.maximum_adjacent_factor_rows,
        impl_->config.maximum_adjacent_factor_coefficients};
    const adjacent_internal::OdomEpochChartPlacement chart{current_seal.ref.odom_epoch, H_map_odom};
    auto exact_factor = adjacent_internal::makeAdjacentBoundaryFactor(
        from_boundary.seal, current_seal, chart, factor_keys, factor_limits);
    if (!exact_factor) {
      return Result::failure(graphError(
          GlobalGraphErrorCode::InvalidSparseLink,
          "exact adjacent factor rejected the sealed transition: " + exact_factor.error().detail));
    }
    auto factor_id = impl_->allocateFactorId(candidate);
    if (!factor_id) {
      return Result::failure(std::move(factor_id).error());
    }
    const GlobalFactorId committed_factor_id = factor_id.value();
    candidate.factors.push_back(exact_factor.value());
    AdjacentBoundaryCheckpoint adjacent_checkpoint{
        committed_factor_id, from_boundary.insertion_index, candidate.next_anchor_index,
        *current_seal.from_previous};
    candidate.known_factors.push_back(
        Impl::KnownFactorRecord{committed_factor_id, Impl::KnownFactorKind::Adjacent,
                                impl_->adjacentMonitoringScale(frozen), exact_factor.value(),
                                std::move(adjacent_checkpoint), std::nullopt});
    candidate.anchor_order.push_back(appended_identity);
    candidate.anchors.emplace(appended_identity,
                              Impl::AnchorRecord{std::move(current_seal), to_C, to_keys, true,
                                                 candidate.next_anchor_index});
    const std::array<gtsam::Key, 7U> connected_keys{from_boundary.keys.velocity,
                                                    from_boundary.keys.gyro_bias,
                                                    from_boundary.keys.accel_bias,
                                                    to_keys.anchor,
                                                    to_keys.velocity,
                                                    to_keys.gyro_bias,
                                                    to_keys.accel_bias};
    for (const gtsam::Key key : connected_keys) {
      candidate.connectivity_edges.emplace_back(from_boundary.keys.anchor, key);
    }
    ++candidate.next_anchor_index;
    ++candidate.adjacent_factors;
    ++candidate.materialized_navigation_boundaries;
    candidate.total_adjacent_rows += frozen_rows;
    candidate.total_adjacent_coefficients += frozen_coefficients;
  }
  return impl_->optimizeAndCommit(std::move(candidate), GlobalTransactionKind::AdjacentInsertion,
                                  consecutive_seals.size());
}

core::Result<GlobalGraphCommit, GlobalGraphError> GlobalGraph::appendGnssBatch(
    GnssBatchAppend append) {
  using Result = core::Result<GlobalGraphCommit, GlobalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(graphError(GlobalGraphErrorCode::NotInitialized,
                                      "initialize the mission gauge before adding GNSS"));
  }
  if (append.constraints.empty()) {
    return Result::failure(graphError(GlobalGraphErrorCode::EmptyGnssBatch,
                                      "GNSS transaction must contain individual fix factors"));
  }
  if (!impl_->state.alignment_exists && !append.initial_alignment) {
    return Result::failure(graphError(
        GlobalGraphErrorCode::AlignmentInitializerRequired,
        "the first GNSS transaction requires an observable four-DoF alignment initializer"));
  }
  if (impl_->state.alignment_exists && append.initial_alignment) {
    return Result::failure(graphError(
        GlobalGraphErrorCode::AlignmentAlreadyInitialized,
        "an established alignment is preserved; a new initializer belongs to reacquisition"));
  }
  if (append.initial_alignment && (!append.initial_alignment->translation_enu.allFinite() ||
                                   !std::isfinite(append.initial_alignment->yaw_enu_map_rad))) {
    return Result::failure(graphError(GlobalGraphErrorCode::InvalidGnssConstraint,
                                      "alignment initializer is non-finite"));
  }
  if (append.constraints.size() > impl_->config.maximum_gnss_factors - impl_->state.gnss_factors) {
    return Result::failure(graphError(GlobalGraphErrorCode::GnssFactorCapacity,
                                      "GNSS-factor hard cap would be exceeded"));
  }

  std::unordered_set<std::uint64_t> batch_observations;
  for (const auto& constraint : append.constraints) {
    if (!validSubmapRef(constraint.submap) || !constraint.observation.valid() ||
        !constraint.antenna_position_submap.allFinite() ||
        !constraint.measured_position_enu.allFinite()) {
      return Result::failure(graphError(
          GlobalGraphErrorCode::InvalidGnssConstraint,
          "GNSS factor identity, submap ref, antenna point, or ENU measurement is invalid"));
    }
    if (!validCovariance(constraint.effective_covariance_enu, impl_->config)) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::InvalidCovariance,
                     "GNSS effective covariance must be finite, symmetric, and positive definite"));
    }
    const AnchorIdentityKey identity = core::sparseSubmapIdentityKey(constraint.submap);
    const auto anchor = impl_->state.anchors.find(identity);
    if (anchor == impl_->state.anchors.end()) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::DisconnectedProposal,
                     "GNSS factor references an anchor outside the connected graph"));
    }
    if (anchor->second.seal.ref != constraint.submap) {
      return Result::failure(graphError(GlobalGraphErrorCode::StaleSubmapReference,
                                        "GNSS factor pins a stale or altered submap ref"));
    }
    if (impl_->state.gnss_observations.contains(constraint.observation.value()) ||
        !batch_observations.insert(constraint.observation.value()).second) {
      return Result::failure(
          graphError(GlobalGraphErrorCode::DuplicateGnssObservation,
                     "each GNSS observation may contribute information exactly once"));
    }
  }

  Impl::State candidate = impl_->state;
  const AlignmentKey alignment_key = Impl::alignmentKey();
  if (!candidate.alignment_exists) {
    const std::size_t current_scalar_dimension =
        6U * candidate.anchors.size() + 9U * candidate.materialized_navigation_boundaries;
    if (current_scalar_dimension > impl_->config.maximum_scalar_dimension ||
        4U > impl_->config.maximum_scalar_dimension - current_scalar_dimension) {
      return Result::failure(graphError(GlobalGraphErrorCode::ScalarDimensionCapacity,
                                        "GNSS alignment exceeds the scalar-dimension cap"));
    }
    gtsam::Vector4 initializer;
    initializer.head<3>() = append.initial_alignment->translation_enu;
    initializer(3) = wrapYaw(append.initial_alignment->yaw_enu_map_rad);
    candidate.values.insert(alignment_key, initializer);
    candidate.alignment_exists = true;
  }
  for (const auto& constraint : append.constraints) {
    const AnchorKey anchor_key =
        candidate.anchors.at(core::sparseSubmapIdentityKey(constraint.submap)).keys.anchor;
    auto factor_id = impl_->allocateFactorId(candidate);
    if (!factor_id) {
      return Result::failure(std::move(factor_id).error());
    }
    const auto gnss_factor = boost::make_shared<GnssAntennaFactor>(
        anchor_key, alignment_key, constraint.antenna_position_submap,
        constraint.measured_position_enu, constraint.effective_covariance_enu);
    candidate.factors.push_back(gnss_factor);
    candidate.known_factors.push_back(
        Impl::KnownFactorRecord{std::move(factor_id).value(), Impl::KnownFactorKind::Gnss,
                                Impl::gnssScale(), gnss_factor, std::nullopt, constraint});
    candidate.connectivity_edges.emplace_back(anchor_key, alignment_key);
    candidate.gnss_observations.insert(constraint.observation.value());
    ++candidate.gnss_factors;
  }
  return impl_->optimizeAndCommit(std::move(candidate), GlobalTransactionKind::GnssInsertion);
}

core::Result<RobustLoopTransactionResult, RobustLoopTransactionError>
GlobalGraph::appendRobustLoopBatch(RobustLoopBatchAppend append,
                                   const LoopConsensusReport& consensus) {
  using Result = core::Result<RobustLoopTransactionResult, RobustLoopTransactionError>;
  if (!impl_->latest_commit) {
    return Result::failure(
        robustLoopError(RobustLoopTransactionErrorCode::NotInitialized,
                        "initialize the mission gauge before evaluating robust loop candidates"));
  }
  if (!append.expected_parent.valid() || append.expected_parent != impl_->latest_commit->revision) {
    return Result::failure(
        robustLoopError(RobustLoopTransactionErrorCode::StaleParentRevision,
                        "loop shadow parent does not equal the current committed graph revision"));
  }
  if (append.candidates.empty()) {
    return Result::failure(
        robustLoopError(RobustLoopTransactionErrorCode::EmptyCandidateBatch,
                        "robust loop transaction requires at least one PCM-admitted candidate"));
  }
  if (append.candidates.size() > impl_->config.maximum_loop_candidates_per_transaction ||
      append.candidates.size() > impl_->config.loop_gnc.maximum_robust_candidates) {
    return Result::failure(
        robustLoopError(RobustLoopTransactionErrorCode::CandidateCapacity,
                        "new loop batch exceeds its deterministic transaction bound"));
  }
  if (impl_->state.loop_factors.size() + append.candidates.size() >
          impl_->config.maximum_loop_factors ||
      impl_->state.loop_factors.size() + append.candidates.size() >
          impl_->config.loop_gnc.maximum_robust_candidates) {
    return Result::failure(
        robustLoopError(RobustLoopTransactionErrorCode::LoopFactorCapacity,
                        "complete active robust-factor shadow would exceed capacity"));
  }
  if (impl_->state.known_factors.size() > impl_->config.loop_gnc.maximum_known_inliers) {
    return Result::failure(robustLoopError(
        RobustLoopTransactionErrorCode::CandidateCapacity,
        "complete known-inlier factor set exceeds the configured GNC shadow bound"));
  }

  std::sort(append.candidates.begin(), append.candidates.end(),
            [](const RobustLoopCandidate& left, const RobustLoopCandidate& right) {
              return left.measurement.proposal < right.measurement.proposal;
            });
  std::unordered_set<std::uint64_t> pcm_decision_ids;
  for (const LoopProposalDecision& decision : consensus.decisions) {
    if (!decision.proposal.valid() || !pcm_decision_ids.insert(decision.proposal.value()).second) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::MissingPcmAdmission,
          "PCM report contains an invalid or duplicate proposal decision", decision.proposal));
    }
  }
  std::unordered_set<std::uint64_t> new_proposals;
  std::optional<core::SessionId> batch_session;
  std::vector<const LoopMeasurement*> validated_new_measurements;
  validated_new_measurements.reserve(append.candidates.size());
  const CoreLoopAncestryApi ancestry_api;
  for (const RobustLoopCandidate& candidate : append.candidates) {
    const ProposalId proposal = candidate.measurement.proposal;
    std::string detail;
    if (!validLoopCandidate(candidate, impl_->config, &detail)) {
      return Result::failure(robustLoopError(RobustLoopTransactionErrorCode::InvalidCandidate,
                                             std::move(detail), proposal));
    }
    const auto pcm_decision = std::find_if(
        consensus.decisions.begin(), consensus.decisions.end(),
        [&](const LoopProposalDecision& decision) { return decision.proposal == proposal; });
    if (pcm_decision == consensus.decisions.end() ||
        pcm_decision->disposition != LoopProposalDisposition::AdmittedToGncCandidateBatch ||
        candidate.measurement.header.created_at > consensus.evaluated_at) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::MissingPcmAdmission,
          "loop candidate lacks a matching non-stale PCM admission decision", proposal));
    }
    if (!batch_session) {
      batch_session = candidate.measurement.header.session;
    } else if (*batch_session != candidate.measurement.header.session) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::InvalidCandidate,
          "one robust loop transaction may contain only one mission session", proposal));
    }
    if (!new_proposals.insert(proposal.value()).second ||
        std::any_of(impl_->state.loop_factors.begin(), impl_->state.loop_factors.end(),
                    [&](const Impl::LoopFactorRecord& factor) {
                      return factor.measurement.proposal == proposal;
                    })) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::DuplicateProposal,
          "loop proposal is duplicated in the batch or already committed", proposal));
    }
    const auto valid_ancestry_with = [&](const LoopMeasurement& other) {
      const LoopAncestryAssessment assessment = ancestry_api.assess(candidate.measurement, other);
      return assessment.independence == AncestryIndependence::Independent &&
             assessment.pair_treatment != PairCorrelationTreatment::Missing;
    };
    if (std::any_of(impl_->state.loop_factors.begin(), impl_->state.loop_factors.end(),
                    [&](const Impl::LoopFactorRecord& factor) {
                      return !valid_ancestry_with(factor.measurement);
                    }) ||
        std::any_of(validated_new_measurements.begin(), validated_new_measurements.end(),
                    [&](const LoopMeasurement* measurement) {
                      return !valid_ancestry_with(*measurement);
                    })) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::NonIndependentLineage,
          "loop lineage overlaps an active candidate or lacks the required pair treatment",
          proposal));
    }
    const AnchorIdentityKey from_identity =
        core::sparseSubmapIdentityKey(candidate.measurement.from);
    const AnchorIdentityKey to_identity = core::sparseSubmapIdentityKey(candidate.measurement.to);
    const auto from = impl_->state.anchors.find(from_identity);
    const auto to = impl_->state.anchors.find(to_identity);
    if (from == impl_->state.anchors.end() || to == impl_->state.anchors.end()) {
      return Result::failure(robustLoopError(RobustLoopTransactionErrorCode::UnknownSubmap,
                                             "loop endpoint is outside the committed anchor graph",
                                             proposal));
    }
    if (from->second.seal.ref != candidate.measurement.from ||
        to->second.seal.ref != candidate.measurement.to) {
      return Result::failure(robustLoopError(
          RobustLoopTransactionErrorCode::StaleSubmapReference,
          "loop proposal pins a stale or altered immutable submap reference", proposal));
    }
    validated_new_measurements.push_back(&candidate.measurement);
  }

  Impl::State shadow = impl_->state;
  const std::size_t active_before = shadow.loop_factors.size();
  for (RobustLoopCandidate& candidate : append.candidates) {
    const ProposalId proposal = candidate.measurement.proposal;
    auto factor_id = impl_->allocateFactorId(shadow);
    const std::optional<CandidateId> candidate_id = Impl::allocateCandidateId(shadow);
    if (!factor_id || !candidate_id) {
      RobustLoopTransactionError failure = robustLoopError(
          RobustLoopTransactionErrorCode::IdentityOverflow,
          "loop transaction exhausted a global factor or GNC candidate identity", proposal);
      if (!factor_id) {
        failure.graph_error = std::move(factor_id).error();
      }
      return Result::failure(std::move(failure));
    }
    const AnchorKey from_key =
        shadow.anchors.at(core::sparseSubmapIdentityKey(candidate.measurement.from)).keys.anchor;
    const AnchorKey to_key =
        shadow.anchors.at(core::sparseSubmapIdentityKey(candidate.measurement.to)).keys.anchor;
    const auto factor = boost::make_shared<RankAwareRelativeFactor>(
        from_key, to_key, candidate.measurement.T_from_to, candidate.measurement.information);
    shadow.loop_factors.push_back(
        Impl::LoopFactorRecord{std::move(factor_id).value(), std::move(candidate.measurement),
                               *candidate_id, candidate.scale, from_key, to_key, factor});
  }
  std::sort(shadow.loop_factors.begin(), shadow.loop_factors.end(),
            [](const Impl::LoopFactorRecord& left, const Impl::LoopFactorRecord& right) {
              return left.candidate < right.candidate;
            });
  impl_->rebuildFactors(shadow);

  std::vector<KnownInlierFactor> known_inliers;
  known_inliers.reserve(shadow.known_factors.size());
  for (const Impl::KnownFactorRecord& factor : shadow.known_factors) {
    known_inliers.push_back(KnownInlierFactor{factor.id, factor.scale});
  }
  std::vector<RobustCandidateFactor> robust_candidates;
  robust_candidates.reserve(shadow.loop_factors.size());
  for (const Impl::LoopFactorRecord& factor : shadow.loop_factors) {
    robust_candidates.push_back(RobustCandidateFactor{factor.candidate, factor.scale});
  }

  Impl::LoopShadowSolver shadow_solver(impl_->config, shadow);
  GncTlsController controller(impl_->config.loop_gnc);
  auto classified = controller.run(known_inliers, robust_candidates, shadow_solver);
  if (!classified) {
    RobustLoopTransactionError error =
        robustLoopError(RobustLoopTransactionErrorCode::GncTlsFailure,
                        "complete robust-loop shadow failed GNC-TLS: " + classified.error().detail);
    error.gnc_error = classified.error();
    return Result::failure(std::move(error));
  }

  RobustLoopTransactionReport report;
  report.evaluated_parent = append.expected_parent;
  report.pcm_evaluated_at = consensus.evaluated_at;
  report.pcm_admitted_candidates = append.candidates.size();
  report.gnc = std::move(classified).value();
  report.validation.known_inlier_factors = shadow.known_factors.size();
  for (const Impl::KnownFactorRecord& factor : shadow.known_factors) {
    switch (factor.kind) {
      case Impl::KnownFactorKind::MissionGauge:
        ++report.validation.mission_gauge_known_inliers;
        break;
      case Impl::KnownFactorKind::Adjacent:
        ++report.validation.adjacent_known_inliers;
        break;
      case Impl::KnownFactorKind::Gnss:
        ++report.validation.gnss_known_inliers;
        break;
    }
  }
  report.validation.active_loop_factors_before = active_before;
  report.validation.evaluated_loop_candidates = shadow.loop_factors.size();
  report.validation.complete_known_inlier_set =
      report.gnc.known_inliers.size() == shadow.known_factors.size();
  report.validation.known_inlier_weights_fixed_to_one =
      std::all_of(report.gnc.known_inliers.begin(), report.gnc.known_inliers.end(),
                  [](const GncTlsKnownInlierReport& factor) { return factor.weight == 1.0; });
  for (const GncTlsKnownInlierReport& factor : report.gnc.known_inliers) {
    if (factor.normalized_squared_cost) {
      report.validation.maximum_known_inlier_normalized_squared_cost =
          std::max(report.validation.maximum_known_inlier_normalized_squared_cost,
                   *factor.normalized_squared_cost);
    }
  }

  report.decisions.reserve(report.gnc.robust_candidates.size());
  for (const GncTlsCandidateReport& classified_factor : report.gnc.robust_candidates) {
    const auto record = std::lower_bound(
        shadow.loop_factors.begin(), shadow.loop_factors.end(), classified_factor.candidate_id,
        [](const Impl::LoopFactorRecord& factor, CandidateId candidate) {
          return factor.candidate < candidate;
        });
    if (record == shadow.loop_factors.end() ||
        record->candidate != classified_factor.candidate_id) {
      RobustLoopTransactionError error =
          robustLoopError(RobustLoopTransactionErrorCode::GncTlsFailure,
                          "GNC-TLS report cannot be joined to the complete shadow candidate set");
      error.gnc_error =
          GncTlsError{GncTlsErrorCode::IncompleteShadowCosts, std::nullopt,
                      classified_factor.candidate_id, std::nullopt, "candidate join failed"};
      return Result::failure(std::move(error));
    }
    const bool previously_committed =
        new_proposals.find(record->measurement.proposal.value()) == new_proposals.end();
    RobustLoopDecision decision;
    decision.proposal = record->measurement.proposal;
    decision.candidate = classified_factor.candidate_id;
    decision.previously_committed = previously_committed;
    decision.cost = classified_factor.cost;
    decision.final_weight = classified_factor.final_weight;
    decision.tls_disposition = classified_factor.disposition;
    if (!previously_committed &&
        classified_factor.disposition == GncTlsCandidateDisposition::TlsInlier) {
      ++report.validation.new_tls_inliers;
      decision.application = RobustLoopApplicationDisposition::TlsInlierNotCommitted;
    } else if (!previously_committed) {
      ++report.validation.new_tls_outliers;
      decision.application = RobustLoopApplicationDisposition::RejectedNewTlsOutlier;
    } else {
      decision.application =
          RobustLoopApplicationDisposition::RetainedCommittedDespiteRejectedTransaction;
    }
    report.decisions.push_back(std::move(decision));
  }

  if (report.validation.new_tls_inliers == 0U) {
    report.outcome = RobustLoopTransactionOutcome::NoNewTlsInlier;
    report.validation.active_loop_factors_after = active_before;
    return Result::success(RobustLoopTransactionResult{std::move(report), std::nullopt});
  }

  std::unordered_set<std::uint64_t> tls_inliers;
  for (const GncTlsCandidateReport& factor : report.gnc.robust_candidates) {
    if (factor.disposition == GncTlsCandidateDisposition::TlsInlier) {
      tls_inliers.insert(factor.candidate_id.value());
    }
  }
  shadow.loop_factors.erase(std::remove_if(shadow.loop_factors.begin(), shadow.loop_factors.end(),
                                           [&](const Impl::LoopFactorRecord& factor) {
                                             return !tls_inliers.contains(factor.candidate.value());
                                           }),
                            shadow.loop_factors.end());
  shadow.values = shadow_solver.workingValues();
  impl_->rebuildFactors(shadow);

  for (RobustLoopDecision& decision : report.decisions) {
    if (decision.tls_disposition == GncTlsCandidateDisposition::TlsInlier) {
      decision.application = decision.previously_committed
                                 ? RobustLoopApplicationDisposition::RetainedCommittedTlsInlier
                                 : RobustLoopApplicationDisposition::CommittedNewTlsInlier;
    } else if (decision.previously_committed) {
      decision.application = RobustLoopApplicationDisposition::RemovedCommittedTlsOutlier;
      ++report.validation.removed_committed_outliers;
    }
  }
  report.validation.active_loop_factors_after = shadow.loop_factors.size();
  auto committed =
      impl_->optimizeAndCommit(std::move(shadow), GlobalTransactionKind::RobustLoopInsertion);
  if (!committed) {
    for (RobustLoopDecision& decision : report.decisions) {
      if (decision.previously_committed) {
        decision.application =
            RobustLoopApplicationDisposition::RetainedCommittedDespiteRejectedTransaction;
      } else if (decision.tls_disposition == GncTlsCandidateDisposition::TlsInlier) {
        decision.application = RobustLoopApplicationDisposition::TlsInlierNotCommitted;
      }
    }
    report.outcome = RobustLoopTransactionOutcome::RejectedFinalGraphValidation;
    report.validation.removed_committed_outliers = 0U;
    report.validation.active_loop_factors_after = active_before;
    RobustLoopTransactionError error =
        robustLoopError(RobustLoopTransactionErrorCode::FinalGraphValidationFailure,
                        "TLS inlier graph failed the ordinary complete graph commit validation: " +
                            committed.error().detail);
    error.graph_error = committed.error();
    error.shadow_report = std::move(report);
    return Result::failure(std::move(error));
  }

  report.outcome = RobustLoopTransactionOutcome::Committed;
  report.validation.final_complete_graph_validation_passed = true;
  return Result::success(
      RobustLoopTransactionResult{std::move(report), std::move(committed).value()});
}

core::Result<GlobalGraphCommit, GlobalGraphCheckpointError> GlobalGraph::restoreCheckpoint(
    const GlobalGraphCheckpoint& checkpoint) {
  using Result = core::Result<GlobalGraphCommit, GlobalGraphCheckpointError>;
  if (impl_->latest_commit) {
    return Result::failure(
        recoveryError(GlobalGraphCheckpointErrorCode::AlreadyInitialized,
                      "a recovery checkpoint can be installed only into an empty global graph"));
  }
  if (!validConfig(impl_->config)) {
    return Result::failure(recoveryError(GlobalGraphCheckpointErrorCode::ConfigurationMismatch,
                                         "global graph recovery configuration is invalid"));
  }
  const GlobalGraphCheckpointLimits limits = checkpoint_internal::limitsForConfig(impl_->config);
  auto verified = checkpoint_internal::verify(checkpoint, limits);
  if (!verified) {
    return Result::failure(std::move(verified).error());
  }
  auto configuration = checkpoint_internal::configurationIdentity(impl_->config, limits);
  if (!configuration) {
    return Result::failure(std::move(configuration).error());
  }
  if (configuration.value().schema_version != checkpoint.configuration.schema_version ||
      configuration.value().checksum != checkpoint.configuration.checksum ||
      checkpoint.mission_gauge.translation_sigma_m !=
          impl_->config.mission_gauge_translation_sigma_m ||
      checkpoint.mission_gauge.rotation_sigma_rad !=
          impl_->config.mission_gauge_rotation_sigma_rad ||
      !exactRecoveryTolerances(checkpoint.recovery.tolerances,
                               checkpoint_internal::recoveryTolerances(impl_->config))) {
    return Result::failure(recoveryError(
        GlobalGraphCheckpointErrorCode::ConfigurationMismatch,
        "checkpoint configuration, mission gauge, or canonical recovery tolerances differ"));
  }

  try {
    Impl::State shadow;
    for (const OdomEpochChartPlacementCheckpoint& placement : checkpoint.chart_placements) {
      shadow.chart_placements.emplace(placement.odom_epoch, placement.H_map_odom);
    }

    std::vector<core::SparseSubmapIdentityKey> slot_identities;
    slot_identities.reserve(checkpoint.boundaries.size());
    for (const BoundaryNavigationCheckpoint& boundary : checkpoint.boundaries) {
      const Impl::BoundaryKeys keys = impl_->nextBoundaryKeys(shadow);
      shadow.values.insert(keys.anchor, toGtsamPose(boundary.T_map_submap));
      const bool navigation_materialized = boundary.velocity_map.has_value();
      if (navigation_materialized) {
        shadow.values.insert(keys.velocity, *boundary.velocity_map);
        shadow.values.insert(keys.gyro_bias, *boundary.gyro_bias);
        shadow.values.insert(keys.accel_bias, *boundary.accel_bias);
        ++shadow.materialized_navigation_boundaries;
      }
      const core::Pose3d C_submap_imu =
          boundary.seal.T_odom_submap.inverse() * boundary.seal.boundary_navigation.T_odom_imu;
      const core::SparseSubmapIdentityKey identity =
          core::sparseSubmapIdentityKey(boundary.seal.ref);
      slot_identities.push_back(identity);
      shadow.anchor_order.push_back(identity);
      shadow.anchors.emplace(
          identity, Impl::AnchorRecord{boundary.seal, C_submap_imu, keys, navigation_materialized,
                                       static_cast<std::size_t>(boundary.slot)});
      ++shadow.next_anchor_index;
    }

    const Impl::AnchorRecord& first = shadow.anchors.at(slot_identities.front());
    const gtsam::Vector6 gauge_sigmas =
        (gtsam::Vector6() << checkpoint.mission_gauge.rotation_sigma_rad,
         checkpoint.mission_gauge.rotation_sigma_rad, checkpoint.mission_gauge.rotation_sigma_rad,
         checkpoint.mission_gauge.translation_sigma_m, checkpoint.mission_gauge.translation_sigma_m,
         checkpoint.mission_gauge.translation_sigma_m)
            .finished();
    const auto gauge_factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(
        first.keys.anchor, toGtsamPose(checkpoint.mission_gauge.T_map_submap),
        gtsam::noiseModel::Diagonal::Sigmas(gauge_sigmas));
    shadow.known_factors.push_back(Impl::KnownFactorRecord{
        checkpoint.mission_gauge.factor, Impl::KnownFactorKind::MissionGauge, std::nullopt,
        gauge_factor, std::nullopt, std::nullopt});

    for (const AdjacentBoundaryCheckpoint& saved : checkpoint.adjacent_factors) {
      Impl::AnchorRecord& from = shadow.anchors.at(slot_identities.at(saved.from_slot));
      Impl::AnchorRecord& to = shadow.anchors.at(slot_identities.at(saved.to_slot));
      const auto placement = shadow.chart_placements.find(to.seal.ref.odom_epoch);
      if (placement == shadow.chart_placements.end()) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                          "adjacent recovery factor has no immutable odom chart placement"));
      }
      // Keep the explicit eight-variable key order visually aligned with the
      // frozen factor layout; no key is recovered from a serialized GTSAM key.
      const adjacent_internal::AdjacentBoundaryKeys factor_keys{
          from.keys.anchor, from.keys.velocity, from.keys.gyro_bias, from.keys.accel_bias,
          to.keys.anchor,   to.keys.velocity,   to.keys.gyro_bias,   to.keys.accel_bias};
      const adjacent_internal::AdjacentBoundaryFactorLimits factor_limits{
          impl_->config.maximum_adjacent_factor_rows,
          impl_->config.maximum_adjacent_factor_coefficients};
      const adjacent_internal::OdomEpochChartPlacement chart{to.seal.ref.odom_epoch,
                                                             placement->second};
      auto factor = adjacent_internal::makeAdjacentBoundaryFactor(from.seal, to.seal, chart,
                                                                  factor_keys, factor_limits);
      if (!factor) {
        return Result::failure(recoveryError(
            GlobalGraphCheckpointErrorCode::ReconstructionFailure,
            "adjacent recovery factor could not be reconstructed: " + factor.error().detail));
      }
      const core::FrozenSquareRootFactor& frozen =
          to.seal.from_previous->local_transition.boundary_factor;
      shadow.known_factors.push_back(Impl::KnownFactorRecord{
          saved.factor, Impl::KnownFactorKind::Adjacent, impl_->adjacentMonitoringScale(frozen),
          factor.value(), saved, std::nullopt});
      const std::array<gtsam::Key, 7U> connected_keys{
          from.keys.velocity, from.keys.gyro_bias, from.keys.accel_bias, to.keys.anchor,
          to.keys.velocity,   to.keys.gyro_bias,   to.keys.accel_bias};
      for (gtsam::Key key : connected_keys) {
        shadow.connectivity_edges.emplace_back(from.keys.anchor, key);
      }
      ++shadow.adjacent_factors;
      shadow.total_adjacent_rows += frozen.rows;
      shadow.total_adjacent_coefficients += frozen.row_major_A.size();
    }

    if (checkpoint.alignment) {
      gtsam::Vector4 alignment;
      alignment.head<3>() = checkpoint.alignment->translation_enu;
      alignment(3) = checkpoint.alignment->yaw_enu_map_rad;
      shadow.values.insert(Impl::alignmentKey(), alignment);
      shadow.alignment_exists = true;
    }
    for (const GnssFactorCheckpoint& saved : checkpoint.gnss_factors) {
      const auto anchor =
          shadow.anchors.find(core::sparseSubmapIdentityKey(saved.constraint.submap));
      if (anchor == shadow.anchors.end() || anchor->second.seal.ref != saved.constraint.submap) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                          "GNSS recovery factor references an absent or stale anchor"));
      }
      const auto factor = boost::make_shared<GnssAntennaFactor>(
          anchor->second.keys.anchor, Impl::alignmentKey(),
          saved.constraint.antenna_position_submap, saved.constraint.measured_position_enu,
          saved.constraint.effective_covariance_enu);
      shadow.known_factors.push_back(
          Impl::KnownFactorRecord{saved.factor, Impl::KnownFactorKind::Gnss, Impl::gnssScale(),
                                  factor, std::nullopt, saved.constraint});
      shadow.connectivity_edges.emplace_back(anchor->second.keys.anchor, Impl::alignmentKey());
      shadow.gnss_observations.insert(saved.constraint.observation.value());
      ++shadow.gnss_factors;
    }

    for (const LoopFactorCheckpoint& saved : checkpoint.loop_factors) {
      std::string detail;
      if (!validLoopCandidate(RobustLoopCandidate{saved.measurement, saved.scale}, impl_->config,
                              &detail)) {
        return Result::failure(recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                                             "loop recovery factor is invalid: " + detail));
      }
      const auto from = shadow.anchors.find(core::sparseSubmapIdentityKey(saved.measurement.from));
      const auto to = shadow.anchors.find(core::sparseSubmapIdentityKey(saved.measurement.to));
      if (from == shadow.anchors.end() || to == shadow.anchors.end() ||
          from->second.seal.ref != saved.measurement.from ||
          to->second.seal.ref != saved.measurement.to) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                          "loop recovery factor references an absent or stale anchor"));
      }
      const auto factor = boost::make_shared<RankAwareRelativeFactor>(
          from->second.keys.anchor, to->second.keys.anchor, saved.measurement.T_from_to,
          saved.measurement.information);
      shadow.loop_factors.push_back(
          Impl::LoopFactorRecord{saved.factor, saved.measurement, saved.candidate, saved.scale,
                                 from->second.keys.anchor, to->second.keys.anchor, factor});
    }
    shadow.next_factor_id = checkpoint.next_factor_id;
    shadow.next_candidate_id = checkpoint.next_candidate_id;
    impl_->rebuildFactors(shadow);

    if (!impl_->connected(shadow) || shadow.factors.size() != checkpoint.factor_order.size()) {
      return Result::failure(
          recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                        "reconstructed checkpoint graph is disconnected or incomplete"));
    }
    for (std::size_t index = 0U; index < checkpoint.factor_order.size(); ++index) {
      const double cost = 2.0 * shadow.factors.at(index)->error(shadow.values);
      const GlobalFactorObjectiveCheckpoint& expected =
          checkpoint.recovery.factor_objectives[index];
      if (expected.factor != checkpoint.factor_order[index] ||
          !withinTolerance(cost, expected.whitened_squared_cost,
                           checkpoint.recovery.tolerances.objective_absolute,
                           checkpoint.recovery.tolerances.objective_relative)) {
        return Result::failure(recoveryError(
            GlobalGraphCheckpointErrorCode::ObjectiveMismatch,
            "reconstructed per-factor objective differs from the canonical checkpoint"));
      }
    }
    const double reconstructed_objective = 2.0 * shadow.factors.error(shadow.values);
    if (!withinTolerance(reconstructed_objective, checkpoint.recovery.whitened_squared_objective,
                         checkpoint.recovery.tolerances.objective_absolute,
                         checkpoint.recovery.tolerances.objective_relative)) {
      return Result::failure(
          recoveryError(GlobalGraphCheckpointErrorCode::ObjectiveMismatch,
                        "reconstructed complete nonlinear objective differs from the checkpoint"));
    }

    gtsam::LevenbergMarquardtOptimizer recovery_optimizer(shadow.factors, shadow.values,
                                                          optimizerParams(impl_->config));
    const gtsam::Values recovered_values = recovery_optimizer.optimize();
    for (std::size_t slot = 0U; slot < checkpoint.boundaries.size(); ++slot) {
      const Impl::AnchorRecord& anchor = shadow.anchors.at(slot_identities[slot]);
      const BoundaryNavigationCheckpoint& expected = checkpoint.boundaries[slot];
      if (!poseWithinTolerance(fromGtsamPose(recovered_values.at<gtsam::Pose3>(anchor.keys.anchor)),
                               expected.T_map_submap,
                               checkpoint.recovery.tolerances.estimate_tangent_absolute) ||
          (anchor.navigation_materialized &&
           ((!matrixWithinTolerance(
                recovered_values.at<gtsam::Vector3>(anchor.keys.velocity), *expected.velocity_map,
                checkpoint.recovery.tolerances.estimate_tangent_absolute, 0.0)) ||
            (!matrixWithinTolerance(
                recovered_values.at<gtsam::Vector3>(anchor.keys.gyro_bias), *expected.gyro_bias,
                checkpoint.recovery.tolerances.estimate_tangent_absolute, 0.0)) ||
            (!matrixWithinTolerance(
                recovered_values.at<gtsam::Vector3>(anchor.keys.accel_bias), *expected.accel_bias,
                checkpoint.recovery.tolerances.estimate_tangent_absolute, 0.0))))) {
        return Result::failure(recoveryError(
            GlobalGraphCheckpointErrorCode::EstimateMismatch,
            "re-optimized boundary estimate differs from its canonical checkpoint value"));
      }
    }
    if (checkpoint.alignment) {
      const gtsam::Vector4 recovered_alignment =
          recovered_values.at<gtsam::Vector4>(Impl::alignmentKey());
      gtsam::Vector4 expected_alignment;
      expected_alignment.head<3>() = checkpoint.alignment->translation_enu;
      expected_alignment(3) = checkpoint.alignment->yaw_enu_map_rad;
      expected_alignment(3) +=
          kTwoPi * std::round((recovered_alignment(3) - expected_alignment(3)) / kTwoPi);
      if (!matrixWithinTolerance(recovered_alignment, expected_alignment,
                                 checkpoint.recovery.tolerances.estimate_tangent_absolute, 0.0)) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::EstimateMismatch,
                          "re-optimized alignment differs from its canonical checkpoint value"));
      }
    }

    const gtsam::Ordering variable_order = impl_->ordering(shadow);
    const auto gaussian = shadow.factors.linearize(shadow.values);
    const auto hessian_and_gradient = gaussian->hessian(variable_order);
    const gtsam::Matrix& hessian = hessian_and_gradient.first;
    Eigen::SelfAdjointEigenSolver<gtsam::Matrix> eigensolver(0.5 * (hessian + hessian.transpose()),
                                                             Eigen::EigenvaluesOnly);
    if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().allFinite()) {
      return Result::failure(recoveryError(GlobalGraphCheckpointErrorCode::RankMismatch,
                                           "reconstructed Hessian decomposition failed"));
    }
    const double largest = eigensolver.eigenvalues().maxCoeff();
    const double threshold =
        std::max(impl_->config.hessian_absolute_rank_tolerance,
                 impl_->config.hessian_relative_rank_tolerance * std::max(0.0, largest));
    std::size_t rank{};
    double smallest_supported = std::numeric_limits<double>::infinity();
    for (double eigenvalue : eigensolver.eigenvalues()) {
      if (eigenvalue > threshold) {
        ++rank;
        smallest_supported = std::min(smallest_supported, eigenvalue);
      }
    }
    const double condition = largest / smallest_supported;
    const double gradient_norm = hessian_and_gradient.second.size() == 0
                                     ? 0.0
                                     : hessian_and_gradient.second.cwiseAbs().maxCoeff();
    if (static_cast<std::size_t>(hessian.rows()) != checkpoint.recovery.scalar_dimension ||
        rank != checkpoint.recovery.numerical_rank ||
        rank != static_cast<std::size_t>(hessian.rows())) {
      return Result::failure(
          recoveryError(GlobalGraphCheckpointErrorCode::RankMismatch,
                        "reconstructed graph has a different scalar dimension or numerical rank"));
    }
    if (!withinTolerance(condition, checkpoint.recovery.hessian_condition, 0.0,
                         checkpoint.recovery.tolerances.condition_relative) ||
        !withinTolerance(gradient_norm, checkpoint.recovery.gradient_infinity_norm,
                         checkpoint.recovery.tolerances.gradient_infinity_absolute,
                         checkpoint.recovery.tolerances.objective_relative)) {
      return Result::failure(recoveryError(
          GlobalGraphCheckpointErrorCode::ObjectiveMismatch,
          "reconstructed condition or objective gradient differs from the checkpoint"));
    }

    gtsam::Marginals marginals(shadow.factors, shadow.values, gtsam::Marginals::QR);
    std::vector<SubmapAnchorEstimate> anchors;
    anchors.reserve(checkpoint.boundaries.size());
    for (std::size_t slot = 0U; slot < checkpoint.boundaries.size(); ++slot) {
      const Impl::AnchorRecord& anchor = shadow.anchors.at(slot_identities[slot]);
      const core::PoseCovariance covariance =
          fromGtsamPoseCovariance(marginals.marginalCovariance(anchor.keys.anchor));
      const core::PoseCovariance& expected =
          checkpoint.recovery.boundary_marginals[slot].covariance;
      if (!matrixWithinTolerance(covariance.matrix, expected.matrix,
                                 checkpoint.recovery.tolerances.covariance_absolute,
                                 checkpoint.recovery.tolerances.covariance_relative)) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::MarginalMismatch,
                          "reconstructed anchor marginal differs from the checkpoint"));
      }
      anchors.push_back(SubmapAnchorEstimate{anchor.seal.ref, anchor.seal.T_odom_submap,
                                             anchor.seal.support_time.end,
                                             checkpoint.boundaries[slot].T_map_submap, expected});
    }
    std::optional<AlignmentCovariance> alignment_covariance;
    if (checkpoint.alignment) {
      const Eigen::Matrix4d recovered_covariance =
          marginals.marginalCovariance(Impl::alignmentKey());
      if (!matrixWithinTolerance(recovered_covariance,
                                 checkpoint.recovery.alignment_covariance->matrix,
                                 checkpoint.recovery.tolerances.covariance_absolute,
                                 checkpoint.recovery.tolerances.covariance_relative)) {
        return Result::failure(
            recoveryError(GlobalGraphCheckpointErrorCode::MarginalMismatch,
                          "reconstructed alignment marginal differs from the checkpoint"));
      }
      alignment_covariance = checkpoint.recovery.alignment_covariance;
    }

    const Impl::AnchorRecord& reference = impl_->referenceAnchor(shadow);
    const auto reference_estimate =
        std::find_if(anchors.begin(), anchors.end(), [&](const SubmapAnchorEstimate& estimate) {
          return estimate.submap == reference.seal.ref;
        });
    if (reference_estimate == anchors.end()) {
      return Result::failure(recoveryError(GlobalGraphCheckpointErrorCode::MarginalMismatch,
                                           "map-to-odom reference anchor is absent"));
    }
    const core::Pose3d T_map_odom =
        reference_estimate->T_map_submap * reference.seal.T_odom_submap.inverse();
    const core::Matrix6d adjoint = reference.seal.T_odom_submap.Adj();
    const core::Matrix6d map_odom_covariance =
        adjoint * reference_estimate->covariance.matrix * adjoint.transpose();
    if (reference.seal.ref != checkpoint.recovery.map_odom.reference_submap ||
        !poseWithinTolerance(T_map_odom, checkpoint.recovery.map_odom.T_map_odom,
                             checkpoint.recovery.tolerances.estimate_tangent_absolute) ||
        !matrixWithinTolerance(map_odom_covariance, checkpoint.recovery.map_odom.covariance.matrix,
                               checkpoint.recovery.tolerances.covariance_absolute,
                               checkpoint.recovery.tolerances.covariance_relative)) {
      return Result::failure(
          recoveryError(GlobalGraphCheckpointErrorCode::MarginalMismatch,
                        "reconstructed map-to-odom estimate differs from the checkpoint"));
    }

    GlobalGraphCommit commit{checkpoint.revision,
                             checkpoint.parent,
                             std::move(anchors),
                             checkpoint.alignment,
                             std::move(alignment_covariance),
                             checkpoint.recovery.map_odom,
                             checkpoint.recovery.committed_solve};
    GlobalGraphCommit returned = commit;
    GlobalGraphCheckpoint installed_checkpoint = checkpoint;
    // The following moves are the only publication point. All reconstruction,
    // nonlinear checks, marginal checks, and potentially allocating copies
    // completed while the live graph was still empty.
    impl_->state = std::move(shadow);
    impl_->latest_commit = std::move(commit);
    impl_->latest_checkpoint = std::move(installed_checkpoint);
    return Result::success(std::move(returned));
  } catch (const gtsam::IndeterminantLinearSystemException& exception) {
    return Result::failure(recoveryError(
        GlobalGraphCheckpointErrorCode::RankMismatch,
        std::string("checkpoint reconstruction is indeterminate: ") + exception.what()));
  } catch (const std::exception& exception) {
    return Result::failure(
        recoveryError(GlobalGraphCheckpointErrorCode::ReconstructionFailure,
                      std::string("checkpoint reconstruction failed: ") + exception.what()));
  }
}

core::Result<GlobalGraphCommit, GlobalGraphError> GlobalGraph::snapshot() const {
  using Result = core::Result<GlobalGraphCommit, GlobalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(
        graphError(GlobalGraphErrorCode::NotInitialized, "global graph has no committed mission"));
  }
  return Result::success(*impl_->latest_commit);
}

core::Result<GlobalGraphCheckpoint, GlobalGraphError> GlobalGraph::checkpoint() const {
  using Result = core::Result<GlobalGraphCheckpoint, GlobalGraphError>;
  if (!impl_->latest_checkpoint) {
    return Result::failure(
        graphError(GlobalGraphErrorCode::NotInitialized, "global graph has no checkpoint"));
  }
  return Result::success(*impl_->latest_checkpoint);
}

bool GlobalGraph::initialized() const noexcept {
  return impl_->latest_commit.has_value();
}

}  // namespace meridian::global
