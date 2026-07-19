#include "meridian/local/graph.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <boost/make_shared.hpp>
#include <cmath>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "candidate_gate.hpp"
#include "candidate_isolated_isam2.hpp"
#include "direct_lidar_factor.hpp"
#include "gtsam_conventions.hpp"
#include "meridian/core/canonical_bytes.hpp"
#include "meridian/local/finalized_lidar_target_map.hpp"
#include "pipeline_timing_internal.hpp"
#include "visual_gtsam_factor.hpp"

namespace meridian::local {
namespace {

using PoseKey = gtsam::Key;
using VelocityKey = gtsam::Key;
using BiasKey = gtsam::Key;
using EtaKey = gtsam::Key;

struct NavigationKeys {
  PoseKey pose;
  VelocityKey velocity;
  BiasKey bias;
};

struct BoundedEtaKey {
  EtaKey key{};
  double minimum_range_m{};
  double maximum_range_m{};
};

[[nodiscard]] NavigationKeys navigationKeys(std::size_t index) {
  return NavigationKeys{gtsam::Symbol('x', index), gtsam::Symbol('v', index),
                        gtsam::Symbol('b', index)};
}

[[nodiscard]] EtaKey visualEtaKey(VisualLandmarkId landmark) {
  return gtsam::Symbol('e', landmark.value());
}

[[nodiscard]] LocalGraphError graphError(
    LocalGraphErrorCode code, std::string detail,
    std::optional<DirectLidarRegistrationReport> lidar_registration = std::nullopt,
    std::optional<LocalSolveReport> rejected_solve = std::nullopt,
    std::vector<DirectLidarPairReport> lidar_pairs = {}) {
  return LocalGraphError{code,
                         std::move(detail),
                         std::move(lidar_registration),
                         std::move(lidar_pairs),
                         std::nullopt,
                         std::move(rejected_solve),
                         std::nullopt};
}

[[nodiscard]] LocalGraphError factorBatchGraphError(LocalGraphErrorCode code, std::string detail,
                                                    SensorFactorBatchRef batch) {
  LocalGraphError error = graphError(code, std::move(detail));
  error.factor_batch = batch;
  return error;
}

[[nodiscard]] bool finitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

inline constexpr std::string_view kFinalPoseCovarianceChecksumDomain{
    "meridian.local.finalized_lidar_target_map.final_pose_covariance"};
inline constexpr std::uint32_t kFinalPoseCovarianceChecksumSchemaVersion{1U};

[[nodiscard]] std::optional<core::ContentHash> finalizedPoseCovarianceChecksum(
    const core::PoseCovariance& covariance) {
  auto encoder = core::CanonicalEncoder::create(kFinalPoseCovarianceChecksumDomain,
                                                kFinalPoseCovarianceChecksumSchemaVersion, 2048U);
  if (!encoder) {
    return std::nullopt;
  }
  const auto ok = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!ok(encoder.value().writeU8(static_cast<std::uint8_t>(covariance.tangent))) ||
      !ok(encoder.value().writeEigenMatrix(covariance.matrix))) {
    return std::nullopt;
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return std::nullopt;
  }
  return encoded.value().digest();
}

[[nodiscard]] bool validConfig(const LocalGraphConfig& config) {
  const auto& imu = config.imu;
  return imu.gravity_odom.allFinite() && imu.gravity_odom.norm() > 0.0 &&
         finitePositive(imu.accelerometer_noise_density_mps2_sqrt_hz) &&
         finitePositive(imu.gyroscope_noise_density_radps_sqrt_hz) &&
         finitePositive(imu.accelerometer_bias_random_walk_mps3_sqrt_hz) &&
         finitePositive(imu.gyroscope_bias_random_walk_radps2_sqrt_hz) &&
         finitePositive(imu.integration_noise_density) &&
         finitePositive(imu.preintegration_accelerometer_bias_variance_m2ps4) &&
         finitePositive(imu.preintegration_gyroscope_bias_variance_rad2ps2) &&
         imu.initial_accelerometer_bias_mean_mps2.allFinite() &&
         imu.initial_gyroscope_bias_mean_radps.allFinite() &&
         finitePositive(imu.initial_accelerometer_bias_sigma_mps2) &&
         finitePositive(imu.initial_gyroscope_bias_sigma_radps) &&
         config.maximum_navigation_states >= 2U && config.target_fixed_lag.nanoseconds > 0 &&
         finitePositive(config.pose_rotation_relinearization_rad) &&
         finitePositive(config.pose_translation_relinearization_m) &&
         finitePositive(config.velocity_relinearization_mps) &&
         finitePositive(config.accelerometer_bias_relinearization_mps2) &&
         finitePositive(config.gyroscope_bias_relinearization_radps) &&
         finitePositive(config.visual_log_inverse_range_relinearization) &&
         config.maximum_nonlinear_iterations > 0U &&
         finitePositive(config.nonlinear_convergence_sigma_fraction) &&
         config.nonlinear_convergence_sigma_fraction <= 1.0 &&
         finitePositive(config.nonlinear_translation_convergence_m) &&
         finitePositive(config.nonlinear_rotation_convergence_rad) &&
         finitePositive(config.nonlinear_velocity_convergence_mps) &&
         finitePositive(config.nonlinear_accelerometer_bias_convergence_mps2) &&
         finitePositive(config.nonlinear_gyroscope_bias_convergence_radps) &&
         finitePositive(config.nonlinear_visual_log_inverse_range_convergence) &&
         config.maximum_nonlinear_backtracking_steps > 0U &&
         finitePositive(config.nonlinear_backtracking_reduction) &&
         config.nonlinear_backtracking_reduction < 1.0 &&
         std::isfinite(config.nonlinear_objective_absolute_convergence) &&
         config.nonlinear_objective_absolute_convergence >= 0.0 &&
         std::isfinite(config.nonlinear_objective_relative_convergence) &&
         config.nonlinear_objective_relative_convergence >= 0.0 &&
         finitePositive(config.maximum_transaction_translation_correction_m) &&
         finitePositive(config.maximum_transaction_rotation_correction_rad) &&
         std::isfinite(config.complete_objective_nonsmooth_absolute_allowance) &&
         config.complete_objective_nonsmooth_absolute_allowance >= 0.0 &&
         std::isfinite(config.complete_objective_nonsmooth_relative_allowance) &&
         config.complete_objective_nonsmooth_relative_allowance >= 0.0 &&
         config.maximum_visual_landmarks_per_transaction > 0U &&
         config.maximum_visual_factors_per_transaction > 0U &&
         config.maximum_visual_factor_retirements_per_transaction > 0U &&
         config.maximum_direct_lidar_factors_per_transaction > 0U &&
         config.maximum_direct_lidar_factors_per_transaction <=
             core::kMaximumDirectionalObservabilityRecords &&
         config.maximum_finalized_lidar_owners_per_factor > 0U &&
         config.maximum_active_factor_batches > 0U &&
         config.maximum_removable_factor_batches > 0U &&
         config.maximum_removable_factor_batches <= config.maximum_active_factor_batches &&
         config.maximum_factor_batches_per_removal_transaction > 0U &&
         config.maximum_factor_batches_per_removal_transaction <=
             config.maximum_removable_factor_batches &&
         config.maximum_terminal_factor_batch_records > 0U;
}

[[nodiscard]] bool finiteState(const core::NavStateEstimate& state) {
  return state.T_odom_imu.matrix().allFinite() && state.velocity_odom.allFinite() &&
         state.gyro_bias.allFinite() && state.accel_bias.allFinite();
}

[[nodiscard]] bool validCovariance(const NavigationCovariance& covariance) {
  if (!covariance.matrix.allFinite() ||
      covariance.order != NavigationCovarianceOrder::RotationVelocityPositionGyroBiasAccelBias) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  const double symmetry_error =
      (covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff();
  if (symmetry_error > 1.0e-10 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eigen_solver(covariance.matrix);
  return eigen_solver.info() == Eigen::Success && eigen_solver.eigenvalues().minCoeff() > 0.0;
}

[[nodiscard]] bool validPoseCovariance(const core::PoseCovariance& covariance) {
  if (covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !covariance.matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  const double symmetry_error =
      (covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff();
  if (symmetry_error > 1.0e-10 * scale) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<core::Matrix6d> eigen_solver(covariance.matrix);
  return eigen_solver.info() == Eigen::Success &&
         eigen_solver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

[[nodiscard]] bool validRemovalReason(FactorBatchRemovalReason reason) noexcept {
  switch (reason) {
    case FactorBatchRemovalReason::SensorFailure:
    case FactorBatchRemovalReason::FrontendInvalidation:
      return true;
  }
  return false;
}

[[nodiscard]] bool primaryLineageOverlaps(const core::ObservationLineage& lhs,
                                          const core::ObservationLineage& rhs) noexcept {
  for (const core::ObservationUsage& left : lhs.usage) {
    if (left.role != core::ObservationRole::PrimaryResidual) {
      continue;
    }
    for (const core::ObservationUsage& right : rhs.usage) {
      if (right.role == core::ObservationRole::PrimaryResidual &&
          left.slice.overlaps(right.slice)) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] const core::FactorBatchMetadata& factorBatchMetadata(
    const LidarDirectFactorBatch& batch) noexcept {
  return batch.metadata;
}

[[nodiscard]] std::size_t factorBatchFactorCount(const LidarDirectFactorBatch& batch) noexcept {
  return batch.pairs.size() + static_cast<std::size_t>(batch.finalized_map.has_value());
}

[[nodiscard]] bool factorBatchTouchesAnyState(const LidarDirectFactorBatch& batch,
                                              const std::set<std::uint64_t>& states) noexcept {
  return states.contains(batch.source_state.value()) ||
         std::any_of(batch.pairs.begin(), batch.pairs.end(),
                     [&](const LidarDirectFactorPairSpec& pair) {
                       return states.contains(pair.target_state.value());
                     });
}

[[nodiscard]] bool factorBatchSourceIsFinal(const LidarDirectFactorBatch& batch,
                                            const std::set<std::uint64_t>& states) noexcept {
  return states.contains(batch.source_state.value());
}

struct FactorBatchValidationFailure {
  LocalGraphErrorCode code{LocalGraphErrorCode::InvalidFactorBatch};
  std::string detail;
};

[[nodiscard]] core::Matrix6d directionalInformation(
    const core::DirectionalObservability& observability) {
  return observability.basis * observability.eigenvalues.asDiagonal() *
         observability.basis.transpose();
}

[[nodiscard]] core::Matrix6d rankAwareInformation(const core::RankAwareInformation& information) {
  return information.basis * information.eigenvalues.asDiagonal() * information.basis.transpose();
}

[[nodiscard]] bool lidarPrimaryLineageNamesSource(const core::ObservationLineage& lineage,
                                                  core::MeasurementId source_sweep) noexcept {
  bool found = false;
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.role != core::ObservationRole::PrimaryResidual) {
      continue;
    }
    if (!std::holds_alternative<core::MeasurementId>(usage.slice.root) ||
        std::get<core::MeasurementId>(usage.slice.root) != source_sweep) {
      return false;
    }
    found = true;
  }
  return found;
}

[[nodiscard]] bool validLidarRegistrationTermination(
    LidarRegistrationTermination termination) noexcept {
  switch (termination) {
    case LidarRegistrationTermination::Converged:
    case LidarRegistrationTermination::IterationLimitReached:
      return true;
  }
  return false;
}

[[nodiscard]] bool validRankAwareInformation(
    const core::RankAwareInformation& information) noexcept {
  if (!information.finite() ||
      information.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      (information.eigenvalues.array() < 0.0).any()) {
    return false;
  }
  for (std::size_t index = 0U; index < 6U; ++index) {
    const double eigenvalue = information.eigenvalues(static_cast<Eigen::Index>(index));
    if ((index < information.rank && !finitePositive(eigenvalue)) ||
        (index >= information.rank && eigenvalue != 0.0)) {
      return false;
    }
  }
  if (information.rank > 0U) {
    const Eigen::Index rank = static_cast<Eigen::Index>(information.rank);
    const Eigen::MatrixXd supported_gram =
        information.basis.leftCols(rank).transpose() * information.basis.leftCols(rank);
    if (!supported_gram.isApprox(Eigen::MatrixXd::Identity(rank, rank), 1.0e-9)) {
      return false;
    }
  }
  const core::Matrix6d matrix = rankAwareInformation(information);
  const Eigen::SelfAdjointEigenSolver<core::Matrix6d> solver(matrix);
  return solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
         solver.eigenvalues().minCoeff() >=
             -1.0e-10 * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
}

[[nodiscard]] bool validRegistrationReport(
    const LidarDirectFactorBatch& batch,
    const LidarRegistrationSnapshotAggregate& aggregate) noexcept {
  const DirectLidarRegistrationReport& report = batch.registration_report;
  const LidarRegistrationDiagnostics& diagnostics = report.diagnostics;
  const LidarRegistrationWorkCounters& work = report.work;
  if (!validLidarRegistrationTermination(report.termination) ||
      !report.T_odom_source.matrix().allFinite() ||
      !report.source_right_correction.matrix().allFinite() ||
      report.source_right_correction.translation().norm() >
          batch.registration.maximum_correction_translation_m + 1.0e-12 ||
      report.source_right_correction.so3().log().norm() >
          batch.registration.maximum_correction_rotation_rad + 1.0e-12 ||
      !std::isfinite(report.initial_robust_cost) || !std::isfinite(report.final_robust_cost) ||
      report.initial_robust_cost < 0.0 || report.final_robust_cost < 0.0 ||
      !std::isfinite(aggregate.final_robust_cost) || aggregate.final_robust_cost < 0.0 ||
      aggregate.target_count != factorBatchFactorCount(batch) ||
      diagnostics.target_count != aggregate.target_count ||
      diagnostics.live_target_count != aggregate.live_target_count ||
      diagnostics.finalized_map_target_count != aggregate.finalized_map_target_count ||
      diagnostics.live_target_count != batch.pairs.size() ||
      diagnostics.finalized_map_target_count !=
          static_cast<std::size_t>(batch.finalized_map.has_value()) ||
      diagnostics.correspondences != aggregate.correspondences ||
      !std::isfinite(diagnostics.overlap_fraction) || diagnostics.overlap_fraction < 0.0 ||
      diagnostics.overlap_fraction > 1.0 || !std::isfinite(diagnostics.effective_correspondences) ||
      diagnostics.effective_correspondences < 0.0 ||
      !std::isfinite(diagnostics.maximum_squared_residual_m2) ||
      diagnostics.maximum_squared_residual_m2 < 0.0 || !finitePositive(diagnostics.huber_delta_m) ||
      !finitePositive(diagnostics.characteristic_length_m) ||
      !finitePositive(diagnostics.normalized_observable_eigenvalue_threshold) ||
      diagnostics.observable_rank < batch.registration.minimum_observable_rank ||
      diagnostics.observable_rank > 6U ||
      !diagnostics.raw_normalized_hessian_eigenvalues.allFinite() ||
      !validRankAwareInformation(diagnostics.normalized_directional_information) ||
      !validRankAwareInformation(diagnostics.physical_information) ||
      diagnostics.normalized_directional_information.rank != diagnostics.observable_rank ||
      diagnostics.physical_information.rank != diagnostics.observable_rank ||
      work.source_points_selected != aggregate.source_point_count ||
      work.source_points_considered < work.source_points_selected ||
      work.source_points_omitted_by_capacity > work.source_points_considered ||
      work.outer_iterations > batch.registration.maximum_outer_iterations ||
      work.accepted_steps > work.gauss_newton_trials + work.lm_damping_trials ||
      (report.termination == LidarRegistrationTermination::IterationLimitReached &&
       work.outer_iterations != batch.registration.maximum_outer_iterations)) {
    return false;
  }
  if (batch.finalized_map &&
      !report.T_odom_source.matrix().isApprox(
          batch.finalized_map->snapshot->associationPose().matrix(), 1.0e-12)) {
    return false;
  }
  const auto same_scalar = [](double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 1.0e-12 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
  };
  return same_scalar(diagnostics.overlap_fraction, aggregate.overlap_fraction) &&
         same_scalar(diagnostics.effective_correspondences, aggregate.effective_correspondences) &&
         same_scalar(diagnostics.maximum_squared_residual_m2,
                     aggregate.maximum_squared_residual_m2) &&
         same_scalar(diagnostics.huber_delta_m, aggregate.huber_delta_m) &&
         same_scalar(report.final_robust_cost, aggregate.final_robust_cost);
}

[[nodiscard]] std::optional<FactorBatchValidationFailure> validateLidarDirectBatchPayload(
    const LidarDirectFactorBatch& batch, const LocalGraphConfig& config) {
  const core::FactorBatchMetadataValidationError metadata_error =
      core::validateFactorBatchMetadata(batch.metadata);
  if (metadata_error != core::FactorBatchMetadataValidationError::None) {
    return FactorBatchValidationFailure{LocalGraphErrorCode::InvalidFactorBatch,
                                        "FactorBatch metadata validation failed with code " +
                                            std::to_string(static_cast<int>(metadata_error))};
  }
  if (batch.metadata.sensor.modality != core::SensorModality::Lidar) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "LidarDirectFactorBatch metadata must name one LiDAR sensor instance"};
  }
  const std::size_t factor_count = factorBatchFactorCount(batch);
  if (!batch.source_state.valid() || batch.source_time != batch.metadata.timing.reference_time ||
      factor_count == 0U || factor_count > config.maximum_direct_lidar_factors_per_transaction ||
      batch.metadata.directional_observability.size() != factor_count ||
      !finitePositive(batch.registration.maximum_translation_information)) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR source, pair count, observability count, reference time, or sweep budget is "
        "invalid"};
  }

  std::optional<core::MeasurementId> source_sweep;
  std::optional<std::size_t> source_points;
  std::optional<core::ContentHash> source_cloud_checksum;
  std::optional<double> huber_delta_m;
  std::set<std::size_t> owned_source_rows;
  std::set<std::uint32_t> owned_stable_source_indices;
  std::set<core::MeasurementId> required_conditioning_measurements;
  std::size_t correspondence_count = 0U;
  const auto same_scalar = [](double lhs, double rhs) {
    return std::abs(lhs - rhs) <= 1.0e-12 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
  };
  std::vector<std::shared_ptr<const LidarFactorSnapshot>> snapshots;
  snapshots.reserve(batch.pairs.size());
  for (std::size_t index = 0U; index < batch.pairs.size(); ++index) {
    const LidarDirectFactorPairSpec& pair = batch.pairs[index];
    const core::DirectionalObservability& observability =
        batch.metadata.directional_observability[index];
    if (!pair.target_state.valid() || !(pair.target_state < batch.source_state) ||
        !(pair.target_time < batch.source_time) || !pair.snapshot ||
        !core::contentHashPresent(pair.snapshot->checksum()) ||
        pair.snapshot->targetState() != pair.target_state ||
        pair.snapshot->targetTime() != pair.target_time ||
        pair.snapshot->sourceState() != batch.source_state ||
        pair.snapshot->sourceTime() != batch.source_time || !pair.snapshot->targetSweep().valid() ||
        !pair.snapshot->sourceSweep().valid() || !finitePositive(pair.information_scale) ||
        pair.information_scale > 1.0) {
      return FactorBatchValidationFailure{LocalGraphErrorCode::InvalidFactorBatch,
                                          "direct LiDAR pair endpoint, immutable snapshot, or "
                                          "information scale is invalid at index " +
                                              std::to_string(index)};
    }
    if (index > 0U && !(batch.pairs[index - 1U].target_state < pair.target_state)) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct LiDAR target states must be unique and strictly increasing"};
    }
    if ((source_sweep && *source_sweep != pair.snapshot->sourceSweep()) ||
        (source_points && *source_points != pair.snapshot->sourcePointCount()) ||
        (source_cloud_checksum && *source_cloud_checksum != pair.snapshot->sourceCloudChecksum()) ||
        (huber_delta_m && !same_scalar(*huber_delta_m, pair.snapshot->huberDeltaM()))) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "all direct LiDAR factors must own rows from one bit-identical source population and "
          "Huber objective"};
    }
    source_sweep = pair.snapshot->sourceSweep();
    source_points = pair.snapshot->sourcePointCount();
    source_cloud_checksum = pair.snapshot->sourceCloudChecksum();
    huber_delta_m = pair.snapshot->huberDeltaM();
    required_conditioning_measurements.insert(pair.snapshot->targetSweep());
    if (*source_points == 0U || pair.snapshot->rows().empty()) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct point ICP pair has no selected source population or frozen rows at index " +
              std::to_string(index)};
    }
    for (const FrozenPointCorrespondence& row : pair.snapshot->rows()) {
      if (!owned_source_rows.insert(row.source_point_storage_index).second ||
          !owned_stable_source_indices.insert(row.source_index).second) {
        return FactorBatchValidationFailure{
            LocalGraphErrorCode::FactorBatchLineageConflict,
            "direct point ICP factors overlap in sparse storage or stable source-index "
            "ownership"};
      }
      ++correspondence_count;
    }

    const auto canonical_information =
        lidarFactorInformation(pair.snapshot, batch.registration, pair.information_scale);
    if (!canonical_information) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct point ICP pair has invalid canonical admission information at index " +
              std::to_string(index) + ": " + canonical_information.error().detail};
    }
    if (observability.rank != canonical_information.value().rank ||
        observability.supported_variables !=
            std::vector<core::DirectionalVariable>{core::DirectionalVariable::PoseTranslation,
                                                   core::DirectionalVariable::PoseRotation} ||
        observability.endpoints !=
            std::vector<core::DirectionalObservabilityEndpoint>{
                {core::DirectionalEndpointRole::Target, pair.target_state, pair.target_time},
                {core::DirectionalEndpointRole::Source, batch.source_state, batch.source_time}}) {
      return FactorBatchValidationFailure{LocalGraphErrorCode::InvalidFactorBatch,
                                          "direct LiDAR observability variables, rank, or "
                                          "endpoints differ from its pair at index " +
                                              std::to_string(index)};
    }
    const core::Matrix6d declared_information = directionalInformation(observability);
    const core::Matrix6d canonical_information_matrix =
        rankAwareInformation(canonical_information.value());
    const double scale = std::max(1.0, canonical_information_matrix.cwiseAbs().maxCoeff());
    if (!declared_information.isApprox(canonical_information_matrix, 1.0e-8 * scale)) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct LiDAR directional observability does not reconstruct pair information at index " +
              std::to_string(index)};
    }
    snapshots.push_back(pair.snapshot);
  }
  if (batch.finalized_map) {
    const LidarFinalizedMapFactorSpec& map_spec = *batch.finalized_map;
    const auto& snapshot = map_spec.snapshot;
    if (!snapshot || !core::contentHashPresent(snapshot->checksum()) ||
        snapshot->sourceState() != batch.source_state ||
        snapshot->sourceTime() != batch.source_time || !snapshot->sourceSweep().valid() ||
        !core::contentHashPresent(snapshot->sourceCloudChecksum()) ||
        snapshot->mapOdomEpoch() != batch.metadata.odom_epoch ||
        snapshot->mapSensor() != batch.metadata.sensor || snapshot->mapVersion() == 0U ||
        !core::contentHashPresent(snapshot->mapChecksum()) ||
        !finitePositive(map_spec.configured_correlation_inflation_floor) ||
        !finitePositive(map_spec.information_scale) || map_spec.information_scale > 1.0 ||
        snapshot->rows().empty() || snapshot->sourcePointCount() == 0U ||
        snapshot->uniqueOwnerCount() == 0U ||
        snapshot->uniqueOwnerCount() > config.maximum_finalized_lidar_owners_per_factor) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct finalized-map LiDAR factor identity, source, map view, scale, or owner bound "
          "is invalid"};
    }
    if ((source_sweep && *source_sweep != snapshot->sourceSweep()) ||
        (source_points && *source_points != snapshot->sourcePointCount()) ||
        (source_cloud_checksum && *source_cloud_checksum != snapshot->sourceCloudChecksum()) ||
        (huber_delta_m && !same_scalar(*huber_delta_m, snapshot->huberDeltaM()))) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "live and finalized-map LiDAR factors do not share one bit-identical source population "
          "and Huber objective"};
    }
    source_sweep = snapshot->sourceSweep();
    source_points = snapshot->sourcePointCount();
    source_cloud_checksum = snapshot->sourceCloudChecksum();
    huber_delta_m = snapshot->huberDeltaM();

    using OwnerIdentity =
        std::tuple<SensorFactorBatchRef, core::StateId, core::MeasurementId,
                   core::LocalGraphRevision, core::ContentHash, core::ContentHash>;
    std::set<OwnerIdentity> unique_owners;
    for (const auto& owner_handle : snapshot->owners()) {
      if (!owner_handle) {
        return FactorBatchValidationFailure{
            LocalGraphErrorCode::InvalidFactorBatch,
            "direct finalized-map owner table contains a null owner"};
      }
      const FinalizedLidarTargetOwner& owner = *owner_handle;
      const FinalizedLidarAdmissionReceipt& accepted = owner.admission;
      const auto lineage_checksum = recomputeAcceptedLidarLineageChecksum(owner.cloud_lineage);
      const auto covariance_checksum =
          finalizedPoseCovarianceChecksum(owner.finalized_state.pose_covariance);
      if (!owner.batch.batch_id.valid() || owner.batch.sensor != snapshot->mapSensor() ||
          accepted.health.sensor != owner.batch.sensor ||
          accepted.odom_epoch != snapshot->mapOdomEpoch() || !accepted.map_eligible ||
          accepted.health.state != core::SensorHealthState::Active ||
          accepted.health.assessed_at < accepted.reference_time ||
          core::validateSensorHealthSnapshot(accepted.health) !=
              core::SensorHealthValidationError::None ||
          accepted.header.schema_version == 0U || !accepted.header.trace.valid() ||
          !accepted.header.producer.valid() || !accepted.header.session.valid() ||
          !accepted.header.config.valid() ||
          accepted.header.direct_calibration != owner.calibration ||
          !accepted.accepted_lineage.valid() ||
          !core::contentHashPresent(accepted.accepted_lineage_checksum) ||
          !core::contentHashPresent(accepted.accepted_batch_metadata_checksum) ||
          owner.finalized_state.odom_epoch != snapshot->mapOdomEpoch() ||
          !owner.finalized_state.state.valid() || !owner.finalized_state.final_revision.valid() ||
          owner.finalized_state.exact_time != accepted.reference_time ||
          !finiteState(owner.finalized_state.final_estimate) ||
          !validPoseCovariance(owner.finalized_state.pose_covariance) || !owner.sweep.valid() ||
          !core::contentHashPresent(owner.cloud_checksum) || !owner.calibration.valid() ||
          !core::contentHashPresent(owner.final_pose_covariance_checksum) || !covariance_checksum ||
          covariance_checksum.value() != owner.final_pose_covariance_checksum ||
          core::validateLineage(owner.cloud_lineage) != core::LineageValidationError::None ||
          !lineage_checksum || lineage_checksum.value() != owner.cloud_lineage.checksum) {
        return FactorBatchValidationFailure{
            LocalGraphErrorCode::InvalidFactorBatch,
            "direct finalized-map row owner has incomplete or inconsistent finality, covariance, "
            "health, calibration, or lineage provenance"};
      }
      std::set<core::MeasurementId> owner_imu_support;
      std::set<core::MeasurementId> nested_conditioning;
      for (const core::ObservationUsage& usage : owner.cloud_lineage.usage) {
        if (usage.role != core::ObservationRole::ConditioningOnly) {
          continue;
        }
        const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
        if (measurement != nullptr) {
          nested_conditioning.insert(*measurement);
        }
      }
      for (const core::MeasurementId imu : owner.imu_support) {
        if (!imu.valid() || !owner_imu_support.insert(imu).second ||
            !nested_conditioning.contains(imu)) {
          return FactorBatchValidationFailure{
              LocalGraphErrorCode::InvalidFactorBatch,
              "direct finalized-map owner IMU ancestry must be valid, unique, and retained in "
              "its nested cloud lineage"};
        }
      }
      unique_owners.emplace(owner.batch, owner.finalized_state.state, owner.sweep,
                            owner.finalized_state.final_revision, owner.cloud_checksum,
                            owner.final_pose_covariance_checksum);
      required_conditioning_measurements.insert(owner.sweep);
    }
    for (const FrozenFinalizedMapPointCorrespondence& row : snapshot->rows()) {
      if (row.owner_index >= snapshot->owners().size() || !row.source_point.allFinite() ||
          !row.target_point_odom.allFinite() ||
          !owned_source_rows.insert(row.source_point_storage_index).second ||
          !owned_stable_source_indices.insert(row.source_index).second) {
        return FactorBatchValidationFailure{
            LocalGraphErrorCode::FactorBatchLineageConflict,
            "direct finalized-map rows have invalid geometry, owner index, or overlap a live "
            "source row"};
      }
      ++correspondence_count;
    }
    if (unique_owners.size() != snapshot->uniqueOwnerCount() ||
        unique_owners.size() > config.maximum_finalized_lidar_owners_per_factor) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct finalized-map unique owner count differs from its sealed bounded provenance"};
    }

    const auto canonical_information = lidarFinalizedMapFactorInformation(
        snapshot, batch.registration, map_spec.information_scale);
    const auto owner_inflation = lidarFinalizedMapOwnerPoseCovarianceInflation(snapshot);
    const core::DirectionalObservability& observability =
        batch.metadata.directional_observability[batch.pairs.size()];
    if (!canonical_information || !owner_inflation ||
        !same_scalar(owner_inflation.value(), snapshot->ownerPoseCovarianceInflation()) ||
        observability.rank != canonical_information.value().rank ||
        observability.supported_variables !=
            std::vector<core::DirectionalVariable>{core::DirectionalVariable::PoseTranslation,
                                                   core::DirectionalVariable::PoseRotation} ||
        observability.endpoints !=
            std::vector<core::DirectionalObservabilityEndpoint>{
                {core::DirectionalEndpointRole::Unary, batch.source_state, batch.source_time}}) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct finalized-map information, owner covariance inflation, or unary observability "
          "endpoint is inconsistent"};
    }
    const core::Matrix6d declared_information = directionalInformation(observability);
    const core::Matrix6d canonical_information_matrix =
        rankAwareInformation(canonical_information.value());
    const double scale = std::max(1.0, canonical_information_matrix.cwiseAbs().maxCoeff());
    if (!declared_information.isApprox(canonical_information_matrix, 1.0e-8 * scale)) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct finalized-map directional observability does not reconstruct unary "
          "information"};
    }
  }
  if (!source_points || correspondence_count > *source_points ||
      owned_source_rows.size() != correspondence_count ||
      owned_stable_source_indices.size() != correspondence_count) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::FactorBatchLineageConflict,
        "direct LiDAR factors exceed or ambiguously own the selected source population"};
  }

  const auto aggregate = summarizeLidarFactorSnapshots(
      snapshots, batch.finalized_map ? batch.finalized_map->snapshot : nullptr);
  if (!aggregate || !source_points || aggregate.value().source_point_count != *source_points ||
      aggregate.value().correspondences != correspondence_count ||
      !validRegistrationReport(batch, aggregate.value())) {
    const auto& diagnostics = batch.registration_report.diagnostics;
    const auto& work = batch.registration_report.work;
    const LidarRegistrationSnapshotAggregate canonical =
        aggregate ? aggregate.value() : LidarRegistrationSnapshotAggregate{};
    const std::string aggregate_error =
        aggregate ? std::string{} : ", aggregate error=" + aggregate.error().detail;
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct point ICP registration report is inconsistent with its immutable target snapshots: "
        "targets report/pairs=" +
            std::to_string(diagnostics.target_count) + "/" + std::to_string(factor_count) +
            ", correspondences report/rows=" + std::to_string(diagnostics.correspondences) + "/" +
            std::to_string(canonical.correspondences) +
            ", source selected/snapshot=" + std::to_string(work.source_points_selected) + "/" +
            std::to_string(canonical.source_point_count) +
            ", overlap report/rows=" + std::to_string(diagnostics.overlap_fraction) + "/" +
            std::to_string(canonical.overlap_fraction) + ", effective report/rows=" +
            std::to_string(diagnostics.effective_correspondences) + "/" +
            std::to_string(canonical.effective_correspondences) + ", max mahalanobis report/rows=" +
            std::to_string(diagnostics.maximum_squared_residual_m2) + "/" +
            std::to_string(canonical.maximum_squared_residual_m2) +
            ", Huber delta report/rows=" + std::to_string(diagnostics.huber_delta_m) + "/" +
            std::to_string(canonical.huber_delta_m) + ", final robust cost report/rows=" +
            std::to_string(batch.registration_report.final_robust_cost) + "/" +
            std::to_string(canonical.final_robust_cost) + ", initial robust cost=" +
            std::to_string(batch.registration_report.initial_robust_cost) +
            ", observable ranks normalized/physical=" +
            std::to_string(diagnostics.normalized_directional_information.rank) + "/" +
            std::to_string(diagnostics.physical_information.rank) + aggregate_error};
  }
  if (!source_sweep || !lidarPrimaryLineageNamesSource(batch.metadata.lineage, *source_sweep)) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR primary lineage must name only its exact source sweep"};
  }

  const core::ObservationLineage& lineage = batch.metadata.lineage;
  if (lineage.correlations.size() != 1U) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR lineage must declare exactly one batch-local correlation policy"};
  }
  const core::CorrelationDeclaration& declaration = lineage.correlations.front();
  const core::CorrelationPolicyRevision expected_policy{batch.finalized_map ? 3U : 1U};
  if (declaration.policy != expected_policy ||
      declaration.treatment != core::CorrelationTreatment::CovarianceInflationAndInformationCap ||
      !std::isfinite(declaration.covariance_inflation) || declaration.covariance_inflation <= 1.0 ||
      !declaration.total_information_cap || !finitePositive(*declaration.total_information_cap)) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR lineage has a missing, unity, or unsupported revision-specific correlation "
        "treatment"};
  }
  std::size_t primary_usages = 0U;
  std::size_t conditioning_usages = 0U;
  std::set<core::MeasurementId> conditioning_measurements;
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.role == core::ObservationRole::PrimaryResidual) {
      const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
      if (measurement == nullptr || *measurement != *source_sweep) {
        return FactorBatchValidationFailure{
            LocalGraphErrorCode::InvalidFactorBatch,
            "direct LiDAR source sweep must be the sole PrimaryResidual observation"};
      }
      ++primary_usages;
      continue;
    }
    if (usage.role != core::ObservationRole::ConditioningOnly || usage.factor_group ||
        usage.correlation_group != declaration.group) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct LiDAR non-primary ancestry must be conditioning-only in its declared group"};
    }
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    if (measurement == nullptr || *measurement == *source_sweep ||
        !conditioning_measurements.insert(*measurement).second) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct LiDAR target and IMU conditioning ancestry must be measurement-based and "
          "deduplicated"};
    }
    ++conditioning_usages;
  }
  if (primary_usages != 1U || conditioning_usages == 0U) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR lineage requires one source PrimaryResidual and target or deskew "
        "conditioning ancestry"};
  }
  for (const core::MeasurementId required : required_conditioning_measurements) {
    if (!conditioning_measurements.contains(required)) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct LiDAR lineage omits a live target, finalized-owner sweep, or current/live IMU "
          "from conditioning ancestry"};
    }
  }

  double owner_pose_covariance_inflation = 1.0;
  if (batch.finalized_map) {
    const auto recomputed =
        lidarFinalizedMapOwnerPoseCovarianceInflation(batch.finalized_map->snapshot);
    if (!recomputed || !finitePositive(recomputed.value())) {
      return FactorBatchValidationFailure{
          LocalGraphErrorCode::InvalidFactorBatch,
          "direct finalized-map owner pose covariance inflation cannot be reconstructed"};
    }
    owner_pose_covariance_inflation = recomputed.value();
  }
  const double expected_information_cap =
      batch.registration.maximum_translation_information / batch.base_covariance_inflation;
  const double expected_live_information_scale = 1.0 / batch.base_covariance_inflation;
  const double configured_map_correlation_floor =
      batch.finalized_map ? batch.finalized_map->configured_correlation_inflation_floor : 1.0;
  const double effective_map_covariance_inflation =
      std::max(owner_pose_covariance_inflation, configured_map_correlation_floor);
  const double expected_map_covariance_inflation =
      batch.base_covariance_inflation * effective_map_covariance_inflation;
  const double expected_map_information_scale = 1.0 / expected_map_covariance_inflation;
  const bool valid_pair_scale =
      std::all_of(batch.pairs.begin(), batch.pairs.end(), [&](const auto& pair) {
        return same_scalar(pair.information_scale, expected_live_information_scale);
      });
  const bool valid_map_scale =
      !batch.finalized_map ||
      same_scalar(batch.finalized_map->information_scale, expected_map_information_scale);
  if (!std::isfinite(batch.base_covariance_inflation) || batch.base_covariance_inflation <= 1.0 ||
      !same_scalar(declaration.covariance_inflation, batch.base_covariance_inflation) ||
      !finitePositive(expected_information_cap) ||
      !finitePositive(expected_live_information_scale) ||
      !finitePositive(configured_map_correlation_floor) ||
      !finitePositive(effective_map_covariance_inflation) ||
      !finitePositive(expected_map_covariance_inflation) ||
      !finitePositive(expected_map_information_scale) ||
      !same_scalar(*declaration.total_information_cap, expected_information_cap) ||
      !valid_pair_scale || !valid_map_scale) {
    return FactorBatchValidationFailure{
        LocalGraphErrorCode::InvalidFactorBatch,
        "direct LiDAR live/map per-channel scaling, base and owner covariance inflation, "
        "configured map correlation floor, policy revision, and declared information cap "
        "disagree"};
  }
  return std::nullopt;
}

[[nodiscard]] bool validLineage(const core::ObservationLineage& lineage) {
  return lineage.id.valid() && core::validateLineage(lineage) == core::LineageValidationError::None;
}

[[nodiscard]] bool validPixelCovariance(const Eigen::Matrix2d& covariance) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-10 * scale) {
    return false;
  }
  return Eigen::LLT<Eigen::Matrix2d>(covariance).info() == Eigen::Success;
}

[[nodiscard]] bool validVisualObservation(const VisualObservationRef& observation) {
  const EquidistantCamera camera(observation.camera_model);
  return observation.frame.valid() && observation.state.valid() && observation.camera.valid() &&
         observation.calibration.valid() && observation.pixel.allFinite() &&
         camera.isInsideImage(observation.pixel) &&
         validPixelCovariance(observation.pixel_covariance) &&
         observation.imu_from_camera.T_imu_camera().matrix().allFinite();
}

[[nodiscard]] bool sameCameraModel(const EquidistantCameraParameters& lhs,
                                   const EquidistantCameraParameters& rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height && lhs.fx == rhs.fx &&
         lhs.fy == rhs.fy && lhs.cx == rhs.cx && lhs.cy == rhs.cy && lhs.k1 == rhs.k1 &&
         lhs.k2 == rhs.k2 && lhs.k3 == rhs.k3 && lhs.k4 == rhs.k4;
}

[[nodiscard]] bool sameVisualObservation(const VisualObservationRef& lhs,
                                         const VisualObservationRef& rhs) {
  return lhs.frame == rhs.frame && lhs.state == rhs.state && lhs.exact_time == rhs.exact_time &&
         lhs.camera == rhs.camera && lhs.calibration == rhs.calibration &&
         (lhs.pixel.array() == rhs.pixel.array()).all() &&
         (lhs.pixel_covariance.array() == rhs.pixel_covariance.array()).all() &&
         sameCameraModel(lhs.camera_model, rhs.camera_model) &&
         lhs.imu_from_camera.T_imu_camera().matrix().isApprox(
             rhs.imu_from_camera.T_imu_camera().matrix(), 0.0);
}

[[nodiscard]] bool lineageContainsVisualObservation(
    const core::ObservationLineage& lineage, const VisualObservationRef& observation,
    core::ObservationRole role, std::optional<core::FactorGroupId> factor_group) {
  return std::any_of(
      lineage.usage.begin(), lineage.usage.end(), [&](const core::ObservationUsage& usage) {
        return std::holds_alternative<core::MeasurementId>(usage.slice.root) &&
               std::get<core::MeasurementId>(usage.slice.root) == observation.frame &&
               usage.slice.kind == core::SliceKind::Whole &&
               usage.slice.calibration == observation.calibration && usage.role == role &&
               usage.factor_group == factor_group;
      });
}

class JointNavigationPriorFactor final
    : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias> {
public:
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Vector3, gtsam::imuBias::ConstantBias>;

  JointNavigationPriorFactor(PoseKey pose_key, VelocityKey velocity_key, BiasKey bias_key,
                             gtsam::Pose3 pose, gtsam::Vector3 velocity,
                             gtsam::imuBias::ConstantBias bias,
                             const Eigen::Matrix<double, 15, 15>& covariance)
      : Base(gtsam::noiseModel::Gaussian::Covariance(covariance), pose_key, velocity_key, bias_key),
        pose_(std::move(pose)),
        velocity_(std::move(velocity)),
        bias_(std::move(bias)) {}

  gtsam::Vector evaluateError(
      const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
      const gtsam::imuBias::ConstantBias& bias,
      boost::optional<gtsam::Matrix&> pose_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> velocity_jacobian = boost::none,
      boost::optional<gtsam::Matrix&> bias_jacobian = boost::none) const override {
    gtsam::Matrix6 between_jacobian;
    const gtsam::Pose3 relative = pose_.between(pose, boost::none, between_jacobian);
    gtsam::Matrix6 log_jacobian;

    gtsam::Vector15 error;
    error.head<6>() = gtsam::Pose3::Logmap(relative, log_jacobian);
    error.segment<3>(6) = velocity - velocity_;
    error.tail<6>() = (bias - bias_).vector();

    if (pose_jacobian) {
      *pose_jacobian = gtsam::Matrix::Zero(15, 6);
      pose_jacobian->topRows<6>() = log_jacobian * between_jacobian;
    }
    if (velocity_jacobian) {
      *velocity_jacobian = gtsam::Matrix::Zero(15, 3);
      velocity_jacobian->block<3, 3>(6, 0).setIdentity();
    }
    if (bias_jacobian) {
      *bias_jacobian = gtsam::Matrix::Zero(15, 6);
      bias_jacobian->bottomRows<6>().setIdentity();
    }
    return error;
  }

private:
  gtsam::Pose3 pose_;
  gtsam::Vector3 velocity_;
  gtsam::imuBias::ConstantBias bias_;
};

[[nodiscard]] gtsam::ISAM2Params solverParams(const LocalGraphConfig& config) {
  gtsam::ISAM2Params params;
  // CandidateIsolatedISAM2 preserves exact nonlinear-factor slots and ordered
  // keys. Immutable direct point ICP factors require no private mutable cache.
  // Keep the configured incremental relinearization policy separate from the
  // physically propagated stopping thresholds and true-cost acceptance gates.
  // GLIM's production local-odometry profile uses iSAM2 Gauss--Newton. A zero
  // wildfire threshold computes the complete absolute Bayes-tree delta and
  // avoids Dogleg's stale-theta trust-region baseline in stock GTSAM 4.2.
  params.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  params.relinearizeSkip = 1;
  params.enableRelinearization = true;
  // The transaction path evaluates the complete nonlinear factor graph at
  // every seed, iteration baseline, and candidate used by its acceptance
  // gates. NonlinearSolveSummary is therefore the authoritative source for
  // public error_before/error_after. Asking iSAM2 to evaluate the same graph
  // inside every update would duplicate those exact full-factor passes.
  params.evaluateNonlinearError = false;
  params.factorization = gtsam::ISAM2Params::QR;
  params.findUnusedFactorSlots = true;
  // Stock GTSAM 4.2 does not value-initialize every scalar in ISAM2Result on
  // empty updates. Detailed per-key status gives deterministic work counters;
  // the remaining scalar counters are admitted only when their update really
  // reeliminated variables and they are within graph-owned bounds.
  params.enableDetailedResults = true;

  gtsam::ISAM2ThresholdMap thresholds;
  thresholds['x'] =
      (gtsam::Vector6() << config.pose_rotation_relinearization_rad,
       config.pose_rotation_relinearization_rad, config.pose_rotation_relinearization_rad,
       config.pose_translation_relinearization_m, config.pose_translation_relinearization_m,
       config.pose_translation_relinearization_m)
          .finished();
  thresholds['v'] = gtsam::Vector3::Constant(config.velocity_relinearization_mps);
  // ConstantBias tangent order is [accelerometer, gyroscope].
  thresholds['b'] =
      (gtsam::Vector6() << config.accelerometer_bias_relinearization_mps2,
       config.accelerometer_bias_relinearization_mps2,
       config.accelerometer_bias_relinearization_mps2, config.gyroscope_bias_relinearization_radps,
       config.gyroscope_bias_relinearization_radps, config.gyroscope_bias_relinearization_radps)
          .finished();
  thresholds['e'] = gtsam::Vector1::Constant(config.visual_log_inverse_range_relinearization);
  params.relinearizeThreshold = std::move(thresholds);
  return params;
}

[[nodiscard]] boost::shared_ptr<gtsam::PreintegrationCombinedParams> preintegrationParams(
    const LocalGraphConfig& config) {
  const auto& imu = config.imu;
  auto params = boost::make_shared<gtsam::PreintegrationCombinedParams>(imu.gravity_odom);
  params->accelerometerCovariance =
      Eigen::Matrix3d::Identity() * std::pow(imu.accelerometer_noise_density_mps2_sqrt_hz, 2);
  params->gyroscopeCovariance =
      Eigen::Matrix3d::Identity() * std::pow(imu.gyroscope_noise_density_radps_sqrt_hz, 2);
  params->integrationCovariance =
      Eigen::Matrix3d::Identity() * std::pow(imu.integration_noise_density, 2);
  params->biasAccCovariance =
      Eigen::Matrix3d::Identity() * std::pow(imu.accelerometer_bias_random_walk_mps3_sqrt_hz, 2);
  params->biasOmegaCovariance =
      Eigen::Matrix3d::Identity() * std::pow(imu.gyroscope_bias_random_walk_radps2_sqrt_hz, 2);
  params->biasAccOmegaInt.setZero();
  params->biasAccOmegaInt.diagonal().head<3>().setConstant(
      imu.preintegration_accelerometer_bias_variance_m2ps4);
  params->biasAccOmegaInt.diagonal().tail<3>().setConstant(
      imu.preintegration_gyroscope_bias_variance_rad2ps2);
  return params;
}

struct VisualSolveCounts {
  std::size_t active_landmarks{};
  std::size_t active_factors{};
  std::size_t landmarks_added{};
  std::size_t factors_added{};
  std::size_t factors_retired{};
  std::size_t landmarks_marginalized{};
};

struct StateCorrectionMetrics {
  double translation_m{};
  double rotation_rad{};
  double velocity_mps{};
  double accelerometer_bias_mps2{};
  double gyroscope_bias_radps{};
  double visual_log_inverse_range{};

  [[nodiscard]] bool finite() const noexcept {
    return std::isfinite(translation_m) && std::isfinite(rotation_rad) &&
           std::isfinite(velocity_mps) && std::isfinite(accelerometer_bias_mps2) &&
           std::isfinite(gyroscope_bias_radps) && std::isfinite(visual_log_inverse_range);
  }
};

struct PoseCorrectionExtrema {
  StateCorrectionMetrics maximum;
  std::optional<std::size_t> translation_navigation_index;
  std::optional<std::size_t> rotation_navigation_index;
};

[[nodiscard]] StateCorrectionMetrics poseCorrection(const gtsam::Pose3& from,
                                                    const gtsam::Pose3& to) {
  StateCorrectionMetrics correction;
  correction.translation_m = (to.translation() - from.translation()).norm();
  correction.rotation_rad = gtsam::Rot3::Logmap(from.rotation().between(to.rotation())).norm();
  return correction;
}

[[nodiscard]] PoseCorrectionExtrema poseCorrectionExtrema(
    const gtsam::Values& from, const gtsam::Values& to,
    const std::vector<NavigationKeys>& navigation_keys) {
  PoseCorrectionExtrema result;
  for (std::size_t index = 0U; index < navigation_keys.size(); ++index) {
    const NavigationKeys& keys = navigation_keys[index];
    const StateCorrectionMetrics correction =
        poseCorrection(from.at<gtsam::Pose3>(keys.pose), to.at<gtsam::Pose3>(keys.pose));
    if (!result.translation_navigation_index ||
        correction.translation_m > result.maximum.translation_m) {
      result.maximum.translation_m = correction.translation_m;
      result.translation_navigation_index = index;
    }
    if (!result.rotation_navigation_index ||
        correction.rotation_rad > result.maximum.rotation_rad) {
      result.maximum.rotation_rad = correction.rotation_rad;
      result.rotation_navigation_index = index;
    }
  }
  return result;
}

struct EffectiveConvergenceThresholds {
  double interval_duration_s{};
  double sigma_fraction{};
  StateCorrectionMetrics limits;
};

[[nodiscard]] EffectiveConvergenceThresholds effectiveConvergenceThresholds(
    const LocalGraphConfig& config, core::Duration interval_duration) {
  const double dt = static_cast<double>(interval_duration.nanoseconds) * 1.0e-9;
  const double sqrt_dt = std::sqrt(dt);
  const double fraction = config.nonlinear_convergence_sigma_fraction;
  const auto& imu = config.imu;

  EffectiveConvergenceThresholds thresholds;
  thresholds.interval_duration_s = dt;
  thresholds.sigma_fraction = fraction;
  // These are one-axis standard deviations under the same continuous white
  // noise convention used by GTSAM preintegration. Integrating white
  // acceleration once gives n_a*sqrt(dt) velocity sigma; integrating it twice
  // gives n_a*dt^(3/2)/sqrt(3) position sigma. Rotation uses the small-angle
  // integral of white gyro noise, and bias densities are continuous random
  // walks. The configured component values remain strict numerical floors.
  thresholds.limits.translation_m = std::max(
      config.nonlinear_translation_convergence_m,
      fraction * imu.accelerometer_noise_density_mps2_sqrt_hz * dt * sqrt_dt / std::sqrt(3.0));
  thresholds.limits.rotation_rad =
      std::max(config.nonlinear_rotation_convergence_rad,
               fraction * imu.gyroscope_noise_density_radps_sqrt_hz * sqrt_dt);
  thresholds.limits.velocity_mps =
      std::max(config.nonlinear_velocity_convergence_mps,
               fraction * imu.accelerometer_noise_density_mps2_sqrt_hz * sqrt_dt);
  thresholds.limits.accelerometer_bias_mps2 =
      std::max(config.nonlinear_accelerometer_bias_convergence_mps2,
               fraction * imu.accelerometer_bias_random_walk_mps3_sqrt_hz * sqrt_dt);
  thresholds.limits.gyroscope_bias_radps =
      std::max(config.nonlinear_gyroscope_bias_convergence_radps,
               fraction * imu.gyroscope_bias_random_walk_radps2_sqrt_hz * sqrt_dt);
  thresholds.limits.visual_log_inverse_range =
      config.nonlinear_visual_log_inverse_range_convergence;
  return thresholds;
}

struct NonlinearSolveSummary {
  std::size_t iterations{};
  std::size_t variables_relinearized{};
  std::size_t variables_reeliminated{};
  std::size_t factors_recalculated{};
  std::size_t cliques{};
  EffectiveConvergenceThresholds convergence;
  StateCorrectionMetrics last_iteration_correction;
  double maximum_iteration_translation_m{};
  double maximum_iteration_rotation_rad{};
  double maximum_transaction_translation_m{};
  double maximum_transaction_rotation_rad{};
  double marginalization_translation_m{};
  double marginalization_rotation_rad{};
  std::size_t full_steps_rejected{};
  std::size_t backtracking_trials{};
  std::size_t cauchy_directions_attempted{};
  std::size_t cauchy_steps_accepted{};
  std::size_t cauchy_backtracking_trials{};
  std::size_t zero_step_terminations{};
  double minimum_step_scale{1.0};
  double last_objective_change{};
  std::optional<double> error_before;
  std::optional<double> error_after;
};

[[nodiscard]] double objectiveConvergenceTolerance(const LocalGraphConfig& config,
                                                   double reference_error) {
  return config.nonlinear_objective_absolute_convergence +
         config.nonlinear_objective_relative_convergence * std::max(1.0, std::abs(reference_error));
}

// Once every physical state increment is below its declared threshold, a
// finite decrease in the complete nonlinear objective is safe convergence,
// not a reason to keep forcing relinearization.  Only an increase needs the
// bounded numerical stabilization allowance.  The converged-transaction gate
// below still re-evaluates the complete objective and all physical limits.
[[nodiscard]] bool objectiveAcceptsPhysicalConvergence(double objective_change,
                                                       double stabilization_tolerance) {
  return std::isfinite(objective_change) &&
         objective_change <= stabilization_tolerance;
}

void accumulateUpdate(const gtsam::ISAM2Result& update, std::size_t maximum_variables,
                      std::size_t maximum_factors, NonlinearSolveSummary& summary) {
  std::size_t variables_relinearized{};
  std::size_t variables_reeliminated{};
  if (update.detail) {
    for (const auto& [key, status] : update.detail->variableStatus) {
      static_cast<void>(key);
      variables_relinearized += status.isRelinearized ? 1U : 0U;
      variables_reeliminated += status.isReeliminated ? 1U : 0U;
    }
  } else {
    variables_relinearized =
        update.variablesRelinearized <= maximum_variables ? update.variablesRelinearized : 0U;
    variables_reeliminated =
        update.variablesReeliminated <= maximum_variables ? update.variablesReeliminated : 0U;
  }
  summary.variables_relinearized += variables_relinearized;
  summary.variables_reeliminated += variables_reeliminated;
  if (variables_reeliminated > 0U && update.factorsRecalculated <= maximum_factors) {
    summary.factors_recalculated += update.factorsRecalculated;
  }
  if (update.cliques <= maximum_variables) {
    summary.cliques = update.cliques;
  }
}

[[nodiscard]] StateCorrectionMetrics stateCorrections(
    const gtsam::Values& from, const gtsam::Values& to,
    const std::vector<NavigationKeys>& navigation_keys,
    const std::vector<BoundedEtaKey>& eta_keys) {
  StateCorrectionMetrics maximum;
  for (const NavigationKeys& keys : navigation_keys) {
    const gtsam::Pose3& from_pose = from.at<gtsam::Pose3>(keys.pose);
    const gtsam::Pose3& to_pose = to.at<gtsam::Pose3>(keys.pose);
    const double rotation =
        gtsam::Rot3::Logmap(from_pose.rotation().between(to_pose.rotation())).norm();
    const double translation = (to_pose.translation() - from_pose.translation()).norm();
    const double velocity =
        (to.at<gtsam::Vector3>(keys.velocity) - from.at<gtsam::Vector3>(keys.velocity)).norm();
    const auto& from_bias = from.at<gtsam::imuBias::ConstantBias>(keys.bias);
    const auto& to_bias = to.at<gtsam::imuBias::ConstantBias>(keys.bias);
    maximum.translation_m = std::max(maximum.translation_m, translation);
    maximum.rotation_rad = std::max(maximum.rotation_rad, rotation);
    maximum.velocity_mps = std::max(maximum.velocity_mps, velocity);
    maximum.accelerometer_bias_mps2 =
        std::max(maximum.accelerometer_bias_mps2,
                 (to_bias.accelerometer() - from_bias.accelerometer()).norm());
    maximum.gyroscope_bias_radps = std::max(maximum.gyroscope_bias_radps,
                                            (to_bias.gyroscope() - from_bias.gyroscope()).norm());
  }
  for (const BoundedEtaKey& variable : eta_keys) {
    const gtsam_api::BoundedEtaValue from_eta = gtsam_api::decodeBoundedEta(
        from.at<double>(variable.key), variable.minimum_range_m, variable.maximum_range_m);
    const gtsam_api::BoundedEtaValue to_eta = gtsam_api::decodeBoundedEta(
        to.at<double>(variable.key), variable.minimum_range_m, variable.maximum_range_m);
    if (!std::isfinite(from_eta.eta) || !std::isfinite(to_eta.eta)) {
      maximum.visual_log_inverse_range = std::numeric_limits<double>::infinity();
      continue;
    }
    maximum.visual_log_inverse_range =
        std::max(maximum.visual_log_inverse_range, std::abs(to_eta.eta - from_eta.eta));
  }
  return maximum;
}

[[nodiscard]] bool converged(const StateCorrectionMetrics& correction,
                             const EffectiveConvergenceThresholds& thresholds) {
  return correction.translation_m <= thresholds.limits.translation_m &&
         correction.rotation_rad <= thresholds.limits.rotation_rad &&
         correction.velocity_mps <= thresholds.limits.velocity_mps &&
         correction.accelerometer_bias_mps2 <= thresholds.limits.accelerometer_bias_mps2 &&
         correction.gyroscope_bias_radps <= thresholds.limits.gyroscope_bias_radps &&
         correction.visual_log_inverse_range <= thresholds.limits.visual_log_inverse_range;
}

[[nodiscard]] LocalSolveReport solveReport(
    const gtsam::ISAM2Result& result, std::size_t navigation_states,
    std::size_t combined_imu_factors, std::size_t maximum_states,
    std::size_t marginalized_navigation_states = 0U, std::size_t marginal_factors_added = 0U,
    std::size_t factors_deleted_by_marginalization = 0U, bool window_cap_applied = false,
    const VisualSolveCounts& visual = {}, const NonlinearSolveSummary* nonlinear = nullptr) {
  LocalSolveReport report;
  report.navigation_states = navigation_states;
  report.joint_initial_priors = 1U;
  report.combined_imu_factors = combined_imu_factors;
  report.active_visual_landmarks = visual.active_landmarks;
  report.active_visual_factors = visual.active_factors;
  report.visual_landmarks_added = visual.landmarks_added;
  report.visual_factors_added = visual.factors_added;
  report.visual_factors_retired = visual.factors_retired;
  report.visual_landmarks_marginalized = visual.landmarks_marginalized;
  report.marginalized_navigation_states = marginalized_navigation_states;
  report.marginal_factors_added = marginal_factors_added;
  report.factors_deleted_by_marginalization = factors_deleted_by_marginalization;
  if (nonlinear != nullptr) {
    report.variables_relinearized = nonlinear->variables_relinearized;
    report.variables_reeliminated = nonlinear->variables_reeliminated;
    report.factors_recalculated = nonlinear->factors_recalculated;
    report.cliques = nonlinear->cliques;
    report.nonlinear_iterations = nonlinear->iterations;
    report.convergence_interval_duration_s = nonlinear->convergence.interval_duration_s;
    report.convergence_sigma_fraction = nonlinear->convergence.sigma_fraction;
    report.effective_translation_convergence_m = nonlinear->convergence.limits.translation_m;
    report.effective_rotation_convergence_rad = nonlinear->convergence.limits.rotation_rad;
    report.effective_velocity_convergence_mps = nonlinear->convergence.limits.velocity_mps;
    report.effective_accelerometer_bias_convergence_mps2 =
        nonlinear->convergence.limits.accelerometer_bias_mps2;
    report.effective_gyroscope_bias_convergence_radps =
        nonlinear->convergence.limits.gyroscope_bias_radps;
    report.effective_visual_log_inverse_range_convergence =
        nonlinear->convergence.limits.visual_log_inverse_range;
    report.last_iteration_translation_correction_m =
        nonlinear->last_iteration_correction.translation_m;
    report.last_iteration_rotation_correction_rad =
        nonlinear->last_iteration_correction.rotation_rad;
    report.last_iteration_velocity_correction_mps =
        nonlinear->last_iteration_correction.velocity_mps;
    report.last_iteration_accelerometer_bias_correction_mps2 =
        nonlinear->last_iteration_correction.accelerometer_bias_mps2;
    report.last_iteration_gyroscope_bias_correction_radps =
        nonlinear->last_iteration_correction.gyroscope_bias_radps;
    report.last_iteration_visual_log_inverse_range_correction =
        nonlinear->last_iteration_correction.visual_log_inverse_range;
    report.maximum_iteration_translation_correction_m = nonlinear->maximum_iteration_translation_m;
    report.maximum_iteration_rotation_correction_rad = nonlinear->maximum_iteration_rotation_rad;
    report.maximum_transaction_translation_correction_m =
        nonlinear->maximum_transaction_translation_m;
    report.maximum_transaction_rotation_correction_rad =
        nonlinear->maximum_transaction_rotation_rad;
    report.marginalization_translation_correction_m = nonlinear->marginalization_translation_m;
    report.marginalization_rotation_correction_rad = nonlinear->marginalization_rotation_rad;
    report.nonlinear_full_steps_rejected = nonlinear->full_steps_rejected;
    report.nonlinear_backtracking_trials = nonlinear->backtracking_trials;
    report.nonlinear_cauchy_directions_attempted = nonlinear->cauchy_directions_attempted;
    report.nonlinear_cauchy_steps_accepted = nonlinear->cauchy_steps_accepted;
    report.nonlinear_cauchy_backtracking_trials = nonlinear->cauchy_backtracking_trials;
    report.nonlinear_zero_step_terminations = nonlinear->zero_step_terminations;
    report.minimum_nonlinear_step_scale = nonlinear->minimum_step_scale;
    report.last_iteration_objective_change = nonlinear->last_objective_change;
    report.error_before = nonlinear->error_before;
    report.error_after = nonlinear->error_after;
  } else {
    report.variables_relinearized = result.variablesRelinearized;
    report.variables_reeliminated = result.variablesReeliminated;
    report.factors_recalculated = result.factorsRecalculated;
    report.cliques = result.cliques;
    report.nonlinear_iterations = 1U;
    if (result.errorBefore) {
      report.error_before = *result.errorBefore;
    }
    if (result.errorAfter) {
      report.error_after = *result.errorAfter;
    }
  }
  report.capacity = navigation_states >= maximum_states ? LocalGraphCapacityStatus::AtHardCap
                                                        : LocalGraphCapacityStatus::WithinHardCap;
  if (marginalized_navigation_states > 0U) {
    report.marginalization = window_cap_applied
                                 ? LocalGraphMarginalizationStatus::AppliedWindowCap
                                 : LocalGraphMarginalizationStatus::AppliedNominalLag;
  }
  return report;
}

void populateFactorBatchCounts(LocalSolveReport* report, std::size_t active_batches,
                               std::size_t active_factors, std::size_t batches_added = 0U,
                               std::size_t factors_added = 0U, std::size_t batches_removed = 0U,
                               std::size_t factors_removed = 0U, std::size_t batches_sealed = 0U,
                               std::size_t factors_sealed = 0U, std::size_t batches_finalized = 0U,
                               std::size_t factors_finalized = 0U) {
  report->active_factor_batches = active_batches;
  report->active_lidar_direct_batch_factors = active_factors;
  report->factor_batches_added = batches_added;
  report->lidar_direct_batch_factors_added = factors_added;
  report->factor_batches_removed = batches_removed;
  report->lidar_direct_batch_factors_removed = factors_removed;
  report->factor_batches_sealed = batches_sealed;
  report->lidar_direct_batch_factors_sealed = factors_sealed;
  report->factor_batches_finalized = batches_finalized;
  report->lidar_direct_batch_factors_finalized = factors_finalized;
}

}  // namespace

namespace gtsam_api {

gtsam::Pose3 toGtsamPose(const core::Pose3d& pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.so3().matrix()), pose.translation());
}

core::Matrix6d toGtsamPoseCovariance(const core::PoseCovariance& covariance) {
  core::Matrix6d converted;
  // GTSAM Pose3 tangents are [rotation, translation]; Meridian public pose
  // tangents are right [translation, rotation]. Preserve every cross term.
  for (Eigen::Index row = 0; row < 6; ++row) {
    const Eigen::Index meridian_row = row < 3 ? row + 3 : row - 3;
    for (Eigen::Index column = 0; column < 6; ++column) {
      const Eigen::Index meridian_column = column < 3 ? column + 3 : column - 3;
      converted(row, column) = covariance.matrix(meridian_row, meridian_column);
    }
  }
  return converted;
}

core::Pose3d fromGtsamPose(const gtsam::Pose3& pose) {
  return core::Pose3d(Sophus::SO3d(pose.rotation().matrix()), pose.translation());
}

gtsam::imuBias::ConstantBias toGtsamBias(const core::NavStateEstimate& state) {
  return gtsam::imuBias::ConstantBias(state.accel_bias, state.gyro_bias);
}

core::NavStateEstimate fromGtsamState(const gtsam::Pose3& pose, const gtsam::Vector3& velocity,
                                      const gtsam::imuBias::ConstantBias& bias) {
  core::NavStateEstimate state;
  state.T_odom_imu = fromGtsamPose(pose);
  state.velocity_odom = velocity;
  state.gyro_bias = bias.gyroscope();
  state.accel_bias = bias.accelerometer();
  return state;
}

Eigen::Matrix<double, 15, 15> toGtsamNavigationCovariance(const NavigationCovariance& covariance) {
  Eigen::Matrix<double, 15, 15> converted;
  for (Eigen::Index row = 0; row < 15; ++row) {
    for (Eigen::Index column = 0; column < 15; ++column) {
      converted(row, column) =
          covariance.matrix(kGtsamIndexToMeridian.at(row), kGtsamIndexToMeridian.at(column));
    }
  }
  return converted;
}

NavigationCovariance fromGtsamNavigationCovariance(
    const Eigen::Matrix<double, 15, 15>& covariance) {
  NavigationCovariance converted;
  for (Eigen::Index row = 0; row < 15; ++row) {
    for (Eigen::Index column = 0; column < 15; ++column) {
      converted.matrix(kGtsamIndexToMeridian.at(row), kGtsamIndexToMeridian.at(column)) =
          covariance(row, column);
    }
  }
  return converted;
}

NavigationCovariance jointNavigationCovarianceFromBayesTree(const gtsam::ISAM2& solver,
                                                            gtsam::Key pose_key,
                                                            gtsam::Key velocity_key,
                                                            gtsam::Key bias_key) {
  // Re-express the existing Bayes tree as its Gaussian conditionals.  This is
  // the exact current incremental posterior, not a relinearization of stored
  // nonlinear factors. In particular, raw-point direct-registration factors
  // are never visited by covariance extraction.
  gtsam::GaussianFactorGraph linear_posterior;
  std::vector<gtsam::ISAM2::sharedClique> pending(solver.roots().begin(), solver.roots().end());
  while (!pending.empty()) {
    const gtsam::ISAM2::sharedClique clique = pending.back();
    pending.pop_back();
    if (!clique || !clique->conditional()) {
      throw std::runtime_error("iSAM2 Bayes tree contains an empty clique");
    }
    linear_posterior.push_back(clique->conditional());
    pending.insert(pending.end(), clique->children.begin(), clique->children.end());
  }
  if (linear_posterior.empty()) {
    throw std::runtime_error("iSAM2 Bayes tree is empty");
  }

  const gtsam::Ordering ordering{pose_key, velocity_key, bias_key};
  const auto joint = linear_posterior.marginalMultifrontalBayesNet(ordering, gtsam::EliminateQR);
  if (!joint || joint->empty()) {
    throw std::runtime_error("iSAM2 Bayes-tree joint marginal is empty");
  }
  const auto matrix_and_rhs = joint->matrix(ordering);
  const Eigen::MatrixXd& square_root = matrix_and_rhs.first;
  constexpr Eigen::Index kJointDimension = 15;
  if (square_root.rows() != kJointDimension || square_root.cols() != kJointDimension ||
      !square_root.allFinite()) {
    throw std::runtime_error("X/V/B Bayes-tree joint has an invalid square-root dimension");
  }

  // The requested ordering need not coincide with the Bayes net's native
  // triangular ordering, so use a rank-revealing QR on the small final joint.
  // For whitened system A, Cov = A^-1 A^-T.
  const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> decomposition(square_root);
  if (decomposition.rank() != kJointDimension) {
    throw std::runtime_error("X/V/B Bayes-tree joint is rank deficient");
  }
  const Eigen::MatrixXd inverse_square_root =
      decomposition.solve(Eigen::MatrixXd::Identity(kJointDimension, kJointDimension));
  if (!inverse_square_root.allFinite()) {
    throw std::runtime_error("X/V/B Bayes-tree covariance solve produced non-finite values");
  }
  Eigen::Matrix<double, 15, 15> covariance = inverse_square_root * inverse_square_root.transpose();
  covariance = 0.5 * (covariance + covariance.transpose());

  if (!covariance.allFinite() ||
      Eigen::LLT<Eigen::Matrix<double, 15, 15>>(covariance).info() != Eigen::Success) {
    throw std::runtime_error(
        "assembled X/V/B Bayes-tree covariance is not finite positive definite");
  }
  return fromGtsamNavigationCovariance(covariance);
}

[[nodiscard]] core::PoseCovariance poseCovarianceFromBayesTree(const gtsam::ISAM2& solver,
                                                               PoseKey pose_key) {
  // ISAM2::marginalCovariance consumes the candidate's current Bayes tree; it
  // does not relinearize the nonlinear factor graph. GTSAM Pose3 covariance is
  // [R,P], while the public Meridian tangent is right [P,R]. Permute all four
  // blocks explicitly so rotation/translation cross terms cannot be lost.
  const gtsam::Matrix covariance = solver.marginalCovariance(pose_key);
  if (covariance.rows() != 6 || covariance.cols() != 6 || !covariance.allFinite()) {
    throw std::runtime_error("pose Bayes-tree marginal has an invalid dimension or value");
  }
  core::PoseCovariance converted;
  converted.matrix.topLeftCorner<3, 3>() = covariance.bottomRightCorner<3, 3>();
  converted.matrix.topRightCorner<3, 3>() = covariance.bottomLeftCorner<3, 3>();
  converted.matrix.bottomLeftCorner<3, 3>() = covariance.topRightCorner<3, 3>();
  converted.matrix.bottomRightCorner<3, 3>() = covariance.topLeftCorner<3, 3>();
  converted.matrix = 0.5 * (converted.matrix + converted.matrix.transpose());
  return converted;
}

Eigen::Matrix<double, 6, 1> toGtsamPoseTangent(
    const Eigen::Matrix<double, 6, 1>& meridian_tangent) {
  Eigen::Matrix<double, 6, 1> result;
  result.head<3>() = meridian_tangent.tail<3>();
  result.tail<3>() = meridian_tangent.head<3>();
  return result;
}

Eigen::Matrix<double, 6, 1> fromGtsamPoseTangent(const Eigen::Matrix<double, 6, 1>& gtsam_tangent) {
  Eigen::Matrix<double, 6, 1> result;
  result.head<3>() = gtsam_tangent.tail<3>();
  result.tail<3>() = gtsam_tangent.head<3>();
  return result;
}

}  // namespace gtsam_api

struct LocalGraph::Impl {
  Impl(LocalGraphConfig graph_config, std::shared_ptr<LocalPipelineTimingRecorder> input_timing)
      : config(std::move(graph_config)),
        solver(std::make_unique<gtsam_api::CandidateIsolatedISAM2>(solverParams(config))),
        preintegration_params(preintegrationParams(config)),
        timing(std::move(input_timing)) {}

  LocalGraphConfig config;
  std::unique_ptr<gtsam_api::CandidateIsolatedISAM2> solver;
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> preintegration_params;
  std::shared_ptr<LocalPipelineTimingRecorder> timing;
  struct NavigationRecord {
    core::StateId state;
    core::FusionTime time;
    std::size_t graph_index{};
  };
  struct VisualLandmarkRecord {
    VisualLandmarkId landmark;
    VisualTrackId track;
    EtaKey eta_key{};
    VisualObservationRef anchor;
    double minimum_range_m{};
    double maximum_range_m{};
  };
  struct VisualFactorRecord {
    core::FactorId factor;
    VisualLandmarkId landmark;
    gtsam::FactorIndex solver_index{};
    core::StateId anchor_state;
    core::StateId observer_state;
  };
  struct FactorBatchRecord {
    FactorBatchProvenance provenance;
    std::vector<gtsam::FactorIndex> solver_indices;
    std::vector<gtsam::KeyVector> solver_keys;
  };
  struct SensorBatchSequence {
    core::FactorBatchId latest_batch;
    core::SensorRecoveryEpoch recovery_epoch;
    std::uint64_t health_transition_sequence{};
    core::FusionTime health_assessed_at;
    core::FusionTime produced_at;
  };
  std::unordered_set<std::uint64_t> state_ids;
  std::deque<NavigationRecord> navigation_records;
  std::map<std::uint64_t, VisualLandmarkRecord> visual_landmarks;
  std::map<std::uint64_t, VisualFactorRecord> visual_factors;
  std::map<SensorFactorBatchRef, FactorBatchRecord> active_factor_batches;
  std::deque<SensorFactorBatchRef> removable_factor_batches;
  std::deque<FactorBatchProvenance> terminal_factor_batches;
  std::map<core::SensorInstanceId, SensorBatchSequence> sensor_batch_sequences;
  std::size_t navigation_states{};
  std::size_t next_graph_index{};
  std::size_t combined_imu_factors{};
  std::size_t active_lidar_direct_factors{};
  std::size_t terminal_factor_batch_records_evicted{};
  core::OdomEpoch odom_epoch;
  core::StateId latest_state;
  core::FusionTime latest_time;
  core::LocalGraphRevision latest_revision;
  std::shared_ptr<const LocalGraphCommit> latest_commit;
};

LocalGraph::LocalGraph(LocalGraphConfig config) : LocalGraph(std::move(config), {}) {}

LocalGraph::LocalGraph(LocalGraphConfig config, std::shared_ptr<LocalPipelineTimingRecorder> timing)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(timing))) {}

LocalGraph::~LocalGraph() = default;
LocalGraph::LocalGraph(LocalGraph&&) noexcept = default;
LocalGraph& LocalGraph::operator=(LocalGraph&&) noexcept = default;

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::initialize(
    LocalGraphInitialization initialization) {
  using Result = core::Result<LocalGraphCommit, LocalGraphError>;
  core::PipelineWorkIdentity work;
  if (initialization.state.valid()) {
    work.state = initialization.state;
  }
  detail::LocalPipelineTimingScope transaction_timing(
      impl_->timing, LocalPipelineTimingStage::GraphTransactionUpdate, work);
  if (impl_->latest_commit) {
    return Result::failure(graphError(LocalGraphErrorCode::AlreadyInitialized,
                                      "a local graph can be initialized only once per odom epoch"));
  }
  if (!validConfig(impl_->config)) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InvalidConfig,
                   "local graph limits, thresholds, gravity, and IMU densities must be "
                   "finite and positive"));
  }
  if (!initialization.odom_epoch.valid() || !initialization.state.valid() ||
      !finiteState(initialization.estimate)) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InvalidInitialization,
                   "initial odom epoch, state identity, or navigation estimate is invalid"));
  }
  if (!validCovariance(initialization.covariance)) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InvalidCovariance,
                   "initial covariance must be finite, symmetric positive definite, and "
                   "ordered [R,V,P,Bg,Ba]"));
  }
  if (!validLineage(initialization.lineage)) {
    return Result::failure(graphError(LocalGraphErrorCode::InvalidLineage,
                                      "initial state ObservationLineage is invalid"));
  }

  try {
    const NavigationKeys keys = navigationKeys(0U);
    const gtsam::Pose3 pose = gtsam_api::toGtsamPose(initialization.estimate.T_odom_imu);
    const gtsam::imuBias::ConstantBias bias = gtsam_api::toGtsamBias(initialization.estimate);
    const auto covariance = gtsam_api::toGtsamNavigationCovariance(initialization.covariance);

    gtsam::NonlinearFactorGraph new_factors;
    new_factors.emplace_shared<JointNavigationPriorFactor>(
        keys.pose, keys.velocity, keys.bias, pose, initialization.estimate.velocity_odom, bias,
        covariance);
    gtsam::Values new_values;
    new_values.insert(keys.pose, pose);
    new_values.insert(keys.velocity, initialization.estimate.velocity_odom);
    new_values.insert(keys.bias, bias);

    auto candidate = std::make_unique<gtsam_api::CandidateIsolatedISAM2>(*impl_->solver);
    const gtsam::ISAM2Result update = candidate->update(new_factors, new_values);
    const gtsam::Values estimate = candidate->calculateEstimate();
    const core::NavStateEstimate state = gtsam_api::fromGtsamState(
        estimate.at<gtsam::Pose3>(keys.pose), estimate.at<gtsam::Vector3>(keys.velocity),
        estimate.at<gtsam::imuBias::ConstantBias>(keys.bias));
    if (!finiteState(state)) {
      return Result::failure(
          graphError(LocalGraphErrorCode::NonFiniteEstimate,
                     "initial graph solve produced a non-finite navigation state"));
    }
    NavigationCovariance marginal;
    try {
      marginal = gtsam_api::jointNavigationCovarianceFromBayesTree(*candidate, keys.pose,
                                                                   keys.velocity, keys.bias);
    } catch (const std::exception& exception) {
      return Result::failure(
          graphError(LocalGraphErrorCode::MarginalCovarianceFailure,
                     std::string("initial joint marginal failed: ") + exception.what()));
    }
    if (!marginal.matrix.allFinite()) {
      return Result::failure(graphError(LocalGraphErrorCode::MarginalCovarianceFailure,
                                        "initial joint marginal is non-finite"));
    }

    LocalGraphCommit commit;
    commit.odom_epoch = initialization.odom_epoch;
    commit.revision = core::LocalGraphRevision{1U};
    commit.parent = core::LocalGraphRevision{};
    commit.state = initialization.state;
    commit.state_time = initialization.exact_time;
    commit.estimate = state;
    commit.covariance = marginal;
    commit.navigation_poses.push_back(
        LocalGraphPoseSnapshot{initialization.state, initialization.exact_time, state.T_odom_imu});
    commit.solve = solveReport(update, 1U, 0U, impl_->config.maximum_navigation_states);
    commit.lineage = std::move(initialization.lineage);
    auto stored_commit = std::make_shared<const LocalGraphCommit>(commit);

    impl_->solver.swap(candidate);
    impl_->state_ids.insert(initialization.state.value());
    impl_->navigation_records.push_back(
        Impl::NavigationRecord{initialization.state, initialization.exact_time, 0U});
    impl_->navigation_states = 1U;
    impl_->next_graph_index = 1U;
    impl_->combined_imu_factors = 0U;
    impl_->odom_epoch = initialization.odom_epoch;
    impl_->latest_state = initialization.state;
    impl_->latest_time = initialization.exact_time;
    impl_->latest_revision = commit.revision;
    impl_->latest_commit.swap(stored_commit);
    transaction_timing.finish();
    return Result::success(std::move(commit));
  } catch (const std::exception& exception) {
    return Result::failure(
        graphError(LocalGraphErrorCode::SolverFailure,
                   std::string("initial iSAM2 QR transaction failed: ") + exception.what()));
  }
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::appendImuKnot(ImuKnotAppend append) {
  SensorKnotAppend transaction;
  transaction.navigation = std::move(append);
  return appendSensorKnot(std::move(transaction));
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::appendSensorKnot(
    SensorKnotAppend append) {
  return appendNavigationKnot(std::move(append));
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::insertFactorBatch(
    LidarDirectFactorBatch batch) {
  return applyFactorBatchTransaction(std::move(batch), std::nullopt);
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::removeFactorBatches(
    FactorBatchRemovalRequest request) {
  return applyFactorBatchTransaction(std::nullopt, std::move(request));
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::applyFactorBatchTransaction(
    std::optional<LidarDirectFactorBatch> insertion,
    std::optional<FactorBatchRemovalRequest> removal) {
  using Result = core::Result<LocalGraphCommit, LocalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NotInitialized, "initialize the local graph first"));
  }
  if (insertion.has_value() == removal.has_value()) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InvalidFactorBatch,
                   "factor-only transaction requires exactly one insertion or removal request"));
  }

  SensorFactorBatchRef diagnostic_ref;
  if (insertion) {
    const core::FactorBatchMetadata& metadata = factorBatchMetadata(*insertion);
    diagnostic_ref = SensorFactorBatchRef{metadata.sensor, metadata.batch_id};
  } else if (!removal->batches.empty()) {
    diagnostic_ref = removal->batches.front();
  }
  core::PipelineWorkIdentity work;
  work.state = impl_->latest_state;
  detail::LocalPipelineTimingScope transaction_timing(
      impl_->timing, LocalPipelineTimingStage::GraphTransactionUpdate, work);

  std::vector<SensorFactorBatchRef> ordered_removals;
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::FactorIndices remove_factor_indices;
  std::vector<gtsam::KeyVector> inserted_solver_keys;
  std::vector<DirectLidarPairReport> inserted_pair_reports;
  std::optional<DirectLidarFinalizedMapReport> inserted_finalized_map_report;
  core::Duration convergence_interval{1LL};
  std::size_t factors_added = 0U;
  std::size_t factors_removed = 0U;
  const auto factor_transaction_error = [&](LocalGraphErrorCode code, std::string detail,
                                            std::optional<LocalSolveReport> rejected_solve =
                                                std::nullopt) {
    LocalGraphError error = factorBatchGraphError(code, std::move(detail), diagnostic_ref);
    if (insertion && inserted_pair_reports.size() == insertion->pairs.size() &&
        inserted_finalized_map_report.has_value() == insertion->finalized_map.has_value()) {
      error.lidar_registration = insertion->registration_report;
      error.lidar_pairs = inserted_pair_reports;
      error.lidar_finalized_map = inserted_finalized_map_report;
    }
    error.rejected_solve = std::move(rejected_solve);
    return error;
  };

  if (insertion) {
    const auto failure = validateLidarDirectBatchPayload(*insertion, impl_->config);
    if (failure) {
      return Result::failure(factorBatchGraphError(failure->code, failure->detail, diagnostic_ref));
    }
    const core::FactorBatchMetadata& metadata = factorBatchMetadata(*insertion);
    if (metadata.odom_epoch != impl_->odom_epoch) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::FactorBatchEpochMismatch,
          "FactorBatch odom epoch does not match the live local graph", diagnostic_ref));
    }
    if (metadata.health.state != core::SensorHealthState::Active) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::FactorBatchHealthUnavailable,
          "only an authoritative Active sensor batch may mutate the live graph; Suspect, Failed, "
          "and Recovering results remain shadow-only",
          diagnostic_ref));
    }
    if (impl_->active_factor_batches.size() >= impl_->config.maximum_active_factor_batches) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::FactorBatchCapacity,
          "active FactorBatch journal reached its configured hard bound", diagnostic_ref));
    }
    if (impl_->active_factor_batches.contains(diagnostic_ref) ||
        std::any_of(impl_->terminal_factor_batches.begin(), impl_->terminal_factor_batches.end(),
                    [&](const FactorBatchProvenance& record) {
                      const core::FactorBatchMetadata& existing = factorBatchMetadata(record.batch);
                      return existing.sensor == diagnostic_ref.sensor &&
                             existing.batch_id == diagnostic_ref.batch_id;
                    })) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::DuplicateFactorBatch,
          "FactorBatch identity has already entered this graph epoch", diagnostic_ref));
    }
    const auto conflicts_with_primary_lineage = [&](const FactorBatchProvenance& record) {
      return record.status != FactorBatchJournalStatus::Removed &&
             primaryLineageOverlaps(metadata.lineage, factorBatchMetadata(record.batch).lineage);
    };
    const bool active_lineage_conflict = std::any_of(
        impl_->active_factor_batches.begin(), impl_->active_factor_batches.end(),
        [&](const auto& entry) { return conflicts_with_primary_lineage(entry.second.provenance); });
    const bool finalized_lineage_conflict =
        std::any_of(impl_->terminal_factor_batches.begin(), impl_->terminal_factor_batches.end(),
                    conflicts_with_primary_lineage);
    if (active_lineage_conflict || finalized_lineage_conflict) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::FactorBatchLineageConflict,
          "FactorBatch primary observations overlap evidence that is already active or embedded "
          "in a marginal prior",
          diagnostic_ref));
    }
    if (const auto sequence = impl_->sensor_batch_sequences.find(metadata.sensor);
        sequence != impl_->sensor_batch_sequences.end()) {
      const Impl::SensorBatchSequence& latest = sequence->second;
      if (metadata.batch_id <= latest.latest_batch ||
          metadata.timing.produced_at <= latest.produced_at ||
          metadata.health.assessed_at <= latest.health_assessed_at ||
          metadata.health.recovery_epoch < latest.recovery_epoch ||
          (metadata.health.recovery_epoch == latest.recovery_epoch &&
           metadata.health.transition_sequence < latest.health_transition_sequence)) {
        return Result::failure(factorBatchGraphError(
            LocalGraphErrorCode::StaleFactorBatch,
            "FactorBatch identity, production time, health assessment, transition, or recovery "
            "epoch is stale for this sensor",
            diagnostic_ref));
      }
      const bool recovery_epoch_advanced = metadata.health.recovery_epoch > latest.recovery_epoch;
      if (recovery_epoch_advanced &&
          (metadata.health.recovery_epoch.value() != latest.recovery_epoch.value() + 1U ||
           metadata.health.transition_sequence <= latest.health_transition_sequence ||
           metadata.health.state != core::SensorHealthState::Active)) {
        return Result::failure(factorBatchGraphError(
            LocalGraphErrorCode::FactorBatchEpochMismatch,
            "a new recovery epoch must advance exactly once with a newer Active health "
            "transition",
            diagnostic_ref));
      }
    }

    const std::size_t insertion_factor_count = factorBatchFactorCount(*insertion);
    new_factors.reserve(insertion_factor_count);
    inserted_solver_keys.reserve(insertion_factor_count);
    inserted_pair_reports.reserve(insertion_factor_count);
    const auto append_lidar_factor = [&](const LidarDirectFactorBatch& batch,
                                         const LidarDirectFactorPairSpec& pair,
                                         std::size_t index) -> std::optional<LocalGraphError> {
      const auto target = std::find_if(
          impl_->navigation_records.begin(), impl_->navigation_records.end(),
          [&](const Impl::NavigationRecord& record) { return record.state == pair.target_state; });
      const auto source = std::find_if(
          impl_->navigation_records.begin(), impl_->navigation_records.end(),
          [&](const Impl::NavigationRecord& record) { return record.state == batch.source_state; });
      if (target == impl_->navigation_records.end() || source == impl_->navigation_records.end()) {
        return factorBatchGraphError(
            LocalGraphErrorCode::FactorBatchStateUnavailable,
            "direct LiDAR factor endpoint is not in the live lag window at batch index " +
                std::to_string(index),
            diagnostic_ref);
      }
      if (target->time != pair.target_time || source->time != batch.source_time) {
        return factorBatchGraphError(LocalGraphErrorCode::FactorBatchReferenceMismatch,
                                     "direct LiDAR factor endpoint time does not match the exact "
                                     "live state at batch index " +
                                         std::to_string(index),
                                     diagnostic_ref);
      }
      const NavigationKeys target_keys = navigationKeys(target->graph_index);
      const NavigationKeys source_keys = navigationKeys(source->graph_index);
      const gtsam::KeyVector keys{target_keys.pose, source_keys.pose};
      inserted_solver_keys.push_back(keys);
      auto factor = boost::make_shared<gtsam_api::DirectLidarFactor>(
          target_keys.pose, source_keys.pose, pair.snapshot, batch.registration,
          pair.information_scale);
      inserted_pair_reports.push_back(DirectLidarPairReport{
          pair.target_state, pair.target_time, pair.snapshot->targetSweep(),
          pair.snapshot->sourceSweep(), pair.snapshot->sourcePointCount(),
          pair.snapshot->rows().size(), pair.snapshot->sourceRowsExcludedByOwnership(),
          pair.snapshot->candidateVoxelLookups(), pair.snapshot->candidatePointsExamined(),
          pair.information_scale, factor->admissionInformation(), pair.snapshot->checksum()});
      new_factors.push_back(std::move(factor));
      return std::nullopt;
    };
    std::optional<LocalGraphError> construction_error;
    for (std::size_t index = 0U; index < insertion->pairs.size() && !construction_error; ++index) {
      construction_error = append_lidar_factor(*insertion, insertion->pairs[index], index);
    }
    if (insertion->finalized_map && !construction_error) {
      const LidarFinalizedMapFactorSpec& spec = *insertion->finalized_map;
      const auto source =
          std::find_if(impl_->navigation_records.begin(), impl_->navigation_records.end(),
                       [&](const Impl::NavigationRecord& record) {
                         return record.state == insertion->source_state;
                       });
      if (source == impl_->navigation_records.end()) {
        construction_error = factorBatchGraphError(
            LocalGraphErrorCode::FactorBatchStateUnavailable,
            "direct finalized-map LiDAR source endpoint is not in the live lag window",
            diagnostic_ref);
      } else if (source->time != insertion->source_time) {
        construction_error = factorBatchGraphError(
            LocalGraphErrorCode::FactorBatchReferenceMismatch,
            "direct finalized-map LiDAR source time does not match the exact live state",
            diagnostic_ref);
      } else {
        const NavigationKeys source_keys = navigationKeys(source->graph_index);
        inserted_solver_keys.push_back(gtsam::KeyVector{source_keys.pose});
        auto factor = boost::make_shared<gtsam_api::DirectLidarFactor>(
            source_keys.pose, spec.snapshot, insertion->registration, spec.information_scale);
        const auto owner_inflation = lidarFinalizedMapOwnerPoseCovarianceInflation(spec.snapshot);
        if (!owner_inflation) {
          construction_error = factorBatchGraphError(
              LocalGraphErrorCode::InvalidFactorBatch,
              "direct finalized-map LiDAR owner covariance inflation became unverifiable",
              diagnostic_ref);
          inserted_solver_keys.pop_back();
        } else {
          inserted_finalized_map_report =
              DirectLidarFinalizedMapReport{spec.snapshot->sourceSweep(),
                                            spec.snapshot->sourcePointCount(),
                                            spec.snapshot->rows().size(),
                                            spec.snapshot->sourceRowsExcludedByOwnership(),
                                            spec.snapshot->candidateVoxelLookups(),
                                            spec.snapshot->candidatePointsExamined(),
                                            spec.snapshot->uniqueOwnerCount(),
                                            spec.snapshot->mapOdomEpoch(),
                                            spec.snapshot->mapSensor(),
                                            spec.snapshot->mapVersion(),
                                            spec.snapshot->mapChecksum(),
                                            owner_inflation.value(),
                                            spec.configured_correlation_inflation_floor,
                                            std::max(owner_inflation.value(),
                                                     spec.configured_correlation_inflation_floor),
                                            spec.information_scale,
                                            factor->admissionInformation(),
                                            spec.snapshot->checksum()};
          new_factors.push_back(std::move(factor));
        }
      }
    }
    if (construction_error) {
      return Result::failure(std::move(*construction_error));
    }
    convergence_interval = metadata.timing.support.duration();
    factors_added = insertion_factor_count;
  } else {
    if (!validRemovalReason(removal->reason) || removal->batches.empty() ||
        removal->batches.size() > impl_->config.maximum_factor_batches_per_removal_transaction) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::InvalidFactorBatch,
          "factor removal reason or bounded batch count is invalid", diagnostic_ref));
    }
    ordered_removals = removal->batches;
    std::sort(ordered_removals.begin(), ordered_removals.end());
    if (std::adjacent_find(ordered_removals.begin(), ordered_removals.end()) !=
        ordered_removals.end()) {
      return Result::failure(
          factorBatchGraphError(LocalGraphErrorCode::FactorBatchRemovalUnavailable,
                                "factor removal batch identities must be unique", diagnostic_ref));
    }
    const gtsam::NonlinearFactorGraph& live_factors = impl_->solver->getFactorsUnsafe();
    std::int64_t maximum_interval_ns = 1LL;
    for (const SensorFactorBatchRef& batch_ref : ordered_removals) {
      const auto record = impl_->active_factor_batches.find(batch_ref);
      if (record == impl_->active_factor_batches.end() || !record->second.provenance.removable) {
        return Result::failure(factorBatchGraphError(
            LocalGraphErrorCode::FactorBatchRemovalUnavailable,
            "FactorBatch is unknown, sealed, or already terminal and cannot be retracted",
            batch_ref));
      }
      if (record->second.solver_indices.size() != record->second.solver_keys.size() ||
          record->second.solver_indices.size() !=
              factorBatchFactorCount(record->second.provenance.batch)) {
        return Result::failure(
            factorBatchGraphError(LocalGraphErrorCode::FactorBatchRemovalUnavailable,
                                  "FactorBatch solver provenance is incomplete", batch_ref));
      }
      maximum_interval_ns =
          std::max(maximum_interval_ns, factorBatchMetadata(record->second.provenance.batch)
                                            .timing.support.duration()
                                            .nanoseconds);
      for (std::size_t index = 0U; index < record->second.solver_indices.size(); ++index) {
        const gtsam::FactorIndex solver_index = record->second.solver_indices[index];
        if (solver_index >= live_factors.size() || !live_factors.at(solver_index) ||
            live_factors.at(solver_index)->keys() != record->second.solver_keys[index]) {
          return Result::failure(factorBatchGraphError(
              LocalGraphErrorCode::FactorBatchRemovalUnavailable,
              "FactorBatch is no longer represented by its exact live solver slots", batch_ref));
        }
        remove_factor_indices.push_back(solver_index);
      }
      factors_removed += record->second.solver_indices.size();
    }
    std::sort(remove_factor_indices.begin(), remove_factor_indices.end());
    if (std::adjacent_find(remove_factor_indices.begin(), remove_factor_indices.end()) !=
        remove_factor_indices.end()) {
      return Result::failure(
          factorBatchGraphError(LocalGraphErrorCode::FactorBatchRemovalUnavailable,
                                "factor removal batches alias a solver slot", diagnostic_ref));
    }
    convergence_interval = core::Duration{maximum_interval_ns};
  }

  try {
    const gtsam::Values transaction_seed = impl_->solver->calculateEstimate();
    auto candidate = std::make_unique<gtsam_api::CandidateIsolatedISAM2>(*impl_->solver);
    gtsam::ISAM2UpdateParams update_parameters;
    update_parameters.removeFactorIndices = remove_factor_indices;
    update_parameters.force_relinearize = true;
    // A factor-only transaction attaches a new residual to states whose
    // published estimate may differ materially from iSAM2's retained
    // linearization point.  `force_relinearize` only bypasses relinearizeSkip;
    // it still applies the configured delta thresholds.  A newly attached
    // factor would therefore be linearized around a stale theta while the
    // complete-objective gate is (correctly) evaluated at calculateEstimate().
    // Force a complete delta solve here so GTSAM relinearizes every non-fixed
    // state at the committed estimate before solving the isolated candidate.
    update_parameters.forceFullSolve = true;
    const gtsam::ISAM2Result update =
        candidate->update(new_factors, gtsam::Values{}, update_parameters);
    if (update.newFactorsIndices.size() != new_factors.size()) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::SolverFailure,
          "iSAM2 did not return one deterministic slot for every FactorBatch factor"));
    }
    if (candidate->candidateCacheSetStamp().stateful_factors != 0U) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::SolverFailure,
          "factor-only transactions require a stateless nonlinear factor graph"));
    }

    const double initial_error = candidate->getFactorsUnsafe().error(transaction_seed);
    if (!std::isfinite(initial_error)) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::NonFiniteEstimate,
          "factor-only candidate objective is non-finite at the committed seed"));
    }
    std::vector<NavigationKeys> navigation_keys;
    navigation_keys.reserve(impl_->navigation_records.size());
    for (const Impl::NavigationRecord& record : impl_->navigation_records) {
      navigation_keys.push_back(navigationKeys(record.graph_index));
    }
    std::vector<BoundedEtaKey> eta_keys;
    eta_keys.reserve(impl_->visual_landmarks.size());
    for (const auto& [landmark_id, landmark] : impl_->visual_landmarks) {
      static_cast<void>(landmark_id);
      eta_keys.push_back(
          BoundedEtaKey{landmark.eta_key, landmark.minimum_range_m, landmark.maximum_range_m});
    }

    NonlinearSolveSummary nonlinear_summary;
    nonlinear_summary.convergence =
        effectiveConvergenceThresholds(impl_->config, convergence_interval);
    nonlinear_summary.error_before = initial_error;
    const std::size_t candidate_variable_count = 3U * navigation_keys.size() + eta_keys.size();
    const std::size_t candidate_factor_count = candidate->getFactorsUnsafe().size();
    const auto rejected_solve_report = [&]() {
      const VisualSolveCounts visual_counts{impl_->visual_landmarks.size(),
                                            impl_->visual_factors.size()};
      LocalSolveReport report =
          solveReport(update, impl_->navigation_states, impl_->combined_imu_factors,
                      impl_->config.maximum_navigation_states, 0U, 0U, 0U, false, visual_counts,
                      &nonlinear_summary);
      populateFactorBatchCounts(&report, impl_->active_factor_batches.size(),
                                impl_->active_lidar_direct_factors, 0U, 0U, 0U, 0U);
      return report;
    };
    gtsam::Values previous_iteration = transaction_seed;
    gtsam::Values settled_estimate;
    bool solve_converged = false;
    double settled_error = initial_error;
    double previous_error = initial_error;
    for (std::size_t iteration = 0U; iteration < impl_->config.maximum_nonlinear_iterations;
         ++iteration) {
      if (iteration == 0U) {
        accumulateUpdate(update, candidate_variable_count, candidate_factor_count,
                         nonlinear_summary);
      } else {
        gtsam::ISAM2UpdateParams iteration_parameters;
        iteration_parameters.force_relinearize = true;
        iteration_parameters.forceFullSolve = true;
        const gtsam::ISAM2Result iteration_update =
            candidate->update(gtsam::NonlinearFactorGraph{}, gtsam::Values{}, iteration_parameters);
        accumulateUpdate(iteration_update, candidate_variable_count, candidate_factor_count,
                         nonlinear_summary);
      }
      gtsam::Values full_step_estimate = candidate->calculateEstimate();
      const double full_step_error = candidate->getFactorsUnsafe().error(full_step_estimate);
      const double objective_stabilization_tolerance =
          objectiveConvergenceTolerance(impl_->config, previous_error);
      const bool full_step_is_physically_converged =
          std::isfinite(full_step_error) && full_step_error > previous_error &&
          full_step_error - previous_error <= objective_stabilization_tolerance &&
          converged(
              stateCorrections(previous_iteration, full_step_estimate, navigation_keys, eta_keys),
              nonlinear_summary.convergence);
      gtsam_api::CandidateGlobalizationResult globalized = candidate->globalizeFullStep(
          previous_iteration, previous_error, std::move(full_step_estimate), full_step_error,
          impl_->config.maximum_nonlinear_backtracking_steps,
          impl_->config.nonlinear_backtracking_reduction, full_step_is_physically_converged,
          objective_stabilization_tolerance);
      settled_estimate = std::move(globalized.estimate);
      settled_error = globalized.error;
      const StateCorrectionMetrics iteration_correction =
          stateCorrections(previous_iteration, settled_estimate, navigation_keys, eta_keys);
      const StateCorrectionMetrics transaction_correction =
          stateCorrections(transaction_seed, settled_estimate, navigation_keys, eta_keys);
      nonlinear_summary.iterations = iteration + 1U;
      nonlinear_summary.last_iteration_correction = iteration_correction;
      nonlinear_summary.error_after = settled_error;
      nonlinear_summary.full_steps_rejected += globalized.rejected_full_step ? 1U : 0U;
      nonlinear_summary.backtracking_trials += globalized.backtracking_trials;
      nonlinear_summary.cauchy_directions_attempted +=
          globalized.cauchy_direction_attempted ? 1U : 0U;
      nonlinear_summary.cauchy_steps_accepted += globalized.cauchy_step_accepted ? 1U : 0U;
      nonlinear_summary.cauchy_backtracking_trials += globalized.cauchy_backtracking_trials;
      nonlinear_summary.zero_step_terminations += globalized.zero_step ? 1U : 0U;
      nonlinear_summary.minimum_step_scale =
          std::min(nonlinear_summary.minimum_step_scale, globalized.step_scale);
      nonlinear_summary.last_objective_change = settled_error - previous_error;
      if (iteration_correction.finite() && transaction_correction.finite()) {
        nonlinear_summary.maximum_iteration_translation_m = std::max(
            nonlinear_summary.maximum_iteration_translation_m, iteration_correction.translation_m);
        nonlinear_summary.maximum_iteration_rotation_rad = std::max(
            nonlinear_summary.maximum_iteration_rotation_rad, iteration_correction.rotation_rad);
        nonlinear_summary.maximum_transaction_translation_m = transaction_correction.translation_m;
        nonlinear_summary.maximum_transaction_rotation_rad = transaction_correction.rotation_rad;
      }
      const detail::CandidateGateDecision gate = detail::evaluateCandidateGate(
          detail::CandidateGateInput{
              initial_error, settled_error, transaction_correction.translation_m,
              transaction_correction.rotation_rad,
              iteration_correction.finite() && transaction_correction.finite(),
              detail::CandidateGatePhase::NonlinearIteration},
          detail::CandidateGateLimits{
              impl_->config.maximum_transaction_translation_correction_m,
              impl_->config.maximum_transaction_rotation_correction_rad,
              impl_->config.complete_objective_nonsmooth_absolute_allowance,
              impl_->config.complete_objective_nonsmooth_relative_allowance});
      if (gate == detail::CandidateGateDecision::NonFinite) {
        return Result::failure(factor_transaction_error(
            LocalGraphErrorCode::NonFiniteEstimate,
            "factor-only candidate produced a non-finite objective or state correction",
            rejected_solve_report()));
      }
      if (gate == detail::CandidateGateDecision::PoseCorrectionLimit) {
        return Result::failure(factor_transaction_error(
            LocalGraphErrorCode::PoseCorrectionLimit,
            "factor-only candidate exceeded the configured graph pose-correction gate",
            rejected_solve_report()));
      }
      const bool objective_accepts_convergence = objectiveAcceptsPhysicalConvergence(
          nonlinear_summary.last_objective_change,
          objectiveConvergenceTolerance(impl_->config, previous_error));
      if (converged(iteration_correction, nonlinear_summary.convergence) &&
          objective_accepts_convergence) {
        solve_converged = true;
        break;
      }
      previous_iteration = settled_estimate;
      previous_error = settled_error;
    }
    if (!solve_converged) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::NonlinearConvergenceFailure,
          "factor-only candidate reached the configured nonlinear iteration bound",
          rejected_solve_report()));
    }
    const detail::CandidateGateDecision final_gate = detail::evaluateCandidateGate(
        detail::CandidateGateInput{initial_error, settled_error,
                                   nonlinear_summary.maximum_transaction_translation_m,
                                   nonlinear_summary.maximum_transaction_rotation_rad, true,
                                   detail::CandidateGatePhase::ConvergedTransaction},
        detail::CandidateGateLimits{impl_->config.maximum_transaction_translation_correction_m,
                                    impl_->config.maximum_transaction_rotation_correction_rad,
                                    impl_->config.complete_objective_nonsmooth_absolute_allowance,
                                    impl_->config.complete_objective_nonsmooth_relative_allowance});
    if (final_gate != detail::CandidateGateDecision::Accepted) {
      const double allowance =
          impl_->config.complete_objective_nonsmooth_absolute_allowance +
          impl_->config.complete_objective_nonsmooth_relative_allowance * std::abs(initial_error);
      const double theta_error =
          candidate->getFactorsUnsafe().error(candidate->getLinearizationPoint());
      const double delta_norm = candidate->getDelta().norm();
      return Result::failure(factor_transaction_error(
          final_gate == detail::CandidateGateDecision::CompleteObjectiveIncrease
              ? LocalGraphErrorCode::NonlinearCostIncrease
              : LocalGraphErrorCode::NonFiniteEstimate,
          "factor-only converged candidate failed its complete-objective gate: seed=" +
              std::to_string(initial_error) + " candidate=" + std::to_string(settled_error) +
              " allowance=" + std::to_string(allowance) + " theta_error=" +
              std::to_string(theta_error) + " delta_norm=" + std::to_string(delta_norm) +
              " iterations=" + std::to_string(nonlinear_summary.iterations) +
              " transaction_translation=" +
              std::to_string(nonlinear_summary.maximum_transaction_translation_m) +
              " transaction_rotation=" +
              std::to_string(nonlinear_summary.maximum_transaction_rotation_rad),
          rejected_solve_report()));
    }

    const NavigationKeys latest_keys = navigationKeys(impl_->navigation_records.back().graph_index);
    const core::NavStateEstimate latest_estimate = gtsam_api::fromGtsamState(
        settled_estimate.at<gtsam::Pose3>(latest_keys.pose),
        settled_estimate.at<gtsam::Vector3>(latest_keys.velocity),
        settled_estimate.at<gtsam::imuBias::ConstantBias>(latest_keys.bias));
    if (!finiteState(latest_estimate)) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::NonFiniteEstimate,
          "factor-only graph solve produced a non-finite latest navigation state",
          rejected_solve_report()));
    }
    NavigationCovariance latest_covariance;
    try {
      latest_covariance = gtsam_api::jointNavigationCovarianceFromBayesTree(
          *candidate, latest_keys.pose, latest_keys.velocity, latest_keys.bias);
    } catch (const std::exception& exception) {
      return Result::failure(factor_transaction_error(
          LocalGraphErrorCode::MarginalCovarianceFailure,
          std::string("factor-only joint marginal failed: ") + exception.what(),
          rejected_solve_report()));
    }

    std::vector<LocalGraphPoseSnapshot> navigation_poses;
    navigation_poses.reserve(impl_->navigation_records.size());
    for (const Impl::NavigationRecord& record : impl_->navigation_records) {
      const core::Pose3d pose = gtsam_api::fromGtsamPose(
          settled_estimate.at<gtsam::Pose3>(navigationKeys(record.graph_index).pose));
      if (!pose.matrix().allFinite()) {
        return Result::failure(factor_transaction_error(
            LocalGraphErrorCode::NonFiniteEstimate,
            "factor-only graph solve produced a non-finite live pose", rejected_solve_report()));
      }
      navigation_poses.push_back(LocalGraphPoseSnapshot{record.state, record.time, pose});
    }

    if (impl_->latest_revision.value() >= core::LocalGraphRevision::kInvalidValue - 1U) {
      return Result::failure(factorBatchGraphError(
          LocalGraphErrorCode::GraphRevisionExhausted,
          "factor-only transaction cannot publish an invalid or wrapped graph revision",
          diagnostic_ref));
    }
    const core::LocalGraphRevision candidate_revision{impl_->latest_revision.value() + 1U};
    auto candidate_active_batches = impl_->active_factor_batches;
    auto candidate_removable_batches = impl_->removable_factor_batches;
    auto candidate_terminal_batches = impl_->terminal_factor_batches;
    auto candidate_sequences = impl_->sensor_batch_sequences;
    std::size_t candidate_active_factor_count = impl_->active_lidar_direct_factors;
    std::size_t candidate_terminal_evictions = impl_->terminal_factor_batch_records_evicted;
    std::vector<FactorBatchProvenance> transitions;

    const auto append_terminal = [&](FactorBatchProvenance provenance) {
      candidate_terminal_batches.push_back(std::move(provenance));
      if (candidate_terminal_batches.size() > impl_->config.maximum_terminal_factor_batch_records) {
        candidate_terminal_batches.pop_front();
        ++candidate_terminal_evictions;
      }
    };

    if (insertion) {
      while (candidate_removable_batches.size() >= impl_->config.maximum_removable_factor_batches) {
        const SensorFactorBatchRef sealed = candidate_removable_batches.front();
        candidate_removable_batches.pop_front();
        if (auto record = candidate_active_batches.find(sealed);
            record != candidate_active_batches.end()) {
          record->second.provenance.removable = false;
        }
      }
      FactorBatchProvenance provenance;
      provenance.batch = *insertion;
      provenance.inserted_revision = candidate_revision;
      provenance.status = FactorBatchJournalStatus::Active;
      provenance.removable = true;
      Impl::FactorBatchRecord record;
      record.provenance = provenance;
      record.solver_indices = update.newFactorsIndices;
      record.solver_keys = inserted_solver_keys;
      candidate_active_batches.emplace(diagnostic_ref, std::move(record));
      candidate_removable_batches.push_back(diagnostic_ref);
      candidate_active_factor_count += factors_added;
      const core::FactorBatchMetadata& metadata = factorBatchMetadata(*insertion);
      candidate_sequences[metadata.sensor] = Impl::SensorBatchSequence{
          metadata.batch_id, metadata.health.recovery_epoch, metadata.health.transition_sequence,
          metadata.health.assessed_at, metadata.timing.produced_at};
      transitions.push_back(std::move(provenance));
    } else {
      for (const SensorFactorBatchRef& batch_ref : ordered_removals) {
        auto record = candidate_active_batches.find(batch_ref);
        FactorBatchProvenance provenance = record->second.provenance;
        provenance.status = FactorBatchJournalStatus::Removed;
        provenance.terminal_revision = candidate_revision;
        provenance.removable = false;
        provenance.removal_reason = removal->reason;
        candidate_active_factor_count -= factorBatchFactorCount(provenance.batch);
        candidate_active_batches.erase(record);
        candidate_removable_batches.erase(std::remove(candidate_removable_batches.begin(),
                                                      candidate_removable_batches.end(), batch_ref),
                                          candidate_removable_batches.end());
        transitions.push_back(provenance);
        append_terminal(std::move(provenance));
      }
    }

    LocalGraphCommit commit;
    commit.odom_epoch = impl_->odom_epoch;
    commit.revision = candidate_revision;
    commit.parent = impl_->latest_revision;
    commit.state = impl_->latest_state;
    commit.state_time = impl_->latest_time;
    commit.estimate = latest_estimate;
    commit.covariance = latest_covariance;
    commit.navigation_poses = std::move(navigation_poses);
    if (insertion) {
      commit.lidar_registration = insertion->registration_report;
      commit.lidar_pairs = inserted_pair_reports;
      commit.lidar_finalized_map = inserted_finalized_map_report;
    }
    const VisualSolveCounts visual_counts{impl_->visual_landmarks.size(),
                                          impl_->visual_factors.size()};
    commit.solve = solveReport(update, impl_->navigation_states, impl_->combined_imu_factors,
                               impl_->config.maximum_navigation_states, 0U, 0U, 0U, false,
                               visual_counts, &nonlinear_summary);
    populateFactorBatchCounts(&commit.solve, candidate_active_batches.size(),
                              candidate_active_factor_count, insertion ? 1U : 0U, factors_added,
                              ordered_removals.size(), factors_removed);
    commit.lineage =
        insertion ? factorBatchMetadata(*insertion).lineage : impl_->latest_commit->lineage;
    commit.factor_batch_transitions = std::move(transitions);

    // Complete every allocation and potentially-throwing copy before the
    // commit boundary. The operations below are swaps or scalar assignments;
    // a failed publication copy must therefore leave both solver and journal
    // bit-for-bit unchanged.
    auto stored_commit = std::make_shared<const LocalGraphCommit>(commit);

    impl_->solver.swap(candidate);
    impl_->active_factor_batches.swap(candidate_active_batches);
    impl_->removable_factor_batches.swap(candidate_removable_batches);
    impl_->terminal_factor_batches.swap(candidate_terminal_batches);
    impl_->sensor_batch_sequences.swap(candidate_sequences);
    impl_->active_lidar_direct_factors = candidate_active_factor_count;
    impl_->terminal_factor_batch_records_evicted = candidate_terminal_evictions;
    impl_->latest_revision = commit.revision;
    impl_->latest_commit.swap(stored_commit);
    transaction_timing.finish();
    return Result::success(std::move(commit));
  } catch (const std::exception& exception) {
    return Result::failure(factor_transaction_error(
        LocalGraphErrorCode::SolverFailure,
        std::string("factor-only iSAM2 QR transaction failed: ") + exception.what()));
  }
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::appendNavigationKnot(
    SensorKnotAppend transaction) {
  using Result = core::Result<LocalGraphCommit, LocalGraphError>;
  ImuKnotAppend& append = transaction.navigation;
  core::PipelineWorkIdentity work;
  if (append.state.valid()) {
    work.state = append.state;
  }
  detail::LocalPipelineTimingScope transaction_timing(
      impl_->timing, LocalPipelineTimingStage::GraphTransactionUpdate, work);
  if (!impl_->latest_commit) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NotInitialized, "initialize the local graph first"));
  }
  if (!append.state.valid()) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NonMonotonicState, "new state identity is invalid"));
  }
  if (impl_->state_ids.contains(append.state.value())) {
    return Result::failure(
        graphError(LocalGraphErrorCode::DuplicateState, "state identity is already in the graph"));
  }
  if (append.state.value() <= impl_->latest_state.value() ||
      append.exact_time <= impl_->latest_time) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NonMonotonicState,
                   "IMU knots require strictly increasing state IDs and fusion times"));
  }
  if (!validLineage(append.lineage)) {
    return Result::failure(graphError(LocalGraphErrorCode::InvalidLineage,
                                      "IMU factor ObservationLineage is invalid"));
  }

  std::size_t marginalize_count = 0U;
  const core::FusionTime nominal_cutoff = append.exact_time - impl_->config.target_fixed_lag;
  while (marginalize_count < impl_->navigation_records.size() &&
         impl_->navigation_records[marginalize_count].time < nominal_cutoff) {
    ++marginalize_count;
  }
  bool window_cap_applied = false;
  while (impl_->navigation_states + 1U - marginalize_count >
         impl_->config.maximum_navigation_states) {
    ++marginalize_count;
    window_cap_applied = true;
  }
  std::set<std::uint64_t> marginalized_state_ids;
  for (std::size_t index = 0U; index < marginalize_count; ++index) {
    marginalized_state_ids.insert(impl_->navigation_records[index].state.value());
  }
  std::vector<SensorFactorBatchRef> factor_batches_sealed_by_marginalization;
  std::vector<SensorFactorBatchRef> factor_batches_finalized_by_marginalization;
  std::size_t lidar_direct_factors_sealed_by_marginalization = 0U;
  std::size_t lidar_direct_factors_finalized_by_marginalization = 0U;
  for (const auto& [batch_ref, record] : impl_->active_factor_batches) {
    const bool touches_final_state =
        factorBatchTouchesAnyState(record.provenance.batch, marginalized_state_ids);
    const bool source_state_final =
        factorBatchSourceIsFinal(record.provenance.batch, marginalized_state_ids);
    const std::size_t factor_count = factorBatchFactorCount(record.provenance.batch);
    if (source_state_final) {
      factor_batches_finalized_by_marginalization.push_back(batch_ref);
      lidar_direct_factors_finalized_by_marginalization += factor_count;
    } else if (touches_final_state &&
               record.provenance.status == FactorBatchJournalStatus::Active) {
      factor_batches_sealed_by_marginalization.push_back(batch_ref);
      lidar_direct_factors_sealed_by_marginalization += factor_count;
    }
  }

  const auto state_time = [&](core::StateId state) -> std::optional<core::FusionTime> {
    if (state == append.state) {
      return append.exact_time;
    }
    const auto record = std::find_if(
        impl_->navigation_records.begin(), impl_->navigation_records.end(),
        [&](const Impl::NavigationRecord& candidate) { return candidate.state == state; });
    if (record == impl_->navigation_records.end()) {
      return std::nullopt;
    }
    return record->time;
  };

  std::vector<core::FactorId> ordered_visual_retirements = transaction.visual_factor_retirements;
  if (ordered_visual_retirements.size() >
      impl_->config.maximum_visual_factor_retirements_per_transaction) {
    return Result::failure(
        graphError(LocalGraphErrorCode::VisualTransactionLimit,
                   "visual factor retirement count exceeds the transaction bound"));
  }
  std::sort(ordered_visual_retirements.begin(), ordered_visual_retirements.end());
  for (std::size_t index = 0U; index < ordered_visual_retirements.size(); ++index) {
    const core::FactorId factor = ordered_visual_retirements[index];
    if (!factor.valid() || (index > 0U && factor == ordered_visual_retirements[index - 1U])) {
      return Result::failure(graphError(LocalGraphErrorCode::VisualFactorRetirementUnavailable,
                                        "visual factor retirements must be valid and unique"));
    }
    if (!impl_->visual_factors.contains(factor.value())) {
      return Result::failure(
          graphError(LocalGraphErrorCode::VisualFactorRetirementUnavailable,
                     "visual factor retirement refers to an unknown or inactive factor"));
    }
  }

  std::vector<const VisualLandmarkSeed*> ordered_visual_landmarks;
  std::vector<const VisualReprojectionFactorSpec*> ordered_visual_factors;
  std::map<std::uint64_t, const VisualLandmarkSeed*> new_visual_landmarks;
  std::map<std::uint64_t, std::size_t> new_landmark_factor_counts;
  std::map<std::uint64_t, std::pair<double, double>> new_landmark_range_bounds;
  if (transaction.visual) {
    const VisualFactorBatch& visual = *transaction.visual;
    if (visual.exact_time != append.exact_time) {
      return Result::failure(
          graphError(LocalGraphErrorCode::VisualReferenceMismatch,
                     "visual batch time must equal the candidate navigation state time"));
    }
    if (visual.new_landmarks.size() > impl_->config.maximum_visual_landmarks_per_transaction ||
        visual.factors.size() > impl_->config.maximum_visual_factors_per_transaction) {
      return Result::failure(
          graphError(LocalGraphErrorCode::VisualTransactionLimit,
                     "visual landmark or factor count exceeds the transaction bound"));
    }

    ordered_visual_landmarks.reserve(visual.new_landmarks.size());
    for (const VisualLandmarkSeed& landmark : visual.new_landmarks) {
      ordered_visual_landmarks.push_back(&landmark);
    }
    std::sort(ordered_visual_landmarks.begin(), ordered_visual_landmarks.end(),
              [](const VisualLandmarkSeed* lhs, const VisualLandmarkSeed* rhs) {
                return lhs->landmark < rhs->landmark;
              });
    for (const VisualLandmarkSeed* landmark : ordered_visual_landmarks) {
      const double expected_range = std::exp(-landmark->eta);
      const auto anchor_time = state_time(landmark->anchor.state);
      if (!landmark->landmark.valid() || !landmark->track.valid() ||
          !std::isfinite(landmark->eta) || !std::isfinite(landmark->initial_range_m) ||
          landmark->initial_range_m <= 0.0 || !std::isfinite(expected_range) ||
          std::abs(landmark->initial_range_m - expected_range) >
              1.0e-12 * std::max(1.0, expected_range) ||
          landmark->triangulation.status != VisualTriangulationStatus::Seeded ||
          !validVisualObservation(landmark->anchor)) {
        return Result::failure(
            graphError(LocalGraphErrorCode::InvalidVisualBatch,
                       "visual landmark seed identity, eta, range, status, or anchor is invalid"));
      }
      if (!anchor_time || *anchor_time != landmark->anchor.exact_time) {
        return Result::failure(graphError(
            LocalGraphErrorCode::VisualStateUnavailable,
            "visual landmark anchor state is unknown, marginalized, or at a different time"));
      }
      if (impl_->visual_landmarks.contains(landmark->landmark.value()) ||
          !new_visual_landmarks.emplace(landmark->landmark.value(), landmark).second) {
        return Result::failure(graphError(LocalGraphErrorCode::DuplicateVisualLandmark,
                                          "visual landmark eta may be initialized exactly once"));
      }
      new_landmark_factor_counts.emplace(landmark->landmark.value(), 0U);
    }

    ordered_visual_factors.reserve(visual.factors.size());
    for (const VisualReprojectionFactorSpec& factor : visual.factors) {
      ordered_visual_factors.push_back(&factor);
    }
    std::sort(ordered_visual_factors.begin(), ordered_visual_factors.end(),
              [](const VisualReprojectionFactorSpec* lhs, const VisualReprojectionFactorSpec* rhs) {
                return lhs->id < rhs->id;
              });
    std::optional<core::FactorId> previous_factor;
    for (const VisualReprojectionFactorSpec* factor : ordered_visual_factors) {
      if (!finitePositive(factor->minimum_range_m) || !finitePositive(factor->maximum_range_m) ||
          factor->minimum_range_m >= factor->maximum_range_m) {
        return Result::failure(
            graphError(LocalGraphErrorCode::InvalidVisualBatch,
                       "visual factor inverse-range bounds must be finite, positive, and ordered"));
      }
      if (!factor->id.valid() || !factor->landmark.valid() || !factor->track.valid() ||
          (previous_factor && *previous_factor == factor->id) ||
          impl_->visual_factors.contains(factor->id.value())) {
        return Result::failure(
            graphError(LocalGraphErrorCode::DuplicateVisualFactor,
                       "visual factor identities must be valid, new, and unique"));
      }
      previous_factor = factor->id;

      const VisualObservationRef* expected_anchor = nullptr;
      VisualTrackId expected_track;
      const auto new_landmark = new_visual_landmarks.find(factor->landmark.value());
      if (new_landmark != new_visual_landmarks.end()) {
        expected_anchor = &new_landmark->second->anchor;
        expected_track = new_landmark->second->track;
        ++new_landmark_factor_counts.at(factor->landmark.value());
        const auto [bounds, inserted] = new_landmark_range_bounds.emplace(
            factor->landmark.value(), std::pair{factor->minimum_range_m, factor->maximum_range_m});
        if (!inserted && (bounds->second.first != factor->minimum_range_m ||
                          bounds->second.second != factor->maximum_range_m)) {
          return Result::failure(
              graphError(LocalGraphErrorCode::InvalidVisualBatch,
                         "all visual factors sharing a landmark must use identical range bounds"));
        }
      } else {
        const auto existing = impl_->visual_landmarks.find(factor->landmark.value());
        if (existing == impl_->visual_landmarks.end()) {
          return Result::failure(
              graphError(LocalGraphErrorCode::VisualLandmarkUnavailable,
                         "visual factor refers to an unknown or marginalized landmark"));
        }
        expected_anchor = &existing->second.anchor;
        expected_track = existing->second.track;
        if (existing->second.minimum_range_m != factor->minimum_range_m ||
            existing->second.maximum_range_m != factor->maximum_range_m) {
          return Result::failure(
              graphError(LocalGraphErrorCode::InvalidVisualBatch,
                         "visual factor range bounds differ from its existing landmark"));
        }
      }
      if (factor->track != expected_track ||
          !sameVisualObservation(factor->anchor, *expected_anchor)) {
        return Result::failure(
            graphError(LocalGraphErrorCode::VisualReferenceMismatch,
                       "visual factor track or immutable anchor does not match its landmark"));
      }

      const auto anchor_time = state_time(factor->anchor.state);
      const auto observer_time = state_time(factor->observer.state);
      if (!anchor_time || !observer_time) {
        return Result::failure(
            graphError(LocalGraphErrorCode::VisualStateUnavailable,
                       "visual factor refers to an unknown or marginalized state"));
      }
      if (*anchor_time != factor->anchor.exact_time ||
          *observer_time != factor->observer.exact_time ||
          factor->observer.exact_time > visual.exact_time) {
        return Result::failure(
            graphError(LocalGraphErrorCode::VisualReferenceMismatch,
                       "visual factor state times do not match exact graph knot times"));
      }
      const core::FactorGroupId factor_group(factor->id.value());
      if (!validLineage(factor->lineage) ||
          !lineageContainsVisualObservation(factor->lineage, factor->anchor,
                                            core::ObservationRole::ConditioningOnly,
                                            std::nullopt) ||
          !lineageContainsVisualObservation(factor->lineage, factor->observer,
                                            core::ObservationRole::PrimaryResidual, factor_group)) {
        return Result::failure(
            graphError(LocalGraphErrorCode::InvalidLineage,
                       "visual factor lineage does not bind its exact anchor and observer"));
      }
    }
    for (const auto& [landmark, factor_count] : new_landmark_factor_counts) {
      if (factor_count == 0U) {
        return Result::failure(graphError(
            LocalGraphErrorCode::InvalidVisualBatch,
            "every new visual eta requires at least one factor in the same transaction"));
      }
      const auto bounds = new_landmark_range_bounds.find(landmark);
      const VisualLandmarkSeed& seed = *new_visual_landmarks.at(landmark);
      if (bounds == new_landmark_range_bounds.end() ||
          seed.initial_range_m < bounds->second.first ||
          seed.initial_range_m > bounds->second.second) {
        return Result::failure(
            graphError(LocalGraphErrorCode::InvalidVisualBatch,
                       "visual landmark seed range is outside its shared factor bounds"));
      }
    }
  }

  const ImuInterval& interval = append.interval;
  if (!interval.support.valid() || interval.knots.size() < 2U) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InvalidImuInterval,
                   "combined preintegration needs a valid interval with at least two knots"));
  }
  if (interval.support.start != impl_->latest_time || interval.support.end != append.exact_time ||
      interval.knots.front().time != impl_->latest_time ||
      interval.knots.back().time != append.exact_time) {
    return Result::failure(
        graphError(LocalGraphErrorCode::InexactImuBoundary,
                   "preintegration support and boundary knots must exactly equal adjacent "
                   "state times"));
  }
  if (interval.contains_saturation) {
    return Result::failure(graphError(LocalGraphErrorCode::SaturatedImu,
                                      "saturated IMU support cannot enter a factor"));
  }
  if (interval.inferred_missing_ticks != 0U) {
    return Result::failure(
        graphError(LocalGraphErrorCode::GapBridgeNotImplemented,
                   "the first graph slice cannot yet identify and inflate each inferred "
                   "missing-tick segment"));
  }
  if (interval.maximum_time_uncertainty.nanoseconds != 0) {
    return Result::failure(
        graphError(LocalGraphErrorCode::TimestampUncertaintyNotImplemented,
                   "timestamp-boundary covariance propagation is not implemented; the "
                   "adapter will not accept an overconfident factor"));
  }

  for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
    const auto& previous = interval.knots[index - 1U];
    const auto& current = interval.knots[index];
    if (current.time <= previous.time || !previous.specific_force_mps2.allFinite() ||
        !previous.angular_velocity_radps.allFinite() || !current.specific_force_mps2.allFinite() ||
        !current.angular_velocity_radps.allFinite()) {
      return Result::failure(
          graphError(LocalGraphErrorCode::InvalidImuInterval,
                     "IMU knots must be strictly ordered with finite measurements"));
    }
  }

  try {
    const std::size_t previous_index = impl_->navigation_records.back().graph_index;
    const std::size_t current_index = impl_->next_graph_index;
    const NavigationKeys previous_keys = navigationKeys(previous_index);
    const NavigationKeys current_keys = navigationKeys(current_index);
    const gtsam::Values previous_values = impl_->solver->calculateEstimate();
    const gtsam::Pose3 previous_pose = previous_values.at<gtsam::Pose3>(previous_keys.pose);
    const gtsam::Vector3 previous_velocity =
        previous_values.at<gtsam::Vector3>(previous_keys.velocity);
    const gtsam::imuBias::ConstantBias previous_bias =
        previous_values.at<gtsam::imuBias::ConstantBias>(previous_keys.bias);

    gtsam::PreintegratedCombinedMeasurements preintegrated(impl_->preintegration_params,
                                                           previous_bias);
    for (std::size_t index = 1U; index < interval.knots.size(); ++index) {
      const auto& previous = interval.knots[index - 1U];
      const auto& current = interval.knots[index];
      const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
      const Eigen::Vector3d acceleration_midpoint =
          0.5 * (previous.specific_force_mps2 + current.specific_force_mps2);
      const Eigen::Vector3d angular_rate_midpoint =
          0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps);
      preintegrated.integrateMeasurement(acceleration_midpoint, angular_rate_midpoint, dt);
    }

    const gtsam::NavState predicted =
        preintegrated.predict(gtsam::NavState(previous_pose, previous_velocity), previous_bias);
    gtsam::NonlinearFactorGraph new_factors;
    new_factors.emplace_shared<gtsam::CombinedImuFactor>(
        previous_keys.pose, previous_keys.velocity, current_keys.pose, current_keys.velocity,
        previous_keys.bias, current_keys.bias, preintegrated);
    gtsam::Pose3 current_pose_seed = predicted.pose();
    gtsam::Values new_values;
    new_values.insert(current_keys.pose, current_pose_seed);
    new_values.insert(current_keys.velocity, predicted.velocity());
    new_values.insert(current_keys.bias, previous_bias);
    const auto candidateError = [&](LocalGraphErrorCode code, std::string detail,
                                    std::optional<LocalSolveReport> rejected_solve = std::nullopt) {
      return graphError(code, std::move(detail), std::nullopt, std::move(rejected_solve));
    };

    std::set<std::uint64_t> marginalized_visual_landmark_ids;
    for (const auto& [landmark_id, landmark] : impl_->visual_landmarks) {
      if (marginalized_state_ids.contains(landmark.anchor.state.value())) {
        marginalized_visual_landmark_ids.insert(landmark_id);
      }
    }
    for (const auto& [factor_id, factor] : impl_->visual_factors) {
      static_cast<void>(factor_id);
      if (marginalized_state_ids.contains(factor.anchor_state.value()) ||
          marginalized_state_ids.contains(factor.observer_state.value())) {
        marginalized_visual_landmark_ids.insert(factor.landmark.value());
      }
    }
    for (const VisualLandmarkSeed* landmark : ordered_visual_landmarks) {
      if (marginalized_state_ids.contains(landmark->anchor.state.value())) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::VisualStateUnavailable,
                           "new visual landmark anchor is scheduled for marginalization"));
      }
    }
    for (const VisualReprojectionFactorSpec* factor : ordered_visual_factors) {
      if (marginalized_state_ids.contains(factor->anchor.state.value()) ||
          marginalized_state_ids.contains(factor->observer.state.value())) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::VisualStateUnavailable,
                           "new visual factor endpoint is scheduled for marginalization"));
      }
      if (marginalized_visual_landmark_ids.contains(factor->landmark.value())) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::VisualLandmarkUnavailable,
                           "new visual factor landmark is scheduled for marginalization"));
      }
    }

    auto candidate_visual_landmarks = impl_->visual_landmarks;
    auto candidate_visual_factors = impl_->visual_factors;
    gtsam::FactorIndices remove_factor_indices;
    remove_factor_indices.reserve(ordered_visual_retirements.size());
    std::set<std::uint64_t> explicitly_retired_visual_factor_ids;
    for (const core::FactorId factor : ordered_visual_retirements) {
      const auto record = impl_->visual_factors.find(factor.value());
      remove_factor_indices.push_back(record->second.solver_index);
      candidate_visual_factors.erase(factor.value());
      explicitly_retired_visual_factor_ids.insert(factor.value());
    }

    std::set<std::uint64_t> landmarks_with_pending_factors;
    for (const VisualReprojectionFactorSpec* factor : ordered_visual_factors) {
      landmarks_with_pending_factors.insert(factor->landmark.value());
    }
    std::set<std::uint64_t> factorless_visual_landmark_ids;
    for (const auto& [landmark_id, landmark] : candidate_visual_landmarks) {
      static_cast<void>(landmark);
      const bool has_active_factor = std::any_of(
          candidate_visual_factors.begin(), candidate_visual_factors.end(),
          [&](const auto& factor) { return factor.second.landmark.value() == landmark_id; });
      if (!has_active_factor && !landmarks_with_pending_factors.contains(landmark_id)) {
        // Removing only this metadata record leaks an unconstrained eta inside
        // iSAM2. Treat factorless retirement as ordinary graph finality so its
        // eta is marginalized in the same candidate transaction.
        marginalized_visual_landmark_ids.insert(landmark_id);
        factorless_visual_landmark_ids.insert(landmark_id);
      }
    }

    std::vector<core::FactorId> finalized_visual_factors;
    for (const auto& [factor_id, factor] : impl_->visual_factors) {
      if (marginalized_visual_landmark_ids.contains(factor.landmark.value()) &&
          !explicitly_retired_visual_factor_ids.contains(factor_id)) {
        finalized_visual_factors.push_back(factor.factor);
      }
    }

    std::vector<EtaKey> marginalized_eta_keys;
    marginalized_eta_keys.reserve(marginalized_visual_landmark_ids.size());
    std::set<EtaKey> factorless_eta_keys;
    std::map<std::uint64_t, double> factorless_final_eta;
    for (const std::uint64_t landmark_id : marginalized_visual_landmark_ids) {
      const auto landmark = impl_->visual_landmarks.find(landmark_id);
      if (landmark != impl_->visual_landmarks.end()) {
        marginalized_eta_keys.push_back(landmark->second.eta_key);
        if (factorless_visual_landmark_ids.contains(landmark_id)) {
          factorless_eta_keys.insert(landmark->second.eta_key);
          factorless_final_eta.emplace(
              landmark_id, gtsam_api::decodeBoundedEta(
                               previous_values.at<double>(landmark->second.eta_key),
                               landmark->second.minimum_range_m, landmark->second.maximum_range_m)
                               .eta);
        }
      }
      candidate_visual_landmarks.erase(landmark_id);
      for (auto factor = candidate_visual_factors.begin();
           factor != candidate_visual_factors.end();) {
        if (factor->second.landmark.value() == landmark_id) {
          factor = candidate_visual_factors.erase(factor);
        } else {
          ++factor;
        }
      }
    }

    const auto state_graph_index = [&](core::StateId state) -> std::optional<std::size_t> {
      if (state == append.state) {
        return current_index;
      }
      const auto record =
          std::find_if(impl_->navigation_records.begin(), impl_->navigation_records.end(),
                       [&](const Impl::NavigationRecord& candidate_record) {
                         return candidate_record.state == state;
                       });
      if (record == impl_->navigation_records.end()) {
        return std::nullopt;
      }
      return record->graph_index;
    };
    const auto state_pose = [&](core::StateId state) -> core::Pose3d {
      if (state == append.state) {
        return gtsam_api::fromGtsamPose(current_pose_seed);
      }
      const std::size_t index = *state_graph_index(state);
      return gtsam_api::fromGtsamPose(previous_values.at<gtsam::Pose3>(navigationKeys(index).pose));
    };

    for (const VisualLandmarkSeed* landmark : ordered_visual_landmarks) {
      const EtaKey eta_key = visualEtaKey(landmark->landmark);
      if (previous_values.exists(eta_key) || new_values.exists(eta_key)) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::DuplicateVisualLandmark,
                           "visual eta key is already initialized in the solver"));
      }
      const auto bounds = new_landmark_range_bounds.at(landmark->landmark.value());
      new_values.insert(eta_key,
                        gtsam_api::encodeBoundedEta(landmark->eta, bounds.first, bounds.second));
      candidate_visual_landmarks.emplace(
          landmark->landmark.value(),
          Impl::VisualLandmarkRecord{landmark->landmark, landmark->track, eta_key, landmark->anchor,
                                     bounds.first, bounds.second});
    }

    const std::size_t visual_factor_offset = new_factors.size();
    for (const VisualReprojectionFactorSpec* factor : ordered_visual_factors) {
      const std::size_t anchor_index = *state_graph_index(factor->anchor.state);
      const std::size_t observer_index = *state_graph_index(factor->observer.state);
      const auto landmark = candidate_visual_landmarks.find(factor->landmark.value());
      const double eta_latent = new_values.exists(landmark->second.eta_key)
                                    ? new_values.at<double>(landmark->second.eta_key)
                                    : previous_values.at<double>(landmark->second.eta_key);
      const double eta = gtsam_api::decodeBoundedEta(eta_latent, landmark->second.minimum_range_m,
                                                     landmark->second.maximum_range_m)
                             .eta;
      const auto evaluation = evaluateVisualReprojection(*factor, state_pose(factor->anchor.state),
                                                         state_pose(factor->observer.state), eta);
      if (!evaluation) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::VisualFactorEvaluationFailure,
                           "visual factor seed evaluation failed: " + evaluation.error().detail));
      }
      new_factors.push_back(boost::make_shared<gtsam_api::AnchoredInverseRangeFactor>(
          navigationKeys(anchor_index).pose, navigationKeys(observer_index).pose,
          landmark->second.eta_key, *factor));
    }

    auto candidate = std::make_unique<gtsam_api::CandidateIsolatedISAM2>(*impl_->solver);
    gtsam::FactorIndices marginal_factor_indices;
    gtsam::FactorIndices deleted_factor_indices;
    gtsam::FastList<gtsam::Key> leaf_keys;
    boost::optional<gtsam::FastMap<gtsam::Key, int>> constrained_keys;
    boost::optional<gtsam::FastList<gtsam::Key>> extra_reeliminate_keys;
    const bool marginalization_required =
        marginalize_count > 0U || marginalized_eta_keys.size() > factorless_eta_keys.size();
    if (marginalization_required) {
      gtsam::FastMap<gtsam::Key, int> ordering_groups;
      for (const auto& record : impl_->navigation_records) {
        const NavigationKeys keys = navigationKeys(record.graph_index);
        ordering_groups[keys.pose] = 2;
        ordering_groups[keys.velocity] = 2;
        ordering_groups[keys.bias] = 2;
      }
      ordering_groups[current_keys.pose] = 2;
      ordering_groups[current_keys.velocity] = 2;
      ordering_groups[current_keys.bias] = 2;

      for (const auto& [landmark_id, landmark] : candidate_visual_landmarks) {
        static_cast<void>(landmark_id);
        ordering_groups[landmark.eta_key] = 2;
      }
      for (const EtaKey eta_key : marginalized_eta_keys) {
        // A factorless eta leaves the nonlinear variable index in the same
        // update that removes its last factor. Passing that orphan to COLAMD's
        // constrained/extra-reelimination maps throws before the atomic
        // marginalization step. It remains a requested leaf below.
        if (!factorless_eta_keys.contains(eta_key)) {
          ordering_groups[eta_key] = 0;
        }
      }

      for (const EtaKey eta_key : marginalized_eta_keys) {
        if (!factorless_eta_keys.contains(eta_key)) {
          leaf_keys.push_back(eta_key);
        }
      }
      for (std::size_t index = 0U; index < marginalize_count; ++index) {
        const NavigationKeys keys = navigationKeys(impl_->navigation_records[index].graph_index);
        ordering_groups[keys.pose] = 1;
        ordering_groups[keys.velocity] = 1;
        ordering_groups[keys.bias] = 1;
        leaf_keys.push_back(keys.pose);
        leaf_keys.push_back(keys.velocity);
        leaf_keys.push_back(keys.bias);
      }

      constrained_keys = std::move(ordering_groups);
      // Match GTSAM's incremental fixed-lag smoother: only Bayes-tree
      // descendants whose separator contains a requested leaf need explicit
      // re-elimination. The constrained ordering still covers the complete
      // candidate, and marginalizeLeaves() below retains the exact leaf set.
      extra_reeliminate_keys = impl_->solver->affectedKeysForLeafMarginalization(leaf_keys);
    }

    // GLIM settles each newly augmented smoother with repeated update() calls
    // before consuming its fixed-lag result.  gtsam_points additionally makes
    // error decrease and update magnitude first-class ISAM2 diagnostics.  Keep
    // those two properties here while retaining stock GTSAM behind the private
    // adapter: every candidate checks the configured iSAM2 relinearization
    // policy, then completes the Bayes-tree delta solve before measuring its
    // physical state correction. Relinearization thresholds and convergence
    // resolution are deliberately separate; there is no secondary batch or
    // fast path.
    gtsam::ISAM2UpdateParams first_update_parameters;
    first_update_parameters.removeFactorIndices = remove_factor_indices;
    first_update_parameters.constrainedKeys = constrained_keys;
    first_update_parameters.extraReelimKeys = extra_reeliminate_keys;
    first_update_parameters.force_relinearize = true;
    const gtsam::ISAM2Result update =
        candidate->update(new_factors, new_values, first_update_parameters);
    if (update.newFactorsIndices.size() != new_factors.size()) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::SolverFailure,
                         "iSAM2 did not return a deterministic index for every new factor"));
    }
    for (std::size_t index = 0U; index < ordered_visual_factors.size(); ++index) {
      const VisualReprojectionFactorSpec& factor = *ordered_visual_factors[index];
      candidate_visual_factors.emplace(
          factor.id.value(),
          Impl::VisualFactorRecord{factor.id, factor.landmark,
                                   update.newFactorsIndices[visual_factor_offset + index],
                                   factor.anchor.state, factor.observer.state});
    }

    gtsam::Values transaction_seed = previous_values;
    transaction_seed.insert(new_values);
    const gtsam_api::CandidateCacheSetStamp candidate_cache_set =
        candidate->candidateCacheSetStamp();
    if (candidate_cache_set.stateful_factors != 0U) {
      return Result::failure(candidateError(
          LocalGraphErrorCode::SolverFailure,
          "complete-objective reuse requires stateless nonlinear factors, but the candidate "
          "contains " +
              std::to_string(candidate_cache_set.stateful_factors) +
              " factor(s) with mutable candidate caches"));
    }
    const double initial_error = candidate->getFactorsUnsafe().error(transaction_seed);
    if (!std::isfinite(initial_error)) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::NonFiniteEstimate,
                         "candidate nonlinear objective is non-finite at the transaction seed"));
    }

    std::vector<NavigationKeys> candidate_navigation_keys;
    candidate_navigation_keys.reserve(impl_->navigation_records.size() + 1U);
    std::vector<core::StateId> candidate_navigation_states;
    candidate_navigation_states.reserve(impl_->navigation_records.size() + 1U);
    for (const auto& record : impl_->navigation_records) {
      candidate_navigation_keys.push_back(navigationKeys(record.graph_index));
      candidate_navigation_states.push_back(record.state);
    }
    candidate_navigation_keys.push_back(current_keys);
    candidate_navigation_states.push_back(append.state);
    std::vector<BoundedEtaKey> candidate_eta_keys;
    candidate_eta_keys.reserve(candidate_visual_landmarks.size());
    for (const auto& [landmark_id, landmark] : candidate_visual_landmarks) {
      static_cast<void>(landmark_id);
      candidate_eta_keys.push_back(
          BoundedEtaKey{landmark.eta_key, landmark.minimum_range_m, landmark.maximum_range_m});
    }
    const std::size_t candidate_variable_count =
        3U * candidate_navigation_keys.size() + candidate_eta_keys.size();
    const std::size_t candidate_factor_count = candidate->getFactorsUnsafe().size();

    NonlinearSolveSummary nonlinear_summary;
    nonlinear_summary.convergence = effectiveConvergenceThresholds(
        impl_->config, interval.support.end - interval.support.start);
    nonlinear_summary.error_before = initial_error;
    gtsam::Values previous_iteration = transaction_seed;
    gtsam::Values settled_estimate;
    const auto rejectedSolveReport = [&]() {
      const VisualSolveCounts visual_counts{
          candidate_visual_landmarks.size(), candidate_visual_factors.size(),
          ordered_visual_landmarks.size(),   ordered_visual_factors.size(),
          ordered_visual_retirements.size(), marginalized_visual_landmark_ids.size()};
      return solveReport(update, impl_->navigation_states + 1U, impl_->combined_imu_factors + 1U,
                         impl_->config.maximum_navigation_states, 0U, 0U, 0U, false, visual_counts,
                         &nonlinear_summary);
    };
    bool solve_converged = false;
    double settled_complete_error = initial_error;
    double previous_complete_error = initial_error;
    bool previous_step_globalized = false;
    std::optional<double> isam_error_before;
    std::optional<double> isam_error_after;
    for (std::size_t iteration = 0U; iteration < impl_->config.maximum_nonlinear_iterations;
         ++iteration) {
      if (iteration == 0U) {
        accumulateUpdate(update, candidate_variable_count, candidate_factor_count,
                         nonlinear_summary);
        if (update.errorBefore) {
          isam_error_before = *update.errorBefore;
        }
        if (update.errorAfter) {
          isam_error_after = *update.errorAfter;
        }
      } else {
        gtsam::ISAM2UpdateParams iteration_parameters;
        // The factor-bearing update above uses extraReelimKeys once to make
        // every requested fixed-lag leaf structurally marginalizable. Later
        // empty updates retain that ordering for any naturally affected
        // subtree, but must not mark the entire active window again.
        iteration_parameters.constrainedKeys = constrained_keys;
        iteration_parameters.force_relinearize = true;
        // A damped delta is a valid trust-region estimate but not the raw
        // Bayes-tree back-substitution. The following iteration must therefore
        // relinearize every non-fixed variable at that accepted estimate.
        iteration_parameters.forceFullSolve = previous_step_globalized;
        const gtsam::ISAM2Result iteration_update =
            candidate->update(gtsam::NonlinearFactorGraph{}, gtsam::Values{}, iteration_parameters);
        accumulateUpdate(iteration_update, candidate_variable_count, candidate_factor_count,
                         nonlinear_summary);
        if (iteration_update.errorBefore) {
          isam_error_before = *iteration_update.errorBefore;
        }
        if (iteration_update.errorAfter) {
          isam_error_after = *iteration_update.errorAfter;
        }
        // The graph-wide invariant above excludes factors whose nonlinear
        // objective changes as a side effect of linearization. Therefore the
        // complete objective already evaluated at the previous Values is the
        // exact reference after this thresholded empty update.
      }

      gtsam::Values full_step_estimate = candidate->calculateEstimate();
      const double full_step_error = candidate->getFactorsUnsafe().error(full_step_estimate);
      const double objective_stabilization_tolerance =
          objectiveConvergenceTolerance(impl_->config, previous_complete_error);
      const bool full_step_is_physically_converged =
          std::isfinite(full_step_error) && full_step_error > previous_complete_error &&
          full_step_error - previous_complete_error <= objective_stabilization_tolerance &&
          converged(stateCorrections(previous_iteration, full_step_estimate,
                                     candidate_navigation_keys, candidate_eta_keys),
                    nonlinear_summary.convergence);
      gtsam_api::CandidateGlobalizationResult globalized = candidate->globalizeFullStep(
          previous_iteration, previous_complete_error, std::move(full_step_estimate),
          full_step_error, impl_->config.maximum_nonlinear_backtracking_steps,
          impl_->config.nonlinear_backtracking_reduction, full_step_is_physically_converged,
          objective_stabilization_tolerance);
      settled_estimate = std::move(globalized.estimate);
      settled_complete_error = globalized.error;
      previous_step_globalized = globalized.rejected_full_step;
      const StateCorrectionMetrics iteration_correction = stateCorrections(
          previous_iteration, settled_estimate, candidate_navigation_keys, candidate_eta_keys);
      const StateCorrectionMetrics transaction_correction = stateCorrections(
          transaction_seed, settled_estimate, candidate_navigation_keys, candidate_eta_keys);
      const StateCorrectionMetrics cumulative_imu_correction =
          poseCorrection(predicted.pose(), settled_estimate.at<gtsam::Pose3>(current_keys.pose));
      const gtsam::NavState candidate_conditioned_imu_prediction = preintegrated.predict(
          gtsam::NavState(settled_estimate.at<gtsam::Pose3>(previous_keys.pose),
                          settled_estimate.at<gtsam::Vector3>(previous_keys.velocity)),
          settled_estimate.at<gtsam::imuBias::ConstantBias>(previous_keys.bias));
      const StateCorrectionMetrics conditioned_imu_correction =
          poseCorrection(candidate_conditioned_imu_prediction.pose(),
                         settled_estimate.at<gtsam::Pose3>(current_keys.pose));
      const StateCorrectionMetrics newest_transition_correction =
          poseCorrection(previous_pose.between(current_pose_seed),
                         settled_estimate.at<gtsam::Pose3>(previous_keys.pose)
                             .between(settled_estimate.at<gtsam::Pose3>(current_keys.pose)));
      // A fixed-lag solve may revise the previous and current states together
      // by a large common odom-frame transform while leaving both the newest
      // relative transition and the IMU residual small. Treating that common
      // smoothing revision as a LiDAR innovation causes a false rejection
      // cascade. The physical transaction gate therefore bounds only the
      // newest transition change and the current state relative to IMU
      // repropagation from the candidate-updated previous state.
      const double gated_transaction_translation = std::max(
          newest_transition_correction.translation_m, conditioned_imu_correction.translation_m);
      const double gated_transaction_rotation = std::max(newest_transition_correction.rotation_rad,
                                                         conditioned_imu_correction.rotation_rad);
      nonlinear_summary.iterations = iteration + 1U;
      nonlinear_summary.last_iteration_correction = iteration_correction;
      nonlinear_summary.error_after = settled_complete_error;
      nonlinear_summary.full_steps_rejected += globalized.rejected_full_step ? 1U : 0U;
      nonlinear_summary.backtracking_trials += globalized.backtracking_trials;
      nonlinear_summary.cauchy_directions_attempted +=
          globalized.cauchy_direction_attempted ? 1U : 0U;
      nonlinear_summary.cauchy_steps_accepted += globalized.cauchy_step_accepted ? 1U : 0U;
      nonlinear_summary.cauchy_backtracking_trials += globalized.cauchy_backtracking_trials;
      nonlinear_summary.zero_step_terminations += globalized.zero_step ? 1U : 0U;
      nonlinear_summary.minimum_step_scale =
          std::min(nonlinear_summary.minimum_step_scale, globalized.step_scale);
      nonlinear_summary.last_objective_change = settled_complete_error - previous_complete_error;
      if (iteration_correction.finite() && transaction_correction.finite() &&
          cumulative_imu_correction.finite()) {
        nonlinear_summary.maximum_iteration_translation_m = std::max(
            nonlinear_summary.maximum_iteration_translation_m, iteration_correction.translation_m);
        nonlinear_summary.maximum_iteration_rotation_rad = std::max(
            nonlinear_summary.maximum_iteration_rotation_rad, iteration_correction.rotation_rad);
        nonlinear_summary.maximum_transaction_translation_m = std::max(
            nonlinear_summary.maximum_transaction_translation_m, gated_transaction_translation);
        nonlinear_summary.maximum_transaction_rotation_rad = std::max(
            nonlinear_summary.maximum_transaction_rotation_rad, gated_transaction_rotation);
      }

      const detail::CandidateGateDecision gate = detail::evaluateCandidateGate(
          detail::CandidateGateInput{
              initial_error, settled_complete_error, gated_transaction_translation,
              gated_transaction_rotation,
              iteration_correction.finite() && transaction_correction.finite() &&
                  cumulative_imu_correction.finite() && conditioned_imu_correction.finite() &&
                  newest_transition_correction.finite(),
              detail::CandidateGatePhase::NonlinearIteration},
          detail::CandidateGateLimits{
              impl_->config.maximum_transaction_translation_correction_m,
              impl_->config.maximum_transaction_rotation_correction_rad,
              impl_->config.complete_objective_nonsmooth_absolute_allowance,
              impl_->config.complete_objective_nonsmooth_relative_allowance});
      if (gate == detail::CandidateGateDecision::NonFinite) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::NonFiniteEstimate,
            "candidate nonlinear iteration produced a non-finite correction or objective",
            rejectedSolveReport()));
      }
      if (gate == detail::CandidateGateDecision::PoseCorrectionLimit) {
        const PoseCorrectionExtrema window_correction =
            poseCorrectionExtrema(transaction_seed, settled_estimate, candidate_navigation_keys);
        const auto stateAt =
            [&candidate_navigation_states](const std::optional<std::size_t>& index) {
              return index && *index < candidate_navigation_states.size()
                         ? std::to_string(candidate_navigation_states[*index].value())
                         : std::string("unset");
            };
        return Result::failure(candidateError(
            LocalGraphErrorCode::PoseCorrectionLimit,
            "candidate newest-transition correction or conditioned IMU innovation exceeded the "
            "configured pose gate: "
            "gated_translation=" +
                std::to_string(gated_transaction_translation) + " m gated_rotation=" +
                std::to_string(gated_transaction_rotation) + " rad window_translation=" +
                std::to_string(window_correction.maximum.translation_m) +
                " m window_translation_state=" +
                stateAt(window_correction.translation_navigation_index) +
                " window_rotation=" + std::to_string(window_correction.maximum.rotation_rad) +
                " rad window_rotation_state=" +
                stateAt(window_correction.rotation_navigation_index) +
                " current_from_imu_translation=" +
                std::to_string(cumulative_imu_correction.translation_m) +
                " m current_from_imu_rotation=" +
                std::to_string(cumulative_imu_correction.rotation_rad) +
                " rad conditioned_imu_translation=" +
                std::to_string(conditioned_imu_correction.translation_m) +
                " m conditioned_imu_rotation=" +
                std::to_string(conditioned_imu_correction.rotation_rad) +
                " rad newest_transition_translation=" +
                std::to_string(newest_transition_correction.translation_m) +
                " m newest_transition_rotation=" +
                std::to_string(newest_transition_correction.rotation_rad) + " rad",
            rejectedSolveReport()));
      }

      const bool objective_accepts_convergence = objectiveAcceptsPhysicalConvergence(
          nonlinear_summary.last_objective_change,
          objectiveConvergenceTolerance(impl_->config, previous_complete_error));
      if (converged(iteration_correction, nonlinear_summary.convergence) &&
          objective_accepts_convergence) {
        solve_converged = true;
        break;
      }
      previous_iteration = settled_estimate;
      previous_complete_error = settled_complete_error;
    }
    if (!solve_converged) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::NonlinearConvergenceFailure,
                         "candidate iSAM2 solve reached the configured nonlinear iteration bound",
                         rejectedSolveReport()));
    }
    const detail::CandidateGateDecision converged_gate = detail::evaluateCandidateGate(
        detail::CandidateGateInput{initial_error, settled_complete_error,
                                   nonlinear_summary.maximum_transaction_translation_m,
                                   nonlinear_summary.maximum_transaction_rotation_rad, true,
                                   detail::CandidateGatePhase::ConvergedTransaction},
        detail::CandidateGateLimits{impl_->config.maximum_transaction_translation_correction_m,
                                    impl_->config.maximum_transaction_rotation_correction_rad,
                                    impl_->config.complete_objective_nonsmooth_absolute_allowance,
                                    impl_->config.complete_objective_nonsmooth_relative_allowance});
    if (converged_gate == detail::CandidateGateDecision::CompleteObjectiveIncrease) {
      const double theta_error =
          candidate->getFactorsUnsafe().error(candidate->getLinearizationPoint());
      const double committed_seed_error = impl_->solver->getFactorsUnsafe().error(previous_values);
      const double committed_theta_error =
          impl_->solver->getFactorsUnsafe().error(impl_->solver->getLinearizationPoint());
      const StateCorrectionMetrics theta_from_seed =
          stateCorrections(transaction_seed, candidate->getLinearizationPoint(),
                           candidate_navigation_keys, candidate_eta_keys);
      const gtsam::VectorValues& solver_delta = candidate->getDelta();
      const double delta_norm = solver_delta.norm();
      const double committed_delta_norm = impl_->solver->getDelta().norm();
      std::map<char, double> maximum_delta_by_symbol;
      for (const auto& [key, value] : solver_delta) {
        maximum_delta_by_symbol[gtsam::Symbol(key).chr()] = std::max(
            maximum_delta_by_symbol[gtsam::Symbol(key).chr()], value.lpNorm<Eigen::Infinity>());
      }
      const auto symbol_delta = [&maximum_delta_by_symbol](char symbol) {
        const auto found = maximum_delta_by_symbol.find(symbol);
        return found == maximum_delta_by_symbol.end() ? 0.0 : found->second;
      };
      return Result::failure(candidateError(
          LocalGraphErrorCode::NonlinearCostIncrease,
          "converged complete nonlinear objective exceeded the transaction-seed allowance: seed=" +
              std::to_string(initial_error) +
              " candidate=" + std::to_string(settled_complete_error) + " isam_error_before=" +
              (isam_error_before ? std::to_string(*isam_error_before) : std::string("unset")) +
              " isam_error_after=" +
              (isam_error_after ? std::to_string(*isam_error_after) : std::string("unset")) +
              " theta_error=" + std::to_string(theta_error) + " delta_norm=" +
              std::to_string(delta_norm) + " max_pose_delta=" + std::to_string(symbol_delta('x')) +
              " max_velocity_delta=" + std::to_string(symbol_delta('v')) +
              " max_bias_delta=" + std::to_string(symbol_delta('b')) + " max_eta_delta=" +
              std::to_string(symbol_delta('e')) + " state=" + std::to_string(append.state.value()) +
              " time_ns=" + std::to_string(append.exact_time.nanoseconds) + " new_imu_factors=1" +
              " new_lidar_factors=0" +
              " new_visual_landmarks=" + std::to_string(ordered_visual_landmarks.size()) +
              " new_visual_factors=" + std::to_string(ordered_visual_factors.size()) +
              " retired_visual_factors=" + std::to_string(ordered_visual_retirements.size()) +
              " marginalized_states=" + std::to_string(marginalize_count) +
              " committed_seed_error=" + std::to_string(committed_seed_error) +
              " committed_theta_error=" + std::to_string(committed_theta_error) +
              " committed_delta_norm=" + std::to_string(committed_delta_norm) +
              " theta_from_seed_translation=" + std::to_string(theta_from_seed.translation_m) +
              " theta_from_seed_rotation=" + std::to_string(theta_from_seed.rotation_rad),
          rejectedSolveReport()));
    }
    // Reserve and materialize final-state snapshots before marginalizeLeaves
    // mutates the candidate Bayes tree. The estimate and covariance therefore
    // describe the exact settled posterior that is about to be frozen into the
    // boundary prior. Nothing is published unless every later transaction
    // gate succeeds and the candidate is atomically committed.
    if (impl_->latest_revision.value() >= core::LocalGraphRevision::kInvalidValue - 1U) {
      return Result::failure(candidateError(
          LocalGraphErrorCode::GraphRevisionExhausted,
          "navigation transaction cannot publish an invalid or wrapped graph revision"));
    }
    const core::LocalGraphRevision candidate_revision{impl_->latest_revision.value() + 1U};
    std::vector<LocalGraphFinalizedState> finalized_states;
    finalized_states.reserve(marginalize_count);
    for (std::size_t index = 0U; index < marginalize_count; ++index) {
      const Impl::NavigationRecord& record = impl_->navigation_records[index];
      const NavigationKeys keys = navigationKeys(record.graph_index);
      const core::NavStateEstimate final_estimate =
          gtsam_api::fromGtsamState(settled_estimate.at<gtsam::Pose3>(keys.pose),
                                    settled_estimate.at<gtsam::Vector3>(keys.velocity),
                                    settled_estimate.at<gtsam::imuBias::ConstantBias>(keys.bias));
      core::PoseCovariance pose_covariance;
      try {
        pose_covariance = gtsam_api::poseCovarianceFromBayesTree(*candidate, keys.pose);
      } catch (const std::exception& exception) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::MarginalCovarianceFailure,
                           std::string("pre-marginalization pose covariance failed for state ") +
                               std::to_string(record.state.value()) + ": " + exception.what()));
      }
      if (!finiteState(final_estimate) || !validPoseCovariance(pose_covariance)) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::FinalityValidationFailure,
            "pre-marginalization final state or right-translation-first covariance is invalid"));
      }
      if (!finalized_states.empty() && (record.state <= finalized_states.back().state ||
                                        record.time <= finalized_states.back().exact_time)) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::FinalityValidationFailure,
            "pre-marginalization final states are not strictly monotonic and unique"));
      }
      finalized_states.push_back(
          LocalGraphFinalizedState{record.state, record.time, impl_->odom_epoch, candidate_revision,
                                   final_estimate, std::move(pose_covariance)});
    }
    if (finalized_states.size() != marginalize_count) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::FinalityValidationFailure,
                         "final-state reservation did not cover every removed state"));
    }

    std::vector<LocalGraphFinalizedVisualLandmark> finalized_visual_landmarks;
    finalized_visual_landmarks.reserve(marginalized_visual_landmark_ids.size());
    for (const std::uint64_t landmark_id : marginalized_visual_landmark_ids) {
      const auto record = impl_->visual_landmarks.find(landmark_id);
      if (record == impl_->visual_landmarks.end()) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::FinalityValidationFailure,
            "finalized visual landmark is absent from the settled pre-marginalization state"));
      }
      double final_eta{};
      if (settled_estimate.exists(record->second.eta_key)) {
        final_eta = gtsam_api::decodeBoundedEta(settled_estimate.at<double>(record->second.eta_key),
                                                record->second.minimum_range_m,
                                                record->second.maximum_range_m)
                        .eta;
      } else {
        const auto factorless_eta = factorless_final_eta.find(landmark_id);
        if (factorless_eta == factorless_final_eta.end()) {
          return Result::failure(candidateError(
              LocalGraphErrorCode::FinalityValidationFailure,
              "non-factorless visual landmark is absent from the settled finality state"));
        }
        // Removing the last factor makes GTSAM erase the now-orphan theta in
        // the first candidate update. Its immutable physical value is the
        // previously settled posterior captured immediately before removal.
        final_eta = factorless_eta->second;
      }
      core::FinalizedLandmarkSegment segment;
      segment.segment = core::LandmarkSegmentId(record->second.landmark.value());
      segment.anchor_state = record->second.anchor.state;
      segment.final_revision = candidate_revision;
      segment.final_log_inverse_range = final_eta;
      segment.reason = core::LandmarkFinalityReason::Marginalized;
      if (!record->second.landmark.valid() ||
          core::validateFinalizedLandmarkSegment(segment) != core::FinalityValidationError::None ||
          (!finalized_visual_landmarks.empty() &&
           record->second.landmark <= finalized_visual_landmarks.back().landmark)) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::FinalityValidationFailure,
                           "finalized visual landmark identity, segment, or ordering is invalid"));
      }
      finalized_visual_landmarks.push_back(
          LocalGraphFinalizedVisualLandmark{record->second.landmark, std::move(segment)});
    }

    // Marginalize only after the nonlinear candidate has settled.  Freezing a
    // boundary linearization earlier is exactly the failure mode this
    // transaction path is designed to avoid.
    if (marginalization_required) {
      candidate->marginalizeLeaves(leaf_keys, marginal_factor_indices, deleted_factor_indices);
    }
    const gtsam::Values estimate = candidate->calculateEstimate();

    std::vector<NavigationKeys> retained_navigation_keys;
    retained_navigation_keys.reserve(impl_->navigation_records.size() + 1U - marginalize_count);
    for (std::size_t index = marginalize_count; index < impl_->navigation_records.size(); ++index) {
      retained_navigation_keys.push_back(
          navigationKeys(impl_->navigation_records[index].graph_index));
    }
    retained_navigation_keys.push_back(current_keys);
    const StateCorrectionMetrics marginalization_correction =
        stateCorrections(settled_estimate, estimate, retained_navigation_keys, candidate_eta_keys);
    if (!marginalization_correction.finite()) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::NonFiniteEstimate,
                         "fixed-lag marginalization produced a non-finite state correction"));
    }
    nonlinear_summary.marginalization_translation_m = marginalization_correction.translation_m;
    nonlinear_summary.marginalization_rotation_rad = marginalization_correction.rotation_rad;
    if (!converged(marginalization_correction, nonlinear_summary.convergence)) {
      return Result::failure(candidateError(
          LocalGraphErrorCode::SolverFailure,
          "fixed-lag marginalization changed the settled estimate beyond convergence tolerance"));
    }
    const StateCorrectionMetrics final_transaction_correction =
        stateCorrections(transaction_seed, estimate, retained_navigation_keys, candidate_eta_keys);
    const StateCorrectionMetrics final_cumulative_imu_correction =
        poseCorrection(predicted.pose(), estimate.at<gtsam::Pose3>(current_keys.pose));
    // The previous endpoint can be one of the leaves removed by this
    // transaction (for example with a one-state hard window).  Its settled
    // pre-marginalization value is then the immutable boundary value; retained
    // endpoints use the post-marginalization estimate.  This keeps the final
    // innovation check valid without reading keys that no longer exist.
    const gtsam::Pose3 final_previous_pose =
        estimate.exists(previous_keys.pose) ? estimate.at<gtsam::Pose3>(previous_keys.pose)
                                            : settled_estimate.at<gtsam::Pose3>(previous_keys.pose);
    const gtsam::Vector3 final_previous_velocity =
        estimate.exists(previous_keys.velocity)
            ? estimate.at<gtsam::Vector3>(previous_keys.velocity)
            : settled_estimate.at<gtsam::Vector3>(previous_keys.velocity);
    const gtsam::imuBias::ConstantBias final_previous_bias =
        estimate.exists(previous_keys.bias)
            ? estimate.at<gtsam::imuBias::ConstantBias>(previous_keys.bias)
            : settled_estimate.at<gtsam::imuBias::ConstantBias>(previous_keys.bias);
    const gtsam::NavState final_conditioned_imu_prediction = preintegrated.predict(
        gtsam::NavState(final_previous_pose, final_previous_velocity), final_previous_bias);
    const StateCorrectionMetrics final_conditioned_imu_correction = poseCorrection(
        final_conditioned_imu_prediction.pose(), estimate.at<gtsam::Pose3>(current_keys.pose));
    const StateCorrectionMetrics final_newest_transition_correction =
        poseCorrection(previous_pose.between(current_pose_seed),
                       final_previous_pose.between(estimate.at<gtsam::Pose3>(current_keys.pose)));
    if (!final_transaction_correction.finite() || !final_cumulative_imu_correction.finite() ||
        !final_conditioned_imu_correction.finite() ||
        !final_newest_transition_correction.finite()) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::NonFiniteEstimate,
                         "post-marginalization transaction correction is non-finite"));
    }
    const double final_gated_translation =
        std::max(final_newest_transition_correction.translation_m,
                 final_conditioned_imu_correction.translation_m);
    const double final_gated_rotation = std::max(final_newest_transition_correction.rotation_rad,
                                                 final_conditioned_imu_correction.rotation_rad);
    nonlinear_summary.maximum_transaction_translation_m =
        std::max(nonlinear_summary.maximum_transaction_translation_m, final_gated_translation);
    nonlinear_summary.maximum_transaction_rotation_rad =
        std::max(nonlinear_summary.maximum_transaction_rotation_rad, final_gated_rotation);
    if (final_gated_translation > impl_->config.maximum_transaction_translation_correction_m ||
        final_gated_rotation > impl_->config.maximum_transaction_rotation_correction_rad) {
      return Result::failure(candidateError(
          LocalGraphErrorCode::PoseCorrectionLimit,
          "post-marginalization newest-transition correction or conditioned IMU innovation "
          "exceeded the configured transaction pose gate"));
    }

    for (const auto& [landmark_id, landmark] : candidate_visual_landmarks) {
      static_cast<void>(landmark_id);
      if (!estimate.exists(landmark.eta_key) ||
          !std::isfinite(estimate.at<double>(landmark.eta_key)) ||
          !std::isfinite(gtsam_api::decodeBoundedEta(estimate.at<double>(landmark.eta_key),
                                                     landmark.minimum_range_m,
                                                     landmark.maximum_range_m)
                             .eta)) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::NonFiniteEstimate,
            "active bounded visual eta is absent or non-finite after the candidate solve"));
      }
    }
    for (const EtaKey eta_key : marginalized_eta_keys) {
      if (estimate.exists(eta_key)) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::SolverFailure,
                           "visual eta remained active after endpoint marginalization"));
      }
    }
    const core::NavStateEstimate state =
        gtsam_api::fromGtsamState(estimate.at<gtsam::Pose3>(current_keys.pose),
                                  estimate.at<gtsam::Vector3>(current_keys.velocity),
                                  estimate.at<gtsam::imuBias::ConstantBias>(current_keys.bias));
    if (!finiteState(state)) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::NonFiniteEstimate,
                         "combined IMU graph solve produced a non-finite navigation state"));
    }

    NavigationCovariance marginal;
    try {
      marginal = gtsam_api::jointNavigationCovarianceFromBayesTree(
          *candidate, current_keys.pose, current_keys.velocity, current_keys.bias);
    } catch (const std::exception& exception) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::MarginalCovarianceFailure,
                         std::string("joint navigation marginal failed: ") + exception.what()));
    }
    if (!marginal.matrix.allFinite()) {
      return Result::failure(candidateError(LocalGraphErrorCode::MarginalCovarianceFailure,
                                            "joint navigation marginal is non-finite"));
    }

    std::vector<LocalGraphPoseSnapshot> navigation_poses;
    navigation_poses.reserve(impl_->navigation_records.size() + 1U);
    for (std::size_t index = 0U; index < impl_->navigation_records.size(); ++index) {
      const Impl::NavigationRecord& record = impl_->navigation_records[index];
      core::Pose3d pose;
      if (index < marginalize_count) {
        pose = finalized_states[index].final_estimate.T_odom_imu;
      } else {
        const NavigationKeys keys = navigationKeys(record.graph_index);
        pose = gtsam_api::fromGtsamPose(estimate.at<gtsam::Pose3>(keys.pose));
      }
      if (!pose.matrix().allFinite() ||
          (!navigation_poses.empty() && (record.state <= navigation_poses.back().state ||
                                         record.time <= navigation_poses.back().exact_time))) {
        return Result::failure(
            candidateError(LocalGraphErrorCode::FinalityValidationFailure,
                           "committed navigation pose snapshot is non-finite or non-monotonic"));
      }
      navigation_poses.push_back(
          LocalGraphPoseSnapshot{record.state, record.time, std::move(pose)});
    }
    if (!navigation_poses.empty() && (append.state <= navigation_poses.back().state ||
                                      append.exact_time <= navigation_poses.back().exact_time)) {
      return Result::failure(
          candidateError(LocalGraphErrorCode::FinalityValidationFailure,
                         "current state does not follow the committed navigation pose snapshot"));
    }
    navigation_poses.push_back(
        LocalGraphPoseSnapshot{append.state, append.exact_time, state.T_odom_imu});

    auto candidate_terminal_factor_batches = impl_->terminal_factor_batches;
    std::size_t candidate_terminal_factor_batch_evictions =
        impl_->terminal_factor_batch_records_evicted;
    std::vector<FactorBatchProvenance> factor_batch_transitions;
    factor_batch_transitions.reserve(factor_batches_sealed_by_marginalization.size() +
                                     factor_batches_finalized_by_marginalization.size());
    for (const SensorFactorBatchRef& batch_ref : factor_batches_sealed_by_marginalization) {
      const auto record = impl_->active_factor_batches.find(batch_ref);
      if (record == impl_->active_factor_batches.end()) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::FinalityValidationFailure,
            "FactorBatch selected for sealing is absent from the active provenance journal"));
      }
      FactorBatchProvenance provenance = record->second.provenance;
      provenance.status = FactorBatchJournalStatus::SealedByMarginalization;
      provenance.sealed_revision = candidate_revision;
      provenance.removable = false;
      factor_batch_transitions.push_back(std::move(provenance));
    }
    for (const SensorFactorBatchRef& batch_ref : factor_batches_finalized_by_marginalization) {
      const auto record = impl_->active_factor_batches.find(batch_ref);
      if (record == impl_->active_factor_batches.end()) {
        return Result::failure(candidateError(
            LocalGraphErrorCode::FinalityValidationFailure,
            "FactorBatch selected for finality is absent from the active provenance journal"));
      }
      FactorBatchProvenance provenance = record->second.provenance;
      provenance.status = FactorBatchJournalStatus::FinalizedByMarginalization;
      if (!provenance.sealed_revision) {
        provenance.sealed_revision = candidate_revision;
      }
      provenance.terminal_revision = candidate_revision;
      provenance.removable = false;
      factor_batch_transitions.push_back(provenance);
      candidate_terminal_factor_batches.push_back(std::move(provenance));
      if (candidate_terminal_factor_batches.size() >
          impl_->config.maximum_terminal_factor_batch_records) {
        candidate_terminal_factor_batches.pop_front();
        ++candidate_terminal_factor_batch_evictions;
      }
    }
    const std::size_t active_factor_batches_after_finality =
        impl_->active_factor_batches.size() - factor_batches_finalized_by_marginalization.size();
    const std::size_t active_lidar_direct_factors_after_finality =
        impl_->active_lidar_direct_factors - lidar_direct_factors_finalized_by_marginalization;

    LocalGraphCommit commit;
    commit.odom_epoch = impl_->odom_epoch;
    commit.revision = candidate_revision;
    commit.parent = impl_->latest_revision;
    commit.state = append.state;
    commit.state_time = append.exact_time;
    commit.estimate = state;
    commit.covariance = marginal;
    commit.navigation_poses = std::move(navigation_poses);
    commit.finalized_states = std::move(finalized_states);
    commit.finalized_visual_factors = std::move(finalized_visual_factors);
    commit.finalized_visual_landmarks = std::move(finalized_visual_landmarks);
    commit.finality.records_reserved = marginalize_count;
    commit.finality.pose_covariances_computed = marginalize_count;
    commit.finality.nominal_cutoff = nominal_cutoff;
    if (!commit.finalized_states.empty()) {
      commit.finality.status = window_cap_applied ? LocalGraphFinalityStatus::PublishedWindowCap
                                                  : LocalGraphFinalityStatus::PublishedNominalLag;
      commit.finality.oldest_finalized_state = commit.finalized_states.front().state;
      commit.finality.newest_finalized_state = commit.finalized_states.back().state;
      commit.finality.oldest_finalized_time = commit.finalized_states.front().exact_time;
      commit.finality.newest_finalized_time = commit.finalized_states.back().exact_time;
    }
    const std::size_t active_navigation_states = impl_->navigation_states + 1U - marginalize_count;
    const VisualSolveCounts visual_counts{
        candidate_visual_landmarks.size(), candidate_visual_factors.size(),
        ordered_visual_landmarks.size(),   ordered_visual_factors.size(),
        ordered_visual_retirements.size(), marginalized_visual_landmark_ids.size()};
    commit.solve = solveReport(update, active_navigation_states, impl_->combined_imu_factors + 1U,
                               impl_->config.maximum_navigation_states, marginalize_count,
                               marginal_factor_indices.size(), deleted_factor_indices.size(),
                               window_cap_applied, visual_counts, &nonlinear_summary);
    populateFactorBatchCounts(&commit.solve, active_factor_batches_after_finality,
                              active_lidar_direct_factors_after_finality, 0U, 0U, 0U, 0U,
                              factor_batches_sealed_by_marginalization.size(),
                              lidar_direct_factors_sealed_by_marginalization,
                              factor_batches_finalized_by_marginalization.size(),
                              lidar_direct_factors_finalized_by_marginalization);
    commit.lineage = std::move(append.lineage);
    commit.factor_batch_transitions = std::move(factor_batch_transitions);
    auto stored_commit = std::make_shared<const LocalGraphCommit>(commit);

    impl_->solver.swap(candidate);
    impl_->visual_landmarks = std::move(candidate_visual_landmarks);
    impl_->visual_factors = std::move(candidate_visual_factors);
    for (const SensorFactorBatchRef& batch_ref : factor_batches_sealed_by_marginalization) {
      auto record = impl_->active_factor_batches.find(batch_ref);
      record->second.provenance.status = FactorBatchJournalStatus::SealedByMarginalization;
      record->second.provenance.sealed_revision = candidate_revision;
      record->second.provenance.removable = false;
      impl_->removable_factor_batches.erase(
          std::remove(impl_->removable_factor_batches.begin(),
                      impl_->removable_factor_batches.end(), batch_ref),
          impl_->removable_factor_batches.end());
    }
    for (const SensorFactorBatchRef& batch_ref : factor_batches_finalized_by_marginalization) {
      impl_->active_factor_batches.erase(batch_ref);
      impl_->removable_factor_batches.erase(
          std::remove(impl_->removable_factor_batches.begin(),
                      impl_->removable_factor_batches.end(), batch_ref),
          impl_->removable_factor_batches.end());
    }
    impl_->terminal_factor_batches.swap(candidate_terminal_factor_batches);
    impl_->active_lidar_direct_factors = active_lidar_direct_factors_after_finality;
    impl_->terminal_factor_batch_records_evicted = candidate_terminal_factor_batch_evictions;
    for (std::size_t index = 0U; index < marginalize_count; ++index) {
      impl_->state_ids.erase(impl_->navigation_records.front().state.value());
      impl_->navigation_records.pop_front();
    }
    impl_->state_ids.insert(append.state.value());
    impl_->navigation_records.push_back(
        Impl::NavigationRecord{append.state, append.exact_time, current_index});
    impl_->navigation_states = active_navigation_states;
    ++impl_->next_graph_index;
    ++impl_->combined_imu_factors;
    impl_->latest_state = append.state;
    impl_->latest_time = append.exact_time;
    impl_->latest_revision = commit.revision;
    impl_->latest_commit.swap(stored_commit);
    transaction_timing.finish();
    return Result::success(std::move(commit));
  } catch (const gtsam_api::VisualFactorEvaluationException& exception) {
    return Result::failure(
        graphError(LocalGraphErrorCode::VisualFactorEvaluationFailure,
                   std::string("visual factor evaluation failed during candidate solve: ") +
                       exception.what()));
  } catch (const std::exception& exception) {
    return Result::failure(
        graphError(LocalGraphErrorCode::SolverFailure,
                   std::string("combined IMU iSAM2 QR transaction failed: ") + exception.what()));
  }
}

core::Result<LocalGraphCommit, LocalGraphError> LocalGraph::estimate() const {
  using Result = core::Result<LocalGraphCommit, LocalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NotInitialized, "local graph has no committed state"));
  }
  return Result::success(*impl_->latest_commit);
}

core::Result<LocalGraphNavigationSnapshot, LocalGraphError> LocalGraph::navigationState(
    core::StateId state) const {
  using Result = core::Result<LocalGraphNavigationSnapshot, LocalGraphError>;
  if (!impl_->latest_commit) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NotInitialized, "local graph has no committed state"));
  }
  if (!state.valid()) {
    return Result::failure(graphError(LocalGraphErrorCode::NavigationStateUnavailable,
                                      "requested navigation state identity is invalid"));
  }
  const auto record = std::find_if(
      impl_->navigation_records.begin(), impl_->navigation_records.end(),
      [&](const Impl::NavigationRecord& candidate) { return candidate.state == state; });
  if (record == impl_->navigation_records.end()) {
    return Result::failure(
        graphError(LocalGraphErrorCode::NavigationStateUnavailable,
                   "requested navigation state is not explicit in the live local graph window"));
  }

  try {
    const NavigationKeys keys = navigationKeys(record->graph_index);
    const gtsam::Values estimate = impl_->solver->calculateEstimate();
    const core::NavStateEstimate navigation = gtsam_api::fromGtsamState(
        estimate.at<gtsam::Pose3>(keys.pose), estimate.at<gtsam::Vector3>(keys.velocity),
        estimate.at<gtsam::imuBias::ConstantBias>(keys.bias));
    if (!finiteState(navigation)) {
      return Result::failure(
          graphError(LocalGraphErrorCode::NonFiniteEstimate,
                     "requested live navigation state has a non-finite optimized estimate"));
    }
    return Result::success(LocalGraphNavigationSnapshot{impl_->odom_epoch, impl_->latest_revision,
                                                        record->state, record->time, navigation});
  } catch (const std::exception& exception) {
    return Result::failure(
        graphError(LocalGraphErrorCode::SolverFailure,
                   std::string("live navigation-state query failed: ") + exception.what()));
  }
}

std::optional<FactorBatchProvenance> LocalGraph::factorBatchProvenance(
    SensorFactorBatchRef batch) const {
  if (const auto active = impl_->active_factor_batches.find(batch);
      active != impl_->active_factor_batches.end()) {
    return active->second.provenance;
  }
  const auto terminal =
      std::find_if(impl_->terminal_factor_batches.begin(), impl_->terminal_factor_batches.end(),
                   [&](const FactorBatchProvenance& record) {
                     const core::FactorBatchMetadata& metadata = factorBatchMetadata(record.batch);
                     return metadata.sensor == batch.sensor && metadata.batch_id == batch.batch_id;
                   });
  return terminal == impl_->terminal_factor_batches.end()
             ? std::nullopt
             : std::optional<FactorBatchProvenance>{*terminal};
}

std::vector<FactorBatchProvenance> LocalGraph::factorBatchJournal() const {
  std::vector<FactorBatchProvenance> journal;
  journal.reserve(impl_->active_factor_batches.size() + impl_->terminal_factor_batches.size());
  journal.insert(journal.end(), impl_->terminal_factor_batches.begin(),
                 impl_->terminal_factor_batches.end());
  for (const auto& [batch, record] : impl_->active_factor_batches) {
    static_cast<void>(batch);
    journal.push_back(record.provenance);
  }
  std::sort(journal.begin(), journal.end(),
            [](const FactorBatchProvenance& lhs, const FactorBatchProvenance& rhs) {
              if (lhs.inserted_revision != rhs.inserted_revision) {
                return lhs.inserted_revision < rhs.inserted_revision;
              }
              const core::FactorBatchMetadata& left = factorBatchMetadata(lhs.batch);
              const core::FactorBatchMetadata& right = factorBatchMetadata(rhs.batch);
              return SensorFactorBatchRef{left.sensor, left.batch_id} <
                     SensorFactorBatchRef{right.sensor, right.batch_id};
            });
  return journal;
}

FactorBatchJournalStats LocalGraph::factorBatchJournalStats() const noexcept {
  const std::size_t sealed_batches = static_cast<std::size_t>(std::count_if(
      impl_->active_factor_batches.begin(), impl_->active_factor_batches.end(),
      [](const auto& entry) {
        return entry.second.provenance.status == FactorBatchJournalStatus::SealedByMarginalization;
      }));
  return FactorBatchJournalStats{
      impl_->active_factor_batches.size(),    impl_->active_lidar_direct_factors,
      impl_->removable_factor_batches.size(), sealed_batches,
      impl_->terminal_factor_batches.size(),  impl_->terminal_factor_batch_records_evicted};
}

bool LocalGraph::initialized() const noexcept {
  return static_cast<bool>(impl_->latest_commit);
}

}  // namespace meridian::local
