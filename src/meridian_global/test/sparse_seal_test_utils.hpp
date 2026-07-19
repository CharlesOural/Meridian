#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <source_location>
#include <utility>
#include <vector>

#include "meridian/core/api.hpp"

namespace meridian::global::test_support {

[[nodiscard]] inline core::ContentHash opaqueHash(std::uint8_t value) {
  core::ContentHash hash{};
  hash.front() = value;
  hash.back() = static_cast<std::uint8_t>(value ^ 0xa5U);
  return hash;
}

template <typename Value, typename Error>
[[nodiscard]] Value required(core::Result<Value, Error> result,
                             std::source_location location = std::source_location::current()) {
  if (!result) {
    std::fprintf(stderr, "canonical test fixture construction failed at %s:%u\n",
                 location.file_name(), location.line());
    std::abort();
  }
  return std::move(result).value();
}

[[nodiscard]] inline core::Pose3d pose(
    const Eigen::Vector3d& translation,
    const Eigen::Vector3d& rotation_vector = Eigen::Vector3d::Zero()) {
  return core::Pose3d(Sophus::SO3d::exp(rotation_vector), translation);
}

[[nodiscard]] inline core::ObservationLineage lineage(std::uint64_t id) {
  core::ObservationLineage output;
  output.id = core::ObservationLineageId(id);
  core::ObservationSlice slice;
  slice.root = core::MeasurementId(1000U + id);
  slice.source_checksum = opaqueHash(static_cast<std::uint8_t>(10U + id));
  slice.calibration = core::CalibrationEpoch(3U);
  output.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::PrimaryResidual,
                                                core::DerivedRecordId(2000U + id),
                                                core::FactorGroupId(3000U + id), std::nullopt});
  output.checksum = required(core::recomputeObservationLineageChecksum(output));
  return output;
}

[[nodiscard]] inline core::BlobRef durableBlob(std::uint64_t id, core::SparsePayloadKind kind) {
  core::BlobRef ref;
  ref.store = core::BlobStoreId(1U);
  ref.id = core::BlobId(id);
  ref.checksum = opaqueHash(static_cast<std::uint8_t>(20U + id));
  ref.layout = core::LayoutId(static_cast<std::uint64_t>(kind) + 1U);
  ref.bytes = 64U + id;
  ref.storage = core::BlobStorage::DurableSpool;
  return ref;
}

[[nodiscard]] inline core::SparsePayloadCatalog payloads(std::uint64_t base) {
  core::SparsePayloadCatalog catalog;
  catalog.entries = {
      {core::SparsePayloadKind::InternalTrajectory,
       durableBlob(base + 0U, core::SparsePayloadKind::InternalTrajectory)},
      {core::SparsePayloadKind::KeyframeIndex,
       durableBlob(base + 1U, core::SparsePayloadKind::KeyframeIndex)},
      {core::SparsePayloadKind::RegistrationProxy,
       durableBlob(base + 2U, core::SparsePayloadKind::RegistrationProxy)},
      {core::SparsePayloadKind::DenseInputIndex,
       durableBlob(base + 3U, core::SparsePayloadKind::DenseInputIndex)},
  };
  catalog.checksum = required(core::recomputeSparsePayloadCatalogChecksum(catalog));
  return catalog;
}

[[nodiscard]] inline core::LocalVariableRef stateVariable(core::LocalVariableKind kind,
                                                          core::StateId state) {
  core::LocalVariableRef variable;
  variable.kind = kind;
  variable.state = state;
  return variable;
}

[[nodiscard]] inline core::FrozenSquareRootFactor boundaryFactor(
    core::StateId from, core::StateId to, std::uint32_t rank = 24U,
    double to_translation_x_delta = 0.0) {
  core::FrozenSquareRootFactor factor;
  factor.rows = rank;
  factor.columns = 30U;
  factor.layout = {
      {stateVariable(core::LocalVariableKind::Pose, from), 0U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, from), 6U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, from), 9U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, from), 12U, 3U},
      {stateVariable(core::LocalVariableKind::Pose, to), 15U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, to), 21U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, to), 24U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, to), 27U, 3U},
  };
  factor.row_major_A.assign(static_cast<std::size_t>(factor.rows) * factor.columns, 0.0);
  for (std::uint32_t row = 0U; row < rank; ++row) {
    factor.row_major_A[static_cast<std::size_t>(row) * factor.columns + 6U + row] = 1.0;
  }
  factor.rhs.assign(factor.rows, 0.0);
  if (rank > 9U) {
    factor.rhs[9U] = to_translation_x_delta;
  }
  factor.constant_squared_error = 0.125;
  factor.numerical_rank = rank;
  factor.absolute_rank_tolerance = 1.0e-12;
  factor.relative_rank_tolerance = 1.0e-9;
  factor.cost_statistics.source_residual_dof = static_cast<std::uint64_t>(rank) + 6U;
  factor.cost_statistics.eliminated_numerical_rank = 6U;
  factor.cost_statistics.effective_dof = rank;
  factor.cost_statistics.calibration_revision = core::ResidualCalibrationRevision(1U);
  factor.cost_statistics.calibrated_total_cost_cutoff = 36.0;
  factor.checksum = required(core::recomputeFrozenSquareRootFactorChecksum(factor));
  return factor;
}

[[nodiscard]] inline core::FrozenSquareRootFactor boundaryFactorOnColumns(
    core::StateId from, core::StateId to, const std::vector<std::uint32_t>& columns) {
  core::FrozenSquareRootFactor factor;
  factor.rows = static_cast<std::uint32_t>(columns.size());
  factor.columns = 30U;
  factor.layout = {
      {stateVariable(core::LocalVariableKind::Pose, from), 0U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, from), 6U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, from), 9U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, from), 12U, 3U},
      {stateVariable(core::LocalVariableKind::Pose, to), 15U, 6U},
      {stateVariable(core::LocalVariableKind::NavigationVelocity, to), 21U, 3U},
      {stateVariable(core::LocalVariableKind::GyroBias, to), 24U, 3U},
      {stateVariable(core::LocalVariableKind::AccelBias, to), 27U, 3U},
  };
  factor.row_major_A.assign(static_cast<std::size_t>(factor.rows) * factor.columns, 0.0);
  for (std::size_t row = 0U; row < columns.size(); ++row) {
    factor.row_major_A[row * factor.columns + columns[row]] = 1.0;
  }
  factor.rhs.assign(factor.rows, 0.0);
  factor.constant_squared_error = 0.125;
  factor.numerical_rank = factor.rows;
  factor.absolute_rank_tolerance = 1.0e-12;
  factor.relative_rank_tolerance = 1.0e-9;
  factor.cost_statistics.source_residual_dof = columns.size() + 6U;
  factor.cost_statistics.eliminated_numerical_rank = 6U;
  factor.cost_statistics.effective_dof = columns.size();
  factor.cost_statistics.calibration_revision = core::ResidualCalibrationRevision(1U);
  factor.cost_statistics.calibrated_total_cost_cutoff = 36.0;
  factor.checksum = required(core::recomputeFrozenSquareRootFactorChecksum(factor));
  return factor;
}

inline void finalizeLocalContent(core::SparseSubmapSeal* seal) {
  seal->ref.local_content_checksum = required(core::recomputeSubmapLocalContentChecksum(*seal));
}

inline void finalizeSealEnvelope(core::SparseSubmapSeal* seal) {
  seal->seal_checksum = required(core::recomputeSparseSubmapSealChecksum(*seal));
}

[[nodiscard]] inline core::SparseSubmapSeal firstSeal(
    std::uint64_t id = 1U, std::int64_t start_ns = 0,
    const core::Pose3d& T_odom_submap = core::Pose3d{},
    const core::Pose3d& T_odom_imu = core::Pose3d{}) {
  core::SparseSubmapSeal seal;
  seal.header.schema_version = 1U;
  seal.header.trace = core::TraceId(100U + id);
  seal.header.producer = core::ProducerId(2U);
  seal.header.session = core::SessionId(1U);
  seal.header.created_at = core::FusionTime{start_ns + 100};
  seal.header.config = core::ConfigRevision(4U);
  seal.header.direct_calibration = core::CalibrationEpoch(3U);
  seal.ref.session = seal.header.session;
  seal.ref.odom_epoch = core::OdomEpoch(2U);
  seal.ref.id = core::SubmapId(id);
  seal.ref.calibration = *seal.header.direct_calibration;
  seal.ref.content_revision = core::SubmapContentRevision(1U);
  seal.final_local_revision = core::LocalGraphRevision(20U + id);
  seal.support_time = core::TimeRange{core::FusionTime{start_ns}, core::FusionTime{start_ns + 100}};
  seal.frame.boundary_state = core::StateId(100U + id);
  seal.frame.boundary_time = seal.support_time.start;
  seal.frame.gravity_up_odom = Eigen::Vector3d::UnitZ();
  seal.frame.boundary_yaw_odom = 0.0;
  seal.T_odom_submap = T_odom_submap;
  seal.boundary_navigation.state = seal.frame.boundary_state;
  seal.boundary_navigation.exact_time = seal.frame.boundary_time;
  seal.boundary_navigation.final_revision = core::LocalGraphRevision(10U + id);
  seal.boundary_navigation.T_odom_imu = T_odom_imu;
  seal.boundary_navigation.velocity_odom = Eigen::Vector3d{0.5, -0.1, 0.2};
  seal.boundary_navigation.gyro_bias = Eigen::Vector3d{0.01, -0.02, 0.03};
  seal.boundary_navigation.accel_bias = Eigen::Vector3d{-0.1, 0.05, 0.02};
  seal.payloads = payloads(id * 10U);
  seal.lineage = lineage(id);
  seal.quality_checksum = opaqueHash(static_cast<std::uint8_t>(120U + id));
  finalizeLocalContent(&seal);
  finalizeSealEnvelope(&seal);
  return seal;
}

[[nodiscard]] inline core::SparseSubmapSeal successor(const core::SparseSubmapSeal& previous,
                                                      std::uint32_t transition_rank = 24U,
                                                      double to_translation_x_delta = 0.0) {
  const std::uint64_t id = previous.ref.id.value() + 1U;
  const core::Pose3d T_odom_submap =
      pose(previous.T_odom_submap.translation() + Eigen::Vector3d{1.0, 0.2, -0.1},
           Eigen::Vector3d{0.0, 0.0, 0.08});
  const core::Pose3d T_odom_imu =
      T_odom_submap * pose(Eigen::Vector3d{0.2, -0.05, 0.1}, Eigen::Vector3d{0.03, -0.02, 0.01});
  core::SparseSubmapSeal current =
      firstSeal(id, previous.support_time.end.nanoseconds, T_odom_submap, T_odom_imu);
  current.header.session = previous.ref.session;
  current.ref.session = previous.ref.session;
  current.ref.odom_epoch = previous.ref.odom_epoch;
  current.ref.calibration = previous.ref.calibration;
  current.header.direct_calibration = current.ref.calibration;
  finalizeLocalContent(&current);

  core::CondensedBoundaryTransition local;
  local.header.schema_version = 1U;
  local.header.trace = core::TraceId(500U + id);
  local.header.producer = core::ProducerId(2U);
  local.header.session = current.ref.session;
  local.header.created_at = current.support_time.start;
  local.header.config = core::ConfigRevision(4U);
  local.header.direct_calibration = current.ref.calibration;
  local.odom_epoch = current.ref.odom_epoch;
  local.from = previous.boundary_navigation;
  local.to = current.boundary_navigation;
  local.boundary_factor =
      boundaryFactor(local.from.state, local.to.state, transition_rank, to_translation_x_delta);
  local.source_factors = {{local.odom_epoch, core::FactorId(1U)},
                          {local.odom_epoch, core::FactorId(2U)}};
  local.lineage = lineage(500U + id);
  local.final_revision = current.boundary_navigation.final_revision;
  local.input_partition_checksum = opaqueHash(93U);
  local.checksum = required(core::recomputeCondensedBoundaryTransitionChecksum(local));

  core::SealedBoundaryTransition transition{previous.ref, current.ref, std::move(local), {}};
  transition.checksum = required(core::recomputeSealedBoundaryTransitionChecksum(transition));
  current.previous = previous.ref;
  current.from_previous = std::move(transition);
  finalizeSealEnvelope(&current);
  return current;
}

inline void replacePredecessorRef(core::SparseSubmapSeal* current,
                                  const core::SubmapRef& replacement) {
  current->previous = replacement;
  current->from_previous->from = replacement;
  current->from_previous->checksum =
      required(core::recomputeSealedBoundaryTransitionChecksum(*current->from_previous));
  finalizeSealEnvelope(current);
}

inline void replaceBoundaryFactor(core::SparseSubmapSeal* current,
                                  core::FrozenSquareRootFactor factor) {
  current->from_previous->local_transition.boundary_factor = std::move(factor);
  current->from_previous->local_transition.checksum = required(
      core::recomputeCondensedBoundaryTransitionChecksum(current->from_previous->local_transition));
  current->from_previous->checksum =
      required(core::recomputeSealedBoundaryTransitionChecksum(*current->from_previous));
  finalizeSealEnvelope(current);
}

}  // namespace meridian::global::test_support
