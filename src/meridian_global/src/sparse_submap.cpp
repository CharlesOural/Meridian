#include "meridian/global/sparse_submap.hpp"

#include "persistence_internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Eigenvalues>

namespace meridian::global {
namespace {

using ResultBool = core::Result<bool, SparseSubmapError>;

[[nodiscard]] SparseSubmapError error(SparseSubmapErrorCode code, std::string detail) {
  return SparseSubmapError{code, std::move(detail)};
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool finiteState(const core::NavStateEstimate& state) noexcept {
  return finitePose(state.T_odom_imu) && state.velocity_odom.allFinite() &&
         state.gyro_bias.allFinite() && state.accel_bias.allFinite();
}

[[nodiscard]] bool zeroHash(const core::ContentHash& hash) noexcept {
  return std::all_of(hash.begin(), hash.end(), [](std::uint8_t value) { return value == 0U; });
}

using persistence_internal::Sha256;

class CanonicalWriter {
 public:
  void unsigned64(std::uint64_t value) noexcept {
    std::array<std::byte, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
      bytes[7U - index] = static_cast<std::byte>(value >> (index * 8U));
    }
    hash_.update(bytes);
  }

  void signed64(std::int64_t value) noexcept {
    unsigned64(std::bit_cast<std::uint64_t>(value));
  }

  void floating(double value) noexcept {
    unsigned64(std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value));
  }

  void bytes(const core::ContentHash& value) noexcept {
    hash_.update(std::as_bytes(std::span(value)));
  }

  void raw(std::span<const std::byte> value) noexcept { hash_.update(value); }

  template <typename Tag>
  void id(core::StrongId<Tag> value) noexcept {
    unsigned64(value.value());
  }

  void pose(const core::Pose3d& value) noexcept {
    const Eigen::Matrix4d matrix = value.matrix();
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
        floating(matrix(row, column));
      }
    }
  }

  template <typename Derived>
  void matrix(const Eigen::MatrixBase<Derived>& value) noexcept {
    unsigned64(static_cast<std::uint64_t>(value.rows()));
    unsigned64(static_cast<std::uint64_t>(value.cols()));
    for (Eigen::Index row = 0; row < value.rows(); ++row) {
      for (Eigen::Index column = 0; column < value.cols(); ++column) {
        floating(value(row, column));
      }
    }
  }

  [[nodiscard]] core::ContentHash finish() noexcept { return hash_.finish(); }

 private:
  Sha256 hash_;
};

[[nodiscard]] core::ContentHash hashBytes(const core::ImmutableBytes& bytes) noexcept {
  return persistence_internal::hashBytes(bytes ? std::span<const std::byte>(*bytes)
                                               : std::span<const std::byte>{});
}

[[nodiscard]] bool sameCorrelation(const core::CorrelationDeclaration& lhs,
                                   const core::CorrelationDeclaration& rhs) noexcept {
  return lhs.group == rhs.group && lhs.policy == rhs.policy && lhs.treatment == rhs.treatment &&
         lhs.covariance_inflation == rhs.covariance_inflation &&
         lhs.total_information_cap == rhs.total_information_cap;
}

void writeSlice(CanonicalWriter& writer, const core::ObservationSlice& slice) noexcept {
  writer.unsigned64(static_cast<std::uint64_t>(slice.root.index()));
  std::visit([&writer](const auto& id) { writer.id(id); }, slice.root);
  writer.unsigned64(static_cast<std::uint64_t>(slice.kind));
  writer.unsigned64(slice.begin);
  writer.unsigned64(slice.end);
  writer.bytes(slice.source_checksum);
  writer.id(slice.calibration);
}

[[nodiscard]] core::ContentHash hashLineage(const core::ObservationLineage& lineage) noexcept {
  CanonicalWriter writer;
  writer.id(lineage.id);
  writer.unsigned64(lineage.usage.size());
  for (const auto& usage : lineage.usage) {
    writeSlice(writer, usage.slice);
    writer.unsigned64(static_cast<std::uint64_t>(usage.role));
    writer.id(usage.consumer);
    writer.unsigned64(usage.factor_group.has_value() ? 1U : 0U);
    if (usage.factor_group) {
      writer.id(*usage.factor_group);
    }
    writer.unsigned64(usage.correlation_group.has_value() ? 1U : 0U);
    if (usage.correlation_group) {
      writer.id(*usage.correlation_group);
    }
  }
  writer.unsigned64(lineage.correlations.size());
  for (const auto& declaration : lineage.correlations) {
    writer.id(declaration.group);
    writer.id(declaration.policy);
    writer.unsigned64(static_cast<std::uint64_t>(declaration.treatment));
    writer.floating(declaration.covariance_inflation);
    writer.unsigned64(declaration.total_information_cap.has_value() ? 1U : 0U);
    if (declaration.total_information_cap) {
      writer.floating(*declaration.total_information_cap);
    }
  }
  return writer.finish();
}

[[nodiscard]] bool checkedAdd(std::uint64_t lhs, std::uint64_t rhs,
                              std::uint64_t* output) noexcept {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return false;
  }
  *output = lhs + rhs;
  return true;
}

[[nodiscard]] std::uint64_t payloadBytes(const PlacePayloadInput& payload) noexcept {
  if (payload.record) {
    return payload.record->bytes;
  }
  return payload.in_memory ? static_cast<std::uint64_t>(payload.in_memory->size()) : 0U;
}

[[nodiscard]] bool validPoseCovariance(const core::PoseCovariance& covariance,
                                       const SparseSubmapConfig& config) noexcept {
  if (covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !covariance.matrix.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  if ((covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff() >
      config.covariance_symmetry_relative_tolerance * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<core::Matrix6d> solver(
      0.5 * (covariance.matrix + covariance.matrix.transpose()), Eigen::EigenvaluesOnly);
  return solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
         solver.eigenvalues().minCoeff() >= -config.minimum_supported_covariance_eigenvalue;
}

[[nodiscard]] bool validHeader(const core::RecordHeader& header) noexcept {
  return header.schema_version > 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid();
}

}  // namespace

struct SparseSubmapCoordinator::Impl {
  struct StateRecord {
    LocalStateContributionInput staged;
    std::optional<FinalizedLocalStateInput> finalized;
  };

  struct Builder {
    Builder(core::SubmapId submap_id, SparseSubmapLifecycle initial_lifecycle,
            core::StateId boundary_state, core::FusionTime boundary_time)
        : id(submap_id),
          lifecycle(initial_lifecycle),
          start_state(boundary_state),
          start_time(boundary_time) {}

    core::SubmapId id;
    SparseSubmapLifecycle lifecycle{SparseSubmapLifecycle::Active};
    core::StateId start_state;
    core::FusionTime start_time;
    std::optional<core::StateId> end_state;
    std::optional<core::FusionTime> end_time;
    std::vector<core::StateId> core_states;
    std::vector<FinalizedFactorInput> factors;
    std::vector<FinalizedLidarKeyframeInput> lidar_keyframes;
    std::vector<FinalizedVisualKeyframeInput> visual_keyframes;
    std::optional<FinalizedBoundaryCondensationInput> condensation;
  };

  struct PendingAdjacent {
    core::SubmapRef from;
    core::StateId to_boundary_state;
    core::Pose3d T_from_to;
    core::PoseCovariance covariance;
    core::RankAwareInformation information;
    core::ObservationLineage lineage;
    std::vector<core::FactorId> eliminated_factor_ids;
    core::LocalGraphRevision final_revision;
  };

  explicit Impl(SparseSubmapConfig input_config) : config(std::move(input_config)) {
    config_error = validateConfig();
  }

  [[nodiscard]] std::optional<SparseSubmapError> validateConfig() const {
    const auto& policy = config.split;
    if (!policy.revision.valid() || policy.revision.value() == 0U ||
        !config.first_submap_id.valid() || config.first_submap_id.value() == 0U ||
        !config.content_revision.valid() || config.content_revision.value() == 0U ||
        !std::isfinite(policy.maximum_travel_m) ||
        policy.maximum_travel_m <= 0.0 || !std::isfinite(policy.maximum_rotation_rad) ||
        policy.maximum_rotation_rad <= 0.0 || policy.maximum_duration.nanoseconds <= 0 ||
        policy.maximum_keyframes == 0U || policy.maximum_payload_bytes == 0U ||
        config.maximum_staged_states < 2U || config.maximum_factors_per_submap == 0U ||
        config.maximum_factor_support_states == 0U || config.maximum_keyframes_per_submap == 0U ||
        config.maximum_lineage_usages_per_submap == 0U ||
        config.maximum_lineage_correlations_per_submap == 0U ||
        config.maximum_builder_bytes < policy.maximum_payload_bytes ||
        config.maximum_place_payload_bytes == 0U ||
        config.maximum_proxy_input_samples_per_submap == 0U ||
        config.maximum_registration_proxy_points == 0U || config.maximum_pending_seals == 0U ||
        config.maximum_seal_identities == 0U || config.maximum_seen_input_ids == 0U ||
        config.maximum_consumed_primary_slices == 0U ||
        !std::isfinite(config.registration_voxel_resolution_m) ||
        config.registration_voxel_resolution_m <= 0.0 ||
        !std::isfinite(config.minimum_heading_projection_norm) ||
        config.minimum_heading_projection_norm <= 0.0 ||
        !std::isfinite(config.covariance_symmetry_relative_tolerance) ||
        config.covariance_symmetry_relative_tolerance <= 0.0 ||
        !std::isfinite(config.minimum_supported_covariance_eigenvalue) ||
        config.minimum_supported_covariance_eigenvalue <= 0.0 ||
        !std::isfinite(config.maximum_supported_covariance_condition) ||
        config.maximum_supported_covariance_condition <= 1.0) {
      return error(SparseSubmapErrorCode::InvalidConfiguration,
                   "sparse submap configuration contains an invalid or inconsistent hard bound");
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<SparseSubmapError> ready() const {
    return config_error;
  }

  [[nodiscard]] std::optional<SparseSubmapError> validateSession(
      const core::RecordHeader& header) const {
    if (!validHeader(header)) {
      return error(SparseSubmapErrorCode::InvalidRecord,
                   "record header contains an invalid schema or typed identity");
    }
    if (session && header.session != *session) {
      return error(SparseSubmapErrorCode::SessionMismatch,
                   "record session does not match the active sparse-submap session");
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<SparseSubmapError> rejectLateRecord(
      core::FusionTime terminal_time) const {
    if (finalized_through && terminal_time < *finalized_through) {
      return error(SparseSubmapErrorCode::RecordOutsideOpenInterval,
                   "record arrived behind the declared local finality barrier");
    }
    return std::nullopt;
  }

  [[nodiscard]] Builder* owner(core::FusionTime terminal_time) noexcept {
    if (finalizing && finalizing->end_time && terminal_time < *finalizing->end_time) {
      return &*finalizing;
    }
    if (active && terminal_time >= active->start_time) {
      return &*active;
    }
    return nullptr;
  }

  [[nodiscard]] const StateRecord* state(core::StateId id) const noexcept {
    const auto found = states.find(id.value());
    return found == states.end() ? nullptr : &found->second;
  }

  [[nodiscard]] StateRecord* state(core::StateId id) noexcept {
    const auto found = states.find(id.value());
    return found == states.end() ? nullptr : &found->second;
  }

  [[nodiscard]] std::optional<SparseSubmapError> validatePayload(
      PlacePayloadInput* payload) const {
    if ((payload->record.has_value() ? 1U : 0U) + (payload->in_memory ? 1U : 0U) != 1U) {
      return error(SparseSubmapErrorCode::InvalidRecord,
                   "place payload must contain exactly one BlobRef or immutable in-memory record");
    }
    const std::uint64_t bytes = payloadBytes(*payload);
    if (bytes == 0U || bytes > config.maximum_place_payload_bytes) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "place payload byte count is zero or exceeds its hard cap");
    }
    if (payload->record) {
      if (!payload->record->store.valid() || !payload->record->id.valid() ||
          !payload->record->layout.valid() || payload->record->bytes == 0U ||
          zeroHash(payload->record->checksum)) {
        return error(SparseSubmapErrorCode::InvalidRecord,
                     "place payload BlobRef identity/checksum/layout is invalid");
      }
      if (!zeroHash(payload->checksum) && payload->checksum != payload->record->checksum) {
        return error(SparseSubmapErrorCode::InvalidRecord,
                     "place payload checksum conflicts with its BlobRef checksum");
      }
      payload->checksum = payload->record->checksum;
    } else {
      const core::ContentHash computed = hashBytes(payload->in_memory);
      if (!zeroHash(payload->checksum) && payload->checksum != computed) {
        return error(SparseSubmapErrorCode::InvalidRecord,
                     "in-memory place payload checksum does not match its immutable bytes");
      }
      payload->checksum = computed;
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<SparseSubmapError> validateLineageBounds(
      const core::ObservationLineage& lineage) const {
    if (!lineage.id.valid() || core::validateLineage(lineage) != core::LineageValidationError::None) {
      return error(SparseSubmapErrorCode::InvalidLineage,
                   "record lineage failed identity, slice, correlation, or duplicate-use validation");
    }
    if (lineage.usage.size() > config.maximum_lineage_usages_per_submap ||
        lineage.correlations.size() > config.maximum_lineage_correlations_per_submap) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "record lineage exceeds a configured usage/correlation hard cap");
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<SparseSubmapError> reserveSeenId(std::uint64_t id,
                                                               const char* kind) {
    if (seen_input_ids.contains(std::pair<std::string, std::uint64_t>{kind, id})) {
      return error(SparseSubmapErrorCode::DuplicateIdentity,
                   std::string(kind) + " identity was already admitted");
    }
    if (seen_input_ids.size() >= config.maximum_seen_input_ids) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "seen input identity index reached its mission hard cap");
    }
    return std::nullopt;
  }

  void commitSeenId(std::uint64_t id, const char* kind) {
    seen_input_ids.emplace(kind, id);
  }

  [[nodiscard]] std::optional<SparseSubmapError> reservePrimarySlices(
      const core::ObservationLineage& lineage) const {
    std::size_t additions = 0U;
    for (const auto& usage : lineage.usage) {
      if (usage.role != core::ObservationRole::PrimaryResidual) {
        continue;
      }
      ++additions;
      for (const auto& existing : primary_slices) {
        if (usage.slice.overlaps(existing)) {
          return error(SparseSubmapErrorCode::NonDisjointLineage,
                       "factor primary observation overlaps an already owned core partition");
        }
      }
    }
    if (additions > config.maximum_consumed_primary_slices - primary_slices.size()) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "primary observation ownership index reached its mission hard cap");
    }
    return std::nullopt;
  }

  void commitPrimarySlices(const core::ObservationLineage& lineage) {
    for (const auto& usage : lineage.usage) {
      if (usage.role == core::ObservationRole::PrimaryResidual) {
        primary_slices.push_back(usage.slice);
      }
    }
  }

  [[nodiscard]] std::optional<SparseSubmapError> builderUsageWithinCapacity(
      const Builder& builder) const {
    std::size_t usages = 0U;
    std::size_t correlations = 0U;
    const auto accumulate = [&](const core::ObservationLineage& lineage) {
      usages += lineage.usage.size();
      correlations += lineage.correlations.size();
    };
    for (const auto& factor : builder.factors) {
      accumulate(factor.lineage);
    }
    for (const auto& keyframe : builder.lidar_keyframes) {
      accumulate(keyframe.lineage);
    }
    for (const auto& keyframe : builder.visual_keyframes) {
      accumulate(keyframe.lineage);
    }
    if (usages > config.maximum_lineage_usages_per_submap ||
        correlations > config.maximum_lineage_correlations_per_submap) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "submap aggregate lineage exceeds its hard cap");
    }
    return std::nullopt;
  }

  [[nodiscard]] core::Result<std::uint64_t, SparseSubmapError> builderBytes(
      const Builder& builder) const {
    using Result = core::Result<std::uint64_t, SparseSubmapError>;
    std::uint64_t total = 0U;
    const auto add = [&total](std::uint64_t value) { return checkedAdd(total, value, &total); };
    for (const core::StateId id : builder.core_states) {
      const StateRecord* record = state(id);
      if (record == nullptr || !add(record->staged.retained_bytes)) {
        return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                     "builder byte accounting overflowed or lost a state"));
      }
    }
    for (const auto& factor : builder.factors) {
      if (!add(factor.retained_bytes)) {
        return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                     "factor byte accounting overflowed"));
      }
    }
    for (const auto& keyframe : builder.lidar_keyframes) {
      const std::uint64_t samples =
          static_cast<std::uint64_t>(keyframe.registration_samples.size()) *
          static_cast<std::uint64_t>(sizeof(LidarProxySample));
      if (!add(keyframe.retained_bytes) || !add(samples) ||
          (keyframe.place_payload && !add(payloadBytes(*keyframe.place_payload)))) {
        return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                     "LiDAR keyframe byte accounting overflowed"));
      }
    }
    for (const auto& keyframe : builder.visual_keyframes) {
      if (!add(keyframe.retained_bytes) || !add(payloadBytes(keyframe.place_payload))) {
        return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                     "visual keyframe byte accounting overflowed"));
      }
    }
    return Result::success(total);
  }

  [[nodiscard]] std::optional<SparseSubmapError> validateBuilderCapacity(
      const Builder& builder) const {
    if (builder.factors.size() > config.maximum_factors_per_submap ||
        builder.lidar_keyframes.size() + builder.visual_keyframes.size() >
            config.maximum_keyframes_per_submap) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "submap factor/keyframe hard cap reached");
    }
    std::size_t proxy_samples = 0U;
    for (const auto& keyframe : builder.lidar_keyframes) {
      if (keyframe.registration_samples.size() >
          config.maximum_proxy_input_samples_per_submap - proxy_samples) {
        return error(SparseSubmapErrorCode::CapacityExceeded,
                     "registration proxy input sample hard cap reached");
      }
      proxy_samples += keyframe.registration_samples.size();
    }
    if (const auto lineage_error = builderUsageWithinCapacity(builder)) {
      return lineage_error;
    }
    const auto bytes = builderBytes(builder);
    if (!bytes) {
      return bytes.error();
    }
    if (bytes.value() > config.maximum_builder_bytes) {
      return error(SparseSubmapErrorCode::CapacityExceeded,
                   "submap retained-byte hard cap reached");
    }
    return std::nullopt;
  }

  [[nodiscard]] core::Result<core::Pose3d, SparseSubmapError> boundaryFrame(
      core::StateId boundary) const {
    using Result = core::Result<core::Pose3d, SparseSubmapError>;
    const StateRecord* record = state(boundary);
    if (record == nullptr || !record->finalized) {
      return Result::failure(error(SparseSubmapErrorCode::BoundaryNotFinalized,
                                   "submap boundary state is not explicitly finalized/out of lag"));
    }
    const Eigen::Vector3d body_x = record->finalized->final_estimate.T_odom_imu.so3().matrix().col(0);
    const double heading_norm = std::hypot(body_x.x(), body_x.y());
    if (!std::isfinite(heading_norm) || heading_norm < config.minimum_heading_projection_norm) {
      return Result::failure(error(SparseSubmapErrorCode::BoundaryYawUnobservable,
                                   "boundary body x-axis has no observable gravity-plane heading"));
    }
    const double yaw = std::atan2(body_x.y(), body_x.x());
    const Sophus::SO3d R_odom_submap =
        Sophus::SO3d::exp(Eigen::Vector3d{0.0, 0.0, yaw});
    return Result::success(core::Pose3d(
        R_odom_submap, record->finalized->final_estimate.T_odom_imu.translation()));
  }

  [[nodiscard]] std::optional<SparseSubmapError> allStatesFinal(const Builder& builder) const {
    std::set<std::uint64_t> required;
    required.insert(builder.start_state.value());
    if (builder.end_state) {
      required.insert(builder.end_state->value());
    }
    for (const core::StateId id : builder.core_states) {
      required.insert(id.value());
    }
    for (const auto& factor : builder.factors) {
      for (const core::StateId id : factor.support_states) {
        required.insert(id.value());
      }
    }
    for (const auto& keyframe : builder.lidar_keyframes) {
      required.insert(keyframe.state.value());
    }
    for (const auto& keyframe : builder.visual_keyframes) {
      required.insert(keyframe.state.value());
    }
    for (const std::uint64_t id : required) {
      const StateRecord* record = state(core::StateId(id));
      if (record == nullptr || !record->finalized ||
          record->finalized->finality != LocalStateFinality::FinalizedAndOutOfLag) {
        return error(SparseSubmapErrorCode::BoundaryNotFinalized,
                     "at least one submap-contributing state remains inside the local lag");
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<SparseSubmapError> finalRevisionsCoverPartition(
      const Builder& builder) const {
    if (!builder.condensation || !finality_revision ||
        *finality_revision < builder.condensation->final_revision) {
      return error(SparseSubmapErrorCode::InvalidCondensation,
                   "finality barrier revision does not cover the boundary condensation");
    }
    const core::LocalGraphRevision condensation_revision =
        builder.condensation->final_revision;
    const auto state_is_covered = [&](core::StateId id) {
      const StateRecord* record = state(id);
      return record != nullptr && record->finalized &&
             record->finalized->final_revision <= condensation_revision;
    };
    if (!state_is_covered(builder.start_state) ||
        (builder.end_state && !state_is_covered(*builder.end_state))) {
      return error(SparseSubmapErrorCode::InvalidCondensation,
                   "condensation revision predates a finalized boundary state");
    }
    for (const auto id : builder.core_states) {
      if (!state_is_covered(id)) {
        return error(SparseSubmapErrorCode::InvalidCondensation,
                     "condensation revision predates a finalized core state");
      }
    }
    for (const auto& factor : builder.factors) {
      if (factor.final_revision > condensation_revision) {
        return error(SparseSubmapErrorCode::InvalidCondensation,
                     "condensation revision predates a finalized factor in its partition");
      }
    }
    for (const auto& keyframe : builder.lidar_keyframes) {
      if (keyframe.final_revision > condensation_revision) {
        return error(SparseSubmapErrorCode::InvalidCondensation,
                     "condensation revision predates a finalized LiDAR keyframe");
      }
    }
    for (const auto& keyframe : builder.visual_keyframes) {
      if (keyframe.final_revision > condensation_revision) {
        return error(SparseSubmapErrorCode::InvalidCondensation,
                     "condensation revision predates a finalized visual keyframe");
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] core::Result<core::ObservationLineage, SparseSubmapError> combinedLineage(
      const Builder& builder) const {
    using Result = core::Result<core::ObservationLineage, SparseSubmapError>;
    core::ObservationLineage output;
    output.id = core::ObservationLineageId(builder.id.value());
    std::vector<const FinalizedFactorInput*> ordered;
    ordered.reserve(builder.factors.size());
    for (const auto& factor : builder.factors) {
      if (factor.kind != FinalizedFactorKind::IncomingMarginalPrior) {
        ordered.push_back(&factor);
      }
    }
    std::sort(ordered.begin(), ordered.end(), [](const auto* lhs, const auto* rhs) {
      return std::tie(lhs->terminal_time, lhs->factor) <
             std::tie(rhs->terminal_time, rhs->factor);
    });
    std::map<std::uint64_t, core::CorrelationDeclaration> declarations;
    for (const auto* factor : ordered) {
      output.usage.insert(output.usage.end(), factor->lineage.usage.begin(),
                          factor->lineage.usage.end());
      for (const auto& declaration : factor->lineage.correlations) {
        const auto found = declarations.find(declaration.group.value());
        if (found != declarations.end() && !sameCorrelation(found->second, declaration)) {
          return Result::failure(error(
              SparseSubmapErrorCode::InvalidLineage,
              "factor lineages contain conflicting declarations for one correlation group"));
        }
        declarations.emplace(declaration.group.value(), declaration);
      }
    }
    for (const auto& [unused, declaration] : declarations) {
      static_cast<void>(unused);
      output.correlations.push_back(declaration);
    }
    if (output.usage.size() > config.maximum_lineage_usages_per_submap ||
        output.correlations.size() > config.maximum_lineage_correlations_per_submap ||
        core::validateLineage(output) != core::LineageValidationError::None) {
      return Result::failure(error(SparseSubmapErrorCode::InvalidLineage,
                                   "combined factor partition lineage is invalid or over capacity"));
    }
    output.checksum = hashLineage(output);
    return Result::success(std::move(output));
  }

  struct RelativeUncertainty {
    core::PoseCovariance covariance;
    core::RankAwareInformation information;
  };

  [[nodiscard]] core::Result<RelativeUncertainty, SparseSubmapError> relativeUncertainty(
      const FinalizedBoundaryCondensationInput& input, const core::Pose3d& T_from_to) const {
    using Result = core::Result<RelativeUncertainty, SparseSubmapError>;
    const double scale = std::max(1.0, input.joint_pose_covariance.cwiseAbs().maxCoeff());
    if (input.covariance_order !=
            BoundaryJointCovarianceOrder::FromThenToRightTranslationFirst ||
        !input.joint_pose_covariance.allFinite() || input.supported_relative_rank == 0U ||
        input.supported_relative_rank > 6U ||
        !input.relative_information_basis.allFinite() ||
        (input.relative_information_basis.transpose() * input.relative_information_basis -
         core::Matrix6d::Identity())
                .cwiseAbs()
                .maxCoeff() > 1.0e-8 ||
        (input.joint_pose_covariance - input.joint_pose_covariance.transpose())
                .cwiseAbs()
                .maxCoeff() >
            config.covariance_symmetry_relative_tolerance * scale) {
      return Result::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                   "boundary joint covariance/order/rank is invalid"));
    }
    const Eigen::Matrix<double, 12, 12> symmetric =
        0.5 * (input.joint_pose_covariance + input.joint_pose_covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> joint_solver(
        symmetric, Eigen::EigenvaluesOnly);
    if (joint_solver.info() != Eigen::Success || !joint_solver.eigenvalues().allFinite() ||
        joint_solver.eigenvalues().minCoeff() <
            -config.minimum_supported_covariance_eigenvalue * scale) {
      return Result::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                   "boundary joint covariance is not positive semidefinite"));
    }

    Eigen::Matrix<double, 6, 12> jacobian;
    jacobian.leftCols<6>() = -T_from_to.inverse().Adj();
    jacobian.rightCols<6>().setIdentity();
    const core::Matrix6d covariance_matrix = jacobian * symmetric * jacobian.transpose();
    const Eigen::Index rank = static_cast<Eigen::Index>(input.supported_relative_rank);
    const Eigen::Matrix<double, 6, Eigen::Dynamic> supported_basis =
        input.relative_information_basis.leftCols(rank);
    const Eigen::MatrixXd supported_covariance =
        supported_basis.transpose() * covariance_matrix * supported_basis;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (supported_covariance + supported_covariance.transpose()));
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
        !solver.eigenvectors().allFinite()) {
      return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                   "relative boundary covariance eigendecomposition failed"));
    }
    const double minimum = solver.eigenvalues()(0);
    const double maximum_supported = solver.eigenvalues()(rank - 1);
    if (minimum < config.minimum_supported_covariance_eigenvalue ||
        maximum_supported / minimum > config.maximum_supported_covariance_condition) {
      return Result::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                   "supported relative covariance is singular or ill-conditioned"));
    }

    RelativeUncertainty output;
    output.covariance.matrix = 0.5 * (covariance_matrix + covariance_matrix.transpose());
    output.information.basis.leftCols(rank) = supported_basis * solver.eigenvectors();
    if (rank < 6) {
      output.information.basis.rightCols(6 - rank) =
          input.relative_information_basis.rightCols(6 - rank);
    }
    output.information.rank = input.supported_relative_rank;
    for (Eigen::Index index = 0; index < rank; ++index) {
      output.information.eigenvalues(index) = 1.0 / solver.eigenvalues()(index);
    }
    return Result::success(std::move(output));
  }

  struct ProxyCandidate {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    RegistrationProxyPoint point;
    core::SweepId sweep;
    std::size_t source_index{};
  };

  [[nodiscard]] core::Result<RegistrationProxy, SparseSubmapError> buildProxy(
      const Builder& builder, const core::Pose3d& T_odom_submap) const {
    using Result = core::Result<RegistrationProxy, SparseSubmapError>;
    std::vector<ProxyCandidate> candidates;
    for (const auto& keyframe : builder.lidar_keyframes) {
      const StateRecord* state_record = state(keyframe.state);
      if (state_record == nullptr || !state_record->finalized) {
        return Result::failure(error(SparseSubmapErrorCode::BoundaryNotFinalized,
                                     "LiDAR proxy references a non-finalized state"));
      }
      const core::Pose3d T_submap_lidar =
          T_odom_submap.inverse() * state_record->finalized->final_estimate.T_odom_imu *
          keyframe.T_imu_lidar;
      for (std::size_t index = 0; index < keyframe.registration_samples.size(); ++index) {
        const auto& sample = keyframe.registration_samples[index];
        RegistrationProxyPoint transformed;
        transformed.point_submap = T_submap_lidar * sample.point_lidar;
        transformed.normal_submap = T_submap_lidar.so3() * sample.normal_lidar.normalized();
        transformed.weight = sample.weight;
        const Eigen::Vector3d scaled =
            transformed.point_submap / config.registration_voxel_resolution_m;
        constexpr double kIndexLimit =
            static_cast<double>(std::numeric_limits<std::int64_t>::max() / 2);
        if (!transformed.point_submap.allFinite() || !transformed.normal_submap.allFinite() ||
            !std::isfinite(transformed.weight) || scaled.cwiseAbs().maxCoeff() > kIndexLimit) {
          return Result::failure(error(SparseSubmapErrorCode::NumericalFailure,
                                       "registration proxy transform or voxel index is non-finite"));
        }
        candidates.push_back(ProxyCandidate{
            static_cast<std::int64_t>(std::floor(scaled.x())),
            static_cast<std::int64_t>(std::floor(scaled.y())),
            static_cast<std::int64_t>(std::floor(scaled.z())), transformed, keyframe.sweep, index});
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const ProxyCandidate& lhs,
                                                       const ProxyCandidate& rhs) {
      return std::tie(lhs.x, lhs.y, lhs.z, lhs.point.point_submap.x(),
                      lhs.point.point_submap.y(), lhs.point.point_submap.z(), lhs.sweep,
                      lhs.source_index) <
             std::tie(rhs.x, rhs.y, rhs.z, rhs.point.point_submap.x(),
                      rhs.point.point_submap.y(), rhs.point.point_submap.z(), rhs.sweep,
                      rhs.source_index);
    });
    RegistrationProxy output;
    output.voxel_resolution_m = config.registration_voxel_resolution_m;
    for (const auto& candidate : candidates) {
      if (!output.points.empty()) {
        const auto& previous = candidates[&candidate - candidates.data() - 1U];
        if (candidate.x == previous.x && candidate.y == previous.y && candidate.z == previous.z) {
          continue;
        }
      }
      if (output.points.size() >= config.maximum_registration_proxy_points) {
        return Result::failure(error(SparseSubmapErrorCode::CapacityExceeded,
                                     "unique registration proxy voxels exceed their hard cap"));
      }
      output.points.push_back(candidate.point);
    }
    CanonicalWriter writer;
    writer.floating(output.voxel_resolution_m);
    writer.unsigned64(output.points.size());
    for (const auto& point : output.points) {
      writer.matrix(point.point_submap);
      writer.matrix(point.normal_submap);
      writer.floating(point.weight);
    }
    output.checksum = writer.finish();
    return Result::success(std::move(output));
  }

  [[nodiscard]] core::Result<SparseSubmapSeal, SparseSubmapError> buildSeal(
      const Builder& builder) const {
    using Result = core::Result<SparseSubmapSeal, SparseSubmapError>;
    if (!builder.end_state || !builder.end_time || !builder.condensation) {
      return Result::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                   "finalizing submap lacks an end boundary or condensation"));
    }
    const auto origin = boundaryFrame(builder.start_state);
    if (!origin) {
      return Result::failure(origin.error());
    }
    const auto endpoint = boundaryFrame(*builder.end_state);
    if (!endpoint) {
      return Result::failure(endpoint.error());
    }
    const core::Pose3d T_from_to = origin.value().inverse() * endpoint.value();
    const auto uncertainty = relativeUncertainty(*builder.condensation, T_from_to);
    if (!uncertainty) {
      return Result::failure(uncertainty.error());
    }
    const auto lineage = combinedLineage(builder);
    if (!lineage) {
      return Result::failure(lineage.error());
    }
    const auto proxy = buildProxy(builder, origin.value());
    if (!proxy) {
      return Result::failure(proxy.error());
    }

    const core::RecordHeader& seal_header = states.at(builder.start_state.value()).staged.header;
    const FinalizedSubmapFrame submap_ref{
        core::SubmapRef{seal_header.session, *odom_epoch, builder.id, *calibration,
                        config.content_revision, {}},
        origin.value(), *builder.end_time};
    auto mutable_seal = std::make_shared<SparseSubmapSealRecord>(submap_ref);
    mutable_seal->header = seal_header;
    mutable_seal->header.created_at = *builder.end_time;
    mutable_seal->core_interval = core::TimeRange{builder.start_time, *builder.end_time};
    mutable_seal->start_boundary_state = builder.start_state;
    mutable_seal->end_boundary_state = *builder.end_state;
    mutable_seal->final_local_revision = builder.condensation->final_revision;
    mutable_seal->calibration_epochs.push_back(*calibration);
    mutable_seal->core_state_ids = builder.core_states;
    std::sort(mutable_seal->core_state_ids.begin(), mutable_seal->core_state_ids.end(),
              [this](core::StateId lhs, core::StateId rhs) {
                const auto& left = states.at(lhs.value()).staged;
                const auto& right = states.at(rhs.value()).staged;
                return std::tie(left.exact_time, left.state) < std::tie(right.exact_time, right.state);
              });
    const Eigen::Matrix3d R_submap_odom = origin.value().so3().inverse().matrix();
    for (const core::StateId id : mutable_seal->core_state_ids) {
      const auto& final_state = *states.at(id.value()).finalized;
      mutable_seal->finalized_trajectory.push_back(FinalizedSubmapStateRecord{
          id,
          final_state.exact_time,
          final_state.final_revision,
          origin.value().inverse() * final_state.final_estimate.T_odom_imu,
          R_submap_odom * final_state.final_estimate.velocity_odom,
          final_state.final_estimate.gyro_bias,
          final_state.final_estimate.accel_bias,
          final_state.pose_covariance});
    }
    mutable_seal->factor_lineage = lineage.value();
    mutable_seal->registration_proxy = proxy.value();

    std::vector<const FinalizedFactorInput*> ordered_factors;
    for (const auto& factor : builder.factors) {
      if (factor.kind != FinalizedFactorKind::IncomingMarginalPrior) {
        ordered_factors.push_back(&factor);
      }
    }
    std::sort(ordered_factors.begin(), ordered_factors.end(), [](const auto* lhs, const auto* rhs) {
      return std::tie(lhs->terminal_time, lhs->factor) <
             std::tie(rhs->terminal_time, rhs->factor);
    });
    for (const auto* factor : ordered_factors) {
      mutable_seal->condensed_factor_ids.push_back(factor->factor);
    }

    std::vector<const FinalizedLidarKeyframeInput*> lidar;
    for (const auto& keyframe : builder.lidar_keyframes) {
      lidar.push_back(&keyframe);
    }
    std::sort(lidar.begin(), lidar.end(), [](const auto* lhs, const auto* rhs) {
      return std::tie(lhs->terminal_time, lhs->sweep) < std::tie(rhs->terminal_time, rhs->sweep);
    });
    for (const auto* keyframe : lidar) {
      if (!keyframe->place_payload) {
        continue;
      }
      const auto& final_state = *states.at(keyframe->state.value()).finalized;
      mutable_seal->lidar_place_index.push_back(LidarPlacePayloadIndexEntry{
          keyframe->sweep, keyframe->state, keyframe->terminal_time,
          origin.value().inverse() * final_state.final_estimate.T_odom_imu * keyframe->T_imu_lidar,
          *keyframe->place_payload, keyframe->lineage});
    }

    std::vector<const FinalizedVisualKeyframeInput*> visual;
    for (const auto& keyframe : builder.visual_keyframes) {
      visual.push_back(&keyframe);
    }
    std::sort(visual.begin(), visual.end(), [](const auto* lhs, const auto* rhs) {
      return std::tie(lhs->terminal_time, lhs->frame) < std::tie(rhs->terminal_time, rhs->frame);
    });
    for (const auto* keyframe : visual) {
      const auto& final_state = *states.at(keyframe->state.value()).finalized;
      mutable_seal->visual_place_index.push_back(VisualPlacePayloadIndexEntry{
          keyframe->frame, keyframe->camera, keyframe->state, keyframe->terminal_time,
          origin.value().inverse() * final_state.final_estimate.T_odom_imu * keyframe->T_imu_camera,
          keyframe->place_payload, keyframe->lineage});
    }

    if (pending_adjacent) {
      if (pending_adjacent->to_boundary_state != builder.start_state) {
        return Result::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                     "pending adjacent edge does not terminate at this submap origin"));
      }
      mutable_seal->incoming_adjacent = AdjacentConstraintRecord{
          AdjacentSubmapAppend{
              mutable_seal->submap,
              RelativeAnchorConstraint{pending_adjacent->from, mutable_seal->submap.ref,
                                       pending_adjacent->T_from_to,
                                       pending_adjacent->information}},
          pending_adjacent->covariance, pending_adjacent->lineage,
          pending_adjacent->eliminated_factor_ids, pending_adjacent->final_revision};
    }

    CanonicalWriter writer;
    writer.id(mutable_seal->header.session);
    writer.unsigned64(mutable_seal->header.schema_version);
    writer.id(mutable_seal->header.producer);
    writer.id(mutable_seal->header.config);
    writer.id(config.split.revision);
    writer.id(builder.id);
    writer.id(config.content_revision);
    writer.id(*odom_epoch);
    writer.signed64(builder.start_time.nanoseconds);
    writer.signed64(builder.end_time->nanoseconds);
    writer.id(builder.start_state);
    writer.id(*builder.end_state);
    writer.id(mutable_seal->final_local_revision);
    writer.id(*calibration);
    writer.pose(origin.value());
    writer.unsigned64(mutable_seal->core_state_ids.size());
    for (const auto id : mutable_seal->core_state_ids) {
      writer.id(id);
    }
    writer.unsigned64(mutable_seal->finalized_trajectory.size());
    for (const auto& state_record : mutable_seal->finalized_trajectory) {
      writer.id(state_record.state);
      writer.signed64(state_record.exact_time.nanoseconds);
      writer.id(state_record.final_local_revision);
      writer.pose(state_record.T_submap_imu);
      writer.matrix(state_record.velocity_submap);
      writer.matrix(state_record.gyro_bias);
      writer.matrix(state_record.accel_bias);
      writer.matrix(state_record.pose_covariance.matrix);
    }
    writer.unsigned64(mutable_seal->condensed_factor_ids.size());
    for (const auto id : mutable_seal->condensed_factor_ids) {
      writer.id(id);
    }
    writer.bytes(mutable_seal->factor_lineage.checksum);
    writer.bytes(mutable_seal->registration_proxy.checksum);
    writer.unsigned64(mutable_seal->lidar_place_index.size());
    for (const auto& entry : mutable_seal->lidar_place_index) {
      writer.id(entry.sweep);
      writer.id(entry.state);
      writer.signed64(entry.terminal_time.nanoseconds);
      writer.pose(entry.T_submap_lidar);
      writer.bytes(entry.payload.checksum);
      writer.bytes(hashLineage(entry.lineage));
    }
    writer.unsigned64(mutable_seal->visual_place_index.size());
    for (const auto& entry : mutable_seal->visual_place_index) {
      writer.id(entry.frame);
      writer.id(entry.camera);
      writer.id(entry.state);
      writer.signed64(entry.terminal_time.nanoseconds);
      writer.pose(entry.T_submap_camera);
      writer.bytes(entry.payload.checksum);
      writer.bytes(hashLineage(entry.lineage));
    }
    writer.unsigned64(mutable_seal->incoming_adjacent.has_value() ? 1U : 0U);
    if (mutable_seal->incoming_adjacent) {
      const auto& incoming = *mutable_seal->incoming_adjacent;
      writer.id(incoming.global_append.constraint.from.session);
      writer.id(incoming.global_append.constraint.from.odom_epoch);
      writer.id(incoming.global_append.constraint.from.id);
      writer.id(incoming.global_append.constraint.from.calibration);
      writer.id(incoming.global_append.constraint.from.content_revision);
      writer.bytes(incoming.global_append.constraint.from.local_content_checksum);
      writer.id(incoming.global_append.constraint.to.session);
      writer.id(incoming.global_append.constraint.to.odom_epoch);
      writer.id(incoming.global_append.constraint.to.id);
      writer.id(incoming.global_append.constraint.to.calibration);
      writer.id(incoming.global_append.constraint.to.content_revision);
      writer.bytes(incoming.global_append.constraint.to.local_content_checksum);
      writer.pose(incoming.global_append.constraint.T_from_to);
      writer.matrix(incoming.relative_covariance.matrix);
      writer.matrix(incoming.global_append.constraint.information.basis);
      writer.matrix(incoming.global_append.constraint.information.eigenvalues);
      writer.unsigned64(incoming.global_append.constraint.information.rank);
      writer.bytes(hashLineage(incoming.lineage));
      writer.unsigned64(incoming.eliminated_factor_ids.size());
      for (const auto id : incoming.eliminated_factor_ids) {
        writer.id(id);
      }
    }
    writer.matrix(uncertainty.value().covariance.matrix);
    writer.matrix(uncertainty.value().information.basis);
    writer.matrix(uncertainty.value().information.eigenvalues);
    writer.unsigned64(uncertainty.value().information.rank);
    const core::ContentHash legacy_content_checksum = writer.finish();
    mutable_seal->submap.ref.local_content_checksum = legacy_content_checksum;
    if (mutable_seal->incoming_adjacent) {
      mutable_seal->incoming_adjacent->global_append.submap = mutable_seal->submap;
      mutable_seal->incoming_adjacent->global_append.constraint.to = mutable_seal->submap.ref;
    }
    mutable_seal->identity =
        core::SparseSubmapSealIdentity{mutable_seal->submap.ref, legacy_content_checksum};
    return Result::success(std::const_pointer_cast<const SparseSubmapSealRecord>(mutable_seal));
  }

  [[nodiscard]] core::Result<bool, SparseSubmapError> trySeal() {
    if (!finalizing || !finalizing->end_time || !finalized_through ||
        *finalized_through < *finalizing->end_time || !finalizing->condensation) {
      return ResultBool::success(false);
    }
    if (const auto states_error = allStatesFinal(*finalizing)) {
      return ResultBool::success(false);
    }
    if (const auto revision_error = finalRevisionsCoverPartition(*finalizing)) {
      return ResultBool::failure(*revision_error);
    }
    if (pending_seals.size() >= config.maximum_pending_seals ||
        seal_identities.size() >= config.maximum_seal_identities) {
      return ResultBool::failure(error(SparseSubmapErrorCode::CapacityExceeded,
                                       "seal output or identity hard cap reached"));
    }
    const auto seal = buildSeal(*finalizing);
    if (!seal) {
      return ResultBool::failure(seal.error());
    }
    const core::SparseSubmapIdentityKey identity_key =
        core::sparseSubmapIdentityKey(seal.value()->identity.ref);
    const auto existing = seal_identities.find(identity_key);
    if (existing != seal_identities.end()) {
      if (existing->second != seal.value()->identity) {
        return ResultBool::failure(error(SparseSubmapErrorCode::SealIdentityConflict,
                                         "same submap ID produced a conflicting content checksum"));
      }
      return ResultBool::success(false);
    }

    const core::Pose3d end_frame = boundaryFrame(*finalizing->end_state).value();
    const core::Pose3d start_frame = boundaryFrame(finalizing->start_state).value();
    const core::Pose3d T_from_to = start_frame.inverse() * end_frame;
    const auto uncertainty = relativeUncertainty(*finalizing->condensation, T_from_to);
    pending_adjacent = PendingAdjacent{
        seal.value()->submap.ref, *finalizing->end_state, T_from_to,
        uncertainty.value().covariance, uncertainty.value().information,
        seal.value()->factor_lineage, seal.value()->condensed_factor_ids,
        finalizing->condensation->final_revision};

    seal_identities.emplace(identity_key, seal.value()->identity);
    pending_seals.push_back(seal.value());
    finalizing.reset();
    pruneStates();
    return ResultBool::success(true);
  }

  void pruneStates() {
    std::set<std::uint64_t> required;
    const auto collect = [&required](const Builder& builder) {
      required.insert(builder.start_state.value());
      if (builder.end_state) {
        required.insert(builder.end_state->value());
      }
      for (const auto id : builder.core_states) {
        required.insert(id.value());
      }
      for (const auto& factor : builder.factors) {
        for (const auto id : factor.support_states) {
          required.insert(id.value());
        }
      }
      for (const auto& keyframe : builder.lidar_keyframes) {
        required.insert(keyframe.state.value());
      }
      for (const auto& keyframe : builder.visual_keyframes) {
        required.insert(keyframe.state.value());
      }
    };
    if (active) {
      collect(*active);
    }
    if (finalizing) {
      collect(*finalizing);
    }
    std::erase_if(states, [&required](const auto& item) { return !required.contains(item.first); });
  }

  SparseSubmapConfig config;
  std::optional<SparseSubmapError> config_error;
  std::optional<core::SessionId> session;
  std::optional<core::OdomEpoch> odom_epoch;
  std::optional<core::CalibrationEpoch> calibration;
  std::optional<core::FusionTime> last_state_time;
  std::optional<core::FusionTime> finalized_through;
  std::optional<core::LocalGraphRevision> finality_revision;
  std::optional<Builder> active;
  std::optional<Builder> finalizing;
  std::optional<PendingAdjacent> pending_adjacent;
  core::SubmapId next_submap_id;
  std::unordered_map<std::uint64_t, StateRecord> states;
  std::set<std::pair<std::string, std::uint64_t>> seen_input_ids;
  std::vector<core::ObservationSlice> primary_slices;
  std::vector<SparseSubmapSeal> pending_seals;
  std::map<core::SparseSubmapIdentityKey, core::SparseSubmapSealIdentity> seal_identities;
};

SparseSubmapCoordinator::SparseSubmapCoordinator(SparseSubmapConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {
  impl_->next_submap_id = impl_->config.first_submap_id;
}

SparseSubmapCoordinator::~SparseSubmapCoordinator() = default;
SparseSubmapCoordinator::SparseSubmapCoordinator(SparseSubmapCoordinator&&) noexcept = default;
SparseSubmapCoordinator& SparseSubmapCoordinator::operator=(SparseSubmapCoordinator&&) noexcept =
    default;

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::stageState(
    LocalStateContributionInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  if (!input.state.valid() || !input.odom_epoch.valid() || !input.calibration.valid() ||
      !input.admitted_revision.valid() || !finiteState(input.provisional_estimate) ||
      input.retained_bytes == 0U ||
      (input.header.direct_calibration && *input.header.direct_calibration != input.calibration)) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "staged local state identity/revision/estimate/bytes is invalid"));
  }
  if (impl_->last_state_time && input.exact_time <= *impl_->last_state_time) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NonMonotonicStateTime,
                                     "local state timestamps must be strictly increasing"));
  }
  if (const auto late = impl_->rejectLateRecord(input.exact_time)) {
    return ResultBool::failure(*late);
  }
  if (impl_->odom_epoch && input.odom_epoch != *impl_->odom_epoch) {
    return ResultBool::failure(error(SparseSubmapErrorCode::OdomEpochMismatch,
                                     "one coordinator instance owns exactly one odom epoch"));
  }
  if (impl_->calibration && input.calibration != *impl_->calibration) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "calibration change requires a submap boundary/new coordinator"));
  }
  if (const auto duplicate = impl_->reserveSeenId(input.state.value(), "state")) {
    return ResultBool::failure(*duplicate);
  }
  if (impl_->states.size() >= impl_->config.maximum_staged_states) {
    return ResultBool::failure(error(SparseSubmapErrorCode::CapacityExceeded,
                                     "staged local state hard cap reached"));
  }

  const bool starting = !impl_->active;
  if (starting) {
    impl_->session = input.header.session;
    impl_->odom_epoch = input.odom_epoch;
    impl_->calibration = input.calibration;
    impl_->active = Impl::Builder{impl_->next_submap_id, SparseSubmapLifecycle::Active,
                                  input.state, input.exact_time};
  }
  Impl::Builder* target = impl_->owner(input.exact_time);
  if (target == nullptr || (impl_->finalizing && target == &*impl_->finalizing &&
                            impl_->finalizing->condensation)) {
    if (starting) {
      impl_->active.reset();
      impl_->session.reset();
      impl_->odom_epoch.reset();
      impl_->calibration.reset();
    }
    return ResultBool::failure(error(SparseSubmapErrorCode::RecordOutsideOpenInterval,
                                     "state has no mutable half-open core owner"));
  }
  target->core_states.push_back(input.state);
  impl_->states.emplace(input.state.value(), Impl::StateRecord{input, std::nullopt});
  if (const auto capacity_error = impl_->validateBuilderCapacity(*target)) {
    target->core_states.pop_back();
    impl_->states.erase(input.state.value());
    if (starting) {
      impl_->active.reset();
      impl_->session.reset();
      impl_->odom_epoch.reset();
      impl_->calibration.reset();
    }
    return ResultBool::failure(*capacity_error);
  }
  impl_->commitSeenId(input.state.value(), "state");
  impl_->last_state_time = input.exact_time;
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::finalizeState(
    FinalizedLocalStateInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  Impl::StateRecord* record = impl_->state(input.state);
  if (record == nullptr) {
    return ResultBool::failure(error(SparseSubmapErrorCode::UnknownState,
                                     "finalization references an unstaged local state"));
  }
  if (input.exact_time != record->staged.exact_time || input.odom_epoch != record->staged.odom_epoch ||
      input.calibration != record->staged.calibration || !input.final_revision.valid() ||
      !finiteState(input.final_estimate) ||
      !validPoseCovariance(input.pose_covariance, impl_->config) ||
      input.finality != LocalStateFinality::FinalizedAndOutOfLag) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "final state conflicts with staged identity or covariance semantics"));
  }
  if (record->finalized) {
    const bool same = record->finalized->final_revision == input.final_revision &&
                      (record->finalized->final_estimate.T_odom_imu.matrix() -
                       input.final_estimate.T_odom_imu.matrix())
                              .cwiseAbs()
                              .maxCoeff() <= 1.0e-12 &&
                      (record->finalized->final_estimate.velocity_odom -
                       input.final_estimate.velocity_odom)
                              .cwiseAbs()
                              .maxCoeff() <= 1.0e-12 &&
                      (record->finalized->final_estimate.gyro_bias -
                       input.final_estimate.gyro_bias)
                              .cwiseAbs()
                              .maxCoeff() <= 1.0e-12 &&
                      (record->finalized->final_estimate.accel_bias -
                       input.final_estimate.accel_bias)
                              .cwiseAbs()
                              .maxCoeff() <= 1.0e-12 &&
                      (record->finalized->pose_covariance.matrix - input.pose_covariance.matrix)
                              .cwiseAbs()
                              .maxCoeff() <= 1.0e-12;
    if (!same) {
      return ResultBool::failure(error(SparseSubmapErrorCode::DuplicateIdentity,
                                       "state was finalized twice with conflicting content"));
    }
    return ResultBool::success(false);
  }
  record->finalized = std::move(input);
  const auto sealed = impl_->trySeal();
  if (!sealed) {
    return ResultBool::failure(sealed.error());
  }
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::stageFactor(
    FinalizedFactorInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->active) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NotStarted,
                                     "stage at least one local state before a factor"));
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  if (!input.factor.valid() || !input.terminal_state.valid() || !input.final_revision.valid() ||
      input.retained_bytes == 0U || input.support_states.empty() ||
      input.support_states.size() > impl_->config.maximum_factor_support_states ||
      std::find(input.support_states.begin(), input.support_states.end(), input.terminal_state) ==
          input.support_states.end()) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "factor identity/support/revision/byte accounting is invalid"));
  }
  const Impl::StateRecord* terminal = impl_->state(input.terminal_state);
  if (terminal == nullptr || terminal->staged.exact_time != input.terminal_time) {
    return ResultBool::failure(error(SparseSubmapErrorCode::UnknownState,
                                     "factor terminal state/time is not an exact staged state"));
  }
  for (const core::StateId support : input.support_states) {
    if (impl_->state(support) == nullptr) {
      return ResultBool::failure(error(SparseSubmapErrorCode::UnknownState,
                                       "factor support contains an unstaged local state"));
    }
  }
  if (const auto late = impl_->rejectLateRecord(input.terminal_time)) {
    return ResultBool::failure(*late);
  }
  if (const auto lineage_error = impl_->validateLineageBounds(input.lineage)) {
    return ResultBool::failure(*lineage_error);
  }
  if (input.kind != FinalizedFactorKind::IncomingMarginalPrior) {
    if (const auto ownership_error = impl_->reservePrimarySlices(input.lineage)) {
      return ResultBool::failure(*ownership_error);
    }
  }
  if (const auto duplicate = impl_->reserveSeenId(input.factor.value(), "factor")) {
    return ResultBool::failure(*duplicate);
  }
  Impl::Builder* target = impl_->owner(input.terminal_time);
  if (target == nullptr || (target->lifecycle == SparseSubmapLifecycle::Finalizing &&
                            target->condensation)) {
    return ResultBool::failure(error(SparseSubmapErrorCode::RecordOutsideOpenInterval,
                                     "factor has no mutable half-open core partition"));
  }
  target->factors.push_back(input);
  if (const auto capacity_error = impl_->validateBuilderCapacity(*target)) {
    target->factors.pop_back();
    return ResultBool::failure(*capacity_error);
  }
  if (input.kind != FinalizedFactorKind::IncomingMarginalPrior) {
    impl_->commitPrimarySlices(input.lineage);
  }
  impl_->commitSeenId(input.factor.value(), "factor");
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::stageLidarKeyframe(
    FinalizedLidarKeyframeInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->active) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NotStarted,
                                     "stage at least one local state before a LiDAR keyframe"));
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  const Impl::StateRecord* record = impl_->state(input.state);
  if (!input.sweep.valid() || record == nullptr || record->staged.exact_time != input.terminal_time ||
      !input.final_revision.valid() || !finitePose(input.T_imu_lidar) ||
      input.retained_bytes == 0U || input.registration_samples.empty()) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "LiDAR keyframe identity/state/extrinsic/samples is invalid"));
  }
  for (const auto& sample : input.registration_samples) {
    if (!sample.point_lidar.allFinite() || !sample.normal_lidar.allFinite() ||
        sample.normal_lidar.norm() <= 1.0e-12 || !std::isfinite(sample.weight) ||
        sample.weight <= 0.0) {
      return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                       "LiDAR proxy sample is non-finite or has invalid normal/weight"));
    }
  }
  if (const auto late = impl_->rejectLateRecord(input.terminal_time)) {
    return ResultBool::failure(*late);
  }
  if (const auto lineage_error = impl_->validateLineageBounds(input.lineage)) {
    return ResultBool::failure(*lineage_error);
  }
  if (input.place_payload) {
    if (const auto payload_error = impl_->validatePayload(&*input.place_payload)) {
      return ResultBool::failure(*payload_error);
    }
  }
  if (const auto duplicate = impl_->reserveSeenId(input.sweep.value(), "sweep")) {
    return ResultBool::failure(*duplicate);
  }
  Impl::Builder* target = impl_->owner(input.terminal_time);
  if (target == nullptr || (target->lifecycle == SparseSubmapLifecycle::Finalizing &&
                            target->condensation)) {
    return ResultBool::failure(error(SparseSubmapErrorCode::RecordOutsideOpenInterval,
                                     "LiDAR keyframe has no mutable half-open core owner"));
  }
  target->lidar_keyframes.push_back(std::move(input));
  if (const auto capacity_error = impl_->validateBuilderCapacity(*target)) {
    target->lidar_keyframes.pop_back();
    return ResultBool::failure(*capacity_error);
  }
  impl_->commitSeenId(target->lidar_keyframes.back().sweep.value(), "sweep");
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::stageVisualKeyframe(
    FinalizedVisualKeyframeInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->active) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NotStarted,
                                     "stage at least one local state before a visual keyframe"));
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  const Impl::StateRecord* record = impl_->state(input.state);
  if (!input.frame.valid() || !input.camera.valid() || record == nullptr ||
      record->staged.exact_time != input.terminal_time || !input.final_revision.valid() ||
      !finitePose(input.T_imu_camera) || input.retained_bytes == 0U) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidRecord,
                                     "visual keyframe identity/state/extrinsic is invalid"));
  }
  if (const auto late = impl_->rejectLateRecord(input.terminal_time)) {
    return ResultBool::failure(*late);
  }
  if (const auto lineage_error = impl_->validateLineageBounds(input.lineage)) {
    return ResultBool::failure(*lineage_error);
  }
  if (const auto payload_error = impl_->validatePayload(&input.place_payload)) {
    return ResultBool::failure(*payload_error);
  }
  if (const auto duplicate = impl_->reserveSeenId(input.frame.value(), "camera_frame")) {
    return ResultBool::failure(*duplicate);
  }
  Impl::Builder* target = impl_->owner(input.terminal_time);
  if (target == nullptr || (target->lifecycle == SparseSubmapLifecycle::Finalizing &&
                            target->condensation)) {
    return ResultBool::failure(error(SparseSubmapErrorCode::RecordOutsideOpenInterval,
                                     "visual keyframe has no mutable half-open core owner"));
  }
  target->visual_keyframes.push_back(std::move(input));
  if (const auto capacity_error = impl_->validateBuilderCapacity(*target)) {
    target->visual_keyframes.pop_back();
    return ResultBool::failure(*capacity_error);
  }
  impl_->commitSeenId(target->visual_keyframes.back().frame.value(), "camera_frame");
  return ResultBool::success(true);
}

core::Result<SparseSubmapSplitReport, SparseSubmapError>
SparseSubmapCoordinator::considerSplit(SparseSubmapSplitRequest request) {
  using Result = core::Result<SparseSubmapSplitReport, SparseSubmapError>;
  if (const auto configuration_error = impl_->ready()) {
    return Result::failure(*configuration_error);
  }
  if (!impl_->active) {
    return Result::failure(error(SparseSubmapErrorCode::NotStarted,
                                 "cannot split before the first local state"));
  }
  if (impl_->finalizing) {
    return Result::failure(error(SparseSubmapErrorCode::SplitAlreadyFinalizing,
                                 "one provisional successor is already buffering the lag tail"));
  }
  const Impl::StateRecord* boundary = impl_->state(request.boundary_state);
  if (boundary == nullptr || boundary->staged.exact_time <= impl_->active->start_time ||
      std::find(impl_->active->core_states.begin(), impl_->active->core_states.end(),
                request.boundary_state) == impl_->active->core_states.end()) {
    return Result::failure(error(SparseSubmapErrorCode::InvalidSplitBoundary,
                                 "split boundary must be a later state in the active core"));
  }
  const core::FusionTime boundary_time = boundary->staged.exact_time;
  SparseSubmapSplitReport report;
  report.boundary_time = boundary_time;
  report.duration = boundary_time - impl_->active->start_time;

  std::vector<const Impl::StateRecord*> ordered_states;
  for (const core::StateId id : impl_->active->core_states) {
    const Impl::StateRecord* record = impl_->state(id);
    if (record != nullptr && record->staged.exact_time <= boundary_time) {
      ordered_states.push_back(record);
    }
  }
  std::sort(ordered_states.begin(), ordered_states.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->staged.exact_time < rhs->staged.exact_time;
  });
  for (std::size_t index = 1U; index < ordered_states.size(); ++index) {
    const core::Pose3d relative =
        ordered_states[index - 1U]->staged.provisional_estimate.T_odom_imu.inverse() *
        ordered_states[index]->staged.provisional_estimate.T_odom_imu;
    report.travel_m += relative.translation().norm();
    report.rotation_rad += relative.so3().log().norm();
  }
  const auto before = [boundary_time](const auto& record) {
    return record.terminal_time < boundary_time;
  };
  report.keyframes =
      static_cast<std::size_t>(std::count_if(impl_->active->lidar_keyframes.begin(),
                                             impl_->active->lidar_keyframes.end(), before)) +
      static_cast<std::size_t>(std::count_if(impl_->active->visual_keyframes.begin(),
                                             impl_->active->visual_keyframes.end(), before));

  Impl::Builder provisional_old = *impl_->active;
  provisional_old.core_states.erase(
      std::remove_if(provisional_old.core_states.begin(), provisional_old.core_states.end(),
                     [&](core::StateId id) {
                       return impl_->state(id)->staged.exact_time >= boundary_time;
                     }),
      provisional_old.core_states.end());
  std::erase_if(provisional_old.factors,
                [&](const auto& factor) { return factor.terminal_time >= boundary_time; });
  std::erase_if(provisional_old.lidar_keyframes,
                [&](const auto& keyframe) { return keyframe.terminal_time >= boundary_time; });
  std::erase_if(provisional_old.visual_keyframes,
                [&](const auto& keyframe) { return keyframe.terminal_time >= boundary_time; });
  const auto bytes = impl_->builderBytes(provisional_old);
  if (!bytes) {
    return Result::failure(bytes.error());
  }
  report.payload_bytes = bytes.value();

  if (request.local_reset) {
    report.reasons.push_back(SparseSubmapSplitReason::LocalReset);
  }
  if (report.travel_m >= impl_->config.split.maximum_travel_m) {
    report.reasons.push_back(SparseSubmapSplitReason::Travel);
  }
  if (report.rotation_rad >= impl_->config.split.maximum_rotation_rad) {
    report.reasons.push_back(SparseSubmapSplitReason::Rotation);
  }
  if (report.duration >= impl_->config.split.maximum_duration) {
    report.reasons.push_back(SparseSubmapSplitReason::Duration);
  }
  if (report.keyframes >= impl_->config.split.maximum_keyframes) {
    report.reasons.push_back(SparseSubmapSplitReason::KeyframeCount);
  }
  if (report.payload_bytes >= impl_->config.split.maximum_payload_bytes) {
    report.reasons.push_back(SparseSubmapSplitReason::PayloadBytes);
  }
  if (report.reasons.empty()) {
    return Result::success(std::move(report));
  }
  if (impl_->active->id.value() >= core::SubmapId::kInvalidValue - 1U) {
    return Result::failure(error(SparseSubmapErrorCode::SubmapIdOverflow,
                                 "submap ID sequence reached its invalid sentinel"));
  }

  Impl::Builder old = *impl_->active;
  old.lifecycle = SparseSubmapLifecycle::Finalizing;
  old.end_state = request.boundary_state;
  old.end_time = boundary_time;
  Impl::Builder successor{core::SubmapId(old.id.value() + 1U), SparseSubmapLifecycle::Active,
                          request.boundary_state, boundary_time};

  const auto move_partition = [boundary_time](auto& source, auto& destination, auto time_of) {
    auto first = std::stable_partition(source.begin(), source.end(), [&](const auto& record) {
      return time_of(record) < boundary_time;
    });
    destination.insert(destination.end(), std::make_move_iterator(first),
                       std::make_move_iterator(source.end()));
    source.erase(first, source.end());
  };
  move_partition(old.core_states, successor.core_states, [&](core::StateId id) {
    return impl_->state(id)->staged.exact_time;
  });
  move_partition(old.factors, successor.factors,
                 [](const auto& factor) { return factor.terminal_time; });
  move_partition(old.lidar_keyframes, successor.lidar_keyframes,
                 [](const auto& keyframe) { return keyframe.terminal_time; });
  move_partition(old.visual_keyframes, successor.visual_keyframes,
                 [](const auto& keyframe) { return keyframe.terminal_time; });
  if (successor.core_states.empty() || successor.core_states.front() != request.boundary_state) {
    return Result::failure(error(SparseSubmapErrorCode::InvalidSplitBoundary,
                                 "successor buffer did not retain the half-open boundary state"));
  }
  if (const auto old_capacity = impl_->validateBuilderCapacity(old)) {
    return Result::failure(*old_capacity);
  }
  if (const auto new_capacity = impl_->validateBuilderCapacity(successor)) {
    return Result::failure(*new_capacity);
  }
  impl_->finalizing = std::move(old);
  impl_->active = std::move(successor);
  impl_->next_submap_id = impl_->active->id;
  report.split_requested = true;
  const auto sealed = impl_->trySeal();
  if (!sealed) {
    return Result::failure(sealed.error());
  }
  return Result::success(std::move(report));
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::abortProvisionalSplit() {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->finalizing || !impl_->active) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NoFinalizingSubmap,
                                     "there is no provisional split to abort"));
  }
  Impl::Builder merged = *impl_->finalizing;
  merged.lifecycle = SparseSubmapLifecycle::Active;
  merged.end_state.reset();
  merged.end_time.reset();
  merged.condensation.reset();
  merged.core_states.insert(merged.core_states.end(), impl_->active->core_states.begin(),
                            impl_->active->core_states.end());
  merged.factors.insert(merged.factors.end(), impl_->active->factors.begin(),
                        impl_->active->factors.end());
  merged.lidar_keyframes.insert(
      merged.lidar_keyframes.end(),
      impl_->active->lidar_keyframes.begin(), impl_->active->lidar_keyframes.end());
  merged.visual_keyframes.insert(
      merged.visual_keyframes.end(),
      impl_->active->visual_keyframes.begin(), impl_->active->visual_keyframes.end());
  if (const auto capacity_error = impl_->validateBuilderCapacity(merged)) {
    return ResultBool::failure(*capacity_error);
  }
  impl_->next_submap_id = impl_->finalizing->id;
  impl_->active = std::move(merged);
  impl_->finalizing.reset();
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::submitBoundaryCondensation(
    FinalizedBoundaryCondensationInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->finalizing || !impl_->finalizing->end_state) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NoFinalizingSubmap,
                                     "boundary condensation requires a finalizing submap"));
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  if (impl_->finalizing->condensation) {
    return ResultBool::failure(error(SparseSubmapErrorCode::DuplicateIdentity,
                                     "finalizing submap already owns a boundary condensation"));
  }
  if (input.from_boundary_state != impl_->finalizing->start_state ||
      input.to_boundary_state != *impl_->finalizing->end_state ||
      !input.final_revision.valid()) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                     "condensation endpoint identities do not match the interval"));
  }
  const double covariance_scale =
      std::max(1.0, input.joint_pose_covariance.cwiseAbs().maxCoeff());
  if (input.covariance_order !=
          BoundaryJointCovarianceOrder::FromThenToRightTranslationFirst ||
      input.supported_relative_rank == 0U || input.supported_relative_rank > 6U ||
      !input.joint_pose_covariance.allFinite() ||
      (input.joint_pose_covariance - input.joint_pose_covariance.transpose())
              .cwiseAbs()
              .maxCoeff() >
          impl_->config.covariance_symmetry_relative_tolerance * covariance_scale ||
      !input.relative_information_basis.allFinite() ||
      (input.relative_information_basis.transpose() * input.relative_information_basis -
       core::Matrix6d::Identity())
              .cwiseAbs()
              .maxCoeff() > 1.0e-8) {
    return ResultBool::failure(error(
        SparseSubmapErrorCode::InvalidCondensation,
        "boundary condensation covariance/order/rank/support basis is invalid"));
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> joint_solver(
      0.5 * (input.joint_pose_covariance + input.joint_pose_covariance.transpose()),
      Eigen::EigenvaluesOnly);
  if (joint_solver.info() != Eigen::Success || !joint_solver.eigenvalues().allFinite() ||
      joint_solver.eigenvalues().minCoeff() <
          -impl_->config.minimum_supported_covariance_eigenvalue * covariance_scale) {
    return ResultBool::failure(error(SparseSubmapErrorCode::InvalidCondensation,
                                     "boundary joint covariance is not positive semidefinite"));
  }
  std::vector<core::FactorId> expected;
  for (const auto& factor : impl_->finalizing->factors) {
    if (factor.kind != FinalizedFactorKind::IncomingMarginalPrior) {
      expected.push_back(factor.factor);
    }
  }
  auto provided = input.eliminated_factor_ids;
  const auto order = [](core::FactorId lhs, core::FactorId rhs) { return lhs < rhs; };
  std::sort(expected.begin(), expected.end(), order);
  std::sort(provided.begin(), provided.end(), order);
  if (std::adjacent_find(provided.begin(), provided.end()) != provided.end() ||
      expected != provided) {
    return ResultBool::failure(error(
        SparseSubmapErrorCode::InvalidCondensation,
        "eliminated factor IDs must equal the disjoint core partition and exclude its incoming prior"));
  }
  const auto start_frame = impl_->boundaryFrame(input.from_boundary_state);
  const auto end_frame = impl_->boundaryFrame(input.to_boundary_state);
  if (start_frame && end_frame) {
    const auto uncertainty =
        impl_->relativeUncertainty(input, start_frame.value().inverse() * end_frame.value());
    if (!uncertainty) {
      return ResultBool::failure(uncertainty.error());
    }
  }
  impl_->finalizing->condensation = std::move(input);
  const auto sealed = impl_->trySeal();
  if (!sealed) {
    return ResultBool::failure(sealed.error());
  }
  return ResultBool::success(true);
}

core::Result<bool, SparseSubmapError> SparseSubmapCoordinator::advanceFinality(
    LocalFinalityBarrierInput input) {
  if (const auto configuration_error = impl_->ready()) {
    return ResultBool::failure(*configuration_error);
  }
  if (!impl_->active) {
    return ResultBool::failure(error(SparseSubmapErrorCode::NotStarted,
                                     "cannot advance finality before the first state"));
  }
  if (const auto header_error = impl_->validateSession(input.header)) {
    return ResultBool::failure(*header_error);
  }
  if (!input.final_revision.valid() ||
      (impl_->last_state_time && input.finalized_through > *impl_->last_state_time) ||
      (impl_->finalized_through && input.finalized_through < *impl_->finalized_through) ||
      (impl_->finality_revision && input.final_revision < *impl_->finality_revision)) {
    return ResultBool::failure(error(
        SparseSubmapErrorCode::FinalityRegression,
        "local finality time/revision must be monotonic and cannot pass the latest staged state"));
  }
  impl_->finalized_through = input.finalized_through;
  impl_->finality_revision = input.final_revision;
  const auto sealed = impl_->trySeal();
  if (!sealed) {
    return ResultBool::failure(sealed.error());
  }
  return ResultBool::success(true);
}

std::vector<SparseSubmapSeal> SparseSubmapCoordinator::takeSealed() {
  std::vector<SparseSubmapSeal> output;
  output.swap(impl_->pending_seals);
  return output;
}

SparseSubmapStatus SparseSubmapCoordinator::status() const noexcept {
  SparseSubmapStatus output;
  output.started = impl_->active.has_value();
  if (impl_->active) {
    output.active_submap = impl_->active->id;
    output.active_lifecycle = impl_->active->lifecycle;
  }
  if (impl_->finalizing) {
    output.finalizing_submap = impl_->finalizing->id;
  }
  output.staged_states = impl_->states.size();
  output.pending_seals = impl_->pending_seals.size();
  output.sealed_identities = impl_->seal_identities.size();
  output.finalized_through = impl_->finalized_through;
  return output;
}

std::optional<core::SparseSubmapSealIdentity> SparseSubmapCoordinator::sealIdentity(
    const core::SubmapRef& submap) const {
  if (core::validateSubmapRef(submap) != core::SubmapRefValidationError::None) {
    return std::nullopt;
  }
  const auto found = impl_->seal_identities.find(core::sparseSubmapIdentityKey(submap));
  return found == impl_->seal_identities.end() || found->second.ref != submap
             ? std::nullopt
             : std::optional<core::SparseSubmapSealIdentity>(found->second);
}

}  // namespace meridian::global
