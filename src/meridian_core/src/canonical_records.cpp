#include "meridian/core/canonical_records.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace meridian::core {
namespace {

[[nodiscard]] CanonicalRecordError recordError(
    CanonicalRecordErrorCode code, CanonicalRecordKind record,
    CanonicalEncodingError encoding_error = CanonicalEncodingError::None) noexcept {
  return CanonicalRecordError{code, record, encoding_error};
}

class RecordWriter {
public:
  using CreateResult = Result<RecordWriter, CanonicalRecordError>;

  [[nodiscard]] static CreateResult create(const CanonicalRecordDomain& domain,
                                           CanonicalRecordLimits limits) {
    if (limits.maximum_output_bytes == 0U || limits.maximum_collection_entries == 0U) {
      return CreateResult::failure(
          recordError(CanonicalRecordErrorCode::InvalidLimits, domain.kind));
    }
    auto encoder =
        CanonicalEncoder::create(domain.tag, domain.schema_version, limits.maximum_output_bytes);
    if (!encoder) {
      const auto code = encoder.error() == CanonicalEncodingError::InvalidMaximumBytes
                            ? CanonicalRecordErrorCode::InvalidLimits
                            : CanonicalRecordErrorCode::EncodingFailure;
      return CreateResult::failure(recordError(code, domain.kind, encoder.error()));
    }
    return CreateResult::success(
        RecordWriter(domain.kind, limits.maximum_collection_entries, std::move(encoder).value()));
  }

  RecordWriter(const RecordWriter&) = delete;
  RecordWriter& operator=(const RecordWriter&) = delete;
  RecordWriter(RecordWriter&&) noexcept = default;
  RecordWriter& operator=(RecordWriter&&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }

  void u8(std::uint8_t value) {
    if (ok()) {
      apply(encoder_.writeU8(value));
    }
  }
  void u32(std::uint32_t value) {
    if (ok()) {
      apply(encoder_.writeU32(value));
    }
  }
  void u64(std::uint64_t value) {
    if (ok()) {
      apply(encoder_.writeU64(value));
    }
  }
  void i64(std::int64_t value) {
    if (ok()) {
      apply(encoder_.writeI64(value));
    }
  }
  void optional(bool present) {
    if (ok()) {
      apply(encoder_.writeOptionalMarker(present));
    }
  }
  void floating(double value) {
    if (ok()) {
      apply(encoder_.writeDouble(value));
    }
  }
  void hash(const ContentHash& value) {
    if (ok()) {
      apply(encoder_.writeHash(value));
    }
  }
  void pose(const Pose3d& value) {
    if (ok()) {
      apply(encoder_.writePose3(value));
    }
  }

  template <typename Derived>
  void vector(const Eigen::MatrixBase<Derived>& value) {
    if (ok()) {
      apply(encoder_.writeEigenVector(value));
    }
  }

  template <typename Id>
  void id(Id value) {
    u64(value.value());
  }

  void count(std::size_t value) {
    if (!ok()) {
      return;
    }
    if (value > remaining_collection_entries_) {
      error_ = recordError(CanonicalRecordErrorCode::CollectionLimitExceeded, kind_);
      return;
    }
    remaining_collection_entries_ -= value;
    u64(static_cast<std::uint64_t>(value));
  }

  void requireHash(const ContentHash& value) {
    if (!ok()) {
      return;
    }
    if (!contentHashPresent(value)) {
      error_ = recordError(CanonicalRecordErrorCode::MissingDependencyChecksum, kind_);
      return;
    }
    hash(value);
  }

  void unsupported() {
    if (ok()) {
      error_ = recordError(CanonicalRecordErrorCode::UnsupportedValue, kind_);
    }
  }

  [[nodiscard]] CanonicalChecksumResult finish() {
    if (error_) {
      return CanonicalChecksumResult::failure(*error_);
    }
    auto result = encoder_.finish();
    if (!result) {
      return CanonicalChecksumResult::failure(
          recordError(CanonicalRecordErrorCode::EncodingFailure, kind_, result.error()));
    }
    return CanonicalChecksumResult::success(result.value().digest());
  }

private:
  RecordWriter(CanonicalRecordKind kind, std::uint64_t maximum_collection_entries,
               CanonicalEncoder encoder)
      : kind_(kind),
        remaining_collection_entries_(maximum_collection_entries),
        encoder_(std::move(encoder)) {}

  void apply(CanonicalEncodingError error) {
    if (!ok() || error == CanonicalEncodingError::None) {
      return;
    }
    error_ = recordError(CanonicalRecordErrorCode::EncodingFailure, kind_, error);
  }

  CanonicalRecordKind kind_;
  std::uint64_t remaining_collection_entries_{};
  CanonicalEncoder encoder_;
  std::optional<CanonicalRecordError> error_;
};

template <typename Function>
[[nodiscard]] CanonicalChecksumResult encode(const CanonicalRecordDomain& domain,
                                             CanonicalRecordLimits limits, Function&& function) {
  auto writer_result = RecordWriter::create(domain, limits);
  if (!writer_result) {
    return CanonicalChecksumResult::failure(writer_result.error());
  }
  RecordWriter writer = std::move(writer_result).value();
  function(writer);
  return writer.finish();
}

void writeRecordHeader(RecordWriter& writer, const RecordHeader& header) {
  writer.u32(header.schema_version);
  writer.id(header.trace);
  writer.id(header.producer);
  writer.id(header.session);
  writer.i64(header.created_at.nanoseconds);
  writer.id(header.config);
  writer.optional(header.direct_calibration.has_value());
  if (header.direct_calibration) {
    writer.id(*header.direct_calibration);
  }
}

void writeLocalContentHeader(RecordWriter& writer, const RecordHeader& header) {
  writer.u32(header.schema_version);
  writer.id(header.producer);
  writer.id(header.session);
  writer.id(header.config);
  writer.optional(header.direct_calibration.has_value());
  if (header.direct_calibration) {
    writer.id(*header.direct_calibration);
  }
}

void writeSubmapRefIdentity(RecordWriter& writer, const SubmapRef& ref) {
  writer.id(ref.session);
  writer.id(ref.odom_epoch);
  writer.id(ref.id);
  writer.id(ref.calibration);
  writer.id(ref.content_revision);
}

void writeSubmapRef(RecordWriter& writer, const SubmapRef& ref) {
  writeSubmapRefIdentity(writer, ref);
  writer.requireHash(ref.local_content_checksum);
}

void writeBoundary(RecordWriter& writer, const BoundaryNavigationLinearization& boundary) {
  writer.id(boundary.state);
  writer.i64(boundary.exact_time.nanoseconds);
  writer.id(boundary.final_revision);
  writer.pose(boundary.T_odom_imu);
  writer.vector(boundary.velocity_odom);
  writer.vector(boundary.gyro_bias);
  writer.vector(boundary.accel_bias);
}

void writeSubmapFrame(RecordWriter& writer, const SubmapFrameDefinition& frame) {
  writer.id(frame.boundary_state);
  writer.i64(frame.boundary_time.nanoseconds);
  writer.vector(frame.gravity_up_odom);
  writer.floating(frame.boundary_yaw_odom);
}

void writeLocalVariableKind(RecordWriter& writer, LocalVariableKind kind) {
  switch (kind) {
    case LocalVariableKind::Pose:
      writer.u8(1U);
      return;
    case LocalVariableKind::NavigationVelocity:
      writer.u8(2U);
      return;
    case LocalVariableKind::GyroBias:
      writer.u8(3U);
      return;
    case LocalVariableKind::AccelBias:
      writer.u8(4U);
      return;
    case LocalVariableKind::LandmarkLogInverseRange:
      writer.u8(5U);
      return;
  }
  writer.unsupported();
}

void writePoseTangent(RecordWriter& writer, PoseTangentConvention tangent) {
  switch (tangent) {
    case PoseTangentConvention::RightTranslationFirst:
      writer.u8(1U);
      return;
  }
  writer.unsupported();
}

void writeSliceKind(RecordWriter& writer, SliceKind kind) {
  switch (kind) {
    case SliceKind::Whole:
      writer.u8(1U);
      return;
    case SliceKind::IndexRange:
      writer.u8(2U);
      return;
  }
  writer.unsupported();
}

void writeObservationRole(RecordWriter& writer, ObservationRole role) {
  switch (role) {
    case ObservationRole::PrimaryResidual:
      writer.u8(1U);
      return;
    case ObservationRole::ConditioningOnly:
      writer.u8(2U);
      return;
    case ObservationRole::RetrievalSeedOnly:
      writer.u8(3U);
      return;
    case ObservationRole::DerivedSummary:
      writer.u8(4U);
      return;
  }
  writer.unsupported();
}

void writeCorrelationTreatment(RecordWriter& writer, CorrelationTreatment treatment) {
  switch (treatment) {
    case CorrelationTreatment::JointCompositeWhitening:
      writer.u8(1U);
      return;
    case CorrelationTreatment::CovarianceInflationAndInformationCap:
      writer.u8(2U);
      return;
    case CorrelationTreatment::NotIndependent:
      writer.u8(3U);
      return;
  }
  writer.unsupported();
}

void writePayloadKind(RecordWriter& writer, SparsePayloadKind kind) {
  switch (kind) {
    case SparsePayloadKind::InternalTrajectory:
      writer.u8(1U);
      return;
    case SparsePayloadKind::KeyframeIndex:
      writer.u8(2U);
      return;
    case SparsePayloadKind::RegistrationProxy:
      writer.u8(3U);
      return;
    case SparsePayloadKind::DenseInputIndex:
      writer.u8(4U);
      return;
    case SparsePayloadKind::VisualPlaceCatalog:
      writer.u8(5U);
      return;
    case SparsePayloadKind::LidarPlaceCatalog:
      writer.u8(6U);
      return;
  }
  writer.unsupported();
}

void writeObservationLineage(RecordWriter& writer, const ObservationLineage& lineage) {
  writer.id(lineage.id);
  writer.count(lineage.usage.size());
  if (!writer.ok()) {
    return;
  }
  for (const ObservationUsage& usage : lineage.usage) {
    if (usage.slice.root.valueless_by_exception()) {
      writer.unsupported();
      return;
    }
    if (std::holds_alternative<MeasurementId>(usage.slice.root)) {
      writer.u8(1U);
      writer.id(std::get<MeasurementId>(usage.slice.root));
    } else if (std::holds_alternative<GnssObservationId>(usage.slice.root)) {
      writer.u8(2U);
      writer.id(std::get<GnssObservationId>(usage.slice.root));
    } else {
      writer.unsupported();
      return;
    }
    writeSliceKind(writer, usage.slice.kind);
    writer.u64(usage.slice.begin);
    writer.u64(usage.slice.end);
    writer.requireHash(usage.slice.source_checksum);
    writer.id(usage.slice.calibration);
    writeObservationRole(writer, usage.role);
    writer.id(usage.consumer);
    writer.optional(usage.factor_group.has_value());
    if (usage.factor_group) {
      writer.id(*usage.factor_group);
    }
    writer.optional(usage.correlation_group.has_value());
    if (usage.correlation_group) {
      writer.id(*usage.correlation_group);
    }
    if (!writer.ok()) {
      return;
    }
  }

  writer.count(lineage.correlations.size());
  if (!writer.ok()) {
    return;
  }
  for (const CorrelationDeclaration& declaration : lineage.correlations) {
    writer.id(declaration.group);
    writer.id(declaration.policy);
    writeCorrelationTreatment(writer, declaration.treatment);
    writer.floating(declaration.covariance_inflation);
    writer.optional(declaration.total_information_cap.has_value());
    if (declaration.total_information_cap) {
      writer.floating(*declaration.total_information_cap);
    }
    if (!writer.ok()) {
      return;
    }
  }
}

void writeFrozenFactor(RecordWriter& writer, const FrozenSquareRootFactor& factor) {
  writePoseTangent(writer, factor.pose_tangent);
  writer.u32(factor.rows);
  writer.u32(factor.columns);
  writer.count(factor.layout.size());
  if (!writer.ok()) {
    return;
  }
  for (const SquareRootColumnBlock& block : factor.layout) {
    writeLocalVariableKind(writer, block.variable.kind);
    writer.optional(block.variable.state.has_value());
    if (block.variable.state) {
      writer.id(*block.variable.state);
    }
    writer.optional(block.variable.landmark.has_value());
    if (block.variable.landmark) {
      writer.id(*block.variable.landmark);
    }
    writer.u32(block.column_offset);
    writer.u32(block.dimension);
    if (!writer.ok()) {
      return;
    }
  }
  writer.count(factor.row_major_A.size());
  if (!writer.ok()) {
    return;
  }
  for (double value : factor.row_major_A) {
    writer.floating(value);
    if (!writer.ok()) {
      return;
    }
  }
  writer.count(factor.rhs.size());
  if (!writer.ok()) {
    return;
  }
  for (double value : factor.rhs) {
    writer.floating(value);
    if (!writer.ok()) {
      return;
    }
  }
  writer.floating(factor.constant_squared_error);
  writer.u32(factor.numerical_rank);
  writer.floating(factor.absolute_rank_tolerance);
  writer.floating(factor.relative_rank_tolerance);
  writer.u64(factor.cost_statistics.source_residual_dof);
  writer.u64(factor.cost_statistics.eliminated_numerical_rank);
  writer.u64(factor.cost_statistics.effective_dof);
  writer.id(factor.cost_statistics.calibration_revision);
  writer.optional(factor.cost_statistics.calibrated_total_cost_cutoff.has_value());
  if (factor.cost_statistics.calibrated_total_cost_cutoff) {
    writer.floating(*factor.cost_statistics.calibrated_total_cost_cutoff);
  }
}

void writePayloadCatalog(RecordWriter& writer, const SparsePayloadCatalog& catalog) {
  writer.count(catalog.entries.size());
  if (!writer.ok()) {
    return;
  }
  for (const SparsePayloadIndexEntry& entry : catalog.entries) {
    writePayloadKind(writer, entry.kind);
    writer.id(entry.root.store);
    writer.id(entry.root.id);
    writer.requireHash(entry.root.checksum);
    writer.id(entry.root.layout);
    writer.u64(entry.root.bytes);
    if (!writer.ok()) {
      return;
    }
  }
}

void writeLocalContent(RecordWriter& writer, const SparseSubmapSeal& seal) {
  writeLocalContentHeader(writer, seal.header);
  writeSubmapRefIdentity(writer, seal.ref);
  writer.id(seal.final_local_revision);
  writer.i64(seal.support_time.start.nanoseconds);
  writer.i64(seal.support_time.end.nanoseconds);
  writeSubmapFrame(writer, seal.frame);
  writer.pose(seal.T_odom_submap);
  writeBoundary(writer, seal.boundary_navigation);
  writer.requireHash(seal.payloads.checksum);
  writer.requireHash(seal.lineage.checksum);
  writer.requireHash(seal.quality_checksum);
}

void writeSealEnvelope(RecordWriter& writer, const SparseSubmapSeal& seal) {
  writeRecordHeader(writer, seal.header);
  writeSubmapRef(writer, seal.ref);
  writer.optional(seal.previous.has_value());
  if (seal.previous) {
    writeSubmapRef(writer, *seal.previous);
  }
  writer.id(seal.final_local_revision);
  writer.i64(seal.support_time.start.nanoseconds);
  writer.i64(seal.support_time.end.nanoseconds);
  writeSubmapFrame(writer, seal.frame);
  writer.pose(seal.T_odom_submap);
  writeBoundary(writer, seal.boundary_navigation);
  writer.requireHash(seal.payloads.checksum);
  writer.optional(seal.from_previous.has_value());
  if (seal.from_previous) {
    writer.requireHash(seal.from_previous->checksum);
  }
  writer.requireHash(seal.lineage.checksum);
  writer.requireHash(seal.quality_checksum);
}

}  // namespace

CanonicalChecksumResult recomputeObservationLineageChecksum(const ObservationLineage& lineage,
                                                            CanonicalRecordLimits limits) {
  return encode(kObservationLineageChecksumDomain, limits,
                [&lineage](RecordWriter& writer) { writeObservationLineage(writer, lineage); });
}

CanonicalChecksumResult recomputeFrozenSquareRootFactorChecksum(
    const FrozenSquareRootFactor& factor, CanonicalRecordLimits limits) {
  return encode(kFrozenSquareRootFactorChecksumDomain, limits,
                [&factor](RecordWriter& writer) { writeFrozenFactor(writer, factor); });
}

CanonicalChecksumResult recomputeCondensedBoundaryTransitionChecksum(
    const CondensedBoundaryTransition& transition, CanonicalRecordLimits limits) {
  return encode(kCondensedBoundaryTransitionChecksumDomain, limits,
                [&transition](RecordWriter& writer) {
                  writeRecordHeader(writer, transition.header);
                  writer.id(transition.odom_epoch);
                  writeBoundary(writer, transition.from);
                  writeBoundary(writer, transition.to);
                  writer.requireHash(transition.boundary_factor.checksum);
                  writer.count(transition.source_factors.size());
                  if (!writer.ok()) {
                    return;
                  }
                  for (const LocalFactorRef& factor : transition.source_factors) {
                    writer.id(factor.odom_epoch);
                    writer.id(factor.factor);
                    if (!writer.ok()) {
                      return;
                    }
                  }
                  writer.requireHash(transition.lineage.checksum);
                  writer.id(transition.final_revision);
                  writer.requireHash(transition.input_partition_checksum);
                });
}

CanonicalChecksumResult recomputeSparsePayloadCatalogChecksum(const SparsePayloadCatalog& catalog,
                                                              CanonicalRecordLimits limits) {
  return encode(kSparsePayloadCatalogChecksumDomain, limits,
                [&catalog](RecordWriter& writer) { writePayloadCatalog(writer, catalog); });
}

CanonicalChecksumResult recomputeSubmapLocalContentChecksum(const SparseSubmapSeal& seal,
                                                            CanonicalRecordLimits limits) {
  return encode(kSubmapLocalContentChecksumDomain, limits,
                [&seal](RecordWriter& writer) { writeLocalContent(writer, seal); });
}

CanonicalChecksumResult recomputeSealedBoundaryTransitionChecksum(
    const SealedBoundaryTransition& transition, CanonicalRecordLimits limits) {
  return encode(kSealedBoundaryTransitionChecksumDomain, limits,
                [&transition](RecordWriter& writer) {
                  writeSubmapRef(writer, transition.from);
                  writeSubmapRef(writer, transition.to);
                  writer.requireHash(transition.local_transition.checksum);
                });
}

CanonicalChecksumResult recomputeSparseSubmapSealChecksum(const SparseSubmapSeal& seal,
                                                          CanonicalRecordLimits limits) {
  return encode(kSparseSubmapSealChecksumDomain, limits,
                [&seal](RecordWriter& writer) { writeSealEnvelope(writer, seal); });
}

}  // namespace meridian::core
