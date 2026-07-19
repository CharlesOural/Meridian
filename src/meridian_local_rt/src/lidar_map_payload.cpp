#include "meridian/local/lidar_map_payload.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <variant>

#include "meridian/core/canonical_bytes.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] LidarMapPayloadError payloadError(LidarMapPayloadErrorCode code, std::string detail) {
  return LidarMapPayloadError{code, std::move(detail)};
}

[[nodiscard]] bool finitePoint(const core::LidarPoint& point) noexcept {
  return std::isfinite(static_cast<double>(point.x)) &&
         std::isfinite(static_cast<double>(point.y)) &&
         std::isfinite(static_cast<double>(point.z)) &&
         std::isfinite(static_cast<double>(point.intensity));
}

[[nodiscard]] bool finitePose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite();
}

[[nodiscard]] bool validLayout(const core::LidarSweep& sweep) noexcept {
  if (!sweep.points || sweep.points->empty() || sweep.layout.width == 0U ||
      sweep.layout.height == 0U) {
    return false;
  }
  const std::size_t width = static_cast<std::size_t>(sweep.layout.width);
  const std::size_t height = static_cast<std::size_t>(sweep.layout.height);
  if (width > std::numeric_limits<std::size_t>::max() / height) {
    return false;
  }
  const std::size_t source_domain = width * height;
  return sweep.points->size() <= source_domain &&
         (!sweep.layout.organized || sweep.points->size() == source_domain);
}

[[nodiscard]] bool lineageReferencesSourceSweep(const core::ObservationLineage& lineage,
                                                core::MeasurementId sweep) noexcept {
  return std::any_of(lineage.usage.begin(), lineage.usage.end(), [&](const auto& usage) {
    const auto* measurement = std::get_if<core::MeasurementId>(&usage.slice.root);
    return measurement != nullptr && *measurement == sweep &&
           (usage.role == core::ObservationRole::PrimaryResidual ||
            usage.role == core::ObservationRole::DerivedSummary);
  });
}

[[nodiscard]] bool writeLineage(core::CanonicalEncoder& encoder,
                                const core::ObservationLineage& lineage) {
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.writeU64(lineage.id.value())) ||
      !write(encoder.writeOptionalMarker(core::contentHashPresent(lineage.checksum))) ||
      (core::contentHashPresent(lineage.checksum) && !write(encoder.writeHash(lineage.checksum))) ||
      !write(encoder.writeU64(static_cast<std::uint64_t>(lineage.usage.size())))) {
    return false;
  }
  for (const core::ObservationUsage& usage : lineage.usage) {
    const bool measurement = std::holds_alternative<core::MeasurementId>(usage.slice.root);
    if (!write(encoder.writeU8(measurement ? 0U : 1U)) ||
        !write(encoder.writeU64(
            measurement ? std::get<core::MeasurementId>(usage.slice.root).value()
                        : std::get<core::GnssObservationId>(usage.slice.root).value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(usage.slice.kind))) ||
        !write(encoder.writeU64(usage.slice.begin)) || !write(encoder.writeU64(usage.slice.end)) ||
        !write(
            encoder.writeOptionalMarker(core::contentHashPresent(usage.slice.source_checksum))) ||
        (core::contentHashPresent(usage.slice.source_checksum) &&
         !write(encoder.writeHash(usage.slice.source_checksum))) ||
        !write(encoder.writeU64(usage.slice.calibration.value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(usage.role))) ||
        !write(encoder.writeU64(usage.consumer.value())) ||
        !write(encoder.writeOptionalMarker(usage.factor_group.has_value())) ||
        (usage.factor_group && !write(encoder.writeU64(usage.factor_group->value()))) ||
        !write(encoder.writeOptionalMarker(usage.correlation_group.has_value())) ||
        (usage.correlation_group && !write(encoder.writeU64(usage.correlation_group->value())))) {
      return false;
    }
  }
  if (!write(encoder.writeU64(static_cast<std::uint64_t>(lineage.correlations.size())))) {
    return false;
  }
  for (const core::CorrelationDeclaration& declaration : lineage.correlations) {
    if (!write(encoder.writeU64(declaration.group.value())) ||
        !write(encoder.writeU64(declaration.policy.value())) ||
        !write(encoder.writeU8(static_cast<std::uint8_t>(declaration.treatment))) ||
        !write(encoder.writeDouble(declaration.covariance_inflation)) ||
        !write(encoder.writeOptionalMarker(declaration.total_information_cap.has_value())) ||
        (declaration.total_information_cap &&
         !write(encoder.writeDouble(*declaration.total_information_cap)))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] core::Result<core::ContentHash, LidarMapPayloadError> rawPayloadChecksum(
    const core::LidarSweep& sweep) {
  using Result = core::Result<core::ContentHash, LidarMapPayloadError>;
  constexpr std::uint64_t kFixedBytes = 1024U;
  constexpr std::uint64_t kBytesPerPoint = 64U;
  const auto count = static_cast<std::uint64_t>(sweep.points->size());
  if (count > (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) / kBytesPerPoint) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "raw LiDAR map payload is too large to checksum"));
  }
  auto encoder = core::CanonicalEncoder::create(kLidarMapRawPayloadChecksumDomain,
                                                kLidarMapRawPayloadChecksumSchemaVersion,
                                                kFixedBytes + count * kBytesPerPoint);
  if (!encoder) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "raw LiDAR map checksum encoder initialization failed"));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  const core::RecordHeader& header = sweep.header;
  const core::SourceStamp& stamp = sweep.stamp;
  if (!write(encoder.value().writeU32(header.schema_version)) ||
      !write(encoder.value().writeU64(header.trace.value())) ||
      !write(encoder.value().writeU64(header.producer.value())) ||
      !write(encoder.value().writeU64(header.session.value())) ||
      !write(encoder.value().writeI64(header.created_at.nanoseconds)) ||
      !write(encoder.value().writeU64(header.config.value())) ||
      !write(encoder.value().writeOptionalMarker(header.direct_calibration.has_value())) ||
      (header.direct_calibration &&
       !write(encoder.value().writeU64(header.direct_calibration->value()))) ||
      !write(encoder.value().writeU64(sweep.id.value())) ||
      !write(encoder.value().writeU64(sweep.lidar.value())) ||
      !write(encoder.value().writeI64(stamp.raw_time.nanoseconds)) ||
      !write(encoder.value().writeI64(stamp.fusion_time.nanoseconds)) ||
      !write(encoder.value().writeI64(stamp.host_arrival_time.nanoseconds)) ||
      !write(encoder.value().writeU64(stamp.clock_revision.value())) ||
      !write(encoder.value().writeU64(stamp.source_epoch.value())) ||
      !write(encoder.value().writeOptionalMarker(stamp.device_sequence.has_value())) ||
      (stamp.device_sequence && !write(encoder.value().writeU64(stamp.device_sequence->value()))) ||
      !write(encoder.value().writeU64(stamp.ingress_sequence.value())) ||
      !write(encoder.value().writeI64(stamp.uncertainty.nanoseconds)) ||
      !write(encoder.value().writeU8(static_cast<std::uint8_t>(stamp.status))) ||
      !write(encoder.value().writeI64(sweep.acquisition.start.nanoseconds)) ||
      !write(encoder.value().writeI64(sweep.acquisition.end.nanoseconds)) ||
      !write(encoder.value().writeU32(sweep.layout.width)) ||
      !write(encoder.value().writeU32(sweep.layout.height)) ||
      !write(encoder.value().writeBool(sweep.layout.organized)) ||
      !write(encoder.value().writeU64(count))) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "raw LiDAR map checksum header encoding failed"));
  }
  for (const core::LidarPoint& point : *sweep.points) {
    if (!write(encoder.value().writeU32(point.source_index)) ||
        !write(encoder.value().writeU32(std::bit_cast<std::uint32_t>(point.x))) ||
        !write(encoder.value().writeU32(std::bit_cast<std::uint32_t>(point.y))) ||
        !write(encoder.value().writeU32(std::bit_cast<std::uint32_t>(point.z))) ||
        !write(encoder.value().writeU32(std::bit_cast<std::uint32_t>(point.intensity))) ||
        !write(encoder.value().writeI32(point.time_offset_ns)) ||
        !write(encoder.value().writeU16(point.ring))) {
      return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                          "raw LiDAR map checksum row encoding failed"));
    }
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "raw LiDAR map checksum finalization failed"));
  }
  return Result::success(encoded.value().digest());
}

[[nodiscard]] core::Result<core::ContentHash, LidarMapPayloadError> payloadChecksum(
    const AcceptedLidarMapInput& input, const core::ContentHash& raw_checksum) {
  using Result = core::Result<core::ContentHash, LidarMapPayloadError>;
  constexpr std::uint64_t kFixedBytes = 4096U;
  constexpr std::uint64_t kBytesPerLineageUsage = 256U;
  constexpr std::uint64_t kBytesPerCorrelation = 128U;
  const auto usage_count = static_cast<std::uint64_t>(input.localization_lineage.usage.size());
  const auto correlation_count =
      static_cast<std::uint64_t>(input.localization_lineage.correlations.size());
  if (usage_count >
      (std::numeric_limits<std::uint64_t>::max() - kFixedBytes) / kBytesPerLineageUsage) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "LiDAR map payload lineage is too large to checksum"));
  }
  std::uint64_t maximum_bytes = kFixedBytes + usage_count * kBytesPerLineageUsage;
  if (correlation_count >
      (std::numeric_limits<std::uint64_t>::max() - maximum_bytes) / kBytesPerCorrelation) {
    return Result::failure(
        payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                     "LiDAR map payload correlations are too large to checksum"));
  }
  maximum_bytes += correlation_count * kBytesPerCorrelation;
  auto encoder = core::CanonicalEncoder::create(
      kLidarMapPayloadChecksumDomain, kLidarMapPayloadChecksumSchemaVersion, maximum_bytes);
  if (!encoder) {
    return Result::failure(
        payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                     "LiDAR map payload checksum encoder initialization failed"));
  }
  const auto write = [](core::CanonicalEncodingError error) {
    return error == core::CanonicalEncodingError::None;
  };
  if (!write(encoder.value().writeHash(raw_checksum)) ||
      !write(encoder.value().writeU64(input.odom_epoch.value())) ||
      !write(encoder.value().writeU64(input.state.value())) ||
      !write(encoder.value().writeU64(input.accepted_revision.value())) ||
      !write(encoder.value().writeU64(input.admitting_batch.value())) ||
      !write(encoder.value().writeU64(input.recovery_epoch.value())) ||
      !write(encoder.value().writePose3(input.T_odom_imu)) ||
      !write(encoder.value().writeU64(input.calibration.value())) ||
      !write(encoder.value().writeHash(input.registration_cloud_checksum)) ||
      !writeLineage(encoder.value(), input.localization_lineage)) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "LiDAR map payload checksum encoding failed"));
  }
  auto encoded = encoder.value().finish();
  if (!encoded || !core::contentHashPresent(encoded.value().digest())) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::ChecksumFailure,
                                        "LiDAR map payload checksum finalization failed"));
  }
  return Result::success(encoded.value().digest());
}

}  // namespace

AcceptedLidarMapInput::AcceptedLidarMapInput(AcceptedLidarMapInputData data)
    : sweep(std::move(data.sweep)),
      odom_epoch(data.odom_epoch),
      state(data.state),
      accepted_revision(data.accepted_revision),
      admitting_batch(data.admitting_batch),
      recovery_epoch(data.recovery_epoch),
      T_odom_imu(std::move(data.T_odom_imu)),
      calibration(data.calibration),
      registration_cloud_checksum(data.registration_cloud_checksum),
      localization_lineage(std::move(data.localization_lineage)) {}

core::Result<std::shared_ptr<const AcceptedLidarMapInput>, LidarMapPayloadError>
AcceptedLidarMapInput::create(AcceptedLidarMapInputData data) {
  using Result = core::Result<std::shared_ptr<const AcceptedLidarMapInput>, LidarMapPayloadError>;
  if (!data.sweep.id.valid() || !data.sweep.lidar.valid() || !data.odom_epoch.valid() ||
      !data.state.valid() || !data.accepted_revision.valid() || !data.admitting_batch.valid() ||
      !data.recovery_epoch.valid() || !data.calibration.valid() ||
      !core::contentHashPresent(data.registration_cloud_checksum)) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::InvalidIdentity,
                                        "accepted LiDAR map input identity is invalid"));
  }
  if (!data.sweep.acquisition.valid() || !validLayout(data.sweep)) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::InvalidSweep,
                                        "accepted LiDAR map input sweep or layout is invalid"));
  }
  if (!finitePose(data.T_odom_imu)) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::InvalidLocalization,
                                        "accepted LiDAR map input pose is non-finite"));
  }
  if (!data.localization_lineage.id.valid() || data.localization_lineage.usage.empty() ||
      core::validateLineage(data.localization_lineage) != core::LineageValidationError::None ||
      !lineageReferencesSourceSweep(data.localization_lineage, data.sweep.id)) {
    return Result::failure(payloadError(
        LidarMapPayloadErrorCode::InvalidLineage,
        "accepted LiDAR map input lineage is invalid or lacks its exact source sweep"));
  }
  auto input =
      std::shared_ptr<const AcceptedLidarMapInput>(new AcceptedLidarMapInput(std::move(data)));
  return Result::success(std::move(input));
}

LidarMapPayload::LidarMapPayload(std::shared_ptr<const AcceptedLidarMapInput> accepted_input,
                                 core::ContentHash raw_checksum, core::ContentHash payload_checksum)
    : input(std::move(accepted_input)),
      raw_payload_checksum(raw_checksum),
      checksum(payload_checksum) {}

core::Result<std::shared_ptr<const LidarMapPayload>, LidarMapPayloadError> LidarMapPayload::seal(
    std::shared_ptr<const AcceptedLidarMapInput> input) {
  using Result = core::Result<std::shared_ptr<const LidarMapPayload>, LidarMapPayloadError>;
  if (!input) {
    return Result::failure(payloadError(LidarMapPayloadErrorCode::InvalidIdentity,
                                        "LiDAR map payload input is absent"));
  }
  const std::size_t source_domain = input->sweep.layout.sourcePointCount();
  std::optional<std::uint32_t> previous_source_index;
  for (const core::LidarPoint& point : *input->sweep.points) {
    if (!finitePoint(point) || static_cast<std::size_t>(point.source_index) >= source_domain ||
        (previous_source_index && point.source_index <= *previous_source_index)) {
      return Result::failure(payloadError(
          LidarMapPayloadErrorCode::InvalidSweep,
          "raw LiDAR map rows must be finite, source-index ordered, unique, and in layout"));
    }
    previous_source_index = point.source_index;
  }
  auto raw_checksum = rawPayloadChecksum(input->sweep);
  if (!raw_checksum) {
    return Result::failure(raw_checksum.error());
  }
  auto payload_checksum = payloadChecksum(*input, raw_checksum.value());
  if (!payload_checksum) {
    return Result::failure(payload_checksum.error());
  }
  auto payload = std::shared_ptr<const LidarMapPayload>(
      new LidarMapPayload(std::move(input), raw_checksum.value(), payload_checksum.value()));
  return Result::success(std::move(payload));
}

}  // namespace meridian::local
