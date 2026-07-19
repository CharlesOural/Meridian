#include "meridian/global/seal_spool.hpp"

#include "persistence_internal.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>

namespace meridian::global {
namespace {

using Bytes = std::vector<std::byte>;

constexpr std::array<std::byte, 8> kSealMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                              std::byte{'N'}, std::byte{'S'}, std::byte{'E'},
                                              std::byte{'A'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kOutboxMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                std::byte{'N'}, std::byte{'O'}, std::byte{'B'},
                                                std::byte{'X'}, std::byte{'1'}};
constexpr std::array<std::byte, 8> kAckMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                             std::byte{'N'}, std::byte{'A'}, std::byte{'C'},
                                             std::byte{'K'}, std::byte{'1'}};
constexpr std::uint32_t kFormatVersion = 2U;
constexpr std::size_t kFrameOverhead = persistence_internal::kFrameOverhead;

[[nodiscard]] SealSpoolError makeError(SealSpoolErrorCode code, std::string detail,
                                       std::optional<OutboxSequence> sequence = std::nullopt,
                                       std::optional<core::SubmapId> submap = std::nullopt) {
  return SealSpoolError{code, sequence, submap, std::move(detail)};
}

[[nodiscard]] bool zeroHash(const core::ContentHash& hash) noexcept {
  return std::all_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte == 0U; });
}


[[nodiscard]] core::ContentHash hashBytes(std::span<const std::byte> bytes) noexcept {
  return persistence_internal::hashBytes(bytes);
}

[[nodiscard]] bool sha256SelfTest() noexcept {
  return persistence_internal::sha256SelfTest();
}

class Writer {
public:
  void raw(std::span<const std::byte> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void u8(std::uint8_t value) { bytes_.push_back(static_cast<std::byte>(value)); }
  void u32(std::uint32_t value) {
    for (std::size_t index = 0U; index < 4U; ++index) {
      bytes_.push_back(static_cast<std::byte>(value >> ((3U - index) * 8U)));
    }
  }
  void u64(std::uint64_t value) {
    for (std::size_t index = 0U; index < 8U; ++index) {
      bytes_.push_back(static_cast<std::byte>(value >> ((7U - index) * 8U)));
    }
  }
  void i64(std::int64_t value) { u64(std::bit_cast<std::uint64_t>(value)); }
  void floating(double value) { u64(std::bit_cast<std::uint64_t>(value == 0.0 ? 0.0 : value)); }
  void hash(const core::ContentHash& value) { raw(std::as_bytes(std::span(value))); }
  template <typename Tag>
  void id(core::StrongId<Tag> value) {
    u64(value.value());
  }
  void boolean(bool value) { u8(value ? 1U : 0U); }
  [[nodiscard]] const Bytes& bytes() const noexcept { return bytes_; }
  [[nodiscard]] Bytes take() && { return std::move(bytes_); }

private:
  Bytes bytes_;
};

class Reader {
public:
  explicit Reader(std::span<const std::byte> bytes, std::size_t maximum_elements)
      : bytes_(bytes), maximum_elements_(maximum_elements) {}

  [[nodiscard]] bool raw(std::span<std::byte> output) noexcept {
    if (output.size() > remaining()) {
      valid_ = false;
      return false;
    }
    std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_), output.size(),
                output.begin());
    offset_ += output.size();
    return true;
  }
  [[nodiscard]] std::uint8_t u8() noexcept {
    if (remaining() < 1U) {
      valid_ = false;
      return 0U;
    }
    return static_cast<std::uint8_t>(bytes_[offset_++]);
  }
  [[nodiscard]] std::uint32_t u32() noexcept {
    std::uint32_t value = 0U;
    for (std::size_t index = 0U; index < 4U; ++index) {
      value = (value << 8U) | u8();
    }
    return value;
  }
  [[nodiscard]] std::uint64_t u64() noexcept {
    std::uint64_t value = 0U;
    for (std::size_t index = 0U; index < 8U; ++index) {
      value = (value << 8U) | u8();
    }
    return value;
  }
  [[nodiscard]] std::int64_t i64() noexcept { return std::bit_cast<std::int64_t>(u64()); }
  [[nodiscard]] double floating() noexcept {
    const double value = std::bit_cast<double>(u64());
    if (!std::isfinite(value)) {
      valid_ = false;
    }
    return value;
  }
  [[nodiscard]] core::ContentHash hash() noexcept {
    core::ContentHash value{};
    (void)raw(std::as_writable_bytes(std::span(value)));
    return value;
  }
  template <typename Id>
  [[nodiscard]] Id id() noexcept {
    return Id(u64());
  }
  [[nodiscard]] bool boolean() noexcept {
    const std::uint8_t value = u8();
    if (value > 1U) {
      valid_ = false;
    }
    return value == 1U;
  }
  [[nodiscard]] std::size_t count(std::size_t minimum_element_bytes = 1U) noexcept {
    const std::uint64_t value = u64();
    if (minimum_element_bytes == 0U || value > remaining_element_budget_ ||
        value > remaining() / minimum_element_bytes ||
        value > std::numeric_limits<std::size_t>::max()) {
      valid_ = false;
      return 0U;
    }
    const auto count = static_cast<std::size_t>(value);
    remaining_element_budget_ -= count;
    return count;
  }
  [[nodiscard]] std::size_t sizeValue(std::size_t explicit_maximum) noexcept {
    const std::uint64_t value = u64();
    if (value > explicit_maximum || value > std::numeric_limits<std::size_t>::max()) {
      valid_ = false;
      return 0U;
    }
    return static_cast<std::size_t>(value);
  }
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] bool complete() const noexcept { return valid_ && offset_ == bytes_.size(); }
  [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
  void invalidate() noexcept { valid_ = false; }

private:
  std::span<const std::byte> bytes_;
  std::size_t maximum_elements_{};
  std::size_t remaining_element_budget_{maximum_elements_};
  std::size_t offset_{};
  bool valid_{true};
};

void writePose(Writer& writer, const core::Pose3d& pose) {
  const Eigen::Matrix4d matrix = pose.matrix();
  for (Eigen::Index row = 0; row < 4; ++row) {
    for (Eigen::Index column = 0; column < 4; ++column) {
      writer.floating(matrix(row, column));
    }
  }
}

[[nodiscard]] core::Pose3d readPose(Reader& reader) {
  Eigen::Matrix4d matrix;
  for (Eigen::Index row = 0; row < 4; ++row) {
    for (Eigen::Index column = 0; column < 4; ++column) {
      matrix(row, column) = reader.floating();
    }
  }

  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  const bool homogeneous_bottom_row =
      matrix.bottomLeftCorner<1, 3>().cwiseAbs().maxCoeff() <= 1.0e-12 &&
      std::abs(matrix(3, 3) - 1.0) <= 1.0e-12;
  const bool valid_rotation =
      matrix.allFinite() &&
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <= 1.0e-8 &&
      std::abs(rotation.determinant() - 1.0) <= 1.0e-8;
  if (!homogeneous_bottom_row || !valid_rotation) {
    // Never hand untrusted persisted coefficients to Sophus: depending on its
    // assertion policy, a malformed rotation can terminate the process before
    // recovery has a chance to return a structured corruption error.
    reader.invalidate();
    return core::Pose3d{};
  }
  return core::Pose3d(Sophus::SO3d(matrix.topLeftCorner<3, 3>()), matrix.topRightCorner<3, 1>());
}

template <typename Derived>
void writeMatrix(Writer& writer, const Eigen::MatrixBase<Derived>& matrix) {
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
      writer.floating(matrix(row, column));
    }
  }
}

template <int Rows, int Columns>
[[nodiscard]] Eigen::Matrix<double, Rows, Columns> readMatrix(Reader& reader) {
  Eigen::Matrix<double, Rows, Columns> matrix;
  for (Eigen::Index row = 0; row < Rows; ++row) {
    for (Eigen::Index column = 0; column < Columns; ++column) {
      matrix(row, column) = reader.floating();
    }
  }
  return matrix;
}

void writeHeader(Writer& writer, const core::RecordHeader& header) {
  writer.u32(header.schema_version);
  writer.id(header.trace);
  writer.id(header.producer);
  writer.id(header.session);
  writer.i64(header.created_at.nanoseconds);
  writer.id(header.config);
  writer.boolean(header.direct_calibration.has_value());
  if (header.direct_calibration) {
    writer.id(*header.direct_calibration);
  }
}

[[nodiscard]] core::RecordHeader readHeader(Reader& reader) {
  core::RecordHeader header;
  header.schema_version = reader.u32();
  header.trace = reader.id<core::TraceId>();
  header.producer = reader.id<core::ProducerId>();
  header.session = reader.id<core::SessionId>();
  header.created_at = core::FusionTime{reader.i64()};
  header.config = reader.id<core::ConfigRevision>();
  if (reader.boolean()) {
    header.direct_calibration = reader.id<core::CalibrationEpoch>();
  }
  return header;
}

void writeSubmapRef(Writer& writer, const core::SubmapRef& submap) {
  writer.id(submap.session);
  writer.id(submap.odom_epoch);
  writer.id(submap.id);
  writer.id(submap.calibration);
  writer.id(submap.content_revision);
  writer.hash(submap.local_content_checksum);
}

[[nodiscard]] core::SubmapRef readSubmapRef(Reader& reader) {
  return core::SubmapRef{reader.id<core::SessionId>(),
                         reader.id<core::OdomEpoch>(),
                         reader.id<core::SubmapId>(),
                         reader.id<core::CalibrationEpoch>(),
                         reader.id<core::SubmapContentRevision>(), reader.hash()};
}

void writeSubmapFrame(Writer& writer, const FinalizedSubmapFrame& submap) {
  writeSubmapRef(writer, submap.ref);
  writePose(writer, submap.T_odom_submap);
  writer.i64(submap.support_end.nanoseconds);
}

[[nodiscard]] FinalizedSubmapFrame readSubmapFrame(Reader& reader) {
  return FinalizedSubmapFrame{readSubmapRef(reader), readPose(reader),
                              core::FusionTime{reader.i64()}};
}

void writeIdentity(Writer& writer, const core::SparseSubmapSealIdentity& identity) {
  writeSubmapRef(writer, identity.ref);
  writer.hash(identity.seal_checksum);
}

[[nodiscard]] core::SparseSubmapSealIdentity readIdentity(Reader& reader) {
  return {readSubmapRef(reader), reader.hash()};
}

void writeLineage(Writer& writer, const core::ObservationLineage& lineage) {
  writer.id(lineage.id);
  writer.u64(lineage.usage.size());
  for (const auto& usage : lineage.usage) {
    writer.u8(static_cast<std::uint8_t>(usage.slice.root.index()));
    std::visit([&writer](const auto& id) { writer.id(id); }, usage.slice.root);
    writer.u8(static_cast<std::uint8_t>(usage.slice.kind));
    writer.u64(usage.slice.begin);
    writer.u64(usage.slice.end);
    writer.hash(usage.slice.source_checksum);
    writer.id(usage.slice.calibration);
    writer.u8(static_cast<std::uint8_t>(usage.role));
    writer.id(usage.consumer);
    writer.boolean(usage.factor_group.has_value());
    if (usage.factor_group) {
      writer.id(*usage.factor_group);
    }
    writer.boolean(usage.correlation_group.has_value());
    if (usage.correlation_group) {
      writer.id(*usage.correlation_group);
    }
  }
  writer.u64(lineage.correlations.size());
  for (const auto& declaration : lineage.correlations) {
    writer.id(declaration.group);
    writer.id(declaration.policy);
    writer.u8(static_cast<std::uint8_t>(declaration.treatment));
    writer.floating(declaration.covariance_inflation);
    writer.boolean(declaration.total_information_cap.has_value());
    if (declaration.total_information_cap) {
      writer.floating(*declaration.total_information_cap);
    }
  }
  writer.hash(lineage.checksum);
}

[[nodiscard]] core::ObservationLineage readLineage(Reader& reader) {
  core::ObservationLineage lineage;
  lineage.id = reader.id<core::ObservationLineageId>();
  const std::size_t usages = reader.count(77U);
  lineage.usage.reserve(usages);
  for (std::size_t index = 0U; index < usages; ++index) {
    core::ObservationUsage usage;
    const std::uint8_t root = reader.u8();
    if (root == 0U) {
      usage.slice.root = reader.id<core::MeasurementId>();
    } else if (root == 1U) {
      usage.slice.root = reader.id<core::GnssObservationId>();
    } else {
      reader.invalidate();
    }
    const std::uint8_t slice_kind = reader.u8();
    if (slice_kind > static_cast<std::uint8_t>(core::SliceKind::IndexRange)) {
      reader.invalidate();
    }
    usage.slice.kind = static_cast<core::SliceKind>(slice_kind);
    usage.slice.begin = reader.u64();
    usage.slice.end = reader.u64();
    usage.slice.source_checksum = reader.hash();
    usage.slice.calibration = reader.id<core::CalibrationEpoch>();
    const std::uint8_t role = reader.u8();
    if (role > static_cast<std::uint8_t>(core::ObservationRole::DerivedSummary)) {
      reader.invalidate();
    }
    usage.role = static_cast<core::ObservationRole>(role);
    usage.consumer = reader.id<core::DerivedRecordId>();
    if (reader.boolean()) {
      usage.factor_group = reader.id<core::FactorGroupId>();
    }
    if (reader.boolean()) {
      usage.correlation_group = reader.id<core::CorrelationGroupId>();
    }
    lineage.usage.push_back(std::move(usage));
  }
  const std::size_t correlations = reader.count(26U);
  lineage.correlations.reserve(correlations);
  for (std::size_t index = 0U; index < correlations; ++index) {
    core::CorrelationDeclaration declaration;
    declaration.group = reader.id<core::CorrelationGroupId>();
    declaration.policy = reader.id<core::CorrelationPolicyRevision>();
    const std::uint8_t treatment = reader.u8();
    if (treatment > static_cast<std::uint8_t>(core::CorrelationTreatment::NotIndependent)) {
      reader.invalidate();
    }
    declaration.treatment = static_cast<core::CorrelationTreatment>(treatment);
    declaration.covariance_inflation = reader.floating();
    if (reader.boolean()) {
      declaration.total_information_cap = reader.floating();
    }
    lineage.correlations.push_back(std::move(declaration));
  }
  lineage.checksum = reader.hash();
  return lineage;
}

void writeBlobRef(Writer& writer, const core::BlobRef& blob) {
  writer.id(blob.store);
  writer.id(blob.id);
  writer.hash(blob.checksum);
  writer.id(blob.layout);
  writer.u64(blob.bytes);
  writer.u8(static_cast<std::uint8_t>(blob.storage));
  writer.boolean(blob.lease_token.has_value());
  if (blob.lease_token) {
    writer.id(blob.lease_token->id);
    writer.id(blob.lease_token->issuing_store_instance);
  }
}

[[nodiscard]] core::BlobRef readBlobRef(Reader& reader) {
  core::BlobRef blob;
  blob.store = reader.id<core::BlobStoreId>();
  blob.id = reader.id<core::BlobId>();
  blob.checksum = reader.hash();
  blob.layout = reader.id<core::LayoutId>();
  blob.bytes = reader.u64();
  const std::uint8_t storage = reader.u8();
  if (storage > static_cast<std::uint8_t>(core::BlobStorage::DurableSpool)) {
    reader.invalidate();
  }
  blob.storage = static_cast<core::BlobStorage>(storage);
  if (reader.boolean()) {
    blob.lease_token = core::LeaseToken{reader.id<core::LeaseTokenId>(),
                                        reader.id<core::StoreInstanceEpoch>()};
  }
  return blob;
}

void writePayload(Writer& writer, const PlacePayloadInput& payload) {
  writer.hash(payload.checksum);
  writer.boolean(payload.record.has_value());
  if (payload.record) {
    writeBlobRef(writer, *payload.record);
  }
  // In-memory records are rejected before serialization. Keeping this marker
  // makes accidental format widening detectable during recovery.
  writer.boolean(payload.in_memory != nullptr);
}

[[nodiscard]] PlacePayloadInput readPayload(Reader& reader) {
  PlacePayloadInput payload;
  payload.checksum = reader.hash();
  if (reader.boolean()) {
    payload.record = readBlobRef(reader);
  }
  if (reader.boolean()) {
    reader.invalidate();
  }
  return payload;
}

void writeInformation(Writer& writer, const core::RankAwareInformation& information) {
  writeMatrix(writer, information.basis);
  writeMatrix(writer, information.eigenvalues);
  writer.u64(information.rank);
  writer.u8(static_cast<std::uint8_t>(information.tangent));
}

[[nodiscard]] core::RankAwareInformation readInformation(Reader& reader) {
  core::RankAwareInformation information;
  information.basis = readMatrix<6, 6>(reader);
  information.eigenvalues = readMatrix<6, 1>(reader);
  information.rank = reader.sizeValue(6U);
  const std::uint8_t tangent = reader.u8();
  if (tangent != static_cast<std::uint8_t>(
                     core::PoseTangentConvention::RightTranslationFirst)) {
    reader.invalidate();
  }
  information.tangent = static_cast<core::PoseTangentConvention>(tangent);
  return information;
}

void writePoseCovariance(Writer& writer, const core::PoseCovariance& covariance) {
  writeMatrix(writer, covariance.matrix);
  writer.u8(static_cast<std::uint8_t>(covariance.tangent));
}

[[nodiscard]] core::PoseCovariance readPoseCovariance(Reader& reader) {
  core::PoseCovariance covariance;
  covariance.matrix = readMatrix<6, 6>(reader);
  const std::uint8_t tangent = reader.u8();
  if (tangent != static_cast<std::uint8_t>(
                     core::PoseTangentConvention::RightTranslationFirst)) {
    reader.invalidate();
  }
  covariance.tangent = static_cast<core::PoseTangentConvention>(tangent);
  return covariance;
}

[[nodiscard]] bool sameSubmapFrame(const FinalizedSubmapFrame& lhs,
                                   const FinalizedSubmapFrame& rhs) noexcept {
  return lhs.ref == rhs.ref && lhs.support_end == rhs.support_end &&
         (lhs.T_odom_submap.matrix() - rhs.T_odom_submap.matrix()).cwiseAbs().maxCoeff() <= 1.0e-12;
}

void writeSealPayload(Writer& writer, const SparseSubmapSealRecord& seal) {
  writeHeader(writer, seal.header);
  writeIdentity(writer, seal.identity);
  writeSubmapFrame(writer, seal.submap);
  writer.i64(seal.core_interval.start.nanoseconds);
  writer.i64(seal.core_interval.end.nanoseconds);
  writer.id(seal.start_boundary_state);
  writer.id(seal.end_boundary_state);
  writer.id(seal.final_local_revision);

  writer.u64(seal.calibration_epochs.size());
  for (const auto calibration : seal.calibration_epochs) {
    writer.id(calibration);
  }
  writer.u64(seal.core_state_ids.size());
  for (const auto state : seal.core_state_ids) {
    writer.id(state);
  }
  writer.u64(seal.finalized_trajectory.size());
  for (const auto& state : seal.finalized_trajectory) {
    writer.id(state.state);
    writer.i64(state.exact_time.nanoseconds);
    writer.id(state.final_local_revision);
    writePose(writer, state.T_submap_imu);
    writeMatrix(writer, state.velocity_submap);
    writeMatrix(writer, state.gyro_bias);
    writeMatrix(writer, state.accel_bias);
    writePoseCovariance(writer, state.pose_covariance);
  }
  writer.u64(seal.condensed_factor_ids.size());
  for (const auto factor : seal.condensed_factor_ids) {
    writer.id(factor);
  }
  writeLineage(writer, seal.factor_lineage);

  writer.floating(seal.registration_proxy.voxel_resolution_m);
  writer.u64(seal.registration_proxy.points.size());
  for (const auto& point : seal.registration_proxy.points) {
    writeMatrix(writer, point.point_submap);
    writeMatrix(writer, point.normal_submap);
    writer.floating(point.weight);
  }
  writer.hash(seal.registration_proxy.checksum);

  writer.u64(seal.lidar_place_index.size());
  for (const auto& entry : seal.lidar_place_index) {
    writer.id(entry.sweep);
    writer.id(entry.state);
    writer.i64(entry.terminal_time.nanoseconds);
    writePose(writer, entry.T_submap_lidar);
    writePayload(writer, entry.payload);
    writeLineage(writer, entry.lineage);
  }
  writer.u64(seal.visual_place_index.size());
  for (const auto& entry : seal.visual_place_index) {
    writer.id(entry.frame);
    writer.id(entry.camera);
    writer.id(entry.state);
    writer.i64(entry.terminal_time.nanoseconds);
    writePose(writer, entry.T_submap_camera);
    writePayload(writer, entry.payload);
    writeLineage(writer, entry.lineage);
  }

  writer.boolean(seal.incoming_adjacent.has_value());
  if (seal.incoming_adjacent) {
    const auto& adjacent = *seal.incoming_adjacent;
    writeSubmapFrame(writer, adjacent.global_append.submap);
    writeSubmapRef(writer, adjacent.global_append.constraint.from);
    writeSubmapRef(writer, adjacent.global_append.constraint.to);
    writePose(writer, adjacent.global_append.constraint.T_from_to);
    writeInformation(writer, adjacent.global_append.constraint.information);
    writePoseCovariance(writer, adjacent.relative_covariance);
    writeLineage(writer, adjacent.lineage);
    writer.u64(adjacent.eliminated_factor_ids.size());
    for (const auto factor : adjacent.eliminated_factor_ids) {
      writer.id(factor);
    }
    writer.id(adjacent.final_local_revision);
  }
  writer.u8(static_cast<std::uint8_t>(seal.storage_semantics));
}

[[nodiscard]] SparseSubmapSeal readSealPayload(Reader& reader) {
  const core::RecordHeader header = readHeader(reader);
  const core::SparseSubmapSealIdentity identity = readIdentity(reader);
  FinalizedSubmapFrame submap = readSubmapFrame(reader);
  auto mutable_seal = std::make_shared<SparseSubmapSealRecord>(submap);
  mutable_seal->header = header;
  mutable_seal->identity = identity;
  mutable_seal->core_interval =
      core::TimeRange{core::FusionTime{reader.i64()}, core::FusionTime{reader.i64()}};
  mutable_seal->start_boundary_state = reader.id<core::StateId>();
  mutable_seal->end_boundary_state = reader.id<core::StateId>();
  mutable_seal->final_local_revision = reader.id<core::LocalGraphRevision>();

  const std::size_t calibrations = reader.count(8U);
  mutable_seal->calibration_epochs.reserve(calibrations);
  for (std::size_t index = 0U; index < calibrations; ++index) {
    mutable_seal->calibration_epochs.push_back(reader.id<core::CalibrationEpoch>());
  }
  const std::size_t states = reader.count(8U);
  mutable_seal->core_state_ids.reserve(states);
  for (std::size_t index = 0U; index < states; ++index) {
    mutable_seal->core_state_ids.push_back(reader.id<core::StateId>());
  }
  const std::size_t trajectory = reader.count(513U);
  mutable_seal->finalized_trajectory.reserve(trajectory);
  for (std::size_t index = 0U; index < trajectory; ++index) {
    FinalizedSubmapStateRecord state;
    state.state = reader.id<core::StateId>();
    state.exact_time = core::FusionTime{reader.i64()};
    state.final_local_revision = reader.id<core::LocalGraphRevision>();
    state.T_submap_imu = readPose(reader);
    state.velocity_submap = readMatrix<3, 1>(reader);
    state.gyro_bias = readMatrix<3, 1>(reader);
    state.accel_bias = readMatrix<3, 1>(reader);
    state.pose_covariance = readPoseCovariance(reader);
    mutable_seal->finalized_trajectory.push_back(std::move(state));
  }
  const std::size_t factors = reader.count(8U);
  mutable_seal->condensed_factor_ids.reserve(factors);
  for (std::size_t index = 0U; index < factors; ++index) {
    mutable_seal->condensed_factor_ids.push_back(reader.id<core::FactorId>());
  }
  mutable_seal->factor_lineage = readLineage(reader);

  mutable_seal->registration_proxy.voxel_resolution_m = reader.floating();
  const std::size_t points = reader.count(56U);
  mutable_seal->registration_proxy.points.reserve(points);
  for (std::size_t index = 0U; index < points; ++index) {
    RegistrationProxyPoint point;
    point.point_submap = readMatrix<3, 1>(reader);
    point.normal_submap = readMatrix<3, 1>(reader);
    point.weight = reader.floating();
    mutable_seal->registration_proxy.points.push_back(std::move(point));
  }
  mutable_seal->registration_proxy.checksum = reader.hash();

  const std::size_t lidar_entries = reader.count(242U);
  mutable_seal->lidar_place_index.reserve(lidar_entries);
  for (std::size_t index = 0U; index < lidar_entries; ++index) {
    LidarPlacePayloadIndexEntry entry;
    entry.sweep = reader.id<core::SweepId>();
    entry.state = reader.id<core::StateId>();
    entry.terminal_time = core::FusionTime{reader.i64()};
    entry.T_submap_lidar = readPose(reader);
    entry.payload = readPayload(reader);
    entry.lineage = readLineage(reader);
    mutable_seal->lidar_place_index.push_back(std::move(entry));
  }
  const std::size_t visual_entries = reader.count(250U);
  mutable_seal->visual_place_index.reserve(visual_entries);
  for (std::size_t index = 0U; index < visual_entries; ++index) {
    VisualPlacePayloadIndexEntry entry;
    entry.frame = reader.id<core::CameraFrameId>();
    entry.camera = reader.id<core::CameraId>();
    entry.state = reader.id<core::StateId>();
    entry.terminal_time = core::FusionTime{reader.i64()};
    entry.T_submap_camera = readPose(reader);
    entry.payload = readPayload(reader);
    entry.lineage = readLineage(reader);
    mutable_seal->visual_place_index.push_back(std::move(entry));
  }

  if (reader.boolean()) {
    FinalizedSubmapFrame appended = readSubmapFrame(reader);
    core::SubmapRef from = readSubmapRef(reader);
    core::SubmapRef to = readSubmapRef(reader);
    core::Pose3d transform = readPose(reader);
    core::RankAwareInformation information = readInformation(reader);
    core::PoseCovariance covariance = readPoseCovariance(reader);
    core::ObservationLineage lineage = readLineage(reader);
    const std::size_t eliminated = reader.count(8U);
    std::vector<core::FactorId> factor_ids;
    factor_ids.reserve(eliminated);
    for (std::size_t index = 0U; index < eliminated; ++index) {
      factor_ids.push_back(reader.id<core::FactorId>());
    }
    const core::LocalGraphRevision revision = reader.id<core::LocalGraphRevision>();
    RelativeAnchorConstraint constraint{std::move(from), std::move(to), std::move(transform),
                                        std::move(information)};
    AdjacentSubmapAppend append{std::move(appended), std::move(constraint)};
    mutable_seal->incoming_adjacent =
        AdjacentConstraintRecord{std::move(append), std::move(covariance), std::move(lineage),
                                 std::move(factor_ids), revision};
  }
  const std::uint8_t storage_semantics = reader.u8();
  if (storage_semantics !=
      static_cast<std::uint8_t>(SparseSealStorageSemantics::VolatileInProcessOnly)) {
    reader.invalidate();
  }
  mutable_seal->storage_semantics = static_cast<SparseSealStorageSemantics>(storage_semantics);
  return std::const_pointer_cast<const SparseSubmapSealRecord>(mutable_seal);
}

[[nodiscard]] bool validPose(const core::Pose3d& pose) noexcept {
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = pose.so3().matrix();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <= 1.0e-8 &&
         std::abs(rotation.determinant() - 1.0) <= 1.0e-8;
}

[[nodiscard]] bool validInformation(const core::RankAwareInformation& information) noexcept {
  if (!information.finite() || information.rank == 0U ||
      information.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      (information.basis.transpose() * information.basis - core::Matrix6d::Identity())
              .cwiseAbs()
              .maxCoeff() > 1.0e-8) {
    return false;
  }
  for (std::size_t index = 0U; index < 6U; ++index) {
    const double value = information.eigenvalues(static_cast<Eigen::Index>(index));
    if ((index < information.rank && value <= 0.0) || (index >= information.rank && value != 0.0)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool validCovariance(const core::PoseCovariance& covariance) noexcept {
  if (!covariance.finite() ||
      covariance.tangent != core::PoseTangentConvention::RightTranslationFirst) {
    return false;
  }
  const double scale = std::max(1.0, covariance.matrix.cwiseAbs().maxCoeff());
  if ((covariance.matrix - covariance.matrix.transpose()).cwiseAbs().maxCoeff() >
      1.0e-9 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<core::Matrix6d> solver(
      0.5 * (covariance.matrix + covariance.matrix.transpose()), Eigen::EigenvaluesOnly);
  return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

[[nodiscard]] bool validPersistedLineage(const core::ObservationLineage& lineage) noexcept {
  for (const auto& usage : lineage.usage) {
    const auto root = usage.slice.root.index();
    if (root > 1U || static_cast<std::uint8_t>(usage.slice.kind) >
                         static_cast<std::uint8_t>(core::SliceKind::IndexRange) ||
        static_cast<std::uint8_t>(usage.role) >
            static_cast<std::uint8_t>(core::ObservationRole::DerivedSummary)) {
      return false;
    }
  }
  for (const auto& declaration : lineage.correlations) {
    if (static_cast<std::uint8_t>(declaration.treatment) >
            static_cast<std::uint8_t>(core::CorrelationTreatment::NotIndependent) ||
        (declaration.total_information_cap &&
         (!std::isfinite(*declaration.total_information_cap) ||
          *declaration.total_information_cap < 0.0))) {
      return false;
    }
  }
  return core::validateLineage(lineage) == core::LineageValidationError::None;
}

[[nodiscard]] std::optional<SealSpoolError> validatePayload(const PlacePayloadInput& payload,
                                                            core::SubmapId submap) {
  if (payload.in_memory) {
    return makeError(
        SealSpoolErrorCode::UnsupportedInMemoryPayload,
        "process-local place payload bytes cannot be claimed as a durable spool reference",
        std::nullopt, submap);
  }
  if (!payload.record) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "place payload has neither a durable BlobRef nor serializable content",
                     std::nullopt, submap);
  }
  const core::BlobRef& blob = *payload.record;
  if (blob.storage != core::BlobStorage::DurableSpool || blob.lease_token.has_value()) {
    return makeError(SealSpoolErrorCode::NonDurablePayloadReference,
                     "final seal payload must name lease-free DurableSpool content", std::nullopt,
                     submap);
  }
  if (!blob.store.valid() || !blob.id.valid() || !blob.layout.valid() || blob.bytes == 0U ||
      zeroHash(blob.checksum) || payload.checksum != blob.checksum) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "durable place payload identity, layout, bytes, or checksum is invalid",
                     std::nullopt, submap);
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> validateSeal(const SparseSubmapSeal& seal,
                                                         const SealSpoolConfig& config) {
  if (!seal) {
    return makeError(SealSpoolErrorCode::InvalidSeal, "sparse seal pointer is null");
  }
  const auto& record = *seal;
  const auto& identity = record.identity;
  if (record.header.schema_version == 0U || !record.header.trace.valid() ||
      !record.header.producer.valid() || !record.header.session.valid() ||
      !record.header.config.valid() ||
      core::validateSubmapRef(identity.ref) != core::SubmapRefValidationError::None ||
      zeroHash(identity.seal_checksum) || identity.ref != record.submap.ref ||
      identity.ref.session != record.header.session || !validPose(record.submap.T_odom_submap) ||
      record.submap.support_end != record.core_interval.end || !record.core_interval.valid() ||
      !record.start_boundary_state.valid() || !record.end_boundary_state.valid() ||
      !record.final_local_revision.valid() || record.calibration_epochs.empty() ||
      record.storage_semantics != SparseSealStorageSemantics::VolatileInProcessOnly) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "seal header, immutable identity, support, or finality fields are invalid",
                     std::nullopt, identity.ref.id);
  }
  if (record.calibration_epochs.size() > config.maximum_vector_elements ||
      record.core_state_ids.size() > config.maximum_vector_elements ||
      record.finalized_trajectory.size() > config.maximum_vector_elements ||
      record.condensed_factor_ids.size() > config.maximum_vector_elements ||
      record.registration_proxy.points.size() > config.maximum_vector_elements ||
      record.lidar_place_index.size() > config.maximum_vector_elements ||
      record.visual_place_index.size() > config.maximum_vector_elements) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "seal contains a vector larger than the configured decode bound", std::nullopt,
                     identity.ref.id);
  }
  if (!std::is_sorted(record.calibration_epochs.begin(), record.calibration_epochs.end()) ||
      std::adjacent_find(record.calibration_epochs.begin(), record.calibration_epochs.end()) !=
          record.calibration_epochs.end() ||
      std::any_of(record.calibration_epochs.begin(), record.calibration_epochs.end(),
                  [](core::CalibrationEpoch epoch) { return !epoch.valid(); }) ||
      !std::binary_search(record.calibration_epochs.begin(), record.calibration_epochs.end(),
                          identity.ref.calibration) ||
      (record.header.direct_calibration &&
       *record.header.direct_calibration != identity.ref.calibration)) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "calibration epochs must be valid, unique, and canonically ordered",
                     std::nullopt, identity.ref.id);
  }
  if (!record.factor_lineage.id.valid() || !validPersistedLineage(record.factor_lineage) ||
      zeroHash(record.factor_lineage.checksum) ||
      !std::isfinite(record.registration_proxy.voxel_resolution_m) ||
      record.registration_proxy.voxel_resolution_m <= 0.0 ||
      zeroHash(record.registration_proxy.checksum)) {
    return makeError(SealSpoolErrorCode::InvalidSeal,
                     "factor lineage or registration proxy is invalid", std::nullopt,
                     identity.ref.id);
  }
  std::set<core::StateId> states;
  for (const auto state : record.core_state_ids) {
    if (!state.valid() || !states.insert(state).second) {
      return makeError(SealSpoolErrorCode::InvalidSeal, "core state IDs must be valid and unique",
                       std::nullopt, identity.ref.id);
    }
  }
  core::FusionTime previous_time = record.core_interval.start;
  for (std::size_t index = 0U; index < record.finalized_trajectory.size(); ++index) {
    const auto& state = record.finalized_trajectory[index];
    if (!state.state.valid() || !state.final_local_revision.valid() ||
        !validPose(state.T_submap_imu) || !state.velocity_submap.allFinite() ||
        !state.gyro_bias.allFinite() || !state.accel_bias.allFinite() ||
        !validCovariance(state.pose_covariance) ||
        !record.core_interval.contains(state.exact_time) ||
        (index > 0U && state.exact_time < previous_time)) {
      return makeError(SealSpoolErrorCode::InvalidSeal,
                       "final trajectory state, time, pose, or covariance is invalid", std::nullopt,
                       identity.ref.id);
    }
    previous_time = state.exact_time;
  }
  for (const auto& point : record.registration_proxy.points) {
    if (!point.point_submap.allFinite() || !point.normal_submap.allFinite() ||
        !std::isfinite(point.weight) || point.weight <= 0.0 ||
        std::abs(point.normal_submap.norm() - 1.0) > 1.0e-5) {
      return makeError(SealSpoolErrorCode::InvalidSeal,
                       "registration proxy contains a non-finite or invalid weighted normal",
                       std::nullopt, identity.ref.id);
    }
  }
  for (const auto& entry : record.lidar_place_index) {
    if (!entry.sweep.valid() || !entry.state.valid() || !validPose(entry.T_submap_lidar) ||
        !entry.lineage.id.valid() ||
        !validPersistedLineage(entry.lineage)) {
      return makeError(SealSpoolErrorCode::InvalidSeal,
                       "LiDAR place entry identity, pose, or lineage is invalid", std::nullopt,
                       identity.ref.id);
    }
    if (const auto payload_error = validatePayload(entry.payload, identity.ref.id)) {
      return payload_error;
    }
  }
  for (const auto& entry : record.visual_place_index) {
    if (!entry.frame.valid() || !entry.camera.valid() || !entry.state.valid() ||
        !validPose(entry.T_submap_camera) || !entry.lineage.id.valid() ||
        !validPersistedLineage(entry.lineage)) {
      return makeError(SealSpoolErrorCode::InvalidSeal,
                       "visual place entry identity, pose, or lineage is invalid", std::nullopt,
                       identity.ref.id);
    }
    if (const auto payload_error = validatePayload(entry.payload, identity.ref.id)) {
      return payload_error;
    }
  }
  if (record.incoming_adjacent) {
    const auto& adjacent = *record.incoming_adjacent;
    if (!sameSubmapFrame(adjacent.global_append.submap, record.submap) ||
        adjacent.global_append.constraint.to != record.submap.ref ||
        adjacent.global_append.constraint.from == record.submap.ref ||
        core::validateSubmapRef(adjacent.global_append.constraint.from) !=
            core::SubmapRefValidationError::None ||
        !validPose(adjacent.global_append.constraint.T_from_to) ||
        !validInformation(adjacent.global_append.constraint.information) ||
        !validCovariance(adjacent.relative_covariance) || !adjacent.lineage.id.valid() ||
        !validPersistedLineage(adjacent.lineage) ||
        !adjacent.final_local_revision.valid()) {
      return makeError(SealSpoolErrorCode::InvalidSeal,
                       "incoming adjacent constraint endpoints, geometry, or lineage is invalid",
                       std::nullopt, identity.ref.id);
    }
  }
  return std::nullopt;
}

[[nodiscard]] Bytes frameBytes(std::span<const std::byte> magic,
                               std::span<const std::byte> payload) {
  return persistence_internal::frameBytes(magic, kFormatVersion, payload);
}

struct DecodedFrame {
  core::ContentHash checksum{};
  std::span<const std::byte> payload;
};

[[nodiscard]] core::Result<DecodedFrame, SealSpoolError> decodeFrame(
    std::span<const std::byte> bytes, std::span<const std::byte> expected_magic,
    std::uint64_t maximum_payload_bytes, const std::filesystem::path& path) {
  const auto decoded = persistence_internal::decodeFrame(
      bytes, expected_magic, kFormatVersion, maximum_payload_bytes, path);
  if (!decoded) {
    const auto code = decoded.error().code == persistence_internal::ErrorCode::ChecksumMismatch
                          ? SealSpoolErrorCode::ChecksumMismatch
                          : SealSpoolErrorCode::RecoveryCorruption;
    return core::Result<DecodedFrame, SealSpoolError>::failure(
        makeError(code, decoded.error().detail));
  }
  return core::Result<DecodedFrame, SealSpoolError>::success(
      DecodedFrame{decoded.value().checksum, decoded.value().payload});
}

[[nodiscard]] std::string hashHex(const core::ContentHash& hash) {
  return persistence_internal::hashHex(hash);
}

[[nodiscard]] std::string sequenceName(OutboxSequence sequence, std::string_view extension) {
  std::array<char, 32> buffer{};
  const int length = std::snprintf(buffer.data(), buffer.size(), "%020llu",
                                   static_cast<unsigned long long>(sequence.value()));
  return std::string(buffer.data(), static_cast<std::size_t>(length)) + std::string(extension);
}

[[nodiscard]] std::optional<OutboxSequence> parseSequence(const std::filesystem::path& path,
                                                          std::string_view extension) {
  const std::string filename = path.filename().string();
  if (filename.size() != 20U + extension.size() ||
      filename.compare(20U, extension.size(), extension) != 0 ||
      !std::all_of(filename.begin(), filename.begin() + 20,
                   [](char value) { return value >= '0' && value <= '9'; })) {
    return std::nullopt;
  }
  std::uint64_t value = 0U;
  for (std::size_t index = 0U; index < 20U; ++index) {
    const std::uint64_t digit = static_cast<std::uint64_t>(filename[index] - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return std::nullopt;
    }
    value = value * 10U + digit;
  }
  if (value == 0U) {
    return std::nullopt;
  }
  return OutboxSequence(value);
}

struct OutboxEntry {
  OutboxSequence sequence;
  core::SparseSubmapSealIdentity identity;
  core::ContentHash seal_checksum{};
  core::ContentHash record_checksum{};
  std::uint64_t record_bytes{};
};

void writeOutboxPayload(Writer& writer, const OutboxEntry& entry) {
  writer.id(entry.sequence);
  writeIdentity(writer, entry.identity);
  writer.hash(entry.seal_checksum);
  writer.hash(entry.record_checksum);
  writer.u64(entry.record_bytes);
}

[[nodiscard]] OutboxEntry readOutboxPayload(Reader& reader) {
  OutboxEntry entry;
  entry.sequence = reader.id<OutboxSequence>();
  entry.identity = readIdentity(reader);
  entry.seal_checksum = reader.hash();
  entry.record_checksum = reader.hash();
  entry.record_bytes = reader.u64();
  return entry;
}

[[nodiscard]] bool sameIdentityKey(const core::SparseSubmapSealIdentity& lhs,
                                   const core::SparseSubmapSealIdentity& rhs) noexcept {
  return core::sparseSubmapIdentityKey(lhs.ref) == core::sparseSubmapIdentityKey(rhs.ref);
}

[[nodiscard]] bool sameEntry(const OutboxEntry& lhs, const OutboxEntry& rhs) noexcept {
  return lhs.sequence == rhs.sequence && lhs.identity == rhs.identity &&
         lhs.seal_checksum == rhs.seal_checksum && lhs.record_checksum == rhs.record_checksum &&
         lhs.record_bytes == rhs.record_bytes;
}

using IdentityKey = core::SparseSubmapIdentityKey;

[[nodiscard]] IdentityKey identityKey(const core::SparseSubmapSealIdentity& identity) noexcept {
  return core::sparseSubmapIdentityKey(identity.ref);
}

[[nodiscard]] core::Result<Bytes, SealSpoolError> readFile(const std::filesystem::path& path,
                                                           std::uint64_t maximum_bytes) {
  const auto result = persistence_internal::readFile(path, maximum_bytes);
  if (!result) {
    const auto code = (result.error().code == persistence_internal::ErrorCode::InvalidRecord ||
                       result.error().code == persistence_internal::ErrorCode::SizeMismatch)
                          ? SealSpoolErrorCode::RecoveryCorruption
                          : SealSpoolErrorCode::IoFailure;
    return core::Result<Bytes, SealSpoolError>::failure(makeError(code, result.error().detail));
  }
  return core::Result<Bytes, SealSpoolError>::success(result.value());
}

[[nodiscard]] std::optional<SealSpoolError> syncDirectory(const std::filesystem::path& path) {
  const auto result = persistence_internal::syncDirectory(path);
  return result ? std::optional(makeError(SealSpoolErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> ensureDirectory(const std::filesystem::path& path) {
  const auto result = persistence_internal::ensureDirectory(path, "spool directory");
  return result ? std::optional(makeError(SealSpoolErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> durableWriteNoReplace(
    const std::filesystem::path& destination, std::span<const std::byte> bytes,
    std::uint64_t* temporary_counter, std::int64_t* elapsed_us) {
  const auto result = persistence_internal::durableWriteNoReplace(destination, bytes,
                                                                   temporary_counter);
  if (!result) {
    SealSpoolErrorCode code = SealSpoolErrorCode::IoFailure;
    if (result.error().code == persistence_internal::ErrorCode::Conflict) {
      code = SealSpoolErrorCode::IdentityConflict;
    } else if (result.error().code == persistence_internal::ErrorCode::AmbiguousIntegrity) {
      code = SealSpoolErrorCode::IntegrityFailure;
    }
    return makeError(code, result.error().detail);
  }
  if (elapsed_us != nullptr) {
    *elapsed_us = result.value().elapsed_us;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> cleanTemporaryFiles(
    const std::filesystem::path& directory, std::size_t* recovered) {
  const auto result = persistence_internal::cleanTemporaryFiles(directory, recovered);
  return result ? std::optional(makeError(SealSpoolErrorCode::IoFailure, result->detail))
                : std::nullopt;
}

[[nodiscard]] Bytes serializeSeal(const SparseSubmapSealRecord& seal,
                                  core::ContentHash* payload_checksum) {
  Writer payload_writer;
  writeSealPayload(payload_writer, seal);
  Bytes payload = std::move(payload_writer).take();
  *payload_checksum = hashBytes(payload);
  return frameBytes(kSealMagic, payload);
}

[[nodiscard]] Bytes serializeEntry(const OutboxEntry& entry, std::span<const std::byte> magic) {
  Writer payload_writer;
  writeOutboxPayload(payload_writer, entry);
  const Bytes payload = std::move(payload_writer).take();
  return frameBytes(magic, payload);
}

[[nodiscard]] std::vector<std::filesystem::path> filesWithExtension(
    const std::filesystem::path& directory, std::string_view extension, std::size_t* scanned,
    std::optional<SealSpoolError>* output_error) {
  std::vector<std::filesystem::path> files;
  std::error_code filesystem_error;
  for (std::filesystem::directory_iterator iterator(directory, filesystem_error), end;
       iterator != end && !filesystem_error; iterator.increment(filesystem_error)) {
    ++*scanned;
    const auto status = iterator->symlink_status(filesystem_error);
    if (filesystem_error) {
      break;
    }
    if (!std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status) ||
        iterator->path().extension() != extension) {
      *output_error =
          makeError(SealSpoolErrorCode::RecoveryCorruption,
                    "unexpected durable spool directory entry: " + iterator->path().string());
      return {};
    }
    files.push_back(iterator->path());
  }
  if (filesystem_error) {
    *output_error = makeError(
        SealSpoolErrorCode::IoFailure,
        "durable spool scan failed in " + directory.string() + ": " + filesystem_error.message());
    return {};
  }
  std::sort(files.begin(), files.end());
  return files;
}

}  // namespace

struct SealSpool::Impl {
  struct StoredSeal {
    SparseSubmapSeal seal;
    core::ContentHash record_checksum{};
    std::uint64_t record_bytes{};
    std::filesystem::path path;
  };

  explicit Impl(SealSpoolConfig input_config)
      : config(std::move(input_config)),
        seals_directory(config.root_directory / "seals"),
        outbox_directory(config.root_directory / "outbox"),
        acknowledgements_directory(config.root_directory / "acks") {}

  [[nodiscard]] std::uint64_t reservedAcknowledgementBytes() const {
    std::uint64_t total = 0U;
    for (const auto& [sequence, entry] : entries) {
      if (!acknowledged.contains(sequence)) {
        total += serializeEntry(entry, kAckMagic).size();
      }
    }
    return total;
  }

  [[nodiscard]] SealSpoolStatus currentStatus() const noexcept {
    SealSpoolStatus output;
    output.head = OutboxSequence(head);
    output.durable_seals = seals.size();
    output.outbox_entries = entries.size();
    output.acknowledged_entries = acknowledged.size();
    output.unacknowledged_entries = entries.size() - acknowledged.size();
    output.durable_bytes = durable_bytes;
    output.reserved_acknowledgement_bytes = reservedAcknowledgementBytes();
    output.maximum_bytes = config.maximum_bytes;
    output.recovered_temporary_files = recovered_temporary_files;
    output.recovery_scanned_files = recovery_scanned_files;
    output.ambiguous_destination = ambiguous_destination;
    for (const auto& [sequence, unused] : entries) {
      (void)unused;
      if (!acknowledged.contains(sequence)) {
        output.oldest_unacknowledged = OutboxSequence(sequence);
        break;
      }
    }
    if (integrity_failure) {
      output.state = SealSpoolState::ReadOnlyIntegrityFailure;
    } else if (io_failure) {
      output.state = SealSpoolState::ReadOnlyIoFailure;
    } else if (capacity_failure || seals.size() >= config.maximum_seals ||
        entries.size() >= config.maximum_outbox_entries ||
        durable_bytes + output.reserved_acknowledgement_bytes >= config.maximum_bytes) {
      output.state = SealSpoolState::DegradedStorageCapacity;
    }
    return output;
  }

  [[nodiscard]] bool writePermitted(const std::filesystem::path& path,
                                    std::span<const std::byte> bytes) const noexcept {
    if (io_failure) {
      return false;
    }
    return !integrity_failure ||
           (ambiguous_destination && ambiguous_checksum && *ambiguous_destination == path &&
            *ambiguous_checksum == hashBytes(bytes));
  }

  void latchWriteFailure(const SealSpoolError& failure, const std::filesystem::path& path,
                         std::span<const std::byte> bytes) {
    if (failure.code == SealSpoolErrorCode::IntegrityFailure) {
      integrity_failure = true;
      ambiguous_destination = path;
      ambiguous_checksum = hashBytes(bytes);
    } else if (failure.code == SealSpoolErrorCode::IoFailure) {
      io_failure = true;
    }
  }

  void resolveAmbiguousWrite(const std::filesystem::path& path,
                             std::span<const std::byte> bytes) noexcept {
    if (integrity_failure && ambiguous_destination && ambiguous_checksum &&
        *ambiguous_destination == path && *ambiguous_checksum == hashBytes(bytes)) {
      integrity_failure = false;
      ambiguous_destination.reset();
      ambiguous_checksum.reset();
    }
  }

  SealSpoolConfig config;
  std::filesystem::path seals_directory;
  std::filesystem::path outbox_directory;
  std::filesystem::path acknowledgements_directory;
  std::map<IdentityKey, StoredSeal> seals;
  std::map<core::ContentHash, IdentityKey> seal_by_record_checksum;
  std::map<std::uint64_t, OutboxEntry> entries;
  std::map<IdentityKey, std::uint64_t> sequence_by_identity;
  std::set<std::uint64_t> acknowledged;
  std::uint64_t head{};
  std::uint64_t durable_bytes{};
  std::uint64_t temporary_counter{};
  std::size_t recovered_temporary_files{};
  std::size_t recovery_scanned_files{};
  bool capacity_failure{false};
  bool io_failure{false};
  bool integrity_failure{false};
  std::optional<std::filesystem::path> ambiguous_destination;
  std::optional<core::ContentHash> ambiguous_checksum;
};

namespace {
[[nodiscard]] std::optional<SealSpoolError> validateConfig(const SealSpoolConfig& config);
[[nodiscard]] std::optional<SealSpoolError> recoverSealFiles(SealSpool::Impl& impl);
[[nodiscard]] std::optional<SealSpoolError> recoverOutboxFiles(SealSpool::Impl& impl);
[[nodiscard]] std::optional<SealSpoolError> recoverAcknowledgements(SealSpool::Impl& impl);
}  // namespace

core::Result<std::unique_ptr<SealSpool>, SealSpoolError> SealSpool::open(SealSpoolConfig config) {
  using Result = core::Result<std::unique_ptr<SealSpool>, SealSpoolError>;
  if (const auto config_error = validateConfig(config)) {
    return Result::failure(*config_error);
  }
  auto impl = std::make_unique<Impl>(std::move(config));
  if (const auto root_error = ensureDirectory(impl->config.root_directory)) {
    return Result::failure(*root_error);
  }
  if (const auto seal_error = ensureDirectory(impl->seals_directory)) {
    return Result::failure(*seal_error);
  }
  if (const auto outbox_error = ensureDirectory(impl->outbox_directory)) {
    return Result::failure(*outbox_error);
  }
  if (const auto acknowledgement_error = ensureDirectory(impl->acknowledgements_directory)) {
    return Result::failure(*acknowledgement_error);
  }
  if (const auto root_sync_error = syncDirectory(impl->config.root_directory)) {
    return Result::failure(*root_sync_error);
  }
  for (const auto& directory :
       {impl->seals_directory, impl->outbox_directory, impl->acknowledgements_directory}) {
    if (const auto cleanup_error =
            cleanTemporaryFiles(directory, &impl->recovered_temporary_files)) {
      return Result::failure(*cleanup_error);
    }
  }
  if (const auto seal_recovery = recoverSealFiles(*impl)) {
    return Result::failure(*seal_recovery);
  }
  if (const auto outbox_recovery = recoverOutboxFiles(*impl)) {
    return Result::failure(*outbox_recovery);
  }
  if (const auto ack_recovery = recoverAcknowledgements(*impl)) {
    return Result::failure(*ack_recovery);
  }
  if (impl->seals.size() > impl->config.maximum_seals ||
      impl->entries.size() > impl->config.maximum_outbox_entries ||
      impl->durable_bytes + impl->reservedAcknowledgementBytes() > impl->config.maximum_bytes) {
    impl->capacity_failure = true;
  }
  return Result::success(std::unique_ptr<SealSpool>(new SealSpool(std::move(impl))));
}

SealSpool::SealSpool(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SealSpool::~SealSpool() = default;
SealSpool::SealSpool(SealSpool&&) noexcept = default;
SealSpool& SealSpool::operator=(SealSpool&&) noexcept = default;

core::Result<SealEnqueueReport, SealSpoolError> SealSpool::enqueue(SparseSubmapSeal seal) {
  using Result = core::Result<SealEnqueueReport, SealSpoolError>;
  if (const auto validation_error = validateSeal(seal, impl_->config)) {
    return Result::failure(*validation_error);
  }

  core::ContentHash record_checksum{};
  const Bytes seal_frame = serializeSeal(*seal, &record_checksum);
  if (seal_frame.size() > impl_->config.maximum_seal_record_bytes) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(SealSpoolErrorCode::CapacityExceeded,
                                     "serialized seal exceeds its per-record byte cap",
                                     std::nullopt, seal->identity.ref.id));
  }
  const IdentityKey key = identityKey(seal->identity);
  auto stored = impl_->seals.find(key);
  bool new_seal = stored == impl_->seals.end();
  if (!new_seal) {
    if (stored->second.seal->identity != seal->identity ||
        stored->second.record_checksum != record_checksum ||
        stored->second.record_bytes != seal_frame.size()) {
      return Result::failure(
          makeError(SealSpoolErrorCode::IdentityConflict,
                    "the same sparse-seal identity was enqueued with different canonical content",
                    std::nullopt, seal->identity.ref.id));
    }
    const auto sequence = impl_->sequence_by_identity.find(key);
    if (sequence != impl_->sequence_by_identity.end()) {
      SealEnqueueReport report;
      report.sequence = OutboxSequence(sequence->second);
      report.identity = seal->identity;
      report.serialized_record_checksum = record_checksum;
      report.serialized_record_bytes = seal_frame.size();
      report.idempotent = true;
      return Result::success(std::move(report));
    }
  }

  if (impl_->head == std::numeric_limits<std::uint64_t>::max()) {
    return Result::failure(makeError(SealSpoolErrorCode::CapacityExceeded,
                                     "outbox sequence space is exhausted", std::nullopt,
                                     seal->identity.ref.id));
  }
  const OutboxSequence sequence(impl_->head + 1U);
  const OutboxEntry entry{sequence, seal->identity, seal->identity.seal_checksum,
                          record_checksum, static_cast<std::uint64_t>(seal_frame.size())};
  const Bytes entry_frame = serializeEntry(entry, kOutboxMagic);
  const Bytes acknowledgement_frame = serializeEntry(entry, kAckMagic);
  const std::uint64_t added_seal_bytes = new_seal ? seal_frame.size() : 0U;
  const std::uint64_t projected_durable =
      impl_->durable_bytes + added_seal_bytes + entry_frame.size();
  const std::uint64_t projected_reserved =
      impl_->reservedAcknowledgementBytes() + acknowledgement_frame.size();
  if ((new_seal && impl_->seals.size() >= impl_->config.maximum_seals) ||
      impl_->entries.size() >= impl_->config.maximum_outbox_entries ||
      projected_durable > impl_->config.maximum_bytes ||
      projected_reserved > impl_->config.maximum_bytes - projected_durable) {
    impl_->capacity_failure = true;
    return Result::failure(makeError(
        SealSpoolErrorCode::CapacityExceeded,
        "enqueue would exceed a seal, outbox, or byte hard cap; no unacknowledged seal was deleted",
        sequence, seal->identity.ref.id));
  }

  SealEnqueueReport report;
  report.sequence = sequence;
  report.identity = seal->identity;
  report.serialized_record_checksum = record_checksum;
  report.serialized_record_bytes = seal_frame.size();
  if (new_seal) {
    const std::filesystem::path seal_path =
        impl_->seals_directory / (hashHex(record_checksum) + ".seal");
    if (!impl_->writePermitted(seal_path, seal_frame)) {
      return Result::failure(makeError(
          impl_->io_failure ? SealSpoolErrorCode::IoFailure
                            : SealSpoolErrorCode::IntegrityFailure,
          "spool is read-only after an unresolved durable write", sequence,
          seal->identity.ref.id));
    }
    if (const auto write_error =
            durableWriteNoReplace(seal_path, seal_frame, &impl_->temporary_counter,
                                  &report.timing.seal_write_and_sync_us)) {
      impl_->latchWriteFailure(*write_error, seal_path, seal_frame);
      return Result::failure(*write_error);
    }
    impl_->resolveAmbiguousWrite(seal_path, seal_frame);
    const auto inserted = impl_->seals.emplace(
        key, Impl::StoredSeal{seal, record_checksum, static_cast<std::uint64_t>(seal_frame.size()),
                              seal_path});
    impl_->seal_by_record_checksum.emplace(record_checksum, key);
    impl_->durable_bytes += seal_frame.size();
    stored = inserted.first;
  }

  const std::filesystem::path entry_path =
      impl_->outbox_directory / sequenceName(sequence, ".entry");
  if (!impl_->writePermitted(entry_path, entry_frame)) {
    return Result::failure(makeError(
        impl_->io_failure ? SealSpoolErrorCode::IoFailure : SealSpoolErrorCode::IntegrityFailure,
        "spool is read-only after an unresolved durable write", sequence,
        seal->identity.ref.id));
  }
  if (const auto write_error =
          durableWriteNoReplace(entry_path, entry_frame, &impl_->temporary_counter,
                                &report.timing.outbox_write_and_sync_us)) {
    // A fully durable orphan seal is intentionally retained. Retrying the
    // same enqueue appends its missing outbox entry without rewriting it.
    impl_->latchWriteFailure(*write_error, entry_path, entry_frame);
    return Result::failure(*write_error);
  }
  impl_->resolveAmbiguousWrite(entry_path, entry_frame);
  impl_->entries.emplace(sequence.value(), entry);
  impl_->sequence_by_identity.emplace(key, sequence.value());
  impl_->head = sequence.value();
  impl_->durable_bytes += entry_frame.size();
  return Result::success(std::move(report));
}

core::Result<std::vector<SealReplayRecord>, SealSpoolError> SealSpool::replaySince(
    OutboxSequence after, std::size_t maximum_records) const {
  using Result = core::Result<std::vector<SealReplayRecord>, SealSpoolError>;
  if (!after.valid() || maximum_records == 0U ||
      maximum_records > impl_->config.maximum_replay_records) {
    return Result::failure(makeError(SealSpoolErrorCode::ReplayLimitExceeded,
                                     "replay sequence or requested record bound is invalid",
                                     after));
  }
  std::vector<SealReplayRecord> output;
  output.reserve(maximum_records);
  for (auto iterator = impl_->entries.upper_bound(after.value());
       iterator != impl_->entries.end() && output.size() < maximum_records; ++iterator) {
    if (impl_->acknowledged.contains(iterator->first)) {
      continue;
    }
    const OutboxEntry& entry = iterator->second;
    const auto stored = impl_->seals.find(identityKey(entry.identity));
    if (stored == impl_->seals.end()) {
      return Result::failure(makeError(SealSpoolErrorCode::RecoveryCorruption,
                                       "live outbox entry lost its durable seal", entry.sequence,
                                       entry.identity.ref.id));
    }
    output.push_back(SealReplayRecord{entry.sequence, entry.identity, entry.seal_checksum,
                                      entry.record_checksum, entry.record_bytes,
                                      stored->second.seal});
  }
  return Result::success(std::move(output));
}

core::Result<bool, SealSpoolError> SealSpool::acknowledge(
    const SealAcknowledgement& acknowledgement, SealSpoolTiming* timing) {
  using Result = core::Result<bool, SealSpoolError>;
  if (!acknowledgement.sequence.valid() || acknowledgement.sequence.value() == 0U) {
    return Result::failure(makeError(SealSpoolErrorCode::SequenceNotFound,
                                     "acknowledgement sequence is invalid",
                                     acknowledgement.sequence));
  }
  const auto match = impl_->entries.find(acknowledgement.sequence.value());
  if (match == impl_->entries.end()) {
    return Result::failure(makeError(SealSpoolErrorCode::SequenceNotFound,
                                     "acknowledgement sequence is not in the durable outbox",
                                     acknowledgement.sequence));
  }
  const OutboxEntry& entry = match->second;
  if (acknowledgement.identity != entry.identity ||
      acknowledgement.seal_checksum != entry.seal_checksum ||
      acknowledgement.seal_checksum != acknowledgement.identity.seal_checksum) {
    return Result::failure(makeError(
        SealSpoolErrorCode::AcknowledgementConflict,
        "acknowledgement identity or seal checksum conflicts with the durable outbox entry",
        acknowledgement.sequence, entry.identity.ref.id));
  }
  if (impl_->acknowledged.contains(acknowledgement.sequence.value())) {
    return Result::success(false);
  }
  const Bytes acknowledgement_frame = serializeEntry(entry, kAckMagic);
  if (impl_->durable_bytes + acknowledgement_frame.size() > impl_->config.maximum_bytes) {
    return Result::failure(makeError(
        SealSpoolErrorCode::IoFailure,
        "reserved acknowledgement bytes are unexpectedly unavailable; spool integrity is suspect",
        acknowledgement.sequence, entry.identity.ref.id));
  }
  std::int64_t elapsed_us = 0;
  const std::filesystem::path acknowledgement_path =
      impl_->acknowledgements_directory / sequenceName(entry.sequence, ".ack");
  if (!impl_->writePermitted(acknowledgement_path, acknowledgement_frame)) {
    return Result::failure(makeError(
        impl_->io_failure ? SealSpoolErrorCode::IoFailure : SealSpoolErrorCode::IntegrityFailure,
        "spool is read-only after an unresolved durable write", acknowledgement.sequence,
        entry.identity.ref.id));
  }
  if (const auto write_error = durableWriteNoReplace(acknowledgement_path, acknowledgement_frame,
                                                     &impl_->temporary_counter, &elapsed_us)) {
    impl_->latchWriteFailure(*write_error, acknowledgement_path, acknowledgement_frame);
    return Result::failure(*write_error);
  }
  impl_->resolveAmbiguousWrite(acknowledgement_path, acknowledgement_frame);
  impl_->acknowledged.insert(acknowledgement.sequence.value());
  impl_->durable_bytes += acknowledgement_frame.size();
  if (timing != nullptr) {
    timing->acknowledgement_write_and_sync_us = elapsed_us;
  }
  return Result::success(true);
}

SealSpoolStatus SealSpool::status() const noexcept {
  return impl_->currentStatus();
}

namespace {

[[nodiscard]] std::optional<SealSpoolError> validateConfig(const SealSpoolConfig& config) {
  if (!sha256SelfTest()) {
    return makeError(SealSpoolErrorCode::SerializationFailure,
                     "SHA-256 implementation failed its startup known-answer test");
  }
  if (config.root_directory.empty() || config.maximum_bytes <= kFrameOverhead ||
      config.maximum_seal_record_bytes <= kFrameOverhead ||
      config.maximum_seal_record_bytes > config.maximum_bytes || config.maximum_seals == 0U ||
      config.maximum_outbox_entries == 0U || config.maximum_replay_records == 0U ||
      config.maximum_vector_elements == 0U) {
    return makeError(SealSpoolErrorCode::InvalidConfiguration,
                     "seal spool path or hard capacity bounds are invalid");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> recoverSealFiles(SealSpool::Impl& impl) {
  std::optional<SealSpoolError> scan_error;
  const auto paths =
      filesWithExtension(impl.seals_directory, ".seal", &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  for (const auto& path : paths) {
    const auto bytes = readFile(path, impl.config.maximum_seal_record_bytes);
    if (!bytes) {
      return bytes.error();
    }
    const auto frame = decodeFrame(bytes.value(), kSealMagic,
                                   impl.config.maximum_seal_record_bytes - kFrameOverhead, path);
    if (!frame) {
      return frame.error();
    }
    if (path.filename().string() != hashHex(frame.value().checksum) + ".seal") {
      return makeError(
          SealSpoolErrorCode::RecoveryCorruption,
          "seal filename does not equal its canonical payload checksum: " + path.string());
    }
    Reader reader(frame.value().payload, impl.config.maximum_vector_elements);
    SparseSubmapSeal seal = readSealPayload(reader);
    if (!reader.complete()) {
      return makeError(
          SealSpoolErrorCode::RecoveryCorruption,
          "seal payload is truncated, overlong, or contains an invalid enum: " + path.string());
    }
    if (const auto validation_error = validateSeal(seal, impl.config)) {
      SealSpoolError recovered = *validation_error;
      recovered.code = SealSpoolErrorCode::RecoveryCorruption;
      recovered.detail = "recovered seal validation failed: " + recovered.detail;
      return recovered;
    }
    core::ContentHash canonical_checksum{};
    const Bytes canonical = serializeSeal(*seal, &canonical_checksum);
    if (canonical_checksum != frame.value().checksum || canonical != bytes.value()) {
      return makeError(
          SealSpoolErrorCode::RecoveryCorruption,
          "seal does not round-trip through canonical serialization: " + path.string());
    }
    const IdentityKey key = identityKey(seal->identity);
    if (impl.seals.contains(key) || impl.seal_by_record_checksum.contains(canonical_checksum)) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "duplicate seal identity or canonical checksum during recovery",
                       std::nullopt, seal->identity.ref.id);
    }
    impl.seals.emplace(
        key, SealSpool::Impl::StoredSeal{seal, canonical_checksum,
                                         static_cast<std::uint64_t>(canonical.size()), path});
    impl.seal_by_record_checksum.emplace(canonical_checksum, key);
    impl.durable_bytes += canonical.size();
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> recoverOutboxFiles(SealSpool::Impl& impl) {
  std::optional<SealSpoolError> scan_error;
  const auto paths = filesWithExtension(impl.outbox_directory, ".entry",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  for (const auto& path : paths) {
    const auto filename_sequence = parseSequence(path, ".entry");
    if (!filename_sequence) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "outbox filename is not a canonical sequence: " + path.string());
    }
    const auto bytes = readFile(path, 4096U);
    if (!bytes) {
      return bytes.error();
    }
    const auto frame = decodeFrame(bytes.value(), kOutboxMagic, 4096U - kFrameOverhead, path);
    if (!frame) {
      return frame.error();
    }
    Reader reader(frame.value().payload, 64U);
    const OutboxEntry entry = readOutboxPayload(reader);
    if (!reader.complete() || entry.sequence != *filename_sequence || !entry.sequence.valid() ||
        entry.sequence.value() == 0U || entry.seal_checksum != entry.identity.seal_checksum ||
        zeroHash(entry.record_checksum)) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "outbox entry identity, sequence, or checksum is invalid: " + path.string());
    }
    const auto checksum_match = impl.seal_by_record_checksum.find(entry.record_checksum);
    if (checksum_match == impl.seal_by_record_checksum.end()) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "outbox entry references a missing durable seal", entry.sequence,
                       entry.identity.ref.id);
    }
    const auto seal_match = impl.seals.find(checksum_match->second);
    if (seal_match == impl.seals.end() ||
        !sameIdentityKey(entry.identity, seal_match->second.seal->identity) ||
        entry.identity != seal_match->second.seal->identity ||
        entry.record_bytes != seal_match->second.record_bytes) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "outbox entry conflicts with its referenced durable seal", entry.sequence,
                       entry.identity.ref.id);
    }
    if (!impl.entries.emplace(entry.sequence.value(), entry).second ||
        !impl.sequence_by_identity.emplace(identityKey(entry.identity), entry.sequence.value())
             .second) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "duplicate outbox sequence or seal identity", entry.sequence,
                       entry.identity.ref.id);
    }
    impl.head = std::max(impl.head, entry.sequence.value());
    impl.durable_bytes += bytes.value().size();
  }
  std::uint64_t expected = 1U;
  for (const auto& [sequence, unused] : impl.entries) {
    (void)unused;
    if (sequence != expected++) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "outbox sequences are not contiguous from one", OutboxSequence(sequence));
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<SealSpoolError> recoverAcknowledgements(SealSpool::Impl& impl) {
  std::optional<SealSpoolError> scan_error;
  const auto paths = filesWithExtension(impl.acknowledgements_directory, ".ack",
                                        &impl.recovery_scanned_files, &scan_error);
  if (scan_error) {
    return scan_error;
  }
  for (const auto& path : paths) {
    const auto filename_sequence = parseSequence(path, ".ack");
    if (!filename_sequence) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "acknowledgement filename is not a canonical sequence: " + path.string());
    }
    const auto bytes = readFile(path, 4096U);
    if (!bytes) {
      return bytes.error();
    }
    const auto frame = decodeFrame(bytes.value(), kAckMagic, 4096U - kFrameOverhead, path);
    if (!frame) {
      return frame.error();
    }
    Reader reader(frame.value().payload, 64U);
    const OutboxEntry acknowledgement = readOutboxPayload(reader);
    const auto entry = impl.entries.find(filename_sequence->value());
    if (!reader.complete() || acknowledgement.sequence != *filename_sequence ||
        entry == impl.entries.end() || !sameEntry(acknowledgement, entry->second) ||
        !impl.acknowledged.insert(filename_sequence->value()).second) {
      return makeError(SealSpoolErrorCode::RecoveryCorruption,
                       "acknowledgement conflicts with its durable outbox entry",
                       *filename_sequence);
    }
    impl.durable_bytes += bytes.value().size();
  }
  return std::nullopt;
}

}  // namespace

}  // namespace meridian::global
