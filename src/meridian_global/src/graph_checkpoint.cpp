#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "graph_checkpoint_internal.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {
namespace {

constexpr std::string_view kCheckpointChecksumDomain =
    "meridian.global.checksum.global_graph_checkpoint";
constexpr std::string_view kCheckpointWireDomain = "meridian.global.wire.global_graph_checkpoint";
constexpr std::string_view kConfigurationChecksumDomain =
    "meridian.global.checksum.global_graph_configuration";

[[nodiscard]] GlobalGraphCheckpointError checkpointError(GlobalGraphCheckpointErrorCode code,
                                                         std::string detail) {
  return GlobalGraphCheckpointError{code, std::move(detail), std::nullopt, std::nullopt};
}

[[nodiscard]] bool validLimits(const GlobalGraphCheckpointLimits& limits) noexcept {
  return limits.maximum_wire_bytes > 0U && limits.maximum_boundaries > 0U &&
         limits.maximum_chart_placements > 0U && limits.maximum_adjacent_factors > 0U &&
         limits.maximum_gnss_factors > 0U && limits.maximum_loop_factors > 0U &&
         limits.maximum_factor_rows > 0U && limits.maximum_factor_coefficients > 0U &&
         limits.maximum_nested_collection_entries > 0U;
}

class Writer {
public:
  using CreateResult = core::Result<Writer, GlobalGraphCheckpointError>;

  [[nodiscard]] static CreateResult create(std::string_view domain, std::uint32_t schema,
                                           GlobalGraphCheckpointLimits limits) {
    if (!validLimits(limits)) {
      return CreateResult::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidLimits,
                                                   "checkpoint limits are zero or invalid"));
    }
    auto encoder = core::CanonicalEncoder::create(domain, schema, limits.maximum_wire_bytes);
    if (!encoder) {
      GlobalGraphCheckpointError error =
          checkpointError(GlobalGraphCheckpointErrorCode::EncodingFailure,
                          "canonical checkpoint encoder could not be created");
      error.encoding_error = encoder.error();
      return CreateResult::failure(std::move(error));
    }
    return CreateResult::success(
        Writer(limits.maximum_nested_collection_entries, std::move(encoder).value()));
  }

  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  Writer(Writer&&) noexcept = default;
  Writer& operator=(Writer&&) noexcept = default;

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  void u8(std::uint8_t value) { apply(encoder_.writeU8(value)); }
  void u32(std::uint32_t value) { apply(encoder_.writeU32(value)); }
  void u64(std::uint64_t value) { apply(encoder_.writeU64(value)); }
  void i64(std::int64_t value) { apply(encoder_.writeI64(value)); }
  void boolean(bool value) { apply(encoder_.writeBool(value)); }
  void floating(double value) { apply(encoder_.writeDouble(value)); }
  void hash(const core::ContentHash& value) { apply(encoder_.writeHash(value)); }
  void pose(const core::Pose3d& value) {
    // The public canonical primitive pins the semantic 4x4 pose. Preserve the
    // exact unit-quaternion chart as an adjunct so decoding does not introduce
    // matrix->quaternion round-off that would invalidate nested core digests.
    apply(encoder_.writePose3(value));
    Eigen::Vector4d coefficients = value.so3().unit_quaternion().coeffs();
    // q and -q represent the same rotation. Select one bit-stable sign so the
    // checkpoint remains canonical over solver-neutral pose semantics.
    const auto first_nonzero = std::find_if(coefficients.data(), coefficients.data() + 4,
                                            [](double component) { return component != 0.0; });
    if (first_nonzero != coefficients.data() + 4 && *first_nonzero < 0.0) {
      coefficients = -coefficients;
    }
    apply(encoder_.writeEigenVector(coefficients));
  }

  template <typename Derived>
  void vector(const Eigen::MatrixBase<Derived>& value) {
    apply(encoder_.writeEigenVector(value));
  }

  template <typename Derived>
  void matrix(const Eigen::MatrixBase<Derived>& value) {
    apply(encoder_.writeEigenMatrix(value));
  }

  template <typename Id>
  void id(Id value) {
    u64(value.value());
  }

  void optional(bool present) { boolean(present); }

  void count(std::size_t value) {
    if (!ok()) {
      return;
    }
    if (value > remaining_entries_) {
      error_ = checkpointError(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                               "checkpoint nested collection budget is exceeded");
      return;
    }
    remaining_entries_ -= value;
    u64(static_cast<std::uint64_t>(value));
  }

  [[nodiscard]] core::Result<core::CanonicalByteSequence, GlobalGraphCheckpointError> finish() {
    using Result = core::Result<core::CanonicalByteSequence, GlobalGraphCheckpointError>;
    if (error_) {
      return Result::failure(*error_);
    }
    auto finished = encoder_.finish();
    if (!finished) {
      GlobalGraphCheckpointError error =
          checkpointError(GlobalGraphCheckpointErrorCode::EncodingFailure,
                          "canonical checkpoint encoding could not be finalized");
      error.encoding_error = finished.error();
      return Result::failure(std::move(error));
    }
    return Result::success(std::move(finished).value());
  }

private:
  Writer(std::size_t entries, core::CanonicalEncoder encoder)
      : remaining_entries_(entries), encoder_(std::move(encoder)) {}

  void apply(core::CanonicalEncodingError result) {
    if (!ok() || result == core::CanonicalEncodingError::None) {
      return;
    }
    GlobalGraphCheckpointError error =
        checkpointError(result == core::CanonicalEncodingError::OutputLimitExceeded
                            ? GlobalGraphCheckpointErrorCode::CapacityExceeded
                            : GlobalGraphCheckpointErrorCode::EncodingFailure,
                        "canonical checkpoint field encoding failed");
    error.encoding_error = result;
    error_ = std::move(error);
  }

  std::size_t remaining_entries_{};
  core::CanonicalEncoder encoder_;
  std::optional<GlobalGraphCheckpointError> error_;
};

class Reader {
public:
  Reader(std::span<const std::byte> bytes, GlobalGraphCheckpointLimits limits)
      : bytes_(bytes), remaining_entries_(limits.maximum_nested_collection_entries) {
    if (!validLimits(limits) || bytes.size() > limits.maximum_wire_bytes) {
      fail(GlobalGraphCheckpointErrorCode::InvalidLimits,
           "wire bytes or checkpoint decode limits are invalid");
    }
  }

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  [[nodiscard]] bool finished() const noexcept { return position_ == bytes_.size(); }
  [[nodiscard]] const GlobalGraphCheckpointError& error() const { return *error_; }

  void expectEnvelope(std::string_view domain, std::uint32_t schema) {
    const std::span<const std::byte> encoded_domain = bytesField();
    if (!ok()) {
      return;
    }
    const auto expected = std::as_bytes(std::span(domain.data(), domain.size()));
    if (!std::ranges::equal(encoded_domain, expected)) {
      fail(GlobalGraphCheckpointErrorCode::UnsupportedSchema,
           "checkpoint wire domain is unsupported");
      return;
    }
    if (u32() != schema && ok()) {
      fail(GlobalGraphCheckpointErrorCode::UnsupportedSchema,
           "checkpoint wire schema is unsupported");
    }
  }

  [[nodiscard]] std::uint8_t u8() { return static_cast<std::uint8_t>(unsignedValue(1U)); }
  [[nodiscard]] std::uint32_t u32() { return static_cast<std::uint32_t>(unsignedValue(4U)); }
  [[nodiscard]] std::uint64_t u64() { return unsignedValue(8U); }
  [[nodiscard]] std::int64_t i64() { return std::bit_cast<std::int64_t>(u64()); }

  [[nodiscard]] bool boolean() {
    const std::uint8_t value = u8();
    if (ok() && value > 1U) {
      fail(GlobalGraphCheckpointErrorCode::InvalidBoolean,
           "checkpoint boolean marker is neither zero nor one");
    }
    return value == 1U;
  }

  [[nodiscard]] double floating() {
    const double value = std::bit_cast<double>(u64());
    if (ok() && !std::isfinite(value)) {
      fail(GlobalGraphCheckpointErrorCode::InvalidFloatingPoint,
           "checkpoint contains a non-finite floating-point value");
    }
    return value;
  }

  [[nodiscard]] core::ContentHash hash() {
    core::ContentHash result{};
    const std::span<const std::byte> field = bytesField();
    if (!ok()) {
      return result;
    }
    if (field.size() != result.size()) {
      fail(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
           "checkpoint hash does not contain exactly 32 bytes");
      return result;
    }
    for (std::size_t index = 0U; index < result.size(); ++index) {
      result[index] = std::to_integer<std::uint8_t>(field[index]);
    }
    return result;
  }

  template <typename Id>
  [[nodiscard]] Id id() {
    return Id(u64());
  }

  [[nodiscard]] std::size_t count(std::size_t maximum, std::string_view name) {
    const std::uint64_t value = u64();
    if (!ok()) {
      return 0U;
    }
    if (value > maximum || value > remaining_entries_ ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      fail(GlobalGraphCheckpointErrorCode::CapacityExceeded,
           std::string(name) + " exceeds its checkpoint decode bound");
      return 0U;
    }
    remaining_entries_ -= static_cast<std::size_t>(value);
    return static_cast<std::size_t>(value);
  }

  template <Eigen::Index Dimension>
  [[nodiscard]] Eigen::Matrix<double, Dimension, 1> fixedVector() {
    Eigen::Matrix<double, Dimension, 1> result = Eigen::Matrix<double, Dimension, 1>::Zero();
    const std::uint64_t count = u64();
    if (ok() && count != static_cast<std::uint64_t>(Dimension)) {
      fail(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
           "checkpoint Eigen vector has the wrong dimension");
      return result;
    }
    for (Eigen::Index index = 0; index < Dimension && ok(); ++index) {
      result(index) = floating();
    }
    return result;
  }

  template <Eigen::Index Rows, Eigen::Index Columns>
  [[nodiscard]] Eigen::Matrix<double, Rows, Columns> fixedMatrix() {
    Eigen::Matrix<double, Rows, Columns> result = Eigen::Matrix<double, Rows, Columns>::Zero();
    const std::uint64_t rows = u64();
    const std::uint64_t columns = u64();
    if (ok() && (rows != static_cast<std::uint64_t>(Rows) ||
                 columns != static_cast<std::uint64_t>(Columns))) {
      fail(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
           "checkpoint Eigen matrix has the wrong dimensions");
      return result;
    }
    for (Eigen::Index row = 0; row < Rows && ok(); ++row) {
      for (Eigen::Index column = 0; column < Columns && ok(); ++column) {
        result(row, column) = floating();
      }
    }
    return result;
  }

  [[nodiscard]] core::Pose3d pose() {
    const Eigen::Matrix4d matrix = fixedMatrix<4, 4>();
    const Eigen::Vector4d quaternion_coefficients = fixedVector<4>();
    if (!ok()) {
      return core::Pose3d{};
    }
    const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
    Eigen::Quaterniond quaternion;
    quaternion.coeffs() = quaternion_coefficients;
    const Eigen::Matrix3d quaternion_rotation = quaternion.toRotationMatrix();
    if (matrix.row(3) != Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0) ||
        std::abs(quaternion.squaredNorm() - 1.0) > 1.0e-12 ||
        (quaternion_rotation - rotation).cwiseAbs().maxCoeff() > 1.0e-12 ||
        (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() >
            1.0e-10 ||
        std::abs(rotation.determinant() - 1.0) > 1.0e-10) {
      fail(GlobalGraphCheckpointErrorCode::InvalidPose,
           "checkpoint pose is not a canonical finite SE(3) matrix");
      return core::Pose3d{};
    }
    return core::Pose3d(Sophus::SO3d(quaternion), matrix.topRightCorner<3, 1>());
  }

  void fail(GlobalGraphCheckpointErrorCode code, std::string detail) {
    if (!error_) {
      error_ = checkpointError(code, std::move(detail));
    }
  }

private:
  [[nodiscard]] std::uint64_t unsignedValue(std::size_t width) {
    if (!ok()) {
      return 0U;
    }
    if (width > bytes_.size() - position_) {
      fail(GlobalGraphCheckpointErrorCode::TruncatedRecord,
           "checkpoint ends inside a fixed-width field");
      return 0U;
    }
    std::uint64_t result{};
    for (std::size_t index = 0U; index < width; ++index) {
      result = (result << 8U) | std::to_integer<std::uint8_t>(bytes_[position_ + index]);
    }
    position_ += width;
    return result;
  }

  [[nodiscard]] std::span<const std::byte> bytesField() {
    const std::uint64_t count = u64();
    if (!ok()) {
      return {};
    }
    if (count > static_cast<std::uint64_t>(bytes_.size() - position_)) {
      fail(GlobalGraphCheckpointErrorCode::TruncatedRecord,
           "checkpoint ends inside a length-delimited field");
      return {};
    }
    const auto size = static_cast<std::size_t>(count);
    const std::span<const std::byte> result = bytes_.subspan(position_, size);
    position_ += size;
    return result;
  }

  std::span<const std::byte> bytes_;
  std::size_t position_{};
  std::size_t remaining_entries_{};
  std::optional<GlobalGraphCheckpointError> error_;
};

template <typename Enum>
void writeEnum(Writer& writer, Enum value,
               std::initializer_list<std::pair<Enum, std::uint8_t>> map) {
  const auto found = std::find_if(map.begin(), map.end(),
                                  [value](const auto& item) { return item.first == value; });
  if (found == map.end()) {
    // An unsupported enum deliberately makes the encoder fail without relying
    // on the implementation-defined underlying representation.
    writer.floating(std::numeric_limits<double>::quiet_NaN());
    return;
  }
  writer.u8(found->second);
}

template <typename Enum>
[[nodiscard]] Enum readEnum(Reader& reader,
                            std::initializer_list<std::pair<std::uint8_t, Enum>> map,
                            Enum fallback) {
  const std::uint8_t encoded = reader.u8();
  const auto found = std::find_if(map.begin(), map.end(),
                                  [encoded](const auto& item) { return item.first == encoded; });
  if (reader.ok() && found == map.end()) {
    reader.fail(GlobalGraphCheckpointErrorCode::InvalidEnum,
                "checkpoint contains an unsupported enumeration value");
    return fallback;
  }
  return found == map.end() ? fallback : found->second;
}

void writeRecordHeader(Writer& writer, const core::RecordHeader& header) {
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

[[nodiscard]] core::RecordHeader readRecordHeader(Reader& reader) {
  core::RecordHeader result;
  result.schema_version = reader.u32();
  result.trace = reader.id<core::TraceId>();
  result.producer = reader.id<core::ProducerId>();
  result.session = reader.id<core::SessionId>();
  result.created_at.nanoseconds = reader.i64();
  result.config = reader.id<core::ConfigRevision>();
  if (reader.boolean()) {
    result.direct_calibration = reader.id<core::CalibrationEpoch>();
  }
  return result;
}

void writeSubmapRef(Writer& writer, const core::SubmapRef& ref) {
  writer.id(ref.session);
  writer.id(ref.odom_epoch);
  writer.id(ref.id);
  writer.id(ref.calibration);
  writer.id(ref.content_revision);
  writer.hash(ref.local_content_checksum);
}

[[nodiscard]] core::SubmapRef readSubmapRef(Reader& reader) {
  core::SubmapRef result;
  result.session = reader.id<core::SessionId>();
  result.odom_epoch = reader.id<core::OdomEpoch>();
  result.id = reader.id<core::SubmapId>();
  result.calibration = reader.id<core::CalibrationEpoch>();
  result.content_revision = reader.id<core::SubmapContentRevision>();
  result.local_content_checksum = reader.hash();
  return result;
}

void writeBoundary(Writer& writer, const core::BoundaryNavigationLinearization& boundary) {
  writer.id(boundary.state);
  writer.i64(boundary.exact_time.nanoseconds);
  writer.id(boundary.final_revision);
  writer.pose(boundary.T_odom_imu);
  writer.vector(boundary.velocity_odom);
  writer.vector(boundary.gyro_bias);
  writer.vector(boundary.accel_bias);
}

[[nodiscard]] core::BoundaryNavigationLinearization readBoundary(Reader& reader) {
  core::BoundaryNavigationLinearization result;
  result.state = reader.id<core::StateId>();
  result.exact_time.nanoseconds = reader.i64();
  result.final_revision = reader.id<core::LocalGraphRevision>();
  result.T_odom_imu = reader.pose();
  result.velocity_odom = reader.fixedVector<3>();
  result.gyro_bias = reader.fixedVector<3>();
  result.accel_bias = reader.fixedVector<3>();
  return result;
}

void writeBlobRef(Writer& writer, const core::BlobRef& ref) {
  writer.id(ref.store);
  writer.id(ref.id);
  writer.hash(ref.checksum);
  writer.id(ref.layout);
  writer.u64(ref.bytes);
  writeEnum(writer, ref.storage,
            {{core::BlobStorage::InProcessPool, 1U},
             {core::BlobStorage::SharedMemoryLease, 2U},
             {core::BlobStorage::DurableSpool, 3U}});
  writer.optional(ref.lease_token.has_value());
  if (ref.lease_token) {
    writer.id(ref.lease_token->id);
    writer.id(ref.lease_token->issuing_store_instance);
  }
}

[[nodiscard]] core::BlobRef readBlobRef(Reader& reader) {
  core::BlobRef result;
  result.store = reader.id<core::BlobStoreId>();
  result.id = reader.id<core::BlobId>();
  result.checksum = reader.hash();
  result.layout = reader.id<core::LayoutId>();
  result.bytes = reader.u64();
  result.storage = readEnum(reader,
                            {{1U, core::BlobStorage::InProcessPool},
                             {2U, core::BlobStorage::SharedMemoryLease},
                             {3U, core::BlobStorage::DurableSpool}},
                            core::BlobStorage::InProcessPool);
  if (reader.boolean()) {
    result.lease_token =
        core::LeaseToken{reader.id<core::LeaseTokenId>(), reader.id<core::StoreInstanceEpoch>()};
  }
  return result;
}

void writeObservationLineage(Writer& writer, const core::ObservationLineage& lineage) {
  writer.id(lineage.id);
  writer.count(lineage.usage.size());
  for (const core::ObservationUsage& usage : lineage.usage) {
    if (usage.slice.root.valueless_by_exception()) {
      writer.floating(std::numeric_limits<double>::quiet_NaN());
      return;
    }
    if (std::holds_alternative<core::MeasurementId>(usage.slice.root)) {
      writer.u8(1U);
      writer.id(std::get<core::MeasurementId>(usage.slice.root));
    } else {
      writer.u8(2U);
      writer.id(std::get<core::GnssObservationId>(usage.slice.root));
    }
    writeEnum(writer, usage.slice.kind,
              {{core::SliceKind::Whole, 1U}, {core::SliceKind::IndexRange, 2U}});
    writer.u64(usage.slice.begin);
    writer.u64(usage.slice.end);
    writer.hash(usage.slice.source_checksum);
    writer.id(usage.slice.calibration);
    writeEnum(writer, usage.role,
              {{core::ObservationRole::PrimaryResidual, 1U},
               {core::ObservationRole::ConditioningOnly, 2U},
               {core::ObservationRole::RetrievalSeedOnly, 3U},
               {core::ObservationRole::DerivedSummary, 4U}});
    writer.id(usage.consumer);
    writer.optional(usage.factor_group.has_value());
    if (usage.factor_group) {
      writer.id(*usage.factor_group);
    }
    writer.optional(usage.correlation_group.has_value());
    if (usage.correlation_group) {
      writer.id(*usage.correlation_group);
    }
  }
  writer.count(lineage.correlations.size());
  for (const core::CorrelationDeclaration& declaration : lineage.correlations) {
    writer.id(declaration.group);
    writer.id(declaration.policy);
    writeEnum(writer, declaration.treatment,
              {{core::CorrelationTreatment::JointCompositeWhitening, 1U},
               {core::CorrelationTreatment::CovarianceInflationAndInformationCap, 2U},
               {core::CorrelationTreatment::NotIndependent, 3U}});
    writer.floating(declaration.covariance_inflation);
    writer.optional(declaration.total_information_cap.has_value());
    if (declaration.total_information_cap) {
      writer.floating(*declaration.total_information_cap);
    }
  }
  writer.hash(lineage.checksum);
}

[[nodiscard]] core::ObservationLineage readObservationLineage(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  core::ObservationLineage result;
  result.id = reader.id<core::ObservationLineageId>();
  const std::size_t usage_count =
      reader.count(limits.maximum_nested_collection_entries, "lineage usage count");
  result.usage.reserve(usage_count);
  for (std::size_t index = 0U; index < usage_count && reader.ok(); ++index) {
    core::ObservationUsage usage;
    const std::uint8_t root_kind = reader.u8();
    if (root_kind == 1U) {
      usage.slice.root = reader.id<core::MeasurementId>();
    } else if (root_kind == 2U) {
      usage.slice.root = reader.id<core::GnssObservationId>();
    } else {
      reader.fail(GlobalGraphCheckpointErrorCode::InvalidEnum, "lineage root kind is unsupported");
    }
    usage.slice.kind =
        readEnum(reader, {{1U, core::SliceKind::Whole}, {2U, core::SliceKind::IndexRange}},
                 core::SliceKind::Whole);
    usage.slice.begin = reader.u64();
    usage.slice.end = reader.u64();
    usage.slice.source_checksum = reader.hash();
    usage.slice.calibration = reader.id<core::CalibrationEpoch>();
    usage.role = readEnum(reader,
                          {{1U, core::ObservationRole::PrimaryResidual},
                           {2U, core::ObservationRole::ConditioningOnly},
                           {3U, core::ObservationRole::RetrievalSeedOnly},
                           {4U, core::ObservationRole::DerivedSummary}},
                          core::ObservationRole::DerivedSummary);
    usage.consumer = reader.id<core::DerivedRecordId>();
    if (reader.boolean()) {
      usage.factor_group = reader.id<core::FactorGroupId>();
    }
    if (reader.boolean()) {
      usage.correlation_group = reader.id<core::CorrelationGroupId>();
    }
    result.usage.push_back(std::move(usage));
  }
  const std::size_t correlation_count =
      reader.count(limits.maximum_nested_collection_entries, "lineage correlation count");
  result.correlations.reserve(correlation_count);
  for (std::size_t index = 0U; index < correlation_count && reader.ok(); ++index) {
    core::CorrelationDeclaration declaration;
    declaration.group = reader.id<core::CorrelationGroupId>();
    declaration.policy = reader.id<core::CorrelationPolicyRevision>();
    declaration.treatment =
        readEnum(reader,
                 {{1U, core::CorrelationTreatment::JointCompositeWhitening},
                  {2U, core::CorrelationTreatment::CovarianceInflationAndInformationCap},
                  {3U, core::CorrelationTreatment::NotIndependent}},
                 core::CorrelationTreatment::NotIndependent);
    declaration.covariance_inflation = reader.floating();
    if (reader.boolean()) {
      declaration.total_information_cap = reader.floating();
    }
    result.correlations.push_back(std::move(declaration));
  }
  result.checksum = reader.hash();
  return result;
}

void writeFrozenFactor(Writer& writer, const core::FrozenSquareRootFactor& factor) {
  writeEnum(writer, factor.pose_tangent,
            {{core::PoseTangentConvention::RightTranslationFirst, 1U}});
  writer.u32(factor.rows);
  writer.u32(factor.columns);
  writer.count(factor.layout.size());
  for (const core::SquareRootColumnBlock& block : factor.layout) {
    writeEnum(writer, block.variable.kind,
              {{core::LocalVariableKind::Pose, 1U},
               {core::LocalVariableKind::NavigationVelocity, 2U},
               {core::LocalVariableKind::GyroBias, 3U},
               {core::LocalVariableKind::AccelBias, 4U},
               {core::LocalVariableKind::LandmarkLogInverseRange, 5U}});
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
  }
  writer.count(factor.row_major_A.size());
  for (double value : factor.row_major_A) {
    writer.floating(value);
  }
  writer.count(factor.rhs.size());
  for (double value : factor.rhs) {
    writer.floating(value);
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
  writer.hash(factor.checksum);
}

[[nodiscard]] core::FrozenSquareRootFactor readFrozenFactor(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  core::FrozenSquareRootFactor result;
  result.pose_tangent = readEnum(reader, {{1U, core::PoseTangentConvention::RightTranslationFirst}},
                                 core::PoseTangentConvention::RightTranslationFirst);
  result.rows = reader.u32();
  result.columns = reader.u32();
  if (reader.ok() && (result.rows > limits.maximum_factor_rows ||
                      (result.columns != 0U &&
                       result.rows > limits.maximum_factor_coefficients / result.columns))) {
    reader.fail(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                "frozen factor dimensions exceed checkpoint bounds");
  }
  const std::size_t layout_count = reader.count(64U, "frozen factor column layout count");
  result.layout.reserve(layout_count);
  for (std::size_t index = 0U; index < layout_count && reader.ok(); ++index) {
    core::SquareRootColumnBlock block;
    block.variable.kind = readEnum(reader,
                                   {{1U, core::LocalVariableKind::Pose},
                                    {2U, core::LocalVariableKind::NavigationVelocity},
                                    {3U, core::LocalVariableKind::GyroBias},
                                    {4U, core::LocalVariableKind::AccelBias},
                                    {5U, core::LocalVariableKind::LandmarkLogInverseRange}},
                                   core::LocalVariableKind::Pose);
    if (reader.boolean()) {
      block.variable.state = reader.id<core::StateId>();
    }
    if (reader.boolean()) {
      block.variable.landmark = reader.id<core::LandmarkSegmentId>();
    }
    block.column_offset = reader.u32();
    block.dimension = reader.u32();
    result.layout.push_back(std::move(block));
  }
  const std::size_t coefficient_count =
      reader.count(limits.maximum_factor_coefficients, "frozen factor coefficient count");
  result.row_major_A.reserve(coefficient_count);
  for (std::size_t index = 0U; index < coefficient_count && reader.ok(); ++index) {
    result.row_major_A.push_back(reader.floating());
  }
  const std::size_t rhs_count =
      reader.count(limits.maximum_factor_rows, "frozen factor right-hand-side count");
  result.rhs.reserve(rhs_count);
  for (std::size_t index = 0U; index < rhs_count && reader.ok(); ++index) {
    result.rhs.push_back(reader.floating());
  }
  result.constant_squared_error = reader.floating();
  result.numerical_rank = reader.u32();
  result.absolute_rank_tolerance = reader.floating();
  result.relative_rank_tolerance = reader.floating();
  result.cost_statistics.source_residual_dof = reader.u64();
  result.cost_statistics.eliminated_numerical_rank = reader.u64();
  result.cost_statistics.effective_dof = reader.u64();
  result.cost_statistics.calibration_revision = reader.id<core::ResidualCalibrationRevision>();
  if (reader.boolean()) {
    result.cost_statistics.calibrated_total_cost_cutoff = reader.floating();
  }
  result.checksum = reader.hash();
  return result;
}

void writeCondensedTransition(Writer& writer, const core::CondensedBoundaryTransition& transition) {
  writeRecordHeader(writer, transition.header);
  writer.id(transition.odom_epoch);
  writeBoundary(writer, transition.from);
  writeBoundary(writer, transition.to);
  writeFrozenFactor(writer, transition.boundary_factor);
  writer.count(transition.source_factors.size());
  for (const core::LocalFactorRef& source : transition.source_factors) {
    writer.id(source.odom_epoch);
    writer.id(source.factor);
  }
  writeObservationLineage(writer, transition.lineage);
  writer.id(transition.final_revision);
  writer.hash(transition.input_partition_checksum);
  writer.hash(transition.checksum);
}

[[nodiscard]] core::CondensedBoundaryTransition readCondensedTransition(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  core::CondensedBoundaryTransition result;
  result.header = readRecordHeader(reader);
  result.odom_epoch = reader.id<core::OdomEpoch>();
  result.from = readBoundary(reader);
  result.to = readBoundary(reader);
  result.boundary_factor = readFrozenFactor(reader, limits);
  const std::size_t source_count =
      reader.count(limits.maximum_nested_collection_entries, "source factor count");
  result.source_factors.reserve(source_count);
  for (std::size_t index = 0U; index < source_count && reader.ok(); ++index) {
    result.source_factors.push_back(
        core::LocalFactorRef{reader.id<core::OdomEpoch>(), reader.id<core::FactorId>()});
  }
  result.lineage = readObservationLineage(reader, limits);
  result.final_revision = reader.id<core::LocalGraphRevision>();
  result.input_partition_checksum = reader.hash();
  result.checksum = reader.hash();
  return result;
}

void writeSealedTransition(Writer& writer, const core::SealedBoundaryTransition& transition) {
  writeSubmapRef(writer, transition.from);
  writeSubmapRef(writer, transition.to);
  writeCondensedTransition(writer, transition.local_transition);
  writer.hash(transition.checksum);
}

[[nodiscard]] core::SealedBoundaryTransition readSealedTransition(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  core::SealedBoundaryTransition result;
  result.from = readSubmapRef(reader);
  result.to = readSubmapRef(reader);
  result.local_transition = readCondensedTransition(reader, limits);
  result.checksum = reader.hash();
  return result;
}

void writePayloadCatalog(Writer& writer, const core::SparsePayloadCatalog& catalog) {
  writer.count(catalog.entries.size());
  for (const core::SparsePayloadIndexEntry& entry : catalog.entries) {
    writeEnum(writer, entry.kind,
              {{core::SparsePayloadKind::InternalTrajectory, 1U},
               {core::SparsePayloadKind::KeyframeIndex, 2U},
               {core::SparsePayloadKind::RegistrationProxy, 3U},
               {core::SparsePayloadKind::DenseInputIndex, 4U},
               {core::SparsePayloadKind::VisualPlaceCatalog, 5U},
               {core::SparsePayloadKind::LidarPlaceCatalog, 6U}});
    writeBlobRef(writer, entry.root);
  }
  writer.hash(catalog.checksum);
}

[[nodiscard]] core::SparsePayloadCatalog readPayloadCatalog(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  core::SparsePayloadCatalog result;
  const std::size_t count = reader.count(6U, "sparse payload catalog count");
  result.entries.reserve(count);
  for (std::size_t index = 0U; index < count && reader.ok(); ++index) {
    core::SparsePayloadIndexEntry entry;
    entry.kind = readEnum(reader,
                          {{1U, core::SparsePayloadKind::InternalTrajectory},
                           {2U, core::SparsePayloadKind::KeyframeIndex},
                           {3U, core::SparsePayloadKind::RegistrationProxy},
                           {4U, core::SparsePayloadKind::DenseInputIndex},
                           {5U, core::SparsePayloadKind::VisualPlaceCatalog},
                           {6U, core::SparsePayloadKind::LidarPlaceCatalog}},
                          core::SparsePayloadKind::InternalTrajectory);
    entry.root = readBlobRef(reader);
    result.entries.push_back(std::move(entry));
  }
  (void)limits;
  result.checksum = reader.hash();
  return result;
}

void writeSparseSeal(Writer& writer, const core::SparseSubmapSeal& seal) {
  writeRecordHeader(writer, seal.header);
  writeSubmapRef(writer, seal.ref);
  writer.optional(seal.previous.has_value());
  if (seal.previous) {
    writeSubmapRef(writer, *seal.previous);
  }
  writer.id(seal.final_local_revision);
  writer.i64(seal.support_time.start.nanoseconds);
  writer.i64(seal.support_time.end.nanoseconds);
  writer.id(seal.frame.boundary_state);
  writer.i64(seal.frame.boundary_time.nanoseconds);
  writer.vector(seal.frame.gravity_up_odom);
  writer.floating(seal.frame.boundary_yaw_odom);
  writer.pose(seal.T_odom_submap);
  writeBoundary(writer, seal.boundary_navigation);
  writePayloadCatalog(writer, seal.payloads);
  writer.optional(seal.from_previous.has_value());
  if (seal.from_previous) {
    writeSealedTransition(writer, *seal.from_previous);
  }
  writeObservationLineage(writer, seal.lineage);
  writer.hash(seal.quality_checksum);
  writer.hash(seal.seal_checksum);
}

[[nodiscard]] core::SparseSubmapSeal readSparseSeal(Reader& reader,
                                                    const GlobalGraphCheckpointLimits& limits) {
  core::SparseSubmapSeal result;
  result.header = readRecordHeader(reader);
  result.ref = readSubmapRef(reader);
  if (reader.boolean()) {
    result.previous = readSubmapRef(reader);
  }
  result.final_local_revision = reader.id<core::LocalGraphRevision>();
  result.support_time.start.nanoseconds = reader.i64();
  result.support_time.end.nanoseconds = reader.i64();
  result.frame.boundary_state = reader.id<core::StateId>();
  result.frame.boundary_time.nanoseconds = reader.i64();
  result.frame.gravity_up_odom = reader.fixedVector<3>();
  result.frame.boundary_yaw_odom = reader.floating();
  result.T_odom_submap = reader.pose();
  result.boundary_navigation = readBoundary(reader);
  result.payloads = readPayloadCatalog(reader, limits);
  if (reader.boolean()) {
    result.from_previous = readSealedTransition(reader, limits);
  }
  result.lineage = readObservationLineage(reader, limits);
  result.quality_checksum = reader.hash();
  result.seal_checksum = reader.hash();
  return result;
}

void writePoseCovariance(Writer& writer, const core::PoseCovariance& covariance) {
  writer.matrix(covariance.matrix);
  writeEnum(writer, covariance.tangent, {{core::PoseTangentConvention::RightTranslationFirst, 1U}});
}

[[nodiscard]] core::PoseCovariance readPoseCovariance(Reader& reader) {
  core::PoseCovariance result;
  result.matrix = reader.fixedMatrix<6, 6>();
  result.tangent = readEnum(reader, {{1U, core::PoseTangentConvention::RightTranslationFirst}},
                            core::PoseTangentConvention::RightTranslationFirst);
  return result;
}

void writeRankAwareInformation(Writer& writer, const core::RankAwareInformation& information) {
  writer.matrix(information.basis);
  writer.vector(information.eigenvalues);
  writer.u64(information.rank);
  writeEnum(writer, information.tangent,
            {{core::PoseTangentConvention::RightTranslationFirst, 1U}});
}

[[nodiscard]] core::RankAwareInformation readRankAwareInformation(Reader& reader) {
  core::RankAwareInformation result;
  result.basis = reader.fixedMatrix<6, 6>();
  result.eigenvalues = reader.fixedVector<6>();
  const std::uint64_t rank = reader.u64();
  if (reader.ok() && rank > 6U) {
    reader.fail(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                "rank-aware loop information rank exceeds six");
  }
  result.rank = static_cast<std::size_t>(rank);
  result.tangent = readEnum(reader, {{1U, core::PoseTangentConvention::RightTranslationFirst}},
                            core::PoseTangentConvention::RightTranslationFirst);
  return result;
}

void writeGncScale(Writer& writer, const GncTlsFactorScale& scale) {
  writer.u32(scale.degrees_of_freedom);
  writer.floating(scale.calibrated_chi_squared_cutoff);
}

[[nodiscard]] GncTlsFactorScale readGncScale(Reader& reader) {
  return GncTlsFactorScale{reader.u32(), reader.floating()};
}

void writeLoopMeasurement(Writer& writer, const LoopMeasurement& measurement) {
  writeRecordHeader(writer, measurement.header);
  writer.id(measurement.proposal);
  writeEnum(writer, measurement.modality, {{LoopModality::Visual, 1U}, {LoopModality::Lidar, 2U}});
  writeSubmapRef(writer, measurement.from);
  writeSubmapRef(writer, measurement.to);
  writer.count(measurement.calibration_epochs.size());
  for (core::CalibrationEpoch epoch : measurement.calibration_epochs) {
    writer.id(epoch);
  }
  writer.pose(measurement.T_from_to);
  writeRankAwareInformation(writer, measurement.information);
  writeObservationLineage(writer, measurement.lineage);
}

[[nodiscard]] LoopMeasurement readLoopMeasurement(Reader& reader,
                                                  const GlobalGraphCheckpointLimits& limits) {
  LoopMeasurement result;
  result.header = readRecordHeader(reader);
  result.proposal = reader.id<ProposalId>();
  result.modality = readEnum(reader, {{1U, LoopModality::Visual}, {2U, LoopModality::Lidar}},
                             LoopModality::Visual);
  result.from = readSubmapRef(reader);
  result.to = readSubmapRef(reader);
  const std::size_t calibration_count = reader.count(256U, "loop calibration epoch count");
  result.calibration_epochs.reserve(calibration_count);
  for (std::size_t index = 0U; index < calibration_count && reader.ok(); ++index) {
    result.calibration_epochs.push_back(reader.id<core::CalibrationEpoch>());
  }
  result.T_from_to = reader.pose();
  result.information = readRankAwareInformation(reader);
  result.lineage = readObservationLineage(reader, limits);
  return result;
}

void writeGnssConstraint(Writer& writer, const GnssAntennaConstraint& constraint) {
  writeSubmapRef(writer, constraint.submap);
  writer.id(constraint.observation);
  writer.vector(constraint.antenna_position_submap);
  writer.vector(constraint.measured_position_enu);
  writer.matrix(constraint.effective_covariance_enu);
}

[[nodiscard]] GnssAntennaConstraint readGnssConstraint(Reader& reader) {
  GnssAntennaConstraint result;
  result.submap = readSubmapRef(reader);
  result.observation = reader.id<core::GnssObservationId>();
  result.antenna_position_submap = reader.fixedVector<3>();
  result.measured_position_enu = reader.fixedVector<3>();
  result.effective_covariance_enu = reader.fixedMatrix<3, 3>();
  return result;
}

void writeSolveReport(Writer& writer, const GlobalSolveReport& report) {
  writeEnum(writer, report.transaction,
            {{GlobalTransactionKind::MissionInitialization, 1U},
             {GlobalTransactionKind::AdjacentInsertion, 2U},
             {GlobalTransactionKind::GnssInsertion, 3U},
             {GlobalTransactionKind::RobustLoopInsertion, 4U}});
  writer.u64(report.anchors);
  writer.u64(report.materialized_navigation_boundaries);
  writer.u64(report.adjacent_seals_in_transaction);
  writer.u64(report.adjacent_factors);
  writer.u64(report.gnss_factors);
  writer.u64(report.loop_factors);
  writer.u64(report.scalar_dimension);
  writer.u64(report.numerical_rank);
  writer.u64(report.solver_iterations);
  writer.floating(report.initial_error);
  writer.floating(report.final_error);
  writer.floating(report.hessian_condition);
  writer.boolean(report.finite);
  writer.boolean(report.converged);
  writer.boolean(report.connected);
}

[[nodiscard]] GlobalSolveReport readSolveReport(Reader& reader) {
  GlobalSolveReport result;
  result.transaction = readEnum(reader,
                                {{1U, GlobalTransactionKind::MissionInitialization},
                                 {2U, GlobalTransactionKind::AdjacentInsertion},
                                 {3U, GlobalTransactionKind::GnssInsertion},
                                 {4U, GlobalTransactionKind::RobustLoopInsertion}},
                                GlobalTransactionKind::MissionInitialization);
  const auto size = [&reader]() -> std::size_t {
    const std::uint64_t value = reader.u64();
    if (reader.ok() &&
        value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      reader.fail(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                  "solve report size does not fit this platform");
      return 0U;
    }
    return static_cast<std::size_t>(value);
  };
  result.anchors = size();
  result.materialized_navigation_boundaries = size();
  result.adjacent_seals_in_transaction = size();
  result.adjacent_factors = size();
  result.gnss_factors = size();
  result.loop_factors = size();
  result.scalar_dimension = size();
  result.numerical_rank = size();
  result.solver_iterations = size();
  result.initial_error = reader.floating();
  result.final_error = reader.floating();
  result.hessian_condition = reader.floating();
  result.finite = reader.boolean();
  result.converged = reader.boolean();
  result.connected = reader.boolean();
  return result;
}

void writeMapOdom(Writer& writer, const MapOdomEstimate& estimate) {
  writer.id(estimate.graph_revision);
  writeSubmapRef(writer, estimate.reference_submap);
  writer.pose(estimate.T_map_odom);
  writePoseCovariance(writer, estimate.covariance);
  writeEnum(writer, estimate.covariance_semantics,
            {{MapOdomCovarianceSemantics::ConditionalOnSealedLocalFrame, 1U}});
}

[[nodiscard]] MapOdomEstimate readMapOdom(Reader& reader) {
  MapOdomEstimate result;
  result.graph_revision = reader.id<GlobalGraphRevision>();
  result.reference_submap = readSubmapRef(reader);
  result.T_map_odom = reader.pose();
  result.covariance = readPoseCovariance(reader);
  result.covariance_semantics =
      readEnum(reader, {{1U, MapOdomCovarianceSemantics::ConditionalOnSealedLocalFrame}},
               MapOdomCovarianceSemantics::ConditionalOnSealedLocalFrame);
  return result;
}

void writeRecoveryTolerances(Writer& writer, const GlobalGraphRecoveryTolerances& tolerances) {
  writer.floating(tolerances.objective_absolute);
  writer.floating(tolerances.objective_relative);
  writer.floating(tolerances.estimate_tangent_absolute);
  writer.floating(tolerances.gradient_infinity_absolute);
  writer.floating(tolerances.covariance_absolute);
  writer.floating(tolerances.covariance_relative);
  writer.floating(tolerances.condition_relative);
}

[[nodiscard]] GlobalGraphRecoveryTolerances readRecoveryTolerances(Reader& reader) {
  GlobalGraphRecoveryTolerances result;
  result.objective_absolute = reader.floating();
  result.objective_relative = reader.floating();
  result.estimate_tangent_absolute = reader.floating();
  result.gradient_infinity_absolute = reader.floating();
  result.covariance_absolute = reader.floating();
  result.covariance_relative = reader.floating();
  result.condition_relative = reader.floating();
  return result;
}

void writeConfigurationFields(Writer& writer, const GlobalGraphConfig& config) {
  writer.u64(config.maximum_anchors);
  writer.u64(config.maximum_scalar_dimension);
  writer.u64(config.maximum_adjacent_factors);
  writer.u64(config.maximum_adjacent_seals_per_transaction);
  writer.u64(config.maximum_adjacent_factor_rows);
  writer.u64(config.maximum_adjacent_factor_coefficients);
  writer.u64(config.maximum_total_adjacent_factor_rows);
  writer.u64(config.maximum_total_adjacent_factor_coefficients);
  writer.u64(config.maximum_gnss_factors);
  writer.u64(config.maximum_loop_factors);
  writer.u64(config.maximum_loop_candidates_per_transaction);
  writer.u64(config.maximum_loop_calibration_epochs);
  writer.u64(config.maximum_loop_lineage_usages);
  writer.u64(config.maximum_loop_lineage_correlations);
  writer.u64(config.maximum_solver_iterations);
  writer.u64(config.loop_gnc.maximum_known_inliers);
  writer.u64(config.loop_gnc.maximum_robust_candidates);
  writer.u64(config.loop_gnc.maximum_iterations);
  writer.u64(config.loop_gnc.maximum_inner_solver_iterations);
  writer.u32(config.loop_gnc.maximum_degrees_of_freedom);
  writer.floating(config.loop_gnc.mu_step);
  writer.floating(config.loop_gnc.minimum_mu);
  writer.floating(config.loop_gnc.maximum_mu);
  writer.floating(config.loop_gnc.binary_weight_tolerance);
  writer.floating(config.loop_gnc.stable_weight_tolerance);
  writer.floating(config.loop_gnc.minimum_chi_squared_cutoff);
  writer.floating(config.loop_gnc.maximum_chi_squared_cutoff);
  writer.floating(config.loop_gnc.maximum_whitened_squared_cost);
  writer.floating(config.loop_gnc.maximum_normalized_squared_cost);
  writer.floating(config.mission_gauge_translation_sigma_m);
  writer.floating(config.mission_gauge_rotation_sigma_rad);
  writer.floating(config.covariance_symmetry_relative_tolerance);
  writer.floating(config.minimum_covariance_eigenvalue);
  writer.floating(config.information_basis_orthonormal_tolerance);
  writer.floating(config.information_zero_tolerance);
  writer.floating(config.hessian_absolute_rank_tolerance);
  writer.floating(config.hessian_relative_rank_tolerance);
  writer.floating(config.maximum_hessian_condition);
  writer.floating(config.solver_relative_error_tolerance);
  writer.floating(config.solver_absolute_error_tolerance);
}

void writeCheckpointFields(Writer& writer, const GlobalGraphCheckpoint& checkpoint,
                           bool include_checksum) {
  writer.u32(checkpoint.schema_version);
  writer.u32(checkpoint.key_schema_version);
  writer.u32(checkpoint.configuration.schema_version);
  writer.hash(checkpoint.configuration.checksum);
  writer.id(checkpoint.mission_session);
  writer.id(checkpoint.revision);
  writer.optional(checkpoint.parent.has_value());
  if (checkpoint.parent) {
    writer.id(*checkpoint.parent);
  }
  writer.id(checkpoint.mission_gauge.factor);
  writer.u64(checkpoint.mission_gauge.boundary_slot);
  writer.pose(checkpoint.mission_gauge.T_map_submap);
  writer.floating(checkpoint.mission_gauge.translation_sigma_m);
  writer.floating(checkpoint.mission_gauge.rotation_sigma_rad);

  writer.count(checkpoint.chart_placements.size());
  for (const OdomEpochChartPlacementCheckpoint& placement : checkpoint.chart_placements) {
    writer.id(placement.odom_epoch);
    writer.pose(placement.H_map_odom);
  }
  writer.count(checkpoint.boundaries.size());
  for (const BoundaryNavigationCheckpoint& boundary : checkpoint.boundaries) {
    writer.u64(boundary.slot);
    writeSparseSeal(writer, boundary.seal);
    writer.pose(boundary.T_map_submap);
    writer.optional(boundary.velocity_map.has_value());
    if (boundary.velocity_map) {
      writer.vector(*boundary.velocity_map);
    }
    writer.optional(boundary.gyro_bias.has_value());
    if (boundary.gyro_bias) {
      writer.vector(*boundary.gyro_bias);
    }
    writer.optional(boundary.accel_bias.has_value());
    if (boundary.accel_bias) {
      writer.vector(*boundary.accel_bias);
    }
  }
  writer.count(checkpoint.adjacent_factors.size());
  for (const AdjacentBoundaryCheckpoint& adjacent : checkpoint.adjacent_factors) {
    writer.id(adjacent.factor);
    writer.u64(adjacent.from_slot);
    writer.u64(adjacent.to_slot);
    writeSealedTransition(writer, adjacent.transition);
  }
  writer.count(checkpoint.gnss_factors.size());
  for (const GnssFactorCheckpoint& gnss : checkpoint.gnss_factors) {
    writer.id(gnss.factor);
    writeGnssConstraint(writer, gnss.constraint);
  }
  writer.count(checkpoint.loop_factors.size());
  for (const LoopFactorCheckpoint& loop : checkpoint.loop_factors) {
    writer.id(loop.factor);
    writeLoopMeasurement(writer, loop.measurement);
    writer.id(loop.candidate);
    writeGncScale(writer, loop.scale);
  }
  writer.count(checkpoint.factor_order.size());
  for (GlobalFactorId factor : checkpoint.factor_order) {
    writer.id(factor);
  }
  writer.optional(checkpoint.alignment.has_value());
  if (checkpoint.alignment) {
    writer.vector(checkpoint.alignment->translation_enu);
    writer.floating(checkpoint.alignment->yaw_enu_map_rad);
  }
  writer.u64(checkpoint.next_boundary_slot);
  writer.u64(checkpoint.next_factor_id);
  writer.u64(checkpoint.next_candidate_id);

  writer.floating(checkpoint.recovery.whitened_squared_objective);
  writer.floating(checkpoint.recovery.gradient_infinity_norm);
  writer.u64(checkpoint.recovery.scalar_dimension);
  writer.u64(checkpoint.recovery.numerical_rank);
  writer.floating(checkpoint.recovery.hessian_condition);
  writer.count(checkpoint.recovery.factor_objectives.size());
  for (const GlobalFactorObjectiveCheckpoint& factor : checkpoint.recovery.factor_objectives) {
    writer.id(factor.factor);
    writer.floating(factor.whitened_squared_cost);
  }
  writer.count(checkpoint.recovery.boundary_marginals.size());
  for (const BoundaryMarginalCheckpoint& marginal : checkpoint.recovery.boundary_marginals) {
    writer.u64(marginal.boundary_slot);
    writePoseCovariance(writer, marginal.covariance);
  }
  writer.optional(checkpoint.recovery.alignment_covariance.has_value());
  if (checkpoint.recovery.alignment_covariance) {
    writer.matrix(checkpoint.recovery.alignment_covariance->matrix);
    writeEnum(writer, checkpoint.recovery.alignment_covariance->tangent,
              {{AlignmentTangentConvention::TranslationEnuThenYaw, 1U}});
  }
  writeMapOdom(writer, checkpoint.recovery.map_odom);
  writeSolveReport(writer, checkpoint.recovery.committed_solve);
  writeRecoveryTolerances(writer, checkpoint.recovery.tolerances);
  if (include_checksum) {
    writer.hash(checkpoint.checksum);
  }
}

[[nodiscard]] std::size_t readSize(Reader& reader, std::string_view name) {
  const std::uint64_t value = reader.u64();
  if (reader.ok() && value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    reader.fail(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                std::string(name) + " does not fit this platform");
    return 0U;
  }
  return static_cast<std::size_t>(value);
}

[[nodiscard]] GlobalGraphCheckpoint readCheckpointFields(
    Reader& reader, const GlobalGraphCheckpointLimits& limits) {
  GlobalGraphCheckpoint result;
  result.schema_version = reader.u32();
  result.key_schema_version = reader.u32();
  result.configuration.schema_version = reader.u32();
  result.configuration.checksum = reader.hash();
  result.mission_session = reader.id<core::SessionId>();
  result.revision = reader.id<GlobalGraphRevision>();
  if (reader.boolean()) {
    result.parent = reader.id<GlobalGraphRevision>();
  }
  result.mission_gauge.factor = reader.id<GlobalFactorId>();
  result.mission_gauge.boundary_slot = reader.u64();
  result.mission_gauge.T_map_submap = reader.pose();
  result.mission_gauge.translation_sigma_m = reader.floating();
  result.mission_gauge.rotation_sigma_rad = reader.floating();

  const std::size_t placement_count =
      reader.count(limits.maximum_chart_placements, "chart placement count");
  result.chart_placements.reserve(placement_count);
  for (std::size_t index = 0U; index < placement_count && reader.ok(); ++index) {
    result.chart_placements.push_back(
        OdomEpochChartPlacementCheckpoint{reader.id<core::OdomEpoch>(), reader.pose()});
  }
  const std::size_t boundary_count = reader.count(limits.maximum_boundaries, "boundary count");
  result.boundaries.reserve(boundary_count);
  for (std::size_t index = 0U; index < boundary_count && reader.ok(); ++index) {
    BoundaryNavigationCheckpoint boundary;
    boundary.slot = reader.u64();
    boundary.seal = readSparseSeal(reader, limits);
    boundary.T_map_submap = reader.pose();
    if (reader.boolean()) {
      boundary.velocity_map = reader.fixedVector<3>();
    }
    if (reader.boolean()) {
      boundary.gyro_bias = reader.fixedVector<3>();
    }
    if (reader.boolean()) {
      boundary.accel_bias = reader.fixedVector<3>();
    }
    result.boundaries.push_back(std::move(boundary));
  }
  const std::size_t adjacent_count =
      reader.count(limits.maximum_adjacent_factors, "adjacent factor count");
  result.adjacent_factors.reserve(adjacent_count);
  for (std::size_t index = 0U; index < adjacent_count && reader.ok(); ++index) {
    AdjacentBoundaryCheckpoint adjacent;
    adjacent.factor = reader.id<GlobalFactorId>();
    adjacent.from_slot = reader.u64();
    adjacent.to_slot = reader.u64();
    adjacent.transition = readSealedTransition(reader, limits);
    result.adjacent_factors.push_back(std::move(adjacent));
  }
  const std::size_t gnss_count = reader.count(limits.maximum_gnss_factors, "GNSS factor count");
  result.gnss_factors.reserve(gnss_count);
  for (std::size_t index = 0U; index < gnss_count && reader.ok(); ++index) {
    result.gnss_factors.push_back(
        GnssFactorCheckpoint{reader.id<GlobalFactorId>(), readGnssConstraint(reader)});
  }
  const std::size_t loop_count = reader.count(limits.maximum_loop_factors, "loop factor count");
  result.loop_factors.reserve(loop_count);
  for (std::size_t index = 0U; index < loop_count && reader.ok(); ++index) {
    LoopFactorCheckpoint loop;
    loop.factor = reader.id<GlobalFactorId>();
    loop.measurement = readLoopMeasurement(reader, limits);
    loop.candidate = reader.id<CandidateId>();
    loop.scale = readGncScale(reader);
    result.loop_factors.push_back(std::move(loop));
  }
  const std::size_t factor_order_count =
      reader.count(1U + limits.maximum_adjacent_factors + limits.maximum_gnss_factors +
                       limits.maximum_loop_factors,
                   "active factor order count");
  result.factor_order.reserve(factor_order_count);
  for (std::size_t index = 0U; index < factor_order_count && reader.ok(); ++index) {
    result.factor_order.push_back(reader.id<GlobalFactorId>());
  }
  if (reader.boolean()) {
    result.alignment = YawTranslation4{reader.fixedVector<3>(), reader.floating()};
  }
  result.next_boundary_slot = reader.u64();
  result.next_factor_id = reader.u64();
  result.next_candidate_id = reader.u64();

  result.recovery.whitened_squared_objective = reader.floating();
  result.recovery.gradient_infinity_norm = reader.floating();
  result.recovery.scalar_dimension = readSize(reader, "recovery scalar dimension");
  result.recovery.numerical_rank = readSize(reader, "recovery numerical rank");
  result.recovery.hessian_condition = reader.floating();
  const std::size_t objective_count =
      reader.count(1U + limits.maximum_adjacent_factors + limits.maximum_gnss_factors +
                       limits.maximum_loop_factors,
                   "factor objective count");
  result.recovery.factor_objectives.reserve(objective_count);
  for (std::size_t index = 0U; index < objective_count && reader.ok(); ++index) {
    result.recovery.factor_objectives.push_back(
        GlobalFactorObjectiveCheckpoint{reader.id<GlobalFactorId>(), reader.floating()});
  }
  const std::size_t marginal_count =
      reader.count(limits.maximum_boundaries, "boundary marginal count");
  result.recovery.boundary_marginals.reserve(marginal_count);
  for (std::size_t index = 0U; index < marginal_count && reader.ok(); ++index) {
    result.recovery.boundary_marginals.push_back(
        BoundaryMarginalCheckpoint{reader.u64(), readPoseCovariance(reader)});
  }
  if (reader.boolean()) {
    AlignmentCovariance covariance;
    covariance.matrix = reader.fixedMatrix<4, 4>();
    covariance.tangent = readEnum(reader, {{1U, AlignmentTangentConvention::TranslationEnuThenYaw}},
                                  AlignmentTangentConvention::TranslationEnuThenYaw);
    result.recovery.alignment_covariance = std::move(covariance);
  }
  result.recovery.map_odom = readMapOdom(reader);
  result.recovery.committed_solve = readSolveReport(reader);
  result.recovery.tolerances = readRecoveryTolerances(reader);
  result.checksum = reader.hash();
  return result;
}

[[nodiscard]] bool validPose(const core::Pose3d& pose) noexcept {
  const Eigen::Matrix4d matrix = pose.matrix();
  if (!matrix.allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = matrix.topLeftCorner<3, 3>();
  return (matrix.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).cwiseAbs().maxCoeff() == 0.0 &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <
             1.0e-10 &&
         std::abs(rotation.determinant() - 1.0) < 1.0e-10;
}

template <typename Matrix>
[[nodiscard]] bool validCovariance(const Matrix& covariance) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-9 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Matrix> solver(0.5 * (covariance + covariance.transpose()),
                                               Eigen::EigenvaluesOnly);
  return solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
         solver.eigenvalues().minCoeff() >= -1.0e-10 * scale;
}

[[nodiscard]] bool exactPose(const core::Pose3d& left, const core::Pose3d& right) noexcept {
  return (left.matrix().array() == right.matrix().array()).all();
}

[[nodiscard]] bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] core::Result<bool, GlobalGraphCheckpointError> validateCheckpointStructure(
    const GlobalGraphCheckpoint& checkpoint, const GlobalGraphCheckpointLimits& limits) {
  using Result = core::Result<bool, GlobalGraphCheckpointError>;
  if (!validLimits(limits)) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidLimits,
                                           "checkpoint limits are invalid"));
  }
  if (checkpoint.schema_version != kGlobalGraphCheckpointSchemaVersion) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::UnsupportedSchema,
                                           "global graph checkpoint schema is unsupported"));
  }
  if (checkpoint.key_schema_version != kGlobalGraphKeySchemaVersion) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::UnsupportedKeySchema,
                                           "global graph key allocation schema is unsupported"));
  }
  if (checkpoint.configuration.schema_version != kGlobalGraphConfigurationSchemaVersion) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::UnsupportedConfigurationSchema,
                        "global graph configuration identity schema is unsupported"));
  }
  if (!checkpoint.mission_session.valid() || !checkpoint.revision.valid() ||
      !core::contentHashPresent(checkpoint.configuration.checksum)) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                        "checkpoint mission, revision, or configuration identity is invalid"));
  }
  if ((checkpoint.revision.value() == 0U && checkpoint.parent.has_value()) ||
      (checkpoint.revision.value() > 0U &&
       (!checkpoint.parent || !checkpoint.parent->valid() ||
        checkpoint.parent->value() + 1U != checkpoint.revision.value()))) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                        "checkpoint revision and parent do not form one canonical chain step"));
  }
  if (checkpoint.boundaries.empty() || checkpoint.boundaries.size() > limits.maximum_boundaries ||
      checkpoint.chart_placements.empty() ||
      checkpoint.chart_placements.size() > limits.maximum_chart_placements ||
      checkpoint.adjacent_factors.size() > limits.maximum_adjacent_factors ||
      checkpoint.gnss_factors.size() > limits.maximum_gnss_factors ||
      checkpoint.loop_factors.size() > limits.maximum_loop_factors) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                                           "checkpoint top-level collection exceeds its bound"));
  }
  if (!checkpoint.mission_gauge.factor.valid() || checkpoint.mission_gauge.boundary_slot != 0U ||
      !validPose(checkpoint.mission_gauge.T_map_submap) ||
      !positiveFinite(checkpoint.mission_gauge.translation_sigma_m) ||
      !positiveFinite(checkpoint.mission_gauge.rotation_sigma_rad)) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "mission gauge record is invalid"));
  }

  std::set<core::OdomEpoch> epochs;
  for (const OdomEpochChartPlacementCheckpoint& placement : checkpoint.chart_placements) {
    if (!placement.odom_epoch.valid() || !validPose(placement.H_map_odom) ||
        !epochs.insert(placement.odom_epoch).second) {
      return Result::failure(
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                          "chart placements contain an invalid, duplicate, or non-finite epoch"));
    }
  }
  if (!std::is_sorted(
          checkpoint.chart_placements.begin(), checkpoint.chart_placements.end(),
          [](const auto& left, const auto& right) { return left.odom_epoch < right.odom_epoch; })) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "chart placements are not in canonical epoch order"));
  }

  std::set<core::SparseSubmapIdentityKey> identities;
  for (std::size_t index = 0U; index < checkpoint.boundaries.size(); ++index) {
    const BoundaryNavigationCheckpoint& boundary = checkpoint.boundaries[index];
    if (boundary.slot != index || boundary.seal.ref.session != checkpoint.mission_session ||
        !validPose(boundary.T_map_submap) ||
        !identities.insert(core::sparseSubmapIdentityKey(boundary.seal.ref)).second ||
        !epochs.contains(boundary.seal.ref.odom_epoch)) {
      return Result::failure(checkpointError(
          GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
          "boundary slots, sessions, identities, estimates, or chart epochs are invalid"));
    }
    const bool velocity = boundary.velocity_map.has_value();
    if (velocity != boundary.gyro_bias.has_value() || velocity != boundary.accel_bias.has_value() ||
        (velocity && (!boundary.velocity_map->allFinite() || !boundary.gyro_bias->allFinite() ||
                      !boundary.accel_bias->allFinite())) ||
        (checkpoint.boundaries.size() > 1U && !velocity)) {
      return Result::failure(
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                          "boundary navigation state is partial, non-finite, or not materialized"));
    }
    auto verified = core::verifyCanonicalSparseSubmapSeal(boundary.seal);
    if (!verified) {
      GlobalGraphCheckpointError error =
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidSparseSeal,
                          "checkpoint boundary seal failed recursive canonical verification");
      error.canonical_verification_error = std::move(verified).error();
      return Result::failure(std::move(error));
    }
    if (index == 0U) {
      if (boundary.seal.previous || boundary.seal.from_previous ||
          !exactPose(checkpoint.mission_gauge.T_map_submap, boundary.seal.T_odom_submap)) {
        return Result::failure(
            checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                            "first boundary or mission gauge target is not canonical"));
      }
    } else if (core::validateSparseSubmapLink(checkpoint.boundaries[index - 1U].seal,
                                              boundary.seal) !=
               core::SparseSubmapLinkValidationError::None) {
      return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidSparseSeal,
                                             "checkpoint boundary chain is not exact"));
    }
  }
  if (checkpoint.next_boundary_slot != checkpoint.boundaries.size()) {
    return Result::failure(checkpointError(
        GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
        "next boundary slot does not follow the canonical contiguous slot sequence"));
  }
  if (checkpoint.adjacent_factors.size() + 1U != checkpoint.boundaries.size()) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                        "adjacent factor count does not span the boundary chain"));
  }

  std::vector<GlobalFactorId> known_factor_ids{checkpoint.mission_gauge.factor};
  known_factor_ids.reserve(1U + checkpoint.adjacent_factors.size() +
                           checkpoint.gnss_factors.size());
  for (std::size_t index = 0U; index < checkpoint.adjacent_factors.size(); ++index) {
    const AdjacentBoundaryCheckpoint& adjacent = checkpoint.adjacent_factors[index];
    const BoundaryNavigationCheckpoint& successor = checkpoint.boundaries[index + 1U];
    if (!adjacent.factor.valid() || adjacent.from_slot != index || adjacent.to_slot != index + 1U ||
        !successor.seal.from_previous ||
        adjacent.transition.checksum != successor.seal.from_previous->checksum ||
        adjacent.transition.from != successor.seal.from_previous->from ||
        adjacent.transition.to != successor.seal.from_previous->to) {
      return Result::failure(checkpointError(
          GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
          "adjacent factor record does not exactly identify its sealed transition"));
    }
    const auto frozen_verified = core::verifyCanonicalFrozenSquareRootFactorChecksum(
        adjacent.transition.local_transition.boundary_factor);
    const auto lineage_verified = core::verifyCanonicalObservationLineageChecksum(
        adjacent.transition.local_transition.lineage);
    const auto condensed_verified = core::verifyCanonicalCondensedBoundaryTransitionChecksum(
        adjacent.transition.local_transition);
    const auto transition_verified =
        core::verifyCanonicalSealedBoundaryTransitionChecksum(adjacent.transition);
    if (!frozen_verified || !lineage_verified || !condensed_verified || !transition_verified) {
      GlobalGraphCheckpointError error =
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidSparseSeal,
                          "adjacent checkpoint transition failed recursive canonical verification");
      if (!frozen_verified) {
        error.canonical_verification_error = frozen_verified.error();
      } else if (!lineage_verified) {
        error.canonical_verification_error = lineage_verified.error();
      } else if (!condensed_verified) {
        error.canonical_verification_error = condensed_verified.error();
      } else {
        error.canonical_verification_error = transition_verified.error();
      }
      return Result::failure(std::move(error));
    }
    known_factor_ids.push_back(adjacent.factor);
  }
  if (!std::is_sorted(
          checkpoint.adjacent_factors.begin(), checkpoint.adjacent_factors.end(),
          [](const auto& left, const auto& right) { return left.factor < right.factor; }) ||
      !std::is_sorted(
          checkpoint.gnss_factors.begin(), checkpoint.gnss_factors.end(),
          [](const auto& left, const auto& right) { return left.factor < right.factor; }) ||
      !std::is_sorted(
          checkpoint.loop_factors.begin(), checkpoint.loop_factors.end(),
          [](const auto& left, const auto& right) { return left.candidate < right.candidate; })) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "typed factors are not in canonical identity order"));
  }

  std::unordered_set<std::uint64_t> observations;
  for (const GnssFactorCheckpoint& gnss : checkpoint.gnss_factors) {
    if (!gnss.factor.valid() || !gnss.constraint.observation.valid() ||
        !observations.insert(gnss.constraint.observation.value()).second ||
        !identities.contains(core::sparseSubmapIdentityKey(gnss.constraint.submap)) ||
        !gnss.constraint.antenna_position_submap.allFinite() ||
        !gnss.constraint.measured_position_enu.allFinite() ||
        !validCovariance(gnss.constraint.effective_covariance_enu)) {
      return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                             "GNSS checkpoint factor is invalid"));
    }
    known_factor_ids.push_back(gnss.factor);
  }
  std::sort(known_factor_ids.begin(), known_factor_ids.end());

  std::vector<GlobalFactorId> expected_order = known_factor_ids;
  std::unordered_set<std::uint64_t> proposals;
  std::unordered_set<std::uint64_t> candidates;
  for (const LoopFactorCheckpoint& loop : checkpoint.loop_factors) {
    if (!loop.factor.valid() || !loop.candidate.valid() || !loop.measurement.proposal.valid() ||
        !proposals.insert(loop.measurement.proposal.value()).second ||
        !candidates.insert(loop.candidate.value()).second ||
        loop.measurement.header.session != checkpoint.mission_session ||
        !identities.contains(core::sparseSubmapIdentityKey(loop.measurement.from)) ||
        !identities.contains(core::sparseSubmapIdentityKey(loop.measurement.to)) ||
        loop.scale.degrees_of_freedom != loop.measurement.information.rank ||
        !positiveFinite(loop.scale.calibrated_chi_squared_cutoff)) {
      return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                             "loop checkpoint factor is invalid"));
    }
    auto lineage_verified =
        core::verifyCanonicalObservationLineageChecksum(loop.measurement.lineage);
    if (!lineage_verified) {
      GlobalGraphCheckpointError error =
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                          "loop checkpoint lineage failed canonical verification");
      error.canonical_verification_error = std::move(lineage_verified).error();
      return Result::failure(std::move(error));
    }
    expected_order.push_back(loop.factor);
  }
  std::unordered_set<std::uint64_t> unique_factors;
  if (std::any_of(expected_order.begin(), expected_order.end(),
                  [&](GlobalFactorId factor) {
                    return !factor.valid() || !unique_factors.insert(factor.value()).second;
                  }) ||
      checkpoint.factor_order != expected_order) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                        "active factor order is duplicate, incomplete, or non-canonical"));
  }
  const auto maximum_factor =
      std::max_element(expected_order.begin(), expected_order.end(),
                       [](GlobalFactorId left, GlobalFactorId right) { return left < right; });
  if (maximum_factor == expected_order.end() ||
      checkpoint.next_factor_id <= maximum_factor->value() ||
      checkpoint.next_factor_id >= GlobalFactorId::kInvalidValue) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "next global factor identity is invalid"));
  }
  if ((!checkpoint.loop_factors.empty() &&
       checkpoint.next_candidate_id <= checkpoint.loop_factors.back().candidate.value()) ||
      checkpoint.next_candidate_id >= CandidateId::kInvalidValue) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "next robust candidate identity is invalid"));
  }
  if (checkpoint.alignment.has_value() != !checkpoint.gnss_factors.empty() ||
      (checkpoint.alignment && (!checkpoint.alignment->translation_enu.allFinite() ||
                                !std::isfinite(checkpoint.alignment->yaw_enu_map_rad)))) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "checkpoint alignment presence or estimate is invalid"));
  }

  const GlobalGraphRecoveryAudit& audit = checkpoint.recovery;
  if (!std::isfinite(audit.whitened_squared_objective) || audit.whitened_squared_objective < 0.0 ||
      !std::isfinite(audit.gradient_infinity_norm) || audit.gradient_infinity_norm < 0.0 ||
      audit.scalar_dimension == 0U || audit.numerical_rank != audit.scalar_dimension ||
      !positiveFinite(audit.hessian_condition) ||
      audit.factor_objectives.size() != checkpoint.factor_order.size() ||
      audit.boundary_marginals.size() != checkpoint.boundaries.size() ||
      audit.alignment_covariance.has_value() != checkpoint.alignment.has_value()) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "checkpoint recovery audit dimensions are invalid"));
  }
  double factor_cost_sum{};
  for (std::size_t index = 0U; index < audit.factor_objectives.size(); ++index) {
    const GlobalFactorObjectiveCheckpoint& objective = audit.factor_objectives[index];
    if (objective.factor != checkpoint.factor_order[index] ||
        !std::isfinite(objective.whitened_squared_cost) || objective.whitened_squared_cost < 0.0) {
      return Result::failure(
          checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                          "per-factor recovery objective is invalid or out of canonical order"));
    }
    factor_cost_sum += objective.whitened_squared_cost;
  }
  const double objective_scale = std::max(1.0, std::abs(audit.whitened_squared_objective));
  if (!std::isfinite(factor_cost_sum) ||
      std::abs(factor_cost_sum - audit.whitened_squared_objective) > 1.0e-10 * objective_scale) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "per-factor objectives do not sum to the total"));
  }
  for (std::size_t index = 0U; index < audit.boundary_marginals.size(); ++index) {
    const BoundaryMarginalCheckpoint& marginal = audit.boundary_marginals[index];
    if (marginal.boundary_slot != index ||
        marginal.covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
        !validCovariance(marginal.covariance.matrix)) {
      return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                             "boundary marginal audit is invalid"));
    }
  }
  if (audit.alignment_covariance &&
      (audit.alignment_covariance->tangent != AlignmentTangentConvention::TranslationEnuThenYaw ||
       !validCovariance(audit.alignment_covariance->matrix))) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "alignment marginal audit is invalid"));
  }
  if (audit.map_odom.graph_revision != checkpoint.revision ||
      !identities.contains(core::sparseSubmapIdentityKey(audit.map_odom.reference_submap)) ||
      !validPose(audit.map_odom.T_map_odom) || !validCovariance(audit.map_odom.covariance.matrix) ||
      audit.committed_solve.anchors != checkpoint.boundaries.size() ||
      audit.committed_solve.adjacent_factors != checkpoint.adjacent_factors.size() ||
      audit.committed_solve.gnss_factors != checkpoint.gnss_factors.size() ||
      audit.committed_solve.loop_factors != checkpoint.loop_factors.size() ||
      audit.committed_solve.scalar_dimension != audit.scalar_dimension ||
      audit.committed_solve.numerical_rank != audit.numerical_rank ||
      !audit.committed_solve.finite || !audit.committed_solve.converged ||
      !audit.committed_solve.connected) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "published recovery audit is internally inconsistent"));
  }
  const GlobalGraphRecoveryTolerances& tolerances = audit.tolerances;
  if (!positiveFinite(tolerances.objective_absolute) ||
      !positiveFinite(tolerances.objective_relative) ||
      !positiveFinite(tolerances.estimate_tangent_absolute) ||
      !positiveFinite(tolerances.gradient_infinity_absolute) ||
      !positiveFinite(tolerances.covariance_absolute) ||
      !positiveFinite(tolerances.covariance_relative) ||
      !positiveFinite(tolerances.condition_relative)) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                                           "recovery tolerances are invalid"));
  }
  return Result::success(true);
}

[[nodiscard]] core::Result<core::ContentHash, GlobalGraphCheckpointError> checkpointChecksum(
    const GlobalGraphCheckpoint& checkpoint, GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<core::ContentHash, GlobalGraphCheckpointError>;
  auto created =
      Writer::create(kCheckpointChecksumDomain, kGlobalGraphCheckpointSchemaVersion, limits);
  if (!created) {
    return Result::failure(std::move(created).error());
  }
  Writer writer = std::move(created).value();
  writeCheckpointFields(writer, checkpoint, false);
  auto encoded = writer.finish();
  if (!encoded) {
    return Result::failure(std::move(encoded).error());
  }
  return Result::success(encoded.value().digest());
}

}  // namespace

namespace checkpoint_internal {

GlobalGraphCheckpointLimits limitsForConfig(const GlobalGraphConfig& config) noexcept {
  GlobalGraphCheckpointLimits limits;
  limits.maximum_boundaries = config.maximum_anchors;
  limits.maximum_chart_placements = config.maximum_anchors;
  limits.maximum_adjacent_factors = config.maximum_adjacent_factors;
  limits.maximum_gnss_factors = config.maximum_gnss_factors;
  limits.maximum_loop_factors = config.maximum_loop_factors;
  limits.maximum_factor_rows = config.maximum_adjacent_factor_rows;
  limits.maximum_factor_coefficients = config.maximum_adjacent_factor_coefficients;
  return limits;
}

core::Result<GlobalGraphConfigurationIdentity, GlobalGraphCheckpointError> configurationIdentity(
    const GlobalGraphConfig& config, GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<GlobalGraphConfigurationIdentity, GlobalGraphCheckpointError>;
  auto created =
      Writer::create(kConfigurationChecksumDomain, kGlobalGraphConfigurationSchemaVersion, limits);
  if (!created) {
    return Result::failure(std::move(created).error());
  }
  Writer writer = std::move(created).value();
  writeConfigurationFields(writer, config);
  auto encoded = writer.finish();
  if (!encoded) {
    return Result::failure(std::move(encoded).error());
  }
  return Result::success(GlobalGraphConfigurationIdentity{kGlobalGraphConfigurationSchemaVersion,
                                                          encoded.value().digest()});
}

GlobalGraphRecoveryTolerances recoveryTolerances(const GlobalGraphConfig& config) noexcept {
  GlobalGraphRecoveryTolerances result;
  result.objective_absolute = std::max(1.0e-9, 10.0 * config.solver_absolute_error_tolerance);
  result.objective_relative = std::max(1.0e-9, 10.0 * config.solver_relative_error_tolerance);
  result.estimate_tangent_absolute = 1.0e-6;
  result.gradient_infinity_absolute =
      std::max(1.0e-6, std::sqrt(config.solver_absolute_error_tolerance));
  result.covariance_absolute = 1.0e-7;
  result.covariance_relative = 1.0e-5;
  result.condition_relative = 1.0e-5;
  return result;
}

core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError> finalize(
    GlobalGraphCheckpoint checkpoint, GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError>;
  auto valid = validateCheckpointStructure(checkpoint, limits);
  if (!valid) {
    return Result::failure(std::move(valid).error());
  }
  auto checksum = checkpointChecksum(checkpoint, limits);
  if (!checksum) {
    return Result::failure(std::move(checksum).error());
  }
  checkpoint.checksum = std::move(checksum).value();
  return Result::success(std::move(checkpoint));
}

core::Result<bool, GlobalGraphCheckpointError> verify(const GlobalGraphCheckpoint& checkpoint,
                                                      GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<bool, GlobalGraphCheckpointError>;
  auto valid = validateCheckpointStructure(checkpoint, limits);
  if (!valid) {
    return Result::failure(std::move(valid).error());
  }
  auto expected = checkpointChecksum(checkpoint, limits);
  if (!expected) {
    return Result::failure(std::move(expected).error());
  }
  if (!core::contentHashPresent(checkpoint.checksum) || expected.value() != checkpoint.checksum) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::ChecksumMismatch,
                                           "global graph checkpoint checksum does not match"));
  }
  return Result::success(true);
}

}  // namespace checkpoint_internal

core::Result<core::CanonicalByteSequence, GlobalGraphCheckpointError> encodeGlobalGraphCheckpoint(
    const GlobalGraphCheckpoint& checkpoint, GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<core::CanonicalByteSequence, GlobalGraphCheckpointError>;
  auto verified = checkpoint_internal::verify(checkpoint, limits);
  if (!verified) {
    return Result::failure(std::move(verified).error());
  }
  auto created = Writer::create(kCheckpointWireDomain, kGlobalGraphCheckpointSchemaVersion, limits);
  if (!created) {
    return Result::failure(std::move(created).error());
  }
  Writer writer = std::move(created).value();
  writeCheckpointFields(writer, checkpoint, true);
  return writer.finish();
}

core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError> decodeGlobalGraphCheckpoint(
    std::span<const std::byte> bytes, GlobalGraphCheckpointLimits limits) {
  using Result = core::Result<GlobalGraphCheckpoint, GlobalGraphCheckpointError>;
  try {
    Reader reader(bytes, limits);
    reader.expectEnvelope(kCheckpointWireDomain, kGlobalGraphCheckpointSchemaVersion);
    GlobalGraphCheckpoint checkpoint = readCheckpointFields(reader, limits);
    if (!reader.ok()) {
      return Result::failure(reader.error());
    }
    if (!reader.finished()) {
      return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::TrailingBytes,
                                             "checkpoint has trailing bytes"));
    }
    auto verified = checkpoint_internal::verify(checkpoint, limits);
    if (!verified) {
      return Result::failure(std::move(verified).error());
    }
    auto canonical = encodeGlobalGraphCheckpoint(checkpoint, limits);
    if (!canonical) {
      return Result::failure(std::move(canonical).error());
    }
    if (!std::ranges::equal(bytes, canonical.value().bytes())) {
      return Result::failure(
          checkpointError(GlobalGraphCheckpointErrorCode::NonCanonicalEncoding,
                          "checkpoint bytes are semantically decodable but not canonical"));
    }
    return Result::success(std::move(checkpoint));
  } catch (const std::bad_alloc&) {
    return Result::failure(checkpointError(GlobalGraphCheckpointErrorCode::CapacityExceeded,
                                           "checkpoint decode allocation failed"));
  } catch (const std::exception& exception) {
    return Result::failure(
        checkpointError(GlobalGraphCheckpointErrorCode::InvalidCheckpoint,
                        std::string("checkpoint decode failed: ") + exception.what()));
  }
}

}  // namespace meridian::global
