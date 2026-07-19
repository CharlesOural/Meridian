#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "meridian/core/result.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::core {

struct ImuSample {
  RecordHeader header;
  MeasurementId id;
  SourceStamp stamp;
  Eigen::Vector3d specific_force_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_radps{Eigen::Vector3d::Zero()};
  bool saturated{};
};

struct LidarPoint {
  float x{};
  float y{};
  float z{};
  float intensity{};
  std::int32_t time_offset_ns{};
  std::uint16_t ring{};
  std::uint32_t source_index{};
};

using LidarPoints = std::vector<LidarPoint>;

struct LidarLayout {
  std::uint32_t width{};
  std::uint32_t height{};
  bool organized{};

  [[nodiscard]] std::size_t sourcePointCount() const noexcept {
    return static_cast<std::size_t>(width) * height;
  }
  [[nodiscard]] bool containsSourceIndex(std::uint32_t index) const noexcept {
    return width > 0U && height > 0U && index < sourcePointCount();
  }
};

struct LidarSweep {
  RecordHeader header;
  MeasurementId id;
  LidarId lidar;
  SourceStamp stamp;
  TimeRange acquisition;
  LidarLayout layout;
  std::shared_ptr<const LidarPoints> points;
};

enum class ImageEncoding {
  Mono8,
  Bgr8,
  Jpeg,
};

struct CameraFrame {
  RecordHeader header;
  MeasurementId id;
  CameraId camera;
  SourceStamp stamp;
  FusionTime exposure_midpoint;
  Duration exposure;
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t stride{};
  ImageEncoding encoding{ImageEncoding::Mono8};
  std::shared_ptr<const std::vector<std::byte>> pixels;
};

enum class GnssValidationErrorCode {
  NonFiniteGeodeticPosition,
  LatitudeOutOfRange,
  LongitudeOutOfRange,
  InvalidCovarianceSource,
  NonFiniteCovariance,
  NonSymmetricCovariance,
  NonPositiveSemidefiniteCovariance,
  NonDiagonalCovariance,
  InvalidServiceMask,
  InvalidIdentity,
  InvalidTimestampMetadata,
  InvalidSolution,
  InvalidStatus,
  InconsistentSourceSemantics,
  InvalidCorrectionAge,
  InvalidDilutionOfPrecision,
  InvalidSatelliteCount,
};

struct GnssValidationError {
  GnssValidationErrorCode code{GnssValidationErrorCode::NonFiniteGeodeticPosition};
  std::string detail;
};

// This value type represents WGS84 ellipsoidal latitude, longitude, and
// altitude only. A future non-WGS84 datum must use a different type or an
// explicit conversion API; it cannot be relabelled as GeodeticPosition.
class GeodeticPosition {
 public:
  [[nodiscard]] static Result<GeodeticPosition, GnssValidationError> fromWgs84Degrees(
      double latitude_deg, double longitude_deg, double altitude_m);

  [[nodiscard]] double latitudeDeg() const noexcept { return latitude_deg_; }
  [[nodiscard]] double longitudeDeg() const noexcept { return longitude_deg_; }
  [[nodiscard]] double altitudeM() const noexcept { return altitude_m_; }

 private:
  GeodeticPosition(double latitude_deg, double longitude_deg, double altitude_m)
      : latitude_deg_(latitude_deg), longitude_deg_(longitude_deg), altitude_m_(altitude_m) {}

  double latitude_deg_{};
  double longitude_deg_{};
  double altitude_m_{};
};

// Describes what the producer actually reported about the ENU covariance.
// Unknown covariance is deliberately not representable because the covariance
// is mandatory on an admitted GnssObservation.
enum class PositionCovarianceSource {
  ReceiverApproximation,
  ReceiverDiagonal,
  ReceiverFull,
};

class PositionCovarianceEnu {
 public:
  [[nodiscard]] static Result<PositionCovarianceEnu, GnssValidationError> fromMatrix(
      const Eigen::Matrix3d& matrix, PositionCovarianceSource source);

  [[nodiscard]] const Eigen::Matrix3d& matrix() const noexcept { return matrix_; }
  [[nodiscard]] PositionCovarianceSource source() const noexcept { return source_; }

 private:
  PositionCovarianceEnu(Eigen::Matrix3d matrix, PositionCovarianceSource source)
      : matrix_(std::move(matrix)), source_(source) {}

  Eigen::Matrix3d matrix_{Eigen::Matrix3d::Zero()};
  PositionCovarianceSource source_{PositionCovarianceSource::ReceiverApproximation};
};

// Receiver-specific adapters may populate the detailed corrected-solution
// states. A generic NavSatFix adapter is restricted to the first three
// concrete states and must keep SBAS/GBAS distinct from RTK or PPP.
enum class GnssSolutionType {
  Unknown,
  Autonomous,
  SbasAugmented,
  GbasAugmented,
  Differential,
  RtkFloat,
  RtkFixed,
  PppFloat,
  PppFixed,
};

enum class GnssFixAvailability {
  Unknown,
  Unavailable,
  Available,
};

enum class GnssIntegrityStatus {
  Unknown,
  Valid,
  Suspect,
  Invalid,
};

enum class GnssCorrectionStatus {
  Unknown,
  NotUsed,
  Current,
  Stale,
};

enum class GnssStatusSource {
  GenericNavSatFix,
  ReceiverSpecific,
};

enum class GnssConstellationService : std::uint16_t {
  Gps = 1U << 0U,
  Glonass = 1U << 1U,
  Compass = 1U << 2U,
  Galileo = 1U << 3U,
};

class GnssServiceSet {
 public:
  [[nodiscard]] static Result<GnssServiceSet, GnssValidationError> fromMask(
      std::uint16_t mask);

  [[nodiscard]] std::uint16_t mask() const noexcept { return mask_; }
  [[nodiscard]] bool contains(GnssConstellationService service) const noexcept {
    const auto bit = static_cast<std::uint16_t>(service);
    return (mask_ & bit) == bit;
  }

 private:
  explicit GnssServiceSet(std::uint16_t mask) : mask_(mask) {}
  std::uint16_t mask_{};
};

struct GnssStatus {
  GnssFixAvailability fix{GnssFixAvailability::Unknown};
  GnssIntegrityStatus integrity{GnssIntegrityStatus::Unknown};
  GnssCorrectionStatus corrections{GnssCorrectionStatus::Unknown};
  GnssStatusSource source{GnssStatusSource::ReceiverSpecific};
  // nullopt means that the producer did not report constellation services.
  std::optional<GnssServiceSet> services;
};

struct GnssObservation {
  RecordHeader header;
  GnssObservationId id;
  SourceStamp stamp;
  GeodeticPosition wgs84;
  PositionCovarianceEnu covariance;
  GnssSolutionType solution{GnssSolutionType::Unknown};
  std::optional<Duration> correction_age;
  std::optional<double> hdop;
  std::optional<double> vdop;
  std::optional<std::uint16_t> satellites;
  GnssStatus status;
};

// Required at every ingress boundary after assembling the record. The two
// value types above already guarantee their local invariants; this check binds
// identity, timing, optional quality fields, and source-specific semantics.
[[nodiscard]] Result<bool, GnssValidationError> validateGnssObservation(
    const GnssObservation& observation);

}  // namespace meridian::core
