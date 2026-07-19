#include "meridian/core/sensor_observations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Eigenvalues>

namespace meridian::core {
namespace {

[[nodiscard]] GnssValidationError validationError(GnssValidationErrorCode code,
                                                   std::string detail) {
  return GnssValidationError{code, std::move(detail)};
}

template <typename Enum>
[[nodiscard]] bool knownEnum(Enum value) noexcept;

template <>
bool knownEnum(PositionCovarianceSource value) noexcept {
  switch (value) {
    case PositionCovarianceSource::ReceiverApproximation:
    case PositionCovarianceSource::ReceiverDiagonal:
    case PositionCovarianceSource::ReceiverFull:
      return true;
  }
  return false;
}

template <>
bool knownEnum(GnssSolutionType value) noexcept {
  switch (value) {
    case GnssSolutionType::Unknown:
    case GnssSolutionType::Autonomous:
    case GnssSolutionType::SbasAugmented:
    case GnssSolutionType::GbasAugmented:
    case GnssSolutionType::Differential:
    case GnssSolutionType::RtkFloat:
    case GnssSolutionType::RtkFixed:
    case GnssSolutionType::PppFloat:
    case GnssSolutionType::PppFixed:
      return true;
  }
  return false;
}

template <>
bool knownEnum(GnssFixAvailability value) noexcept {
  switch (value) {
    case GnssFixAvailability::Unknown:
    case GnssFixAvailability::Unavailable:
    case GnssFixAvailability::Available:
      return true;
  }
  return false;
}

template <>
bool knownEnum(GnssIntegrityStatus value) noexcept {
  switch (value) {
    case GnssIntegrityStatus::Unknown:
    case GnssIntegrityStatus::Valid:
    case GnssIntegrityStatus::Suspect:
    case GnssIntegrityStatus::Invalid:
      return true;
  }
  return false;
}

template <>
bool knownEnum(GnssCorrectionStatus value) noexcept {
  switch (value) {
    case GnssCorrectionStatus::Unknown:
    case GnssCorrectionStatus::NotUsed:
    case GnssCorrectionStatus::Current:
    case GnssCorrectionStatus::Stale:
      return true;
  }
  return false;
}

template <>
bool knownEnum(GnssStatusSource value) noexcept {
  switch (value) {
    case GnssStatusSource::GenericNavSatFix:
    case GnssStatusSource::ReceiverSpecific:
      return true;
  }
  return false;
}

[[nodiscard]] bool validTimeMappingStatus(TimeMappingStatus value) noexcept {
  switch (value) {
    case TimeMappingStatus::Valid:
    case TimeMappingStatus::Estimated:
    case TimeMappingStatus::Discontinuous:
    case TimeMappingStatus::Uncertain:
      return true;
  }
  return false;
}

[[nodiscard]] bool genericNavSatSolution(GnssSolutionType solution) noexcept {
  return solution == GnssSolutionType::Autonomous ||
         solution == GnssSolutionType::SbasAugmented ||
         solution == GnssSolutionType::GbasAugmented;
}

}  // namespace

Result<GeodeticPosition, GnssValidationError> GeodeticPosition::fromWgs84Degrees(
    double latitude_deg, double longitude_deg, double altitude_m) {
  using PositionResult = Result<GeodeticPosition, GnssValidationError>;
  if (!std::isfinite(latitude_deg) || !std::isfinite(longitude_deg) ||
      !std::isfinite(altitude_m)) {
    return PositionResult::failure(
        validationError(GnssValidationErrorCode::NonFiniteGeodeticPosition,
                        "WGS84 latitude, longitude, and altitude must be finite"));
  }
  if (latitude_deg < -90.0 || latitude_deg > 90.0) {
    return PositionResult::failure(
        validationError(GnssValidationErrorCode::LatitudeOutOfRange,
                        "WGS84 latitude must be in [-90, 90] degrees"));
  }
  if (longitude_deg < -180.0 || longitude_deg > 180.0) {
    return PositionResult::failure(
        validationError(GnssValidationErrorCode::LongitudeOutOfRange,
                        "WGS84 longitude must be in [-180, 180] degrees"));
  }
  return PositionResult::success(
      GeodeticPosition(latitude_deg, longitude_deg, altitude_m));
}

Result<PositionCovarianceEnu, GnssValidationError> PositionCovarianceEnu::fromMatrix(
    const Eigen::Matrix3d& matrix, PositionCovarianceSource source) {
  using CovarianceResult = Result<PositionCovarianceEnu, GnssValidationError>;
  if (!knownEnum(source)) {
    return CovarianceResult::failure(
        validationError(GnssValidationErrorCode::InvalidCovarianceSource,
                        "ENU covariance source is not a supported domain value"));
  }
  if (!matrix.allFinite()) {
    return CovarianceResult::failure(
        validationError(GnssValidationErrorCode::NonFiniteCovariance,
                        "ENU covariance must contain only finite values"));
  }

  const double scale = std::max(1.0, matrix.cwiseAbs().maxCoeff());
  const double symmetry_tolerance = 1.0e-10 * scale;
  if ((matrix - matrix.transpose()).cwiseAbs().maxCoeff() > symmetry_tolerance) {
    return CovarianceResult::failure(
        validationError(GnssValidationErrorCode::NonSymmetricCovariance,
                        "ENU covariance must be symmetric"));
  }

  Eigen::Matrix3d canonical = 0.5 * (matrix + matrix.transpose());
  if (source == PositionCovarianceSource::ReceiverDiagonal) {
    Eigen::Matrix3d off_diagonal = canonical;
    off_diagonal.diagonal().setZero();
    if (off_diagonal.cwiseAbs().maxCoeff() > symmetry_tolerance) {
      return CovarianceResult::failure(
          validationError(GnssValidationErrorCode::NonDiagonalCovariance,
                          "a receiver-diagonal covariance cannot contain cross terms"));
    }
    canonical = canonical.diagonal().asDiagonal();
  }

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(canonical);
  const double psd_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  if (eigen_solver.info() != Eigen::Success || !eigen_solver.eigenvalues().allFinite() ||
      eigen_solver.eigenvalues().minCoeff() < -psd_tolerance) {
    return CovarianceResult::failure(
        validationError(GnssValidationErrorCode::NonPositiveSemidefiniteCovariance,
                        "ENU covariance must be positive semidefinite"));
  }
  if (eigen_solver.eigenvalues().minCoeff() < 0.0) {
    // A covariance admitted through the numerical tolerance must still be PSD
    // when stored. Clamp only tolerance-sized round-off modes and reconstruct
    // in the original ENU basis; materially negative modes were rejected above.
    const Eigen::Vector3d nonnegative_eigenvalues =
        eigen_solver.eigenvalues().cwiseMax(0.0);
    canonical = eigen_solver.eigenvectors() * nonnegative_eigenvalues.asDiagonal() *
                eigen_solver.eigenvectors().transpose();
    canonical = 0.5 * (canonical + canonical.transpose());
  }
  return CovarianceResult::success(PositionCovarianceEnu(std::move(canonical), source));
}

Result<GnssServiceSet, GnssValidationError> GnssServiceSet::fromMask(std::uint16_t mask) {
  using ServiceResult = Result<GnssServiceSet, GnssValidationError>;
  constexpr std::uint16_t kKnownMask =
      static_cast<std::uint16_t>(GnssConstellationService::Gps) |
      static_cast<std::uint16_t>(GnssConstellationService::Glonass) |
      static_cast<std::uint16_t>(GnssConstellationService::Compass) |
      static_cast<std::uint16_t>(GnssConstellationService::Galileo);
  if (mask == 0U || (mask & static_cast<std::uint16_t>(~kKnownMask)) != 0U) {
    return ServiceResult::failure(
        validationError(GnssValidationErrorCode::InvalidServiceMask,
                        "GNSS service mask must contain only known non-zero constellation bits"));
  }
  return ServiceResult::success(GnssServiceSet(mask));
}

Result<bool, GnssValidationError> validateGnssObservation(
    const GnssObservation& observation) {
  using ValidationResult = Result<bool, GnssValidationError>;
  const RecordHeader& header = observation.header;
  if (header.schema_version == 0U || !header.trace.valid() || !header.producer.valid() ||
      !header.session.valid() || !header.config.valid() || !header.direct_calibration ||
      !header.direct_calibration->valid() || !observation.id.valid()) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidIdentity,
                        "GNSS record identity and direct calibration must be valid"));
  }

  const SourceStamp& stamp = observation.stamp;
  if (!stamp.clock_revision.valid() || !stamp.source_epoch.valid() ||
      !stamp.ingress_sequence.valid() ||
      (stamp.device_sequence && !stamp.device_sequence->valid()) ||
      stamp.uncertainty.nanoseconds < 0 || !validTimeMappingStatus(stamp.status)) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidTimestampMetadata,
                        "GNSS source timestamp metadata is invalid"));
  }
  if (!knownEnum(observation.solution)) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidSolution,
                        "GNSS solution is not a supported domain value"));
  }
  if (!knownEnum(observation.status.fix) || !knownEnum(observation.status.integrity) ||
      !knownEnum(observation.status.corrections) || !knownEnum(observation.status.source)) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidStatus,
                        "GNSS status contains an unsupported domain value"));
  }
  if (observation.status.fix != GnssFixAvailability::Available) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidStatus,
                        "an admitted GNSS position observation must report an available fix"));
  }

  if (observation.status.source == GnssStatusSource::GenericNavSatFix &&
      (!genericNavSatSolution(observation.solution) ||
       observation.status.integrity != GnssIntegrityStatus::Unknown ||
       observation.status.corrections != GnssCorrectionStatus::Unknown ||
       observation.correction_age || observation.hdop || observation.vdop ||
       observation.satellites)) {
    return ValidationResult::failure(validationError(
        GnssValidationErrorCode::InconsistentSourceSemantics,
        "generic NavSatFix metadata cannot claim receiver integrity, corrections, DOP, "
        "satellite count, RTK, or PPP state"));
  }

  if (observation.correction_age && observation.correction_age->nanoseconds < 0) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidCorrectionAge,
                        "GNSS correction age cannot be negative"));
  }
  if (observation.correction_age &&
      observation.status.corrections != GnssCorrectionStatus::Current &&
      observation.status.corrections != GnssCorrectionStatus::Stale) {
    return ValidationResult::failure(validationError(
        GnssValidationErrorCode::InconsistentSourceSemantics,
        "GNSS correction age requires a current or stale correction status"));
  }
  const auto invalid_dop = [](const std::optional<double>& dop) {
    return dop && (!std::isfinite(*dop) || *dop < 0.0);
  };
  if (invalid_dop(observation.hdop) || invalid_dop(observation.vdop)) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidDilutionOfPrecision,
                        "GNSS dilution-of-precision values must be finite and non-negative"));
  }
  if (observation.satellites && *observation.satellites == 0U) {
    return ValidationResult::failure(
        validationError(GnssValidationErrorCode::InvalidSatelliteCount,
                        "a known GNSS satellite count must be non-zero"));
  }
  return ValidationResult::success(true);
}

}  // namespace meridian::core
