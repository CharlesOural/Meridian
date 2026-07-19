#include "meridian/global/gnss.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <utility>

namespace meridian::global {
namespace {

constexpr double kWgs84SemiMajorM = 6378137.0;
constexpr double kWgs84InverseFlattening = 298.257223563;
constexpr double kWgs84Flattening = 1.0 / kWgs84InverseFlattening;
constexpr double kWgs84SemiMinorM = kWgs84SemiMajorM * (1.0 - kWgs84Flattening);
constexpr double kWgs84EccentricitySquared = kWgs84Flattening * (2.0 - kWgs84Flattening);
constexpr double kWgs84SecondEccentricitySquared =
    kWgs84EccentricitySquared / (1.0 - kWgs84EccentricitySquared);
constexpr double kNanosecondsPerSecond = 1.0e9;

[[nodiscard]] bool finite(double value) noexcept {
  return std::isfinite(value);
}

[[nodiscard]] double normalizeAngle(double angle) noexcept {
  return std::remainder(angle, 2.0 * std::numbers::pi);
}

[[nodiscard]] Eigen::Matrix3d rotationZ(double yaw) noexcept {
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  Eigen::Matrix3d rotation;
  rotation << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return rotation;
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) noexcept {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return matrix;
}

[[nodiscard]] bool validCovariance(const Eigen::Matrix3d& covariance) {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-10 * scale) {
    return false;
  }
  Eigen::LLT<Eigen::Matrix3d> llt(0.5 * (covariance + covariance.transpose()));
  return llt.info() == Eigen::Success;
}

[[nodiscard]] Eigen::Matrix3d covarianceInformation(const Eigen::Matrix3d& covariance) {
  return covariance.ldlt().solve(Eigen::Matrix3d::Identity());
}

[[nodiscard]] double huberWeight(double residual_sigma, double threshold) noexcept {
  if (residual_sigma <= threshold || residual_sigma <= std::numeric_limits<double>::epsilon()) {
    return 1.0;
  }
  return threshold / residual_sigma;
}

struct PreparedCorrespondence {
  Eigen::Vector3d map;
  Eigen::Vector3d enu;
  Eigen::Matrix3d information;
  double scalar_weight{};
};

[[nodiscard]] double weightedMedian(std::vector<std::pair<double, double>> values) {
  std::stable_sort(values.begin(), values.end(),
                   [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  double total_weight = 0.0;
  for (const auto& [value, weight] : values) {
    (void)value;
    total_weight += weight;
  }
  const double midpoint = 0.5 * total_weight;
  double accumulated = 0.0;
  for (const auto& [value, weight] : values) {
    accumulated += weight;
    if (accumulated >= midpoint) {
      return value;
    }
  }
  return values.back().first;
}

[[nodiscard]] Eigen::Vector3d robustTranslationForYaw(
    std::span<const PreparedCorrespondence> correspondences, double yaw) {
  const Eigen::Matrix3d rotation = rotationZ(yaw);
  Eigen::Vector3d translation;
  for (Eigen::Index axis = 0; axis < 3; ++axis) {
    std::vector<std::pair<double, double>> values;
    values.reserve(correspondences.size());
    for (const auto& correspondence : correspondences) {
      values.emplace_back((correspondence.enu - rotation * correspondence.map)(axis),
                          correspondence.scalar_weight);
    }
    translation(axis) = weightedMedian(std::move(values));
  }
  return translation;
}

struct AlignmentLinearization {
  Eigen::Matrix4d normal{Eigen::Matrix4d::Zero()};
  Eigen::Vector4d gradient{Eigen::Vector4d::Zero()};
  double weighted_squared_error{};
  double maximum_residual_sigma{};
  std::size_t inliers{};
};

[[nodiscard]] AlignmentLinearization linearizeAlignment(
    std::span<const PreparedCorrespondence> correspondences, double yaw,
    const Eigen::Vector3d& translation, double huber_threshold) {
  AlignmentLinearization result;
  const Eigen::Matrix3d rotation = rotationZ(yaw);
  for (const auto& correspondence : correspondences) {
    const Eigen::Vector3d rotated = rotation * correspondence.map;
    const Eigen::Vector3d residual = rotated + translation - correspondence.enu;
    const double squared_sigma = std::max(0.0, residual.dot(correspondence.information * residual));
    const double residual_sigma = std::sqrt(squared_sigma);
    const double robust_weight = huberWeight(residual_sigma, huber_threshold);

    Eigen::Matrix<double, 3, 4> jacobian;
    jacobian.leftCols<3>().setIdentity();
    jacobian.col(3) = Eigen::Vector3d{-rotated.y(), rotated.x(), 0.0};
    const Eigen::Matrix3d weighted_information = robust_weight * correspondence.information;
    result.normal.noalias() += jacobian.transpose() * weighted_information * jacobian;
    result.gradient.noalias() += jacobian.transpose() * weighted_information * residual;
    result.weighted_squared_error += robust_weight * squared_sigma;
    result.maximum_residual_sigma = std::max(result.maximum_residual_sigma, residual_sigma);
    if (residual_sigma <= huber_threshold) {
      ++result.inliers;
    }
  }
  return result;
}

[[nodiscard]] std::optional<double> matrixCondition(const Eigen::Matrix4d& matrix) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(matrix, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return std::nullopt;
  }
  const double smallest = solver.eigenvalues().minCoeff();
  const double largest = solver.eigenvalues().maxCoeff();
  if (!(smallest > 0.0) || !(largest > 0.0)) {
    return std::nullopt;
  }
  return largest / smallest;
}

[[nodiscard]] bool validAlignmentConfig(const AlignmentConfig& config) noexcept {
  return config.minimum_correspondences >= 2U && finite(config.minimum_time_span_s) &&
         config.minimum_time_span_s >= 0.0 && finite(config.minimum_horizontal_rms_m) &&
         config.minimum_horizontal_rms_m > 0.0 && finite(config.minimum_pair_baseline_m) &&
         config.minimum_pair_baseline_m > 0.0 && finite(config.huber_threshold_sigma) &&
         config.huber_threshold_sigma > 0.0 && config.maximum_iterations > 0U &&
         finite(config.yaw_convergence_rad) && config.yaw_convergence_rad > 0.0 &&
         finite(config.translation_convergence_m) && config.translation_convergence_m > 0.0 &&
         finite(config.maximum_normal_condition) && config.maximum_normal_condition > 1.0;
}

[[nodiscard]] bool validHealthConfig(const GnssHealthConfig& config) noexcept {
  return config.consistent_samples_for_admission > 0U && config.minimum_satellites > 0U &&
         finite(config.maximum_position_std_m) && config.maximum_position_std_m > 0.0 &&
         finite(config.healthy_innovation_chi2) && config.healthy_innovation_chi2 > 0.0 &&
         finite(config.reacquisition_consistency_chi2) &&
         config.reacquisition_consistency_chi2 > 0.0 && finite(config.outage_timeout_s) &&
         config.outage_timeout_s > 0.0 && finite(config.maximum_reacquisition_sample_gap_s) &&
         config.maximum_reacquisition_sample_gap_s > 0.0;
}

}  // namespace

core::Result<Wgs84Lla, GeodesyError> Wgs84Lla::fromRadians(double latitude_rad,
                                                           double longitude_rad,
                                                           double altitude_m) {
  if (!finite(latitude_rad) || !finite(longitude_rad) || !finite(altitude_m)) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::NonFiniteInput});
  }
  if (latitude_rad < -0.5 * std::numbers::pi || latitude_rad > 0.5 * std::numbers::pi) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::LatitudeOutOfRange});
  }
  if (longitude_rad < -std::numbers::pi || longitude_rad > std::numbers::pi) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::LongitudeOutOfRange});
  }
  return core::Result<Wgs84Lla, GeodesyError>::success(
      Wgs84Lla(latitude_rad, longitude_rad, altitude_m));
}

core::Result<Wgs84Lla, GeodesyError> Wgs84Lla::fromDegrees(double latitude_deg,
                                                           double longitude_deg,
                                                           double altitude_m) {
  if (!finite(latitude_deg) || !finite(longitude_deg) || !finite(altitude_m)) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::NonFiniteInput});
  }
  constexpr double kRadiansPerDegree = std::numbers::pi / 180.0;
  return fromRadians(latitude_deg * kRadiansPerDegree, longitude_deg * kRadiansPerDegree,
                     altitude_m);
}

double Wgs84Lla::latitudeDeg() const noexcept {
  return latitude_rad_ * 180.0 / std::numbers::pi;
}

double Wgs84Lla::longitudeDeg() const noexcept {
  return longitude_rad_ * 180.0 / std::numbers::pi;
}

core::Result<EcefPosition, GeodesyError> EcefPosition::fromMeters(const Eigen::Vector3d& meters) {
  if (!meters.allFinite()) {
    return core::Result<EcefPosition, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::NonFiniteInput});
  }
  return core::Result<EcefPosition, GeodesyError>::success(EcefPosition(meters));
}

core::Result<EnuPosition, GeodesyError> EnuPosition::fromMeters(const Eigen::Vector3d& meters) {
  if (!meters.allFinite()) {
    return core::Result<EnuPosition, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::NonFiniteInput});
  }
  return core::Result<EnuPosition, GeodesyError>::success(EnuPosition(meters));
}

core::Result<EcefPosition, GeodesyError> wgs84LlaToEcef(const Wgs84Lla& lla) {
  const double sin_latitude = std::sin(lla.latitudeRad());
  const double cos_latitude = std::cos(lla.latitudeRad());
  const double sin_longitude = std::sin(lla.longitudeRad());
  const double cos_longitude = std::cos(lla.longitudeRad());
  const double prime_vertical_radius =
      kWgs84SemiMajorM / std::sqrt(1.0 - kWgs84EccentricitySquared * sin_latitude * sin_latitude);
  const double xy_radius = prime_vertical_radius + lla.altitudeM();
  Eigen::Vector3d ecef;
  ecef.x() = xy_radius * cos_latitude * cos_longitude;
  ecef.y() = xy_radius * cos_latitude * sin_longitude;
  ecef.z() =
      (prime_vertical_radius * (1.0 - kWgs84EccentricitySquared) + lla.altitudeM()) * sin_latitude;
  return EcefPosition::fromMeters(ecef);
}

core::Result<Wgs84Lla, GeodesyError> wgs84EcefToLla(const EcefPosition& ecef) {
  const Eigen::Vector3d& point = ecef.meters();
  const double horizontal = std::hypot(point.x(), point.y());
  if (point.norm() < 1.0) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::UndefinedAtEarthCenter});
  }

  if (horizontal < 1.0e-9) {
    const double latitude = std::copysign(0.5 * std::numbers::pi, point.z());
    return Wgs84Lla::fromRadians(latitude, 0.0, std::abs(point.z()) - kWgs84SemiMinorM);
  }

  const double longitude = std::atan2(point.y(), point.x());
  const double bowring_angle =
      std::atan2(point.z() * kWgs84SemiMajorM, horizontal * kWgs84SemiMinorM);
  const double sin_bowring = std::sin(bowring_angle);
  const double cos_bowring = std::cos(bowring_angle);
  double latitude = std::atan2(point.z() + kWgs84SecondEccentricitySquared * kWgs84SemiMinorM *
                                               sin_bowring * sin_bowring * sin_bowring,
                               horizontal - kWgs84EccentricitySquared * kWgs84SemiMajorM *
                                                cos_bowring * cos_bowring * cos_bowring);

  bool converged = false;
  double altitude = 0.0;
  for (std::size_t iteration = 0; iteration < 15U; ++iteration) {
    const double sin_latitude = std::sin(latitude);
    const double cos_latitude = std::cos(latitude);
    const double prime_vertical_radius =
        kWgs84SemiMajorM / std::sqrt(1.0 - kWgs84EccentricitySquared * sin_latitude * sin_latitude);
    altitude = horizontal / cos_latitude - prime_vertical_radius;
    const double next_latitude = std::atan2(
        point.z(), horizontal * (1.0 - kWgs84EccentricitySquared * prime_vertical_radius /
                                           (prime_vertical_radius + altitude)));
    if (std::abs(next_latitude - latitude) < 1.0e-14) {
      latitude = next_latitude;
      converged = true;
      break;
    }
    latitude = next_latitude;
  }
  if (!converged) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(
        GeodesyError{GeodesyErrorCode::InverseDidNotConverge});
  }

  const double sin_latitude = std::sin(latitude);
  const double prime_vertical_radius =
      kWgs84SemiMajorM / std::sqrt(1.0 - kWgs84EccentricitySquared * sin_latitude * sin_latitude);
  altitude = horizontal / std::cos(latitude) - prime_vertical_radius;
  return Wgs84Lla::fromRadians(latitude, longitude, altitude);
}

core::Result<LocalEnuDatum, GeodesyError> LocalEnuDatum::create(const Wgs84Lla& origin) {
  auto origin_ecef = wgs84LlaToEcef(origin);
  if (!origin_ecef) {
    return core::Result<LocalEnuDatum, GeodesyError>::failure(origin_ecef.error());
  }
  const double sin_latitude = std::sin(origin.latitudeRad());
  const double cos_latitude = std::cos(origin.latitudeRad());
  const double sin_longitude = std::sin(origin.longitudeRad());
  const double cos_longitude = std::cos(origin.longitudeRad());
  Eigen::Matrix3d rotation;
  rotation << -sin_longitude, cos_longitude, 0.0, -sin_latitude * cos_longitude,
      -sin_latitude * sin_longitude, cos_latitude, cos_latitude * cos_longitude,
      cos_latitude * sin_longitude, sin_latitude;
  return core::Result<LocalEnuDatum, GeodesyError>::success(
      LocalEnuDatum(origin, std::move(origin_ecef).value(), rotation));
}

core::Result<EnuPosition, GeodesyError> LocalEnuDatum::toEnu(const EcefPosition& ecef) const {
  return EnuPosition::fromMeters(R_enu_ecef_ * (ecef.meters() - origin_ecef_.meters()));
}

core::Result<EcefPosition, GeodesyError> LocalEnuDatum::toEcef(const EnuPosition& enu) const {
  return EcefPosition::fromMeters(origin_ecef_.meters() + R_enu_ecef_.transpose() * enu.meters());
}

core::Result<EnuPosition, GeodesyError> LocalEnuDatum::toEnu(const Wgs84Lla& lla) const {
  auto ecef = wgs84LlaToEcef(lla);
  if (!ecef) {
    return core::Result<EnuPosition, GeodesyError>::failure(ecef.error());
  }
  return toEnu(ecef.value());
}

core::Result<Wgs84Lla, GeodesyError> LocalEnuDatum::toLla(const EnuPosition& enu) const {
  auto ecef = toEcef(enu);
  if (!ecef) {
    return core::Result<Wgs84Lla, GeodesyError>::failure(ecef.error());
  }
  return wgs84EcefToLla(ecef.value());
}

core::Result<AntennaLeverArm, GnssModelError> AntennaLeverArm::fromImuToAntenna(
    const Eigen::Vector3d& r_imu_antenna_in_imu) {
  if (!r_imu_antenna_in_imu.allFinite()) {
    return core::Result<AntennaLeverArm, GnssModelError>::failure(
        GnssModelError{GnssModelErrorCode::NonFiniteInput});
  }
  return core::Result<AntennaLeverArm, GnssModelError>::success(
      AntennaLeverArm(r_imu_antenna_in_imu));
}

core::Result<GravityAlignedTransform, GnssModelError> GravityAlignedTransform::create(
    double yaw_enu_map_rad, const Eigen::Vector3d& translation_enu) {
  if (!finite(yaw_enu_map_rad) || !translation_enu.allFinite()) {
    return core::Result<GravityAlignedTransform, GnssModelError>::failure(
        GnssModelError{GnssModelErrorCode::NonFiniteInput});
  }
  return core::Result<GravityAlignedTransform, GnssModelError>::success(
      GravityAlignedTransform(normalizeAngle(yaw_enu_map_rad), translation_enu));
}

Eigen::Matrix3d GravityAlignedTransform::REnuMap() const noexcept {
  return rotationZ(yaw_enu_map_rad_);
}

Eigen::Vector3d GravityAlignedTransform::apply(const Eigen::Vector3d& position_map) const noexcept {
  return REnuMap() * position_map + translation_enu_;
}

Eigen::Vector3d GravityAlignedTransform::inverseApply(
    const Eigen::Vector3d& position_enu) const noexcept {
  return REnuMap().transpose() * (position_enu - translation_enu_);
}

core::Result<AntennaPrediction, GnssModelError> predictAntennaPosition(
    const core::Pose3d& T_map_imu, const AntennaLeverArm& lever_arm,
    const GravityAlignedTransform& T_enu_map) {
  if (!T_map_imu.matrix().allFinite()) {
    return core::Result<AntennaPrediction, GnssModelError>::failure(
        GnssModelError{GnssModelErrorCode::InvalidTransform});
  }
  const Eigen::Matrix3d R_map_imu = T_map_imu.so3().matrix();
  const Eigen::Vector3d& lever = lever_arm.imuToAntennaInImu();
  const Eigen::Vector3d position_map = T_map_imu.translation() + R_map_imu * lever;
  const Eigen::Matrix3d R_enu_map = T_enu_map.REnuMap();
  const Eigen::Vector3d position_enu = T_enu_map.apply(position_map);

  Eigen::Matrix<double, 3, 6> J_map_pose;
  J_map_pose.leftCols<3>() = R_map_imu;
  J_map_pose.rightCols<3>() = -R_map_imu * skew(lever);
  const Eigen::Matrix<double, 3, 6> J_enu_pose = R_enu_map * J_map_pose;

  Eigen::Matrix<double, 3, 4> J_enu_alignment;
  const Eigen::Vector3d rotated = R_enu_map * position_map;
  J_enu_alignment.leftCols<3>().setIdentity();
  J_enu_alignment.col(3) = Eigen::Vector3d{-rotated.y(), rotated.x(), 0.0};

  return core::Result<AntennaPrediction, GnssModelError>::success(
      AntennaPrediction(position_map, position_enu, J_enu_pose, J_enu_alignment));
}

core::Result<AlignmentEstimate, AlignmentError> estimateGravityAlignedTransform(
    std::span<const AlignmentCorrespondence> correspondences, const AlignmentConfig& config) {
  if (!validAlignmentConfig(config)) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::InvalidConfiguration, 0U});
  }
  if (correspondences.size() < config.minimum_correspondences) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::TooFewCorrespondences, correspondences.size()});
  }

  std::vector<PreparedCorrespondence> prepared;
  prepared.reserve(correspondences.size());
  for (std::size_t index = 0; index < correspondences.size(); ++index) {
    const auto& correspondence = correspondences[index];
    if (!correspondence.antenna_position_map.allFinite() ||
        !correspondence.antenna_position_enu.allFinite()) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::NonFiniteInput, index});
    }
    if (!validCovariance(correspondence.covariance_enu)) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::InvalidCovariance, index});
    }
    if (index > 0U && correspondence.stamp <= correspondences[index - 1U].stamp) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::NonMonotonicTimestamps, index});
    }
    const double mean_variance = correspondence.covariance_enu.trace() / 3.0;
    prepared.push_back(PreparedCorrespondence{
        correspondence.antenna_position_map, correspondence.antenna_position_enu,
        covarianceInformation(correspondence.covariance_enu), 1.0 / mean_variance});
  }

  const double time_span_s =
      static_cast<double>(
          (correspondences.back().stamp - correspondences.front().stamp).nanoseconds) /
      kNanosecondsPerSecond;
  if (time_span_s < config.minimum_time_span_s) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::InsufficientTimeSpan, 0U});
  }

  double total_weight = 0.0;
  Eigen::Vector2d map_centroid = Eigen::Vector2d::Zero();
  Eigen::Vector2d enu_centroid = Eigen::Vector2d::Zero();
  for (const auto& correspondence : prepared) {
    total_weight += correspondence.scalar_weight;
    map_centroid.noalias() += correspondence.scalar_weight * correspondence.map.head<2>();
    enu_centroid.noalias() += correspondence.scalar_weight * correspondence.enu.head<2>();
  }
  map_centroid /= total_weight;
  enu_centroid /= total_weight;

  double horizontal_scatter = 0.0;
  double procrustes_cosine = 0.0;
  double procrustes_sine = 0.0;
  for (const auto& correspondence : prepared) {
    const Eigen::Vector2d centered_map = correspondence.map.head<2>() - map_centroid;
    const Eigen::Vector2d centered_enu = correspondence.enu.head<2>() - enu_centroid;
    horizontal_scatter += correspondence.scalar_weight * centered_map.squaredNorm();
    procrustes_cosine += correspondence.scalar_weight * centered_map.dot(centered_enu);
    procrustes_sine += correspondence.scalar_weight *
                       (centered_map.x() * centered_enu.y() - centered_map.y() * centered_enu.x());
  }
  const double horizontal_rms = std::sqrt(horizontal_scatter / total_weight);
  if (!finite(horizontal_rms) || horizontal_rms < config.minimum_horizontal_rms_m ||
      std::hypot(procrustes_cosine, procrustes_sine) <= std::numeric_limits<double>::epsilon()) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::InsufficientHorizontalExcitation, 0U});
  }

  std::vector<std::pair<double, double>> pair_sines;
  std::vector<std::pair<double, double>> pair_cosines;
  const std::size_t maximum_pair_count = prepared.size() * (prepared.size() - 1U) / 2U;
  pair_sines.reserve(maximum_pair_count);
  pair_cosines.reserve(maximum_pair_count);
  for (std::size_t first = 0; first < prepared.size(); ++first) {
    for (std::size_t second = first + 1U; second < prepared.size(); ++second) {
      const Eigen::Vector2d delta_map =
          prepared[second].map.head<2>() - prepared[first].map.head<2>();
      if (delta_map.norm() < config.minimum_pair_baseline_m) {
        continue;
      }
      const Eigen::Vector2d delta_enu =
          prepared[second].enu.head<2>() - prepared[first].enu.head<2>();
      if (delta_enu.norm() <= std::numeric_limits<double>::epsilon()) {
        continue;
      }
      const double candidate_yaw = std::atan2(
          delta_map.x() * delta_enu.y() - delta_map.y() * delta_enu.x(), delta_map.dot(delta_enu));
      const double pair_weight = 2.0 * prepared[first].scalar_weight *
                                 prepared[second].scalar_weight /
                                 (prepared[first].scalar_weight + prepared[second].scalar_weight);
      pair_sines.emplace_back(std::sin(candidate_yaw), pair_weight);
      pair_cosines.emplace_back(std::cos(candidate_yaw), pair_weight);
    }
  }

  double yaw = std::atan2(procrustes_sine, procrustes_cosine);
  if (!pair_sines.empty()) {
    const double robust_sine = weightedMedian(std::move(pair_sines));
    const double robust_cosine = weightedMedian(std::move(pair_cosines));
    if (std::hypot(robust_sine, robust_cosine) > std::numeric_limits<double>::epsilon()) {
      yaw = std::atan2(robust_sine, robust_cosine);
    }
  }
  Eigen::Vector3d translation = robustTranslationForYaw(prepared, yaw);

  bool converged = false;
  std::size_t iterations = 0U;
  AlignmentLinearization linearization;
  for (; iterations < config.maximum_iterations; ++iterations) {
    linearization = linearizeAlignment(prepared, yaw, translation, config.huber_threshold_sigma);
    const auto condition = matrixCondition(linearization.normal);
    if (!condition || *condition > config.maximum_normal_condition) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::IllConditioned, 0U});
    }
    Eigen::LDLT<Eigen::Matrix4d> ldlt(linearization.normal);
    if (ldlt.info() != Eigen::Success) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::IllConditioned, 0U});
    }
    const Eigen::Vector4d increment = ldlt.solve(-linearization.gradient);
    if (!increment.allFinite()) {
      return core::Result<AlignmentEstimate, AlignmentError>::failure(
          AlignmentError{AlignmentErrorCode::IllConditioned, 0U});
    }
    translation += increment.head<3>();
    yaw = normalizeAngle(yaw + increment(3));
    if (std::abs(increment(3)) <= config.yaw_convergence_rad &&
        increment.head<3>().norm() <= config.translation_convergence_m) {
      converged = true;
      ++iterations;
      break;
    }
  }
  if (!converged) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::DidNotConverge, 0U});
  }

  linearization = linearizeAlignment(prepared, yaw, translation, config.huber_threshold_sigma);
  const auto final_condition = matrixCondition(linearization.normal);
  if (!final_condition || *final_condition > config.maximum_normal_condition) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::IllConditioned, 0U});
  }
  Eigen::LDLT<Eigen::Matrix4d> final_ldlt(linearization.normal);
  Eigen::Matrix4d covariance = final_ldlt.solve(Eigen::Matrix4d::Identity());
  if (final_ldlt.info() != Eigen::Success || !covariance.allFinite()) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::IllConditioned, 0U});
  }
  const double degrees_of_freedom = std::max(1.0, 3.0 * static_cast<double>(prepared.size()) - 4.0);
  const double reduced_chi_squared = linearization.weighted_squared_error / degrees_of_freedom;
  covariance *= std::max(1.0, reduced_chi_squared);

  auto transform = GravityAlignedTransform::create(yaw, translation);
  if (!transform) {
    return core::Result<AlignmentEstimate, AlignmentError>::failure(
        AlignmentError{AlignmentErrorCode::NonFiniteInput, 0U});
  }
  AlignmentDiagnostics diagnostics;
  diagnostics.count_ = prepared.size();
  diagnostics.inliers_ = linearization.inliers;
  diagnostics.outliers_ = prepared.size() - linearization.inliers;
  diagnostics.iterations_ = iterations;
  diagnostics.time_span_s_ = time_span_s;
  diagnostics.horizontal_rms_m_ = horizontal_rms;
  diagnostics.condition_ = *final_condition;
  diagnostics.weighted_rms_sigma_ = std::sqrt(linearization.weighted_squared_error /
                                              (3.0 * static_cast<double>(prepared.size())));
  diagnostics.maximum_residual_sigma_ = linearization.maximum_residual_sigma;
  return core::Result<AlignmentEstimate, AlignmentError>::success(
      AlignmentEstimate(std::move(transform).value(), covariance, std::move(diagnostics)));
}

core::Result<GnssHealthMonitor, GnssHealthError> GnssHealthMonitor::create(
    const GnssHealthConfig& config) {
  if (!validHealthConfig(config)) {
    return core::Result<GnssHealthMonitor, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::InvalidConfiguration});
  }
  return core::Result<GnssHealthMonitor, GnssHealthError>::success(GnssHealthMonitor(config));
}

GnssHealthDecision GnssHealthMonitor::decision(
    GnssDisposition disposition, GnssHealthReason reason,
    std::vector<core::GnssObservationId> released) const {
  const std::size_t consistent_count =
      released.empty() ? reacquisition_batch_.size() : released.size();
  return GnssHealthDecision(state_, disposition, reason, consistent_count, std::move(released));
}

std::optional<GnssHealthReason> GnssHealthMonitor::basicRejectionReason(
    const GnssHealthSample& sample) const {
  if (sample.quality.solution() == ReceiverReportedSolution::NoFix ||
      sample.quality.solution() == ReceiverReportedSolution::Unknown) {
    return GnssHealthReason::ReceiverReportedNoFix;
  }
  if (sample.quality.integrity() != ReceiverIntegrity::Valid) {
    return GnssHealthReason::ReceiverIntegrityNotValid;
  }
  if (sample.quality.satellites() && *sample.quality.satellites() < config_.minimum_satellites) {
    return GnssHealthReason::TooFewSatellites;
  }
  if (std::sqrt(sample.innovation_covariance_enu.diagonal().maxCoeff()) >
      config_.maximum_position_std_m) {
    return GnssHealthReason::PositionUncertaintyTooLarge;
  }
  return std::nullopt;
}

bool GnssHealthMonitor::appendOrRestartReacquisitionBatch(const GnssHealthSample& sample) {
  bool consistent = true;
  if (!reacquisition_batch_.empty()) {
    const double gap_s =
        static_cast<double>((sample.stamp - reacquisition_batch_.back().stamp).nanoseconds) /
        kNanosecondsPerSecond;
    if (gap_s > config_.maximum_reacquisition_sample_gap_s) {
      consistent = false;
    } else {
      Eigen::Matrix3d information_sum = Eigen::Matrix3d::Zero();
      Eigen::Vector3d information_weighted_sum = Eigen::Vector3d::Zero();
      for (const auto& batch_sample : reacquisition_batch_) {
        const Eigen::Matrix3d information = covarianceInformation(batch_sample.covariance_enu);
        information_sum += information;
        information_weighted_sum.noalias() += information * batch_sample.innovation_enu;
      }
      const Eigen::Matrix3d mean_covariance =
          information_sum.ldlt().solve(Eigen::Matrix3d::Identity());
      const Eigen::Vector3d mean_innovation = mean_covariance * information_weighted_sum;
      const Eigen::Vector3d delta = sample.innovation_enu - mean_innovation;
      const Eigen::Matrix3d comparison_covariance =
          sample.innovation_covariance_enu + mean_covariance;
      const double consistency_chi2 = delta.dot(comparison_covariance.ldlt().solve(delta));
      consistent =
          finite(consistency_chi2) && consistency_chi2 <= config_.reacquisition_consistency_chi2;
    }
  }
  if (!consistent) {
    reacquisition_batch_.clear();
  }
  reacquisition_batch_.push_back(BatchSample{sample.id, sample.stamp, sample.innovation_enu,
                                             sample.innovation_covariance_enu});
  return consistent;
}

core::Result<GnssHealthDecision, GnssHealthError> GnssHealthMonitor::observe(
    const GnssHealthSample& sample) {
  if (!sample.id.valid()) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::InvalidObservationId});
  }
  if (!sample.innovation_enu.allFinite()) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::NonFiniteInput});
  }
  if (!validCovariance(sample.innovation_covariance_enu)) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::InvalidCovariance});
  }
  if ((sample.quality.hdop() &&
       (!finite(*sample.quality.hdop()) || *sample.quality.hdop() < 0.0)) ||
      (sample.quality.correctionAgeS() &&
       (!finite(*sample.quality.correctionAgeS()) || *sample.quality.correctionAgeS() < 0.0))) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::NonFiniteInput});
  }
  if (last_observation_time_ && sample.stamp <= *last_observation_time_) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::NonMonotonicTimestamp});
  }

  bool observation_gap = false;
  if (last_observation_time_) {
    const double gap_s = static_cast<double>((sample.stamp - *last_observation_time_).nanoseconds) /
                         kNanosecondsPerSecond;
    if (gap_s > config_.outage_timeout_s) {
      state_ = GnssHealthState::Outage;
      reacquisition_batch_.clear();
      observation_gap = true;
    }
  }
  last_observation_time_ = sample.stamp;

  if (const auto rejection = basicRejectionReason(sample)) {
    if (!observation_gap) {
      state_ = GnssHealthState::Quarantined;
    }
    reacquisition_batch_.clear();
    return core::Result<GnssHealthDecision, GnssHealthError>::success(
        decision(GnssDisposition::QuarantineCurrent, *rejection));
  }

  const Eigen::Matrix3d information = covarianceInformation(sample.innovation_covariance_enu);
  const double innovation_chi2 = sample.innovation_enu.dot(information * sample.innovation_enu);
  if (state_ == GnssHealthState::Healthy && !observation_gap &&
      innovation_chi2 <= config_.healthy_innovation_chi2) {
    return core::Result<GnssHealthDecision, GnssHealthError>::success(
        decision(GnssDisposition::AdmitCurrent, GnssHealthReason::Accepted));
  }

  const bool had_batch = !reacquisition_batch_.empty();
  state_ = GnssHealthState::Reacquiring;
  const bool consistent = appendOrRestartReacquisitionBatch(sample);
  if (reacquisition_batch_.size() >= config_.consistent_samples_for_admission) {
    std::vector<core::GnssObservationId> released;
    released.reserve(reacquisition_batch_.size());
    for (const auto& batch_sample : reacquisition_batch_) {
      released.push_back(batch_sample.id);
    }
    reacquisition_batch_.clear();
    state_ = GnssHealthState::Healthy;
    return core::Result<GnssHealthDecision, GnssHealthError>::success(decision(
        GnssDisposition::ReleaseConsistentBatch, GnssHealthReason::Accepted, std::move(released)));
  }

  const GnssHealthReason reason = had_batch && !consistent
                                      ? GnssHealthReason::ReacquisitionBatchInconsistent
                                      : (state_ == GnssHealthState::Reacquiring &&
                                                 innovation_chi2 > config_.healthy_innovation_chi2
                                             ? GnssHealthReason::InnovationRejected
                                             : GnssHealthReason::AwaitingConsistentBatch);
  return core::Result<GnssHealthDecision, GnssHealthError>::success(
      decision(GnssDisposition::QuarantineCurrent, reason));
}

core::Result<GnssHealthDecision, GnssHealthError> GnssHealthMonitor::advanceTime(
    core::FusionTime now) {
  if (last_observation_time_ && now < *last_observation_time_) {
    return core::Result<GnssHealthDecision, GnssHealthError>::failure(
        GnssHealthError{GnssHealthErrorCode::NonMonotonicTimestamp});
  }
  if (last_observation_time_) {
    const double gap_s =
        static_cast<double>((now - *last_observation_time_).nanoseconds) / kNanosecondsPerSecond;
    if (gap_s > config_.outage_timeout_s) {
      state_ = GnssHealthState::Outage;
      reacquisition_batch_.clear();
      return core::Result<GnssHealthDecision, GnssHealthError>::success(
          decision(GnssDisposition::NoObservation, GnssHealthReason::ObservationGap));
    }
  }
  return core::Result<GnssHealthDecision, GnssHealthError>::success(
      decision(GnssDisposition::NoObservation, GnssHealthReason::Accepted));
}

}  // namespace meridian::global
