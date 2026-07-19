#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/result.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"

namespace meridian::global {

enum class GeodesyErrorCode {
  NonFiniteInput,
  LatitudeOutOfRange,
  LongitudeOutOfRange,
  UndefinedAtEarthCenter,
  InverseDidNotConverge,
};

struct GeodesyError {
  GeodesyErrorCode code{GeodesyErrorCode::NonFiniteInput};
};

// WGS84 ellipsoidal latitude, longitude, and height. Angles are stored in
// radians; the named factories make the unit explicit at every call site.
class Wgs84Lla {
public:
  [[nodiscard]] static core::Result<Wgs84Lla, GeodesyError> fromRadians(double latitude_rad,
                                                                        double longitude_rad,
                                                                        double altitude_m);
  [[nodiscard]] static core::Result<Wgs84Lla, GeodesyError> fromDegrees(double latitude_deg,
                                                                        double longitude_deg,
                                                                        double altitude_m);

  [[nodiscard]] double latitudeRad() const noexcept { return latitude_rad_; }
  [[nodiscard]] double longitudeRad() const noexcept { return longitude_rad_; }
  [[nodiscard]] double latitudeDeg() const noexcept;
  [[nodiscard]] double longitudeDeg() const noexcept;
  [[nodiscard]] double altitudeM() const noexcept { return altitude_m_; }

private:
  Wgs84Lla(double latitude_rad, double longitude_rad, double altitude_m)
      : latitude_rad_(latitude_rad), longitude_rad_(longitude_rad), altitude_m_(altitude_m) {}

  double latitude_rad_{};
  double longitude_rad_{};
  double altitude_m_{};
};

class EcefPosition {
public:
  [[nodiscard]] static core::Result<EcefPosition, GeodesyError> fromMeters(
      const Eigen::Vector3d& meters);
  [[nodiscard]] const Eigen::Vector3d& meters() const noexcept { return meters_; }

private:
  explicit EcefPosition(Eigen::Vector3d meters) : meters_(std::move(meters)) {}
  Eigen::Vector3d meters_{Eigen::Vector3d::Zero()};
};

class EnuPosition {
public:
  [[nodiscard]] static core::Result<EnuPosition, GeodesyError> fromMeters(
      const Eigen::Vector3d& meters);
  [[nodiscard]] const Eigen::Vector3d& meters() const noexcept { return meters_; }

private:
  explicit EnuPosition(Eigen::Vector3d meters) : meters_(std::move(meters)) {}
  Eigen::Vector3d meters_{Eigen::Vector3d::Zero()};
};

[[nodiscard]] core::Result<EcefPosition, GeodesyError> wgs84LlaToEcef(const Wgs84Lla& lla);
[[nodiscard]] core::Result<Wgs84Lla, GeodesyError> wgs84EcefToLla(const EcefPosition& ecef);

class LocalEnuDatum {
public:
  [[nodiscard]] static core::Result<LocalEnuDatum, GeodesyError> create(const Wgs84Lla& origin);

  [[nodiscard]] const Wgs84Lla& originLla() const noexcept { return origin_lla_; }
  [[nodiscard]] const EcefPosition& originEcef() const noexcept { return origin_ecef_; }
  [[nodiscard]] const Eigen::Matrix3d& REnuEcef() const noexcept { return R_enu_ecef_; }
  [[nodiscard]] core::Result<EnuPosition, GeodesyError> toEnu(const EcefPosition& ecef) const;
  [[nodiscard]] core::Result<EcefPosition, GeodesyError> toEcef(const EnuPosition& enu) const;
  [[nodiscard]] core::Result<EnuPosition, GeodesyError> toEnu(const Wgs84Lla& lla) const;
  [[nodiscard]] core::Result<Wgs84Lla, GeodesyError> toLla(const EnuPosition& enu) const;

private:
  LocalEnuDatum(Wgs84Lla origin_lla, EcefPosition origin_ecef, Eigen::Matrix3d R_enu_ecef)
      : origin_lla_(std::move(origin_lla)),
        origin_ecef_(std::move(origin_ecef)),
        R_enu_ecef_(std::move(R_enu_ecef)) {}

  Wgs84Lla origin_lla_;
  EcefPosition origin_ecef_;
  Eigen::Matrix3d R_enu_ecef_{Eigen::Matrix3d::Identity()};
};

// This classification is receiver-reported metadata. In particular, callers
// must never manufacture RtkFloat/RtkFixed from a generic position-fix status.
enum class ReceiverReportedSolution {
  Unknown,
  NoFix,
  Autonomous,
  Differential,
  RtkFloat,
  RtkFixed,
  PppFloat,
  PppFixed,
};

enum class ReceiverIntegrity {
  Unknown,
  Valid,
  Suspect,
  Invalid,
};

enum class CorrectionStreamState {
  Unknown,
  NotUsed,
  Current,
  Stale,
};

class GnssQuality {
public:
  GnssQuality(ReceiverReportedSolution solution, ReceiverIntegrity integrity,
              CorrectionStreamState corrections, std::optional<std::uint32_t> satellites,
              std::optional<double> hdop, std::optional<double> correction_age_s)
      : solution_(solution),
        integrity_(integrity),
        corrections_(corrections),
        satellites_(satellites),
        hdop_(hdop),
        correction_age_s_(correction_age_s) {}

  [[nodiscard]] ReceiverReportedSolution solution() const noexcept { return solution_; }
  [[nodiscard]] ReceiverIntegrity integrity() const noexcept { return integrity_; }
  [[nodiscard]] CorrectionStreamState corrections() const noexcept { return corrections_; }
  [[nodiscard]] std::optional<std::uint32_t> satellites() const noexcept { return satellites_; }
  [[nodiscard]] std::optional<double> hdop() const noexcept { return hdop_; }
  [[nodiscard]] std::optional<double> correctionAgeS() const noexcept { return correction_age_s_; }

private:
  ReceiverReportedSolution solution_{ReceiverReportedSolution::Unknown};
  ReceiverIntegrity integrity_{ReceiverIntegrity::Unknown};
  CorrectionStreamState corrections_{CorrectionStreamState::Unknown};
  std::optional<std::uint32_t> satellites_;
  std::optional<double> hdop_;
  std::optional<double> correction_age_s_;
};

enum class GnssModelErrorCode {
  NonFiniteInput,
  InvalidTransform,
};

struct GnssModelError {
  GnssModelErrorCode code{GnssModelErrorCode::NonFiniteInput};
};

class AntennaLeverArm {
public:
  // Vector from the IMU origin to the GNSS antenna phase center, expressed in
  // the IMU frame.
  [[nodiscard]] static core::Result<AntennaLeverArm, GnssModelError> fromImuToAntenna(
      const Eigen::Vector3d& r_imu_antenna_in_imu);
  [[nodiscard]] const Eigen::Vector3d& imuToAntennaInImu() const noexcept {
    return r_imu_antenna_in_imu_;
  }

private:
  explicit AntennaLeverArm(Eigen::Vector3d value) : r_imu_antenna_in_imu_(std::move(value)) {}
  Eigen::Vector3d r_imu_antenna_in_imu_{Eigen::Vector3d::Zero()};
};

class GravityAlignedTransform {
public:
  // p_enu = Rz(yaw_enu_map) * p_map + translation_enu.
  [[nodiscard]] static core::Result<GravityAlignedTransform, GnssModelError> create(
      double yaw_enu_map_rad, const Eigen::Vector3d& translation_enu);

  [[nodiscard]] double yawEnuMapRad() const noexcept { return yaw_enu_map_rad_; }
  [[nodiscard]] const Eigen::Vector3d& translationEnu() const noexcept { return translation_enu_; }
  [[nodiscard]] Eigen::Matrix3d REnuMap() const noexcept;
  [[nodiscard]] Eigen::Vector3d apply(const Eigen::Vector3d& position_map) const noexcept;
  [[nodiscard]] Eigen::Vector3d inverseApply(const Eigen::Vector3d& position_enu) const noexcept;

private:
  GravityAlignedTransform(double yaw_enu_map_rad, Eigen::Vector3d translation_enu)
      : yaw_enu_map_rad_(yaw_enu_map_rad), translation_enu_(std::move(translation_enu)) {}

  double yaw_enu_map_rad_{};
  Eigen::Vector3d translation_enu_{Eigen::Vector3d::Zero()};
};

enum class PoseJacobianConvention {
  SophusRightTranslationThenRotation,
};

enum class AlignmentJacobianConvention {
  TranslationEnuThenYaw,
};

class AntennaPrediction {
public:
  [[nodiscard]] const Eigen::Vector3d& positionMap() const noexcept { return position_map_; }
  [[nodiscard]] const Eigen::Vector3d& positionEnu() const noexcept { return position_enu_; }
  [[nodiscard]] const Eigen::Matrix<double, 3, 6>& positionEnuJacobianPose() const noexcept {
    return J_enu_pose_;
  }
  [[nodiscard]] const Eigen::Matrix<double, 3, 4>& positionEnuJacobianAlignment() const noexcept {
    return J_enu_alignment_;
  }
  [[nodiscard]] PoseJacobianConvention poseJacobianConvention() const noexcept {
    return PoseJacobianConvention::SophusRightTranslationThenRotation;
  }
  [[nodiscard]] AlignmentJacobianConvention alignmentJacobianConvention() const noexcept {
    return AlignmentJacobianConvention::TranslationEnuThenYaw;
  }

private:
  friend core::Result<AntennaPrediction, GnssModelError> predictAntennaPosition(
      const core::Pose3d&, const AntennaLeverArm&, const GravityAlignedTransform&);
  AntennaPrediction(Eigen::Vector3d position_map, Eigen::Vector3d position_enu,
                    Eigen::Matrix<double, 3, 6> J_enu_pose,
                    Eigen::Matrix<double, 3, 4> J_enu_alignment)
      : position_map_(std::move(position_map)),
        position_enu_(std::move(position_enu)),
        J_enu_pose_(std::move(J_enu_pose)),
        J_enu_alignment_(std::move(J_enu_alignment)) {}

  Eigen::Vector3d position_map_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d position_enu_{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 3, 6> J_enu_pose_{Eigen::Matrix<double, 3, 6>::Zero()};
  Eigen::Matrix<double, 3, 4> J_enu_alignment_{Eigen::Matrix<double, 3, 4>::Zero()};
};

[[nodiscard]] core::Result<AntennaPrediction, GnssModelError> predictAntennaPosition(
    const core::Pose3d& T_map_imu, const AntennaLeverArm& lever_arm,
    const GravityAlignedTransform& T_enu_map);

struct AlignmentCorrespondence {
  core::FusionTime stamp;
  Eigen::Vector3d antenna_position_map{Eigen::Vector3d::Zero()};
  Eigen::Vector3d antenna_position_enu{Eigen::Vector3d::Zero()};
  // Covariance of the ENU alignment residual. Callers should include both the
  // GNSS position uncertainty and propagated local-position uncertainty.
  Eigen::Matrix3d covariance_enu{Eigen::Matrix3d::Identity()};
};

struct AlignmentConfig {
  std::size_t minimum_correspondences{6};
  double minimum_time_span_s{2.0};
  double minimum_horizontal_rms_m{1.0};
  double minimum_pair_baseline_m{0.5};
  double huber_threshold_sigma{2.5};
  std::size_t maximum_iterations{30};
  double yaw_convergence_rad{1.0e-10};
  double translation_convergence_m{1.0e-8};
  double maximum_normal_condition{1.0e12};
};

enum class AlignmentErrorCode {
  InvalidConfiguration,
  TooFewCorrespondences,
  NonFiniteInput,
  InvalidCovariance,
  NonMonotonicTimestamps,
  InsufficientTimeSpan,
  InsufficientHorizontalExcitation,
  IllConditioned,
  DidNotConverge,
};

struct AlignmentError {
  AlignmentErrorCode code{AlignmentErrorCode::NonFiniteInput};
  std::size_t correspondence_index{};
};

class AlignmentDiagnostics {
public:
  [[nodiscard]] std::size_t correspondenceCount() const noexcept { return count_; }
  [[nodiscard]] std::size_t robustInlierCount() const noexcept { return inliers_; }
  [[nodiscard]] std::size_t robustOutlierCount() const noexcept { return outliers_; }
  [[nodiscard]] std::size_t iterations() const noexcept { return iterations_; }
  [[nodiscard]] double timeSpanS() const noexcept { return time_span_s_; }
  [[nodiscard]] double horizontalRmsM() const noexcept { return horizontal_rms_m_; }
  [[nodiscard]] double normalCondition() const noexcept { return condition_; }
  [[nodiscard]] double weightedRmsSigma() const noexcept { return weighted_rms_sigma_; }
  [[nodiscard]] double maximumResidualSigma() const noexcept { return maximum_residual_sigma_; }

private:
  friend core::Result<class AlignmentEstimate, AlignmentError> estimateGravityAlignedTransform(
      std::span<const AlignmentCorrespondence>, const AlignmentConfig&);
  std::size_t count_{};
  std::size_t inliers_{};
  std::size_t outliers_{};
  std::size_t iterations_{};
  double time_span_s_{};
  double horizontal_rms_m_{};
  double condition_{};
  double weighted_rms_sigma_{};
  double maximum_residual_sigma_{};
};

class AlignmentEstimate {
public:
  [[nodiscard]] const GravityAlignedTransform& transform() const noexcept { return transform_; }
  // Parameter order is [translation_e, translation_n, translation_u,
  // yaw_enu_map], matching Meridian's translation-first pose convention.
  [[nodiscard]] const Eigen::Matrix4d& covariance() const noexcept { return covariance_; }
  [[nodiscard]] const AlignmentDiagnostics& diagnostics() const noexcept { return diagnostics_; }

private:
  friend core::Result<AlignmentEstimate, AlignmentError> estimateGravityAlignedTransform(
      std::span<const AlignmentCorrespondence>, const AlignmentConfig&);
  AlignmentEstimate(GravityAlignedTransform transform, Eigen::Matrix4d covariance,
                    AlignmentDiagnostics diagnostics)
      : transform_(std::move(transform)),
        covariance_(std::move(covariance)),
        diagnostics_(std::move(diagnostics)) {}

  GravityAlignedTransform transform_;
  Eigen::Matrix4d covariance_{Eigen::Matrix4d::Zero()};
  AlignmentDiagnostics diagnostics_;
};

[[nodiscard]] core::Result<AlignmentEstimate, AlignmentError> estimateGravityAlignedTransform(
    std::span<const AlignmentCorrespondence> correspondences, const AlignmentConfig& config = {});

enum class GnssHealthState {
  Initializing,
  Healthy,
  Quarantined,
  Outage,
  Reacquiring,
};

enum class GnssDisposition {
  AdmitCurrent,
  QuarantineCurrent,
  ReleaseConsistentBatch,
  NoObservation,
};

enum class GnssHealthReason {
  Accepted,
  AwaitingConsistentBatch,
  ReceiverReportedNoFix,
  ReceiverIntegrityNotValid,
  TooFewSatellites,
  PositionUncertaintyTooLarge,
  InnovationRejected,
  ReacquisitionBatchInconsistent,
  ObservationGap,
};

struct GnssHealthSample {
  core::GnssObservationId id;
  core::FusionTime stamp;
  GnssQuality quality{ReceiverReportedSolution::Unknown,
                      ReceiverIntegrity::Unknown,
                      CorrectionStreamState::Unknown,
                      std::nullopt,
                      std::nullopt,
                      std::nullopt};
  // Innovation is measured antenna position minus the current global-model
  // prediction. A stable non-zero offset is allowed during reacquisition.
  Eigen::Vector3d innovation_enu{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d innovation_covariance_enu{Eigen::Matrix3d::Identity()};
};

struct GnssHealthConfig {
  std::size_t consistent_samples_for_admission{5};
  std::uint32_t minimum_satellites{5};
  double maximum_position_std_m{10.0};
  double healthy_innovation_chi2{14.156};  // 3 DoF, 99.73%.
  double reacquisition_consistency_chi2{14.156};
  double outage_timeout_s{2.0};
  double maximum_reacquisition_sample_gap_s{1.0};
};

enum class GnssHealthErrorCode {
  InvalidConfiguration,
  NonFiniteInput,
  InvalidCovariance,
  InvalidObservationId,
  NonMonotonicTimestamp,
};

struct GnssHealthError {
  GnssHealthErrorCode code{GnssHealthErrorCode::NonFiniteInput};
};

class GnssHealthDecision {
public:
  [[nodiscard]] GnssHealthState state() const noexcept { return state_; }
  [[nodiscard]] GnssDisposition disposition() const noexcept { return disposition_; }
  [[nodiscard]] GnssHealthReason reason() const noexcept { return reason_; }
  [[nodiscard]] std::size_t consistentSampleCount() const noexcept { return consistent_count_; }
  [[nodiscard]] const std::vector<core::GnssObservationId>& releasedObservationIds()
      const noexcept {
    return released_ids_;
  }

private:
  friend class GnssHealthMonitor;
  GnssHealthDecision(GnssHealthState state, GnssDisposition disposition, GnssHealthReason reason,
                     std::size_t consistent_count,
                     std::vector<core::GnssObservationId> released_ids)
      : state_(state),
        disposition_(disposition),
        reason_(reason),
        consistent_count_(consistent_count),
        released_ids_(std::move(released_ids)) {}

  GnssHealthState state_{GnssHealthState::Initializing};
  GnssDisposition disposition_{GnssDisposition::NoObservation};
  GnssHealthReason reason_{GnssHealthReason::AwaitingConsistentBatch};
  std::size_t consistent_count_{};
  std::vector<core::GnssObservationId> released_ids_;
};

class GnssHealthMonitor {
public:
  [[nodiscard]] static core::Result<GnssHealthMonitor, GnssHealthError> create(
      const GnssHealthConfig& config = {});

  [[nodiscard]] core::Result<GnssHealthDecision, GnssHealthError> observe(
      const GnssHealthSample& sample);
  [[nodiscard]] core::Result<GnssHealthDecision, GnssHealthError> advanceTime(core::FusionTime now);
  [[nodiscard]] GnssHealthState state() const noexcept { return state_; }

private:
  struct BatchSample {
    core::GnssObservationId id;
    core::FusionTime stamp;
    Eigen::Vector3d innovation_enu;
    Eigen::Matrix3d covariance_enu;
  };

  explicit GnssHealthMonitor(GnssHealthConfig config) : config_(std::move(config)) {}
  [[nodiscard]] GnssHealthDecision decision(
      GnssDisposition disposition, GnssHealthReason reason,
      std::vector<core::GnssObservationId> released = {}) const;
  [[nodiscard]] std::optional<GnssHealthReason> basicRejectionReason(
      const GnssHealthSample& sample) const;
  [[nodiscard]] bool appendOrRestartReacquisitionBatch(const GnssHealthSample& sample);

  GnssHealthConfig config_;
  GnssHealthState state_{GnssHealthState::Initializing};
  std::optional<core::FusionTime> last_observation_time_;
  std::vector<BatchSample> reacquisition_batch_;
};

}  // namespace meridian::global
