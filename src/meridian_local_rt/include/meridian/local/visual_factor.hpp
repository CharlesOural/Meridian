#pragma once

#include <Eigen/Core>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/equidistant_camera.hpp"
#include "meridian/local/visual_frontend.hpp"

namespace meridian::local {

class VisualLandmarkId {
public:
  constexpr VisualLandmarkId() = default;
  explicit constexpr VisualLandmarkId(std::uint64_t value) : value_(value) {}
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalidValue; }
  auto operator<=>(const VisualLandmarkId&) const = default;

  static constexpr std::uint64_t kInvalidValue = std::numeric_limits<std::uint64_t>::max();

private:
  std::uint64_t value_{kInvalidValue};
};

// Immutable measurement-side data for one visual observation. The typed
// extrinsic maps camera coordinates into the optimized IMU body frame:
// p_imu = T_imu_camera * p_camera.
struct VisualObservationRef {
  core::MeasurementId frame;
  core::StateId state;
  core::FusionTime exact_time;
  core::CameraId camera;
  core::CalibrationEpoch calibration;
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  Eigen::Matrix2d pixel_covariance{Eigen::Matrix2d::Identity()};
  EquidistantCameraParameters camera_model;
  core::ImuFromCameraTransform imu_from_camera{core::Pose3d{}};
};

struct VisualReprojectionFactorSpec {
  core::FactorId id;
  VisualLandmarkId landmark;
  VisualTrackId track;
  VisualObservationRef anchor;
  VisualObservationRef observer;
  double minimum_range_m{0.3};
  double maximum_range_m{200.0};
  // The anchor pixel conditions the scalar inverse-range parameterization; it
  // is not added again as an independent residual. Its covariance is retained
  // above for observability diagnostics and future covariance-aware
  // reanchoring.
  // The observer covariance whitens this 2-vector, then GTSAM applies exactly
  // one Huber kernel to its norm.
  double huber_delta_sigma{2.5};
  core::ObservationLineage lineage;
};

enum class VisualReprojectionErrorCode {
  InvalidIdentity,
  InvalidTimeOrder,
  InvalidCameraModel,
  InvalidExtrinsic,
  InvalidPixel,
  InvalidCovariance,
  InvalidLineage,
  InvalidRangeBounds,
  NonFinitePose,
  NonFiniteEta,
  RangeOutsideBounds,
  AnchorUnprojectionFailure,
  ObserverCheirality,
  ObserverProjectionFailure,
  ObserverOutsideImage,
  NumericalFailure,
};

struct VisualReprojectionError {
  VisualReprojectionErrorCode code{};
  std::string detail;
};

struct VisualReprojectionEvaluation {
  Eigen::Vector2d predicted_pixel{Eigen::Vector2d::Zero()};
  Eigen::Vector2d residual{Eigen::Vector2d::Zero()};
  // Public pose Jacobians use Meridian's binding convention: right
  // perturbations with [translation, rotation] columns.
  Eigen::Matrix<double, 2, 6> anchor_pose_jacobian{Eigen::Matrix<double, 2, 6>::Zero()};
  Eigen::Matrix<double, 2, 6> observer_pose_jacobian{Eigen::Matrix<double, 2, 6>::Zero()};
  Eigen::Matrix<double, 2, 1> eta_jacobian{Eigen::Matrix<double, 2, 1>::Zero()};
  double anchor_range_m{};
  double observer_depth_m{};
  double squared_mahalanobis{};
};

// Pure geometry path shared by diagnostics and the private GTSAM factor.
// eta=log(rho), rho is inverse Euclidean range along the anchor unit ray.
[[nodiscard]] core::Result<VisualReprojectionEvaluation, VisualReprojectionError>
evaluateVisualReprojection(const VisualReprojectionFactorSpec& spec,
                           const core::Pose3d& T_odom_imu_anchor,
                           const core::Pose3d& T_odom_imu_observer, double eta);

struct VisualKeyframeContext {
  core::StateId state;
  core::FusionTime exact_time;
  core::CameraId camera;
  core::CalibrationEpoch calibration;
  EquidistantCameraParameters camera_model;
  core::ImuFromCameraTransform imu_from_camera{core::Pose3d{}};
  // Estimate used only for deterministic triangulation and admission. It is
  // not copied into factor specifications and is not an independent VIO pose.
  core::Pose3d T_odom_imu;
};

struct VisualFactorBuilderConfig {
  std::size_t minimum_non_anchor_observations{2U};
  std::size_t maximum_pending_observations{8U};
  std::size_t maximum_observations_per_track{20U};
  std::size_t maximum_active_tracks{500U};
  std::size_t maximum_missed_keyframes{2U};
  double minimum_baseline_m{0.05};
  double minimum_parallax_rad{0.02617993877991494};  // 1.5 degrees
  double minimum_range_m{0.3};
  double maximum_range_m{200.0};
  double triangulation_huber_angle_rad{0.01};
  std::size_t triangulation_iterations{4U};
  double maximum_triangulation_condition{1.0e8};
  double maximum_inlier_reprojection_error_px{4.0};
  double maximum_reprojection_rmse_px{2.0};
  double huber_delta_sigma{2.5};
  double outlier_chi_squared_gate{9.210340371976184};
  std::size_t outlier_commits_before_retirement{2U};
};

enum class VisualTriangulationStatus {
  Seeded,
  InsufficientObservations,
  InsufficientBaseline,
  InsufficientParallax,
  IllConditioned,
  BehindCamera,
  RangeOutsideBounds,
  TooFewInliers,
  ReprojectionRmseTooLarge,
  NumericalFailure,
};

struct VisualTriangulationDecision {
  VisualTrackId track;
  VisualTriangulationStatus status{VisualTriangulationStatus::InsufficientObservations};
  std::size_t observations{};
  std::size_t inliers{};
  double maximum_baseline_m{};
  double maximum_parallax_rad{};
  double condition_number{};
  double reprojection_rmse_px{};
};

struct VisualLandmarkSeed {
  VisualLandmarkId landmark;
  VisualTrackId track;
  VisualObservationRef anchor;
  double eta{};
  double initial_range_m{};
  VisualTriangulationDecision triangulation;
};

enum class VisualTrackRetirementReason {
  Lost,
  MaximumLength,
  Capacity,
};

struct VisualTrackRetirement {
  VisualTrackId track;
  VisualLandmarkId landmark;
  VisualTrackRetirementReason reason{VisualTrackRetirementReason::Lost};
};

struct VisualFactorBatchReport {
  std::size_t input_features{};
  std::size_t tracks_created{};
  std::size_t tracks_pending{};
  std::size_t tracks_initialized{};
  std::size_t tracks_rejected_capacity{};
  std::size_t factors_emitted{};
  std::size_t observations_rejected{};
  std::size_t active_tracks{};
  std::vector<VisualTriangulationDecision> triangulation;
};

struct VisualFactorBatch {
  core::FusionTime exact_time;
  std::vector<VisualLandmarkSeed> new_landmarks;
  std::vector<VisualReprojectionFactorSpec> factors;
  std::vector<VisualTrackRetirement> retired_tracks;
  VisualFactorBatchReport report;
};

struct VisualUncommittedBatchDiscardReport {
  std::size_t landmark_seeds_discarded{};
  std::size_t active_landmark_initializations_rolled_back{};
  std::size_t stale_track_observations_discarded{};
  std::size_t factors_discarded{};
  std::size_t factor_health_entries_removed{};
};

// Lane-local identities translated from one successful graph commit.  Empty
// updates are intentional: every lane observes every graph revision, even
// when that revision finalized no visual work owned by the lane.
struct VisualFinalityUpdate {
  core::LocalGraphRevision revision;
  std::vector<core::FactorId> finalized_factors;
  std::vector<VisualLandmarkId> finalized_landmarks;
};

struct VisualFactorFinalityReport {
  core::LocalGraphRevision revision;
  std::size_t finalized_factors_observed{};
  std::size_t finalized_landmarks_observed{};
  std::size_t accepted_tracks_pruned{};
  std::size_t factor_health_entries_pruned{};
};

struct VisualResidualFeedback {
  core::FactorId factor;
  double squared_mahalanobis{};
};

struct VisualResidualFeedbackReport {
  core::LocalGraphRevision revision;
  std::size_t evaluated{};
  std::size_t unknown_or_inactive{};
  std::vector<core::FactorId> retired_observation_factors;
};

enum class VisualFactorBuilderErrorCode {
  InvalidConfig,
  NonKeyframeInput,
  InvalidContext,
  InvalidLineage,
  DuplicateTrack,
  NonMonotonicKeyframe,
  StateOrCameraMismatch,
  InvalidFeature,
  InvalidFeedbackRevision,
  DuplicateFeedback,
  InvalidFinality,
  NumericalFailure,
};

struct VisualFactorBuilderError {
  VisualFactorBuilderErrorCode code{};
  std::string detail;
};

// Deterministic, bounded track lifecycle. It creates landmark seeds and direct
// reprojection factor specifications; it never estimates or publishes a pose.
class VisualFactorBatchBuilder {
public:
  explicit VisualFactorBatchBuilder(VisualFactorBuilderConfig config = {});
  ~VisualFactorBatchBuilder();

  VisualFactorBatchBuilder(VisualFactorBatchBuilder&&) noexcept;
  VisualFactorBatchBuilder& operator=(VisualFactorBatchBuilder&&) noexcept;
  VisualFactorBatchBuilder(const VisualFactorBatchBuilder&) = delete;
  VisualFactorBatchBuilder& operator=(const VisualFactorBatchBuilder&) = delete;

  [[nodiscard]] core::Result<VisualFactorBatch, VisualFactorBuilderError> processKeyframe(
      const VisualFrontendOutput& frontend, const VisualKeyframeContext& context);

  // Called once per accepted graph commit. Each observation factor owns its
  // own outlier streak; retiring one never deletes the landmark or modality.
  [[nodiscard]] core::Result<VisualResidualFeedbackReport, VisualFactorBuilderError>
  applyAcceptedResiduals(core::LocalGraphRevision revision,
                         const std::vector<VisualResidualFeedback>& feedback);

  // A graph rejection can invalidate a queued landmark seed and every later
  // uncommitted factor that depends on it. This removes only measurement work
  // that never entered the graph; accepted factor health and track history are
  // preserved. A rolled-back landmark keeps only its newest observation so a
  // later seed cannot retain an anchor state that was never graph-accepted or
  // has since left the fixed-lag window.
  [[nodiscard]] VisualUncommittedBatchDiscardReport discardUncommittedBatch(
      const VisualFactorBatch& batch) noexcept;

  // Applies graph finality to accepted builder state. Factor health is removed
  // before landmark-owned tracks so reconciliation is dependency-safe. The
  // update is candidate-copy transactional and never rolls identity counters
  // back, therefore a finalized ID cannot be reused.
  [[nodiscard]] core::Result<VisualFactorFinalityReport, VisualFactorBuilderError>
  reconcileFinality(const VisualFinalityUpdate& update);

  void reset();
  [[nodiscard]] std::size_t activeTracks() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace meridian::local
