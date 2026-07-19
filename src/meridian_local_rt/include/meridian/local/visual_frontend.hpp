#pragma once

#include <Eigen/Core>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/local/equidistant_camera.hpp"

namespace meridian::local {

class VisualTrackId {
public:
  constexpr VisualTrackId() = default;
  explicit constexpr VisualTrackId(std::uint64_t value) : value_(value) {}
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
  [[nodiscard]] constexpr bool valid() const noexcept { return value_ != kInvalidValue; }
  auto operator<=>(const VisualTrackId&) const = default;

  static constexpr std::uint64_t kInvalidValue = std::numeric_limits<std::uint64_t>::max();

private:
  std::uint64_t value_{kInvalidValue};
};

struct VisualRotationPrior {
  // Coordinate convention: bearing_current = R_current_previous *
  // bearing_previous. The endpoints must exactly match consecutive exposure
  // midpoints; a mismatched or invalid prior is reported and safely ignored.
  Eigen::Matrix3d R_current_previous{Eigen::Matrix3d::Identity()};
  core::FusionTime previous_exposure_midpoint;
  core::FusionTime current_exposure_midpoint;
  std::vector<core::MeasurementId> imu_support;
  core::CalibrationEpoch imu_calibration;
};

struct VisualFrontendConfig {
  std::uint32_t grid_columns{8U};
  std::uint32_t grid_rows{6U};
  std::uint32_t features_per_cell{8U};
  double detector_quality{0.01};
  double minimum_feature_distance_px{15.0};
  std::uint32_t detector_block_size{3U};
  std::uint32_t image_border_px{16U};
  std::uint32_t klt_window_size{21U};
  std::uint32_t klt_pyramid_levels{3U};
  std::uint32_t klt_max_iterations{30U};
  double klt_epsilon{0.01};
  double maximum_klt_error{30.0};
  double maximum_forward_backward_error_px{0.75};
  std::uint32_t recovery_minimum_tracks{48U};
  double recovery_minimum_retention{0.35};
  std::uint32_t keyframe_minimum_tracks{120U};
  double keyframe_minimum_spatial_coverage{0.5};
  double keyframe_maximum_overlap{0.55};
  double keyframe_minimum_parallax_px{15.0};
  core::Duration minimum_keyframe_interval{150'000'000LL};
  core::Duration maximum_keyframe_interval{500'000'000LL};
  core::Duration maximum_time_uncertainty{2'000'000LL};
  double base_pixel_sigma{0.8};

  [[nodiscard]] bool valid(const EquidistantCamera& camera) const noexcept;
};

enum class RotationPriorStatus {
  NotProvided,
  Applied,
  TimeMismatch,
  InvalidRotation,
  MissingCalibration,
  InvalidSupport,
};

enum class VisualKeyframeReason {
  None,
  FirstFrame,
  Requested,
  Recovery,
  InsufficientCoverage,
  LowTrackOverlap,
  SufficientParallax,
  MaximumInterval,
};

enum class VisualTrackingQuality {
  Nominal,
  Degraded,
  Recovery,
};

struct VisualFrontendTiming {
  std::int64_t image_decode_us{};
  std::int64_t optical_flow_us{};
  std::int64_t replenishment_us{};
  std::int64_t descriptor_us{};
  std::int64_t total_us{};
};

struct VisualTrackingReport {
  std::size_t input_tracks{};
  std::size_t rotation_seeded_tracks{};
  std::size_t forward_flow_successes{};
  std::size_t backward_flow_successes{};
  std::size_t rejected_by_forward_flow{};
  std::size_t rejected_by_backward_flow{};
  std::size_t rejected_by_border{};
  std::size_t rejected_by_klt_error{};
  std::size_t rejected_by_forward_backward_error{};
  std::size_t rejected_by_grid_capacity{};
  std::size_t retained_tracks{};
  std::size_t replenished_tracks{};
  std::size_t output_tracks{};
  std::size_t descriptor_tracks{};
  std::size_t grid_cells_below_target{};
  double retained_fraction{1.0};
  double spatial_coverage{};
  double overlap_with_last_keyframe{1.0};
  double median_keyframe_parallax_px{};
  bool rotation_compensated_parallax_available{};
  double mean_klt_error{};
  double median_forward_backward_error_px{};
  RotationPriorStatus rotation_prior{RotationPriorStatus::NotProvided};
  VisualKeyframeReason keyframe_reason{VisualKeyframeReason::None};
  VisualTrackingQuality quality{VisualTrackingQuality::Nominal};
  core::Duration time_uncertainty;
  VisualFrontendTiming timing;
};

struct VisualFeatureObservation {
  VisualTrackId track;
  Eigen::Vector2d distorted_pixel{Eigen::Vector2d::Zero()};
  Eigen::Vector3d unit_bearing{Eigen::Vector3d::UnitZ()};
  Eigen::Vector2d pixel_velocity_per_second{Eigen::Vector2d::Zero()};
  Eigen::Matrix2d pixel_covariance{Eigen::Matrix2d::Identity()};
  std::uint32_t age{};
  float klt_error{};
  float forward_backward_error_px{};
  bool newly_detected{};
};

inline constexpr std::size_t kBriskDescriptorBytes = 64U;

struct BriskTrackDescriptor {
  VisualTrackId track;
  std::array<std::uint8_t, kBriskDescriptorBytes> bytes{};
};

struct VisualFrontendOutput {
  core::MeasurementId source_frame;
  core::CameraId camera;
  core::FusionTime exposure_midpoint;
  bool keyframe{};
  std::vector<VisualFeatureObservation> features;
  // Empty for non-keyframes. Descriptors are deliberately computed only for
  // keyframes (including recovery keyframes) to isolate real-time tracking
  // from global retrieval work.
  std::vector<BriskTrackDescriptor> descriptors;
  core::ObservationLineage lineage;
  VisualTrackingReport report;
};

enum class VisualFrontendErrorCode {
  InvalidConfiguration,
  InvalidCameraFrame,
  UnsupportedImageEncoding,
  InvalidImageLayout,
  MissingCalibration,
  CameraChanged,
  CalibrationChanged,
  ResolutionChanged,
  NonMonotonicTime,
  SourceEpochChanged,
  TimeUncertain,
  CameraModelFailure,
  OpticalFlowFailure,
  DescriptorFailure,
};

struct VisualFrontendError {
  VisualFrontendErrorCode code{};
  std::string detail;
};

class GridKltVisualFrontend {
public:
  GridKltVisualFrontend(EquidistantCamera camera, VisualFrontendConfig config = {});
  ~GridKltVisualFrontend();

  GridKltVisualFrontend(const GridKltVisualFrontend&) = delete;
  GridKltVisualFrontend& operator=(const GridKltVisualFrontend&) = delete;
  GridKltVisualFrontend(GridKltVisualFrontend&&) noexcept;
  GridKltVisualFrontend& operator=(GridKltVisualFrontend&&) noexcept;

  [[nodiscard]] core::Result<VisualFrontendOutput, VisualFrontendError> process(
      const core::CameraFrame& frame,
      std::optional<VisualRotationPrior> rotation_prior = std::nullopt,
      bool request_keyframe = false);

  void reset();
  [[nodiscard]] bool initialized() const noexcept;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace meridian::local
