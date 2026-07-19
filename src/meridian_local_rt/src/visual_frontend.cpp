#include "meridian/local/visual_frontend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <unordered_map>
#include <utility>

namespace meridian::local {
namespace {

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] std::int64_t elapsedMicroseconds(SteadyClock::time_point start,
                                               SteadyClock::time_point end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

[[nodiscard]] double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() % 2U) != 0U) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1U] + values[middle]);
}

[[nodiscard]] bool validRotation(const Eigen::Matrix3d& rotation) {
  if (!rotation.allFinite()) {
    return false;
  }
  constexpr double kRotationTolerance = 1.0e-3;
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <
             kRotationTolerance &&
         std::abs(rotation.determinant() - 1.0) < kRotationTolerance;
}

[[nodiscard]] cv::Point2f toCv(const Eigen::Vector2d& point) {
  return cv::Point2f{static_cast<float>(point.x()), static_cast<float>(point.y())};
}

[[nodiscard]] Eigen::Vector2d toEigen(const cv::Point2f& point) {
  return Eigen::Vector2d{static_cast<double>(point.x), static_cast<double>(point.y)};
}

struct TrackState {
  VisualTrackId id;
  cv::Point2f pixel;
  cv::Point2f previous_pixel;
  std::uint32_t age{1U};
  float klt_error{};
  float forward_backward_error{};
  bool newly_detected{true};
};

}  // namespace

bool VisualFrontendConfig::valid(const EquidistantCamera& camera) const noexcept {
  if (!camera.valid()) {
    return false;
  }
  const auto& model = camera.parameters();
  const bool dimensions_fit_opencv =
      model.width <= static_cast<std::uint32_t>(std::numeric_limits<int>::max()) &&
      model.height <= static_cast<std::uint32_t>(std::numeric_limits<int>::max());
  const std::uint64_t track_capacity =
      static_cast<std::uint64_t>(grid_columns) * grid_rows * features_per_cell;
  const bool valid_grid = grid_columns > 0U && grid_rows > 0U && features_per_cell > 0U &&
                          grid_columns <= model.width && grid_rows <= model.height &&
                          track_capacity <= 10'000U && recovery_minimum_tracks <= track_capacity &&
                          keyframe_minimum_tracks <= track_capacity;
  const bool valid_detector =
      std::isfinite(detector_quality) && detector_quality > 0.0 && detector_quality <= 1.0 &&
      std::isfinite(minimum_feature_distance_px) && minimum_feature_distance_px >= 1.0 &&
      detector_block_size >= 3U && (detector_block_size % 2U) == 1U;
  const bool valid_border = image_border_px > 0U && 2U * image_border_px < model.width &&
                            2U * image_border_px < model.height;
  const bool valid_klt =
      klt_window_size >= 3U && (klt_window_size % 2U) == 1U && klt_pyramid_levels <= 8U &&
      klt_max_iterations > 0U && std::isfinite(klt_epsilon) && klt_epsilon > 0.0 &&
      std::isfinite(maximum_klt_error) && maximum_klt_error > 0.0 &&
      std::isfinite(maximum_forward_backward_error_px) && maximum_forward_backward_error_px > 0.0;
  const bool valid_decisions =
      std::isfinite(recovery_minimum_retention) && recovery_minimum_retention >= 0.0 &&
      recovery_minimum_retention <= 1.0 && std::isfinite(keyframe_minimum_spatial_coverage) &&
      keyframe_minimum_spatial_coverage >= 0.0 && keyframe_minimum_spatial_coverage <= 1.0 &&
      std::isfinite(keyframe_maximum_overlap) && keyframe_maximum_overlap >= 0.0 &&
      keyframe_maximum_overlap <= 1.0 && std::isfinite(keyframe_minimum_parallax_px) &&
      keyframe_minimum_parallax_px >= 0.0 && minimum_keyframe_interval.nanoseconds >= 0 &&
      maximum_keyframe_interval.nanoseconds >= minimum_keyframe_interval.nanoseconds &&
      maximum_time_uncertainty.nanoseconds >= 0 && std::isfinite(base_pixel_sigma) &&
      base_pixel_sigma > 0.0;
  return dimensions_fit_opencv && valid_grid && valid_detector && valid_border && valid_klt &&
         valid_decisions;
}

struct GridKltVisualFrontend::Implementation {
  Implementation(EquidistantCamera camera_in, VisualFrontendConfig config_in)
      : camera(std::move(camera_in)), config(config_in), brisk(cv::BRISK::create()) {}

  [[nodiscard]] std::size_t cellIndex(const cv::Point2f& point) const {
    const auto& model = camera.parameters();
    const double clamped_x =
        std::clamp(static_cast<double>(point.x), 0.0, static_cast<double>(model.width) - 1.0);
    const double clamped_y =
        std::clamp(static_cast<double>(point.y), 0.0, static_cast<double>(model.height) - 1.0);
    const auto column = std::min(config.grid_columns - 1U,
                                 static_cast<std::uint32_t>(clamped_x * config.grid_columns /
                                                            static_cast<double>(model.width)));
    const auto row = std::min(config.grid_rows - 1U,
                              static_cast<std::uint32_t>(clamped_y * config.grid_rows /
                                                         static_cast<double>(model.height)));
    return static_cast<std::size_t>(row) * config.grid_columns + column;
  }

  [[nodiscard]] core::Result<cv::Mat, VisualFrontendError> decodeImage(
      const core::CameraFrame& frame) const {
    if (!frame.pixels) {
      return core::Result<cv::Mat, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::InvalidImageLayout,
           "camera frame has no immutable pixel storage"});
    }

    std::uint64_t minimum_stride = 0U;
    int cv_type = 0;
    switch (frame.encoding) {
      case core::ImageEncoding::Mono8:
        minimum_stride = frame.width;
        cv_type = CV_8UC1;
        break;
      case core::ImageEncoding::Bgr8:
        minimum_stride = static_cast<std::uint64_t>(frame.width) * 3U;
        cv_type = CV_8UC3;
        break;
      case core::ImageEncoding::Jpeg:
        return core::Result<cv::Mat, VisualFrontendError>::failure(
            {VisualFrontendErrorCode::UnsupportedImageEncoding,
             "compressed JPEG must be decoded by the ROS adapter"});
    }
    if (frame.stride < minimum_stride) {
      return core::Result<cv::Mat, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::InvalidImageLayout,
           "camera stride is smaller than one image row"});
    }
    const std::uint64_t required_bytes = static_cast<std::uint64_t>(frame.stride) * frame.height;
    if (required_bytes > frame.pixels->size()) {
      return core::Result<cv::Mat, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::InvalidImageLayout,
           "camera storage is smaller than stride times height"});
    }

    const auto rows = static_cast<int>(frame.height);
    const auto columns = static_cast<int>(frame.width);
    cv::Mat view(rows, columns, cv_type, const_cast<std::byte*>(frame.pixels->data()),
                 frame.stride);
    if (frame.encoding == core::ImageEncoding::Mono8) {
      return core::Result<cv::Mat, VisualFrontendError>::success(view.clone());
    }
    cv::Mat grayscale;
    cv::cvtColor(view, grayscale, cv::COLOR_BGR2GRAY);
    return core::Result<cv::Mat, VisualFrontendError>::success(std::move(grayscale));
  }

  void pruneToGridCapacity(VisualTrackingReport& report) {
    const std::size_t cell_count = static_cast<std::size_t>(config.grid_columns) * config.grid_rows;
    std::vector<std::vector<TrackState>> cells(cell_count);
    for (TrackState& track : tracks) {
      cells[cellIndex(track.pixel)].push_back(std::move(track));
    }

    tracks.clear();
    for (auto& cell : cells) {
      std::sort(cell.begin(), cell.end(), [](const TrackState& lhs, const TrackState& rhs) {
        if (lhs.age != rhs.age) {
          return lhs.age > rhs.age;
        }
        return lhs.id < rhs.id;
      });
      if (cell.size() > config.features_per_cell) {
        report.rejected_by_grid_capacity += cell.size() - config.features_per_cell;
        cell.resize(config.features_per_cell);
      }
      for (TrackState& track : cell) {
        tracks.push_back(std::move(track));
      }
    }
  }

  void replenish(const cv::Mat& image, VisualTrackingReport& report) {
    const auto& model = camera.parameters();
    const std::size_t cell_count = static_cast<std::size_t>(config.grid_columns) * config.grid_rows;
    std::vector<std::uint32_t> occupancy(cell_count, 0U);
    cv::Mat mask(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    const int border = static_cast<int>(config.image_border_px);
    mask.rowRange(0, border).setTo(0);
    mask.rowRange(image.rows - border, image.rows).setTo(0);
    mask.colRange(0, border).setTo(0);
    mask.colRange(image.cols - border, image.cols).setTo(0);

    const int exclusion_radius =
        std::max(1, static_cast<int>(std::ceil(config.minimum_feature_distance_px)));
    for (const TrackState& track : tracks) {
      ++occupancy[cellIndex(track.pixel)];
      cv::circle(mask, track.pixel, exclusion_radius, cv::Scalar(0), cv::FILLED, cv::LINE_8);
    }

    for (std::uint32_t row = 0U; row < config.grid_rows; ++row) {
      for (std::uint32_t column = 0U; column < config.grid_columns; ++column) {
        const std::size_t index = static_cast<std::size_t>(row) * config.grid_columns + column;
        if (occupancy[index] >= config.features_per_cell) {
          continue;
        }
        const int x_begin = static_cast<int>(static_cast<std::uint64_t>(column) * model.width /
                                             config.grid_columns);
        const int x_end = static_cast<int>(static_cast<std::uint64_t>(column + 1U) * model.width /
                                           config.grid_columns);
        const int y_begin =
            static_cast<int>(static_cast<std::uint64_t>(row) * model.height / config.grid_rows);
        const int y_end = static_cast<int>(static_cast<std::uint64_t>(row + 1U) * model.height /
                                           config.grid_rows);
        const cv::Rect region{x_begin, y_begin, x_end - x_begin, y_end - y_begin};
        std::vector<cv::Point2f> candidates;
        const int missing = static_cast<int>(config.features_per_cell - occupancy[index]);
        cv::goodFeaturesToTrack(image(region), candidates, missing, config.detector_quality,
                                config.minimum_feature_distance_px, mask(region),
                                static_cast<int>(config.detector_block_size), false, 0.04);
        for (cv::Point2f& candidate : candidates) {
          candidate.x += static_cast<float>(x_begin);
          candidate.y += static_cast<float>(y_begin);
        }
        // OpenCV ranks corners by response. IDs are nevertheless assigned in
        // geometric order, eliminating response-tie nondeterminism.
        std::sort(candidates.begin(), candidates.end(),
                  [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                    if (lhs.y != rhs.y) {
                      return lhs.y < rhs.y;
                    }
                    return lhs.x < rhs.x;
                  });
        for (const cv::Point2f& candidate : candidates) {
          if (!camera.isInsideImage(toEigen(candidate), config.image_border_px)) {
            continue;
          }
          tracks.push_back(TrackState{VisualTrackId(next_track_id++), candidate, candidate, 1U,
                                      0.0F, 0.0F, true});
          ++occupancy[index];
          ++report.replenished_tracks;
          cv::circle(mask, candidate, exclusion_radius, cv::Scalar(0), cv::FILLED, cv::LINE_8);
        }
      }
    }
    report.grid_cells_below_target = static_cast<std::size_t>(
        std::count_if(occupancy.begin(), occupancy.end(),
                      [this](std::uint32_t count) { return count < config.features_per_cell; }));
    const std::size_t occupied_cells = static_cast<std::size_t>(std::count_if(
        occupancy.begin(), occupancy.end(), [](std::uint32_t count) { return count > 0U; }));
    report.spatial_coverage =
        static_cast<double>(occupied_cells) / static_cast<double>(occupancy.size());
  }

  void computeKeyframeMetrics(VisualTrackingReport& report) const {
    if (last_keyframe_pixels.empty()) {
      report.overlap_with_last_keyframe = 1.0;
      report.median_keyframe_parallax_px = 0.0;
      report.rotation_compensated_parallax_available = false;
      return;
    }
    std::size_t common = 0U;
    std::vector<double> parallax;
    parallax.reserve(tracks.size());
    for (const TrackState& track : tracks) {
      const auto match = last_keyframe_pixels.find(track.id.value());
      if (match == last_keyframe_pixels.end()) {
        continue;
      }
      ++common;
      if (rotation_chain_valid) {
        const auto keyframe_ray = camera.unproject(toEigen(match->second));
        if (!keyframe_ray) {
          continue;
        }
        const auto rotation_only_prediction =
            camera.project(R_current_last_keyframe * keyframe_ray.value().unit_ray);
        if (!rotation_only_prediction) {
          continue;
        }
        parallax.push_back((toEigen(track.pixel) - rotation_only_prediction.value().pixel).norm());
      }
    }
    report.overlap_with_last_keyframe =
        static_cast<double>(common) / static_cast<double>(last_keyframe_pixels.size());
    report.rotation_compensated_parallax_available = rotation_chain_valid;
    report.median_keyframe_parallax_px = rotation_chain_valid ? median(std::move(parallax)) : 0.0;
  }

  [[nodiscard]] VisualKeyframeReason chooseKeyframeReason(
      bool first_frame, bool request_keyframe, bool recovery, core::FusionTime time,
      const VisualTrackingReport& report) const {
    if (first_frame) {
      return VisualKeyframeReason::FirstFrame;
    }
    if (request_keyframe) {
      return VisualKeyframeReason::Requested;
    }
    const core::Duration since_keyframe = time - last_keyframe_time;
    if (since_keyframe.nanoseconds < config.minimum_keyframe_interval.nanoseconds) {
      return VisualKeyframeReason::None;
    }
    if (recovery) {
      return VisualKeyframeReason::Recovery;
    }
    if (report.output_tracks < config.keyframe_minimum_tracks ||
        report.spatial_coverage < config.keyframe_minimum_spatial_coverage) {
      return VisualKeyframeReason::InsufficientCoverage;
    }
    if (since_keyframe.nanoseconds >= config.maximum_keyframe_interval.nanoseconds) {
      return VisualKeyframeReason::MaximumInterval;
    }
    if (report.overlap_with_last_keyframe < config.keyframe_maximum_overlap) {
      return VisualKeyframeReason::LowTrackOverlap;
    }
    if (report.rotation_compensated_parallax_available &&
        report.median_keyframe_parallax_px >= config.keyframe_minimum_parallax_px) {
      return VisualKeyframeReason::SufficientParallax;
    }
    return VisualKeyframeReason::None;
  }

  [[nodiscard]] core::Result<std::vector<BriskTrackDescriptor>, VisualFrontendError>
  computeDescriptors(const cv::Mat& image) const {
    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(tracks.size());
    for (std::size_t index = 0U; index < tracks.size(); ++index) {
      cv::KeyPoint keypoint(tracks[index].pixel, 12.0F);
      keypoint.class_id = static_cast<int>(index);
      keypoints.push_back(keypoint);
    }
    cv::Mat descriptor_matrix;
    brisk->compute(image, keypoints, descriptor_matrix);
    if (!descriptor_matrix.empty() &&
        (descriptor_matrix.type() != CV_8UC1 ||
         descriptor_matrix.cols != static_cast<int>(kBriskDescriptorBytes) ||
         descriptor_matrix.rows != static_cast<int>(keypoints.size()))) {
      return core::Result<std::vector<BriskTrackDescriptor>, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::DescriptorFailure,
           "OpenCV BRISK returned an unexpected descriptor layout"});
    }

    std::vector<BriskTrackDescriptor> descriptors;
    descriptors.reserve(keypoints.size());
    for (int row = 0; row < descriptor_matrix.rows; ++row) {
      const int track_index = keypoints[static_cast<std::size_t>(row)].class_id;
      if (track_index < 0 || track_index >= static_cast<int>(tracks.size())) {
        return core::Result<std::vector<BriskTrackDescriptor>, VisualFrontendError>::failure(
            {VisualFrontendErrorCode::DescriptorFailure,
             "OpenCV BRISK did not preserve keypoint identity"});
      }
      BriskTrackDescriptor descriptor;
      descriptor.track = tracks[static_cast<std::size_t>(track_index)].id;
      std::memcpy(descriptor.bytes.data(), descriptor_matrix.ptr(row), kBriskDescriptorBytes);
      descriptors.push_back(descriptor);
    }
    std::sort(descriptors.begin(), descriptors.end(),
              [](const BriskTrackDescriptor& lhs, const BriskTrackDescriptor& rhs) {
                return lhs.track < rhs.track;
              });
    return core::Result<std::vector<BriskTrackDescriptor>, VisualFrontendError>::success(
        std::move(descriptors));
  }

  [[nodiscard]] core::ObservationLineage makeLineage(
      const core::CameraFrame& frame, const std::optional<VisualRotationPrior>& prior,
      RotationPriorStatus prior_status) {
    const core::DerivedRecordId consumer(next_derived_record_id++);
    core::ObservationLineage lineage;
    lineage.id = core::ObservationLineageId(next_lineage_id++);

    core::ObservationUsage image_usage;
    image_usage.slice.root = frame.id;
    image_usage.slice.kind = core::SliceKind::Whole;
    image_usage.slice.calibration = *frame.header.direct_calibration;
    image_usage.role = core::ObservationRole::DerivedSummary;
    image_usage.consumer = consumer;
    lineage.usage.push_back(image_usage);

    if (prior && prior_status == RotationPriorStatus::Applied) {
      for (const core::MeasurementId measurement : prior->imu_support) {
        core::ObservationUsage imu_usage;
        imu_usage.slice.root = measurement;
        imu_usage.slice.kind = core::SliceKind::Whole;
        imu_usage.slice.calibration = prior->imu_calibration;
        imu_usage.role = core::ObservationRole::ConditioningOnly;
        imu_usage.consumer = consumer;
        lineage.usage.push_back(imu_usage);
      }
    }
    return lineage;
  }

  EquidistantCamera camera;
  VisualFrontendConfig config;
  cv::Ptr<cv::BRISK> brisk;
  cv::Mat previous_image;
  std::vector<TrackState> tracks;
  std::optional<core::CameraId> active_camera;
  std::optional<core::CalibrationEpoch> calibration_epoch;
  std::optional<core::SourceEpoch> source_epoch;
  core::FusionTime previous_time;
  core::FusionTime last_keyframe_time;
  std::unordered_map<std::uint64_t, cv::Point2f> last_keyframe_pixels;
  Eigen::Matrix3d R_current_last_keyframe{Eigen::Matrix3d::Identity()};
  bool rotation_chain_valid{true};
  std::uint64_t next_track_id{};
  std::uint64_t next_derived_record_id{};
  std::uint64_t next_lineage_id{};
};

GridKltVisualFrontend::GridKltVisualFrontend(EquidistantCamera camera, VisualFrontendConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(camera), config)) {}

GridKltVisualFrontend::~GridKltVisualFrontend() = default;
GridKltVisualFrontend::GridKltVisualFrontend(GridKltVisualFrontend&&) noexcept = default;
GridKltVisualFrontend& GridKltVisualFrontend::operator=(GridKltVisualFrontend&&) noexcept = default;

core::Result<VisualFrontendOutput, VisualFrontendError> GridKltVisualFrontend::process(
    const core::CameraFrame& frame, std::optional<VisualRotationPrior> rotation_prior,
    bool request_keyframe) {
  const auto total_start = SteadyClock::now();
  // Frontend updates are transactional: malformed inputs and OpenCV failures
  // leave the last accepted image, tracks, lineage counters, and keyframe
  // rotation chain untouched.
  Implementation working_state = *implementation_;
  Implementation& state = working_state;
  if (!state.config.valid(state.camera)) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::InvalidConfiguration,
         "visual frontend or camera configuration is invalid"});
  }
  if (!frame.id.valid() || !frame.camera.valid() || !frame.stamp.source_epoch.valid()) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::InvalidCameraFrame,
         "camera frame has an invalid measurement, camera, or epoch ID"});
  }
  if (!frame.header.direct_calibration || !frame.header.direct_calibration->valid()) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::MissingCalibration,
         "camera frame must identify its direct calibration epoch"});
  }
  if (frame.stamp.uncertainty.nanoseconds < 0 ||
      frame.stamp.uncertainty.nanoseconds > state.config.maximum_time_uncertainty.nanoseconds ||
      frame.stamp.status == core::TimeMappingStatus::Discontinuous ||
      frame.stamp.status == core::TimeMappingStatus::Uncertain) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::TimeUncertain,
         "camera time mapping is discontinuous or exceeds the configured "
         "uncertainty bound"});
  }
  if (frame.width != state.camera.parameters().width ||
      frame.height != state.camera.parameters().height) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::ResolutionChanged,
         "camera frame resolution differs from calibrated resolution"});
  }
  if (state.active_camera && frame.camera != *state.active_camera) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::CameraChanged,
         "one visual frontend instance may track only one camera stream"});
  }
  if (state.calibration_epoch && *frame.header.direct_calibration != *state.calibration_epoch) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::CalibrationChanged,
         "camera calibration epoch changed; construct a frontend with the new "
         "camera model before continuing"});
  }
  if (state.source_epoch && frame.stamp.source_epoch != *state.source_epoch) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::SourceEpochChanged,
         "camera source epoch changed; reset the frontend before continuing"});
  }
  const bool first_frame = state.previous_image.empty();
  if (!first_frame && frame.exposure_midpoint <= state.previous_time) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::NonMonotonicTime,
         "camera exposure midpoint is not strictly increasing"});
  }

  VisualTrackingReport report;
  const auto decode_start = SteadyClock::now();
  core::Result<cv::Mat, VisualFrontendError> decoded =
      core::Result<cv::Mat, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::InvalidImageLayout, "uninitialized"});
  try {
    decoded = state.decodeImage(frame);
  } catch (const cv::Exception& exception) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::InvalidImageLayout, exception.what()});
  }
  if (!decoded) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(decoded.error());
  }
  cv::Mat current_image = std::move(decoded).value();
  report.timing.image_decode_us = elapsedMicroseconds(decode_start, SteadyClock::now());

  report.input_tracks = state.tracks.size();
  const auto flow_start = SteadyClock::now();
  std::vector<double> accepted_fb_errors;
  double accepted_klt_error_sum = 0.0;
  RotationPriorStatus prior_status = RotationPriorStatus::NotProvided;
  bool apply_rotation_prior = false;
  if (rotation_prior) {
    if (first_frame || rotation_prior->previous_exposure_midpoint != state.previous_time ||
        rotation_prior->current_exposure_midpoint != frame.exposure_midpoint) {
      prior_status = RotationPriorStatus::TimeMismatch;
    } else if (!rotation_prior->imu_calibration.valid()) {
      prior_status = RotationPriorStatus::MissingCalibration;
    } else if (rotation_prior->imu_support.empty() ||
               std::any_of(rotation_prior->imu_support.begin(), rotation_prior->imu_support.end(),
                           [](core::MeasurementId measurement) { return !measurement.valid(); })) {
      prior_status = RotationPriorStatus::InvalidSupport;
    } else if (!validRotation(rotation_prior->R_current_previous)) {
      prior_status = RotationPriorStatus::InvalidRotation;
    } else {
      prior_status = RotationPriorStatus::Applied;
      apply_rotation_prior = true;
    }
  }
  report.rotation_prior = prior_status;
  report.time_uncertainty = frame.stamp.uncertainty;
  if (first_frame) {
    state.R_current_last_keyframe.setIdentity();
    state.rotation_chain_valid = true;
  } else if (apply_rotation_prior && state.rotation_chain_valid) {
    state.R_current_last_keyframe =
        rotation_prior->R_current_previous * state.R_current_last_keyframe;
  } else {
    state.rotation_chain_valid = false;
  }

  if (!first_frame && !state.tracks.empty()) {
    std::vector<cv::Point2f> previous_points;
    previous_points.reserve(state.tracks.size());
    for (const TrackState& track : state.tracks) {
      previous_points.push_back(track.pixel);
    }

    std::vector<cv::Point2f> current_points;
    int flow_flags = 0;
    if (apply_rotation_prior) {
      current_points = previous_points;
      flow_flags = cv::OPTFLOW_USE_INITIAL_FLOW;
      for (std::size_t index = 0U; index < state.tracks.size(); ++index) {
        const auto ray = state.camera.unproject(toEigen(previous_points[index]));
        if (!ray) {
          continue;
        }
        const Eigen::Vector3d predicted_ray =
            rotation_prior->R_current_previous * ray.value().unit_ray;
        const auto projection = state.camera.project(predicted_ray);
        if (!projection ||
            !state.camera.isInsideImage(projection.value().pixel, state.config.image_border_px)) {
          continue;
        }
        current_points[index] = toCv(projection.value().pixel);
        ++report.rotation_seeded_tracks;
      }
    }

    std::vector<unsigned char> forward_status;
    std::vector<float> forward_error;
    std::vector<cv::Point2f> backward_points;
    std::vector<unsigned char> backward_status;
    std::vector<float> backward_error;
    try {
      cv::calcOpticalFlowPyrLK(state.previous_image, current_image, previous_points, current_points,
                               forward_status, forward_error,
                               cv::Size(static_cast<int>(state.config.klt_window_size),
                                        static_cast<int>(state.config.klt_window_size)),
                               static_cast<int>(state.config.klt_pyramid_levels),
                               cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                                static_cast<int>(state.config.klt_max_iterations),
                                                state.config.klt_epsilon),
                               flow_flags);
      cv::calcOpticalFlowPyrLK(current_image, state.previous_image, current_points, backward_points,
                               backward_status, backward_error,
                               cv::Size(static_cast<int>(state.config.klt_window_size),
                                        static_cast<int>(state.config.klt_window_size)),
                               static_cast<int>(state.config.klt_pyramid_levels),
                               cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                                static_cast<int>(state.config.klt_max_iterations),
                                                state.config.klt_epsilon));
    } catch (const cv::Exception& exception) {
      return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::OpticalFlowFailure, exception.what()});
    }

    std::vector<TrackState> retained;
    retained.reserve(state.tracks.size());
    for (std::size_t index = 0U; index < state.tracks.size(); ++index) {
      if (forward_status[index] == 0U) {
        ++report.rejected_by_forward_flow;
        continue;
      }
      ++report.forward_flow_successes;
      if (!state.camera.isInsideImage(toEigen(current_points[index]),
                                      state.config.image_border_px)) {
        ++report.rejected_by_border;
        continue;
      }
      if (!std::isfinite(forward_error[index]) ||
          forward_error[index] > state.config.maximum_klt_error) {
        ++report.rejected_by_klt_error;
        continue;
      }
      if (backward_status[index] == 0U) {
        ++report.rejected_by_backward_flow;
        continue;
      }
      ++report.backward_flow_successes;
      const float fb_error = cv::norm(backward_points[index] - previous_points[index]);
      if (!std::isfinite(fb_error) || fb_error > state.config.maximum_forward_backward_error_px) {
        ++report.rejected_by_forward_backward_error;
        continue;
      }

      TrackState track = state.tracks[index];
      track.previous_pixel = track.pixel;
      track.pixel = current_points[index];
      track.age =
          track.age == std::numeric_limits<std::uint32_t>::max() ? track.age : track.age + 1U;
      track.klt_error = forward_error[index];
      track.forward_backward_error = fb_error;
      track.newly_detected = false;
      accepted_klt_error_sum += forward_error[index];
      accepted_fb_errors.push_back(fb_error);
      retained.push_back(track);
    }
    state.tracks = std::move(retained);
    state.pruneToGridCapacity(report);
  }
  report.retained_tracks = state.tracks.size();
  if (report.input_tracks > 0U) {
    report.retained_fraction =
        static_cast<double>(report.retained_tracks) / static_cast<double>(report.input_tracks);
  }
  if (report.retained_tracks > 0U) {
    report.mean_klt_error =
        accepted_klt_error_sum /
        static_cast<double>(report.retained_tracks + report.rejected_by_grid_capacity);
  }
  report.median_forward_backward_error_px = median(accepted_fb_errors);
  report.timing.optical_flow_us = elapsedMicroseconds(flow_start, SteadyClock::now());

  const bool recovery =
      !first_frame && (report.retained_tracks < state.config.recovery_minimum_tracks ||
                       report.retained_fraction < state.config.recovery_minimum_retention);
  const auto replenish_start = SteadyClock::now();
  try {
    state.replenish(current_image, report);
  } catch (const cv::Exception& exception) {
    return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
        {VisualFrontendErrorCode::OpticalFlowFailure, exception.what()});
  }
  std::sort(state.tracks.begin(), state.tracks.end(),
            [](const TrackState& lhs, const TrackState& rhs) { return lhs.id < rhs.id; });
  report.output_tracks = state.tracks.size();
  report.timing.replenishment_us = elapsedMicroseconds(replenish_start, SteadyClock::now());

  state.computeKeyframeMetrics(report);
  report.keyframe_reason = state.chooseKeyframeReason(first_frame, request_keyframe, recovery,
                                                      frame.exposure_midpoint, report);
  const bool keyframe = report.keyframe_reason != VisualKeyframeReason::None;
  if (recovery) {
    report.quality = VisualTrackingQuality::Recovery;
  } else if (report.retained_fraction < 0.65 ||
             report.output_tracks < state.config.recovery_minimum_tracks) {
    report.quality = VisualTrackingQuality::Degraded;
  } else {
    report.quality = VisualTrackingQuality::Nominal;
  }

  std::vector<BriskTrackDescriptor> descriptors;
  const auto descriptor_start = SteadyClock::now();
  if (keyframe) {
    try {
      auto descriptor_result = state.computeDescriptors(current_image);
      if (!descriptor_result) {
        return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
            descriptor_result.error());
      }
      descriptors = std::move(descriptor_result).value();
    } catch (const cv::Exception& exception) {
      return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::DescriptorFailure, exception.what()});
    }
  }
  report.descriptor_tracks = descriptors.size();
  report.timing.descriptor_us = elapsedMicroseconds(descriptor_start, SteadyClock::now());

  VisualFrontendOutput output;
  output.source_frame = frame.id;
  output.camera = frame.camera;
  output.exposure_midpoint = frame.exposure_midpoint;
  output.keyframe = keyframe;
  output.descriptors = std::move(descriptors);
  output.lineage = state.makeLineage(frame, rotation_prior, prior_status);
  output.features.reserve(state.tracks.size());
  const double delta_seconds =
      first_frame
          ? 0.0
          : static_cast<double>((frame.exposure_midpoint - state.previous_time).nanoseconds) /
                1.0e9;
  for (const TrackState& track : state.tracks) {
    auto ray = state.camera.unproject(toEigen(track.pixel));
    if (!ray) {
      return core::Result<VisualFrontendOutput, VisualFrontendError>::failure(
          {VisualFrontendErrorCode::CameraModelFailure, ray.error().detail});
    }
    VisualFeatureObservation observation;
    observation.track = track.id;
    observation.distorted_pixel = toEigen(track.pixel);
    observation.unit_bearing = ray.value().unit_ray;
    if (!track.newly_detected && delta_seconds > 0.0) {
      observation.pixel_velocity_per_second =
          (toEigen(track.pixel) - toEigen(track.previous_pixel)) / delta_seconds;
    }
    const double normalized_error =
        std::clamp(static_cast<double>(track.klt_error) / state.config.maximum_klt_error, 0.0, 2.0);
    const double sigma = state.config.base_pixel_sigma * (1.0 + normalized_error);
    observation.pixel_covariance = sigma * sigma * Eigen::Matrix2d::Identity();
    const double time_sigma_seconds =
        static_cast<double>(frame.stamp.uncertainty.nanoseconds) / 1.0e9;
    observation.pixel_covariance +=
        time_sigma_seconds * time_sigma_seconds *
        (observation.pixel_velocity_per_second * observation.pixel_velocity_per_second.transpose());
    observation.age = track.age;
    observation.klt_error = track.klt_error;
    observation.forward_backward_error_px = track.forward_backward_error;
    observation.newly_detected = track.newly_detected;
    output.features.push_back(observation);
  }

  if (keyframe) {
    state.last_keyframe_pixels.clear();
    state.last_keyframe_pixels.reserve(state.tracks.size());
    for (const TrackState& track : state.tracks) {
      state.last_keyframe_pixels.emplace(track.id.value(), track.pixel);
    }
    state.last_keyframe_time = frame.exposure_midpoint;
    state.R_current_last_keyframe.setIdentity();
    state.rotation_chain_valid = true;
  }
  state.previous_image = std::move(current_image);
  state.previous_time = frame.exposure_midpoint;
  state.active_camera = frame.camera;
  state.calibration_epoch = *frame.header.direct_calibration;
  state.source_epoch = frame.stamp.source_epoch;
  report.timing.total_us = elapsedMicroseconds(total_start, SteadyClock::now());
  output.report = report;
  *implementation_ = std::move(working_state);

  return core::Result<VisualFrontendOutput, VisualFrontendError>::success(std::move(output));
}

void GridKltVisualFrontend::reset() {
  implementation_->previous_image.release();
  implementation_->tracks.clear();
  implementation_->active_camera.reset();
  implementation_->calibration_epoch.reset();
  implementation_->source_epoch.reset();
  implementation_->previous_time = {};
  implementation_->last_keyframe_time = {};
  implementation_->last_keyframe_pixels.clear();
  implementation_->R_current_last_keyframe.setIdentity();
  implementation_->rotation_chain_valid = true;
}

bool GridKltVisualFrontend::initialized() const noexcept {
  return !implementation_->previous_image.empty();
}

}  // namespace meridian::local
