#include "meridian/local_rt/initialization/bootstrap_odometry.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "meridian/local_rt/lidar/voxel_grid.hpp"

namespace meridian::local_rt::initialization {
namespace {

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

void validate(const BootstrapOdometryOptions& options) {
  if (!positiveFinite(options.downsample_voxel_size_m) || options.minimum_points < 3U ||
      !positiveFinite(options.maximum_accepted_rmse_m) ||
      !positiveFinite(options.maximum_accepted_condition_number)) {
    throw std::invalid_argument("bootstrap odometry acceptance options are invalid");
  }
  if (options.minimum_points > options.registration.max_source_points ||
      options.minimum_points < options.registration.minimum_correspondences) {
    throw std::invalid_argument(
        "bootstrap minimum_points is incompatible with registration point bounds");
  }
  const std::size_t points_per_voxel = options.target.max_points_per_voxel;
  if (points_per_voxel == 0U) {
    throw std::invalid_argument("bootstrap target point capacity is invalid");
  }
  const std::size_t required_voxels =
      options.minimum_points / points_per_voxel +
      static_cast<std::size_t>(options.minimum_points % points_per_voxel != 0U);
  if (options.target.max_voxels < required_voxels) {
    throw std::invalid_argument("bootstrap target cannot hold minimum_points");
  }
}

}  // namespace

BootstrapOdometry::BootstrapOdometry(BootstrapOdometryOptions options)
    : options_(std::move(options)), target_(options_.target), registration_(options_.registration) {
  validate(options_);
}

void BootstrapOdometry::reset() {
  target_.clear();
  accepted_poses_.clear();
}

Sophus::SE3d BootstrapOdometry::constantVelocityGuess(core::TimeNs time) const {
  if (accepted_poses_.empty()) {
    return {};
  }
  const AcceptedPose& last = accepted_poses_.back();
  if (accepted_poses_.size() == 1U) {
    return last.odom_from_lidar;
  }

  const AcceptedPose& previous = accepted_poses_[accepted_poses_.size() - 2U];
  const std::optional<std::int64_t> previous_duration =
      core::TimeNs::checkedDifference(last.time, previous.time);
  const std::optional<std::int64_t> prediction_duration =
      core::TimeNs::checkedDifference(time, last.time);
  if (!previous_duration.has_value() || *previous_duration <= 0 ||
      !prediction_duration.has_value() || *prediction_duration <= 0) {
    return last.odom_from_lidar;
  }

  const double duration_ratio =
      static_cast<double>(*prediction_duration) / static_cast<double>(*previous_duration);
  const auto relative_motion = (previous.odom_from_lidar.inverse() * last.odom_from_lidar).log();
  return last.odom_from_lidar * Sophus::SE3d::exp(duration_ratio * relative_motion);
}

BootstrapFrame BootstrapOdometry::add(const core::LidarSweep& sweep,
                                      std::span<const lidar::Point3d> points_at_sweep_end) {
  lidar::VoxelDownsampleResult downsampled =
      lidar::deterministicVoxelDownsample(points_at_sweep_end, options_.downsample_voxel_size_m);
  BootstrapFrame frame{
      .measurement_id = sweep.header().measurementId(),
      .time = sweep.acquisitionEnd(),
      .initial_odom_from_lidar = Sophus::SE3d{},
      .odom_from_lidar = Sophus::SE3d{},
      .registration_points = std::move(downsampled.points),
      .downsample = downsampled.stats,
      .quality = {},
      .registration_status = std::nullopt,
      .target_update = std::nullopt,
      .outcome = BootstrapFrameOutcome::kRejectedPointCount,
      .accepted = false,
  };
  frame.quality.input_source_points = downsampled.stats.input_points;
  frame.quality.finite_source_points = downsampled.stats.finite_points;

  if (!accepted_poses_.empty()) {
    const std::optional<std::int64_t> elapsed =
        core::TimeNs::checkedDifference(frame.time, accepted_poses_.back().time);
    if (!elapsed.has_value() || *elapsed <= 0) {
      frame.outcome = BootstrapFrameOutcome::kRejectedTimestamp;
      return frame;
    }
  }
  frame.initial_odom_from_lidar = constantVelocityGuess(frame.time);
  frame.odom_from_lidar = frame.initial_odom_from_lidar;

  if (frame.registration_points.size() < options_.minimum_points) {
    return frame;
  }
  if (frame.registration_points.size() > options_.registration.max_source_points) {
    frame.outcome = BootstrapFrameOutcome::kRejectedSourcePointLimit;
    return frame;
  }

  if (target_.empty()) {
    frame.initial_odom_from_lidar = Sophus::SE3d{};
    frame.odom_from_lidar = Sophus::SE3d();
    lidar::BoundedVoxelTarget staged_target = target_;
    frame.target_update =
        staged_target.updateTargetFrame(frame.registration_points, Eigen::Vector3d::Zero());
    if (staged_target.pointCount() < options_.minimum_points) {
      frame.outcome = BootstrapFrameOutcome::kRejectedAnchorAdmission;
      return frame;
    }
    target_ = std::move(staged_target);
    frame.outcome = BootstrapFrameOutcome::kAcceptedAnchor;
    frame.accepted = true;
    accepted_poses_.push_back(AcceptedPose{frame.time, frame.odom_from_lidar});
    return frame;
  }

  lidar::PointToPointRegistrationResult result =
      registration_.align(frame.registration_points, target_, frame.initial_odom_from_lidar);
  frame.odom_from_lidar = result.T_target_source;
  frame.quality = result.quality;
  frame.registration_status = result.status;
  frame.accepted =
      result.hasUsableEstimate() && frame.quality.correspondences >= options_.minimum_points &&
      std::isfinite(frame.quality.point_rmse_m) &&
      frame.quality.point_rmse_m <= options_.maximum_accepted_rmse_m &&
      std::isfinite(frame.quality.hessian_condition_number) &&
      frame.quality.hessian_condition_number <= options_.maximum_accepted_condition_number;
  if (!frame.accepted) {
    frame.outcome = BootstrapFrameOutcome::kRejectedRegistration;
    return frame;
  }

  lidar::BoundedVoxelTarget staged_target = target_;
  frame.target_update = staged_target.update(frame.registration_points, frame.odom_from_lidar,
                                             frame.odom_from_lidar.translation());
  if (staged_target.pointCount() < options_.minimum_points) {
    frame.accepted = false;
    frame.outcome = BootstrapFrameOutcome::kRejectedTargetAdmission;
    return frame;
  }

  target_ = std::move(staged_target);
  frame.outcome = BootstrapFrameOutcome::kAcceptedRegistration;
  accepted_poses_.push_back(AcceptedPose{frame.time, frame.odom_from_lidar});
  return frame;
}

}  // namespace meridian::local_rt::initialization
