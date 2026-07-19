#include "meridian/local/lidar_deskew.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace meridian::local {
namespace {

[[nodiscard]] DeskewError deskewError(DeskewErrorCode code,
                                      std::string detail) {
  return DeskewError{code, std::move(detail)};
}

class DiscretePoseInterpolator {
 public:
  explicit DiscretePoseInterpolator(
      const std::vector<TimedNavState>& trajectory)
      : trajectory_(trajectory), segment_logs_(trajectory.size() - 1U) {
    for (std::size_t index = 0U; index + 1U < trajectory_.size(); ++index) {
      const core::Pose3d& left = trajectory_[index].state.T_odom_imu;
      const core::Pose3d& right = trajectory_[index + 1U].state.T_odom_imu;
      if (!left.matrix().allFinite() || !right.matrix().allFinite()) {
        continue;
      }
      const core::Pose3d relative = left.inverse() * right;
      if (!relative.matrix().allFinite()) {
        continue;
      }
      const core::Pose3d::Tangent relative_log = relative.log();
      if (relative_log.allFinite()) {
        segment_logs_[index] = relative_log;
      }
    }
  }

  [[nodiscard]] core::Result<core::Pose3d, DeskewError> poseAt(
      core::FusionTime time) const {
    using Result = core::Result<core::Pose3d, DeskewError>;
    if (time < trajectory_.front().time || time > trajectory_.back().time) {
      return Result::failure(deskewError(
          DeskewErrorCode::PointOutsideTrajectory,
          "requested deskew time is outside the discrete trajectory"));
    }
    const auto right = std::lower_bound(
        trajectory_.begin(), trajectory_.end(), time,
        [](const TimedNavState& state, core::FusionTime query) {
          return state.time < query;
        });
    if (right == trajectory_.end()) {
      return Result::success(trajectory_.back().state.T_odom_imu);
    }
    if (right->time == time || right == trajectory_.begin()) {
      return Result::success(right->state.T_odom_imu);
    }
    const auto left = std::prev(right);
    const std::size_t segment_index =
        static_cast<std::size_t>(std::distance(trajectory_.begin(), left));
    const double alpha =
        static_cast<double>((time - left->time).nanoseconds) /
        static_cast<double>((right->time - left->time).nanoseconds);
    if (segment_logs_[segment_index].has_value()) {
      return Result::success(
          left->state.T_odom_imu *
          core::Pose3d::exp(alpha * *segment_logs_[segment_index]));
    }

    // Preserve the scalar path's non-finite behavior for an invalid pose
    // segment. Finite segments always use the precomputed logarithm above.
    const core::Pose3d relative =
        left->state.T_odom_imu.inverse() * right->state.T_odom_imu;
    return Result::success(
        left->state.T_odom_imu *
        core::Pose3d::exp(alpha * relative.log()));
  }

 private:
  const std::vector<TimedNavState>& trajectory_;
  std::vector<std::optional<core::Pose3d::Tangent>> segment_logs_;
};

}  // namespace

core::Result<DeskewedSweep, DeskewError> deskewLidarSweep(
    const core::LidarSweep& sweep, core::FusionTime reference_time,
    const core::Pose3d& T_imu_lidar,
    const std::vector<TimedNavState>& trajectory,
    std::vector<core::MeasurementId> imu_support) {
  using Result = core::Result<DeskewedSweep, DeskewError>;
  if (!sweep.points || sweep.points->empty() || !sweep.acquisition.valid()) {
    return Result::failure(deskewError(
        DeskewErrorCode::EmptySweep,
        "deskew requires a non-empty sweep with valid acquisition support"));
  }
  if (trajectory.empty()) {
    return Result::failure(deskewError(DeskewErrorCode::EmptyTrajectory,
                                       "deskew trajectory is empty"));
  }
  for (std::size_t index = 1U; index < trajectory.size(); ++index) {
    if (trajectory[index].time <= trajectory[index - 1U].time) {
      return Result::failure(deskewError(
          DeskewErrorCode::InvalidTrajectory,
          "deskew trajectory timestamps must be strictly increasing"));
    }
  }
  if (reference_time < trajectory.front().time ||
      reference_time > trajectory.back().time) {
    return Result::failure(deskewError(
        DeskewErrorCode::ReferenceOutsideTrajectory,
        "deskew reference time is outside the discrete trajectory"));
  }
  if (!T_imu_lidar.matrix().allFinite()) {
    return Result::failure(deskewError(
        DeskewErrorCode::NonFiniteTransform,
        "T_imu_lidar contains a non-finite value"));
  }

  const DiscretePoseInterpolator pose_interpolator(trajectory);
  auto reference_pose_result = pose_interpolator.poseAt(reference_time);
  if (!reference_pose_result) {
    return Result::failure(reference_pose_result.error());
  }
  const core::Pose3d T_odom_imu_reference = reference_pose_result.value();
  const core::Pose3d T_reference_odom = T_odom_imu_reference.inverse();

  auto output = std::make_unique<core::LidarPoints>();
  output->reserve(sweep.points->size());
  std::unordered_map<std::int32_t, core::Pose3d>
      T_reference_imu_lidar_by_offset;
  const std::size_t expected_unique_offsets =
      sweep.layout.organized && sweep.layout.width > 0U
          ? std::min(sweep.points->size(),
                     static_cast<std::size_t>(sweep.layout.width))
          : sweep.points->size();
  T_reference_imu_lidar_by_offset.reserve(expected_unique_offsets);
  std::size_t pose_interpolations = 0U;
  for (const auto& point : *sweep.points) {
    const core::FusionTime point_time =
        sweep.acquisition.start + core::Duration{point.time_offset_ns};
    if (!sweep.acquisition.contains(point_time)) {
      return Result::failure(deskewError(
          DeskewErrorCode::PointOutsideTrajectory,
          "point offset lies outside the immutable sweep acquisition interval"));
    }

    auto cached_transform =
        T_reference_imu_lidar_by_offset.find(point.time_offset_ns);
    if (cached_transform == T_reference_imu_lidar_by_offset.end()) {
      auto point_pose_result = pose_interpolator.poseAt(point_time);
      if (!point_pose_result) {
        return Result::failure(point_pose_result.error());
      }
      const core::Pose3d T_reference_imu_lidar =
          T_reference_odom * point_pose_result.value() * T_imu_lidar;
      cached_transform =
          T_reference_imu_lidar_by_offset
              .emplace(point.time_offset_ns, T_reference_imu_lidar)
              .first;
      ++pose_interpolations;
    }

    const Eigen::Vector3d point_lidar{point.x, point.y, point.z};
    const Eigen::Vector3d point_reference_imu =
        cached_transform->second * point_lidar;
    if (!point_reference_imu.allFinite()) {
      return Result::failure(deskewError(
          DeskewErrorCode::NonFiniteTransform,
          "deskew generated a non-finite point"));
    }
    core::LidarPoint transformed = point;
    transformed.x = static_cast<float>(point_reference_imu.x());
    transformed.y = static_cast<float>(point_reference_imu.y());
    transformed.z = static_cast<float>(point_reference_imu.z());
    output->push_back(transformed);
  }

  DeskewedSweep deskewed;
  deskewed.source = sweep.id;
  deskewed.reference_time = reference_time;
  deskewed.T_odom_imu_reference = T_odom_imu_reference;
  deskewed.layout = sweep.layout;
  deskewed.points_in_reference_imu = std::move(output);
  deskewed.imu_support = std::move(imu_support);
  deskewed.pose_interpolations = pose_interpolations;
  return Result::success(std::move(deskewed));
}

}  // namespace meridian::local
