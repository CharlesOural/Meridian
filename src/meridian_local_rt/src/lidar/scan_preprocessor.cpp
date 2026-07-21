#include "meridian/local_rt/lidar/scan_preprocessor.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "meridian/local_rt/lidar/voxel_grid.hpp"

namespace meridian::local_rt::lidar {
namespace {

using Clock = std::chrono::steady_clock;

std::int64_t elapsedNs(const Clock::time_point begin) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count();
}

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

Sophus::SE3d sophus(const core::Pose3d& pose) {
  const core::Quaterniond& q = pose.rotation();
  const core::Vec3d& p = pose.translation();
  return Sophus::SE3d(Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z()),
                      Eigen::Vector3d(p.x, p.y, p.z));
}

Sophus::SE3d interpolate(const core::Pose3d& first, const core::Pose3d& second, double alpha) {
  const Sophus::SE3d a = sophus(first);
  const Sophus::SE3d b = sophus(second);
  const Eigen::Quaterniond qa(a.unit_quaternion());
  const Eigen::Quaterniond qb(b.unit_quaternion());
  return Sophus::SE3d(qa.slerp(alpha, qb).normalized(),
                      (1.0 - alpha) * a.translation() + alpha * b.translation());
}

struct TimedPose final {
  core::TimeNs time;
  core::Pose3d pose;
};

std::optional<Sophus::SE3d> poseAt(std::span<const TimedPose> poses, core::TimeNs time) {
  if (poses.empty() || time < poses.front().time || time > poses.back().time) {
    return std::nullopt;
  }
  const auto upper =
      std::lower_bound(poses.begin(), poses.end(), time,
                       [](const TimedPose& item, core::TimeNs query) { return item.time < query; });
  if (upper == poses.end()) {
    return sophus(poses.back().pose);
  }
  if (upper->time == time || upper == poses.begin()) {
    return sophus(upper->pose);
  }
  const TimedPose& before = *std::prev(upper);
  const auto numerator = core::TimeNs::checkedDifference(time, before.time);
  const auto denominator = core::TimeNs::checkedDifference(upper->time, before.time);
  if (!numerator.has_value() || !denominator.has_value() || *denominator <= 0) {
    return std::nullopt;
  }
  const double alpha = static_cast<double>(*numerator) / static_cast<double>(*denominator);
  return interpolate(before.pose, upper->pose, alpha);
}

PointCloud filterPoints(std::span<const Point3d> points, double minimum_range_m,
                        double maximum_range_m, std::size_t& rejected) {
  PointCloud filtered;
  filtered.reserve(points.size());
  const double minimum_squared = minimum_range_m * minimum_range_m;
  const double maximum_squared = maximum_range_m * maximum_range_m;
  for (const Point3d& point : points) {
    const double range_squared = point.squaredNorm();
    if (!point.allFinite() || !std::isfinite(range_squared) || range_squared < minimum_squared ||
        range_squared > maximum_squared) {
      ++rejected;
      continue;
    }
    filtered.push_back(point);
  }
  return filtered;
}

PreparedScan finish(core::TimeNs time, core::StateId state_id, PointCloud points,
                    const ScanPreprocessorOptions& options, ScanPreprocessorTiming timing,
                    ScanPreprocessorStats stats, const Clock::time_point total_begin) {
  const Clock::time_point target_begin = Clock::now();
  VoxelDownsampleResult target =
      deterministicVoxelDownsample(points, options.target_downsample_voxel_m);
  timing.target_downsample_ns = elapsedNs(target_begin);

  const Clock::time_point source_begin = Clock::now();
  VoxelDownsampleResult source =
      deterministicVoxelDownsample(points, options.source_downsample_voxel_m);
  timing.source_downsample_ns = elapsedNs(source_begin);
  stats.target_points = target.points.size();
  stats.source_points = source.points.size();
  timing.total_ns = elapsedNs(total_begin);
  return PreparedScan{
      .frame = ScanFrame{.state_id = state_id,
                         .time = time,
                         .target_points_lidar = std::move(target.points),
                         .source_points_lidar = std::move(source.points)},
      .timing = timing,
      .stats = stats,
  };
}

}  // namespace

ScanPreprocessor::ScanPreprocessor(ScanPreprocessorOptions options) : options_(options) {
  if (!positiveFinite(options_.minimum_range_m) || !positiveFinite(options_.maximum_range_m) ||
      options_.minimum_range_m >= options_.maximum_range_m ||
      !positiveFinite(options_.target_downsample_voxel_m) ||
      !positiveFinite(options_.source_downsample_voxel_m)) {
    throw std::invalid_argument("scan preprocessor options are incomplete or nonphysical");
  }
}

PreparedScan ScanPreprocessor::prepare(const core::LidarSweep& sweep,
                                       const core::NavigationState& propagation_seed,
                                       core::StateId state_id,
                                       const DensePropagation& propagation) const {
  const Clock::time_point total_begin = Clock::now();
  ScanPreprocessorTiming timing;
  ScanPreprocessorStats stats;
  stats.input_points = sweep.points().size();

  std::vector<TimedPose> poses;
  poses.reserve(propagation.samples.size() + 1U);
  if (propagation.samples.empty()) {
    throw std::invalid_argument("deskew requires a non-empty dense propagation");
  }
  poses.push_back({propagation_seed.time(), propagation_seed.odomFromImu()});
  for (const DenseImuSample& sample : propagation.samples) {
    poses.push_back({sample.time, sample.odom_from_imu});
  }
  if (poses.back().time != propagation.endpoint.time()) {
    throw std::invalid_argument("dense propagation endpoint does not close its pose path");
  }

  const Clock::time_point filter_begin = Clock::now();
  struct TimedPoint final {
    Point3d point;
    core::TimeNs time;
  };
  std::vector<TimedPoint, Eigen::aligned_allocator<TimedPoint>> filtered;
  filtered.reserve(sweep.points().size());
  const double minimum_squared = options_.minimum_range_m * options_.minimum_range_m;
  const double maximum_squared = options_.maximum_range_m * options_.maximum_range_m;
  for (const core::LidarPoint& raw : sweep.points()) {
    const Point3d point(static_cast<double>(raw.x), static_cast<double>(raw.y),
                        static_cast<double>(raw.z));
    const double squared_range = point.squaredNorm();
    if (!point.allFinite() || !std::isfinite(squared_range) || squared_range < minimum_squared ||
        squared_range > maximum_squared) {
      ++stats.range_rejected_points;
      continue;
    }
    const auto point_time =
        core::TimeNs::checkedAdd(sweep.header().measurementTime(), raw.time_offset_ns);
    const auto sensor_time =
        point_time.has_value()
            ? core::TimeNs::checkedAdd(*point_time, options_.lidar_time_offset_to_imu_ns)
            : std::nullopt;
    if (!sensor_time.has_value()) {
      ++stats.support_rejected_points;
      continue;
    }
    filtered.push_back({point, *sensor_time});
  }
  timing.range_filter_ns = elapsedNs(filter_begin);

  const Clock::time_point deskew_begin = Clock::now();
  const Sophus::SE3d odom_from_imu_end = sophus(propagation.endpoint.odomFromImu());
  const Sophus::SE3d imu_from_lidar = sophus(options_.T_imu_lidar);
  const Sophus::SE3d lidar_end_from_odom = (odom_from_imu_end * imu_from_lidar).inverse();
  PointCloud deskewed;
  deskewed.reserve(filtered.size());
  for (const TimedPoint& timed : filtered) {
    const auto odom_from_imu_at_point = poseAt(poses, timed.time);
    if (!odom_from_imu_at_point.has_value()) {
      ++stats.support_rejected_points;
      continue;
    }
    deskewed.push_back(lidar_end_from_odom * *odom_from_imu_at_point * imu_from_lidar *
                       timed.point);
  }
  timing.deskew_ns = elapsedNs(deskew_begin);
  stats.deskewed_points = deskewed.size();
  return finish(propagation.endpoint.time(), state_id, std::move(deskewed), options_, timing, stats,
                total_begin);
}

PreparedScan ScanPreprocessor::prepareSweepEndPoints(core::TimeNs time, core::StateId state_id,
                                                     std::span<const Point3d> points) const {
  const Clock::time_point total_begin = Clock::now();
  ScanPreprocessorTiming timing;
  ScanPreprocessorStats stats;
  stats.input_points = points.size();
  const Clock::time_point filter_begin = Clock::now();
  PointCloud filtered = filterPoints(points, options_.minimum_range_m, options_.maximum_range_m,
                                     stats.range_rejected_points);
  timing.range_filter_ns = elapsedNs(filter_begin);
  stats.deskewed_points = filtered.size();
  return finish(time, state_id, std::move(filtered), options_, timing, stats, total_begin);
}

}  // namespace meridian::local_rt::lidar
