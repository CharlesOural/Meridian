#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cstddef>
#include <vector>

#include "meridian/core/geometry.hpp"
#include "meridian/core/ids.hpp"
#include "meridian/core/navigation.hpp"
#include "meridian/core/time.hpp"

namespace meridian::local_rt {

class ImuIntegrationSegment final {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuIntegrationSegment(core::TimeRange support, Eigen::Vector3d angular_velocity_rad_s,
                        Eigen::Vector3d specific_force_m_s2);

  [[nodiscard]] const core::TimeRange& support() const noexcept { return support_; }
  [[nodiscard]] const Eigen::Vector3d& angularVelocityRadS() const noexcept {
    return angular_velocity_rad_s_;
  }
  [[nodiscard]] const Eigen::Vector3d& specificForceMS2() const noexcept {
    return specific_force_m_s2_;
  }
  [[nodiscard]] double durationSeconds() const noexcept;

private:
  core::TimeRange support_;
  Eigen::Vector3d angular_velocity_rad_s_;
  Eigen::Vector3d specific_force_m_s2_;
};

using ImuIntegrationSegments =
    std::vector<ImuIntegrationSegment, Eigen::aligned_allocator<ImuIntegrationSegment>>;

class ImuInterval final {
public:
  ImuInterval(core::TimeRange support, ImuIntegrationSegments segments,
              std::size_t source_sample_count);

  [[nodiscard]] const core::TimeRange& support() const noexcept { return support_; }
  [[nodiscard]] const ImuIntegrationSegments& segments() const noexcept { return segments_; }
  [[nodiscard]] std::size_t sourceSampleCount() const noexcept { return source_sample_count_; }
  [[nodiscard]] double durationSeconds() const noexcept;

private:
  core::TimeRange support_;
  ImuIntegrationSegments segments_;
  std::size_t source_sample_count_{};
};

struct DenseImuSample final {
  core::TimeNs time;
  core::Pose3d odom_from_imu;
  core::Vec3d velocity_odom_m_s;
};

struct DensePropagation final {
  std::vector<DenseImuSample> samples;
  core::NavigationState endpoint;
};

}  // namespace meridian::local_rt
