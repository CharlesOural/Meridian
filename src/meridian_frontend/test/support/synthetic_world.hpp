#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

#include "meridian/common/cloud.hpp"
#include "meridian/common/frame.hpp"
#include "meridian/common/measure_group.hpp"
#include "meridian/common/pose.hpp"
#include "meridian/common/preprocessed_group.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"

namespace meridian::lio::test {

// Deterministic synthetic rig: an axis-aligned box room sampled by a ray-cast spinning
// LiDAR from an analytic trajectory, with IMU samples derived from the same trajectory
// as body-frame specific force (gravity included) and angular rate. All randomness
// (range noise) comes from one fixed-seed generator consumed in generation order, so
// the same configuration always produces a bit-identical measurement stream.
//
// The trajectory holds still for hold_s (long enough for static initialization), then
// drives a horizontal circle whose angular speed ramps linearly over ramp_s — velocity
// is continuous, so no impulse escapes the IMU. radius_m == 0 keeps a permanent
// standstill. The body axes stay level (yaw-only attitude tracking the tangent).

constexpr double kWorldGravity = 9.81;
constexpr Timestamp kWorldEpoch = 1'000'000'000;

struct WorldConfig {
  // room interior; the trajectory must stay inside
  Eigen::Vector3d room_min{-4.0, -4.0, 0.0};
  Eigen::Vector3d room_max{4.0, 4.0, 4.0};
  bool floor_only = false;  // single-plane scene: only the z = room_min.z() plane returns

  // trajectory
  double hold_s = 1.5;
  double ramp_s = 2.0;
  double radius_m = 2.0;
  double rate_rad_s = 0.21;  // steady-state angular speed of the circuit
  double height_m = 1.5;

  // sensors
  double lidar_rate_hz = 10.0;
  int n_azimuth = 240;
  int n_rings = 16;
  double fov_up_deg = 30.0;
  double fov_down_deg = -30.0;
  double max_range_m = 30.0;
  double range_noise_std = 0.005;
  double imu_rate_hz = 200.0;
  unsigned seed = 42;
  bool attach_images = true;
};

class SyntheticWorld {
public:
  explicit SyntheticWorld(WorldConfig cfg = {})
      : cfg_(cfg),
        lidar_period_ns_(static_cast<Duration>(std::llround(1e9 / cfg.lidar_rate_hz))),
        imu_period_ns_(static_cast<Duration>(std::llround(1e9 / cfg.imu_rate_hz))),
        rng_(cfg.seed),
        range_noise_(0.0, cfg.range_noise_std > 0.0 ? cfg.range_noise_std : 1.0) {
    image_data_ = std::make_shared<const std::vector<std::uint8_t>>(16, 128);
  }

  Duration lidarPeriod() const { return lidar_period_ns_; }
  Timestamp groupBegin(int k) const { return kWorldEpoch + k * lidar_period_ns_; }
  Timestamp groupEnd(int k) const { return groupBegin(k) + lidar_period_ns_; }

  // ---- ground truth ----

  Pose poseAt(Timestamp stamp) const {
    const Kin k = kinAt(secondsOf(stamp));
    const Eigen::Vector3d p(cfg_.radius_m * std::cos(k.theta), cfg_.radius_m * std::sin(k.theta),
                            cfg_.height_m);
    const double yaw = k.theta + std::numbers::pi / 2.0;  // heading along the tangent
    return Pose{Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())), p};
  }

  Eigen::Vector3d velWorldAt(Timestamp stamp) const {
    const Kin k = kinAt(secondsOf(stamp));
    return cfg_.radius_m * k.sigma * Eigen::Vector3d(-std::sin(k.theta), std::cos(k.theta), 0.0);
  }

  Eigen::Vector3d accWorldAt(Timestamp stamp) const {
    const Kin k = kinAt(secondsOf(stamp));
    const Eigen::Vector3d tangent(-std::sin(k.theta), std::cos(k.theta), 0.0);
    const Eigen::Vector3d inward(-std::cos(k.theta), -std::sin(k.theta), 0.0);
    return cfg_.radius_m * (k.sigma_dot * tangent + k.sigma * k.sigma * inward);
  }

  // Specific force in the body frame: f = R^T (a - g), the quantity an ideal
  // accelerometer reads (at rest f = R^T * (0, 0, +9.81)).
  ImuSample imuAt(Timestamp stamp) const {
    const Kin k = kinAt(secondsOf(stamp));
    ImuSample s;
    s.stamp = stamp;
    const Eigen::Vector3d g_world(0.0, 0.0, -kWorldGravity);
    s.acc = poseAt(stamp).q.conjugate() * (accWorldAt(stamp) - g_world);
    s.gyro = Eigen::Vector3d(0.0, 0.0, k.sigma);  // level attitude: body rate is the yaw rate
    return s;
  }

  // Absolute distance from a world point to the nearest room surface plane.
  double surfaceDistance(const Eigen::Vector3d& p) const {
    if (cfg_.floor_only) {
      return std::abs(p.z() - cfg_.room_min.z());
    }
    double best = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
      best = std::min(best, std::abs(p[a] - cfg_.room_min[a]));
      best = std::min(best, std::abs(p[a] - cfg_.room_max[a]));
    }
    return best;
  }

  // ---- measurement stream ----

  // One sweep: azimuth columns sample the trajectory at their own time offsets, so the
  // raw points carry real motion skew. Consumes range noise in generation order.
  LidarScan scanAt(Timestamp t_begin) {
    auto cloud = std::make_shared<PointCloud>();
    for (int j = 0; j < cfg_.n_azimuth; ++j) {
      const Duration offset = lidar_period_ns_ * j / cfg_.n_azimuth;
      const Pose T = poseAt(t_begin + offset);
      const double azim = 2.0 * std::numbers::pi * j / cfg_.n_azimuth;
      for (int k = 0; k < cfg_.n_rings; ++k) {
        const double elev_deg = cfg_.fov_down_deg + (cfg_.fov_up_deg - cfg_.fov_down_deg) * k /
                                                        std::max(cfg_.n_rings - 1, 1);
        const double elev = elev_deg * std::numbers::pi / 180.0;
        const Eigen::Vector3d d_sensor(std::cos(elev) * std::cos(azim),
                                       std::cos(elev) * std::sin(azim), std::sin(elev));
        const double range = castRay(T.t, T.q * d_sensor);
        if (!(range > 0.0) || range > cfg_.max_range_m) {
          continue;
        }
        const double noisy = cfg_.range_noise_std > 0.0 ? range + range_noise_(rng_) : range;
        LidarPoint pt;
        pt.xyz = (noisy * d_sensor).cast<float>();
        pt.t_offset_ns = static_cast<std::int32_t>(offset);
        pt.ring = static_cast<std::uint16_t>(k);
        pt.range = static_cast<float>(noisy);
        cloud->push_back(pt);
      }
    }
    LidarScan scan;
    scan.stamp_start = t_begin;
    scan.sweep_duration = lidar_period_ns_;
    scan.sensor_frame = Frame::OsSensor0;
    scan.points = std::move(cloud);
    return scan;
  }

  PreprocessedGroup groupAt(int k) {
    PreprocessedGroup pg;
    MeasureGroup& g = pg.group;
    g.t_begin = groupBegin(k);
    g.t_end = groupEnd(k);
    g.scan = scanAt(g.t_begin);
    // IMU on a fixed global lattice; both group bounds land on it, so the boundary
    // sample re-delivers across consecutive groups (the straddle the tracker dedups).
    const std::int64_t i0 = (g.t_begin - kWorldEpoch) / imu_period_ns_;
    const std::int64_t i1 = (g.t_end - kWorldEpoch) / imu_period_ns_;
    for (std::int64_t i = i0; i <= i1; ++i) {
      g.imu.push_back(imuAt(kWorldEpoch + i * imu_period_ns_));
    }
    if (cfg_.attach_images) {
      CameraFrame f;
      f.stamp = g.t_begin + lidar_period_ns_ / 2;
      f.width = 4;
      f.height = 4;
      f.encoding = CameraFrame::Encoding::Mono8;
      f.data = image_data_;
      g.image = f;
    }
    return pg;
  }

  std::vector<PreprocessedGroup> groups(int count) {
    std::vector<PreprocessedGroup> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
      out.push_back(groupAt(k));
    }
    return out;
  }

private:
  struct Kin {
    double theta = 0.0;      // circuit angle [rad]
    double sigma = 0.0;      // its rate [rad/s]
    double sigma_dot = 0.0;  // its acceleration [rad/s^2]
  };

  double secondsOf(Timestamp stamp) const { return to_seconds(stamp - kWorldEpoch); }

  // Hold, then a linear angular-speed ramp to the steady rate: theta is C1, so the
  // velocity is continuous and the IMU sees only finite accelerations.
  Kin kinAt(double t) const {
    Kin k;
    const double tau = t - cfg_.hold_s;
    if (cfg_.radius_m <= 0.0 || tau <= 0.0) {
      return k;
    }
    if (tau < cfg_.ramp_s) {
      k.sigma_dot = cfg_.rate_rad_s / cfg_.ramp_s;
      k.sigma = k.sigma_dot * tau;
      k.theta = 0.5 * k.sigma_dot * tau * tau;
      return k;
    }
    k.sigma = cfg_.rate_rad_s;
    k.theta = cfg_.rate_rad_s * (tau - 0.5 * cfg_.ramp_s);
    return k;
  }

  // First positive intersection with the room planes seen from inside (infinity when
  // the ray escapes, e.g. upward beams in the floor-only scene).
  double castRay(const Eigen::Vector3d& origin, const Eigen::Vector3d& dir) const {
    constexpr double kEps = 1e-9;
    double best = std::numeric_limits<double>::infinity();
    if (cfg_.floor_only) {
      if (dir.z() < -kEps) {
        const double t = (cfg_.room_min.z() - origin.z()) / dir.z();
        if (t > 0.0) {
          best = t;
        }
      }
      return best;
    }
    for (int a = 0; a < 3; ++a) {
      double t = std::numeric_limits<double>::infinity();
      if (dir[a] > kEps) {
        t = (cfg_.room_max[a] - origin[a]) / dir[a];
      } else if (dir[a] < -kEps) {
        t = (cfg_.room_min[a] - origin[a]) / dir[a];
      }
      if (t > 0.0 && t < best) {
        best = t;
      }
    }
    return best;
  }

  WorldConfig cfg_;
  Duration lidar_period_ns_;
  Duration imu_period_ns_;
  std::mt19937 rng_;
  std::normal_distribution<double> range_noise_;
  std::shared_ptr<const std::vector<std::uint8_t>> image_data_;
};

}  // namespace meridian::lio::test
