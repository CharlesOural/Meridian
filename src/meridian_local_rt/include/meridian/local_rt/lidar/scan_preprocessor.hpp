#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "meridian/core/observations.hpp"
#include "meridian/local_rt/imu_types.hpp"
#include "meridian/local_rt/lidar/scan_to_map.hpp"

namespace meridian::local_rt::lidar {

struct ScanPreprocessorOptions final {
  double minimum_range_m{0.1};
  double maximum_range_m{100.0};
  double target_downsample_voxel_m{0.4};
  double source_downsample_voxel_m{1.0};
  std::int64_t lidar_time_offset_to_imu_ns{};
  core::Pose3d T_imu_lidar{};
};

struct ScanPreprocessorTiming final {
  std::int64_t range_filter_ns{};
  std::int64_t deskew_ns{};
  std::int64_t target_downsample_ns{};
  std::int64_t source_downsample_ns{};
  std::int64_t total_ns{};
};

struct ScanPreprocessorStats final {
  std::size_t input_points{};
  std::size_t range_rejected_points{};
  std::size_t support_rejected_points{};
  std::size_t deskewed_points{};
  std::size_t target_points{};
  std::size_t source_points{};
};

struct PreparedScan final {
  ScanFrame frame;
  ScanPreprocessorTiming timing;
  ScanPreprocessorStats stats;
};

// Converts a raw acquisition into two immutable sweep-end point sets: a denser
// owner-local target and a sparser registration source. Point range is gated in
// the physical acquisition frame before continuous-time deskew.
class ScanPreprocessor final {
public:
  explicit ScanPreprocessor(ScanPreprocessorOptions options);

  [[nodiscard]] PreparedScan prepare(const core::LidarSweep& sweep,
                                     const core::NavigationState& propagation_seed,
                                     core::StateId state_id,
                                     const DensePropagation& propagation) const;

  // Initialization already supplies sweep-end geometry (dynamic mode), or no
  // motion estimate is available yet (static mode's first target). This path
  // performs the same range/downsample stages without applying deskew again.
  [[nodiscard]] PreparedScan prepareSweepEndPoints(core::TimeNs time, core::StateId state_id,
                                                   std::span<const Point3d> points) const;

private:
  ScanPreprocessorOptions options_;
};

}  // namespace meridian::local_rt::lidar
