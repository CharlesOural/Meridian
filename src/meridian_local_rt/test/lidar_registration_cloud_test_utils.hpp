#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <vector>

#include "meridian/local/lidar_registration_cloud.hpp"

namespace meridian::local::test {

[[nodiscard]] inline core::ObservationLineage lidarLineage(
    core::MeasurementId sweep, std::uint64_t lineage_id = 0U,
    std::span<const core::MeasurementId> imu_support = {}) {
  core::ObservationSlice source_slice;
  source_slice.root = sweep;
  source_slice.kind = core::SliceKind::Whole;
  source_slice.calibration = core::CalibrationEpoch{1U};
  source_slice.source_checksum.fill(
      static_cast<std::uint8_t>((sweep.value() % static_cast<std::uint64_t>(251U)) + 1U));

  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId{lineage_id == 0U ? sweep.value() : lineage_id};
  lineage.usage.push_back(
      core::ObservationUsage{source_slice, core::ObservationRole::DerivedSummary,
                             core::DerivedRecordId{sweep.value()}, std::nullopt, std::nullopt});

  const std::set<core::MeasurementId> canonical_support(imu_support.begin(), imu_support.end());
  for (const core::MeasurementId measurement : canonical_support) {
    core::ObservationSlice conditioning_slice;
    conditioning_slice.root = measurement;
    conditioning_slice.kind = core::SliceKind::Whole;
    conditioning_slice.calibration = core::CalibrationEpoch{1U};
    conditioning_slice.source_checksum.fill(
        static_cast<std::uint8_t>((measurement.value() % static_cast<std::uint64_t>(251U)) + 1U));
    lineage.usage.push_back(
        core::ObservationUsage{conditioning_slice, core::ObservationRole::ConditioningOnly,
                               core::DerivedRecordId{sweep.value()}, std::nullopt, std::nullopt});
  }
  return lineage;
}

[[nodiscard]] inline std::shared_ptr<const LidarRegistrationCloud> sealedLidarRegistrationCloud(
    const std::vector<Eigen::Vector3d>& positions, core::MeasurementId sweep, core::FusionTime time,
    const core::Pose3d& T_odom_imu_seed, double exact_index_resolution_m = 1.0,
    core::ObservationLineage lineage = {}, std::vector<core::MeasurementId> imu_support = {}) {
  LidarRegistrationCloudData data;
  data.source_sweep = sweep;
  data.reference_time = time;
  data.T_odom_imu_seed = T_odom_imu_seed;
  data.layout.width = static_cast<std::uint32_t>(positions.size());
  data.layout.height = 1U;
  data.layout.organized = false;
  data.points_in_reference_imu = std::make_unique<core::LidarPoints>();
  data.points_in_reference_imu->reserve(positions.size());
  data.points.reserve(positions.size());
  for (std::size_t index = 0U; index < positions.size(); ++index) {
    core::LidarPoint raw;
    raw.x = static_cast<float>(positions[index].x());
    raw.y = static_cast<float>(positions[index].y());
    raw.z = static_cast<float>(positions[index].z());
    raw.intensity = static_cast<float>(index) + 0.25F;
    raw.time_offset_ns = static_cast<std::int32_t>(index);
    raw.ring = static_cast<std::uint16_t>(index % 64U);
    raw.source_index = static_cast<std::uint32_t>(index);
    data.points_in_reference_imu->push_back(raw);

    data.points.push_back(LidarRegistrationPoint{
        Eigen::Vector3d{static_cast<double>(raw.x), static_cast<double>(raw.y),
                        static_cast<double>(raw.z)},
        raw.source_index, raw.intensity, raw.ring});
  }
  data.stats.input_points = positions.size();
  data.stats.valid_range_points = positions.size();
  data.stats.deterministic_voxel_points = positions.size();
  data.imu_support = std::move(imu_support);
  data.lineage =
      lineage.id.valid() ? std::move(lineage) : lidarLineage(sweep, 0U, data.imu_support);

  auto sealed = LidarRegistrationCloud::create(
      std::move(data), LidarRegistrationIndexConfig{exact_index_resolution_m});
  if (!sealed) {
    throw std::runtime_error("test LiDAR registration cloud sealing failed: " +
                             sealed.error().detail);
  }
  return std::move(sealed).value();
}

}  // namespace meridian::local::test
