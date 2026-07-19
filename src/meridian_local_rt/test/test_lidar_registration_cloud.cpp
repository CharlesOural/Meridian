#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/lidar_registration_cloud.hpp"

namespace meridian::local {
namespace {

static_assert(!std::is_aggregate_v<LidarRegistrationCloud>);
static_assert(!std::is_default_constructible_v<LidarRegistrationCloud>);
static_assert(!std::is_copy_constructible_v<LidarRegistrationCloud>);
static_assert(!std::is_move_constructible_v<LidarRegistrationCloud>);

[[nodiscard]] LidarRegistrationCloudData registrationCloudData(
    const std::vector<Eigen::Vector3d>& positions, const std::vector<std::size_t>& insertion_order,
    core::MeasurementId sweep = core::MeasurementId{80U},
    core::FusionTime time = core::FusionTime{800U}, const core::Pose3d& seed = core::Pose3d{}) {
  if (positions.size() != insertion_order.size()) {
    throw std::logic_error("test insertion order must cover every point");
  }
  LidarRegistrationCloudData data;
  data.source_sweep = sweep;
  data.reference_time = time;
  data.T_odom_imu_seed = seed;
  data.layout = core::LidarLayout{static_cast<std::uint32_t>(positions.size()), 1U, false};
  data.points_in_reference_imu = std::make_unique<core::LidarPoints>();
  data.points_in_reference_imu->reserve(positions.size());
  data.points.reserve(positions.size());
  for (const std::size_t index : insertion_order) {
    if (index >= positions.size()) {
      throw std::logic_error("test insertion order contains an invalid index");
    }
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
  data.lineage = test::lidarLineage(sweep);
  return data;
}

[[nodiscard]] DeskewedSweep spatialSweep(bool reverse_input) {
  auto points = std::make_unique<core::LidarPoints>();
  constexpr std::size_t kRegionCount = 4U;
  constexpr std::size_t kSide = 6U;
  points->reserve(kRegionCount * kSide * kSide);
  std::uint32_t source_index = 0U;
  for (std::size_t region = 0U; region < kRegionCount; ++region) {
    for (std::size_t row = 0U; row < kSide; ++row) {
      for (std::size_t column = 0U; column < kSide; ++column) {
        core::LidarPoint point;
        point.x = 5.0F + 10.0F * static_cast<float>(region) + 0.31F * static_cast<float>(column) +
                  0.019F * static_cast<float>((row + column) % 2U);
        point.y = -0.9F + 0.31F * static_cast<float>(row);
        point.z = 0.07F * static_cast<float>((2U * row + column) % 4U);
        point.intensity = static_cast<float>(source_index) + 0.5F;
        point.time_offset_ns = static_cast<std::int32_t>(source_index);
        point.ring = static_cast<std::uint16_t>(row);
        point.source_index = source_index++;
        points->push_back(point);
      }
    }
  }
  if (reverse_input) {
    std::reverse(points->begin(), points->end());
  }
  DeskewedSweep sweep;
  sweep.source = core::MeasurementId{40U};
  sweep.reference_time = core::FusionTime{4'000U};
  sweep.T_odom_imu_reference = core::Pose3d{Sophus::SO3d::exp(Eigen::Vector3d{0.01, -0.02, 0.03}),
                                            Eigen::Vector3d{1.0, 2.0, 3.0}};
  sweep.layout = core::LidarLayout{source_index, 1U, false};
  sweep.points_in_reference_imu = std::move(points);
  sweep.imu_support = reverse_input ? std::vector{core::MeasurementId{3U}, core::MeasurementId{9U}}
                                    : std::vector{core::MeasurementId{9U}, core::MeasurementId{3U},
                                                  core::MeasurementId{9U}};
  return sweep;
}

void expectCloudRowsBitIdentical(const LidarRegistrationCloud& expected,
                                 const LidarRegistrationCloud& actual) {
  EXPECT_EQ(actual.source_sweep, expected.source_sweep);
  EXPECT_EQ(actual.reference_time, expected.reference_time);
  EXPECT_TRUE(
      (actual.T_odom_imu_seed.matrix().array() == expected.T_odom_imu_seed.matrix().array()).all());
  EXPECT_EQ(actual.layout.width, expected.layout.width);
  EXPECT_EQ(actual.layout.height, expected.layout.height);
  EXPECT_EQ(actual.layout.organized, expected.layout.organized);
  EXPECT_EQ(actual.stats.input_points, expected.stats.input_points);
  EXPECT_EQ(actual.stats.valid_range_points, expected.stats.valid_range_points);
  EXPECT_EQ(actual.stats.deterministic_voxel_points, expected.stats.deterministic_voxel_points);
  EXPECT_EQ(actual.imu_support, expected.imu_support);
  EXPECT_EQ(actual.checksum, expected.checksum);
  EXPECT_DOUBLE_EQ(actual.exactIndexVoxelResolutionM(), expected.exactIndexVoxelResolutionM());
  EXPECT_EQ(actual.exactIndexVoxelCount(), expected.exactIndexVoxelCount());

  ASSERT_EQ(actual.points.size(), expected.points.size());
  for (std::size_t index = 0U; index < expected.points.size(); ++index) {
    const LidarRegistrationPoint& lhs = expected.points[index];
    const LidarRegistrationPoint& rhs = actual.points[index];
    EXPECT_EQ(rhs.source_index, lhs.source_index);
    EXPECT_EQ(rhs.intensity, lhs.intensity);
    EXPECT_EQ(rhs.ring, lhs.ring);
    EXPECT_TRUE((rhs.point.array() == lhs.point.array()).all());
  }
}

[[nodiscard]] ExactLidarNeighbor bruteForceNearest(const LidarRegistrationCloud& cloud,
                                                   const Eigen::Vector3d& query,
                                                   double maximum_distance_m) {
  ExactLidarNeighbor best;
  best.distance_squared_m2 = std::numeric_limits<double>::infinity();
  if (!query.allFinite() || !std::isfinite(maximum_distance_m) || maximum_distance_m <= 0.0) {
    return best;
  }
  const double maximum_squared = maximum_distance_m * maximum_distance_m;
  for (std::size_t index = 0U; index < cloud.points.size(); ++index) {
    const LidarRegistrationPoint& point = cloud.points[index];
    const double distance_squared = (point.point - query).squaredNorm();
    if (distance_squared > maximum_squared) {
      continue;
    }
    if (!best.found || distance_squared < best.distance_squared_m2 ||
        (distance_squared == best.distance_squared_m2 &&
         std::tie(point.source_index, index) <
             std::tie(best.source_index, best.point_storage_index))) {
      best.found = true;
      best.point_storage_index = index;
      best.source_index = point.source_index;
      best.distance_squared_m2 = distance_squared;
    }
  }
  return best;
}

void expectSameNeighbor(const ExactLidarNeighbor& expected, const ExactLidarNeighbor& actual,
                        bool compare_work = false) {
  EXPECT_EQ(actual.found, expected.found);
  EXPECT_EQ(actual.point_storage_index, expected.point_storage_index);
  EXPECT_EQ(actual.source_index, expected.source_index);
  EXPECT_EQ(actual.distance_squared_m2, expected.distance_squared_m2);
  if (compare_work) {
    EXPECT_EQ(actual.voxel_lookups, expected.voxel_lookups);
    EXPECT_EQ(actual.points_examined, expected.points_examined);
  }
}

TEST(LidarRegistrationCloud, VoxelCapChecksumsAndRowsAreInputAndWorkerDeterministic) {
  LidarPreprocessConfig single_config;
  single_config.parallel_worker_count = 1U;
  single_config.minimum_range_m = 1.0;
  single_config.maximum_range_m = 80.0;
  single_config.voxel_size_m = 0.20;
  single_config.maximum_output_points = 24U;
  const std::array imu_support{core::MeasurementId{3U}, core::MeasurementId{9U}};

  auto single = buildLidarRegistrationCloud(
      spatialSweep(false), test::lidarLineage(core::MeasurementId{40U}, 400U, imu_support),
      single_config, LidarRegistrationIndexConfig{0.5});
  ASSERT_TRUE(single) << single.error().detail;
  ASSERT_EQ(single.value()->points.size(), single_config.maximum_output_points);
  EXPECT_EQ(single.value()->stats.input_points, 144U);
  EXPECT_GT(single.value()->stats.valid_range_points, single.value()->points.size());
  EXPECT_TRUE(core::contentHashPresent(single.value()->checksum));
  EXPECT_EQ(single.value()->imu_support,
            (std::vector{core::MeasurementId{3U}, core::MeasurementId{9U}}));

  LidarPreprocessConfig parallel_config = single_config;
  parallel_config.parallel_worker_count = 4U;
  auto reordered = buildLidarRegistrationCloud(
      spatialSweep(true), test::lidarLineage(core::MeasurementId{40U}, 400U, imu_support),
      parallel_config, LidarRegistrationIndexConfig{0.5});
  ASSERT_TRUE(reordered) << reordered.error().detail;
  expectCloudRowsBitIdentical(*single.value(), *reordered.value());

  auto different_index = buildLidarRegistrationCloud(
      spatialSweep(false), test::lidarLineage(core::MeasurementId{40U}, 400U, imu_support),
      single_config, LidarRegistrationIndexConfig{0.25});
  ASSERT_TRUE(different_index) << different_index.error().detail;
  EXPECT_NE(different_index.value()->checksum, single.value()->checksum);
}

TEST(LidarRegistrationCloud, EqualCenterDistanceWinnerUsesCanonicalSourceIdentity) {
  const auto make_sweep = [](bool reverse_storage) {
    std::array<core::LidarPoint, 2U> rows;
    rows[0].x = 5.75F;
    rows[0].y = 0.5F;
    rows[0].z = 0.5F;
    rows[0].intensity = 10.0F;
    rows[0].source_index = 0U;
    rows[1].x = 5.25F;
    rows[1].y = 0.5F;
    rows[1].z = 0.5F;
    rows[1].intensity = 20.0F;
    rows[1].source_index = 1U;
    if (reverse_storage) {
      std::reverse(rows.begin(), rows.end());
    }
    DeskewedSweep sweep;
    sweep.source = core::MeasurementId{41U};
    sweep.reference_time = core::FusionTime{4'100U};
    sweep.layout = core::LidarLayout{2U, 1U, false};
    sweep.points_in_reference_imu = std::make_unique<core::LidarPoints>(rows.begin(), rows.end());
    return sweep;
  };

  LidarPreprocessConfig config;
  config.minimum_range_m = 1.0;
  config.maximum_range_m = 80.0;
  config.voxel_size_m = 1.0;
  config.maximum_output_points = 2U;
  auto canonical =
      buildLidarRegistrationCloud(make_sweep(false), test::lidarLineage(core::MeasurementId{41U}),
                                  config, LidarRegistrationIndexConfig{1.0});
  auto reversed =
      buildLidarRegistrationCloud(make_sweep(true), test::lidarLineage(core::MeasurementId{41U}),
                                  config, LidarRegistrationIndexConfig{1.0});
  ASSERT_TRUE(canonical) << canonical.error().detail;
  ASSERT_TRUE(reversed) << reversed.error().detail;
  ASSERT_EQ(canonical.value()->points.size(), 1U);
  EXPECT_EQ(canonical.value()->points.front().source_index, 0U);
  expectCloudRowsBitIdentical(*canonical.value(), *reversed.value());
}

TEST(LidarRegistrationCloud, ExactIndexMatchesBruteForceTiesGeneralRadiusAndConcurrentReads) {
  std::vector<Eigen::Vector3d> positions{{-0.10, 0.0, 0.0}, {0.10, 0.0, 0.0}};
  for (int row = -5; row <= 5; ++row) {
    for (int column = -5; column <= 5; ++column) {
      positions.emplace_back(
          0.37 * static_cast<double>(column) + 0.013 * static_cast<double>((row + column + 11) % 3),
          0.33 * static_cast<double>(row) -
              0.017 * static_cast<double>((2 * row - column + 22) % 5),
          0.09 * static_cast<double>((row - column + 11) % 4));
    }
  }
  const auto cloud = test::sealedLidarRegistrationCloud(
      positions, core::MeasurementId{81U}, core::FusionTime{810U}, core::Pose3d{}, 0.25);

  const ExactLidarNeighbor tie = cloud->nearestExact(Eigen::Vector3d::Zero(), 0.20);
  ASSERT_TRUE(tie.found);
  EXPECT_EQ(tie.source_index, 0U);

  std::mt19937_64 generator(0x4d6572696469616eULL);
  std::uniform_real_distribution<double> xy(-2.25, 2.25);
  std::uniform_real_distribution<double> z(-0.25, 0.50);
  std::vector<Eigen::Vector3d> queries;
  queries.reserve(64U);
  for (std::size_t index = 0U; index < 64U; ++index) {
    queries.emplace_back(xy(generator), xy(generator), z(generator));
  }

  constexpr double kMaximumDistanceM = 0.80;
  std::vector<ExactLidarNeighbor> serial_results;
  serial_results.reserve(queries.size());
  for (const Eigen::Vector3d& query : queries) {
    const ExactLidarNeighbor indexed = cloud->nearestExact(query, kMaximumDistanceM);
    const ExactLidarNeighbor brute_force = bruteForceNearest(*cloud, query, kMaximumDistanceM);
    expectSameNeighbor(brute_force, indexed);
    EXPECT_GT(indexed.voxel_lookups, 27U);
    serial_results.push_back(indexed);
  }

  constexpr std::size_t kThreadCount = 4U;
  constexpr std::size_t kRepeats = 8U;
  std::vector<ExactLidarNeighbor> concurrent_results(kThreadCount * kRepeats * queries.size());
  std::vector<std::thread> readers;
  readers.reserve(kThreadCount);
  for (std::size_t thread = 0U; thread < kThreadCount; ++thread) {
    readers.emplace_back([&, thread] {
      for (std::size_t repeat = 0U; repeat < kRepeats; ++repeat) {
        for (std::size_t query = 0U; query < queries.size(); ++query) {
          const std::size_t output = (thread * kRepeats + repeat) * queries.size() + query;
          concurrent_results[output] = cloud->nearestExact(queries[query], kMaximumDistanceM);
        }
      }
    });
  }
  for (std::thread& reader : readers) {
    reader.join();
  }
  for (std::size_t index = 0U; index < concurrent_results.size(); ++index) {
    expectSameNeighbor(serial_results[index % queries.size()], concurrent_results[index], true);
  }

  const ExactLidarNeighbor invalid =
      cloud->nearestExact(Eigen::Vector3d::Zero(), std::numeric_limits<double>::max());
  EXPECT_FALSE(invalid.found);
  EXPECT_EQ(invalid.voxel_lookups, 0U);
  EXPECT_EQ(invalid.points_examined, 0U);
}

TEST(LidarRegistrationCloud, FactoryCanonicalizesRowsAndRejectsMalformedLineageAndRows) {
  const std::vector<Eigen::Vector3d> positions{
      {-0.10, 0.0, 0.0}, {0.10, 0.0, 0.0}, {0.60, 0.1, -0.2}};
  std::vector<std::size_t> canonical_order(positions.size());
  std::iota(canonical_order.begin(), canonical_order.end(), 0U);
  std::vector<std::size_t> reverse_order = canonical_order;
  std::reverse(reverse_order.begin(), reverse_order.end());

  auto canonical = LidarRegistrationCloud::create(registrationCloudData(positions, canonical_order),
                                                  LidarRegistrationIndexConfig{0.25});
  auto reordered = LidarRegistrationCloud::create(registrationCloudData(positions, reverse_order),
                                                  LidarRegistrationIndexConfig{0.25});
  ASSERT_TRUE(canonical) << canonical.error().detail;
  ASSERT_TRUE(reordered) << reordered.error().detail;
  expectCloudRowsBitIdentical(*canonical.value(), *reordered.value());

  LidarRegistrationCloudData mismatched = registrationCloudData(positions, canonical_order);
  mismatched.points.front().point.x() =
      std::nextafter(mismatched.points.front().point.x(), std::numeric_limits<double>::infinity());
  auto mismatched_result = LidarRegistrationCloud::create(std::move(mismatched));
  ASSERT_FALSE(mismatched_result);
  EXPECT_EQ(mismatched_result.error().code, LidarPreprocessErrorCode::NoUsablePoints);

  LidarRegistrationCloudData non_finite = registrationCloudData(positions, canonical_order);
  non_finite.points.front().point.z() = std::numeric_limits<double>::quiet_NaN();
  auto non_finite_result = LidarRegistrationCloud::create(std::move(non_finite));
  ASSERT_FALSE(non_finite_result);
  EXPECT_EQ(non_finite_result.error().code, LidarPreprocessErrorCode::NoUsablePoints);

  LidarRegistrationCloudData malformed_lineage = registrationCloudData(positions, canonical_order);
  malformed_lineage.lineage.usage.front().consumer = core::DerivedRecordId{};
  auto lineage_result = LidarRegistrationCloud::create(std::move(malformed_lineage));
  ASSERT_FALSE(lineage_result);
  EXPECT_EQ(lineage_result.error().code, LidarPreprocessErrorCode::InvalidLineage);

  LidarRegistrationCloudData missing_conditioning =
      registrationCloudData(positions, canonical_order);
  missing_conditioning.imu_support = {core::MeasurementId{9U}};
  auto conditioning_result = LidarRegistrationCloud::create(std::move(missing_conditioning));
  ASSERT_FALSE(conditioning_result);
  EXPECT_EQ(conditioning_result.error().code, LidarPreprocessErrorCode::InvalidLineage);

  LidarRegistrationCloudData duplicate_source = registrationCloudData(positions, canonical_order);
  duplicate_source.points_in_reference_imu->back().source_index =
      duplicate_source.points_in_reference_imu->front().source_index;
  auto duplicate_result = LidarRegistrationCloud::create(std::move(duplicate_source));
  ASSERT_FALSE(duplicate_result);
  EXPECT_EQ(duplicate_result.error().code, LidarPreprocessErrorCode::InvalidLayout);

  auto invalid_index = LidarRegistrationCloud::create(
      registrationCloudData(positions, canonical_order), LidarRegistrationIndexConfig{0.0});
  ASSERT_FALSE(invalid_index);
  EXPECT_EQ(invalid_index.error().code, LidarPreprocessErrorCode::InvalidConfig);
}

}  // namespace
}  // namespace meridian::local
