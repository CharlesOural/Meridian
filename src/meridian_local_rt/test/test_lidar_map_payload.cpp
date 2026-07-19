#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <utility>

#include "meridian/local/lidar_map_payload.hpp"

namespace meridian::local {
namespace {

[[nodiscard]] core::ContentHash presentHash(std::uint8_t value) {
  core::ContentHash hash{};
  hash.front() = value;
  return hash;
}

[[nodiscard]] core::LidarSweep sweep() {
  core::LidarSweep result;
  result.header.trace = core::TraceId{1U};
  result.header.producer = core::ProducerId{2U};
  result.header.session = core::SessionId{3U};
  result.header.created_at = core::FusionTime{240LL};
  result.header.config = core::ConfigRevision{4U};
  result.header.direct_calibration = core::CalibrationEpoch{5U};
  result.id = core::MeasurementId{31U};
  result.lidar = core::LidarId{8U};
  result.stamp.raw_time = core::RawDeviceTime{90LL};
  result.stamp.fusion_time = core::FusionTime{100LL};
  result.stamp.host_arrival_time = core::ArrivalTime{110LL};
  result.stamp.clock_revision = core::ClockRevision{1U};
  result.stamp.source_epoch = core::SourceEpoch{1U};
  result.stamp.device_sequence = core::SourceSequence{6U};
  result.stamp.ingress_sequence = core::IngressSequence{7U};
  result.stamp.uncertainty = core::Duration{1LL};
  result.acquisition = core::TimeRange{core::FusionTime{100LL}, core::FusionTime{200LL}};
  result.layout = core::LidarLayout{2U, 1U, true};
  auto points = std::make_shared<core::LidarPoints>();
  points->push_back(core::LidarPoint{1.0F, 2.0F, 3.0F, 4.0F, 0, 5U, 0U});
  points->push_back(core::LidarPoint{6.0F, 7.0F, 8.0F, 9.0F, 10, 6U, 1U});
  result.points = std::move(points);
  return result;
}

[[nodiscard]] core::ObservationLineage lineage(core::MeasurementId source) {
  core::ObservationLineage result;
  result.id = core::ObservationLineageId{30U};
  result.checksum = presentHash(1U);
  core::ObservationSlice slice;
  slice.root = source;
  slice.calibration = core::CalibrationEpoch{5U};
  result.usage.push_back(core::ObservationUsage{slice, core::ObservationRole::DerivedSummary,
                                                core::DerivedRecordId{32U}, std::nullopt,
                                                std::nullopt});
  return result;
}

[[nodiscard]] AcceptedLidarMapInputData data() {
  AcceptedLidarMapInputData result;
  result.sweep = sweep();
  result.odom_epoch = core::OdomEpoch{7U};
  result.state = core::StateId{10U};
  result.accepted_revision = core::LocalGraphRevision{9U};
  result.admitting_batch = core::FactorBatchId{6U};
  result.recovery_epoch = core::SensorRecoveryEpoch{1U};
  result.T_odom_imu = core::Pose3d{};
  result.calibration = core::CalibrationEpoch{5U};
  result.registration_cloud_checksum = presentHash(2U);
  result.localization_lineage = lineage(result.sweep.id);
  return result;
}

TEST(LidarMapPayload, AcceptedInputSharesRawSweepAndDownstreamSealIsDeterministic) {
  const AcceptedLidarMapInputData original = data();
  const core::LidarPoints* const raw_points = original.sweep.points.get();
  auto first_input = AcceptedLidarMapInput::create(original);
  auto second_input = AcceptedLidarMapInput::create(original);
  ASSERT_TRUE(first_input);
  ASSERT_TRUE(second_input);
  EXPECT_EQ(first_input.value()->sweep.points.get(), raw_points);
  auto first = LidarMapPayload::seal(first_input.value());
  auto second = LidarMapPayload::seal(second_input.value());
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first.value()->input, first_input.value());
  EXPECT_EQ(first.value()->raw_payload_checksum, second.value()->raw_payload_checksum);
  EXPECT_EQ(first.value()->checksum, second.value()->checksum);
  EXPECT_TRUE(core::contentHashPresent(first.value()->raw_payload_checksum));
  EXPECT_TRUE(core::contentHashPresent(first.value()->checksum));

  AcceptedLidarMapInputData relocated = original;
  relocated.T_odom_imu = core::Pose3d{Sophus::SO3d{}, Eigen::Vector3d{1.0, -2.0, 3.0}};
  auto relocated_input = AcceptedLidarMapInput::create(std::move(relocated));
  ASSERT_TRUE(relocated_input);
  auto accepted_elsewhere = LidarMapPayload::seal(relocated_input.value());
  ASSERT_TRUE(accepted_elsewhere);
  EXPECT_EQ(first.value()->raw_payload_checksum, accepted_elsewhere.value()->raw_payload_checksum);
  EXPECT_NE(first.value()->checksum, accepted_elsewhere.value()->checksum);
}

TEST(LidarMapPayload, RawPointMutationChangesRawAndEnvelopeChecksums) {
  const AcceptedLidarMapInputData original = data();
  auto first_input = AcceptedLidarMapInput::create(original);
  ASSERT_TRUE(first_input);
  auto first = LidarMapPayload::seal(first_input.value());
  ASSERT_TRUE(first);

  AcceptedLidarMapInputData modified = original;
  auto modified_points = std::make_shared<core::LidarPoints>(*modified.sweep.points);
  modified_points->back().intensity += 1.0F;
  modified.sweep.points = std::move(modified_points);
  auto second_input = AcceptedLidarMapInput::create(std::move(modified));
  ASSERT_TRUE(second_input);
  auto second = LidarMapPayload::seal(second_input.value());
  ASSERT_TRUE(second);
  EXPECT_NE(first.value()->raw_payload_checksum, second.value()->raw_payload_checksum);
  EXPECT_NE(first.value()->checksum, second.value()->checksum);
}

TEST(LidarMapPayload, DefersRawRowWorkUntilSealButRejectsWrongSourceLineageImmediately) {
  AcceptedLidarMapInputData noncanonical = data();
  auto rows = std::make_shared<core::LidarPoints>(*noncanonical.sweep.points);
  rows->back().source_index = rows->front().source_index;
  noncanonical.sweep.points = std::move(rows);
  auto accepted_rows = AcceptedLidarMapInput::create(std::move(noncanonical));
  ASSERT_TRUE(accepted_rows);
  auto rejected_rows = LidarMapPayload::seal(accepted_rows.value());
  ASSERT_FALSE(rejected_rows);
  EXPECT_EQ(rejected_rows.error().code, LidarMapPayloadErrorCode::InvalidSweep);

  AcceptedLidarMapInputData wrong_lineage = data();
  wrong_lineage.localization_lineage = lineage(core::MeasurementId{99U});
  auto rejected_lineage = AcceptedLidarMapInput::create(std::move(wrong_lineage));
  ASSERT_FALSE(rejected_lineage);
  EXPECT_EQ(rejected_lineage.error().code, LidarMapPayloadErrorCode::InvalidLineage);
}

}  // namespace
}  // namespace meridian::local
