#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <vector>

#include "meridian/ros/sensor_conversions.hpp"

namespace meridian::ros {
namespace {

ObservationContext context() {
  ObservationContext result;
  result.header.trace = core::TraceId{1};
  result.header.producer = core::ProducerId{2};
  result.header.session = core::SessionId{3};
  result.header.config = core::ConfigRevision{4};
  result.header.direct_calibration = core::CalibrationEpoch{5};
  result.measurement = core::MeasurementId{6};
  result.clock.revision = core::ClockRevision{7};
  result.clock.source_epoch = core::SourceEpoch{8};
  result.clock.rate = 1.0;
  result.ingress_sequence = core::IngressSequence{9};
  return result;
}

template <typename Scalar>
void writeScalar(std::vector<std::uint8_t>* destination, std::size_t offset, Scalar value,
                 bool big_endian) {
  std::array<std::uint8_t, sizeof(Scalar)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(Scalar));
  constexpr bool host_big_endian = std::endian::native == std::endian::big;
  if (big_endian != host_big_endian && sizeof(Scalar) > 1U) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(destination->data() + offset, bytes.data(), bytes.size());
}

sensor_msgs::msg::PointCloud2 cloud(bool big_endian = false) {
  using sensor_msgs::msg::PointField;
  sensor_msgs::msg::PointCloud2 message;
  message.header.stamp.sec = 10;
  message.height = 1;
  message.width = 2;
  message.is_bigendian = big_endian;
  message.point_step = 20;
  message.row_step = 44;  // Includes four bytes of row padding.
  message.fields = {
      PointField{}.set__name("x").set__offset(0).set__datatype(PointField::FLOAT32).set__count(1),
      PointField{}.set__name("y").set__offset(4).set__datatype(PointField::FLOAT32).set__count(1),
      PointField{}.set__name("z").set__offset(8).set__datatype(PointField::FLOAT32).set__count(1),
      PointField{}.set__name("t").set__offset(12).set__datatype(PointField::UINT32).set__count(1),
      PointField{}
          .set__name("ring")
          .set__offset(16)
          .set__datatype(PointField::UINT16)
          .set__count(1),
      PointField{}
          .set__name("reflectivity")
          .set__offset(18)
          .set__datatype(PointField::UINT16)
          .set__count(1),
  };
  message.data.resize(message.row_step);
  writeScalar(&message.data, 0, 1.0F, big_endian);
  writeScalar(&message.data, 4, 2.0F, big_endian);
  writeScalar(&message.data, 8, 3.0F, big_endian);
  writeScalar(&message.data, 12, std::uint32_t{100}, big_endian);
  writeScalar(&message.data, 16, std::uint16_t{7}, big_endian);
  writeScalar(&message.data, 18, std::uint16_t{21}, big_endian);
  writeScalar(&message.data, 20, 4.0F, big_endian);
  writeScalar(&message.data, 24, 5.0F, big_endian);
  writeScalar(&message.data, 28, 6.0F, big_endian);
  writeScalar(&message.data, 32, std::uint32_t{1'000}, big_endian);
  writeScalar(&message.data, 36, std::uint16_t{8}, big_endian);
  writeScalar(&message.data, 38, std::uint16_t{22}, big_endian);
  return message;
}

sensor_msgs::msg::PointCloud2 organizedCloud() {
  sensor_msgs::msg::PointCloud2 message = cloud();
  message.height = 2U;
  message.data.resize(static_cast<std::size_t>(message.row_step) * message.height);

  const std::size_t second_row = message.row_step;
  writeScalar(&message.data, second_row + 0U, 7.0F, false);
  writeScalar(&message.data, second_row + 4U, 8.0F, false);
  writeScalar(&message.data, second_row + 8U, 9.0F, false);
  writeScalar(&message.data, second_row + 12U, std::uint32_t{2'000}, false);
  writeScalar(&message.data, second_row + 16U, std::uint16_t{9}, false);
  writeScalar(&message.data, second_row + 18U, std::uint16_t{23}, false);

  const std::size_t second_point = second_row + message.point_step;
  writeScalar(&message.data, second_point + 0U, 10.0F, false);
  writeScalar(&message.data, second_point + 4U, 11.0F, false);
  writeScalar(&message.data, second_point + 8U, 12.0F, false);
  writeScalar(&message.data, second_point + 12U, std::uint32_t{3'000}, false);
  writeScalar(&message.data, second_point + 16U, std::uint16_t{10}, false);
  writeScalar(&message.data, second_point + 18U, std::uint16_t{24}, false);
  return message;
}

TEST(ImuConversion, RejectsNonFiniteSamples) {
  sensor_msgs::msg::Imu message;
  message.linear_acceleration.x = std::numeric_limits<double>::quiet_NaN();
  const auto converted = convertImu(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::NonFiniteMeasurement);
}

TEST(LidarConversion, HonorsPaddingAndNormalizesPointTime) {
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};
  const auto converted = convertLidar(cloud(), conversion_context);
  ASSERT_TRUE(converted);
  ASSERT_EQ(converted.value().record.points->size(), 2U);
  EXPECT_EQ(converted.value().record.points->at(0).time_offset_ns, 0);
  EXPECT_EQ(converted.value().record.points->at(1).time_offset_ns, 900);
  EXPECT_EQ(converted.value().record.points->at(1).ring, 8U);
  EXPECT_FLOAT_EQ(converted.value().record.points->at(0).intensity, 21.0F);
  EXPECT_EQ(converted.value().record.acquisition.start.nanoseconds, 10'000'000'100LL);
  EXPECT_EQ(converted.value().record.acquisition.end.nanoseconds, 10'000'001'001LL);
}

TEST(LidarConversion, ReadsBigEndianFields) {
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};
  const auto converted = convertLidar(cloud(true), conversion_context);
  ASSERT_TRUE(converted);
  EXPECT_FLOAT_EQ(converted.value().record.points->at(1).x, 4.0F);
  EXPECT_EQ(converted.value().record.points->at(1).ring, 8U);
}

TEST(LidarConversion, PreservesCompleteOrganizedRasterIdentity) {
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};

  const auto converted = convertLidar(organizedCloud(), conversion_context);

  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().record.layout.width, 2U);
  EXPECT_EQ(converted.value().record.layout.height, 2U);
  EXPECT_TRUE(converted.value().record.layout.organized);
  ASSERT_EQ(converted.value().record.points->size(), 4U);
  for (std::size_t index = 0U; index < converted.value().record.points->size(); ++index) {
    EXPECT_EQ(converted.value().record.points->at(index).source_index, index);
  }
  EXPECT_EQ(converted.value().stats.input_elements, 4U);
  EXPECT_EQ(converted.value().stats.output_elements, 4U);
  EXPECT_EQ(converted.value().stats.discarded_non_finite, 0U);
}

TEST(LidarConversion, SparseFilteredRasterRetainsDimensionsAndOriginalSourceIndices) {
  auto message = organizedCloud();
  writeScalar(&message.data, 28U, std::numeric_limits<float>::quiet_NaN(), false);
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};

  const auto converted = convertLidar(message, conversion_context);

  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().record.layout.width, message.width);
  EXPECT_EQ(converted.value().record.layout.height, message.height);
  EXPECT_FALSE(converted.value().record.layout.organized);
  ASSERT_EQ(converted.value().record.points->size(), 3U);
  EXPECT_EQ(converted.value().record.points->at(0).source_index, 0U);
  EXPECT_EQ(converted.value().record.points->at(1).source_index, 2U);
  EXPECT_EQ(converted.value().record.points->at(2).source_index, 3U);
  EXPECT_EQ(converted.value().record.points->at(0).time_offset_ns, 0);
  EXPECT_EQ(converted.value().record.points->at(1).time_offset_ns, 1'900);
  EXPECT_EQ(converted.value().record.points->at(2).time_offset_ns, 2'900);
  EXPECT_EQ(converted.value().stats.input_elements, 4U);
  EXPECT_EQ(converted.value().stats.output_elements, 3U);
  EXPECT_EQ(converted.value().stats.discarded_non_finite, 1U);
}

TEST(LidarConversion, RequiresPerPointTiming) {
  auto message = cloud();
  message.fields.erase(message.fields.begin() + 3);
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};
  const auto converted = convertLidar(message, conversion_context);
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::MissingPerPointTime);
}

TEST(ImageConversion, PacksRowsAndCanonicalizesPositiveTimingOffset) {
  sensor_msgs::msg::Image message;
  message.header.stamp.sec = 12;
  message.header.stamp.nanosec = 400U;
  message.width = 2;
  message.height = 2;
  message.step = 4;
  message.encoding = sensor_msgs::image_encodings::MONO8;
  message.data = {1, 2, 99, 99, 3, 4, 99, 99};
  ObservationContext observation = context();
  observation.host_arrival_time = core::ArrivalTime{98'765};
  observation.device_sequence = core::SourceSequence{42U};
  const CameraConversionContext conversion_context{observation, core::CameraId{2},
                                                   core::Duration{50}, core::Duration{1'000}};
  const auto converted = convertImage(message, conversion_context);
  ASSERT_TRUE(converted);
  ASSERT_EQ(converted.value().pixels->size(), 4U);
  EXPECT_EQ(std::to_integer<unsigned>(converted.value().pixels->at(2)), 3U);
  EXPECT_EQ(converted.value().stamp.raw_time.nanoseconds, 12'000'000'400LL);
  EXPECT_EQ(converted.value().stamp.host_arrival_time.nanoseconds, 98'765);
  ASSERT_TRUE(converted.value().stamp.device_sequence);
  EXPECT_EQ(converted.value().stamp.device_sequence->value(), 42U);
  EXPECT_EQ(converted.value().stamp.clock_revision, observation.clock.revision);
  EXPECT_EQ(converted.value().stamp.source_epoch, observation.clock.source_epoch);
  EXPECT_EQ(converted.value().stamp.ingress_sequence, observation.ingress_sequence);
  EXPECT_EQ(converted.value().stamp.fusion_time.nanoseconds, 12'000'000'450LL);
  EXPECT_EQ(converted.value().exposure_midpoint.nanoseconds, 12'000'000'450LL);
  // Required by VisualLane: there is one canonical camera observation time.
  EXPECT_EQ(converted.value().stamp.fusion_time, converted.value().exposure_midpoint);
  EXPECT_EQ(converted.value().exposure, core::Duration{1'000});
}

TEST(ImageConversion, CanonicalizesNegativeTimingOffsetWithoutRewritingClockMetadata) {
  sensor_msgs::msg::Image message;
  message.header.stamp.sec = 12;
  message.header.stamp.nanosec = 400U;
  message.width = 2;
  message.height = 1;
  message.step = 2;
  message.encoding = sensor_msgs::image_encodings::MONO8;
  message.data = {7, 8};
  ObservationContext observation = context();
  observation.host_arrival_time = core::ArrivalTime{123'456};
  const CameraConversionContext conversion_context{observation, core::CameraId{2},
                                                   core::Duration{-750}, core::Duration{2'000}};

  const auto converted = convertImage(message, conversion_context);

  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().stamp.raw_time.nanoseconds, 12'000'000'400LL);
  EXPECT_EQ(converted.value().stamp.host_arrival_time.nanoseconds, 123'456);
  EXPECT_EQ(converted.value().stamp.fusion_time.nanoseconds, 11'999'999'650LL);
  EXPECT_EQ(converted.value().exposure_midpoint.nanoseconds, 11'999'999'650LL);
  EXPECT_EQ(converted.value().stamp.fusion_time, converted.value().exposure_midpoint);
  EXPECT_EQ(converted.value().exposure, core::Duration{2'000});
}

TEST(LidarConversion, RejectsTruncatedPayload) {
  auto message = cloud();
  message.data.resize(10);
  const LidarConversionContext conversion_context{context(), core::LidarId{1}, {}};
  const auto converted = convertLidar(message, conversion_context);
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::TruncatedData);
}

sensor_msgs::msg::NavSatFix validGnssFix() {
  sensor_msgs::msg::NavSatFix message;
  message.header.stamp.sec = 42;
  message.header.stamp.nanosec = 123U;
  message.status.status = sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX;
  message.status.service = sensor_msgs::msg::NavSatStatus::SERVICE_GPS;
  message.latitude = 48.8566;
  message.longitude = 2.3522;
  message.altitude = 37.5;
  message.position_covariance = {0.04, 0.01, 0.0, 0.01, 0.09, 0.0, 0.0, 0.0, 0.25};
  message.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN;
  return message;
}

TEST(GnssConversion, PreservesMappedTimeIdentityAndPsdEnuCovariance) {
  ObservationContext observation_context = context();
  observation_context.host_arrival_time = core::ArrivalTime{55};
  const auto converted = convertGnss(validGnssFix(), observation_context);

  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().id, core::GnssObservationId{observation_context.measurement.value()});
  EXPECT_EQ(converted.value().stamp.fusion_time, core::FusionTime{42'000'000'123LL});
  EXPECT_EQ(converted.value().stamp.host_arrival_time.nanoseconds,
            observation_context.host_arrival_time.nanoseconds);
  EXPECT_DOUBLE_EQ(converted.value().wgs84.latitudeDeg(), 48.8566);
  EXPECT_DOUBLE_EQ(converted.value().wgs84.longitudeDeg(), 2.3522);
  EXPECT_DOUBLE_EQ(converted.value().wgs84.altitudeM(), 37.5);
  EXPECT_TRUE(converted.value().covariance.matrix().isApprox(
      converted.value().covariance.matrix().transpose(), 0.0));
  EXPECT_EQ(converted.value().covariance.source(), core::PositionCovarianceSource::ReceiverFull);
  EXPECT_EQ(converted.value().solution, core::GnssSolutionType::GbasAugmented);
  EXPECT_EQ(converted.value().status.fix, core::GnssFixAvailability::Available);
  EXPECT_EQ(converted.value().status.integrity, core::GnssIntegrityStatus::Unknown);
  EXPECT_EQ(converted.value().status.corrections, core::GnssCorrectionStatus::Unknown);
  EXPECT_EQ(converted.value().status.source, core::GnssStatusSource::GenericNavSatFix);
  ASSERT_TRUE(converted.value().status.services);
  EXPECT_TRUE(converted.value().status.services->contains(core::GnssConstellationService::Gps));
  EXPECT_FALSE(converted.value().correction_age);
  EXPECT_FALSE(converted.value().hdop);
  EXPECT_FALSE(converted.value().vdop);
  EXPECT_FALSE(converted.value().satellites);
}

TEST(GnssConversion, MapsOnlyCoarseGenericFixClassesAndPreservesUnknownFields) {
  auto message = validGnssFix();
  message.status.service = 0U;

  message.status.status = sensor_msgs::msg::NavSatStatus::STATUS_FIX;
  auto converted = convertGnss(message, context());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().solution, core::GnssSolutionType::Autonomous);
  EXPECT_FALSE(converted.value().status.services);

  message.status.status = sensor_msgs::msg::NavSatStatus::STATUS_SBAS_FIX;
  converted = convertGnss(message, context());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().solution, core::GnssSolutionType::SbasAugmented);

  message.status.status = sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX;
  converted = convertGnss(message, context());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted.value().solution, core::GnssSolutionType::GbasAugmented);
  EXPECT_NE(converted.value().solution, core::GnssSolutionType::RtkFloat);
  EXPECT_NE(converted.value().solution, core::GnssSolutionType::RtkFixed);
  EXPECT_NE(converted.value().solution, core::GnssSolutionType::PppFloat);
  EXPECT_NE(converted.value().solution, core::GnssSolutionType::PppFixed);
}

TEST(GnssConversion, RejectsNoFixAndUnknownCovariance) {
  auto message = validGnssFix();
  message.status.status = sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX;
  auto converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::UnavailableMeasurement);

  message = validGnssFix();
  message.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN;
  converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::UnknownCovariance);
}

TEST(GnssConversion, RejectsUnknownWireEnumerationValues) {
  auto message = validGnssFix();
  message.status.status = 3;
  auto converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::UnsupportedFieldType);

  message = validGnssFix();
  message.position_covariance_type = 4U;
  converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::UnsupportedFieldType);
}

TEST(GnssConversion, RejectsIndefiniteCovarianceWithPositiveDiagonal) {
  auto message = validGnssFix();
  // Every diagonal entry is positive, but the leading 2x2 block has one
  // negative eigenvalue. A diagonal-only ingress check would accept it.
  message.position_covariance = {1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const auto converted = convertGnss(message, context());

  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::InvalidCovariance);
}

TEST(GnssConversion, RejectsNonFiniteGeodeticAndCovarianceValues) {
  auto message = validGnssFix();
  message.latitude = std::numeric_limits<double>::quiet_NaN();
  auto converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::NonFiniteMeasurement);

  message = validGnssFix();
  message.position_covariance[4] = std::numeric_limits<double>::infinity();
  converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::InvalidCovariance);
}

TEST(GnssConversion, RejectsAsymmetricAndMislabelledDiagonalCovariance) {
  auto message = validGnssFix();
  message.position_covariance[1] = 0.02;
  auto converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::InvalidCovariance);

  message = validGnssFix();
  message.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_DIAGONAL_KNOWN;
  converted = convertGnss(message, context());
  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::InvalidCovariance);
}

TEST(GnssConversion, RejectsUnknownConstellationServiceBits) {
  auto message = validGnssFix();
  message.status.service = 1U << 8U;

  const auto converted = convertGnss(message, context());

  ASSERT_FALSE(converted);
  EXPECT_EQ(converted.error().code, ConversionErrorCode::UnsupportedFieldType);
}

}  // namespace
}  // namespace meridian::ros
