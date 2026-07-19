#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <limits>
#include <rclcpp/time.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "meridian/ros/newer_college_calibration.hpp"
#include "meridian/tools/bag_replay.hpp"

namespace meridian::tools {
namespace {

static_assert(!std::is_copy_constructible_v<DomainEvent>);
static_assert(!std::is_copy_assignable_v<DomainEvent>);
static_assert(std::is_move_constructible_v<DomainEvent>);
static_assert(!std::is_move_assignable_v<DomainEvent>);

class TemporaryRos2Bag {
public:
  TemporaryRos2Bag() {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto clock_value = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("meridian-gnss-replay-" + std::to_string(clock_value) + "-" +
             std::to_string(sequence.fetch_add(1U)));
  }

  TemporaryRos2Bag(const TemporaryRos2Bag&) = delete;
  TemporaryRos2Bag& operator=(const TemporaryRos2Bag&) = delete;

  ~TemporaryRos2Bag() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

[[nodiscard]] std::filesystem::path repositoryRoot() {
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
}

[[nodiscard]] core::ImuCalibration replayOnlyImuCalibration() {
  // Replay only needs the sensor identity and timing. Explicit caller-owned
  // values keep the calibration loader honest; no estimator consumes them in
  // this package.
  return core::ImuCalibration("alphasense_imu", "/alphasense_driver_ros/imu",
                              core::ImuSensorModel::BoschBmi085, 200.0, 9.80665,
                              core::ImuNoiseModel(0.0011002607647952406, 0.00022632861789099884,
                                                  3.390710627779767e-05, 8.252445860125436e-06));
}

template <typename Scalar>
void writeScalar(std::vector<std::uint8_t>* destination, std::size_t offset, Scalar value) {
  std::array<std::uint8_t, sizeof(Scalar)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(Scalar));
  if constexpr (std::endian::native == std::endian::big) {
    std::reverse(bytes.begin(), bytes.end());
  }
  std::memcpy(destination->data() + offset, bytes.data(), bytes.size());
}

[[nodiscard]] sensor_msgs::msg::PointCloud2 lidarCloudWithOneNonFinitePoint() {
  using sensor_msgs::msg::PointField;
  sensor_msgs::msg::PointCloud2 message;
  message.header.stamp.sec = 42;
  message.header.stamp.nanosec = 123U;
  message.height = 1U;
  message.width = 3U;
  message.is_bigendian = false;
  message.point_step = 20U;
  message.row_step = message.width * message.point_step;
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
  for (std::size_t point = 0U; point < message.width; ++point) {
    const std::size_t base = point * message.point_step;
    writeScalar(
        &message.data, base,
        point == 1U ? std::numeric_limits<float>::quiet_NaN() : static_cast<float>(point + 1U));
    writeScalar(&message.data, base + 4U, static_cast<float>(point + 2U));
    writeScalar(&message.data, base + 8U, static_cast<float>(point + 3U));
    writeScalar(&message.data, base + 12U, static_cast<std::uint32_t>(100U + 100U * point));
    writeScalar(&message.data, base + 16U, static_cast<std::uint16_t>(point));
    writeScalar(&message.data, base + 18U, static_cast<std::uint16_t>(20U + point));
  }
  return message;
}

TEST(ReplayProfile, RejectsDuplicateTopics) {
  core::AffineClockModel clock;
  clock.revision = core::ClockRevision{1U};
  clock.source_epoch = core::SourceEpoch{1U};
  clock.rate = 1.0;
  const ReplaySourceIdentity identity{"imu", core::ProducerId{1U}, clock};
  std::vector<ReplaySource> sources;
  sources.emplace_back(ImuReplaySource{"/imu", identity, true});
  sources.emplace_back(ImuReplaySource{"/imu", identity, true});

  auto profile =
      ReplayProfile::create("duplicate", core::SessionId{1U}, core::ConfigRevision{1U},
                            core::CalibrationEpoch{1U}, ReplayBounds{10U, 20U}, std::move(sources));

  ASSERT_FALSE(profile);
  EXPECT_EQ(profile.error().code, ReplayProfileErrorCode::DuplicateTopic);
}

TEST(Ros2BagReplay, GenericGnssIngressPreservesTypedSemantics) {
  constexpr char kTopic[] = "/gnss/fix";
  TemporaryRos2Bag bag;

  sensor_msgs::msg::NavSatFix fix;
  fix.header.stamp.sec = 42;
  fix.header.stamp.nanosec = 123U;
  fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_SBAS_FIX;
  // Zero is intentionally retained as unknown service metadata.
  fix.status.service = 0U;
  fix.latitude = 48.8566;
  fix.longitude = 2.3522;
  fix.altitude = 37.5;
  fix.position_covariance = {0.04, 0.01, 0.0, 0.01, 0.09, 0.0, 0.0, 0.0, 0.25};
  fix.position_covariance_type = sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_KNOWN;

  {
    rosbag2_cpp::Writer writer;
    writer.open(bag.path().string());
    auto no_fix = fix;
    no_fix.header.stamp.sec = 41;
    no_fix.status.status = sensor_msgs::msg::NavSatStatus::STATUS_NO_FIX;
    writer.write(no_fix, kTopic, rclcpp::Time{41'000'000'123LL});
    writer.write(fix, kTopic, rclcpp::Time{42'000'000'123LL});
    writer.close();
  }

  core::AffineClockModel clock;
  clock.revision = core::ClockRevision{1U};
  clock.source_epoch = core::SourceEpoch{2U};
  clock.rate = 1.0;
  const ReplaySourceIdentity identity{"generic_navsat", core::ProducerId{3U}, clock};
  std::vector<ReplaySource> sources;
  sources.emplace_back(GnssReplaySource{kTopic, identity, true});
  auto profile =
      ReplayProfile::create("gnss", core::SessionId{4U}, core::ConfigRevision{5U},
                            core::CalibrationEpoch{6U}, ReplayBounds{4U, 4U}, std::move(sources));
  ASSERT_TRUE(profile) << profile.error().detail;

  std::size_t emitted = 0U;
  const auto replayed = replayRos2Bag(bag.path(), profile.value(), [&](DomainEvent&& event) {
    const auto* observation = std::get_if<core::GnssObservation>(&event.observation());
    if (observation == nullptr) {
      return ReplayVisitResult::failure(
          ReplayVisitorError{"GNSS replay emitted the wrong domain observation type"});
    }
    ++emitted;
    EXPECT_DOUBLE_EQ(observation->wgs84.latitudeDeg(), fix.latitude);
    EXPECT_EQ(observation->solution, core::GnssSolutionType::SbasAugmented);
    EXPECT_EQ(observation->status.source, core::GnssStatusSource::GenericNavSatFix);
    EXPECT_EQ(observation->status.integrity, core::GnssIntegrityStatus::Unknown);
    EXPECT_EQ(observation->status.corrections, core::GnssCorrectionStatus::Unknown);
    EXPECT_FALSE(observation->status.services);
    EXPECT_FALSE(observation->correction_age);
    EXPECT_FALSE(observation->hdop);
    EXPECT_FALSE(observation->vdop);
    EXPECT_FALSE(observation->satellites);
    if (!event.metadata().conversion_accounting) {
      return ReplayVisitResult::failure(
          ReplayVisitorError{"GNSS identity conversion accounting is absent"});
    }
    const ReplayConversionAccounting& accounting = *event.metadata().conversion_accounting;
    EXPECT_EQ(accounting.kind, ReplayConversionElementKind::Record);
    EXPECT_EQ(accounting.input_elements, 1U);
    EXPECT_EQ(accounting.output_elements, 1U);
    EXPECT_EQ(accounting.discarded_non_finite, 0U);
    EXPECT_TRUE(accounting.valid());
    return ReplayVisitResult::success(ReplayVisitAction::Continue);
  });

  ASSERT_TRUE(replayed) << replayed.error().detail;
  EXPECT_EQ(replayed.value().completion, ReplayCompletion::EndOfBag);
  EXPECT_EQ(emitted, 1U);
  EXPECT_EQ(replayed.value().stats.bag_messages_read, 2U);
  EXPECT_EQ(replayed.value().stats.events_emitted, 1U);
  EXPECT_EQ(replayed.value().stats.conversion_input_elements, 1U);
  EXPECT_EQ(replayed.value().stats.conversion_output_elements, 1U);
  EXPECT_EQ(replayed.value().stats.conversion_discarded_elements, 0U);
  const auto topic_stats = replayed.value().stats.configured_topics.find(kTopic);
  ASSERT_NE(topic_stats, replayed.value().stats.configured_topics.end());
  EXPECT_EQ(topic_stats->second.messages_seen, 2U);
  EXPECT_EQ(topic_stats->second.conversion_errors, 1U);
  EXPECT_EQ(topic_stats->second.events_emitted, 1U);
  EXPECT_EQ(topic_stats->second.conversion_input_elements, 1U);
  EXPECT_EQ(topic_stats->second.conversion_output_elements, 1U);
  EXPECT_EQ(topic_stats->second.conversion_discarded_elements, 0U);
}

TEST(Ros2BagReplay, PreservesExactLidarPointAccountingWithoutSilentLoss) {
  constexpr char kTopic[] = "/lidar/points";
  TemporaryRos2Bag bag;
  const sensor_msgs::msg::PointCloud2 cloud = lidarCloudWithOneNonFinitePoint();
  {
    rosbag2_cpp::Writer writer;
    writer.open(bag.path().string());
    writer.write(cloud, kTopic, rclcpp::Time{42'000'000'123LL});
    writer.close();
  }

  core::AffineClockModel clock;
  clock.revision = core::ClockRevision{1U};
  clock.source_epoch = core::SourceEpoch{2U};
  clock.rate = 1.0;
  std::vector<ReplaySource> sources;
  sources.emplace_back(LidarReplaySource{kTopic,
                                         ReplaySourceIdentity{"lidar", core::ProducerId{3U}, clock},
                                         core::LidarId{4U},
                                         {},
                                         true});
  auto profile =
      ReplayProfile::create("lidar-accounting", core::SessionId{4U}, core::ConfigRevision{5U},
                            core::CalibrationEpoch{6U}, ReplayBounds{2U, 2U}, std::move(sources));
  ASSERT_TRUE(profile) << profile.error().detail;

  std::size_t visits = 0U;
  const auto replayed = replayRos2Bag(bag.path(), profile.value(), [&](DomainEvent&& event) {
    ++visits;
    const auto* sweep = std::get_if<core::LidarSweep>(&event.observation());
    if (sweep == nullptr || !sweep->points) {
      return ReplayVisitResult::failure(
          ReplayVisitorError{"LiDAR replay emitted no immutable point payload"});
    }
    EXPECT_EQ(sweep->points->size(), 2U);
    if (!event.metadata().conversion_accounting) {
      return ReplayVisitResult::failure(
          ReplayVisitorError{"LiDAR point conversion accounting is absent"});
    }
    const ReplayConversionAccounting& accounting = *event.metadata().conversion_accounting;
    EXPECT_EQ(accounting.kind, ReplayConversionElementKind::LidarPoint);
    EXPECT_EQ(accounting.input_elements, 3U);
    EXPECT_EQ(accounting.output_elements, 2U);
    EXPECT_EQ(accounting.discarded_non_finite, 1U);
    EXPECT_TRUE(accounting.valid());
    EXPECT_EQ(accounting.input_elements,
              accounting.output_elements + accounting.discarded_non_finite);
    return ReplayVisitResult::success(ReplayVisitAction::Continue);
  });

  ASSERT_TRUE(replayed) << replayed.error().detail;
  EXPECT_EQ(visits, 1U);
  EXPECT_EQ(replayed.value().stats.bag_messages_read, 1U);
  EXPECT_EQ(replayed.value().stats.configured_messages_seen, 1U);
  EXPECT_EQ(replayed.value().stats.events_emitted, 1U);
  EXPECT_EQ(replayed.value().stats.conversion_input_elements, 3U);
  EXPECT_EQ(replayed.value().stats.conversion_output_elements, 2U);
  EXPECT_EQ(replayed.value().stats.conversion_discarded_elements, 1U);
  const auto found = replayed.value().stats.configured_topics.find(kTopic);
  ASSERT_NE(found, replayed.value().stats.configured_topics.end());
  EXPECT_EQ(found->second.messages_seen, 1U);
  EXPECT_EQ(found->second.events_emitted, 1U);
  EXPECT_EQ(found->second.conversion_input_elements, 3U);
  EXPECT_EQ(found->second.conversion_output_elements, 2U);
  EXPECT_EQ(found->second.conversion_discarded_elements, 1U);
}

TEST(Ros2BagReplay, QuadEasyCappedSmoke) {
  const std::filesystem::path root = repositoryRoot();
  const std::filesystem::path bag = root / "bags/newer-college/quad-easy";
  ASSERT_TRUE(std::filesystem::exists(bag / "metadata.yaml"));

  auto calibration = ros::loadNewerCollegeCalibration(
      root / "bags/newer-college/calib", ros::NewerCollegeCollection::Collection1,
      core::CalibrationEpoch{1U}, replayOnlyImuCalibration());
  ASSERT_TRUE(calibration) << calibration.error().detail;

  NewerCollegeReplayOptions options;
  options.bounds = ReplayBounds{12U, 1'024U};
  options.session = core::SessionId{7U};
  options.config = core::ConfigRevision{3U};
  options.clock_revision = core::ClockRevision{2U};
  options.first_source_epoch = core::SourceEpoch{10U};
  options.first_producer = core::ProducerId{20U};
  auto profile = makeNewerCollegeRos2ReplayProfile(calibration.value(), options);
  ASSERT_TRUE(profile) << profile.error().detail;

  using EventSignature = std::tuple<std::string, std::string, std::int64_t, std::uint64_t,
                                    std::uint64_t, std::size_t, std::int64_t>;
  const auto collect = [&](std::vector<EventSignature>* signatures) {
    std::uint64_t previous_ingress = 0U;
    return replayRos2Bag(bag, profile.value(), [&](DomainEvent&& event) {
      EXPECT_TRUE(event.metadata().ingress_sequence.valid());
      EXPECT_TRUE(event.metadata().measurement_sequence.valid());
      EXPECT_EQ(event.metadata().recorded_timestamp.nanoseconds,
                event.metadata().recorded_arrival.nanoseconds);
      EXPECT_FALSE(event.metadata().topic.empty());
      EXPECT_FALSE(event.metadata().source_name.empty());
      if (!signatures->empty()) {
        EXPECT_GT(event.metadata().ingress_sequence.value(), previous_ingress);
      }
      previous_ingress = event.metadata().ingress_sequence.value();
      const std::int64_t fusion_time = std::visit(
          [](const auto& observation) {
            EXPECT_EQ(observation.header.created_at.nanoseconds,
                      observation.stamp.fusion_time.nanoseconds);
            return observation.stamp.fusion_time.nanoseconds;
          },
          event.observation());
      signatures->emplace_back(event.metadata().topic, event.metadata().source_name,
                               event.metadata().recorded_timestamp.nanoseconds,
                               event.metadata().ingress_sequence.value(),
                               event.metadata().measurement_sequence.value(),
                               event.observation().index(), fusion_time);
      return ReplayVisitResult::success(ReplayVisitAction::Continue);
    });
  };

  std::vector<EventSignature> first_signatures;
  auto result = collect(&first_signatures);

  ASSERT_TRUE(result) << result.error().detail;
  EXPECT_EQ(result.value().completion, ReplayCompletion::EventLimit);
  EXPECT_EQ(first_signatures.size(), 12U);
  EXPECT_EQ(result.value().stats.events_emitted, 12U);
  EXPECT_EQ(result.value().stats.configured_messages_seen, 12U);
  EXPECT_EQ(result.value().stats.bag_messages_read, result.value().stats.configured_messages_seen);
  EXPECT_TRUE(result.value().stats.unknown_topics.empty());
  for (const auto& [topic, stats] : result.value().stats.configured_topics) {
    EXPECT_FALSE(topic.empty());
    EXPECT_EQ(stats.deserialization_errors, 0U);
    EXPECT_EQ(stats.conversion_errors, 0U);
    EXPECT_EQ(stats.image_decode_errors, 0U);
    EXPECT_EQ(stats.clock_guard_errors, 0U);
  }

  std::vector<EventSignature> second_signatures;
  auto repeated = collect(&second_signatures);
  ASSERT_TRUE(repeated) << repeated.error().detail;
  EXPECT_EQ(repeated.value().completion, ReplayCompletion::EventLimit);
  EXPECT_EQ(second_signatures, first_signatures);
}

TEST(Ros2BagReplay, NewerCollegeFrontendIsolationRemovesExcludedTopicsBeforeDecode) {
  const std::filesystem::path root = repositoryRoot();
  auto calibration = ros::loadNewerCollegeCalibration(
      root / "bags/newer-college/calib", ros::NewerCollegeCollection::Collection1,
      core::CalibrationEpoch{1U}, replayOnlyImuCalibration());
  ASSERT_TRUE(calibration) << calibration.error().detail;

  NewerCollegeReplayOptions lidar_imu_options;
  lidar_imu_options.bounds = ReplayBounds{12U, 1'024U};
  lidar_imu_options.session = core::SessionId{7U};
  lidar_imu_options.config = core::ConfigRevision{3U};
  lidar_imu_options.clock_revision = core::ClockRevision{2U};
  lidar_imu_options.first_source_epoch = core::SourceEpoch{10U};
  lidar_imu_options.first_producer = core::ProducerId{20U};
  lidar_imu_options.include_cameras = false;
  auto lidar_imu = makeNewerCollegeRos2ReplayProfile(calibration.value(), lidar_imu_options);
  ASSERT_TRUE(lidar_imu) << lidar_imu.error().detail;
  ASSERT_EQ(lidar_imu.value().sources().size(), 2U);
  EXPECT_TRUE(std::holds_alternative<ImuReplaySource>(lidar_imu.value().sources()[0]));
  EXPECT_TRUE(std::holds_alternative<LidarReplaySource>(lidar_imu.value().sources()[1]));

  NewerCollegeReplayOptions imu_options = lidar_imu_options;
  imu_options.include_lidar = false;
  auto imu_only = makeNewerCollegeRos2ReplayProfile(calibration.value(), imu_options);
  ASSERT_TRUE(imu_only) << imu_only.error().detail;
  ASSERT_EQ(imu_only.value().sources().size(), 1U);
  EXPECT_TRUE(std::holds_alternative<ImuReplaySource>(imu_only.value().sources().front()));
}

}  // namespace
}  // namespace meridian::tools
