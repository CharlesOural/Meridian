#include "meridian/tools/bag_replay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "meridian/ros/sensor_conversions.hpp"

namespace meridian::tools {
namespace {

template <class... Visitors>
struct Overloaded : Visitors... {
  using Visitors::operator()...;
};
template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

struct SourceView {
  const std::string* topic{};
  const ReplaySourceIdentity* identity{};
  bool required{};
};

[[nodiscard]] SourceView sourceView(const ReplaySource& source) {
  return std::visit(
      [](const auto& typed) {
        return SourceView{&typed.topic, &typed.source, typed.required};
      },
      source);
}

[[nodiscard]] bool sourceIdValid(const ReplaySource& source) {
  return std::visit(Overloaded{
                        [](const ImuReplaySource&) { return true; },
                        [](const LidarReplaySource& lidar) { return lidar.lidar.valid(); },
                        [](const CameraReplaySource& camera) {
                          return camera.camera.valid() && camera.exposure.nanoseconds >= 0;
                        },
                        [](const GnssReplaySource&) { return true; },
                    },
                    source);
}

[[nodiscard]] std::string_view expectedRosType(const ReplaySource& source) {
  return std::visit(
      Overloaded{
          [](const ImuReplaySource&) -> std::string_view { return "sensor_msgs/msg/Imu"; },
          [](const LidarReplaySource&) -> std::string_view {
            return "sensor_msgs/msg/PointCloud2";
          },
          [](const CameraReplaySource& camera) -> std::string_view {
            return camera.wire_format == CameraWireFormat::Image
                       ? "sensor_msgs/msg/Image"
                       : "sensor_msgs/msg/CompressedImage";
          },
          [](const GnssReplaySource&) -> std::string_view { return "sensor_msgs/msg/NavSatFix"; },
      },
      source);
}

[[nodiscard]] ReplayProfileError profileError(ReplayProfileErrorCode code, std::string detail) {
  return ReplayProfileError{code, std::move(detail)};
}

[[nodiscard]] ReplayError replayError(ReplayErrorCode code, std::string detail, ReplayStats stats) {
  return ReplayError{code, std::move(detail), std::move(stats)};
}

[[nodiscard]] std::string compressedTopic(std::string topic) {
  constexpr std::string_view suffix = "/compressed";
  if (topic.size() >= suffix.size() &&
      topic.compare(topic.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return topic;
  }
  return topic + std::string(suffix);
}

[[nodiscard]] bool nextSequence(std::uint64_t* sequence) noexcept {
  if (*sequence == core::MeasurementId::kInvalidValue - 1U) {
    return false;
  }
  ++(*sequence);
  return true;
}

template <typename Message>
[[nodiscard]] std::optional<Message> deserializeMessage(
    const rosbag2_storage::SerializedBagMessage& serialized, std::string* error_detail) {
  try {
    rclcpp::SerializedMessage bytes(*serialized.serialized_data);
    rclcpp::Serialization<Message> serializer;
    Message message;
    serializer.deserialize_message(&bytes, &message);
    return message;
  } catch (const std::exception& exception) {
    *error_detail = exception.what();
    return std::nullopt;
  }
}

[[nodiscard]] core::RecordHeader makeHeader(const ReplayProfile& profile,
                                            const ReplaySourceIdentity& source,
                                            core::MeasurementId measurement,
                                            std::int64_t recorded_timestamp) {
  core::RecordHeader header;
  header.trace = core::TraceId{measurement.value()};
  header.producer = source.producer;
  header.session = profile.session();
  header.created_at = core::FusionTime{recorded_timestamp};
  header.config = profile.config();
  header.direct_calibration = profile.calibration();
  return header;
}

[[nodiscard]] ros::ObservationContext makeObservationContext(const ReplayProfile& profile,
                                                             const ReplaySourceIdentity& source,
                                                             core::IngressSequence ingress,
                                                             core::MeasurementId measurement,
                                                             std::int64_t recorded_timestamp) {
  ros::ObservationContext context;
  context.header = makeHeader(profile, source, measurement, recorded_timestamp);
  context.clock = source.clock;
  context.host_arrival_time = core::ArrivalTime{recorded_timestamp};
  context.ingress_sequence = ingress;
  context.measurement = measurement;
  // ROS 2 sensor messages do not carry a device sequence. Do not relabel a
  // generated replay counter as one; ingress_sequence is the generated order.
  context.device_sequence = std::nullopt;
  return context;
}

template <typename Record>
void setCreatedAtToFusionTime(Record* record) {
  record->header.created_at = record->stamp.fusion_time;
}

struct ConvertedObservation {
  DomainObservation observation;
  std::optional<ReplayConversionAccounting> accounting;
};

using ConvertResult = core::Result<ConvertedObservation, ros::ConversionError>;

[[nodiscard]] ReplayConversionAccounting identityConversionAccounting() noexcept {
  return ReplayConversionAccounting{ReplayConversionElementKind::Record, 1U, 1U, 0U};
}

template <typename Message>
[[nodiscard]] ConvertResult convertMessage(const Message& message, const ReplaySource& source,
                                           const ros::ObservationContext& observation_context) {
  return std::visit(
      Overloaded{
          [&](const ImuReplaySource&) -> ConvertResult {
            if constexpr (!std::is_same_v<Message, sensor_msgs::msg::Imu>) {
              return ConvertResult::failure({ros::ConversionErrorCode::UnsupportedFieldType,
                                             "internal replay/source IMU type mismatch"});
            } else {
              auto converted = ros::convertImu(message, observation_context);
              if (!converted) {
                return ConvertResult::failure(converted.error());
              }
              core::ImuSample record = std::move(converted).value();
              setCreatedAtToFusionTime(&record);
              DomainObservation observation(std::move(record));
              return ConvertResult::success(
                  {std::move(observation), identityConversionAccounting()});
            }
          },
          [&](const LidarReplaySource& lidar) -> ConvertResult {
            if constexpr (!std::is_same_v<Message, sensor_msgs::msg::PointCloud2>) {
              return ConvertResult::failure({ros::ConversionErrorCode::UnsupportedFieldType,
                                             "internal replay/source lidar type mismatch"});
            } else {
              ros::LidarConversionContext context;
              context.observation = observation_context;
              context.lidar = lidar.lidar;
              context.header_to_point_time_origin = lidar.header_to_point_time_origin;
              auto converted = ros::convertLidar(message, context);
              if (!converted) {
                return ConvertResult::failure(converted.error());
              }
              auto converted_value = std::move(converted).value();
              const ReplayConversionAccounting accounting{
                  ReplayConversionElementKind::LidarPoint, converted_value.stats.input_elements,
                  converted_value.stats.output_elements,
                  converted_value.stats.discarded_non_finite};
              core::LidarSweep record = std::move(converted_value.record);
              setCreatedAtToFusionTime(&record);
              DomainObservation observation(std::move(record));
              return ConvertResult::success({std::move(observation), accounting});
            }
          },
          [&](const CameraReplaySource& camera) -> ConvertResult {
            ros::CameraConversionContext context;
            context.observation = observation_context;
            context.camera = camera.camera;
            context.stamp_to_exposure_midpoint = camera.stamp_to_exposure_midpoint;
            context.exposure = camera.exposure;
            ros::CameraResult converted = [&]() {
              if constexpr (std::is_same_v<Message, sensor_msgs::msg::Image>) {
                return ros::convertImage(message, context);
              } else if constexpr (std::is_same_v<Message, sensor_msgs::msg::CompressedImage>) {
                return ros::convertCompressedImage(message, context);
              } else {
                return ros::CameraResult::failure({ros::ConversionErrorCode::UnsupportedFieldType,
                                                   "internal replay/source camera type mismatch"});
              }
            }();
            if (!converted) {
              return ConvertResult::failure(converted.error());
            }
            core::CameraFrame record = std::move(converted).value();
            setCreatedAtToFusionTime(&record);
            DomainObservation observation(std::move(record));
            return ConvertResult::success({std::move(observation), identityConversionAccounting()});
          },
          [&](const GnssReplaySource&) -> ConvertResult {
            if constexpr (!std::is_same_v<Message, sensor_msgs::msg::NavSatFix>) {
              return ConvertResult::failure({ros::ConversionErrorCode::UnsupportedFieldType,
                                             "internal replay/source GNSS type mismatch"});
            } else {
              auto converted = ros::convertGnss(message, observation_context);
              if (!converted) {
                return ConvertResult::failure(converted.error());
              }
              core::GnssObservation record = std::move(converted).value();
              setCreatedAtToFusionTime(&record);
              DomainObservation observation(std::move(record));
              return ConvertResult::success(
                  {std::move(observation), identityConversionAccounting()});
            }
          },
      },
      source);
}

template <typename Message>
[[nodiscard]] std::optional<ConvertedObservation> deserializeAndConvert(
    const rosbag2_storage::SerializedBagMessage& serialized, const ReplaySource& source,
    const ros::ObservationContext& observation_context, TopicReplayStats* stats) {
  std::string detail;
  auto message = deserializeMessage<Message>(serialized, &detail);
  if (!message) {
    ++stats->deserialization_errors;
    return std::nullopt;
  }
  std::optional<ConvertResult> converted_result;
  try {
    converted_result.emplace(convertMessage(*message, source, observation_context));
  } catch (const std::exception&) {
    ++stats->conversion_errors;
    if constexpr (std::is_same_v<Message, sensor_msgs::msg::CompressedImage>) {
      ++stats->image_decode_errors;
    }
    return std::nullopt;
  }
  ConvertResult& converted = *converted_result;
  if (!converted) {
    ++stats->conversion_errors;
    if (converted.error().code == ros::ConversionErrorCode::ImageDecodeFailed) {
      ++stats->image_decode_errors;
    }
    return std::nullopt;
  }
  return std::move(converted).value();
}

[[nodiscard]] bool additionFits(std::size_t current, std::size_t increment) noexcept {
  return increment <= std::numeric_limits<std::size_t>::max() - current;
}

[[nodiscard]] bool conversionAccountingFits(const ReplayConversionAccounting& accounting,
                                            const TopicReplayStats& topic,
                                            const ReplayStats& total) noexcept {
  return accounting.valid() &&
         additionFits(topic.conversion_input_elements, accounting.input_elements) &&
         additionFits(topic.conversion_output_elements, accounting.output_elements) &&
         additionFits(topic.conversion_discarded_elements, accounting.discarded_non_finite) &&
         additionFits(total.conversion_input_elements, accounting.input_elements) &&
         additionFits(total.conversion_output_elements, accounting.output_elements) &&
         additionFits(total.conversion_discarded_elements, accounting.discarded_non_finite);
}

void accumulateConversionAccounting(const ReplayConversionAccounting& accounting,
                                    TopicReplayStats* topic, ReplayStats* total) noexcept {
  topic->conversion_input_elements += accounting.input_elements;
  topic->conversion_output_elements += accounting.output_elements;
  topic->conversion_discarded_elements += accounting.discarded_non_finite;
  total->conversion_input_elements += accounting.input_elements;
  total->conversion_output_elements += accounting.output_elements;
  total->conversion_discarded_elements += accounting.discarded_non_finite;
}

}  // namespace

ReplayProfile::ReplayProfile(std::string name, core::SessionId session, core::ConfigRevision config,
                             core::CalibrationEpoch calibration, ReplayBounds bounds,
                             std::vector<ReplaySource> sources)
    : name_(std::move(name)),
      session_(session),
      config_(config),
      calibration_(calibration),
      bounds_(bounds),
      sources_(std::move(sources)) {}

core::Result<ReplayProfile, ReplayProfileError> ReplayProfile::create(
    std::string name, core::SessionId session, core::ConfigRevision config,
    core::CalibrationEpoch calibration, ReplayBounds bounds, std::vector<ReplaySource> sources) {
  using ProfileResult = core::Result<ReplayProfile, ReplayProfileError>;
  if (name.empty() || !session.valid() || !config.valid() || !calibration.valid()) {
    return ProfileResult::failure(
        profileError(ReplayProfileErrorCode::InvalidIdentity,
                     "profile name, session, config, and calibration must be valid"));
  }
  if (bounds.max_events == 0U || bounds.max_bag_messages == 0U) {
    return ProfileResult::failure(
        profileError(ReplayProfileErrorCode::InvalidBounds,
                     "event and bag-message replay limits must both be non-zero"));
  }
  if (sources.empty()) {
    return ProfileResult::failure(
        profileError(ReplayProfileErrorCode::EmptySources,
                     "a replay profile requires at least one sensor source"));
  }

  std::unordered_map<std::string, bool> topics;
  for (const ReplaySource& source : sources) {
    const SourceView view = sourceView(source);
    if (view.topic->empty() || view.identity->name.empty() || !view.identity->producer.valid()) {
      return ProfileResult::failure(
          profileError(ReplayProfileErrorCode::InvalidIdentity,
                       "each replay source needs a topic, name, and producer"));
    }
    if (!view.identity->clock.valid()) {
      return ProfileResult::failure(
          profileError(ReplayProfileErrorCode::InvalidClock,
                       "source " + view.identity->name + " has an invalid affine clock"));
    }
    if (!sourceIdValid(source)) {
      return ProfileResult::failure(
          profileError(ReplayProfileErrorCode::InvalidId,
                       "source " + view.identity->name + " has an invalid sensor ID"));
    }
    if (!topics.emplace(*view.topic, true).second) {
      return ProfileResult::failure(
          profileError(ReplayProfileErrorCode::DuplicateTopic,
                       "topic " + *view.topic + " is configured more than once"));
    }
  }
  return ProfileResult::success(
      ReplayProfile(std::move(name), session, config, calibration, bounds, std::move(sources)));
}

core::Result<ReplayProfile, ReplayProfileError> makeNewerCollegeRos2ReplayProfile(
    const core::CalibrationBundle& calibration, const NewerCollegeReplayOptions& options) {
  const std::size_t source_count = 1U + (options.include_lidar ? 1U : 0U) +
                                   (options.include_cameras ? calibration.cameras().size() : 0U);
  const auto has_id_range = [source_count](std::uint64_t first, std::uint64_t invalid) {
    return first != invalid && source_count <= static_cast<std::size_t>(invalid - first);
  };
  if (!options.clock_revision.valid() ||
      !has_id_range(options.first_source_epoch.value(), core::SourceEpoch::kInvalidValue) ||
      !has_id_range(options.first_producer.value(), core::ProducerId::kInvalidValue)) {
    return core::Result<ReplayProfile, ReplayProfileError>::failure(
        profileError(ReplayProfileErrorCode::InvalidId,
                     "Newer College source identity range is invalid or "
                     "would overflow"));
  }
  std::vector<ReplaySource> sources;
  sources.reserve(source_count);
  std::uint64_t epoch = options.first_source_epoch.value();
  std::uint64_t producer = options.first_producer.value();

  const auto identity = [&](std::string name) {
    core::AffineClockModel clock;
    clock.revision = options.clock_revision;
    clock.source_epoch = core::SourceEpoch{epoch++};
    clock.raw_reference = core::RawDeviceTime{0};
    clock.fusion_reference = core::FusionTime{0};
    clock.rate = 1.0;
    clock.uncertainty = core::Duration{0};
    clock.status = core::TimeMappingStatus::Valid;
    return ReplaySourceIdentity{std::move(name), core::ProducerId{producer++}, clock};
  };

  sources.emplace_back(
      ImuReplaySource{calibration.imu().sourceTopic(), identity(calibration.imu().name()), true});
  if (options.include_lidar) {
    sources.emplace_back(LidarReplaySource{calibration.lidar().sourceTopic(),
                                           identity(calibration.lidar().name()),
                                           calibration.lidar().id(), core::Duration{0}, true});
  }
  if (options.include_cameras) {
    for (const core::CameraCalibration& camera : calibration.cameras()) {
      sources.emplace_back(CameraReplaySource{
          compressedTopic(camera.sourceTopic()), identity(camera.name()), camera.id(),
          CameraWireFormat::CompressedImage, camera.timing().imuTimeMinusCameraTime(),
          options.image_exposure, true});
    }
  }

  return ReplayProfile::create("newer-college-ros2", options.session, options.config,
                               calibration.epoch(), options.bounds, std::move(sources));
}

DomainEvent::DomainEvent(ReplayEventMetadata metadata, DomainObservation observation)
    : metadata_(std::move(metadata)), observation_(std::move(observation)) {}

std::size_t estimateDomainEventBytes(const DomainEvent& event) noexcept {
  const auto saturatingAdd = [](std::size_t lhs, std::size_t rhs) {
    return rhs > std::numeric_limits<std::size_t>::max() - lhs
               ? std::numeric_limits<std::size_t>::max()
               : lhs + rhs;
  };
  const auto saturatingMultiply = [](std::size_t lhs, std::size_t rhs) {
    return lhs != 0U && rhs > std::numeric_limits<std::size_t>::max() / lhs
               ? std::numeric_limits<std::size_t>::max()
               : lhs * rhs;
  };

  std::size_t bytes = sizeof(DomainEvent);
  // Conversion accounting is inline in ReplayEventMetadata and is therefore
  // already represented by sizeof(DomainEvent).
  bytes = saturatingAdd(bytes, event.metadata().topic.size());
  bytes = saturatingAdd(bytes, event.metadata().source_name.size());
  const std::size_t payload_bytes = std::visit(
      Overloaded{
          [](const core::ImuSample&) { return std::size_t{0U}; },
          [&](const core::LidarSweep& sweep) {
            return sweep.points ? saturatingMultiply(sweep.points->size(), sizeof(core::LidarPoint))
                                : std::size_t{0U};
          },
          [](const core::CameraFrame& frame) {
            return frame.pixels ? frame.pixels->size() : std::size_t{0U};
          },
          [](const core::GnssObservation&) { return std::size_t{0U}; },
      },
      event.observation());
  return saturatingAdd(bytes, payload_bytes);
}

ReplayResult replayRos2Bag(const std::filesystem::path& bag_uri, const ReplayProfile& profile,
                           DomainEventVisitor visitor) {
  ReplayStats stats;
  if (!visitor) {
    return ReplayResult::failure(replayError(ReplayErrorCode::InvalidVisitor,
                                             "domain-event visitor is empty", std::move(stats)));
  }

  rosbag2_cpp::Reader reader;
  try {
    rosbag2_storage::StorageOptions storage;
    storage.uri = bag_uri.string();
    storage.storage_id = "sqlite3";
    rosbag2_cpp::ConverterOptions converter;
    converter.input_serialization_format = "cdr";
    converter.output_serialization_format = "cdr";
    reader.open(storage, converter);
  } catch (const std::exception& exception) {
    return ReplayResult::failure(
        replayError(ReplayErrorCode::BagOpenFailed,
                    "failed to open ROS 2 bag " + bag_uri.string() + ": " + exception.what(),
                    std::move(stats)));
  }

  std::unordered_map<std::string, std::string> bag_topics;
  try {
    for (const auto& topic : reader.get_all_topics_and_types()) {
      bag_topics.emplace(topic.name, topic.type);
    }
  } catch (const std::exception& exception) {
    return ReplayResult::failure(
        replayError(ReplayErrorCode::BagReadFailed,
                    "failed to read ROS 2 bag topic metadata: " + std::string(exception.what()),
                    std::move(stats)));
  }

  std::unordered_map<std::string, std::size_t> source_by_topic;
  source_by_topic.reserve(profile.sources().size());
  for (std::size_t index = 0; index < profile.sources().size(); ++index) {
    const ReplaySource& source = profile.sources()[index];
    const SourceView view = sourceView(source);
    source_by_topic.emplace(*view.topic, index);
    const auto found = bag_topics.find(*view.topic);
    if (found == bag_topics.end()) {
      if (view.required) {
        return ReplayResult::failure(replayError(
            ReplayErrorCode::RequiredTopicMissing,
            "required topic " + *view.topic + " is absent from the bag", std::move(stats)));
      }
      continue;
    }
    if (found->second != expectedRosType(source)) {
      return ReplayResult::failure(replayError(ReplayErrorCode::TopicTypeMismatch,
                                               "topic " + *view.topic + " has type " +
                                                   found->second + ", expected " +
                                                   std::string(expectedRosType(source)),
                                               std::move(stats)));
    }
  }

  // Apply modality isolation before SQLite returns serialized payloads. This
  // is materially different from ignoring an already-read camera event: a
  // LiDAR--IMU benchmark must not pay image I/O or deserialization cost, just
  // as the deployed local node will not subscribe to camera topics in that
  // mode.
  rosbag2_storage::StorageFilter storage_filter;
  storage_filter.topics.reserve(profile.sources().size());
  for (const ReplaySource& source : profile.sources()) {
    storage_filter.topics.push_back(*sourceView(source).topic);
  }
  try {
    reader.set_filter(storage_filter);
  } catch (const std::exception& exception) {
    return ReplayResult::failure(
        replayError(ReplayErrorCode::BagReadFailed,
                    "failed to install ROS 2 bag topic filter: " + std::string(exception.what()),
                    std::move(stats)));
  }

  std::vector<core::SourceClockGuard> clock_guards(profile.sources().size());
  std::uint64_t ingress_value = 0U;
  std::uint64_t measurement_value = 0U;
  const ReplayBounds bounds = profile.bounds();

  while (stats.bag_messages_read < bounds.max_bag_messages &&
         stats.events_emitted < bounds.max_events) {
    bool has_next = false;
    try {
      has_next = reader.has_next();
    } catch (const std::exception& exception) {
      return ReplayResult::failure(replayError(
          ReplayErrorCode::BagReadFailed,
          "failed while checking the next ROS 2 bag message: " + std::string(exception.what()),
          std::move(stats)));
    }
    if (!has_next) {
      return ReplayResult::success(ReplayReport{ReplayCompletion::EndOfBag, std::move(stats)});
    }

    std::shared_ptr<rosbag2_storage::SerializedBagMessage> serialized;
    try {
      serialized = reader.read_next();
    } catch (const std::exception& exception) {
      return ReplayResult::failure(
          replayError(ReplayErrorCode::BagReadFailed,
                      "failed to read the next ROS 2 bag message: " + std::string(exception.what()),
                      std::move(stats)));
    }
    ++stats.bag_messages_read;
    const core::IngressSequence ingress{ingress_value};
    if (!nextSequence(&ingress_value)) {
      return ReplayResult::failure(replayError(ReplayErrorCode::SequenceOverflow,
                                               "global bag ingress sequence exhausted",
                                               std::move(stats)));
    }

    const auto source_found = source_by_topic.find(serialized->topic_name);
    if (source_found == source_by_topic.end()) {
      ++stats.unknown_topics[serialized->topic_name];
      continue;
    }

    const std::size_t source_index = source_found->second;
    const ReplaySource& source = profile.sources()[source_index];
    const SourceView view = sourceView(source);
    TopicReplayStats& topic_stats = stats.configured_topics[*view.topic];
    ++topic_stats.messages_seen;
    ++stats.configured_messages_seen;
    const core::MeasurementId measurement{measurement_value};
    if (!nextSequence(&measurement_value)) {
      return ReplayResult::failure(replayError(ReplayErrorCode::SequenceOverflow,
                                               "global measurement sequence exhausted",
                                               std::move(stats)));
    }

    const ros::ObservationContext context = makeObservationContext(
        profile, *view.identity, ingress, measurement, serialized->time_stamp);
    std::optional<ConvertedObservation> converted =
        std::visit(Overloaded{
                       [&](const ImuReplaySource&) {
                         return deserializeAndConvert<sensor_msgs::msg::Imu>(*serialized, source,
                                                                             context, &topic_stats);
                       },
                       [&](const LidarReplaySource&) {
                         return deserializeAndConvert<sensor_msgs::msg::PointCloud2>(
                             *serialized, source, context, &topic_stats);
                       },
                       [&](const CameraReplaySource& camera) {
                         if (camera.wire_format == CameraWireFormat::Image) {
                           return deserializeAndConvert<sensor_msgs::msg::Image>(
                               *serialized, source, context, &topic_stats);
                         }
                         return deserializeAndConvert<sensor_msgs::msg::CompressedImage>(
                             *serialized, source, context, &topic_stats);
                       },
                       [&](const GnssReplaySource&) {
                         return deserializeAndConvert<sensor_msgs::msg::NavSatFix>(
                             *serialized, source, context, &topic_stats);
                       },
                   },
                   source);
    if (!converted) {
      continue;
    }

    const core::SourceStamp& converted_stamp = std::visit(
        [](const auto& observation) -> const core::SourceStamp& { return observation.stamp; },
        converted->observation);
    auto guarded = clock_guards[source_index].admit(converted_stamp);
    if (!guarded) {
      ++topic_stats.clock_guard_errors;
      continue;
    }

    const std::optional<ReplayConversionAccounting> conversion_accounting = converted->accounting;
    if (conversion_accounting &&
        !conversionAccountingFits(*conversion_accounting, topic_stats, stats)) {
      return ReplayResult::failure(replayError(
          conversion_accounting->valid() ? ReplayErrorCode::SequenceOverflow
                                         : ReplayErrorCode::BagReadFailed,
          conversion_accounting->valid() ? "conversion element accounting counters exhausted"
                                         : "converter produced inconsistent element accounting",
          std::move(stats)));
    }

    ReplayEventMetadata metadata;
    metadata.topic = *view.topic;
    metadata.source_name = view.identity->name;
    metadata.recorded_timestamp = RecordedBagTimestamp{serialized->time_stamp};
    metadata.recorded_arrival = core::ArrivalTime{serialized->time_stamp};
    metadata.ingress_sequence = ingress;
    metadata.measurement_sequence = measurement;
    metadata.conversion_accounting = conversion_accounting;
    DomainEvent event(std::move(metadata), std::move(converted->observation));

    ReplayVisitResult decision = [&]() {
      try {
        return visitor(std::move(event));
      } catch (const std::exception& exception) {
        return ReplayVisitResult::failure(
            ReplayVisitorError{"visitor threw an exception: " + std::string(exception.what())});
      }
    }();
    if (!decision) {
      return ReplayResult::failure(
          replayError(ReplayErrorCode::VisitorFailed, decision.error().detail, std::move(stats)));
    }
    if (conversion_accounting) {
      accumulateConversionAccounting(*conversion_accounting, &topic_stats, &stats);
    }
    ++topic_stats.events_emitted;
    ++stats.events_emitted;
    if (decision.value() == ReplayVisitAction::Stop) {
      return ReplayResult::success(ReplayReport{ReplayCompletion::VisitorStop, std::move(stats)});
    }
  }

  const ReplayCompletion completion = stats.events_emitted >= bounds.max_events
                                          ? ReplayCompletion::EventLimit
                                          : ReplayCompletion::BagMessageLimit;
  return ReplayResult::success(ReplayReport{completion, std::move(stats)});
}

namespace {

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] core::PipelineStage scheduledStage(std::uint64_t id, std::string_view name) {
  const auto stage = core::makePipelineStage(core::PipelineStageId{id}, name);
  if (!stage) {
    throw std::logic_error("invalid built-in scheduled replay stage");
  }
  return *stage;
}

[[nodiscard]] const core::PipelineStage& producerTotalStage() {
  static const core::PipelineStage stage = scheduledStage(10'001U, "replay.producer.total");
  return stage;
}

[[nodiscard]] const core::PipelineStage& producerEventStage() {
  static const core::PipelineStage stage =
      scheduledStage(10'002U, "replay.producer.schedule_enqueue");
  return stage;
}

[[nodiscard]] const core::PipelineStage& consumerTotalStage() {
  static const core::PipelineStage stage = scheduledStage(10'003U, "replay.consumer.total");
  return stage;
}

[[nodiscard]] const core::PipelineStage& consumerEventStage() {
  static const core::PipelineStage stage = scheduledStage(10'004U, "replay.consumer.visit");
  return stage;
}

[[nodiscard]] core::PipelineTimingSample timingSample(
    const core::PipelineStage& stage, const core::CpuWallDuration& duration,
    core::PipelineDisposition disposition,
    std::optional<core::MeasurementId> measurement = std::nullopt) {
  core::PipelineTimingSample sample;
  sample.stage = stage;
  sample.wall_duration = duration.wall;
  sample.thread_cpu_duration = duration.thread_cpu;
  sample.work.measurement = measurement;
  sample.disposition = disposition;
  return sample;
}

enum class ScheduledQueuePushStatus {
  Accepted,
  CountLimit,
  ByteLimit,
  Cancelled,
};

class ScheduledEventQueue {
public:
  ScheduledEventQueue(std::size_t count_capacity, std::size_t byte_capacity)
      : count_capacity_(count_capacity),
        byte_capacity_(byte_capacity),
        name_(*core::PipelineQueueName::make("scheduled_replay.events")) {}

  [[nodiscard]] ScheduledQueuePushStatus push(DomainEvent&& event, std::size_t bytes) {
    const SteadyClock::time_point now = SteadyClock::now();
    std::unique_lock lock(mutex_);
    updateOldestAgeLocked(now);
    if (cancelled_) {
      return ScheduledQueuePushStatus::Cancelled;
    }
    if (queue_.size() >= count_capacity_) {
      ++rejected_;
      return ScheduledQueuePushStatus::CountLimit;
    }
    if (bytes > byte_capacity_ || current_bytes_ > byte_capacity_ - bytes) {
      ++rejected_;
      return ScheduledQueuePushStatus::ByteLimit;
    }
    queue_.emplace_back(std::move(event), bytes, now);
    current_bytes_ += bytes;
    ++accepted_;
    maximum_count_ = std::max(maximum_count_, queue_.size());
    maximum_bytes_ = std::max(maximum_bytes_, current_bytes_);
    lock.unlock();
    condition_.notify_one();
    return ScheduledQueuePushStatus::Accepted;
  }

  struct Item {
    Item(DomainEvent event_value, std::size_t byte_count,
         SteadyClock::time_point enqueue_time_value)
        : event(std::move(event_value)), bytes(byte_count), enqueue_time(enqueue_time_value) {}

    DomainEvent event;
    std::size_t bytes{};
    SteadyClock::time_point enqueue_time;
  };

  [[nodiscard]] std::optional<Item> waitPop() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&]() { return cancelled_ || producer_closed_ || !queue_.empty(); });
    if (cancelled_ || queue_.empty()) {
      return std::nullopt;
    }
    const SteadyClock::time_point now = SteadyClock::now();
    updateOldestAgeLocked(now);
    Item item = std::move(queue_.front());
    queue_.pop_front();
    current_bytes_ -= item.bytes;
    return std::optional<Item>(std::in_place, std::move(item));
  }

  [[nodiscard]] bool waitUntil(SteadyClock::time_point deadline) {
    std::unique_lock lock(mutex_);
    condition_.wait_until(lock, deadline, [&]() { return cancelled_; });
    return !cancelled_;
  }

  void closeProducer() {
    {
      std::scoped_lock lock(mutex_);
      updateOldestAgeLocked(SteadyClock::now());
      producer_closed_ = true;
    }
    condition_.notify_all();
  }

  void cancelAndAccountPending() {
    {
      std::scoped_lock lock(mutex_);
      updateOldestAgeLocked(SteadyClock::now());
      skipped_policy_ += static_cast<std::uint64_t>(queue_.size());
      queue_.clear();
      current_bytes_ = 0U;
      cancelled_ = true;
    }
    condition_.notify_all();
  }

  [[nodiscard]] ScheduledReplayQueueReport report() {
    std::scoped_lock lock(mutex_);
    updateOldestAgeLocked(SteadyClock::now());
    core::PipelineQueueSnapshot terminal;
    terminal.name = name_;
    terminal.count = queue_.size();
    terminal.bytes = current_bytes_;
    terminal.count_capacity = count_capacity_;
    terminal.byte_capacity = byte_capacity_;
    if (!queue_.empty()) {
      terminal.oldest_age = core::Duration{std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               SteadyClock::now() - queue_.front().enqueue_time)
                                               .count()};
    }
    terminal.accepted = accepted_;
    terminal.rejected = rejected_;
    terminal.skipped_policy = skipped_policy_;
    return ScheduledReplayQueueReport{terminal, maximum_count_, maximum_bytes_,
                                      maximum_oldest_age_};
  }

private:
  void updateOldestAgeLocked(SteadyClock::time_point now) {
    if (queue_.empty()) {
      return;
    }
    const core::Duration age{
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - queue_.front().enqueue_time)
            .count()};
    if (!maximum_oldest_age_ || age.nanoseconds > maximum_oldest_age_->nanoseconds) {
      maximum_oldest_age_ = age;
    }
  }

  std::size_t count_capacity_{};
  std::size_t byte_capacity_{};
  core::PipelineQueueName name_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<Item> queue_;
  std::size_t current_bytes_{};
  std::size_t maximum_count_{};
  std::size_t maximum_bytes_{};
  std::optional<core::Duration> maximum_oldest_age_;
  std::uint64_t accepted_{};
  std::uint64_t rejected_{};
  std::uint64_t skipped_policy_{};
  bool producer_closed_{};
  bool cancelled_{};
};

enum class ConsumerControlState {
  Running,
  Stop,
  Failed,
};

struct ConsumerControlSnapshot {
  ConsumerControlState state{ConsumerControlState::Running};
  std::string detail;
};

class ConsumerControl {
public:
  void requestStop() {
    std::scoped_lock lock(mutex_);
    if (state_ == ConsumerControlState::Running) {
      state_ = ConsumerControlState::Stop;
    }
  }

  void fail(std::string detail) {
    std::scoped_lock lock(mutex_);
    if (state_ == ConsumerControlState::Running) {
      state_ = ConsumerControlState::Failed;
      detail_ = std::move(detail);
    }
  }

  [[nodiscard]] ConsumerControlSnapshot snapshot() const {
    std::scoped_lock lock(mutex_);
    return ConsumerControlSnapshot{state_, detail_};
  }

private:
  mutable std::mutex mutex_;
  ConsumerControlState state_{ConsumerControlState::Running};
  std::string detail_;
};

enum class RecordedScheduleStatus {
  Ready,
  Cancelled,
  NonMonotonic,
  RangeOverflow,
};

class RecordedArrivalScheduler {
public:
  explicit RecordedArrivalScheduler(double playback_rate) : playback_rate_(playback_rate) {}

  [[nodiscard]] RecordedScheduleStatus wait(core::ArrivalTime recorded_arrival,
                                            ScheduledEventQueue* queue) {
    if (!first_arrival_) {
      first_arrival_ = recorded_arrival;
      previous_arrival_ = recorded_arrival;
      wall_anchor_ = SteadyClock::now();
      return RecordedScheduleStatus::Ready;
    }
    if (recorded_arrival.nanoseconds < previous_arrival_->nanoseconds) {
      return RecordedScheduleStatus::NonMonotonic;
    }
    previous_arrival_ = recorded_arrival;

    const long double recorded_delta = static_cast<long double>(recorded_arrival.nanoseconds) -
                                       static_cast<long double>(first_arrival_->nanoseconds);
    const long double scheduled_delta = recorded_delta / playback_rate_;
    if (!std::isfinite(scheduled_delta) || scheduled_delta < 0.0L ||
        scheduled_delta > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
      return RecordedScheduleStatus::RangeOverflow;
    }
    const auto delay =
        std::chrono::nanoseconds{static_cast<std::int64_t>(std::llround(scheduled_delta))};
    return queue->waitUntil(wall_anchor_ + delay) ? RecordedScheduleStatus::Ready
                                                  : RecordedScheduleStatus::Cancelled;
  }

private:
  double playback_rate_{};
  std::optional<core::ArrivalTime> first_arrival_;
  std::optional<core::ArrivalTime> previous_arrival_;
  SteadyClock::time_point wall_anchor_;
};

[[nodiscard]] ReplayVisitResult cancelledProducerDecision(const ConsumerControl& control) {
  const ConsumerControlSnapshot outcome = control.snapshot();
  if (outcome.state == ConsumerControlState::Stop) {
    return ReplayVisitResult::success(ReplayVisitAction::Stop);
  }
  if (outcome.state == ConsumerControlState::Failed) {
    return ReplayVisitResult::failure(ReplayVisitorError{outcome.detail});
  }
  return ReplayVisitResult::failure(ReplayVisitorError{"scheduled replay queue was cancelled"});
}

[[nodiscard]] ScheduledReplayRuntimeReport makeScheduledRuntime(
    ScheduledEventQueue* queue, core::PipelineTimingSample producer_total,
    core::PipelineTimingSample consumer_total,
    const core::BoundedPipelineTimingAccumulator& producer_events,
    const core::BoundedPipelineTimingAccumulator& consumer_events) {
  ScheduledReplayQueueReport queue_report = queue->report();
  producer_total.queue = queue_report.terminal;
  consumer_total.queue = queue_report.terminal;
  ScheduledReplayRuntimeReport output;
  output.queue = std::move(queue_report);
  output.producer_total = std::move(producer_total);
  output.consumer_total = std::move(consumer_total);
  output.producer_schedule_enqueue = producer_events.snapshot();
  output.consumer_visit = consumer_events.snapshot();
  return output;
}

[[nodiscard]] ReplayStats replayStatsFrom(const ReplayResult& result) {
  return result ? result.value().stats : result.error().stats;
}

}  // namespace

ScheduledReplayResult replayRos2BagScheduled(const std::filesystem::path& bag_uri,
                                             const ReplayProfile& profile,
                                             const ScheduledReplayOptions& options,
                                             DomainEventVisitor visitor) {
  ReplayStats empty_stats;
  if (!visitor) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::InvalidVisitor, "domain-event visitor is empty",
                             std::move(empty_stats), std::nullopt});
  }
  if (!std::isfinite(options.playback_rate) || options.playback_rate <= 0.0 ||
      options.queue_count_capacity == 0U || options.queue_byte_capacity == 0U ||
      options.timing_window_capacity == 0U) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::InvalidSchedule,
                             "scheduled replay requires a finite positive rate and non-zero queue "
                             "and timing capacities",
                             std::move(empty_stats), std::nullopt});
  }

  ScheduledEventQueue queue(options.queue_count_capacity, options.queue_byte_capacity);
  ConsumerControl consumer_control;
  core::BoundedPipelineTimingAccumulator producer_events(producerEventStage(),
                                                         options.timing_window_capacity);
  core::BoundedPipelineTimingAccumulator consumer_events(consumerEventStage(),
                                                         options.timing_window_capacity);

  std::optional<ReplayResult> producer_result;
  std::optional<core::PipelineTimingSample> producer_total;
  std::optional<core::PipelineTimingSample> consumer_total;
  std::optional<std::string> queue_overflow_detail;
  std::optional<std::string> scheduling_error_detail;
  std::optional<std::string> consumer_error_detail;
  bool consumer_stopped = false;

  std::optional<std::thread> consumer_thread;
  try {
    consumer_thread.emplace([&]() {
      core::ThreadCpuWallTimer total_timer;
      try {
        while (std::optional<ScheduledEventQueue::Item> item = queue.waitPop()) {
          core::ThreadCpuWallTimer event_timer;
          const std::optional<core::MeasurementId> measurement =
              item->event.metadata().measurement_sequence;
          ReplayVisitResult decision = [&]() {
            try {
              return visitor(std::move(item->event));
            } catch (const std::exception& exception) {
              return ReplayVisitResult::failure(ReplayVisitorError{"visitor threw an exception: " +
                                                                   std::string(exception.what())});
            } catch (...) {
              return ReplayVisitResult::failure(
                  ReplayVisitorError{"visitor threw a non-standard exception"});
            }
          }();
          core::PipelineDisposition disposition = core::PipelineDisposition::Completed;
          if (!decision) {
            disposition = core::PipelineDisposition::Failed;
          } else if (decision.value() == ReplayVisitAction::Stop) {
            disposition = core::PipelineDisposition::Accepted;
          }
          const auto observed = consumer_events.observe(
              timingSample(consumerEventStage(), event_timer.elapsed(), disposition, measurement));
          static_cast<void>(observed);

          if (!decision) {
            consumer_error_detail = decision.error().detail;
            consumer_control.fail(*consumer_error_detail);
            queue.cancelAndAccountPending();
            break;
          }
          if (decision.value() == ReplayVisitAction::Stop) {
            consumer_stopped = true;
            consumer_control.requestStop();
            queue.cancelAndAccountPending();
            break;
          }
        }
      } catch (const std::exception& exception) {
        consumer_error_detail =
            "scheduled replay consumer threw an exception: " + std::string(exception.what());
        consumer_control.fail(*consumer_error_detail);
        queue.cancelAndAccountPending();
      } catch (...) {
        consumer_error_detail = "scheduled replay consumer threw a non-standard exception";
        consumer_control.fail(*consumer_error_detail);
        queue.cancelAndAccountPending();
      }
      const core::PipelineDisposition disposition =
          consumer_error_detail ? core::PipelineDisposition::Failed
                                : (consumer_stopped ? core::PipelineDisposition::Accepted
                                                    : core::PipelineDisposition::Completed);
      consumer_total = timingSample(consumerTotalStage(), total_timer.elapsed(), disposition);
    });
  } catch (const std::system_error& exception) {
    return ScheduledReplayResult::failure(ScheduledReplayError{
        ReplayErrorCode::ThreadStartFailed,
        "failed to start scheduled replay consumer: " + std::string(exception.what()),
        ReplayStats{}, std::nullopt});
  }

  std::optional<std::thread> producer_thread;
  try {
    producer_thread.emplace([&]() {
      core::ThreadCpuWallTimer total_timer;
      RecordedArrivalScheduler scheduler(options.playback_rate);
      try {
        producer_result.emplace(replayRos2Bag(bag_uri, profile, [&](DomainEvent&& event) {
          core::ThreadCpuWallTimer event_timer;
          const core::MeasurementId measurement = event.metadata().measurement_sequence;
          const RecordedScheduleStatus schedule_status =
              scheduler.wait(event.metadata().recorded_arrival, &queue);
          if (schedule_status == RecordedScheduleStatus::Cancelled) {
            const auto observed = producer_events.observe(
                timingSample(producerEventStage(), event_timer.elapsed(),
                             core::PipelineDisposition::Skipped, measurement));
            static_cast<void>(observed);
            return cancelledProducerDecision(consumer_control);
          }
          if (schedule_status == RecordedScheduleStatus::NonMonotonic ||
              schedule_status == RecordedScheduleStatus::RangeOverflow) {
            scheduling_error_detail = schedule_status == RecordedScheduleStatus::NonMonotonic
                                          ? "recorded arrival timestamps are non-monotonic"
                                          : "recorded arrival schedule exceeds steady-clock range";
            const auto observed = producer_events.observe(
                timingSample(producerEventStage(), event_timer.elapsed(),
                             core::PipelineDisposition::Failed, measurement));
            static_cast<void>(observed);
            return ReplayVisitResult::failure(ReplayVisitorError{*scheduling_error_detail});
          }

          const std::size_t bytes = estimateDomainEventBytes(event);
          const ScheduledQueuePushStatus push_status = queue.push(std::move(event), bytes);
          if (push_status == ScheduledQueuePushStatus::Accepted) {
            const auto observed = producer_events.observe(
                timingSample(producerEventStage(), event_timer.elapsed(),
                             core::PipelineDisposition::Accepted, measurement));
            static_cast<void>(observed);
            return ReplayVisitResult::success(ReplayVisitAction::Continue);
          }
          if (push_status == ScheduledQueuePushStatus::Cancelled) {
            const auto observed = producer_events.observe(
                timingSample(producerEventStage(), event_timer.elapsed(),
                             core::PipelineDisposition::Skipped, measurement));
            static_cast<void>(observed);
            return cancelledProducerDecision(consumer_control);
          }

          queue_overflow_detail = push_status == ScheduledQueuePushStatus::CountLimit
                                      ? "scheduled replay event queue count capacity exceeded"
                                      : "scheduled replay event queue byte capacity exceeded";
          const auto observed = producer_events.observe(
              timingSample(producerEventStage(), event_timer.elapsed(),
                           core::PipelineDisposition::Rejected, measurement));
          static_cast<void>(observed);
          return ReplayVisitResult::failure(ReplayVisitorError{*queue_overflow_detail});
        }));
      } catch (const std::exception& exception) {
        producer_result.emplace(ReplayResult::failure(replayError(
            ReplayErrorCode::BagReadFailed,
            "scheduled replay producer threw an exception: " + std::string(exception.what()),
            ReplayStats{})));
      } catch (...) {
        producer_result.emplace(ReplayResult::failure(replayError(
            ReplayErrorCode::BagReadFailed,
            "scheduled replay producer threw a non-standard exception", ReplayStats{})));
      }
      queue.closeProducer();
      const core::PipelineDisposition disposition = producer_result && *producer_result
                                                        ? core::PipelineDisposition::Completed
                                                        : core::PipelineDisposition::Failed;
      producer_total = timingSample(producerTotalStage(), total_timer.elapsed(), disposition);
    });
  } catch (const std::system_error& exception) {
    queue.cancelAndAccountPending();
    consumer_thread->join();
    return ScheduledReplayResult::failure(ScheduledReplayError{
        ReplayErrorCode::ThreadStartFailed,
        "failed to start scheduled replay producer: " + std::string(exception.what()),
        ReplayStats{}, std::nullopt});
  }

  producer_thread->join();
  consumer_thread->join();

  if (!producer_result || !producer_total || !consumer_total) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::ThreadStartFailed,
                             "scheduled replay worker terminated without a complete result",
                             ReplayStats{}, std::nullopt});
  }

  ScheduledReplayRuntimeReport runtime = makeScheduledRuntime(
      &queue, *producer_total, *consumer_total, producer_events, consumer_events);
  ReplayStats stats = replayStatsFrom(*producer_result);

  if (queue_overflow_detail) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::QueueOverflow, *queue_overflow_detail,
                             std::move(stats), std::move(runtime)});
  }
  if (scheduling_error_detail) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::InvalidSchedule, *scheduling_error_detail,
                             std::move(stats), std::move(runtime)});
  }
  if (consumer_error_detail) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{ReplayErrorCode::VisitorFailed, *consumer_error_detail,
                             std::move(stats), std::move(runtime)});
  }
  if (!*producer_result) {
    return ScheduledReplayResult::failure(
        ScheduledReplayError{producer_result->error().code, producer_result->error().detail,
                             std::move(stats), std::move(runtime)});
  }

  ReplayCompletion completion = producer_result->value().completion;
  if (consumer_stopped) {
    completion = ReplayCompletion::VisitorStop;
  }
  return ScheduledReplayResult::success(
      ScheduledReplayReport{completion, std::move(stats), std::move(runtime)});
}

}  // namespace meridian::tools
