#include "meridian/global/lidar_place.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <set>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace meridian::global {
namespace {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

constexpr std::size_t kDescriptorBytes = 32U;
constexpr std::size_t kMultiIndexTables = 16U;

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] bool validHeader(const core::RecordHeader& header) noexcept {
  return header.schema_version > 0U && header.trace.valid() && header.producer.valid() &&
         header.session.valid() && header.config.valid() &&
         (!header.direct_calibration || header.direct_calibration->valid());
}

[[nodiscard]] bool validPose(const core::Pose3d& pose) noexcept {
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d rotation = pose.so3().matrix();
  return (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <
             1.0e-9 &&
         std::abs(rotation.determinant() - 1.0) < 1.0e-9;
}

[[nodiscard]] bool validSubmap(const core::SubmapRef& submap) noexcept {
  return core::validateSubmapRef(submap) == core::SubmapRefValidationError::None;
}

[[nodiscard]] bool validSubmap(const FinalizedSubmapFrame& submap) noexcept {
  return validSubmap(submap.ref) && validPose(submap.T_odom_submap);
}

[[nodiscard]] bool zeroHash(const core::ContentHash& hash) noexcept {
  return std::all_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte == 0U; });
}

[[nodiscard]] bool sameSubmapObject(const core::SubmapRef& left,
                                    const core::SubmapRef& right) noexcept {
  return left.session == right.session && left.odom_epoch == right.odom_epoch &&
         left.id == right.id;
}

[[nodiscard]] bool sameSubmap(const FinalizedSubmapFrame& left,
                              const FinalizedSubmapFrame& right) noexcept {
  return left.ref == right.ref && left.support_end == right.support_end &&
         (left.T_odom_submap.matrix() - right.T_odom_submap.matrix()).cwiseAbs().maxCoeff() <=
             1.0e-12;
}

[[nodiscard]] LidarPlaceDescriptorError descriptorError(LidarPlaceDescriptorErrorCode code,
                                                        std::string detail) {
  return LidarPlaceDescriptorError{code, std::move(detail)};
}

[[nodiscard]] LidarPlaceIndexError indexError(
    LidarPlaceIndexErrorCode code, std::string detail,
    std::optional<LidarPlaceEntryKey> key = std::nullopt) {
  LidarPlaceIndexError result;
  result.code = code;
  result.key = key;
  result.detail = std::move(detail);
  return result;
}

[[nodiscard]] LidarLoopVerifierError verifierError(LidarLoopVerifierErrorCode code,
                                                   std::string detail) {
  return LidarLoopVerifierError{code, std::move(detail)};
}

[[nodiscard]] bool validDescriptorConfig(const LidarPlaceDescriptorConfig& config) noexcept {
  return config.model_revision.valid() && config.config_revision.valid() &&
         finitePositive(config.density_resolution_m) &&
         std::isfinite(config.minimum_density_fraction) && config.minimum_density_fraction >= 0.0 &&
         config.minimum_density_fraction < 1.0 && config.minimum_proxy_points > 0U &&
         config.maximum_proxy_points >= config.minimum_proxy_points &&
         config.maximum_bev_cells > 0U && config.maximum_features > 0U &&
         config.minimum_features > 0U && config.minimum_features <= config.maximum_features &&
         config.maximum_features <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         config.orb_edge_threshold_pixels > 0 && config.orb_patch_size_pixels >= 3 &&
         config.orb_fast_threshold >= 0 && config.self_similarity_hamming_threshold >= 0 &&
         config.self_similarity_hamming_threshold <= 256;
}

[[nodiscard]] bool validProxySample(const RegistrationProxyPoint& sample) noexcept {
  const double normal_norm = sample.normal_submap.norm();
  return sample.point_submap.allFinite() && sample.normal_submap.allFinite() &&
         std::isfinite(normal_norm) && normal_norm > 1.0e-9 && finitePositive(sample.weight);
}

[[nodiscard]] bool validDescriptor(const LidarPlaceDescriptor& descriptor) noexcept {
  if (!validHeader(descriptor.header) || !validSubmap(descriptor.submap) ||
      descriptor.header.session != descriptor.submap.ref.session ||
      (descriptor.header.direct_calibration &&
       *descriptor.header.direct_calibration != descriptor.submap.ref.calibration) ||
      !descriptor.model_revision.valid() || !descriptor.config_revision.valid() ||
      zeroHash(descriptor.registration_proxy_checksum) ||
      !std::isfinite(descriptor.ground_height_submap_m) || descriptor.features.empty()) {
    return false;
  }
  return std::all_of(
      descriptor.features.begin(), descriptor.features.end(), [](const LidarBevFeature& feature) {
        return feature.position_submap_m.allFinite() && std::isfinite(feature.response);
      });
}

[[nodiscard]] bool sameDescriptorContent(const LidarPlaceDescriptor& left,
                                         const LidarPlaceDescriptor& right) noexcept {
  if (!sameSubmap(left.submap, right.submap) || left.model_revision != right.model_revision ||
      left.config_revision != right.config_revision ||
      left.registration_proxy_checksum != right.registration_proxy_checksum ||
      left.ground_height_submap_m != right.ground_height_submap_m ||
      left.features.size() != right.features.size()) {
    return false;
  }
  return std::equal(left.features.begin(), left.features.end(), right.features.begin(),
                    [](const LidarBevFeature& first, const LidarBevFeature& second) {
                      return first.position_submap_m == second.position_submap_m &&
                             first.binary_descriptor == second.binary_descriptor &&
                             first.response == second.response;
                    });
}

[[nodiscard]] int hammingDistance(
    const std::array<std::uint8_t, kDescriptorBytes>& left,
    const std::array<std::uint8_t, kDescriptorBytes>& right) noexcept {
  int distance = 0;
  for (std::size_t index = 0U; index < kDescriptorBytes; ++index) {
    distance += std::popcount(static_cast<unsigned int>(left[index] ^ right[index]));
  }
  return distance;
}

[[nodiscard]] std::uint32_t multiIndexKey(
    const std::array<std::uint8_t, kDescriptorBytes>& descriptor, std::size_t table) noexcept {
  const std::size_t byte = table * 2U;
  const std::uint32_t word = static_cast<std::uint32_t>(descriptor[byte]) |
                             (static_cast<std::uint32_t>(descriptor[byte + 1U]) << 8U);
  return static_cast<std::uint32_t>((table << 16U) | word);
}

[[nodiscard]] std::uint64_t orderedNanoseconds(std::int64_t value) noexcept {
  return std::bit_cast<std::uint64_t>(value) ^ (std::uint64_t{1} << 63U);
}

[[nodiscard]] bool absoluteTimeDifferenceLess(core::FusionTime left, core::FusionTime right,
                                              core::Duration threshold) noexcept {
  const std::uint64_t left_ordered = orderedNanoseconds(left.nanoseconds);
  const std::uint64_t right_ordered = orderedNanoseconds(right.nanoseconds);
  const std::uint64_t difference =
      left_ordered >= right_ordered ? left_ordered - right_ordered : right_ordered - left_ordered;
  return difference < static_cast<std::uint64_t>(threshold.nanoseconds);
}

[[nodiscard]] core::FusionTime saturatingAdd(core::FusionTime time,
                                             core::Duration duration) noexcept {
  if (duration.nanoseconds > 0 &&
      time.nanoseconds > std::numeric_limits<std::int64_t>::max() - duration.nanoseconds) {
    return core::FusionTime{std::numeric_limits<std::int64_t>::max()};
  }
  if (duration.nanoseconds < 0 &&
      time.nanoseconds < std::numeric_limits<std::int64_t>::min() - duration.nanoseconds) {
    return core::FusionTime{std::numeric_limits<std::int64_t>::min()};
  }
  return core::FusionTime{time.nanoseconds + duration.nanoseconds};
}

struct BevBounds {
  std::int64_t minimum_x{};
  std::int64_t minimum_y{};
  std::size_t columns{};
  std::size_t rows{};
};

[[nodiscard]] core::Result<BevBounds, LidarPlaceDescriptorError> bevBounds(
    std::span<const RegistrationProxyPoint> samples, double resolution, std::size_t maximum_cells) {
  using Result = core::Result<BevBounds, LidarPlaceDescriptorError>;
  std::int64_t minimum_x = std::numeric_limits<std::int64_t>::max();
  std::int64_t minimum_y = std::numeric_limits<std::int64_t>::max();
  std::int64_t maximum_x = std::numeric_limits<std::int64_t>::min();
  std::int64_t maximum_y = std::numeric_limits<std::int64_t>::min();
  for (const RegistrationProxyPoint& sample : samples) {
    if (!validProxySample(sample)) {
      continue;
    }
    const double cell_x = std::floor(sample.point_submap.x() / resolution);
    const double cell_y = std::floor(sample.point_submap.y() / resolution);
    if (cell_x < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        cell_x > static_cast<double>(std::numeric_limits<std::int32_t>::max()) ||
        cell_y < static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
        cell_y > static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
      return Result::failure(
          descriptorError(LidarPlaceDescriptorErrorCode::NumericalFailure,
                          "BEV cell coordinate exceeds the bounded signed index range"));
    }
    const auto x = static_cast<std::int64_t>(cell_x);
    const auto y = static_cast<std::int64_t>(cell_y);
    minimum_x = std::min(minimum_x, x);
    minimum_y = std::min(minimum_y, y);
    maximum_x = std::max(maximum_x, x);
    maximum_y = std::max(maximum_y, y);
  }
  if (minimum_x > maximum_x || minimum_y > maximum_y) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                        "registration proxy contains no valid sample for a BEV"));
  }
  const std::uint64_t columns = static_cast<std::uint64_t>(maximum_x - minimum_x) + 1U;
  const std::uint64_t rows = static_cast<std::uint64_t>(maximum_y - minimum_y) + 1U;
  if (rows > maximum_cells || columns > maximum_cells || rows > maximum_cells / columns ||
      rows > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::BevCapacity,
                        "density BEV exceeds configured cell or OpenCV dimension capacity"));
  }
  return Result::success(BevBounds{minimum_x, minimum_y, static_cast<std::size_t>(columns),
                                   static_cast<std::size_t>(rows)});
}

[[nodiscard]] double quantile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double position = fraction * static_cast<double>(values.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower);
  return (1.0 - alpha) * values[lower] + alpha * values[upper];
}

struct PointPair2d {
  Eigen::Vector2d from;
  Eigen::Vector2d to;
  int descriptor_distance{};
  std::size_t from_feature{};
  std::size_t to_feature{};
};

struct Alignment2d {
  double yaw{};
  Eigen::Vector2d translation{Eigen::Vector2d::Zero()};

  [[nodiscard]] Eigen::Vector2d apply(const Eigen::Vector2d& point) const noexcept {
    const double cosine = std::cos(yaw);
    const double sine = std::sin(yaw);
    return Eigen::Vector2d(cosine * point.x() - sine * point.y(),
                           sine * point.x() + cosine * point.y()) +
           translation;
  }
};

[[nodiscard]] Alignment2d fitAlignment2d(std::span<const PointPair2d> matches,
                                         std::span<const std::size_t> inliers) {
  Eigen::Vector2d from_mean = Eigen::Vector2d::Zero();
  Eigen::Vector2d to_mean = Eigen::Vector2d::Zero();
  for (std::size_t index : inliers) {
    from_mean += matches[index].from;
    to_mean += matches[index].to;
  }
  const double count = static_cast<double>(inliers.size());
  from_mean /= count;
  to_mean /= count;
  double dot = 0.0;
  double cross = 0.0;
  for (std::size_t index : inliers) {
    const Eigen::Vector2d from = matches[index].from - from_mean;
    const Eigen::Vector2d to = matches[index].to - to_mean;
    dot += to.dot(from);
    cross += to.x() * from.y() - to.y() * from.x();
  }
  const double yaw = std::atan2(cross, dot);
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  Eigen::Matrix2d rotation;
  rotation << cosine, -sine, sine, cosine;
  return Alignment2d{yaw, from_mean - rotation * to_mean};
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

struct Ransac2dResult {
  Alignment2d alignment;
  std::size_t inliers{};
  double rmse_m{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] Ransac2dResult deterministicRansac2d(std::span<const PointPair2d> matches,
                                                   std::size_t maximum_trials,
                                                   double minimum_baseline_m,
                                                   double inlier_distance_m, std::uint64_t seed) {
  struct Sample {
    std::size_t first{};
    std::size_t second{};
    std::uint64_t order{};
  };
  std::vector<Sample> samples;
  const std::size_t reserve =
      matches.size() > 1U ? matches.size() * (matches.size() - 1U) / 2U : 0U;
  samples.reserve(std::min(reserve, maximum_trials * 4U));
  for (std::size_t first = 0U; first < matches.size(); ++first) {
    for (std::size_t second = first + 1U; second < matches.size(); ++second) {
      const double from_baseline = (matches[second].from - matches[first].from).norm();
      const double to_baseline = (matches[second].to - matches[first].to).norm();
      if (std::min(from_baseline, to_baseline) < minimum_baseline_m) {
        continue;
      }
      const std::uint64_t pair_key =
          (static_cast<std::uint64_t>(first) << 32U) | static_cast<std::uint64_t>(second);
      samples.push_back(Sample{first, second, mix64(seed ^ pair_key)});
    }
  }
  std::sort(samples.begin(), samples.end(), [](const Sample& left, const Sample& right) {
    return std::tie(left.order, left.first, left.second) <
           std::tie(right.order, right.first, right.second);
  });
  if (samples.size() > maximum_trials) {
    samples.resize(maximum_trials);
  }

  const double squared_threshold = inlier_distance_m * inlier_distance_m;
  std::vector<std::size_t> best_inliers;
  double best_squared_error = std::numeric_limits<double>::infinity();
  for (const Sample& sample : samples) {
    const std::array<std::size_t, 2> sample_indices{sample.first, sample.second};
    const Alignment2d hypothesis = fitAlignment2d(matches, sample_indices);
    std::vector<std::size_t> inliers;
    inliers.reserve(matches.size());
    double squared_error = 0.0;
    for (std::size_t index = 0U; index < matches.size(); ++index) {
      const double error =
          (hypothesis.apply(matches[index].to) - matches[index].from).squaredNorm();
      if (error <= squared_threshold) {
        inliers.push_back(index);
        squared_error += error;
      }
    }
    if (inliers.size() > best_inliers.size() ||
        (inliers.size() == best_inliers.size() && squared_error < best_squared_error)) {
      best_inliers = std::move(inliers);
      best_squared_error = squared_error;
    }
  }
  if (best_inliers.size() < 2U) {
    return {};
  }
  const Alignment2d refined = fitAlignment2d(matches, best_inliers);
  double squared_error = 0.0;
  for (std::size_t index : best_inliers) {
    squared_error += (refined.apply(matches[index].to) - matches[index].from).squaredNorm();
  }
  return Ransac2dResult{refined, best_inliers.size(),
                        std::sqrt(squared_error / static_cast<double>(best_inliers.size()))};
}

}  // namespace

LidarPlaceEntryKey lidarPlaceEntryKey(const LidarPlaceDescriptor& descriptor) noexcept {
  return LidarPlaceEntryKey{descriptor.submap.ref};
}

core::Result<ImmutableLidarPlaceDescriptor, LidarPlaceDescriptorError> buildLidarPlaceDescriptor(
    const core::RecordHeader& header, const FinalizedSubmapFrame& submap,
    const RegistrationProxy& registration_proxy, const LidarPlaceDescriptorConfig& config) {
  using Result = core::Result<ImmutableLidarPlaceDescriptor, LidarPlaceDescriptorError>;
  if (!validDescriptorConfig(config)) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InvalidConfiguration,
                        "LiDAR place descriptor configuration is invalid or unbounded"));
  }
  if (!validHeader(header)) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InvalidHeader,
                        "LiDAR place descriptor requires a valid immutable record header"));
  }
  if (!validSubmap(submap) || header.session != submap.ref.session ||
      (header.direct_calibration && *header.direct_calibration != submap.ref.calibration)) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InvalidSubmap,
                        "LiDAR place descriptor requires a valid immutable submap reference"));
  }
  if (!finitePositive(registration_proxy.voxel_resolution_m) ||
      zeroHash(registration_proxy.checksum)) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InvalidRegistrationProxy,
                        "registration proxy resolution and checksum must be valid"));
  }
  if (registration_proxy.points.size() > config.maximum_proxy_points) {
    return Result::failure(descriptorError(LidarPlaceDescriptorErrorCode::ProxyCapacity,
                                           "registration proxy exceeds descriptor input capacity"));
  }
  const std::size_t valid_points = static_cast<std::size_t>(std::count_if(
      registration_proxy.points.begin(), registration_proxy.points.end(), validProxySample));
  if (valid_points < config.minimum_proxy_points) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                        "registration proxy has insufficient valid support for a BEV"));
  }
  const auto bounds =
      bevBounds(registration_proxy.points, config.density_resolution_m, config.maximum_bev_cells);
  if (!bounds) {
    return Result::failure(bounds.error());
  }

  cv::Mat counts(static_cast<int>(bounds.value().rows), static_cast<int>(bounds.value().columns),
                 CV_32SC1, cv::Scalar(0));
  std::vector<double> heights;
  heights.reserve(valid_points);
  std::uint32_t maximum_count = 0U;
  for (const RegistrationProxyPoint& sample : registration_proxy.points) {
    if (!validProxySample(sample)) {
      continue;
    }
    const auto x = static_cast<std::int64_t>(
        std::floor(sample.point_submap.x() / config.density_resolution_m));
    const auto y = static_cast<std::int64_t>(
        std::floor(sample.point_submap.y() / config.density_resolution_m));
    const int column = static_cast<int>(x - bounds.value().minimum_x);
    const int row = static_cast<int>(y - bounds.value().minimum_y);
    int& count = counts.at<int>(row, column);
    if (count < std::numeric_limits<int>::max()) {
      ++count;
    }
    maximum_count = std::max(maximum_count, static_cast<std::uint32_t>(count));
    heights.push_back(sample.point_submap.z());
  }
  if (maximum_count == 0U) {
    return Result::failure(descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                                           "registration proxy generated an empty density BEV"));
  }

  cv::Mat density(counts.rows, counts.cols, CV_8UC1, cv::Scalar(0));
  const double denominator = std::log1p(static_cast<double>(maximum_count));
  const double minimum_value = config.minimum_density_fraction * 255.0;
  std::size_t occupied_cells = 0U;
  for (int row = 0; row < counts.rows; ++row) {
    for (int column = 0; column < counts.cols; ++column) {
      const int count = counts.at<int>(row, column);
      if (count <= 0) {
        continue;
      }
      const double normalized = 255.0 * std::log1p(static_cast<double>(count)) / denominator;
      if (normalized >= minimum_value) {
        density.at<std::uint8_t>(row, column) =
            static_cast<std::uint8_t>(std::clamp(std::lround(normalized), 0L, 255L));
        ++occupied_cells;
      }
    }
  }
  if (occupied_cells == 0U) {
    return Result::failure(descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                                           "density threshold removed every occupied BEV cell"));
  }

  const auto orb = cv::ORB::create(static_cast<int>(config.maximum_features), 1.0F, 1,
                                   config.orb_edge_threshold_pixels, 0, 2, cv::ORB::HARRIS_SCORE,
                                   config.orb_patch_size_pixels, config.orb_fast_threshold);
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  orb->detectAndCompute(density, cv::noArray(), keypoints, descriptors);
  if (descriptors.empty() || descriptors.cols != 32 || descriptors.type() != CV_8UC1 ||
      descriptors.rows != static_cast<int>(keypoints.size())) {
    return Result::failure(descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                                           "density BEV produced no valid 256-bit ORB descriptor"));
  }

  std::vector<bool> retain(keypoints.size(), true);
  std::size_t self_similarity_pruned = 0U;
  if (descriptors.rows >= 2) {
    cv::BFMatcher self_matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> self_matches;
    self_matcher.knnMatch(descriptors, descriptors, self_matches, 2);
    for (const auto& matches : self_matches) {
      if (matches.size() < 2U) {
        continue;
      }
      const std::size_t query = static_cast<std::size_t>(matches.front().queryIdx);
      if (matches[1].distance <= static_cast<float>(config.self_similarity_hamming_threshold) &&
          query < retain.size()) {
        retain[query] = false;
        ++self_similarity_pruned;
      }
    }
  }

  auto descriptor = std::make_shared<LidarPlaceDescriptor>(header, submap);
  descriptor->model_revision = config.model_revision;
  descriptor->config_revision = config.config_revision;
  descriptor->registration_proxy_checksum = registration_proxy.checksum;
  descriptor->ground_height_submap_m = quantile(std::move(heights), 0.10);
  descriptor->features.reserve(keypoints.size() - self_similarity_pruned);
  for (std::size_t index = 0U; index < keypoints.size(); ++index) {
    if (!retain[index]) {
      continue;
    }
    const cv::KeyPoint& keypoint = keypoints[index];
    LidarBevFeature feature;
    feature.position_submap_m.x() =
        (static_cast<double>(bounds.value().minimum_x) + static_cast<double>(keypoint.pt.x) + 0.5) *
        config.density_resolution_m;
    feature.position_submap_m.y() =
        (static_cast<double>(bounds.value().minimum_y) + static_cast<double>(keypoint.pt.y) + 0.5) *
        config.density_resolution_m;
    std::copy_n(descriptors.ptr<std::uint8_t>(static_cast<int>(index)), kDescriptorBytes,
                feature.binary_descriptor.begin());
    feature.response = keypoint.response;
    descriptor->features.push_back(feature);
  }
  if (descriptor->features.size() < config.minimum_features) {
    return Result::failure(
        descriptorError(LidarPlaceDescriptorErrorCode::InsufficientSupport,
                        "self-similarity-pruned BEV has too few place features"));
  }
  descriptor->build = LidarPlaceDescriptorBuildReport{config.model_revision,
                                                      config.config_revision,
                                                      registration_proxy.points.size(),
                                                      valid_points,
                                                      occupied_cells,
                                                      bounds.value().rows,
                                                      bounds.value().columns,
                                                      keypoints.size(),
                                                      self_similarity_pruned,
                                                      descriptor->features.size(),
                                                      config.density_resolution_m,
                                                      descriptor->ground_height_submap_m};
  return Result::success(std::shared_ptr<const LidarPlaceDescriptor>(std::move(descriptor)));
}

struct LidarPlaceIndex::Impl {
  struct FeatureHandle {
    std::size_t entry{};
    std::size_t feature{};
  };

  explicit Impl(LidarPlaceIndexConfig index_config) : config(std::move(index_config)) {}

  LidarPlaceIndexConfig config;
  std::vector<ImmutableLidarPlaceDescriptor> entries;
  std::map<core::SparseSubmapIdentityKey, std::size_t> entry_lookup;
  std::unordered_map<std::uint32_t, std::vector<FeatureHandle>> buckets;
  std::size_t feature_count{};
  std::size_t reference_count{};
};

namespace {

[[nodiscard]] bool validIndexConfig(const LidarPlaceIndexConfig& config) noexcept {
  return config.model_revision.valid() && config.config_revision.valid() &&
         config.maximum_entries > 0U && config.maximum_total_features > 0U &&
         config.maximum_features_per_entry > 0U &&
         config.maximum_index_references >= kMultiIndexTables &&
         config.maximum_descriptor_comparisons_per_query > 0U &&
         config.maximum_ransac_candidates > 0U && config.maximum_ransac_trials_per_candidate > 0U &&
         config.maximum_top_k > 0U && config.minimum_descriptor_matches >= 2U &&
         config.minimum_ransac_inliers >= 2U &&
         config.minimum_ransac_inliers <= config.minimum_descriptor_matches &&
         config.maximum_hamming_distance >= 0 && config.maximum_hamming_distance <= 256 &&
         std::isfinite(config.minimum_ransac_inlier_ratio) &&
         config.minimum_ransac_inlier_ratio > 0.0 && config.minimum_ransac_inlier_ratio <= 1.0 &&
         finitePositive(config.minimum_ransac_baseline_m) &&
         finitePositive(config.ransac_inlier_distance_m) &&
         config.minimum_temporal_separation.nanoseconds >= 0 &&
         config.candidate_ttl.nanoseconds > 0;
}

[[nodiscard]] bool containsKey(std::span<const LidarPlaceEntryKey> keys,
                               const LidarPlaceEntryKey& key) {
  return std::binary_search(keys.begin(), keys.end(), key);
}

struct RawFeatureMatch {
  std::size_t query_feature{};
  std::size_t target_feature{};
  int distance{};
};

[[nodiscard]] std::vector<PointPair2d> uniqueFeatureMatches(std::vector<RawFeatureMatch> raw,
                                                            const LidarPlaceDescriptor& target,
                                                            const LidarPlaceDescriptor& query) {
  std::sort(raw.begin(), raw.end(), [](const RawFeatureMatch& left, const RawFeatureMatch& right) {
    return std::tie(left.distance, left.query_feature, left.target_feature) <
           std::tie(right.distance, right.query_feature, right.target_feature);
  });
  std::vector<bool> query_used(query.features.size(), false);
  std::vector<bool> target_used(target.features.size(), false);
  std::vector<PointPair2d> result;
  result.reserve(std::min(query.features.size(), target.features.size()));
  for (const RawFeatureMatch& match : raw) {
    if (match.query_feature >= query_used.size() || match.target_feature >= target_used.size() ||
        query_used[match.query_feature] || target_used[match.target_feature]) {
      continue;
    }
    query_used[match.query_feature] = true;
    target_used[match.target_feature] = true;
    result.push_back(PointPair2d{target.features[match.target_feature].position_submap_m,
                                 query.features[match.query_feature].position_submap_m,
                                 match.distance, match.target_feature, match.query_feature});
  }
  return result;
}

}  // namespace

LidarPlaceIndex::LidarPlaceIndex(LidarPlaceIndexConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

LidarPlaceIndex::~LidarPlaceIndex() = default;
LidarPlaceIndex::LidarPlaceIndex(LidarPlaceIndex&&) noexcept = default;
LidarPlaceIndex& LidarPlaceIndex::operator=(LidarPlaceIndex&&) noexcept = default;

core::Result<LidarPlaceIndexInsertReport, LidarPlaceIndexError> LidarPlaceIndex::insert(
    ImmutableLidarPlaceDescriptor descriptor) {
  using Result = core::Result<LidarPlaceIndexInsertReport, LidarPlaceIndexError>;
  if (!validIndexConfig(impl_->config)) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::InvalidConfiguration,
                                      "LiDAR place index configuration is invalid or unbounded"));
  }
  if (!descriptor || !validDescriptor(*descriptor) ||
      descriptor->features.size() > impl_->config.maximum_features_per_entry) {
    return Result::failure(
        indexError(LidarPlaceIndexErrorCode::InvalidDescriptor,
                   "LiDAR place index requires a bounded valid immutable descriptor"));
  }
  const LidarPlaceEntryKey key = lidarPlaceEntryKey(*descriptor);
  const core::SparseSubmapIdentityKey identity_key =
      core::sparseSubmapIdentityKey(key.submap);
  if (descriptor->model_revision != impl_->config.model_revision) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::ModelRevisionMismatch,
                                      "descriptor model revision does not match the index", key));
  }
  if (descriptor->config_revision != impl_->config.config_revision) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::ConfigRevisionMismatch,
                                      "descriptor configuration revision does not match the index",
                                      key));
  }
  const auto existing = impl_->entry_lookup.find(identity_key);
  if (existing != impl_->entry_lookup.end()) {
    const auto& retained = *impl_->entries.at(existing->second);
    if (!sameDescriptorContent(retained, *descriptor)) {
      return Result::failure(
          indexError(LidarPlaceIndexErrorCode::IdentityConflict,
                     "same LiDAR place identity names different immutable content", key));
    }
    return Result::success(LidarPlaceIndexInsertReport{
        impl_->config.model_revision, impl_->config.config_revision, key,
        LidarPlaceIndexInsertDisposition::AlreadyPresent, impl_->entries.size(),
        impl_->feature_count, impl_->reference_count});
  }
  if (impl_->entries.size() >= impl_->config.maximum_entries) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::EntryCapacity,
                                      "LiDAR place index entry capacity is exhausted", key));
  }
  if (descriptor->features.size() > impl_->config.maximum_total_features - impl_->feature_count) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::FeatureCapacity,
                                      "LiDAR place index feature capacity is exhausted", key));
  }
  if (descriptor->features.size() >
      (impl_->config.maximum_index_references - impl_->reference_count) / kMultiIndexTables) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::IndexReferenceCapacity,
                                      "LiDAR place multi-index reference capacity is exhausted",
                                      key));
  }

  const std::size_t entry_index = impl_->entries.size();
  impl_->entries.push_back(std::move(descriptor));
  impl_->entry_lookup.emplace(identity_key, entry_index);
  const auto& inserted = *impl_->entries.back();
  for (std::size_t feature_index = 0U; feature_index < inserted.features.size(); ++feature_index) {
    for (std::size_t table = 0U; table < kMultiIndexTables; ++table) {
      impl_->buckets[multiIndexKey(inserted.features[feature_index].binary_descriptor, table)]
          .push_back(Impl::FeatureHandle{entry_index, feature_index});
    }
  }
  impl_->feature_count += inserted.features.size();
  impl_->reference_count += inserted.features.size() * kMultiIndexTables;
  return Result::success(
      LidarPlaceIndexInsertReport{impl_->config.model_revision, impl_->config.config_revision, key,
                                  LidarPlaceIndexInsertDisposition::Inserted, impl_->entries.size(),
                                  impl_->feature_count, impl_->reference_count});
}

core::Result<LidarPlaceRetrievalResult, LidarPlaceIndexError> LidarPlaceIndex::retrieve(
    const LidarPlaceRetrievalRequest& request) const {
  using Result = core::Result<LidarPlaceRetrievalResult, LidarPlaceIndexError>;
  if (!validIndexConfig(impl_->config)) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::InvalidConfiguration,
                                      "LiDAR place index configuration is invalid or unbounded"));
  }
  if (!request.query || !validDescriptor(*request.query) ||
      request.query->features.size() > impl_->config.maximum_features_per_entry ||
      request.top_k == 0U || request.top_k > impl_->config.maximum_top_k ||
      request.now < request.query->submap.support_end) {
    return Result::failure(
        indexError(LidarPlaceIndexErrorCode::InvalidQuery,
                   "LiDAR retrieval query, revisions, top-K, or evaluation time is invalid"));
  }
  if (request.query->model_revision != impl_->config.model_revision) {
    return Result::failure(indexError(LidarPlaceIndexErrorCode::ModelRevisionMismatch,
                                      "query descriptor model revision does not match the index",
                                      lidarPlaceEntryKey(*request.query)));
  }
  if (request.query->config_revision != impl_->config.config_revision) {
    return Result::failure(
        indexError(LidarPlaceIndexErrorCode::ConfigRevisionMismatch,
                   "query descriptor configuration revision does not match the index",
                   lidarPlaceEntryKey(*request.query)));
  }
  std::vector<LidarPlaceEntryKey> excluded = request.excluded;
  std::sort(excluded.begin(), excluded.end());
  if (std::adjacent_find(excluded.begin(), excluded.end()) != excluded.end() ||
      std::any_of(excluded.begin(), excluded.end(), [](const LidarPlaceEntryKey& key) {
        return core::validateSubmapRef(key.submap) != core::SubmapRefValidationError::None;
      })) {
    return Result::failure(
        indexError(LidarPlaceIndexErrorCode::InvalidQuery,
                   "LiDAR retrieval exclusion list must not contain duplicate identities"));
  }

  LidarPlaceRetrievalResult result;
  result.report.model_revision = impl_->config.model_revision;
  result.report.config_revision = impl_->config.config_revision;
  result.report.query = lidarPlaceEntryKey(*request.query);
  result.report.evaluated_at = request.now;
  result.report.candidates_valid_until = saturatingAdd(request.now, impl_->config.candidate_ttl);
  result.report.requested_top_k = request.top_k;
  result.report.indexed_entries = impl_->entries.size();

  std::vector<bool> eligible(impl_->entries.size(), true);
  for (std::size_t entry_index = 0U; entry_index < impl_->entries.size(); ++entry_index) {
    const LidarPlaceDescriptor& entry = *impl_->entries[entry_index];
    const LidarPlaceEntryKey key = lidarPlaceEntryKey(entry);
    if (key == result.report.query) {
      eligible[entry_index] = false;
      ++result.report.excluded_same_identity;
      continue;
    }
    if (entry.submap.ref.session != request.query->submap.ref.session) {
      eligible[entry_index] = false;
      ++result.report.excluded_other_session;
      continue;
    }
    if (containsKey(excluded, key)) {
      eligible[entry_index] = false;
      ++result.report.excluded_explicit_policy;
      continue;
    }
    if (entry.submap.ref.session == request.query->submap.ref.session &&
        absoluteTimeDifferenceLess(entry.submap.support_end, request.query->submap.support_end,
                                   impl_->config.minimum_temporal_separation)) {
      eligible[entry_index] = false;
      ++result.report.excluded_temporal_separation;
      continue;
    }
    ++result.report.entries_examined;
  }

  std::size_t bucket_references = 0U;
  for (const LidarBevFeature& query_feature : request.query->features) {
    for (std::size_t table = 0U; table < kMultiIndexTables; ++table) {
      const auto bucket =
          impl_->buckets.find(multiIndexKey(query_feature.binary_descriptor, table));
      if (bucket == impl_->buckets.end()) {
        continue;
      }
      const std::size_t eligible_references = static_cast<std::size_t>(
          std::count_if(bucket->second.begin(), bucket->second.end(),
                        [&](const Impl::FeatureHandle& handle) { return eligible[handle.entry]; }));
      if (eligible_references >
          impl_->config.maximum_descriptor_comparisons_per_query - bucket_references) {
        return Result::failure(
            indexError(LidarPlaceIndexErrorCode::QueryResourceLimit,
                       "complete LiDAR multi-index query exceeds the descriptor comparison budget",
                       result.report.query));
      }
      bucket_references += eligible_references;
    }
  }
  result.report.multi_index_bucket_references = bucket_references;

  std::map<std::size_t, std::vector<RawFeatureMatch>> raw_matches;
  std::set<std::tuple<std::size_t, std::size_t, std::size_t>> compared;
  for (std::size_t query_feature_index = 0U; query_feature_index < request.query->features.size();
       ++query_feature_index) {
    const auto& query_feature = request.query->features[query_feature_index];
    for (std::size_t table = 0U; table < kMultiIndexTables; ++table) {
      const auto bucket =
          impl_->buckets.find(multiIndexKey(query_feature.binary_descriptor, table));
      if (bucket == impl_->buckets.end()) {
        continue;
      }
      for (const Impl::FeatureHandle& handle : bucket->second) {
        if (!eligible[handle.entry] ||
            !compared.emplace(handle.entry, handle.feature, query_feature_index).second) {
          continue;
        }
        ++result.report.exact_hamming_comparisons;
        const auto& target_feature = impl_->entries[handle.entry]->features[handle.feature];
        const int distance =
            hammingDistance(target_feature.binary_descriptor, query_feature.binary_descriptor);
        if (distance <= impl_->config.maximum_hamming_distance) {
          raw_matches[handle.entry].push_back(
              RawFeatureMatch{query_feature_index, handle.feature, distance});
        }
      }
    }
  }

  struct CandidateWork {
    std::size_t entry{};
    std::vector<PointPair2d> matches;
  };
  std::vector<CandidateWork> candidate_work;
  candidate_work.reserve(raw_matches.size());
  for (auto& [entry, matches] : raw_matches) {
    std::vector<PointPair2d> unique =
        uniqueFeatureMatches(std::move(matches), *impl_->entries[entry], *request.query);
    if (unique.size() >= impl_->config.minimum_descriptor_matches) {
      candidate_work.push_back(CandidateWork{entry, std::move(unique)});
    }
  }
  result.report.entries_with_descriptor_votes = candidate_work.size();
  std::sort(candidate_work.begin(), candidate_work.end(),
            [&](const CandidateWork& left, const CandidateWork& right) {
              if (left.matches.size() != right.matches.size()) {
                return left.matches.size() > right.matches.size();
              }
              return lidarPlaceEntryKey(*impl_->entries[left.entry]) <
                     lidarPlaceEntryKey(*impl_->entries[right.entry]);
            });
  if (candidate_work.size() > impl_->config.maximum_ransac_candidates) {
    candidate_work.resize(impl_->config.maximum_ransac_candidates);
  }

  struct AcceptedSeed {
    LidarRetrievalSeed seed;
    LidarPlaceCandidateReport report;
  };
  std::vector<AcceptedSeed> accepted;
  for (const CandidateWork& candidate : candidate_work) {
    const auto& target = *impl_->entries[candidate.entry];
    const std::uint64_t deterministic_seed =
        mix64(target.submap.ref.session.value() ^ target.submap.ref.id.value() ^
              (request.query->submap.ref.id.value() << 1U));
    const Ransac2dResult ransac =
        deterministicRansac2d(candidate.matches, impl_->config.maximum_ransac_trials_per_candidate,
                              impl_->config.minimum_ransac_baseline_m,
                              impl_->config.ransac_inlier_distance_m, deterministic_seed);
    ++result.report.ransac_candidates_evaluated;
    LidarPlaceCandidateReport candidate_report;
    candidate_report.model_revision = impl_->config.model_revision;
    candidate_report.config_revision = impl_->config.config_revision;
    candidate_report.candidate = lidarPlaceEntryKey(target);
    candidate_report.descriptor_matches = candidate.matches.size();
    candidate_report.ransac_inliers = ransac.inliers;
    candidate_report.ransac_inlier_ratio =
        static_cast<double>(ransac.inliers) / static_cast<double>(candidate.matches.size());
    candidate_report.ransac_rmse_m = ransac.rmse_m;
    candidate_report.passed_geometric_seed_gate =
        ransac.inliers >= impl_->config.minimum_ransac_inliers &&
        candidate_report.ransac_inlier_ratio >= impl_->config.minimum_ransac_inlier_ratio;
    result.report.candidates.push_back(candidate_report);
    if (!candidate_report.passed_geometric_seed_gate) {
      continue;
    }

    core::RecordHeader seed_header = request.query->header;
    seed_header.created_at = request.now;
    LidarRetrievalSeed seed(seed_header, target.submap.ref, request.query->submap.ref);
    seed.model_revision = impl_->config.model_revision;
    seed.config_revision = impl_->config.config_revision;
    seed.from_proxy_checksum = target.registration_proxy_checksum;
    seed.to_proxy_checksum = request.query->registration_proxy_checksum;
    const double cosine = std::cos(ransac.alignment.yaw);
    const double sine = std::sin(ransac.alignment.yaw);
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation.topLeftCorner<2, 2>() << cosine, -sine, sine, cosine;
    Eigen::Vector3d translation;
    translation.head<2>() = ransac.alignment.translation;
    translation.z() = target.ground_height_submap_m - request.query->ground_height_submap_m;
    seed.T_from_to_seed = core::Pose3d(Sophus::SO3d(rotation), translation);
    seed.valid_until = result.report.candidates_valid_until;
    seed.descriptor_matches = candidate.matches.size();
    seed.ransac_inliers = ransac.inliers;
    seed.ransac_inlier_ratio = candidate_report.ransac_inlier_ratio;
    seed.ransac_rmse_m = ransac.rmse_m;
    seed.retrieval_score = static_cast<double>(ransac.inliers) - 0.25 * ransac.rmse_m;
    accepted.push_back(AcceptedSeed{std::move(seed), std::move(candidate_report)});
  }
  std::sort(accepted.begin(), accepted.end(),
            [](const AcceptedSeed& left, const AcceptedSeed& right) {
              if (left.seed.ransac_inliers != right.seed.ransac_inliers) {
                return left.seed.ransac_inliers > right.seed.ransac_inliers;
              }
              if (left.seed.ransac_rmse_m != right.seed.ransac_rmse_m) {
                return left.seed.ransac_rmse_m < right.seed.ransac_rmse_m;
              }
              return left.seed.from < right.seed.from;
            });
  if (accepted.size() > request.top_k) {
    accepted.erase(accepted.begin() + static_cast<std::ptrdiff_t>(request.top_k), accepted.end());
  }
  result.seeds.reserve(accepted.size());
  for (AcceptedSeed& item : accepted) {
    result.seeds.push_back(std::move(item.seed));
  }
  result.report.emitted_top_k = result.seeds.size();
  return Result::success(std::move(result));
}

std::size_t LidarPlaceIndex::size() const noexcept {
  return impl_->entries.size();
}

std::size_t LidarPlaceIndex::featureCount() const noexcept {
  return impl_->feature_count;
}

const LidarPlaceIndexConfig& LidarPlaceIndex::config() const noexcept {
  return impl_->config;
}

namespace {

[[nodiscard]] bool validVerifierConfig(const LidarLoopVerifierConfig& config) noexcept {
  return config.model_revision.valid() && config.config_revision.valid() &&
         config.maximum_proxy_points_per_submap > 0U &&
         config.minimum_proxy_points_per_submap > 0U &&
         config.maximum_proxy_points_per_submap >= config.minimum_proxy_points_per_submap &&
         config.maximum_iterations > 0U && config.maximum_damping_trials > 0U &&
         config.minimum_correspondences_each_direction > 0U && config.support_grid_dimension > 0U &&
         config.minimum_support_cells_each_direction > 0U &&
         config.minimum_support_cells_each_direction <=
             config.support_grid_dimension * config.support_grid_dimension &&
         config.minimum_information_rank > 0U && config.minimum_information_rank <= 6U &&
         finitePositive(config.maximum_correspondence_distance_m) &&
         std::isfinite(config.minimum_bidirectional_overlap) &&
         config.minimum_bidirectional_overlap > 0.0 &&
         config.minimum_bidirectional_overlap <= 1.0 &&
         std::isfinite(config.minimum_normal_consistency) &&
         config.minimum_normal_consistency >= 0.0 && config.minimum_normal_consistency <= 1.0 &&
         std::isfinite(config.minimum_pair_normal_cosine) &&
         config.minimum_pair_normal_cosine >= 0.0 && config.minimum_pair_normal_cosine <= 1.0 &&
         finitePositive(config.maximum_residual_median_m) &&
         finitePositive(config.maximum_residual_quantile_m) &&
         config.maximum_residual_quantile_m >= config.maximum_residual_median_m &&
         std::isfinite(config.residual_upper_quantile) && config.residual_upper_quantile > 0.5 &&
         config.residual_upper_quantile < 1.0 &&
         finitePositive(config.normal_standard_deviation_m) &&
         finitePositive(config.tangential_standard_deviation_m) &&
         config.tangential_standard_deviation_m >= config.normal_standard_deviation_m &&
         finitePositive(config.model_standard_deviation_m) &&
         finitePositive(config.huber_delta_sigma) &&
         finitePositive(config.initial_relative_damping) &&
         std::isfinite(config.damping_increase) && config.damping_increase > 1.0 &&
         std::isfinite(config.damping_decrease) && config.damping_decrease > 0.0 &&
         config.damping_decrease < 1.0 && finitePositive(config.translation_convergence_m) &&
         finitePositive(config.rotation_convergence_rad) &&
         finitePositive(config.maximum_iteration_translation_step_m) &&
         finitePositive(config.maximum_iteration_rotation_step_rad) &&
         finitePositive(config.maximum_seed_translation_correction_m) &&
         finitePositive(config.maximum_seed_rotation_correction_rad) &&
         finitePositive(config.maximum_gravity_tilt_rad) &&
         config.maximum_gravity_tilt_rad < std::numbers::pi &&
         finitePositive(config.absolute_observable_eigenvalue) &&
         std::isfinite(config.relative_observable_eigenvalue) &&
         config.relative_observable_eigenvalue >= 0.0 &&
         config.relative_observable_eigenvalue < 1.0 &&
         finitePositive(config.maximum_information_eigenvalue) &&
         config.maximum_information_eigenvalue >= config.absolute_observable_eigenvalue &&
         finitePositive(config.correspondence_information_inflation) &&
         config.correspondence_information_inflation >= 1.0 &&
         finitePositive(config.maximum_proxy_weight);
}

[[nodiscard]] bool validProxy(const RegistrationProxy& proxy,
                              const LidarLoopVerifierConfig& config) noexcept {
  if (!finitePositive(proxy.voxel_resolution_m) || zeroHash(proxy.checksum) ||
      proxy.points.size() < config.minimum_proxy_points_per_submap ||
      proxy.points.size() > config.maximum_proxy_points_per_submap) {
    return false;
  }
  return std::all_of(proxy.points.begin(), proxy.points.end(), [&](const auto& sample) {
    if (!validProxySample(sample) || sample.weight > config.maximum_proxy_weight) {
      return false;
    }
    const Eigen::Array3d cell =
        (sample.point_submap / config.maximum_correspondence_distance_m).array().floor();
    const double minimum = static_cast<double>(std::numeric_limits<std::int32_t>::min()) + 1.0;
    const double maximum = static_cast<double>(std::numeric_limits<std::int32_t>::max()) - 1.0;
    return cell.allFinite() && (cell >= minimum).all() && (cell <= maximum).all();
  });
}

struct VoxelKey {
  std::int32_t x{};
  std::int32_t y{};
  std::int32_t z{};

  bool operator==(const VoxelKey&) const = default;
};

struct VoxelKeyHash {
  [[nodiscard]] std::size_t operator()(const VoxelKey& key) const noexcept {
    std::uint64_t value = static_cast<std::uint32_t>(key.x) * 0x9e3779b1U;
    value ^= static_cast<std::uint32_t>(key.y) * 0x85ebca77U;
    value ^= static_cast<std::uint32_t>(key.z) * 0xc2b2ae3dU;
    return static_cast<std::size_t>(mix64(value));
  }
};

[[nodiscard]] std::optional<VoxelKey> voxelKey(const Eigen::Vector3d& point,
                                               double resolution) noexcept {
  const Eigen::Array3d coordinate = (point / resolution).array().floor();
  if (!coordinate.allFinite() ||
      (coordinate < static_cast<double>(std::numeric_limits<std::int32_t>::min())).any() ||
      (coordinate > static_cast<double>(std::numeric_limits<std::int32_t>::max())).any()) {
    return std::nullopt;
  }
  return VoxelKey{static_cast<std::int32_t>(coordinate.x()),
                  static_cast<std::int32_t>(coordinate.y()),
                  static_cast<std::int32_t>(coordinate.z())};
}

class ProxyVoxelIndex {
public:
  ProxyVoxelIndex(const RegistrationProxy& proxy, double resolution)
      : proxy_(proxy), resolution_(resolution) {
    buckets_.reserve(proxy.points.size());
    for (std::size_t index = 0U; index < proxy.points.size(); ++index) {
      const auto key = voxelKey(proxy.points[index].point_submap, resolution_);
      if (key) {
        buckets_[*key].push_back(index);
      }
    }
  }

  [[nodiscard]] std::optional<std::size_t> nearest(const Eigen::Vector3d& point,
                                                   double maximum_distance) const noexcept {
    const auto center = voxelKey(point, resolution_);
    if (!center) {
      return std::nullopt;
    }
    double best_squared = maximum_distance * maximum_distance;
    std::optional<std::size_t> best;
    for (std::int32_t dx = -1; dx <= 1; ++dx) {
      for (std::int32_t dy = -1; dy <= 1; ++dy) {
        for (std::int32_t dz = -1; dz <= 1; ++dz) {
          const VoxelKey key{static_cast<std::int32_t>(center->x + dx),
                             static_cast<std::int32_t>(center->y + dy),
                             static_cast<std::int32_t>(center->z + dz)};
          const auto bucket = buckets_.find(key);
          if (bucket == buckets_.end()) {
            continue;
          }
          for (std::size_t index : bucket->second) {
            const double squared = (proxy_.points[index].point_submap - point).squaredNorm();
            if (squared < best_squared || (squared == best_squared && (!best || index < *best))) {
              best_squared = squared;
              best = index;
            }
          }
        }
      }
    }
    return best;
  }

private:
  const RegistrationProxy& proxy_;
  double resolution_{};
  std::unordered_map<VoxelKey, std::vector<std::size_t>, VoxelKeyHash> buckets_;
};

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) noexcept {
  Eigen::Matrix3d result;
  result << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return result;
}

[[nodiscard]] Eigen::Matrix3d surfelCovariance(const Eigen::Vector3d& unit_normal,
                                               const LidarLoopVerifierConfig& config) noexcept {
  const double normal_variance =
      config.normal_standard_deviation_m * config.normal_standard_deviation_m;
  const double tangential_variance =
      config.tangential_standard_deviation_m * config.tangential_standard_deviation_m;
  return tangential_variance * Eigen::Matrix3d::Identity() +
         (normal_variance - tangential_variance) * unit_normal * unit_normal.transpose();
}

struct GicpPair {
  std::size_t from_index{};
  std::size_t to_index{};
  bool forward{};
  double euclidean_residual_m{};
  double normal_cosine{};
};

struct GicpEvaluation {
  Matrix6 hessian{Matrix6::Zero()};
  Vector6 gradient{Vector6::Zero()};
  std::vector<GicpPair> pairs;
  std::size_t forward_correspondences{};
  std::size_t reverse_correspondences{};
  std::size_t normal_consistent_correspondences{};
  double cost{};
};

[[nodiscard]] GicpEvaluation evaluateGicp(const core::Pose3d& T_from_to,
                                          const RegistrationProxy& from_proxy,
                                          const RegistrationProxy& to_proxy,
                                          const ProxyVoxelIndex& from_index,
                                          const ProxyVoxelIndex& to_index,
                                          const LidarLoopVerifierConfig& config, bool derivatives) {
  GicpEvaluation result;
  result.pairs.reserve(from_proxy.points.size() + to_proxy.points.size());
  const Eigen::Matrix3d rotation = T_from_to.so3().matrix();
  const Eigen::Vector3d translation = T_from_to.translation();
  const core::Pose3d T_to_from = T_from_to.inverse();

  for (std::size_t to = 0U; to < to_proxy.points.size(); ++to) {
    const Eigen::Vector3d transformed = rotation * to_proxy.points[to].point_submap + translation;
    const auto from = from_index.nearest(transformed, config.maximum_correspondence_distance_m);
    if (!from) {
      continue;
    }
    const double residual = (transformed - from_proxy.points[*from].point_submap).norm();
    const double normal_cosine = std::abs(from_proxy.points[*from].normal_submap.normalized().dot(
        rotation * to_proxy.points[to].normal_submap.normalized()));
    result.pairs.push_back(GicpPair{*from, to, true, residual, normal_cosine});
    ++result.forward_correspondences;
  }
  for (std::size_t from = 0U; from < from_proxy.points.size(); ++from) {
    const Eigen::Vector3d point_to = T_to_from * from_proxy.points[from].point_submap;
    const auto to = to_index.nearest(point_to, config.maximum_correspondence_distance_m);
    if (!to) {
      continue;
    }
    const Eigen::Vector3d transformed = rotation * to_proxy.points[*to].point_submap + translation;
    const double residual = (transformed - from_proxy.points[from].point_submap).norm();
    const double normal_cosine = std::abs(from_proxy.points[from].normal_submap.normalized().dot(
        rotation * to_proxy.points[*to].normal_submap.normalized()));
    result.pairs.push_back(GicpPair{from, *to, false, residual, normal_cosine});
    ++result.reverse_correspondences;
  }

  const double model_variance =
      config.model_standard_deviation_m * config.model_standard_deviation_m;
  for (const GicpPair& pair : result.pairs) {
    if (pair.normal_cosine < config.minimum_pair_normal_cosine) {
      continue;
    }
    ++result.normal_consistent_correspondences;
    const auto& from = from_proxy.points[pair.from_index];
    const auto& to = to_proxy.points[pair.to_index];
    const Eigen::Vector3d transformed = rotation * to.point_submap + translation;
    const Eigen::Vector3d residual = transformed - from.point_submap;
    const Eigen::Vector3d from_normal = from.normal_submap.normalized();
    const Eigen::Vector3d to_normal = to.normal_submap.normalized();
    Eigen::Matrix3d covariance =
        surfelCovariance(from_normal, config) +
        rotation * surfelCovariance(to_normal, config) * rotation.transpose();
    covariance.diagonal().array() += model_variance;
    covariance = 0.5 * (covariance + covariance.transpose());
    const Eigen::LLT<Eigen::Matrix3d> cholesky(covariance);
    if (cholesky.info() != Eigen::Success) {
      continue;
    }
    const Eigen::Matrix3d mahalanobis = cholesky.solve(Eigen::Matrix3d::Identity());
    const double squared = std::max(0.0, residual.dot(mahalanobis * residual));
    const double sigma = std::sqrt(squared);
    const double huber_weight =
        sigma <= config.huber_delta_sigma || sigma <= std::numeric_limits<double>::epsilon()
            ? 1.0
            : config.huber_delta_sigma / sigma;
    const double proxy_weight = std::sqrt(from.weight * to.weight);
    result.cost +=
        proxy_weight * (sigma <= config.huber_delta_sigma
                            ? 0.5 * squared
                            : config.huber_delta_sigma * (sigma - 0.5 * config.huber_delta_sigma));
    if (!derivatives) {
      continue;
    }
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = rotation;
    jacobian.rightCols<3>() = -rotation * skew(to.point_submap);
    const double weight = proxy_weight * huber_weight;
    result.hessian.noalias() += weight * jacobian.transpose() * mahalanobis * jacobian;
    result.gradient.noalias() += weight * jacobian.transpose() * mahalanobis * residual;
  }
  // A pose cannot lower the objective simply by moving support outside the
  // correspondence radius. Missing directional support pays the Huber loss at
  // the configured maximum correspondence distance and model scale.
  const std::size_t expected = from_proxy.points.size() + to_proxy.points.size();
  const std::size_t missing = expected - result.normal_consistent_correspondences;
  const double missing_sigma =
      config.maximum_correspondence_distance_m / config.model_standard_deviation_m;
  const double missing_cost =
      missing_sigma <= config.huber_delta_sigma
          ? 0.5 * missing_sigma * missing_sigma
          : config.huber_delta_sigma * (missing_sigma - 0.5 * config.huber_delta_sigma);
  result.cost += static_cast<double>(missing) * missing_cost;
  result.hessian = 0.5 * (result.hessian + result.hessian.transpose());
  return result;
}

[[nodiscard]] std::size_t spatialSupportCells(const RegistrationProxy& proxy,
                                              std::span<const std::size_t> indices,
                                              std::size_t dimension) {
  if (indices.empty()) {
    return 0U;
  }
  Eigen::Vector2d minimum = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
  Eigen::Vector2d maximum = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
  for (const auto& point : proxy.points) {
    minimum = minimum.cwiseMin(point.point_submap.head<2>());
    maximum = maximum.cwiseMax(point.point_submap.head<2>());
  }
  const Eigen::Vector2d extent = (maximum - minimum).cwiseMax(1.0e-9);
  std::vector<bool> occupied(dimension * dimension, false);
  for (std::size_t index : indices) {
    const Eigen::Array2d normalized =
        ((proxy.points[index].point_submap.head<2>() - minimum).cwiseQuotient(extent)).array();
    const std::size_t x = std::min(
        dimension - 1U, static_cast<std::size_t>(std::floor(std::clamp(normalized.x(), 0.0, 1.0) *
                                                            static_cast<double>(dimension))));
    const std::size_t y = std::min(
        dimension - 1U, static_cast<std::size_t>(std::floor(std::clamp(normalized.y(), 0.0, 1.0) *
                                                            static_cast<double>(dimension))));
    occupied[y * dimension + x] = true;
  }
  return static_cast<std::size_t>(std::count(occupied.begin(), occupied.end(), true));
}

struct FinalInformation {
  core::RankAwareInformation information;
  Vector6 raw_eigenvalues{Vector6::Zero()};
  double calibrated_sigma_m{};
};

[[nodiscard]] FinalInformation finalRankAwareInformation(const core::Pose3d& T_from_to,
                                                         const RegistrationProxy& from_proxy,
                                                         const RegistrationProxy& to_proxy,
                                                         std::span<const GicpPair> pairs,
                                                         const LidarLoopVerifierConfig& config) {
  const Eigen::Matrix3d rotation = T_from_to.so3().matrix();
  const Eigen::Vector3d translation = T_from_to.translation();
  std::set<std::pair<std::size_t, std::size_t>> unique;
  std::vector<double> absolute_normal_residuals;
  absolute_normal_residuals.reserve(pairs.size());
  struct NormalRow {
    Eigen::Matrix<double, 1, 6> jacobian;
    double residual{};
    double proxy_weight{};
  };
  std::vector<NormalRow> rows;
  rows.reserve(pairs.size());
  for (const GicpPair& pair : pairs) {
    if (pair.normal_cosine < config.minimum_pair_normal_cosine) {
      continue;
    }
    if (!unique.emplace(pair.from_index, pair.to_index).second) {
      continue;
    }
    const auto& from = from_proxy.points[pair.from_index];
    const auto& to = to_proxy.points[pair.to_index];
    const Eigen::Vector3d transformed_normal = rotation * to.normal_submap.normalized();
    const Eigen::Vector3d from_normal = from.normal_submap.normalized();
    const double sign = from_normal.dot(transformed_normal) >= 0.0 ? 1.0 : -1.0;
    Eigen::Vector3d normal = from_normal + sign * transformed_normal;
    if (normal.norm() <= 1.0e-9) {
      continue;
    }
    normal.normalize();
    const Eigen::Vector3d residual_vector =
        rotation * to.point_submap + translation - from.point_submap;
    const double residual = normal.dot(residual_vector);
    Eigen::Matrix<double, 3, 6> point_jacobian;
    point_jacobian.leftCols<3>() = rotation;
    point_jacobian.rightCols<3>() = -rotation * skew(to.point_submap);
    rows.push_back(NormalRow{normal.transpose() * point_jacobian, residual,
                             std::sqrt(from.weight * to.weight)});
    absolute_normal_residuals.push_back(std::abs(residual));
  }
  FinalInformation result;
  result.calibrated_sigma_m = std::max(config.normal_standard_deviation_m,
                                       1.4826 * quantile(absolute_normal_residuals, 0.50));
  Matrix6 information = Matrix6::Zero();
  for (const NormalRow& row : rows) {
    const double normalized = std::abs(row.residual) / result.calibrated_sigma_m;
    const double robust_weight = normalized <= config.huber_delta_sigma ||
                                         normalized <= std::numeric_limits<double>::epsilon()
                                     ? 1.0
                                     : config.huber_delta_sigma / normalized;
    information.noalias() += row.proxy_weight * robust_weight /
                             (result.calibrated_sigma_m * result.calibrated_sigma_m) *
                             row.jacobian.transpose() * row.jacobian;
  }
  information /= config.correspondence_information_inflation;
  information = 0.5 * (information + information.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6> eigen_solver(information);
  if (eigen_solver.info() != Eigen::Success || !eigen_solver.eigenvalues().allFinite() ||
      !eigen_solver.eigenvectors().allFinite()) {
    return result;
  }
  const double maximum = std::max(0.0, eigen_solver.eigenvalues().maxCoeff());
  const double threshold = std::max(config.absolute_observable_eigenvalue,
                                    config.relative_observable_eigenvalue * maximum);
  std::size_t output = 0U;
  std::size_t raw_output = 0U;
  for (Eigen::Index input = 5; input >= 0; --input) {
    const double raw = std::max(0.0, eigen_solver.eigenvalues()(input));
    result.raw_eigenvalues(static_cast<Eigen::Index>(raw_output)) = raw;
    ++raw_output;
    if (raw >= threshold) {
      result.information.basis.col(static_cast<Eigen::Index>(output)) =
          eigen_solver.eigenvectors().col(input);
      result.information.eigenvalues(static_cast<Eigen::Index>(output)) =
          std::min(raw, config.maximum_information_eigenvalue);
      ++result.information.rank;
      ++output;
    }
    if (input == 0) {
      break;
    }
  }
  // Complete the basis without changing support so the public matrix remains
  // orthonormal even when observable eigenvectors were not contiguous.
  for (Eigen::Index input = 5; input >= 0 && output < 6U; --input) {
    const double raw = std::max(0.0, eigen_solver.eigenvalues()(input));
    if (raw < threshold) {
      result.information.basis.col(static_cast<Eigen::Index>(output)) =
          eigen_solver.eigenvectors().col(input);
      ++output;
    }
    if (input == 0) {
      break;
    }
  }
  result.information.tangent = core::PoseTangentConvention::RightTranslationFirst;
  return result;
}

[[nodiscard]] bool containsCalibration(std::span<const core::CalibrationEpoch> epochs,
                                       core::CalibrationEpoch calibration) noexcept {
  return std::binary_search(epochs.begin(), epochs.end(), calibration);
}

}  // namespace

core::Result<VerifiedLidarLoop, LidarLoopVerifierError> verifyLidarLoop(
    const LidarLoopVerificationRequest& request, const RegistrationProxy& from_proxy,
    const RegistrationProxy& to_proxy, const LidarLoopVerifierConfig& config) {
  using Result = core::Result<VerifiedLidarLoop, LidarLoopVerifierError>;
  if (!validVerifierConfig(config)) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::InvalidConfiguration,
                      "LiDAR loop verifier configuration is invalid or unbounded"));
  }
  if (!validHeader(request.header) || !request.proposal.valid() ||
      !validHeader(request.seed.header) || !validSubmap(request.seed.from) ||
      !validSubmap(request.seed.to) || sameSubmapObject(request.seed.from, request.seed.to) ||
      !validPose(request.seed.T_from_to_seed) ||
      request.evaluated_at < request.seed.header.created_at) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::InvalidRequest,
                      "LiDAR verification request, endpoints, seed pose, or time is invalid"));
  }
  if (request.evaluated_at > request.seed.valid_until) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::SeedExpired,
                      "LiDAR retrieval seed expired before geometric verification"));
  }
  if (request.seed.model_revision != config.model_revision ||
      request.seed.config_revision != config.config_revision) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::RevisionMismatch,
                      "LiDAR retrieval and verifier model/configuration revisions differ"));
  }
  if (request.header.session != request.seed.header.session ||
      request.header.session != request.seed.from.session ||
      request.header.session != request.seed.to.session) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::EndpointMismatch,
                      "LiDAR verification header and retrieval seed name different sessions"));
  }
  if (from_proxy.checksum != request.seed.from_proxy_checksum ||
      to_proxy.checksum != request.seed.to_proxy_checksum) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::ProxyChecksumMismatch,
                      "LiDAR verification proxy checksum differs from the retrieval seed"));
  }
  if (from_proxy.points.size() > config.maximum_proxy_points_per_submap ||
      to_proxy.points.size() > config.maximum_proxy_points_per_submap) {
    return Result::failure(verifierError(LidarLoopVerifierErrorCode::ProxyCapacity,
                                         "LiDAR verification proxy exceeds configured capacity"));
  }
  if (!validProxy(from_proxy, config) || !validProxy(to_proxy, config)) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::InvalidRegistrationProxy,
                      "LiDAR verification requires finite, normalized-capable immutable proxies"));
  }
  if (request.calibration_epochs.empty() ||
      !std::is_sorted(request.calibration_epochs.begin(), request.calibration_epochs.end()) ||
      std::adjacent_find(request.calibration_epochs.begin(), request.calibration_epochs.end()) !=
          request.calibration_epochs.end() ||
      std::any_of(request.calibration_epochs.begin(), request.calibration_epochs.end(),
                  [](core::CalibrationEpoch epoch) { return !epoch.valid(); }) ||
      (request.header.direct_calibration &&
       !containsCalibration(request.calibration_epochs, *request.header.direct_calibration)) ||
      !containsCalibration(request.calibration_epochs, request.seed.from.calibration) ||
      !containsCalibration(request.calibration_epochs, request.seed.to.calibration)) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::InvalidRequest,
                      "LiDAR verification calibration epoch set is invalid or non-canonical"));
  }
  const bool has_primary =
      std::any_of(request.lineage.usage.begin(), request.lineage.usage.end(),
                  [](const core::ObservationUsage& usage) {
                    return usage.role == core::ObservationRole::PrimaryResidual;
                  });
  if (!request.lineage.id.valid() || request.lineage.usage.empty() || !has_primary ||
      core::validateLineage(request.lineage) != core::LineageValidationError::None ||
      std::any_of(request.lineage.usage.begin(), request.lineage.usage.end(),
                  [&](const core::ObservationUsage& usage) {
                    return !containsCalibration(request.calibration_epochs,
                                                usage.slice.calibration);
                  })) {
    return Result::failure(
        verifierError(LidarLoopVerifierErrorCode::InvalidLineage,
                      "LiDAR verification lineage lacks valid primary proxy observations"));
  }

  LidarLoopVerificationReport report;
  report.model_revision = config.model_revision;
  report.config_revision = config.config_revision;
  report.from_proxy_points = from_proxy.points.size();
  report.to_proxy_points = to_proxy.points.size();
  const ProxyVoxelIndex from_index(from_proxy, config.maximum_correspondence_distance_m);
  const ProxyVoxelIndex to_index(to_proxy, config.maximum_correspondence_distance_m);
  core::Pose3d estimate = request.seed.T_from_to_seed;
  GicpEvaluation evaluation =
      evaluateGicp(estimate, from_proxy, to_proxy, from_index, to_index, config, true);
  report.initial_cost = evaluation.cost;
  double damping = config.initial_relative_damping;
  bool converged = false;
  for (std::size_t iteration = 0U; iteration < config.maximum_iterations; ++iteration) {
    report.iterations = iteration + 1U;
    if (evaluation.forward_correspondences < config.minimum_correspondences_each_direction ||
        evaluation.reverse_correspondences < config.minimum_correspondences_each_direction ||
        !evaluation.hessian.allFinite() || !evaluation.gradient.allFinite() ||
        !std::isfinite(evaluation.cost)) {
      break;
    }
    const double diagonal_scale =
        std::max(1.0, evaluation.hessian.diagonal().cwiseAbs().maxCoeff());
    bool accepted = false;
    for (std::size_t trial = 0U; trial < config.maximum_damping_trials; ++trial) {
      Matrix6 conditioned = evaluation.hessian;
      conditioned.diagonal().array() += damping * diagonal_scale;
      const Eigen::LDLT<Matrix6> solve(conditioned);
      if (solve.info() != Eigen::Success) {
        damping *= config.damping_increase;
        ++report.rejected_steps;
        continue;
      }
      const Vector6 delta = solve.solve(-evaluation.gradient);
      if (!delta.allFinite()) {
        return Result::failure(
            verifierError(LidarLoopVerifierErrorCode::NumericalFailure,
                          "LiDAR GICP damping solve generated a non-finite update"));
      }
      const double translation_step = delta.head<3>().norm();
      const double rotation_step = delta.tail<3>().norm();
      if (translation_step <= config.translation_convergence_m &&
          rotation_step <= config.rotation_convergence_rad) {
        converged = true;
        accepted = true;
        break;
      }
      if (translation_step > config.maximum_iteration_translation_step_m ||
          rotation_step > config.maximum_iteration_rotation_step_rad) {
        damping *= config.damping_increase;
        ++report.rejected_steps;
        continue;
      }
      const core::Pose3d candidate = estimate * core::Pose3d::exp(delta);
      GicpEvaluation candidate_evaluation =
          evaluateGicp(candidate, from_proxy, to_proxy, from_index, to_index, config, true);
      const double decrease_tolerance = 1.0e-12 * std::max(1.0, std::abs(evaluation.cost));
      if (std::isfinite(candidate_evaluation.cost) &&
          candidate_evaluation.cost < evaluation.cost - decrease_tolerance) {
        estimate = candidate;
        evaluation = std::move(candidate_evaluation);
        damping = std::max(1.0e-12, damping * config.damping_decrease);
        ++report.accepted_steps;
        accepted = true;
        if (translation_step <= config.translation_convergence_m &&
            rotation_step <= config.rotation_convergence_rad) {
          converged = true;
        }
        break;
      }
      damping *= config.damping_increase;
      ++report.rejected_steps;
    }
    if (converged || !accepted) {
      break;
    }
  }

  // This fresh evaluation is binding for every gate and for final information.
  // No association or Hessian from an earlier optimizer iteration is reused.
  const GicpEvaluation final =
      evaluateGicp(estimate, from_proxy, to_proxy, from_index, to_index, config, false);
  report.final_cost = final.cost;
  report.forward_correspondences = final.forward_correspondences;
  report.reverse_correspondences = final.reverse_correspondences;
  report.forward_overlap = static_cast<double>(final.forward_correspondences) /
                           static_cast<double>(to_proxy.points.size());
  report.reverse_overlap = static_cast<double>(final.reverse_correspondences) /
                           static_cast<double>(from_proxy.points.size());
  std::set<std::pair<std::size_t, std::size_t>> unique_pairs;
  std::vector<std::size_t> forward_support;
  std::vector<std::size_t> reverse_support;
  std::vector<double> residuals;
  std::size_t consistent_normals = 0U;
  for (const GicpPair& pair : final.pairs) {
    unique_pairs.emplace(pair.from_index, pair.to_index);
    residuals.push_back(pair.euclidean_residual_m);
    if (pair.forward) {
      forward_support.push_back(pair.to_index);
    } else {
      reverse_support.push_back(pair.from_index);
    }
    if (pair.normal_cosine >= config.minimum_pair_normal_cosine) {
      ++consistent_normals;
    }
  }
  report.unique_correspondences = unique_pairs.size();
  report.normal_consistent_correspondences = consistent_normals;
  report.forward_support_cells =
      spatialSupportCells(to_proxy, forward_support, config.support_grid_dimension);
  report.reverse_support_cells =
      spatialSupportCells(from_proxy, reverse_support, config.support_grid_dimension);
  report.residual_median_m = quantile(residuals, 0.50);
  report.residual_upper_quantile_m = quantile(residuals, config.residual_upper_quantile);
  report.normal_consistency = final.pairs.empty() ? 0.0
                                                  : static_cast<double>(consistent_normals) /
                                                        static_cast<double>(final.pairs.size());
  report.dynamic_fraction = std::nullopt;
  report.dynamic_label_status = DynamicProxyLabelStatus::UnavailableNoSemanticLabel;

  const Vector6 seed_correction = (request.seed.T_from_to_seed.inverse() * estimate).log();
  report.seed_translation_correction_m = seed_correction.head<3>().norm();
  report.seed_rotation_correction_rad = seed_correction.tail<3>().norm();
  report.final_gravity_tilt_rad = std::acos(
      std::clamp(estimate.so3().matrix().col(2).dot(Eigen::Vector3d::UnitZ()), -1.0, 1.0));
  const FinalInformation information =
      finalRankAwareInformation(estimate, from_proxy, to_proxy, final.pairs, config);
  report.information = information.information;
  report.raw_information_eigenvalues = information.raw_eigenvalues;
  report.calibrated_normal_residual_sigma_m = information.calibrated_sigma_m;

  const auto minimum_gate = [&](LidarLoopGate gate, double measured, double threshold) {
    report.gates.push_back(LidarLoopGateResult{
        gate, std::isfinite(measured) && measured >= threshold, measured, threshold});
  };
  const auto maximum_gate = [&](LidarLoopGate gate, double measured, double threshold) {
    report.gates.push_back(LidarLoopGateResult{
        gate, std::isfinite(measured) && measured <= threshold, measured, threshold});
  };
  minimum_gate(LidarLoopGate::BidirectionalOverlap,
               std::min(report.forward_overlap, report.reverse_overlap),
               config.minimum_bidirectional_overlap);
  minimum_gate(
      LidarLoopGate::SpatialSupport,
      static_cast<double>(std::min(report.forward_support_cells, report.reverse_support_cells)),
      static_cast<double>(config.minimum_support_cells_each_direction));
  maximum_gate(LidarLoopGate::ResidualMedian, report.residual_median_m,
               config.maximum_residual_median_m);
  maximum_gate(LidarLoopGate::ResidualUpperQuantile, report.residual_upper_quantile_m,
               config.maximum_residual_quantile_m);
  minimum_gate(LidarLoopGate::NormalConsistency, report.normal_consistency,
               config.minimum_normal_consistency);
  const double normalized_seed_correction =
      std::max(report.seed_translation_correction_m / config.maximum_seed_translation_correction_m,
               report.seed_rotation_correction_rad / config.maximum_seed_rotation_correction_rad);
  maximum_gate(LidarLoopGate::SeedCorrection, normalized_seed_correction, 1.0);
  maximum_gate(LidarLoopGate::GravityAlignment, report.final_gravity_tilt_rad,
               config.maximum_gravity_tilt_rad);
  minimum_gate(LidarLoopGate::InformationRank, static_cast<double>(report.information.rank),
               static_cast<double>(config.minimum_information_rank));

  VerifiedLidarLoop result(request.seed, report);
  const bool gates_pass = std::all_of(result.report.gates.begin(), result.report.gates.end(),
                                      [](const LidarLoopGateResult& gate) { return gate.passed; });
  if (!converged) {
    result.report.disposition = LidarLoopVerificationDisposition::RejectedDidNotConverge;
    return Result::success(std::move(result));
  }
  if (!gates_pass) {
    const bool rank_failed =
        std::any_of(result.report.gates.begin(), result.report.gates.end(),
                    [](const LidarLoopGateResult& gate) {
                      return gate.gate == LidarLoopGate::InformationRank && !gate.passed;
                    });
    result.report.disposition = rank_failed
                                    ? LidarLoopVerificationDisposition::RejectedDegenerate
                                    : LidarLoopVerificationDisposition::RejectedGeometricGates;
    return Result::success(std::move(result));
  }

  result.report.disposition = LidarLoopVerificationDisposition::Accepted;
  result.measurement.emplace(LoopMeasurement{
      request.header, request.proposal, LoopModality::Lidar, request.seed.from, request.seed.to,
      request.calibration_epochs, estimate, result.report.information, request.lineage});
  return Result::success(std::move(result));
}

}  // namespace meridian::global
