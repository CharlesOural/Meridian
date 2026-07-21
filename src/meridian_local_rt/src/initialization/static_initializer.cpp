#include "meridian/local_rt/initialization/static_initializer.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace meridian::local_rt::initialization {
namespace {

Eigen::Vector3d eigen(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

core::Vec3d coreVector(const Eigen::Vector3d& vector) {
  return {.x = vector.x(), .y = vector.y(), .z = vector.z()};
}

Eigen::Quaterniond eigen(const core::Quaterniond& quaternion) {
  return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

core::Pose3d corePose(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation) {
  const Eigen::Quaterniond quaternion = Eigen::Quaterniond(rotation).normalized();
  return core::Pose3d(coreVector(translation), core::Quaterniond(quaternion.w(), quaternion.x(),
                                                                 quaternion.y(), quaternion.z()));
}

Eigen::Matrix3d removeYaw(const Eigen::Matrix3d& rotation) {
  const double projected_x_norm = std::hypot(rotation(0, 0), rotation(1, 0));
  double yaw = 0.0;
  if (projected_x_norm > 1.0e-12) {
    yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  } else {
    // At a +/-90 degree pitch the base x-axis has no horizontal heading.
    // Use the y-axis to choose the otherwise unobservable yaw gauge.
    yaw = std::atan2(-rotation(0, 1), rotation(1, 1));
  }
  return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * rotation;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("median requires at least one value");
  }
  const std::size_t middle = values.size() / 2U;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                   values.end());
  const double upper = values[middle];
  if (values.size() % 2U != 0U) {
    return upper;
  }
  const double lower =
      *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
  return 0.5 * (lower + upper);
}

Eigen::Vector3d componentMedian(const std::vector<Eigen::Vector3d>& values) {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  x.reserve(values.size());
  y.reserve(values.size());
  z.reserve(values.size());
  for (const Eigen::Vector3d& value : values) {
    x.push_back(value.x());
    y.push_back(value.y());
    z.push_back(value.z());
  }
  return {median(std::move(x)), median(std::move(y)), median(std::move(z))};
}

struct BlockMeans final {
  Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d specific_force{Eigen::Vector3d::Zero()};
};

bool positiveFinite(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

bool sampleIsFinite(const core::ImuSample& sample) noexcept {
  return sample.angularVelocityRadS().isFinite() && sample.specificForceMS2().isFinite();
}

bool sampleIsSaturated(const core::ImuSample& sample,
                       const StaticInitializerOptions& options) noexcept {
  const Eigen::Vector3d angular_velocity = eigen(sample.angularVelocityRadS());
  const Eigen::Vector3d specific_force = eigen(sample.specificForceMS2());
  return angular_velocity.cwiseAbs().maxCoeff() >= options.gyroscope_saturation_rad_s ||
         specific_force.cwiseAbs().maxCoeff() >= options.accelerometer_saturation_m_s2;
}

void validate(const StaticInitializerOptions& options) {
  if (options.window_duration_ns <= 0 || options.block_duration_ns <= 0 ||
      options.block_duration_ns > options.window_duration_ns) {
    throw std::invalid_argument("static initialization durations are invalid");
  }
  if (options.minimum_samples < 2U || options.minimum_blocks < 2U) {
    throw std::invalid_argument("static initialization support limits are invalid");
  }
  if (options.maximum_sample_gap_ns <= 0 || !positiveFinite(options.gravity_m_s2) ||
      !positiveFinite(options.gyroscope_saturation_rad_s) ||
      !positiveFinite(options.accelerometer_saturation_m_s2) ||
      !positiveFinite(options.maximum_mean_angular_rate_rad_s) ||
      !positiveFinite(options.maximum_block_angular_dispersion_rad_s) ||
      !positiveFinite(options.maximum_specific_force_norm_error_m_s2) ||
      !positiveFinite(options.maximum_block_direction_dispersion_rad)) {
    throw std::invalid_argument("static initialization gates must be finite and positive");
  }
}

StaticInitializationUpdate collecting(std::string reason, std::size_t sample_count) {
  StaticInitializationUpdate update;
  update.reason = std::move(reason);
  update.quality.imu_sample_count = static_cast<std::uint64_t>(sample_count);
  return update;
}

}  // namespace

StaticInitializer::StaticInitializer(StaticInitializerOptions options)
    : options_(std::move(options)) {
  validate(options_);
}

void StaticInitializer::reset() noexcept {
  samples_.clear();
  accepted_result_.reset();
}

StaticInitializationUpdate StaticInitializer::add(const core::ImuSample& sample) {
  if (accepted_result_.has_value()) {
    return StaticInitializationUpdate{
        .status = core::InitializationStatus::kAccepted,
        .reason = "static initialization was already accepted",
        .quality = accepted_result_->quality(),
        .result = accepted_result_,
    };
  }

  if (!sampleIsFinite(sample)) {
    samples_.clear();
    return collecting("non-finite IMU data reset the static window", samples_.size());
  }
  if (sampleIsSaturated(sample, options_)) {
    samples_.clear();
    return collecting("IMU saturation reset the static window", samples_.size());
  }

  const core::TimeNs time = sample.header().measurementTime();
  if (!samples_.empty() && time <= samples_.back().header().measurementTime()) {
    samples_.clear();
    samples_.push_back(sample);
    return collecting("non-monotonic IMU time reset the static window", samples_.size());
  }
  if (!samples_.empty()) {
    const auto gap =
        core::TimeNs::checkedDifference(time, samples_.back().header().measurementTime());
    if (!gap.has_value() || *gap > options_.maximum_sample_gap_ns) {
      samples_.clear();
      samples_.push_back(sample);
      return collecting("an IMU gap reset the static window", samples_.size());
    }
  }
  samples_.push_back(sample);

  while (samples_.size() > 1U) {
    const auto second_age =
        core::TimeNs::checkedDifference(time, samples_[1U].header().measurementTime());
    if (!second_age.has_value()) {
      samples_.clear();
      samples_.push_back(sample);
      return collecting("invalid IMU time span reset the static window", samples_.size());
    }
    if (*second_age < options_.window_duration_ns) {
      break;
    }
    samples_.pop_front();
  }

  if (samples_.size() < options_.minimum_samples) {
    return collecting("waiting for the configured static IMU sample count", samples_.size());
  }
  const auto duration =
      core::TimeNs::checkedDifference(time, samples_.front().header().measurementTime());
  if (!duration.has_value() || *duration < options_.window_duration_ns) {
    return collecting("waiting for the configured static support duration", samples_.size());
  }

  const core::TimeNs begin = samples_.front().header().measurementTime();
  const std::size_t block_count =
      static_cast<std::size_t>(*duration / options_.block_duration_ns) +
      static_cast<std::size_t>(*duration % options_.block_duration_ns != 0);
  std::vector<BlockMeans> blocks;
  std::vector<std::size_t> block_counts;
  blocks.resize(block_count);
  block_counts.resize(block_count);
  for (std::size_t index = 0; index < samples_.size(); ++index) {
    const core::ImuSample& item = samples_[index];
    const Eigen::Vector3d angular_velocity = eigen(item.angularVelocityRadS());
    const Eigen::Vector3d specific_force = eigen(item.specificForceMS2());

    const auto offset = core::TimeNs::checkedDifference(item.header().measurementTime(), begin);
    if (!offset.has_value() || *offset < 0) {
      return collecting("invalid static-window time support", samples_.size());
    }
    // The support end is a real sample and therefore inclusive for these
    // statistics. Clamp an exact final block boundary into the last block
    // instead of creating a one-sample (N + 1)th block.
    const std::size_t block_index =
        std::min(static_cast<std::size_t>(*offset / options_.block_duration_ns), block_count - 1U);
    const double next_count = static_cast<double>(block_counts[block_index] + 1U);
    blocks[block_index].angular_velocity +=
        (angular_velocity - blocks[block_index].angular_velocity) / next_count;
    blocks[block_index].specific_force +=
        (specific_force - blocks[block_index].specific_force) / next_count;
    ++block_counts[block_index];
  }

  std::vector<Eigen::Vector3d> block_gyro;
  std::vector<Eigen::Vector3d> block_accel;
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    if (block_counts[index] == 0U) {
      continue;
    }
    block_gyro.push_back(blocks[index].angular_velocity);
    block_accel.push_back(blocks[index].specific_force -
                          eigen(options_.calibrated_bias_prior.accelerometerMS2()));
  }
  if (block_gyro.size() < options_.minimum_blocks) {
    return collecting("waiting for enough populated static statistic blocks", samples_.size());
  }

  const Eigen::Vector3d mean_gyro = componentMedian(block_gyro);
  const Eigen::Vector3d calibrated_specific_force = componentMedian(block_accel);
  std::vector<double> gyro_deviations;
  std::vector<double> direction_deviations;
  std::vector<double> gravity_norm_errors;
  gyro_deviations.reserve(block_gyro.size());
  direction_deviations.reserve(block_accel.size());
  gravity_norm_errors.reserve(block_accel.size());
  const double raw_gravity_magnitude = calibrated_specific_force.norm();
  if (!std::isfinite(raw_gravity_magnitude) || raw_gravity_magnitude <= 0.0) {
    return collecting("specific-force norm rejects the static assertion", samples_.size());
  }
  const Eigen::Vector3d accel_direction = calibrated_specific_force / raw_gravity_magnitude;
  for (std::size_t index = 0; index < block_gyro.size(); ++index) {
    gyro_deviations.push_back((block_gyro[index] - mean_gyro).norm());
    const double block_accel_norm = block_accel[index].norm();
    if (!std::isfinite(block_accel_norm) || block_accel_norm <= 0.0) {
      return collecting("a block has invalid calibrated specific force", samples_.size());
    }
    const Eigen::Vector3d direction = block_accel[index] / block_accel_norm;
    direction_deviations.push_back(
        std::acos(std::clamp(direction.dot(accel_direction), -1.0, 1.0)));
    gravity_norm_errors.push_back(std::abs(block_accel_norm - options_.gravity_m_s2));
  }

  double gyro_dispersion = median(std::move(gyro_deviations));
  double direction_dispersion = median(std::move(direction_deviations));
  double gravity_norm_error = median(std::move(gravity_norm_errors));

  // Block means reject slow motion robustly, but alone they can alias an
  // oscillation whose period happens to divide a block. Include robust
  // sample-level dispersions so the static decision is not block-phase
  // dependent while still tolerating isolated IMU outliers.
  std::vector<double> sample_gyro_deviations;
  std::vector<double> sample_direction_deviations;
  std::vector<double> sample_gravity_norm_errors;
  sample_gyro_deviations.reserve(samples_.size());
  sample_direction_deviations.reserve(samples_.size());
  sample_gravity_norm_errors.reserve(samples_.size());
  for (const core::ImuSample& item : samples_) {
    const Eigen::Vector3d angular_velocity = eigen(item.angularVelocityRadS());
    const Eigen::Vector3d calibrated_acceleration =
        eigen(item.specificForceMS2()) - eigen(options_.calibrated_bias_prior.accelerometerMS2());
    const double acceleration_norm = calibrated_acceleration.norm();
    if (!std::isfinite(acceleration_norm) || acceleration_norm <= 0.0) {
      return collecting("a sample has invalid calibrated specific force", samples_.size());
    }
    sample_gyro_deviations.push_back((angular_velocity - mean_gyro).norm());
    sample_direction_deviations.push_back(std::acos(
        std::clamp((calibrated_acceleration / acceleration_norm).dot(accel_direction), -1.0, 1.0)));
    sample_gravity_norm_errors.push_back(std::abs(acceleration_norm - options_.gravity_m_s2));
  }
  gyro_dispersion = std::max(gyro_dispersion, median(std::move(sample_gyro_deviations)));
  direction_dispersion =
      std::max(direction_dispersion, median(std::move(sample_direction_deviations)));
  gravity_norm_error = std::max(gravity_norm_error, median(std::move(sample_gravity_norm_errors)));
  gravity_norm_error =
      std::max(gravity_norm_error, std::abs(raw_gravity_magnitude - options_.gravity_m_s2));
  if (!mean_gyro.allFinite() || !std::isfinite(gyro_dispersion) ||
      mean_gyro.norm() > options_.maximum_mean_angular_rate_rad_s) {
    return collecting("mean angular rate rejects the static assertion", samples_.size());
  }
  if (gyro_dispersion > options_.maximum_block_angular_dispersion_rad_s) {
    return collecting("angular-rate dispersion rejects the static assertion", samples_.size());
  }
  if (gravity_norm_error > options_.maximum_specific_force_norm_error_m_s2) {
    return collecting("specific-force norm rejects the static assertion", samples_.size());
  }
  if (direction_dispersion > options_.maximum_block_direction_dispersion_rad) {
    return collecting("specific-force direction rejects the static assertion", samples_.size());
  }

  const Eigen::Quaterniond odom_from_imu_alignment = Eigen::Quaterniond::FromTwoVectors(
      calibrated_specific_force, Eigen::Vector3d::UnitZ() * options_.gravity_m_s2);
  const Eigen::Matrix3d base_from_imu_rotation = eigen(options_.base_from_imu.rotation()).matrix();
  const Eigen::Matrix3d raw_odom_from_base =
      odom_from_imu_alignment.toRotationMatrix() * base_from_imu_rotation.transpose();
  const Eigen::Matrix3d odom_from_base_rotation = removeYaw(raw_odom_from_base);
  const Eigen::Matrix3d odom_from_imu_rotation = odom_from_base_rotation * base_from_imu_rotation;
  const Eigen::Vector3d odom_from_imu_translation =
      odom_from_base_rotation * eigen(options_.base_from_imu.translation());

  const auto support_end = core::TimeNs::checkedAdd(time, 1);
  if (!support_end.has_value()) {
    return collecting("static anchor cannot form half-open support", samples_.size());
  }
  core::InitializationQuality quality;
  quality.imu_sample_count = static_cast<std::uint64_t>(samples_.size());
  quality.raw_gravity_magnitude_m_s2 = raw_gravity_magnitude;
  quality.all_required_gates_passed = true;

  const core::ImuBias bias(coreVector(mean_gyro),
                           options_.calibrated_bias_prior.accelerometerMS2());
  core::NavigationState seed(core::StateId(1U), time,
                             corePose(odom_from_imu_rotation, odom_from_imu_translation), {}, bias);
  core::InitializationResult result(core::InitializationMode::kStatic, time, std::move(seed),
                                    core::TimeRange(begin, *support_end), quality);
  accepted_result_ = result;
  return StaticInitializationUpdate{
      .status = core::InitializationStatus::kAccepted,
      .reason = "static zero-motion assertion and all IMU gates accepted",
      .quality = quality,
      .result = std::move(result),
  };
}

}  // namespace meridian::local_rt::initialization
