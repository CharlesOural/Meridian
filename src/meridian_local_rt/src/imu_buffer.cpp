#include "meridian/local_rt/imu_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace meridian::local_rt {
namespace {

constexpr double kNanosecondsPerSecond = 1.0e9;

Eigen::Vector3d eigen(const core::Vec3d& vector) {
  return {vector.x, vector.y, vector.z};
}

bool finite(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

std::int64_t durationNs(core::TimeNs end, core::TimeNs begin) {
  const std::optional<std::int64_t> duration = core::TimeNs::checkedDifference(end, begin);
  if (!duration.has_value()) {
    throw std::overflow_error("IMU time difference overflows int64 nanoseconds");
  }
  return *duration;
}

}  // namespace

ImuIntegrationSegment::ImuIntegrationSegment(core::TimeRange support,
                                             Eigen::Vector3d angular_velocity_rad_s,
                                             Eigen::Vector3d specific_force_m_s2)
    : support_(std::move(support)),
      angular_velocity_rad_s_(std::move(angular_velocity_rad_s)),
      specific_force_m_s2_(std::move(specific_force_m_s2)) {
  if (support_.empty()) {
    throw std::invalid_argument("an IMU integration segment requires non-empty support");
  }
  if (!finite(angular_velocity_rad_s_) || !finite(specific_force_m_s2_)) {
    throw std::invalid_argument("an IMU integration segment requires finite measurements");
  }
}

double ImuIntegrationSegment::durationSeconds() const noexcept {
  const std::optional<std::int64_t> duration = support_.durationNs();
  return duration.has_value() ? static_cast<double>(*duration) / kNanosecondsPerSecond : 0.0;
}

ImuInterval::ImuInterval(core::TimeRange support, ImuIntegrationSegments segments,
                         std::size_t source_sample_count)
    : support_(std::move(support)),
      segments_(std::move(segments)),
      source_sample_count_(source_sample_count) {
  if (support_.empty() || segments_.empty()) {
    throw std::invalid_argument("an IMU interval requires non-empty exact support");
  }
  if (segments_.front().support().begin() != support_.begin() ||
      segments_.back().support().end() != support_.end()) {
    throw std::invalid_argument("IMU segments do not span the requested support");
  }
  for (std::size_t index = 1U; index < segments_.size(); ++index) {
    if (segments_[index - 1U].support().end() != segments_[index].support().begin()) {
      throw std::invalid_argument("IMU segments are not contiguous");
    }
  }
}

double ImuInterval::durationSeconds() const noexcept {
  const std::optional<std::int64_t> duration = support_.durationNs();
  return duration.has_value() ? static_cast<double>(*duration) / kNanosecondsPerSecond : 0.0;
}

ImuIntervalResult::ImuIntervalResult(ImuInterval interval) : result_(std::move(interval)) {}

ImuIntervalResult::ImuIntervalResult(ImuIntervalFailure failure) : result_(std::move(failure)) {}

bool ImuIntervalResult::ok() const noexcept {
  return std::holds_alternative<ImuInterval>(result_);
}

const ImuInterval* ImuIntervalResult::value() const noexcept {
  return std::get_if<ImuInterval>(&result_);
}

const ImuIntervalFailure* ImuIntervalResult::error() const noexcept {
  return std::get_if<ImuIntervalFailure>(&result_);
}

ImuBuffer::ImuBuffer(ImuBufferConfig config) : config_(config) {
  if (config_.capacity == 0U || config_.maximum_gap.count() <= 0) {
    throw std::invalid_argument("IMU buffer capacity and maximum gap must be positive");
  }
}

ImuInsertResult ImuBuffer::insert(const core::ImuSample& sample) {
  const core::TimeNs time = sample.header().measurementTime();
  const Eigen::Vector3d angular_velocity = eigen(sample.angularVelocityRadS());
  const Eigen::Vector3d specific_force = eigen(sample.specificForceMS2());
  if (!finite(angular_velocity) || !finite(specific_force)) {
    return {.status = ImuInsertStatus::kNonFinite};
  }
  if (!samples_.empty()) {
    if (time == samples_.back().time) {
      return {.status = ImuInsertStatus::kDuplicateTimestamp};
    }
    if (time < samples_.back().time) {
      return {.status = ImuInsertStatus::kOutOfOrder};
    }
  }

  samples_.push_back({time, angular_velocity, specific_force});
  std::size_t evicted_samples = 0U;
  while (samples_.size() > config_.capacity) {
    samples_.pop_front();
    ++evicted_samples;
  }
  return {.status = ImuInsertStatus::kInserted, .evicted_samples = evicted_samples};
}

std::optional<core::TimeNs> ImuBuffer::latestTime() const noexcept {
  return samples_.empty() ? std::nullopt : std::optional<core::TimeNs>(samples_.back().time);
}

ImuIntervalResult ImuBuffer::interval(core::TimeNs begin, core::TimeNs end) const {
  if (end <= begin) {
    return ImuIntervalResult(
        ImuIntervalFailure{.code = ImuIntervalErrorCode::kInvalidRange,
                           .message = "an IMU integration interval must have positive duration",
                           .offending_gap = std::nullopt});
  }
  if (samples_.empty()) {
    return ImuIntervalResult(ImuIntervalFailure{.code = ImuIntervalErrorCode::kEmptyBuffer,
                                                .message = "the IMU buffer is empty",
                                                .offending_gap = std::nullopt});
  }

  const auto atOrAfter = [this](core::TimeNs time) {
    return std::lower_bound(
        samples_.begin(), samples_.end(), time,
        [](const BufferedSample& sample, core::TimeNs target) { return sample.time < target; });
  };
  const auto begin_right = atOrAfter(begin);
  if (begin_right == samples_.end() ||
      (begin_right->time != begin && begin_right == samples_.begin())) {
    return ImuIntervalResult(
        ImuIntervalFailure{.code = ImuIntervalErrorCode::kBeginNotBracketed,
                           .message = "the IMU buffer does not bracket the interval begin",
                           .offending_gap = std::nullopt});
  }
  const auto end_right = atOrAfter(end);
  if (end_right == samples_.end()) {
    return ImuIntervalResult(
        ImuIntervalFailure{.code = ImuIntervalErrorCode::kEndNotBracketed,
                           .message = "the IMU buffer does not bracket the interval end",
                           .offending_gap = std::nullopt});
  }

  const auto begin_left = begin_right->time == begin ? begin_right : std::prev(begin_right);
  const auto end_left = end_right->time == end ? end_right : std::prev(end_right);
  for (auto left = begin_left; left != end_right; ++left) {
    const auto right = std::next(left);
    const std::int64_t gap_ns = durationNs(right->time, left->time);
    if (gap_ns > config_.maximum_gap.count()) {
      return ImuIntervalResult(
          ImuIntervalFailure{.code = ImuIntervalErrorCode::kSourceGapTooLarge,
                             .message = "a source IMU gap exceeds the configured maximum",
                             .offending_gap = core::TimeRange(left->time, right->time)});
    }
  }

  const auto interpolate = [](const BufferedSample& left, const BufferedSample& right,
                              core::TimeNs time) {
    if (time == left.time) {
      return BufferedSample{time, left.angular_velocity_rad_s, left.specific_force_m_s2};
    }
    if (time == right.time) {
      return BufferedSample{time, right.angular_velocity_rad_s, right.specific_force_m_s2};
    }
    const double alpha = static_cast<double>(durationNs(time, left.time)) /
                         static_cast<double>(durationNs(right.time, left.time));
    return BufferedSample{
        time,
        left.angular_velocity_rad_s +
            alpha * (right.angular_velocity_rad_s - left.angular_velocity_rad_s),
        left.specific_force_m_s2 + alpha * (right.specific_force_m_s2 - left.specific_force_m_s2)};
  };

  std::vector<BufferedSample, Eigen::aligned_allocator<BufferedSample>> knots;
  knots.reserve(static_cast<std::size_t>(std::distance(begin_left, end_right)) + 2U);
  knots.push_back(interpolate(*begin_left, *begin_right, begin));
  for (auto item = std::next(begin_left); item != end_right; ++item) {
    if (item->time > begin && item->time < end) {
      knots.push_back(*item);
    }
  }
  knots.push_back(interpolate(*end_left, *end_right, end));

  ImuIntegrationSegments segments;
  segments.reserve(knots.size() - 1U);
  for (std::size_t index = 1U; index < knots.size(); ++index) {
    segments.emplace_back(
        core::TimeRange(knots[index - 1U].time, knots[index].time),
        0.5 * (knots[index - 1U].angular_velocity_rad_s + knots[index].angular_velocity_rad_s),
        0.5 * (knots[index - 1U].specific_force_m_s2 + knots[index].specific_force_m_s2));
  }

  const std::size_t source_sample_count =
      static_cast<std::size_t>(std::distance(begin_left, end_right)) + 1U;
  return ImuIntervalResult(
      ImuInterval(core::TimeRange(begin, end), std::move(segments), source_sample_count));
}

}  // namespace meridian::local_rt
