#include "meridian/global/visual_place.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace meridian::global {
namespace {

constexpr double kTiny = 1.0e-12;

[[nodiscard]] bool nonzeroHash(const core::ContentHash& hash) noexcept {
  return std::any_of(hash.begin(), hash.end(), [](std::uint8_t byte) { return byte != 0U; });
}

[[nodiscard]] bool sameSubmap(const core::SubmapRef& lhs,
                              const core::SubmapRef& rhs) noexcept {
  return lhs == rhs;
}

[[nodiscard]] bool sameSubmapObject(const core::SubmapRef& lhs,
                                    const core::SubmapRef& rhs) noexcept {
  return lhs.session == rhs.session && lhs.odom_epoch == rhs.odom_epoch && lhs.id == rhs.id;
}

[[nodiscard]] bool containsCalibration(std::span<const core::CalibrationEpoch> epochs,
                                       core::CalibrationEpoch calibration) noexcept {
  return std::binary_search(epochs.begin(), epochs.end(), calibration);
}

[[nodiscard]] bool containsSubmap(std::span<const core::SubmapRef> ids,
                                  const core::SubmapRef& id) noexcept {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

[[nodiscard]] std::uint32_t hamming(const VisualBriskDescriptor& lhs,
                                    const VisualBriskDescriptor& rhs) noexcept {
  std::uint32_t distance = 0U;
  for (std::size_t index = 0U; index < lhs.size(); ++index) {
    distance += static_cast<std::uint32_t>(
        std::popcount(static_cast<unsigned int>(lhs[index] ^ rhs[index])));
  }
  return distance;
}

[[nodiscard]] std::int64_t absoluteNanoseconds(core::FusionTime lhs,
                                               core::FusionTime rhs) noexcept {
  const std::uint64_t lhs_ordered =
      std::bit_cast<std::uint64_t>(lhs.nanoseconds) ^ (std::uint64_t{1} << 63U);
  const std::uint64_t rhs_ordered =
      std::bit_cast<std::uint64_t>(rhs.nanoseconds) ^ (std::uint64_t{1} << 63U);
  const std::uint64_t difference =
      lhs_ordered >= rhs_ordered ? lhs_ordered - rhs_ordered : rhs_ordered - lhs_ordered;
  if (difference > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(difference);
}

[[nodiscard]] core::FusionTime saturatingAdd(core::FusionTime time,
                                             core::Duration duration) noexcept {
  if (duration.nanoseconds > 0 &&
      time.nanoseconds > std::numeric_limits<std::int64_t>::max() - duration.nanoseconds) {
    return {std::numeric_limits<std::int64_t>::max()};
  }
  if (duration.nanoseconds < 0 &&
      time.nanoseconds < std::numeric_limits<std::int64_t>::min() - duration.nanoseconds) {
    return {std::numeric_limits<std::int64_t>::min()};
  }
  return {time.nanoseconds + duration.nanoseconds};
}

[[nodiscard]] bool validPose(const core::Pose3d& pose) noexcept {
  return pose.matrix().allFinite() && std::abs(pose.so3().matrix().determinant() - 1.0) < 1.0e-6 &&
         (pose.so3().matrix().transpose() * pose.so3().matrix() - Eigen::Matrix3d::Identity())
                 .norm() < 1.0e-6;
}

[[nodiscard]] bool validRecordHeader(
    const core::RecordHeader& header, core::FusionTime now,
    std::span<const core::CalibrationEpoch> calibration_epochs) noexcept {
  if (header.schema_version != 1U || !header.trace.valid() || !header.producer.valid() ||
      !header.session.valid() || !header.config.valid() || header.created_at > now) {
    return false;
  }
  if (!header.direct_calibration.has_value()) {
    return true;
  }
  return header.direct_calibration->valid() &&
         std::find(calibration_epochs.begin(), calibration_epochs.end(),
                   *header.direct_calibration) != calibration_epochs.end();
}

[[nodiscard]] bool validPointCovariance(const Eigen::Matrix3d& covariance) noexcept {
  if (!covariance.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, covariance.norm());
  if ((covariance - covariance.transpose()).norm() > 1.0e-10 * scale) {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(0.5 * (covariance + covariance.transpose()),
                                                        Eigen::EigenvaluesOnly);
  return solver.info() == Eigen::Success && solver.eigenvalues().minCoeff() >= -1.0e-12 * scale;
}

[[nodiscard]] double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  return values.size() % 2U == 1U ? values[middle] : 0.5 * (values[middle - 1U] + values[middle]);
}

[[nodiscard]] Eigen::Vector3d tangentAxis(const Eigen::Vector3d& bearing) noexcept {
  Eigen::Vector3d axis =
      std::abs(bearing.z()) < 0.8 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
  axis -= bearing * bearing.dot(axis);
  const double norm = axis.norm();
  if (norm > kTiny) {
    return axis / norm;
  }
  return Eigen::Vector3d::UnitY();
}

struct BowEntry {
  std::uint32_t word{};
  double weight{};
};
using BowVector = std::vector<BowEntry>;

[[nodiscard]] bool sameDescriptorSet(std::span<const VisualBriskDescriptor> lhs,
                                     std::span<const VisualBriskDescriptor> rhs) noexcept {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

[[nodiscard]] bool sameRetrievalRecord(const VisualRetrievalSubmap& lhs,
                                       const VisualRetrievalSubmap& rhs) noexcept {
  if (!sameSubmap(lhs.submap, rhs.submap) || lhs.model_revision != rhs.model_revision ||
      lhs.model_checksum != rhs.model_checksum || lhs.config_revision != rhs.config_revision ||
      lhs.overlap_or_adjacent != rhs.overlap_or_adjacent ||
      lhs.keyframes.size() != rhs.keyframes.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < lhs.keyframes.size(); ++index) {
    const auto& a = lhs.keyframes[index];
    const auto& b = rhs.keyframes[index];
    if (a.id != b.id || a.time != b.time || a.covisibility_group != b.covisibility_group ||
        !sameDescriptorSet(a.descriptors, b.descriptors)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool validIndexConfig(const VisualPlaceIndexConfig& config) noexcept {
  return config.revision.valid() && config.maximum_submaps > 0U && config.maximum_keyframes > 0U &&
         config.maximum_descriptors > 0U && config.maximum_words > 1U &&
         config.maximum_descriptors_per_keyframe > 0U && config.maximum_keyframes_per_submap > 0U &&
         config.maximum_frame_hits > 0U && config.frame_top_k > 0U && config.submap_top_k > 0U &&
         config.minimum_query_keyframe_votes > 0U && config.minimum_candidate_keyframe_votes > 0U &&
         std::isfinite(config.minimum_frame_score) && config.minimum_frame_score >= 0.0 &&
         config.minimum_frame_score <= 1.0 && config.temporal_vote_window.nanoseconds >= 0 &&
         config.candidate_ttl.nanoseconds > 0;
}

[[nodiscard]] bool validVerifierConfig(const VisualGeometricVerifierConfig& config) noexcept {
  return config.maximum_landmarks > 0U && config.maximum_query_features > 0U &&
         config.maximum_descriptors_per_landmark > 0U && config.maximum_correspondences >= 4U &&
         config.maximum_ransac_hypotheses > 0U && config.maximum_refinement_iterations > 0U &&
         config.maximum_hamming_distance <= 512U && std::isfinite(config.maximum_ratio) &&
         config.maximum_ratio > 0.0 && config.maximum_ratio < 1.0 &&
         std::isfinite(config.ransac_angular_threshold_rad) &&
         config.ransac_angular_threshold_rad > 0.0 && config.minimum_ransac_inliers >= 4U &&
         std::isfinite(config.minimum_ransac_inlier_ratio) &&
         config.minimum_ransac_inlier_ratio > 0.0 && config.minimum_ransac_inlier_ratio <= 1.0 &&
         config.grid_columns > 0U && config.grid_rows > 0U &&
         std::isfinite(config.minimum_grid_coverage) && config.minimum_grid_coverage > 0.0 &&
         config.minimum_grid_coverage <= 1.0 && std::isfinite(config.minimum_bearing_spread_rad) &&
         config.minimum_bearing_spread_rad > 0.0 && std::isfinite(config.minimum_median_depth_m) &&
         config.minimum_median_depth_m > 0.0 && std::isfinite(config.maximum_median_depth_m) &&
         config.maximum_median_depth_m > config.minimum_median_depth_m &&
         std::isfinite(config.minimum_median_parallax_rad) &&
         config.minimum_median_parallax_rad >= 0.0 && std::isfinite(config.robust_loss_scale_rad) &&
         config.robust_loss_scale_rad > 0.0 && std::isfinite(config.final_inlier_threshold_rad) &&
         config.final_inlier_threshold_rad > 0.0 && std::isfinite(config.base_bearing_sigma_rad) &&
         config.base_bearing_sigma_rad > 0.0 && std::isfinite(config.covariance_inflation) &&
         config.covariance_inflation >= 1.0 &&
         std::isfinite(config.hessian_absolute_rank_tolerance) &&
         config.hessian_absolute_rank_tolerance > 0.0 &&
         std::isfinite(config.hessian_relative_rank_tolerance) &&
         config.hessian_relative_rank_tolerance > 0.0 && config.minimum_information_rank > 0U &&
         config.minimum_information_rank <= 6U &&
         std::isfinite(config.maximum_information_eigenvalue) &&
         config.maximum_information_eigenvalue > 0.0 &&
         std::isfinite(config.maximum_information_condition) &&
         config.maximum_information_condition > 1.0;
}

}  // namespace

struct VisualPlaceIndex::Impl {
  struct StoredKeyframe {
    std::size_t submap_index{};
    VisualPlaceKeyframeId id;
    core::FusionTime time;
    std::uint64_t covisibility_group{};
    BowVector bow;
    std::size_t descriptor_count{};
  };

  VisualVocabulary vocabulary;
  VisualPlaceIndexConfig config;
  bool valid{};
  std::string invalid_detail;
  std::vector<VisualRetrievalSubmap> submaps;
  std::vector<StoredKeyframe> keyframes;
  std::vector<std::vector<std::pair<std::size_t, double>>> inverted;
  std::size_t descriptor_count{};

  explicit Impl(VisualVocabulary vocabulary_in, VisualPlaceIndexConfig config_in)
      : vocabulary(std::move(vocabulary_in)), config(config_in) {
    if (!validIndexConfig(config)) {
      invalid_detail = "visual place index configuration is invalid";
      return;
    }
    if (!vocabulary.revision.valid() || !nonzeroHash(vocabulary.checksum) ||
        vocabulary.words.size() < 2U || vocabulary.words.size() > config.maximum_words ||
        vocabulary.inverse_document_frequency.size() != vocabulary.words.size()) {
      invalid_detail = "fixed vocabulary revision, checksum, word count, or IDF layout is invalid";
      return;
    }
    for (double weight : vocabulary.inverse_document_frequency) {
      if (!std::isfinite(weight) || weight <= 0.0) {
        invalid_detail = "fixed vocabulary IDF weights must be finite and positive";
        return;
      }
    }
    inverted.resize(vocabulary.words.size());
    valid = true;
  }

  [[nodiscard]] BowVector transform(std::span<const VisualBriskDescriptor> descriptors) const {
    std::map<std::uint32_t, std::size_t> counts;
    for (const auto& descriptor : descriptors) {
      std::uint32_t best_word = 0U;
      std::uint32_t best_distance = std::numeric_limits<std::uint32_t>::max();
      for (std::uint32_t word = 0U; word < vocabulary.words.size(); ++word) {
        const std::uint32_t distance = hamming(descriptor, vocabulary.words[word]);
        if (distance < best_distance) {
          best_distance = distance;
          best_word = word;
        }
      }
      ++counts[best_word];
    }
    BowVector bow;
    double normalization = 0.0;
    for (const auto& [word, count] : counts) {
      const double weight =
          static_cast<double>(count) * vocabulary.inverse_document_frequency[word];
      bow.push_back({word, weight});
      normalization += weight;
    }
    if (normalization > 0.0) {
      for (BowEntry& entry : bow) {
        entry.weight /= normalization;
      }
    }
    return bow;
  }

  [[nodiscard]] core::Result<bool, VisualPlaceError> validateRecord(
      const VisualRetrievalSubmap& record) const {
    if (core::validateSubmapRef(record.submap) != core::SubmapRefValidationError::None ||
        record.model_revision != vocabulary.revision ||
        record.model_checksum != vocabulary.checksum || record.config_revision != config.revision ||
        record.keyframes.empty() || record.keyframes.size() > config.maximum_keyframes_per_submap) {
      return core::Result<bool, VisualPlaceError>::failure(
          {record.model_revision != vocabulary.revision ||
                   record.model_checksum != vocabulary.checksum
               ? VisualPlaceErrorCode::ModelMismatch
               : (record.config_revision != config.revision ? VisualPlaceErrorCode::ConfigMismatch
                                                            : VisualPlaceErrorCode::InvalidRecord),
           "retrieval submap identity, fixed model, or keyframe count is invalid"});
    }
    std::set<VisualPlaceKeyframeId> ids;
    core::FusionTime previous = record.keyframes.front().time;
    for (std::size_t index = 0U; index < record.keyframes.size(); ++index) {
      const auto& keyframe = record.keyframes[index];
      if (!keyframe.id.valid() || keyframe.descriptors.empty() ||
          keyframe.descriptors.size() > config.maximum_descriptors_per_keyframe ||
          !ids.insert(keyframe.id).second || (index > 0U && keyframe.time < previous)) {
        return core::Result<bool, VisualPlaceError>::failure(
            {VisualPlaceErrorCode::InvalidRecord,
             "retrieval keyframes require unique IDs, ordered time, and bounded descriptors"});
      }
      previous = keyframe.time;
    }
    std::set<core::SubmapRef> exclusions;
    for (const core::SubmapRef& id : record.overlap_or_adjacent) {
      if (core::validateSubmapRef(id) != core::SubmapRefValidationError::None ||
          id == record.submap || !exclusions.insert(id).second) {
        return core::Result<bool, VisualPlaceError>::failure(
            {VisualPlaceErrorCode::InvalidRecord,
             "overlap/adjacency exclusions must be valid, unique, and not self"});
      }
    }
    return core::Result<bool, VisualPlaceError>::success(true);
  }
};

VisualPlaceIndex::VisualPlaceIndex(VisualVocabulary vocabulary, VisualPlaceIndexConfig config)
    : impl_(std::make_unique<Impl>(std::move(vocabulary), config)) {}

VisualPlaceIndex::~VisualPlaceIndex() = default;
VisualPlaceIndex::VisualPlaceIndex(VisualPlaceIndex&&) noexcept = default;
VisualPlaceIndex& VisualPlaceIndex::operator=(VisualPlaceIndex&&) noexcept = default;

core::Result<bool, VisualPlaceError> VisualPlaceIndex::add(VisualRetrievalSubmap record) {
  if (!impl_->valid) {
    return core::Result<bool, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::InvalidVocabulary, impl_->invalid_detail});
  }
  auto validation = impl_->validateRecord(record);
  if (!validation) {
    return core::Result<bool, VisualPlaceError>::failure(validation.error());
  }
  for (const auto& existing : impl_->submaps) {
    if (core::sparseSubmapIdentityKey(existing.submap) ==
        core::sparseSubmapIdentityKey(record.submap)) {
      if (sameRetrievalRecord(existing, record)) {
        return core::Result<bool, VisualPlaceError>::success(false);
      }
      return core::Result<bool, VisualPlaceError>::failure(
          {VisualPlaceErrorCode::DuplicateSubmapConflict,
           "the same submap identity was re-delivered with different visual place content"});
    }
  }
  std::size_t new_descriptors = 0U;
  for (const auto& keyframe : record.keyframes) {
    new_descriptors += keyframe.descriptors.size();
  }
  if (impl_->submaps.size() + 1U > impl_->config.maximum_submaps ||
      impl_->keyframes.size() + record.keyframes.size() > impl_->config.maximum_keyframes ||
      impl_->descriptor_count + new_descriptors > impl_->config.maximum_descriptors) {
    return core::Result<bool, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::CapacityExceeded,
         "adding the visual submap would exceed an index hard bound"});
  }

  const std::size_t submap_index = impl_->submaps.size();
  const std::size_t keyframe_begin = impl_->keyframes.size();
  impl_->submaps.push_back(record);
  for (const auto& keyframe : record.keyframes) {
    Impl::StoredKeyframe stored;
    stored.submap_index = submap_index;
    stored.id = keyframe.id;
    stored.time = keyframe.time;
    stored.covisibility_group = keyframe.covisibility_group;
    stored.bow = impl_->transform(keyframe.descriptors);
    stored.descriptor_count = keyframe.descriptors.size();
    impl_->keyframes.push_back(std::move(stored));
  }
  for (std::size_t index = keyframe_begin; index < impl_->keyframes.size(); ++index) {
    for (const BowEntry& entry : impl_->keyframes[index].bow) {
      impl_->inverted[entry.word].emplace_back(index, entry.weight);
    }
  }
  impl_->descriptor_count += new_descriptors;
  return core::Result<bool, VisualPlaceError>::success(true);
}

core::Result<VisualRetrievalReport, VisualPlaceError> VisualPlaceIndex::query(
    const VisualRetrievalSubmap& query_record, core::FusionTime now) const {
  if (!impl_->valid) {
    return core::Result<VisualRetrievalReport, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::InvalidVocabulary, impl_->invalid_detail});
  }
  auto validation = impl_->validateRecord(query_record);
  if (!validation) {
    return core::Result<VisualRetrievalReport, VisualPlaceError>::failure(validation.error());
  }

  struct Vote {
    std::size_t query_index{};
    std::size_t database_index{};
    double score{};
  };
  std::map<std::size_t, std::vector<Vote>> votes;
  VisualRetrievalReport report;
  report.model_revision = impl_->vocabulary.revision;
  report.model_checksum = impl_->vocabulary.checksum;
  report.config_revision = impl_->config.revision;
  report.query_submap = query_record.submap;
  report.evaluated_at = now;
  report.indexed_submaps = impl_->submaps.size();
  report.indexed_keyframes = impl_->keyframes.size();
  report.indexed_descriptors = impl_->descriptor_count;

  for (std::size_t query_index = 0U; query_index < query_record.keyframes.size(); ++query_index) {
    const auto query_bow = impl_->transform(query_record.keyframes[query_index].descriptors);
    std::map<std::size_t, double> scores;
    for (const BowEntry& query_entry : query_bow) {
      for (const auto& [database_index, database_weight] : impl_->inverted[query_entry.word]) {
        scores[database_index] += std::min(query_entry.weight, database_weight);
      }
    }
    std::vector<std::pair<std::size_t, double>> ordered(scores.begin(), scores.end());
    std::sort(ordered.begin(), ordered.end(), [&](const auto& lhs, const auto& rhs) {
      if (lhs.second != rhs.second) {
        return lhs.second > rhs.second;
      }
      const auto& lhs_keyframe = impl_->keyframes[lhs.first];
      const auto& rhs_keyframe = impl_->keyframes[rhs.first];
      if (lhs_keyframe.submap_index != rhs_keyframe.submap_index) {
        return impl_->submaps[lhs_keyframe.submap_index].submap <
               impl_->submaps[rhs_keyframe.submap_index].submap;
      }
      return lhs_keyframe.id < rhs_keyframe.id;
    });
    std::size_t retained = 0U;
    for (const auto& [database_index, score] : ordered) {
      ++report.evaluated_frame_pairs;
      const auto& database_keyframe = impl_->keyframes[database_index];
      const auto& candidate_record = impl_->submaps[database_keyframe.submap_index];
      if (candidate_record.submap == query_record.submap) {
        ++report.same_submap_exclusions;
        continue;
      }
      if (containsSubmap(query_record.overlap_or_adjacent, candidate_record.submap) ||
          containsSubmap(candidate_record.overlap_or_adjacent, query_record.submap)) {
        ++report.overlap_or_adjacent_exclusions;
        continue;
      }
      if (score < impl_->config.minimum_frame_score) {
        ++report.below_score_exclusions;
        continue;
      }
      if (retained >= impl_->config.frame_top_k) {
        ++report.top_k_exclusions;
        continue;
      }
      ++retained;
      votes[database_keyframe.submap_index].push_back({query_index, database_index, score});
      if (report.frame_hits.size() < impl_->config.maximum_frame_hits) {
        report.frame_hits.push_back({query_record.keyframes[query_index].id, database_keyframe.id,
                                     candidate_record.submap, score,
                                     VisualRetrievalExclusion::None});
      }
    }
  }

  for (auto& [submap_index, candidate_votes] : votes) {
    std::sort(
        candidate_votes.begin(), candidate_votes.end(), [&](const Vote& lhs, const Vote& rhs) {
          if (lhs.score != rhs.score) {
            return lhs.score > rhs.score;
          }
          const auto& lhs_query = query_record.keyframes[lhs.query_index];
          const auto& rhs_query = query_record.keyframes[rhs.query_index];
          if (lhs_query.id != rhs_query.id) {
            return lhs_query.id < rhs_query.id;
          }
          return impl_->keyframes[lhs.database_index].id < impl_->keyframes[rhs.database_index].id;
        });

    std::vector<Vote> best_group;
    for (std::size_t anchor = 0U; anchor < candidate_votes.size(); ++anchor) {
      std::vector<Vote> group;
      std::set<VisualPlaceKeyframeId> query_ids;
      std::set<VisualPlaceKeyframeId> candidate_ids;
      for (const Vote& vote : candidate_votes) {
        const auto& q_anchor = query_record.keyframes[candidate_votes[anchor].query_index];
        const auto& c_anchor = impl_->keyframes[candidate_votes[anchor].database_index];
        const auto& q = query_record.keyframes[vote.query_index];
        const auto& c = impl_->keyframes[vote.database_index];
        const bool query_compatible = (q_anchor.covisibility_group != 0U &&
                                       q_anchor.covisibility_group == q.covisibility_group) ||
                                      absoluteNanoseconds(q_anchor.time, q.time) <=
                                          impl_->config.temporal_vote_window.nanoseconds;
        const bool candidate_compatible = (c_anchor.covisibility_group != 0U &&
                                           c_anchor.covisibility_group == c.covisibility_group) ||
                                          absoluteNanoseconds(c_anchor.time, c.time) <=
                                              impl_->config.temporal_vote_window.nanoseconds;
        if (query_compatible && candidate_compatible && query_ids.insert(q.id).second &&
            candidate_ids.insert(c.id).second) {
          group.push_back(vote);
        }
      }
      const double group_score =
          std::accumulate(group.begin(), group.end(), 0.0,
                          [](double sum, const Vote& vote) { return sum + vote.score; });
      const double best_score =
          std::accumulate(best_group.begin(), best_group.end(), 0.0,
                          [](double sum, const Vote& vote) { return sum + vote.score; });
      if (group.size() > best_group.size() ||
          (group.size() == best_group.size() && group_score > best_score)) {
        best_group = std::move(group);
      }
    }

    std::set<VisualPlaceKeyframeId> query_support;
    std::set<VisualPlaceKeyframeId> candidate_support;
    double score_sum = 0.0;
    for (const Vote& vote : best_group) {
      query_support.insert(query_record.keyframes[vote.query_index].id);
      candidate_support.insert(impl_->keyframes[vote.database_index].id);
      score_sum += vote.score;
    }
    if (query_support.size() < impl_->config.minimum_query_keyframe_votes ||
        candidate_support.size() < impl_->config.minimum_candidate_keyframe_votes) {
      ++report.voting_exclusions;
      continue;
    }
    VisualPlaceSeed seed{impl_->submaps[submap_index].submap,
                         query_record.submap,
                         impl_->vocabulary.revision,
                         impl_->vocabulary.checksum,
                         impl_->config.revision,
                         now,
                         saturatingAdd(now, impl_->config.candidate_ttl),
                         0.0,
                         0U,
                         0U,
                         {},
                         {}};
    seed.scheduling_score = score_sum / static_cast<double>(best_group.size());
    seed.query_keyframe_votes = query_support.size();
    seed.candidate_keyframe_votes = candidate_support.size();
    seed.query_support.assign(query_support.begin(), query_support.end());
    seed.candidate_support.assign(candidate_support.begin(), candidate_support.end());
    report.candidates.push_back(std::move(seed));
  }
  std::sort(report.candidates.begin(), report.candidates.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.scheduling_score != rhs.scheduling_score) {
                return lhs.scheduling_score > rhs.scheduling_score;
              }
              return lhs.candidate < rhs.candidate;
            });
  if (report.candidates.size() > impl_->config.submap_top_k) {
    report.top_k_exclusions += report.candidates.size() - impl_->config.submap_top_k;
    report.candidates.erase(
        report.candidates.begin() + static_cast<std::ptrdiff_t>(impl_->config.submap_top_k),
        report.candidates.end());
  }
  return core::Result<VisualRetrievalReport, VisualPlaceError>::success(std::move(report));
}

namespace {

struct Correspondence {
  std::size_t feature_index{};
  std::size_t landmark_index{};
  std::uint32_t distance{};
};

[[nodiscard]] double angularResidual(const core::Pose3d& T_candidate_query,
                                     const MatureVisualLandmark& landmark,
                                     const QueryVisualFeature& feature,
                                     Eigen::Vector2d* tangent_residual = nullptr,
                                     double* depth = nullptr) noexcept {
  const Eigen::Vector3d point_query =
      T_candidate_query.inverse() * landmark.position_candidate_submap;
  const Eigen::Vector3d point_camera = feature.T_query_submap_camera.inverse() * point_query;
  const double point_norm = point_camera.norm();
  if (!(point_norm > kTiny) || !std::isfinite(point_norm)) {
    return std::numeric_limits<double>::infinity();
  }
  if (depth != nullptr) {
    *depth = point_camera.z();
  }
  const Eigen::Vector3d predicted = point_camera / point_norm;
  const Eigen::Vector3d bearing = feature.bearing_camera.normalized();
  const double cosine = std::clamp(predicted.dot(bearing), -1.0, 1.0);
  if (tangent_residual != nullptr) {
    const Eigen::Vector3d first = tangentAxis(bearing);
    const Eigen::Vector3d second = bearing.cross(first).normalized();
    (*tangent_residual) << first.dot(predicted), second.dot(predicted);
  }
  return std::acos(cosine);
}

[[nodiscard]] Eigen::Matrix<double, 2, 6> numericalJacobian(
    const core::Pose3d& pose, const MatureVisualLandmark& landmark,
    const QueryVisualFeature& feature) noexcept {
  Eigen::Matrix<double, 2, 6> jacobian;
  constexpr double step = 1.0e-6;
  for (int column = 0; column < 6; ++column) {
    Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
    delta(column) = step;
    Eigen::Vector2d plus;
    Eigen::Vector2d minus;
    (void)angularResidual(pose * core::Pose3d::exp(delta), landmark, feature, &plus);
    (void)angularResidual(pose * core::Pose3d::exp(-delta), landmark, feature, &minus);
    jacobian.col(column) = (plus - minus) / (2.0 * step);
  }
  return jacobian;
}

[[nodiscard]] double correspondenceVariance(const core::Pose3d& pose,
                                            const MatureVisualLandmark& landmark,
                                            const QueryVisualFeature& feature,
                                            double base_sigma) noexcept {
  const core::Pose3d T_camera_candidate = feature.T_query_submap_camera.inverse() * pose.inverse();
  const Eigen::Vector3d point_camera = T_camera_candidate * landmark.position_candidate_submap;
  const double range = point_camera.norm();
  if (!(range > kTiny) || !std::isfinite(range)) {
    return std::numeric_limits<double>::infinity();
  }
  const Eigen::Vector3d predicted = point_camera / range;
  const Eigen::Vector3d bearing = feature.bearing_camera.normalized();
  const Eigen::Vector3d first = tangentAxis(bearing);
  const Eigen::Vector3d second = bearing.cross(first).normalized();
  Eigen::Matrix<double, 2, 3> tangent;
  tangent.row(0) = first.transpose();
  tangent.row(1) = second.transpose();
  const Eigen::Matrix3d unit_jacobian =
      (Eigen::Matrix3d::Identity() - predicted * predicted.transpose()) / range;
  const Eigen::Matrix<double, 2, 3> point_jacobian =
      tangent * unit_jacobian * T_camera_candidate.so3().matrix();
  const Eigen::Matrix2d projected =
      point_jacobian * landmark.covariance_candidate_submap * point_jacobian.transpose();
  const double landmark_variance = std::max(0.0, 0.5 * projected.trace());
  return base_sigma * base_sigma + landmark_variance;
}

struct PoseScore {
  core::Pose3d pose;
  std::vector<std::size_t> inliers;
  double residual_sum{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] PoseScore scorePose(const core::Pose3d& pose,
                                  std::span<const Correspondence> correspondences,
                                  std::span<const MatureVisualLandmark> landmarks,
                                  std::span<const QueryVisualFeature> features, double threshold) {
  PoseScore score;
  score.pose = pose;
  score.residual_sum = 0.0;
  for (std::size_t index = 0U; index < correspondences.size(); ++index) {
    const auto& correspondence = correspondences[index];
    double depth = 0.0;
    const double angle = angularResidual(pose, landmarks[correspondence.landmark_index],
                                         features[correspondence.feature_index], nullptr, &depth);
    if (depth > 0.0 && angle <= threshold) {
      score.inliers.push_back(index);
      score.residual_sum += angle;
    }
  }
  return score;
}

[[nodiscard]] bool betterScore(const PoseScore& candidate, const PoseScore& incumbent) noexcept {
  return candidate.inliers.size() > incumbent.inliers.size() ||
         (candidate.inliers.size() == incumbent.inliers.size() &&
          candidate.residual_sum < incumbent.residual_sum);
}

[[nodiscard]] std::uint64_t mixSeed(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] std::array<std::size_t, 4> deterministicSubset(std::size_t count,
                                                             std::uint64_t seed) noexcept {
  std::array<std::size_t, 4> subset{};
  for (std::size_t item = 0U; item < subset.size(); ++item) {
    for (;;) {
      seed = mixSeed(seed + item);
      const std::size_t index = static_cast<std::size_t>(seed % count);
      if (std::find(subset.begin(), subset.begin() + static_cast<std::ptrdiff_t>(item), index) ==
          subset.begin() + static_cast<std::ptrdiff_t>(item)) {
        subset[item] = index;
        break;
      }
    }
  }
  std::sort(subset.begin(), subset.end());
  return subset;
}

[[nodiscard]] double robustCost(const core::Pose3d& pose,
                                std::span<const Correspondence> correspondences,
                                std::span<const MatureVisualLandmark> landmarks,
                                std::span<const QueryVisualFeature> features, double scale) {
  double cost = 0.0;
  const double scale_squared = scale * scale;
  for (const auto& correspondence : correspondences) {
    Eigen::Vector2d residual;
    (void)angularResidual(pose, landmarks[correspondence.landmark_index],
                          features[correspondence.feature_index], &residual);
    cost += scale_squared * std::log1p(residual.squaredNorm() / scale_squared);
  }
  return cost;
}

}  // namespace

VisualGeometricVerifier::VisualGeometricVerifier(VisualGeometricVerifierConfig config)
    : config_(config) {}

VisualVerificationOutcome VisualGeometricVerifier::rejectScaleLessMonocular() const noexcept {
  VisualVerificationOutcome outcome;
  outcome.report.disposition = VisualVerificationDisposition::RejectedMonocular2d2dScaleLess;
  return outcome;
}

core::Result<VisualVerificationOutcome, VisualPlaceError> VisualGeometricVerifier::verify(
    const VisualVerificationInput& input, core::FusionTime now) const {
  if (!validVerifierConfig(config_)) {
    return core::Result<VisualVerificationOutcome, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::InvalidConfiguration,
         "visual geometric verifier configuration is invalid"});
  }
  VisualVerificationOutcome outcome;
  outcome.report.config_revision = input.seed.config_revision;
  auto reject = [&](VisualVerificationDisposition disposition) {
    outcome.report.disposition = disposition;
    return core::Result<VisualVerificationOutcome, VisualPlaceError>::success(std::move(outcome));
  };
  if (!input.proposal.valid() || !input.seed.model_revision.valid() ||
      !nonzeroHash(input.seed.model_checksum) || input.seed.query_support.empty() ||
      input.seed.candidate_support.empty() ||
      input.candidate_landmarks.size() > config_.maximum_landmarks ||
      input.query_features.size() > config_.maximum_query_features ||
      input.calibration_epochs.empty() ||
      !std::is_sorted(input.calibration_epochs.begin(), input.calibration_epochs.end()) ||
      std::adjacent_find(input.calibration_epochs.begin(), input.calibration_epochs.end()) !=
          input.calibration_epochs.end() ||
      std::any_of(input.calibration_epochs.begin(), input.calibration_epochs.end(),
                  [](core::CalibrationEpoch epoch) { return !epoch.valid(); }) ||
      core::validateSubmapRef(input.seed.candidate) != core::SubmapRefValidationError::None ||
      core::validateSubmapRef(input.seed.query) != core::SubmapRefValidationError::None ||
      core::validateSubmapRef(input.candidate_submap) != core::SubmapRefValidationError::None ||
      core::validateSubmapRef(input.query_submap) != core::SubmapRefValidationError::None ||
      input.header.session != input.seed.candidate.session ||
      input.header.session != input.seed.query.session ||
      !containsCalibration(input.calibration_epochs, input.seed.candidate.calibration) ||
      !containsCalibration(input.calibration_epochs, input.seed.query.calibration) ||
      !validRecordHeader(input.header, now, input.calibration_epochs) ||
      core::validateLineage(input.lineage) != core::LineageValidationError::None ||
      std::any_of(input.lineage.usage.begin(), input.lineage.usage.end(),
                  [&](const core::ObservationUsage& usage) {
                    return !containsCalibration(input.calibration_epochs,
                                                usage.slice.calibration);
                  })) {
    return core::Result<VisualVerificationOutcome, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::InvalidVerificationInput,
         "proposal, model, support, capacity, or observation lineage is invalid"});
  }
  if (now > input.seed.valid_until || now < input.seed.created_at) {
    return reject(VisualVerificationDisposition::RejectedExpiredSeed);
  }
  if (!sameSubmap(input.seed.candidate, input.candidate_submap) ||
      !sameSubmap(input.seed.query, input.query_submap) ||
      sameSubmapObject(input.candidate_submap, input.query_submap)) {
    return reject(VisualVerificationDisposition::RejectedEndpointMismatch);
  }
  if (input.candidate_model_revision != input.seed.model_revision ||
      input.query_model_revision != input.seed.model_revision ||
      input.candidate_model_checksum != input.seed.model_checksum ||
      input.query_model_checksum != input.seed.model_checksum) {
    return reject(VisualVerificationDisposition::RejectedModelMismatch);
  }
  if (input.candidate_config_revision != input.seed.config_revision ||
      input.query_config_revision != input.seed.config_revision ||
      !input.seed.config_revision.valid()) {
    return reject(VisualVerificationDisposition::RejectedConfigMismatch);
  }

  std::vector<std::size_t> mature;
  mature.reserve(input.candidate_landmarks.size());
  for (std::size_t index = 0U; index < input.candidate_landmarks.size(); ++index) {
    const auto& landmark = input.candidate_landmarks[index];
    if (landmark.mature && !landmark.dynamic && landmark.id.valid() &&
        landmark.observation_count >= 2U && landmark.position_candidate_submap.allFinite() &&
        validPointCovariance(landmark.covariance_candidate_submap) &&
        landmark.reference_camera_center_candidate_submap.allFinite() &&
        !landmark.descriptors.empty() &&
        landmark.descriptors.size() <= config_.maximum_descriptors_per_landmark) {
      mature.push_back(index);
    } else if (landmark.dynamic) {
      ++outcome.report.dynamic_rejections;
    }
  }
  outcome.report.mature_landmarks = mature.size();
  if (mature.size() < 2U) {
    return reject(VisualVerificationDisposition::RejectedNoMatureMetricLandmarks);
  }

  std::vector<Correspondence> tentative;
  tentative.reserve(input.query_features.size());
  std::vector<std::uint32_t> reciprocal_distance(input.candidate_landmarks.size(),
                                                 std::numeric_limits<std::uint32_t>::max());
  std::vector<std::size_t> reciprocal_feature(input.candidate_landmarks.size(),
                                              std::numeric_limits<std::size_t>::max());
  for (std::size_t feature_index = 0U; feature_index < input.query_features.size();
       ++feature_index) {
    const auto& feature = input.query_features[feature_index];
    if (feature.dynamic_masked) {
      ++outcome.report.dynamic_rejections;
      continue;
    }
    if (!feature.keyframe.valid() || !feature.camera.valid() ||
        !validPose(feature.T_query_submap_camera) || !feature.bearing_camera.allFinite() ||
        feature.bearing_camera.norm() < 0.99 || feature.image_width == 0U ||
        feature.image_height == 0U || !feature.pixel.allFinite()) {
      continue;
    }
    std::size_t best_landmark = 0U;
    std::uint32_t best = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t second = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t landmark_index : mature) {
      std::uint32_t landmark_best = std::numeric_limits<std::uint32_t>::max();
      for (const auto& descriptor : input.candidate_landmarks[landmark_index].descriptors) {
        landmark_best = std::min(landmark_best, hamming(feature.descriptor, descriptor));
        ++outcome.report.descriptor_comparisons;
      }
      if (landmark_best < reciprocal_distance[landmark_index] ||
          (landmark_best == reciprocal_distance[landmark_index] &&
           feature_index < reciprocal_feature[landmark_index])) {
        reciprocal_distance[landmark_index] = landmark_best;
        reciprocal_feature[landmark_index] = feature_index;
      }
      if (landmark_best < best ||
          (landmark_best == best && input.candidate_landmarks[landmark_index].id <
                                        input.candidate_landmarks[best_landmark].id)) {
        second = best;
        best = landmark_best;
        best_landmark = landmark_index;
      } else if (landmark_best < second) {
        second = landmark_best;
      }
    }
    if (best <= config_.maximum_hamming_distance &&
        second != std::numeric_limits<std::uint32_t>::max() &&
        static_cast<double>(best) < config_.maximum_ratio * static_cast<double>(second)) {
      ++outcome.report.ratio_passes;
      tentative.push_back({feature_index, best_landmark, best});
    }
  }
  if (tentative.empty()) {
    return reject(VisualVerificationDisposition::RejectedDescriptorAmbiguity);
  }

  std::map<std::size_t, Correspondence> landmark_best;
  for (const auto& correspondence : tentative) {
    if (reciprocal_feature[correspondence.landmark_index] != correspondence.feature_index) {
      continue;
    }
    auto iterator = landmark_best.find(correspondence.landmark_index);
    if (iterator == landmark_best.end() || correspondence.distance < iterator->second.distance ||
        (correspondence.distance == iterator->second.distance &&
         correspondence.feature_index < iterator->second.feature_index)) {
      landmark_best[correspondence.landmark_index] = correspondence;
    }
  }
  std::vector<Correspondence> correspondences;
  correspondences.reserve(landmark_best.size());
  for (const auto& [unused, correspondence] : landmark_best) {
    (void)unused;
    correspondences.push_back(correspondence);
  }
  std::sort(correspondences.begin(), correspondences.end(), [&](const auto& lhs, const auto& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    const auto& lhs_landmark = input.candidate_landmarks[lhs.landmark_index];
    const auto& rhs_landmark = input.candidate_landmarks[rhs.landmark_index];
    if (lhs_landmark.id != rhs_landmark.id) {
      return lhs_landmark.id < rhs_landmark.id;
    }
    return lhs.feature_index < rhs.feature_index;
  });
  if (correspondences.size() > config_.maximum_correspondences) {
    correspondences.resize(config_.maximum_correspondences);
  }
  outcome.report.mutual_matches = correspondences.size();
  if (correspondences.size() < 4U) {
    return reject(VisualVerificationDisposition::RejectedInsufficientCorrespondences);
  }

  std::map<std::pair<std::uint64_t, VisualPlaceKeyframeId>, std::vector<std::size_t>> camera_groups;
  for (std::size_t index = 0U; index < correspondences.size(); ++index) {
    const auto& feature = input.query_features[correspondences[index].feature_index];
    if (feature.bearing_camera.z() > 1.0e-4) {
      camera_groups[{feature.camera.value(), feature.keyframe}].push_back(index);
    }
  }
  std::vector<const std::vector<std::size_t>*> usable_groups;
  for (const auto& [unused, group] : camera_groups) {
    (void)unused;
    if (group.size() >= 4U) {
      usable_groups.push_back(&group);
    }
  }
  if (usable_groups.empty()) {
    return reject(VisualVerificationDisposition::RejectedRansac);
  }

  PoseScore best_pose;
  best_pose.residual_sum = std::numeric_limits<double>::infinity();
  for (std::size_t hypothesis = 0U; hypothesis < config_.maximum_ransac_hypotheses; ++hypothesis) {
    const auto& group = *usable_groups[hypothesis % usable_groups.size()];
    const std::uint64_t seed =
        input.proposal.value() ^ (input.seed.candidate.id.value() << 1U) ^
        (input.seed.query.id.value() << 17U) ^ hypothesis;
    const auto subset = deterministicSubset(group.size(), seed);
    std::vector<cv::Point3d> object_points;
    std::vector<cv::Point2d> image_points;
    object_points.reserve(4U);
    image_points.reserve(4U);
    for (std::size_t local_index : subset) {
      const auto& correspondence = correspondences[group[local_index]];
      const auto& point =
          input.candidate_landmarks[correspondence.landmark_index].position_candidate_submap;
      const auto bearing =
          input.query_features[correspondence.feature_index].bearing_camera.normalized();
      object_points.emplace_back(point.x(), point.y(), point.z());
      image_points.emplace_back(bearing.x() / bearing.z(), bearing.y() / bearing.z());
    }
    std::vector<cv::Mat> rotations;
    std::vector<cv::Mat> translations;
    int solutions = 0;
    try {
      solutions =
          cv::solvePnPGeneric(object_points, image_points, cv::Mat::eye(3, 3, CV_64F),
                              cv::noArray(), rotations, translations, false, cv::SOLVEPNP_AP3P);
    } catch (const cv::Exception&) {
      // Algebraically degenerate four-point subsets are expected RANSAC
      // outcomes, not process-level errors.
      solutions = 0;
    }
    ++outcome.report.ransac_hypotheses;
    const auto& seed_feature =
        input.query_features[correspondences[group[subset[0]]].feature_index];
    for (int solution = 0; solution < solutions; ++solution) {
      cv::Mat rotation_matrix;
      cv::Rodrigues(rotations[static_cast<std::size_t>(solution)], rotation_matrix);
      Eigen::Matrix3d R_camera_candidate;
      Eigen::Vector3d t_camera_candidate;
      for (int row = 0; row < 3; ++row) {
        t_camera_candidate(row) = translations[static_cast<std::size_t>(solution)].at<double>(row);
        for (int column = 0; column < 3; ++column) {
          R_camera_candidate(row, column) = rotation_matrix.at<double>(row, column);
        }
      }
      core::Pose3d T_camera_candidate(R_camera_candidate, t_camera_candidate);
      const core::Pose3d T_query_candidate =
          seed_feature.T_query_submap_camera * T_camera_candidate;
      const core::Pose3d T_candidate_query = T_query_candidate.inverse();
      if (!validPose(T_candidate_query)) {
        continue;
      }
      auto scored = scorePose(T_candidate_query, correspondences, input.candidate_landmarks,
                              input.query_features, config_.ransac_angular_threshold_rad);
      if (betterScore(scored, best_pose)) {
        best_pose = std::move(scored);
      }
    }
  }
  outcome.report.ransac_inliers = best_pose.inliers.size();
  outcome.report.ransac_inlier_ratio =
      static_cast<double>(best_pose.inliers.size()) / static_cast<double>(correspondences.size());
  if (best_pose.inliers.size() < config_.minimum_ransac_inliers ||
      outcome.report.ransac_inlier_ratio < config_.minimum_ransac_inlier_ratio) {
    return reject(VisualVerificationDisposition::RejectedRansac);
  }

  std::vector<Correspondence> inliers;
  inliers.reserve(best_pose.inliers.size());
  for (std::size_t index : best_pose.inliers) {
    inliers.push_back(correspondences[index]);
  }
  core::Pose3d pose = best_pose.pose;
  outcome.report.robust_cost_before =
      robustCost(pose, inliers, input.candidate_landmarks, input.query_features,
                 config_.robust_loss_scale_rad);
  double previous_cost = outcome.report.robust_cost_before;
  bool refinement_valid = true;
  for (std::size_t iteration = 0U; iteration < config_.maximum_refinement_iterations; ++iteration) {
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    const double scale_squared = config_.robust_loss_scale_rad * config_.robust_loss_scale_rad;
    for (const auto& correspondence : inliers) {
      Eigen::Vector2d residual;
      (void)angularResidual(pose, input.candidate_landmarks[correspondence.landmark_index],
                            input.query_features[correspondence.feature_index], &residual);
      const auto jacobian =
          numericalJacobian(pose, input.candidate_landmarks[correspondence.landmark_index],
                            input.query_features[correspondence.feature_index]);
      const double weight = 1.0 / (1.0 + residual.squaredNorm() / scale_squared);
      hessian.noalias() += weight * jacobian.transpose() * jacobian;
      gradient.noalias() += weight * jacobian.transpose() * residual;
    }
    hessian.diagonal().array() += 1.0e-10;
    Eigen::LDLT<Eigen::Matrix<double, 6, 6>> decomposition(hessian);
    if (decomposition.info() != Eigen::Success) {
      refinement_valid = false;
      break;
    }
    Eigen::Matrix<double, 6, 1> step = decomposition.solve(-gradient);
    if (!step.allFinite()) {
      refinement_valid = false;
      break;
    }
    const double translation_norm = step.head<3>().norm();
    const double rotation_norm = step.tail<3>().norm();
    if (translation_norm > 1.0) {
      step.head<3>() *= 1.0 / translation_norm;
    }
    if (rotation_norm > 0.25) {
      step.tail<3>() *= 0.25 / rotation_norm;
    }
    core::Pose3d candidate = pose * core::Pose3d::exp(step);
    const double candidate_cost = robustCost(candidate, inliers, input.candidate_landmarks,
                                             input.query_features, config_.robust_loss_scale_rad);
    ++outcome.report.refinement_iterations;
    if (!std::isfinite(candidate_cost) || candidate_cost > previous_cost + 1.0e-12) {
      break;
    }
    pose = std::move(candidate);
    const double reduction = previous_cost - candidate_cost;
    previous_cost = candidate_cost;
    if (step.norm() < 1.0e-7 || reduction < 1.0e-12) {
      break;
    }
  }
  outcome.report.robust_cost_after = previous_cost;
  if (!refinement_valid || !validPose(pose) ||
      previous_cost > outcome.report.robust_cost_before + 1.0e-12) {
    return reject(VisualVerificationDisposition::RejectedRefinement);
  }

  std::vector<Correspondence> final_inliers;
  std::vector<double> depths;
  std::vector<double> parallaxes;
  std::set<std::pair<std::uint32_t, std::uint32_t>> occupied_cells;
  std::vector<Eigen::Vector3d> bearings;
  for (const auto& correspondence : correspondences) {
    const auto& feature = input.query_features[correspondence.feature_index];
    const auto& landmark = input.candidate_landmarks[correspondence.landmark_index];
    double depth = 0.0;
    const double angle = angularResidual(pose, landmark, feature, nullptr, &depth);
    if (!(depth > 0.0) || angle > config_.final_inlier_threshold_rad) {
      continue;
    }
    final_inliers.push_back(correspondence);
    depths.push_back(depth);
    bearings.push_back(feature.bearing_camera.normalized());
    const auto column = std::min(
        config_.grid_columns - 1U,
        static_cast<std::uint32_t>(
            std::clamp(feature.pixel.x(), 0.0, static_cast<double>(feature.image_width - 1U)) *
            config_.grid_columns / feature.image_width));
    const auto row = std::min(
        config_.grid_rows - 1U,
        static_cast<std::uint32_t>(
            std::clamp(feature.pixel.y(), 0.0, static_cast<double>(feature.image_height - 1U)) *
            config_.grid_rows / feature.image_height));
    occupied_cells.insert({column, row});
    const Eigen::Vector3d query_camera_candidate =
        pose * feature.T_query_submap_camera.translation();
    const Eigen::Vector3d first =
        (landmark.position_candidate_submap - landmark.reference_camera_center_candidate_submap)
            .normalized();
    const Eigen::Vector3d second =
        (landmark.position_candidate_submap - query_camera_candidate).normalized();
    parallaxes.push_back(std::acos(std::clamp(first.dot(second), -1.0, 1.0)));
  }
  outcome.report.final_inliers = final_inliers.size();
  outcome.report.final_inlier_ratio =
      static_cast<double>(final_inliers.size()) / static_cast<double>(correspondences.size());
  if (final_inliers.size() < config_.minimum_ransac_inliers ||
      outcome.report.final_inlier_ratio < config_.minimum_ransac_inlier_ratio) {
    return reject(VisualVerificationDisposition::RejectedInlierSupport);
  }
  outcome.report.grid_coverage = static_cast<double>(occupied_cells.size()) /
                                 static_cast<double>(config_.grid_columns * config_.grid_rows);
  if (outcome.report.grid_coverage < config_.minimum_grid_coverage) {
    return reject(VisualVerificationDisposition::RejectedGridCoverage);
  }
  Eigen::Vector3d mean_bearing = Eigen::Vector3d::Zero();
  for (const auto& bearing : bearings) {
    mean_bearing += bearing;
  }
  mean_bearing.normalize();
  std::vector<double> bearing_angles;
  for (const auto& bearing : bearings) {
    bearing_angles.push_back(std::acos(std::clamp(mean_bearing.dot(bearing), -1.0, 1.0)));
  }
  outcome.report.bearing_spread_rad = median(std::move(bearing_angles));
  if (outcome.report.bearing_spread_rad < config_.minimum_bearing_spread_rad) {
    return reject(VisualVerificationDisposition::RejectedBearingDiversity);
  }
  outcome.report.median_depth_m = median(std::move(depths));
  outcome.report.median_parallax_rad = median(std::move(parallaxes));
  if (outcome.report.median_depth_m < config_.minimum_median_depth_m ||
      outcome.report.median_depth_m > config_.maximum_median_depth_m ||
      outcome.report.median_parallax_rad < config_.minimum_median_parallax_rad) {
    return reject(VisualVerificationDisposition::RejectedDepthParallax);
  }

  Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
  for (const auto& correspondence : final_inliers) {
    const auto jacobian =
        numericalJacobian(pose, input.candidate_landmarks[correspondence.landmark_index],
                          input.query_features[correspondence.feature_index]);
    const double variance = correspondenceVariance(
        pose, input.candidate_landmarks[correspondence.landmark_index],
        input.query_features[correspondence.feature_index], config_.base_bearing_sigma_rad);
    if (!std::isfinite(variance) || variance <= 0.0) {
      return core::Result<VisualVerificationOutcome, VisualPlaceError>::failure(
          {VisualPlaceErrorCode::NumericalFailure,
           "landmark covariance projection produced an invalid bearing variance"});
    }
    hessian.noalias() +=
        jacobian.transpose() * jacobian / (variance * config_.covariance_inflation);
  }
  hessian = 0.5 * (hessian + hessian.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigen_solver(hessian);
  if (eigen_solver.info() != Eigen::Success || !eigen_solver.eigenvalues().allFinite()) {
    return core::Result<VisualVerificationOutcome, VisualPlaceError>::failure(
        {VisualPlaceErrorCode::NumericalFailure,
         "final visual verification Hessian eigendecomposition failed"});
  }
  core::RankAwareInformation information;
  const double maximum = std::max(0.0, eigen_solver.eigenvalues()(5));
  const double threshold = std::max(config_.hessian_absolute_rank_tolerance,
                                    maximum * config_.hessian_relative_rank_tolerance);
  double smallest_supported = std::numeric_limits<double>::infinity();
  for (std::size_t column = 0U; column < 6U; ++column) {
    const int source = 5 - static_cast<int>(column);
    const double value = eigen_solver.eigenvalues()(source);
    outcome.report.hessian_eigenvalues(static_cast<int>(column)) = value;
    information.basis.col(static_cast<int>(column)) = eigen_solver.eigenvectors().col(source);
    if (value > threshold && value > 0.0) {
      information.eigenvalues(static_cast<int>(column)) =
          std::min(value, config_.maximum_information_eigenvalue);
      smallest_supported =
          std::min(smallest_supported, information.eigenvalues(static_cast<int>(column)));
      ++information.rank;
    }
  }
  outcome.report.information_rank = information.rank;
  outcome.report.information_condition = information.rank > 0U
                                             ? information.eigenvalues(0) / smallest_supported
                                             : std::numeric_limits<double>::infinity();
  if (information.rank < config_.minimum_information_rank ||
      !std::isfinite(outcome.report.information_condition) ||
      outcome.report.information_condition > config_.maximum_information_condition) {
    return reject(VisualVerificationDisposition::RejectedDegenerateInformation);
  }

  LoopMeasurement measurement{input.header,
                              input.proposal,
                              LoopModality::Visual,
                              input.candidate_submap,
                              input.query_submap,
                              input.calibration_epochs,
                              pose,
                              information,
                              input.lineage};
  std::sort(measurement.calibration_epochs.begin(), measurement.calibration_epochs.end());
  measurement.calibration_epochs.erase(
      std::unique(measurement.calibration_epochs.begin(), measurement.calibration_epochs.end()),
      measurement.calibration_epochs.end());
  outcome.measurement = std::move(measurement);
  outcome.report.disposition = VisualVerificationDisposition::AcceptedMetric3d2d;
  return core::Result<VisualVerificationOutcome, VisualPlaceError>::success(std::move(outcome));
}

}  // namespace meridian::global
