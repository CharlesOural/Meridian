#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

inline constexpr std::size_t kVisualBriskDescriptorBytes = 64U;
using VisualBriskDescriptor = std::array<std::uint8_t, kVisualBriskDescriptorBytes>;

struct VisualPlaceKeyframeIdTag;
struct VisualPlaceLandmarkIdTag;
struct DescriptorModelRevisionTag;
struct VisualPlaceConfigRevisionTag;
using VisualPlaceKeyframeId = core::StrongId<VisualPlaceKeyframeIdTag>;
using VisualPlaceLandmarkId = core::StrongId<VisualPlaceLandmarkIdTag>;
using DescriptorModelRevision = core::StrongId<DescriptorModelRevisionTag>;
using VisualPlaceConfigRevision = core::StrongId<VisualPlaceConfigRevisionTag>;

// A fixed, offline-produced vocabulary. Query or database descriptors are
// never used to retrain it at runtime. Word order is part of the checksum.
struct VisualVocabulary {
  DescriptorModelRevision revision;
  core::ContentHash checksum{};
  std::vector<VisualBriskDescriptor> words;
  // Positive inverse-document-frequency weight for every word.
  std::vector<double> inverse_document_frequency;
};

struct VisualRetrievalKeyframe {
  VisualPlaceKeyframeId id;
  core::FusionTime time;
  // Non-zero IDs identify keyframes connected by sealed covisibility.
  std::uint64_t covisibility_group{};
  std::vector<VisualBriskDescriptor> descriptors;
};

struct VisualRetrievalSubmap {
  core::SubmapRef submap;
  DescriptorModelRevision model_revision;
  core::ContentHash model_checksum{};
  VisualPlaceConfigRevision config_revision;
  std::vector<VisualRetrievalKeyframe> keyframes;
  // Explicit symmetric policy input. It contains overlapping and immediately
  // adjacent submaps; no spatial-distance heuristic substitutes for it.
  std::vector<core::SubmapRef> overlap_or_adjacent;
};

struct VisualPlaceIndexConfig {
  VisualPlaceConfigRevision revision{1U};
  std::size_t maximum_submaps{4096U};
  std::size_t maximum_keyframes{65'536U};
  std::size_t maximum_descriptors{8'000'000U};
  std::size_t maximum_words{65'536U};
  std::size_t maximum_descriptors_per_keyframe{2048U};
  std::size_t maximum_keyframes_per_submap{256U};
  std::size_t maximum_frame_hits{4096U};
  std::size_t frame_top_k{12U};
  std::size_t submap_top_k{5U};
  std::size_t minimum_query_keyframe_votes{2U};
  std::size_t minimum_candidate_keyframe_votes{2U};
  double minimum_frame_score{0.08};
  core::Duration temporal_vote_window{1'500'000'000LL};
  core::Duration candidate_ttl{10'000'000'000LL};
};

enum class VisualRetrievalExclusion {
  None,
  SameSubmap,
  OverlapOrAdjacent,
  ModelMismatch,
  InsufficientTemporalCovisibilityVotes,
  BelowScore,
  TopK,
};

struct VisualFrameRetrievalHit {
  VisualPlaceKeyframeId query_keyframe;
  VisualPlaceKeyframeId candidate_keyframe;
  core::SubmapRef candidate_submap;
  double score{};
  VisualRetrievalExclusion exclusion{VisualRetrievalExclusion::None};
};

// Replaceable scheduling record. Scores and votes are intentionally absent
// from LoopMeasurement and never influence geometric information.
struct VisualPlaceSeed {
  core::SubmapRef candidate;
  core::SubmapRef query;
  DescriptorModelRevision model_revision;
  core::ContentHash model_checksum{};
  VisualPlaceConfigRevision config_revision;
  core::FusionTime created_at;
  core::FusionTime valid_until;
  double scheduling_score{};
  std::size_t query_keyframe_votes{};
  std::size_t candidate_keyframe_votes{};
  std::vector<VisualPlaceKeyframeId> query_support;
  std::vector<VisualPlaceKeyframeId> candidate_support;
};

struct VisualRetrievalReport {
  DescriptorModelRevision model_revision;
  core::ContentHash model_checksum{};
  VisualPlaceConfigRevision config_revision;
  core::SubmapRef query_submap;
  core::FusionTime evaluated_at;
  std::size_t indexed_submaps{};
  std::size_t indexed_keyframes{};
  std::size_t indexed_descriptors{};
  std::size_t evaluated_frame_pairs{};
  std::size_t same_submap_exclusions{};
  std::size_t overlap_or_adjacent_exclusions{};
  std::size_t below_score_exclusions{};
  std::size_t voting_exclusions{};
  std::size_t top_k_exclusions{};
  std::vector<VisualFrameRetrievalHit> frame_hits;
  std::vector<VisualPlaceSeed> candidates;
};

enum class VisualPlaceErrorCode {
  InvalidConfiguration,
  InvalidVocabulary,
  InvalidRecord,
  ModelMismatch,
  ConfigMismatch,
  DuplicateSubmapConflict,
  CapacityExceeded,
  InvalidVerificationInput,
  NumericalFailure,
};

struct VisualPlaceError {
  VisualPlaceErrorCode code{VisualPlaceErrorCode::InvalidRecord};
  std::string detail;
};

// Single-writer insertion with const deterministic queries. Stored records are
// immutable and bounded; adding an identical submap is idempotent.
class VisualPlaceIndex {
public:
  explicit VisualPlaceIndex(VisualVocabulary vocabulary, VisualPlaceIndexConfig config = {});
  ~VisualPlaceIndex();

  VisualPlaceIndex(VisualPlaceIndex&&) noexcept;
  VisualPlaceIndex& operator=(VisualPlaceIndex&&) noexcept;
  VisualPlaceIndex(const VisualPlaceIndex&) = delete;
  VisualPlaceIndex& operator=(const VisualPlaceIndex&) = delete;

  [[nodiscard]] core::Result<bool, VisualPlaceError> add(VisualRetrievalSubmap record);
  [[nodiscard]] core::Result<VisualRetrievalReport, VisualPlaceError> query(
      const VisualRetrievalSubmap& query, core::FusionTime now) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct MatureVisualLandmark {
  VisualPlaceLandmarkId id;
  core::Vector3d position_candidate_submap{core::Vector3d::Zero()};
  Eigen::Matrix3d covariance_candidate_submap{Eigen::Matrix3d::Identity()};
  // A sealed observing camera centre used for the final parallax gate.
  core::Vector3d reference_camera_center_candidate_submap{core::Vector3d::Zero()};
  std::uint32_t observation_count{};
  bool mature{false};
  bool dynamic{false};
  std::vector<VisualBriskDescriptor> descriptors;
};

struct QueryVisualFeature {
  VisualPlaceKeyframeId keyframe;
  core::CameraId camera;
  // Maps this camera coordinate system into the query submap.
  core::Pose3d T_query_submap_camera;
  Eigen::Vector3d bearing_camera{Eigen::Vector3d::UnitZ()};
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  std::uint32_t image_width{};
  std::uint32_t image_height{};
  VisualBriskDescriptor descriptor{};
  bool dynamic_masked{false};
};

struct VisualVerificationInput {
  core::RecordHeader header;
  ProposalId proposal;
  VisualPlaceSeed seed;
  // Payload identities are repeated deliberately: a replaceable retrieval
  // seed cannot authorize stale content revisions at verification time.
  core::SubmapRef candidate_submap;
  core::SubmapRef query_submap;
  DescriptorModelRevision candidate_model_revision;
  DescriptorModelRevision query_model_revision;
  core::ContentHash candidate_model_checksum{};
  core::ContentHash query_model_checksum{};
  VisualPlaceConfigRevision candidate_config_revision;
  VisualPlaceConfigRevision query_config_revision;
  std::vector<core::CalibrationEpoch> calibration_epochs;
  std::vector<MatureVisualLandmark> candidate_landmarks;
  std::vector<QueryVisualFeature> query_features;
  core::ObservationLineage lineage;
};

struct VisualGeometricVerifierConfig {
  std::size_t maximum_landmarks{100'000U};
  std::size_t maximum_query_features{10'000U};
  std::size_t maximum_descriptors_per_landmark{16U};
  std::size_t maximum_correspondences{4096U};
  std::size_t maximum_ransac_hypotheses{256U};
  std::size_t maximum_refinement_iterations{15U};
  std::uint32_t maximum_hamming_distance{120U};
  double maximum_ratio{0.80};
  double ransac_angular_threshold_rad{0.012};
  std::size_t minimum_ransac_inliers{16U};
  double minimum_ransac_inlier_ratio{0.40};
  std::uint32_t grid_columns{8U};
  std::uint32_t grid_rows{6U};
  double minimum_grid_coverage{0.25};
  double minimum_bearing_spread_rad{0.08};
  double minimum_median_depth_m{0.50};
  double maximum_median_depth_m{150.0};
  double minimum_median_parallax_rad{0.008};
  double robust_loss_scale_rad{0.010};
  double final_inlier_threshold_rad{0.015};
  double base_bearing_sigma_rad{0.0025};
  double covariance_inflation{4.0};
  double hessian_absolute_rank_tolerance{1.0e-8};
  double hessian_relative_rank_tolerance{1.0e-6};
  std::size_t minimum_information_rank{6U};
  double maximum_information_eigenvalue{1.0e6};
  double maximum_information_condition{1.0e10};
};

enum class VisualVerificationDisposition {
  AcceptedMetric3d2d,
  RejectedExpiredSeed,
  RejectedEndpointMismatch,
  RejectedModelMismatch,
  RejectedConfigMismatch,
  RejectedNoMatureMetricLandmarks,
  RejectedDescriptorAmbiguity,
  RejectedInsufficientCorrespondences,
  RejectedRansac,
  RejectedInlierSupport,
  RejectedGridCoverage,
  RejectedBearingDiversity,
  RejectedDepthParallax,
  RejectedRefinement,
  RejectedDegenerateInformation,
  RejectedMonocular2d2dScaleLess,
};

struct VisualVerificationReport {
  VisualPlaceConfigRevision config_revision;
  VisualVerificationDisposition disposition{
      VisualVerificationDisposition::RejectedInsufficientCorrespondences};
  std::size_t mature_landmarks{};
  std::size_t descriptor_comparisons{};
  std::size_t ratio_passes{};
  std::size_t mutual_matches{};
  std::size_t dynamic_rejections{};
  std::size_t ransac_hypotheses{};
  std::size_t ransac_inliers{};
  double ransac_inlier_ratio{};
  std::size_t final_inliers{};
  double final_inlier_ratio{};
  double grid_coverage{};
  double bearing_spread_rad{};
  double median_depth_m{};
  double median_parallax_rad{};
  double robust_cost_before{};
  double robust_cost_after{};
  std::size_t refinement_iterations{};
  // Descending in the declared right, translation-first relative tangent.
  Eigen::Matrix<double, 6, 1> hessian_eigenvalues{Eigen::Matrix<double, 6, 1>::Zero()};
  std::size_t information_rank{};
  double information_condition{};
  bool used_odometry_seed{false};
  bool retrieval_score_used_as_information{false};
};

struct VisualVerificationOutcome {
  VisualVerificationReport report;
  std::optional<LoopMeasurement> measurement;
};

// Stateless, ROS-free verifier. RANSAC candidates are produced per calibrated
// central camera only; bearings from different camera centres are never pooled
// into a central PnP. All cameras are then evaluated and refined jointly in one
// common submap-to-submap pose.
class VisualGeometricVerifier {
public:
  explicit VisualGeometricVerifier(VisualGeometricVerifierConfig config = {});

  [[nodiscard]] core::Result<VisualVerificationOutcome, VisualPlaceError> verify(
      const VisualVerificationInput& input, core::FusionTime now) const;

  // Explicitly documents the first-slice policy when only 2D--2D data exist.
  [[nodiscard]] VisualVerificationOutcome rejectScaleLessMonocular() const noexcept;

private:
  VisualGeometricVerifierConfig config_;
};

}  // namespace meridian::global
