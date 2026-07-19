#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"
#include "meridian/global/sparse_submap.hpp"

namespace meridian::global {

struct LidarDescriptorModelRevisionTag;
struct LidarPlaceConfigRevisionTag;

using LidarDescriptorModelRevision = core::StrongId<LidarDescriptorModelRevisionTag>;
using LidarPlaceConfigRevision = core::StrongId<LidarPlaceConfigRevisionTag>;

// One ORB feature extracted from a gravity-aligned, density-valued BEV. The
// position remains metric in the immutable submap frame; the image itself is
// deliberately not retained by the global index.
struct LidarBevFeature {
  Eigen::Vector2d position_submap_m{Eigen::Vector2d::Zero()};
  std::array<std::uint8_t, 32> binary_descriptor{};
  float response{};
};

struct LidarPlaceDescriptorBuildReport {
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  std::size_t input_proxy_points{};
  std::size_t valid_proxy_points{};
  std::size_t occupied_bev_cells{};
  std::size_t bev_rows{};
  std::size_t bev_columns{};
  std::size_t orb_features{};
  std::size_t self_similarity_pruned{};
  std::size_t retained_features{};
  double density_resolution_m{};
  double ground_height_submap_m{};
};

// Immutable descriptor record. Public fields make durable serialization and
// deterministic replay straightforward; production creation goes through
// buildLidarPlaceDescriptor(), which validates every bound.
struct LidarPlaceDescriptor {
  LidarPlaceDescriptor(core::RecordHeader record_header, FinalizedSubmapFrame immutable_submap)
      : header(std::move(record_header)), submap(std::move(immutable_submap)) {}

  core::RecordHeader header;
  FinalizedSubmapFrame submap;
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  core::ContentHash registration_proxy_checksum{};
  double ground_height_submap_m{};
  std::vector<LidarBevFeature> features;
  LidarPlaceDescriptorBuildReport build;
};

using ImmutableLidarPlaceDescriptor = std::shared_ptr<const LidarPlaceDescriptor>;

struct LidarPlaceDescriptorConfig {
  LidarDescriptorModelRevision model_revision{1U};
  LidarPlaceConfigRevision config_revision{1U};
  double density_resolution_m{0.50};
  double minimum_density_fraction{0.05};
  std::size_t minimum_proxy_points{80U};
  std::size_t maximum_proxy_points{100'000U};
  std::size_t maximum_bev_cells{1'000'000U};
  std::size_t maximum_features{500U};
  std::size_t minimum_features{8U};
  int orb_edge_threshold_pixels{15};
  int orb_patch_size_pixels{31};
  int orb_fast_threshold{20};
  int self_similarity_hamming_threshold{35};
};

enum class LidarPlaceDescriptorErrorCode {
  InvalidConfiguration,
  InvalidHeader,
  InvalidSubmap,
  InvalidRegistrationProxy,
  ProxyCapacity,
  BevCapacity,
  InsufficientSupport,
  NumericalFailure,
};

struct LidarPlaceDescriptorError {
  LidarPlaceDescriptorErrorCode code{LidarPlaceDescriptorErrorCode::InvalidConfiguration};
  std::string detail;
};

[[nodiscard]] core::Result<ImmutableLidarPlaceDescriptor, LidarPlaceDescriptorError>
buildLidarPlaceDescriptor(const core::RecordHeader& header, const FinalizedSubmapFrame& submap,
                          const RegistrationProxy& registration_proxy,
                          const LidarPlaceDescriptorConfig& config = {});

struct LidarPlaceEntryKey {
  core::SubmapRef submap;

  auto operator<=>(const LidarPlaceEntryKey&) const = default;
};

[[nodiscard]] LidarPlaceEntryKey lidarPlaceEntryKey(
    const LidarPlaceDescriptor& descriptor) noexcept;

struct LidarPlaceIndexConfig {
  LidarDescriptorModelRevision model_revision{1U};
  LidarPlaceConfigRevision config_revision{1U};
  std::size_t maximum_entries{4096U};
  std::size_t maximum_total_features{1'500'000U};
  std::size_t maximum_features_per_entry{500U};
  std::size_t maximum_index_references{24'000'000U};
  std::size_t maximum_descriptor_comparisons_per_query{2'000'000U};
  std::size_t maximum_ransac_candidates{32U};
  std::size_t maximum_ransac_trials_per_candidate{512U};
  std::size_t maximum_top_k{10U};
  std::size_t minimum_descriptor_matches{8U};
  std::size_t minimum_ransac_inliers{6U};
  int maximum_hamming_distance{50};
  double minimum_ransac_inlier_ratio{0.20};
  double minimum_ransac_baseline_m{1.5};
  double ransac_inlier_distance_m{1.5};
  core::Duration minimum_temporal_separation{15'000'000'000LL};
  core::Duration candidate_ttl{30'000'000'000LL};
};

enum class LidarPlaceIndexInsertDisposition {
  Inserted,
  AlreadyPresent,
};

struct LidarPlaceIndexInsertReport {
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  LidarPlaceEntryKey key;
  LidarPlaceIndexInsertDisposition disposition{LidarPlaceIndexInsertDisposition::Inserted};
  std::size_t entries{};
  std::size_t indexed_features{};
  std::size_t index_references{};
};

struct LidarPlaceRetrievalRequest {
  LidarPlaceRetrievalRequest(ImmutableLidarPlaceDescriptor query_descriptor,
                             core::FusionTime query_time)
      : query(std::move(query_descriptor)), now(query_time) {}

  ImmutableLidarPlaceDescriptor query;
  core::FusionTime now;
  std::size_t top_k{5U};
  // Same, overlapping, covisible, and adjacent submaps are supplied by the
  // owner explicitly. The index additionally enforces identity and temporal
  // separation without assuming sequential integer IDs.
  std::vector<LidarPlaceEntryKey> excluded;
};

struct LidarRetrievalSeed {
  LidarRetrievalSeed(core::RecordHeader record_header, core::SubmapRef candidate_submap,
                     core::SubmapRef query_submap)
      : header(std::move(record_header)),
        from(std::move(candidate_submap)),
        to(std::move(query_submap)) {}

  core::RecordHeader header;
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  core::SubmapRef from;
  core::SubmapRef to;
  core::ContentHash from_proxy_checksum{};
  core::ContentHash to_proxy_checksum{};
  core::Pose3d T_from_to_seed;
  core::FusionTime valid_until;
  std::size_t descriptor_matches{};
  std::size_t ransac_inliers{};
  double ransac_inlier_ratio{};
  double ransac_rmse_m{};
  // Scheduling evidence only. It is never consumed by the geometric
  // verifier's information calculation or by the global graph.
  double retrieval_score{};
};

struct LidarPlaceCandidateReport {
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  LidarPlaceEntryKey candidate;
  std::size_t descriptor_matches{};
  std::size_t ransac_inliers{};
  double ransac_inlier_ratio{};
  double ransac_rmse_m{};
  bool passed_geometric_seed_gate{false};
};

struct LidarPlaceRetrievalReport {
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  LidarPlaceEntryKey query;
  core::FusionTime evaluated_at;
  core::FusionTime candidates_valid_until;
  std::size_t requested_top_k{};
  std::size_t emitted_top_k{};
  std::size_t indexed_entries{};
  std::size_t entries_examined{};
  std::size_t excluded_same_identity{};
  std::size_t excluded_other_session{};
  std::size_t excluded_explicit_policy{};
  std::size_t excluded_temporal_separation{};
  std::size_t multi_index_bucket_references{};
  std::size_t exact_hamming_comparisons{};
  std::size_t entries_with_descriptor_votes{};
  std::size_t ransac_candidates_evaluated{};
  std::vector<LidarPlaceCandidateReport> candidates;
};

struct LidarPlaceRetrievalResult {
  std::vector<LidarRetrievalSeed> seeds;
  LidarPlaceRetrievalReport report;
};

enum class LidarPlaceIndexErrorCode {
  InvalidConfiguration,
  InvalidDescriptor,
  ModelRevisionMismatch,
  ConfigRevisionMismatch,
  IdentityConflict,
  EntryCapacity,
  FeatureCapacity,
  IndexReferenceCapacity,
  InvalidQuery,
  QueryResourceLimit,
};

struct LidarPlaceIndexError {
  LidarPlaceIndexErrorCode code{LidarPlaceIndexErrorCode::InvalidConfiguration};
  std::optional<LidarPlaceEntryKey> key;
  std::string detail;
};

// Bounded single-writer descriptor database. The 16-table binary multi-index
// is Meridian-owned and deterministic; it replaces MapClosures' unavailable
// FetchContent-only HBST dependency while preserving exact Hamming validation.
class LidarPlaceIndex {
public:
  explicit LidarPlaceIndex(LidarPlaceIndexConfig config = {});
  ~LidarPlaceIndex();

  LidarPlaceIndex(LidarPlaceIndex&&) noexcept;
  LidarPlaceIndex& operator=(LidarPlaceIndex&&) noexcept;
  LidarPlaceIndex(const LidarPlaceIndex&) = delete;
  LidarPlaceIndex& operator=(const LidarPlaceIndex&) = delete;

  [[nodiscard]] core::Result<LidarPlaceIndexInsertReport, LidarPlaceIndexError> insert(
      ImmutableLidarPlaceDescriptor descriptor);

  [[nodiscard]] core::Result<LidarPlaceRetrievalResult, LidarPlaceIndexError> retrieve(
      const LidarPlaceRetrievalRequest& request) const;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t featureCount() const noexcept;
  [[nodiscard]] const LidarPlaceIndexConfig& config() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class DynamicProxyLabelStatus {
  UnavailableNoSemanticLabel,
};

struct LidarLoopVerifierConfig {
  LidarDescriptorModelRevision model_revision{1U};
  LidarPlaceConfigRevision config_revision{1U};
  std::size_t maximum_proxy_points_per_submap{100'000U};
  std::size_t minimum_proxy_points_per_submap{80U};
  std::size_t maximum_iterations{20U};
  std::size_t maximum_damping_trials{8U};
  std::size_t minimum_correspondences_each_direction{50U};
  std::size_t support_grid_dimension{4U};
  std::size_t minimum_support_cells_each_direction{6U};
  std::size_t minimum_information_rank{3U};
  double maximum_correspondence_distance_m{1.5};
  double minimum_bidirectional_overlap{0.30};
  double minimum_normal_consistency{0.65};
  double minimum_pair_normal_cosine{0.35};
  double maximum_residual_median_m{0.30};
  double maximum_residual_quantile_m{0.75};
  double residual_upper_quantile{0.90};
  double normal_standard_deviation_m{0.05};
  double tangential_standard_deviation_m{0.20};
  double model_standard_deviation_m{0.04};
  double huber_delta_sigma{2.5};
  double initial_relative_damping{1.0e-3};
  double damping_increase{10.0};
  double damping_decrease{0.25};
  double translation_convergence_m{1.0e-4};
  double rotation_convergence_rad{1.0e-4};
  double maximum_iteration_translation_step_m{1.0};
  double maximum_iteration_rotation_step_rad{0.35};
  double maximum_seed_translation_correction_m{8.0};
  double maximum_seed_rotation_correction_rad{1.0};
  double maximum_gravity_tilt_rad{0.35};
  double absolute_observable_eigenvalue{1.0e-6};
  double relative_observable_eigenvalue{1.0e-4};
  double maximum_information_eigenvalue{1.0e5};
  // Final normal-projected correspondences are spatially correlated. This
  // fixed calibrated seed prevents raw point count from becoming confidence.
  double correspondence_information_inflation{4.0};
  double maximum_proxy_weight{1.0e6};
};

enum class LidarLoopGate {
  BidirectionalOverlap,
  SpatialSupport,
  ResidualMedian,
  ResidualUpperQuantile,
  NormalConsistency,
  SeedCorrection,
  GravityAlignment,
  InformationRank,
};

struct LidarLoopGateResult {
  LidarLoopGate gate{LidarLoopGate::BidirectionalOverlap};
  bool passed{false};
  double measured{};
  double threshold{};
};

enum class LidarLoopVerificationDisposition {
  Accepted,
  RejectedGeometricGates,
  RejectedDegenerate,
  RejectedDidNotConverge,
};

struct LidarLoopVerificationReport {
  LidarDescriptorModelRevision model_revision;
  LidarPlaceConfigRevision config_revision;
  LidarLoopVerificationDisposition disposition{
      LidarLoopVerificationDisposition::RejectedGeometricGates};
  std::size_t from_proxy_points{};
  std::size_t to_proxy_points{};
  std::size_t iterations{};
  std::size_t accepted_steps{};
  std::size_t rejected_steps{};
  std::size_t forward_correspondences{};
  std::size_t reverse_correspondences{};
  std::size_t unique_correspondences{};
  std::size_t normal_consistent_correspondences{};
  std::size_t forward_support_cells{};
  std::size_t reverse_support_cells{};
  double initial_cost{};
  double final_cost{};
  double forward_overlap{};
  double reverse_overlap{};
  double residual_median_m{};
  double residual_upper_quantile_m{};
  double normal_consistency{};
  std::optional<double> dynamic_fraction;
  DynamicProxyLabelStatus dynamic_label_status{DynamicProxyLabelStatus::UnavailableNoSemanticLabel};
  double seed_translation_correction_m{};
  double seed_rotation_correction_rad{};
  double final_gravity_tilt_rad{};
  double calibrated_normal_residual_sigma_m{};
  Eigen::Matrix<double, 6, 1> raw_information_eigenvalues{Eigen::Matrix<double, 6, 1>::Zero()};
  core::RankAwareInformation information;
  std::vector<LidarLoopGateResult> gates;
};

struct LidarLoopVerificationRequest {
  LidarLoopVerificationRequest(core::RecordHeader record_header, ProposalId proposal_id,
                               LidarRetrievalSeed retrieval_seed)
      : header(std::move(record_header)), proposal(proposal_id), seed(std::move(retrieval_seed)) {}

  core::RecordHeader header;
  ProposalId proposal;
  LidarRetrievalSeed seed;
  core::FusionTime evaluated_at;
  std::vector<core::CalibrationEpoch> calibration_epochs;
  core::ObservationLineage lineage;
};

struct VerifiedLidarLoop {
  VerifiedLidarLoop(LidarRetrievalSeed retrieval_seed,
                    LidarLoopVerificationReport verification_report)
      : seed(std::move(retrieval_seed)), report(std::move(verification_report)) {}

  LidarRetrievalSeed seed;
  LidarLoopVerificationReport report;
  std::optional<LoopMeasurement> measurement;
};

enum class LidarLoopVerifierErrorCode {
  InvalidConfiguration,
  InvalidRequest,
  SeedExpired,
  RevisionMismatch,
  EndpointMismatch,
  ProxyChecksumMismatch,
  InvalidRegistrationProxy,
  ProxyCapacity,
  InvalidLineage,
  NumericalFailure,
};

struct LidarLoopVerifierError {
  LidarLoopVerifierErrorCode code{LidarLoopVerifierErrorCode::InvalidConfiguration};
  std::string detail;
};

// Independent loop verifier. It never reads current corrected anchor distance
// and never derives information from retrieval score. Final correspondences
// are rebuilt at the accepted pose before the graph-ready measurement is made.
[[nodiscard]] core::Result<VerifiedLidarLoop, LidarLoopVerifierError> verifyLidarLoop(
    const LidarLoopVerificationRequest& request, const RegistrationProxy& from_proxy,
    const RegistrationProxy& to_proxy, const LidarLoopVerifierConfig& config = {});

}  // namespace meridian::global
