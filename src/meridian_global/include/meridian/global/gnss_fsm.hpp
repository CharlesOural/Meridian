#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "meridian/core/result.hpp"
#include "meridian/core/strong_id.hpp"
#include "meridian/core/time.hpp"
#include "meridian/global/gnss.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

// GNSS is a global-backend modality. This state machine consumes finalized
// global proposal inputs and can only release global graph batches; it has no
// API capable of changing local odometry or feeding a pose back to it.
enum class GnssFsmState {
  Uninitialized,
  Aligning,
  Trusted,
  Suspect,
  Reacquiring,
  Unavailable,
};

enum class GnssFsmDisposition {
  NoObservation,
  RejectedCurrent,
  BufferedForAlignment,
  QuarantinedCurrent,
  AdmitTrustedCurrent,
  ShadowRequested,
  ReleaseGraphBatch,
};

enum class GnssFsmReason {
  None,
  StreamStarted,
  StreamReturnedWithoutAlignment,
  StreamReturnedWithAlignment,
  AwaitingQualityWindow,
  AwaitingAlignmentExcitation,
  AlignmentObservable,
  TrustedNisAccepted,
  TrustedNisUnavailable,
  TrustedNisIndeterminate,
  TrustedNisRejected,
  ReceiverSolutionRejected,
  ReceiverIntegrityRejected,
  ReceiverCorrectionRejected,
  ReceiverMetadataUnavailable,
  SatelliteCountRejected,
  HdopRejected,
  CorrectionAgeRejected,
  ReceiverCovarianceRejected,
  QualityFailureStreak,
  QualifiedWindowStarted,
  QualifiedWindowInterrupted,
  QualifiedWindowGap,
  ShadowAlreadyPending,
  ShadowValidationPassed,
  ShadowOptimizationFailed,
  ShadowGraphValidationFailed,
  ShadowResourceLimitFailed,
  ShadowPostSolveNisRejected,
  StreamTimedOut,
};

struct GnssShadowAttemptIdTag;
using GnssShadowAttemptId = core::StrongId<GnssShadowAttemptIdTag>;

enum class GnssShadowKind {
  InitialAlignment,
  Reacquisition,
};

// Receiver metadata is never inferred. Optional requirements below mean
// "require the receiver to report this field"; when disabled, a field that is
// actually present is still checked for validity.
struct GnssReceiverGateConfig {
  ReceiverReportedSolution minimum_solution{ReceiverReportedSolution::Autonomous};
  ReceiverIntegrity required_integrity{ReceiverIntegrity::Valid};
  std::optional<std::uint32_t> minimum_satellites{6U};
  std::optional<double> maximum_hdop{3.0};
  std::optional<double> maximum_correction_age_s{2.0};
  bool require_current_corrections_for_corrected_solutions{true};
  double maximum_horizontal_position_std_m{5.0};
  double maximum_vertical_position_std_m{8.0};
  double maximum_receiver_covariance_condition{1.0e8};
};

struct GnssFsmConfig {
  GnssReceiverGateConfig receiver;
  AlignmentConfig alignment;

  // Every uncommitted window is hard bounded. Configuration should leave
  // enough room for the excitation checks in AlignmentConfig.
  std::size_t maximum_buffered_observations{64U};
  std::size_t consecutive_failures_to_suspect{3U};
  std::size_t consecutive_trusted_accepts_to_clear_failures{2U};
  std::size_t consecutive_post_solve_accepts{5U};

  core::Duration outage_timeout{2'000'000'000LL};
  core::Duration maximum_qualified_sample_gap{1'000'000'000LL};

  // Three-dimensional position NIS hysteresis. Values below the acceptance
  // gate are admitted, values above the rejection gate count toward SUSPECT,
  // and the band between them is quarantined without increasing the streak.
  double trusted_nis_accept_threshold{11.34486673};  // chi2(3), 99%
  double trusted_nis_reject_threshold{16.26623620};  // chi2(3), 99.9%
  double post_solve_nis_accept_threshold{11.34486673};
  double post_solve_nis_reject_threshold{16.26623620};
};

// The correspondence and graph constraint must have been constructed from
// the same exact-time finalized trajectory association. The FSM deliberately
// does not perform nearest-pose association or WGS84 conversion.
struct GnssFsmObservation {
  GnssQuality quality;
  Eigen::Matrix3d receiver_covariance_enu{Eigen::Matrix3d::Identity()};
  AlignmentCorrespondence alignment_correspondence;
  GnssAntennaConstraint graph_constraint;
  // Present only when a committed alignment and anchor prediction exist.
  std::optional<double> committed_alignment_nis;

  GnssFsmObservation(GnssQuality quality_in, Eigen::Matrix3d receiver_covariance_enu_in,
                     AlignmentCorrespondence alignment_correspondence_in,
                     GnssAntennaConstraint graph_constraint_in,
                     std::optional<double> committed_alignment_nis_in = std::nullopt)
      : quality(std::move(quality_in)),
        receiver_covariance_enu(std::move(receiver_covariance_enu_in)),
        alignment_correspondence(std::move(alignment_correspondence_in)),
        graph_constraint(std::move(graph_constraint_in)),
        committed_alignment_nis(committed_alignment_nis_in) {}
};

struct GnssObservationNis {
  core::GnssObservationId observation;
  double nis{};
};

// This is a candidate for a complete bounded graph shadow transaction. The
// robust fit is only a seed and covariance report; information still enters
// exactly once through candidate_batch.constraints.
struct GnssShadowRequest {
  GnssShadowAttemptId attempt;
  GnssShadowKind kind{GnssShadowKind::InitialAlignment};
  core::TimeRange source_window;
  YawTranslation4 alignment_seed;
  AlignmentCovariance alignment_covariance;
  AlignmentDiagnostics alignment_diagnostics;
  GnssBatchAppend candidate_batch;
};

struct GnssShadowValidation {
  GnssShadowAttemptId attempt;
  core::FusionTime completed_at;
  bool optimizer_converged{false};
  bool graph_validation_passed{false};
  bool resource_limits_satisfied{false};
  // Exact candidate order and identity are required. This makes the
  // consecutive post-solve consistency check deterministic.
  std::vector<GnssObservationNis> post_solve_nis;
};

struct GnssFsmTransition {
  std::uint64_t sequence{};
  GnssFsmState from{GnssFsmState::Uninitialized};
  GnssFsmState to{GnssFsmState::Uninitialized};
  GnssFsmReason reason{GnssFsmReason::None};
  core::FusionTime at;
  std::optional<core::GnssObservationId> trigger;
};

struct GnssFsmSnapshot {
  GnssFsmState state{GnssFsmState::Uninitialized};
  bool alignment_committed{false};
  bool quarantine_active{false};
  std::optional<core::FusionTime> quarantine_start;
  std::optional<core::FusionTime> quarantine_latest;
  std::optional<core::FusionTime> last_observation_time;
  std::optional<core::FusionTime> last_admitted_time;
  std::optional<core::GnssObservationId> last_admitted_observation;
  std::optional<GnssShadowAttemptId> pending_shadow;
  std::size_t buffered_observations{};
  std::size_t consecutive_failures{};
  std::size_t consecutive_trusted_accepts{};
  std::uint64_t transition_sequence{};
  std::uint64_t capacity_resets{};
};

struct GnssFsmReport {
  GnssFsmDisposition disposition{GnssFsmDisposition::NoObservation};
  GnssFsmReason reason{GnssFsmReason::None};
  core::FusionTime at;
  std::optional<core::GnssObservationId> observation;
  std::optional<double> nis;
  std::optional<AlignmentErrorCode> alignment_wait_reason;
  std::vector<GnssFsmTransition> transitions;
  std::optional<GnssShadowRequest> shadow_request;
  std::optional<GnssBatchAppend> released_batch;
  GnssFsmSnapshot snapshot;
};

enum class GnssFsmErrorCode {
  InvalidConfiguration,
  InvalidObservationIdentity,
  InvalidObservationAssociation,
  NonFiniteInput,
  InvalidCovariance,
  NonMonotonicTime,
  CapacityExceeded,
  InvalidShadowAttempt,
  ShadowResultIdentityMismatch,
};

struct GnssFsmError {
  GnssFsmErrorCode code{GnssFsmErrorCode::NonFiniteInput};
  std::string detail;
  GnssFsmState state{GnssFsmState::Uninitialized};
  std::size_t buffered_observations{};
  bool uncommitted_window_cleared{false};
};

class GnssAdmissionFsm {
public:
  [[nodiscard]] static core::Result<GnssAdmissionFsm, GnssFsmError> create(
      GnssFsmConfig config = {});

  [[nodiscard]] core::Result<GnssFsmReport, GnssFsmError> observe(
      const GnssFsmObservation& observation);
  [[nodiscard]] core::Result<GnssFsmReport, GnssFsmError> advanceTime(core::FusionTime now);
  [[nodiscard]] core::Result<GnssFsmReport, GnssFsmError> resolveShadow(
      const GnssShadowValidation& validation);

  [[nodiscard]] GnssFsmState state() const noexcept { return state_; }
  [[nodiscard]] GnssFsmSnapshot snapshot() const;

private:
  struct BufferedObservation {
    core::GnssObservationId id;
    core::FusionTime stamp;
    AlignmentCorrespondence alignment;
    GnssAntennaConstraint factor;
  };

  struct PendingShadow {
    GnssShadowRequest request;
  };

  explicit GnssAdmissionFsm(GnssFsmConfig config) : config_(std::move(config)) {}

  [[nodiscard]] GnssFsmError error(GnssFsmErrorCode code, std::string detail,
                                   bool window_cleared = false) const;
  [[nodiscard]] GnssFsmReport report(GnssFsmDisposition disposition, GnssFsmReason reason,
                                     core::FusionTime at,
                                     std::optional<core::GnssObservationId> observation,
                                     std::vector<GnssFsmTransition> transitions = {}) const;
  void transitionTo(GnssFsmState next, GnssFsmReason reason, core::FusionTime at,
                    std::optional<core::GnssObservationId> trigger,
                    std::vector<GnssFsmTransition>& transitions);
  void beginQuarantine(core::FusionTime at) noexcept;
  void clearUncommittedWindow() noexcept;

  GnssFsmConfig config_;
  GnssFsmState state_{GnssFsmState::Uninitialized};
  bool alignment_committed_{false};
  bool quarantine_active_{false};
  std::optional<core::FusionTime> quarantine_start_;
  std::optional<core::FusionTime> quarantine_latest_;
  std::optional<core::FusionTime> last_observation_time_;
  std::optional<core::FusionTime> last_event_time_;
  std::optional<core::FusionTime> last_admitted_time_;
  std::optional<core::GnssObservationId> last_admitted_observation_;
  std::deque<BufferedObservation> window_;
  std::optional<PendingShadow> pending_shadow_;
  std::size_t consecutive_failures_{};
  std::size_t consecutive_trusted_accepts_{};
  std::uint64_t transition_sequence_{};
  std::uint64_t next_shadow_attempt_{1U};
  std::uint64_t capacity_resets_{};
};

}  // namespace meridian::global
