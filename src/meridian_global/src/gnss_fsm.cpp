#include "meridian/global/gnss_fsm.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

namespace meridian::global {
namespace {

using Result = core::Result<GnssFsmReport, GnssFsmError>;

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] int solutionRank(ReceiverReportedSolution solution) noexcept {
  switch (solution) {
    case ReceiverReportedSolution::Unknown:
    case ReceiverReportedSolution::NoFix:
      return 0;
    case ReceiverReportedSolution::Autonomous:
      return 1;
    case ReceiverReportedSolution::Differential:
      return 2;
    case ReceiverReportedSolution::RtkFloat:
    case ReceiverReportedSolution::PppFloat:
      return 3;
    case ReceiverReportedSolution::RtkFixed:
    case ReceiverReportedSolution::PppFixed:
      return 4;
  }
  return 0;
}

[[nodiscard]] bool correctedSolution(ReceiverReportedSolution solution) noexcept {
  return solution == ReceiverReportedSolution::Differential ||
         solution == ReceiverReportedSolution::RtkFloat ||
         solution == ReceiverReportedSolution::RtkFixed ||
         solution == ReceiverReportedSolution::PppFloat ||
         solution == ReceiverReportedSolution::PppFixed;
}

struct CovarianceCheck {
  bool valid{false};
  double condition{std::numeric_limits<double>::infinity()};
};

[[nodiscard]] CovarianceCheck checkCovariance(const Eigen::Matrix3d& covariance) {
  if (!covariance.allFinite()) {
    return {};
  }
  const double scale = std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > 1.0e-10 * scale) {
    return {};
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(0.5 * (covariance + covariance.transpose()),
                                                        Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
      !(solver.eigenvalues().minCoeff() > 0.0)) {
    return {};
  }
  return CovarianceCheck{true, solver.eigenvalues().maxCoeff() / solver.eigenvalues().minCoeff()};
}

[[nodiscard]] bool samePosition(const Eigen::Vector3d& first,
                                const Eigen::Vector3d& second) noexcept {
  const double scale = std::max({1.0, first.norm(), second.norm()});
  return (first - second).norm() <= 1.0e-10 * scale;
}

[[nodiscard]] bool validReceiverConfiguration(const GnssReceiverGateConfig& config) {
  if (solutionRank(config.minimum_solution) == 0 ||
      !finitePositive(config.maximum_horizontal_position_std_m) ||
      !finitePositive(config.maximum_vertical_position_std_m) ||
      !finitePositive(config.maximum_receiver_covariance_condition) ||
      config.maximum_receiver_covariance_condition < 1.0) {
    return false;
  }
  if (config.minimum_satellites && *config.minimum_satellites == 0U) {
    return false;
  }
  if (config.maximum_hdop && !finitePositive(*config.maximum_hdop)) {
    return false;
  }
  if (config.maximum_correction_age_s && !finitePositive(*config.maximum_correction_age_s)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool validConfiguration(const GnssFsmConfig& config) {
  const auto alignment_probe =
      estimateGravityAlignedTransform(std::span<const AlignmentCorrespondence>{}, config.alignment);
  const bool alignment_config_valid =
      !alignment_probe && alignment_probe.error().code == AlignmentErrorCode::TooFewCorrespondences;
  return alignment_config_valid && validReceiverConfiguration(config.receiver) &&
         config.maximum_buffered_observations >= config.alignment.minimum_correspondences &&
         config.maximum_buffered_observations >= config.consecutive_post_solve_accepts &&
         config.consecutive_failures_to_suspect > 0U &&
         config.consecutive_trusted_accepts_to_clear_failures > 0U &&
         config.consecutive_post_solve_accepts > 0U && config.outage_timeout.nanoseconds > 0 &&
         config.maximum_qualified_sample_gap.nanoseconds > 0 &&
         finitePositive(config.trusted_nis_accept_threshold) &&
         finitePositive(config.trusted_nis_reject_threshold) &&
         config.trusted_nis_accept_threshold < config.trusted_nis_reject_threshold &&
         finitePositive(config.post_solve_nis_accept_threshold) &&
         finitePositive(config.post_solve_nis_reject_threshold) &&
         config.post_solve_nis_accept_threshold < config.post_solve_nis_reject_threshold;
}

[[nodiscard]] std::optional<GnssFsmReason> receiverRejection(const GnssFsmObservation& observation,
                                                             const GnssReceiverGateConfig& config,
                                                             double receiver_covariance_condition) {
  const auto& quality = observation.quality;
  if (solutionRank(quality.solution()) < solutionRank(config.minimum_solution)) {
    return GnssFsmReason::ReceiverSolutionRejected;
  }
  if (quality.integrity() != config.required_integrity) {
    return GnssFsmReason::ReceiverIntegrityRejected;
  }

  if (correctedSolution(quality.solution()) &&
      config.require_current_corrections_for_corrected_solutions &&
      quality.corrections() != CorrectionStreamState::Current) {
    return quality.corrections() == CorrectionStreamState::Unknown
               ? GnssFsmReason::ReceiverMetadataUnavailable
               : GnssFsmReason::ReceiverCorrectionRejected;
  }
  if (quality.corrections() == CorrectionStreamState::Stale) {
    return GnssFsmReason::ReceiverCorrectionRejected;
  }

  if (config.minimum_satellites) {
    if (!quality.satellites()) {
      return GnssFsmReason::ReceiverMetadataUnavailable;
    }
    if (*quality.satellites() < *config.minimum_satellites) {
      return GnssFsmReason::SatelliteCountRejected;
    }
  }
  if (config.maximum_hdop) {
    if (!quality.hdop()) {
      return GnssFsmReason::ReceiverMetadataUnavailable;
    }
    if (*quality.hdop() > *config.maximum_hdop) {
      return GnssFsmReason::HdopRejected;
    }
  }
  if (config.maximum_correction_age_s && correctedSolution(quality.solution())) {
    if (!quality.correctionAgeS()) {
      return GnssFsmReason::ReceiverMetadataUnavailable;
    }
    if (*quality.correctionAgeS() > *config.maximum_correction_age_s) {
      return GnssFsmReason::CorrectionAgeRejected;
    }
  }

  const Eigen::Vector3d standard_deviation =
      observation.receiver_covariance_enu.diagonal().cwiseSqrt();
  if (standard_deviation.head<2>().maxCoeff() > config.maximum_horizontal_position_std_m ||
      standard_deviation.z() > config.maximum_vertical_position_std_m ||
      receiver_covariance_condition > config.maximum_receiver_covariance_condition) {
    return GnssFsmReason::ReceiverCovarianceRejected;
  }
  return std::nullopt;
}

[[nodiscard]] bool expectedAlignmentWait(AlignmentErrorCode code) noexcept {
  return code == AlignmentErrorCode::TooFewCorrespondences ||
         code == AlignmentErrorCode::InsufficientTimeSpan ||
         code == AlignmentErrorCode::InsufficientHorizontalExcitation ||
         code == AlignmentErrorCode::IllConditioned || code == AlignmentErrorCode::DidNotConverge;
}

}  // namespace

core::Result<GnssAdmissionFsm, GnssFsmError> GnssAdmissionFsm::create(GnssFsmConfig config) {
  if (!validConfiguration(config)) {
    return core::Result<GnssAdmissionFsm, GnssFsmError>::failure(GnssFsmError{
        GnssFsmErrorCode::InvalidConfiguration,
        "GNSS FSM bounds, receiver gates, alignment fit, durations, or NIS hysteresis are invalid",
        GnssFsmState::Uninitialized, 0U, false});
  }
  return core::Result<GnssAdmissionFsm, GnssFsmError>::success(GnssAdmissionFsm(std::move(config)));
}

GnssFsmError GnssAdmissionFsm::error(GnssFsmErrorCode code, std::string detail,
                                     bool window_cleared) const {
  return GnssFsmError{code, std::move(detail), state_, window_.size(), window_cleared};
}

GnssFsmSnapshot GnssAdmissionFsm::snapshot() const {
  return GnssFsmSnapshot{
      state_,
      alignment_committed_,
      quarantine_active_,
      quarantine_start_,
      quarantine_latest_,
      last_observation_time_,
      last_admitted_time_,
      last_admitted_observation_,
      pending_shadow_ ? std::optional(pending_shadow_->request.attempt) : std::nullopt,
      window_.size(),
      consecutive_failures_,
      consecutive_trusted_accepts_,
      transition_sequence_,
      capacity_resets_};
}

GnssFsmReport GnssAdmissionFsm::report(GnssFsmDisposition disposition, GnssFsmReason reason,
                                       core::FusionTime at,
                                       std::optional<core::GnssObservationId> observation,
                                       std::vector<GnssFsmTransition> transitions) const {
  GnssFsmReport result;
  result.disposition = disposition;
  result.reason = reason;
  result.at = at;
  result.observation = observation;
  result.transitions = std::move(transitions);
  result.snapshot = snapshot();
  return result;
}

void GnssAdmissionFsm::transitionTo(GnssFsmState next, GnssFsmReason reason, core::FusionTime at,
                                    std::optional<core::GnssObservationId> trigger,
                                    std::vector<GnssFsmTransition>& transitions) {
  if (state_ == next) {
    return;
  }
  const GnssFsmState previous = state_;
  state_ = next;
  transitions.push_back(
      GnssFsmTransition{++transition_sequence_, previous, next, reason, at, trigger});
}

void GnssAdmissionFsm::beginQuarantine(core::FusionTime at) noexcept {
  if (!quarantine_active_) {
    quarantine_start_ = at;
    quarantine_active_ = true;
  }
  quarantine_latest_ = at;
}

void GnssAdmissionFsm::clearUncommittedWindow() noexcept {
  window_.clear();
  pending_shadow_.reset();
}

core::Result<GnssFsmReport, GnssFsmError> GnssAdmissionFsm::observe(
    const GnssFsmObservation& observation) {
  const auto id = observation.graph_constraint.observation;
  const core::FusionTime stamp = observation.alignment_correspondence.stamp;
  if (!id.valid() ||
      core::validateSubmapRef(observation.graph_constraint.submap) !=
          core::SubmapRefValidationError::None) {
    return Result::failure(error(GnssFsmErrorCode::InvalidObservationIdentity,
                                 "GNSS observation or finalized submap identity is invalid"));
  }
  if (!observation.alignment_correspondence.antenna_position_map.allFinite() ||
      !observation.alignment_correspondence.antenna_position_enu.allFinite() ||
      !observation.graph_constraint.antenna_position_submap.allFinite() ||
      !observation.graph_constraint.measured_position_enu.allFinite() ||
      (observation.committed_alignment_nis &&
       (!std::isfinite(*observation.committed_alignment_nis) ||
        *observation.committed_alignment_nis < 0.0))) {
    return Result::failure(
        error(GnssFsmErrorCode::NonFiniteInput, "GNSS proposal or NIS is non-finite"));
  }
  if (!samePosition(observation.alignment_correspondence.antenna_position_enu,
                    observation.graph_constraint.measured_position_enu)) {
    return Result::failure(
        error(GnssFsmErrorCode::InvalidObservationAssociation,
              "alignment correspondence and graph factor do not carry the same ENU observation"));
  }

  const CovarianceCheck receiver_covariance = checkCovariance(observation.receiver_covariance_enu);
  const CovarianceCheck alignment_covariance =
      checkCovariance(observation.alignment_correspondence.covariance_enu);
  const CovarianceCheck effective_covariance =
      checkCovariance(observation.graph_constraint.effective_covariance_enu);
  if (!receiver_covariance.valid || !alignment_covariance.valid || !effective_covariance.valid) {
    return Result::failure(error(
        GnssFsmErrorCode::InvalidCovariance,
        "receiver, alignment-residual, and graph-factor covariances must be positive definite"));
  }
  if ((observation.quality.hdop() &&
       (!std::isfinite(*observation.quality.hdop()) || *observation.quality.hdop() < 0.0)) ||
      (observation.quality.correctionAgeS() &&
       (!std::isfinite(*observation.quality.correctionAgeS()) ||
        *observation.quality.correctionAgeS() < 0.0))) {
    return Result::failure(
        error(GnssFsmErrorCode::NonFiniteInput, "reported HDOP or correction age is invalid"));
  }
  if ((last_observation_time_ && stamp <= *last_observation_time_) ||
      (last_event_time_ && stamp < *last_event_time_)) {
    return Result::failure(
        error(GnssFsmErrorCode::NonMonotonicTime, "GNSS observation time is not strictly ordered"));
  }

  std::vector<GnssFsmTransition> transitions;
  const bool observation_gap =
      last_observation_time_ &&
      (stamp - *last_observation_time_).nanoseconds >= config_.outage_timeout.nanoseconds;
  if (observation_gap) {
    clearUncommittedWindow();
    if (alignment_committed_) {
      beginQuarantine(stamp);
    }
    transitionTo(GnssFsmState::Unavailable, GnssFsmReason::StreamTimedOut, stamp, id, transitions);
  }

  if (state_ == GnssFsmState::Uninitialized) {
    transitionTo(GnssFsmState::Aligning, GnssFsmReason::StreamStarted, stamp, id, transitions);
  } else if (state_ == GnssFsmState::Unavailable) {
    const GnssFsmState next =
        alignment_committed_ ? GnssFsmState::Reacquiring : GnssFsmState::Aligning;
    const GnssFsmReason reason = alignment_committed_
                                     ? GnssFsmReason::StreamReturnedWithAlignment
                                     : GnssFsmReason::StreamReturnedWithoutAlignment;
    transitionTo(next, reason, stamp, id, transitions);
    if (alignment_committed_) {
      beginQuarantine(stamp);
    }
  }
  last_observation_time_ = stamp;
  last_event_time_ = stamp;

  if (const auto rejection =
          receiverRejection(observation, config_.receiver, receiver_covariance.condition)) {
    if (state_ == GnssFsmState::Trusted) {
      beginQuarantine(stamp);
      ++consecutive_failures_;
      consecutive_trusted_accepts_ = 0U;
      if (consecutive_failures_ >= config_.consecutive_failures_to_suspect) {
        transitionTo(GnssFsmState::Suspect, GnssFsmReason::QualityFailureStreak, stamp, id,
                     transitions);
      }
    } else if (state_ == GnssFsmState::Reacquiring) {
      beginQuarantine(stamp);
      window_.clear();
      transitionTo(GnssFsmState::Suspect, GnssFsmReason::QualifiedWindowInterrupted, stamp, id,
                   transitions);
    } else if (state_ == GnssFsmState::Aligning) {
      window_.clear();
    } else if (state_ == GnssFsmState::Suspect) {
      beginQuarantine(stamp);
      window_.clear();
    }
    const auto disposition = state_ == GnssFsmState::Aligning
                                 ? GnssFsmDisposition::RejectedCurrent
                                 : GnssFsmDisposition::QuarantinedCurrent;
    return Result::success(report(disposition, *rejection, stamp, id, std::move(transitions)));
  }

  if (pending_shadow_) {
    if (state_ == GnssFsmState::Reacquiring) {
      beginQuarantine(stamp);
    }
    const auto disposition = state_ == GnssFsmState::Aligning
                                 ? GnssFsmDisposition::RejectedCurrent
                                 : GnssFsmDisposition::QuarantinedCurrent;
    return Result::success(report(disposition, GnssFsmReason::ShadowAlreadyPending, stamp, id,
                                  std::move(transitions)));
  }

  if (state_ == GnssFsmState::Trusted) {
    if (!observation.committed_alignment_nis) {
      beginQuarantine(stamp);
      ++consecutive_failures_;
      consecutive_trusted_accepts_ = 0U;
      if (consecutive_failures_ >= config_.consecutive_failures_to_suspect) {
        transitionTo(GnssFsmState::Suspect, GnssFsmReason::QualityFailureStreak, stamp, id,
                     transitions);
      }
      return Result::success(report(GnssFsmDisposition::QuarantinedCurrent,
                                    GnssFsmReason::TrustedNisUnavailable, stamp, id,
                                    std::move(transitions)));
    }

    const double nis = *observation.committed_alignment_nis;
    if (nis <= config_.trusted_nis_accept_threshold) {
      ++consecutive_trusted_accepts_;
      if (consecutive_trusted_accepts_ >= config_.consecutive_trusted_accepts_to_clear_failures) {
        consecutive_failures_ = 0U;
        quarantine_active_ = false;
      }
      last_admitted_observation_ = id;
      last_admitted_time_ = stamp;
      GnssFsmReport accepted =
          report(GnssFsmDisposition::AdmitTrustedCurrent, GnssFsmReason::TrustedNisAccepted, stamp,
                 id, std::move(transitions));
      accepted.nis = nis;
      accepted.released_batch = GnssBatchAppend{std::nullopt, {observation.graph_constraint}};
      accepted.snapshot = snapshot();
      return Result::success(std::move(accepted));
    }

    beginQuarantine(stamp);
    consecutive_trusted_accepts_ = 0U;
    GnssFsmReason reason = GnssFsmReason::TrustedNisIndeterminate;
    if (nis >= config_.trusted_nis_reject_threshold) {
      reason = GnssFsmReason::TrustedNisRejected;
      ++consecutive_failures_;
      if (consecutive_failures_ >= config_.consecutive_failures_to_suspect) {
        transitionTo(GnssFsmState::Suspect, GnssFsmReason::QualityFailureStreak, stamp, id,
                     transitions);
      }
    }
    GnssFsmReport rejected =
        report(GnssFsmDisposition::QuarantinedCurrent, reason, stamp, id, std::move(transitions));
    rejected.nis = nis;
    return Result::success(std::move(rejected));
  }

  if (state_ == GnssFsmState::Suspect) {
    beginQuarantine(stamp);
    transitionTo(GnssFsmState::Reacquiring, GnssFsmReason::QualifiedWindowStarted, stamp, id,
                 transitions);
  }

  if (state_ != GnssFsmState::Aligning && state_ != GnssFsmState::Reacquiring) {
    return Result::failure(error(GnssFsmErrorCode::InvalidObservationAssociation,
                                 "qualified GNSS sample reached an invalid FSM state"));
  }

  if (!window_.empty() && (stamp - window_.back().stamp).nanoseconds >
                              config_.maximum_qualified_sample_gap.nanoseconds) {
    window_.clear();
    if (state_ == GnssFsmState::Reacquiring) {
      transitionTo(GnssFsmState::Suspect, GnssFsmReason::QualifiedWindowGap, stamp, id,
                   transitions);
      return Result::success(report(GnssFsmDisposition::QuarantinedCurrent,
                                    GnssFsmReason::QualifiedWindowGap, stamp, id,
                                    std::move(transitions)));
    }
  }

  if (window_.size() >= config_.maximum_buffered_observations) {
    GnssFsmError capacity =
        error(GnssFsmErrorCode::CapacityExceeded,
              "GNSS alignment window reached its hard observation cap before becoming observable");
    window_.clear();
    ++capacity_resets_;
    capacity.uncommitted_window_cleared = true;
    return Result::failure(std::move(capacity));
  }

  window_.push_back(BufferedObservation{id, stamp, observation.alignment_correspondence,
                                        observation.graph_constraint});
  const std::size_t minimum_candidate_size =
      std::max(config_.alignment.minimum_correspondences, config_.consecutive_post_solve_accepts);
  if (window_.size() < minimum_candidate_size) {
    const auto disposition = state_ == GnssFsmState::Aligning
                                 ? GnssFsmDisposition::BufferedForAlignment
                                 : GnssFsmDisposition::QuarantinedCurrent;
    return Result::success(report(disposition, GnssFsmReason::AwaitingQualityWindow, stamp, id,
                                  std::move(transitions)));
  }

  std::vector<AlignmentCorrespondence> correspondences;
  correspondences.reserve(window_.size());
  for (const auto& buffered : window_) {
    correspondences.push_back(buffered.alignment);
  }
  const auto estimate = estimateGravityAlignedTransform(correspondences, config_.alignment);
  if (!estimate) {
    if (!expectedAlignmentWait(estimate.error().code)) {
      return Result::failure(error(GnssFsmErrorCode::InvalidObservationAssociation,
                                   "validated GNSS window failed alignment input validation"));
    }
    const auto disposition = state_ == GnssFsmState::Aligning
                                 ? GnssFsmDisposition::BufferedForAlignment
                                 : GnssFsmDisposition::QuarantinedCurrent;
    GnssFsmReport waiting = report(disposition, GnssFsmReason::AwaitingAlignmentExcitation, stamp,
                                   id, std::move(transitions));
    waiting.alignment_wait_reason = estimate.error().code;
    return Result::success(std::move(waiting));
  }

  const GnssShadowKind kind =
      alignment_committed_ ? GnssShadowKind::Reacquisition : GnssShadowKind::InitialAlignment;
  const auto& transform = estimate.value().transform();
  const YawTranslation4 alignment_seed{transform.translationEnu(), transform.yawEnuMapRad()};
  GnssBatchAppend candidate;
  if (kind == GnssShadowKind::InitialAlignment) {
    candidate.initial_alignment = alignment_seed;
  }
  candidate.constraints.reserve(window_.size());
  for (const auto& buffered : window_) {
    candidate.constraints.push_back(buffered.factor);
  }
  const core::FusionTime source_end = window_.back().stamp + core::Duration{1};
  GnssShadowRequest request{GnssShadowAttemptId(next_shadow_attempt_++),
                            kind,
                            core::TimeRange{window_.front().stamp, source_end},
                            alignment_seed,
                            AlignmentCovariance{estimate.value().covariance()},
                            estimate.value().diagnostics(),
                            std::move(candidate)};
  pending_shadow_ = PendingShadow{request};
  window_.clear();

  GnssFsmReport ready =
      report(GnssFsmDisposition::ShadowRequested, GnssFsmReason::AlignmentObservable, stamp, id,
             std::move(transitions));
  ready.shadow_request = std::move(request);
  ready.snapshot = snapshot();
  return Result::success(std::move(ready));
}

core::Result<GnssFsmReport, GnssFsmError> GnssAdmissionFsm::advanceTime(core::FusionTime now) {
  if (last_event_time_ && now < *last_event_time_) {
    return Result::failure(
        error(GnssFsmErrorCode::NonMonotonicTime, "GNSS FSM time cannot move backwards"));
  }
  last_event_time_ = now;

  const bool timed_out = !last_observation_time_ || (now - *last_observation_time_).nanoseconds >=
                                                        config_.outage_timeout.nanoseconds;
  if (!timed_out) {
    return Result::success(
        report(GnssFsmDisposition::NoObservation, GnssFsmReason::None, now, std::nullopt));
  }

  clearUncommittedWindow();
  if (alignment_committed_) {
    beginQuarantine(now);
  }
  std::vector<GnssFsmTransition> transitions;
  transitionTo(GnssFsmState::Unavailable, GnssFsmReason::StreamTimedOut, now, std::nullopt,
               transitions);
  return Result::success(report(GnssFsmDisposition::NoObservation, GnssFsmReason::StreamTimedOut,
                                now, std::nullopt, std::move(transitions)));
}

core::Result<GnssFsmReport, GnssFsmError> GnssAdmissionFsm::resolveShadow(
    const GnssShadowValidation& validation) {
  if (!validation.attempt.valid()) {
    return Result::failure(
        error(GnssFsmErrorCode::InvalidShadowAttempt, "GNSS shadow attempt identity is invalid"));
  }
  if (!pending_shadow_ || pending_shadow_->request.attempt != validation.attempt) {
    return Result::failure(error(GnssFsmErrorCode::InvalidShadowAttempt,
                                 "GNSS shadow result does not match the pending attempt"));
  }
  if (last_event_time_ && validation.completed_at < *last_event_time_) {
    return Result::failure(error(GnssFsmErrorCode::NonMonotonicTime,
                                 "GNSS shadow completion time precedes processed input"));
  }
  if (validation.post_solve_nis.size() > config_.maximum_buffered_observations) {
    return Result::failure(error(GnssFsmErrorCode::CapacityExceeded,
                                 "GNSS shadow NIS report exceeds the configured hard cap"));
  }

  const GnssShadowRequest request = pending_shadow_->request;
  if (validation.post_solve_nis.size() != request.candidate_batch.constraints.size()) {
    return Result::failure(error(GnssFsmErrorCode::ShadowResultIdentityMismatch,
                                 "GNSS shadow NIS report does not cover every candidate factor"));
  }
  for (std::size_t index = 0; index < validation.post_solve_nis.size(); ++index) {
    const auto& entry = validation.post_solve_nis[index];
    if (!entry.observation.valid() ||
        entry.observation != request.candidate_batch.constraints[index].observation) {
      return Result::failure(error(GnssFsmErrorCode::ShadowResultIdentityMismatch,
                                   "GNSS shadow NIS identity or order differs from the request"));
    }
    if (!std::isfinite(entry.nis) || entry.nis < 0.0) {
      return Result::failure(
          error(GnssFsmErrorCode::NonFiniteInput, "GNSS post-solve NIS is invalid"));
    }
  }
  last_event_time_ = validation.completed_at;

  GnssFsmReason failure_reason = GnssFsmReason::None;
  if (!validation.optimizer_converged) {
    failure_reason = GnssFsmReason::ShadowOptimizationFailed;
  } else if (!validation.graph_validation_passed) {
    failure_reason = GnssFsmReason::ShadowGraphValidationFailed;
  } else if (!validation.resource_limits_satisfied) {
    failure_reason = GnssFsmReason::ShadowResourceLimitFailed;
  } else {
    std::size_t trailing_accepts = 0U;
    bool hard_nis_failure = false;
    for (const auto& entry : validation.post_solve_nis) {
      if (entry.nis >= config_.post_solve_nis_reject_threshold) {
        hard_nis_failure = true;
      }
    }
    for (auto iterator = validation.post_solve_nis.rbegin();
         iterator != validation.post_solve_nis.rend() &&
         iterator->nis <= config_.post_solve_nis_accept_threshold;
         ++iterator) {
      ++trailing_accepts;
    }
    if (hard_nis_failure || trailing_accepts < config_.consecutive_post_solve_accepts) {
      failure_reason = GnssFsmReason::ShadowPostSolveNisRejected;
    }
  }

  std::vector<GnssFsmTransition> transitions;
  pending_shadow_.reset();
  if (failure_reason != GnssFsmReason::None) {
    if (request.kind == GnssShadowKind::Reacquisition) {
      beginQuarantine(validation.completed_at);
      transitionTo(GnssFsmState::Suspect, failure_reason, validation.completed_at, std::nullopt,
                   transitions);
    } else {
      transitionTo(GnssFsmState::Aligning, failure_reason, validation.completed_at, std::nullopt,
                   transitions);
    }
    return Result::success(report(GnssFsmDisposition::QuarantinedCurrent, failure_reason,
                                  validation.completed_at, std::nullopt, std::move(transitions)));
  }

  alignment_committed_ = true;
  quarantine_active_ = false;
  consecutive_failures_ = 0U;
  consecutive_trusted_accepts_ = 0U;
  const auto& last_constraint = request.candidate_batch.constraints.back();
  last_admitted_observation_ = last_constraint.observation;
  last_admitted_time_ = request.source_window.end - core::Duration{1};
  transitionTo(GnssFsmState::Trusted, GnssFsmReason::ShadowValidationPassed,
               validation.completed_at, last_admitted_observation_, transitions);

  GnssFsmReport released =
      report(GnssFsmDisposition::ReleaseGraphBatch, GnssFsmReason::ShadowValidationPassed,
             validation.completed_at, last_admitted_observation_, std::move(transitions));
  released.released_batch = request.candidate_batch;
  released.snapshot = snapshot();
  return Result::success(std::move(released));
}

}  // namespace meridian::global
