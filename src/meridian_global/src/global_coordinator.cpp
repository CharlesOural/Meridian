#include "meridian/global/global_coordinator.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace meridian::global {
namespace {

[[nodiscard]] GlobalCoordinatorError makeError(GlobalCoordinatorErrorCode code,
                                               std::string detail) {
  GlobalCoordinatorError error;
  error.code = code;
  error.detail = std::move(detail);
  return error;
}

[[nodiscard]] GlobalCoordinatorError graphError(GlobalGraphError error) {
  GlobalCoordinatorError output =
      makeError(GlobalCoordinatorErrorCode::GraphTransactionRejected, error.detail);
  output.graph_error = std::move(error);
  return output;
}

[[nodiscard]] GlobalCoordinatorError robustLoopError(RobustLoopTransactionError error) {
  GlobalCoordinatorError output =
      makeError(GlobalCoordinatorErrorCode::RobustLoopTransactionRejected, error.detail);
  output.robust_loop_error = std::move(error);
  return output;
}

[[nodiscard]] std::optional<GlobalCoordinatorError> validateSealRecord(
    const core::SparseSubmapSeal& seal) {
  auto verified = core::verifyCanonicalSparseSubmapSeal(seal);
  if (!verified) {
    GlobalCoordinatorError error = makeError(GlobalCoordinatorErrorCode::InvalidSeal,
                                             "core sparse submap seal canonical admission failed");
    error.canonical_verification_error = std::move(verified).error();
    return error;
  }
  return std::nullopt;
}

}  // namespace

struct GlobalCoordinator::Impl {
  struct AcceptedSeal {
    core::SparseSubmapSeal seal;
    std::optional<GlobalGraphRevision> graph_revision;
  };

  struct EpochRecord {
    core::OdomEpoch odom_epoch;
    GlobalEpochConnectivity connectivity{GlobalEpochConnectivity::PendingUnconnected};
    std::vector<AcceptedSeal> seals;
  };

  explicit Impl(GlobalCoordinatorConfig input_config)
      : config(std::move(input_config)), graph(config.graph) {
    valid_config = config.maximum_accepted_seals > 0U && config.maximum_pending_epochs > 0U &&
                   config.maximum_pending_seals > 0U;
  }

  [[nodiscard]] EpochRecord* findEpoch(core::OdomEpoch odom_epoch) noexcept {
    const auto found = std::find_if(epochs.begin(), epochs.end(), [&](const EpochRecord& epoch) {
      return epoch.odom_epoch == odom_epoch;
    });
    return found == epochs.end() ? nullptr : &*found;
  }

  [[nodiscard]] const EpochRecord* findEpoch(core::OdomEpoch odom_epoch) const noexcept {
    const auto found = std::find_if(epochs.begin(), epochs.end(), [&](const EpochRecord& epoch) {
      return epoch.odom_epoch == odom_epoch;
    });
    return found == epochs.end() ? nullptr : &*found;
  }

  [[nodiscard]] const AcceptedSeal* findAcceptedSeal(const core::SubmapRef& submap) const noexcept {
    for (const EpochRecord& epoch : epochs) {
      for (const AcceptedSeal& accepted : epoch.seals) {
        if (accepted.seal.ref == submap) {
          return &accepted;
        }
      }
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t pendingEpochCount() const noexcept {
    return static_cast<std::size_t>(
        std::count_if(epochs.begin(), epochs.end(), [](const EpochRecord& epoch) {
          return epoch.connectivity == GlobalEpochConnectivity::PendingUnconnected;
        }));
  }

  [[nodiscard]] std::size_t pendingSealCount() const noexcept {
    std::size_t count = 0U;
    for (const EpochRecord& epoch : epochs) {
      if (epoch.connectivity == GlobalEpochConnectivity::PendingUnconnected) {
        count += epoch.seals.size();
      }
    }
    return count;
  }

  [[nodiscard]] std::size_t connectedSealCount() const noexcept {
    std::size_t count = 0U;
    for (const EpochRecord& epoch : epochs) {
      if (epoch.connectivity == GlobalEpochConnectivity::CommittedConnected) {
        count += epoch.seals.size();
      }
    }
    return count;
  }

  [[nodiscard]] const AcceptedSeal* findSameSubmapIdentity(
      const core::SparseSubmapSealIdentity& identity) const noexcept {
    for (const EpochRecord& epoch : epochs) {
      for (const AcceptedSeal& accepted : epoch.seals) {
        const core::SparseSubmapSealIdentity candidate =
            core::sparseSubmapSealIdentity(accepted.seal);
        if (core::sparseSubmapIdentityKey(candidate.ref) ==
            core::sparseSubmapIdentityKey(identity.ref)) {
          return &accepted;
        }
      }
    }
    return nullptr;
  }

  GlobalCoordinatorConfig config;
  GlobalGraph graph;
  bool valid_config{false};
  std::optional<core::SessionId> session;
  std::optional<core::OdomEpoch> connected_epoch;
  std::optional<GlobalGraphRevision> latest_graph_revision;
  std::vector<EpochRecord> epochs;
  std::size_t accepted_seals{};
};

GlobalCoordinator::GlobalCoordinator(GlobalCoordinatorConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

GlobalCoordinator::~GlobalCoordinator() = default;
GlobalCoordinator::GlobalCoordinator(GlobalCoordinator&&) noexcept = default;
GlobalCoordinator& GlobalCoordinator::operator=(GlobalCoordinator&&) noexcept = default;

core::Result<GlobalSealIngestionReport, GlobalCoordinatorError> GlobalCoordinator::ingestSeal(
    core::SparseSubmapSeal seal) {
  if (!impl_->valid_config) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::InvalidConfiguration,
                  "global coordinator capacity limits must be nonzero"));
  }
  if (const auto validation = validateSealRecord(seal); validation.has_value()) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(*validation);
  }
  const core::SparseSubmapSealIdentity identity = core::sparseSubmapSealIdentity(seal);
  if (impl_->session.has_value() && seal.ref.session != *impl_->session) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::SessionMismatch,
                  "seal belongs to another global coordinator session"));
  }

  if (const Impl::AcceptedSeal* duplicate = impl_->findSameSubmapIdentity(identity);
      duplicate != nullptr) {
    if (core::classifySparseSealRedelivery(core::sparseSubmapSealIdentity(duplicate->seal),
                                           identity) !=
        core::SparseSealRedeliveryRelation::Idempotent) {
      return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::ConflictingDuplicate,
                    "accepted submap identity was redelivered with a different "
                    "content revision or checksum"));
    }
    const Impl::EpochRecord* epoch = impl_->findEpoch(seal.ref.odom_epoch);
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::success(
        GlobalSealIngestionReport{identity, epoch->connectivity,
                                  GlobalSealDisposition::DuplicateIdempotent, false,
                                  duplicate->graph_revision, std::nullopt});
  }

  if (impl_->accepted_seals >= impl_->config.maximum_accepted_seals) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::CapacityExceeded,
                  "accepted sparse-seal capacity is exhausted"));
  }

  Impl::EpochRecord* epoch = impl_->findEpoch(seal.ref.odom_epoch);
  if (epoch == nullptr) {
    if (seal.from_previous.has_value()) {
      return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::MissingEpochStart,
                    "first delivered seal of an odom epoch has an incoming "
                    "adjacent constraint"));
    }

    if (!impl_->graph.initialized()) {
      Impl::EpochRecord candidate;
      candidate.odom_epoch = seal.ref.odom_epoch;
      candidate.connectivity = GlobalEpochConnectivity::CommittedConnected;
      candidate.seals.push_back(Impl::AcceptedSeal{seal, std::nullopt});
      impl_->epochs.push_back(std::move(candidate));
      Impl::EpochRecord& stored_epoch = impl_->epochs.back();

      auto initialized = impl_->graph.initializeMission(seal);
      if (!initialized) {
        impl_->epochs.pop_back();
        return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
            graphError(std::move(initialized.error())));
      }
      GlobalGraphCommit commit = std::move(initialized).value();
      stored_epoch.seals.back().graph_revision = commit.revision;
      impl_->session = seal.ref.session;
      impl_->connected_epoch = seal.ref.odom_epoch;
      impl_->latest_graph_revision = commit.revision;
      ++impl_->accepted_seals;
      return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::success(
          GlobalSealIngestionReport{identity, GlobalEpochConnectivity::CommittedConnected,
                                    GlobalSealDisposition::MissionInitialized, true,
                                    commit.revision, std::move(commit)});
    }

    if (impl_->pendingEpochCount() >= impl_->config.maximum_pending_epochs ||
        impl_->pendingSealCount() >= impl_->config.maximum_pending_seals) {
      return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::CapacityExceeded,
                    "pending unconnected epoch capacity is exhausted"));
    }
    Impl::EpochRecord pending;
    pending.odom_epoch = seal.ref.odom_epoch;
    pending.connectivity = GlobalEpochConnectivity::PendingUnconnected;
    pending.seals.push_back(Impl::AcceptedSeal{seal, std::nullopt});
    impl_->epochs.push_back(std::move(pending));
    ++impl_->accepted_seals;
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::success(
        GlobalSealIngestionReport{identity, GlobalEpochConnectivity::PendingUnconnected,
                                  GlobalSealDisposition::PendingUnconnectedStored, false,
                                  std::nullopt, std::nullopt});
  }

  if (!seal.from_previous.has_value()) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::MissingAdjacency,
                  "non-initial seal of an odom epoch has no incoming adjacent "
                  "constraint"));
  }
  const Impl::AcceptedSeal& previous = epoch->seals.back();
  if (core::validateSparseSubmapLink(previous.seal, seal) !=
      core::SparseSubmapLinkValidationError::None) {
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::StaleOrOutOfOrderSeal,
                  "incoming adjacent constraint does not extend the latest "
                  "accepted half-open submap chain exactly"));
  }

  if (epoch->connectivity == GlobalEpochConnectivity::PendingUnconnected) {
    if (impl_->pendingSealCount() >= impl_->config.maximum_pending_seals) {
      return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::CapacityExceeded,
                    "pending unconnected seal capacity is exhausted"));
    }
    epoch->seals.push_back(Impl::AcceptedSeal{seal, std::nullopt});
    ++impl_->accepted_seals;
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::success(
        GlobalSealIngestionReport{identity, GlobalEpochConnectivity::PendingUnconnected,
                                  GlobalSealDisposition::PendingUnconnectedStored, false,
                                  std::nullopt, std::nullopt});
  }

  epoch->seals.push_back(Impl::AcceptedSeal{seal, std::nullopt});
  auto appended = impl_->graph.appendAdjacent(seal);
  if (!appended) {
    epoch->seals.pop_back();
    return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::failure(
        graphError(std::move(appended.error())));
  }
  GlobalGraphCommit commit = std::move(appended).value();
  epoch->seals.back().graph_revision = commit.revision;
  impl_->latest_graph_revision = commit.revision;
  ++impl_->accepted_seals;
  return core::Result<GlobalSealIngestionReport, GlobalCoordinatorError>::success(
      GlobalSealIngestionReport{identity, GlobalEpochConnectivity::CommittedConnected,
                                GlobalSealDisposition::AdjacentCommitted, true, commit.revision,
                                std::move(commit)});
}

core::Result<GlobalGnssBatchReport, GlobalCoordinatorError> GlobalCoordinator::applyGnssBatch(
    GraphReadyGnssBatch batch) {
  if (!impl_->valid_config) {
    return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::InvalidConfiguration,
                  "global coordinator capacity limits must be nonzero"));
  }
  if (!impl_->graph.initialized() || !impl_->latest_graph_revision.has_value()) {
    return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(makeError(
        GlobalCoordinatorErrorCode::NotInitialized, "global graph has no connected mission"));
  }
  if (batch.expected_parent != *impl_->latest_graph_revision) {
    return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::StaleParentRevision,
                  "GNSS batch was prepared against a stale graph revision"));
  }
  for (const GnssAntennaConstraint& constraint : batch.append.constraints) {
    const Impl::AcceptedSeal* accepted = impl_->findAcceptedSeal(constraint.submap);
    if (accepted == nullptr) {
      return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::UnknownSubmap,
                    "GNSS batch references an unknown or stale SubmapRef"));
    }
    const Impl::EpochRecord* epoch = impl_->findEpoch(accepted->seal.ref.odom_epoch);
    if (epoch->connectivity == GlobalEpochConnectivity::PendingUnconnected) {
      return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::PendingEpochConnectorNotImplemented,
                    "qualified GNSS connection of a pending epoch requires an atomic "
                    "whole-chain GlobalGraph transaction"));
    }
  }

  auto appended = impl_->graph.appendGnssBatch(std::move(batch.append));
  if (!appended) {
    return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::failure(
        graphError(std::move(appended.error())));
  }
  GlobalGraphCommit commit = std::move(appended).value();
  impl_->latest_graph_revision = commit.revision;
  return core::Result<GlobalGnssBatchReport, GlobalCoordinatorError>::success(
      GlobalGnssBatchReport{batch.expected_parent, std::move(commit)});
}

core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>
GlobalCoordinator::applyRobustLoopBatch(RobustLoopBatchAppend batch,
                                        const LoopConsensusReport& consensus) {
  if (!impl_->valid_config) {
    return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::InvalidConfiguration,
                  "global coordinator capacity limits must be nonzero"));
  }
  if (!impl_->graph.initialized() || !impl_->latest_graph_revision.has_value()) {
    return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(makeError(
        GlobalCoordinatorErrorCode::NotInitialized, "global graph has no connected mission"));
  }
  if (batch.expected_parent != *impl_->latest_graph_revision) {
    return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
        makeError(GlobalCoordinatorErrorCode::StaleParentRevision,
                  "robust loop batch was prepared against a stale graph revision"));
  }

  for (const RobustLoopCandidate& candidate : batch.candidates) {
    if (!impl_->session.has_value() || candidate.measurement.header.session != *impl_->session) {
      return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::SessionMismatch,
                    "robust loop batch contains a proposal from another session"));
    }
    const Impl::AcceptedSeal* from = impl_->findAcceptedSeal(candidate.measurement.from);
    const Impl::AcceptedSeal* to = impl_->findAcceptedSeal(candidate.measurement.to);
    if (from == nullptr || to == nullptr) {
      return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::UnknownSubmap,
                    "robust loop batch references an unknown or stale SubmapRef"));
    }
    const Impl::EpochRecord* from_epoch = impl_->findEpoch(from->seal.ref.odom_epoch);
    const Impl::EpochRecord* to_epoch = impl_->findEpoch(to->seal.ref.odom_epoch);
    if (from_epoch->connectivity == GlobalEpochConnectivity::PendingUnconnected ||
        to_epoch->connectivity == GlobalEpochConnectivity::PendingUnconnected) {
      return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
          makeError(GlobalCoordinatorErrorCode::PendingEpochConnectorNotImplemented,
                    "verified loop connection of a pending epoch requires an atomic "
                    "whole-chain GlobalGraph transaction"));
    }
  }

  const GlobalGraphRevision evaluated_parent = batch.expected_parent;
  auto appended = impl_->graph.appendRobustLoopBatch(std::move(batch), consensus);
  if (!appended) {
    return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::failure(
        robustLoopError(std::move(appended.error())));
  }
  RobustLoopTransactionResult transaction = std::move(appended).value();
  std::optional<GlobalGraphRevision> committed_revision;
  if (transaction.commit.has_value()) {
    committed_revision = transaction.commit->revision;
    impl_->latest_graph_revision = transaction.commit->revision;
  }
  return core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>::success(
      GlobalRobustLoopBatchReport{evaluated_parent, committed_revision, std::move(transaction)});
}

core::Result<GlobalGraphCommit, GlobalCoordinatorError> GlobalCoordinator::graphSnapshot() const {
  if (!impl_->graph.initialized()) {
    return core::Result<GlobalGraphCommit, GlobalCoordinatorError>::failure(makeError(
        GlobalCoordinatorErrorCode::NotInitialized, "global graph has no connected mission"));
  }
  auto snapshot = impl_->graph.snapshot();
  if (!snapshot) {
    return core::Result<GlobalGraphCommit, GlobalCoordinatorError>::failure(
        graphError(std::move(snapshot.error())));
  }
  return core::Result<GlobalGraphCommit, GlobalCoordinatorError>::success(
      std::move(snapshot).value());
}

std::optional<MapOdomEstimate> GlobalCoordinator::mapOdom(core::OdomEpoch odom_epoch) const {
  const Impl::EpochRecord* epoch = impl_->findEpoch(odom_epoch);
  if (epoch == nullptr || epoch->connectivity != GlobalEpochConnectivity::CommittedConnected) {
    return std::nullopt;
  }
  auto snapshot = impl_->graph.snapshot();
  if (!snapshot || snapshot.value().map_odom.reference_submap.odom_epoch != odom_epoch) {
    return std::nullopt;
  }
  return snapshot.value().map_odom;
}

std::optional<GlobalEpochStatus> GlobalCoordinator::epochStatus(core::OdomEpoch odom_epoch) const {
  const Impl::EpochRecord* epoch = impl_->findEpoch(odom_epoch);
  if (epoch == nullptr || epoch->seals.empty()) {
    return std::nullopt;
  }
  return GlobalEpochStatus{epoch->odom_epoch,
                           epoch->connectivity,
                           epoch->seals.size(),
                           core::sparseSubmapSealIdentity(epoch->seals.front().seal),
                           core::sparseSubmapSealIdentity(epoch->seals.back().seal),
                           epoch->seals.back().graph_revision};
}

std::vector<core::SparseSubmapSeal> GlobalCoordinator::pendingSeals(
    core::OdomEpoch odom_epoch) const {
  std::vector<core::SparseSubmapSeal> output;
  const Impl::EpochRecord* epoch = impl_->findEpoch(odom_epoch);
  if (epoch == nullptr || epoch->connectivity != GlobalEpochConnectivity::PendingUnconnected) {
    return output;
  }
  output.reserve(epoch->seals.size());
  for (const Impl::AcceptedSeal& accepted : epoch->seals) {
    output.push_back(accepted.seal);
  }
  return output;
}

GlobalCoordinatorStatus GlobalCoordinator::status() const noexcept {
  return GlobalCoordinatorStatus{impl_->graph.initialized(), impl_->session,
                                 impl_->connected_epoch,     impl_->latest_graph_revision,
                                 impl_->accepted_seals,      impl_->connectedSealCount(),
                                 impl_->pendingEpochCount(), impl_->pendingSealCount()};
}

}  // namespace meridian::global
