#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "meridian/core/api.hpp"
#include "meridian/global/graph.hpp"

namespace meridian::global {

// The first odom epoch admitted to a coordinator instance is the only epoch
// that can enter the current GlobalGraph API. Later epochs are retained
// explicitly until a future whole-chain shadow transaction can connect them.
enum class GlobalEpochConnectivity {
  CommittedConnected,
  PendingUnconnected,
};

struct GlobalCoordinatorConfig {
  GlobalGraphConfig graph;
  std::size_t maximum_accepted_seals{4096U};
  std::size_t maximum_pending_epochs{16U};
  std::size_t maximum_pending_seals{1024U};
};

enum class GlobalSealDisposition {
  MissionInitialized,
  AdjacentCommitted,
  PendingUnconnectedStored,
  DuplicateIdempotent,
};

struct GlobalSealIngestionReport {
  core::SparseSubmapSealIdentity identity;
  GlobalEpochConnectivity epoch_connectivity{GlobalEpochConnectivity::PendingUnconnected};
  GlobalSealDisposition disposition{GlobalSealDisposition::PendingUnconnectedStored};
  bool graph_mutated{false};
  // The revision at which this seal entered the graph. It is absent for a
  // pending epoch and preserved on an idempotent re-delivery.
  std::optional<GlobalGraphRevision> seal_graph_revision;
  // Present only when this call committed an in-memory graph shadow.
  // Persistence belongs to the later GraphJournal adapter.
  std::optional<GlobalGraphCommit> graph_commit;
};

struct GlobalEpochStatus {
  core::OdomEpoch odom_epoch;
  GlobalEpochConnectivity connectivity{GlobalEpochConnectivity::PendingUnconnected};
  std::size_t accepted_seals{};
  core::SparseSubmapSealIdentity first;
  core::SparseSubmapSealIdentity latest;
  std::optional<GlobalGraphRevision> latest_seal_graph_revision;
};

struct GlobalCoordinatorStatus {
  bool graph_initialized{false};
  std::optional<core::SessionId> session;
  std::optional<core::OdomEpoch> connected_epoch;
  std::optional<GlobalGraphRevision> committed_graph_revision;
  std::size_t accepted_seals{};
  std::size_t connected_seals{};
  std::size_t pending_epochs{};
  std::size_t pending_seals{};
};

// GNSS proposal construction/alignment remains outside this coordinator. A
// graph-ready batch names the graph revision against which it was prepared.
struct GraphReadyGnssBatch {
  GlobalGraphRevision expected_parent;
  GnssBatchAppend append;
};

struct GlobalGnssBatchReport {
  GlobalGraphRevision evaluated_parent;
  GlobalGraphCommit commit;
};

struct GlobalRobustLoopBatchReport {
  GlobalGraphRevision evaluated_parent;
  std::optional<GlobalGraphRevision> committed_revision;
  RobustLoopTransactionResult transaction;
};

enum class GlobalCoordinatorErrorCode {
  InvalidConfiguration,
  NotInitialized,
  InvalidSeal,
  SessionMismatch,
  OdomEpochMismatch,
  ConflictingDuplicate,
  MissingEpochStart,
  MissingAdjacency,
  StaleOrOutOfOrderSeal,
  UnknownSubmap,
  StaleParentRevision,
  CapacityExceeded,
  // Connecting a retained epoch requires one atomic transaction containing
  // its complete adjacent chain and the verified connector. GlobalGraph does
  // not expose that transaction yet, so this is an explicit capability result.
  PendingEpochConnectorNotImplemented,
  GraphTransactionRejected,
  RobustLoopTransactionRejected,
};

struct GlobalCoordinatorError {
  GlobalCoordinatorErrorCode code{GlobalCoordinatorErrorCode::InvalidSeal};
  std::optional<core::CanonicalVerificationError> canonical_verification_error;
  std::optional<GlobalGraphError> graph_error;
  std::optional<RobustLoopTransactionError> robust_loop_error;
  std::string detail;
};

// ROS-free, move-only, single-writer owner. It never writes LocalGraph state
// or feeds a global correction into local odometry. All GlobalGraph mutation
// is delegated through its atomic shadow APIs, and coordinator bookkeeping is
// changed only after the graph transaction succeeds.
class GlobalCoordinator {
public:
  explicit GlobalCoordinator(GlobalCoordinatorConfig config = {});
  ~GlobalCoordinator();

  GlobalCoordinator(GlobalCoordinator&&) noexcept;
  GlobalCoordinator& operator=(GlobalCoordinator&&) noexcept;
  GlobalCoordinator(const GlobalCoordinator&) = delete;
  GlobalCoordinator& operator=(const GlobalCoordinator&) = delete;

  [[nodiscard]] core::Result<GlobalSealIngestionReport, GlobalCoordinatorError> ingestSeal(
      core::SparseSubmapSeal seal);

  [[nodiscard]] core::Result<GlobalGnssBatchReport, GlobalCoordinatorError> applyGnssBatch(
      GraphReadyGnssBatch batch);

  [[nodiscard]] core::Result<GlobalRobustLoopBatchReport, GlobalCoordinatorError>
  applyRobustLoopBatch(RobustLoopBatchAppend batch, const LoopConsensusReport& consensus);

  [[nodiscard]] core::Result<GlobalGraphCommit, GlobalCoordinatorError> graphSnapshot() const;
  [[nodiscard]] std::optional<MapOdomEstimate> mapOdom(core::OdomEpoch odom_epoch) const;
  [[nodiscard]] std::optional<GlobalEpochStatus> epochStatus(core::OdomEpoch odom_epoch) const;
  [[nodiscard]] std::vector<core::SparseSubmapSeal> pendingSeals(core::OdomEpoch odom_epoch) const;
  [[nodiscard]] GlobalCoordinatorStatus status() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::global
