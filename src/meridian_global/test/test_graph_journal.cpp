#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../src/persistence_internal.hpp"
#include "meridian/global/graph_journal.hpp"
#include "sparse_seal_test_utils.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::ContentHash contentHash(std::uint8_t value) {
  core::ContentHash output{};
  output[0] = value;
  output[15] = static_cast<std::uint8_t>(value ^ 0x5aU);
  output[31] = static_cast<std::uint8_t>(value ^ 0xa5U);
  return output;
}

[[nodiscard]] GlobalGraphCheckpoint checkpoint(std::uint64_t revision_value) {
  GlobalGraph graph;
  core::SparseSubmapSeal seal = test_support::firstSeal();
  (void)test_support::required(graph.initializeMission(seal));
  for (std::uint64_t revision = 1U; revision <= revision_value; ++revision) {
    seal = test_support::successor(seal);
    (void)test_support::required(graph.appendAdjacent(seal));
  }
  return test_support::required(graph.checkpoint());
}

[[nodiscard]] GlobalGraphCheckpoint alignedCheckpoint() {
  GlobalGraph graph;
  const core::SparseSubmapSeal first = test_support::firstSeal();
  (void)test_support::required(graph.initializeMission(first));
  const YawTranslation4 truth{{12.0, -4.0, 1.5}, 0.35};
  const std::array<Eigen::Vector3d, 6> antenna_points{
      Eigen::Vector3d{0.0, 0.0, 0.0},  Eigen::Vector3d{10.0, 0.0, 0.4},
      Eigen::Vector3d{0.0, 8.0, -0.2}, Eigen::Vector3d{5.0, -4.0, 0.8},
      Eigen::Vector3d{-3.0, 6.0, 0.2}, Eigen::Vector3d{13.0, 9.0, -0.5}};
  GnssBatchAppend gnss;
  gnss.initial_alignment = YawTranslation4{{11.0, -3.0, 1.0}, 0.25};
  for (std::size_t index = 0U; index < antenna_points.size(); ++index) {
    gnss.constraints.push_back(GnssAntennaConstraint{
        first.ref, core::GnssObservationId(index + 1U), antenna_points[index],
        truth.apply(antenna_points[index]), Eigen::Matrix3d::Identity() * 0.01});
  }
  (void)test_support::required(graph.appendGnssBatch(std::move(gnss)));
  return test_support::required(graph.checkpoint());
}

[[nodiscard]] GraphJournalPrepareRequest request(GlobalGraphCheckpoint graph_checkpoint,
                                                 std::uint8_t input_value = 1U) {
  const std::uint64_t revision = graph_checkpoint.revision.value();
  std::vector<GraphJournalInput> inputs{
      GraphJournalInput{GraphJournalInputKind::SparseSubmapSeal, 100U + revision,
                        contentHash(input_value), GraphJournalDisposition::Admitted},
      GraphJournalInput{GraphJournalInputKind::AdjacentConstraint, 200U + revision,
                        contentHash(static_cast<std::uint8_t>(input_value + 1U)),
                        GraphJournalDisposition::Retained}};
  return GraphJournalPrepareRequest{graph_checkpoint.parent, std::move(graph_checkpoint),
                                    std::move(inputs)};
}

[[nodiscard]] GraphJournalPrepareRequest request(std::uint64_t revision,
                                                 std::uint8_t input_value = 1U) {
  return request(checkpoint(revision), input_value);
}

class TemporaryJournal {
public:
  TemporaryJournal() {
    static std::atomic<std::uint64_t> counter{0U};
    path =
        std::filesystem::temp_directory_path() /
        ("meridian_graph_journal_" + std::to_string(static_cast<unsigned long long>(::getpid())) +
         "_" + std::to_string(counter.fetch_add(1U)));
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  ~TemporaryJournal() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  [[nodiscard]] GraphJournalConfig config() const {
    GraphJournalConfig output;
    output.root_directory = path;
    output.maximum_bytes = 8U * 1024U * 1024U;
    output.checkpoint_limits.maximum_wire_bytes = 2U * 1024U * 1024U;
    output.checkpoint_limits.maximum_boundaries = 32U;
    output.checkpoint_limits.maximum_chart_placements = 32U;
    output.checkpoint_limits.maximum_adjacent_factors = 32U;
    output.checkpoint_limits.maximum_gnss_factors = 64U;
    output.checkpoint_limits.maximum_loop_factors = 32U;
    output.maximum_revisions = 16U;
    output.maximum_prepared_transactions = 32U;
    output.maximum_inputs_per_transaction = 64U;
    output.maximum_replay_records = 16U;
    return output;
  }

  std::filesystem::path path;
};

[[nodiscard]] std::filesystem::path onlyFile(const std::filesystem::path& directory,
                                             std::string_view extension) {
  std::filesystem::path found;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == extension) {
      EXPECT_TRUE(found.empty());
      found = entry.path();
    }
  }
  EXPECT_FALSE(found.empty());
  return found;
}

void flipLastByte(const std::filesystem::path& path) {
  std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(file.good());
  file.seekg(-1, std::ios::end);
  char value{};
  file.read(&value, 1);
  value = static_cast<char>(static_cast<unsigned char>(value) ^ 0x01U);
  file.seekp(-1, std::ios::end);
  file.write(&value, 1);
  file.flush();
  ASSERT_TRUE(file.good());
}

void truncateLastByte(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  ASSERT_GT(size, 0U);
  std::filesystem::resize_file(path, size - 1U);
}

void writeBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.flush();
  ASSERT_TRUE(output.good());
}

[[nodiscard]] std::vector<std::byte> readFrameBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  EXPECT_TRUE(input.good());
  const auto size = input.tellg();
  EXPECT_GE(size, 0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  EXPECT_TRUE(input.good());
  return bytes;
}

[[nodiscard]] std::filesystem::path rewriteFramedRecord(const std::filesystem::path& path,
                                                        std::vector<std::byte> bytes,
                                                        std::string_view extension) {
  constexpr std::size_t kChecksumOffset = 8U + 4U + 8U;
  const auto checksum = persistence_internal::hashBytes(
      std::span<const std::byte>(bytes).subspan(persistence_internal::kFrameOverhead));
  for (std::size_t index = 0U; index < checksum.size(); ++index) {
    bytes[kChecksumOffset + index] = static_cast<std::byte>(checksum[index]);
  }
  const auto replacement =
      path.parent_path() / (persistence_internal::hashHex(checksum) + std::string(extension));
  if (replacement != path) {
    std::filesystem::rename(path, replacement);
  }
  std::ofstream output(replacement, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  output.flush();
  EXPECT_TRUE(output.good());
  return replacement;
}

TEST(GraphJournal, CanonicalCheckpointRoundTripsAndReplaysExactRevisions) {
  TemporaryJournal temporary;
  core::ContentHash revision_zero_checksum{};
  const GlobalGraphCheckpoint expected_latest = checkpoint(1U);
  const auto expected_latest_bytes =
      encodeGlobalGraphCheckpoint(expected_latest, temporary.config().checkpoint_limits);
  ASSERT_TRUE(expected_latest_bytes);
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto first_prepare = opened.value()->prepare(request(0U, 10U));
    ASSERT_TRUE(first_prepare);
    EXPECT_FALSE(first_prepare.value().idempotent);
    revision_zero_checksum = first_prepare.value().checkpoint_checksum;
    auto reversed_request = request(0U, 10U);
    std::reverse(reversed_request.inputs.begin(), reversed_request.inputs.end());
    const auto repeated_prepare = opened.value()->prepare(std::move(reversed_request));
    ASSERT_TRUE(repeated_prepare);
    EXPECT_TRUE(repeated_prepare.value().idempotent);
    EXPECT_EQ(repeated_prepare.value().checkpoint_checksum, revision_zero_checksum);
    const auto first_commit = opened.value()->commitPrepared(first_prepare.value());
    ASSERT_TRUE(first_commit);
    EXPECT_FALSE(first_commit.value().idempotent);
    const auto repeated_commit = opened.value()->commitPrepared(first_prepare.value());
    ASSERT_TRUE(repeated_commit);
    EXPECT_TRUE(repeated_commit.value().idempotent);

    const auto second_prepare = opened.value()->prepare(request(1U, 20U));
    ASSERT_TRUE(second_prepare);
    ASSERT_TRUE(opened.value()->commitPrepared(second_prepare.value()));
  }

  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  const GraphJournalStatus status = recovered.value()->status();
  ASSERT_TRUE(status.committed_head.has_value());
  EXPECT_EQ(*status.committed_head, GlobalGraphRevision(1U));
  EXPECT_EQ(status.committed_revisions, 2U);
  EXPECT_EQ(status.uncommitted_prepared_transactions, 0U);
  const auto latest = recovered.value()->latestCommit();
  ASSERT_TRUE(latest);
  ASSERT_TRUE(latest.value().has_value());
  EXPECT_EQ(latest.value()->revision, GlobalGraphRevision(1U));
  EXPECT_EQ(latest.value()->checkpoint.boundaries.size(), 2U);
  EXPECT_EQ(latest.value()->checkpoint.recovery.map_odom.graph_revision, GlobalGraphRevision(1U));
  const auto recovered_bytes =
      encodeGlobalGraphCheckpoint(latest.value()->checkpoint, temporary.config().checkpoint_limits);
  ASSERT_TRUE(recovered_bytes);
  EXPECT_TRUE(
      std::ranges::equal(expected_latest_bytes.value().bytes(), recovered_bytes.value().bytes()));

  GlobalGraph restored;
  const auto restored_commit = restored.restoreCheckpoint(latest.value()->checkpoint);
  ASSERT_TRUE(restored_commit) << restored_commit.error().detail;
  EXPECT_EQ(test_support::required(restored.checkpoint()).checksum, expected_latest.checksum);

  const auto all = recovered.value()->commitsSince(std::nullopt, 16U);
  ASSERT_TRUE(all);
  ASSERT_EQ(all.value().size(), 2U);
  EXPECT_EQ(all.value().front().checkpoint_checksum, revision_zero_checksum);
  const auto after_zero = recovered.value()->commitsSince(GlobalGraphRevision(0U), 16U);
  ASSERT_TRUE(after_zero);
  ASSERT_EQ(after_zero.value().size(), 1U);
  EXPECT_EQ(after_zero.value().front().revision, GlobalGraphRevision(1U));
}

TEST(GraphJournal, RecoveredCheckpointRestoresIdentityAllocatorsWithoutDrift) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto first = opened.value()->prepare(request(0U, 0xa0U));
    ASSERT_TRUE(first);
    ASSERT_TRUE(opened.value()->commitPrepared(first.value()));
    const auto second = opened.value()->prepare(request(1U, 0xa1U));
    ASSERT_TRUE(second);
    ASSERT_TRUE(opened.value()->commitPrepared(second.value()));
  }

  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  const auto latest = recovered.value()->latestCommit();
  ASSERT_TRUE(latest);
  ASSERT_TRUE(latest.value().has_value());
  const GlobalGraphCheckpoint durable_checkpoint = latest.value()->checkpoint;

  GlobalGraph restored;
  ASSERT_TRUE(restored.restoreCheckpoint(durable_checkpoint));
  core::SparseSubmapSeal second = test_support::successor(test_support::firstSeal());
  const core::SparseSubmapSeal third = test_support::successor(second);
  const auto appended = restored.appendAdjacent(third);
  ASSERT_TRUE(appended) << appended.error().detail;
  const GlobalGraphCheckpoint continued = test_support::required(restored.checkpoint());
  const GlobalGraphCheckpoint uninterrupted = checkpoint(2U);

  EXPECT_EQ(continued.checksum, uninterrupted.checksum);
  EXPECT_EQ(continued.revision, GlobalGraphRevision(durable_checkpoint.revision.value() + 1U));
  EXPECT_EQ(continued.next_boundary_slot, durable_checkpoint.next_boundary_slot + 1U);
  EXPECT_EQ(continued.next_factor_id, durable_checkpoint.next_factor_id + 1U);
  EXPECT_EQ(continued.next_candidate_id, durable_checkpoint.next_candidate_id);
  ASSERT_FALSE(continued.adjacent_factors.empty());
  EXPECT_EQ(continued.adjacent_factors.back().factor,
            GlobalFactorId(durable_checkpoint.next_factor_id));
}

TEST(GraphJournal, StoresExactCheckpointProducedByCurrentGlobalGraph) {
  TemporaryJournal temporary;
  const core::Pose3d T_odom_submap(Sophus::SO3d::exp(Eigen::Vector3d{0.1, -0.05, 0.4}),
                                   Eigen::Vector3d{4.0, -2.0, 0.8});
  const core::SparseSubmapSeal first_submap =
      test_support::firstSeal(91U, 0, T_odom_submap, T_odom_submap);
  GlobalGraph graph;
  auto graph_commit = graph.initializeMission(first_submap);
  ASSERT_TRUE(graph_commit);
  auto graph_checkpoint = graph.checkpoint();
  ASSERT_TRUE(graph_checkpoint);

  std::vector<GraphJournalInput> inputs{GraphJournalInput{GraphJournalInputKind::SparseSubmapSeal,
                                                          91U, contentHash(0xb1U),
                                                          GraphJournalDisposition::Admitted}};
  GraphJournalPrepareRequest prepare_request{graph_checkpoint.value().parent,
                                             graph_checkpoint.value(), std::move(inputs)};
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  const auto prepared = opened.value()->prepare(std::move(prepare_request));
  ASSERT_TRUE(prepared) << prepared.error().detail;
  const auto committed = opened.value()->commitPrepared(prepared.value());
  ASSERT_TRUE(committed) << committed.error().detail;
  EXPECT_EQ(committed.value().checkpoint.checksum, graph_checkpoint.value().checksum);
  const auto encoded =
      encodeGlobalGraphCheckpoint(graph_checkpoint.value(), temporary.config().checkpoint_limits);
  ASSERT_TRUE(encoded);
  const auto durable_bytes =
      readFrameBytes(onlyFile(temporary.path / "checkpoints", ".checkpoint"));
  EXPECT_TRUE(std::ranges::equal(encoded.value().bytes(), durable_bytes));
  EXPECT_LT((committed.value().checkpoint.recovery.map_odom.T_map_odom.inverse() * core::Pose3d{})
                .log()
                .norm(),
            1.0e-8);
}

TEST(GraphJournal, RoundTripsOptionalGravityAlignedEnuStateAndCovariance) {
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  const auto initialized = opened.value()->prepare(request(0U, 0xc0U));
  ASSERT_TRUE(initialized);
  ASSERT_TRUE(opened.value()->commitPrepared(initialized.value()));
  const GlobalGraphCheckpoint aligned = alignedCheckpoint();
  ASSERT_TRUE(aligned.alignment.has_value());
  ASSERT_TRUE(aligned.recovery.alignment_covariance.has_value());
  const auto prepared = opened.value()->prepare(request(aligned, 0xc1U));
  ASSERT_TRUE(prepared) << prepared.error().detail;
  const auto committed = opened.value()->commitPrepared(prepared.value());
  ASSERT_TRUE(committed);
  ASSERT_TRUE(committed.value().checkpoint.alignment.has_value());
  ASSERT_TRUE(committed.value().checkpoint.recovery.alignment_covariance.has_value());
  EXPECT_NEAR(committed.value().checkpoint.alignment->yaw_enu_map_rad,
              aligned.alignment->yaw_enu_map_rad, 1.0e-15);
  EXPECT_LT(
      (committed.value().checkpoint.alignment->translation_enu - aligned.alignment->translation_enu)
          .norm(),
      1.0e-15);
  EXPECT_LT((committed.value().checkpoint.recovery.alignment_covariance->matrix -
             aligned.recovery.alignment_covariance->matrix)
                .norm(),
            1.0e-15);
}

TEST(GraphJournal, CrashBeforeCommitMarkerLeavesPreparedRetryableButInvisible) {
  TemporaryJournal temporary;
  PreparedGraphCommit token;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    auto prepared = opened.value()->prepare(request(0U, 30U));
    ASSERT_TRUE(prepared);
    token = prepared.value();
    const auto latest = opened.value()->latestCommit();
    ASSERT_TRUE(latest);
    EXPECT_FALSE(latest.value().has_value());
  }

  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_FALSE(recovered.value()->latestCommit().value().has_value());
  EXPECT_EQ(recovered.value()->status().uncommitted_prepared_transactions, 1U);
  const auto committed = recovered.value()->commitPrepared(token);
  ASSERT_TRUE(committed);
  EXPECT_EQ(committed.value().revision, GlobalGraphRevision(0U));
}

TEST(GraphJournal, DurableMarkerRecoversRevisionBeforeAnyPublicationState) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 40U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
  }
  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  const auto latest = recovered.value()->latestCommit();
  ASSERT_TRUE(latest);
  ASSERT_TRUE(latest.value().has_value());
  EXPECT_EQ(latest.value()->revision, GlobalGraphRevision(0U));
  EXPECT_EQ(latest.value()->checkpoint.recovery.committed_solve.transaction,
            GlobalTransactionKind::MissionInitialization);
}

TEST(GraphJournal, CompetingPreparedCandidatesUseCommitMarkerCompareAndSwap) {
  TemporaryJournal temporary;
  auto config = temporary.config();
  config.maximum_prepared_transactions = 3U;
  auto opened = GraphJournal::open(config);
  ASSERT_TRUE(opened);
  const auto initial = opened.value()->prepare(request(0U, 50U));
  ASSERT_TRUE(initial);
  ASSERT_TRUE(opened.value()->commitPrepared(initial.value()));

  const auto candidate_a = opened.value()->prepare(request(1U, 51U));
  const auto candidate_b = opened.value()->prepare(request(1U, 52U));
  ASSERT_TRUE(candidate_a);
  ASSERT_TRUE(candidate_b);
  EXPECT_NE(candidate_a.value().prepared_checksum, candidate_b.value().prepared_checksum);
  ASSERT_TRUE(opened.value()->commitPrepared(candidate_a.value()));
  const auto rejected = opened.value()->commitPrepared(candidate_b.value());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::PreparedNotFound);
  EXPECT_EQ(opened.value()->latestCommit().value()->prepared_checksum,
            candidate_a.value().prepared_checksum);
  EXPECT_EQ(opened.value()->status().uncommitted_prepared_transactions, 0U);
  EXPECT_GE(opened.value()->status().retired_unreachable_files, 1U);

  const auto stale = opened.value()->prepare(request(1U, 53U));
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, GraphJournalErrorCode::StaleParentRevision);

  const auto next = opened.value()->prepare(request(2U, 54U));
  ASSERT_TRUE(next) << next.error().detail;
}

TEST(GraphJournal, RecoveryRetiresAStalePreparedSiblingLeftAfterDurableCommit) {
  TemporaryJournal temporary;
  std::filesystem::path losing_path;
  std::vector<char> losing_bytes;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto initial = opened.value()->prepare(request(0U, 55U));
    ASSERT_TRUE(initial);
    ASSERT_TRUE(opened.value()->commitPrepared(initial.value()));
    const auto winner = opened.value()->prepare(request(1U, 56U));
    const auto loser = opened.value()->prepare(request(1U, 57U));
    ASSERT_TRUE(winner);
    ASSERT_TRUE(loser);
    losing_path = temporary.path / "prepared" /
                  (persistence_internal::hashHex(loser.value().prepared_checksum) + ".prepared");
    std::ifstream input(losing_path, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(input.good());
    const auto size = input.tellg();
    losing_bytes.resize(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(losing_bytes.data(), static_cast<std::streamsize>(size));
    ASSERT_TRUE(input.good());
    ASSERT_TRUE(opened.value()->commitPrepared(winner.value()));
    EXPECT_FALSE(std::filesystem::exists(losing_path));
  }
  {
    std::ofstream output(losing_path, std::ios::binary);
    output.write(losing_bytes.data(), static_cast<std::streamsize>(losing_bytes.size()));
    output.flush();
    ASSERT_TRUE(output.good());
  }
  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_FALSE(std::filesystem::exists(losing_path));
  EXPECT_EQ(recovered.value()->status().uncommitted_prepared_transactions, 0U);
  EXPECT_GE(recovered.value()->status().retired_unreachable_files, 1U);
}

TEST(GraphJournal, TemporaryCrashFilesAreRemovedBeforeStrictRecoveryScan) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
  }
  const std::vector<std::filesystem::path> torn{
      temporary.path / "checkpoints" / "orphan.checkpoint.tmp.1",
      temporary.path / "prepared" / "orphan.prepared.tmp.2",
      temporary.path / "committed" / "orphan.committed.tmp.3"};
  for (const auto& path : torn) {
    std::ofstream output(path, std::ios::binary);
    output << "partial";
  }
  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().recovered_temporary_files, torn.size());
  for (const auto& path : torn) {
    EXPECT_FALSE(std::filesystem::exists(path));
  }
}

TEST(GraphJournal, LegacyCheckpointAndRecordFormatsRequireExplicitMigration) {
  constexpr std::array<std::byte, 8> kLegacyCheckpointMagic{
      std::byte{'M'}, std::byte{'R'}, std::byte{'D'}, std::byte{'N'},
      std::byte{'G'}, std::byte{'C'}, std::byte{'P'}, std::byte{'1'}};
  constexpr std::array<std::byte, 8> kPreparedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                    std::byte{'N'}, std::byte{'G'}, std::byte{'P'},
                                                    std::byte{'R'}, std::byte{'1'}};
  constexpr std::array<std::byte, 8> kCommittedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                     std::byte{'N'}, std::byte{'G'}, std::byte{'C'},
                                                     std::byte{'M'}, std::byte{'1'}};

  {
    TemporaryJournal temporary;
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    opened.value().reset();
    const auto legacy =
        persistence_internal::frameBytes(kLegacyCheckpointMagic, 4U, std::span<const std::byte>{});
    writeBytes(temporary.path / "checkpoints" / "legacy.checkpoint", legacy);
    const auto recovered = GraphJournal::open(temporary.config());
    ASSERT_FALSE(recovered);
    EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::MigrationRequired);
  }
  {
    TemporaryJournal temporary;
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    opened.value().reset();
    const auto legacy =
        persistence_internal::frameBytes(kPreparedMagic, 4U, std::span<const std::byte>{});
    writeBytes(temporary.path / "prepared" / "legacy.prepared", legacy);
    const auto recovered = GraphJournal::open(temporary.config());
    ASSERT_FALSE(recovered);
    EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::MigrationRequired);
  }
  {
    TemporaryJournal temporary;
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    opened.value().reset();
    const auto legacy =
        persistence_internal::frameBytes(kCommittedMagic, 4U, std::span<const std::byte>{});
    writeBytes(temporary.path / "committed" / "00000000000000000000.committed", legacy);
    const auto recovered = GraphJournal::open(temporary.config());
    ASSERT_FALSE(recovered);
    EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::MigrationRequired);
  }
}

TEST(GraphJournal, UnknownFutureRecordFormatIsRejectedWithoutGuessing) {
  constexpr std::array<std::byte, 8> kCommittedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                     std::byte{'N'}, std::byte{'G'}, std::byte{'C'},
                                                     std::byte{'M'}, std::byte{'1'}};
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  opened.value().reset();
  const auto future =
      persistence_internal::frameBytes(kCommittedMagic, 99U, std::span<const std::byte>{});
  writeBytes(temporary.path / "committed" / "00000000000000000000.committed", future);
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::UnsupportedFormat);
}

TEST(GraphJournal, CheckpointChecksumCorruptionFailsClosedOnRestart) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 60U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
  }
  flipLastByte(onlyFile(temporary.path / "checkpoints", ".checkpoint"));
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::IntegrityFailure);
}

TEST(GraphJournal, TruncatedCommittedCheckpointFailsClosedOnRestart) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 61U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
  }
  truncateLastByte(onlyFile(temporary.path / "checkpoints", ".checkpoint"));
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::IntegrityFailure);
}

TEST(GraphJournal, PreparedChecksumCorruptionFailsClosedEvenWithoutCommitMarker) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->prepare(request(0U, 65U)));
  }
  flipLastByte(onlyFile(temporary.path / "prepared", ".prepared"));
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::ChecksumMismatch);
}

TEST(GraphJournal, TruncatedPreparedRecordFailsClosedBeforeCommit) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->prepare(request(0U, 66U)));
  }
  truncateLastByte(onlyFile(temporary.path / "prepared", ".prepared"));
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::RecoveryCorruption);
}

TEST(GraphJournal, TornFrameBeforeCompleteVersionIsCorruptionNotVersionRejection) {
  constexpr std::array<std::byte, 8> kPreparedMagic{std::byte{'M'}, std::byte{'R'}, std::byte{'D'},
                                                    std::byte{'N'}, std::byte{'G'}, std::byte{'P'},
                                                    std::byte{'R'}, std::byte{'1'}};
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  opened.value().reset();
  writeBytes(temporary.path / "prepared" / "torn.prepared", kPreparedMagic);
  const auto recovered = GraphJournal::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::RecoveryCorruption);
}

TEST(GraphJournal, FirstMarkerCorruptionFailsAndLaterForkExposesVerifiedPrefixReadOnly) {
  {
    TemporaryJournal temporary;
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 70U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
    opened.value().reset();
    flipLastByte(onlyFile(temporary.path / "committed", ".committed"));
    const auto recovered = GraphJournal::open(temporary.config());
    ASSERT_FALSE(recovered);
    EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::IntegrityFailure);
  }
  {
    TemporaryJournal temporary;
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 71U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
    opened.value().reset();
    const auto marker = onlyFile(temporary.path / "committed", ".committed");
    const auto conflicting = temporary.path / "committed" / "00000000000000000001.committed";
    std::filesystem::copy_file(marker, conflicting);
    const auto recovered = GraphJournal::open(temporary.config());
    ASSERT_TRUE(recovered);
    EXPECT_EQ(recovered.value()->status().state, GraphJournalState::ReadOnlyIntegrityFailure);
    ASSERT_TRUE(recovered.value()->status().integrity_detail.has_value());
    ASSERT_TRUE(recovered.value()->latestCommit().value().has_value());
    EXPECT_EQ(recovered.value()->latestCommit().value()->revision, GlobalGraphRevision(0U));
    const auto rejected = recovered.value()->prepare(request(1U, 72U));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::IntegrityFailure);
  }
}

TEST(GraphJournal, CorruptUnreachableCheckpointIsDiscardedWithoutHidingCommittedPrefix) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 73U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
  }
  const auto committed_checkpoint = onlyFile(temporary.path / "checkpoints", ".checkpoint");
  const auto unreachable = temporary.path / "checkpoints" / "orphan.checkpoint";
  std::filesystem::copy_file(committed_checkpoint, unreachable);
  flipLastByte(unreachable);

  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_FALSE(std::filesystem::exists(unreachable));
  EXPECT_GE(recovered.value()->status().retired_unreachable_files, 1U);
  ASSERT_TRUE(recovered.value()->latestCommit().value().has_value());
  EXPECT_EQ(recovered.value()->latestCommit().value()->revision, GlobalGraphRevision(0U));
}

TEST(GraphJournal, CorruptSecondCommittedMarkerRetainsOnlyVerifiedRevisionReadOnly) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto first = opened.value()->prepare(request(0U, 74U));
    ASSERT_TRUE(first);
    ASSERT_TRUE(opened.value()->commitPrepared(first.value()));
    const auto second = opened.value()->prepare(request(1U, 75U));
    ASSERT_TRUE(second);
    ASSERT_TRUE(opened.value()->commitPrepared(second.value()));
  }
  flipLastByte(temporary.path / "committed" / "00000000000000000001.committed");
  auto recovered = GraphJournal::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().state, GraphJournalState::ReadOnlyIntegrityFailure);
  EXPECT_EQ(recovered.value()->status().committed_revisions, 1U);
  ASSERT_TRUE(recovered.value()->latestCommit().value().has_value());
  EXPECT_EQ(recovered.value()->latestCommit().value()->revision, GlobalGraphRevision(0U));
}

TEST(GraphJournal, RecomputedPreparedChecksumCannotHideMalformedEnum) {
  TemporaryJournal temporary;
  const auto config = temporary.config();
  {
    auto opened = GraphJournal::open(config);
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->prepare(request(0U, 76U)));
  }
  auto path = onlyFile(temporary.path / "prepared", ".prepared");
  auto bytes = readFrameBytes(path);
  bytes.back() = std::byte{0xff};
  path = rewriteFramedRecord(path, std::move(bytes), ".prepared");
  (void)path;
  const auto recovered = GraphJournal::open(config);
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::RecoveryCorruption);
}

TEST(GraphJournal, NonFinitePoseAndCovarianceFailBeforeDurableMutation) {
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  auto invalid_pose = request(0U, 80U);
  invalid_pose.checkpoint.boundaries.front().T_map_submap.translation().x() =
      std::numeric_limits<double>::quiet_NaN();
  const auto rejected_pose = opened.value()->prepare(std::move(invalid_pose));
  ASSERT_FALSE(rejected_pose);
  EXPECT_EQ(rejected_pose.error().code, GraphJournalErrorCode::InvalidCheckpoint);
  EXPECT_EQ(opened.value()->status().durable_bytes, 0U);

  auto invalid_covariance = request(0U, 81U);
  invalid_covariance.checkpoint.recovery.boundary_marginals.front().covariance.matrix(0, 0) =
      std::numeric_limits<double>::infinity();
  const auto rejected_covariance = opened.value()->prepare(std::move(invalid_covariance));
  ASSERT_FALSE(rejected_covariance);
  EXPECT_EQ(rejected_covariance.error().code, GraphJournalErrorCode::InvalidCheckpoint);
  EXPECT_EQ(opened.value()->status().durable_bytes, 0U);

  auto indefinite_covariance = request(0U, 82U);
  indefinite_covariance.checkpoint.recovery.boundary_marginals.front().covariance.matrix(0, 0) =
      -1.0;
  const auto rejected_indefinite = opened.value()->prepare(std::move(indefinite_covariance));
  ASSERT_FALSE(rejected_indefinite);
  EXPECT_EQ(rejected_indefinite.error().code, GraphJournalErrorCode::InvalidCheckpoint);
  EXPECT_EQ(opened.value()->status().durable_bytes, 0U);
}

TEST(GraphJournal, DuplicateInputIdentityFailsBeforeDurableMutation) {
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  auto duplicate = request(0U, 85U);
  duplicate.inputs.push_back(duplicate.inputs.front());
  duplicate.inputs.back().disposition = GraphJournalDisposition::Rejected;
  const auto rejected = opened.value()->prepare(std::move(duplicate));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::InvalidTransaction);
  EXPECT_EQ(opened.value()->status().durable_bytes, 0U);
}

TEST(GraphJournal, PreparedCapabilityTamperingFailsClosed) {
  TemporaryJournal temporary;
  auto opened = GraphJournal::open(temporary.config());
  ASSERT_TRUE(opened);
  const auto prepared = opened.value()->prepare(request(0U, 90U));
  ASSERT_TRUE(prepared);
  PreparedGraphCommit tampered = prepared.value();
  tampered.checkpoint_checksum[0] ^= 0x01U;
  const auto rejected = opened.value()->commitPrepared(tampered);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::PreparedChecksumConflict);
  EXPECT_FALSE(opened.value()->latestCommit().value().has_value());
}

TEST(GraphJournal, RevisionCapacityBecomesReadOnlyWithoutDeletingRecoveryPath) {
  TemporaryJournal temporary;
  auto config = temporary.config();
  config.maximum_revisions = 1U;
  {
    auto opened = GraphJournal::open(config);
    ASSERT_TRUE(opened);
    const auto prepared = opened.value()->prepare(request(0U, 100U));
    ASSERT_TRUE(prepared);
    ASSERT_TRUE(opened.value()->commitPrepared(prepared.value()));
    EXPECT_EQ(opened.value()->status().state, GraphJournalState::ReadOnlyCapacityDegraded);
    const auto rejected = opened.value()->prepare(request(1U, 101U));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::CapacityExceeded);
    EXPECT_EQ(opened.value()->status().committed_revisions, 1U);
  }
  auto recovered = GraphJournal::open(config);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().state, GraphJournalState::ReadOnlyCapacityDegraded);
  ASSERT_TRUE(recovered.value()->latestCommit().value().has_value());
  EXPECT_EQ(recovered.value()->latestCommit().value()->revision, GlobalGraphRevision(0U));
}

TEST(GraphJournal, ReducedCheckpointBoundsRejectDurableStateOnRestart) {
  TemporaryJournal temporary;
  {
    auto opened = GraphJournal::open(temporary.config());
    ASSERT_TRUE(opened);
    const auto first = opened.value()->prepare(request(0U, 105U));
    ASSERT_TRUE(first);
    ASSERT_TRUE(opened.value()->commitPrepared(first.value()));
    const auto second = opened.value()->prepare(request(1U, 106U));
    ASSERT_TRUE(second);
    ASSERT_TRUE(opened.value()->commitPrepared(second.value()));
  }

  auto reduced = temporary.config();
  reduced.checkpoint_limits.maximum_boundaries = 1U;
  const auto recovered = GraphJournal::open(reduced);
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, GraphJournalErrorCode::CapacityExceeded);
}

TEST(GraphJournal, SmallerRestartCapacityCannotCommitPreviouslyPreparedNextRevision) {
  TemporaryJournal temporary;
  auto original_config = temporary.config();
  original_config.maximum_revisions = 2U;
  PreparedGraphCommit next_revision;
  {
    auto opened = GraphJournal::open(original_config);
    ASSERT_TRUE(opened);
    const auto initial = opened.value()->prepare(request(0U, 110U));
    ASSERT_TRUE(initial);
    ASSERT_TRUE(opened.value()->commitPrepared(initial.value()));
    const auto prepared = opened.value()->prepare(request(1U, 111U));
    ASSERT_TRUE(prepared);
    next_revision = prepared.value();
  }

  auto reduced_config = original_config;
  reduced_config.maximum_revisions = 1U;
  auto recovered = GraphJournal::open(reduced_config);
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().state, GraphJournalState::ReadOnlyCapacityDegraded);
  const auto rejected = recovered.value()->commitPrepared(next_revision);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::CapacityExceeded);
  ASSERT_TRUE(recovered.value()->latestCommit().value().has_value());
  EXPECT_EQ(recovered.value()->latestCommit().value()->revision, GlobalGraphRevision(0U));
}

TEST(GraphJournal, ReplayRequestsAreStrictlyBounded) {
  TemporaryJournal temporary;
  auto config = temporary.config();
  config.maximum_replay_records = 1U;
  auto opened = GraphJournal::open(config);
  ASSERT_TRUE(opened);
  const auto rejected = opened.value()->commitsSince(std::nullopt, 2U);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, GraphJournalErrorCode::ReplayLimitExceeded);
}

}  // namespace
}  // namespace meridian::global
