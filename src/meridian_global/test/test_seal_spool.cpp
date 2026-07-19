#include <gtest/gtest.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "meridian/global/seal_spool.hpp"
#include "../src/persistence_internal.hpp"

namespace meridian::global {
namespace {

[[nodiscard]] core::ContentHash hash(std::uint8_t value) {
  core::ContentHash output{};
  output[0] = value;
  output[15] = static_cast<std::uint8_t>(value ^ 0x5aU);
  output[31] = static_cast<std::uint8_t>(value ^ 0xa5U);
  return output;
}

[[nodiscard]] core::ObservationLineage emptyLineage(std::uint64_t id) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId(id);
  lineage.checksum = hash(static_cast<std::uint8_t>(id));
  return lineage;
}

[[nodiscard]] SparseSubmapSeal makeSeal(std::uint64_t id, std::uint8_t content,
                                        bool durable_place_payload = false,
                                        bool in_memory_place_payload = false,
                                        std::uint64_t trace = 1U) {
  const FinalizedSubmapFrame submap{
      core::SubmapRef{core::SessionId(5U), core::OdomEpoch(3U), core::SubmapId(id),
                      core::CalibrationEpoch(7U), core::SubmapContentRevision(2U), hash(content)},
      core::Pose3d{}, core::FusionTime{100}};
  auto seal = std::make_shared<SparseSubmapSealRecord>(submap);
  seal->header.schema_version = 1U;
  seal->header.trace = core::TraceId(trace);
  seal->header.producer = core::ProducerId(4U);
  seal->header.session = core::SessionId(5U);
  seal->header.created_at = core::FusionTime{100};
  seal->header.config = core::ConfigRevision(6U);
  seal->header.direct_calibration = core::CalibrationEpoch(7U);
  seal->identity = core::SparseSubmapSealIdentity{submap.ref, hash(content)};
  seal->core_interval = {core::FusionTime{0}, core::FusionTime{100}};
  seal->start_boundary_state = core::StateId(id * 10U + 1U);
  seal->end_boundary_state = core::StateId(id * 10U + 2U);
  seal->final_local_revision = core::LocalGraphRevision(8U);
  seal->calibration_epochs = {core::CalibrationEpoch(7U)};
  seal->core_state_ids = {seal->start_boundary_state};

  FinalizedSubmapStateRecord state;
  state.state = seal->start_boundary_state;
  state.exact_time = core::FusionTime{0};
  state.final_local_revision = core::LocalGraphRevision(8U);
  state.T_submap_imu = core::Pose3d{};
  state.pose_covariance.matrix = core::Matrix6d::Identity() * 0.01;
  seal->finalized_trajectory.push_back(state);
  seal->condensed_factor_ids = {core::FactorId(id * 100U + 1U)};
  seal->factor_lineage = emptyLineage(id * 1000U + 1U);
  seal->registration_proxy.voxel_resolution_m = 0.4;
  seal->registration_proxy.points.push_back(
      {core::Vector3d{1.0, 2.0, 3.0}, core::Vector3d::UnitZ(), 1.0});
  seal->registration_proxy.checksum = hash(static_cast<std::uint8_t>(content + 1U));

  if (durable_place_payload || in_memory_place_payload) {
    VisualPlacePayloadIndexEntry entry;
    entry.frame = core::CameraFrameId(id * 100U + 1U);
    entry.camera = core::CameraId(0U);
    entry.state = seal->start_boundary_state;
    entry.terminal_time = core::FusionTime{50};
    entry.T_submap_camera = core::Pose3d{};
    entry.payload.checksum = hash(static_cast<std::uint8_t>(content + 2U));
    if (durable_place_payload) {
      core::BlobRef blob;
      blob.store = core::BlobStoreId(11U);
      blob.id = core::BlobId(id);
      blob.checksum = entry.payload.checksum;
      blob.layout = core::LayoutId(12U);
      blob.bytes = 64U;
      blob.storage = core::BlobStorage::DurableSpool;
      entry.payload.record = blob;
    } else {
      auto bytes = std::make_shared<std::vector<std::byte>>(64U, std::byte{0x4a});
      entry.payload.in_memory = std::const_pointer_cast<const std::vector<std::byte>>(bytes);
    }
    entry.lineage = emptyLineage(id * 1000U + 2U);
    seal->visual_place_index.push_back(std::move(entry));
  }
  return std::const_pointer_cast<const SparseSubmapSealRecord>(seal);
}

class TemporarySpool {
public:
  TemporarySpool() {
    static std::atomic<std::uint64_t> counter{0U};
    path = std::filesystem::temp_directory_path() /
           ("meridian_seal_spool_" + std::to_string(static_cast<unsigned long long>(::getpid())) +
            "_" + std::to_string(counter.fetch_add(1U)));
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  ~TemporarySpool() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }

  [[nodiscard]] SealSpoolConfig config() const {
    SealSpoolConfig output;
    output.root_directory = path;
    output.maximum_bytes = 8U * 1024U * 1024U;
    output.maximum_seal_record_bytes = 2U * 1024U * 1024U;
    output.maximum_seals = 16U;
    output.maximum_outbox_entries = 16U;
    output.maximum_replay_records = 16U;
    output.maximum_vector_elements = 1024U;
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

[[nodiscard]] std::vector<std::byte> readBytes(const std::filesystem::path& path) {
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

[[nodiscard]] std::filesystem::path rewriteSealWithValidFrameChecksum(
    const std::filesystem::path& path, std::vector<std::byte> bytes) {
  constexpr std::size_t kChecksumOffset = 8U + 4U + 8U;
  constexpr std::size_t kPayloadOffset = persistence_internal::kFrameOverhead;
  EXPECT_GT(bytes.size(), kPayloadOffset);
  const auto checksum = persistence_internal::hashBytes(
      std::span<const std::byte>(bytes).subspan(kPayloadOffset));
  for (std::size_t index = 0U; index < checksum.size(); ++index) {
    bytes[kChecksumOffset + index] = static_cast<std::byte>(checksum[index]);
  }
  const auto replacement = path.parent_path() /
                           (persistence_internal::hashHex(checksum) + ".seal");
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

TEST(SealSpool, RecoversDurableSealAndPreservesCanonicalSequenceAcrossRestart) {
  TemporarySpool temporary;
  const SparseSubmapSeal seal = makeSeal(1U, 20U, true);
  core::ContentHash first_checksum{};
  {
    auto opened = SealSpool::open(temporary.config());
    ASSERT_TRUE(opened);
    auto enqueue = opened.value()->enqueue(seal);
    ASSERT_TRUE(enqueue);
    EXPECT_EQ(enqueue.value().sequence, OutboxSequence(1U));
    EXPECT_FALSE(enqueue.value().idempotent);
    EXPECT_GT(enqueue.value().serialized_record_bytes, 0U);
    first_checksum = enqueue.value().serialized_record_checksum;
  }

  auto recovered = SealSpool::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().head, OutboxSequence(1U));
  const auto replay = recovered.value()->replaySince(OutboxSequence(0U), 16U);
  ASSERT_TRUE(replay);
  ASSERT_EQ(replay.value().size(), 1U);
  EXPECT_EQ(replay.value().front().serialized_record_checksum, first_checksum);
  EXPECT_EQ(replay.value().front().seal->identity, seal->identity);
  ASSERT_EQ(replay.value().front().seal->visual_place_index.size(), 1U);
  ASSERT_TRUE(replay.value().front().seal->visual_place_index.front().payload.record.has_value());
  EXPECT_EQ(replay.value().front().seal->visual_place_index.front().payload.record->storage,
            core::BlobStorage::DurableSpool);

  const auto idempotent = recovered.value()->enqueue(seal);
  ASSERT_TRUE(idempotent);
  EXPECT_TRUE(idempotent.value().idempotent);
  EXPECT_EQ(idempotent.value().sequence, OutboxSequence(1U));
  EXPECT_EQ(idempotent.value().serialized_record_checksum, first_checksum);
}

TEST(SealSpool, RemovesTornTemporaryFilesBeforeRecovery) {
  TemporarySpool temporary;
  {
    auto opened = SealSpool::open(temporary.config());
    ASSERT_TRUE(opened);
  }
  const auto torn_seal = temporary.path / "seals" / "orphan.seal.tmp.42";
  const auto torn_entry = temporary.path / "outbox" / "entry.tmp.42";
  {
    std::ofstream output(torn_seal, std::ios::binary);
    output << "partial";
  }
  {
    std::ofstream output(torn_entry, std::ios::binary);
    output << "partial";
  }
  auto recovered = SealSpool::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().recovered_temporary_files, 2U);
  EXPECT_FALSE(std::filesystem::exists(torn_seal));
  EXPECT_FALSE(std::filesystem::exists(torn_entry));
}

TEST(SealSpool, ChecksumMismatchFailsClosedDuringRestart) {
  TemporarySpool temporary;
  {
    auto opened = SealSpool::open(temporary.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->enqueue(makeSeal(1U, 21U)));
  }
  const auto seal_file = onlyFile(temporary.path / "seals", ".seal");
  {
    std::fstream file(seal_file, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.good());
    file.seekg(-1, std::ios::end);
    char value{};
    file.read(&value, 1);
    value = static_cast<char>(static_cast<unsigned char>(value) ^ 0x01U);
    file.seekp(-1, std::ios::end);
    file.write(&value, 1);
    file.flush();
  }
  const auto recovered = SealSpool::open(temporary.config());
  ASSERT_FALSE(recovered);
  EXPECT_EQ(recovered.error().code, SealSpoolErrorCode::ChecksumMismatch);
}

TEST(SealSpool, DuplicateIdentityAndCanonicalContentConflictsAreRejected) {
  TemporarySpool temporary;
  auto opened = SealSpool::open(temporary.config());
  ASSERT_TRUE(opened);
  const SparseSubmapSeal original = makeSeal(1U, 22U);
  ASSERT_TRUE(opened.value()->enqueue(original));

  const auto changed_content = opened.value()->enqueue(makeSeal(1U, 23U));
  ASSERT_FALSE(changed_content);
  EXPECT_EQ(changed_content.error().code, SealSpoolErrorCode::IdentityConflict);

  const auto changed_record = opened.value()->enqueue(makeSeal(1U, 22U, false, false, 99U));
  ASSERT_FALSE(changed_record);
  EXPECT_EQ(changed_record.error().code, SealSpoolErrorCode::IdentityConflict);
  EXPECT_EQ(opened.value()->status().unacknowledged_entries, 1U);
}

TEST(SealSpool, AcknowledgementIsExactDurableAndIdempotent) {
  TemporarySpool temporary;
  SealAcknowledgement acknowledgement;
  {
    auto opened = SealSpool::open(temporary.config());
    ASSERT_TRUE(opened);
    auto enqueue = opened.value()->enqueue(makeSeal(1U, 24U));
    ASSERT_TRUE(enqueue);
    acknowledgement = {enqueue.value().sequence, enqueue.value().identity,
                       enqueue.value().identity.seal_checksum};
    auto conflicting = acknowledgement;
    conflicting.seal_checksum = hash(99U);
    const auto rejected = opened.value()->acknowledge(conflicting);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, SealSpoolErrorCode::AcknowledgementConflict);

    SealSpoolTiming timing;
    const auto first = opened.value()->acknowledge(acknowledgement, &timing);
    ASSERT_TRUE(first);
    EXPECT_TRUE(first.value());
    EXPECT_GE(timing.acknowledgement_write_and_sync_us, 0);
    const auto repeated = opened.value()->acknowledge(acknowledgement);
    ASSERT_TRUE(repeated);
    EXPECT_FALSE(repeated.value());
    const auto replay = opened.value()->replaySince(OutboxSequence(0U), 16U);
    ASSERT_TRUE(replay);
    EXPECT_TRUE(replay.value().empty());
  }
  auto recovered = SealSpool::open(temporary.config());
  ASSERT_TRUE(recovered);
  EXPECT_EQ(recovered.value()->status().acknowledged_entries, 1U);
  EXPECT_EQ(recovered.value()->status().unacknowledged_entries, 0U);
  const auto repeated = recovered.value()->acknowledge(acknowledgement);
  ASSERT_TRUE(repeated);
  EXPECT_FALSE(repeated.value());
}

TEST(SealSpool, CapacityFailureNeverDeletesUnacknowledgedSeal) {
  TemporarySpool temporary;
  auto config = temporary.config();
  config.maximum_seals = 1U;
  auto opened = SealSpool::open(config);
  ASSERT_TRUE(opened);
  ASSERT_TRUE(opened.value()->enqueue(makeSeal(1U, 25U)));
  const auto rejected = opened.value()->enqueue(makeSeal(2U, 26U));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, SealSpoolErrorCode::CapacityExceeded);
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::DegradedStorageCapacity);
  EXPECT_EQ(opened.value()->status().durable_seals, 1U);
  EXPECT_EQ(opened.value()->status().unacknowledged_entries, 1U);
  const auto replay = opened.value()->replaySince(OutboxSequence(0U), 16U);
  ASSERT_TRUE(replay);
  ASSERT_EQ(replay.value().size(), 1U);
  EXPECT_EQ(replay.value().front().identity.ref.id, core::SubmapId(1U));
}

TEST(SealSpool, InMemoryPlacePayloadFailsBeforeAnyDurableMutation) {
  TemporarySpool temporary;
  auto opened = SealSpool::open(temporary.config());
  ASSERT_TRUE(opened);
  const auto rejected = opened.value()->enqueue(makeSeal(1U, 27U, false, true));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, SealSpoolErrorCode::UnsupportedInMemoryPayload);
  EXPECT_EQ(opened.value()->status().durable_seals, 0U);
  EXPECT_EQ(opened.value()->status().outbox_entries, 0U);
}

TEST(SealSpool, ExistingIdenticalSealReconcilesACompletedRenameAndContinues) {
  TemporarySpool staging;
  TemporarySpool target;
  const auto seal = makeSeal(3U, 31U);
  {
    auto opened = SealSpool::open(staging.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->enqueue(seal));
  }
  const auto staged_file = onlyFile(staging.path / "seals", ".seal");
  auto opened = SealSpool::open(target.config());
  ASSERT_TRUE(opened);
  std::filesystem::copy_file(staged_file, target.path / "seals" / staged_file.filename());
  const auto reconciled = opened.value()->enqueue(seal);
  ASSERT_TRUE(reconciled) << reconciled.error().detail;
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::Ready);
  EXPECT_EQ(opened.value()->status().unacknowledged_entries, 1U);
}

TEST(SealSpool, ExistingDifferentImmutableBytesAreAConflictNotAmbiguousIo) {
  TemporarySpool staging;
  TemporarySpool target;
  const auto seal = makeSeal(9U, 37U);
  {
    auto opened = SealSpool::open(staging.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->enqueue(seal));
  }
  const auto staged_file = onlyFile(staging.path / "seals", ".seal");
  auto opened = SealSpool::open(target.config());
  ASSERT_TRUE(opened);
  const auto conflicting = target.path / "seals" / staged_file.filename();
  std::filesystem::copy_file(staged_file, conflicting);
  {
    std::fstream file(conflicting, std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(file.good());
    file.seekp(-1, std::ios::end);
    file.put(static_cast<char>(0x7f));
    file.flush();
    ASSERT_TRUE(file.good());
  }
  const auto rejected = opened.value()->enqueue(seal);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, SealSpoolErrorCode::IdentityConflict);
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::Ready);
  EXPECT_FALSE(opened.value()->status().ambiguous_destination.has_value());
}

TEST(SealSpool, AmbiguousDestinationLatchesIntegrityAndExactRetryClearsIt) {
  TemporarySpool staging;
  TemporarySpool target;
  const auto seal = makeSeal(4U, 32U);
  {
    auto opened = SealSpool::open(staging.config());
    ASSERT_TRUE(opened);
    ASSERT_TRUE(opened.value()->enqueue(seal));
  }
  const auto staged_file = onlyFile(staging.path / "seals", ".seal");
  auto opened = SealSpool::open(target.config());
  ASSERT_TRUE(opened);
  const auto ambiguous = target.path / "seals" / staged_file.filename();
  std::filesystem::create_directory(ambiguous);

  const auto rejected = opened.value()->enqueue(seal);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, SealSpoolErrorCode::IntegrityFailure);
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::ReadOnlyIntegrityFailure);
  ASSERT_EQ(opened.value()->status().ambiguous_destination, ambiguous);

  std::filesystem::remove_all(ambiguous);
  const auto retried = opened.value()->enqueue(seal);
  ASSERT_TRUE(retried) << retried.error().detail;
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::Ready);
  EXPECT_FALSE(opened.value()->status().ambiguous_destination.has_value());
}

TEST(SealSpool, RetryAfterRenameBeforeDirectorySyncReconcilesExactDestination) {
  TemporarySpool temporary;
  auto opened = SealSpool::open(temporary.config());
  ASSERT_TRUE(opened);
  const auto seal = makeSeal(8U, 36U);
  persistence_internal::setAtomicWriteFailpointForTesting(
      persistence_internal::AtomicWriteFailpoint::AfterRenameBeforeDirectorySync);

  const auto interrupted = opened.value()->enqueue(seal);
  ASSERT_FALSE(interrupted);
  EXPECT_EQ(interrupted.error().code, SealSpoolErrorCode::IntegrityFailure);
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::ReadOnlyIntegrityFailure);
  ASSERT_TRUE(opened.value()->status().ambiguous_destination.has_value());
  EXPECT_TRUE(std::filesystem::is_regular_file(
      *opened.value()->status().ambiguous_destination));

  const auto retried = opened.value()->enqueue(seal);
  ASSERT_TRUE(retried) << retried.error().detail;
  EXPECT_EQ(opened.value()->status().state, SealSpoolState::Ready);
  EXPECT_EQ(opened.value()->status().durable_seals, 1U);
  EXPECT_EQ(opened.value()->status().unacknowledged_entries, 1U);
}

TEST(SealSpool, InvalidInformationCapAndIndefiniteSealCovarianceFailBeforeMutation) {
  TemporarySpool temporary;
  auto opened = SealSpool::open(temporary.config());
  ASSERT_TRUE(opened);
  auto invalid_cap = std::const_pointer_cast<SparseSubmapSealRecord>(makeSeal(5U, 33U));
  core::CorrelationDeclaration declaration;
  declaration.group = core::CorrelationGroupId(1U);
  declaration.policy = core::CorrelationPolicyRevision(1U);
  declaration.treatment = core::CorrelationTreatment::CovarianceInflationAndInformationCap;
  declaration.covariance_inflation = 1.0;
  declaration.total_information_cap = -1.0;
  invalid_cap->factor_lineage.correlations.push_back(declaration);
  const auto cap_rejected = opened.value()->enqueue(
      std::const_pointer_cast<const SparseSubmapSealRecord>(invalid_cap));
  ASSERT_FALSE(cap_rejected);
  EXPECT_EQ(cap_rejected.error().code, SealSpoolErrorCode::InvalidSeal);

  auto indefinite = std::const_pointer_cast<SparseSubmapSealRecord>(makeSeal(6U, 34U));
  indefinite->finalized_trajectory.front().pose_covariance.matrix(0, 0) = -0.1;
  const auto covariance_rejected = opened.value()->enqueue(
      std::const_pointer_cast<const SparseSubmapSealRecord>(indefinite));
  ASSERT_FALSE(covariance_rejected);
  EXPECT_EQ(covariance_rejected.error().code, SealSpoolErrorCode::InvalidSeal);
  EXPECT_EQ(opened.value()->status().durable_bytes, 0U);
}

TEST(SealSpool, RecomputedChecksumCannotHideMalformedEnumNanOrCount) {
  const auto run_case = [](const auto& mutate, std::size_t maximum_elements = 1024U) {
    TemporarySpool temporary;
    auto config = temporary.config();
    config.maximum_vector_elements = maximum_elements;
    {
      auto opened = SealSpool::open(config);
      ASSERT_TRUE(opened);
      ASSERT_TRUE(opened.value()->enqueue(makeSeal(7U, 35U)));
    }
    auto path = onlyFile(temporary.path / "seals", ".seal");
    auto bytes = readBytes(path);
    mutate(bytes);
    path = rewriteSealWithValidFrameChecksum(path, std::move(bytes));
    (void)path;
    const auto recovered = SealSpool::open(config);
    ASSERT_FALSE(recovered);
    EXPECT_EQ(recovered.error().code, SealSpoolErrorCode::RecoveryCorruption);
  };

  run_case([](std::vector<std::byte>& bytes) { bytes.back() = std::byte{0x7f}; });

  run_case([](std::vector<std::byte>& bytes) {
    constexpr std::array<std::byte, 8> kPointFour{
        std::byte{0x3f}, std::byte{0xd9}, std::byte{0x99}, std::byte{0x99},
        std::byte{0x99}, std::byte{0x99}, std::byte{0x99}, std::byte{0x9a}};
    const auto match = std::search(bytes.begin() +
                                       static_cast<std::ptrdiff_t>(persistence_internal::kFrameOverhead),
                                   bytes.end(), kPointFour.begin(), kPointFour.end());
    ASSERT_NE(match, bytes.end());
    constexpr std::array<std::byte, 8> kQuietNan{
        std::byte{0x7f}, std::byte{0xf8}, std::byte{0x00}, std::byte{0x00},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
    std::copy(kQuietNan.begin(), kQuietNan.end(), match);
  });

  run_case(
      [](std::vector<std::byte>& bytes) {
        constexpr std::size_t kCalibrationCountPayloadOffset = 317U;
        const std::size_t offset = persistence_internal::kFrameOverhead +
                                   kCalibrationCountPayloadOffset;
        constexpr std::uint64_t kImpossibleCount = 1'000'000U;
        for (std::size_t index = 0U; index < 8U; ++index) {
          bytes[offset + index] =
              static_cast<std::byte>(kImpossibleCount >> ((7U - index) * 8U));
        }
      },
      2'000'000U);
}

}  // namespace
}  // namespace meridian::global
