#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "meridian/core/api.hpp"

namespace meridian::local {

inline constexpr std::string_view kLidarMapRawPayloadChecksumDomain{
    "meridian.local.lidar_map_payload.raw_sweep"};
inline constexpr std::uint32_t kLidarMapRawPayloadChecksumSchemaVersion{1U};
inline constexpr std::string_view kLidarMapPayloadChecksumDomain{
    "meridian.local.lidar_map_payload.accepted"};
inline constexpr std::uint32_t kLidarMapPayloadChecksumSchemaVersion{1U};

enum class LidarMapPayloadErrorCode {
  InvalidIdentity,
  InvalidSweep,
  InvalidLocalization,
  InvalidLineage,
  ChecksumFailure,
};

struct LidarMapPayloadError {
  LidarMapPayloadErrorCode code{};
  std::string detail;
};

// Cheap construction input at the localization/map-admission boundary. The
// raw sweep payload is already immutable and therefore remains shared without
// a point copy. Creation is O(1) in point count: dense-map validation and
// hashing belong to the downstream sealing step, never the local estimator.
struct AcceptedLidarMapInputData {
  core::LidarSweep sweep;
  core::OdomEpoch odom_epoch;
  core::StateId state;
  core::LocalGraphRevision accepted_revision;
  core::FactorBatchId admitting_batch;
  core::SensorRecoveryEpoch recovery_epoch;
  core::Pose3d T_odom_imu;
  core::CalibrationEpoch calibration;
  core::ContentHash registration_cloud_checksum{};
  core::ObservationLineage localization_lineage;
};

// Immutable ticket emitted only after localization and MapAdmissionGate both
// accept the LiDAR batch. Dense mapping must re-deskew this raw sweep using
// accepted/finalized localization; it must never consume tracking deskew.
class AcceptedLidarMapInput final {
public:
  [[nodiscard]] static core::Result<std::shared_ptr<const AcceptedLidarMapInput>,
                                    LidarMapPayloadError>
  create(AcceptedLidarMapInputData data);

  AcceptedLidarMapInput(const AcceptedLidarMapInput&) = delete;
  AcceptedLidarMapInput& operator=(const AcceptedLidarMapInput&) = delete;
  AcceptedLidarMapInput(AcceptedLidarMapInput&&) = delete;
  AcceptedLidarMapInput& operator=(AcceptedLidarMapInput&&) = delete;

  const core::LidarSweep sweep;
  const core::OdomEpoch odom_epoch;
  const core::StateId state;
  const core::LocalGraphRevision accepted_revision;
  const core::FactorBatchId admitting_batch;
  const core::SensorRecoveryEpoch recovery_epoch;
  const core::Pose3d T_odom_imu;
  const core::CalibrationEpoch calibration;
  const core::ContentHash registration_cloud_checksum;
  const core::ObservationLineage localization_lineage;

private:
  explicit AcceptedLidarMapInput(AcceptedLidarMapInputData data);
};

// Canonically sealed form created explicitly by a downstream map worker. This
// O(N) operation validates and hashes every raw row; it is intentionally not
// called by the localization estimator.
class LidarMapPayload final {
public:
  [[nodiscard]] static core::Result<std::shared_ptr<const LidarMapPayload>, LidarMapPayloadError>
  seal(std::shared_ptr<const AcceptedLidarMapInput> input);

  LidarMapPayload(const LidarMapPayload&) = delete;
  LidarMapPayload& operator=(const LidarMapPayload&) = delete;
  LidarMapPayload(LidarMapPayload&&) = delete;
  LidarMapPayload& operator=(LidarMapPayload&&) = delete;

  const std::shared_ptr<const AcceptedLidarMapInput> input;
  const core::ContentHash raw_payload_checksum;
  const core::ContentHash checksum;

private:
  LidarMapPayload(std::shared_ptr<const AcceptedLidarMapInput> input,
                  core::ContentHash raw_checksum, core::ContentHash payload_checksum);
};

}  // namespace meridian::local
