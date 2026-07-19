#pragma once

// Private GTSAM seam. This header is intentionally not installed.

#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "meridian/local/lidar_registration.hpp"

namespace meridian::local::gtsam_api {

// Stateless direct point-to-point factor with either
//
//   T_target_source = inverse(T_odom_target) * T_odom_source.
//
// for a live binary target, or
//
//   T_target_source = T_odom_source
//
// for the graph-finalized odom-map unary target.
//
// Nearest-neighbor identities, robust admission weights, supported
// directions, and the physical information cap are sealed before this factor
// is constructed. The factor therefore owns no mutable correspondence or
// linearization cache and is safe to clone into an isolated iSAM2 candidate.
// Its six-or-fewer residual modes are a deterministic row-space compression
// of the accepted weighted 3-D point residuals, not a pose measurement
// synthesized by another estimator.
class DirectLidarFactor final : public gtsam::NonlinearFactor {
public:
  DirectLidarFactor(gtsam::Key target_pose_key, gtsam::Key source_pose_key,
                    std::shared_ptr<const LidarFactorSnapshot> snapshot,
                    LidarRegistrationConfig config = {}, double information_scale = 1.0);
  DirectLidarFactor(gtsam::Key source_pose_key,
                    std::shared_ptr<const FinalizedMapLidarFactorSnapshot> snapshot,
                    LidarRegistrationConfig config = {}, double information_scale = 1.0);

  [[nodiscard]] double error(const gtsam::Values& values) const override;
  [[nodiscard]] gtsam::GaussianFactor::shared_ptr linearize(
      const gtsam::Values& values) const override;
  [[nodiscard]] std::size_t dim() const override { return rank_; }
  [[nodiscard]] gtsam::NonlinearFactor::shared_ptr clone() const override;

  [[nodiscard]] const std::shared_ptr<const LidarFactorSnapshot>& snapshot() const noexcept {
    return snapshot_;
  }
  [[nodiscard]] const std::shared_ptr<const FinalizedMapLidarFactorSnapshot>& finalizedMapSnapshot()
      const noexcept {
    return finalized_map_snapshot_;
  }
  [[nodiscard]] bool unary() const noexcept { return static_cast<bool>(finalized_map_snapshot_); }
  [[nodiscard]] const LidarRegistrationConfig& config() const noexcept { return config_; }
  [[nodiscard]] double informationScale() const noexcept { return information_scale_; }
  [[nodiscard]] double characteristicLengthM() const noexcept { return characteristic_length_m_; }
  [[nodiscard]] const core::RankAwareInformation& admissionInformation() const noexcept {
    return admission_information_;
  }

  void print(const std::string& prefix = "",
             const gtsam::KeyFormatter& formatter = gtsam::DefaultKeyFormatter) const override;

private:
  struct CompressedPointRow {
    Eigen::Vector3d source_point{Eigen::Vector3d::Zero()};
    Eigen::Vector3d target_point{Eigen::Vector3d::Zero()};
    double association_distance_squared_m2{};
    double association_huber_weight{1.0};
  };

  void initializeCompressed(const core::Pose3d& association_pose,
                            std::span<const CompressedPointRow> rows,
                            double residual_standard_deviation_m, double huber_delta_m,
                            double characteristic_length_m);
  void evaluateRelative(const core::Pose3d& T_target_source, Eigen::Matrix<double, 6, 1>* residual,
                        Eigen::Matrix<double, 6, 6>* jacobian) const;

  friend struct DirectLidarFactorSufficientStatisticsTestAccess;

  std::shared_ptr<const LidarFactorSnapshot> snapshot_;
  std::shared_ptr<const FinalizedMapLidarFactorSnapshot> finalized_map_snapshot_;
  LidarRegistrationConfig config_;
  double information_scale_{1.0};
  double characteristic_length_m_{};
  std::size_t rank_{};
  core::RankAwareInformation admission_information_;

  // For frozen row i, let
  //
  //   W_i = sqrt(huber_weight_i) / residual_standard_deviation,
  //   P_i = its supported-mode projection.
  //
  // These sums encode sum_i P_i W_i (q_i - T p_i) exactly. Construction is
  // linear in correspondence count; every later optimizer probe is bounded by
  // six residual modes and does not revisit the point rows.
  Eigen::Matrix<double, 6, 1> compressed_target_sum_{Eigen::Matrix<double, 6, 1>::Zero()};
  Eigen::Matrix<double, 6, 3> compressed_coefficient_sum_{Eigen::Matrix<double, 6, 3>::Zero()};
  std::array<Eigen::Matrix3d, 6> compressed_source_moments_{
      Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(),
      Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero(), Eigen::Matrix3d::Zero()};
  std::size_t compressed_row_count_{};
};

}  // namespace meridian::local::gtsam_api
