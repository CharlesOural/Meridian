#include "direct_lidar_factor.hpp"

#include <gtsam/linear/HessianFactor.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <boost/make_shared.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gtsam_conventions.hpp"

namespace meridian::local::gtsam_api {
namespace {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

[[nodiscard]] const Matrix6& tangentPermutation() {
  // Meridian/Sophus: [translation, rotation].
  // GTSAM Pose3:     [rotation, translation].
  static const Matrix6 permutation = [] {
    Matrix6 value = Matrix6::Zero();
    value.topRightCorner<3, 3>().setIdentity();
    value.bottomLeftCorner<3, 3>().setIdentity();
    return value;
  }();
  return permutation;
}

[[nodiscard]] Eigen::Matrix3d skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d matrix;
  matrix << 0.0, -vector.z(), vector.y(), vector.z(), 0.0, -vector.x(), -vector.y(), vector.x(),
      0.0;
  return matrix;
}

[[nodiscard]] Matrix6 symmetrized(const Matrix6& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

[[nodiscard]] core::Pose3d relativePose(const gtsam::Values& values, const gtsam::KeyVector& keys) {
  return fromGtsamPose(
      values.at<gtsam::Pose3>(keys.at(0)).between(values.at<gtsam::Pose3>(keys.at(1))));
}

[[nodiscard]] bool finitePositive(double value) noexcept {
  return std::isfinite(value) && value > 0.0;
}

[[nodiscard]] double huberWeight(double norm, double delta) noexcept {
  return norm <= delta || norm <= std::numeric_limits<double>::epsilon() ? 1.0 : delta / norm;
}

struct AdmissionRow {
  double whitener{};
  Eigen::Matrix<double, 3, 6> robust_jacobian{Eigen::Matrix<double, 3, 6>::Zero()};
};

}  // namespace

DirectLidarFactor::DirectLidarFactor(gtsam::Key target_pose_key, gtsam::Key source_pose_key,
                                     std::shared_ptr<const LidarFactorSnapshot> snapshot,
                                     LidarRegistrationConfig config, double information_scale)
    : gtsam::NonlinearFactor(gtsam::KeyVector{target_pose_key, source_pose_key}),
      snapshot_(std::move(snapshot)),
      config_(std::move(config)),
      information_scale_(information_scale) {
  if (target_pose_key == source_pose_key) {
    throw std::invalid_argument("direct LiDAR target and source pose keys must be distinct");
  }
  if (!snapshot_ || !core::contentHashPresent(snapshot_->checksum()) ||
      !snapshot_->targetState().valid() || !snapshot_->sourceState().valid() ||
      !(snapshot_->targetState() < snapshot_->sourceState()) ||
      !(snapshot_->targetTime() < snapshot_->sourceTime()) ||
      !snapshot_->associationPose().matrix().allFinite() ||
      !finitePositive(snapshot_->residualStandardDeviationM()) ||
      !finitePositive(snapshot_->huberDeltaM())) {
    throw std::invalid_argument(
        "direct LiDAR factor requires one canonical checksum-bearing pose-separated snapshot");
  }
  auto admission = lidarFactorInformation(snapshot_, config_, information_scale_);
  if (!admission) {
    throw std::invalid_argument("direct LiDAR factor admission failed: " +
                                admission.error().detail);
  }
  admission_information_ = std::move(admission).value();
  rank_ = admission_information_.rank;

  std::vector<CompressedPointRow> rows;
  rows.reserve(snapshot_->rows().size());
  for (const FrozenPointCorrespondence& row : snapshot_->rows()) {
    rows.push_back(CompressedPointRow{row.source_point, row.target_point,
                                      row.association_distance_squared_m2,
                                      row.association_huber_weight});
  }
  initializeCompressed(snapshot_->associationPose(), rows, snapshot_->residualStandardDeviationM(),
                       snapshot_->huberDeltaM(), snapshot_->characteristicLengthM());
}

DirectLidarFactor::DirectLidarFactor(
    gtsam::Key source_pose_key, std::shared_ptr<const FinalizedMapLidarFactorSnapshot> snapshot,
    LidarRegistrationConfig config, double information_scale)
    : gtsam::NonlinearFactor(gtsam::KeyVector{source_pose_key}),
      finalized_map_snapshot_(std::move(snapshot)),
      config_(std::move(config)),
      information_scale_(information_scale) {
  if (!finalized_map_snapshot_ || !core::contentHashPresent(finalized_map_snapshot_->checksum()) ||
      !finalized_map_snapshot_->sourceState().valid() ||
      !finalized_map_snapshot_->mapOdomEpoch().valid() ||
      !finalized_map_snapshot_->mapSensor().valid() ||
      finalized_map_snapshot_->mapVersion() == 0U ||
      !core::contentHashPresent(finalized_map_snapshot_->mapChecksum()) ||
      !finalized_map_snapshot_->associationPose().matrix().allFinite() ||
      !finitePositive(finalized_map_snapshot_->residualStandardDeviationM()) ||
      !finitePositive(finalized_map_snapshot_->huberDeltaM())) {
    throw std::invalid_argument(
        "direct finalized-map LiDAR factor requires one canonical checksum-bearing unary "
        "snapshot");
  }
  auto admission =
      lidarFinalizedMapFactorInformation(finalized_map_snapshot_, config_, information_scale_);
  if (!admission) {
    throw std::invalid_argument("direct finalized-map LiDAR factor admission failed: " +
                                admission.error().detail);
  }
  admission_information_ = std::move(admission).value();
  rank_ = admission_information_.rank;

  std::vector<CompressedPointRow> rows;
  rows.reserve(finalized_map_snapshot_->rows().size());
  for (const FrozenFinalizedMapPointCorrespondence& row : finalized_map_snapshot_->rows()) {
    rows.push_back(CompressedPointRow{row.source_point, row.target_point_odom,
                                      row.association_distance_squared_m2,
                                      row.association_huber_weight});
  }
  initializeCompressed(finalized_map_snapshot_->associationPose(), rows,
                       finalized_map_snapshot_->residualStandardDeviationM(),
                       finalized_map_snapshot_->huberDeltaM(),
                       finalized_map_snapshot_->characteristicLengthM());
}

void DirectLidarFactor::initializeCompressed(const core::Pose3d& association_pose,
                                             std::span<const CompressedPointRow> rows,
                                             double residual_standard_deviation_m,
                                             double huber_delta_m, double characteristic_length_m) {
  if (rank_ == 0U || rank_ > 6U || rows.empty()) {
    throw std::runtime_error("direct LiDAR factor has invalid supported rows or rank");
  }

  characteristic_length_m_ = characteristic_length_m;
  if (!finitePositive(characteristic_length_m_)) {
    throw std::runtime_error("direct LiDAR factor characteristic length is invalid");
  }

  const Eigen::Matrix3d association_rotation = association_pose.rotationMatrix();
  const Eigen::Vector3d association_translation = association_pose.translation();
  const double inverse_standard_deviation = 1.0 / residual_standard_deviation_m;
  std::vector<AdmissionRow> admission_rows;
  admission_rows.reserve(rows.size());
  Matrix6 raw_hessian = Matrix6::Zero();
  for (const CompressedPointRow& row : rows) {
    if (!row.source_point.allFinite() || !row.target_point.allFinite() ||
        !std::isfinite(row.association_distance_squared_m2) ||
        row.association_distance_squared_m2 < 0.0 ||
        !finitePositive(row.association_huber_weight) || row.association_huber_weight > 1.0) {
      throw std::runtime_error("direct LiDAR factor contains an invalid frozen point row");
    }

    const Eigen::Vector3d association_residual =
        row.target_point - (association_rotation * row.source_point + association_translation);
    const double distance_squared = std::max(0.0, association_residual.squaredNorm());
    const double expected_weight = huberWeight(std::sqrt(distance_squared), huber_delta_m);
    const double scalar_tolerance =
        1.0e-9 * std::max({1.0, distance_squared, row.association_distance_squared_m2});
    if (std::abs(distance_squared - row.association_distance_squared_m2) > scalar_tolerance ||
        std::abs(expected_weight - row.association_huber_weight) > 1.0e-12) {
      throw std::runtime_error(
          "direct LiDAR factor snapshot distance or robust admission weight is stale");
    }

    AdmissionRow admitted;
    admitted.whitener = std::sqrt(row.association_huber_weight) * inverse_standard_deviation;
    Eigen::Matrix<double, 3, 6> geometric_jacobian;
    geometric_jacobian.leftCols<3>() = -association_rotation;
    geometric_jacobian.rightCols<3>() = association_rotation * skew(row.source_point);
    admitted.robust_jacobian = admitted.whitener * geometric_jacobian;
    raw_hessian.noalias() += admitted.robust_jacobian.transpose() * admitted.robust_jacobian;
    admission_rows.push_back(std::move(admitted));
  }
  raw_hessian = symmetrized(raw_hessian);
  if (!raw_hessian.allFinite()) {
    throw std::runtime_error("direct LiDAR factor raw admission information is non-finite");
  }

  Matrix6 capped_information = Matrix6::Zero();
  for (std::size_t mode = 0U; mode < rank_; ++mode) {
    const Eigen::Index index = static_cast<Eigen::Index>(mode);
    const Vector6 basis = admission_information_.basis.col(index);
    const double eigenvalue = admission_information_.eigenvalues(index) / information_scale_;
    if (!finitePositive(eigenvalue)) {
      throw std::runtime_error("direct LiDAR factor supported information is non-positive");
    }
    capped_information.noalias() += eigenvalue * basis * basis.transpose();
  }

  const Eigen::SelfAdjointEigenSolver<Matrix6> raw_solver(raw_hessian);
  if (raw_solver.info() != Eigen::Success || !raw_solver.eigenvalues().allFinite() ||
      !raw_solver.eigenvectors().allFinite()) {
    throw std::runtime_error("direct LiDAR factor raw information pseudoinverse failed");
  }
  const double raw_threshold =
      1.0e-10 * std::max(1.0, raw_solver.eigenvalues().cwiseAbs().maxCoeff());
  Matrix6 raw_pseudoinverse = Matrix6::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    if (raw_solver.eigenvalues()(index) <= raw_threshold) {
      continue;
    }
    const Vector6 basis = raw_solver.eigenvectors().col(index);
    raw_pseudoinverse.noalias() +=
        (1.0 / raw_solver.eigenvalues()(index)) * basis * basis.transpose();
  }

  Matrix6 reconstructed_jacobian = Matrix6::Zero();
  for (std::size_t row_index = 0U; row_index < admission_rows.size(); ++row_index) {
    const AdmissionRow& admission_row = admission_rows[row_index];
    Eigen::Matrix<double, 6, 3> mode_coefficients = Eigen::Matrix<double, 6, 3>::Zero();
    for (std::size_t mode = 0U; mode < rank_; ++mode) {
      const Eigen::Index index = static_cast<Eigen::Index>(mode);
      const double eigenvalue = admission_information_.eigenvalues(index) / information_scale_;
      const Vector6 basis = admission_information_.basis.col(index);
      mode_coefficients.row(index) = std::sqrt(eigenvalue) * basis.transpose() * raw_pseudoinverse *
                                     admission_row.robust_jacobian.transpose();
    }
    if (!mode_coefficients.allFinite()) {
      throw std::runtime_error("direct LiDAR factor row-space compression is non-finite");
    }
    reconstructed_jacobian.noalias() += mode_coefficients * admission_row.robust_jacobian;

    const CompressedPointRow& row = rows[row_index];
    const Eigen::Matrix<double, 6, 3> coefficient = admission_row.whitener * mode_coefficients;
    compressed_target_sum_.noalias() += coefficient * row.target_point;
    compressed_coefficient_sum_ += coefficient;
    for (std::size_t mode = 0U; mode < rank_; ++mode) {
      compressed_source_moments_[mode].noalias() +=
          coefficient.row(static_cast<Eigen::Index>(mode)).transpose() *
          row.source_point.transpose();
    }
  }
  const Matrix6 reconstructed_information =
      symmetrized(reconstructed_jacobian.topRows(static_cast<Eigen::Index>(rank_)).transpose() *
                  reconstructed_jacobian.topRows(static_cast<Eigen::Index>(rank_)));
  const double reconstruction_scale = std::max(1.0, capped_information.cwiseAbs().maxCoeff());
  if (!reconstructed_information.isApprox(capped_information, 1.0e-7 * reconstruction_scale)) {
    throw std::runtime_error(
        "direct LiDAR factor supported row-space reconstruction is inconsistent");
  }

  compressed_row_count_ = rows.size();
  if (!compressed_target_sum_.allFinite() || !compressed_coefficient_sum_.allFinite()) {
    throw std::runtime_error("direct LiDAR factor sufficient statistics are non-finite");
  }
  for (std::size_t mode = 0U; mode < rank_; ++mode) {
    if (!compressed_source_moments_[mode].allFinite()) {
      throw std::runtime_error("direct LiDAR factor source sufficient statistics are non-finite");
    }
  }
}

void DirectLidarFactor::evaluateRelative(const core::Pose3d& T_target_source, Vector6* residual,
                                         Matrix6* jacobian) const {
  if (residual == nullptr) {
    throw std::invalid_argument("direct LiDAR factor residual output must be present");
  }
  residual->setZero();
  if (jacobian != nullptr) {
    jacobian->setZero();
  }
  const std::size_t sealed_row_count =
      snapshot_ ? snapshot_->rows().size() : finalized_map_snapshot_->rows().size();
  if (!T_target_source.matrix().allFinite() || compressed_row_count_ != sealed_row_count) {
    throw std::runtime_error("direct LiDAR factor evaluation input is invalid");
  }

  const Eigen::Matrix3d rotation = T_target_source.rotationMatrix();
  const Eigen::Vector3d translation = T_target_source.translation();
  *residual = compressed_target_sum_ - compressed_coefficient_sum_ * translation;
  for (std::size_t mode = 0U; mode < rank_; ++mode) {
    const Eigen::Index index = static_cast<Eigen::Index>(mode);
    (*residual)(index) -= compressed_source_moments_[mode].cwiseProduct(rotation).sum();
  }
  if (jacobian != nullptr) {
    jacobian->leftCols<3>().noalias() = -compressed_coefficient_sum_ * rotation;
    for (std::size_t mode = 0U; mode < rank_; ++mode) {
      // If M = R^T D_k, these three components are exactly
      // sum_i A_i(k,:) R [p_i]_x.
      const Eigen::Matrix3d rotation_source_moment =
          rotation.transpose() * compressed_source_moments_[mode];
      const Eigen::Index index = static_cast<Eigen::Index>(mode);
      (*jacobian)(index, 3) = rotation_source_moment(1, 2) - rotation_source_moment(2, 1);
      (*jacobian)(index, 4) = rotation_source_moment(2, 0) - rotation_source_moment(0, 2);
      (*jacobian)(index, 5) = rotation_source_moment(0, 1) - rotation_source_moment(1, 0);
    }
  }
  if (!residual->allFinite() || (jacobian != nullptr && !jacobian->allFinite())) {
    throw std::runtime_error("direct LiDAR factor compressed residual is non-finite");
  }
}

double DirectLidarFactor::error(const gtsam::Values& values) const {
  try {
    Vector6 residual;
    const core::Pose3d pose = unary() ? fromGtsamPose(values.at<gtsam::Pose3>(keys().at(0)))
                                      : relativePose(values, keys());
    evaluateRelative(pose, &residual, nullptr);
    const double value =
        0.5 * information_scale_ * residual.head(static_cast<Eigen::Index>(rank_)).squaredNorm();
    if (std::isfinite(value) && value >= 0.0) {
      return value;
    }
  } catch (const std::exception&) {
    // Invalid optimizer probes are rejectable without mutating factor state.
  }
  return std::numeric_limits<double>::infinity();
}

gtsam::GaussianFactor::shared_ptr DirectLidarFactor::linearize(const gtsam::Values& values) const {
  Vector6 residual;
  Matrix6 relative_jacobian_meridian;
  if (unary()) {
    const gtsam::Pose3& source_pose = values.at<gtsam::Pose3>(keys().at(0));
    evaluateRelative(fromGtsamPose(source_pose), &residual, &relative_jacobian_meridian);
  } else {
    const gtsam::Pose3& target_pose = values.at<gtsam::Pose3>(keys().at(0));
    const gtsam::Pose3& source_pose = values.at<gtsam::Pose3>(keys().at(1));
    const gtsam::Pose3 relative = target_pose.between(source_pose);
    evaluateRelative(fromGtsamPose(relative), &residual, &relative_jacobian_meridian);
  }
  const Eigen::Index rank = static_cast<Eigen::Index>(rank_);
  const double square_root_scale = std::sqrt(information_scale_);
  const Eigen::Matrix<double, Eigen::Dynamic, 6> relative_jacobian_gtsam =
      square_root_scale * relative_jacobian_meridian.topRows(rank) * tangentPermutation();
  const Eigen::VectorXd scaled_residual = square_root_scale * residual.head(rank);

  if (unary()) {
    const Matrix6 source_information =
        symmetrized(relative_jacobian_gtsam.transpose() * relative_jacobian_gtsam);
    const Vector6 source_gradient = relative_jacobian_gtsam.transpose() * scaled_residual;
    const double constant = scaled_residual.squaredNorm();
    if (!source_information.allFinite() || !source_gradient.allFinite() ||
        !std::isfinite(constant)) {
      throw std::runtime_error("direct finalized-map LiDAR factor linearization is non-finite");
    }
    return boost::make_shared<gtsam::HessianFactor>(keys().at(0), source_information,
                                                    -source_gradient, constant);
  }

  const gtsam::Pose3& target_pose = values.at<gtsam::Pose3>(keys().at(0));
  const gtsam::Pose3& source_pose = values.at<gtsam::Pose3>(keys().at(1));
  gtsam::Matrix66 relative_from_target;
  gtsam::Matrix66 relative_from_source;
  static_cast<void>(target_pose.between(source_pose, relative_from_target, relative_from_source));
  const Eigen::Matrix<double, Eigen::Dynamic, 6> target_jacobian =
      relative_jacobian_gtsam * relative_from_target;
  const Eigen::Matrix<double, Eigen::Dynamic, 6> source_jacobian =
      relative_jacobian_gtsam * relative_from_source;

  const Matrix6 target_information = symmetrized(target_jacobian.transpose() * target_jacobian);
  const Matrix6 target_source_information = target_jacobian.transpose() * source_jacobian;
  const Matrix6 source_information = symmetrized(source_jacobian.transpose() * source_jacobian);
  const Vector6 target_gradient = target_jacobian.transpose() * scaled_residual;
  const Vector6 source_gradient = source_jacobian.transpose() * scaled_residual;
  const double constant = scaled_residual.squaredNorm();
  if (!target_information.allFinite() || !target_source_information.allFinite() ||
      !source_information.allFinite() || !target_gradient.allFinite() ||
      !source_gradient.allFinite() || !std::isfinite(constant)) {
    throw std::runtime_error("direct LiDAR factor linearization is non-finite");
  }

  return boost::make_shared<gtsam::HessianFactor>(keys().at(0), keys().at(1), target_information,
                                                  target_source_information, -target_gradient,
                                                  source_information, -source_gradient, constant);
}

gtsam::NonlinearFactor::shared_ptr DirectLidarFactor::clone() const {
  return boost::make_shared<DirectLidarFactor>(*this);
}

void DirectLidarFactor::print(const std::string& prefix,
                              const gtsam::KeyFormatter& formatter) const {
  if (unary()) {
    std::cout << prefix << "DirectLidarFactor(" << formatter(keys().at(0))
              << ") source=" << finalized_map_snapshot_->sourceSweep().value()
              << " finalized_map_version=" << finalized_map_snapshot_->mapVersion()
              << " correspondences=" << finalized_map_snapshot_->rows().size() << " rank=" << rank_
              << " characteristic_length_m=" << characteristic_length_m_
              << " information_scale=" << information_scale_
              << " checksum=" << core::sha256Hex(finalized_map_snapshot_->checksum()) << '\n';
    return;
  }
  std::cout << prefix << "DirectLidarFactor(" << formatter(keys().at(0)) << ", "
            << formatter(keys().at(1)) << ") source=" << snapshot_->sourceSweep().value()
            << " target=" << snapshot_->targetSweep().value()
            << " correspondences=" << snapshot_->rows().size() << " rank=" << rank_
            << " characteristic_length_m=" << characteristic_length_m_
            << " information_scale=" << information_scale_
            << " checksum=" << core::sha256Hex(snapshot_->checksum()) << '\n';
}

}  // namespace meridian::local::gtsam_api
