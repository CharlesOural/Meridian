#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <boost/pointer_cast.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "direct_lidar_factor.hpp"
#include "finalized_lidar_factor_test_utils.hpp"
#include "gtsam_conventions.hpp"
#include "lidar_registration_cloud_test_utils.hpp"
#include "meridian/local/lidar_registration.hpp"

namespace meridian::local::gtsam_api {

struct DirectLidarFactorSufficientStatisticsTestAccess {
  static void evaluate(const DirectLidarFactor& factor, const core::Pose3d& T_target_source,
                       Eigen::Matrix<double, 6, 1>* residual,
                       Eigen::Matrix<double, 6, 6>* jacobian) {
    factor.evaluateRelative(T_target_source, residual, jacobian);
  }
};

}  // namespace meridian::local::gtsam_api

namespace meridian::local {
namespace {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Matrix36 = Eigen::Matrix<double, 3, 6>;
using Matrix63 = Eigen::Matrix<double, 6, 3>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

struct FactorFixture {
  std::shared_ptr<const LidarFactorSnapshot> snapshot;
  LidarRegistrationConfig config;
};

struct DirectRowEvaluation {
  Vector6 residual{Vector6::Zero()};
  Matrix6 jacobian{Matrix6::Zero()};
};

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
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

[[nodiscard]] Matrix6 tangentPermutation() {
  Matrix6 permutation = Matrix6::Zero();
  permutation.topRightCorner<3, 3>().setIdentity();
  permutation.bottomLeftCorner<3, 3>().setIdentity();
  return permutation;
}

[[nodiscard]] std::vector<Eigen::Vector3d> volumePoints() {
  std::vector<Eigen::Vector3d> points;
  for (int x = -3; x <= 3; ++x) {
    for (int y = -2; y <= 2; ++y) {
      for (int z = -1; z <= 1; ++z) {
        points.emplace_back(0.38 * static_cast<double>(x), 0.43 * static_cast<double>(y),
                            0.47 * static_cast<double>(z) + 0.025 * static_cast<double>(x * y));
      }
    }
  }
  return points;
}

[[nodiscard]] FactorFixture makeFixture() {
  const std::vector<Eigen::Vector3d> source_points = volumePoints();
  const core::Pose3d truth =
      pose(Eigen::Vector3d{0.11, -0.07, 0.045}, Eigen::Vector3d{0.021, -0.014, 0.017});
  const core::Pose3d seed =
      pose(Eigen::Vector3d{0.055, -0.025, 0.018}, Eigen::Vector3d{0.012, -0.007, 0.009});

  std::vector<Eigen::Vector3d> target_points;
  target_points.reserve(source_points.size());
  for (std::size_t index = 0U; index < source_points.size(); ++index) {
    Eigen::Vector3d target = truth * source_points[index];
    if (index % 11U == 0U) {
      target.z() += 0.20;
    }
    target_points.push_back(target);
  }

  constexpr double kIndexResolutionM = 0.25;
  const auto source = test::sealedLidarRegistrationCloud(
      source_points, core::MeasurementId{61U}, core::FusionTime{2'000}, seed, kIndexResolutionM);
  const auto target = test::sealedLidarRegistrationCloud(target_points, core::MeasurementId{60U},
                                                         core::FusionTime{1'000}, core::Pose3d{},
                                                         kIndexResolutionM);
  const LidarRegistrationTarget record{core::StateId{19U}, target->reference_time, target,
                                       target->T_odom_imu_seed,
                                       target->T_odom_imu_seed.inverse() * source->T_odom_imu_seed};

  LidarRegistrationConfig config;
  config.target_voxel_resolution_m = kIndexResolutionM;
  config.source_voxel_size_m = 0.15;
  config.maximum_correspondence_distance_m = 0.60;
  config.minimum_huber_delta_m = 0.05;
  config.maximum_huber_delta_m = 0.30;
  config.relative_normalized_observable_eigenvalue = 1.0e-5;
  config.maximum_translation_information = 80.0;
  const auto registered =
      registerLidarScan(core::StateId{20U}, source, std::span{&record, 1U}, config);
  if (!registered) {
    throw std::runtime_error("direct LiDAR sufficient-statistics fixture registration failed: " +
                             registered.error().detail);
  }
  if (registered.value().target_snapshots.size() != 1U) {
    throw std::runtime_error(
        "direct LiDAR sufficient-statistics fixture did not produce one target snapshot");
  }
  return FactorFixture{registered.value().target_snapshots.front(), config};
}

// Independent point-row oracle. It reconstructs the admission projection from
// the immutable snapshot, then revisits every frozen scalar-weighted P2P row.
// It intentionally consumes none of DirectLidarFactor's sufficient statistics.
template <typename Snapshot, typename TargetPointAccessor>
[[nodiscard]] DirectRowEvaluation directRowOracleImpl(
    const Snapshot& snapshot, const core::RankAwareInformation& admission_information,
    std::size_t rank, double information_scale, const core::Pose3d& T_target_source,
    TargetPointAccessor target_point) {
  struct AdmissionRow {
    double whitener{};
    Matrix36 robust_jacobian{Matrix36::Zero()};
  };

  const auto rows = snapshot.rows();
  const Eigen::Matrix3d association_rotation = snapshot.associationPose().rotationMatrix();
  std::vector<AdmissionRow> admission_rows;
  admission_rows.reserve(rows.size());
  Matrix6 raw_hessian = Matrix6::Zero();
  for (const auto& row : rows) {
    AdmissionRow admitted;
    admitted.whitener =
        std::sqrt(row.association_huber_weight) / snapshot.residualStandardDeviationM();
    admitted.robust_jacobian.template leftCols<3>() = -admitted.whitener * association_rotation;
    admitted.robust_jacobian.template rightCols<3>() =
        admitted.whitener * association_rotation * skew(row.source_point);
    raw_hessian.noalias() += admitted.robust_jacobian.transpose() * admitted.robust_jacobian;
    admission_rows.push_back(admitted);
  }

  const Eigen::SelfAdjointEigenSolver<Matrix6> solver(symmetrized(raw_hessian));
  EXPECT_EQ(solver.info(), Eigen::Success);
  const double threshold = 1.0e-10 * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
  Matrix6 pseudoinverse = Matrix6::Zero();
  for (Eigen::Index index = 0; index < 6; ++index) {
    if (solver.eigenvalues()(index) > threshold) {
      const Vector6 basis = solver.eigenvectors().col(index);
      pseudoinverse.noalias() += (1.0 / solver.eigenvalues()(index)) * basis * basis.transpose();
    }
  }

  const Eigen::Matrix3d rotation = T_target_source.rotationMatrix();
  const Eigen::Vector3d translation = T_target_source.translation();
  DirectRowEvaluation evaluation;
  for (std::size_t row_index = 0U; row_index < rows.size(); ++row_index) {
    Matrix63 projection = Matrix63::Zero();
    for (std::size_t mode = 0U; mode < rank; ++mode) {
      const Eigen::Index index = static_cast<Eigen::Index>(mode);
      const double eigenvalue = admission_information.eigenvalues(index) / information_scale;
      projection.row(index) = std::sqrt(eigenvalue) *
                              admission_information.basis.col(index).transpose() * pseudoinverse *
                              admission_rows[row_index].robust_jacobian.transpose();
    }

    const auto& row = rows[row_index];
    const Eigen::Vector3d geometric_residual =
        target_point(row) - (rotation * row.source_point + translation);
    evaluation.residual.noalias() +=
        projection * admission_rows[row_index].whitener * geometric_residual;

    Matrix36 geometric_jacobian;
    geometric_jacobian.leftCols<3>() = -rotation;
    geometric_jacobian.rightCols<3>() = rotation * skew(row.source_point);
    evaluation.jacobian.noalias() +=
        projection * admission_rows[row_index].whitener * geometric_jacobian;
  }
  return evaluation;
}

[[nodiscard]] DirectRowEvaluation directRowOracle(
    const LidarFactorSnapshot& snapshot, const core::RankAwareInformation& admission_information,
    std::size_t rank, double information_scale, const core::Pose3d& T_target_source) {
  return directRowOracleImpl(snapshot, admission_information, rank, information_scale,
                             T_target_source,
                             [](const FrozenPointCorrespondence& row) -> const Eigen::Vector3d& {
                               return row.target_point;
                             });
}

[[nodiscard]] DirectRowEvaluation directRowOracle(
    const FinalizedMapLidarFactorSnapshot& snapshot,
    const core::RankAwareInformation& admission_information, std::size_t rank,
    double information_scale, const core::Pose3d& T_odom_source) {
  return directRowOracleImpl(
      snapshot, admission_information, rank, information_scale, T_odom_source,
      [](const FrozenFinalizedMapPointCorrespondence& row) -> const Eigen::Vector3d& {
        return row.target_point_odom;
      });
}

TEST(DirectLidarFactorSufficientStatistics, MatchesEveryFrozenWeightedPointRowAtMultiplePoses) {
  const FactorFixture fixture = makeFixture();
  ASSERT_TRUE(fixture.snapshot);
  EXPECT_TRUE(std::any_of(
      fixture.snapshot->rows().begin(), fixture.snapshot->rows().end(),
      [](const FrozenPointCorrespondence& row) { return row.association_huber_weight < 1.0; }));

  constexpr double kInformationScale = 0.37;
  const gtsam::Key target_key = gtsam::Symbol('x', 0U);
  const gtsam::Key source_key = gtsam::Symbol('x', 1U);
  const gtsam_api::DirectLidarFactor factor(target_key, source_key, fixture.snapshot,
                                            fixture.config, kInformationScale);
  ASSERT_GE(factor.dim(), 1U);

  const std::array<Vector6, 3> offsets{
      (Vector6() << 0.013, -0.009, 0.006, 0.008, -0.005, 0.004).finished(),
      (Vector6() << -0.021, 0.014, -0.011, -0.006, 0.009, -0.007).finished(),
      (Vector6() << 0.034, 0.018, -0.016, 0.012, 0.007, -0.010).finished()};
  const gtsam::Pose3 target_pose(gtsam::Rot3::RzRyRx(0.19, -0.11, 0.24),
                                 gtsam::Point3(1.2, -0.8, 0.45));

  for (const Vector6& offset : offsets) {
    const core::Pose3d relative = fixture.snapshot->associationPose() * core::Pose3d::exp(offset);
    const DirectRowEvaluation oracle =
        directRowOracle(*fixture.snapshot, factor.admissionInformation(), factor.dim(),
                        kInformationScale, relative);
    Vector6 residual;
    Matrix6 jacobian;
    gtsam_api::DirectLidarFactorSufficientStatisticsTestAccess::evaluate(factor, relative,
                                                                         &residual, &jacobian);
    EXPECT_TRUE(residual.isApprox(oracle.residual, 2.0e-9))
        << "sufficient=" << residual.transpose() << " direct=" << oracle.residual.transpose();
    EXPECT_TRUE(jacobian.isApprox(oracle.jacobian, 2.0e-9)) << "sufficient=\n"
                                                            << jacobian << "\ndirect=\n"
                                                            << oracle.jacobian;

    const gtsam::Pose3 source_pose = target_pose.compose(gtsam_api::toGtsamPose(relative));
    gtsam::Values values;
    values.insert(target_key, target_pose);
    values.insert(source_key, source_pose);
    const Eigen::Index rank = static_cast<Eigen::Index>(factor.dim());
    const double expected_error =
        0.5 * kInformationScale * oracle.residual.head(rank).squaredNorm();
    EXPECT_NEAR(factor.error(values), expected_error, 2.0e-9 * std::max(1.0, expected_error));

    gtsam::Matrix66 relative_from_target;
    gtsam::Matrix66 relative_from_source;
    static_cast<void>(target_pose.between(source_pose, relative_from_target, relative_from_source));
    const Eigen::Matrix<double, Eigen::Dynamic, 6> relative_jacobian =
        std::sqrt(kInformationScale) * oracle.jacobian.topRows(rank) * tangentPermutation();
    const Eigen::VectorXd scaled_residual =
        std::sqrt(kInformationScale) * oracle.residual.head(rank);
    const Eigen::Matrix<double, Eigen::Dynamic, 6> target_jacobian =
        relative_jacobian * relative_from_target;
    const Eigen::Matrix<double, Eigen::Dynamic, 6> source_jacobian =
        relative_jacobian * relative_from_source;

    Eigen::Matrix<double, 12, 12> expected_information = Eigen::Matrix<double, 12, 12>::Zero();
    expected_information.topLeftCorner<6, 6>() =
        symmetrized(target_jacobian.transpose() * target_jacobian);
    expected_information.topRightCorner<6, 6>() = target_jacobian.transpose() * source_jacobian;
    expected_information.bottomLeftCorner<6, 6>() =
        expected_information.topRightCorner<6, 6>().transpose();
    expected_information.bottomRightCorner<6, 6>() =
        symmetrized(source_jacobian.transpose() * source_jacobian);
    const Vector6 expected_target_gradient = target_jacobian.transpose() * scaled_residual;
    const Vector6 expected_source_gradient = source_jacobian.transpose() * scaled_residual;

    const auto linear = boost::dynamic_pointer_cast<gtsam::HessianFactor>(factor.linearize(values));
    ASSERT_TRUE(linear);
    EXPECT_TRUE(linear->information().isApprox(expected_information, 2.0e-9));
    EXPECT_TRUE(linear->linearTerm(linear->begin()).isApprox(-expected_target_gradient, 2.0e-9));
    EXPECT_TRUE(
        linear->linearTerm(linear->begin() + 1).isApprox(-expected_source_gradient, 2.0e-9));
    EXPECT_NEAR(linear->constantTerm(), scaled_residual.squaredNorm(),
                2.0e-9 * std::max(1.0, scaled_residual.squaredNorm()));
  }
}

TEST(DirectLidarFactorSufficientStatistics,
     UnaryFinalizedMapMatchesEveryFrozenWeightedPointRowAtMultiplePoses) {
  const test::FinalizedMapRegistrationFixture fixture =
      test::finalizedMapRegistrationFixture(core::StateId{20U}, core::FusionTime{3'000});
  const auto snapshot = fixture.registration.finalized_map_snapshot;
  ASSERT_TRUE(snapshot);
  ASSERT_FALSE(snapshot->rows().empty());

  constexpr double kInformationScale = 0.23;
  const gtsam::Key source_key = gtsam::Symbol('x', 4U);
  const gtsam_api::DirectLidarFactor factor(source_key, snapshot, fixture.config,
                                            kInformationScale);
  ASSERT_TRUE(factor.unary());
  ASSERT_EQ(factor.keys().size(), 1U);
  ASSERT_GE(factor.dim(), 1U);

  const std::array<Vector6, 3> offsets{
      (Vector6() << 0.011, -0.007, 0.005, 0.006, -0.004, 0.003).finished(),
      (Vector6() << -0.018, 0.012, -0.009, -0.005, 0.007, -0.006).finished(),
      (Vector6() << 0.027, 0.014, -0.013, 0.009, 0.006, -0.008).finished()};
  for (const Vector6& offset : offsets) {
    const core::Pose3d T_odom_source = snapshot->associationPose() * core::Pose3d::exp(offset);
    const DirectRowEvaluation oracle = directRowOracle(
        *snapshot, factor.admissionInformation(), factor.dim(), kInformationScale, T_odom_source);
    Vector6 residual;
    Matrix6 jacobian;
    gtsam_api::DirectLidarFactorSufficientStatisticsTestAccess::evaluate(factor, T_odom_source,
                                                                         &residual, &jacobian);
    EXPECT_TRUE(residual.isApprox(oracle.residual, 2.0e-9));
    EXPECT_TRUE(jacobian.isApprox(oracle.jacobian, 2.0e-9));

    gtsam::Values values;
    values.insert(source_key, gtsam_api::toGtsamPose(T_odom_source));
    const Eigen::Index rank = static_cast<Eigen::Index>(factor.dim());
    const double expected_error =
        0.5 * kInformationScale * oracle.residual.head(rank).squaredNorm();
    EXPECT_NEAR(factor.error(values), expected_error, 2.0e-9 * std::max(1.0, expected_error));

    const Eigen::Matrix<double, Eigen::Dynamic, 6> expected_jacobian =
        std::sqrt(kInformationScale) * oracle.jacobian.topRows(rank) * tangentPermutation();
    const Eigen::VectorXd scaled_residual =
        std::sqrt(kInformationScale) * oracle.residual.head(rank);
    const auto linear = boost::dynamic_pointer_cast<gtsam::HessianFactor>(factor.linearize(values));
    ASSERT_TRUE(linear);
    EXPECT_EQ(linear->keys().size(), 1U);
    EXPECT_TRUE(linear->information().isApprox(
        symmetrized(expected_jacobian.transpose() * expected_jacobian), 2.0e-9));
    EXPECT_TRUE(linear->linearTerm(linear->begin())
                    .isApprox(-expected_jacobian.transpose() * scaled_residual, 2.0e-9));
    EXPECT_NEAR(linear->constantTerm(), scaled_residual.squaredNorm(),
                2.0e-9 * std::max(1.0, scaled_residual.squaredNorm()));
  }
}

TEST(DirectLidarFactorSufficientStatistics, RelativeJacobianMatchesRightPerturbationDifference) {
  const FactorFixture fixture = makeFixture();
  ASSERT_TRUE(fixture.snapshot);
  const gtsam_api::DirectLidarFactor factor(gtsam::Symbol('x', 0U), gtsam::Symbol('x', 1U),
                                            fixture.snapshot, fixture.config);
  const Vector6 offset = (Vector6() << 0.026, -0.017, 0.012, 0.011, -0.008, 0.006).finished();
  const core::Pose3d relative = fixture.snapshot->associationPose() * core::Pose3d::exp(offset);
  Vector6 residual;
  Matrix6 analytic;
  gtsam_api::DirectLidarFactorSufficientStatisticsTestAccess::evaluate(factor, relative, &residual,
                                                                       &analytic);

  Matrix6 numeric = Matrix6::Zero();
  constexpr double kStep = 1.0e-7;
  for (Eigen::Index index = 0; index < 6; ++index) {
    Vector6 delta = Vector6::Zero();
    delta(index) = kStep;
    Vector6 plus;
    Vector6 minus;
    gtsam_api::DirectLidarFactorSufficientStatisticsTestAccess::evaluate(
        factor, relative * core::Pose3d::exp(delta), &plus, nullptr);
    gtsam_api::DirectLidarFactorSufficientStatisticsTestAccess::evaluate(
        factor, relative * core::Pose3d::exp(-delta), &minus, nullptr);
    numeric.col(index) = (plus - minus) / (2.0 * kStep);
  }
  EXPECT_TRUE(analytic.isApprox(numeric, 2.0e-6)) << "analytic=\n"
                                                  << analytic << "\nnumeric=\n"
                                                  << numeric;
}

}  // namespace
}  // namespace meridian::local
