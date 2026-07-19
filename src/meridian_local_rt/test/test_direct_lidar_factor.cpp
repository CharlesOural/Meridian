#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
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

namespace meridian::local {
namespace {

using Matrix6 = Eigen::Matrix<double, 6, 6>;
using Vector6 = Eigen::Matrix<double, 6, 1>;

struct FactorFixture {
  std::shared_ptr<const LidarFactorSnapshot> snapshot;
  LidarRegistrationConfig config;
};

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                const Eigen::Vector3d& rotation) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
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
      // A sparse, repeatable heavy-tailed return population ensures that the
      // factor exercises frozen Huber weights as well as exact inliers.
      target.z() += 0.20;
    }
    target_points.push_back(target);
  }

  constexpr double kIndexResolutionM = 0.25;
  const auto source = test::sealedLidarRegistrationCloud(
      source_points, core::MeasurementId{41U}, core::FusionTime{2'000}, seed, kIndexResolutionM);
  const auto target = test::sealedLidarRegistrationCloud(target_points, core::MeasurementId{40U},
                                                         core::FusionTime{1'000}, core::Pose3d{},
                                                         kIndexResolutionM);
  const LidarRegistrationTarget record{core::StateId{9U}, target->reference_time, target,
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
      registerLidarScan(core::StateId{10U}, source, std::span{&record, 1U}, config);
  if (!registered) {
    throw std::runtime_error("direct LiDAR factor fixture registration failed: " +
                             registered.error().detail);
  }
  if (registered.value().target_snapshots.size() != 1U) {
    throw std::runtime_error("direct LiDAR factor fixture did not produce one target snapshot");
  }
  return FactorFixture{registered.value().target_snapshots.front(), config};
}

[[nodiscard]] Matrix6 informationMatrix(const core::RankAwareInformation& information) {
  return information.basis * information.eigenvalues.asDiagonal() * information.basis.transpose();
}

[[nodiscard]] bool positiveSemidefinite(const Eigen::MatrixXd& matrix) {
  if (matrix.rows() != matrix.cols() || !matrix.allFinite()) {
    return false;
  }
  const Eigen::MatrixXd symmetric = 0.5 * (matrix + matrix.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
  return solver.info() == Eigen::Success && solver.eigenvalues().allFinite() &&
         solver.eigenvalues().minCoeff() >=
             -1.0e-8 * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
}

[[nodiscard]] bool augmentedQuadraticIsPositiveSemidefinite(const gtsam::HessianFactor& factor) {
  const Eigen::MatrixXd information = factor.information();
  const Eigen::VectorXd gradient = factor.linearTerm();
  Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(information.rows() + 1, information.cols() + 1);
  augmented.topLeftCorner(information.rows(), information.cols()) = information;
  augmented.topRightCorner(information.rows(), 1) = -gradient;
  augmented.bottomLeftCorner(1, information.cols()) = -gradient.transpose();
  augmented(augmented.rows() - 1, augmented.cols() - 1) = factor.constantTerm();
  return positiveSemidefinite(augmented);
}

TEST(DirectLidarFactor, EvaluationIsStatelessAndClonePreservesSealedInputs) {
  const FactorFixture fixture = makeFixture();
  ASSERT_TRUE(fixture.snapshot);
  const gtsam::Key target_key = gtsam::Symbol('x', 7U);
  const gtsam::Key source_key = gtsam::Symbol('x', 11U);
  constexpr double kInformationScale = 0.42;
  gtsam_api::DirectLidarFactor factor(target_key, source_key, fixture.snapshot, fixture.config,
                                      kInformationScale);

  ASSERT_EQ(factor.keys().size(), 2U);
  EXPECT_EQ(factor.keys().at(0), target_key);
  EXPECT_EQ(factor.keys().at(1), source_key);
  EXPECT_EQ(factor.snapshot().get(), fixture.snapshot.get());
  EXPECT_EQ(factor.config(), fixture.config);
  EXPECT_DOUBLE_EQ(factor.informationScale(), kInformationScale);
  EXPECT_GT(factor.characteristicLengthM(), 0.0);
  EXPECT_EQ(factor.dim(), factor.admissionInformation().rank);
  EXPECT_GE(factor.dim(), 1U);
  EXPECT_LE(factor.dim(), 6U);

  const core::ContentHash checksum_before = fixture.snapshot->checksum();
  const gtsam::Pose3 target_pose(gtsam::Rot3::RzRyRx(0.17, -0.09, 0.22),
                                 gtsam::Point3(1.2, -0.8, 0.45));
  const core::Pose3d relative =
      fixture.snapshot->associationPose() *
      core::Pose3d::exp((Vector6() << 0.009, -0.006, 0.004, 0.005, -0.003, 0.002).finished());
  const gtsam::Pose3 source_pose = target_pose.compose(gtsam_api::toGtsamPose(relative));
  gtsam::Values values;
  values.insert(target_key, target_pose);
  values.insert(source_key, source_pose);

  const double first_error = factor.error(values);
  const auto first_linear = factor.linearize(values);
  const double second_error = factor.error(values);
  const auto second_linear = factor.linearize(values);
  EXPECT_TRUE(std::isfinite(first_error));
  EXPECT_DOUBLE_EQ(first_error, second_error);
  EXPECT_TRUE(first_linear->equals(*second_linear, 0.0));
  EXPECT_EQ(fixture.snapshot->checksum(), checksum_before);

  const auto clone = boost::dynamic_pointer_cast<gtsam_api::DirectLidarFactor>(factor.clone());
  ASSERT_TRUE(clone);
  EXPECT_NE(clone.get(), &factor);
  EXPECT_EQ(clone->snapshot().get(), fixture.snapshot.get());
  EXPECT_EQ(clone->config(), fixture.config);
  EXPECT_DOUBLE_EQ(clone->informationScale(), kInformationScale);
  EXPECT_DOUBLE_EQ(clone->error(values), first_error);
  EXPECT_TRUE(clone->linearize(values)->equals(*first_linear, 0.0));
}

TEST(DirectLidarFactor, BinaryLinearizationMatchesFiniteDifferenceAndIsPositiveSemidefinite) {
  const FactorFixture fixture = makeFixture();
  ASSERT_TRUE(fixture.snapshot);
  const gtsam::Key target_key = gtsam::Symbol('x', 2U);
  const gtsam::Key source_key = gtsam::Symbol('x', 5U);
  gtsam_api::DirectLidarFactor factor(target_key, source_key, fixture.snapshot, fixture.config);

  const gtsam::Pose3 target_pose(gtsam::Rot3::RzRyRx(0.15, -0.08, 0.21),
                                 gtsam::Point3(1.0, 2.0, -0.5));
  const core::Pose3d relative =
      fixture.snapshot->associationPose() *
      core::Pose3d::exp((Vector6() << 0.006, -0.004, 0.005, 0.003, -0.002, 0.0025).finished());
  const gtsam::Pose3 source_pose = target_pose.compose(gtsam_api::toGtsamPose(relative));
  gtsam::Values values;
  values.insert(target_key, target_pose);
  values.insert(source_key, source_pose);

  const auto linear = boost::dynamic_pointer_cast<gtsam::HessianFactor>(factor.linearize(values));
  ASSERT_TRUE(linear);
  const Vector6 analytic_target = -linear->linearTerm(linear->begin());
  const Vector6 analytic_source = -linear->linearTerm(linear->begin() + 1);
  Vector6 numeric_target = Vector6::Zero();
  Vector6 numeric_source = Vector6::Zero();
  constexpr double kStep = 1.0e-6;
  for (Eigen::Index index = 0; index < 6; ++index) {
    Vector6 delta = Vector6::Zero();
    delta(index) = kStep;

    gtsam::Values target_plus(values);
    gtsam::Values target_minus(values);
    target_plus.update(target_key, target_pose.retract(delta));
    target_minus.update(target_key, target_pose.retract(-delta));
    numeric_target(index) =
        (factor.error(target_plus) - factor.error(target_minus)) / (2.0 * kStep);

    gtsam::Values source_plus(values);
    gtsam::Values source_minus(values);
    source_plus.update(source_key, source_pose.retract(delta));
    source_minus.update(source_key, source_pose.retract(-delta));
    numeric_source(index) =
        (factor.error(source_plus) - factor.error(source_minus)) / (2.0 * kStep);
  }
  EXPECT_TRUE(analytic_target.isApprox(numeric_target, 5.0e-5))
      << "analytic=" << analytic_target.transpose() << " numeric=" << numeric_target.transpose();
  EXPECT_TRUE(analytic_source.isApprox(numeric_source, 5.0e-5))
      << "analytic=" << analytic_source.transpose() << " numeric=" << numeric_source.transpose();

  gtsam::VectorValues zero;
  zero.insert(target_key, Vector6::Zero());
  zero.insert(source_key, Vector6::Zero());
  EXPECT_NEAR(linear->error(zero), factor.error(values), 1.0e-12);
  EXPECT_TRUE(positiveSemidefinite(linear->information()));
  EXPECT_TRUE(augmentedQuadraticIsPositiveSemidefinite(*linear));

  const Matrix6 admitted = informationMatrix(factor.admissionInformation());
  EXPECT_TRUE(positiveSemidefinite(admitted));
  EXPECT_LE(fixture.snapshot->geometricInformationScale(), 1.0);
  EXPECT_GT(factor.admissionInformation()
                .eigenvalues.head(static_cast<Eigen::Index>(factor.admissionInformation().rank))
                .maxCoeff(),
            factor.admissionInformation()
                    .eigenvalues.head(static_cast<Eigen::Index>(factor.admissionInformation().rank))
                    .minCoeff() *
                1.01);
}

TEST(DirectLidarFactor,
     UnaryFinalizedMapLinearizationMatchesFiniteDifferenceAndIsPositiveSemidefinite) {
  const test::FinalizedMapRegistrationFixture fixture =
      test::finalizedMapRegistrationFixture(core::StateId{30U}, core::FusionTime{4'000});
  const auto snapshot = fixture.registration.finalized_map_snapshot;
  ASSERT_TRUE(snapshot);
  const gtsam::Key source_key = gtsam::Symbol('x', 8U);
  constexpr double kInformationScale = 0.19;
  gtsam_api::DirectLidarFactor factor(source_key, snapshot, fixture.config, kInformationScale);
  ASSERT_TRUE(factor.unary());
  ASSERT_EQ(factor.keys(), gtsam::KeyVector{source_key});
  EXPECT_EQ(factor.snapshot(), nullptr);
  EXPECT_EQ(factor.finalizedMapSnapshot().get(), snapshot.get());

  const core::Pose3d source_pose_meridian =
      snapshot->associationPose() *
      core::Pose3d::exp((Vector6() << 0.008, -0.005, 0.004, 0.004, -0.003, 0.002).finished());
  const gtsam::Pose3 source_pose = gtsam_api::toGtsamPose(source_pose_meridian);
  gtsam::Values values;
  values.insert(source_key, source_pose);

  const auto linear = boost::dynamic_pointer_cast<gtsam::HessianFactor>(factor.linearize(values));
  ASSERT_TRUE(linear);
  ASSERT_EQ(linear->keys().size(), 1U);
  const Vector6 analytic = -linear->linearTerm(linear->begin());
  Vector6 numeric = Vector6::Zero();
  constexpr double kStep = 1.0e-6;
  for (Eigen::Index index = 0; index < 6; ++index) {
    Vector6 delta = Vector6::Zero();
    delta(index) = kStep;
    gtsam::Values plus(values);
    gtsam::Values minus(values);
    plus.update(source_key, source_pose.retract(delta));
    minus.update(source_key, source_pose.retract(-delta));
    numeric(index) = (factor.error(plus) - factor.error(minus)) / (2.0 * kStep);
  }
  EXPECT_TRUE(analytic.isApprox(numeric, 5.0e-5))
      << "analytic=" << analytic.transpose() << " numeric=" << numeric.transpose();

  gtsam::VectorValues zero;
  zero.insert(source_key, Vector6::Zero());
  EXPECT_NEAR(linear->error(zero), factor.error(values), 1.0e-12);
  EXPECT_TRUE(positiveSemidefinite(linear->information()));
  EXPECT_TRUE(augmentedQuadraticIsPositiveSemidefinite(*linear));
  EXPECT_TRUE(positiveSemidefinite(informationMatrix(factor.admissionInformation())));

  const auto clone = boost::dynamic_pointer_cast<gtsam_api::DirectLidarFactor>(factor.clone());
  ASSERT_TRUE(clone);
  EXPECT_TRUE(clone->unary());
  EXPECT_EQ(clone->finalizedMapSnapshot().get(), snapshot.get());
  EXPECT_DOUBLE_EQ(clone->error(values), factor.error(values));
  EXPECT_TRUE(clone->linearize(values)->equals(*linear, 0.0));

  EXPECT_THROW(gtsam_api::DirectLidarFactor(source_key, nullptr, fixture.config),
               std::invalid_argument);
  LidarRegistrationConfig stale = fixture.config;
  stale.residual_standard_deviation_m *= 2.0;
  EXPECT_THROW(gtsam_api::DirectLidarFactor(source_key, snapshot, stale), std::invalid_argument);
}

TEST(DirectLidarFactor, RegistrationSnapshotRejectsStaleOrMismatchedAdmissionProfiles) {
  const FactorFixture fixture = makeFixture();
  ASSERT_TRUE(fixture.snapshot);
  const gtsam::Key target_key = gtsam::Symbol('x', 0U);
  const gtsam::Key source_key = gtsam::Symbol('x', 1U);

  EXPECT_THROW(
      gtsam_api::DirectLidarFactor(target_key, target_key, fixture.snapshot, fixture.config),
      std::invalid_argument);
  EXPECT_THROW(gtsam_api::DirectLidarFactor(target_key, source_key, nullptr, fixture.config),
               std::invalid_argument);
  EXPECT_THROW(
      gtsam_api::DirectLidarFactor(target_key, source_key, fixture.snapshot, fixture.config, 0.0),
      std::invalid_argument);

  LidarRegistrationConfig stale_noise_profile = fixture.config;
  stale_noise_profile.residual_standard_deviation_m *= 2.0;
  EXPECT_THROW(
      gtsam_api::DirectLidarFactor(target_key, source_key, fixture.snapshot, stale_noise_profile),
      std::invalid_argument);

  LidarRegistrationConfig stale_robust_profile = fixture.config;
  stale_robust_profile.minimum_huber_delta_m = fixture.snapshot->huberDeltaM() + 0.01;
  stale_robust_profile.maximum_huber_delta_m = std::max(stale_robust_profile.minimum_huber_delta_m,
                                                        stale_robust_profile.maximum_huber_delta_m);
  ASSERT_TRUE(isValidLidarRegistrationConfig(stale_robust_profile));
  EXPECT_THROW(
      gtsam_api::DirectLidarFactor(target_key, source_key, fixture.snapshot, stale_robust_profile),
      std::invalid_argument);

  LidarRegistrationConfig stale_support_profile = fixture.config;
  stale_support_profile.minimum_correspondences = fixture.snapshot->rows().size() + 1U;
  ASSERT_TRUE(isValidLidarRegistrationConfig(stale_support_profile));
  EXPECT_THROW(
      gtsam_api::DirectLidarFactor(target_key, source_key, fixture.snapshot, stale_support_profile),
      std::invalid_argument);
}

}  // namespace
}  // namespace meridian::local
