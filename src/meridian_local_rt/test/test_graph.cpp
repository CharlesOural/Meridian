#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <Eigen/Eigenvalues>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "candidate_gate.hpp"
#include "candidate_isolated_isam2.hpp"
#include "gtsam_conventions.hpp"
#include "meridian/local/graph.hpp"

namespace meridian::local {
namespace {

constexpr double kGravity = 9.80665;

class ExponentialScalarFactor final : public gtsam::NoiseModelFactorN<double> {
public:
  using Base = gtsam::NoiseModelFactorN<double>;

  explicit ExponentialScalarFactor(gtsam::Key key)
      : Base(gtsam::noiseModel::Isotropic::Sigma(1U, 1.0), key) {}

  gtsam::Vector evaluateError(
      const double& value, boost::optional<gtsam::Matrix&> jacobian = boost::none) const override {
    const double exponential = std::exp(value);
    if (jacobian) {
      *jacobian = gtsam::Matrix::Constant(1, 1, exponential);
    }
    return gtsam::Vector1(exponential - 1.0);
  }
};

class StationaryScalarFactor final : public gtsam::NoiseModelFactorN<double> {
public:
  using Base = gtsam::NoiseModelFactorN<double>;

  explicit StationaryScalarFactor(gtsam::Key key)
      : Base(gtsam::noiseModel::Isotropic::Sigma(1U, 1.0), key) {}

  gtsam::Vector evaluateError(
      const double& value, boost::optional<gtsam::Matrix&> jacobian = boost::none) const override {
    if (jacobian) {
      *jacobian = gtsam::Matrix::Constant(1, 1, 2.0 * value);
    }
    return gtsam::Vector1(value * value + 1.0);
  }
};

TEST(CandidateGate, AcceptsPointZeroFivePercentOnlyWithFiniteBoundedCorrection) {
  const detail::CandidateGateLimits limits{1.0, 0.35, 1.0e-9, 1.0e-3};
  detail::CandidateGateInput input{1000.0, 1000.5, 0.2, 0.05, true};
  EXPECT_EQ(detail::evaluateCandidateGate(input, limits), detail::CandidateGateDecision::Accepted);

  input.all_state_corrections_finite = false;
  EXPECT_EQ(detail::evaluateCandidateGate(input, limits), detail::CandidateGateDecision::NonFinite);
  input.all_state_corrections_finite = true;
  input.transaction_translation_correction_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(detail::evaluateCandidateGate(input, limits), detail::CandidateGateDecision::NonFinite);
  input.transaction_translation_correction_m = 1.01;
  EXPECT_EQ(detail::evaluateCandidateGate(input, limits),
            detail::CandidateGateDecision::PoseCorrectionLimit);
}

TEST(CandidateGate, RejectsPointTwoPercentAndPreservesSmoothDecrease) {
  const detail::CandidateGateLimits limits{1.0, 0.35, 1.0e-9, 1.0e-3};
  EXPECT_EQ(detail::evaluateCandidateGate({1000.0, 1002.0, 0.2, 0.05, true}, limits),
            detail::CandidateGateDecision::CompleteObjectiveIncrease);
  EXPECT_EQ(detail::evaluateCandidateGate({1000.0, 900.0, 0.2, 0.05, true}, limits),
            detail::CandidateGateDecision::Accepted);
}

TEST(CandidateGate, DefersObjectiveAcceptanceUntilTheTransactionConverges) {
  const detail::CandidateGateLimits limits{1.0, 0.35, 1.0e-9, 1.0e-3};
  EXPECT_EQ(
      detail::evaluateCandidateGate(
          {100.0, 130.0, 0.2, 0.05, true, detail::CandidateGatePhase::NonlinearIteration}, limits),
      detail::CandidateGateDecision::Accepted);
  EXPECT_EQ(detail::evaluateCandidateGate(
                {1000.0, 99.0, 0.2, 0.05, true, detail::CandidateGatePhase::ConvergedTransaction},
                limits),
            detail::CandidateGateDecision::Accepted);
  EXPECT_EQ(detail::evaluateCandidateGate(
                {1000.0, 1002.0, 0.2, 0.05, true, detail::CandidateGatePhase::ConvergedTransaction},
                limits),
            detail::CandidateGateDecision::CompleteObjectiveIncrease);
}

TEST(CandidateGate, CumulativeImuGateRejectsReinforcingSubLimitCorrections) {
  const detail::CandidateGateLimits limits{1.0, 0.35, 1.0e-9, 1.0e-3};
  const gtsam::Pose3 imu_prediction;
  const gtsam::Pose3 proposal(gtsam::Rot3{}, gtsam::Point3(0.6, 0.0, 0.0));
  const gtsam::Pose3 settled(gtsam::Rot3{}, gtsam::Point3(1.2, 0.0, 0.0));
  const double proposal_to_settled = (settled.translation() - proposal.translation()).norm();
  const double imu_to_settled = (settled.translation() - imu_prediction.translation()).norm();
  EXPECT_EQ(detail::evaluateCandidateGate({10.0, 9.0, proposal_to_settled, 0.0, true}, limits),
            detail::CandidateGateDecision::Accepted);
  EXPECT_EQ(detail::evaluateCandidateGate(
                {10.0, 9.0, std::max(proposal_to_settled, imu_to_settled), 0.0, true}, limits),
            detail::CandidateGateDecision::PoseCorrectionLimit);
}

TEST(CandidateGate, CumulativeImuGateAllowsCancellingSubLimitCorrections) {
  const detail::CandidateGateLimits limits{1.0, 0.35, 1.0e-9, 1.0e-3};
  const gtsam::Pose3 imu_prediction;
  const gtsam::Pose3 proposal(gtsam::Rot3{}, gtsam::Point3(0.6, 0.0, 0.0));
  const gtsam::Pose3 settled(gtsam::Rot3{}, gtsam::Point3(0.1, 0.0, 0.0));
  const double proposal_to_settled = (settled.translation() - proposal.translation()).norm();
  const double imu_to_settled = (settled.translation() - imu_prediction.translation()).norm();
  EXPECT_EQ(detail::evaluateCandidateGate(
                {10.0, 9.0, std::max(proposal_to_settled, imu_to_settled), 0.0, true}, limits),
            detail::CandidateGateDecision::Accepted);
}

[[nodiscard]] core::ObservationLineage lineage(std::uint64_t id) {
  core::ObservationLineage result;
  result.id = core::ObservationLineageId{id};
  return result;
}

[[nodiscard]] NavigationCovariance diagonalCovariance(double scale = 1.0e-3) {
  NavigationCovariance covariance;
  for (Eigen::Index index = 0; index < 15; ++index) {
    covariance.matrix(index, index) = scale * static_cast<double>(index + 1);
  }
  return covariance;
}

[[nodiscard]] NavigationCovariance correlatedPoseCovariance() {
  NavigationCovariance covariance;
  covariance.matrix.setZero();
  for (Eigen::Index index = 0; index < 15; ++index) {
    covariance.matrix(index, index) = 0.05 + 0.01 * static_cast<double>(index);
  }
  Eigen::Matrix3d rotation_position_cross;
  rotation_position_cross << 0.004, -0.003, 0.002, 0.001, 0.005, -0.002, -0.003, 0.002, 0.006;
  covariance.matrix.block<3, 3>(0, 6) = rotation_position_cross;
  covariance.matrix.block<3, 3>(6, 0) = rotation_position_cross.transpose();
  return covariance;
}

[[nodiscard]] core::PoseCovariance poseCovarianceFromNavigation(
    const NavigationCovariance& navigation) {
  core::PoseCovariance pose;
  pose.matrix.topLeftCorner<3, 3>() = navigation.matrix.block<3, 3>(6, 6);
  pose.matrix.topRightCorner<3, 3>() = navigation.matrix.block<3, 3>(6, 0);
  pose.matrix.bottomLeftCorner<3, 3>() = navigation.matrix.block<3, 3>(0, 6);
  pose.matrix.bottomRightCorner<3, 3>() = navigation.matrix.block<3, 3>(0, 0);
  return pose;
}

[[nodiscard]] bool positiveSemidefinite(const core::PoseCovariance& covariance) {
  if (covariance.tangent != core::PoseTangentConvention::RightTranslationFirst ||
      !covariance.matrix.allFinite() ||
      !covariance.matrix.isApprox(covariance.matrix.transpose(), 1.0e-10)) {
    return false;
  }
  const Eigen::SelfAdjointEigenSolver<core::Matrix6d> eigen_solver(covariance.matrix);
  return eigen_solver.info() == Eigen::Success && eigen_solver.eigenvalues().minCoeff() >= -1.0e-10;
}

[[nodiscard]] NavigationCovariance fullGraphNavigationCovariance(const gtsam::ISAM2& solver,
                                                                 const gtsam::Values& linearization,
                                                                 gtsam::Key pose_key,
                                                                 gtsam::Key velocity_key,
                                                                 gtsam::Key bias_key) {
  const gtsam::Marginals marginals(solver.getFactorsUnsafe(), linearization, gtsam::Marginals::QR);
  const gtsam::KeyVector requested{pose_key, velocity_key, bias_key};
  const gtsam::JointMarginal joint = marginals.jointMarginalCovariance(requested);
  Eigen::Matrix<double, 15, 15> covariance;
  covariance.setZero();
  covariance.block<6, 6>(0, 0) = joint(pose_key, pose_key);
  covariance.block<6, 3>(0, 6) = joint(pose_key, velocity_key);
  covariance.block<6, 6>(0, 9) = joint(pose_key, bias_key);
  covariance.block<3, 6>(6, 0) = joint(velocity_key, pose_key);
  covariance.block<3, 3>(6, 6) = joint(velocity_key, velocity_key);
  covariance.block<3, 6>(6, 9) = joint(velocity_key, bias_key);
  covariance.block<6, 6>(9, 0) = joint(bias_key, pose_key);
  covariance.block<6, 3>(9, 6) = joint(bias_key, velocity_key);
  covariance.block<6, 6>(9, 9) = joint(bias_key, bias_key);
  return gtsam_api::fromGtsamNavigationCovariance(covariance);
}

[[nodiscard]] bool positiveDefinite(const NavigationCovariance& covariance) {
  return covariance.matrix.allFinite() &&
         Eigen::LLT<Eigen::Matrix<double, 15, 15>>(covariance.matrix).info() == Eigen::Success;
}

[[nodiscard]] LocalGraphInitialization initialization(core::NavStateEstimate state = {}) {
  return LocalGraphInitialization{core::OdomEpoch{1}, core::StateId{10},    core::FusionTime{0},
                                  std::move(state),   diagonalCovariance(), lineage(1)};
}

[[nodiscard]] ImuInterval imuInterval(std::int64_t start_ns, std::int64_t end_ns,
                                      const Eigen::Vector3d& specific_force,
                                      const Eigen::Vector3d& angular_rate = Eigen::Vector3d::Zero(),
                                      std::size_t segments = 100U) {
  ImuInterval interval;
  interval.support = core::TimeRange{{start_ns}, {end_ns}};
  for (std::size_t index = 0; index <= segments; ++index) {
    const auto numerator = static_cast<std::int64_t>(index);
    const auto denominator = static_cast<std::int64_t>(segments);
    const std::int64_t time = start_ns + (end_ns - start_ns) * numerator / denominator;
    interval.knots.push_back(InterpolatedImuSample{core::FusionTime{time}, specific_force,
                                                   angular_rate, core::MeasurementId{index + 1U},
                                                   core::MeasurementId{index + 1U}});
  }
  interval.maximum_raw_gap =
      core::Duration{(end_ns - start_ns) / static_cast<std::int64_t>(segments)};
  return interval;
}

[[nodiscard]] EquidistantCameraParameters visualCameraParameters() {
  return EquidistantCameraParameters{720U,    540U,     352.779,  354.609, 359.035,
                                     260.546, -0.04217, -0.00413, 0.00179, -0.00063};
}

[[nodiscard]] VisualObservationRef visualObservation(core::MeasurementId frame, core::StateId state,
                                                     core::FusionTime time,
                                                     const core::Pose3d& T_odom_imu,
                                                     const Eigen::Vector3d& point_odom) {
  VisualObservationRef observation;
  observation.frame = frame;
  observation.state = state;
  observation.exact_time = time;
  observation.camera = core::CameraId(0U);
  observation.calibration = core::CalibrationEpoch(7U);
  observation.pixel_covariance = Eigen::Matrix2d::Identity();
  observation.camera_model = visualCameraParameters();
  observation.imu_from_camera = core::ImuFromCameraTransform(core::Pose3d{});
  const EquidistantCamera camera(observation.camera_model);
  const auto projection = camera.project(T_odom_imu.inverse() * point_odom);
  EXPECT_TRUE(projection) << projection.error().detail;
  if (projection) {
    observation.pixel = projection.value().pixel;
  }
  return observation;
}

[[nodiscard]] core::ObservationLineage visualFactorLineage(core::FactorId factor,
                                                           const VisualObservationRef& anchor,
                                                           const VisualObservationRef& observer) {
  core::ObservationLineage result;
  result.id = core::ObservationLineageId(1'000U + factor.value());
  const core::DerivedRecordId consumer(1'000U + factor.value());
  core::ObservationUsage anchor_usage;
  anchor_usage.slice.root = anchor.frame;
  anchor_usage.slice.kind = core::SliceKind::Whole;
  anchor_usage.slice.calibration = anchor.calibration;
  anchor_usage.role = core::ObservationRole::ConditioningOnly;
  anchor_usage.consumer = consumer;
  result.usage.push_back(anchor_usage);
  core::ObservationUsage observer_usage;
  observer_usage.slice.root = observer.frame;
  observer_usage.slice.kind = core::SliceKind::Whole;
  observer_usage.slice.calibration = observer.calibration;
  observer_usage.role = core::ObservationRole::PrimaryResidual;
  observer_usage.consumer = consumer;
  observer_usage.factor_group = core::FactorGroupId(factor.value());
  result.usage.push_back(observer_usage);
  return result;
}

[[nodiscard]] VisualReprojectionFactorSpec visualFactor(core::FactorId factor_id,
                                                        VisualLandmarkId landmark,
                                                        VisualTrackId track,
                                                        VisualObservationRef anchor,
                                                        VisualObservationRef observer) {
  VisualReprojectionFactorSpec factor;
  factor.id = factor_id;
  factor.landmark = landmark;
  factor.track = track;
  factor.anchor = std::move(anchor);
  factor.observer = std::move(observer);
  factor.lineage = visualFactorLineage(factor.id, factor.anchor, factor.observer);
  return factor;
}

[[nodiscard]] VisualFactorBatch seededVisualBatch(
    core::FusionTime batch_time, VisualLandmarkId landmark, VisualTrackId track,
    core::FactorId factor_id, VisualObservationRef anchor, VisualObservationRef observer,
    const Eigen::Vector3d& point_odom, const core::Pose3d& anchor_pose) {
  VisualFactorBatch batch;
  batch.exact_time = batch_time;
  VisualLandmarkSeed seed;
  seed.landmark = landmark;
  seed.track = track;
  seed.anchor = anchor;
  seed.initial_range_m = (anchor_pose.inverse() * point_odom).norm();
  seed.eta = -std::log(seed.initial_range_m);
  seed.triangulation.track = track;
  seed.triangulation.status = VisualTriangulationStatus::Seeded;
  seed.triangulation.observations = 2U;
  seed.triangulation.inliers = 2U;
  batch.new_landmarks.push_back(seed);
  batch.factors.push_back(
      visualFactor(factor_id, landmark, track, std::move(anchor), std::move(observer)));
  return batch;
}

TEST(GtsamApiConventions, PoseTangentIsRightAndExplicitlyPermuted) {
  const core::Pose3d base(Sophus::SO3d::exp(Eigen::Vector3d{0.2, -0.1, 0.3}),
                          Eigen::Vector3d{3.0, -2.0, 1.0});
  Eigen::Matrix<double, 6, 1> meridian_tangent;
  meridian_tangent << 0.4, -0.5, 0.6, 0.01, -0.02, 0.03;

  const gtsam::Pose3 gtsam_base = gtsam_api::toGtsamPose(base);
  const gtsam::Vector6 gtsam_tangent = gtsam_api::toGtsamPoseTangent(meridian_tangent);
  EXPECT_TRUE(gtsam_tangent.head<3>().isApprox(meridian_tangent.tail<3>()));
  EXPECT_TRUE(gtsam_tangent.tail<3>().isApprox(meridian_tangent.head<3>()));

  const gtsam::Pose3 perturbed = gtsam::traits<gtsam::Pose3>::Retract(gtsam_base, gtsam_tangent);
  const gtsam::Vector6 recovered_gtsam = gtsam::traits<gtsam::Pose3>::Local(gtsam_base, perturbed);
  const Eigen::Matrix<double, 6, 1> recovered_meridian =
      gtsam_api::fromGtsamPoseTangent(recovered_gtsam);
  EXPECT_TRUE(recovered_meridian.isApprox(meridian_tangent, 1.0e-10));
}

TEST(GtsamApiConventions, NavigationCovariancePermutationRoundTrips) {
  NavigationCovariance meridian;
  Eigen::Matrix<double, 15, 15> generator;
  for (Eigen::Index row = 0; row < 15; ++row) {
    for (Eigen::Index column = 0; column < 15; ++column) {
      generator(row, column) = 0.01 * static_cast<double>(1 + row * 15 + column);
    }
  }
  meridian.matrix =
      generator * generator.transpose() + 0.1 * Eigen::Matrix<double, 15, 15>::Identity();
  const auto gtsam = gtsam_api::toGtsamNavigationCovariance(meridian);

  EXPECT_DOUBLE_EQ(gtsam(3, 3), meridian.matrix(6, 6));
  EXPECT_DOUBLE_EQ(gtsam(6, 6), meridian.matrix(3, 3));
  EXPECT_DOUBLE_EQ(gtsam(9, 9), meridian.matrix(12, 12));
  EXPECT_DOUBLE_EQ(gtsam(12, 12), meridian.matrix(9, 9));
  EXPECT_TRUE(
      gtsam_api::fromGtsamNavigationCovariance(gtsam).matrix.isApprox(meridian.matrix, 0.0));
}

TEST(GtsamApiCovariance, BayesTreeMatchesFullQrForDenseCorrelatedNavigationPosterior) {
  const gtsam::Key pose_key = gtsam::Symbol('x', 0U);
  const gtsam::Key velocity_key = gtsam::Symbol('v', 0U);
  const gtsam::Key bias_key = gtsam::Symbol('b', 0U);

  Eigen::Matrix<double, 15, 15> generator;
  for (Eigen::Index row = 0; row < generator.rows(); ++row) {
    for (Eigen::Index column = 0; column < generator.cols(); ++column) {
      generator(row, column) =
          row == column ? 0.7 + 0.03 * static_cast<double>(row)
                        : 0.015 * std::sin(static_cast<double>(1 + row * 17 + column * 11));
    }
  }
  const Eigen::Matrix<double, 15, 15> expected_covariance =
      generator * generator.transpose() + 0.05 * Eigen::Matrix<double, 15, 15>::Identity();
  const Eigen::Matrix<double, 15, 15> information = expected_covariance.inverse();

  gtsam::Values values;
  values.insert(pose_key, gtsam::Pose3{});
  const gtsam::Vector3 zero_velocity = gtsam::Vector3::Zero();
  values.insert(velocity_key, zero_velocity);
  values.insert(bias_key, gtsam::imuBias::ConstantBias{});
  const gtsam::HessianFactor dense_factor(
      pose_key, velocity_key, bias_key, information.block<6, 6>(0, 0),
      information.block<6, 3>(0, 6), information.block<6, 6>(0, 9), gtsam::Vector6::Zero(),
      information.block<3, 3>(6, 6), information.block<3, 6>(6, 9), gtsam::Vector3::Zero(),
      information.block<6, 6>(9, 9), gtsam::Vector6::Zero(), 0.0);
  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::LinearContainerFactor>(dense_factor, values);

  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam::ISAM2 solver(parameters);
  solver.update(factors, values);

  const NavigationCovariance incremental =
      gtsam_api::jointNavigationCovarianceFromBayesTree(solver, pose_key, velocity_key, bias_key);
  const NavigationCovariance full = fullGraphNavigationCovariance(
      solver, solver.getLinearizationPoint(), pose_key, velocity_key, bias_key);
  const double scale = std::max(1.0, full.matrix.cwiseAbs().maxCoeff());
  EXPECT_LE((incremental.matrix - full.matrix).cwiseAbs().maxCoeff(), 1.0e-10 * scale);
  EXPECT_TRUE(positiveDefinite(incremental));

  const Eigen::Matrix<double, 15, 15> incremental_gtsam =
      gtsam_api::toGtsamNavigationCovariance(incremental);
  EXPECT_GT((incremental_gtsam.block<6, 3>(0, 6).norm()), 1.0e-4);
  EXPECT_GT((incremental_gtsam.block<6, 6>(0, 9).norm()), 1.0e-4);
  EXPECT_GT((incremental_gtsam.block<3, 6>(6, 9).norm()), 1.0e-4);
}

TEST(GtsamApiCovariance, BayesTreeMatchesFullQrForCombinedImuAndPoseGraphPosterior) {
  const gtsam::Key pose0 = gtsam::Symbol('x', 0U);
  const gtsam::Key velocity0 = gtsam::Symbol('v', 0U);
  const gtsam::Key bias0 = gtsam::Symbol('b', 0U);
  const gtsam::Key pose1 = gtsam::Symbol('x', 1U);
  const gtsam::Key velocity1 = gtsam::Symbol('v', 1U);
  const gtsam::Key bias1 = gtsam::Symbol('b', 1U);

  auto preintegration_parameters =
      boost::make_shared<gtsam::PreintegrationCombinedParams>(Eigen::Vector3d{0.0, 0.0, -kGravity});
  preintegration_parameters->accelerometerCovariance = 4.0e-4 * Eigen::Matrix3d::Identity();
  preintegration_parameters->gyroscopeCovariance = 2.25e-6 * Eigen::Matrix3d::Identity();
  preintegration_parameters->integrationCovariance = 1.0e-8 * Eigen::Matrix3d::Identity();
  preintegration_parameters->biasAccCovariance = 4.0e-8 * Eigen::Matrix3d::Identity();
  preintegration_parameters->biasOmegaCovariance = 4.0e-10 * Eigen::Matrix3d::Identity();
  preintegration_parameters->biasAccOmegaInt = 1.0e-6 * Eigen::Matrix<double, 6, 6>::Identity();
  const gtsam::imuBias::ConstantBias initial_bias(Eigen::Vector3d{0.01, -0.02, 0.015},
                                                  Eigen::Vector3d{0.001, -0.002, 0.0005});
  gtsam::PreintegratedCombinedMeasurements preintegrated(preintegration_parameters, initial_bias);
  for (std::size_t index = 0U; index < 40U; ++index) {
    preintegrated.integrateMeasurement(
        Eigen::Vector3d{0.15, -0.08, kGravity + 0.04} + initial_bias.accelerometer(),
        Eigen::Vector3d{0.03, -0.01, 0.12} + initial_bias.gyroscope(), 0.01);
  }

  const gtsam::Pose3 initial_pose(gtsam::Rot3::RzRyRx(0.08, -0.04, 0.15),
                                  gtsam::Point3{1.0, -0.5, 0.2});
  const gtsam::Vector3 initial_velocity{0.4, -0.1, 0.05};
  const gtsam::NavState predicted =
      preintegrated.predict(gtsam::NavState(initial_pose, initial_velocity), initial_bias);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(
      pose0, initial_pose,
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector6() << 0.03, 0.04, 0.05, 0.08, 0.09, 0.10).finished()));
  factors.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(
      velocity0, initial_velocity,
      gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector3() << 0.08, 0.10, 0.12).finished()));
  factors.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(
      bias0, initial_bias,
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector6() << 0.03, 0.04, 0.05, 0.003, 0.004, 0.005).finished()));
  factors.emplace_shared<gtsam::CombinedImuFactor>(pose0, velocity0, pose1, velocity1, bias0, bias1,
                                                   preintegrated);
  factors.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
      pose0, pose1, initial_pose.between(predicted.pose()),
      gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector6() << 0.02, 0.025, 0.03, 0.04, 0.05, 0.06).finished()));

  gtsam::Vector6 pose_perturbation;
  pose_perturbation << 0.01, -0.005, 0.008, 0.03, -0.02, 0.01;
  gtsam::Values values;
  values.insert(pose0, initial_pose);
  values.insert(velocity0, initial_velocity);
  values.insert(bias0, initial_bias);
  values.insert(pose1, predicted.pose().retract(pose_perturbation));
  const gtsam::Vector3 perturbed_velocity =
      predicted.velocity() + Eigen::Vector3d{0.02, -0.01, 0.015};
  values.insert(velocity1, perturbed_velocity);
  values.insert(bias1, initial_bias);

  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.relinearizeSkip = 1;
  parameters.relinearizeThreshold = 0.0;
  parameters.enableRelinearization = true;
  parameters.evaluateNonlinearError = false;
  gtsam::ISAM2 solver(parameters);
  solver.update(factors, values);
  solver.update();

  const NavigationCovariance incremental =
      gtsam_api::jointNavigationCovarianceFromBayesTree(solver, pose1, velocity1, bias1);
  const NavigationCovariance full = fullGraphNavigationCovariance(
      solver, solver.getLinearizationPoint(), pose1, velocity1, bias1);
  const double scale = std::max(1.0, full.matrix.cwiseAbs().maxCoeff());
  EXPECT_LE((incremental.matrix - full.matrix).cwiseAbs().maxCoeff(), 2.0e-8 * scale);
  EXPECT_TRUE(positiveDefinite(incremental));

  const Eigen::Matrix<double, 15, 15> incremental_gtsam =
      gtsam_api::toGtsamNavigationCovariance(incremental);
  EXPECT_GT((incremental_gtsam.block<6, 3>(0, 6).norm()), 1.0e-8);
  EXPECT_GT((incremental_gtsam.block<6, 6>(0, 9).norm()), 1.0e-8);
  EXPECT_GT((incremental_gtsam.block<3, 6>(6, 9).norm()), 1.0e-8);
}

TEST(GtsamDependency, FixedLagLeafMarginalsReuseVacantNonlinearFactorSlots) {
  gtsam::ISAM2Params parameters;
  parameters.findUnusedFactorSlots = true;
  gtsam::ISAM2 solver(parameters);

  const auto model = gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e-2);
  const gtsam::Key initial_key = gtsam::Symbol('s', 0U);
  gtsam::NonlinearFactorGraph initial_factors;
  initial_factors.emplace_shared<gtsam::PriorFactor<double>>(initial_key, 0.0, model);
  gtsam::Values initial_values;
  initial_values.insert(initial_key, 0.0);
  ASSERT_NO_THROW(solver.update(initial_factors, initial_values));

  // A one-state fixed-lag chain always contains one active marginal factor.
  // While augmenting it with the next between factor, two nonlinear-factor
  // slots are sufficient forever. GTSAM 4.2.1 made marginalizeLeaves honor
  // findUnusedFactorSlots for the replacement LinearContainerFactor; older
  // releases appended that factor after deleting its predecessors and grew
  // this storage by one slot on every marginalization.
  constexpr std::size_t kTwoStateFactorSlotBound = 2U;
  constexpr std::uint64_t kMarginalizationTransactions = 32U;
  for (std::uint64_t step = 1U; step <= kMarginalizationTransactions; ++step) {
    const gtsam::Key previous_key = gtsam::Symbol('s', step - 1U);
    const gtsam::Key current_key = gtsam::Symbol('s', step);

    gtsam::NonlinearFactorGraph new_factors;
    new_factors.emplace_shared<gtsam::BetweenFactor<double>>(previous_key, current_key, 1.0, model);
    gtsam::Values new_values;
    new_values.insert(current_key, static_cast<double>(step));

    gtsam::FastMap<gtsam::Key, int> ordering_groups;
    ordering_groups[previous_key] = 0;
    ordering_groups[current_key] = 1;
    gtsam::ISAM2UpdateParams update_parameters;
    update_parameters.constrainedKeys = ordering_groups;
    ASSERT_NO_THROW(solver.update(new_factors, new_values, update_parameters));
    ASSERT_LE(solver.getFactorsUnsafe().size(), kTwoStateFactorSlotBound);

    gtsam::FastList<gtsam::Key> leaf_keys{previous_key};
    gtsam::FactorIndices marginal_factor_indices;
    gtsam::FactorIndices deleted_factor_indices;
    ASSERT_NO_THROW(
        solver.marginalizeLeaves(leaf_keys, marginal_factor_indices, deleted_factor_indices));

    ASSERT_EQ(marginal_factor_indices.size(), 1U);
    EXPECT_LT(marginal_factor_indices.front(), kTwoStateFactorSlotBound);
    EXPECT_LE(solver.getFactorsUnsafe().size(), kTwoStateFactorSlotBound);
    EXPECT_FALSE(solver.valueExists(previous_key));
    EXPECT_TRUE(solver.valueExists(current_key));

    std::size_t active_factors = 0U;
    for (const auto& factor : solver.getFactorsUnsafe()) {
      active_factors += factor ? 1U : 0U;
    }
    EXPECT_EQ(active_factors, 1U);
  }
}

TEST(GtsamApiConventions, CombinedResidualOrderIsRotationPositionVelocityAccGyro) {
  auto params = boost::make_shared<gtsam::PreintegrationCombinedParams>(gtsam::Vector3::Zero());
  params->accelerometerCovariance = 1.0e-4 * gtsam::I_3x3;
  params->gyroscopeCovariance = 1.0e-4 * gtsam::I_3x3;
  params->integrationCovariance = 1.0e-8 * gtsam::I_3x3;
  params->biasAccCovariance = 1.0e-8 * gtsam::I_3x3;
  params->biasOmegaCovariance = 1.0e-8 * gtsam::I_3x3;
  params->biasAccOmegaInt = 1.0e-8 * gtsam::I_6x6;
  gtsam::PreintegratedCombinedMeasurements preintegrated(params);

  const gtsam::CombinedImuFactor factor(1, 2, 3, 4, 5, 6, preintegrated);
  const gtsam::Pose3 next_pose(gtsam::Rot3{}, gtsam::Point3(1.0, 2.0, 3.0));
  const gtsam::Vector3 next_velocity(4.0, 5.0, 6.0);
  const gtsam::imuBias::ConstantBias previous_bias;
  const gtsam::Vector3 accel_bias(0.1, 0.2, 0.3);
  const gtsam::Vector3 gyro_bias(0.4, 0.5, 0.6);
  const gtsam::imuBias::ConstantBias next_bias(accel_bias, gyro_bias);
  const gtsam::Vector error = factor.evaluateError(
      gtsam::Pose3{}, gtsam::Vector3::Zero(), next_pose, next_velocity, previous_bias, next_bias);

  ASSERT_EQ(error.size(), 15);
  // The linked build evaluates predicted-minus-actual.  The sign is immaterial
  // to the squared factor cost, but the block order is binding at our adapter.
  EXPECT_TRUE(error.head<3>().isZero(1.0e-12));
  EXPECT_TRUE(error.segment<3>(3).isApprox(-gtsam::Vector3(1.0, 2.0, 3.0), 1.0e-12));
  EXPECT_TRUE(error.segment<3>(6).isApprox(-next_velocity, 1.0e-12));
  EXPECT_TRUE(error.segment<3>(9).isApprox(-accel_bias, 1.0e-12));
  EXPECT_TRUE(error.segment<3>(12).isApprox(-gyro_bias, 1.0e-12));

  const gtsam::Vector rotation_error =
      factor.evaluateError(gtsam::Pose3{}, gtsam::Vector3::Zero(),
                           gtsam::Pose3(gtsam::Rot3::Rz(0.2), gtsam::Point3::Zero()),
                           gtsam::Vector3::Zero(), previous_bias, previous_bias);
  EXPECT_NEAR(rotation_error(2), -0.2, 1.0e-12);
  EXPECT_TRUE(rotation_error.segment<12>(3).isZero(1.0e-12));
}

TEST(GtsamDependency, AffectedLeafTraversalExcludesDisconnectedBayesTreeBranch) {
  const gtsam::Key a0 = gtsam::Symbol('a', 0U);
  const gtsam::Key a1 = gtsam::Symbol('a', 1U);
  const gtsam::Key a2 = gtsam::Symbol('a', 2U);
  const gtsam::Key b0 = gtsam::Symbol('c', 0U);
  const gtsam::Key b1 = gtsam::Symbol('c', 1U);
  const gtsam::Key b2 = gtsam::Symbol('c', 2U);
  const auto prior_model = gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e-3);
  const auto between_model = gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e-2);

  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 solver(parameters);

  gtsam::NonlinearFactorGraph initial_factors;
  initial_factors.emplace_shared<gtsam::PriorFactor<double>>(a0, 0.0, prior_model);
  initial_factors.emplace_shared<gtsam::PriorFactor<double>>(b0, 0.0, prior_model);
  gtsam::Values initial_values;
  initial_values.insert(a0, 0.0);
  initial_values.insert(b0, 0.0);
  ASSERT_NO_THROW(solver.update(initial_factors, initial_values));

  gtsam::NonlinearFactorGraph middle_factors;
  middle_factors.emplace_shared<gtsam::BetweenFactor<double>>(a0, a1, 1.0, between_model);
  middle_factors.emplace_shared<gtsam::BetweenFactor<double>>(b0, b1, 1.0, between_model);
  gtsam::Values middle_values;
  middle_values.insert(a1, 1.0);
  middle_values.insert(b1, 1.0);
  ASSERT_NO_THROW(solver.update(middle_factors, middle_values));

  gtsam::NonlinearFactorGraph final_factors;
  final_factors.emplace_shared<gtsam::BetweenFactor<double>>(a1, a2, 1.0, between_model);
  final_factors.emplace_shared<gtsam::BetweenFactor<double>>(b1, b2, 1.0, between_model);
  gtsam::Values final_values;
  final_values.insert(a2, 2.0);
  final_values.insert(b2, 2.0);
  ASSERT_NO_THROW(solver.update(final_factors, final_values));
  EXPECT_EQ(solver.candidateCacheSetStamp().stateful_factors, 0U);

  // Discover a key that currently has descendants; clique orientation is an
  // implementation detail and must not be hard-coded by this regression.
  std::optional<gtsam::Key> requested_nonleaf;
  std::set<gtsam::Key> affected_set;
  for (const gtsam::Key candidate_key : std::array{a0, a1, a2}) {
    gtsam::FastList<gtsam::Key> requested_leaves;
    requested_leaves.push_back(candidate_key);
    const gtsam::FastList<gtsam::Key> affected =
        solver.affectedKeysForLeafMarginalization(requested_leaves);
    if (!affected.empty()) {
      requested_nonleaf = candidate_key;
      affected_set.insert(affected.begin(), affected.end());
      break;
    }
  }
  ASSERT_TRUE(requested_nonleaf);
  EXPECT_FALSE(affected_set.contains(*requested_nonleaf));
  EXPECT_TRUE(std::all_of(affected_set.begin(), affected_set.end(),
                          [&](gtsam::Key key) { return key == a0 || key == a1 || key == a2; }));
  EXPECT_FALSE(affected_set.contains(b0));
  EXPECT_FALSE(affected_set.contains(b1));
  EXPECT_FALSE(affected_set.contains(b2));
}

TEST(GtsamDependency, GlobalizedEstimatePreservesCandidateGraphAndIsExactlyPublished) {
  const gtsam::Key x = gtsam::Symbol('d', 0U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 solver(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<double>>(x, 0.0,
                                                     gtsam::noiseModel::Isotropic::Sigma(1U, 1.0));
  gtsam::Values values;
  values.insert(x, 2.0);
  ASSERT_NO_THROW(solver.update(factors, values));
  const std::size_t factor_slots = solver.getFactorsUnsafe().size();
  const gtsam::Values theta_before = solver.getLinearizationPoint();

  gtsam::Values damped = solver.calculateEstimate();
  damped.update(x, 1.25);
  ASSERT_NO_THROW(solver.setGlobalizedEstimate(damped));
  EXPECT_DOUBLE_EQ(solver.calculateEstimate().at<double>(x), 1.25);
  EXPECT_EQ(solver.getFactorsUnsafe().size(), factor_slots);
  EXPECT_TRUE(solver.getLinearizationPoint().equals(theta_before, 1.0e-15));
  EXPECT_EQ(solver.candidateCacheSetStamp().stateful_factors, 0U);

  gtsam::ISAM2UpdateParams update_parameters;
  update_parameters.force_relinearize = true;
  update_parameters.forceFullSolve = true;
  ASSERT_NO_THROW(solver.update(gtsam::NonlinearFactorGraph{}, gtsam::Values{}, update_parameters));
  EXPECT_NEAR(solver.calculateEstimate().at<double>(x), 0.0, 1.0e-12);
}

TEST(GtsamDependency, PhysicallyConvergedRoundoffIncreaseRetainsPreviousWithoutTrials) {
  const gtsam::Key x = gtsam::Symbol('d', 1U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 candidate(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.push_back(boost::make_shared<StationaryScalarFactor>(x));
  gtsam::Values seed;
  seed.insert(x, 0.0);
  ASSERT_NO_THROW(candidate.update(factors, seed));
  const double seed_error = candidate.getFactorsUnsafe().error(seed);

  gtsam::Values roundoff_step = seed;
  roundoff_step.update(x, 1.0e-6);
  const double roundoff_error = candidate.getFactorsUnsafe().error(roundoff_step);
  constexpr double kObjectiveStabilizationTolerance = 1.0e-9;
  ASSERT_GT(roundoff_error, seed_error);
  ASSERT_LE(roundoff_error - seed_error, kObjectiveStabilizationTolerance);

  const gtsam_api::CandidateGlobalizationResult retained =
      candidate.globalizeFullStep(seed, seed_error, std::move(roundoff_step), roundoff_error, 8U,
                                  0.5, true, kObjectiveStabilizationTolerance);

  EXPECT_TRUE(retained.rejected_full_step);
  EXPECT_TRUE(retained.zero_step);
  EXPECT_EQ(retained.backtracking_trials, 0U);
  EXPECT_FALSE(retained.cauchy_direction_attempted);
  EXPECT_FALSE(retained.cauchy_step_accepted);
  EXPECT_EQ(retained.cauchy_backtracking_trials, 0U);
  EXPECT_DOUBLE_EQ(retained.step_scale, 0.0);
  EXPECT_DOUBLE_EQ(retained.error, seed_error);
  EXPECT_DOUBLE_EQ(candidate.calculateEstimate().at<double>(x), seed.at<double>(x));
}

TEST(GtsamDependency, NonstationaryNearEqualCostDirectionStillBacktracks) {
  const gtsam::Key x = gtsam::Symbol('d', 2U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 candidate(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<double>>(x, 0.0,
                                                     gtsam::noiseModel::Isotropic::Sigma(1U, 1.0));
  gtsam::Values seed;
  seed.insert(x, -1.0);
  ASSERT_NO_THROW(candidate.update(factors, seed));
  const double seed_error = candidate.getFactorsUnsafe().error(seed);

  gtsam::Values near_equal_full_step = seed;
  near_equal_full_step.update(x, 1.0 + 1.0e-10);
  const double full_step_error = candidate.getFactorsUnsafe().error(near_equal_full_step);
  constexpr double kObjectiveStabilizationTolerance = 1.0e-9;
  ASSERT_GT(full_step_error, seed_error);
  ASSERT_LE(full_step_error - seed_error, kObjectiveStabilizationTolerance);

  const gtsam_api::CandidateGlobalizationResult damped = candidate.globalizeFullStep(
      seed, seed_error, std::move(near_equal_full_step), full_step_error, 8U, 0.5, false,
      kObjectiveStabilizationTolerance);

  EXPECT_TRUE(damped.rejected_full_step);
  EXPECT_FALSE(damped.zero_step);
  EXPECT_EQ(damped.backtracking_trials, 1U);
  EXPECT_FALSE(damped.cauchy_direction_attempted);
  EXPECT_DOUBLE_EQ(damped.step_scale, 0.5);
  EXPECT_LT(damped.error, seed_error);
  EXPECT_NEAR(candidate.calculateEstimate().at<double>(x), 5.0e-11, 1.0e-15);
}

TEST(GtsamDependency, ActualObjectiveBacktrackingDampsAnOvershootingGaussNewtonStep) {
  const gtsam::Key x = gtsam::Symbol('d', 3U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 committed(parameters);

  gtsam::NonlinearFactorGraph prior;
  prior.emplace_shared<gtsam::PriorFactor<double>>(x, -2.0,
                                                   gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e6));
  gtsam::Values initial;
  initial.insert(x, -2.0);
  ASSERT_NO_THROW(committed.update(prior, initial));
  const gtsam::Values seed = committed.calculateEstimate();

  auto candidate = std::make_unique<gtsam_api::CandidateIsolatedISAM2>(committed);
  gtsam::NonlinearFactorGraph nonlinear;
  nonlinear.push_back(boost::make_shared<ExponentialScalarFactor>(x));
  gtsam::ISAM2UpdateParams update_parameters;
  update_parameters.force_relinearize = true;
  update_parameters.forceFullSolve = true;
  ASSERT_NO_THROW(candidate->update(nonlinear, gtsam::Values{}, update_parameters));
  const double seed_error = candidate->getFactorsUnsafe().error(seed);
  gtsam::Values full_step = candidate->calculateEstimate();
  const double full_step_error = candidate->getFactorsUnsafe().error(full_step);
  ASSERT_GT(full_step_error, seed_error);

  const gtsam_api::CandidateGlobalizationResult accepted = candidate->globalizeFullStep(
      seed, seed_error, std::move(full_step), full_step_error, 8U, 0.5);
  EXPECT_TRUE(accepted.rejected_full_step);
  EXPECT_FALSE(accepted.zero_step);
  EXPECT_FALSE(accepted.cauchy_direction_attempted);
  EXPECT_FALSE(accepted.cauchy_step_accepted);
  EXPECT_EQ(accepted.cauchy_backtracking_trials, 0U);
  EXPECT_EQ(accepted.backtracking_trials, 2U);
  EXPECT_DOUBLE_EQ(accepted.step_scale, 0.25);
  EXPECT_LT(accepted.error, seed_error);
  EXPECT_DOUBLE_EQ(candidate->getFactorsUnsafe().error(candidate->calculateEstimate()),
                   accepted.error);
}

TEST(GtsamDependency, ActualObjectiveBacktrackingRecoversFromAnInfiniteFullStepCost) {
  const gtsam::Key x = gtsam::Symbol('d', 4U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 candidate(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<double>>(
      x, -2.0, gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e6));
  factors.push_back(boost::make_shared<ExponentialScalarFactor>(x));
  gtsam::Values seed;
  seed.insert(x, -2.0);
  ASSERT_NO_THROW(candidate.update(factors, seed));
  const double seed_error = candidate.getFactorsUnsafe().error(seed);

  gtsam::Values finite_full_step = seed;
  finite_full_step.update(x, 600.0);
  const double infinite_error = candidate.getFactorsUnsafe().error(finite_full_step);
  ASSERT_TRUE(std::isinf(infinite_error));
  const gtsam_api::CandidateGlobalizationResult recovered = candidate.globalizeFullStep(
      seed, seed_error, std::move(finite_full_step), infinite_error, 8U, 0.5);

  EXPECT_TRUE(recovered.rejected_full_step);
  EXPECT_EQ(recovered.backtracking_trials, 8U);
  EXPECT_FALSE(recovered.cauchy_direction_attempted);
  EXPECT_FALSE(recovered.cauchy_step_accepted);
  EXPECT_EQ(recovered.cauchy_backtracking_trials, 0U);
  EXPECT_FALSE(recovered.zero_step);
  EXPECT_DOUBLE_EQ(recovered.step_scale, 1.0 / 256.0);
  EXPECT_TRUE(std::isfinite(recovered.error));
  EXPECT_LT(recovered.error, seed_error);
  EXPECT_DOUBLE_EQ(candidate.getFactorsUnsafe().error(candidate.calculateEstimate()),
                   recovered.error);
}

TEST(GtsamDependency, ActualObjectiveCauchyDirectionRescuesANonDescentSuppliedFullStep) {
  const gtsam::Key x = gtsam::Symbol('d', 5U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 candidate(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<double>>(
      x, -2.0, gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e6));
  factors.push_back(boost::make_shared<ExponentialScalarFactor>(x));
  gtsam::Values initial;
  initial.insert(x, -2.0);
  gtsam::ISAM2UpdateParams update_parameters;
  update_parameters.force_relinearize = true;
  update_parameters.forceFullSolve = true;
  ASSERT_NO_THROW(candidate.update(factors, initial, update_parameters));
  gtsam::Values seed = initial;
  const double seed_error = candidate.getFactorsUnsafe().error(seed);

  gtsam::Values non_descent = seed;
  non_descent.update(x, -3.0);
  const double non_descent_error = candidate.getFactorsUnsafe().error(non_descent);
  ASSERT_GT(non_descent_error, seed_error);
  const gtsam_api::CandidateGlobalizationResult rescued = candidate.globalizeFullStep(
      seed, seed_error, std::move(non_descent), non_descent_error, 8U, 0.5);

  EXPECT_TRUE(rescued.rejected_full_step);
  EXPECT_EQ(rescued.backtracking_trials, 8U);
  EXPECT_TRUE(rescued.cauchy_direction_attempted);
  EXPECT_TRUE(rescued.cauchy_step_accepted);
  EXPECT_EQ(rescued.cauchy_backtracking_trials, 3U);
  EXPECT_FALSE(rescued.zero_step);
  EXPECT_DOUBLE_EQ(rescued.step_scale, 0.25);
  EXPECT_LT(rescued.error, seed_error);
  EXPECT_DOUBLE_EQ(candidate.getFactorsUnsafe().error(candidate.calculateEstimate()),
                   rescued.error);
}

TEST(GtsamDependency, ActualObjectiveCauchyStationaryPointRetainsTheAcceptedEstimate) {
  const gtsam::Key x = gtsam::Symbol('d', 6U);
  gtsam::ISAM2Params parameters;
  parameters.optimizationParams = gtsam::ISAM2GaussNewtonParams{0.0};
  parameters.factorization = gtsam::ISAM2Params::QR;
  parameters.evaluateNonlinearError = false;
  gtsam_api::CandidateIsolatedISAM2 candidate(parameters);

  gtsam::NonlinearFactorGraph factors;
  factors.emplace_shared<gtsam::PriorFactor<double>>(
      x, 0.0, gtsam::noiseModel::Isotropic::Sigma(1U, 1.0e6));
  factors.push_back(boost::make_shared<StationaryScalarFactor>(x));
  gtsam::Values seed;
  seed.insert(x, 0.0);
  gtsam::ISAM2UpdateParams update_parameters;
  update_parameters.force_relinearize = true;
  update_parameters.forceFullSolve = true;
  ASSERT_NO_THROW(candidate.update(factors, seed, update_parameters));
  const double seed_error = candidate.getFactorsUnsafe().error(seed);

  gtsam::Values increasing = seed;
  increasing.update(x, 1.0);
  const double increasing_error = candidate.getFactorsUnsafe().error(increasing);
  ASSERT_GT(increasing_error, seed_error);
  const gtsam_api::CandidateGlobalizationResult zero = candidate.globalizeFullStep(
      seed, seed_error, std::move(increasing), increasing_error, 4U, 0.5);

  EXPECT_TRUE(zero.rejected_full_step);
  EXPECT_EQ(zero.backtracking_trials, 4U);
  EXPECT_TRUE(zero.cauchy_direction_attempted);
  EXPECT_FALSE(zero.cauchy_step_accepted);
  EXPECT_EQ(zero.cauchy_backtracking_trials, 0U);
  EXPECT_TRUE(zero.zero_step);
  EXPECT_DOUBLE_EQ(zero.step_scale, 0.0);
  EXPECT_DOUBLE_EQ(zero.error, seed_error);
  EXPECT_DOUBLE_EQ(candidate.calculateEstimate().at<double>(x), seed.at<double>(x));
  EXPECT_EQ(candidate.getFactorsUnsafe().size(), factors.size());
}

TEST(LocalGraph, InitializationPreservesMeridianCovarianceOrder) {
  LocalGraph graph;
  const NavigationCovariance expected = diagonalCovariance();
  auto request = initialization();
  request.covariance = expected;
  const auto commit = graph.initialize(std::move(request));
  ASSERT_TRUE(commit) << commit.error().detail;
  EXPECT_EQ(commit.value().revision, core::LocalGraphRevision{1});
  EXPECT_FALSE(commit.value().parent.valid());
  EXPECT_EQ(commit.value().solve.joint_initial_priors, 1U);
  EXPECT_EQ(commit.value().solve.combined_imu_factors, 0U);
  EXPECT_TRUE(commit.value().solve.qr_factorization);
  EXPECT_EQ(commit.value().solve.relinearize_skip, 1U);
  EXPECT_EQ(commit.value().solve.marginalization,
            LocalGraphMarginalizationStatus::InactiveWithinLag);
  EXPECT_TRUE(commit.value().covariance.matrix.isApprox(expected.matrix, 1.0e-9));
  ASSERT_EQ(commit.value().navigation_poses.size(), 1U);
  EXPECT_EQ(commit.value().navigation_poses.front().state, commit.value().state);
  EXPECT_EQ(commit.value().navigation_poses.front().exact_time, commit.value().state_time);
  EXPECT_TRUE(commit.value().navigation_poses.front().T_odom_imu.matrix().isApprox(
      commit.value().estimate.T_odom_imu.matrix(), 0.0));
  EXPECT_TRUE(commit.value().finalized_states.empty());
  EXPECT_EQ(commit.value().finality.status, LocalGraphFinalityStatus::InactiveWithinWindow);
  EXPECT_EQ(commit.value().finality.records_reserved, 0U);
  EXPECT_EQ(commit.value().finality.pose_covariances_computed, 0U);
}

TEST(LocalGraph, StationaryCombinedPreintegrationDoesNotDrift) {
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization()));
  ImuKnotAppend append;
  append.state = core::StateId{11};
  append.exact_time = core::FusionTime{1'000'000'000LL};
  append.interval = imuInterval(0, append.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  append.lineage = lineage(2);
  const auto commit = graph.appendImuKnot(std::move(append));
  ASSERT_TRUE(commit) << commit.error().detail;

  EXPECT_TRUE(commit.value().estimate.T_odom_imu.translation().isZero(1.0e-7));
  EXPECT_TRUE(commit.value().estimate.T_odom_imu.so3().log().isZero(1.0e-7));
  EXPECT_TRUE(commit.value().estimate.velocity_odom.isZero(1.0e-7));
  EXPECT_EQ(commit.value().solve.combined_imu_factors, 1U);
}

TEST(LocalGraph, ConstantTurnUsesEveryExactMidpointSegment) {
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization()));
  ImuKnotAppend append;
  append.state = core::StateId{11};
  append.exact_time = core::FusionTime{2'000'000'000LL};
  append.interval =
      imuInterval(0, append.exact_time.nanoseconds, {0.0, 0.0, kGravity}, {0.0, 0.0, 0.25}, 200U);
  append.lineage = lineage(2);
  const auto commit = graph.appendImuKnot(std::move(append));
  ASSERT_TRUE(commit) << commit.error().detail;

  const Eigen::Vector3d rotation = commit.value().estimate.T_odom_imu.so3().log();
  EXPECT_NEAR(rotation.x(), 0.0, 1.0e-7);
  EXPECT_NEAR(rotation.y(), 0.0, 1.0e-7);
  EXPECT_NEAR(rotation.z(), 0.5, 1.0e-6);
  EXPECT_TRUE(commit.value().estimate.T_odom_imu.translation().isZero(1.0e-6));
}

TEST(LocalGraph, MapsGyroscopeAndAccelerometerBiasWithoutSwapping) {
  core::NavStateEstimate seed;
  seed.accel_bias = {0.11, -0.22, 0.33};
  seed.gyro_bias = {-0.012, 0.023, -0.034};
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  ImuKnotAppend append;
  append.state = core::StateId{11};
  append.exact_time = core::FusionTime{1'000'000'000LL};
  append.interval =
      imuInterval(0, append.exact_time.nanoseconds,
                  Eigen::Vector3d{0.0, 0.0, kGravity} + seed.accel_bias, seed.gyro_bias);
  append.lineage = lineage(2);
  const auto commit = graph.appendImuKnot(std::move(append));
  ASSERT_TRUE(commit) << commit.error().detail;

  EXPECT_TRUE(commit.value().estimate.accel_bias.isApprox(seed.accel_bias, 1.0e-8));
  EXPECT_TRUE(commit.value().estimate.gyro_bias.isApprox(seed.gyro_bias, 1.0e-8));
  EXPECT_TRUE(commit.value().estimate.velocity_odom.isZero(1.0e-6));
}

TEST(LocalGraph, RejectsInexactSupportWithoutMutatingLatestCommit) {
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization()));
  ImuKnotAppend append;
  append.state = core::StateId{11};
  append.exact_time = core::FusionTime{1'000'000'000LL};
  append.interval = imuInterval(1, append.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  append.lineage = lineage(2);
  const auto rejected = graph.appendImuKnot(std::move(append));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::InexactImuBoundary);
  const auto latest = graph.estimate();
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest.value().state, core::StateId{10});
  EXPECT_EQ(latest.value().revision, core::LocalGraphRevision{1});
}

TEST(LocalGraph, MarginalizesOldestStateAtHardCap) {
  LocalGraphConfig config;
  config.maximum_navigation_states = 2U;
  LocalGraph graph(config);
  ASSERT_TRUE(graph.initialize(initialization()));

  ImuKnotAppend first;
  first.state = core::StateId{11};
  first.exact_time = core::FusionTime{1'000'000'000LL};
  first.interval = imuInterval(0, first.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  first.lineage = lineage(2);
  const auto at_cap = graph.appendImuKnot(std::move(first));
  ASSERT_TRUE(at_cap) << at_cap.error().detail;
  EXPECT_EQ(at_cap.value().solve.capacity, LocalGraphCapacityStatus::AtHardCap);
  EXPECT_EQ(at_cap.value().solve.marginalization,
            LocalGraphMarginalizationStatus::InactiveWithinLag);
  EXPECT_TRUE(at_cap.value().finalized_states.empty());
  EXPECT_EQ(at_cap.value().finality.status, LocalGraphFinalityStatus::InactiveWithinWindow);

  ImuKnotAppend second;
  second.state = core::StateId{12};
  second.exact_time = core::FusionTime{2'000'000'000LL};
  second.interval =
      imuInterval(1'000'000'000LL, second.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  second.lineage = lineage(3);
  const auto marginalized = graph.appendImuKnot(std::move(second));
  ASSERT_TRUE(marginalized) << marginalized.error().detail;
  EXPECT_EQ(marginalized.value().revision, core::LocalGraphRevision{3});
  EXPECT_EQ(marginalized.value().solve.navigation_states, 2U);
  EXPECT_EQ(marginalized.value().solve.marginalized_navigation_states, 1U);
  EXPECT_GT(marginalized.value().solve.marginal_factors_added, 0U);
  EXPECT_EQ(marginalized.value().solve.marginalization,
            LocalGraphMarginalizationStatus::AppliedWindowCap);
  EXPECT_TRUE(marginalized.value().estimate.T_odom_imu.translation().isZero(1.0e-6));
  ASSERT_EQ(marginalized.value().finalized_states.size(), 1U);
  const LocalGraphFinalizedState& finalized = marginalized.value().finalized_states.front();
  EXPECT_EQ(finalized.state, core::StateId{10U});
  EXPECT_EQ(finalized.exact_time, core::FusionTime{0});
  EXPECT_EQ(finalized.odom_epoch, core::OdomEpoch{1U});
  EXPECT_EQ(finalized.final_revision, marginalized.value().revision);
  EXPECT_TRUE(finalized.final_estimate.T_odom_imu.translation().isZero(1.0e-6));
  EXPECT_TRUE(positiveSemidefinite(finalized.pose_covariance));
  EXPECT_EQ(marginalized.value().finality.status, LocalGraphFinalityStatus::PublishedWindowCap);
  EXPECT_EQ(marginalized.value().finality.records_reserved, 1U);
  EXPECT_EQ(marginalized.value().finality.pose_covariances_computed, 1U);
  EXPECT_EQ(marginalized.value().finality.oldest_finalized_state, finalized.state);
  EXPECT_EQ(marginalized.value().finality.newest_finalized_state, finalized.state);
  ASSERT_EQ(marginalized.value().navigation_poses.size(), 3U);
  EXPECT_EQ(marginalized.value().navigation_poses.at(0U).state, core::StateId{10U});
  EXPECT_EQ(marginalized.value().navigation_poses.at(1U).state, core::StateId{11U});
  EXPECT_EQ(marginalized.value().navigation_poses.at(2U).state, core::StateId{12U});
  EXPECT_TRUE(marginalized.value().navigation_poses.front().T_odom_imu.matrix().isApprox(
      finalized.final_estimate.T_odom_imu.matrix(), 0.0));
  EXPECT_TRUE(marginalized.value().navigation_poses.back().T_odom_imu.matrix().isApprox(
      marginalized.value().estimate.T_odom_imu.matrix(), 0.0));
}

TEST(LocalGraph, AppliesNominalTimeLagWithoutTrajectoryDiscontinuity) {
  LocalGraphConfig config;
  config.maximum_navigation_states = 64U;
  config.target_fixed_lag = core::Duration{1'500'000'000LL};
  LocalGraph graph(config);
  ASSERT_TRUE(graph.initialize(initialization()));

  for (std::uint64_t step = 1U; step <= 4U; ++step) {
    ImuKnotAppend append;
    append.state = core::StateId{10U + step};
    append.exact_time = core::FusionTime{static_cast<std::int64_t>(step) * 1'000'000'000LL};
    append.interval = imuInterval(append.exact_time.nanoseconds - 1'000'000'000LL,
                                  append.exact_time.nanoseconds, {0.0, 0.0, kGravity});
    append.lineage = lineage(1U + step);
    const auto commit = graph.appendImuKnot(std::move(append));
    ASSERT_TRUE(commit) << commit.error().detail;
    EXPECT_TRUE(commit.value().estimate.T_odom_imu.translation().isZero(1.0e-6));
    EXPECT_TRUE(commit.value().estimate.velocity_odom.isZero(1.0e-6));
    if (step == 1U) {
      EXPECT_TRUE(commit.value().finalized_states.empty());
      EXPECT_EQ(commit.value().finality.status, LocalGraphFinalityStatus::InactiveWithinWindow);
    } else {
      EXPECT_EQ(commit.value().solve.marginalization,
                LocalGraphMarginalizationStatus::AppliedNominalLag);
      EXPECT_EQ(commit.value().solve.navigation_states, 2U);
      ASSERT_EQ(commit.value().finalized_states.size(), 1U);
      const LocalGraphFinalizedState& finalized = commit.value().finalized_states.front();
      EXPECT_EQ(finalized.state, core::StateId{8U + step});
      EXPECT_EQ(finalized.exact_time,
                core::FusionTime{static_cast<std::int64_t>(step - 2U) * 1'000'000'000LL});
      EXPECT_EQ(finalized.final_revision, commit.value().revision);
      EXPECT_EQ(commit.value().finality.status, LocalGraphFinalityStatus::PublishedNominalLag);
      EXPECT_EQ(commit.value().finality.records_reserved, 1U);
      EXPECT_EQ(commit.value().finality.pose_covariances_computed, 1U);
      EXPECT_LT(finalized.exact_time, *commit.value().finality.nominal_cutoff);
      EXPECT_TRUE(positiveSemidefinite(finalized.pose_covariance));
    }
  }
}

TEST(LocalGraph, FinalityPoseCovariancePermutesRotationPositionCrossTerms) {
  LocalGraphConfig config;
  config.target_fixed_lag = core::Duration{1};
  LocalGraph graph(config);
  auto request = initialization();
  request.covariance = correlatedPoseCovariance();
  const auto initialized = graph.initialize(std::move(request));
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const core::PoseCovariance expected =
      poseCovarianceFromNavigation(initialized.value().covariance);
  ASSERT_GT((expected.matrix.topRightCorner<3, 3>().norm()), 1.0e-3);

  ImuKnotAppend append;
  append.state = core::StateId{11U};
  append.exact_time = core::FusionTime{1'000'000'000LL};
  append.interval = imuInterval(0, append.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  append.lineage = lineage(2U);
  const auto commit = graph.appendImuKnot(std::move(append));
  ASSERT_TRUE(commit) << commit.error().detail;
  ASSERT_EQ(commit.value().finalized_states.size(), 1U);
  const core::PoseCovariance& actual = commit.value().finalized_states.front().pose_covariance;
  const double scale = std::max(1.0, expected.matrix.cwiseAbs().maxCoeff());
  EXPECT_LE((actual.matrix - expected.matrix).cwiseAbs().maxCoeff(), 1.0e-9 * scale);
  EXPECT_TRUE((actual.matrix.topRightCorner<3, 3>().isApprox(expected.matrix.topRightCorner<3, 3>(),
                                                             1.0e-10)));
  EXPECT_TRUE((actual.matrix.bottomLeftCorner<3, 3>().isApprox(
      expected.matrix.bottomLeftCorner<3, 3>(), 1.0e-10)));
  EXPECT_TRUE(positiveSemidefinite(actual));
}

TEST(LocalGraph, RejectedCapTransactionPublishesNoFinalityAndConsumesNoRevision) {
  LocalGraphConfig config;
  config.maximum_navigation_states = 2U;
  config.target_fixed_lag = core::Duration{60'000'000'000LL};
  LocalGraph graph(config);
  ASSERT_TRUE(graph.initialize(initialization()));

  ImuKnotAppend first;
  first.state = core::StateId{11U};
  first.exact_time = core::FusionTime{1'000'000'000LL};
  first.interval = imuInterval(0, first.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  first.lineage = lineage(2U);
  const auto before_rejection = graph.appendImuKnot(std::move(first));
  ASSERT_TRUE(before_rejection) << before_rejection.error().detail;
  ASSERT_TRUE(before_rejection.value().finalized_states.empty());

  ImuKnotAppend invalid;
  invalid.state = core::StateId{12U};
  invalid.exact_time = core::FusionTime{2'000'000'000LL};
  invalid.interval =
      imuInterval(1'000'000'001LL, invalid.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  invalid.lineage = lineage(3U);
  const auto rejected = graph.appendImuKnot(std::move(invalid));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::InexactImuBoundary);
  const auto unchanged = graph.estimate();
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(unchanged.value().revision, core::LocalGraphRevision{2U});
  EXPECT_EQ(unchanged.value().state, core::StateId{11U});
  EXPECT_TRUE(unchanged.value().finalized_states.empty());
  EXPECT_EQ(unchanged.value().finality.status, LocalGraphFinalityStatus::InactiveWithinWindow);

  ImuKnotAppend valid;
  valid.state = core::StateId{12U};
  valid.exact_time = core::FusionTime{2'000'000'000LL};
  valid.interval = imuInterval(1'000'000'000LL, valid.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  valid.lineage = lineage(3U);
  const auto committed = graph.appendImuKnot(std::move(valid));
  ASSERT_TRUE(committed) << committed.error().detail;
  EXPECT_EQ(committed.value().revision, core::LocalGraphRevision{3U});
  ASSERT_EQ(committed.value().finalized_states.size(), 1U);
  EXPECT_EQ(committed.value().finalized_states.front().state, core::StateId{10U});
  EXPECT_EQ(committed.value().finalized_states.front().final_revision,
            core::LocalGraphRevision{3U});
}

TEST(LocalGraph, AcceptsCorrectionsBelowExactlyPropagatedImuProcessThresholds) {
  LocalGraphConfig config;
  config.imu.accelerometer_noise_density_mps2_sqrt_hz = 0.12;
  config.imu.gyroscope_noise_density_radps_sqrt_hz = 0.08;
  config.imu.accelerometer_bias_random_walk_mps3_sqrt_hz = 0.04;
  config.imu.gyroscope_bias_random_walk_radps2_sqrt_hz = 0.02;
  config.nonlinear_translation_convergence_m = 1.0e-12;
  config.nonlinear_rotation_convergence_rad = 1.0e-12;
  config.nonlinear_velocity_convergence_mps = 1.0e-12;
  config.nonlinear_accelerometer_bias_convergence_mps2 = 1.0e-12;
  config.nonlinear_gyroscope_bias_convergence_radps = 1.0e-12;
  config.nonlinear_visual_log_inverse_range_convergence = 1.0e-12;
  LocalGraph graph(config);
  ASSERT_TRUE(graph.initialize(initialization()));

  ImuKnotAppend navigation;
  navigation.state = core::StateId{11U};
  navigation.exact_time = core::FusionTime{4'000'000'000LL};
  navigation.interval = imuInterval(0, navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  navigation.lineage = lineage(2U);
  const auto accepted = graph.appendImuKnot(std::move(navigation));
  ASSERT_TRUE(accepted) << accepted.error().detail;

  const LocalSolveReport& solve = accepted.value().solve;
  constexpr double kDt = 4.0;
  constexpr double kSqrtDt = 2.0;
  EXPECT_DOUBLE_EQ(solve.convergence_interval_duration_s, kDt);
  EXPECT_DOUBLE_EQ(solve.convergence_sigma_fraction, 0.25);
  EXPECT_NEAR(solve.effective_translation_convergence_m,
              0.25 * 0.12 * kDt * kSqrtDt / std::sqrt(3.0), 1.0e-15);
  EXPECT_NEAR(solve.effective_rotation_convergence_rad, 0.25 * 0.08 * kSqrtDt, 1.0e-15);
  EXPECT_NEAR(solve.effective_velocity_convergence_mps, 0.25 * 0.12 * kSqrtDt, 1.0e-15);
  EXPECT_NEAR(solve.effective_accelerometer_bias_convergence_mps2, 0.25 * 0.04 * kSqrtDt, 1.0e-15);
  EXPECT_NEAR(solve.effective_gyroscope_bias_convergence_radps, 0.25 * 0.02 * kSqrtDt, 1.0e-15);
  EXPECT_DOUBLE_EQ(solve.effective_visual_log_inverse_range_convergence, 1.0e-12);
  EXPECT_LE(solve.last_iteration_translation_correction_m,
            solve.effective_translation_convergence_m);
  EXPECT_LE(solve.last_iteration_rotation_correction_rad, solve.effective_rotation_convergence_rad);
  EXPECT_LE(solve.last_iteration_velocity_correction_mps, solve.effective_velocity_convergence_mps);
  EXPECT_LE(solve.last_iteration_accelerometer_bias_correction_mps2,
            solve.effective_accelerometer_bias_convergence_mps2);
  EXPECT_LE(solve.last_iteration_gyroscope_bias_correction_radps,
            solve.effective_gyroscope_bias_convergence_radps);
}

TEST(LocalGraph, CommitsVisualSeedAndEtaInOneSensorTransaction) {
  core::NavStateEstimate seed;
  seed.velocity_odom = {0.2, 0.0, 0.0};
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  const Eigen::Vector3d point_odom{0.1, -0.05, 5.0};
  const core::Pose3d anchor_pose;
  const core::Pose3d observer_pose(Sophus::SO3d{}, Eigen::Vector3d{0.2, 0.0, 0.0});
  SensorKnotAppend transaction;
  transaction.navigation.state = core::StateId{11};
  transaction.navigation.exact_time = core::FusionTime{1'000'000'000LL};
  transaction.navigation.interval =
      imuInterval(0, transaction.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  transaction.navigation.lineage = lineage(2U);
  transaction.visual = seededVisualBatch(
      transaction.navigation.exact_time, VisualLandmarkId(20U), VisualTrackId(30U),
      core::FactorId(40U),
      visualObservation(core::MeasurementId(700U), core::StateId(10U), core::FusionTime{0LL},
                        anchor_pose, point_odom),
      visualObservation(core::MeasurementId(701U), core::StateId(11U),
                        transaction.navigation.exact_time, observer_pose, point_odom),
      point_odom, anchor_pose);

  const auto commit = graph.appendSensorKnot(std::move(transaction));
  ASSERT_TRUE(commit) << commit.error().detail;
  EXPECT_FALSE(commit.value().lidar_registration);
  EXPECT_EQ(commit.value().solve.active_visual_landmarks, 1U);
  EXPECT_EQ(commit.value().solve.active_visual_factors, 1U);
  EXPECT_EQ(commit.value().solve.visual_landmarks_added, 1U);
  EXPECT_EQ(commit.value().solve.visual_factors_added, 1U);
  EXPECT_NEAR(commit.value().estimate.T_odom_imu.translation().x(), 0.2, 1.0e-5);
}

TEST(LocalGraph, RejectsInvalidVisualStateWithoutMutatingCandidateGraph) {
  core::NavStateEstimate seed;
  seed.velocity_odom = {0.2, 0.0, 0.0};
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  const Eigen::Vector3d point_odom{0.0, 0.0, 5.0};
  const core::Pose3d anchor_pose;
  const core::Pose3d observer_pose(Sophus::SO3d{}, Eigen::Vector3d{0.2, 0.0, 0.0});
  SensorKnotAppend transaction;
  transaction.navigation.state = core::StateId{11U};
  transaction.navigation.exact_time = core::FusionTime{1'000'000'000LL};
  transaction.navigation.interval =
      imuInterval(0, transaction.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  transaction.navigation.lineage = lineage(2U);
  transaction.visual = seededVisualBatch(
      transaction.navigation.exact_time, VisualLandmarkId(22U), VisualTrackId(32U),
      core::FactorId(42U),
      visualObservation(core::MeasurementId(720U), core::StateId(10U), core::FusionTime{0LL},
                        anchor_pose, point_odom),
      visualObservation(core::MeasurementId(721U), core::StateId(999U),
                        transaction.navigation.exact_time, observer_pose, point_odom),
      point_odom, anchor_pose);

  SensorKnotAppend navigation_retry = transaction;
  navigation_retry.visual.reset();

  const auto rejected = graph.appendSensorKnot(std::move(transaction));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::VisualStateUnavailable);
  const auto latest = graph.estimate();
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest.value().revision, core::LocalGraphRevision(1U));
  EXPECT_EQ(latest.value().state, core::StateId(10U));
  EXPECT_EQ(latest.value().solve.active_visual_landmarks, 0U);

  const auto recovered = graph.appendSensorKnot(std::move(navigation_retry));
  ASSERT_TRUE(recovered) << recovered.error().detail;
  EXPECT_EQ(recovered.value().revision, core::LocalGraphRevision(2U));
  EXPECT_EQ(recovered.value().state, core::StateId(11U));
  EXPECT_EQ(recovered.value().solve.active_visual_landmarks, 0U);
  EXPECT_EQ(recovered.value().solve.active_visual_factors, 0U);
}

TEST(LocalGraph, RejectsInconsistentBoundsAcrossFactorsSharingOneLandmark) {
  core::NavStateEstimate seed;
  seed.velocity_odom = {0.2, 0.0, 0.0};
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  const Eigen::Vector3d point_odom{0.0, 0.0, 5.0};
  const core::Pose3d anchor_pose;
  const core::Pose3d observer_pose(Sophus::SO3d{}, Eigen::Vector3d{0.2, 0.0, 0.0});
  SensorKnotAppend transaction;
  transaction.navigation.state = core::StateId{11U};
  transaction.navigation.exact_time = core::FusionTime{1'000'000'000LL};
  transaction.navigation.interval =
      imuInterval(0, transaction.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  transaction.navigation.lineage = lineage(2U);
  transaction.visual = seededVisualBatch(
      transaction.navigation.exact_time, VisualLandmarkId(25U), VisualTrackId(35U),
      core::FactorId(47U),
      visualObservation(core::MeasurementId(750U), core::StateId(10U), core::FusionTime{0LL},
                        anchor_pose, point_odom),
      visualObservation(core::MeasurementId(751U), core::StateId(11U),
                        transaction.navigation.exact_time, observer_pose, point_odom),
      point_odom, anchor_pose);
  VisualReprojectionFactorSpec inconsistent = transaction.visual->factors.front();
  inconsistent.id = core::FactorId(48U);
  inconsistent.maximum_range_m = 100.0;
  inconsistent.lineage =
      visualFactorLineage(inconsistent.id, inconsistent.anchor, inconsistent.observer);
  transaction.visual->factors.push_back(std::move(inconsistent));

  const auto rejected = graph.appendSensorKnot(std::move(transaction));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::InvalidVisualBatch);
  EXPECT_NE(rejected.error().detail.find("identical range bounds"), std::string::npos);
  ASSERT_TRUE(graph.estimate());
  EXPECT_EQ(graph.estimate().value().revision, core::LocalGraphRevision(1U));
}

TEST(LocalGraph, RetiresMappedVisualFactorAndItsNowUnobservedEta) {
  core::NavStateEstimate seed;
  seed.velocity_odom = {0.2, 0.0, 0.0};
  LocalGraph graph;
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  const Eigen::Vector3d point_odom{0.0, 0.0, 5.0};
  const core::Pose3d anchor_pose;
  const VisualLandmarkId landmark(23U);
  const VisualTrackId track(33U);
  const VisualObservationRef anchor =
      visualObservation(core::MeasurementId(730U), core::StateId(10U), core::FusionTime{0LL},
                        anchor_pose, point_odom);

  SensorKnotAppend seeded;
  seeded.navigation.state = core::StateId(11U);
  seeded.navigation.exact_time = core::FusionTime{1'000'000'000LL};
  seeded.navigation.interval =
      imuInterval(0, seeded.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  seeded.navigation.lineage = lineage(2U);
  seeded.visual = seededVisualBatch(
      seeded.navigation.exact_time, landmark, track, core::FactorId(43U), anchor,
      visualObservation(core::MeasurementId(731U), core::StateId(11U), seeded.navigation.exact_time,
                        core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{0.2, 0.0, 0.0}), point_odom),
      point_odom, anchor_pose);
  ASSERT_TRUE(graph.appendSensorKnot(std::move(seeded)));

  SensorKnotAppend retired;
  retired.navigation.state = core::StateId(12U);
  retired.navigation.exact_time = core::FusionTime{2'000'000'000LL};
  retired.navigation.interval =
      imuInterval(1'000'000'000LL, retired.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  retired.navigation.lineage = lineage(3U);
  retired.visual_factor_retirements.push_back(core::FactorId(43U));
  const auto retirement = graph.appendSensorKnot(std::move(retired));
  ASSERT_TRUE(retirement) << retirement.error().detail;
  EXPECT_EQ(retirement.value().solve.visual_factors_retired, 1U);
  EXPECT_EQ(retirement.value().solve.active_visual_factors, 0U);
  EXPECT_EQ(retirement.value().solve.active_visual_landmarks, 0U);
  EXPECT_EQ(retirement.value().solve.visual_landmarks_marginalized, 1U);
  EXPECT_TRUE(retirement.value().finalized_visual_factors.empty());
  ASSERT_EQ(retirement.value().finalized_visual_landmarks.size(), 1U);
  const LocalGraphFinalizedVisualLandmark& finalized =
      retirement.value().finalized_visual_landmarks.front();
  EXPECT_EQ(finalized.landmark, landmark);
  EXPECT_EQ(finalized.segment.segment, core::LandmarkSegmentId(landmark.value()));
  EXPECT_EQ(finalized.segment.anchor_state, anchor.state);
  EXPECT_EQ(finalized.segment.final_revision, retirement.value().revision);
  EXPECT_NEAR(finalized.segment.final_log_inverse_range, -std::log(5.0), 1.0e-10);
  EXPECT_EQ(finalized.segment.reason, core::LandmarkFinalityReason::Marginalized);
}

TEST(LocalGraph, MarginalizesEtaBeforeItsAnchorAndRejectsFutureReuse) {
  LocalGraphConfig config;
  config.maximum_navigation_states = 2U;
  core::NavStateEstimate seed;
  seed.velocity_odom = {0.2, 0.0, 0.0};
  LocalGraph graph(config);
  ASSERT_TRUE(graph.initialize(initialization(seed)));

  const Eigen::Vector3d point_odom{0.0, 0.0, 5.0};
  const core::Pose3d anchor_pose;
  const VisualLandmarkId landmark(24U);
  const VisualTrackId track(34U);
  const VisualObservationRef anchor =
      visualObservation(core::MeasurementId(740U), core::StateId(10U), core::FusionTime{0LL},
                        anchor_pose, point_odom);
  SensorKnotAppend first;
  first.navigation.state = core::StateId(11U);
  first.navigation.exact_time = core::FusionTime{1'000'000'000LL};
  first.navigation.interval =
      imuInterval(0, first.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  first.navigation.lineage = lineage(2U);
  first.visual = seededVisualBatch(
      first.navigation.exact_time, landmark, track, core::FactorId(45U), anchor,
      visualObservation(core::MeasurementId(741U), core::StateId(11U), first.navigation.exact_time,
                        core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{0.2, 0.0, 0.0}), point_odom),
      point_odom, anchor_pose);
  ASSERT_TRUE(graph.appendSensorKnot(std::move(first)));

  ImuKnotAppend second;
  second.state = core::StateId(12U);
  second.exact_time = core::FusionTime{2'000'000'000LL};
  second.interval =
      imuInterval(1'000'000'000LL, second.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  second.lineage = lineage(3U);
  const auto marginalized = graph.appendImuKnot(std::move(second));
  ASSERT_TRUE(marginalized) << marginalized.error().detail;
  EXPECT_EQ(marginalized.value().solve.visual_landmarks_marginalized, 1U);
  EXPECT_EQ(marginalized.value().solve.active_visual_landmarks, 0U);
  EXPECT_EQ(marginalized.value().solve.active_visual_factors, 0U);
  ASSERT_EQ(marginalized.value().finalized_visual_factors,
            std::vector<core::FactorId>{core::FactorId(45U)});
  ASSERT_EQ(marginalized.value().finalized_visual_landmarks.size(), 1U);
  EXPECT_EQ(marginalized.value().finalized_visual_landmarks.front().landmark, landmark);
  EXPECT_EQ(marginalized.value().finalized_visual_landmarks.front().segment.anchor_state,
            anchor.state);
  EXPECT_EQ(marginalized.value().finalized_visual_landmarks.front().segment.final_revision,
            marginalized.value().revision);

  SensorKnotAppend future;
  future.navigation.state = core::StateId(13U);
  future.navigation.exact_time = core::FusionTime{3'000'000'000LL};
  future.navigation.interval =
      imuInterval(2'000'000'000LL, future.navigation.exact_time.nanoseconds, {0.0, 0.0, kGravity});
  future.navigation.lineage = lineage(4U);
  VisualFactorBatch visual;
  visual.exact_time = future.navigation.exact_time;
  visual.factors.push_back(visualFactor(
      core::FactorId(46U), landmark, track, anchor,
      visualObservation(core::MeasurementId(743U), core::StateId(13U), future.navigation.exact_time,
                        core::Pose3d(Sophus::SO3d{}, Eigen::Vector3d{0.6, 0.0, 0.0}), point_odom)));
  future.visual = std::move(visual);
  const auto rejected = graph.appendSensorKnot(std::move(future));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, LocalGraphErrorCode::VisualLandmarkUnavailable);
  const auto latest = graph.estimate();
  ASSERT_TRUE(latest);
  EXPECT_EQ(latest.value().state, core::StateId(12U));
  EXPECT_EQ(latest.value().revision, core::LocalGraphRevision(3U));
}

}  // namespace
}  // namespace meridian::local
