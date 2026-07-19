#include <gtest/gtest.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/NavState.h>

#include <Eigen/Eigenvalues>
#include <array>
#include <boost/make_shared.hpp>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <set>
#include <utility>

#include "meridian/local/motion_initializer.hpp"

namespace meridian::local {
namespace {

constexpr double kGravity = 9.80665;
constexpr std::int64_t kSegmentNanoseconds = 250'000'000LL;
constexpr std::size_t kSegments = 10U;
constexpr std::size_t kStepsPerSegment = 25U;

enum class SyntheticExcitation {
  RotationAndAcceleration,
  RotationOnly,
  AccelerationOnly,
  ConstantAttitudeAcceleration,
};

struct SyntheticBatch {
  MotionInitializationRequest request;
  core::NavStateEstimate expected_final;
  Eigen::Vector3d expected_accel_bias;
  Eigen::Vector3d expected_gyro_bias;
};

[[nodiscard]] core::ObservationLineage lidarLineage(
    std::size_t segment, core::MeasurementId id,
    const std::vector<core::MeasurementId>& conditioned_imu) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId(100U + segment);
  core::ObservationUsage usage;
  usage.slice.root = id;
  usage.slice.kind = core::SliceKind::Whole;
  usage.slice.calibration = core::CalibrationEpoch(1U);
  usage.role = core::ObservationRole::PrimaryResidual;
  usage.consumer = core::DerivedRecordId(100U + segment);
  usage.factor_group = core::FactorGroupId(100U + segment);
  lineage.usage.push_back(usage);
  const core::CorrelationGroupId correlation{100U + segment};
  for (core::MeasurementId measurement : conditioned_imu) {
    core::ObservationUsage conditioning;
    conditioning.slice.root = measurement;
    conditioning.slice.kind = core::SliceKind::Whole;
    conditioning.slice.calibration = core::CalibrationEpoch{1U};
    conditioning.role = core::ObservationRole::ConditioningOnly;
    conditioning.consumer = core::DerivedRecordId{100U + segment};
    conditioning.correlation_group = correlation;
    lineage.usage.push_back(conditioning);
  }
  lineage.correlations.push_back(core::CorrelationDeclaration{
      correlation, core::CorrelationPolicyRevision{1U},
      core::CorrelationTreatment::CovarianceInflationAndInformationCap, 4.0, 1.0e6});
  return lineage;
}

[[nodiscard]] core::ObservationLineage imuLineage(std::size_t segment,
                                                  const std::vector<core::MeasurementId>& ids) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId(1'000U + segment);
  for (core::MeasurementId id : ids) {
    core::ObservationUsage usage;
    usage.slice.root = id;
    usage.slice.kind = core::SliceKind::Whole;
    usage.slice.calibration = core::CalibrationEpoch(1U);
    usage.role = core::ObservationRole::PrimaryResidual;
    usage.consumer = core::DerivedRecordId(1'000U + segment);
    usage.factor_group = core::FactorGroupId(1'000U + segment);
    lineage.usage.push_back(usage);
  }
  return lineage;
}

[[nodiscard]] core::RankAwareInformation fullPoseInformation() {
  core::RankAwareInformation information;
  information.basis.setIdentity();
  information.eigenvalues << 2'500.0, 2'500.0, 2'500.0, 10'000.0, 10'000.0, 10'000.0;
  information.rank = 6U;
  return information;
}

[[nodiscard]] core::RankAwareInformation partiallyDegeneratePoseInformation(
    std::size_t omitted_direction) {
  const core::RankAwareInformation full = fullPoseInformation();
  core::RankAwareInformation information;
  std::size_t output = 0U;
  for (std::size_t input = 0U; input < 6U; ++input) {
    if (input == omitted_direction) {
      continue;
    }
    information.basis.col(static_cast<Eigen::Index>(output)) =
        full.basis.col(static_cast<Eigen::Index>(input));
    information.eigenvalues(static_cast<Eigen::Index>(output)) =
        full.eigenvalues(static_cast<Eigen::Index>(input));
    ++output;
  }
  information.basis.col(5) = full.basis.col(static_cast<Eigen::Index>(omitted_direction));
  information.rank = 5U;
  return information;
}

[[nodiscard]] core::RankAwareInformation singleDirectionPoseInformation() {
  core::RankAwareInformation information;
  information.basis.setIdentity();
  information.eigenvalues.setZero();
  information.eigenvalues(0) = 2'500.0;
  information.rank = 1U;
  return information;
}

[[nodiscard]] Eigen::Vector3d unbiasedSpecificForce(double time_seconds,
                                                    SyntheticExcitation excitation) {
  if (excitation == SyntheticExcitation::RotationOnly) {
    return {0.0, 0.0, kGravity};
  }
  return Eigen::Vector3d{0.9 * std::sin(2.3 * time_seconds) + 0.35 * std::cos(5.1 * time_seconds),
                         0.7 * std::cos(1.7 * time_seconds) - 0.25 * std::sin(4.2 * time_seconds),
                         kGravity + 0.65 * std::sin(2.9 * time_seconds)};
}

[[nodiscard]] Eigen::Vector3d unbiasedAngularRate(double time_seconds,
                                                  SyntheticExcitation excitation) {
  if (excitation == SyntheticExcitation::RotationOnly) {
    return {0.0, 0.0, 0.35};
  }
  if (excitation == SyntheticExcitation::ConstantAttitudeAcceleration) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d dynamic_rate{0.20 + 0.09 * std::sin(2.1 * time_seconds),
                                     -0.14 + 0.08 * std::cos(1.6 * time_seconds),
                                     0.24 + 0.07 * std::sin(2.8 * time_seconds)};
  // Keep angular motion physically nonzero so the downstream data Hessian
  // can establish all inertial directions, while remaining deliberately
  // below the rotation pre-gate. Only acceleration makes this fixture
  // eligible for the solve.
  return excitation == SyntheticExcitation::AccelerationOnly ? 0.0325 * dynamic_rate : dynamic_rate;
}

[[nodiscard]] SyntheticBatch dynamicBatch(
    SyntheticExcitation excitation = SyntheticExcitation::RotationAndAcceleration) {
  SyntheticBatch batch;
  batch.request.imu_noise.gravity_odom = {0.0, 0.0, -kGravity};
  batch.request.imu_noise.accelerometer_noise_density_mps2_sqrt_hz = 0.02;
  batch.request.imu_noise.gyroscope_noise_density_radps_sqrt_hz = 0.0015;
  batch.request.imu_noise.integration_noise_density = 1.0e-8;
  batch.expected_accel_bias = {0.08, -0.05, 0.04};
  batch.expected_gyro_bias = {0.012, -0.008, 0.006};
  batch.request.imu_noise.accelerometer_bias_prior_mean_mps2 = batch.expected_accel_bias;
  batch.request.imu_noise.gyroscope_bias_prior_mean_radps = batch.expected_gyro_bias;
  const gtsam::imuBias::ConstantBias true_bias(batch.expected_accel_bias, batch.expected_gyro_bias);

  auto parameters =
      boost::make_shared<gtsam::PreintegrationParams>(batch.request.imu_noise.gravity_odom);
  parameters->accelerometerCovariance =
      Eigen::Matrix3d::Identity() *
      std::pow(batch.request.imu_noise.accelerometer_noise_density_mps2_sqrt_hz, 2);
  parameters->gyroscopeCovariance =
      Eigen::Matrix3d::Identity() *
      std::pow(batch.request.imu_noise.gyroscope_noise_density_radps_sqrt_hz, 2);
  parameters->integrationCovariance =
      Eigen::Matrix3d::Identity() * std::pow(batch.request.imu_noise.integration_noise_density, 2);

  gtsam::NavState state(gtsam::Pose3{}, gtsam::Vector3(1.1, -0.45, 0.25));
  for (std::size_t segment_index = 0U; segment_index < kSegments; ++segment_index) {
    const std::int64_t start_ns = static_cast<std::int64_t>(segment_index) * kSegmentNanoseconds;
    const std::int64_t end_ns = start_ns + kSegmentNanoseconds;
    ImuInterval interval;
    interval.support = core::TimeRange{{start_ns}, {end_ns}};
    interval.maximum_raw_gap =
        core::Duration{kSegmentNanoseconds / static_cast<std::int64_t>(kStepsPerSegment)};

    for (std::size_t step = 0U; step <= kStepsPerSegment; ++step) {
      const std::int64_t stamp_ns = start_ns + kSegmentNanoseconds *
                                                   static_cast<std::int64_t>(step) /
                                                   static_cast<std::int64_t>(kStepsPerSegment);
      const double time_seconds = static_cast<double>(stamp_ns) * 1.0e-9;
      const core::MeasurementId id(10'000U + segment_index * 100U + step);
      interval.raw_measurements.push_back(id);
      interval.knots.push_back(InterpolatedImuSample{
          core::FusionTime{stamp_ns},
          unbiasedSpecificForce(time_seconds, excitation) + batch.expected_accel_bias,
          unbiasedAngularRate(time_seconds, excitation) + batch.expected_gyro_bias, id, id});
    }

    gtsam::PreintegratedImuMeasurements preintegrated(parameters, true_bias);
    for (std::size_t step = 1U; step < interval.knots.size(); ++step) {
      const auto& previous = interval.knots[step - 1U];
      const auto& current = interval.knots[step];
      const double dt = static_cast<double>((current.time - previous.time).nanoseconds) * 1.0e-9;
      preintegrated.integrateMeasurement(
          0.5 * (previous.specific_force_mps2 + current.specific_force_mps2),
          0.5 * (previous.angular_velocity_radps + current.angular_velocity_radps), dt);
    }
    const gtsam::NavState next = preintegrated.predict(state, true_bias);
    const core::Pose3d relative(Sophus::SO3d(state.pose().between(next.pose()).rotation().matrix()),
                                state.pose().between(next.pose()).translation());

    MotionInitializationSegment segment;
    segment.lidar.start_time = core::FusionTime{start_ns};
    segment.lidar.end_time = core::FusionTime{end_ns};
    segment.lidar.T_imu_start_imu_end = relative;
    // Every individual scan-registration result is deficient in one
    // direction. The missing direction rotates across the batch, exercising
    // the initializer's aggregate observability decision.
    segment.lidar.information = partiallyDegeneratePoseInformation(segment_index % 6U);
    segment.lidar.lineage = lidarLineage(
        segment_index, core::MeasurementId(50'000U + segment_index), interval.raw_measurements);
    segment.lidar.imu_conditioning_covariance_inflation = 4.0;
    segment.lidar.applied_covariance_inflation = 4.0;
    segment.imu = std::move(interval);
    segment.imu_lineage = imuLineage(segment_index, segment.imu.raw_measurements);
    batch.request.segments.push_back(std::move(segment));
    state = next;
  }

  batch.expected_final.T_odom_imu =
      core::Pose3d(Sophus::SO3d(state.pose().rotation().matrix()), state.pose().translation());
  batch.expected_final.velocity_odom = state.velocity();
  batch.expected_final.accel_bias = batch.expected_accel_bias;
  batch.expected_final.gyro_bias = batch.expected_gyro_bias;
  return batch;
}

[[nodiscard]] MotionInitializationRequest stationaryBatch() {
  MotionInitializationRequest request;
  request.imu_noise.gravity_odom = {0.0, 0.0, -kGravity};
  for (std::size_t segment_index = 0U; segment_index < kSegments; ++segment_index) {
    const std::int64_t start_ns = static_cast<std::int64_t>(segment_index) * kSegmentNanoseconds;
    const std::int64_t end_ns = start_ns + kSegmentNanoseconds;
    MotionInitializationSegment segment;
    segment.lidar.start_time = core::FusionTime{start_ns};
    segment.lidar.end_time = core::FusionTime{end_ns};
    segment.lidar.information = fullPoseInformation();
    segment.imu.support = core::TimeRange{{start_ns}, {end_ns}};
    const core::MeasurementId left(80'000U + 2U * segment_index);
    const core::MeasurementId right(80'001U + 2U * segment_index);
    segment.imu.raw_measurements = {left, right};
    segment.imu.knots = {
        InterpolatedImuSample{
            core::FusionTime{start_ns}, {0.0, 0.0, kGravity}, Eigen::Vector3d::Zero(), left, left},
        InterpolatedImuSample{
            core::FusionTime{end_ns}, {0.0, 0.0, kGravity}, Eigen::Vector3d::Zero(), right, right}};
    segment.imu.maximum_raw_gap = core::Duration{kSegmentNanoseconds};
    segment.lidar.lineage = lidarLineage(
        segment_index, core::MeasurementId(70'000U + segment_index), segment.imu.raw_measurements);
    segment.lidar.imu_conditioning_covariance_inflation = 4.0;
    segment.lidar.applied_covariance_inflation = 4.0;
    segment.imu_lineage = imuLineage(segment_index, segment.imu.raw_measurements);
    request.segments.push_back(std::move(segment));
  }
  return request;
}

TEST(MotionInitializer, DynamicBatchRecoversObservableNavigationState) {
  const SyntheticBatch synthetic = dynamicBatch();
  const MotionInitializer initializer;
  const auto initialized = initializer.initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const MotionInitialization& result = initialized.value();

  EXPECT_LT((synthetic.expected_final.T_odom_imu.inverse() * result.state.T_odom_imu).log().norm(),
            2.0e-3);
  EXPECT_LT((result.state.velocity_odom - synthetic.expected_final.velocity_odom).norm(), 3.0e-3);
  EXPECT_LT((result.state.gyro_bias - synthetic.expected_gyro_bias).norm(), 5.0e-4);
  EXPECT_LT((result.state.accel_bias - synthetic.expected_accel_bias).norm(), 3.0e-3);
  EXPECT_GT(result.state.velocity_odom.norm(), 0.5);
  EXPECT_GT(result.state.gyro_bias.norm(), 0.005);
  EXPECT_GT(result.state.accel_bias.norm(), 0.03);
  EXPECT_EQ(result.exact_time, synthetic.request.segments.back().lidar.end_time);
  EXPECT_EQ(result.lineage_uses.size(), 2U * kSegments);
  EXPECT_EQ(result.diagnostics.data_rank, result.diagnostics.expected_data_rank);
  EXPECT_EQ(result.diagnostics.calibrated_data_rank, result.diagnostics.expected_data_rank);
  EXPECT_EQ(result.diagnostics.full_rank, result.diagnostics.scalar_dimension);
  EXPECT_EQ(result.diagnostics.prior_resolved_accel_tilt_modes, 0U);
  EXPECT_EQ(result.diagnostics.observability_class,
            MotionInitializationObservabilityClass::SensorObservable);
  EXPECT_LT(result.diagnostics.bias_prior_mahalanobis, 5.0);
  EXPECT_TRUE(result.diagnostics.statistically_compatible);
  EXPECT_EQ(result.diagnostics.holdout_lidar_segments, 1U);
  EXPECT_EQ(result.diagnostics.holdout_residual_dimension,
            synthetic.request.segments.back().lidar.information.rank);
  EXPECT_EQ(result.diagnostics.lidar_residual_dimension,
            std::accumulate(synthetic.request.segments.begin(),
                            std::prev(synthetic.request.segments.end()), std::size_t{0U},
                            [](std::size_t dimension, const MotionInitializationSegment& segment) {
                              return dimension + segment.lidar.information.rank;
                            }));
  ASSERT_EQ(result.lineage_uses.size(), 2U * kSegments);
  EXPECT_EQ(result.lineage_uses[result.lineage_uses.size() - 2U].role,
            MotionInitializationLineageRole::ValidationGate);
  EXPECT_EQ(result.lineage_uses.back().role, MotionInitializationLineageRole::FusedResidual);
}

TEST(MotionInitializer, RotationOnlyExcitationPassesPreGateAndAuthoritativeChecks) {
  const SyntheticBatch synthetic = dynamicBatch(SyntheticExcitation::RotationOnly);
  const MotionInitializerConfig config;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const MotionInitializationDiagnostics& diagnostics = initialized.value().diagnostics;
  EXPECT_GE(diagnostics.rotation_excitation_rad, config.minimum_rotation_excitation_rad);
  EXPECT_LT(diagnostics.acceleration_excitation_mps2, config.minimum_acceleration_excitation_mps2);
  EXPECT_EQ(diagnostics.data_rank, diagnostics.expected_data_rank);
  EXPECT_EQ(diagnostics.full_rank, diagnostics.scalar_dimension);
  EXPECT_TRUE(diagnostics.statistically_compatible);
}

TEST(MotionInitializer, AccelerationOnlyExcitationPassesPreGateAndAuthoritativeChecks) {
  const SyntheticBatch synthetic = dynamicBatch(SyntheticExcitation::AccelerationOnly);
  const MotionInitializerConfig config;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const MotionInitializationDiagnostics& diagnostics = initialized.value().diagnostics;
  EXPECT_LT(diagnostics.rotation_excitation_rad, config.minimum_rotation_excitation_rad);
  EXPECT_GE(diagnostics.acceleration_excitation_mps2, config.minimum_acceleration_excitation_mps2);
  EXPECT_EQ(diagnostics.data_rank, diagnostics.expected_data_rank);
  EXPECT_EQ(diagnostics.full_rank, diagnostics.scalar_dimension);
  EXPECT_TRUE(diagnostics.statistically_compatible);
}

TEST(MotionInitializer, ConstantAttitudeAccelerationIsExplicitlyPriorResolved) {
  const SyntheticBatch synthetic = dynamicBatch(SyntheticExcitation::ConstantAttitudeAcceleration);
  const MotionInitializerConfig config;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;

  const MotionInitializationDiagnostics& diagnostics = initialized.value().diagnostics;
  ASSERT_GE(diagnostics.expected_data_rank, 2U);
  EXPECT_LT(diagnostics.rotation_excitation_rad, config.minimum_rotation_excitation_rad);
  EXPECT_GE(diagnostics.acceleration_excitation_mps2, config.minimum_acceleration_excitation_mps2);
  EXPECT_EQ(diagnostics.data_rank, diagnostics.expected_data_rank - 2U);
  EXPECT_EQ(diagnostics.calibrated_data_rank, diagnostics.expected_data_rank);
  EXPECT_EQ(diagnostics.full_rank, diagnostics.scalar_dimension);
  EXPECT_EQ(diagnostics.prior_resolved_accel_tilt_modes, 2U);
  EXPECT_EQ(diagnostics.observability_class,
            MotionInitializationObservabilityClass::PriorResolvedAccelerometerTilt);

  MotionInitializerConfig sensor_only_config = config;
  sensor_only_config.maximum_prior_resolved_accel_tilt_modes = 0U;
  const auto sensor_only = MotionInitializer(sensor_only_config).initialize(synthetic.request);
  ASSERT_FALSE(sensor_only);
  EXPECT_EQ(sensor_only.error().code, MotionInitializationErrorCode::RankDeficientBatch)
      << sensor_only.error().detail;
}

TEST(MotionInitializer, RichAttitudeExcitationRemainsSensorObservable) {
  const SyntheticBatch synthetic = dynamicBatch(SyntheticExcitation::RotationAndAcceleration);
  const auto initialized = MotionInitializer{}.initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;

  const MotionInitializationDiagnostics& diagnostics = initialized.value().diagnostics;
  EXPECT_EQ(diagnostics.data_rank, diagnostics.expected_data_rank);
  EXPECT_EQ(diagnostics.calibrated_data_rank, diagnostics.expected_data_rank);
  EXPECT_EQ(diagnostics.full_rank, diagnostics.scalar_dimension);
  EXPECT_EQ(diagnostics.prior_resolved_accel_tilt_modes, 0U);
  EXPECT_EQ(diagnostics.observability_class,
            MotionInitializationObservabilityClass::SensorObservable);
}

TEST(MotionInitializer, AccelerometerBiasPriorDoesNotMaskUnrelatedPoseDeficiency) {
  SyntheticBatch synthetic = dynamicBatch();
  const MotionInitializerConfig config;
  const auto observable = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_TRUE(observable) << observable.error().detail;
  EXPECT_GE(observable.value().diagnostics.rotation_excitation_rad,
            config.minimum_rotation_excitation_rad);
  EXPECT_GE(observable.value().diagnostics.acceleration_excitation_mps2,
            config.minimum_acceleration_excitation_mps2);
  EXPECT_EQ(observable.value().diagnostics.data_rank,
            observable.value().diagnostics.expected_data_rank);
  EXPECT_EQ(observable.value().diagnostics.prior_resolved_accel_tilt_modes, 0U);

  for (MotionInitializationSegment& segment : synthetic.request.segments) {
    segment.lidar.information = singleDirectionPoseInformation();
  }
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::RankDeficientBatch)
      << initialized.error().detail;
}

TEST(MotionInitializer, RejectsSystematicallyInvertedRelativePoseDirection) {
  SyntheticBatch synthetic = dynamicBatch();
  for (MotionInitializationSegment& segment : synthetic.request.segments) {
    segment.lidar.T_imu_start_imu_end = segment.lidar.T_imu_start_imu_end.inverse();
  }

  MotionInitializerConfig config;
  config.maximum_lidar_mean_squared_whitened_residual = 0.25;
  config.maximum_imu_mean_squared_whitened_residual = 0.25;
  config.maximum_reduced_chi_square = 0.25;
  config.maximum_holdout_mean_squared_whitened_residual = 0.25;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::BiasPlausibilityFailed)
      << initialized.error().detail;
}

TEST(MotionInitializer, RejectsModelIncompatibleBatchByNormalizedResidual) {
  SyntheticBatch synthetic = dynamicBatch();
  synthetic.request.segments[4U].lidar.T_imu_start_imu_end =
      synthetic.request.segments[4U].lidar.T_imu_start_imu_end *
      core::Pose3d::exp(
          (Eigen::Matrix<double, 6, 1>() << 0.45, -0.30, 0.25, 0.08, -0.06, 0.05).finished());
  MotionInitializerConfig config;
  config.maximum_lidar_mean_squared_whitened_residual = 0.25;
  config.maximum_imu_mean_squared_whitened_residual = 0.25;
  config.maximum_reduced_chi_square = 0.25;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::StatisticalCompatibilityFailed)
      << initialized.error().detail;
}

TEST(MotionInitializer, RejectsWithheldFutureRegistrationMismatch) {
  SyntheticBatch synthetic = dynamicBatch();
  synthetic.request.segments.back().lidar.T_imu_start_imu_end =
      synthetic.request.segments.back().lidar.T_imu_start_imu_end *
      core::Pose3d::exp(
          (Eigen::Matrix<double, 6, 1>() << 0.50, -0.35, 0.20, 0.10, -0.08, 0.06).finished());
  MotionInitializerConfig config;
  config.maximum_lidar_mean_squared_whitened_residual = 1.0e6;
  config.maximum_imu_mean_squared_whitened_residual = 1.0e6;
  config.maximum_reduced_chi_square = 1.0e6;
  config.maximum_holdout_mean_squared_whitened_residual = 0.25;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::HoldoutCompatibilityFailed)
      << initialized.error().detail;
}

TEST(MotionInitializer, RejectsBiasOutsideCalibratedPriorSupport) {
  SyntheticBatch synthetic = dynamicBatch();
  synthetic.request.imu_noise.accelerometer_bias_prior_mean_mps2.setZero();
  synthetic.request.imu_noise.gyroscope_bias_prior_mean_radps.setZero();
  MotionInitializerConfig config;
  config.maximum_bias_prior_mahalanobis = 0.05;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::BiasPlausibilityFailed);
}

TEST(MotionInitializer, RejectsWhenNeitherRotationNorAccelerationPassesPreGate) {
  MotionInitializerConfig config;
  config.maximum_raw_imu_gap = core::Duration{kSegmentNanoseconds};
  const MotionInitializer initializer(config);
  const auto initialized = initializer.initialize(stationaryBatch());
  ASSERT_FALSE(initialized);
  EXPECT_EQ(initialized.error().code, MotionInitializationErrorCode::InsufficientExcitation);
}

TEST(MotionInitializer, RejectsInexactSupportAndIncompleteRawLineage) {
  SyntheticBatch support_case = dynamicBatch();
  support_case.request.segments[3U].imu.support.end =
      support_case.request.segments[3U].imu.support.end + core::Duration{1};
  const MotionInitializer initializer;
  const auto inexact = initializer.initialize(support_case.request);
  ASSERT_FALSE(inexact);
  EXPECT_EQ(inexact.error().code, MotionInitializationErrorCode::InexactImuSupport);
  ASSERT_EQ(inexact.error().segment, 3U);

  SyntheticBatch lineage_case = dynamicBatch();
  lineage_case.request.segments[2U].imu_lineage.usage.pop_back();
  const auto missing = initializer.initialize(lineage_case.request);
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code, MotionInitializationErrorCode::LineageSupportMismatch);
  ASSERT_EQ(missing.error().segment, 2U);
}

TEST(MotionInitializer, ReturnsFiniteJointCovarianceInPublicOrder) {
  const SyntheticBatch synthetic = dynamicBatch();
  const auto initialized = MotionInitializer{}.initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const NavigationCovariance& covariance = initialized.value().covariance;
  EXPECT_EQ(covariance.order, NavigationCovarianceOrder::RotationVelocityPositionGyroBiasAccelBias);
  EXPECT_TRUE(covariance.matrix.allFinite());
  EXPECT_TRUE(covariance.matrix.isApprox(covariance.matrix.transpose(), 1.0e-10));
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 15, 15>> eigensolver(covariance.matrix,
                                                                           Eigen::EigenvaluesOnly);
  ASSERT_EQ(eigensolver.info(), Eigen::Success);
  EXPECT_GT(eigensolver.eigenvalues().minCoeff(), -1.0e-10);
  EXPECT_GT(covariance.matrix.diagonal().segment<3>(3).minCoeff(), 0.0);
  EXPECT_GT(covariance.matrix.diagonal().segment<3>(6).minCoeff(), 0.0);
  EXPECT_GT(covariance.matrix.diagonal().segment<3>(9).minCoeff(), 0.0);
  EXPECT_GT(covariance.matrix.diagonal().segment<3>(12).minCoeff(), 0.0);
}

TEST(MotionInitializer, FloorsEveryCorrelatedThreeAxisMarginalEigenvalue) {
  const SyntheticBatch synthetic = dynamicBatch();
  MotionInitializerConfig config;
  config.minimum_orientation_variance_rad2 = 0.25;
  config.minimum_velocity_variance_m2ps2 = 0.36;
  config.minimum_position_variance_m2 = 0.49;
  config.minimum_gyro_bias_variance_rad2ps2 = 0.16;
  config.minimum_accel_bias_variance_m2ps4 = 0.64;
  const auto initialized = MotionInitializer(config).initialize(synthetic.request);
  ASSERT_TRUE(initialized) << initialized.error().detail;
  const NavigationCovariance& covariance = initialized.value().covariance;
  const std::array<std::pair<Eigen::Index, double>, 5> blocks{{
      {0, config.minimum_orientation_variance_rad2},
      {3, config.minimum_velocity_variance_m2ps2},
      {6, config.minimum_position_variance_m2},
      {9, config.minimum_gyro_bias_variance_rad2ps2},
      {12, config.minimum_accel_bias_variance_m2ps4},
  }};
  bool has_within_block_correlation = false;
  for (const auto& [offset, floor] : blocks) {
    const Eigen::Matrix3d marginal = covariance.matrix.block<3, 3>(offset, offset);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(marginal, Eigen::EigenvaluesOnly);
    ASSERT_EQ(eigensolver.info(), Eigen::Success);
    EXPECT_GE(eigensolver.eigenvalues().minCoeff(), floor - 1.0e-10);
    Eigen::Matrix3d off_diagonal = marginal;
    off_diagonal.diagonal().setZero();
    has_within_block_correlation =
        has_within_block_correlation || off_diagonal.cwiseAbs().maxCoeff() > 1.0e-10;
  }
  EXPECT_TRUE(has_within_block_correlation);
}

}  // namespace
}  // namespace meridian::local
