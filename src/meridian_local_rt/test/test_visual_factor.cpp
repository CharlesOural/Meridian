#include <gtest/gtest.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "meridian/local/visual_factor.hpp"
#include "visual_gtsam_factor.hpp"

namespace meridian::local {
namespace {

using Vector6 = Eigen::Matrix<double, 6, 1>;

[[nodiscard]] EquidistantCameraParameters cameraParameters() {
  return EquidistantCameraParameters{720U,    540U,     352.779,  354.609, 359.035,
                                     260.546, -0.04217, -0.00413, 0.00179, -0.00063};
}

[[nodiscard]] core::ObservationLineage frontendLineage(std::uint64_t id, core::MeasurementId frame,
                                                       core::CalibrationEpoch calibration) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId(id);
  core::ObservationUsage usage;
  usage.slice.root = frame;
  usage.slice.kind = core::SliceKind::Whole;
  usage.slice.calibration = calibration;
  usage.role = core::ObservationRole::DerivedSummary;
  usage.consumer = core::DerivedRecordId(id);
  lineage.usage.push_back(usage);
  return lineage;
}

[[nodiscard]] core::ObservationLineage factorLineage(core::MeasurementId anchor,
                                                     core::MeasurementId observer,
                                                     core::CalibrationEpoch calibration) {
  core::ObservationLineage lineage;
  lineage.id = core::ObservationLineageId(91U);
  core::ObservationUsage anchor_usage;
  anchor_usage.slice.root = anchor;
  anchor_usage.slice.kind = core::SliceKind::Whole;
  anchor_usage.slice.calibration = calibration;
  anchor_usage.role = core::ObservationRole::ConditioningOnly;
  anchor_usage.consumer = core::DerivedRecordId(91U);
  lineage.usage.push_back(anchor_usage);
  core::ObservationUsage observer_usage;
  observer_usage.slice.root = observer;
  observer_usage.slice.kind = core::SliceKind::Whole;
  observer_usage.slice.calibration = calibration;
  observer_usage.role = core::ObservationRole::PrimaryResidual;
  observer_usage.consumer = core::DerivedRecordId(91U);
  observer_usage.factor_group = core::FactorGroupId(31U);
  lineage.usage.push_back(observer_usage);
  return lineage;
}

[[nodiscard]] core::Pose3d pose(const Eigen::Vector3d& translation,
                                Eigen::Vector3d rotation = Eigen::Vector3d::Zero()) {
  return core::Pose3d(Sophus::SO3d::exp(rotation), translation);
}

[[nodiscard]] gtsam::Pose3 toGtsam(const core::Pose3d& value) {
  return gtsam::Pose3(gtsam::Rot3(value.so3().matrix()), value.translation());
}

[[nodiscard]] VisualReprojectionFactorSpec nominalSpec(
    const core::Pose3d& anchor_pose, const core::Pose3d& observer_pose, double eta,
    Eigen::Vector2d residual = Eigen::Vector2d{0.4, -0.3},
    Eigen::Vector2d anchor_pixel = Eigen::Vector2d{390.0, 245.0}) {
  const core::CalibrationEpoch calibration(7U);
  VisualReprojectionFactorSpec spec;
  spec.id = core::FactorId(31U);
  spec.landmark = VisualLandmarkId(8U);
  spec.track = VisualTrackId(4U);
  spec.anchor.frame = core::MeasurementId(10U);
  spec.anchor.state = core::StateId(1U);
  spec.anchor.exact_time = core::FusionTime{0LL};
  spec.anchor.camera = core::CameraId(0U);
  spec.anchor.calibration = calibration;
  spec.anchor.pixel = anchor_pixel;
  spec.anchor.pixel_covariance = (Eigen::Matrix2d() << 0.8, 0.1, 0.1, 1.2).finished();
  spec.anchor.camera_model = cameraParameters();
  spec.anchor.imu_from_camera =
      core::ImuFromCameraTransform(pose({0.04, -0.01, 0.02}, {0.01, -0.015, 0.005}));
  spec.observer = spec.anchor;
  spec.observer.frame = core::MeasurementId(11U);
  spec.observer.state = core::StateId(2U);
  spec.observer.exact_time = core::FusionTime{200'000'000LL};
  spec.observer.pixel = Eigen::Vector2d{350.0, 260.0};
  spec.observer.pixel_covariance = (Eigen::Matrix2d() << 1.1, -0.15, -0.15, 0.9).finished();
  spec.lineage = factorLineage(spec.anchor.frame, spec.observer.frame, calibration);

  const auto initial = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta);
  EXPECT_TRUE(initial) << initial.error().detail;
  if (initial) {
    spec.observer.pixel = initial.value().predicted_pixel + residual;
  }
  return spec;
}

TEST(VisualReprojection, AnalyticJacobiansMatchCentralDifferences) {
  const core::Pose3d anchor_pose = pose({1.2, -0.7, 0.4}, {0.12, -0.08, 0.05});
  const core::Pose3d observer_pose = anchor_pose * pose({0.35, -0.06, 0.03}, {0.015, -0.025, 0.01});
  const double eta = -std::log(6.0);
  const VisualReprojectionFactorSpec spec = nominalSpec(anchor_pose, observer_pose, eta);
  const auto evaluation = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta);
  ASSERT_TRUE(evaluation) << evaluation.error().detail;

  Eigen::Matrix<double, 2, 6> numeric_anchor;
  Eigen::Matrix<double, 2, 6> numeric_observer;
  constexpr double kStep = 1.0e-6;
  for (Eigen::Index index = 0; index < 6; ++index) {
    Vector6 delta = Vector6::Zero();
    delta(index) = kStep;
    const auto anchor_plus = evaluateVisualReprojection(
        spec, anchor_pose * core::Pose3d::exp(delta), observer_pose, eta);
    const auto anchor_minus = evaluateVisualReprojection(
        spec, anchor_pose * core::Pose3d::exp(-delta), observer_pose, eta);
    ASSERT_TRUE(anchor_plus);
    ASSERT_TRUE(anchor_minus);
    numeric_anchor.col(index) =
        (anchor_plus.value().residual - anchor_minus.value().residual) / (2.0 * kStep);

    const auto observer_plus = evaluateVisualReprojection(
        spec, anchor_pose, observer_pose * core::Pose3d::exp(delta), eta);
    const auto observer_minus = evaluateVisualReprojection(
        spec, anchor_pose, observer_pose * core::Pose3d::exp(-delta), eta);
    ASSERT_TRUE(observer_plus);
    ASSERT_TRUE(observer_minus);
    numeric_observer.col(index) =
        (observer_plus.value().residual - observer_minus.value().residual) / (2.0 * kStep);
  }
  const auto eta_plus = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta + kStep);
  const auto eta_minus = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta - kStep);
  ASSERT_TRUE(eta_plus);
  ASSERT_TRUE(eta_minus);
  const Eigen::Vector2d numeric_eta =
      (eta_plus.value().residual - eta_minus.value().residual) / (2.0 * kStep);

  EXPECT_TRUE(evaluation.value().anchor_pose_jacobian.isApprox(numeric_anchor, 2.0e-5))
      << evaluation.value().anchor_pose_jacobian << "\nvs\n"
      << numeric_anchor;
  EXPECT_TRUE(evaluation.value().observer_pose_jacobian.isApprox(numeric_observer, 2.0e-5))
      << evaluation.value().observer_pose_jacobian << "\nvs\n"
      << numeric_observer;
  EXPECT_TRUE(evaluation.value().eta_jacobian.isApprox(numeric_eta, 2.0e-5));
}

TEST(VisualReprojection, AdversarialEdgeGeometryJacobiansRemainCorrect) {
  const core::Pose3d anchor_pose = pose({-2.0, 1.0, 0.7}, {0.4, -0.2, 0.3});
  const core::Pose3d observer_pose = anchor_pose * pose({0.8, 0.15, -0.04}, {-0.08, 0.06, -0.04});
  const double eta = -std::log(18.0);
  const auto spec = nominalSpec(anchor_pose, observer_pose, eta, {0.0, 0.0}, {620.0, 420.0});
  const auto evaluation = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta);
  ASSERT_TRUE(evaluation) << evaluation.error().detail;

  constexpr double kStep = 5.0e-7;
  for (Eigen::Index index = 0; index < 6; ++index) {
    Vector6 delta = Vector6::Zero();
    delta(index) = kStep;
    const auto plus = evaluateVisualReprojection(spec, anchor_pose,
                                                 observer_pose * core::Pose3d::exp(delta), eta);
    const auto minus = evaluateVisualReprojection(spec, anchor_pose,
                                                  observer_pose * core::Pose3d::exp(-delta), eta);
    ASSERT_TRUE(plus);
    ASSERT_TRUE(minus);
    const Eigen::Vector2d numeric =
        (plus.value().residual - minus.value().residual) / (2.0 * kStep);
    EXPECT_TRUE(evaluation.value().observer_pose_jacobian.col(index).isApprox(numeric, 4.0e-5));
  }
}

TEST(VisualReprojection, CommonWorldTransformDoesNotChangeResidual) {
  const core::Pose3d anchor_pose = pose({0.0, 0.0, 0.0});
  const core::Pose3d observer_pose = pose({0.3, -0.05, 0.02}, {0.01, 0.0, 0.0});
  const double eta = -std::log(5.0);
  const auto spec = nominalSpec(anchor_pose, observer_pose, eta);
  const auto first = evaluateVisualReprojection(spec, anchor_pose, observer_pose, eta);
  const core::Pose3d T_new_odom = pose({4.0, -3.0, 1.0}, {0.3, -0.1, 0.2});
  const auto transformed =
      evaluateVisualReprojection(spec, T_new_odom * anchor_pose, T_new_odom * observer_pose, eta);
  ASSERT_TRUE(first);
  ASSERT_TRUE(transformed);
  EXPECT_TRUE(first.value().residual.isApprox(transformed.value().residual, 1.0e-10));
}

TEST(VisualReprojection, CheiralityAndRangeFailuresAreTyped) {
  const core::Pose3d anchor_pose;
  const double eta = -std::log(5.0);
  const core::Pose3d nominal_observer = pose({0.2, 0.0, 0.0});
  const auto spec = nominalSpec(anchor_pose, nominal_observer, eta);

  const auto behind = evaluateVisualReprojection(spec, anchor_pose, pose({0.0, 0.0, 10.0}), eta);
  ASSERT_FALSE(behind);
  EXPECT_EQ(behind.error().code, VisualReprojectionErrorCode::ObserverCheirality);

  const auto range =
      evaluateVisualReprojection(spec, anchor_pose, nominal_observer, -std::log(500.0));
  ASSERT_FALSE(range);
  EXPECT_EQ(range.error().code, VisualReprojectionErrorCode::RangeOutsideBounds);
}

TEST(AnchoredInverseRangeFactor, GtsamRightTangentPermutationIsExact) {
  const core::Pose3d anchor_pose = pose({0.4, -0.2, 0.1}, {0.1, -0.04, 0.03});
  const core::Pose3d observer_pose = anchor_pose * pose({0.3, 0.02, 0.0}, {0.01, -0.02, 0.015});
  const double eta = -std::log(7.0);
  const auto spec = nominalSpec(anchor_pose, observer_pose, eta);
  gtsam_api::AnchoredInverseRangeFactor factor(gtsam::Symbol('x', 1U), gtsam::Symbol('x', 2U),
                                               gtsam::Symbol('l', 8U), spec);
  const double bounded_eta_latent =
      gtsam_api::encodeBoundedEta(eta, spec.minimum_range_m, spec.maximum_range_m);

  const gtsam::Pose3 gtsam_anchor = toGtsam(anchor_pose);
  const gtsam::Pose3 gtsam_observer = toGtsam(observer_pose);
  gtsam::Matrix anchor_jacobian;
  gtsam::Matrix observer_jacobian;
  gtsam::Matrix eta_jacobian;
  const gtsam::Vector residual = factor.evaluateError(
      gtsam_anchor, gtsam_observer, bounded_eta_latent, anchor_jacobian, observer_jacobian,
      eta_jacobian);
  EXPECT_TRUE(residual.allFinite());

  Eigen::Matrix<double, 2, 6> numeric_anchor;
  Eigen::Matrix<double, 2, 6> numeric_observer;
  constexpr double kStep = 1.0e-6;
  for (Eigen::Index index = 0; index < 6; ++index) {
    gtsam::Vector6 delta = gtsam::Vector6::Zero();
    delta(index) = kStep;
    numeric_anchor.col(index) =
        (factor.evaluateError(gtsam_anchor.retract(delta), gtsam_observer, bounded_eta_latent) -
         factor.evaluateError(gtsam_anchor.retract(-delta), gtsam_observer, bounded_eta_latent)) /
        (2.0 * kStep);
    numeric_observer.col(index) =
        (factor.evaluateError(gtsam_anchor, gtsam_observer.retract(delta), bounded_eta_latent) -
         factor.evaluateError(gtsam_anchor, gtsam_observer.retract(-delta), bounded_eta_latent)) /
        (2.0 * kStep);
  }
  const Eigen::Vector2d numeric_eta =
      (factor.evaluateError(gtsam_anchor, gtsam_observer, bounded_eta_latent + kStep) -
       factor.evaluateError(gtsam_anchor, gtsam_observer, bounded_eta_latent - kStep)) /
      (2.0 * kStep);
  EXPECT_TRUE(anchor_jacobian.isApprox(numeric_anchor, 2.0e-5));
  EXPECT_TRUE(observer_jacobian.isApprox(numeric_observer, 2.0e-5));
  EXPECT_TRUE(eta_jacobian.isApprox(numeric_eta, 2.0e-5));
}

TEST(BoundedEtaParameterization, RoundTripAndDerivativeMatchCentralDifference) {
  constexpr double kMinimumRange = 0.3;
  constexpr double kMaximumRange = 200.0;
  const double eta = -std::log(7.0);
  const double latent =
      gtsam_api::encodeBoundedEta(eta, kMinimumRange, kMaximumRange);
  const auto decoded =
      gtsam_api::decodeBoundedEta(latent, kMinimumRange, kMaximumRange);
  EXPECT_NEAR(decoded.eta, eta, 1.0e-12);

  constexpr double kStep = 1.0e-6;
  const double numeric =
      (gtsam_api::decodeBoundedEta(latent + kStep, kMinimumRange, kMaximumRange).eta -
       gtsam_api::decodeBoundedEta(latent - kStep, kMinimumRange, kMaximumRange).eta) /
      (2.0 * kStep);
  EXPECT_NEAR(decoded.derivative_wrt_latent, numeric, 1.0e-9);
}

TEST(BoundedEtaParameterization, ExtremeFiniteLatentsRemainStrictlyInsideRangeBounds) {
  constexpr double kMinimumRange = 0.3;
  constexpr double kMaximumRange = 200.0;
  for (const double latent : {-std::numeric_limits<double>::max(), -1.0e6, 1.0e6,
                              std::numeric_limits<double>::max()}) {
    const auto decoded =
        gtsam_api::decodeBoundedEta(latent, kMinimumRange, kMaximumRange);
    const double range = std::exp(-decoded.eta);
    EXPECT_TRUE(std::isfinite(decoded.eta));
    EXPECT_TRUE(std::isfinite(decoded.derivative_wrt_latent));
    EXPECT_GE(decoded.derivative_wrt_latent, 0.0);
    EXPECT_GT(range, kMinimumRange);
    EXPECT_LT(range, kMaximumRange);
  }
}

TEST(AnchoredInverseRangeFactor, ExtremeFiniteLatentsHaveFiniteResidualAndJacobian) {
  const core::Pose3d anchor_pose;
  const core::Pose3d observer_pose;
  const double eta = -std::log(5.0);
  const auto spec = nominalSpec(anchor_pose, observer_pose, eta);
  gtsam_api::AnchoredInverseRangeFactor factor(gtsam::Symbol('x', 1U), gtsam::Symbol('x', 2U),
                                               gtsam::Symbol('l', 8U), spec);
  for (const double latent : {-1.0e6, 1.0e6}) {
    gtsam::Matrix latent_jacobian;
    const gtsam::Vector residual = factor.evaluateError(
        toGtsam(anchor_pose), toGtsam(observer_pose), latent, boost::none, boost::none,
        latent_jacobian);
    EXPECT_TRUE(residual.allFinite());
    EXPECT_TRUE(latent_jacobian.allFinite());
  }
}

[[nodiscard]] VisualKeyframeContext keyframeContext(std::uint64_t state, std::int64_t time_ns,
                                                    double x) {
  VisualKeyframeContext context;
  context.state = core::StateId(state);
  context.exact_time = core::FusionTime{time_ns};
  context.camera = core::CameraId(0U);
  context.calibration = core::CalibrationEpoch(7U);
  context.camera_model = cameraParameters();
  context.imu_from_camera = core::ImuFromCameraTransform(core::Pose3d{});
  context.T_odom_imu = pose({x, 0.0, 0.0});
  return context;
}

[[nodiscard]] VisualFrontendOutput keyframeOutput(
    std::uint64_t frame_id, std::int64_t time_ns, const VisualKeyframeContext& context,
    const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>>& points,
    std::optional<std::pair<VisualTrackId, Eigen::Vector2d>> pixel_override = std::nullopt) {
  VisualFrontendOutput output;
  output.source_frame = core::MeasurementId(frame_id);
  output.camera = core::CameraId(0U);
  output.exposure_midpoint = core::FusionTime{time_ns};
  output.keyframe = true;
  output.lineage = frontendLineage(frame_id, output.source_frame, context.calibration);
  const EquidistantCamera camera(context.camera_model);
  const core::Pose3d T_camera_odom =
      (context.T_odom_imu * context.imu_from_camera.T_imu_camera()).inverse();
  for (const auto& [track, point_odom] : points) {
    const auto projection = camera.project(T_camera_odom * point_odom);
    EXPECT_TRUE(projection);
    if (!projection) {
      continue;
    }
    VisualFeatureObservation feature;
    feature.track = track;
    feature.distorted_pixel = projection.value().pixel;
    if (pixel_override && pixel_override->first == track) {
      feature.distorted_pixel += pixel_override->second;
    }
    const auto ray = camera.unproject(feature.distorted_pixel);
    EXPECT_TRUE(ray);
    feature.unit_bearing = ray.value().unit_ray;
    feature.pixel_covariance = Eigen::Matrix2d::Identity();
    feature.age = 5U;
    output.features.push_back(feature);
  }
  return output;
}

TEST(VisualFactorBatchBuilder, SeedsAfterThreeViewsAndEmitsDirectFactors) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(5U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.1, -0.05, 5.0}}};

  const auto first_context = keyframeContext(1U, 0LL, 0.0);
  const auto first =
      builder.processKeyframe(keyframeOutput(101U, 0LL, first_context, points), first_context);
  ASSERT_TRUE(first) << first.error().detail;
  EXPECT_TRUE(first.value().new_landmarks.empty());
  EXPECT_TRUE(first.value().factors.empty());

  const auto second_context = keyframeContext(2U, 200'000'000LL, 0.2);
  const auto second = builder.processKeyframe(
      keyframeOutput(102U, 200'000'000LL, second_context, points), second_context);
  ASSERT_TRUE(second) << second.error().detail;
  EXPECT_TRUE(second.value().new_landmarks.empty());

  const auto third_context = keyframeContext(3U, 400'000'000LL, 0.4);
  const auto third = builder.processKeyframe(
      keyframeOutput(103U, 400'000'000LL, third_context, points), third_context);
  ASSERT_TRUE(third) << third.error().detail;
  ASSERT_EQ(third.value().new_landmarks.size(), 1U);
  ASSERT_EQ(third.value().factors.size(), 2U);
  EXPECT_NEAR(third.value().new_landmarks.front().initial_range_m,
              std::sqrt(25.0 + 0.1 * 0.1 + 0.05 * 0.05), 1.0e-6);
  EXPECT_EQ(third.value().factors.front().anchor.state, core::StateId(1U));
  EXPECT_EQ(third.value().factors.front().observer.state, core::StateId(2U));
  EXPECT_EQ(third.value().factors.back().observer.state, core::StateId(3U));
  for (const auto& factor : third.value().factors) {
    EXPECT_EQ(core::validateLineage(factor.lineage), core::LineageValidationError::None);
    EXPECT_TRUE(factor.anchor.imu_from_camera.T_imu_camera().matrix().isApprox(
        core::Pose3d{}.matrix(), 0.0));
  }
}

TEST(VisualFactorBatchBuilder, RejectsCameraContextMismatchTransactionally) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(15U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.0, 0.0, 5.0}}};
  auto context = keyframeContext(1U, 0LL, 0.0);
  const VisualFrontendOutput frontend = keyframeOutput(151U, 0LL, context, points);
  context.camera = core::CameraId(1U);

  const auto rejected = builder.processKeyframe(frontend, context);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, VisualFactorBuilderErrorCode::StateOrCameraMismatch);
  EXPECT_EQ(builder.activeTracks(), 0U);
}

TEST(VisualFactorBatchBuilder, RobustTriangulationRejectsOneBadView) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(6U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.0, 0.0, 6.0}}};
  for (std::size_t index = 0U; index < 4U; ++index) {
    const auto context =
        keyframeContext(index + 1U, static_cast<std::int64_t>(index) * 200'000'000LL,
                        0.18 * static_cast<double>(index));
    const auto override = index == 1U ? std::optional<std::pair<VisualTrackId, Eigen::Vector2d>>(
                                            std::pair{track, Eigen::Vector2d{28.0, -18.0}})
                                      : std::nullopt;
    const auto batch = builder.processKeyframe(
        keyframeOutput(201U + index, context.exact_time.nanoseconds, context, points, override),
        context);
    ASSERT_TRUE(batch) << batch.error().detail;
    if (index == 3U) {
      ASSERT_FALSE(batch.value().report.triangulation.empty());
      const VisualTriangulationDecision& decision = batch.value().report.triangulation.back();
      ASSERT_EQ(batch.value().new_landmarks.size(), 1U)
          << "status=" << static_cast<int>(decision.status) << " inliers=" << decision.inliers
          << " condition=" << decision.condition_number
          << " rmse=" << decision.reprojection_rmse_px;
      EXPECT_EQ(batch.value().new_landmarks.front().triangulation.inliers, 3U);
      EXPECT_EQ(batch.value().factors.size(), 2U);
    }
  }
}

TEST(VisualFactorBatchBuilder, RobustTriangulationReanchorsPastBadFirstView) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(16U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.0, 0.0, 6.0}}};
  VisualFactorBatch initialized;
  for (std::size_t index = 0U; index < 4U; ++index) {
    const auto context =
        keyframeContext(index + 1U, static_cast<std::int64_t>(index) * 200'000'000LL,
                        0.18 * static_cast<double>(index));
    const auto override = index == 0U ? std::optional<std::pair<VisualTrackId, Eigen::Vector2d>>(
                                            std::pair{track, Eigen::Vector2d{30.0, -20.0}})
                                      : std::nullopt;
    const auto batch = builder.processKeyframe(
        keyframeOutput(251U + index, context.exact_time.nanoseconds, context, points, override),
        context);
    ASSERT_TRUE(batch) << batch.error().detail;
    if (index == 3U) {
      initialized = batch.value();
    }
  }
  ASSERT_EQ(initialized.new_landmarks.size(), 1U);
  EXPECT_EQ(initialized.new_landmarks.front().anchor.state, core::StateId(2U));
  EXPECT_EQ(initialized.new_landmarks.front().triangulation.inliers, 3U);
  ASSERT_EQ(initialized.factors.size(), 2U);
  EXPECT_EQ(initialized.factors.front().observer.state, core::StateId(3U));
  EXPECT_EQ(initialized.factors.back().observer.state, core::StateId(4U));
}

TEST(VisualFactorBatchBuilder, RetiresOnlyPersistentlyBadObservationFactors) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(7U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.0, 0.0, 5.0}}};
  VisualFactorBatch seeded;
  for (std::size_t index = 0U; index < 3U; ++index) {
    const auto context =
        keyframeContext(index + 1U, static_cast<std::int64_t>(index) * 200'000'000LL,
                        0.2 * static_cast<double>(index));
    const auto batch = builder.processKeyframe(
        keyframeOutput(301U + index, context.exact_time.nanoseconds, context, points), context);
    ASSERT_TRUE(batch);
    if (index == 2U) {
      seeded = batch.value();
    }
  }
  ASSERT_EQ(seeded.factors.size(), 2U);
  const core::FactorId bad_factor = seeded.factors.front().id;
  const core::FactorId good_factor = seeded.factors.back().id;

  const auto first = builder.applyAcceptedResiduals(core::LocalGraphRevision(1U),
                                                    {{bad_factor, 20.0}, {good_factor, 1.0}});
  ASSERT_TRUE(first);
  EXPECT_TRUE(first.value().retired_observation_factors.empty());
  const auto second = builder.applyAcceptedResiduals(core::LocalGraphRevision(2U),
                                                     {{bad_factor, 20.0}, {good_factor, 1.0}});
  ASSERT_TRUE(second);
  ASSERT_EQ(second.value().retired_observation_factors.size(), 1U);
  EXPECT_EQ(second.value().retired_observation_factors.front(), bad_factor);
  EXPECT_EQ(builder.activeTracks(), 1U);
}

TEST(VisualFactorBatchBuilder, FinalityPrunesAcceptedTrackHealthAndNeverReusesIds) {
  VisualFactorBatchBuilder builder;
  const VisualTrackId track(17U);
  const std::vector<std::pair<VisualTrackId, Eigen::Vector3d>> points{
      {track, Eigen::Vector3d{0.0, 0.0, 5.0}}};
  VisualFactorBatch seeded;
  for (std::size_t index = 0U; index < 3U; ++index) {
    const auto context =
        keyframeContext(index + 1U, static_cast<std::int64_t>(index) * 200'000'000LL,
                        0.2 * static_cast<double>(index));
    auto batch = builder.processKeyframe(
        keyframeOutput(401U + index, context.exact_time.nanoseconds, context, points), context);
    ASSERT_TRUE(batch) << batch.error().detail;
    if (index == 2U) {
      seeded = std::move(batch).value();
    }
  }
  ASSERT_EQ(seeded.new_landmarks.size(), 1U);
  ASSERT_EQ(seeded.factors.size(), 2U);
  const VisualLandmarkId finalized_landmark = seeded.new_landmarks.front().landmark;
  std::vector<core::FactorId> finalized_factors;
  for (const VisualReprojectionFactorSpec& factor : seeded.factors) {
    finalized_factors.push_back(factor.id);
  }
  std::sort(finalized_factors.begin(), finalized_factors.end());

  VisualFinalityUpdate finality;
  finality.revision = core::LocalGraphRevision(1U);
  finality.finalized_factors = finalized_factors;
  finality.finalized_landmarks = {finalized_landmark};
  const auto reconciled = builder.reconcileFinality(finality);
  ASSERT_TRUE(reconciled) << reconciled.error().detail;
  EXPECT_EQ(reconciled.value().accepted_tracks_pruned, 1U);
  EXPECT_EQ(reconciled.value().factor_health_entries_pruned, 2U);
  EXPECT_EQ(builder.activeTracks(), 0U);

  VisualFinalityUpdate empty;
  empty.revision = core::LocalGraphRevision(2U);
  ASSERT_TRUE(builder.reconcileFinality(empty));
  const auto repeated = builder.reconcileFinality(empty);
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, VisualFactorBuilderErrorCode::InvalidFinality);

  VisualFactorBatch reseeded;
  for (std::size_t index = 0U; index < 3U; ++index) {
    const auto context = keyframeContext(index + 4U,
                                         static_cast<std::int64_t>(index + 3U) * 200'000'000LL,
                                         0.2 * static_cast<double>(index + 3U));
    auto batch = builder.processKeyframe(
        keyframeOutput(404U + index, context.exact_time.nanoseconds, context, points), context);
    ASSERT_TRUE(batch) << batch.error().detail;
    if (index == 2U) {
      reseeded = std::move(batch).value();
    }
  }
  ASSERT_EQ(reseeded.new_landmarks.size(), 1U);
  EXPECT_GT(reseeded.new_landmarks.front().landmark, finalized_landmark);
  ASSERT_FALSE(reseeded.factors.empty());
  EXPECT_GT(reseeded.factors.front().id, finalized_factors.back());
}

}  // namespace
}  // namespace meridian::local
