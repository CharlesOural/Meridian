#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>

#include "meridian/core/initialization.hpp"

namespace meridian::core {
namespace {

NavigationState makeSeed(TimeNs time = TimeNs(10)) {
  return NavigationState(StateId(1), time, Pose3d(), Vec3d{}, ImuBias());
}

InitializationQuality makeAcceptedDynamicQuality() {
  return InitializationQuality{
      .imu_sample_count = 400,
      .lidar_sweep_count = 20,
      .fitted_transition_count = 18,
      .rejected_transition_count = 0,
      .registration_min_singular_value = 0.2,
      .registration_condition_number = 12.0,
      .gyro_bias_min_singular_value = 0.1,
      .gyro_bias_condition_number = 8.0,
      .gravity_min_singular_value = 0.05,
      .gravity_condition_number = 20.0,
      .raw_gravity_magnitude_m_s2 = 9.81,
      .gyro_bias_correction_norm_rad_s = 0.003,
      .alignment_residual_rms = 0.02,
      .held_out_rotation_error_rad = 0.01,
      .held_out_translation_error_m = 0.04,
      .refinement_rotation_change_rad = 0.005,
      .refinement_translation_change_m = 0.01,
      .all_required_gates_passed = true,
  };
}

TEST(InitializationStatus, HasStableDiagnosticNames) {
  EXPECT_STREQ(toString(InitializationMode::kStatic), "static");
  EXPECT_STREQ(toString(InitializationMode::kDynamic), "dynamic");
  EXPECT_STREQ(toString(InitializationStatus::kCollecting), "collecting");
  EXPECT_STREQ(toString(InitializationStatus::kAccepted), "accepted");
  EXPECT_STREQ(toString(InitializationStatus::kFailed), "failed");
}

TEST(InitializationQuality, RejectsImpossibleCountsAndInvalidMetrics) {
  InitializationQuality impossible_counts;
  impossible_counts.fitted_transition_count = 1;
  EXPECT_FALSE(impossible_counts.isValid());

  InitializationQuality bad_condition = makeAcceptedDynamicQuality();
  bad_condition.gravity_condition_number = 0.5;
  EXPECT_FALSE(bad_condition.isValid());

  InitializationQuality nonfinite = makeAcceptedDynamicQuality();
  nonfinite.alignment_residual_rms = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(nonfinite.isValid());
}

TEST(InitializationResult, PreservesAnAcceptedDynamicWarmStart) {
  const InitializationResult result(InitializationMode::kDynamic, TimeNs(10), makeSeed(),
                                    TimeRange(TimeNs(10), TimeNs(30)),
                                    makeAcceptedDynamicQuality());
  EXPECT_EQ(result.method(), InitializationMode::kDynamic);
  EXPECT_EQ(result.anchorTime(), TimeNs(10));
  EXPECT_EQ(result.seedState().time(), TimeNs(10));
  EXPECT_EQ(result.support(), TimeRange(TimeNs(10), TimeNs(30)));
  EXPECT_EQ(result.quality().fitted_transition_count, 18U);
}

TEST(InitializationResult, RejectsPartialOrInconsistentResults) {
  InitializationQuality failed_gate = makeAcceptedDynamicQuality();
  failed_gate.all_required_gates_passed = false;
  EXPECT_THROW(
      static_cast<void>(InitializationResult(InitializationMode::kDynamic, TimeNs(10), makeSeed(),
                                             TimeRange(TimeNs(10), TimeNs(30)), failed_gate)),
      std::invalid_argument);

  InitializationQuality no_holdout = makeAcceptedDynamicQuality();
  no_holdout.held_out_rotation_error_rad.reset();
  EXPECT_THROW(
      static_cast<void>(InitializationResult(InitializationMode::kDynamic, TimeNs(10), makeSeed(),
                                             TimeRange(TimeNs(10), TimeNs(30)), no_holdout)),
      std::invalid_argument);

  InitializationQuality no_supporting_samples = makeAcceptedDynamicQuality();
  no_supporting_samples.imu_sample_count = 0;
  EXPECT_THROW(static_cast<void>(InitializationResult(InitializationMode::kDynamic, TimeNs(10),
                                                      makeSeed(), TimeRange(TimeNs(10), TimeNs(30)),
                                                      no_supporting_samples)),
               std::invalid_argument);

  InitializationQuality no_reserved_transition = makeAcceptedDynamicQuality();
  no_reserved_transition.fitted_transition_count = 19;
  EXPECT_THROW(static_cast<void>(InitializationResult(InitializationMode::kDynamic, TimeNs(10),
                                                      makeSeed(), TimeRange(TimeNs(10), TimeNs(30)),
                                                      no_reserved_transition)),
               std::invalid_argument);

  EXPECT_THROW(static_cast<void>(InitializationResult(
                   InitializationMode::kDynamic, TimeNs(10), makeSeed(TimeNs(11)),
                   TimeRange(TimeNs(10), TimeNs(30)), makeAcceptedDynamicQuality())),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(InitializationResult(
                   InitializationMode::kDynamic, TimeNs(30), makeSeed(TimeNs(30)),
                   TimeRange(TimeNs(10), TimeNs(30)), makeAcceptedDynamicQuality())),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(InitializationResult(InitializationMode::kDynamic, TimeNs(10),
                                                      makeSeed(), TimeRange(TimeNs(10), TimeNs(10)),
                                                      makeAcceptedDynamicQuality())),
               std::invalid_argument);
}

TEST(InitializationResult, AllowsStaticQualityWithoutDynamicHeldOutMetrics) {
  InitializationQuality quality;
  quality.imu_sample_count = 200;
  quality.raw_gravity_magnitude_m_s2 = 9.81;
  quality.all_required_gates_passed = true;

  EXPECT_NO_THROW(
      static_cast<void>(InitializationResult(InitializationMode::kStatic, TimeNs(10), makeSeed(),
                                             TimeRange(TimeNs(10), TimeNs(20)), quality)));
}

}  // namespace
}  // namespace meridian::core
