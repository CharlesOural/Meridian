#include <gtest/gtest.h>

#include <Eigen/Core>

#include "gnss_gate.hpp"
#include "meridian/common/sample.hpp"
#include "meridian/config/config.hpp"

using meridian::BackendConfig;
using meridian::GnssFix;
using meridian::backend::GnssGate;

namespace {

// A tight, good-quality fix the default gate accepts: small ENU covariance and an
// RTK_Fixed solution.
GnssFix goodFix() {
  GnssFix fix;
  fix.fix = GnssFix::FixType::RTK_Fixed;
  fix.cov_enu = 0.01 * Eigen::Matrix3d::Identity();  // trace = 0.03 m^2
  return fix;
}

// A config with the spacing and skip-if-confident gates active at their defaults.
BackendConfig cfg() {
  BackendConfig c;
  c.gnss_max_cov = 25.0;
  c.gnss_skip_if_confident = true;
  c.gnss_skip_confidence_k = 1.0;
  c.gnss_min_spacing = 1.0;
  return c;
}

// A marginal trace large enough that the skip-if-confident gate can never fire.
constexpr double kHugeMarginal = 1e9;

}  // namespace

TEST(GnssGate, AcceptsGoodFixWithTravelAndLooseMarginal) {
  GnssGate gate;
  const auto d = gate.evaluate(goodFix(), kHugeMarginal, /*travelled=*/2.0, cfg());
  EXPECT_EQ(d, GnssGate::Decision::Accept);
}

TEST(GnssGate, RejectsNoneFix) {
  GnssGate gate;
  GnssFix fix = goodFix();
  fix.fix = GnssFix::FixType::None;
  const auto d = gate.evaluate(fix, kHugeMarginal, /*travelled=*/2.0, cfg());
  EXPECT_EQ(d, GnssGate::Decision::RejectQuality);
}

TEST(GnssGate, RejectsCovarianceAboveMax) {
  GnssGate gate;
  GnssFix fix = goodFix();
  fix.cov_enu = 20.0 * Eigen::Matrix3d::Identity();  // trace = 60 > gnss_max_cov
  const auto d = gate.evaluate(fix, kHugeMarginal, /*travelled=*/2.0, cfg());
  EXPECT_EQ(d, GnssGate::Decision::RejectQuality);
}

// Skip-if-confident fires when the fix covariance is no tighter than the back-end's own
// position marginal (k = 1): a loose fix against a tight marginal is redundant.
TEST(GnssGate, SkipsWhenFixLooserThanTightMarginal) {
  GnssGate gate;
  GnssFix fix = goodFix();
  fix.cov_enu = 1.0 * Eigen::Matrix3d::Identity();  // trace = 3 m^2
  const double tight_marginal = 0.03;               // back-end is far tighter
  const auto d = gate.evaluate(fix, tight_marginal, /*travelled=*/2.0, cfg());
  EXPECT_EQ(d, GnssGate::Decision::SkipConfident);
}

// The same loose fix is admitted when the marginal trace is huge (gate disabled), proving
// the bypass path: a poorly-known back-end welcomes even a loose fix.
TEST(GnssGate, BypassesSkipWhenMarginalHuge) {
  GnssGate gate;
  GnssFix fix = goodFix();
  fix.cov_enu = 1.0 * Eigen::Matrix3d::Identity();
  const auto d = gate.evaluate(fix, kHugeMarginal, /*travelled=*/2.0, cfg());
  EXPECT_EQ(d, GnssGate::Decision::Accept);
}

// Spacing decimation: a fix within min_spacing of the last admitted one is skipped; once
// the caller has travelled past the threshold the same fix is accepted.
TEST(GnssGate, SkipsThenAcceptsAcrossSpacing) {
  GnssGate gate;
  const BackendConfig c = cfg();  // gnss_min_spacing = 1.0 m

  const auto near = gate.evaluate(goodFix(), kHugeMarginal, /*travelled=*/0.5, c);
  EXPECT_EQ(near, GnssGate::Decision::SkipSpacing);

  // The caller notes the (eventual) admission and resets its spacing baseline; after
  // travelling past the threshold the next evaluation accepts.
  gate.note_admitted();
  const auto far = gate.evaluate(goodFix(), kHugeMarginal, /*travelled=*/1.5, c);
  EXPECT_EQ(far, GnssGate::Decision::Accept);
}
