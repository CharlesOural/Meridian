#include "meridian/debug/telemetry_keys.hpp"

#include <set>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using meridian::keys::catalog;
using meridian::keys::unit_string;

// Every wire key is unique: a duplicated id means two channels collide on one topic.
TEST(TelemetryKeys, CatalogHasNoDuplicateIds) {
  std::set<std::string_view> seen;
  for (const auto& k : catalog()) {
    ASSERT_NE(k.id, nullptr);
    EXPECT_TRUE(seen.insert(k.view()).second) << "duplicate key id: " << k.id;
  }
  EXPECT_FALSE(catalog().empty());
}

// unit_string resolves through the catalog (exact match) and returns the declared unit.
TEST(TelemetryKeys, UnitStringMatchesCatalog) {
  for (const auto& k : catalog()) {
    EXPECT_STREQ(unit_string(k.view()), unit_string(k.id))
        << "view()/id disagree for " << k.id;
  }
  EXPECT_STREQ(unit_string("frontend/solve_ms"), "ms");
  EXPECT_STREQ(unit_string("backend/n_factors"), "count");
  EXPECT_STREQ(unit_string("frontend/observability"), "ratio");
  EXPECT_STREQ(unit_string("frontend/gnss/innovation_m"), "m");
}

// Dynamic-suffix keys resolve their unit through the registered prefix.
TEST(TelemetryKeys, DynamicPrefixResolvesUnit) {
  EXPECT_STREQ(unit_string("backend/observability/42"), "ratio");
  EXPECT_STREQ(unit_string("sensors/rate/imu"), "hz");
  EXPECT_STREQ(unit_string("sensors/validator/lidar/range_count"), "count");
}

// An unknown key is unit-less, never a wrong guess.
TEST(TelemetryKeys, UnknownKeyHasNoUnit) {
  EXPECT_STREQ(unit_string("not/a/real/key"), "");
  EXPECT_STREQ(unit_string(""), "");
}

}  // namespace
