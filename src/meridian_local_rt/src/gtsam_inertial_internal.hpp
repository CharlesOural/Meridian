#pragma once

#include <gtsam/navigation/CombinedImuFactor.h>

#include <utility>

#include "meridian/local_rt/combined_preintegration.hpp"

namespace meridian::local_rt {

struct CombinedPreintegration::Impl final {
  explicit Impl(gtsam::PreintegratedCombinedMeasurements value) : pim(std::move(value)) {}

  gtsam::PreintegratedCombinedMeasurements pim;
};

}  // namespace meridian::local_rt
