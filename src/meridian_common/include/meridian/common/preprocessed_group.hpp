#pragma once

#include "meridian/common/measure_group.hpp"

namespace meridian {

// The L1->L2 currency: the assembled MeasureGroup. A wrapper rather than the group
// itself so future L1 products can ride along without an IFrontEnd signature change.
struct PreprocessedGroup {
  MeasureGroup group;
};

}  // namespace meridian
