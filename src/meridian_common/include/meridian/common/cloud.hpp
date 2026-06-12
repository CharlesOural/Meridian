#pragma once

#include <memory>
#include <vector>

#include "meridian/common/point.hpp"

namespace meridian {

// The cloud currency: a contiguous vector of LiDAR returns. It is published behind a
// shared_ptr<const> so it is filled once and read concurrently by L2/L4/L5 without copy.
using PointCloud = std::vector<LidarPoint>;
using PointCloudPtr = std::shared_ptr<const PointCloud>;

}  // namespace meridian
