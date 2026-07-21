#pragma once

#include <rclcpp/node.hpp>

#include "meridian/local_rt/pipeline.hpp"

namespace meridian::apps {

// Declares the complete local_rt.* ROS parameter schema on node, validates the
// representation-level constraints, and translates it into the ROS-free local
// pipeline configuration. Dataset profiles belong in YAML, not in this loader.
[[nodiscard]] local_rt::LocalRtPipelineConfig loadLocalRtPipelineConfig(rclcpp::Node& node);

}  // namespace meridian::apps
