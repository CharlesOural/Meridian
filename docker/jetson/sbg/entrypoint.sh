#!/usr/bin/env bash
# Source the ROS distro and the sbg_ros2_driver overlay, then exec the command.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/sbg_ws/install/setup.bash
exec "$@"
