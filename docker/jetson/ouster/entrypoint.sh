#!/usr/bin/env bash
# Source the ROS distro and the ouster-ros overlay, then exec the compose command.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/ouster_ws/install/setup.bash
exec "$@"
