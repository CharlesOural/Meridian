#!/usr/bin/env bash
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/ouster_ws/install/setup.bash
source /opt/sbg_ws/install/setup.bash
exec "$@"
