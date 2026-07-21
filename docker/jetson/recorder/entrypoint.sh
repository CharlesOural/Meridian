#!/usr/bin/env bash
# Source the ROS distro and every message overlay the rig publishes, then exec the
# compose/run command. Sourcing sbg after ouster is order-independent: the overlays
# are disjoint message sets, not competing versions of one package.
set -e
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source /opt/ouster_ws/install/setup.bash
source /opt/sbg_ws/install/setup.bash
exec "$@"
