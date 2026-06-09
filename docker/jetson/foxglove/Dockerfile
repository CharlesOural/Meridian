# syntax=docker/dockerfile:1
#
# Minimal Foxglove bridge: exposes the local ROS 2 graph as a WebSocket on :8765
# so the Foxglove app (on the Mac) can preview topics. Temporary viz aid — safe to
# delete together with its compose service.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-foxglove-bridge \
 && rm -rf /var/lib/apt/lists/*

CMD ["bash", "-lc", "source /opt/ros/${ROS_DISTRO}/setup.bash && exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765"]
