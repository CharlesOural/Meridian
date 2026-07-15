# syntax=docker/dockerfile:1
#
# Minimal Foxglove bridge: exposes the Jetson ROS 2 graph as a WebSocket on
# port 8765 for a viewer running on a remote workstation.
ARG ROS_DISTRO=humble

# The bridge sends each topic's schema to the client, so it must be able to resolve every
# message type on the graph -- without sbg_driver's definitions the whole /sbg/* namespace
# fails to load and is invisible in the app. Only the built messages are copied, not the
# driver or its serial dependencies.
FROM meridian-sbg:humble AS sbg

FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# nmea_msgs/rtcm_msgs carry the NTRIP correction stream and the GGA sent back to the
# caster; without their definitions those topics cannot be inspected to confirm RTK
# corrections are actually arriving.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-foxglove-bridge \
      ros-${ROS_DISTRO}-nmea-msgs \
      ros-${ROS_DISTRO}-rtcm-msgs \
 && rm -rf /var/lib/apt/lists/*

COPY --from=sbg /opt/sbg_ws/install /opt/sbg_ws/install

CMD ["bash", "-lc", "source /opt/ros/${ROS_DISTRO}/setup.bash && source /opt/sbg_ws/install/setup.bash && exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765"]
