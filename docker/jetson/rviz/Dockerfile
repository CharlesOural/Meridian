# syntax=docker/dockerfile:1
#
# RViz for inspecting a replayed bag. Separate from the recorder because rviz2 drags in
# the whole desktop stack (~1 GB) that has no business in the recording path.
#
# Needs an X display: either the Jetson's own (DISPLAY=:0, when sitting at the machine)
# or an `ssh -X` forwarded one. Carries the rig's message packages so /sbg/* and
# /ouster/telemetry resolve; a type it cannot load is simply invisible in the panel.
ARG ROS_DISTRO=humble

FROM meridian-sbg:humble AS sbg
FROM meridian-ouster:humble

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-rviz2 \
      ros-${ROS_DISTRO}-rosbag2-storage-mcap \
      ros-${ROS_DISTRO}-nmea-msgs \
      ros-${ROS_DISTRO}-rtcm-msgs \
 && rm -rf /var/lib/apt/lists/*

COPY --from=sbg /opt/sbg_ws/install /opt/sbg_ws/install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "rviz2", "rviz2"]
