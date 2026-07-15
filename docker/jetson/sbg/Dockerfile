# syntax=docker/dockerfile:1
#
# Standalone SBG Systems INS/IMU driver (sbg_ros2_driver). Talks to the unit over
# serial and publishes sensor_msgs/Imu (+ SBG-native topics) to ROS 2. CPU-only;
# native arm64 on the Jetson.
#
# Also carries the NTRIP client that feeds RTCM corrections to the unit for RTK, and
# sbg_probe, a read-only dump of the device's stored configuration (see sbg_config.yaml
# for why reading before writing matters here).
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

ARG SBG_ROS_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake python3-rosdep \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/sbg_ws/src \
 && git clone --recurse-submodules -b ${SBG_ROS_REF} \
      https://github.com/SBG-Systems/sbg_ros2_driver.git /opt/sbg_ws/src/sbg_ros2_driver

RUN apt-get update \
 && (rosdep init 2>/dev/null || true) \
 && rosdep update --rosdistro ${ROS_DISTRO} \
 && rosdep install --from-paths /opt/sbg_ws/src --ignore-src -y \
 && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && cd /opt/sbg_ws \
 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && rm -rf build log

# NTRIP client: pulls RTCM from a caster and republishes it as rtcm_msgs/Message, which
# the sbg driver forwards to the unit. rtcm_msgs comes in via the driver's rosdep.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ntrip-client \
 && rm -rf /var/lib/apt/lists/*

# Read-only config dump; links the sbgECom the driver already vendors and built.
COPY probe/sbg_probe.c /tmp/sbg_probe.c
RUN SBG=/opt/sbg_ws/src/sbg_ros2_driver/external/sbgECom \
 && gcc /tmp/sbg_probe.c -o /usr/local/bin/sbg_probe \
      -I$SBG/src -I$SBG/common -I$SBG/common/include \
      -L/opt/sbg_ws/install/sbg_driver/lib -lsbgECom -lm -pthread \
 && rm /tmp/sbg_probe.c

COPY rtk_monitor.py /rtk_monitor.py
RUN chmod +x /rtk_monitor.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
