# syntax=docker/dockerfile:1
#
# Standalone SBG Systems INS/IMU driver (sbg_ros2_driver). Talks to the unit over
# serial and publishes sensor_msgs/Imu (+ SBG-native topics) to ROS 2. CPU-only;
# native arm64 on the Jetson.
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

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
