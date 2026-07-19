# syntax=docker/dockerfile:1
#
# Standalone Ouster OS-series LiDAR driver. Builds ouster-ros (with its ouster-sdk
# submodule) from source into /opt/ouster_ws and publishes PointCloud2 + IMU to
# ROS 2. CPU-only, no CUDA; native arm64 on the Jetson.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

ARG OUSTER_ROS_REF=ros2

RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential cmake python3-rosdep \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ouster_ws/src \
 && git clone --recurse-submodules -b ${OUSTER_ROS_REF} \
      https://github.com/ouster-lidar/ouster-ros.git /opt/ouster_ws/src/ouster-ros

# rosdep resolves the driver + SDK system deps (jsoncpp, eigen, spdlog, curl,
# libtins, pcap, pcl) from the cloned package.xml metadata.
RUN apt-get update \
 && (rosdep init 2>/dev/null || true) \
 && rosdep update --rosdistro ${ROS_DISTRO} \
 && rosdep install --from-paths /opt/ouster_ws/src --ignore-src -y \
 && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
 && cd /opt/ouster_ws \
 && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && rm -rf build log

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
