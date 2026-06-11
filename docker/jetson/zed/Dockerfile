# syntax=docker/dockerfile:1
#
# ZED 2i as a plain UVC stereo camera. No ZED SDK is used: Stereolabs ships no SDK
# build for this board's L4T (r39.2 / JetPack 7.2), and Meridian's visual front-end
# consumes raw images anyway. v4l2_camera publishes the side-by-side stereo frame;
# image_transport plugins add a JPEG stream so Foxglove renders it without a YUV
# decoder. v4l-utils is kept for inspecting the device. CPU-only; native arm64.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-v4l2-camera \
      ros-${ROS_DISTRO}-image-transport-plugins \
      v4l-utils \
 && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
