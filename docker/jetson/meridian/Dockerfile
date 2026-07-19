# syntax=docker/dockerfile:1
#
# Meridian localization image for Jetson. Sensor drivers and Foxglove Bridge
# remain separate services in compose.jetson.yaml.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ARG MERIDIAN_BUILD_JOBS=6
ARG SOPHUS_REF=""
ARG GTSAM_REF=""
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates pkg-config \
      python3-colcon-common-extensions python3-rosdep \
      ros-${ROS_DISTRO}-diagnostic-updater \
      libeigen3-dev libyaml-cpp-dev libboost-all-dev libfmt-dev libtbb-dev \
      libceres-dev libgoogle-glog-dev libgflags-dev \
      libatlas-base-dev libsuitesparse-dev libmetis-dev libsqlite3-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/strasdat/Sophus.git /tmp/Sophus \
 && if [ -n "${SOPHUS_REF}" ]; then git -C /tmp/Sophus checkout "${SOPHUS_REF}"; fi \
 && cmake -S /tmp/Sophus -B /tmp/Sophus/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SOPHUS_TESTS=OFF \
      -DBUILD_SOPHUS_EXAMPLES=OFF \
      -DSOPHUS_USE_BASIC_LOGGING=ON \
 && cmake --build /tmp/Sophus/build --target install --parallel "${MERIDIAN_BUILD_JOBS}" \
 && rm -rf /tmp/Sophus

RUN git clone https://github.com/borglab/gtsam.git /tmp/gtsam \
 && if [ -n "${GTSAM_REF}" ]; then git -C /tmp/gtsam checkout "${GTSAM_REF}"; fi \
 && cmake -S /tmp/gtsam -B /tmp/gtsam/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGTSAM_USE_SYSTEM_EIGEN=ON \
      -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
      -DGTSAM_WITH_TBB=OFF \
      -DGTSAM_BUILD_TESTS=OFF \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
      -DGTSAM_BUILD_UNSTABLE=OFF \
 && cmake --build /tmp/gtsam/build --target install --parallel "${MERIDIAN_BUILD_JOBS}" \
 && rm -rf /tmp/gtsam \
 && ldconfig

ARG USERNAME=meridian
ARG USER_UID=1000
ARG USER_GID=1000
RUN (groupadd --gid ${USER_GID} ${USERNAME} || true) \
 && (useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash || true) \
 && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc \
 && echo '[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' >> /home/${USERNAME}/.bashrc

WORKDIR /workspace
CMD ["bash"]
