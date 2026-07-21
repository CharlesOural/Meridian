# syntax=docker/dockerfile:1
#
# Meridian development image.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ARG MERIDIAN_BUILD_JOBS=6
ARG SOPHUS_REF=""
ARG GTSAM_REF=""
ARG CERES_REF="2.1.0"
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
     && apt-get install -y --no-install-recommends \
     build-essential cmake ninja-build git wget curl unzip ca-certificates gnupg \
     pkg-config sudo gdb nano vim tmux less \
     clangd clang-tidy clang-format \
     python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip python3-venv \
     python3-numpy python3-yaml \
     ros-${ROS_DISTRO}-rosbag2 \
     ros-${ROS_DISTRO}-rosbag2-storage-default-plugins \
     ros-${ROS_DISTRO}-rosbag2-storage-mcap \
     ros-${ROS_DISTRO}-launch \
     ros-${ROS_DISTRO}-launch-ros \
     ros-${ROS_DISTRO}-sensor-msgs \
     ros-${ROS_DISTRO}-foxglove-bridge \
     ros-${ROS_DISTRO}-tf2-tools \
     ros-${ROS_DISTRO}-rqt-tf-tree \
     ros-${ROS_DISTRO}-diagnostic-updater \
     libeigen3-dev libyaml-cpp-dev libboost-all-dev libfmt-dev libtbb-dev \
     libceres-dev libgoogle-glog-dev libgflags-dev \
     libatlas-base-dev libsuitesparse-dev libmetis-dev libsqlite3-dev \
     && rm -rf /var/lib/apt/lists/*

ENV RERUN_VENV=/opt/rerun
RUN python3 -m venv --system-site-packages "${RERUN_VENV}" \
     && "${RERUN_VENV}/bin/pip" install --no-cache-dir --upgrade pip \
     && "${RERUN_VENV}/bin/pip" install --no-cache-dir cmake "rerun-sdk[dataplatform]"
ENV PATH="${RERUN_VENV}/bin:${PATH}"

# Default ceres Ubuntu installation does not work, need >=2.1
RUN git clone https://github.com/ceres-solver/ceres-solver.git /tmp/ceres \
     && git -C /tmp/ceres checkout "${CERES_REF}" \
     && cmake -S /tmp/ceres -B /tmp/ceres/build -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DBUILD_TESTING=OFF \
     -DBUILD_EXAMPLES=OFF \
     -DBUILD_BENCHMARKS=OFF \
     -DUSE_CUDA=OFF \
     && cmake --build /tmp/ceres/build --target install --parallel "${MERIDIAN_BUILD_JOBS}" \
     && rm -rf /tmp/ceres \
     && ldconfig

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

# The latest Sophus checkout may require a newer configure-time CMake than
# Ubuntu 22.04 provides. After installing the native dependencies, return to
# the distro CMake for ROS/ament builds; the current Rerun SDK supports it and
# its vendored Arrow dependencies are not yet compatible with CMake 4.
RUN "${RERUN_VENV}/bin/pip" uninstall --yes cmake

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000
RUN (groupadd --gid ${USER_GID} ${USERNAME} || true) \
     && (useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash || true) \
     && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
     && chmod 0440 /etc/sudoers.d/${USERNAME} \
     && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc \
     && echo '[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' >> /home/${USERNAME}/.bashrc

WORKDIR /workspace
EXPOSE 8765
CMD ["bash"]
