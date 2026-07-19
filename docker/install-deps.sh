#!/usr/bin/env bash
#
# Shared dependency installer for the Meridian dev image
#
# What it installs:
#   - apt:    build tooling, ROS extras, and the apt-provided libs
#             (Eigen 3.4, OpenCV 4, yaml-cpp, Boost, glog/gflags,
#              SuiteSparse, METIS, linuxptp)
#   - source: Sophus, GTSAM


set -euo pipefail

# Source dependencies follow each upstream repository's default branch. A
# caller may still set an explicit ref to reproduce or investigate a build;
# Meridian does not impose one as a repository policy.
SOPHUS_REF="${SOPHUS_REF:-}"
GTSAM_REF="${GTSAM_REF:-}"

export DEBIAN_FRONTEND=noninteractive

# --- build parallelism --------------------------------------------------------
# Number of parallel compile jobs, as the first argument (or MERIDIAN_BUILD_JOBS,
# or the default). Ninja otherwise fans out to every core; GTSAM units are
# Eigen-template-heavy and each peak well over 1.5 GB, so on a low-RAM host the
# default exhausts memory and swaps. Lower this if the build freezes the machine.
JOBS="${1:-${MERIDIAN_BUILD_JOBS:-6}}"
echo "==> Building third-party deps with ${JOBS} parallel jobs."

# --- apt: ROS extras, build tooling, and the apt-provided dep canon ----------
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git wget curl ca-certificates gnupg \
  pkg-config sudo gdb nano vim tmux less \
  clangd clang-tidy clang-format \
  python3-colcon-common-extensions python3-rosdep python3-vcstool python3-pip \
  "ros-${ROS_DISTRO}-foxglove-bridge" \
  "ros-${ROS_DISTRO}-tf2-tools" \
  "ros-${ROS_DISTRO}-rqt-tf-tree" \
  "ros-${ROS_DISTRO}-diagnostic-updater" \
  libeigen3-dev \
  libopencv-dev \
  libyaml-cpp-dev \
  libboost-all-dev \
  libfmt-dev \
  libtbb-dev \
  libgoogle-glog-dev libgflags-dev \
  libatlas-base-dev libsuitesparse-dev \
  libmetis-dev \
  libsqlite3-dev \
  linuxptp
rm -rf /var/lib/apt/lists/*

# --- evo (offline ATE/RPE eval — pip, offline tooling only ---------
pip3 install --no-cache-dir evo

# --- Sophus (core SO(3)/SE(3) representation) --------------------------------
git clone https://github.com/strasdat/Sophus.git /tmp/Sophus
if [[ -n "${SOPHUS_REF}" ]]; then
  git -C /tmp/Sophus checkout "${SOPHUS_REF}"
fi
cmake -S /tmp/Sophus -B /tmp/Sophus/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF \
  -DSOPHUS_USE_BASIC_LOGGING=ON
cmake --build /tmp/Sophus/build --target install --parallel "${JOBS}"
rm -rf /tmp/Sophus

# --- GTSAM (private local/global graph adapters) -----------------------------
# system Eigen (one Eigen for the whole tree), no -march=native (portable image),
# TBB off. Meridian requires GTSAM >= 4.2.1, the first release with the ISAM2
# marginal-factor slot-reuse fix. The package configure checks that minimum;
# the source ref remains unpinned.
git clone https://github.com/borglab/gtsam.git /tmp/gtsam
if [[ -n "${GTSAM_REF}" ]]; then
  git -C /tmp/gtsam checkout "${GTSAM_REF}"
fi
cmake -S /tmp/gtsam -B /tmp/gtsam/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=ON
cmake --build /tmp/gtsam/build --target install --parallel "${JOBS}"
rm -rf /tmp/gtsam

ldconfig
echo "==> Meridian dependency canon installed."
