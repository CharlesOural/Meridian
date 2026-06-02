#!/usr/bin/env bash
#
# Shared dependency installer for the Meridian dev image
#
# What it installs:
#   - apt:    build tooling, ROS extras, and the apt-provided libs
#             (Eigen 3.4, PCL 1.12, OpenCV 4, yaml-cpp, Boost, glog/gflags,
#              SuiteSparse, METIS, linuxptp)
#   - source: Sophus, Ceres 2.1 (Manifold API), GTSAM 4.2, small_gicp


set -euo pipefail

# --- pinned versions --------------------------
SOPHUS_TAG="${SOPHUS_TAG:-1.22.10}"
CERES_TAG="${CERES_TAG:-2.1.0}"
GTSAM_TAG="${GTSAM_TAG:-4.2.0}"
# TODO: pin small_gicp to a commit SHA for full reproducibility.
SMALL_GICP_REF="${SMALL_GICP_REF:-master}"

export DEBIAN_FRONTEND=noninteractive

# --- build parallelism --------------------------------------------------------
# Number of parallel compile jobs, as the first argument (or MERIDIAN_BUILD_JOBS,
# or the default). Ninja otherwise fans out to every core; Ceres/GTSAM units are
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
  libpcl-dev \
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

# --- Sophus (Lie groups; SO3/SE3 the splines store as control points) --------
git clone https://github.com/strasdat/Sophus.git /tmp/Sophus
git -C /tmp/Sophus checkout "${SOPHUS_TAG}"
cmake -S /tmp/Sophus -B /tmp/Sophus/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF \
  -DSOPHUS_USE_BASIC_LOGGING=ON
cmake --build /tmp/Sophus/build --target install --parallel "${JOBS}"
rm -rf /tmp/Sophus

# --- Ceres 2.1 (CT-window NLLS solver; needs the Manifold API → ≥2.1) --------
git clone https://github.com/ceres-solver/ceres-solver.git /tmp/ceres
git -C /tmp/ceres checkout "${CERES_TAG}"
cmake -S /tmp/ceres -B /tmp/ceres/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build /tmp/ceres/build --target install --parallel "${JOBS}"
rm -rf /tmp/ceres

# --- GTSAM 4.2 (iSAM2 back-end) ---------------------
# system Eigen (one Eigen for the whole tree), no -march=native (portable image),
# TBB off
git clone https://github.com/borglab/gtsam.git /tmp/gtsam
git -C /tmp/gtsam checkout "${GTSAM_TAG}"
cmake -S /tmp/gtsam -B /tmp/gtsam/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGTSAM_USE_SYSTEM_EIGEN=ON \
  -DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF \
  -DGTSAM_WITH_TBB=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_UNSTABLE=OFF
cmake --build /tmp/gtsam/build --target install --parallel "${JOBS}"
rm -rf /tmp/gtsam

# --- small_gicp (loop-closure GICP verify) -------------------------------
git clone https://github.com/koide3/small_gicp.git /tmp/small_gicp
git -C /tmp/small_gicp checkout "${SMALL_GICP_REF}"
cmake -S /tmp/small_gicp -B /tmp/small_gicp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build /tmp/small_gicp/build --target install --parallel "${JOBS}"
rm -rf /tmp/small_gicp

ldconfig
echo "==> Meridian dependency canon installed."
