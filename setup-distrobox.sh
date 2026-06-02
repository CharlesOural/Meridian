#!/usr/bin/env bash
#
# Build the reproducible Meridian GPU image (docker/Dockerfile.gpu) and create a
# Distrobox from it.
#   ./setup-distrobox.sh
#
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="meridian:humble-gpu"
BOX="meridian"

echo "==> Building $IMAGE (Meridian dep canon + CUDA 12 toolkit) from docker/Dockerfile.gpu"
echo "    (first build is slow: GTSAM/Ceres/Sophus/small_gicp compile from source)"
docker build -t "$IMAGE" \
  -f docker/Dockerfile.gpu \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  docker/

echo "==> Creating distrobox '$BOX' (NVIDIA GPU + home dir + GUI auto-wired)"
distrobox create --name "$BOX" --image "$IMAGE" --nvidia --yes