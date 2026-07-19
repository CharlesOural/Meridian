#!/usr/bin/env bash
#
# Build the Meridian GPU image (docker/Dockerfile.gpu) and create a
# Distrobox from it.
#   ./setup-distrobox.sh
#
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="meridian:humble-gpu"
BOX="meridian"

export DBX_CONTAINER_MANAGER=docker

echo "==> Building $IMAGE (Meridian dep canon + CUDA 12 toolkit) from docker/Dockerfile.gpu"
echo "    (first build is slow: GTSAM/Sophus compile from source)"
docker build -t "$IMAGE" \
  -f docker/Dockerfile.gpu \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  docker/

# A rootful (docker) container makes distrobox stage a first-shell password setup,
# gated on /var/tmp/.<user>.passwd.initialize. Its passwd loops forever when the
# host's password policy rejects a short password, so clear the sentinel on every
# container start — the init hook runs after distrobox writes it, before login.
echo "==> Creating distrobox '$BOX' (NVIDIA GPU + home dir + GUI auto-wired)"
distrobox create --name "$BOX" --image "$IMAGE" --nvidia --yes \
  --init-hooks 'rm -f /var/tmp/.*.passwd.initialize'
