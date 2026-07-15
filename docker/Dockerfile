# syntax=docker/dockerfile:1
#
# Meridian dev image — CPU, CROSS-PLATFORM. The base is a multi-arch ROS image,
# so it builds natively on linux/amd64 (x86 dev laptop, CPU layers) AND
# linux/arm64 (Apple Silicon, and the Jetson's CPU layers) — no emulation. The
# dependency stack is the shared install-deps.sh; the GPU image (Dockerfile.gpu)
# runs the same script on a CUDA-capable base.
#
# Base is the `perception` variant (ROS base + OpenCV/image tooling), which
# publishes both amd64 and arm64. `desktop-full` is osrf-only and amd64-only, so
# it cannot build native on arm64; rviz2 (a runtime-only dep) is absent here, and
# viz goes through the Foxglove bridge instead.
#
# Covers CPU contract, estimator, global-graph, ROS adapter, and replay work.
# GPU registration and the future nvblox dense candidate require the GPU image.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-perception

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# Meridian dependency canon — one shared installer (see docs/DEVELOPMENT.md).
# MERIDIAN_BUILD_JOBS caps parallel compile jobs; lower it if the build OOMs.
ARG MERIDIAN_BUILD_JOBS=6
COPY install-deps.sh /tmp/install-deps.sh
RUN bash /tmp/install-deps.sh "${MERIDIAN_BUILD_JOBS}" && rm -f /tmp/install-deps.sh

# --- non-root dev user (compose / VS Code devcontainer use it) ------------------
# Distrobox ignores this and injects YOUR host user at `enter` time — expected.
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
# Default user stays root so Distrobox is happy; compose/devcontainer select `dev`.
CMD ["bash"]
