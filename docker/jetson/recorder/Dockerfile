# syntax=docker/dockerfile:1
#
# Dataset recorder: `ros2 bag record` with every message type the rig publishes.
#
# rosbag2 creates a generic subscription per topic, which needs the type's support
# library resolvable at runtime -- a topic whose type cannot be loaded is skipped with
# a warning, so a recorder missing sbg_driver would silently drop the whole /sbg/*
# namespace from the dataset. This image therefore carries the union of the rig's
# message packages: ouster (from the base), sbg (copied), and the NTRIP types.
ARG ROS_DISTRO=humble

FROM meridian-sbg:humble AS sbg

# The Ouster driver image already provides ouster_sensor_msgs (/ouster/telemetry) in
# its overlay; building on it avoids rebuilding the SDK just to get the definitions.
FROM meridian-ouster:humble

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

# rosbag2-storage-mcap: MCAP is the recording format. The default sqlite3 plugin wraps
# every message in a transaction against a single-writer B-tree, which stalls on the
# 6.3 MB/scan cloud stream; MCAP is an append-only chunked log built for this rate and
# is self-describing, so the bag stays readable without this workspace's message
# packages installed.
# nmea/rtcm-msgs: the NTRIP correction stream and the GGA sent back to the caster --
# needed to reconstruct why RTK was (or was not) fixed at any point in the dataset.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-rosbag2-storage-mcap \
      ros-${ROS_DISTRO}-nmea-msgs \
      ros-${ROS_DISTRO}-rtcm-msgs \
 && rm -rf /var/lib/apt/lists/*

COPY --from=sbg /opt/sbg_ws/install /opt/sbg_ws/install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
