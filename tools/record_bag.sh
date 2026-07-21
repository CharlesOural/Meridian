#!/usr/bin/env bash
# Record the live Jetson sensor graph into a ROS 2 bag under bags/ (gitignored).
#
# Runs `ros2 bag record` inside meridian-recorder:humble -- the only image carrying every
# message type on the rig -- on the host ROS graph (host networking + shared /dev/shm), so
# the 6.3 MB clouds move over FastDDS shared memory rather than loopback UDP. Ctrl-C stops
# and finalizes the bag, then each topic's recorded count is checked against its expected
# rate, so loss is caught here rather than months later in analysis.
#
#   tools/record_bag.sh                    # LIO set -> bags/rig_<timestamp>
#   tools/record_bag.sh full campus        # + camera/GNSS/RTK -> bags/campus_<timestamp>
#   tools/record_bag.sh lio calib          # LIO set, named calib
#   TOPICS="/ouster/points /imu/data" tools/record_bag.sh   # explicit topic list
#   BAGS_DIR=/mnt/usb/bags tools/record_bag.sh              # record elsewhere
#
# Storage is MCAP, uncompressed: the SSD sustains ~2.3 GB/s against a ~55 MB/s (lio) or
# ~190 MB/s (full) stream, so compression would only spend CPU the drivers need and risk
# backpressure into the DDS receive path.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to the 2 TB NVMe explicitly rather than a repo-relative bags/: a second checkout
# lives on the 57 GB eMMC, where the full set (~700 GB/h) would fill the root filesystem in
# minutes. Falls back to the repo only if the SSD is not mounted.
SSD_BAGS="/media/agx/ssd/bags"
if [ -z "${BAGS_DIR:-}" ]; then
  if mountpoint -q /media/agx/ssd; then BAGS_DIR="$SSD_BAGS"; else
    echo "WARNING: /media/agx/ssd not mounted -- falling back to $HERE/bags"
    BAGS_DIR="$HERE/bags"
  fi
fi
IMAGE="${RECORD_IMAGE:-meridian-recorder:humble}"
DOMAIN="${ROS_DOMAIN_ID:-0}"

# The same FastDDS profile the drivers load. The recorder is just another participant: if
# it keeps the default 536 KB shared-memory segment while the drivers use the tuned one,
# the mismatch drops messages on both sides -- the bag still looks valid, just short.
DDS_PROFILE="$HERE/docker/jetson/dds/fastdds_large.xml"
[ -f "$DDS_PROFILE" ] || { echo "ERROR: DDS profile missing: $DDS_PROFILE"; exit 1; }

# Core LIO: what the odometry node consumes, plus the sensor intrinsics.
# /ouster/metadata carries the beam angle intrinsics and is latched (transient_local);
# without it the cloud cannot be reprojected, so it is never optional.
TOPICS_LIO="/ouster/points /ouster/imu /ouster/metadata /imu/data /tf /tf_static"

# Full rig: adds camera, GNSS/RTK and the raw SBG frames. The sbg/* frames carry the
# per-solution status and covariances that /imu/data flattens away, and /ntrip_client/rtcm
# records the corrections themselves -- together they let RTK quality be reconstructed
# offline as a ground-truth confidence signal.
TOPICS_FULL="${TOPICS_LIO} \
/zed/image_raw /zed/camera_info \
/imu/nav_sat_fix /imu/mag /imu/utc_ref \
/sbg/gps_pos /sbg/gps_vel /sbg/ekf_quat /sbg/status /sbg/utc_time /sbg/imu_data \
/ntrip_client/rtcm /ntrip_client/nmea /diagnostics"

case "${1:-lio}" in
  lio)  SELECTED="$TOPICS_LIO";  shift ;;
  full) SELECTED="$TOPICS_FULL"; shift ;;
  *)    SELECTED="$TOPICS_LIO"        ;;   # no set given: first arg is the name
esac
NAME="${1:-rig}"
shift || true
TOPICS="${TOPICS:-$SELECTED}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${NAME}_${STAMP}"
mkdir -p "$BAGS_DIR"

# Fail before recording rather than halfway through: a bag truncated by ENOSPC is not
# salvageable, and the full set burns ~700 GB/h.
AVAIL_GB=$(df -BG --output=avail "$BAGS_DIR" | tail -1 | tr -dc '0-9')
echo "Recording -> $BAGS_DIR/$OUT"
echo "Topics    : $(echo $TOPICS | wc -w) topics"
echo "Free space: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 20 ]; then echo "ERROR: <20 GB free, refusing to start"; exit 1; fi
echo "(Ctrl-C to stop and finalize)"

# Runs as root, deliberately. The drivers run as root and FastDDS creates its shared
# memory segments in /dev/shm mode 0644 root-owned, so a non-root participant cannot open
# them: the SHM transport then fails and every topic from a root-owned driver silently
# records zero messages while the bag still looks valid. The bag is chowned back below.
#
# --max-cache-size 512MiB (default 100 MiB): the writer flushes from a cache the
# subscriptions fill. At 6.3 MB/message a 100 MiB cache holds ~16 clouds, so one SSD
# write stall spills straight into dropped messages; 512 MiB absorbs a multi-second stall.
# -t only when there is a terminal: docker refuses to allocate a TTY otherwise, which
# would break the script under a non-interactive SSH or from another script. -i is kept
# either way so Ctrl-C reaches rosbag2 and the bag is finalized rather than truncated.
TTY_FLAG="-i"
[ -t 0 ] && TTY_FLAG="-it"

docker run --rm $TTY_FLAG \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID="$DOMAIN" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds.xml \
  -v "$DDS_PROFILE":/dds.xml:ro \
  -v "$BAGS_DIR":/bags \
  -w /bags \
  "$IMAGE" \
  ros2 bag record \
    --storage mcap \
    --max-cache-size 536870912 \
    -o "$OUT" \
    $TOPICS || true      # rosbag2 exits non-zero on SIGINT; the bag is still finalized

# The bag is written by root inside the container; hand it back to the invoking user.
docker run --rm -v "$BAGS_DIR":/bags "$IMAGE" \
  chown -R "$(id -u):$(id -g)" "/bags/$OUT"

echo
echo "=== Verifying $OUT ==="
docker run --rm \
  -e HOME=/tmp \
  -v "$BAGS_DIR":/bags \
  -w /bags \
  "$IMAGE" \
  ros2 bag info "$OUT" 2>&1 | tee "$BAGS_DIR/$OUT/bag_info.txt"

python3 "$HERE/tools/check_bag.py" "$BAGS_DIR/$OUT/bag_info.txt" $TOPICS || true
