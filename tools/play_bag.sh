#!/usr/bin/env bash
# Replay a recorded bag onto the local ROS graph for inspection (Foxglove or RViz).
#
# The mirror of tools/record_bag.sh: same image, same FastDDS profile, so the replayed
# 6.3 MB clouds move over shared memory exactly as the live ones do.
#
#   tools/play_bag.sh                       # newest bag under the SSD bags dir
#   tools/play_bag.sh campus_20260717_1503  # a named bag (or an absolute path)
#   RATE=0.5 tools/play_bag.sh              # half speed
#   LOOP=0 tools/play_bag.sh                # play once and exit (default loops)
#
# Refuses to start while the live drivers are up: they publish the same topic names, so a
# replay alongside them interleaves recorded and live messages on one topic and what you
# end up inspecting is neither.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAGS_DIR="${BAGS_DIR:-/media/agx/ssd/bags}"
[ -d "$BAGS_DIR" ] || BAGS_DIR="$HERE/bags"
IMAGE="${RECORD_IMAGE:-meridian-recorder:humble}"
DOMAIN="${ROS_DOMAIN_ID:-0}"
RATE="${RATE:-1.0}"
LOOP="${LOOP:-1}"

DDS_PROFILE="$HERE/docker/jetson/dds/fastdds_large.xml"
[ -f "$DDS_PROFILE" ] || { echo "ERROR: DDS profile missing: $DDS_PROFILE"; exit 1; }

LIVE=$(docker ps --filter name=meridian-ouster --filter name=meridian-sbg \
       --filter name=meridian-zed --format '{{.Names}}' | tr '\n' ' ')
if [ -n "$LIVE" ]; then
  echo "ERROR: live drivers are running: $LIVE"
  echo "They publish the same topics as the bag. Stop them first:"
  echo "  docker compose -f compose.jetson.yaml --profile camera stop ouster sbg zed"
  exit 1
fi

BAG="${1:-}"
if [ -z "$BAG" ]; then
  BAG=$(ls -dt "$BAGS_DIR"/*/ 2>/dev/null | head -1 || true)
  [ -n "$BAG" ] || { echo "ERROR: no bags under $BAGS_DIR"; exit 1; }
  echo "No bag given; using newest: $(basename "$BAG")"
fi
case "$BAG" in /*) ;; *) BAG="$BAGS_DIR/$BAG" ;; esac
[ -d "$BAG" ] || { echo "ERROR: no such bag: $BAG"; exit 1; }

# --clock so a consumer set to use_sim_time follows the bag's timeline rather than wall
# clock. Anything reading stamps off the messages themselves is unaffected either way.
ARGS=(--clock --rate "$RATE")
[ "$LOOP" = "1" ] && ARGS+=(--loop)

echo "Playing   : $(basename "$BAG")  (rate ${RATE}x, loop=${LOOP})"
echo "Foxglove  : ws://$(hostname -I | awk '{print $1}'):8765   (bridge must be up)"
echo "(Ctrl-C to stop)"

TTY_FLAG="-i"
[ -t 0 ] && TTY_FLAG="-it"

exec docker run --rm $TTY_FLAG \
  --network host \
  --ipc host \
  -e ROS_DOMAIN_ID="$DOMAIN" \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds.xml \
  -v "$DDS_PROFILE":/dds.xml:ro \
  -v "$(dirname "$BAG")":/bags \
  -w /bags \
  "$IMAGE" \
  ros2 bag play "/bags/$(basename "$BAG")" "${ARGS[@]}"
