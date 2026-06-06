#!/usr/bin/env bash
# Headless Meridian bag run: odometry_node (no rviz) + bag play, recording the
# trajectory (TUM), telemetry, and events for drift/divergence analysis.
#
# Hygiene: `ros2 run` wraps the real binary, so killing only its PID orphans the
# node; every kill below targets the odometry_node binary itself and verifies
# death. A pre-flight check refuses to run while any stale node is alive —
# ghosts subscribe to the same topics and publish to the same outputs, silently
# poisoning the recording AND starving the CPU (drops at best-effort QoS).
source /opt/ros/humble/setup.bash
cd ~/Meridian
source install/setup.bash

if pgrep -f 'lib/meridian_ros/odometry_node' > /dev/null; then
  echo "FATAL: stale odometry_node process(es) alive:" >&2
  pgrep -fa 'lib/meridian_ros/odometry_node' >&2
  echo "kill them first: pkill -f lib/meridian_ros/odometry_node" >&2
  exit 1
fi

OUT=${1:-/tmp/meridian_bagrun}
mkdir -p "$OUT"

BAG=bags/canteen_day_fixed
echo "bag: $BAG" | tee "$OUT/run_info.txt"

# Exec the binary directly (no ros2-run wrapper) so $NODE_PID is the node itself.
"$HOME/Meridian/install/meridian_ros/lib/meridian_ros/odometry_node" --ros-args \
  -p config_file:=$PWD/src/meridian_ros/config/fusionportable.yaml \
  -p use_sim_time:=true \
  > "$OUT/node.log" 2>&1 &
NODE_PID=$!
sleep 3

python3 tools/record_tum.py "$OUT/traj_tum.txt" > "$OUT/record_tum.log" 2>&1 &
TUM_PID=$!
ros2 topic echo /meridian/events > "$OUT/events.yaml" 2>&1 &
EV_PID=$!
ros2 topic echo /meridian/stage_timing > "$OUT/timing.yaml" 2>&1 &
TM_PID=$!
ros2 topic echo /meridian/telemetry > "$OUT/telemetry.yaml" 2>&1 &
TL_PID=$!

ros2 bag play "$BAG" --clock > "$OUT/bagplay.log" 2>&1
sleep 3

kill $TUM_PID $EV_PID $TM_PID $TL_PID 2>/dev/null
sleep 1
kill -INT $NODE_PID 2>/dev/null
for i in $(seq 1 20); do
  kill -0 $NODE_PID 2>/dev/null || break
  sleep 0.5
done
kill -9 $NODE_PID 2>/dev/null
wait 2>/dev/null

if pgrep -f 'lib/meridian_ros/odometry_node' > /dev/null; then
  echo "WARNING: an odometry_node survived shutdown — kill it before the next run" >&2
fi

echo "--- artifacts ---"
wc -l "$OUT/traj_tum.txt" "$OUT/events.yaml" 2>/dev/null
echo "done"
