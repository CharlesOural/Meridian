#!/usr/bin/env bash
# Headless Meridian eval run that captures EVERYTHING needed for diagnose_run.py:
# the odometry track (TUM), the event stream, stage timing, telemetry, the node
# log, and a manifest (config, rate, git SHA, bag, window) for reproducibility.
# Optionally runs the diagnosis at the end.
#
# Unlike run_bag_headless.sh (distrobox/~ paths), this is portable: it derives the
# workspace from its own location, so it works in the Docker container too.
#
# Usage (inside the dev container):
#   tools/run_eval.sh --config CFG.yaml --rate 0.25 --secs 485 --name myrun \
#       [--bag bags/legged_underground_ros2] [--gt GT.tum] [--out DIR] [--diagnose]
set +u
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

CFG=""; RATE=1.0; SECS=0; NAME="run"; BAG="$WS/bags/legged_underground_ros2"
GT=""; OUT="/tmp/meridian_eval"; DIAG=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CFG="$2"; shift 2;;
    --rate) RATE="$2"; shift 2;;
    --secs) SECS="$2"; shift 2;;     # wall-clock cap; 0 = whole bag
    --name) NAME="$2"; shift 2;;
    --bag) BAG="$2"; shift 2;;
    --gt) GT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --diagnose) DIAG=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -z "$CFG" ] && { echo "error: --config required" >&2; exit 2; }
NODE="$WS/install/meridian_ros/lib/meridian_ros/odometry_node"
mkdir -p "$OUT"

if pgrep -f 'lib/meridian_ros/odometry_node' >/dev/null; then
  echo "FATAL: a stale odometry_node is alive; kill it first" >&2
  pgrep -fa 'lib/meridian_ros/odometry_node' >&2; exit 1
fi

# Manifest: everything needed to reproduce and compare this run.
SHA="$(git -C "$WS" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$OUT/${NAME}_manifest.json" <<EOF
{ "name": "$NAME", "config": "$CFG", "rate": $RATE, "secs": $SECS,
  "bag": "$BAG", "git_sha": "$SHA" }
EOF

"$NODE" --ros-args -p config_file:="$CFG" -p use_sim_time:=true \
  > "$OUT/${NAME}_node.log" 2>&1 &
NPID=$!
sleep 3
CAP=$([ "$SECS" -gt 0 ] && echo "$((SECS+25))" || echo 100000)
timeout --signal=INT "$CAP" python3 "$WS/tools/record_tum.py" "$OUT/${NAME}.tum" \
  > "$OUT/${NAME}_rec.log" 2>&1 &
TPID=$!
timeout "$CAP" ros2 topic echo --qos-depth 2000 /meridian/events       > "$OUT/${NAME}_events.txt" 2>&1 &
EPID=$!
timeout "$CAP" ros2 topic echo --qos-depth 2000 /meridian/stage_timing > "$OUT/${NAME}_stage.txt" 2>&1 &
SPID=$!
timeout "$CAP" ros2 topic echo --qos-depth 4000 /meridian/telemetry    > "$OUT/${NAME}_telemetry.txt" 2>&1 &
MPID=$!

if [ "$SECS" -gt 0 ]; then
  timeout --signal=INT "$SECS" ros2 bag play "$BAG" --clock --rate "$RATE" \
    > "$OUT/${NAME}_play.log" 2>&1
else
  ros2 bag play "$BAG" --clock --rate "$RATE" > "$OUT/${NAME}_play.log" 2>&1
fi
sleep 4
kill -INT $TPID $EPID $SPID $MPID 2>/dev/null; sleep 2
kill -INT $NPID 2>/dev/null
for i in $(seq 1 20); do kill -0 $NPID 2>/dev/null || break; sleep 0.5; done
kill -9 $NPID 2>/dev/null; pkill -9 -f 'lib/meridian_ros/odometry_node' 2>/dev/null
wait 2>/dev/null
echo "[$NAME] done: $(wc -l < "$OUT/${NAME}.tum" 2>/dev/null) poses -> $OUT"

if [ "$DIAG" -eq 1 ]; then
  python3 "$WS/tools/diagnose_run.py" --est "$OUT/${NAME}.tum" \
    ${GT:+--gt "$GT"} --events "$OUT/${NAME}_events.txt" \
    --stage-timing "$OUT/${NAME}_stage.txt" --telemetry "$OUT/${NAME}_telemetry.txt" \
    --name "$NAME" --out "$OUT"
  # Full front-end debug surface (12-panel figure; panels for off debug groups
  # render as "group off" so figures stay comparable across postures).
  python3 "$WS/tools/plot_frontend.py" --telemetry "$OUT/${NAME}_telemetry.txt" \
    --events "$OUT/${NAME}_events.txt" --stage-timing "$OUT/${NAME}_stage.txt" \
    --name "$NAME" --out "$OUT"
fi
