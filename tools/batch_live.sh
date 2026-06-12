#!/usr/bin/env bash
# Characterize the nondeterministic live (--realtime) path: run a config N times on
# the same bag and analyze each, so the failure-mode DISTRIBUTION (clean / uniformly
# degraded / diverged) and any shared signature become visible. One run proves
# nothing on a nondeterministic path; the distribution does.
#
# Usage: batch_live.sh <config.yaml> <bag> <gt.tum> <N> <outdir> [extra replay args...]
# Run from the workspace root, inside the container (needs replay_runner + python3/numpy).
set -u
CFG=$1; BAG=$2; GT=$3; N=$4; OUT=$5; shift 5; EXTRA="${*:-}"
RR=./install/meridian_ros/lib/meridian_ros/replay_runner
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"
for i in $(seq 1 "$N"); do
  RUN="$OUT/run$i"; rm -rf "$RUN"; mkdir -p "$RUN"
  $RR "$CFG" "$BAG" "$RUN/run.tum" 0 --realtime 1 $EXTRA > "$RUN/run.log" 2>&1
  L=$(grep -c loop_accepted "$RUN/run.log")
  V=$(python3 tools/analyze_run.py "$RUN" "$GT" 2>&1 | tail -1)
  echo "run$i loops=$L | $V" | tee -a "$SUMMARY"
done
echo "=== DISTRIBUTION ===" | tee -a "$SUMMARY"
python3 - "$SUMMARY" <<'EOF' | tee -a "$SUMMARY"
import sys, re, statistics as st
clean = unif = div = 0; rmses = []
for line in open(sys.argv[1]):
    m = re.search(r"RMSE=([\d.]+)m", line)
    if m: rmses.append(float(m.group(1)))
    if "CLEAN" in line: clean += 1
    elif "UNIFORMLY" in line: unif += 1
    elif "onset" in line: div += 1
print(f"runs={len(rmses)}  clean={clean}  uniform-degraded={unif}  diverged={div}")
if rmses:
    print(f"RMSE  median={st.median(rmses):.2f}m  min={min(rmses):.2f}m  max={max(rmses):.2f}m")
EOF
echo "done -> $SUMMARY"
