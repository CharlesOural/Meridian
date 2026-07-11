# Real-time debugging (front-end, on target)

Field guide for "the trajectory looks wrong on the Jetson." Specs own the design;
this is triage.

## Tooling (run this before interpreting anything)

```bash
# Offline replay — THE way to evaluate accuracy / run A/Bs.
# Reads the bag directly (no ROS transport), pipeline in Replay mode: lossless,
# parallel-safe.
install/meridian_ros/lib/meridian_ros/replay_runner \
    <cfg.yaml> <bag_dir> out.tum [max_content_secs]

# Live-path run (nondeterministic: lossy transport + arrival timing; use for
# integration/viz, NOT for accuracy conclusions) -> full report + figure:
tools/run_eval.sh --config <cfg.yaml> --rate 0.25 --secs 485 --name myrun \
    --gt <GT.tum> --diagnose

# Re-diagnose existing artifacts without re-running:
tools/diagnose_run.py --est myrun.tum --gt GT.tum --events myrun_events.txt \
    --stage-timing myrun_stage.txt --name myrun

# A/B several runs at a glance (ATE / onset / phantom table + overlay plot):
tools/compare_runs.py --gt GT.tum a=a.tum b=b.tum c=c.tum
```

Always look at `diagnose_run`'s sections **in order**: a wrong trajectory is
useless to interpret until [1] confirms the GT is trustworthy (it flags an
identity-quaternion GT and disables heading metrics) and [2] says what the
platform physically did (stationary / in-place-turn / forward). The
**phantom-motion** metric (estimate translating while the platform is still)
catches rotation-as-translation failures that ATE alone hides. Plot first.

Other tools: `tools/run_bag_headless.sh` (legacy distrobox runner),
`tools/eval_ate.py`.

## The failure mode you'll hit: overload, not divergence

The front-end must finish each sweep in **~100 ms** (10 Hz). When it can't,
`Q_meas` evicts whole sweeps → gaps in the measurement stream → past
`lio.max_gap_s` the front-end bridges on constant velocity and arms a reseed →
the estimate **coasts while the platform walks away**, and if the bridge lands
off the map, the map is cleared and the keyframe chain re-anchors. Looks like
drift/explosion; it's actually starvation. Fix the budget, not the estimator.

## Preflight — before trusting any number

- No ghost `odometry_node`: `pgrep -x odometry_node` must be empty. Run via
  `tools/run_bag_headless.sh`, which execs the binary and verifies death.
- Quiet box: `ps aux --sort=-%cpu | head`, load average ≲ 1. Background load
  starves the *input intake* (player / executor / preprocess thread) — no
  front-end change can fix it. The tell: `health/degrade rate_low` high while
  the front-end stage timing stays healthy. Never compare runs across load
  regimes; A/B back-to-back.
- No `TRANSPORT LOSS` warning in the wrapper log (the standing gate, below). If
  it fired, fix transport before reading anything else.
- Runner defaults to `bags/newer-college/quad-easy` (override with `BAG=`).

## Triage (first 5 minutes)

1. Preflight above.
2. `ros2 topic echo /meridian/events` — `pipeline/q_meas_dropped_sweep` firing
   repeatedly ⇒ **overloaded**, go to 3. `frontend/lio/gap` + `frontend/lio/reseed`
   (needs the `lio` debug group on — their emission is group-gated at the
   source) without q_meas drops ⇒ upstream holes (run the intake waterfall below).
   Silent ⇒ **tracking bug**, go to 4. Also overloaded: recorded trajectory
   shorter (in seconds) than the bag (running below 1× real time).
3. **Overloaded:** read `/meridian/stage_timing` (`preprocess`,
   `frontend.lio.ingest`), confirm which side is over budget, apply the first
   knob below that helps, re-run until drops stop.
4. **Tracking:** `eval_ate.py` vs GT, then read the instruments in order:
   `frontend/assoc/n_matched` (association dying?), `frontend/solver/gn_iters`
   pinned at `icp_max_iterations` (non-convergence?), `frontend/obs_min`
   (degenerate geometry?), `frontend/lio/beta` (gravity anchor fighting real
   motion?), `frontend/state/vel_norm` (velocity blow-up?).

## Per-sweep budget (`/meridian/stage_timing`, ms)

The whole front-end is one synchronous stage: `frontend.lio.ingest` = deskew +
downsample + GN solve (association re-runs every iteration — the dominant
cost) + map insert/clip + telemetry. The deskew/solve split is in
`diagnostics()` (`deskew_time_ms` / `solve_time_ms`, surfaced by run_eval
artifacts). Budget numbers on the dev PC and Jetson are to be measured by the
eval campaign; record them here when they exist. Cost scales with keypoints ×
GN iterations; map size affects only memory and the NN constant factor, never
the asymptotics (27-cell probe).

## Tuning knobs when overloaded (cheapest first; `frontend.lio.*` in the deployed config, `src/meridian_ros/config/*.yaml`)

| knob | effect |
|---|---|
| `keypoint_voxel_factor` ↑ (1.5 → 2.0) | fewer keypoints → linear cut in assoc+GN cost; coarser constraint |
| `icp_max_iterations` ↓ | bounds the worst-case sweep; watch for non-convergence rejects (`frontend/lio/reject`) |
| `max_range_m` ↓ | smaller resident map + fewer deskewed returns; cuts memory and clip cost |
| `voxel_size_m` ↑ | coarser map → fewer voxels, cheaper NN; coarser observability normals |
| `max_points_per_voxel` ↓ | shorter in-cell scans per NN probe |

Every change gets a row in `docs/OPTIMIZE.md` (the ledger convention).

## Debug groups & costs (the deep front-end surface)

Spec 09 owns the design; this is the operator card. Four config-seeded
key-prefix wildcard groups under `debug:` gate the deep instrumentation;
everything else (state norms, observability, keyframe count) is always on and
cheap. Off cost per group = **one hash lookup per sweep** (nothing is built).
All stamps are measurement time, so plots align across runs and rates.

| group (`debug.<g>.enable`) | keys | what it answers | cost when ON |
|---|---|---|---|
| `assoc` | `frontend/assoc/{n_attempted,n_matched}` | the association funnel: keypoints offered vs correspondences found — the first thing to read on any tracking failure | negligible (two scalars/sweep) |
| `solver` | `frontend/solver/{gn_iters,dx_norm,chi}` | is GN converging, how hard, and at what final misfit | negligible |
| `lio` | `frontend/lio/{beta,accel_var,n_corr,deskew_span_t_ms}` + events `init_backlog,init_done,gap,reject,reseed` | the LIO internals: gravity-anchor weight vs excitation, init/gap/reseed lifecycle | negligible |
| `map_health` (default ON) | `frontend/map/{voxels,points}` | map growth / clip pathology | negligible |

Always-on (ungrouped): `frontend/state/{vel_norm,bias_gyr_norm,bias_acc_norm}`,
`frontend/obs` (vec6) + `frontend/obs_min`, `frontend/keyframe_count`, and the
ungated `frontend/lio/error` event. Heavy payloads: `body/scan` (the deskewed
sweep on `/meridian/cloud_body`, gated by `debug.publish_clouds`, rate-limited).

Live toggle (no restart, same wildcard the config seeds):

```bash
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey \
    "{key: 'frontend/lio/*', enable: true, max_hz: 10.0}"
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey \
    "{key: 'body/scan', enable: true, max_hz: 2.0}"   # single key works too
```

`src/meridian_ros/config/newer-college-quad.yaml` ships all groups ON (it is the
accuracy-hunt exemplar); every other config keeps the default-off posture. The
offline replay (`replay_runner`) honours the same `debug:` groups from the
same YAML — a replay records exactly what the live posture would (no rate limit:
every sample lands in `*_telemetry.txt`).

**The odometry path:** `/meridian/path` (nav_msgs/Path) is the registered pose
stream (`frontend/path_sample`) sampled at `debug.path_sample_hz` (30 Hz
default), aggregated by the wrapper, republished at `path_publish_hz`, capped
at `path_max_poses`. Add a Path display in rviz (fixed frame `odom`) to see the
trajectory; pair it with `body/scan` and the registered cloud to see whether a
divergence is pose or map.

**One figure per run:** `tools/plot_frontend.py` renders the whole surface
(association funnel, solver trace, LIO internals, observability, state norms,
map health, queues, stage timing) from the run artifacts; `tools/run_eval.sh
--diagnose` writes `<name>_frontend.png` automatically. Panels for off groups
render as "group off", so figures stay comparable across postures.

## Telemetry capture hygiene (live only — replay/FileSink is always exact)

Two ways a live capture silently thins out, both transport-level, both fixed:

- **Token-bucket beating:** a per-key rate limit sitting exactly on the sweep
  cadence (e.g. `telemetry_rate_hz: 10` against 10 Hz sweeps) aliases against
  per-key jitter — measured capture chaos of 1–98% per key. Set the telemetry
  rate well above the cadence you care about (quad config ships 50).
- **Subscriber-side burst overflow:** per-sweep telemetry goes out as a burst
  of small messages; the publisher queue is deep (512, reliable), so the loss
  point is the *subscriber* — `ros2 topic echo` defaults to a tiny queue. Use
  `--qos-depth 512` (or a real recorder) when capturing live.

When in doubt, don't capture live at all: `replay_runner`'s FileSink writes
every sample exactly and is the instrument the numbers should come from.

## The intake audit: where sweeps actually die (and the standing gate)

**"Drops" has two different meters — never conflate them.**
`pipeline/q_meas_dropped` (telemetry) counts sweeps the FRONT-END evicted under
compute pressure — the estimator metric. `sweep_gap_bridged` log lines count ANY
hole in the arriving stamp stream, including scans that died before the node
ever saw them.

**The waterfall** — a counter at every hop, so the loss point is arithmetic:

```
ros2 bag info                       published N      (ground truth)
wrapper/lidar/cb_n                  delivered to our callback
wrapper/lidar/lost_upstream_n       died BEFORE the callback (the stamp stream ticks at
                                    the nominal period, so a k-period hole = k-1 scans
                                    lost upstream — measures the one layer we don't own)
wrapper/lidar/convert_rejected_n    refused by to_raw_lidar
preprocess (timing.yaml count)      reached the preprocess stage
pipeline/q_meas_dropped             evicted by the pipeline (FE compute pressure)
frontend.lio.ingest count           solved by the front-end
```

cb_n + lost_upstream_n must ≈ N. Any unexplained difference means a new hole.

**QoS pairing — both sides, the standing fix.** Best-effort delivery of the
fragmented ~8 MB scans silently loses ~14% under full-pipeline load, uniformly,
for every subscriber on the topic; no application counter sees it — only the
stamp audit does. So: the player publishes the LiDAR topic RELIABLE
(`tools/replay_qos_overrides.yaml`, passed by the runner) and the node
subscribes RELIABLE (`sensors.lidar.qos_reliable: true` in the replay config).
Reliability is per-pairing — a reliable writer does NOT retransmit for a
best-effort reader, and a reliable reader will not pair with a best-effort
writer — so the knob must match the publisher (set false against a best-effort
sensor driver on the robot; check the driver's QoS on the Jetson and prefer
reliable if offered).

**The standing gate:** the wrapper logs `TRANSPORT LOSS: n scans (x%) never
reached this node` (throttled, threshold 1%) the moment upstream loss appears;
the waterfall counters are permanent telemetry. If that warning shows up in any
run — replay or live — fix the transport before reading any other number.
Downstream, a lost scan is just a stamp-stream hole: small holes ride through
on dead reckoning, holes past `lio.max_gap_s` arm a reseed, and a failed
post-gap registration clears the map and re-anchors — the run's ATE is then
poisoned by the landing error.

## Lessons (measured once — don't re-litigate)

- **Reading `sweep_gap_bridged` as FE drops cost a week** — it counts transport
  holes too. Always run the waterfall before blaming the estimator.
- **Background load mis-attributed a "regression"** (a streaming encoder at
  ~65% CPU doubled the drop count); hence the preflight load check.
- **Never `colcon build` while a replay is running** — with `--symlink-install`
  the rebuild swaps the shared libraries under the live process, which then
  wedges or silently runs mixed-version code. Finish or kill replays first.
- **Short clips lie.** Queue-pressure drops and slow failure modes only appear
  on full sequences; every accuracy claim is a full-sequence claim.

## Back-end (L3) & GNSS-referenced harness

The back-end has its own offline loop, decoupled from the front-end replay, via
`tools/backend_dev.sh` (every subcommand runs inside the `meridian` container):

```bash
# Dump the front-end's back-end input stream once (keyframes + GNSS + loops), then iterate
# on the back-end alone at ~thousands of folds/second (no front-end, no ROS):
backend_dev.sh dump  <cfg> <bag_dir> bags/run            # -> bags/run.packets.bin (+ .index.txt)
backend_dev.sh run   <cfg> bags/run.packets.bin out.tum  # [--inject-loops L.yaml] [--g2o g.g2o]
backend_dev.sh check <cfg> bags/run.packets.bin
backend_dev.sh loops bags/run.packets.bin.index.txt L.yaml --gt GT.tum   # truth-anchored loops
```

`backend_runner --g2o <file>` dumps the pose sub-graph (SE3 vertices + relative edges) for
offline inspection. Key back-end telemetry: `backend/chi2` (graph health), `backend/update_ms`,
`backend/relin_count` + the `backend/relinearize` event (loop-thrash), `backend/n_loops` /
`backend/n_gnss`, `backend/fallback_count` (FM-3b rebuilds — a field-survival alarm), the
`backend/loop_edge` marker (green accepted / red rejected), and `backend/gnss/datum_locked`.

**GNSS as a drift reference (no GT file needed).** When a sequence has GNSS, the fixes are an
independent absolute reference for the *front-end*:

```bash
# Capped front-end replay -> packet dump -> front-end drift vs GNSS (rmse_fit is the A/B metric):
backend_dev.sh gnss-eval <cfg> <bag_dir> <max_secs> [--fit-window-s S] [--onset-thresh M]
```

It fits the 4-DoF map<-ENU datum on an early trusted window (the same fit the back-end locks)
and reports residual growth; `rmse_fit` is the baseline drift a tuning A/B minimises and
`onset_kf` is the divergence onset GNSS sees. Cross-checked against the back-end datum lock
(fitted yaw matches to <0.2°).
