# Real-time debugging (front-end, on target)

Field guide for "the trajectory looks wrong on the Jetson." Specs own the design;
this is triage.

## Tooling (run this before interpreting anything)

```bash
# Deterministic offline replay — THE way to evaluate accuracy / run A/Bs.
# Reads the bag directly (no ROS transport), pipeline in Replay mode: lossless,
# solver deadline off, byte-identical output across runs, parallel-safe.
install/meridian_ros/lib/meridian_ros/replay_runner \
    <cfg.yaml> <bag_dir> out.tum [max_content_secs]

# Live-path run (nondeterministic: lossy transport + wall-clock budget; use for
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

The front-end must finish each sweep in **~100 ms** (10 Hz). When it can't, `Q_meas`
drops whole sweeps → gaps → `reseedAfterGap` coasts on constant velocity without
solving → the estimate **barely moves while the platform walks away**. Looks like
drift/explosion; it's actually starvation. Fix the budget, not the estimator.

## Preflight — before trusting any number

- No ghost `odometry_node`: `pgrep -x odometry_node` must be empty. Run via
  `tools/run_bag_headless.sh`, which execs the binary and verifies death.
- Quiet box: `ps aux --sort=-%cpu | head`, load average ≲ 1. Background load starves
  the *input intake* (player / executor / T1) — no front-end change can fix it. The
  tell: `health/degrade rate_low` high while T2 stage timings stay healthy. Never
  compare runs across load regimes; A/B back-to-back (`MERIDIAN_SYNC_KEYFRAME_COV`
  env knob flips sync/async keyframe-cov in the same binary for exactly this).
- No `TRANSPORT LOSS` warning in the wrapper log (the standing gate, below). If it
  fired, fix transport before reading anything else.
- Determinism: only the deterministic replay harness (replay mode, single-threaded
  solve) is bit-reproducible — there, a run-to-run change on the same bag means
  nondeterminism leaked in. LIVE bag replay through the node is NOT run-to-run
  deterministic (threaded solve + arrival timing); compare live runs statistically,
  never bit-wise.
- Runner defaults to `bags/newer-college/quad-easy` (override with `BAG=`).

## Triage (first 5 minutes)

1. Preflight above.
2. `ros2 topic echo /meridian/events` — `q_meas_dropped_sweep` or `window_restart`
   firing repeatedly ⇒ **overloaded**, go to 3. Silent ⇒ **tracking bug**, go to 4.
   Also overloaded: `frontend/deadline_hit` pinned at 1.0, or recorded trajectory
   shorter (in seconds) than the bag (running below 1× real time).
3. **Overloaded:** read `/meridian/stage_timing`, find the stage over budget, apply
   the first knob below that helps, re-run until drops stop.
4. **Tracking:** `eval_ate.py` vs GT, then bisect by ablation: `visual.enable: false`
   gives the clean-LIO baseline (isolates the photometric stage); the Jacobian parity
   tests (`test_residuals_{lidar,visual,imu}`) catch analytic-Jacobian regressions.

## Per-sweep budget (`/meridian/stage_timing`, ms)

`assoc ~30` + `visual ~15` + `solve ~10-20` + `map ~15` + `visual_map ~30` ≈ the
critical path. Sums to ~100 ms on the dev PC — **the Jetson is slower, so headroom is
the goal.** `frontend/solve_ms` (whole solveWindow) ≫ `frontend.ct.solve` (Ceres only)
means the cost is association/map maintenance, not the optimizer.

## Tuning knobs when overloaded (cheapest first; `frontend.*` in the deployed config, `src/meridian_ros/config/*.yaml`)

| knob | effect |
|---|---|
| `visual.enable: false` | drops the whole photometric stage → clean LIO (~7.5 cm). Safe fallback + the test for "is visual the cause?" |
| `lidar.max_lidar_factors` ↓ (1500→1000) | fewer LiDAR residuals; normal-stratified so observability holds |
| camera → DAVIS (`event_cam00`) | 346×260 vs FLIR 1024×768 ≈ 8× fewer pixels |
| `spline.n_cp_max` | keep at 1 (>1 diverges pending warm-start redesign, see below) |
| `solver.time_limit_ms` (90) | hard solve cap; do **not** lower (measured: 60 ms starves the solve, ATE 0.56 m; after parallel association 90 is not binding) |

`visual_map` cost must scale with what the camera *sees* (≤ candidates), never with
total `map_points` — if `frontend.ct.visual_map` climbs with run length, that
regressed (see `VisualMap::selectVisibleIds`). `visual.active_box_m` bounds memory,
not per-sweep cost.

## Debug groups & costs (the deep front-end surface)

Spec 09 §5.1/§11 owns the design; this is the operator card. Five config-seeded
key-prefix wildcard groups under `debug:` gate the deep instrumentation; everything
else (counts, residual means, biases, innovation, observability, IMU consistency)
is always on and cheap. Off cost per group = **one hash lookup per sweep** (nothing
is built). All stamps are measurement time, so plots align across runs and rates.

| group (`debug.<g>.enable`) | keys | what it answers | cost when ON |
|---|---|---|---|
| `assoc` | `frontend/assoc/*`: attempted/matched funnel, reject_{nn,dist,plane,score}, res_p95 + res_hist(16), nn_dist_*, plane_rms_*, strata_kept, outliers cloud | where association loses points; misfit distribution vs time; directional balance of the factor set | ~0.1–0.3 ms/sweep stats + one rejected-cloud build (2 Hz publish) |
| `solver` | `frontend/solver/*`: iter_cost/step/grad/tr_radius vecs, cost_initial, termination; upgrades `frontend/resid/*` to every sweep + cost_frac | is the solve converging, what terminates it, which residual family dominates the cost | ~1–3 ms/sweep (post-solve `Problem::Evaluate`; cannot starve the solve, but mind the Jetson budget) |
| `deskew` | `frontend/deskew/*`: corr_{mean,p95,max}_m, sweep_{trans_m,rot_deg}, 'pre' cloud | how much CT motion compensation actually moves points; before/after deskew view vs `map/registered` | ~0.1 ms + one cloud build (2 Hz publish) |
| `spline` | `frontend/spline/*`: vel/acc/omega vec3, acc/omega span-RMS, jerk_rms | vibration & knot-cadence aliasing (gait frequency beating the knot grid) | ~0.2 ms (≈40 analytic spline evals/sweep) |
| `map_health` (default ON) | `frontend/map/*`: size, n_inserted, insert_rejected | map growth rate / insert pathology | negligible |

Live toggle (no restart, same wildcard the config seeds):

```bash
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey \
    "{key: 'frontend/assoc/*', enable: true, max_hz: 10.0}"
ros2 service call /meridian/set_debug_key meridian_msgs/SetDebugKey \
    "{key: 'frontend/deskew/pre', enable: true, max_hz: 2.0}"   # single key works too
```

`src/meridian_ros/config/newer-college-quad.yaml` ships all groups ON (it is the
accuracy-hunt exemplar); every other config keeps the default-off posture. The
deterministic replay (`replay_runner`) honours the same `debug:` groups from the
same YAML — a replay records exactly what the live posture would (no rate limit:
every sample lands in `*_telemetry.txt`).

**The spline path:** `/meridian/path` (nav_msgs/Path) is the solved B-spline
sampled at `debug.path_sample_hz` (30 Hz default), republished at
`path_publish_hz`, capped at `path_max_poses`. Add a Path display in rviz (fixed
frame `odom`) to see the continuous trajectory — inter-sweep jitter shows here
that keyframe-rate displays hide. Poses near the window's leading edge are still
being refined, so small kinks vs. the final trajectory are expected.

**One figure per run:** `tools/plot_frontend.py` renders the whole surface
(12 panels: funnel, per-family RMS, residual heatmap, solver, innovation, deskew,
kinematics, IMU consistency + biases, observability, queues, stage timing,
visual/GNSS) from the run artifacts; `tools/run_eval.sh --diagnose` writes
`<name>_frontend.png` automatically. Panels for off groups render as "group off",
so figures stay comparable across postures.

## Autodiff vs analytic Jacobians

LiDAR, visual **and IMU** factors use hand-derived analytic Jacobians; the autodiff
versions are kept behind `*Autodiff` factories as parity twins. If convergence looks
wrong after a Jacobian change, suspect these first. Guard:
`test_residuals_{lidar,visual,imu}` assert analytic == autodiff to the
finite-difference floor (~1e-7..1e-9).

**`spline.n_cp_max` stays at 1.** `n_cp_max > 1` DIVERGES on real data (measured:
ATE 1173 m at n_cp=2). The Jacobians are NOT the cause — they are n_cp-correct
(autodiff parity ~3e-11); the failure is the **dense-knot warm start** (a wiggly
dense-knot seed corrupts association). Do not flip it before the warm-start
redesign lands (task #62; see docs/OPTIMIZE.md tested-and-rejected).

## The intake audit: where sweeps actually die (and the standing gate)

**"Drops" has two different meters — never conflate them.** `pipeline/q_meas_dropped`
(telemetry) counts sweeps the FRONT-END evicted under compute pressure — the estimator
metric. `sweep_gap_bridged` log lines count ANY hole in the arriving stamp stream,
including scans that died before the node ever saw them.

**The waterfall** — a counter at every hop, so the loss point is arithmetic:

```
ros2 bag info                       published N      (ground truth)
wrapper/lidar/cb_n                  delivered to our callback
wrapper/lidar/lost_upstream_n       died BEFORE the callback (the stamp stream ticks at
                                    the nominal period, so a k-period hole = k-1 scans
                                    lost upstream — measures the one layer we don't own)
wrapper/lidar/convert_rejected_n    refused by to_raw_lidar
preprocess (timing.yaml count)      reached T1
pipeline/q_meas_dropped             evicted by the pipeline (FE compute pressure)
frontend.ct.marg count              solved by T2
```

cb_n + lost_upstream_n must ≈ N. Any unexplained difference means a new hole.

**QoS pairing — both sides, the standing fix.** Best-effort delivery of the fragmented
~8 MB scans silently loses ~14% under full-pipeline load, uniformly, for every
subscriber on the topic; no application counter sees it — only the stamp audit does.
So: the player publishes the LiDAR topic RELIABLE (`tools/replay_qos_overrides.yaml`,
passed by the runner) and the node subscribes RELIABLE (`sensors.lidar.qos_reliable:
true` in the replay config). Reliability is per-pairing — a reliable writer does NOT
retransmit for a best-effort reader, and a reliable reader will not pair with a
best-effort writer — so the knob must match the publisher (set false against a
best-effort sensor driver on the robot; check the driver's QoS on the Jetson and
prefer reliable if offered).

**The standing gate:** the wrapper logs `TRANSPORT LOSS: n scans (x%) never reached
this node` (throttled, threshold 1%) the moment upstream loss appears; the waterfall
counters are permanent telemetry. If that warning shows up in any run — replay or
live — fix the transport before reading any other number. Downstream, the loss
masquerades as IMU-bridged gaps (graceful, ATE holds ~0.10 m) until holes cluster
≥6 deep, the bridge horizon is exceeded, the window reseeds on a constant-velocity
guess, and the run's ATE is poisoned by the landing error. Bridges also widen the
marginalization blankets, so sustained loss inflates the `marg` tail.

## Lessons (measured once — don't re-litigate)

- **Parallel association (OpenMP) is the decisive sub-100 ms lever** (assoc ~21→7 ms,
  drops 205→7). Deterministic by construction: `schedule(static)` + per-thread buffers
  merged in thread order = bit-identical to serial (the `ParallelAssociate*` test
  guards it); ikd-Tree is held safe by a `shared_mutex` (shared on `fitPlane`,
  exclusive on `insert`/`trimAround`).
- **iVox map backend is a wash** at our 2.2 m gate / 0.6 m voxel ratio. The map
  structure is not the lever; the serial association loop was.
- **Reading `sweep_gap_bridged` as FE drops cost a week** — it counts transport holes
  too. Always run the waterfall before blaming the estimator.
- **Background load mis-attributed a "regression"** (294→752 drops; a streaming encoder
  at ~65% CPU). The matched A/B showed the async cov worker *reduces* restarts 25→0-1;
  hence the preflight load check.
- **Never `colcon build` while a replay is running** — with `--symlink-install` the
  rebuild swaps the shared libraries under the live process, which then wedges or
  silently runs mixed-version code. Finish or kill replays first.
