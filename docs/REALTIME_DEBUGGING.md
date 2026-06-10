# Real-time debugging (front-end, on target)

Field guide for "the trajectory looks wrong on the Jetson." Specs own the design;
this is triage. Tools: `tools/run_bag_headless.sh`, `tools/eval_ate.py`.

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
