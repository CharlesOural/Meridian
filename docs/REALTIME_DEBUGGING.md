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

## Triage (first 5 minutes)

1. `pgrep -x odometry_node` → must be empty before a run (ghosts poison + starve).
2. `ros2 topic echo /meridian/events` — `q_meas_dropped_sweep` or `window_restart`
   firing repeatedly ⇒ **overloaded**, go to 4. Silent ⇒ **tracking bug**, go to 5.
3. `frontend/deadline_hit` pinned at 1.0, or recorded trajectory shorter (in seconds)
   than the bag ⇒ running below 1× real time ⇒ overloaded.
4. **Overloaded:** read `/meridian/stage_timing`, find the stage over budget, apply
   the first knob below that helps, re-run until drops stop.
5. **Tracking:** `eval_ate.py` vs GT; run `frontend.kind: iekf_oracle` on the same bag
   — if the oracle tracks and CT doesn't, the bug is in the CT solve, not the shared
   LiDAR/IMU front matter.

## Per-sweep budget (`/meridian/stage_timing`, ms)

`assoc ~30` + `visual ~15` + `solve ~10-20` + `map ~15` + `visual_map ~30` ≈ the
critical path. Sums to ~100 ms on the dev PC — **the Jetson is slower, so headroom is
the goal.** `frontend/solve_ms` (whole solveWindow) ≫ `frontend.ct.solve` (Ceres only)
means the cost is association/map maintenance, not the optimizer.

## Tuning knobs when overloaded (cheapest first; `frontend.*` in fusionportable.yaml)

| knob | effect |
|---|---|
| `visual.enable: false` | drops the whole photometric stage → clean LIO (~7.5 cm). Safe fallback + the test for "is visual the cause?" |
| `lidar.max_lidar_factors` ↓ (1500→1000) | fewer LiDAR residuals; normal-stratified so observability holds |
| camera → DAVIS (`event_cam00`) | 346×260 vs FLIR 1024×768 ≈ 8× fewer pixels |
| `spline.n_cp_max` | keep at 1 (adaptive density needs the analytic factors first) |
| `solver.time_limit_ms` (90) | hard solve cap; lowering bounds latency but shallows the solve |

`visual_map` cost must scale with what the camera *sees* (≤ candidates), never with
total `map_points` — if `frontend.ct.visual_map` climbs with run length, that
regressed (see `VisualMap::selectVisibleIds`). `visual.active_box_m` bounds memory,
not per-sweep cost.

## Autodiff vs analytic Jacobians

LiDAR & visual factors use hand-derived analytic Jacobians (faster than autodiff). If
convergence looks wrong after a Jacobian change, suspect these first: they're in their
own commit ("analytic spline Jacobians") — revert it to fall back to numerical
autodiff. Guard: `test_residuals_{lidar,visual}` assert analytic == autodiff to ~1e-7.
IMU factor is still autodiff.

## Hygiene before trusting any number

- No ghost `odometry_node` (run via `tools/run_bag_headless.sh`, which execs + verifies).
- Replay is deterministic: a run-to-run change on the same bag = nondeterminism leaked in.
- **Never `colcon build` while a replay is running** — with `--symlink-install` the
  rebuild swaps the shared libraries under the live process, which then wedges or
  silently runs mixed-version code. Finish or kill replays first.
- Runner plays `bags/canteen_day_fixed` (column-clock-repaired). A raw download has
  unrepaired per-point LiDAR times.
