# Real-time debugging (front-end, on target)

Field guide for "the trajectory looks wrong on the Jetson." Specs own the design;
this is triage. Tools: `tools/run_bag_headless.sh`, `tools/eval_ate.py`.

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
- Runner plays `bags/canteen_day_fixed` (column-clock-repaired). A raw download has
  unrepaired per-point LiDAR times.
