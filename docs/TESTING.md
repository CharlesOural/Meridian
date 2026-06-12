# Testing Meridian against a Newer College bag

End-to-end check of the live system (L0 sensors → L1 preprocessing → L2 CT
front-end), driven by a Newer College multi-cam sequence through the ROS node.
Everything runs inside the dev container (see `DEVELOPMENT.md`). Dataset
download, conversion, layout and per-collection calibration: `docs/DATASET.md`.

**Current scope: LIO only.** The visual stage is switched off in the configs
(`frontend.visual.enable: false`): the Alphasense cameras suffer heavy
vignetting, so the benchmark is LiDAR-inertial first; re-enabling visual on
this dataset is a future task. The camera calibration stays in the configs so
the flip is one line.

## Testing policy (mandatory)

- **Routine validation runs ONE sequence: `quad-easy`.** Every day-to-day
  change is judged on it alone.
- **The 3-sequence pass (`quad-easy` + `math-medium` + `park`) runs ONLY
  before committing large work** — not per-iteration.
- **`quad-hard` is a HOLDOUT.** Never tune on it, never use it to pick between
  candidates; it is reserved for milestone evaluations, and any run on it is
  recorded as a validation result (see `docs/DATASET.md`).

| sequence | config (`src/meridian_ros/config/`) | role |
|---|---|---|
| quad-easy | `newer-college-quad.yaml` | routine validation |
| math-medium | `newer-college-math.yaml` | pre-commit pass |
| park | `newer-college-park.yaml` | pre-commit pass |
| quad-hard | `newer-college-quad.yaml` | **HOLDOUT** — milestones only |

The configs differ per collection (the rig was recalibrated between them);
never run park with the quad config.

## 0. Build + unit gates

```bash
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select meridian_frontend meridian_pipeline meridian_config \
  && colcon test-result --verbose
```

The front-end unit/integration tests (voxel map, IMU tracker, registration,
covariance chain, end-to-end synthetic tracking) must be green before judging a
bag run.

## 1. Preflight

Follow the preflight in `docs/REALTIME_DEBUGGING.md` before trusting any
number — in short: quiet box (load ≲ 1), no ghost `odometry_node`, and the
`TRANSPORT LOSS` gate in the node log must stay silent for the whole run. The
runner refuses to start while a stale node is alive, but it cannot see
background load for you.

## 2. Headless run + ATE (the canonical loop)

```bash
tools/run_bag_headless.sh /tmp/nc_quad_easy          # defaults: quad-easy bag + quad config
python3 tools/eval_ate.py \
    bags/newer-college/gt/tum_asimu/gt-nc-quad-easy.csv /tmp/nc_quad_easy/traj_tum.txt
```

The runner starts the node headless, plays the bag with the player-side QoS
override (`tools/replay_qos_overrides.yaml` — reliable LiDAR writer, paired
with `sensors.lidar.qos_reliable: true` in the configs), records the TUM
trajectory plus `node.log` / `events.yaml` / `timing.yaml` / `telemetry.yaml`,
and verifies the node died. Other sequences:

```bash
BAG=bags/newer-college/math-medium CONFIG=src/meridian_ros/config/newer-college-math.yaml \
  tools/run_bag_headless.sh /tmp/nc_math_medium
BAG=bags/newer-college/park CONFIG=src/meridian_ros/config/newer-college-park.yaml \
  tools/run_bag_headless.sh /tmp/nc_park
```

**Ground truth:** always score against `gt/tum_asimu/*.csv` (GT re-expressed in
the Alphasense-IMU estimation frame), never `gt/tum/`. The `tum_asimu` files
are plain TUM (`stamp x y z qx qy qz qw`, whitespace-separated, no header) —
`eval_ate.py` reads them directly, no conversion step.

Healthy quad-easy run, for calibration of expectations: all ~1991 scans reach
the node (`wrapper/lidar/cb_n` ≈ published count, `lost_upstream_n` = 0), no
`TRANSPORT LOSS` warning, no `frontend/lio/{gap,reject,reseed}` events,
`pipeline/q_meas_dropped` absent
or ~0, ATE in the 0.1–0.3 m band (reference: ~0.2 m rmse full bag).

## 3. Interactive run (rviz)

Terminal A — the node (+ rviz, fixed frame `odom`):

```bash
ros2 launch meridian_ros meridian.launch.py use_sim_time:=true rviz:=true
# config_file:=$(pwd)/src/meridian_ros/config/<other>.yaml for non-quad sequences
```

Terminal B — the bag, publishing the sim clock, with the same QoS override the
runner uses (without it, best-effort delivery silently drops scans):

```bash
ros2 bag play bags/newer-college/quad-easy --clock \
    --qos-profile-overrides-path tools/replay_qos_overrides.yaml
```

What you should see:

- **World-stable geometry**: walls/ground in `/meridian/cloud_registered` stay
  put as the rig moves — no smearing or swimming per sweep.
- **Smooth odom track**: `/meridian/odom` traces a continuous path, no
  teleports; the stationary start holds still.
- **Events**: one `frontend/lio/init_done` (static init needs the ~10 s
  still period at the sequence start), recurring `frontend/keyframe`, and
  **no** `frontend/lio/{gap,reject,reseed}` events.

```bash
ros2 topic hz /meridian/odom               # ≈10 Hz (sweep rate)
ros2 topic echo /meridian/events           # init / health / drop events
ros2 topic echo /meridian/stage_timing     # per-stage wall time (assoc/solve/marg/map)
ros2 topic echo /meridian/telemetry        # rates, queue gauges, waterfall counters
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `TRANSPORT LOSS` in the node log | QoS pairing broken — player override not passed, or `sensors.lidar.qos_reliable` edited; see `docs/REALTIME_DEBUGGING.md` |
| `q_meas_dropped` / `frontend/lio/gap` events | overload — triage with the stage-timing budget in `docs/REALTIME_DEBUGGING.md` |
| `dropping scan: missing ... per-point time field` | wrong LiDAR topic, or a bag prepared without the verification pass in `docs/DATASET.md` |
| nothing at all | `use_sim_time:=true` on the node but the bag played without `--clock` (or vice versa) |
| ATE meters-scale on a healthy log | wrong per-collection config for the sequence, or scored against `gt/tum/` instead of `gt/tum_asimu/` |
