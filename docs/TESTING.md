# Testing Meridian against a FusionPortable bag

End-to-end check of the Phase-1 system (L0 sensors → L1 preprocessing → aggregated,
cold-start-deskewed sweeps), driven by a real dataset through the live ROS node.
Everything runs inside the dev container (see `DEVELOPMENT.md`); on the Mac the flow
is identical minus rviz (use Foxglove on `ws://localhost:8765`).

## 0. Build the workspace

```bash
colcon build --symlink-install            # Linux/GPU box
source install/setup.bash
```

## 1. Get a sequence

FusionPortable is the primary development dataset (`docs/DATASET.md`). Start with a
handheld/campus sequence — `20220216_garden_day` is the bring-up reference. Download
the ROS1 bag (and the Kalibr calibration files) from the FusionPortable dataset site
(linked from the paper in `docs/DATASET.md`).

## 2. Convert ROS1 → ROS2

The datasets ship as ROS 1 bags; convert once with the `rosbags` tool (pure Python,
no ROS 1 install needed):

```bash
pip install rosbags
rosbags-convert --src 20220216_garden_day.bag --dst bags/garden_day
```

## 3. Confirm the topic names

```bash
ros2 bag info bags/garden_day
```

The default config (`src/meridian_ros/config/fusionportable.yaml`) expects:

| sensor | topic |
|---|---|
| LiDAR (Ouster OS1-128) | `/os_cloud_node/points` |
| IMU | `/imu/data` |
| camera (left) | `/stereo/left/image` |
| GNSS (disabled by default) | `/ublox/fix` |

If your bag differs, edit the four `topic:` keys. While there, paste the Kalibr
LiDAR-IMU extrinsic into `sensors.lidar.extrinsic_T/extrinsic_R` (identity is fine
for a first visual check, required before judging geometry).

## 4. Run

Terminal A — the node (+ rviz):

```bash
ros2 launch meridian_ros meridian.launch.py \
    config_file:=$(pwd)/src/meridian_ros/config/fusionportable.yaml \
    use_sim_time:=true rviz:=true
```

Terminal B — the bag, publishing the sim clock:

```bash
ros2 bag play bags/garden_day --clock
```

## 5. What you should see

- **rviz** (fixed frame `body`): the deskewed sweep streaming on
  `/meridian/cloud_body`, coloured by intensity. (No TF yet — there is no pose
  estimate until the L2 front-end lands; each sweep is shown in the body frame at
  its own scan-end.)
- **Node log**: a `group #N: ... deskewed=1` line every 5 s, and a
  `preprocess/imu_init_done` event once the static IMU init converges. The init
  needs ~10 IMU samples with the rig near-still; sequences that start in motion
  retry until a quiet window appears (watch `/meridian/events`).
- **Telemetry**:

```bash
ros2 topic echo /meridian/stage_timing          # preprocess + deskew wall time
ros2 topic echo /meridian/telemetry             # rates, group sizes, queue gauges
ros2 topic echo /meridian/events                # init / health / drop events
```

- **Runtime debug control** (no restart):

```bash
ros2 service call /meridian/set_debug_key meridian_msgs/srv/SetDebugKey \
    "{key: 'body/scan', enable: true, max_hz: 5.0}"
ros2 service call /meridian/set_log_level meridian_msgs/srv/SetLogLevel \
    "{module: '', level: 1}"
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `dropping scan: missing ... per-point time field` | wrong LiDAR topic, or the bag's cloud has no `t`/`time` field — check `ros2 bag info` and the field list with `ros2 topic echo --once <topic> \| head` |
| no `/meridian/cloud_body` | IMU init not converged (see `/meridian/events`), or `debug.publish_clouds: false` |
| health events report degraded sync | expected in bag replay: there is no live PTP discipline, recorded stamps pass through unchanged |
| nothing at all | `use_sim_time:=true` on the node but the bag played without `--clock` (or vice versa) |

## Scope note

This exercises sensing + preprocessing (Phase 1). Odometry, TF, the map and drift
metrics arrive with the L2 front-end and later layers; the evaluation harness
(`docs/specs/10_evaluation_harness.md`) then replays the same sequences off-ROS for
ATE/RPE scoring.

## Front-end verification

The L2 CT front-end now runs inside the pipeline (`frontend.kind: ct_livo` in the
config). It owns a trajectory, so rviz's fixed frame is `odom`, the node publishes
the `odom -> body` TF, and `/meridian/cloud_registered` is the world-stable map view.
The body-frame `/meridian/cloud_body` stays available (rviz display **DeskewedBodyScan**,
disabled by default) as a debug view.

### Build/test gates

```bash
colcon test --packages-select meridian_frontend && colcon test-result --verbose
colcon test --packages-select meridian_pipeline meridian_config
```

The front-end's own unit/integration tests (spline, IMU/LiDAR residuals,
marginalization, oracle differential) must be green before judging a bag run.

### Bag run

Same launch as above (`fusionportable.yaml` already carries the `frontend:` block).

```bash
ros2 launch meridian_ros meridian.launch.py \
    config_file:=$(pwd)/src/meridian_ros/config/fusionportable.yaml \
    use_sim_time:=true rviz:=true
ros2 bag play bags/garden_day --clock          # second terminal
```

### Visual checklist (rviz, fixed frame `odom`)

- **World-stable geometry**: walls/ground in `/meridian/cloud_registered` stay put as
  the rig moves — they do not smear or swim with each sweep.
- **Smooth odom track**: the **Odometry** arrows (`/meridian/odom`, last ~100) trace a
  continuous path with no teleports between sweeps.
- **Stationary start holds**: while the rig is still at the start, the `odom -> body`
  TF and the odom track do not drift.
- **No restarts**: `/meridian/events` shows the one-time `preprocess/imu_init_done` and
  recurring `frontend/keyframe`, and **no** window-restart / no-effective-points events.

### Topic checklist

```bash
ros2 topic hz /meridian/odom               # ~ sweep rate (≈10 Hz on this bag)
ros2 topic hz /meridian/cloud_registered   # heavy key, rate-limited (≈2 Hz)
ros2 topic echo /meridian/odom --once      # pose in frame_id "odom", child "body"
ros2 run tf2_ros tf2_echo odom body        # the live transform, updating at sweep rate
ros2 topic echo /meridian/telemetry        # frontend/keyframe_count climbs; queue gauges
```

Expected ranges: `/meridian/odom` near the LiDAR rate; `pipeline/q_meas_dropped` should
stay absent (front-end keeps up); `frontend/keyframe_count` increments roughly on the
keyframe cadence (`frontend.keyframe.{dist_m,rot_deg,time_s}`).

### TUM export + ATE

Record the track to a TUM file (runs inside the box; flushes on Ctrl-C):

```bash
python3 tools/record_tum.py /tmp/meridian_odom.tum     # start before the bag
# ... play the bag, Ctrl-C the recorder when it finishes ...
```

Compare against the FusionPortable ground truth with `evo` (optional — not assumed
installed; `pip install evo` if missing):

```bash
evo_ape tum GT.tum /tmp/meridian_odom.tum -a            # align (Sim(3)) and report ATE
```

The ground-truth TUM comes with the sequence (see `docs/DATASET.md`); `-a` removes the
arbitrary odom-origin offset before scoring.
