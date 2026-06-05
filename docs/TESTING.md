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
