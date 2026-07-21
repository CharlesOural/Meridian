# Barakuda ENSTA dataset

This runbook prepares and evaluates the Taurus/Mobilex Barakuda recordings from
17 July 2026. The first bring-up sequence is `ensta_small_loop`; use
`ensta_front_loop` only after that run completes cleanly.

## Recorded data

Archives are ROS 2 Humble MCAP bags. The estimation inputs verified from
the recorded message schemas are:

- `/ouster/points`: `sensor_msgs/msg/PointCloud2`, 10 Hz, frame `os_sensor`,
  128 x 1024 points with `x/y/z: float32`, `t: uint32` nanoseconds,
  `intensity: float32`, and `ring: uint16`;
- `/imu/data`: `sensor_msgs/msg/Imu`, about 200 messages/s, frame `sbg_imu`;
- `/tf`: position-only `rtk_enu -> rtk_antenna` at about 5 Hz.

The committed profile is
`src/meridian_apps/config/barakuda.yaml`. It uses the supplied Taurus/Mobilex
SBG noise densities and measured `sbg_imu_link -> os_sensor` extrinsic. The base
evaluated by Meridian is the SBG origin.

## Extract a bag

The archives already contain the rosbag directory and `metadata.yaml`. From the
repository root, extract the small loop with:

```bash
tar --extract \
  --file bags/barakuda/XXX \
  --directory bags/barakuda \
  --no-same-owner
```

Then verify it inside the development container:

```bash
ros2 bag info bags/barakuda/ensta_small_loop_20260717_162230
```

## Generate RTK ground truth

The RTK TF contains antenna position but no attitude. The extractor interpolates
the SBG orientation from `/imu/data` at each RTK timestamp and converts antenna
position to the SBG origin using the measured lever arm:

```text
p_rtk_enu_sbg = p_rtk_enu_antenna - R_rtk_enu_sbg * r_sbg_antenna
```

Replace `X Y Z` below with the vector from `sbg_imu_link` to the RTK antenna
phase centre, expressed in `sbg_imu_link`, in metres:

```bash
mkdir -p bags/barakuda/gt/tum bags/barakuda/gt/state
python3 tools/barakuda_bag_to_tum.py \
  bags/barakuda/ensta_small_loop_20260717_162230 \
  bags/barakuda/gt/tum/gt-barakuda-ensta-small-loop.tum \
  --state-output bags/barakuda/gt/state/gt-barakuda-ensta-small-loop.csv \
  --imu-to-antenna-m X Y Z
```

The tool also accepts the original `.tar` directly, which avoids extracting
43 GiB just to create the trajectory:

```bash
python3 tools/barakuda_bag_to_tum.py \
  bags/barakuda/ensta_small_loop_20260717_162230.tar \
  bags/barakuda/gt/tum/gt-barakuda-ensta-small-loop.tum \
  --state-output bags/barakuda/gt/state/gt-barakuda-ensta-small-loop.csv \
  --imu-to-antenna-m X Y Z
```

Omitting `--imu-to-antenna-m` deliberately emits a warning and creates only a
provisional zero-lever trajectory. Such a trajectory is useful for plumbing
but is not an authoritative ATE reference because vehicle rotation makes the
antenna-to-IMU position error time-varying.

## Replay through Meridian

The development image includes the ROS 2 MCAP storage plugin. Rebuild it once
after pulling this profile, then build Meridian:

```bash
docker compose -f compose.dev.yaml up -d --build
docker compose -f compose.dev.yaml exec meridian bash
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install --packages-up-to meridian_apps
source install/setup.bash
mkdir -p out
```

Run the small loop at recorded speed:

```bash
ros2 launch meridian_apps bag_debug.launch.py \
  bag:=/workspace/bags/barakuda/ensta_small_loop_20260717_162230 \
  config:=/workspace/src/meridian_apps/config/barakuda.yaml \
  rrd:=/workspace/out/barakuda_small_loop.rrd \
  topics:="/ouster/points /imu/data" \
  rate:=1.0
```

The topic allow-list avoids requiring the SBG, NMEA, and RTCM custom message
packages merely to replay the two estimator inputs. It does not bypass ROS:
the selected messages still pass through ordinary `ros2 bag play`, DDS, and
the production ingress subscriptions.

For a physical live run with the same topics and calibration, omit the player:

```bash
ros2 launch meridian_apps input_debug.launch.py \
  config:=/workspace/src/meridian_apps/config/barakuda.yaml \
  rrd:=/workspace/out/barakuda_live.rrd
```

## Calculate ATE

Analyze the completed RRD against the generated state trajectory:

```bash
python3 tools/analyze_ingress_rrd.py out/barakuda_small_loop.rrd \
  --bag bags/barakuda/ensta_small_loop_20260717_162230 \
  --config src/meridian_apps/config/barakuda.yaml \
  --ground-truth-state bags/barakuda/gt/state/gt-barakuda-ensta-small-loop.csv
```

The state CSV and TUM file describe `T_rtk_enu_sbg_imu`; the YAML makes the
Meridian base coincide with that SBG origin, so estimate and reference use the
same body frame. Do not report the resulting ATE as calibrated until the RTK
antenna lever arm has been supplied and used to regenerate both files.

## Timing caveat

Although `/imu/data` averages about 200 messages/s, its headers were stamped on
ROS arrival and arrive in batches. The small loop contains a maximum observed
header gap of 109.725 ms. The profile retains a 50 ms missing-data gate so this
defect remains observable; it may reject an affected LiDAR interval. This bag
is suitable for an end-to-end robustness test, but it is not evidence for the
live device-time synchronization contract.
