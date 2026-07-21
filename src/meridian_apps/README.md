# meridian_apps

`meridian_apps` composes the generic ROS-to-core ingress, the transport-agnostic
`meridian_local_rt` initializer and fixed-lag local estimator, and its Rerun
recorder. The current bag-debug executable records the causal and terminal
fixed-lag trajectories but does not add a heavy ROS registration-map or
diagnostics topic. That is a choice for this audit executable, not a restriction
on publishing lightweight downstream products such as odometry from another
composition. Foxglove Bridge lets a viewer inspect the bag's original ROS
topics during replay.

Run Newer College through the same generic launch used by any other bag:

```bash
mkdir -p /workspace/out
ros2 launch meridian_apps bag_debug.launch.py \
  bag:=/workspace/bags/newer-college/quad-easy \
  config:=/workspace/src/meridian_apps/config/newer_college.yaml \
  rrd:=/workspace/out/quad_easy_local_rt.rrd
```

The launch starts the ingress and Foxglove Bridge, then invokes ordinary
`ros2 bag play`. It plays all original topics by default, without a custom
replay process, QoS override, completion service, or in-process expected
counts. An optional `topics` argument passes an explicit allow-list to rosbag2
for bags containing unrelated custom types. When the player exits, ingress
drains its LiDAR decoder, `meridian_local_rt` drains observations with complete
IMU support, and Rerun writes its footer.

The RRD contains:

- full-rate compact IMU values and conversion timings;
- compact metadata for each converted LiDAR scan;
- conversion, queue, and downstream ingress failures;
- a decimated LiDAR preview at 1 Hz, capped at 4096 points; and
- initialization status, quality, accepted seed, and temporary bootstrap poses;
- compact 2 Hz samples of full-rate adjacent IMU preintegration;
- causal and terminal fixed-lag odometry trajectories plus estimator quality;
- stage and sub-stage timings for preprocessing, association, solving,
  marginalization, map maintenance, and recording;
- complete local registration-map snapshots at `/local_rt/map/registration`,
  recorded once per accepted estimator revision at the estimator's native
  cadence (without an additional ROS map topic in this executable); and
- Rerun queue-drop and logging-error counters.

The preview exists only inside the RRD. There is no ROS preview or diagnostics
publisher. Connect Foxglove to `ws://<development-host>:8765` to inspect the
original ROS topics directly.

Post-run checks stay outside the runtime:

```bash
python3 tools/analyze_ingress_rrd.py out/quad_easy_local_rt.rrd \
  --bag bags/newer-college/quad-easy \
  --config src/meridian_apps/config/newer_college.yaml
```

The analyzer reads the configured input topics, compares RRD rows with rosbag2
metadata as an advisory, reports timing/point/conversion statistics plus local
initialization, preintegration, estimator-quality, and registration-map
summaries, and verifies the RRD.
