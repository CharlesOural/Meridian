# meridian_apps

`meridian_apps` composes the generic ROS-to-core ingress with its Rerun
recorder. It performs no odometry and publishes no Meridian debug topics.
Foxglove Bridge is started only so a viewer can inspect the bag's original ROS
topics.

Run Newer College through the same generic launch used by any other bag:

```bash
mkdir -p /workspace/out
ros2 launch meridian_apps bag_debug.launch.py \
  bag:=/workspace/bags/newer-college/quad-easy \
  config:=/workspace/src/meridian_apps/config/newer_college.yaml \
  rrd:=/workspace/out/quad_easy_ingress.rrd
```

The launch starts the ingress and Foxglove Bridge, then invokes ordinary
`ros2 bag play`. It plays the original topics without a custom replay process,
topic filter, QoS override, completion service, or in-process expected counts.
When the player exits, the ingress drains its bounded LiDAR and Rerun queues.

The RRD contains:

- full-rate compact IMU values and conversion timings;
- compact metadata for each converted LiDAR scan;
- conversion, queue, and downstream ingress failures;
- a decimated LiDAR preview at 1 Hz, capped at 4096 points; and
- Rerun queue-drop and logging-error counters.

The preview exists only inside the RRD. There is no ROS preview or diagnostics
publisher. Connect Foxglove to `ws://<development-host>:8765` to inspect the
original ROS topics directly.

Post-run checks stay outside the runtime:

```bash
python3 tools/analyze_ingress_rrd.py out/quad_easy_ingress.rrd \
  --bag bags/newer-college/quad-easy \
  --config src/meridian_apps/config/newer_college.yaml
```

The analyzer reads the configured input topics, compares RRD rows with rosbag2
metadata as an advisory, reports timing/point/conversion statistics, and
verifies the RRD.
