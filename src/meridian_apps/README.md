# meridian_apps

`meridian_apps` contains executable composition roots. Algorithms remain in
their ROS-free packages; this package loads calibration, selects adapters,
owns output files, and reports exactly what was run.

## Offline Newer College multisensor localization

Build, source the workspace, then run:

```bash
ros2 run meridian_apps meridian_bag_localize \
  --bag /data/newer-college/quad-easy \
  --calib /data/newer-college/calib \
  --collection 1 \
  --sensor-mode full \
  --initialization dynamic \
  --output /tmp/meridian-quad-easy \
  --max-events 100000 \
  --max-bag-messages 500000 \
  --replay-mode scheduled \
  --playback-rate 1 \
  --visual-graph enabled
```

Collection `1` is used by the current Quad bags; collection `2` is used by the
current Park recording. Both replay bounds and an explicit `--replay-mode` are
mandatory. `scheduled` requires a finite positive `--playback-rate`: a rate of
`1` follows recorded arrival time, while larger values accelerate the producer
and preserve recorded ordering. `unpaced` forbids `--playback-rate` and runs
the same localization pipeline as quickly as the consumer can process it. Use
it for complete algorithm-throughput and ATE runs that must not fail merely
because the current implementation exceeds the recorded wall-clock budget:

```bash
ros2 run meridian_apps meridian_bag_localize \
  --bag /data/newer-college/quad-easy \
  --calib /data/newer-college/calib \
  --collection 1 --sensor-mode lidar-imu \
  --initialization dynamic \
  --output /tmp/meridian-quad-easy-unpaced \
  --max-events 100000 --max-bag-messages 500000 \
  --replay-mode unpaced
```

The bounded producer/consumer queue defaults to 128 events and 256 MiB, and
the rolling timing window defaults to 1024 samples. They can be overridden
with `--queue-count-capacity`, `--queue-byte-capacity`, and
`--timing-window-capacity`; every effective value is recorded in the run
profile. Reaching either bound is a normal diagnostic-prefix completion but
explicitly reported as partial coverage. Quad Easy acceptance instead requires
end-of-bag with 41,754 domain events and `full_bag_consumed=true`; short
prefixes are debugging tools, never accuracy evidence. The CLI exits with code
`5` if that prefix never produces an
initialized state, so automation cannot mistake an empty trajectory for a
localization result. The CLI requires one explicit sensor mode:

- `imu`: configure and deserialize only the IMU topic;
- `lidar-imu`: configure only IMU and LiDAR topics;
- `full`: configure IMU, LiDAR, and every calibrated camera.

Excluded modalities are removed from the replay profile before
deserialization and conversion. `--visual-graph` is valid only with `full`;
within that mode it defaults to `enabled`, while `disabled` retains camera
decode/tracking but suppresses visual knot and factor submission.

Initialization defaults to `dynamic`, which directly uses the bounded moving
LiDAR–IMU initializer and never runs a stationary probe. Selecting
`--initialization static` is an explicit replay-operator zero-motion assertion:
the IMU verifier must accept it and no dynamic fallback is allowed.
`supervised-auto` carries the same assertion but keeps the dynamic branch
authorized if verification rejects it. The translated ROS-free estimator
profile and typed prior source are recorded in the run report.

Outputs are replaced for each invocation:

- `trajectory_imu.tum`: accepted `T_odom_imu` states;
- `trajectory_base.tum`: the same states transformed with
  `T_odom_base = T_odom_imu * inverse(T_base_imu)`;
- `trajectory_imu_fixed_lag.tum`: finalized IMU-frame states plus the clean
  shutdown tail;
- `trajectory_base_fixed_lag.tum`: each state written after it leaves the
  fixed-lag window, followed by the final active tail at shutdown; use it to
  measure the settled local smoother trajectory without growing the live
  window;
- `navigation_diagnostics.csv`: one schema-v18 row per committed state with
  navigation state, covariance diagonal, graph work, pose-aware direct-ICP termination,
  live/finalized-map channel counts, accepted registration pose and right correction,
  independent live-factor and finalized-map-factor information scales,
  directional observability, physical information, bounded work counters and errors,
  accepted and rejected finalized-map owner/version/checksum/information provenance,
  accepted and rejected effective process-noise convergence thresholds,
  actual-objective Gauss--Newton/Cauchy trials and accepted scale, rejected-candidate
  corrections, visual graph work, and the moving initializer's
  sensor/calibrated/full ranks, prior-resolved mode count and class, per-modality
  residual, held-out compatibility, deskew-refinement, and covariance-calibration data;
- `run_report.json`: schema-v25 completion/error state, full versus partial coverage,
  wall and processed durations, replay/error counts, estimator rejection,
  degradation and registration counts, aggregate and per-camera visual
  activity, committed/rejected local-solver globalization aggregates,
  target-state-unavailable reseed counts, the last accepted direct-ICP
  diagnostics and last typed ICP error, every effective
  replay/calibration/local profile variable, and a sensor-mode proof covering
  configured topics, visual lanes, decode/event/ingest activity, and terminal
  isolation status. It also records the scheduled queue terminal and
  high-water states (count, logical bytes, oldest age, reject/drop/skip
  counters), producer/consumer total and rolling wall/thread-CPU timing, and
  driver-stage timing for IMU ingress, LiDAR enqueue, camera ingress,
  `processReady`, and report/output consumption. IMU ingress additionally
  reports bounded fusion-time newest-gap percentiles, the lifetime maximum
  newest gap, and the retained-sample high watermark. The report also carries
  bounded local-stage wall/thread-CPU distributions for initialization,
  deskew, deterministic ICP cloud preparation, target updates, registration,
  finalized-target maintenance, moving-batch refinement, and graph transactions. The
  finalized-target block accumulates finality, rollback, insertion, pruning, and freeze
  activity and records the terminal pending/ready/retained/version/checksum/health snapshot.

Schema v23 records the single pose-aware direct point-to-point path across disjoint live and
finalized-map target channels: fixed termination counts, complete bounded work counters,
directional observability, physical information, accepted frontend poses, and typed
registration errors. It also exposes the LiDAR health state and recovery
epoch transitions, shadow-mode quarantines, and bounded faulty-batch removals.

Any nonempty scheduled terminal queue, accepted/consumed count mismatch,
reject, drop, stale skip, or policy skip makes a scheduled benchmark fail.
Unpaced mode has no producer/consumer queue; its scheduled runtime block is
`null`, while driver and local-stage timing remain populated and its effective
queue/timing settings remain recorded for run-profile parity.
Live ROS diagnostics remain the responsibility of `meridian_ros`; this
offline application does not create a ROS node.

In `full` mode camera frames are tracked immediately with IMU rotation seeds
when support is available; admitted visual keyframes share the same graph-owned
state schedule as LiDAR. Isolated modes create no visual lanes. A camera event,
decode, ingest, or configured camera topic in either isolated mode is a
terminal sensor-mode violation; `imu` applies the same rule to LiDAR.
Moving starts use one bounded discrete path: rotation-only IMU deskew and an
adjacent direct point-to-point batch, full translation/rotation re-deskew from those provisional states,
then a final batch with a held-out LiDAR compatibility gate. The run report
records the applied covariance inflation for LiDAR registrations that reuse
raw IMU support.
If a rolling target outlives the active fixed-lag state window, the estimator
commits the current IMU/visual fallback, replaces the stale target with that
current sweep, and reports the typed reseed separately from ordinary
registration rejection.
When a process report references one state from both modalities, output is
normalized by state ID and written exactly once, with LiDAR disposition and
registration metadata retained. ROS 1 input is intentionally not read or
converted; the unconverted ROS 1 bags remain deferred benchmark variety.
