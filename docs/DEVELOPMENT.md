# Developing Meridian

This is the operational runbook for the V3 workspace. Architecture and
implementation decisions belong in [SYSTEM_SPECS.md](SYSTEM_SPECS.md), while
measured parameter and algorithm experiments belong in
[OPTIMIZE.md](OPTIMIZE.md).

## Development container

Docker Engine with the Compose plugin is the only host requirement. From the
repository root, build the image and start the persistent development container:

```bash
docker compose -f compose.dev.yaml up -d --build
```

Later starts can omit `--build`:

```bash
docker compose -f compose.dev.yaml up -d
```

The repository is mounted at `/workspace`. Open an interactive shell inside the
running container with:

```bash
docker compose -f compose.dev.yaml exec meridian bash
```

The container allocates 1 GiB of shared memory so ROS 2 transport has room for
the multi-megabyte organized OS0 point clouds during bag playback.

Commands can also be executed without opening a shell:

```bash
docker compose -f compose.dev.yaml exec meridian \
  bash -lc 'source /opt/ros/humble/setup.bash && colcon list'
```

The image user defaults to UID/GID `1000`. On a host using different IDs, pass
them when the image is first built:

```bash
USER_UID=$(id -u) USER_GID=$(id -g) \
  docker compose -f compose.dev.yaml up -d --build
```

## Build and test

Inside the container, build and test the complete first ingress slice and its
dependency closure with:

```bash
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install --packages-up-to meridian_apps
source install/setup.bash
colcon test --packages-up-to meridian_apps --event-handlers console_direct+
colcon test-result --verbose
```

Source ROS before enabling Bash `nounset`: ROS Humble's generated setup scripts
expect some variables to be initially unset. For later incremental work, build
the smallest affected dependency closure:

```bash
colcon build --symlink-install --packages-up-to <package>
colcon test --packages-select <package> --event-handlers console_direct+
colcon test-result --verbose
```

Generated `build/`, `install/`, `log/`, and `compile_commands.json` paths are
local artifacts and are ignored by Git.

## Foxglove Bridge

The complete launch below starts `foxglove_bridge` inside the container on its
fixed internal port `8765`; the viewer runs on the development host or another
machine. Connect it to:

```text
ws://<development-host>:8765
```

`FOXGLOVE_PORT` changes only the port published by the host. For example, this
maps host port `9000` to the bridge's unchanged container port and makes the
viewer URL `ws://<development-host>:9000`:

```bash
FOXGLOVE_PORT=9000 docker compose -f compose.dev.yaml up -d
```

## Generic bag-to-RRD run

The image contains the ROS 2 bag CLI and SQLite3 storage plugin used by the
Newer College bags. Confirm the selected bag and its topics with:

```bash
ros2 bag info /workspace/bags/newer-college/quad-easy
```

Build first, source the resulting workspace, and launch the complete session:

```bash
source /opt/ros/humble/setup.bash
cd /workspace
source install/setup.bash
mkdir -p out
ros2 launch meridian_apps bag_debug.launch.py \
  bag:=/workspace/bags/newer-college/quad-easy \
  config:=/workspace/src/meridian_apps/config/newer_college.yaml \
  rrd:=/workspace/out/quad_easy_ingress.rrd \
  rate:=1.0
```

This launch uses standard `ros2 bag play`; there is no Meridian replay
executable. By default it does not filter bag topics. The optional `topics`
argument supplies a whitespace-separated rosbag allow-list when a recording
contains unrelated custom message types; selected topics still use normal ROS
transport. The launch does not install QoS/count acceptance machinery. A short
startup delay gives DDS discovery time but is not a delivery guarantee; after
the player exits, a short grace period lets ROS callbacks settle and the node
drains its bounded LiDAR and Rerun queues. Use `rate:=0.25` for slower visual
inspection.

Foxglove can inspect the original bag topics, including the original
`/os_cloud_node/points` cloud. Meridian publishes no preview or diagnostics
topic. Its separate 1 Hz, 4096-point preview exists only inside the RRD. Raw
LiDAR scans are never copied into the recording.

## RRD validation

The image also contains the Rerun CLI, Viewer, and Python dataframe-query
dependencies in an isolated virtual environment at `/opt/rerun`. That
environment is first on `PATH`, so both commands below use the same SDK:

```bash
rerun --version
python3 -c 'import rerun as rr; print(rr.__version__)'
```

Meridian records files with the `.rrd` extension. Keep run artifacts below
`/workspace/out/` so they survive container recreation. Verify and analyze a
completed recording with:

```bash
rerun rrd verify /workspace/out/quad_easy_ingress.rrd
du -h /workspace/out/quad_easy_ingress.rrd
python3 tools/analyze_ingress_rrd.py out/quad_easy_ingress.rrd \
  --bag bags/newer-college/quad-easy \
  --config src/meridian_apps/config/newer_college.yaml
```

The analyzer runs Rerun's footer/manifest verification and reports schemas,
sensor-time rates and gaps, conversion timings, point statistics, preview
size, and RRD/bag size ratio. When both bag and config are supplied, it reads
the two configured topic counts from rosbag2 metadata and compares them with
the accepted RRD rows. A difference is reported as information and does not
fail an otherwise valid artifact: this is a post-run transport debug check,
not a runtime contract.

Open the same artifact in the Rerun viewer when deeper offline inspection is
useful:

```bash
rerun /workspace/out/quad_easy_ingress.rrd
```

The Rerun C++ SDK is acquired by the Meridian package through CMake
`FetchContent`, following the upstream integration. Consequently, the first
configure/build needs network access and takes longer while the SDK and its
Arrow dependency are fetched and compiled; later builds reuse CMake's build
tree cache.

## Container lifecycle

Inspect the service and its logs:

```bash
docker compose -f compose.dev.yaml ps
docker compose -f compose.dev.yaml logs -f meridian
```

Stop the container while preserving the image:

```bash
docker compose -f compose.dev.yaml down
```

Rebuild after changing `docker/dev/Dockerfile`:

```bash
docker compose -f compose.dev.yaml build --no-cache meridian
docker compose -f compose.dev.yaml up -d
```

## Jetson environment

The Jetson Compose is a deployment scaffold for a later phase. Its Meridian
runtime service is intentionally disabled while v3 is developed and tested on
Newer College, so the development Compose above is the supported environment
for the current slice.

```bash
docker compose -f compose.dev.yaml up -d --build
docker compose -f compose.dev.yaml exec meridian bash
```

## Implementation workflow

For each V3 implementation slice:

1. Start from the relevant decision and reference implementation named in the
   system specification.
2. Add the smallest production target with focused numerical and failure tests.
3. Validate the component independently before composing the full pipeline.
4. Run complete-sequence benchmarks before promoting an algorithm or default.
5. Record tunables, results, rejected attempts, and runtime effects in
   `OPTIMIZE.md`.
