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

Inside the container, the current clean V3 foundation builds with:

```bash
source /opt/ros/humble/setup.bash
cd /workspace
colcon build --symlink-install --packages-select meridian_cmake
colcon test --packages-select meridian_cmake
colcon test-result --verbose
```

As implementation packages are added, build the smallest affected dependency
closure:

```bash
colcon build --symlink-install --packages-up-to <package>
colcon test --packages-select <package> --event-handlers console_direct+
colcon test-result --verbose
```

Generated `build/`, `install/`, `log/`, and `compile_commands.json` paths are
local artifacts and are ignored by Git.

## Foxglove Bridge

The development image contains the `foxglove_bridge` ROS node; the viewer runs
on another machine. Start the bridge inside the container:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

Connect the remote Foxglove viewer to:

```text
ws://<development-host>:8765
```

The host port can be changed through `FOXGLOVE_PORT` when starting Compose.

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

The Jetson Compose contains a separate Meridian image without Foxglove Bridge;
Foxglove is provided by its dedicated service. Start the stack and open a shell
in the localization container with:

```bash
docker compose -f compose.jetson.yaml up -d --build
docker compose -f compose.jetson.yaml exec meridian bash
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
