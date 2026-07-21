# Meridian

Meridian is a research SLAM framework for long-range off-road robots, targeting
ROS 2 Humble and NVIDIA Jetson Orin.

The `v3` branch is a clean implementation baseline. Architecture and historical
evidence are kept separately:

- [V3 system specification](docs/SYSTEM_SPECS.md)
- [V1/V2 engineering retrospective](docs/V1_V2_RETEX.md)
- [Voxel-SLAM reference and comparison](docs/VOXEL_SLAM_COMPARISON.md)
- [Development runbook](docs/DEVELOPMENT.md)
- [Barakuda dataset runbook](docs/BARAKUDA_DATASET.md)
- [Optimization ledger](docs/OPTIMIZE.md)

The immutable V2 source snapshot is commit
[`1f7a789`](https://github.com/CharlesOural/Meridian/tree/1f7a789cddb1f27d768e6ed097c1f21a8bfbbf44).
Selected upstream research code lives in `../slam-reference`; it is reading and
porting context, not a Meridian build dependency.

Development environment:

```bash
docker compose -f compose.dev.yaml up -d --build
docker compose -f compose.dev.yaml exec meridian bash
```

The image includes the `foxglove_bridge` ROS node and exposes port `8765`. Launch
the bridge inside the container when required, then connect the Foxglove viewer
from another machine to `ws://<development-host>:8765`:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
```

The Jetson Compose remains a deployment scaffold; its Meridian runtime is
intentionally disabled while this first v3 slice is validated on Newer College.
