# Meridian

Meridian is a research-grade, discrete-time LiDAR–visual–inertial SLAM system with GNSS and robust global loop closure, targeting NVIDIA Jetson Orin and ROS 2 Humble.

The repository is currently at the v2 implementation-specification cutover: the coupled v1 estimator has been removed, and implementation restarts from reviewed contracts and benchmark gates. The sole architecture and implementation specification is [docs/SYSTEM_SPECS.md](docs/SYSTEM_SPECS.md).

Operational documentation:

- [Development](docs/DEVELOPMENT.md)
- [Testing and benchmarks](docs/TESTING.md)
- [Real-time debugging](docs/REALTIME_DEBUGGING.md)
- [Optimization/decision ledger](docs/OPTIMIZE.md)

The selected upstream research implementations used as reading and porting context live in `../slam-reference`; they are not Meridian build dependencies. Git commit `f5ca513158c95aaf88223486ec481c1d42730a21` is the exact legacy extraction baseline.

Current active build:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select meridian_cmake
```

See the development runbook before adding packages or dependencies.
