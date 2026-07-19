# Meridian

Meridian is a research SLAM framework for long-range off-road robots, targeting
ROS 2 Humble and NVIDIA Jetson Orin.

This repository snapshot preserves the V2 implementation and the evidence used
to design its clean successor:

- [V3 system specification](docs/SYSTEM_SPECS.md)
- [V1/V2 engineering retrospective](docs/V1_V2_RETEX.md)

The source packages, focused tests, benchmark scenarios, and trajectory tools
remain available as implementation evidence. Selected upstream research code
lives in `../slam-reference`; it is reading and porting context, not a Meridian
build dependency.
