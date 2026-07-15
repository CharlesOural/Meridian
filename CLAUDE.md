# Repository guidance

Meridian v2 is being rebuilt from reviewed, research-grounded contracts. Read [docs/SYSTEM_SPECS.md](docs/SYSTEM_SPECS.md) before implementation work; it is the only architecture/implementation specification. The files in `docs/` for development, testing, debugging, and optimization are operational runbooks and may not redefine the design.

## Current state

The v1 implementation was deliberately removed. Git commit `f5ca513158c95aaf88223486ec481c1d42730a21` is the extraction baseline. Retrieve and adapt only a selected algorithm/test seam; do not restore a legacy package or compatibility API.

Do not add empty target packages. Follow the implementation slices in the specification. At this cutover, only `meridian_cmake` is an active Meridian package.

## Engineering rules

- C++20, target-scoped dependencies, warnings-as-errors, deterministic tests.
- ROS-free core/local/global/dense/store. ROS types, QoS, TF, lifecycle, and conversions stay in adapters.
- No GTSAM, CUDA, nvblox, or ROS type in public domain contracts.
- `T_A_B` maps `B` into `A`; public covariances/information carry frame, tangent, units, and rank semantics.
- Single writer per mutable state; immutable snapshots/records across owners; every queue is bounded by count, bytes, and age.
- Live and replay call the same core. Expected failures use typed outcomes and observable capability transitions.
- All external code reuse records upstream URL, license, adapted paths, and behavioral differences.
- A threshold from a paper/reference is a benchmark seed, not a Meridian default.
- Changed tunables and algorithm choices are recorded in `docs/OPTIMIZE.md` with reproducible artifacts.
- New failure behavior gets tests and an operator action in `docs/REALTIME_DEBUGGING.md`.

## Environment

Use the repository containers; do not install dependencies permanently on the host. See `docs/DEVELOPMENT.md` for the current honest build and test commands. The reference repositories in `../slam-reference` are reading/porting context, not dependencies.

## Comments

Comments explain non-obvious invariants, units, numerical reasoning, and failure behavior. Research/design rationale and citations belong in the specification/package README, not scattered through implementation comments.
