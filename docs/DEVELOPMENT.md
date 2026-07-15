# Developing Meridian

This is the operational development runbook. The authoritative design and implementation order are in [SYSTEM_SPECS.md](SYSTEM_SPECS.md). If this file and the specification disagree, the specification wins.

Current repository state: the v1 implementation has been removed. Only shared workspace infrastructure remains while Slice 0 (contracts and replay harness) is built. Commands naming future packages are marked as such; do not mistake an empty workspace for a completed estimator.

## 1. Development rules

- Develop and test in the provided container; do not install persistent dependencies on the host.
- Keep algorithm packages ROS-free. ROS messages, QoS, TF, lifecycle, and conversion stay in `meridian_ros`/`meridian_msgs`.
- Keep GTSAM, CUDA, nvblox, OpenCV, and other implementation libraries out of public domain headers.
- Add a package only when it contains a real target and tests. Do not create empty architecture placeholders.
- All queues are bounded; all state has one writer; all cross-module records are immutable or move-only.
- A new tunable or changed default requires an entry in [OPTIMIZE.md](OPTIMIZE.md).
- A new runtime failure path requires a typed reason, a test, and an operator action in [REALTIME_DEBUGGING.md](REALTIME_DEBUGGING.md).
- Live and replay must call the same core implementation.

## 2. Repository layout

```text
Meridian/
  docs/SYSTEM_SPECS.md       sole architecture/implementation specification
  docs/DEVELOPMENT.md        this runbook
  docs/OPTIMIZE.md           experiment and tunable ledger
  docs/REALTIME_DEBUGGING.md operator/debug runbook
  docs/TESTING.md            test and benchmark runbook
  docker/                    CPU/GPU development images
  docker/jetson/             sensor-driver deployment containers
  src/                       active colcon packages only
  tools/                     generic active tools only
  bags/                      unversioned test data
  ../slam-reference/         selected upstream research code/paper context
```

The intended package tree is specified in section 5 of `SYSTEM_SPECS.md`. Git commit `f5ca513158c95aaf88223486ec481c1d42730a21` is the v1 extraction baseline; retrieve a selected file with `git show`, then adapt it behind the v2 contract and tests. Do not restore a whole legacy package.

## 3. Entering the environment

Linux/Jetson-capable development:

```bash
./setup-distrobox.sh
distrobox enter meridian
```

Explicit Docker alternative:

```bash
docker compose -f compose.yaml -f compose.linux-gpu.yaml up -d
docker compose -f compose.yaml -f compose.linux-gpu.yaml exec meridian bash
```

CPU-only development:

```bash
docker compose up -d
docker compose exec meridian bash
```

The workspace path is the repository root. Always source ROS before building:

```bash
source /opt/ros/humble/setup.bash
cd ~/Meridian
```

## 4. Dependency handling

`docker/install-deps.sh` is the source-build/apt dependency canon. `dependencies.repos` is only for source dependencies that must live in the workspace. Every dependency must have:

- upstream URL and license;
- package/feature that owns it;
- CPU/GPU and target-platform constraints;
- a clean-build test.

Dependency references are owned by their build or deployment definition. A benchmark or deployment manifest records what actually resolved for that run when the information is available. The research clones in `../slam-reference` are reading material, not build dependencies.

nvblox is not required for the current pre-dense implementation slices. Import and build it only when working on the later dense candidate:

```bash
vcs import src < dependencies.repos
```

Do not initialize old vendor submodules; v2 has none.

## 5. Build and test

During the pre-Slice-0 cleanup, the honest active Meridian build is:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=6 colcon build --symlink-install \
  --parallel-workers 1 --packages-select meridian_cmake
colcon test --packages-select meridian_cmake
colcon test-result --verbose
```

As slices add packages, build the smallest affected closure:

```bash
colcon build --symlink-install --packages-up-to <package>
colcon test --packages-select <package> --event-handlers console_cohesion+
colcon test-result --verbose
```

For a clean validation, remove generated `build/`, `install/`, `log/`, and the merged `compile_commands.json`, then rebuild. An old install overlay can make deleted dependencies appear available.

After a build:

```bash
source install/setup.bash
python3 tools/merge_compile_commands.py
```

Never rebuild a package while a replay/live process is using its symlink-installed library.

## 6. Required local gates

Run gates in proportion to the change:

1. format and warnings-as-errors;
2. unit/property/Jacobian tests for the changed target;
3. package dependency and public-header self-containment checks;
4. ROS-free include/dependency scan for core/local/global/dense/store;
5. deterministic component replay;
6. relevant fault cases;
7. full-sequence benchmark and target-Jetson timing for algorithm/default changes.

CPU debug profiles should include ASan/UBSan and deterministic single-thread modes. GPU changes require a clean dependency import, target-architecture build, analytic CPU/reference comparison where meaningful, CUDA error checks, and target-device tests.

## 7. Implementation workflow

For each slice in `SYSTEM_SPECS.md`:

1. freeze/review the public domain contracts;
2. add the smallest real package target and unit tests;
3. build a deterministic standalone component/replay harness;
4. implement the provisional default and declared challengers behind one interface;
5. run the decision benchmark with a registered manifest;
6. record artifacts/results in `OPTIMIZE.md`;
7. promote one default through code review;
8. add ROS conversion/lifecycle wiring only after core behavior is green;
9. update testing/debug runbooks with executable commands and expected reports.

## 8. Reference-reading workflow

Before implementing a research-derived seam:

- read the primary paper and the exact relevant source files in `../slam-reference`;
- record upstream coordinate/time/residual conventions;
- identify state ownership and hidden runtime assumptions;
- reproduce a minimal upstream behavior in an independent test;
- write Meridian's own interface and analytic/property tests;
- document adapted paths, license, upstream source, and intentional differences in the new package README.

The main routing table is section 2 of `SYSTEM_SPECS.md`. Add a reference repository only when a named implementation or benchmark lane needs its source.

## 9. Commit/review checklist

- [ ] Change follows a reviewed `SYSTEM_SPECS.md` decision or updates the spec first.
- [ ] No legacy compatibility API was restored.
- [ ] Public records include units, frames, time, revisions, and ownership.
- [ ] Failure/overflow behavior is typed and tested.
- [ ] No raw matrix crosses a covariance/information boundary without semantics.
- [ ] Deterministic replay remains green.
- [ ] Tunables/results are in `OPTIMIZE.md`.
- [ ] Operator-visible behavior is in `REALTIME_DEBUGGING.md`.
- [ ] Test manifest and artifacts are reproducible.
