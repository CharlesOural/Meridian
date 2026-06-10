# Meridian — Development Environment

ROS and the whole SLAM toolchain never touch your host OS. Everything lives in a
container built from one reproducible spec (`docker/install-deps.sh`, wrapped by
two Dockerfiles). Break it → delete → recreate.

```
Meridian/                       # this repo IS the colcon workspace (mounted at /workspace)
├── docker/
│   ├── install-deps.sh         # THE dependency canon (spec 11) — one source of truth
│   ├── Dockerfile              # CPU base, cross-platform (Linux amd64 + Apple Silicon arm64)
│   └── Dockerfile.gpu          # Linux: CPU base + CUDA 12 toolkit (for nvblox / L4)
├── compose.yaml                # cross-platform base (CPU, no X11) — what the Mac runs
├── compose.linux-gpu.yaml      # Linux override: NVIDIA GPU + X11/RViz + USB sensors
├── setup-distrobox.sh          # Linux: build GPU image + create the Distrobox
├── dependencies.repos          # vcs deps built in-workspace (nvblox GPU, ouster-ros)
├── src/                        # the colcon source space (meridian_* packages go here)
└── bags/                       # benchmark bags + ground truth (see docs/DATASET.md)
```

**What the image contains** (the dependency canon from
[`docs/specs/11_build_system_libraries.md`](specs/11_build_system_libraries.md)):
ROS 2 Humble (desktop-full), C++20 toolchain (GCC 11), colcon/rosdep/vcstool,
Eigen 3.4, Sophus 1.22.10, **Ceres 2.1**, **GTSAM 4.2**, PCL 1.12, OpenCV 4,
small_gicp, yaml-cpp, linuxptp, evo, and the Foxglove bridge. The GPU image adds
the **CUDA 12 toolkit**; **nvblox** is built in the workspace from
`dependencies.repos`.

> **The one platform caveat — CUDA.** `meridian_map` (L4, the nvblox GPU
> TSDF+colour+mesh) is **CUDA-only with no CPU fallback** (spec 11 §7). It builds
> and runs only on the Linux/GPU image. **Apple Silicon has no CUDA**, so on the
> Mac you build everything *except* L4 (and the packages that link it). This is a
> hardware limit, the same category as RViz being Linux-only — not a bug.

---

## Linux (NVidia), Distrobox

```bash
./setup-distrobox.sh            # builds meridian:humble-gpu, creates the box
distrobox enter meridian
```

Then, one-time workspace bring-up from the repo root:

```bash
git submodule update --init          # vendor/ (basalt-headers, ikd-Tree, scancontext)
vcs import src < dependencies.repos              # nvblox (GPU) + ouster-ros
vcs custom src --git --args submodule update --init --recursive   # nested submodules (ouster-sdk)
rosdep install --from-paths src --ignore-src -y
CMAKE_BUILD_PARALLEL_LEVEL=6 colcon build --symlink-install \
    --parallel-workers 1 \
    --cmake-args -DCMAKE_CUDA_ARCHITECTURES="86;87"
```

> **Build parallelism.** `CMAKE_BUILD_PARALLEL_LEVEL` caps the compile threads
> (default **6** here); `--parallel-workers 1` builds one package at a time, so
> total concurrent compiles stay ≈ 6. The bare `colcon build` instead fans out to
> every core × every package, and nvblox's CUDA units peak at several GB each —
> enough to swap-freeze a 16 GB host. Lower the number if the build OOMs; raise
> it if you have RAM to spare. Same knob at image-build time:
> `--build-arg MERIDIAN_BUILD_JOBS=<n>`.

- RViz just works: `rviz2`
- GPU visible inside: `nvidia-smi`
- Nuke and recreate (~minutes): `distrobox rm -f meridian && ./setup-distrobox.sh`

### Or the explicit Docker path (same image, full GPU + X11)

```bash
xhost +local:root               # once per login, lets the container open RViz
docker compose -f compose.yaml -f compose.linux-gpu.yaml up -d
docker compose -f compose.yaml -f compose.linux-gpu.yaml exec meridian bash
# ...
docker compose -f compose.yaml -f compose.linux-gpu.yaml down -v   # destroy
```

---

## Mac (Apple Silicon)

Distrobox is Linux-only, so on Mac use **plain Docker** + **Foxglove Studio** for
viz. The CPU image builds natively as `arm64` (fast). No NVIDIA/CUDA on Mac.

```bash
docker compose up -d                        # base file only (CPU, no GPU)
docker compose exec meridian bash
```

Workspace bring-up (note: **skip the GPU layer**):

```bash
git submodule update --init
# do NOT `vcs import` nvblox on Mac — it needs CUDA.
CMAKE_BUILD_PARALLEL_LEVEL=6 colcon build --symlink-install \
    --parallel-workers 1 \
    --packages-skip meridian_map meridian_pipeline meridian_ros meridian_tools
```

This builds and unit-tests every **CPU algorithm layer** in isolation — L0
sensors, L1 preprocessing, and **L2 the CT front-end** (plus the cross-cutting
packages). The skip list covers `meridian_map` (L4, CUDA-only) and the
integration/ROS packages; integrated pipeline runs and bag replay happen on the
Linux/GPU box. (colcon just warns about skip names not yet in the tree.)

Visualization on Mac (and Linux too — the shared viz tool):

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Then open **Foxglove Studio** (free Mac app) → connect to `ws://localhost:8765`
for the 3D / map / point-cloud view. Tear down: `docker compose down -v`.

---

## Reproducibility

`docker/install-deps.sh` is the contract — the single place the dependency stack
is defined; both Dockerfiles run it. Add a library by editing that script (or
`dependencies.repos` for a workspace-built one) and rebuilding the image — never
`apt install` permanently inside a running container, or you lose reproducibility.
Version pins live in `install-deps.sh` (source builds) and follow spec 11 §3.

> **Pinning TODO.** A few refs are not yet locked to a SHA (`small_gicp` in
> `install-deps.sh`; `nvblox` and `ouster-ros` in `dependencies.repos`). Pin them
> per spec 11 §3 before any release/air-gapped build so nothing floats on a
> moving branch. (The `vendor/` submodules are already SHA-pinned by gitlink.)

---

## Testing with a dataset

The benchmark set is Newer College 2021 under `bags/newer-college/` (`quad-easy`
is the routine sequence). `docs/DATASET.md` covers download, `rosbags-convert`,
and the local layout; `docs/TESTING.md` covers running against it — the headless
run + ATE loop, or driving the live node with `ros2 bag play --clock`.
