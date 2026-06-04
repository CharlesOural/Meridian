# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**Meridian** is a from-scratch LiDAR-Inertial-Visual-GNSS SLAM system targeting
the **NVIDIA Jetson Orin**. It is proprietary (see `LICENSE`).

**Current state: docs + dev-environment only — there is no source code yet.**
`src/` is the (empty) colcon source space. The system is fully specified in
`docs/specs/` *before* being built. **The specs are the contract — read the
relevant spec first and do not contradict it.** This file is a map and a runbook;
it deliberately does **not** restate the design — the specs own that, so the
single source of truth never drifts. When in doubt, follow the spec, not this file.

## Where the design lives (authoritative — read these, not this file)

- `docs/SYSTEM_OVERVIEW.md` — end-to-end narrative (photons → mesh) and the *why*.
- `docs/specs/00_architecture.md` — package layout & dependency rules (§2, §4),
  threading (§11), the no-ROS rule (§1), the `KeyframePacket` contract (§6), build
  order (§13), and the **non-negotiable invariants** (§14). Start here to build.
- `docs/specs/01_interfaces_and_data_types.md` — **the canonical type authority**:
  every value type + interface, the box-plus convention, and tangent ordering
  (§3, §6). Later specs amend this one; where any doc disagrees, spec 01 wins.
- `docs/specs/11_build_system_libraries.md` — library canon, version pins, the
  colcon workspace, and the CUDA/nvblox build.
- `docs/specs/02..10` — per-layer specs: sensors/time, preprocessing, front-end,
  back-end, mapping, loop closure, calibration, debug, evaluation.
- `docs/grounding/` — `file:line` + paper citations into the **reference** systems
  (FAST-LIO2, FAST-LIVO2, nvblox, iSAM2, Coco-LIC); these describe *those* systems,
  not Meridian's API. `docs/course/` — textbook-depth math.
- `DEVELOPMENT.md` — the dev-environment runbook (summarised below).

## Dev environment & commands

The toolchain lives in a container — never install on the host. `docker/install-deps.sh`
is the single source of truth for the dependency stack (ROS 2 Humble, C++20,
Eigen/Sophus/Ceres 2.1/GTSAM 4.2/PCL/OpenCV/small_gicp, clang tooling); both
Dockerfiles run it. CUDA/nvblox is Linux-GPU only, so Apple Silicon builds
everything except `meridian_map` and the packages that link it.

**`DEVELOPMENT.md` is the runbook** — container setup (distrobox / Docker), the
one-time workspace bring-up (`git submodule update`, `vcs import src < dependencies.repos`),
build invocations, and viz. Follow it rather than duplicating commands here. Tests
run via ament/GoogleTest: `colcon test`; `--packages-select <pkg>` for one package;
`--ctest-args -R <name>` for one test.

CI gates to keep green are defined in spec 11 §9.3 / spec 00 §9.4 (no-ROS grep,
dependency lint, no CUDA outside `meridian_map`, clang-tidy/clang-format `-Werror`).

## Conventions

- C++20, `CMAKE_CXX_EXTENSIONS OFF`. One public class ≈ one header + one `.cpp`.
- Public headers under `include/meridian/<module>/`; interfaces are `I*.hpp`
  pure-virtual + a factory free function; impls live in `src/` and are not exported.
- **Tangent ordering differs by type and is a classic silent bug.** Do not rely on
  memory — consult spec 01 §3.1/§3.4 and §6 (and the per-type comment) before
  touching any covariance, information matrix, or Jacobian.
- Validation datasets and the replay==live harness: see `docs/DATASET.md` and spec 10.

## Comment discipline

Comments describe **what the code does and why the logic is correct** — control
flow that isn't obvious, invariants, edge cases, units, and technical/mathematical
reasoning (derivations, why a formula holds, numerical caveats). That is the only
thing they are for.

Do **not** put in comments:
- **External references** — no `per spec 11 §3`, `see grounding/...`, paper/equation
  citations, or doc section pointers. The code must read self-contained.
- **Project rationale / decisions** — no "chosen for reproducibility", "matches the
  reference", "simplicity mandate". Design rationale belongs in the specs/docs and
  in commit messages, never in code.
- **Environment or personal context** — no "your daily driver", "RTX 3070",
  "the Mac friend", build-host asides.

Bad:  `// Ceres 2.1 needed for the Manifold API (spec 11 §3)`
Bad:  `// clear-and-reintegrate because the TSDF isn't reversible (see grounding 07)`
Good: `// running-average TSDF can't subtract a stale pose, so clear the region first`
Good: `// r = n·(R·p_L + t) + d  — signed point-to-plane distance in world frame`

Keep them sparse: prefer self-explanatory names and structure; comment the
non-obvious, not the line-by-line.
