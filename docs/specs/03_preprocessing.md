# 03 — L1 Per-Sensor Preprocessing (Implementation Spec)

> **Spec status:** normative implementation spec for **layer L1** (`meridian_preprocess`).
> This document tells you *how to build* the per-sensor conditioning stage: the data
> structures, the interfaces, the algorithms (in pseudocode), the parameters, the
> failure modes, and the debug hooks. It is **not** a tutorial — the front-end math
> (IMU model, deskew transform) is derived in
> `04_frontend_estimation.md`. Read that for the *why*; this spec is the *what* and
> the *exactly how*. Reference-system grounding for the engineering/ROS patterns this
> spec validated against is in Appendix R.
>
> **What L1 is, in one line.** L1 turns the raw, time-synced samples L0 produces for
> **one LiDAR + one IMU + one camera + GNSS** into clean, estimator-ready
> measurements for the **discrete LIO** front-end (L2). It does per-modality
> conditioning in parallel; deskew is **not** an L1 concern — the front-end warps
> each point internally from the group's own IMU samples (§4).
>
> **Dependencies (read first, this spec is consistent with them):**
> - `00_architecture.md` — package layout, deskew placement (§7), threading (§11),
>   the telemetry bus (§10), `Config` (§8).
> - `01_interfaces_and_data_types.md` — `ImuSample`, `LidarScan`/`LidarPoint`,
>   `CameraFrame`, `GnssFix`, `IntrinsicsCamera`, `Extrinsic`, `CalibrationSet`,
>   `ILidarPreprocessor` (§7.2), `Frame`, `Timestamp`, ownership/threading
>   conventions. **Every type named here is defined there.**
> - `04_frontend_estimation.md` — the consumer. L2 owns deskew entirely; L1 hands
>   it an *undeskewed* validated scan plus per-point time.
>
> **Grounding rule.** Code claims are cited `file:line` against the reference trees
> in `/home/user/slam-reference/`; the cited files were read byte-accurately. Paper
> equations are referenced by their published numbers. The engineering/ROS reference
> grounding is collected in Appendix R (non-normative).
>
> **Scope note.** Meridian is a **direct** system (point-to-plane LiDAR + sparse-direct
> photometric, FAST-LIVO2 lineage). L1 therefore does **no LOAM-style feature
> extraction** (`01_*` §7.2: "NO feature extraction by default"). The reference
> `Preprocess::give_feature` / `plane_judge` machinery
> (`FAST_LIO/src/preprocess.cpp:483-957`) is documented in §3.6 as a *deliberately
> rejected* path so a future maintainer does not re-import it by reflex.

---

## Table of contents

1. [What L1 is, and the rules that shape it](#1-what-l1-is-and-the-rules-that-shape-it)
2. [Module layout, interfaces, threading](#2-module-layout-interfaces-threading)
3. [LiDAR preprocessing](#3-lidar-preprocessing)
4. [Deskew: where it lives (L2), and what L1 guarantees instead](#4-deskew-where-it-lives-l2-and-what-l1-guarantees-instead)
5. [Camera preprocessing: photometric calibration + rectification](#5-camera-preprocessing-photometric-calibration--rectification)
6. [IMU preprocessing: passthrough + Allan-variance noise](#6-imu-preprocessing-passthrough--allan-variance-noise)
7. [GNSS preprocessing: gating + spoofing check](#7-gnss-preprocessing-gating--spoofing-check)
8. [Parameters (the L1 slice of `Config`)](#8-parameters-the-l1-slice-of-config)
9. [Failure modes (consolidated)](#9-failure-modes-consolidated)
10. [Debug hooks (the L1 telemetry surface)](#10-debug-hooks-the-l1-telemetry-surface)
11. [Test plan](#11-test-plan)
12. [Appendix A — `L1Config` schema](#appendix-a--l1config-schema)
13. [Appendix B — the LiDAR adapter seam](#appendix-b--the-lidar-adapter-seam)

---

## 1. What L1 is, and the rules that shape it

L1 turns the **raw, time-synced sensor samples** L0 produces into **clean,
estimator-ready measurements** for L2, doing per-modality conditioning *in
parallel* and *independently* — there is no cross-modality coupling anywhere in
L1. Concretely, per modality:

| Modality | L0 gives L1 (`01_*` §4) | L1 gives L2 |
|---|---|---|
| LiDAR | raw `LidarScan` (sensor frame, per-point `t_offset_ns`, not deskewed) | filtered+validated+surf-voxel-downsampled `LidarScan` (still sensor frame, points time-sorted, **undeskewed**) — L2 registers each point at its true time |
| Camera | raw `CameraFrame` (Bayer/Mono, exposure/gain) | photometrically-normalised, rectified `CameraFrame` + a built image pyramid |
| IMU | raw `ImuSample` (specific force + rate, bias included) | the same `ImuSample`, *passed through*, **plus** an `ImuNoiseModel` (load-time) and a saturation flag |
| GNSS | raw `GnssFix` (LLA + ENU cov + fix type) | a **gated, anti-spoof-checked** `GnssFix` with an explicit `accept/reject + reason` verdict |

**Rule 1 — deskew lives inside L2, full stop.** The front-end warps each LiDAR
point to the sweep-end instant with a constant-screw model built from the
group's own IMU samples (`04_*`; `00_*` §7.1). L1 therefore hands L2 the
**undeskewed** validated scan plus the mandatory per-point `t_offset_ns` —
time-sorted, so the warp can stream the points in order — and **does not**
transform points into a common frame. L1 holds no deskew routine, no trajectory
query, and no cold-start policy; nothing in `meridian_preprocess` ever needs a
pose, which is what keeps the build graph and the data flow strictly bottom-up
(`00_*` §4 R5).

**Rule 2 — L1 is sensor-conditioning behind a clean interface, not a switch
statement.** The reference `Preprocess::process` is a `switch (lidar_type)` over
`OUST64 / VELO16 / ...` (`FAST_LIO/src/preprocess.cpp:71-88`); FAST-LIVO2 grew that
same switch to seven cases (`FAST-LIVO2/src/preprocess.cpp:62-91`). Meridian runs **one
LiDAR**, so there is exactly one active decode path — but decode still lives behind
a small `ILidarAdapter` seam (Appendix B) so the native-layout quirks of *this*
sensor (Ouster OS1-128) are isolated from the shared, sensor-agnostic validity /
deskew / telemetry stages. The seam exists for cleanliness and unit-testing of
decode, not to host a menu of sensors. (Multi-LiDAR is a one-line future extension
behind the same seam — not designed now.)

---

## 2. Module layout, interfaces, threading

### 2.1 Files (`meridian_preprocess`, no ROS — `00_*` §1.1, §2)

```
meridian_preprocess/
  include/meridian/preprocess/
    ilidar_preprocessor.hpp  # = 01_* §7.2 ILidarPreprocessor
    lidar_adapter.hpp        # ILidarAdapter (per-sensor decode seam, Appendix B)
    lidar_validity.hpp       # range/intensity/self-hit gate + organized layout
    image_preprocessor.hpp   # photometric calib + rectify + pyramid
    imu_preprocessor.hpp     # passthrough + saturation flag (noise model lives in CalibrationSet)
    gnss_gate.hpp            # gating + spoofing check + verdict
  src/
    <one .cpp per header>
    adapters/ouster.cpp      # the ILidarAdapter impl (Appendix B)
  test/   # §11
```

`meridian_preprocess` depends only on `meridian_common`, `meridian_config`,
`meridian_debug`, `meridian_calib`, Eigen, OpenCV (image
only) — never `rclcpp` (`00_*` §4 R1/R4, enforced by the no-ROS CI gate §9.4).

### 2.2 Interfaces

`ILidarPreprocessor` is the boundary contract already declared in `01_*` §7.2.
This spec adds two *internal* collaborators (both swappable, both behind small
interfaces per `00_*` §5):

```cpp
// include/meridian/preprocess/lidar_adapter.hpp
// Per-sensor DECODE only: native packet -> meridian::LidarScan with per-point time.
// (Replaces FAST_LIO's switch + per-type handler; preprocess.cpp:71-88. One impl:
//  Ouster OS1-128, the rig's single LiDAR.)
class ILidarAdapter {
public:
  virtual ~ILidarAdapter() = default;
  // raw: a typed view of the sensor's native point layout (ring, t, range...).
  // Returns a LidarScan with: xyz in sensor frame, t_offset_ns set, ring/range/
  // intensity populated, points NOT yet validity-filtered (validity is a separate,
  // shared stage).
  virtual LidarScan decode(const RawLidarBuffer& raw, const SensorInfo&) const = 0;
  virtual SensorModel model() const = 0;     // OusterOS1_128 (for time-unit, ring count)
};

// include/meridian/preprocess/lidar_validity.hpp
// Shared (sensor-agnostic) validity gate + organized-cloud construction.
class LidarValidityFilter {
public:
  // nominal_rate_hz comes from the LiDAR's SensorInfo (spec 02 §3.2); it sets the
  // sweep_duration floor (§3.5a) and nothing else here.
  LidarValidityFilter(const L1Config::Lidar&, const Extrinsic& T_imu_lidar,
                      double nominal_rate_hz);
  // In-place: drops invalid points, sorts by t_offset_ns, optionally builds organized
  // index, floors sweep_duration (§3.5a). Fills ValidityStats for telemetry (§10).
  LidarScan apply(LidarScan&& in, ValidityStats* out_stats) const;
};
```

**Where deskew is actually invoked.** Per `04_*` and `00_*` §7.1, the
**front-end (L2) owns deskew entirely**: it warps each point to the sweep end with
a constant-screw model built from the group's own IMU samples, inside `ingest`.
L1 contains no deskew interface, no transform routine, and no trajectory consumer;
its only obligations to the warp are the per-point `t_offset_ns`, the time-sorted
point order, and the `sweep_duration` floor (§3.5a).

### 2.3 Threading

All of L1 runs on the **L0/L1 stage thread T1** (`00_*` §11.1: "L0/L1 stage:
time-sync + preprocess"). It is *thread-confined* — not required to be internally
thread-safe (`00_*` §11.2). The one cross-thread fact: L0 hands L1 clouds/images as
**Shared-immutable** `shared_ptr<const ...>` (`01_*` §2.4, §4.2, §4.3); L1 **must
copy-on-write** when it filters (it allocates a new Shared-immutable buffer; it must
never mutate the input bytes, because L4's retained store may alias them —
`01_*` §4.2). The IMU passthrough does *not* copy (it forwards the Value
`ImuSample`).

---

## 3. LiDAR preprocessing

LiDAR L1 is four sequential stages: **decode** (per-sensor, §3.1) → **validity
gate** (shared, §3.2) → **surf-voxel downsample** (shared, §3.2a) →
**organized-cloud construction** (optional, §3.3). The output is an *undeskewed*,
density-reduced, validated scan; deskew (§4) happens inside L2, not inline here.

### 3.1 Decode (per-sensor adapter)

The adapter's only job is to turn the native packet layout into `meridian::LidarScan`
with the contract `01_*` §4.2 requires: `xyz` in the **sensor** frame (raw, not
deskewed), `t_offset_ns` **signed ns from `stamp_start`**, `ring`, `range`,
`intensity`. The reference does this decode but buries it in the same function as
filtering and time-unit conversion (`oust64_handler`,
`FAST_LIO/src/preprocess.cpp:189-282`). We separate decode from filtering.

**Per-point time is mandatory and is the contract anchor.** Meridian stores
`t_offset_ns` as `int32` ns from scan start (`01_*` §4.2). The reference abuses
the PCL `curvature` field to carry per-point time in **milliseconds**
(`added_pt.curvature = pl_orig.points[i].t * time_unit_scale`,
`FAST_LIO/src/preprocess.cpp:226,275`, with `time_unit_scale` chosen from a
`TIME_UNIT{SEC,MS,US,NS}` enum at `:52-69`). That implicit, undocumented
ms-in-`curvature` convention is a classic silent-unit gotcha. **Meridian uses an
explicit, typed `int32 t_offset_ns` field** — no unit ambiguity, no field abuse.
The adapter converts the sensor's native unit to ns once, here.

**Native time-unit handling (Ouster).** Ouster `Point::t` is a `uint32` ns offset
(`FAST_LIO/src/preprocess.h:63`, `FAST-LIVO2/include/preprocess.h:89`), so the
adapter does `t_offset_ns = int32(pt.t)` directly — no scale, no ambiguity. This is
the rig's single LiDAR; there is no per-type unit table to maintain.

**Missing per-point time → reject (default).** A scan whose `t_offset_ns` are all
zero cannot be registered at true point times and is **rejected at L1**
(`01_*` §4.2: "A scan without per-point time cannot be deskewed correctly and is
rejected at L1"), with `event(WARN, "lidar/no_point_time")`. (The reference's
azimuth-reconstruction path for time-less Velodyne dumps,
`FAST_LIO/src/preprocess.cpp:304-322,346-373`, is not relevant for an Ouster that
always reports per-point ns; it is not implemented.)

```text
ALGORITHM decode(raw, info):                       # ILidarAdapter
  scan.stamp_start = info.scan_start_ns            # from L0 PTP stamp
  scan.sensor_frame= info.frame                    # os_sensor0
  pts = reserve(raw.count)
  for p in raw:                                    # native iteration order preserved
     q.xyz       = (p.x, p.y, p.z)                 # sensor frame, raw
     q.intensity = p.intensity
     q.ring      = p.ring
     q.range     = p.range>0 ? p.range : norm(q.xyz)
     q.t_offset_ns = int32(p.t)                    # Ouster: native ns offset, no scale
     pts.push(q)
  scan.sweep_duration = max(pts.t_offset_ns)        # sweep-end offset; offsets are
                                                    # not guaranteed column-ordered, and
                                                    # subtracting the min would move t_end
  scan.points = make_shared<const vector<LidarPoint>>(move(pts))
  return scan
```

### 3.2 Validity gate (range / intensity / self-hit), shared across stages

This is the sensor-agnostic filter. Five checks, in this order (cheapest first, so
most points die early):

1. **NaN/Inf reject.** `!isfinite(x|y|z)` → drop. The reference does this only in
   some handlers (`std::isnan` guards, `FAST-LIVO2/src/preprocess.cpp:727`,
   `Point-LIO/src/preprocess.cpp:383`) and *not* in others — an inconsistency we
   remove by making it the first, unconditional check.

2. **Blind / near-range reject (self-hit core).** Drop points with
   `range² < blind²`. This is the reference `blind` check
   (`range < (blind*blind)`, `FAST_LIO/src/preprocess.cpp:210,264`;
   `blind_sqr`, `FAST-LIVO2/src/preprocess.cpp:265`). Default `blind = 0.5 m`
   (Meridian `Config` value, `00_*` §8.2) vs the reference's tiny `0.01 m` default
   (`FAST_LIO/src/preprocess.cpp:7`) — Meridian's rig is larger and self-occludes
   more, so the blind radius is bigger.

3. **Far-range reject.** Drop points with `range > det_range` (the reference
   `det_range` clip, present in Point-LIO/FAST-LIVO2:
   `range > det_range*det_range`, `Point-LIO/src/preprocess.cpp:383`). Default
   `det_range = 120 m` (Ouster OS1-128 useful range). Removes atmospheric returns
   and wild outliers before they reach registration.

4. **Self-hit geometry reject (the rig structure).** The architecture's L1
   responsibility includes self-hit removal — points that hit the
   vehicle/mast/sensor housing. The reference has **no** explicit self-hit model
   (blind radius is its only proxy). Meridian adds an explicit, calibrated **self-hit
   mask**: a set of axis-aligned boxes / cylinders in the **sensor frame** (loaded
   from `CalibrationSet`/`L1Config`) describing where the rig occludes the LiDAR. A
   point inside any mask volume is dropped. This catches consistent returns off the
   platform that the blind radius alone misses.

5. **Intensity / reflectivity sanity (optional).** Drop points with intensity
   outside `[intensity_min, intensity_max]` when `intensity_gate=true`
   (default false). Zero-intensity returns are often dropout/no-return artifacts;
   saturated returns (retroreflectors) destabilise the photometric colourisation
   later. Off by default because Ouster intensity is generally trustworthy; the
   gate exists for dirty environments.

A surviving point is **kept once per `point_filter_num`** (decimation): the
reference uses `i % point_filter_num` (`FAST_LIO/src/preprocess.cpp:260`) or
`valid_num % point_filter_num` (`:168`). Meridian uses the **valid-count** form
(`valid_num % N == 0`) so decimation is uniform over *valid* points, not over raw
indices (the index form decimates unevenly when many points are invalid). Default
`point_filter_num = 3` (`00_*` §8.2).

After filtering, **sort points ascending by `t_offset_ns`** — a precondition for
L2's streaming per-point deskew (the reference sorts by
`curvature`, `FAST_LIO/src/IMU_Processing.hpp:234`; FAST-LIVO2 sorts in-handler,
`FAST-LIVO2/src/preprocess.cpp:336-338`).

```text
ALGORITHM validity_apply(scan, cfg, mask):         # LidarValidityFilter
  out = []; valid = 0; stats = {}
  for p in scan.points:
    if !isfinite(p.xyz):                stats.n_nan++;        continue
    r2 = dot(p.xyz,p.xyz)
    if r2 < cfg.blind*cfg.blind:        stats.n_blind++;      continue
    if r2 > cfg.det_range^2:            stats.n_far++;        continue
    if mask.contains(p.xyz):            stats.n_selfhit++;    continue
    if cfg.intensity_gate and (p.intensity<cfg.i_min or p.intensity>cfg.i_max):
                                        stats.n_intensity++;  continue
    valid++
    if valid % cfg.point_filter_num != 0:  continue           # decimate by valid count
    out.push(p)
  sort(out by t_offset_ns ascending)
  stats.n_in = scan.points.size(); stats.n_out = out.size()
  # surviving sweep-end, floored so a sparse/over-filtered scan cannot collapse the
  # deskew horizon below a sane minimum (§3.5a):
  surviving_end = out.empty() ? 0 : max(out.t_offset_ns)
  floor_ns      = round(cfg.sweep_floor_frac * 1e9 / nominal_rate_hz)   # rate from SensorInfo
  scan.sweep_duration = max(surviving_end, min(floor_ns, scan_in.sweep_duration))
  scan.points = make_shared<const vector<LidarPoint>>(move(out))
  return scan, stats
```

### 3.2a Surf-voxel downsample (deterministic representative per cell)

`point_filter_num` (§3.2) thins by valid-count, which does not bound *spatial*
density — a near surface still arrives at full resolution while a far one is sparse.
The architecture's L1 contract is a **downsampled** cloud (`00_*` §3 L1 row), so
after the validity gate L1 applies a **surf-voxel downsample**: bucket the surviving
points into a uniform voxel grid of edge `voxel_surf_m` (the `preprocess.voxel_surf_m`
key, `00_*` §8.2; default 0.5 m, with `tsdf_voxel_m <= voxel_surf_m` enforced in
`Config::validate`, `00_*` §8.3) and keep **at most `surf_max_pts` points per voxel**
(default 1; a small cap > 1 retains a little within-cell structure for normal
estimation when `organize` is on). This is the density the front-end's
registration consumes; it is **not** the persistent registration map (that store, its
per-voxel point cap, and its eviction policy are L4/Tier R, owned by `06_mapping.md`
§3.2 / §3.2a — L1 only conditions the *incoming* sweep).

**The representative is chosen deterministically — never newest-wins.** The obvious
implementation (keep the last point that lands in a cell, overwrite as you iterate)
makes the surviving set a function of *iteration order*: two runs over the same scan,
or a reordered decode, can keep different points and produce a different cloud — an
order-dependent swing in behaviour that the regression harness cannot tolerate. L1
instead selects per cell by a **fixed, order-independent rule**:

- **`surf_max_pts == 1` (default):** keep the point **nearest the voxel centre**,
  ties broken by ascending `t_offset_ns` then by the point's decode index. This is
  the deterministic analogue of the reference systems' on-insert map downsample
  (which keeps the point nearest the cell centre) and depends on no
  traversal order.
- **`surf_max_pts > 1`:** when a cell receives more than the cap, retain a bounded
  sample by **reservoir sampling with a per-process seeded RNG** (`surf_seed`,
  default 0): the draw is uniform over the cell's points, unbiased toward early or
  late returns. This mirrors the registration store's seeded reservoir eviction
  (`06_mapping.md` §3.2a). The reservoir draw is the only randomness in L1 and is
  confined to the eviction decision; it never touches a point's coordinates or time.

The downsample runs after the validity gate (so dead points never occupy a cell)
and keeps the gate's already-floored `sweep_duration` (§3.5a): thinning only removes
points, none of which exceeds the anchor, so the horizon stays valid. It preserves
the time-sort (re-sort the survivors by `t_offset_ns`, since cell bucketing reorders
them) and the copy-on-write rule (§2.3 — a new Shared-immutable buffer, input bytes
untouched). `ValidityStats` gains `n_voxel_out` (points after downsample) alongside
`n_out` (points after the validity gate).

### 3.3 Organized-cloud construction (optional, ring-indexed)

For the ring-structured Ouster (`ring` ∈ [0, N_SCANS)), L1 can build an
**organized** layout — a `ring × column` 2-D index over the validated points —
behind a flag (`organize=true`, default **false** for the direct pipeline). The
reference builds per-ring buffers `pl_buff[ring]`
(`FAST_LIO/src/preprocess.cpp:227-229`) *only* for the (rejected, §3.6) feature
path. Meridian keeps an organized index available because two *future* consumers want
it cheaply: (a) a ring-aware self-hit / shadow filter, and (b) image-style
neighbour lookups for normal estimation. It is off by default because direct
point-to-plane registration (L2/L4) queries a spatial voxel map, not a ring grid,
so the organized layout is dead weight in the default path. When on, it is stored
as a side structure (`OrganizedIndex { uint16 n_rings; vector<Range> row_spans; }`),
never as a reordering of `scan.points` (which must stay time-sorted for the
front-end's streaming deskew).

### 3.4 Intensity handling

Intensity is **carried through unchanged** by default (calibrated upstream by the
Ouster firmware). L1 does *not* do intensity calibration for LiDAR (unlike the
camera, §5) because Ouster reflectivity is already sensor-calibrated and the
direct LiDAR residual is geometric (point-to-plane), not photometric — intensity
is used only for (a) the optional intensity gate (§3.2 check 5) and (b) downstream
map colour/visualisation. We keep the field so Scan-Context-style intensity
descriptors (L5) and intensity-aided colourisation (L4) have it.

### 3.5 What L1 outputs to L2 (LiDAR)

A validated `LidarScan`: sensor frame, raw `xyz`, time-sorted, decimated,
surf-voxel-downsampled (§3.2a), self-hit-free, with `sensor_frame` set and
`sweep_duration` recomputed post-filter
as the **max surviving `t_offset_ns`**, floored as in §3.5a (the sweep-end offset,
never max−min of the survivors): culling the earliest columns must not pull `t_end`
below points still in the cloud, and a point whose offset exceeds the recomputed end
would fall outside the deskew horizon. (When nothing is dropped, `point_filter_num
== 1`, no voxel cell exceeds `surf_max_pts`, and the input is already time-sorted,
the original buffer and its `sweep_duration` pass through untouched to avoid a
copy.)
**Undeskewed** — L2 warps each point at its true `t_offset_ns` internally
(§4 explains why deskew lives there). Plus a `ValidityStats` for telemetry
(§10).

### 3.5a `sweep_duration` floor (the deskew-horizon guard)

`sweep_duration` is the span the deskew warp must cover: `t_end = stamp_start +
sweep_duration` is the anchor instant the front-end warps every point to (§4).
Recomputing it as
the *max surviving* `t_offset_ns` is correct for the common case, but a scan that
filtering reduces to a few early-column points — a near-empty sweep through a
tunnel, a self-hit storm, an aggressive `point_filter_num` on a sparse return — can
collapse `sweep_duration` to a small value (or, when every survivor shares the first
column, to ~0). A collapsed anchor span misstates the sweep's true extent and
distorts the warp interval even though the trajectory was fine.

L1 therefore **floors** the recomputed value at a fraction of the nominal sweep
period:

```
floor_ns        = round(sweep_floor_frac / nominal_rate_hz * 1e9)
sweep_duration  = max( max_surviving_t_offset_ns,
                       min(floor_ns, original_sweep_duration) )
```

with `sweep_floor_frac` default `0.5` (§8 / Appendix A) and `nominal_rate_hz` from
`SensorInfo`. Two clamps, both load-bearing:

- The floor only ever **raises** `sweep_duration` toward a sane minimum, so the
  surviving sweep-end (an upper bound on real point times) is always honoured — no
  surviving point is ever pushed outside the horizon.
- The floor is itself capped at the scan's **original** (pre-filter, L0)
  `sweep_duration`: the physical sweep never lasted longer than the device reported,
  so the anchor is never invented past the true sweep end. The padding it adds — at
  most up to half a nominal period, and never beyond the real sweep — only extends
  the warp interval by a few hundred microseconds past the last surviving point,
  which the constant-screw model covers by construction.

The floor never changes any point's `t_offset_ns`, only the anchor span; the
front-end's per-point warp is unaffected for every real point.

### 3.6 Explicitly rejected: LOAM feature extraction

The reference `give_feature` / `plane_judge` / `edge_jump_judge`
(`FAST_LIO/src/preprocess.cpp:483-957`) classify points into
`{Real_Plane, Edge_Jump, Wire, ...}` and emit a reduced `pl_surf`/`pl_corn`. This
is the LOAM heritage and is **gated off** in the reference itself for the LIO path
(`feature_enabled=false` is the default; `FAST_LIO/src/preprocess.cpp:7`). Meridian
**does not implement it.** Rationale, grounded: with a 128-beam Ouster producing
millions of points/s, discarding ~99% to keep a few hundred features is
indefensible, and direct registration is more robust in feature-poor scenes
(`04_frontend_estimation.md` §3.1). This subsection exists so a maintainer does not
"helpfully" port `give_feature` back in — it is a *decision*, not an omission.

---

## 4. Deskew: where it lives (L2), and what L1 guarantees instead

Deskew (motion compensation) is performed **inside the front-end**, not in L1.
There is no deskew interface, no provider, no L1 warp routine, and no L2→L1 data
flow of any kind.

### 4.1 The L2 mechanism, in one paragraph (normative home: spec 04)

The front-end models the body's motion across one sweep as a **constant screw**:
the interval-mean angular rate from the group's own IMU samples plus the
propagated body-frame velocity. For a point sampled at offset $\delta t$ before
the sweep end it applies the relative motion
$\big(\mathrm{Exp}(-\omega\,\delta t),\; -v\,\delta t\big)$ (composed with the
LiDAR→body extrinsic) to express the point in the body frame **at `t_end`**, then
registers the warped point against the local map. The warp happens per point, per
sweep, inside `ingest`; the reference's separate backward-propagation pass
(`FAST_LIO/src/IMU_Processing.hpp:323-335`) is replaced by this single
closed-form screw. Accuracy of the constant-screw approximation under
high-frequency motion is an L2 concern, measured by the evaluation harness
(spec 10), not an L1 contract.

### 4.2 What L1 guarantees the warp (the actual contract)

L1's deskew obligations are exactly three output properties, all already
specified in §3:

1. **Per-point time** — every surviving point carries a valid `t_offset_ns`
   (§3.2; a scan without per-point time is rejected).
2. **Time-sorted order** — points ascend by `t_offset_ns` (§3.2), so the warp
   can stream the sweep in order.
3. **A sane anchor span** — `sweep_duration` is recomputed post-filter and
   floored (§3.5a), so `t_end = stamp_start + sweep_duration` is never collapsed
   by aggressive filtering and never invented past the true sweep end.

Nothing else: the scan crosses L1→L2 **undeskewed**, in the sensor frame, and no
L1 stage ever needs a pose. Cold-start is likewise not an L1 concern — the
front-end holds groups internally until its static initialization completes
(`00_*` §7.2); L1 and the pipeline forward every group immediately, with no
bootstrap buffer and no init gate anywhere below L2.

---

## 5. Camera preprocessing: photometric calibration + rectification

The planned visual stage is **FAST-LIVO2-style sparse-direct photometric**
(dormant: the current front-end fuses LiDAR+IMU only and passes images through
on keyframes). Its residual is a raw image difference
$r = I_{\text{ref}} - I_{\text{cur}}(\pi(\cdot))$ (FAST-LIVO2 `vio.cpp:312`), which
assumes **brightness constancy**. Auto-exposure, vignetting, and gamma all break
that assumption, so camera L1 normalises photometry now, keeping the seam ready.
The reference FAST-LIVO2 `preprocess.cpp` handles only *LiDAR* (its `l515_handler`
just copies RGB into normals, `FAST-LIVO2/src/preprocess.cpp:203-241`) — the
photometric conditioning there lives in the VIO module, not preprocess. Meridian pulls
it into L1 where it belongs (per-sensor conditioning before fusion).

Camera L1 is four stages: **debayer/convert → photometric normalise → geometric
rectify → pyramid build.**

### 5.1 Debayer / colour-space convert

From `CameraFrame.encoding` (`01_*` §4.3: `Mono8 | Bayer_RGGB8 | RGB8`): debayer
Bayer to RGB (OpenCV `cvtColor`), keep Mono as-is. The direct photometric residual
operates on **intensity** (luminance); colour is retained separately only for L4
map colourisation (nvblox colour integration, `06_mapping.md`). So L1 produces
*both* a single-channel intensity image (for L2) and the colour image (forwarded for
L4), sharing storage where possible.

### 5.2 Photometric calibration (the brightness-constancy fix)

Three corrections, applied to the intensity image, parameters from
`IntrinsicsCamera`/`L1Config::Camera`:

1. **Vignetting** — divide by a precomputed per-pixel gain map $V(u,v)$
   (radial falloff, calibrated offline). $I' = I / V$.
2. **Gamma / response linearization** — apply the inverse camera response
   function (CRF) so pixel values are linear in radiance: $I'' = g^{-1}(I')$.
   For an 8-bit sensor a 256-entry LUT makes this free.
3. **Exposure/gain normalisation** — the `CameraFrame` carries `exposure_s` and
   `gain` (`01_*` §4.3, kept "so L2 can normalise intensities"). Normalise to a
   reference exposure: $I''' = I'' \cdot (e_{\text{ref}}/e) \cdot (g_{\text{ref}}/g)$.
   When `exposure_s == 0` (unknown), skip this term and set a telemetry flag; L2
   then falls back to its online affine-brightness estimate (FAST-LIVO2 models an
   affine `a*I+b` per frame — that estimate is L2's, but it converges far faster
   when L1 has already removed the deterministic part).

If no photometric calibration is provided (`photometric_calib=false`; default
**true** for the rig, false for an unmodelled camera), L1 passes intensity through
and emits `event(WARN,"camera/no_photometric_calib")` — the visual residual will be
less stable, which the operator should know.

### 5.3 Geometric rectification / undistortion

Using `IntrinsicsCamera` (`fx,fy,cx,cy`, distortion `RadTan | Equidistant`,
`01_*` §5.1): build the undistort-rectify map once (OpenCV
`initUndistortRectifyMap`) and `remap` each frame. The output is a pinhole image
with a known rectified `K`, so L2's projection $\pi$ and its Jacobian
$\partial\pi/\partial p = \begin{bmatrix} f_x/z & 0 & -f_x x/z^2\\ 0 & f_y/z &
-f_y y/z^2\end{bmatrix}$ (FAST-LIVO2 `vio.cpp:196-208`) are valid without a
per-point distortion term. The rectified `K` and image size are attached to the
output frame so L2 never re-derives them.

**Rolling shutter:** the spec assumes **global shutter** (`01_*` §4.3:
"global shutter: well-defined" mid-exposure stamp). A rolling-shutter camera would
need per-row time carried (analogous to LiDAR `t_offset_ns`) and a per-row pose
query — a future camera capability, flagged here as a known gap, not implemented
now.

### 5.4 Pyramid build

Build a Gaussian image pyramid (default `pyramid_levels = 3`, halving each level)
for coarse-to-fine photometric alignment in L2. Stored on the output `CameraFrame`
as Shared-immutable (built once in L1, read many times in L2). Meridian builds it in L1
so the cost is paid once on the T1 thread, not repeatedly in the hot L2 loop.

### 5.5 Output (camera)

A `CameraFrame` with: rectified single-channel intensity (photometrically
normalised) + pyramid for L2; rectified colour for L4; rectified `K`, size,
mid-exposure `stamp` unchanged; flags for `{photometric_calibrated,
exposure_known}`.

---

## 6. IMU preprocessing: passthrough + Allan-variance noise

IMU L1 is deliberately thin: the IMU is **passed through** to L2 (which owns
initialization, propagation, and de-biasing — `01_*` §4.1: "de-biasing happens in
L2"). L1 does exactly two things: (1) supply the **Allan-variance noise model**
(via `CalibrationSet`, at load time), (2) per-sample **saturation flagging**.

### 6.1 Why L1 does not de-bias, integrate, or initialize

The reference couples init, propagation, and deskew in one `ImuProcess` class
(`FAST_LIO/src/IMU_Processing.hpp`). Meridian puts all three in the front-end:
*static initialization* (gravity alignment + at-rest biases over the first
`frontend.lio.init_stationary_s` of standstill) is the estimator seeding its own
state, *propagation* is estimator state evolution, and *deskew* consumes both —
so they live together in L2 (`04_*`; `00_*` §7.2), and L1 needs no init gate, no
staging buffer, and no startup state machine. The `ImuSample` that crosses L1→L2
is the raw specific force + rate (`01_*` §4.1) — L1 must not subtract bias,
because the bias is a *live estimated variable* in L2 and subtracting an
init-time estimate would double-correct.

**No accelerometer scale normalization (deliberate).** The reference rescales every
raw accelerometer reading at propagation time by $G/\lVert\bar a\rVert$
(`acc_avr = acc_avr * G_m_s2 / mean_acc.norm()`,
`FAST_LIO/src/IMU_Processing.hpp:266`; `FAST-LIVO2/src/IMU_Processing.cpp:353`) —
an implicit unit conversion that couples the runtime model to the init-time static
norm and silently corrects an accelerometer reporting in $g$ rather than m/s². Meridian
does **not** apply this rescale anywhere: the `ImuSample` is consumed in raw m/s² by
the front-end, whose static init checks $\lVert\bar a\rVert \approx G$ before
accepting, so a genuine unit/scale fault is surfaced (init refuses to complete),
not absorbed.

### 6.2 Allan-variance noise model

The continuous-time IMU noise densities feed the front-end's propagation and
prior weighting (the reference fills `Q` blocks from `cov_gyr/cov_acc/cov_bias_*`,
`FAST_LIO/src/IMU_Processing.hpp:280-283`; `process_noise_cov()`,
`use-ikfom.hpp:35-43`). The four parameters are the IMU spec sheet /
Allan-deviation values:

| Symbol | Meaning | Reference field | `CalibrationSet` field (`01_*` §5.3) |
|---|---|---|---|
| $\sigma_g$ | gyro white noise (ARW) | `cov_gyr` | `imu_gyr_noise` |
| $\sigma_a$ | accel white noise (VRW) | `cov_acc` | `imu_acc_noise` |
| $\sigma_{bg}$ | gyro bias random-walk | `cov_bias_gyr` | `imu_gyr_bias_rw` |
| $\sigma_{ba}$ | accel bias random-walk | `cov_bias_acc` | `imu_acc_bias_rw` |

**Source-of-truth decision (deliberate, vs the reference foot-gun).** The
reference *measures* `cov_acc/cov_gyr` during init and then **overwrites** them
with configured `cov_*_scale` constants
(`FAST_LIO/src/IMU_Processing.hpp:368-372`): the measured value is immediately
overwritten, so in practice the run uses the configured scales and the empirical
variance is silently discarded. Meridian makes this explicit: the
**Allan-variance values from `CalibrationSet` are the source of truth**, loaded
once from config (`sensors.imu.cov_*`, squared densities the loader square-roots)
and never overwritten at runtime. The at-rest variance the front-end measures
during its static init is its own acceptance gate and telemetry
(`frontend/lio/accel_var`), not a noise-model source — a measured/configured
mismatch shows up there, never as a hidden overwrite.

### 6.3 Saturation flagging

Each `ImuSample` is checked against the sensor's full-scale range
(`imu_acc_fs`, `imu_gyr_fs` from `L1Config`): if $\lVert a\rVert \ge$ FS or any
$|\omega_i| \ge$ FS, set a saturation flag the front-end uses to down-weight or
skip that sample (a saturated IMU reading is a lie, not a measurement). The
reference has no saturation handling; this is a tactical-robustness addition for
the high-dynamics envelope (0–30 m/s with sharp yaw). The flag rides as out-of-band
metadata (not a new `ImuSample` field — the boundary type is fixed by `01_*`; the
flag is carried in an `ImuSampleMeta` sidecar on the L1→L2 queue, or the sample is
dropped at L1 when `drop_saturated=true`).

### 6.4 Output (IMU)

The raw `ImuSample` passed through (Value, no copy), plus a per-sample
saturation verdict. The noise model reaches the front-end through
`CalibrationSet`, not through the sample stream.

---

## 7. GNSS preprocessing: gating + spoofing check

GNSS is the one modality with an **active adversary**. GNSS L1 produces a `GnssFix`
plus an explicit **accept/reject verdict with a reason**, so L3 only ever sees fixes
that passed both a quality gate and an anti-spoof check. There is no GNSS handling in
the reference systems (FAST-LIO is LiDAR-inertial only); this section is grounded in
the architecture (`00_*` §3 L1 row, `04_*` §3.4 GNSS residual) and standard GNSS
practice.

### 7.1 Quality gate (cheap, first)

Reject a fix unless **all** hold (parameters in `L1Config::Gnss`):

- `fix.fix >= min_fix_type` — e.g. require at least `DGPS`, ideally `RTK_Fixed`
  (`GnssFix::FixType`, `01_*` §4.4). An `SPP` fix with metres of error must not
  anchor a sub-decimetre map unless nothing better exists.
- `fix.num_sats >= min_sats` (default 6).
- `trace(cov_enu) <= max_pos_var` — reject fixes whose own reported covariance is
  too loose (default 25 m² ≈ 5 m σ).
- HDOP/PDOP `<= max_dop` if available.

A fix failing the quality gate is rejected with
`reason = WeakFix|FewSats|HighCov|HighDop`.

### 7.2 Spoofing check (GNSS-vs-IMU velocity consistency)

The architecture specifies it precisely: *compare GNSS-derived velocity to
IMU-derived velocity over a 1 s window; large disagreement → drop and flag.*
Implementation:

1. Maintain a 1 s ring of accepted GNSS positions; derive
   $v_{\text{gnss}} = (p_k - p_{k-W}) / \Delta t$ in local ENU (positions via the
   LLA→ENU conversion §7.3).
2. Get $v_{\text{imu}}$ over the same window from the front-end's live `NavState`
   velocity (or, pre-trajectory, from integrated IMU). This is a **read of L2
   state** — like deskew, a feedback read, injected as a small `IVelocitySource`
   so `meridian_preprocess` stays decoupled from `meridian_frontend`.
3. If $\lVert v_{\text{gnss}} - v_{\text{imu}}\rVert > $ `spoof_vel_thresh`
   (default 3 m/s, tunable to dynamics) for `spoof_persist` consecutive windows,
   declare **spoofing**: reject the fix with `reason = SpoofVelocity` and raise
   `event(WARN,"gnss/spoof_suspected")`.

Additional cheap spoof heuristics (all opt-in): **clock-jump** (GNSS time vs PTP
host time drift > `spoof_clock_thresh` — a spoofer often resets receiver time),
**position-jump** (fix teleports faster than dynamically possible), and
**C/N0 anomaly** (uniformly high carrier-to-noise across all sats is a classic
spoof signature) if the receiver reports per-sat C/N0.

A fix surviving both gates is still only a *candidate* — L3 wraps it in a
**switchable constraint** (`00_*` §3 L3 row) so the optimiser can still disable a
fix that disagrees with everything else. L1's job is to stop the obvious attacks
cheaply, before the optimiser; defence in depth.

### 7.3 LLA → ENU conversion + datum

Convert WGS84 `lat/lon/alt` → ECEF → local **ENU** centred on the **first accepted
fix** (the working map frame). L1 computes the metric ENU position + propagates
`cov_enu` (already ENU in `GnssFix`, `01_*` §4.4) into the verdict. The ENU datum
origin is a **system decision the back-end owns** (`01_*` §4.4) — L1 may compute a
*provisional* ENU for the spoof check, but the authoritative datum is fixed by L3;
L1 reports the geodetic fix + provisional ENU + verdict, and L3 re-projects into the
canonical `map` datum.

### 7.4 Output (GNSS)

The `GnssFix` (unchanged geodetic data + `cov_enu`) plus a
`GnssVerdict { bool accepted; Reason reason; Eigen::Vector3d enu_provisional; }`. Only accepted
fixes are forwarded toward L3; rejected fixes are dropped but **counted and logged**
so denial/spoofing episodes are visible (§10).

---

## 8. Parameters (the L1 slice of `Config`)

L1 reads its slice of the one typed `Config` tree (`00_*` §8). No `nh.param`
scatter, no launch/YAML split, validated on load (`00_*` §8.3; the reference's
silent-default foot-gun — `nh.param` falls back to a hard-coded default on a
missing/misspelled key with no validation, no unknown-key warning, no resolved-config
dump — is catalogued in Appendix R.2). The L1 keys extend the `preprocess:` block
sketched in `00_*` §8.2
(`blind, point_filter_num, voxel_surf_m`). Full schema in Appendix A. Cross-field
validations L1 adds to `Config::validate()`:

- `blind < det_range`; both positive.
- `point_filter_num >= 1`.
- `0 < sweep_floor_frac <= 1` (the floor is a fraction of one nominal period, never
  more — §3.5a).
- `voxel_surf_m > 0` and `tsdf_voxel_m <= voxel_surf_m` (the map cannot be finer
  than the cloud feeding it, `00_*` §8.3); `surf_max_pts >= 1` (§3.2a).
- camera `pyramid_levels >= 1`; intrinsics present iff a camera is configured.
- gnss `min_fix_type` valid; `spoof_vel_thresh > 0` if `spoof_check=true`.
- the LiDAR has an `Extrinsic` in `CalibrationSet` and (if `selfhit_mask`
  referenced) a mask definition.

---

## 9. Failure modes (consolidated)

| # | Failure | Detection | L1 response | Telemetry |
|---|---|---|---|---|
| F1 | Scan with no per-point time | adapter: all `t_offset_ns==0` | reject scan | `event(WARN,"lidar/no_point_time")` |
| F2 | Empty scan after filtering | `n_out == 0` | drop scan, do not forward | `event(WARN,"lidar/empty_after_filter")` + stats |
| F3 | Self-hit storm (sensor sees rig) | `n_selfhit / n_in` high | rely on mask; if mask missing, blind radius only | `scalar("lidar/selfhit_frac")` |
| F4 | IMU saturation | $\lVert a\rVert$/$\lvert\omega\rvert \ge$ FS | flag (or drop if `drop_saturated`) | `scalar("imu/saturated_frac")` |
| F5 | Camera over/under-exposed | mean/clip ratio out of band | forward + flag; L2 down-weights | `event(WARN,"camera/exposure_bad")` |
| F6 | Missing photometric calib | `photometric_calib=false` | pass-through intensity | `event(WARN,"camera/no_photometric_calib")` |
| F7 | GNSS weak fix | quality gate (§7.1) | reject, count | `event(INFO,"gnss/rejected", reason)` |
| F8 | GNSS spoof suspected | vel/clock/jump check (§7.2) | reject + flag | `event(WARN,"gnss/spoof_suspected")` |
| F9 | GNSS denial (no fixes) | no accepted fix for `T` | none (system runs without GNSS) | `scalar("gnss/accept_rate")` |

(Registration failure, sensor-gap reseed, and IMU-init-while-moving are **L2**
failure modes — the front-end owns deskew and initialization, spec 04 — and are
surfaced as `frontend/lio/*` events, not here.)

The governing principle: **L1 never forwards a measurement it cannot vouch for,
and every drop is counted and surfaced** — never a silent `continue`/`ROS_WARN`, the
reference anti-pattern catalogued in Appendix R.2.

---

## 10. Debug hooks (the L1 telemetry surface)

L1 writes to the `TelemetrySink` (`00_*` §10.1; pure interface, ROS bound by the
wrapper). Cost discipline: `NullSink` makes every call a no-op (`00_*` §10.6).
Each stage is wrapped in a `ScopedTimer`. The L1 surface:

**Timing** (`timing(stage, ms)` → `/meridian/stage_timing`, `00_*` §10.2):
`preprocess.lidar.decode`, `preprocess.lidar.validity`, `preprocess.camera.rectify`,
`preprocess.camera.pyramid`, `preprocess.gnss.gate`. This is the live, per-stage
breakdown the
reference only dumped to CSV at shutdown (per-scan preprocess time stashed in the
fixed global array `s_plot11[scan_count]`, `FAST_LIO/src/laserMapping.cpp:295,331`;
Appendix R.3).

**Scalars** (`scalar(key, v, t)` → `/meridian/telemetry`):
- `lidar/n_in`, `lidar/n_out`, `lidar/n_voxel_out`, `lidar/n_nan`, `lidar/n_blind`,
  `lidar/n_far`, `lidar/n_selfhit`, `lidar/selfhit_frac` — the validity +
  downsample funnel (`n_voxel_out` is the point count after §3.2a).
- `imu/saturated_frac`.
- `gnss/accept_rate`, `gnss/spoof_score`.

**Clouds** (`cloud(key, ...)` → PC2, rate-limited, `00_*` §10.6):
- `preprocess/lidar_valid` — the validated cloud (sensor frame), so an operator
  sees what survived filtering.
- `preprocess/lidar_dropped` — the rejected points coloured by reason
  (nan/blind/far/selfhit), so a bad self-hit mask is *visible* (the reference has
  no equivalent).
- (The deskewed sweep is published by the front-end as `body/scan`, spec 09 —
  deskew is L2's, so its visual check lives there.)

**Markers** (`marker(...)` → `MarkerArray`):
- `preprocess/selfhit_mask` — the self-hit volumes, so a mis-calibrated mask is
  obvious in rviz.
- `preprocess/gnss_enu` — accepted (green) vs rejected (red, labelled by reason)
  fixes in the ENU frame.

**Events** (`event(level, tag, msg, t)` → `/meridian/events`): every row of the §9
table.

**Runtime control:** all clouds/markers gated by `debug.publish_*` + the
token-bucket rate limiter (`00_*` §10.5/§10.6), toggleable via `DebugControl`
without restart, so heavy point clouds stay off the wire in production.

---

## 11. Test plan

`meridian_preprocess` is pure C++ + Eigen/PCL/OpenCV — unit-testable with no ROS,
no clock, no middleware (`00_*` §1, the whole point of the split). Tests run under
`colcon test` with a `RecordingSink` capturing telemetry for assertions.

- **Decode adapter (§3.1):** feed a synthetic Ouster buffer with known
  `t`/`ring`/`range`; assert `t_offset_ns` ns-exact, sensor-frame `xyz` unchanged,
  `sweep_duration` correct. A regression replay of a real Ouster scan
  (`10_evaluation_harness.md`) asserts decode stability.
- **Validity gate (§3.2):** construct a cloud spanning all reject reasons
  (NaN, sub-blind, beyond det_range, inside a self-hit box, out-of-band intensity)
  + valid points; assert exactly the expected survivors, the funnel stats, and that
  points are time-sorted afterward. Property test: filtering is idempotent; output
  never aliases input bytes (copy-on-write).
- **Surf-voxel downsample (§3.2a):** feed a dense cell of points; assert
  `surf_max_pts == 1` keeps the point nearest the voxel centre (with the documented
  tie-break) and that the result is invariant under a **shuffle of the input order**
  — the determinism property newest-wins fails. With `surf_max_pts > 1`, assert the
  reservoir keeps exactly the cap and that the survivors are re-sorted by
  `t_offset_ns`. Assert `n_voxel_out` matches the survivor count.
- **`sweep_duration` floor (§3.5a):** filter a scan down to a handful of
  early-column survivors; assert the recomputed `sweep_duration` is raised to
  `sweep_floor_frac / nominal_rate_hz`, that it never exceeds the original
  (pre-filter) span, and that no surviving point's `t_offset_ns` exceeds the floored
  end. A scan whose survivors already span past the floor passes through unchanged.
- **Deskew:** *not tested here* — deskew, static init, and reseed are front-end
  behaviour, covered by the L2 unit suite (spec 04 / spec 10). L1's contribution
  is the three output properties of §4.2, asserted by the validity-gate and
  `sweep_duration`-floor tests above.
- **IMU noise (§6.2):** assert the CalibrationSet Allan values flow through to
  the front-end unchanged (the loader square-roots the squared-density config
  keys exactly once).
- **Camera (§5):** feed a frame through a known vignette+CRF+exposure; assert
  brightness-constancy is restored (a flat-field input becomes flat output);
  assert rectified `K` and pyramid level sizes.
- **GNSS (§7):** quality gate truth table; spoof test feeding GNSS velocity that
  diverges from a scripted IMU velocity, asserting `spoof_suspected` after
  `spoof_persist` windows and not before; LLA→ENU against a known reference point.
- **Order-independence (`00_*` §11.2):** in single-thread mode, the whole L1
  pipeline on a fixed input is order-independent (L1 carries no parallel reduction).

---

## Appendix A — `L1Config` schema

```cpp
// The L1 slice of meridian::Config (00_* §8). Plain fields, validated on load.
// Single LiDAR + single IMU + single camera + GNSS.
struct L1Config {
  struct Lidar {
    SensorModel model = SensorModel::OusterOS1_128;
    double  blind          = 0.5;   // m   (self-hit core; ref default 0.01, preprocess.cpp:7)
    double  det_range      = 120.0; // m   (far-range clip; ref det_range)
    int     point_filter_num = 3;   // keep 1 per N VALID points (ref preprocess.cpp:260)
    double  sweep_floor_frac = 0.5; // min sweep_duration as fraction of nominal period (§3.5a)
    double  voxel_surf_m   = 0.5;   // surf-voxel downsample edge (§3.2a; tsdf_voxel_m <= this)
    int     surf_max_pts   = 1;     // points kept per surf voxel (§3.2a)
    std::uint64_t surf_seed = 0;    // seed for deterministic reservoir when surf_max_pts > 1 (§3.2a)
    bool    intensity_gate = false; double i_min=0, i_max=1e9;
    bool    organize       = false; // build ring×col index (§3.3)
    std::string selfhit_mask;       // name of mask volume set in CalibrationSet
  };
  struct Camera {
    bool    photometric_calib = true;
    std::string vignette_map;          // path to V(u,v)
    std::string crf_lut;               // inverse camera response LUT
    double  ref_exposure_s = 0.01; double ref_gain = 1.0;
    int     pyramid_levels = 3;
  };
  struct Imu {
    // Noise densities live in CalibrationSet (sensors.imu.cov_* keys, squared
    // densities the loader square-roots); init thresholds live in frontend.lio.
    double  imu_acc_fs = 156.0; double imu_gyr_fs = 34.9; // full-scale (m/s^2, rad/s)
    bool    drop_saturated = false;
  };
  struct Gnss {
    bool    enable = true;
    GnssFix::FixType min_fix_type = GnssFix::FixType::DGPS;
    int     min_sats = 6;
    double  max_pos_var = 25.0;   // m^2 (trace cov_enu)
    double  max_dop = 5.0;
    bool    spoof_check = true;
    double  spoof_vel_thresh = 3.0;   // m/s GNSS-vs-IMU disagreement
    int     spoof_persist = 3;        // consecutive windows before declaring spoof
    int     spoof_window_ms = 1000;   // the 1 s comparison window
    bool    spoof_clock_check = false; double spoof_clock_thresh = 0.05; // s
  };
  Lidar lidar; Deskew deskew; Camera camera; Imu imu; Gnss gnss;
};
```

## Appendix B — the LiDAR adapter seam

Replaces the reference `switch(lidar_type)` (`FAST_LIO/src/preprocess.cpp:71-88`;
grown to 7 cases in FAST-LIVO2, `FAST-LIVO2/src/preprocess.cpp:62-91`) with a single
clean adapter. The rig has **one LiDAR** (Ouster OS1-128), so there is one
implementation; the seam exists to isolate that sensor's native-layout quirks from
the shared validity / deskew / telemetry stages and to unit-test decode in isolation.

```cpp
// Native Ouster point layout (from the reference, 01_* §4.2 mirrors ouster_ros::Point):
//   Ouster : x,y,z,intensity, uint32 t(ns), uint16 reflectivity, uint8 ring,
//            uint16 ambient, uint32 range      (preprocess.h:60-69)
class OusterAdapter : public ILidarAdapter { ... };  // t_offset_ns = int32(p.t)

// Built once by the pipeline from CalibrationSet.
ILidarAdapter::Ptr makeLidarAdapter(SensorModel);     // OusterOS1_128 -> OusterAdapter
```

The adapter is the *only* place the sensor's native quirks live (the native ns time
unit, ring count, the Ouster `range`-precomputed-vs-recompute choice). Validity
(§3.2), the deskew mechanism (§4), and the telemetry surface (§10) are
**sensor-agnostic** and shared — the modularity the reference's monolithic per-type
handlers lack. **Future extension (not designed now):** a second LiDAR would register
its own `ILidarAdapter` behind the same seam and produce a second tagged `LidarScan`;
how/whether scans merge would be an L2 decision. One LiDAR is the committed design.

---

## Appendix R — SOTA reference grounding (non-normative)

This appendix is evidence, not contract: curated digests of the reference systems
this spec's design was validated against. Nothing here binds Meridian's behavior —
the normative sections above own the design. Each block names the reference checkout
it was verified against; the clones live in /home/user/slam-reference.

### R.1 Per-sensor LiDAR config: the values that actually differ
*verified against FAST_LIO@7cc4175*

Each reference LiDAR ships one YAML (`config/{avia,horizon,ouster64,velodyne,…}.yaml`)
that differs mainly in topic, `lidar_type`, `scan_line`, `timestamp_unit`,
`blind`, `fov_degree`, `det_range`, `extrinsic_T/R`. Verified divergences:

| sensor | det_range | fov | scan_line | timestamp_unit |
|---|---|---|---|---|
| avia | 450 | 90 | — | — |
| velodyne | 100 | 180 | 32 | 2 (US) |
| ouster64 | 150 | 180 | 64 | 3 (NS) |

The duplicated covariance / publish / `pcd_save` blocks are copied verbatim across
files and drift independently — the motivation for Meridian's single validated
`Config` tree with per-sensor profiles over a shared base (§8, Appendix A). Meridian
runs one LiDAR (Ouster OS1-128, native ns `t`), so there is no per-type unit table.

### R.2 Parameter loading & the silent-default foot-gun
*verified against FAST_LIO@7cc4175, FAST-LIVO2@0d2c034; Point-LIO digest below (source NOT in /home/user/slam-reference)*

| sharp edge | reference behaviour |
|---|---|
| silent defaults | `nh.param(name, var, default)` falls back to the hard-coded default on a missing/misspelled key — no validation, no range check, no unknown-key warning, no resolved-config dump. A YAML typo → a silently wrong run. |
| params split YAML↔launch | `point_filter_num`, `max_iteration`, `filter_size_*`, `cube_side_length`, logging toggle live in `mapping_*.launch`; topics/covariances/extrinsics in `config/*.yaml`. Undocumented, drift-prone: launch sets `cube_side_length=1000` while the code default is **200** (`laserMapping.cpp:774`). |
| silent recovery | a dropped/invalid measurement is a bare `continue`/`ROS_WARN`, never counted or surfaced. |

**Point-LIO digest (source absent from the clone set, recorded from a prior read):**
factored param loading into `parameters.cpp` (`readParameters`) and init into
`li_initialization.cpp` (good "one concern per file" instinct) but stitched them via
`extern` globals in `parameters.h` — one shared mutable namespace spread across
files — and carried dual-mode flags (`use_imu_as_input`, `prop_at_freq_of_imu`) that
force global-flag branching throughout `laserMapping.cpp`. Treat as unverified.

### R.3 Timing / logging instrumentation (what to retain, what to fix)
*verified against FAST_LIO@7cc4175*

FAST_LIO times pipeline stages with `omp_get_wtime()` snapshots and accumulators
(`match_time`, `solve_time`, `kdtree_*_time`), and stashes per-scan stats into 12
fixed global arrays `s_plot…s_plot11[MAXN]` with `MAXN=720000` (`laserMapping.cpp:65,70`)
— per-scan **preprocess** time goes to `s_plot11[scan_count]`
(`laserMapping.cpp:295,331`) — dumped to `Log/*.csv` only at shutdown
(`:1040-1051`). Sharp edges Meridian fixes: all output is `printf`/`fprintf` to
fixed-size global arrays (≈69 MB, overflows on long runs) and flat text — no levels,
no per-line timestamps, no live timing stream. Meridian emits per-stage timing live
(§10) through a leveled structured sink instead (spec 09).

---

*End of spec 03. Upstream: `01_interfaces_and_data_types.md` (types),
`02_sensors_timesync.md` (L0 time-synced samples). Downstream:
`04_frontend_estimation.md` (consumes L1's validated undeskewed scans + per-point
time, owns deskew, initialization, and the LIO estimator) and `06_mapping.md`
(nvblox consumes deskewed keyframe clouds + colour).*
