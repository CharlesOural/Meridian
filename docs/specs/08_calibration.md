# 08 — Cross-cutting: Calibration

> **Spec status:** normative (implementation spec). This document specifies
> Meridian's **calibration subsystem** — the cross-cutting machinery in
> `meridian_calib` that produces, stores, seeds, serves, and (online) refines the
> intrinsic and extrinsic parameters every other layer needs to turn raw sensor
> measurements into a single, geometrically consistent estimate. It is the
> concrete realisation of the architecture's cross-cutting goal: *offline prior +
> online extrinsic refinement as graph variables* (arch §3 cross-cutting row,
> arch §0 mental model).
>
> **The rig is fixed and singular.** Meridian fuses **one LiDAR + one IMU + one
> camera + GNSS** (arch §2, §3). This spec calibrates exactly that rig: four
> sensor frames, one estimation frame. There is no multi-LiDAR merge and no
> per-sensor residual-stream bookkeeping — those do not exist in the system, so they
> do not exist here. (The surface map's `nvblox`/`cpu` backends share one extrinsic
> set; the backend choice does not touch calibration.) (A second LiDAR, if ever added,
> would slot behind the same `Extrinsic` / `ICalibrationProvider` machinery with
> no new concepts; that is a one-line future extension, not a design point.)
>
> **Reading order.** This spec **implements and extends** the calibration value
> types declared in [`01_interfaces_and_data_types.md`](01_interfaces_and_data_types.md)
> §5 (`IntrinsicsCamera`, `Extrinsic`, `CalibrationSet`, plus the `refine_online`
> flag and the L3→L2 versioned-snapshot rule) and obeys the architecture in
> [`00_architecture.md`](00_architecture.md) (ROS-agnostic core §1; package
> `meridian_calib` §2, §4; immutable typed config §8; telemetry bus §10). It
> consumes the time model of [`02_sensors_timesync.md`](02_sensors_timesync.md)
> (the `ClockModel`, `StampSource`, and the **time-offset estimation** of §7 —
> *temporal* calibration lives partly there and is referenced, not duplicated,
> here). It is consumed by **L2** (the LIO front-end, which transforms
> every sensor into the estimation frame $F_e$), **L3** (back-end, the *authority* for online-refined
> extrinsics held as graph variables), and **L4** (nvblox colourisation needs
> `T_body_cam` and camera intrinsics). Where this spec needs a type not yet in
> spec 01, it says so explicitly and the change is **amended into spec 01**, never
> redefined here (spec 01 R-rules).
>
> **Grounding.** Calibration *as state* is grounded in the apex references.
> FAST-LIO carries the LiDAR→IMU extrinsic **inside the filter state**
> (`offset_R_L_I`, `offset_T_L_I` in `state_ikfom`, `use-ikfom.hpp:15–16`) and
> toggles its online estimation with a single boolean `extrinsic_est_en` loaded
> from config (`laserMapping.cpp:786`), seeded by an offline prior `extrinsic_T` /
> `extrinsic_R` (`laserMapping.cpp:803–804`), logged for forensics
> (`laserMapping.cpp:138`). **FAST-LIVO2** additionally carries a **photometric**
> intrinsic — the inverse exposure time — and refines it online alongside the
> geometric extrinsic, plus per-frame exposure compensation for brightness
> constancy. We keep what these get right (a tight offline prior, an
> online-refinement switch, provenance logging, online photometric estimation) and
> fix what they leave implicit: the prior carries **no uncertainty**, there is no
> **observability gate** on whether a parameter is currently estimable, and the
> offline-calibrate → online-refine handoff is undocumented. The offline-tooling
> convention is **Kalibr** (target-board intrinsics + camera–IMU spatiotemporal
> extrinsics + IMU Allan-variance noise), the de-facto standard the benchmark
> dataset ships in ([`DATASET.md`](../DATASET.md): per-collection Kalibr
> camchain-imucam files plus the rig-transform YAML — these *seed the priors*, §8).
>
> **Notation** follows spec 01 §0: $T_{A\_B}\in SE(3)$ maps a point
> from frame $B$ to frame $A$ ($p_A = T_{A\_B}\,p_B$); the estimation frame is
> $F_e$ (normally `imu_link`, spec 01 §2.3); tangent ordering is **translation
> first, rotation second** $\xi=[\rho;\phi]\in\mathbb{R}^6$ with the **right**
> $\boxplus$ convention (spec 01 §3.1); information $\Omega=\Sigma^{-1}$.

---

## Table of contents

1. [Why calibration is a cross-cutting subsystem](#1-why-calibration-is-a-cross-cutting-subsystem)
2. [The three stages (and the one timeline they live on)](#2-the-three-stages-and-the-one-timeline-they-live-on)
3. [Data model (recap of spec 01 §5 + the extensions this spec amends in)](#3-data-model-recap-of-spec-01-5--the-extensions-this-spec-amends-in)
4. [Stage 1 — factory intrinsics](#4-stage-1--factory-intrinsics)
5. [Stage 2 — per-deployment extrinsics as PRIORS (Kalibr / target / hand-eye)](#5-stage-2--per-deployment-extrinsics-as-priors-kalibr--target--hand-eye)
6. [Stage 3 — online refinement as L3 graph variables](#6-stage-3--online-refinement-as-l3-graph-variables)
7. [Temporal calibration (time offsets)](#7-temporal-calibration-time-offsets)
8. [Seeding the priors from dataset calibration files](#8-seeding-the-priors-from-dataset-calibration-files)
9. [The `ICalibrationProvider` interface and snapshot service](#9-the-icalibrationprovider-interface-and-snapshot-service)
10. [Configuration schema](#10-configuration-schema)
11. [Failure modes and recovery](#11-failure-modes-and-recovery)
12. [Debug / introspection hooks](#12-debug--introspection-hooks)
13. [Threading, ownership, lifetime summary](#13-threading-ownership-lifetime-summary)
14. [Worked example: the camera extrinsic from Kalibr file to refined graph variable](#14-worked-example-the-camera-extrinsic-from-kalibr-file-to-refined-graph-variable)
15. [Test plan](#15-test-plan)

---

## 1. Why calibration is a cross-cutting subsystem

Calibration is not a layer; it is a *substrate* every layer reads and one layer
(L3) writes. A tightly-coupled multi-sensor estimator is, at root, the assertion
that several sensors observe **the same rigid body** through **known relative
transforms**. Every fused residual in Meridian's front-end is
conditioned on a calibration parameter:

* The LiDAR registration residual (spec 04) deskews and
  registers each point at its true sample time, bringing a point $p_L$ from
  the LiDAR frame into $F_e$ through the **extrinsic** $T_{F_e\_L}$. A 1° error in
  $T_{F_e\_L}$ rotation at 30 m range is a ~0.5 m phantom shift — orders of
  magnitude above the few-cm map resolution.
* The FAST-LIVO2-style sparse-direct **photometric** residual
  $r = I_k(\pi(K_{cam}, T_{F_e\_C}^{-1}\,p)) - I_{\text{ref}}$ depends on the
  **camera intrinsics** $K_{cam}$ + distortion, the **extrinsic** $T_{F_e\_C}$,
  *and* a **photometric** intrinsic (exposure/gain) for brightness constancy.
* The GNSS factor needs the lever-arm **extrinsic** $T_{F_e\_G}$ from $F_e$ to the
  antenna phase centre, or the absolute factor is biased by the lever arm times
  the orientation.
* Every cross-sensor fuse needs a **temporal** calibration — the per-sensor time
  offset that places measurements on one timeline (spec 02 §5, §7) so the
  estimator integrates and warps each at the right instant.

Because *all* of these condition the estimate, the architecture (arch §1.1, R1)
forbids scattering them through the layers the way FAST-LIO inlines a single
extrinsic into the EKF state and a single boolean into the node body. Meridian
centralises them in **one queryable, versioned `CalibrationSet`** (spec 01 §5.3)
served by `meridian_calib`, with exactly one writer for the online-refined subset
(L3) and many read-only readers (L1, L2, L4). This is the same discipline as the
`KeyframePacket` boundary (spec 01 §6): make the thing that conditions everything
*explicit, typed, and single-sourced* so its correctness can be audited.

`meridian_calib` is a **leaf cross-cutting library** (arch §4 R1): it depends only on
third-party (Eigen, Sophus, yaml-cpp) and on `meridian_common` value types; it
depends on **no layer**, and **no `rclcpp`** (arch §1.1). It owns the
calibration *models*, the *file loaders* (Kalibr + Meridian-native), the *seeding*
of priors, the *snapshot service* (`ICalibrationProvider`), and the *offline
calibration apps* (in `meridian_tools`, which links `meridian_calib`). It does **not**
own the *online* optimisation — that is L3's factor graph — but it owns the
**types and the contract** by which L3 publishes refined values back (§6, §9).

---

## 2. The three stages (and the one timeline they live on)

Meridian's calibration has **three temporally-ordered stages**, each with a
different cadence, owner, and trust model. The single most important design idea
is that **stage 2 produces a *prior* (a mean **and** a covariance), never a hard
constant**, so stage 3 can move it; and stage 3 produces an authoritative
*posterior* that is *fed back* as a fresh prior for the next mission. The stages
form a loop, not a pipeline.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │ STAGE 1  FACTORY INTRINSICS              (per sensor unit; ~once/lifetime)│
   │   camera K + distortion, IMU Allan noise, LiDAR range/intensity model    │
   │   owner: vendor + one-time lab cal     trust: high, slowly-aging         │
   └───────────────────────────────┬──────────────────────────────────────┘
                                    │ feeds (fixed, or weak-prior if refined)
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ STAGE 2  PER-DEPLOYMENT EXTRINSICS as PRIORS  (per rig build / mission)  │
   │   Kalibr / target-board / hand-eye  →  T_Fe_sensor + 6×6 prior cov        │
   │   + temporal offset prior (spec 02 §7)                                    │
   │   owner: meridian_calib offline apps      trust: medium; the SEED for §6     │
   └───────────────────────────────┬──────────────────────────────────────┘
                                    │ seeds CalibrationSet v0  (mean+cov+refine flag)
                                    ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │ STAGE 3  ONLINE REFINEMENT as L3 GRAPH VARIABLES  (per keyframe; live)   │
   │   refine-flagged extrinsics become GTSAM variables anchored by the §5     │
   │   prior; observability-gated; published back as versioned snapshots       │
   │   owner: L3 back-end (authority)        trust: highest WHEN observable     │
   └───────────────────────────────┬──────────────────────────────────────┘
                                    │ refined posterior  →  persisted as next mission's STAGE 2 prior
                                    └───────────────────────────────────────► (loop)
```

| Stage | Parameters | Cadence | Owner / writer | Output | Trust model |
|---|---|---|---|---|---|
| 1 factory intrinsics | camera $K$+distortion; IMU noise densities + bias RW; LiDAR range/intensity | once per unit (re-cal yearly) | vendor + one-time lab; `meridian_calib` loader | `IntrinsicsCamera`, IMU noise, LiDAR model | high; treated **fixed** by default (camera intrinsics MAY be weak-prior-refined, §4.3) |
| 2 per-deployment extrinsics | $T_{F_e\_\text{sensor}}$ for LiDAR, camera, GNSS; temporal offset prior | per rig build / per mission | `meridian_calib` offline apps (Kalibr import / target / hand-eye) | `Extrinsic{T, prior_cov, refine_online, calibrated_at}` per sensor | medium; the **prior** that anchors §6 |
| 3 online refinement | refine-flagged $T_{F_e\_\text{sensor}}$ (+ optional temporal) | per keyframe (live) | **L3 back-end** (authority) | refined `CalibrationSet` snapshot (versioned) | highest **iff** observability-gated (§6.4) |

The cross-stage invariant (the thing you must not break):

> **A calibration parameter exists in exactly one of two regimes at any instant:
> FIXED (stage-1/2 value, never moved this mission) or REFINED (stage-3 graph
> variable anchored by its stage-2 prior). The regime is the `refine_online` flag
> on its `Extrinsic` (spec 01 §5.2). It never silently transitions; a transition
> is a telemetry event (§12).**

This is the explicit answer to FAST-LIO's implicit coupling, where the single
extrinsic is *always* both an offline-loaded value *and* a filter state with no
recorded uncertainty and no observability gate.

---

## 3. Data model (recap of spec 01 §5 + the extensions this spec amends in)

Spec 01 §5 already defines `IntrinsicsCamera`, `Extrinsic`, and `CalibrationSet`.
This spec **does not redefine them**; it adds the small number of fields the three
stages require, and these are **amended into spec 01 §5** (per the R-rules). The
extensions are minimal and additive (default-constructible, so old call sites
compile unchanged).

### 3.1 `Extrinsic` extension — provenance, source, and refinement bounds

```cpp
// AMENDS spec 01 §5.2. New fields are additive; existing fields unchanged.
enum class CalibSource : std::uint8_t {
  Unknown = 0,
  Vendor,          // shipped by the sensor vendor (intrinsics mostly)
  Kalibr,          // imported from a Kalibr YAML (§5, §8)
  TargetBoard,     // meridian_calib target-board app (checkerboard/Apriltag)
  HandEye,         // meridian_calib motion-based hand-eye (no target)
  OnlineRefined,   // a posterior written back by L3 (§6) — a SEED for next mission
  Manual,          // hand-entered (CAD / tape measure) — loosest prior
};

struct Extrinsic {                 // (spec 01 §5.2 fields shown for context)
  Frame   child  = Frame::Unknown; // os_sensor0 (LiDAR) | cam_link | gnss_link
  Frame   parent = Frame::ImuLink; // F_e
  Pose    T_parent_child;          // the offline prior mean (T_Fe_sensor)
  PoseCov6 prior_cov;              // 6-DoF prior uncertainty (Σ), ordering [ρ;φ]
  bool    refine_online = false;   // regime flag (§2 invariant)
  Timestamp calibrated_at = 0;     // provenance: when this prior was established

  // --- NEW (this spec) ---
  CalibSource source = CalibSource::Unknown;   // where the prior came from
  std::uint32_t version = 0;        // monotonically increasing across refinements (§9)
  // Hard bounds: refinement is REJECTED if it tries to leave this gate around the
  // prior mean. Protects against an unobservable variable drifting (§6.4, §11).
  double  max_drift_trans_m = 0.10; // ‖Δp‖ cap from prior mean
  double  max_drift_rot_deg = 2.0;  // ‖Δφ‖ cap from prior mean
  // Per-sensor temporal offset prior (the geometric extrinsic's time twin, §7).
  double  time_offset_ns      = 0.0;   // td: t_sensor = t_Fe + td  (prior mean)
  double  time_offset_std_ns  = 0.0;   // prior 1σ; 0 ⇒ effectively fixed
  bool    refine_time_online  = false; // refine td as a graph variable? (§7.3)
};
```

**Why bounds, not just a prior covariance.** A Gaussian prior alone does not stop
an *unobservable* extrinsic from drifting arbitrarily far in its null space when
the data is uninformative (the corridor / planar-ground degeneracy). The hard
`max_drift_*` gate is a cheap, interpretable safety net layered on top of the
soft prior and the observability gate (§6.4). It is the calibration analogue of
the per-axis observability flowing into back-end noise (spec 01 §3.4) — belt and
braces, because a bad extrinsic silently corrupts the *entire* map.

### 3.2 `IntrinsicsCamera` extension — the photometric intrinsic

```cpp
// AMENDS spec 01 §5.1. Additive; geometric fields unchanged.
struct IntrinsicsCamera {            // (geometric fields from spec 01 §5.1)
  double fx, fy, cx, cy;
  enum class Distortion { None, RadTan, Equidistant } model = Distortion::RadTan;
  std::array<double,5> coeffs = {0,0,0,0,0};
  int width = 0, height = 0;

  // --- NEW (this spec): photometric intrinsic for direct visual (FAST-LIVO2) ---
  // Brightness constancy is broken by auto-exposure; FAST-LIVO2 estimates the
  // inverse exposure time online. We carry it as an intrinsic with a prior so a
  // calibrated camera seeds it and the future visual stage refines it per frame (§6.5).
  double inv_expo_prior   = 1.0;   // 1/exposure scale prior (1.0 = none)
  double inv_expo_std     = 0.0;   // prior 1σ; 0 ⇒ fixed
  bool   refine_photometric_online = true;   // FAST-LIVO2 path: on (§6.5)
  // Optional response/vignette handles (deferred seam; pointers null for now).
};
```

### 3.3 The `CalibrationSet` is the single served value

`CalibrationSet` (spec 01 §5.3) is unchanged in shape; this spec pins down two
things about it that §6/§9 depend on:

* It carries a **`version` counter** (already implied by spec 01 §5.3 "version
  lets the front-end detect 'calibration changed'"). Concretely: the set's
  `version` is the **sum of member `Extrinsic::version`s plus an epoch**, so any
  single refinement bumps the set version monotonically (§9.2).
* It is always served **Shared-immutable** (spec 01 §2.4): a reader holds a
  `shared_ptr<const CalibrationSet>` valid for as long as it needs; a refinement
  publishes a **new** set, never mutates the old (§9). This is the data-race fix
  spec 01 §5.3 mandates between L3's optimiser and L2's per-scan reads.

---

## 4. Stage 1 — factory intrinsics

**What:** the per-unit, slowly-aging parameters that are properties of one
physical sensor, independent of how it is mounted. These are loaded once and, by
default, held **fixed** for the mission.

### 4.1 Camera intrinsics (geometric)

Pinhole $K_{cam} = \begin{psmallmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 &
1\end{psmallmatrix}$ plus a distortion model (`RadTan` $k_1,k_2,p_1,p_2,k_3$ or
`Equidistant`/fisheye $k_1..k_4$ — the Newer College Alphasense case: Kalibr
`pinhole-equi`). Source: a **Kalibr** target-board
intrinsics calibration (`pinhole-radtan` / `pinhole-equi` model), or the vendor
factory file. The front-end's direct photometric residual projects through
$\pi(K_{cam},\cdot)$; an error here is a systematic per-pixel reprojection bias,
so intrinsics are tightened first and refined last.

> **Equidistant handling.** Equidistant/fisheye distortion is awkward for direct
> visual methods, so Meridian supports it as a first-class intrinsic model.
> The stance (spec 02 §4.3 note: rectify in L1, not L0) is that `meridian_calib`
> stores the **native** intrinsics (`Equidistant`)
> and L1 builds a rectified pinhole virtual camera; the *rectified* $K$ is a
> derived intrinsic `meridian_calib` computes and caches, keyed by the source
> intrinsic version, so L2 always sees a pinhole. The benchmark rig's Alphasense
> cameras are exactly this case ([`DATASET.md`](../DATASET.md)): until the rectify
> map (or native equidistant projection) is wired into L1, the visual stage
> auto-disables on them and the system runs LIO-only.

### 4.2 IMU intrinsics (noise model)

The four Allan-variance noise parameters spec 01 §5.3 already lists on
`CalibrationSet` — `imu_acc_noise`, `imu_gyr_noise` (continuous-time noise
densities) and `imu_acc_bias_rw`, `imu_gyr_bias_rw` (bias random-walk) — *are*
the IMU's factory intrinsics. They feed the CT IMU-derivative residual noise and
the restart-fallback preintegration noise (the analogue of FAST-LIO
`set_gyr_cov` / `set_acc_cov` / `process_noise_cov`, `use-ikfom.hpp:35–43`).

> **Units convention — the `CalibrationSet` fields carry standard deviations, the
> config carries variances.** The `imu_acc_noise` / `imu_gyr_noise` /
> `imu_acc_bias_rw` / `imu_gyr_bias_rw` fields on `CalibrationSet` are the
> **continuous-time noise densities as standard deviations** (e.g.
> $\mathrm{(m/s^2)}/\sqrt{\mathrm{Hz}}$, $\mathrm{(rad/s)}/\sqrt{\mathrm{Hz}}$),
> because the residual assemblers weight by $1/\sigma$ and $1/(\sigma\sqrt{dt})$
> directly. The runtime config (spec 02 §10) instead carries the **squared**
> densities — `sensors.imu.cov_acc` / `cov_gyr` (and `b_acc_cov` / `b_gyr_cov`) are
> **variances** ($\mathrm{(m/s^2)^2}$, $\mathrm{(rad/s)^2}$). The bootstrap importer
> `calibrationFromConfig` therefore takes the **square root** of each configured
> value when it populates the `CalibrationSet`. A config author writes variances; a
> reader of `CalibrationSet` gets std-devs. Crossing the two (feeding a variance
> where a std-dev is expected, or vice versa) silently mis-weights every IMU
> residual — keep the convention straight by field, not by intuition.
Source: an **Allan-variance** run (Kalibr's `imu_utils` / `kalibr_allan`) or the
dataset's published IMU model. **Sanity-check imported densities against the
sensor datasheet before trusting them** — a mis-shipped density silently
mis-weights every IMU residual until the estimate diverges (the benchmark
dataset shipped a gyro density duplicated from the accel field; the deployed
configs carry the corrected value, [`DATASET.md`](../DATASET.md)).
These are held **fixed** (online IMU-intrinsic
estimation — scale/misalignment — is a deferred seam, §11). Meridian additionally
lets L0/L2 **inflate** these by the timing uncertainty $\sigma_{o_s}$ from spec 02
§7 (a poorly-timed IMU is a noisier IMU); that inflation is a runtime fold, not a
re-calibration.

### 4.3 Camera intrinsics: fixed, with an optional weak-prior refine

Geometric camera intrinsics are **fixed by default** (`refine_online`-equivalent
off). Meridian allows an **optional** weak-prior online refinement of $f_x,f_y$ only
(the most thermally-sensitive), gated identically to extrinsics (§6.4); it is off
unless explicitly enabled in config. The **photometric** intrinsic
(`inv_expo_prior`, §3.2) is different: it *is* refined live, because brightness
constancy without it is hopeless under auto-exposure (FAST-LIVO2 estimates it per
frame). It belongs to the **front-end's visual stage** (a per-frame nuisance, not a
slowly-varying rig parameter), not to L3 — see §6.5; dormant until that stage
exists.

### 4.4 LiDAR intrinsics

Range bias / intensity-calibration curves and the per-beam angle table (e.g.
`calib/beam_intrinsics_os0-128.json`, consumed only when decoding raw LiDAR
packets rather than driver point clouds) are vendor-supplied for the Ouster
LiDAR and held **fixed**; Meridian does not estimate them (out of scope, and
Ouster's factory calibration is good). The only "intrinsic" Meridian acts on is the
per-point time field, which is a *temporal* property handled in L0 (spec 02 §4.1)
— there is no LiDAR geometric intrinsic to refine here.

---

## 5. Stage 2 — per-deployment extrinsics as PRIORS (Kalibr / target / hand-eye)

**What:** the rigid transforms $T_{F_e\_\text{sensor}}$ that say where each
sensor sits relative to the estimation frame — one each for the **LiDAR**, the
**camera**, and the **GNSS antenna** — established **once per rig build** (or per
mission if the rig is reconfigured). The output is an `Extrinsic` per sensor: a
**mean and a covariance and a `refine_online` flag**, i.e. a *prior*, not a
constant. This is the single biggest departure from the reference, where
`extrinsic_T`/`extrinsic_R` (`laserMapping.cpp:803–804`) are loaded as bare
values with no uncertainty.

### 5.1 Three acquisition methods (offline apps in `meridian_tools`)

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │  meridian_calib offline acquisition (one chosen per sensor pair)         │
   ├─────────────────────────────────────────────────────────────────────┤
   │ (a) KALIBR IMPORT     parse a Kalibr YAML → Extrinsic (§8). Preferred  │
   │                       when a dataset/lab Kalibr run exists.            │
   │ (b) TARGET BOARD      checkerboard/Apriltag co-observed by cam+LiDAR;  │
   │                       PnP / plane-to-plane → T_Fe_cam, T_Fe_lidar.     │
   │ (c) HAND-EYE          motion-based AX=XB on logged trajectories;       │
   │                       targetless; for LiDAR↔IMU and cam↔IMU.          │
   └─────────────────────────────────────────────────────────────────────┘
```

**(a) Kalibr import** — the default path, because the benchmark dataset ships it
and Kalibr is the field standard. Kalibr's `camchain-imucam` YAML gives
$T_{\text{cam}\_\text{imu}}$ plus the time offset $t_d$; the importer
inverts/composes into Meridian's $T_{F_e\_\text{cam}} = T_{\text{imu}\_\text{cam}} =
(T_{\text{cam}\_\text{imu}})^{-1}$ (with $F_e =$ `imu_link`), and copies Kalibr's
reported parameter covariance into `prior_cov` (or a configured default if Kalibr
only reported a point estimate). Full field mapping in §8.

**(b) Target board** — a checkerboard or Apriltag grid co-visible to camera and
LiDAR. The camera side is PnP ($T_{\text{cam}\_\text{target}}$); the LiDAR side
fits the board plane(s) ($T_{\text{lidar}\_\text{target}}$); the extrinsic is the
composition $T_{\text{cam}\_\text{lidar}} = T_{\text{cam}\_\text{target}}
T_{\text{target}\_\text{lidar}}$, then chained through the IMU. `prior_cov` comes
from the bundle-adjustment marginal.

**(c) Hand-eye (targetless)** — solve $A_i X = X B_i$ where $A_i$ is the IMU
motion and $B_i$ the other sensor's motion over the same interval, for the unknown
extrinsic $X = T_{F_e\_\text{sensor}}$. Used for **LiDAR↔IMU** and **cam↔IMU**
when no target is available in the field. The classic Tsai–Lenz or screw-
congruence solution seeds a small NLLS refinement; `prior_cov` is the
refinement's Hessian inverse. This is also exactly the structure a **degenerate**
rig motion will under-constrain — hand-eye needs rotation about ≥2 axes — so the
app **emits an observability report** (§5.3) and refuses to write a tight prior it
cannot justify.

### 5.2 The prior is a mean *and* a covariance

Every method must emit `prior_cov`. The covariance is what lets stage 3 *move*
the extrinsic the right amount: a tight factory mount → small `prior_cov` → L3
barely moves it; a quick-release or field-adjusted mount → large `prior_cov` → L3
is free to refine it. The mapping from "how it was calibrated" to a default
`prior_cov` when a method does not report one:

| Source | Default trans 1σ | Default rot 1σ | `refine_online` default |
|---|---|---|---|
| `Kalibr` (with reported cov) | from Kalibr marginal | from Kalibr marginal | per-config (usually true for cam, false for rigid LiDAR mount) |
| `TargetBoard` | 5 mm | 0.2° | per-config |
| `HandEye` | 2 cm | 0.5° | true (targetless ⇒ less certain) |
| `Manual` (CAD/tape) | 2 cm | 2° | true (loose; let the graph fix it) |
| `OnlineRefined` (last mission) | last posterior σ | last posterior σ | true |

These are **defaults**; the config (§10) overrides per sensor. The numbers are
documented engineering priors, not measurements — they exist so a rig with a
hand-entered extrinsic still runs (loosely) instead of trusting CAD as gospel.

### 5.3 Observability-aware acquisition

Stage 2 apps **must** report which DoF of the extrinsic the calibration motion
actually constrained, as an `ObservabilityReport` (spec 01 §3.4) in the sensor's
named frame. An under-excited hand-eye run constrains rotation well but
translation poorly along the unexcited axes; writing a tight isotropic
`prior_cov` would lie to stage 3. So the app sets `prior_cov` **anisotropically**
from the acquisition Hessian and warns if any axis is below a conditioning
threshold. This is the offline twin of the online observability gate (§6.4), and
it is the discipline FAST-LIO's bare-value extrinsic load entirely lacks.

---

## 6. Stage 3 — online refinement as L3 graph variables

**What:** the refine-flagged extrinsics become **GTSAM variables** in the L3
factor graph (spec 01 §7.4, the back-end), anchored by their stage-2 prior,
observed through every keyframe's geometric factors, and **gated by online
observability** so they only move when the data can move them. L3 is the **single
authority**: it owns the posterior, and it publishes refined `CalibrationSet`
snapshots that L2/L4 pick up at keyframe boundaries (spec 01 §5.3, §9 below).

This is where Meridian most deliberately generalises the reference. FAST-LIO appends
**one** extrinsic to the **filter** state and toggles it with **one** boolean
`extrinsic_est_en` (`laserMapping.cpp:786`), with no observability gate and no
visibility to the global optimiser. Meridian makes each refine-flagged extrinsic a
**graph variable** — gated, visible to global optimisation and loop closure,
published with provenance. The architecture demanded exactly this (arch §3:
"online extrinsic variables" in L3; spec 01 §5.3 design note on *why extrinsics
are not in `NavState`*).

### 6.1 Variable keys and the anchoring prior factor

For each `Extrinsic e` with `refine_online == true`, L3 creates a GTSAM `Pose3`
variable with a dedicated key namespace, e.g. `X_calib(sensor_id)`, distinct from
the keyframe pose keys `X(kf_id)`. At graph initialisation L3 adds **one**
`PriorFactor<Pose3>` on `X_calib(sensor_id)`:

$$
r_{\text{prior}} = \mathrm{Log}\!\big(\,\hat T_{F_e\_s}^{-1}\, T_{F_e\_s}\,\big)
\in \mathbb{R}^6,
\qquad
\Omega_{\text{prior}} = \big(\texttt{prior\_cov}\big)^{-1},
$$

with $\hat T_{F_e\_s}$ the stage-2 prior mean. This prior is what keeps an
*unexcited* extrinsic pinned to its offline value: with no informative geometric
factor, the MAP estimate stays at the prior (zero net force). It is the formal
realisation of the §2 invariant "REFINED variables are anchored by their stage-2
prior."

### 6.2 The geometric observation factor (how a keyframe constrains an extrinsic)

The information about $T_{F_e\_s}$ enters through the same registration residuals
that constrain the trajectory, now differentiated **also** w.r.t. the extrinsic.
For a LiDAR point $p_L$ matched to a map plane $(n,d)$ in `map`, with keyframe
pose $T_{\text{map}\_F_e}$ (variable `X(kf)`) and extrinsic $T_{F_e\_s}$ (variable
`X_calib(s)`):

$$
r(p_L) = n^\top\!\Big( T_{\text{map}\_F_e}\, T_{F_e\_s}\, p_L \Big) + d .
$$

The Jacobian w.r.t. the extrinsic's right-tangent $\delta\xi_s=[\rho_s;\phi_s]$
(spec 01 §3.1 convention) is, writing $q = T_{F_e\_s}\,p_L$ and
$R_{mF} = R(T_{\text{map}\_F_e})$:

$$
\frac{\partial r}{\partial \delta\xi_s}
= n^\top R_{mF}\,\big[\, R_{F_e\_s} \;\big|\; -R_{F_e\_s}\,(p_L)^\wedge \,\big]
\;\in\;\mathbb{R}^{1\times 6},
$$

i.e. the translation block is $n^\top R_{mF} R_{F_e\_s}$ and the rotation block
is $-\,n^\top R_{mF} R_{F_e\_s}\,(p_L)^\wedge$ (ordering $[\rho;\phi]$). This is
the standard point-to-plane Jacobian extended by one more right-multiplied
transform; it is identical in form to the trajectory Jacobian, which is why L3
can assemble it with the same factor machinery (spec 04 derives the trajectory
half; this spec adds the extrinsic column). The camera photometric factor
contributes the analogous $\partial I/\partial\delta\xi_s$ through the image
gradient and projection Jacobian.

> **Static, shared extrinsic across keyframes.** The extrinsic is **one** variable
> shared by *every* keyframe factor for that sensor (it is rig-constant within a
> regime). That accumulation across many keyframes is precisely what makes a
> weakly-per-keyframe-observable extrinsic globally observable over a trajectory
> with enough motion diversity — the same reason hand-eye needs multi-axis
> rotation (§5.1c). It also means a loop closure that re-optimises old keyframes
> *re-informs* the extrinsic, a strict gain over FAST-LIO's filter, where the
> extrinsic only ever sees the sliding window.

### 6.3 Marginalisation and the snapshot

After each iSAM2 update (spec 01 §7.4 `optimize()`), L3 reads back each
`X_calib(s)` estimate **and its marginal covariance**, packs them into a fresh
`CalibrationSet` (mean → `Extrinsic::T_parent_child`, marginal → updated
`prior_cov` for downstream consumers' information, bumped `version`,
`source = OnlineRefined`), and publishes it (§9). The published `prior_cov` is the
*current posterior* covariance, so a consumer can see the extrinsic tightening
over time — and so the **next mission** can seed its stage-2 prior from this
posterior (the §2 loop).

### 6.4 The online observability gate (do not refine what you cannot see)

A refine-flagged extrinsic is only *moved* when it is currently observable.
After each update L3 computes, for each `X_calib(s)`, the per-axis conditioning of
its marginal information $\Omega_s = \Sigma_s^{-1}$ (eigenvalues of the $6\times6$
block, or the diagonal in the sensor frame), normalised to $[0,1]$ exactly as the
front-end's `ObservabilityReport` (spec 01 §3.4). The gate, per axis:

* **score $\geq \theta_{\text{hi}}$** (default 0.5): the axis is well-observed;
  accept the refined value on that axis.
* **score $< \theta_{\text{lo}}$** (default 0.1): the axis is degenerate; **clamp**
  that axis to the prior (do not let the prior's null space drift), and emit
  `event("calib/axis_unobservable", sensor, axis)`.
* **between**: blend toward the prior proportionally (the soft prior already does
  this via $\Omega_{\text{prior}}$; the gate is the explicit, plottable signal).

On top of the soft gate, the **hard bounds** `max_drift_trans_m` /
`max_drift_rot_deg` (§3.1) are checked: a refined value outside the box around the
prior is **rejected**, the variable reset to the prior, and
`event(WARN, "calib/refine_rejected_bounds", sensor)` raised (§11). Belt
(soft prior) + braces (hard box) + observability signal — because a silently bad
extrinsic poisons the entire map and every loop closure.

### 6.5 Photometric intrinsic refinement (front-end visual stage, not L3)

The inverse-exposure intrinsic (`inv_expo_prior`, §3.2) is a **per-frame
nuisance**, not a rig constant, so it is estimated **inside L2** (the FAST-LIVO2
sparse-direct path) when the visual stage exists — today the front-end is
LiDAR-inertial only and this refinement is **dormant** — and is
**not** crossed to L3 as a graph variable (R1: implementation-internal). It is
reported on the telemetry bus (§12) and its prior is re-seeded from a slow running
mean so a clean exposure recovers the prior. This keeps L3's graph minimal (the
same discipline as `kinematics_included = false` by default, spec 01 §6.4) and
matches FAST-LIVO2, which estimates exposure per frame in the front-end.

### 6.6 What crosses L2↔L3 as a live variable (and what does not)

To preserve the clean information budget (spec 01 §6.4, MUST-FIX #3): a refined
extrinsic crosses **L3→L2** only as a *versioned snapshot* (spec 01 §5.3), never
as a live shared mutable. Each geometric extrinsic (LiDAR, camera, GNSS lever-arm)
that is refine-flagged is refined in **exactly one** place — **L3**, the
authoritative multi-sensor owner. The front-end consumes the published
snapshot and resets its linearisation; it does not separately ship the same
extrinsic back as an L3 variable, so no information is double-counted.
**Rule:** any single
extrinsic is refined in exactly one place; in the production CT system that place
is L3.

---

> **Identity-default guard on configured extrinsics.** A defaulted (identity)
> extrinsic is a syntactically valid value, so absence is only knowable at parse
> time: the loader records `extrinsic_set` per sensor. A missing camera extrinsic
> withholds the CamLink entry from the `CalibrationSet`, which makes the front-end
> disable the visual stage loudly (`frontend/visual/disabled`) instead of running
> with a sideways frustum; a missing LiDAR extrinsic warns once at startup
> (`sensors/lidar/extrinsic_default`) since LIO can legitimately run at identity on
> a bench rig but must never mistake it for a calibration.

## 7. Temporal calibration (time offsets)

Temporal calibration is the *time* twin of the geometric extrinsic: the
per-sensor offset $t_d$ such that an event the sensor stamps at $t_{\text{sensor}}$
occurred at $t_{F_e} = t_{\text{sensor}} - t_d$ on the estimation timeline. It is
**split across two specs by responsibility**, and this section states the seam.

### 7.1 What lives in L0 (spec 02 §7) vs. here

* **L0 (`meridian_time`, spec 02 §5, §7)** owns the *clock-level* timeline: PTP/PPS
  hardware discipline (spec 02 §5.2), the per-device offset/skew `ClockModel`
  (spec 02 §5.3), and the **software time-offset estimator** (spec 02 §7: motion
  cross-correlation + a 2-state offset/skew Kalman filter) that places every raw
  sample on the Meridian timeline $t_H$ *before* it leaves L0. The residual timing
  uncertainty $\sigma_{o_s}$ it publishes is folded into measurement noise
  (spec 02 §9.4).
* **`meridian_calib` (here)** owns the *calibration-level* time offset: the
  `Extrinsic::time_offset_ns` **prior** (a calibrated $t_d$ from Kalibr's
  camera–IMU time offset, §8) and the policy for whether $t_d$ is **refined
  online** as a graph variable (§7.3). It treats $t_d$ as a calibration parameter
  exactly parallel to the geometric extrinsic: a prior with uncertainty, a
  refine flag, an observability gate.

The relationship: L0's `ClockModel` corrects *clock drift/skew* (a hardware
property), while `meridian_calib`'s $t_d$ captures the *fixed sensor-to-$F_e$ phase*
(a mounting/pipeline property — e.g. the camera's fixed exposure-readout latency
relative to the IMU). They compose:
$t_{F_e} = \text{ClockModel}.to\_meridian(t_{\text{dev}}) - t_d$. L0 makes all
sensors agree on *when now is*; `meridian_calib` makes the **fused geometry** agree
on *which $F_e$ instant a sensor's measurement corresponds to* — which the
estimator needs to place each sensor's measurement on the trajectory correctly.

### 7.2 The temporal-offset prior

`time_offset_ns` / `time_offset_std_ns` (§3.1) are seeded from the Kalibr
camera–IMU time offset (§8). For a hardware-synced rig
([`DATASET.md`](../DATASET.md): the Newer College rig is FPGA-synchronized — the
Ouster and the Alphasense share the FPGA clock) the prior
is $t_d \approx 0$ with small $\sigma$; for a software-synced rig the prior is
whatever L0's §7 estimator converged to, with its $\sigma_{o_s}$.

> **Resolved — constant per-sensor stamp correction is wired through the intake.**
> `sensors.camera.time_offset_ms` and `sensors.lidar.time_offset_ms` apply a
> constant correction onto the body-IMU timeline ($t_{corr} = t_{sensor} +
> \text{offset}$) once, in the pipeline's `ingest()` overloads — *before*
> validation and aggregation, so every stamp-driven decision (monotonicity, group
> assembly, image–sweep matching) sees corrected time, identically live and in
> replay. Online $t_d$ refinement remains off by default (§7.3).
>
> **Calibration-session timeshifts do NOT automatically transfer to a recording**
> — this is a hard rule, established twice on an earlier benchmark set whose
> Kalibr sessions reported >100 ms shifts while the bag A/B verdict was **0 ms**
> (the recorded streams were already stamp-disciplined; applying the session
> value degraded patch convergence and ATE, the opposite sign pushed every image
> outside its sweep's matching window). Hardware sync paths differ per
> sensor and per recording: set a `time_offset_ms` only after an empirical A/B on
> the actual data (visual-funnel telemetry + ATE), never from the session value
> alone. The importer therefore reports the session timeshift but does not
> own the config key. (The Newer College camchains report +1.8…+2.0 ms cam→imu
> over the same FPGA path as the recordings, so the value is expected to transfer
> — it still gets the A/B before sub-centimetre ATE deltas are trusted.)

### 7.3 Online temporal refinement (optional)

When `refine_time_online == true`, $t_d$ becomes a scalar graph variable. Its
observation Jacobian is the time-derivative of the geometric residual: a point's
residual sensitivity to $t_d$ is the residual gradient times the platform velocity
at that instant,

$$
\frac{\partial r}{\partial t_d}
= n^\top R_{mF}\,\Big( \tfrac{d}{dt}\big[T_{F_e\_s}(t)\,p_L\big]\Big)
\approx n^\top R_{mF}\,\big(\, v_{F_e} + \omega_{F_e}\times q \,\big),
$$

i.e. **time offset is only observable under motion** (the velocity term must be
nonzero), exactly mirroring the L0 cross-correlation gate (spec 02 §7.2: "runs
only during motion"). So $t_d$ refinement is observability-gated identically to
§6.4, and clamped to the prior at low motion. By default online $t_d$ refinement
is **off** (the Kalibr prior + L0's estimator suffice); it is a documented,
config-switchable seam.

---

## 8. Seeding the priors from dataset calibration files

This is the concrete bridge from [`DATASET.md`](../DATASET.md) — the benchmark
ships per-collection Kalibr `camchain-imucam` files (the rig was recalibrated
between collections, so park must not run with the quad calibration), the shared
rig-transform YAML, and the OS0-128 beam table — to a populated
`CalibrationSet v0`. The dataset calibration seeds the offline stage; Meridian's
back-end then refines extrinsics online, so it is a *prior*, not a hard constant.

### 8.1 The Kalibr importer (`meridian_calib::load_kalibr`)

`meridian_calib` ships a loader that parses the two Kalibr YAML families and produces
a `CalibrationSet` (until it lands, `tools/import_ncd.py` performs the same
seeding offline, writing the calibration blocks of the deployed
`newer-college-*.yaml` configs from the same files):

* **`*-camchain-imucam.yaml`** → the camera: `IntrinsicsCamera`
  (`intrinsics: [fx, fy, cx, cy]`, `distortion_model`, `distortion_coeffs`,
  `resolution` — the Newer College Alphasense cam0 imports as pinhole +
  `Equidistant` $k_1..k_4$, 720×540), and `T_cam_imu` → inverted to
  `Extrinsic{parent=ImuLink,
  child=CamLink, T_parent_child = inv(T_cam_imu)}`; plus `timeshift_cam_imu` →
  `time_offset_ns` (Kalibr reports seconds; convert to int64 ns). The imported
  `time_offset_ns` populates the **prior**; the constant intake correction is the
  config's `sensors.*.time_offset_ms` (§7.2), which is set only after the bag A/B
  — the importer reports the session value, it does not write the config key.
* **`*-imu.yaml`** → `imu_acc_noise = accelerometer_noise_density`,
  `imu_gyr_noise = gyroscope_noise_density`,
  `imu_acc_bias_rw = accelerometer_random_walk`,
  `imu_gyr_bias_rw = gyroscope_random_walk` (spec 01 §5.3 fields). When the
  dataset ships no Kalibr IMU file (Newer College does not), the IMU noise rides
  the config `cov_*` (variance) path of §4.2 instead — subject to the same
  datasheet sanity check.
* **LiDAR↔IMU**: Kalibr does not natively calibrate LiDAR; the dataset ships the
  rig chain in its own format. The
  importer has a per-dataset adapter (`load_newer_college_extrinsics`) that fills
  `Extrinsic{child=OsSensor0}` from
  `calib/os_imu_lidar_transforms.yaml` — the `os_sensor_to_as_imu` entry, since
  the bags' clouds are stamped in the `os_sensor` frame — tagging
  `source = Vendor`/`Manual` as appropriate (the chain mixes Ouster factory
  numbers with CAD-derived links). Two caveats the adapter owns: the file's
  quaternions are **(qx, qy, qz, qw)**, qx first; and the os→Alphasense rotation
  is a 180° flip about X, not identity.

### 8.2 Field mapping table (Kalibr → Meridian)

| Kalibr field | Meridian target | Note |
|---|---|---|
| `cam0.intrinsics [fx,fy,cx,cy]` | `IntrinsicsCamera{fx,fy,cx,cy}` | direct |
| `cam0.distortion_model` | `IntrinsicsCamera::model` | `radtan`→`RadTan`, `equidistant`→`Equidistant` |
| `cam0.distortion_coeffs` | `IntrinsicsCamera::coeffs` | radtan: $k_1,k_2,p_1,p_2[,k_3]$ |
| `cam0.resolution` | `IntrinsicsCamera{width,height}` | direct |
| `cam0.T_cam_imu` (4×4) | `Extrinsic::T_parent_child = inv(·)` | Meridian parent=$F_e$=imu; Kalibr gives cam←imu |
| `cam0.timeshift_cam_imu` (s) | `Extrinsic::time_offset_ns` | ×1e9, round to int64; `time_offset_std_ns` from config or Kalibr report |
| `imu.accelerometer_noise_density` | `CalibrationSet::imu_acc_noise` | continuous-time **std-dev**; copied directly (Kalibr already reports a density) |
| `imu.gyroscope_noise_density` | `CalibrationSet::imu_gyr_noise` | continuous-time **std-dev**; copied directly |
| `imu.accelerometer_random_walk` | `CalibrationSet::imu_acc_bias_rw` | bias RW **std-dev**; copied directly |
| `imu.gyroscope_random_walk` | `CalibrationSet::imu_gyr_bias_rw` | bias RW **std-dev**; copied directly |

> **Kalibr vs. config-YAML path differ on the IMU-noise convention.** Kalibr's
> `*-imu.yaml` reports the densities as **standard deviations**, so `load_kalibr`
> copies them straight into the (std-dev) `CalibrationSet` fields — no square root.
> The runtime config path is different: `sensors.imu.cov_acc` / `cov_gyr` (and the
> bias-RW twins) are **variances**, and `calibrationFromConfig` takes their square
> root before populating the same fields (§4.2). Same destination field, two source
> conventions — Kalibr already-std, config-YAML squared.

### 8.3 Prior covariance and refine flags on import

Kalibr reports point estimates; it does not, by default, emit a usable parameter
covariance for the extrinsic. The importer therefore sets `prior_cov` from the
§5.2 default table keyed on `source`, **overridable** by the config block
(§10), and sets `refine_online` from config per sensor (default: camera–IMU
extrinsic `true` because thermal/mounting drift is real; rigid LiDAR–IMU mount
`false`; GNSS lever-arm `false`, measured by tape). `calibrated_at` is set to the
dataset's recording epoch (provenance), `version = 0`, `source = Kalibr`. The
resulting `CalibrationSet v0` is the seed handed to the pipeline at construction
(spec 01 §5.3: "owned by the system bootstrap, shared read-only with the
front-end"), and it is precisely the prior that L3 then refines — exercising the
online path on every dataset run.

---

## 9. The `ICalibrationProvider` interface and snapshot service

The calibration substrate is served behind one small interface so consumers never
touch a mutable shared object and the L3-writer / many-reader contract is
mechanical. This interface is **owned by `meridian_calib`** (it depends on nothing in
L0–L6; it deals only in `meridian_common` value types).

### 9.1 The interface

```cpp
// meridian_calib/include/meridian/calib/icalibration_provider.hpp   (NO ros)
namespace meridian {

// Read side: any layer holds this and pulls an immutable snapshot. The returned
// shared_ptr<const> is valid as long as the holder keeps it (spec 01 §2.4).
class ICalibrationProvider {
public:
  virtual ~ICalibrationProvider() = default;
  // Current calibration. Cheap: returns the latest published snapshot pointer.
  virtual std::shared_ptr<const CalibrationSet> current() const = 0;
  // The monotonically increasing version of `current()` (spec 01 §5.3). A reader
  // compares this to its cached version to know "calibration changed, re-cache."
  virtual std::uint32_t version() const = 0;
};

// Write side: implemented by meridian_calib, the L3 back-end calls publish() after
// each optimize() that moved a refine-flagged extrinsic (§6.3). Exactly ONE
// writer (the back-end thread); many readers (L1/L2/L4 threads).
class CalibrationStore final : public ICalibrationProvider {
public:
  explicit CalibrationStore(std::shared_ptr<const CalibrationSet> seed_v0);  // §8
  std::shared_ptr<const CalibrationSet> current() const override;   // lock-free read
  std::uint32_t version() const override;
  // Publish a NEW immutable set (copy-on-write); bumps version atomically.
  void publish(std::shared_ptr<const CalibrationSet> refined);      // back-end only
};

} // namespace meridian
```

`CalibrationStore::current()` is an atomic load of a `shared_ptr<const ...>`
(or a seqlock around it); `publish()` is an atomic store of a freshly-built set.
No reader ever sees a half-updated extrinsic. This is the concrete mechanism for
spec 01 §5.3's "versioned `CalibrationSet` snapshot, never a live shared mutable"
and the L3→L2 boundary row in spec 01 §9.

### 9.2 Version semantics

* The seed is `version = 0` (§8.3).
* Each `publish()` increments the set version. A reader (e.g. the front-end at a
  keyframe boundary, spec 01 §7.3 `set_calibration`) compares
  `provider->version()` to its cached value; on change it copies what it needs and
  **resets its linearisation** (spec 01 §5.3: "version counter lets the front-end
  detect 'calibration changed, reset linearization'").
* The packet provenance field `KeyframePacket::calib_version` (spec 01 §6.1)
  records *which* snapshot produced a keyframe, so a developer reading a bag can
  attribute a geometry shift to a recalibration. This closes the provenance loop:
  every keyframe knows its calibration version; every refined extrinsic knows its
  source and version (§3.1).

### 9.3 Who reads what (consumer map)

| Consumer | Reads | When | Reaction to a version bump |
|---|---|---|---|
| L1 preprocess | (none directly; deskew is internal to L2) | — | — |
| L2 front-end | extrinsics (transform sensors → $F_e$), cam intrinsics, IMU noise | every scan; re-cache at keyframe boundary | reset linearisation (§9.2) |
| L3 back-end | the refine-flagged extrinsics it *owns*; priors | each `optimize()` | it is the writer |
| L4 map | `T_body_cam` + cam intrinsics for **colourisation** | per integrated keyframe | use the snapshot pinned in `KeyframePacket::T_body_cam` (spec 01 §6.1) for reproducibility, *not* live |

> **Colourisation reproducibility.** L4 colourises using the **`T_body_cam`
> snapshot carried in the `KeyframePacket`** (spec 01 §6.1 field 6), not the live
> provider, so a later extrinsic refinement does not retroactively mis-colour an
> already-integrated keyframe. When a loop correction triggers an nvblox region
> rebuild (MUST-FIX #4, spec 01 §7.5), the rebuild re-reads the stored per-keyframe
> `T_body_cam`, keeping colour consistent with the geometry of that keyframe.

---

## 10. Configuration schema

Calibration config lives under the `meridian:` tree (arch §8.2). The file paths to
the Kalibr/dataset calibration are themselves config fields, so retargeting a
dataset (or a Newer College collection — the rig was recalibrated between them)
is a per-config path change, not code.

```yaml
calib:
  # --- where the stage-1/2 priors come from (§8) ---
  source_file: bags/newer-college/calib/collection1/cam0-1/_2021-07-01-13-36-53-cam0-1-camchain-imucam.yaml
  source_kind: kalibr            # kalibr | newer_college | meridian_native
  rig_file: bags/newer-college/calib/os_imu_lidar_transforms.yaml  # LiDAR<->IMU chain (§8.1)
  # imu_noise_file: <Kalibr *-imu.yaml>  # Allan-variance (§4.2); Newer College ships
  #                                      # none — IMU noise rides sensors.imu.cov_*
  estimation_frame: imu_link     # F_e (spec 01 §2.3)

  # --- global default; per-sensor override below (§6) ---
  refine_online_default: false

  # --- per-sensor extrinsic priors + refine flags (override §5.2 defaults) ---
  # Exactly three extrinsics: one LiDAR, one camera, one GNSS antenna.
  extrinsics:
    - child: os_sensor0          # the LiDAR
      refine_online: false       # rigid factory mount
      prior_trans_std_m: 0.005
      prior_rot_std_deg: 0.2
      max_drift_trans_m: 0.05
      max_drift_rot_deg: 1.0
    - child: cam0                # the camera
      # T_imu_cam, as [tx,ty,tz,qx,qy,qz,qw]. REQUIRED for the visual stage: with it
      # absent the importer leaves an identity transform, which silently breaks
      # promotion of map points into the camera frame and the visual map stays empty.
      # validate() fails fast on a missing camera extrinsic rather than defaulting it.
      refine_online: true        # thermal/mounting drift is real
      prior_trans_std_m: 0.01
      prior_rot_std_deg: 0.3
      max_drift_trans_m: 0.05
      max_drift_rot_deg: 1.5
    - child: gnss_link           # antenna lever arm
      refine_online: false       # measured; usually fixed
      prior_trans_std_m: 0.02

  # --- temporal calibration (§7) ---
  temporal:
    refine_time_online: false    # off by default; Kalibr prior + L0 estimator suffice
    cam_imu_offset_std_ns: 200000 # 0.2 ms prior σ if Kalibr didn't report one

  # --- photometric intrinsic (§4.3, §6.5) ---
  photometric:
    refine_inv_expo_online: true # FAST-LIVO2 path (future visual stage; dormant)
    inv_expo_std: 0.1

  # --- online refinement gate (§6.4) ---
  refine_gate:
    obs_score_hi: 0.5            # accept above
    obs_score_lo: 0.1            # clamp-to-prior below
    publish_min_version_gap_s: 1.0  # rate-limit snapshot publishes (§9, §12.cost)
```

### Validation rules (`Config::validate()`, arch §8.3)

* `estimation_frame` must be a frame named in `sensors:` (cross-spec consistency).
* Each of the three sensors in `sensors:` (spec 02 §10) — LiDAR, camera, GNSS —
  must have either a prior in `calib.extrinsics` or a default; a sensor with
  **no** extrinsic and `refine_online=false` fails fast (you cannot fuse a sensor
  you cannot place).
* **The camera extrinsic (`T_imu_cam`, `[tx,ty,tz,qx,qy,qz,qw]`) is mandatory
  whenever the visual stage is enabled — `validate()` must fail fast on its
  absence, never silently default to identity.** An identity `T_imu_cam` does *not*
  raise `have_cam_extrinsic_` to false (the importer still inserts a `CamLink`
  `Extrinsic`, so the visual stage thinks it has a placement); instead it places
  every map point at the IMU origin, which silently breaks promotion and leaves the
  visual map empty for the whole run. A missing-but-defaulted extrinsic is the worst
  failure class — it looks healthy and produces nothing — so the validator rejects
  it up front rather than letting the default-constructed identity through.
* `max_drift_*` must be ≥ a few prior σ (a box tighter than the prior is a
  configuration error).
* An extrinsic is refined in exactly one place — L3 (§6.6). `validate()` rejects
  any configuration that names a second refinement owner for the same extrinsic.

---

## 11. Failure modes and recovery

| # | Failure | Detection | Recovery | Telemetry |
|---|---|---|---|---|
| F1 | **Missing/ malformed calibration file** | loader parse error at bootstrap | refuse to start (no silent default extrinsic — unlike a mistyped FAST-LIO `nh.param`) with a precise message | log `ERROR`, exit |
| F2 | **Prior with no covariance** (bare Kalibr point estimate) | importer sees no cov | apply §5.2 default `prior_cov` by `source`; warn | `event(WARN,"calib/default_prior_cov",sensor)` |
| F3 | **Online extrinsic drifts in its null space** (unobservable axis) | §6.4 axis score `< θ_lo` | clamp that axis to prior; keep observable axes | `event("calib/axis_unobservable",sensor,axis)` |
| F4 | **Refined value escapes hard bounds** | §3.1 `max_drift_*` exceeded | **reject** update, reset variable to prior, freeze its refinement for a cool-down | `event(WARN,"calib/refine_rejected_bounds",sensor)` |
| F5 | **Refinement diverges / NaN** in iSAM2 block | non-finite marginal | drop the calib variable from the graph for the mission, fall back to FIXED prior, continue | `event(ERROR,"calib/refine_diverged",sensor)` |
| F6 | **Two owners refine the same extrinsic** (config error) | `validate()` cross-check (§10) | fail fast at start | `ERROR` |
| F7 | **Snapshot publish storm** (extrinsic oscillating) | publish rate > limit | rate-limit `publish()` to `publish_min_version_gap_s` (§10); coalesce | `scalar("calib/publish_rate_hz")` |
| F8 | **Temporal prior wrong sign/scale** (Kalibr s vs ns, or sign convention) | residual time-jitter spikes; L0 §7.3 `SyncResidualHigh` | importer unit-test asserts the sign on a known dataset (§15); at runtime, large $t_d$ residual → fall to FIXED prior | `event(WARN,"calib/temporal_suspect",sensor)` |
| F9 | **Mid-mission rig change** (sensor physically moved) | persistent large refinement against tight prior, or operator command | promote that extrinsic to a loose prior (a *new regime*), re-anchor; never silent | `event("calib/regime_change",sensor)` |

The governing principle, matching the architecture's robustness goals
(arch §13.4, switchable constraints, GNC, observability→noise): **a calibration
parameter that cannot be trusted *right now* is reverted to its FIXED prior and
the reversion is visible** — never silently estimated into garbage. This is the
calibration analogue of dropping a Failed sensor from the fuse (spec 02 §9.4).

---

## 12. Debug / introspection hooks

Calibration is invisible in FAST-LIO except for a one-line log of
`offset_R_L_I` / `offset_T_L_I` to a file (`laserMapping.cpp:138`). Meridian routes
calibration introspection through the telemetry bus (arch §10.1, `TelemetrySink`)
so the operator can *watch every extrinsic converge*. Keys (all gated by
`debug.publish_*` and rate-limited, arch §10.6):

| Signal | Telemetry channel & key | Notes / FAST-LIO contrast |
|---|---|---|
| Each online extrinsic (live) | `pose("calib/T_imu_sensor/<sensor>")` per sensor | arch §10.2 already promises `pose("calib/T_imu_lidar")` / `pose("calib/T_imu_cam")`; FAST-LIO logs only the one LiDAR extrinsic, to a file |
| Extrinsic posterior σ trend | `vec("calib/cov_diag/<sensor>", 6)` | 6-axis 1σ; watch it tighten — degeneracy is **plottable** |
| Per-axis observability of the calib variable | `vec("calib/observability/<sensor>", 6)` | rendered as the same 6-bar hexagon marker as the trajectory (arch §10.4) |
| Drift from prior | `vec("calib/drift_from_prior/<sensor>", 6)` + `marker` showing the box | shows headroom before the §3.1 hard bound |
| Snapshot publish | `event("calib/snapshot_published", version)` + `scalar("calib/version")` | every consumer can correlate a geometry shift to a version |
| Temporal offset | `scalar("calib/time_offset_ns/<sensor>")` (+ σ) | the §7 $t_d$, live |
| Photometric intrinsic | `scalar("calib/inv_expo")` (+ σ) | the §6.5 FAST-LIVO2 nuisance, visible |
| Regime/gate events | `event(...)` for F3/F4/F5/F9 (§11) | recovery is **visible**, not a silent `continue` |
| Offline acquisition report | `event("calib/offline_obs", sensor, scores)` | stage-2 apps emit their §5.3 observability so a bad calibration is caught **before** the mission |

A dedicated **rviz overlay** (arch §10.4) draws, per refine-flagged sensor: the
current $T_{F_e\_s}$ frame triad, the prior triad (ghosted), the drift box, and
the 6-bar observability hexagon — so an operator literally sees the camera's
extrinsic settle, with red bars where motion has not yet excited an axis.

---

## 13. Threading, ownership, lifetime summary

| Item | Ownership (spec 01 §2.4) | Thread | Lifetime |
|---|---|---|---|
| `CalibrationSet v0` (seed) | Shared-immutable; built by bootstrap (§8) | constructed on main, read everywhere | whole mission (until superseded) |
| `CalibrationStore` | owned by `meridian_pipeline` (one instance) | written: back-end thread; read: all | whole mission |
| Published snapshot | Shared-immutable, copy-on-write | published by back-end; read by L1/L2/L4 | until the next reader releases it (refcount) |
| Calib graph variables (`X_calib`) | GTSAM `Values` inside L3 (implementation type, does **not** cross a boundary, R1) | back-end thread | whole mission |
| Photometric intrinsic (`inv_expo`) | front-end-internal window variable (does not cross a boundary, R1) | front-end thread | window lifetime |
| `Extrinsic::T_body_cam` snapshot in a `KeyframePacket` | Value, copied into the packet (spec 01 §6.1) | front-end → back-end → map | with the keyframe |
| Offline calibration apps | standalone (`meridian_tools`) | their own process | offline only |

The single-writer/many-reader rule (back-end writes, everyone reads) plus
Shared-immutable copy-on-write is what makes the live extrinsic safe to read on
the hot front-end thread without a lock (spec 01 §5.3 data-race fix). It is the
same ownership shape as the keyframe clouds (spec 01 §6.3): one writer publishes
an immutable thing, many readers share it by `shared_ptr`.

---

## 14. Worked example: the camera extrinsic from Kalibr file to refined graph variable

Concrete trace, to make the three stages tangible (mirrors spec 01 §10's packet
life). The camera extrinsic is the natural worked example because it is the
default refine-flagged transform (thermal/mounting drift is real) and it is
observed through the FAST-LIVO2 photometric factors.

1. **Bootstrap.** `meridian_calib::load_kalibr("…/camchain-imucam.yaml")` +
   `load_newer_college_extrinsics(...)` build `CalibrationSet v0`. Config sets
   the camera (`cam0`) `refine_online: true`, `prior_trans_std_m: 0.01`,
   `prior_rot_std_deg: 0.3`, `max_drift_trans_m: 0.05`, `max_drift_rot_deg: 1.5`.
   Its `Extrinsic{source=Kalibr, version=0, T_parent_child = inv(T_cam_imu)}` is
   created. The LiDAR `os_sensor0` gets `refine_online: false` (rigid); the GNSS
   antenna `gnss_link` likewise. `CalibrationStore` is seeded at `version = 0`.

2. **Stage-2 sanity (offline, optional).** Before the mission a targetless
   cam↔IMU hand-eye app (§5.1c) on a short wiggle log confirms the camera prior and
   reports `obs = [tx .9, ty .9, tz .4, rx .8, ry .8, rz .3]` — translation-Z and
   roll weakly excited. The app writes an **anisotropic** `prior_cov` reflecting
   that and emits `event("calib/offline_obs", cam0, scores)`. (FAST-LIVO2 has no
   such step; it would load a bare value.)

3. **Live, t = 0–30 s.** The front-end's visual stage (dormant today; the flow
   is shown for the full design) projects through the live $T_{F_e\_C}$
   from `provider->current()` for every sparse-direct photometric residual. L3 has
   created `X_calib(cam0)` with a `PriorFactor` (§6.1). Early keyframes are mostly
   forward driving → camera-roll and camera-Z stay near the prior (gate §6.4
   clamps them, `obs` bars red on the rviz hexagon).

4. **t = 35 s, a turn + grade.** The vehicle yaws through a junction and crests a
   ramp, exciting roll and Z. Over the next few keyframes `X_calib(cam0)` refines,
   its marginal σ tightens (`vec("calib/cov_diag/cam0")` visibly drops), the gate
   opens. L3 `publish()`es `CalibrationSet v1`; `event("calib/snapshot_published",
   1)`.

5. **Front-end picks it up.** At the next keyframe boundary the front-end sees
   `provider->version() == 1 ≠` its cached `0`, re-caches $T_{F_e\_C}$, resets its
   linearisation (§9.2). Subsequent `KeyframePacket`s carry `calib_version = 1`.

6. **t = 80 s, bounds guard.** A brief visually-degenerate stretch (a textureless
   wall) tries to push camera-Z 0.07 m — beyond `max_drift_trans_m = 0.05`. L3
   **rejects** the update, resets camera-Z to the prior, raises
   `event(WARN, "calib/refine_rejected_bounds", cam0)` (F4). The map and the
   colourisation are protected.

7. **Mission end.** The camera's posterior ($T_{F_e\_C}$ v_final + its marginal)
   is persisted as the **next** mission's stage-2 prior with
   `source = OnlineRefined` (§2 loop, §5.2 table). The next run starts
   already-tight.

At no point did L1/L4 see a mutable extrinsic; they read immutable snapshots. L3
was the only writer. The camera went from a loose Kalibr prior to a graph-refined,
observability-gated, bounds-protected extrinsic — the architecture's "offline
prior + online refinement as graph variables" made concrete.

---

## 15. Test plan

Per arch §9.4 (unit + replay under `colcon test`), ROS-free (the core is testable
without a middleware, arch §1).

**Unit (no ROS, synthetic):**
* **Kalibr round-trip:** load a known `camchain-imucam.yaml`, assert
  `IntrinsicsCamera` and the **inverted** `T_imu_cam`, and the `timeshift`→
  `time_offset_ns` sign+scale (guards F8). Golden file: the Newer College
  collection-1 camchain (`_2021-07-01-13-36-53-cam0-1-camchain-imucam.yaml`) —
  pinhole + `Equidistant` $k_1..k_4$, 720×540, timeshift +1.8008 ms.
* **Rig-chain import:** load `os_imu_lidar_transforms.yaml`, assert the
  `(qx, qy, qz, qw)` quaternion order and that `os_sensor_to_as_imu` comes out as
  the 180° X-flip, **not** identity — guards the classic wxyz/xyzw swap, which
  would silently mount the LiDAR upside-down.
* **Prior→PriorFactor:** build an `Extrinsic` with a known `prior_cov`, assert the
  GTSAM `PriorFactor` information equals `prior_cov.inverse()` in the right
  ordering (spec 01 §3.1).
* **Extrinsic Jacobian:** finite-difference the §6.2 point-to-plane-w.r.t.-extrinsic
  Jacobian against the analytic form; tolerance 1e-6.
* **Observability gate:** feed a marginal with a forced-degenerate axis; assert
  that axis clamps to prior and the others pass (§6.4).
* **Hard bounds:** drive a refinement past `max_drift_*`; assert reject+reset+event
  (F4).
* **Snapshot atomicity:** hammer `current()` from N reader threads while
  `publish()` runs; assert every read is a complete, self-consistent set and the
  version is monotonic (§9.1).

**Replay (the Newer College benchmark set, [`DATASET.md`](../DATASET.md);
ATE via `tools/eval_ate.py` against `gt/tum_asimu/`; tuning sequences only —
quad-hard is the holdout):**
* **Prior-vs-refined ATE:** run quad-easy / math-medium with `refine_online:
  false` (all fixed) and with the LiDAR extrinsic `refine_online: true`; assert
  ATE does **not regress** and improves on sequences with rotation diversity —
  proving the online path helps, not hurts. The camera twin of this test
  activates with the visual stage (auto-disabled on the equidistant cameras,
  §4.1).
* **Perturbed-prior recovery:** inject a 1° / 3 cm error into the LiDAR prior;
  assert the online refinement recovers it to within the posterior σ over a
  sequence with enough motion. On a degenerate stretch assert it **stays clamped**
  (does not chase noise).
* **Temporal seed:** assert the Kalibr `timeshift` seed plus L0's §7 estimator
  keeps L0 `SyncResidualHigh` (spec 02 §7.3) from firing on the FPGA-synced
  Newer College bags.

---

*End of spec 08. This spec amends spec 01 §5 with the `Extrinsic` /
`IntrinsicsCamera` extensions (§3) and adds the `ICalibrationProvider` /
`CalibrationStore` interface (§9) to the cross-cutting `meridian_calib` package
(arch §2, §4). It consumes spec 02 §7 (temporal) and is consumed by L2 (the CT
LIVO+GNSS front-end), L3, and L4.*
