# 01 — Interfaces & Data Types (The Contracts)

> **Spec status:** normative. This is the foundational spec: it fixes the
> **boundaries** between Meridian's layers L0–L6. Every other spec implements or
> consumes types declared here. If a later spec needs a new field on a boundary
> type, it must amend *this* document, not redefine the type locally.
>
> **Scope.** A ROS-agnostic core behind interfaces is only as good as the data
> types that cross them. This document defines those types *and* the abstract
> interfaces, and states for each boundary **the one thing it passes**, who
> **owns** it, on which **thread** it is produced/consumed, and how **long** it
> lives. The keystone is the L2→L3 keyframe-handoff contract (**MUST-FIX #1**),
> which is what makes the front-end swappable behind `IFrontEnd`. Meridian's
> front-end is the discrete LIO estimator (spec 04); the FAST-LIO2-style iEKF
> test oracle and the continuous-time estimator that previously sat
> behind this contract have both been removed, but §8 keeps the two-estimator
> demonstration because it is the proof the boundary is estimator-agnostic.
>
> Notation follows the shared block (course section 02): state $x$; rotation
> $R \in SO(3)$; position $p$; velocity $v$; gyro/accel biases $b_g, b_a$;
> gravity $g$; box-plus/minus $\boxplus, \boxminus$; Lie maps $\mathrm{Exp}/\mathrm{Log}$
> (capitalised, $\mathbb{R}^3 \leftrightarrow SO(3)$) and $\exp/\log$; hat $(\cdot)^\wedge$;
> residual $r$, measurement $z$, prediction $h(x)$, Jacobian $H$, covariance
> $\Sigma$, information $\Omega = \Sigma^{-1}$, Kalman gain $K$; LiDAR point $p_L$,
> plane $(n, d)$ with $n\cdot x + d = 0$; camera intensity $I$, projection $\pi$,
> intrinsics $K_{cam}$; trajectory pose $T \in SE(3)$.

---

## Table of contents

1. [Design philosophy: contracts, not classes](#1-design-philosophy-contracts-not-classes)
2. [Conventions: units, frames, time, ownership, threading](#2-conventions-units-frames-time-ownership-threading)
3. [Core math types](#3-core-math-types)
4. [Sensor sample types (the L0→L1 currency)](#4-sensor-sample-types-the-l0l1-currency)
5. [Calibration & extrinsic types](#5-calibration--extrinsic-types)
6. [The KeyframePacket — MUST-FIX #1](#6-the-keyframepacket--must-fix-1)
7. [Abstract interfaces](#7-abstract-interfaces)
8. [How two different estimators satisfy `IFrontEnd`](#8-how-two-different-estimators-satisfy-ifrontend)
9. [Boundary summary: the one thing each edge passes](#9-boundary-summary-the-one-thing-each-edge-passes)
10. [Worked example: a packet's life](#10-worked-example-a-packets-life)

---

## 1. Design philosophy: contracts, not classes

A SLAM system is a pipeline of estimators. If those estimators reach into each
other's representations — if L3 reads the front-end's internal state directly,
or L4 reads L3's GTSAM `Values` object — then swapping any one means rewriting
its neighbours, and the handoff between them can double-count information with
no contract to audit against. The reference systems already separate estimator
from measurement model (FAST-LIO's `esekfom` filter knows nothing about LiDAR;
its `h_share_model` functor is the only supplier of residuals $r$ and Jacobians
$H$). Meridian generalises that seed: every layer is a pure abstract base class
whose methods exchange the concrete value types defined in this document, and
nothing else.

Three rules make this real:

* **R1 — Value types cross boundaries; implementation types do not.** A
  `KeyframePacket` (§6) crosses L2→L3. A GTSAM `NonlinearFactorGraph`, a
  filter's internal covariance, a voxel-hash map cell — these are
  *implementation* types and never appear in an interface signature.

* **R2 — Each boundary passes exactly one thing.** Not "a pose and a cloud and
  some biases and a covariance scattered across three calls" — *one* aggregate
  value with a named, versioned schema. §9 tabulates the single currency of
  every edge. This is what lets us reason about double-counting (MUST-FIX #3):
  if information only ever crosses L2→L3 *inside a `KeyframePacket`*, we can
  audit exactly what is in it.

* **R3 — Ownership, thread, and lifetime are part of the type's contract, not
  folklore.** Every type below carries an explicit statement of who allocates
  it, who frees it, which thread touches it, and how long it must remain valid.
  Heavy buffers (point clouds, images) are passed by `shared_ptr` to immutable
  data so that the same bytes can be retained by L4's map store, consumed by
  L5's GICP, and meshed by L4's Poisson stage without a copy (this is the data
  structure MUST-FIX #4 requires).

The code in this spec is **C++-ish pseudo-declaration**: it is precise about
types, `const`-ness, and ownership, but elides obvious boilerplate
(`#include`s, namespaces beyond `meridian::`, trivial getters). The real headers
live in `meridian_common/include/meridian/common/` (value types) and in each
interface's owning layer package (spec 00 §5), e.g.
`meridian_frontend/include/meridian/frontend/ifrontend.hpp`.

---

## 2. Conventions: units, frames, time, ownership, threading

### 2.1 Units (SI, always)

| Quantity | Unit | Type |
|---|---|---|
| Length / position | metre | `double` |
| Angle (internal) | radian | (manifold; never stored as Euler) |
| Time | **nanoseconds since epoch**, integer | `Timestamp = int64_t` |
| Duration | nanoseconds, integer | `Duration = int64_t` |
| Linear velocity | m/s | `double` |
| Angular velocity | rad/s | `double` |
| Acceleration | m/s² | `double` |
| Intensity / reflectivity | sensor-native, documented per sensor | `float` |

**Why integer nanoseconds for time.** Floating-point seconds lose precision
over long missions: a `double` has ~15–16 significant digits, so a timestamp of
$1.7\times10^{9}$ s resolves to ~$0.1\,\mu s$ — borderline for PTP-disciplined
sub-microsecond sync (§4). Meridian stores absolute time as `int64_t` ns (the
ROS 2 `rclcpp::Time` convention). Per-point offsets (§4.2) are `int32_t` ns
relative to scan start, which spans ±2.1 s — far more than any single sweep.

### 2.2 Frames (REP-105 aligned)

```
map        global, gravity-aligned, ENU-ish; fixed; back-end (L3) owns it.
odom       smooth, drift-prone, continuous; front-end (L2) owns it.
base_link  robot body frame.
imu_link, os_sensorN, cam_link, gnss_link   sensor frames.
```

A transform is written $T_{A\_B} \in SE(3)$ meaning "maps a point expressed in
frame $B$ into frame $A$" (so $p_A = T_{A\_B}\, p_B$); this matches the
FAST-LIO `offset_R_L_I`/`offset_T_L_I` convention. **Every `Pose`-typed field
below names its two frames in a comment.** A pose with unnamed frames is a bug.

### 2.3 The "estimation frame"

The front-end estimates the trajectory of one canonical body frame, the
**estimation frame** $F_e$ (normally `imu_link`, since IMU
propagation/preintegration is cheapest there). Sensor measurements are brought
into $F_e$ via extrinsics (§5). **The `KeyframePacket` pose is always
$T_{\text{ref}\_F_e}$** for a named reference frame `ref` (§6), so L3 never has
to guess which physical frame a keyframe pose describes.

### 2.4 Ownership & threading vocabulary

We use four ownership categories, stated for every type:

* **Value** — cheap, copied freely, no heap aliasing concerns (`Pose`,
  `NavState`, `ImuSample`). Lives on the stack or inline in containers.
* **Owned-unique** — heap buffer with a single owner (`std::unique_ptr`);
  ownership *moves* across a boundary (e.g. a freshly preprocessed scan handed
  L1→L2).
* **Shared-immutable** — heap buffer behind `std::shared_ptr<const T>`; multiple
  readers, **no writer after publication**. This is how clouds/images in a
  `KeyframePacket` are passed (§6) so L4/L5/meshing share bytes (MUST-FIX #4,
  R3).
* **Borrowed** — a `const T&` valid only for the duration of the call; the
  callee must copy or `shared_ptr`-retain anything it needs to keep.

Threading: Meridian runs (at least) a **front-end thread** (L0–L2, real-time,
soft-deadline per scan), a **back-end thread** (L3 + L5, best-effort,
iSAM2/loop closure), and a **map/mesh thread** (L4 integration + Marching
Cubes). Boundaries that cross threads are flagged **[TS]** (thread-crossing) and
*must* hand over Shared-immutable or moved Owned-unique data through a queue;
they may not pass Borrowed references.

---

## 3. Core math types

These are the atoms. They are header-only, depend only on Eigen + Sophus, and
have **no ROS and no PCL dependency** — this is what makes the core library
testable in isolation (architecture principle 1).

### 3.1 `Pose` — a rigid transform $T \in SE(3)$

```cpp
namespace meridian {

using Timestamp = std::int64_t;   // nanoseconds since epoch
using Duration  = std::int64_t;   // nanoseconds

// A rigid-body transform T_target_source in SE(3).
// Semantics: p_target = R * p_source + t.
// Stored as a unit quaternion + translation (NOT a 4x4 matrix and NOT Euler):
// the quaternion avoids gimbal lock and is the minimal drift-free rotation store.
struct Pose {
  Eigen::Quaterniond q  = Eigen::Quaterniond::Identity();  // rotation R, normalized
  Eigen::Vector3d    t  = Eigen::Vector3d::Zero();         // translation p [m]

  // Group ops
  Pose   operator*(const Pose& rhs) const;   // composition  T_a_b * T_b_c = T_a_c
  Pose   inverse() const;                    // T_target_source -> T_source_target
  Eigen::Vector3d operator*(const Eigen::Vector3d& p) const;  // apply to a point

  // Manifold ops on the 6-DoF tangent xi = [rho (trans); phi (rot)] in R^6.
  // Right convention (perturbation in the SOURCE/body frame):
  //   pose ⊞ xi  ==  pose * Exp(xi)
  //   a    ⊟ b   ==  Log(b.inverse() * a)            // result in R^6
  Pose            boxplus (const Eigen::Matrix<double,6,1>& xi) const;   // ⊞
  Eigen::Matrix<double,6,1> boxminus(const Pose& rhs) const;            // ⊟

  Eigen::Matrix3d R() const { return q.toRotationMatrix(); }
  Eigen::Matrix4d matrix() const;            // 4x4 homogeneous (for export only)
};
```

**Ownership/lifetime:** Value. Trivially copyable. Lives wherever its holder
lives.

**Field semantics.** `q` is a *unit* quaternion; all mutating ops renormalize
(a stored rotation matrix can silently drift off $SO(3)$ under repeated
multiplication; a quaternion is one normalization away from valid). The 6-DoF
tangent orders **translation first, rotation second**
($\xi = [\rho;\ \phi]$); every Jacobian in later specs uses this order. The
right ($\boxplus$ on the right) convention means perturbations live in the body
frame, which is what point-to-plane LiDAR Jacobians want (the plane residual
differentiates w.r.t. a body-frame twist).

> **Pitfall fixed here:** mixing left- and right-perturbation conventions
> between front-end and back-end is a classic source of sign-flipped Jacobians.
> Meridian mandates the *right* convention everywhere; if a back-end library (GTSAM)
> internally uses a different one, the adapter at the L3 boundary converts —
> never the math in the core.

### 3.2 `Twist` and `NavState` — the kinematic state $x$

```cpp
// Body-frame spatial velocity (optional companion to a Pose for CT queries).
struct Twist {
  Eigen::Vector3d v_lin = Eigen::Vector3d::Zero();   // linear  velocity [m/s], in frame named by holder
  Eigen::Vector3d v_ang = Eigen::Vector3d::Zero();   // angular velocity [rad/s]
};

// The full navigation state the front-end estimates, expressed as plain data.
struct NavState {
  Timestamp       stamp = 0;                          // valid time of this state [ns]
  Pose            T_world_body;                       // T_<ref>_<estimation-frame>; ref named by Frame field
  Eigen::Vector3d v_world = Eigen::Vector3d::Zero();  // velocity in ref frame [m/s]
  Eigen::Vector3d b_g     = Eigen::Vector3d::Zero();  // gyro  bias [rad/s]
  Eigen::Vector3d b_a     = Eigen::Vector3d::Zero();  // accel bias [m/s^2]
  Eigen::Vector3d g_world = {0, 0, -9.81};            // gravity in ref frame [m/s^2]

  Frame ref_frame = Frame::Odom;                      // which frame T_world_body / v_world / g_world live in

  // Tangent-space dimension and box ops on the 18-DoF error state
  // [ p(3) | R(3) | v(3) | b_g(3) | b_a(3) | g(3) ].  (g may be handled as
  // 2-DoF S2 internally; exposed here as 3-DoF for a uniform interface.)
  // This boxplus is the BOUNDARY chart: the pose part uses the coupled SE(3)
  // exponential (Pose::boxplus, §3.1). An impl MAY use a different INTERNAL
  // error chart for its covariance/Jacobians (e.g. decoupled world-additive
  // position + SO(3)-exp rotation, which keeps analytic H rows exact); that
  // internal chart never crosses the boundary — states (de)serialise through
  // this coupled boxplus.
  static constexpr int kDof = 18;
  NavState boxplus(const Eigen::Matrix<double, kDof, 1>& dx) const;   // ⊞
  Eigen::Matrix<double, kDof, 1> boxminus(const NavState& rhs) const; // ⊟
};
```

**Ownership/lifetime:** Value (≈200 bytes). Copied freely.

**Field semantics & design choices.**

* `T_world_body`, `v_world`, `g_world` all live in `ref_frame`. For a live
  front-end estimate `ref_frame == Odom`; after the back-end relinearizes,
  corrected states are expressed in `Map`. **Bundling the frame tag with the
  state prevents the single most common multi-sensor SLAM bug:** silently mixing
  odom-frame and map-frame quantities.

* We **do not** put angular/linear *acceleration* in `NavState`. A
  Point-LIO-style front-end models the IMU as an *output* and carries `omg/acc`
  in its state — an *internal* modelling choice it keeps private. The Meridian
  `NavState` is the *interface* state: the common denominator any front-end
  (iEKF, Point-LIO-style, CT) can fill. This is R1 applied to the state vector.

* `kDof = 18`. This is the *navigation* error state only; **extrinsics are not
  in `NavState`** even though FAST-LIO/Point-LIO carry the LiDAR→IMU offset in
  their filter states. In Meridian, online-refined extrinsics are *calibration
  variables* (§5.3) that the front-end may optimise internally and the back-end
  may hold as graph variables; they have their own type and their own boundary
  so we can turn online refinement on/off without changing the state dimension.

### 3.3 Covariance & information blocks

```cpp
// Square symmetric PSD matrix tagged with what its rows/cols mean.
template <int N>
struct GaussianBlock {
  enum class Form { Covariance, Information } form = Form::Covariance;  // Σ or Ω=Σ^{-1}
  Eigen::Matrix<double, N, N> M = Eigen::Matrix<double, N, N>::Zero();  // the matrix
  // Ordering of the N tangent dims is fixed by the type that owns this block
  // (e.g. Pose order [rho; phi], NavState order [p|R|v|bg|ba|g]).
  // EXCEPTION: KeyframePacket.constraint_cov (§6.1) is ordered rotation-first
  // [rx,ry,rz,tx,ty,tz] to match the GTSAM Pose3 boundary — see §6.1.
};

using PoseCov6   = GaussianBlock<6>;    // for a relative/absolute Pose
using NavCov18   = GaussianBlock<18>;   // for a full NavState
```

**Ownership/lifetime:** Value. **Why carry the `Form` tag:** a back-end (GTSAM)
wants *noise models* it can invert; a filter front-end naturally produces a
*covariance* (the marginal $\Sigma$ from the EKF) but an optimiser produces an
*information* matrix (the Hessian $\Omega = H^\top \Sigma_z^{-1} H$). Tagging the
form makes the L2→L3 adapter's job explicit and auditable, which directly
supports the MUST-FIX #3 information-accounting argument (§6.4): we can see, at
the boundary, whether we are shipping $\Sigma$ or $\Omega$ and ensure we ship it
*once*.

### 3.4 `ObservabilityReport` — per-axis degeneracy

```cpp
// Per-axis observability/degeneracy of the front-end's most recent solve,
// expressed in a NAMED reference frame so the back-end can rotate it correctly.
// This is the X-ICP / D2-LIO-style signal that flows into back-end noise inflation.
struct ObservabilityReport {
  Frame           frame = Frame::Body;          // frame in which the 6 axes are expressed
  // Six scores, one per DoF, order [tx, ty, tz, rx, ry, rz] in `frame`.
  // Score in [0,1]: 1 = fully observable/well-conditioned, 0 = fully degenerate.
  // Defined as a normalized conditioning measure of the registration Hessian
  // (e.g. per-direction eigenvalue ratio of H^T H, or the X-ICP localizability
  // metric). Exact definition is the front-end's; the CONTRACT is the [0,1]
  // semantics and the named frame.
  std::array<double, 6> score = {1,1,1,1,1,1};
  // Optional: the eigenvectors (columns) if the degenerate directions are not
  // axis-aligned. If present, `score` are the eigenvalues' normalized conditioning
  // along these directions instead of along frame axes.
  std::optional<Eigen::Matrix<double,6,6>> eigvecs;
};
```

**Ownership/lifetime:** Value. **Why named-frame + optional eigenvectors:**
degeneracy in a long corridor is along the corridor *axis*, not along
`base_link`'s X unless the robot happens to be aligned. Passing scores without a
frame (a flaw in many systems) makes them un-rotatable and therefore unusable by
the back-end. The eigenvector escape hatch handles non-axis-aligned degeneracy
(e.g. a tunnel at 30°). This report rides inside the `KeyframePacket` (§6) and
later spec 04 (robustness) defines exactly how `score` maps to noise inflation.

---

## 4. Sensor sample types (the L0→L1 currency)

These are produced by `ISensorSource` (§7.1) implementations in L0 and consumed
by L1 preprocessing. They are deliberately **raw**: no deskew, no feature
extraction, no frame transforms. They carry only what the hardware produced plus
the synchronised timestamp.

### 4.1 `ImuSample`

```cpp
struct ImuSample {
  Timestamp       stamp = 0;                 // PTP-disciplined valid time [ns]
  Eigen::Vector3d acc   = Eigen::Vector3d::Zero();   // specific force [m/s^2], imu_link, raw (bias included)
  Eigen::Vector3d gyro  = Eigen::Vector3d::Zero();   // angular rate  [rad/s], imu_link, raw (bias included)
  std::uint8_t    sensor_id = 0;             // which IMU (multi-IMU rigs)
  // NO orientation field: Meridian never trusts a vendor AHRS; orientation is estimated.
};
```

**Ownership/lifetime:** Value (≈40 bytes). Streamed at 100–1000 Hz on the
front-end thread. **Why no quaternion:** vendor on-board fusion is opaque and
uncalibrated for our rig; the estimator must own orientation. `acc` includes
gravity and bias (it is the *raw* specific force); de-biasing happens in L2.

### 4.2 `LidarScan` and `LidarPoint` — per-point time is mandatory

```cpp
// One LiDAR return (Ouster-style layout).
struct LidarPoint {
  Eigen::Vector3f xyz;                 // [m] in the LiDAR sensor frame os_sensorN, raw (NOT deskewed)
  float           intensity = 0.f;     // calibrated intensity
  std::int32_t    t_offset_ns = 0;     // time of THIS return MINUS scan.stamp_start [ns]
  std::uint16_t   ring = 0;            // laser/row id (beam), for ring-aware ops
  std::uint16_t   ambient = 0;         // ambient/NIR (Ouster), 0 if N/A
  float           range = 0.f;         // [m] precomputed range, 0 if invalid
};

struct LidarScan {
  Timestamp     stamp_start = 0;        // valid time of the FIRST point [ns]
  Duration      sweep_duration = 0;     // span first..last point [ns] (~100 ms @ 10 Hz)
  std::uint8_t  sensor_id = 0;          // single LiDAR → always 0 (field kept for a future multi-LiDAR extension)
  Frame         sensor_frame = Frame::Unknown;  // os_sensor0 for extrinsic lookup
  std::shared_ptr<const std::vector<LidarPoint>> points;   // Shared-immutable [TS]
};
```

**Ownership/lifetime:** the `LidarScan` struct itself is a Value, but `points`
is **Shared-immutable** [TS]: L0 fills the vector, then publishes it behind
`shared_ptr<const ...>` and never mutates it again. This lets L1's preprocessed
output and L4's retained keyframe store reference the *same bytes* when no
filtering is needed (a copy is made only when L1 actually filters).

**Why per-point time is non-negotiable.** Deskew (motion compensation) requires
knowing *when each point was sampled* so it can be transformed by the trajectory
at that instant. A scan **without** per-point time cannot be deskewed correctly
and is rejected at L1. `t_offset_ns` is a signed 32-bit offset from
`stamp_start` (±2.1 s range) rather than an absolute 64-bit time per point,
saving 4 bytes/point on multi-million-point clouds while keeping nanosecond
resolution.

> **MUST-FIX #2 hook.** Deskew happens **inside L2**, never in L0/L1: the
> front-end warps each point to the sweep-end instant with a constant-screw
> model built from the group's own IMU samples (spec 04), so no layer below L2
> ever needs a trajectory and no feedback edge exists. The `LidarScan` type
> therefore stays raw end-to-end — per-point time is the only deskew input it
> must carry.

### 4.3 `CameraFrame`

```cpp
// One image. Meridian uses FAST-LIVO2-style sparse-direct photometric residuals,
// so we keep the raw image (a pyramid is built in L1, not stored here) plus
// exposure metadata for photometric consistency.
struct CameraFrame {
  Timestamp     stamp = 0;              // mid-exposure time [ns] (global shutter: well-defined)
  std::uint8_t  sensor_id = 0;
  Frame         sensor_frame = Frame::CamLink;
  int           width = 0, height = 0;
  enum class Encoding { Mono8, Bayer_RGGB8, RGB8 } encoding = Encoding::Mono8;
  std::shared_ptr<const std::vector<std::uint8_t>> data;   // Shared-immutable [TS], row-major
  float         exposure_s = 0.f;      // exposure time [s], for photometric normalization (0 = unknown)
  float         gain = 1.f;            // analog/digital gain, for photometric normalization
};
```

**Ownership/lifetime:** Value struct; `data` is Shared-immutable [TS]. **Why
exposure/gain:** FAST-LIVO2's direct photometric residual $r = I_1(\pi(\cdot)) -
I_2(\pi(\cdot))$ assumes brightness constancy; auto-exposure breaks it. Carrying
`exposure_s`/`gain` lets L2 normalise intensities (or estimate an affine
brightness transform) — the data the photometric front-end needs, kept at the
boundary so L0 stays dumb. We store the **mid-exposure** timestamp because that
is the instant a camera pose should be queried at (the IMU-propagated pose of
whatever estimator consumes the frame).

### 4.4 `GnssFix`

```cpp
struct GnssFix {
  Timestamp       stamp = 0;             // PPS-disciplined time [ns]
  std::uint8_t    sensor_id = 0;
  Frame           sensor_frame = Frame::GnssLink;
  double          lat_deg = 0, lon_deg = 0;   // WGS84
  double          alt_m   = 0;                // ellipsoidal height [m]
  Eigen::Matrix3d cov_enu = Eigen::Matrix3d::Identity();  // position covariance in local ENU [m^2]
  enum class FixType { None, SPP, DGPS, RTK_Float, RTK_Fixed } fix = FixType::None;
  std::uint8_t    num_sats = 0;
};
```

**Ownership/lifetime:** Value. **Why lat/lon/alt + ENU covariance rather than a
metric position:** the local datum (ENU origin) is a *system* decision the
back-end owns; L0 reports the raw geodetic fix and its covariance, and the L3
adapter projects into the `map` datum. `fix` and `cov_enu` together let the
back-end gate weak fixes — an `SPP` fix with metres of covariance should weigh
far less than an `RTK_Fixed` centimetre fix, and GNC robust kernels (spec 04)
key off exactly this.

---

## 5. Calibration & extrinsic types

Cross-cutting concern: every sensor measurement must be expressed in the
estimation frame, which requires extrinsics; and Meridian refines extrinsics
**online** (architecture: "offline prior + online extrinsic refinement as graph
variables").

### 5.1 `IntrinsicsCamera`

```cpp
struct IntrinsicsCamera {
  double fx, fy, cx, cy;                       // pinhole K_cam
  enum class Distortion { None, RadTan, Equidistant } model = Distortion::RadTan;
  std::array<double, 5> coeffs = {0,0,0,0,0};  // k1,k2,p1,p2,k3 (RadTan) or k1..k4 (Equi)
  int width = 0, height = 0;

  // Photometric intrinsic for direct visual (FAST-LIVO2); full semantics in spec 08 §3.2.
  // Auto-exposure breaks brightness constancy, so the inverse exposure time is
  // carried as an intrinsic with a prior and refined online per frame.
  double inv_expo_prior = 1.0;                 // 1/exposure scale prior (1.0 = none)
  double inv_expo_std   = 0.0;                 // prior 1σ; 0 ⇒ fixed
  bool   refine_photometric_online = true;     // FAST-LIVO2 path: on
};
```

### 5.2 `Extrinsic` — a calibrated transform with uncertainty and a refinement flag

```cpp
// Where an extrinsic prior came from (provenance). Detailed in spec 08 §3.1.
enum class CalibSource : std::uint8_t {
  Unknown = 0, Vendor, Kalibr, TargetBoard, HandEye, OnlineRefined, Manual
};

// A sensor-to-estimation-frame transform with its prior uncertainty and whether
// the estimator is allowed to refine it online.
struct Extrinsic {
  Frame   child  = Frame::Unknown;      // e.g. os_sensor0
  Frame   parent = Frame::ImuLink;      // the estimation frame F_e
  Pose    T_parent_child;               // T_<F_e>_<sensor>, the offline prior mean
  PoseCov6 prior_cov;                   // 6-DoF prior uncertainty (Σ); tight if factory-calibrated
  bool    refine_online = false;        // if true, becomes a calibration variable (§5.3)
  Timestamp calibrated_at = 0;          // provenance: when this prior was established

  // Calibration metadata + refinement gates (full semantics in spec 08 §3.1).
  CalibSource   source  = CalibSource::Unknown;  // where the prior came from
  std::uint32_t version = 0;            // bumps on each online refinement (spec 08 §9)
  double max_drift_trans_m = 0.10;      // hard gate: reject refinement leaving this ‖Δp‖ of prior
  double max_drift_rot_deg = 2.0;       // hard gate: ‖Δφ‖ cap from prior mean
  // Per-sensor temporal offset (the time twin of the geometric extrinsic, spec 08 §7).
  double time_offset_ns     = 0.0;      // td: t_sensor = t_Fe + td  (prior mean)
  double time_offset_std_ns = 0.0;      // prior 1σ; 0 ⇒ effectively fixed
  bool   refine_time_online = false;    // refine td as a graph variable?
};
```

**Field semantics.** `T_parent_child` is the offline prior (the
factory/target-board calibration), `prior_cov` is how much we trust it, and
`refine_online` selects whether the estimator may move it. A factory-rigid
LiDAR→IMU mount could set `refine_online = false` with tight `prior_cov`; Meridian
instead enables online extrinsic refinement by default (`refine_online = true`,
looser prior) so the graph absorbs thermal/vibration drift in the field. This is
the type-level switch the architecture's online-calibration and "switchable
constraints" robustness goals need.

### 5.3 `CalibrationSet` and online refinement

```cpp
// The complete, queryable calibration of the rig. Front-end and back-end both
// read it; the back-end may WRITE refined values back (online refinement).
struct CalibrationSet {
  Frame estimation_frame = Frame::ImuLink;                 // F_e
  std::vector<Extrinsic>           extrinsics;             // all sensor->F_e transforms
  std::unordered_map<std::uint8_t, IntrinsicsCamera> cam_intrinsics;  // by camera sensor_id
  // IMU noise parameters (Allan-variance derived), feed the preintegration noise:
  double imu_acc_noise, imu_gyr_noise;                     // continuous-time noise density (std, not variance)
  // The configured sensors.imu.cov_acc/cov_gyr are the SQUARED densities (variance
  // convention); calibration_from_config takes sqrt to fill these std-convention fields.
  double imu_acc_bias_rw, imu_gyr_bias_rw;                 // bias random-walk

  std::uint32_t version = 0;          // snapshot id; any refinement bumps it (§5.3 prose)

  const Extrinsic& extrinsic(Frame sensor) const;          // lookup, throws if absent
};
```

**Ownership/lifetime & threading.** The `CalibrationSet` is owned by the system
bootstrap and shared **read-only** with the front-end (Shared-immutable). When
`refine_online` extrinsics are enabled, the **back-end** owns the authoritative
refined values: it holds them as graph variables (per the architecture), and
publishes updated snapshots that the front-end picks up at keyframe boundaries.
**The refined extrinsic crosses L3→L2 as a versioned `CalibrationSet` snapshot,
never as a live shared mutable** — this avoids a data race between the back-end's
optimiser and the front-end's per-scan reads. The version counter lets the
front-end detect "calibration changed, reset linearization."

> **Design note — why extrinsics are *not* in `NavState`.** Putting all
> extrinsics in the front-end state (as FAST-LIO/Point-LIO do for their single
> LiDAR) would bloat the state and hide them from the back-end's global
> optimisation. As a separate `CalibrationSet` with a `refine_online` flag, a
> front-end *may* still optimise the ones it cares about internally, while the
> *contract* keeps them out of `NavState` so the dimension is fixed and the
> back-end remains the authority (R1 again).

---

## 6. The KeyframePacket — MUST-FIX #1

This is the heart of the spec. **The `KeyframePacket` is the one and only thing
L2 (front-end) hands to L3 (back-end).** Defining it concretely is MUST-FIX #1;
defining *what is and is not in it* is MUST-FIX #3 (no double-counting).

### 6.1 Declaration

```cpp
// The sole L2 -> L3 currency. Emitted by IFrontEnd when it decides a new
// keyframe is warranted. Self-contained: the back-end needs nothing else from
// the front-end to add this keyframe to the graph.
struct KeyframePacket {
  // --- Identity & time ---
  std::uint64_t  id = 0;                 // monotonically increasing keyframe id
  Timestamp      stamp = 0;              // keyframe valid time [ns] (a real measurement instant)

  // --- (1) Pose: the keyframe's estimated pose ---
  Frame          ref_frame = Frame::Odom;   // frame the pose is expressed in (Odom for a live front-end)
  Pose           T_ref_body;                // T_<ref_frame>_<F_e>: pose of the estimation frame

  // --- (2) Optional kinematics: velocity + biases, with an "included" flag ---
  bool           kinematics_included = false;   // does this packet carry v + biases?
  Eigen::Vector3d v_ref   = Eigen::Vector3d::Zero();   // velocity in ref_frame [m/s]   (valid iff included)
  Eigen::Vector3d b_g     = Eigen::Vector3d::Zero();   // gyro bias  [rad/s]            (valid iff included)
  Eigen::Vector3d b_a     = Eigen::Vector3d::Zero();   // accel bias [m/s^2]            (valid iff included)

  // --- (3) Uncertainty: ONE block, tagged covariance-or-information ---
  // Constraint kind tells the back-end how to USE this block (see §6.3, §6.4):
  enum class ConstraintKind {
    RelativeBetween,    // block is the 6-DoF marginal cov of T_prev_this, or a documented
                        // conservative upper bound of it (§6.4) — the DEFAULT clean contract
    AbsolutePrior,      // block is the 6-DoF marginal cov of T_ref_body in ref_frame (first KF / GNSS-anchored)
    ImuPreintegration   // RESTART-FALLBACK ONLY: block accompanies a raw IMU summary (§6.5), mutually exclusive
  } constraint_kind = ConstraintKind::RelativeBetween;
  std::uint64_t  rel_to_id = 0;          // for RelativeBetween: the id this pose is relative TO
  Pose           T_relto_this;           // for RelativeBetween: the relative transform itself
  GaussianBlock<6> constraint_cov;       // 6-DoF block; meaning set by constraint_kind. Form per kind (Σ default).
                                         // ORDERED ROTATION-FIRST [rx,ry,rz,tx,ty,tz] to match the GTSAM Pose3
                                         // boundary — the ONE exception to the translation-first core convention,
                                         // so it is a bare GaussianBlock<6>, NOT a PoseCov6. The front-end's
                                         // translation-first marginal is reordered exactly once when packing.

  // --- (4) Observability (per-axis degeneracy), for back-end noise inflation ---
  ObservabilityReport observability;     // 6 scores in a NAMED frame (§3.4)

  // --- (5) Handle to the keyframe's point cloud (retained store; MUST-FIX #4) ---
  // Deskewed, in the F_e/body frame at `stamp`. Shared-immutable so L4 (re-integration),
  // L5 (GICP), and final Poisson meshing share the SAME bytes (R3).
  std::shared_ptr<const std::vector<LidarPoint>> cloud_body;   // [TS]

  // --- (6) Camera / RGB handle (for colourisation + visual loop cues) ---
  std::shared_ptr<const CameraFrame> image;                    // [TS], may be null if no cam at this KF
  Pose           T_body_cam;             // extrinsic snapshot used at this KF (so L4 can colourise without a lookup)

  // --- (7) Restart-fallback IMU summary (ONLY when constraint_kind==ImuPreintegration) ---
  std::optional<ImuPreintegrationSummary> imu_summary;         // §6.5; null otherwise

  // --- Provenance ---
  std::uint32_t  calib_version = 0;      // which CalibrationSet snapshot produced this (extrinsic provenance)
  std::uint32_t  frontend_kind = 1;      // 1 = retired CT front-end, 2 = LIO (the only live producer);
                                         // diagnostics only — the back-end must not branch on it
};
```

### 6.2 Field-by-field semantics

* **id / stamp.** `id` is the graph node key. `stamp` is a *real measurement
  instant* (a scan-end or a chosen anchor time), so the back-end can attach
  GNSS/loop factors that are time-correlated.

* **(1) T_ref_body.** Always $T_{\text{ref}\_F_e}$ — the pose of the *estimation
  frame* in the named `ref_frame`. By fixing this, any estimator — a discrete
  LIO solving one pose per sweep, an iEKF whose native state is the IMU pose, a
  continuous-time trajectory sampled at `stamp` — produces *the same typed
  thing*. The back-end never has to know which.

* **(2) kinematics_included.** This is the explicit flag MUST-FIX #1 demands.
  Velocity and biases are *optional* across the boundary: whether they cross as
  live variables is a contract decision (§6.4). When `false`, the back-end does
  **not** create velocity/bias nodes for this keyframe — it treats the keyframe
  as a pure pose node. When `true` (used only with the restart-fallback IMU
  factor, §6.5), the back-end *may* create velocity/bias variables linked by an
  IMU factor.

* **(3) constraint_kind + constraint_cov.** This is the MUST-FIX #3 lever
  (§6.4). It tells the back-end **what factor to build** and **what the cov
  block means**, guaranteeing the information is added exactly once.

* **(4) observability.** Flows into per-axis noise inflation (spec 04). Carried
  here, in a named frame, so the back-end can rotate the degenerate directions
  into the graph frame before inflating.

* **(5) cloud_body.** The retained per-keyframe cloud (MUST-FIX #4). It is
  **deskewed** and in the body frame at `stamp` (so re-integrating it at a
  corrected pose is just one `Pose` multiply). Shared-immutable: the L4 map store
  keeps this `shared_ptr`, L5's loop detector runs GICP on it, and final Poisson
  meshing reads it — all without copying.

* **(6) image + T_body_cam.** The RGB used to colourise the surface at this
  keyframe, plus the *exact* body→cam extrinsic snapshot so L4 colourisation is
  reproducible even after online extrinsic refinement changes the live value.

* **(7) imu_summary.** Present *only* in the restart fallback (§6.5); `nullopt`
  in the normal path. Its presence is mutually exclusive with the normal
  `RelativeBetween` factor (§6.4).

* **calib_version / frontend_kind.** Provenance. `frontend_kind` is for
  diagnostics/logging only; **the back-end must not branch on it** — that would
  break the abstraction (R1). It exists so a developer reading a rosbag can see
  which estimator produced a node.

### 6.3 Ownership, threading, lifetime

* The `KeyframePacket` struct is **moved** L2→L3 through a thread-safe queue
  **[TS]** (front-end thread → back-end thread). After the move, the front-end
  no longer owns it.
* `cloud_body` and `image` are **Shared-immutable**: the front-end created them
  during deskew/preprocessing and will not touch them again; the back-end
  forwards the same `shared_ptr`s to L4/L5. The cloud therefore lives as long as
  *any* of {graph node, map store, loop detector} retains it — reference counting
  gives us the lifetime MUST-FIX #4 needs for free.
* Everything else in the packet is plain Value data copied with the struct.

### 6.4 The clean contract — no double-counting (MUST-FIX #3)

Shipping multiple factors built from the same measurements for the same pair of
keyframes (an absolute prior *plus* a relative `BetweenFactor` *plus* an IMU
factor) counts the same information more than once, making the graph
over-confident and inconsistent. Meridian's contract, encoded by
`constraint_kind`, **picks exactly one**:

* **Normal path — `RelativeBetween` (default).** The front-end has already fused
  IMU + LiDAR + visual internally and *summarised* the result as a single
  relative transform $T_{\text{relto}\_this}$ between consecutive keyframes with
  its marginal covariance `constraint_cov` ($\Sigma$). The back-end adds **one**
  `BetweenFactor(rel_to_id → id)` with that noise model and **nothing else**. No
  IMU factor, no separate absolute prior. The IMU measurements have already done
  their work *inside* the front-end and their information is *in* the relative
  covariance. This is the recommended contract and the default.

  A front-end **may ship a conservative upper bound** of this marginal instead of
  the exact value — e.g. the sum of the two endpoint pose marginals with their
  (positive) cross-covariance dropped, the current CT implementation — when the
  exact joint extraction is not affordable per keyframe. The bound's direction is
  safe (odometry is never over-trusted; the no-double-counting goal is preserved),
  but it proportionally loosens every covariance-calibrated gate downstream (PCM,
  GNSS skip-if-confident — spec 05 §4.2/§7.1), so the exact marginal is preferred
  when available, and the choice must be documented at the packing site.

* **Anchor — `AbsolutePrior`.** Used for the **first** keyframe (to fix the
  gauge), when an external absolute reference applies (a fresh GNSS-anchored
  pose), and as a **chain break** when the front-end cannot vouch for a relative
  edge to the previous keyframe (e.g. the first keyframe after a hard reseed,
  where the pose was carried across a data hole on a prediction whose
  error no covariance accounts for). The block is the marginal covariance of
  $T_{\text{ref}\_body}$ itself. This does not double-count because there is no
  companion relative factor for an anchor; the very next keyframe goes back to
  `RelativeBetween`. A chain-break anchor in a *relative* `ref_frame`
  (`Frame::Odom`) carries no map-frame information — the consumer treats it like
  the first-keyframe gauge anchor for the segment it opens, not as an absolute
  measurement.

* **Restart fallback — `ImuPreintegration` (mutually exclusive, reserved).**
  For a producer that can vouch for the IMU across a tracking gap but not for a
  fused relative pose, the packet ships the raw `imu_summary` (§6.5) and
  `kinematics_included = true`, and the back-end builds a single GTSAM
  **`CombinedImuFactor`** (bias random-walk folded in — see spec 05 §5.1) across
  the gap *instead of* the `RelativeBetween` factor. The current LIO front-end
  **never emits this kind** — its restart path is the `AbsolutePrior` chain
  break above — but the contract keeps it for future producers. Because
  `constraint_kind` is a single enum, the kinds are **mutually exclusive by
  construction** — you cannot ship both.

In all three cases, **exactly one constraint** carries the L2 information for a
given keyframe edge. The `Form`-tagged `GaussianBlock` (§3.3) makes the adapter
that converts $\Sigma$→noise-model explicit and auditable. This is the concrete
resolution of MUST-FIX #3.

> **Does velocity/bias cross as a live variable?** Decision: **No, in the normal
> path.** `kinematics_included = false` normally; velocity and biases stay
> *inside* the front-end and their effect is folded into the relative
> covariance. They cross as live variables **only** in the restart fallback
> (`ImuPreintegration`), where the IMU factor genuinely needs velocity/bias
> nodes on both endpoints. This keeps the back-end graph minimal (pose-only
> nodes) in the common case and avoids the back-end having to maintain
> bias/velocity dynamics that the front-end already models better.

### 6.5 `ImuPreintegrationSummary` (restart fallback only)

```cpp
// A self-contained IMU preintegration between two keyframe stamps, used ONLY on
// the restart fallback so the back-end can bridge a gap the front-end
// could not summarise as a relative transform. Mirrors the classic
// (ΔR, Δv, Δp) preintegrated measurements + their covariance + bias Jacobians.
struct ImuPreintegrationSummary {
  Timestamp        t_i = 0, t_j = 0;     // integration interval [ns]
  Eigen::Quaterniond delta_R;            // ΔR_ij  (rotation increment)
  Eigen::Vector3d  delta_v;              // Δv_ij  [m/s]
  Eigen::Vector3d  delta_p;              // Δp_ij  [m]
  Eigen::Vector3d  bias_g_lin, bias_a_lin;   // bias linearization point
  Eigen::Matrix<double,3,3> dR_dbg, dv_dbg, dv_dba, dp_dbg, dp_dba;  // first-order bias Jacobians
  GaussianBlock<9> preint_cov;           // 9-DoF cov of [ΔR|Δv|Δp]
  double           gravity_mag = 9.81;   // gravity magnitude used
};
```

**Ownership/lifetime:** Value, carried by `std::optional` in the packet; present
only when `constraint_kind == ImuPreintegration`. This is the standard
on-manifold preintegration the back-end (GTSAM `PreintegratedImuMeasurements`)
can consume directly.

---

## 7. Abstract interfaces

Every layer boundary is a pure abstract base class; the interface header lives
in its owning layer's package, and the value types it exchanges live in
`meridian_common` (spec 00 §5). Implementations are selected at construction
(dependency injection); `meridian_pipeline` wires them together. Below, each
interface states **the one boundary value** it deals in, plus
ownership/threading.

### 7.1 `ISensorSource` (L0)

```cpp
// One per physical sensor stream. Produces raw, time-synced samples.
// Implementations: OusterLidarSource, ImuSource, CameraSource, GnssSource;
// offline bag replay drives the same sources through the ROS wrapper
// (see docs/DATASET.md).
template <class SampleT>   // SampleT in {ImuSample, LidarScan, CameraFrame, GnssFix}
class ISensorSource {
public:
  virtual ~ISensorSource() = default;

  // Push model: the source invokes `cb` on its own thread when a sample is ready.
  // The callback receives an OWNED-UNIQUE or VALUE sample (heavy buffers behind
  // shared_ptr<const> as in §4). The source must have applied PTP time sync
  // BEFORE calling back; stamps are final.
  using Callback = std::function<void(SampleT&&)>;
  virtual void set_callback(Callback cb) = 0;

  virtual void start() = 0;     // begin streaming
  virtual void stop()  = 0;     // stop; no callbacks after stop() returns

  virtual SensorInfo info() const = 0;   // static descriptor: id, frame, rates, model
};
```

**One thing it passes:** a single raw `SampleT` per callback. **Threading:**
each source owns its acquisition thread; callbacks fire on that thread, so the
consumer (L1) must be thread-safe or marshal onto the front-end thread.
**Ownership:** the sample is *moved* to the callback; the source keeps nothing.
**Why a template:** the four sample types share no base class (they are POD-ish
value types, R1), so a CRTP/template keeps them concrete and copy-free rather
than forcing a `SensorSampleBase*` hierarchy with virtual dispatch on hot data.

### 7.2 `IPreprocessor` (L1)

```cpp
// Per-sensor preprocessing: range/intensity filtering, downsampling, point-time
// validation, pyramid build for images. NO feature extraction (Meridian is direct).
class ILidarPreprocessor {
public:
  virtual ~ILidarPreprocessor() = default;
  // Input: raw scan (Shared-immutable points). Output: filtered scan, ready for
  // deskew by L2. Deskew is NOT done here (it is L2-internal; MUST-FIX #2).
  virtual LidarScan process(const LidarScan& raw) const = 0;
};
```

**One thing it passes:** a filtered `LidarScan` (still raw-frame, not deskewed).
**Threading:** front-end thread, synchronous. **Ownership:** returns a new
`LidarScan`; reuses `raw.points` (Shared-immutable) when no filtering occurs,
else allocates a new Shared-immutable buffer. **Note:** deskew is deliberately
*not* in L1 — it lives inside the front-end (MUST-FIX #2, §4.2).

### 7.3 `IFrontEnd` (L2) — the swappable estimator

This is the interface that keeps the estimator swappable: Meridian's front-end is
the discrete LIO estimator (§8.2); a FAST-LIO2-style iEKF (§8.1) satisfied the
identical contract as the bring-up differential-test oracle (since removed), as
did the retired continuous-time estimator. All are **drop-in** because
only the boundary types are constrained, never the internal estimator.

```cpp
class IFrontEnd {
public:
  virtual ~IFrontEnd() = default;

  // --- Configuration / calibration ---
  // Called once at start, and again whenever the back-end publishes a refined
  // CalibrationSet snapshot (§5.3). The front-end copies what it needs.
  virtual void set_calibration(std::shared_ptr<const CalibrationSet> calib) = 0;

  // --- Ingest: the front-end consumes a per-sweep PreprocessedGroup ---
  // The PRIMARY input. L1 bundles one LiDAR sweep + the IMU spanning it (+ any
  // image/GNSS in the interval) into a MeasureGroup (§ "MeasureGroup", spec 02 §8);
  // the pipeline wraps it as a PreprocessedGroup (Appendix A) and pushes it once
  // per sweep. This call triggers the per-sweep deskew + registration solve. The
  // scan arrives raw (undeskewed, time-sorted); the front-end deskews it
  // internally from the group's own IMU samples (MUST-FIX #2, §4.2).
  virtual void ingest(const PreprocessedGroup& group) = 0;  // one sweep + spanning IMU/image/GNSS

  // High-rate IMU path (between sweeps), for live-odometry propagation ONLY — it
  // updates live_state() at IMU rate and does NOT trigger a solve. The same
  // samples also arrive (bundled) in the next PreprocessedGroup, which is what the
  // estimator actually optimises against; this path is purely for low-latency output.
  virtual void ingest_imu_live(const ImuSample& imu) = 0;

  // --- Back-end correction feedback (loop closure / relinearization) ---
  // Applied by the pipeline at a safe point between ingests on the front-end
  // thread. The front-end re-anchors its odom frame and shifts its state by the
  // correction; it never restarts the estimator.
  virtual void apply_correction(const GraphUpdate& update) = 0;

  // --- Live output: the IMU-rate propagated odom-frame state ---
  // Advanced by ingest_imu_live between sweeps and rebased onto each solved pose
  // by re-propagating the buffered post-sweep samples, so it tracks the solve
  // without waiting on it. For introspection, control, and L4 live integration.
  // NOT the keyframe handoff.
  virtual NavState live_state() const = 0;

  // --- Keyframe handoff: THE one thing L2 gives L3 (§6) ---
  // The front-end decides internally when a keyframe is warranted and invokes
  // this sink. Exactly one KeyframePacket per keyframe, MOVED across the queue.
  using KeyframeSink = std::function<void(KeyframePacket&&)>;
  virtual void set_keyframe_sink(KeyframeSink sink) = 0;

  // --- Introspection (debug topics / timing) ---
  virtual FrontEndDiagnostics diagnostics() const = 0;   // residuals, eff. points, per-stage timing, observability
};
```

**One thing it passes (downward to L3):** a `KeyframePacket` (§6). **One thing
it exposes (sideways):** the live `NavState`. **Threading:** all `ingest`/
`live_state`/`diagnostics` run on the front-end thread; `set_keyframe_sink`'s
callback fires on the front-end thread but *enqueues* to the back-end thread
**[TS]**. **Ownership:** ingested samples are Borrowed (the front-end copies what
it must); emitted packets are Moved.

**The contract that makes any front-end drop-in.** `IFrontEnd` says nothing
about *how* the trajectory is represented:

* It never exposes the state *representation* — only the typed `NavState`
  (sampled at a time) and the typed `KeyframePacket`. An iEKF's representation is
  the 18-DoF error-state covariance (FAST-LIO `esekfom`); the LIO front-end's is
  a single pose + velocity + biases and a voxel-hash local map. Neither leaks.
* It never exposes *how* measurements are fused — only that you `ingest` them.
  An iEKF runs `esekfom`'s iterated update with the `h_share_model` residual
  functor (esekfom.hpp); the LIO front-end runs a per-sweep Gauss-Newton ICP
  against its local map. Both reduce, at a keyframe instant, to *a pose + a
  relative covariance + an observability report + a deskewed cloud* — which is
  exactly the `KeyframePacket`.
* The keyframe *time* is a real instant; either estimator reports its solved
  state at that instant (the sweep end). Same output type.

### 7.4 `IBackEnd` (L3)

```cpp
class IBackEnd {
public:
  virtual ~IBackEnd() = default;

  // Consume the SOLE input from L2. Builds the factor per constraint_kind (§6.4):
  // RelativeBetween -> BetweenFactor; AbsolutePrior -> PriorFactor;
  // ImuPreintegration -> a single CombinedImuFactor ACROSS THE RESTART GAP, mutually exclusive.
  virtual void add_keyframe(KeyframePacket&& kf) = 0;       // [TS] from front-end queue

  // Loop closures arrive from L5 as relative constraints (§7.6).
  virtual void add_loop_constraint(const LoopConstraint& lc) = 0;

  // GNSS / absolute factors (if not folded into AbsolutePrior packets).
  virtual void add_absolute(const GnssFix& fix, std::uint64_t nearest_kf_id) = 0;

  // Run iSAM2 incremental update; returns which keyframes moved (for L4 re-integration).
  virtual GraphUpdate optimize() = 0;

  // Authoritative corrected trajectory + (optionally) refined calibration snapshot.
  virtual std::vector<StampedPose> corrected_trajectory() const = 0;   // map-frame
  virtual std::shared_ptr<const CalibrationSet> refined_calibration() const = 0;  // §5.3

  virtual BackEndDiagnostics diagnostics() const = 0;
};
```

**One thing it consumes:** the `KeyframePacket`. **One thing it emits to L4:** a
`GraphUpdate` listing *which keyframe poses changed and to what* — exactly the
trigger L4 needs to clear-and-rebuild affected map regions (MUST-FIX #4, §7.5).
**Threading:** back-end thread; `add_keyframe`/`add_loop_constraint` enqueue
**[TS]**, `optimize` runs on the back-end thread. **Ownership:** consumes a Moved
packet (and forwards its Shared-immutable cloud/image handles to L4/L5).

```cpp
struct GraphUpdate {
  // For each moved keyframe: its id, new map-frame pose, and (optionally) new cov.
  struct Moved { std::uint64_t id; Pose new_T_map_body; std::optional<PoseCov6> cov; };
  std::vector<Moved> moved;          // empty if optimize() changed nothing materially
  bool loop_closed = false;          // a loop just snapped (large rigid correction)
};
```

### 7.5 `IMapLayer` (L4)

```cpp
// The layered map. ONE facade over a stack: VoxelHashMap (registration), the
// ISurfaceMap surface tier (TSDF+colour, backend-pluggable: nvblox GPU / cpu host /
// deferred vulkan — spec 06 §0), and the mesh stage that reads it. They all consume
// KeyframePackets' clouds + GraphUpdates.
class IMapLayer {
public:
  virtual ~IMapLayer() = default;

  // Integrate a keyframe's deskewed cloud (+ optional RGB) at its CURRENT pose.
  // Shares the cloud bytes via shared_ptr (no copy). For TSDF this is the
  // running-average fusion, identical across surface backends (spec 06 §4.3).
  virtual void integrate(const KeyframePacket& kf, const Pose& T_map_body) = 0;

  // Loop-closure de-integration (MUST-FIX #4): when GraphUpdate says keyframes
  // moved, CLEAR the affected region's voxels/TSDF and REBUILD from the retained
  // per-keyframe clouds at their corrected poses. We do NOT do per-voxel
  // subtraction (it does not match TSDF running-average semantics); we clear and
  // re-integrate. The retained clouds are the KeyframePacket::cloud_body handles
  // the map kept (Shared-immutable), looked up by id.
  virtual void apply_graph_update(const GraphUpdate& update,
                                  const KeyframeStore& store) = 0;

  // Query for registration (voxel-hash impl): nearest plane to a query point.
  virtual std::optional<PlaneHit> query_plane(const Eigen::Vector3f& p_map) const = 0;

  virtual MapDiagnostics diagnostics() const = 0;
};
```

**One thing it consumes:** `KeyframePacket` clouds (for integration) +
`GraphUpdate` (for de-integration). **The retained store is explicit** — the
`KeyframeStore` below is the data structure MUST-FIX #4 said was missing.

```cpp
// The retained per-keyframe point-cloud store. Consumed by L4 re-integration,
// L5 GICP, and final Poisson meshing. It owns nothing the packet didn't already
// own: it holds the SAME Shared-immutable cloud handles, keyed by id.
class KeyframeStore {
public:
  void put(std::uint64_t id, std::shared_ptr<const std::vector<LidarPoint>> cloud,
           std::shared_ptr<const CameraFrame> image, const Pose& T_init_map_body);
  std::shared_ptr<const std::vector<LidarPoint>> cloud(std::uint64_t id) const;
  std::shared_ptr<const CameraFrame>             image(std::uint64_t id) const;
  // Spatial index over keyframe poses, so apply_graph_update can find the
  // keyframes whose clouds touch a moved region.
  std::vector<std::uint64_t> within_radius(const Eigen::Vector3f& c, float r) const;
};
```

**Ownership/lifetime:** the `KeyframeStore` is the **long-lived owner** of the
keyframe clouds/images (it holds the `shared_ptr`s for the mission's duration, or
until a retention policy evicts old keyframes). This is what gives MUST-FIX #4
the data it needs at correction time: when a loop snaps, `apply_graph_update`
asks the store `within_radius` for the affected keyframes, clears those voxels,
and re-integrates `store.cloud(id)` at `update`'s corrected poses. **Threading:**
map/mesh thread; reads from the store are concurrent-safe (Shared-immutable
clouds).

### 7.6 `ILoopDetector` (L5)

```cpp
// Place recognition: Scan Context++ -> STD/BTC candidates -> GICP verify -> PCM.
// Operates on retained keyframe clouds (from KeyframeStore), NOT live scans.
class ILoopDetector {
public:
  virtual ~ILoopDetector() = default;

  // Offer a new keyframe for descriptor extraction + candidate search.
  virtual void add_keyframe(std::uint64_t id,
                            std::shared_ptr<const std::vector<LidarPoint>> cloud,
                            const Pose& T_map_body) = 0;

  // Returns verified loop constraints (post GICP + PCM). May be empty.
  virtual std::vector<LoopConstraint> detect() = 0;

  virtual LoopDiagnostics diagnostics() const = 0;
};

struct LoopConstraint {
  std::uint64_t from_id = 0, to_id = 0;   // the two keyframes
  Pose          T_from_to;                // GICP-refined relative transform
  PoseCov6      cov;                      // its covariance (post-PCM weighting)
  double        fitness = 0;              // GICP fitness / inlier ratio (for the back-end's GNC)
};
```

**One thing it emits:** a `LoopConstraint` (relative pose + cov + fitness) that
L3 ingests as a `BetweenFactor` under a GNC robust kernel. **Threading:**
back-end thread (best-effort). **Ownership:** reads Shared-immutable clouds from
the store; emits Value constraints.

### 7.7 `IMeshExtractor` (L4 mesh stage / L6 feed)

```cpp
// Marching Cubes over the TSDF -> colourised triangle mesh (the first-pass
// deliverable). ESDF / path planning and semantics are DEFERRED (designed to
// slot onto the same TSDF substrate later, not built now).
class IMeshExtractor {
public:
  virtual ~IMeshExtractor() = default;
  // Extract a colour mesh from the current TSDF+RGB map. Incremental: only
  // re-meshes blocks marked dirty since the last call.
  virtual ColorMesh extract(const IMapLayer& tsdf_map) = 0;
};

struct ColorMesh {
  std::vector<Eigen::Vector3f> vertices;     // map frame [m]
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector3<std::uint8_t>> colors;   // RGB per vertex
  std::vector<std::array<std::uint32_t,3>>  triangles;
  // Optional per-vertex confidence for the operator overlay (L6).
  std::vector<float> confidence;
};
```

**One thing it emits:** a `ColorMesh` — the program's first-pass deliverable and
the L6 operator-interface payload. **Threading:** map/mesh thread.

---

## 8. How two different estimators satisfy `IFrontEnd`

This section is the explicit demonstration MUST-FIX #1 calls for. The
`IFrontEnd` contract (§7.3) constrains only the *boundary types*, not the
internal estimator. Meridian ships the discrete LIO front-end (§8.2); the iEKF
(§8.1) filled the same contract as the bring-up cross-check oracle (since
removed) and is kept here as the proof the boundary is estimator-agnostic.

### 8.1 The iEKF (FAST-LIO2-style) — retired test oracle

**Internal representation.** An 18-DoF error-state on the manifold (FAST-LIO's
`state_ikfom`, `use-ikfom.hpp:12–21`: `pos, rot, vel, bg, ba, grav`, plus its
own copy of the LiDAR extrinsic if `refine_online`), with covariance $P$. The
update is the iterated EKF in `esekfom.hpp`: it repeatedly linearizes the
measurement model, computes the Kalman gain $K = P H^\top (H P H^\top + R)^{-1}$,
and applies $x \leftarrow x \boxplus K r$ until convergence.

**Filling the contract:**

| `IFrontEnd` element | iEKF realisation (all inside the per-sweep `ingest(group)`) |
|---|---|
| group's `imu` set | forward-propagate the state + covariance across the sweep (FAST-LIO `IMU_Processing` prediction); `ingest_imu_live` propagates a copy for low-latency output only |
| group's `scan` | deskew via the propagated trajectory (`UndistortPcl`, IMU_Processing.hpp:27), build point-to-plane residuals $r = n\cdot(R p_L + p) + d$, run the iterated update (esekfom.hpp) |
| group's `image` | FAST-LIVO2-style sparse-direct photometric residual as an extra block in the same iterated update |
| `live_state()` | read the current `state_ikfom` into a `NavState` |
| keyframe `T_ref_body` | the converged filter pose at the keyframe stamp |
| keyframe `constraint_cov` | the marginal covariance over consecutive keyframe poses → relative cov (Schur-marginalise to a 6-DoF `RelativeBetween`) |
| `observability` | per-axis conditioning of $H^\top H$ from the registration block |
| `cloud_body` | the deskewed scan, retained Shared-immutable |
| `kinematics_included` | `false` normally; `true` + `imu_summary` only on a restart |

### 8.2 The discrete LIO front-end — Meridian's front-end

**Internal representation.** A single `NavState` (pose, velocity, biases,
gravity) propagated at IMU rate, plus a voxel-hash local map of past sweeps.
Each `ingest` solves one Gauss-Newton point-to-point ICP for the sweep-end pose,
regularised toward the IMU-propagated prediction (spec 04).

**Filling the contract — the differences are all internal:**

| `IFrontEnd` element | LIO realisation (all inside the per-sweep `ingest(group)`) |
|---|---|
| group's `imu` set | dead-reckoning propagation across the sweep; interval-averaged angular rate + acceleration variance feed the deskew screw and the adaptive gravity regulariser |
| group's `scan` | constant-screw deskew to the sweep end, voxel downsample to keypoints, GN point-to-point ICP against the voxel-hash map |
| group's `image` | retained as the latest frame; rides the next `KeyframePacket` as passthrough (no photometric residual) |
| `live_state()` | the IMU-rate propagated state, rebased onto each solved pose |
| keyframe `T_ref_body` | the converged GN pose at the sweep-end stamp |
| keyframe `constraint_cov` | per-scan Laplace covariance of the GN solve, transported and composed over the interval since the previous keyframe |
| `observability` | per-axis conditioning of the GN data Hessian in the body frame |
| `cloud_body` | the deskewed sweep, body frame at `stamp`, retained Shared-immutable |
| `kinematics_included` | `false` always (same contract) |

**The punchline.** The only thing the back-end ever sees is a `KeyframePacket`
with a `RelativeBetween` constraint (or, rarely, an `AbsolutePrior` re-anchor).
Whether the pose came from a discrete GN solve or from iterating a Kalman gain
is invisible to L3–L6 — which is what lets front-ends be replaced with **zero**
changes downstream.

---

## 9. Boundary summary: the one thing each edge passes

| Edge | Producer | Consumer | **The one value** | Ownership | Thread |
|---|---|---|---|---|---|
| L0→L1 | `ISensorSource` | `IPreprocessor` | one raw `ImuSample` / `LidarScan` / `CameraFrame` / `GnssFix` | moved (Value) / Shared-immut. buffers | source thread [TS] |
| L1→L2 | L1 aggregator (pipeline) | `IFrontEnd` | `PreprocessedGroup` (one sweep + spanning IMU/image/GNSS) | Value + Shared-immut. points | front-end |
| **L2→L3** | **`IFrontEnd`** | **`IBackEnd`** | **`KeyframePacket`** (§6) | **moved; clouds Shared-immut.** | **front-end→back-end [TS]** |
| L2→(live) | `IFrontEnd` | L4 live / L6 / debug | `NavState` (odom frame) | Value | front-end |
| L3→L4 | `IBackEnd` | `IMapLayer` | `GraphUpdate` (who moved) | Value | back-end→map [TS] |
| L3→L2 | `IBackEnd` | `IFrontEnd` | refined `CalibrationSet` snapshot | Shared-immut., versioned | back-end→front-end [TS] |
| (store) | L2/L4 | L4/L5/mesh | retained clouds via `KeyframeStore` | Shared-immut., id-keyed | map thread |
| L5→L3 | `ILoopDetector` | `IBackEnd` | `LoopConstraint` (rel pose+cov+fitness) | Value | back-end |
| L4→L6 | `IMeshExtractor` | operator UI | `ColorMesh` | moved | map→UI [TS] |

The **bold row is MUST-FIX #1**: one type, one direction, self-contained, with
the information-accounting discipline of MUST-FIX #3 baked into its
`constraint_kind` enum, the de-integration data (clouds) of MUST-FIX #4 baked
into its `cloud_body` handle, and the internal deskew of MUST-FIX #2 hidden
behind `IFrontEnd::ingest` so the packet type is bootstrap-agnostic.

---

## 10. Worked example: a packet's life

Concrete trace of one keyframe, to make the contracts tangible.

1. **t = 1000.000 s.** `OusterLidarSource` (an `ISensorSource<LidarScan>`)
   finishes a sweep, stamps it via PTP, and moves a `LidarScan` (Shared-immutable
   `points`, `stamp_start = 1000.000 s`, per-point `t_offset_ns` from 0 to
   ~100 ms) to its callback. **[TS]** onto the front-end thread.

2. `ILidarPreprocessor::process` filters and decimates, returning a smaller
   `LidarScan`. The L1 aggregator (spec 02 §8) bundles it with the IMU spanning
   the sweep (and any image/GNSS in the interval) into a `MeasureGroup`; the
   pipeline wraps it as a `PreprocessedGroup` and forwards it immediately.

3. `IFrontEnd::ingest(group)`. The LIO front-end propagates across the sweep on
   the group's IMU, deskews each point to the sweep end with the constant-screw
   model, downsamples to keypoints, and runs the Gauss-Newton ICP against its
   voxel-hash local map. Live `NavState` (the rebased IMU-rate propagation) is
   published on `odom`. The front-end decides this sweep is a keyframe
   (translation since last KF > 0.5 m).

4. The front-end assembles a `KeyframePacket`:
   `id = 412`, `stamp = 1000.100 s`, `ref_frame = Odom`,
   `T_ref_body = ` converged pose, `kinematics_included = false`,
   `constraint_kind = RelativeBetween`, `rel_to_id = 411`,
   `T_relto_this = ` relative pose, `constraint_cov = ` 6-DoF marginal $\Sigma$,
   `observability = ` scores (say `tz = 0.2` — a flat parking lot, weak in Z),
   `cloud_body = ` the deskewed Shared-immutable cloud,
   `image = ` the nearest `CameraFrame`, `T_body_cam = ` extrinsic snapshot.
   It **moves** the packet into the keyframe sink. **[TS]** to the back-end queue.

5. `IBackEnd::add_keyframe(std::move(kf))`. Sees `RelativeBetween`, adds **one**
   `BetweenFactor(411→412)` with the marginal cov as its noise model — *no* IMU
   factor, *no* absolute prior (MUST-FIX #3 satisfied). It forwards
   `cloud_body`/`image` to the `KeyframeStore` and to `ILoopDetector`.

6. `IMapLayer::integrate(kf, T_map_body)` fuses the cloud into the NVBlox
   TSDF+RGB at the current pose. The store retains the cloud handle.

7. **t = 1042 s.** `ILoopDetector::detect()` recognises keyframe 412 closes a
   loop with keyframe 27; GICP on the two retained clouds yields a
   `LoopConstraint`; PCM passes it. `IBackEnd::add_loop_constraint` →
   `optimize()` snaps the trajectory; `GraphUpdate` reports keyframes 200–412
   moved.

8. `IMapLayer::apply_graph_update(update, store)`: clears the voxels in the moved
   region and **re-integrates** `store.cloud(id)` for each moved keyframe at its
   corrected pose (MUST-FIX #4 — clear-and-rebuild, not per-voxel subtraction).

9. `IMeshExtractor::extract` re-meshes the dirty blocks → updated `ColorMesh` for
   the operator (L6), with `confidence` low where `observability.tz` was weak.

At no point did L3–L6 know whether L2 was an iEKF or a discrete LIO. That is the
contract working.

---

## Appendix A — enum and small support types referenced above

```cpp
enum class Frame : std::uint16_t {
  Unknown = 0, Map, Odom, BaseLink, ImuLink, CamLink, GnssLink,
  OsSensor0,            // the single LiDAR (room reserved for OsSensor1.. if a multi-LiDAR extension is ever added)
  Body /* alias of F_e */
};
// Frame tagging rule for F_e: `Body` is the canonical tag for estimation-frame
// quantities emitted by L2+ (ObservabilityReport, deskewed clouds, debug topics).
// `ImuLink` names the physical sensor frame and appears only in extrinsic lookups
// (CalibrationSet.estimation_frame pins which physical frame F_e is). Consumers
// must not compare a Body-tagged quantity against ImuLink by enum equality.

struct SensorInfo {
  Frame frame; std::uint8_t id; double nominal_rate_hz; std::string model;
};

struct StampedPose { Timestamp stamp; std::uint64_t kf_id; Pose T_map_body; };

// Canonical PlaneHit — the L4→L2 return of IMapLayer::query_plane (§7.5).
// Spec 06 §3.5 implements exactly this shape; do not redefine it locally.
struct PlaneHit {
  Eigen::Vector3f n;          // unit normal, MAP frame (n·x + d = 0)
  float           d;          // plane offset
  float           rms;        // plane-fit residual (confidence, for residual weighting)
  Eigen::Vector3f centroid;   // voxel centroid
  std::uint8_t    n_pts;      // support count
};

// MeasureGroup — the per-sweep measurement bundle: one LiDAR sweep plus the IMU
// spanning it (and any image/GNSS in the interval). Built by the L1 aggregator
// (spec 02 §8). A multi-LiDAR rig — future extension only — would add sweeps
// behind the same type; not designed now, so there is no secondary-sweep field.
struct MeasureGroup {
  Timestamp                  t_begin = 0;   // sweep start (= scan.stamp_start)
  Timestamp                  t_end   = 0;   // sweep end   (= stamp_start + sweep_duration)
  LidarScan                  scan;          // the LiDAR sweep (Shared-immutable points)
  std::vector<ImuSample>     imu;           // samples in (lower, t_end] with
                                            // lower = min(prev_t_end, t_begin), plus the
                                            // one sample straddling t_begin; overlapped
                                            // sweeps (stamp jitter) share their window
  std::optional<CameraFrame> image;         // image whose mid-exposure falls in the interval, if any
  std::vector<GnssFix>       gnss;          // fixes in the interval (usually 0 or 1)
};

// PreprocessedGroup — the L1→L2 currency: the assembled MeasureGroup. A wrapper
// rather than the group itself so future L1 products can ride along without an
// IFrontEnd signature change. The front-end ingests one PreprocessedGroup per
// sweep (§7.3); the scan inside is raw (undeskewed, time-sorted).
struct PreprocessedGroup {
  MeasureGroup group;
};

struct FrontEndDiagnostics {
  double  scan_time_ms, solve_time_ms, deskew_time_ms;   // per-stage timing
  int     effective_points;                              // points that found a correspondence
  double  mean_residual, final_residual;
  ObservabilityReport observability;
  int     iterations, outer_iters;        // solver iterations / re-association passes
  bool    restarted;                      // restart flag (MUST-FIX #2)
  bool    deadline_hit;                   // wall-clock solve cut off early (always false on deterministic replay)
};
struct BackEndDiagnostics { int num_keyframes, num_loops; double isam_update_ms; bool last_optimize_diverged; };
struct MapDiagnostics     { std::size_t num_voxels, num_blocks; double integrate_ms, mesh_ms; };
struct LoopDiagnostics    { int candidates, verified, rejected_pcm; double sc_query_ms, gicp_ms; };
```

These support types are *diagnostics/identity* helpers, not boundary currencies;
they may grow without affecting the core contracts. The diagnostics structs are
the concrete realisation of the "debug/introspection in the right places"
principle — they are what the ROS 2 wrapper publishes as debug topics and what
rviz markers and structured logs are built from (per-stage timing, effective
points, residuals, observability, restart flags).

---

*End of spec 01. Next: `specs/04_frontend_estimation.md` (the discrete LIO
formulation — internal screw deskew, the GN system, the covariance chain)
consumes the types defined here.*
