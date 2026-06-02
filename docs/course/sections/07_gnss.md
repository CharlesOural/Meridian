## 07. GNSS & absolute residuals: frames, ENU, switchable

> **Course chapter:** *Tightly-Coupled Multi-Sensor State Estimation & Residuals*
> **This section:** 07 of 12. Builds on §02 (manifolds, $\boxplus/\boxminus$, the
> error state and its Jacobians), §03 (MAP = nonlinear least squares, Gaussians,
> information form), and §04 (the body/IMU state $x$ and its short-term
> propagation). **Feeds:** §08–§09 (how this residual is *solved*, batch and
> recursive), §11 (degeneracy, robust kernels, GNC, PCM — the machinery that
> makes GNSS *safe*), and §12 (one full estimator step end-to-end).
> Frame conventions are normative in [`docs/specs/00_architecture.md`](../../specs/00_architecture.md)
> (the architecture/frames spec); the GNSS message and factor data types are in
> [`docs/specs/01_interfaces_and_data_types.md`](../../specs/01_interfaces_and_data_types.md).

Every residual we have built so far is **relative**. The IMU residual (§04)
ties two consecutive states; the LiDAR point-to-plane residual (§05) ties the
body to a *local* map; the photometric residual (§06) ties the body to a *local*
set of visual points. None of them knows where "north" is, none of them knows
where the robot is on Earth, and all of them **drift**: integrate enough relative
measurements and the estimate slowly walks away from truth, with an error that
grows without bound. This is not a flaw to be fixed by better tuning — it is
intrinsic to dead reckoning. Position error in a LiDAR-inertial system grows
roughly with distance travelled; yaw is the weakest degree of freedom and leaks
the fastest.

GNSS is different *in kind*. It is an **absolute** sensor: it pins the estimate
to a global, Earth-fixed frame with an error that is *bounded* — noisy, sometimes
badly so, but not drifting. One good fix per minute is enough to stop a
LiDAR-inertial system's slow positional and heading drift from accumulating over
a multi-kilometre traverse. That is the prize.

The danger is the mirror image of the prize. Because GNSS is the one sensor that
can *anchor* the whole graph, a single bad fix — multipath off a building, a
stale value held through signal loss, or a deliberate **spoof** — can yank the
entire estimate to a wrong place and corrupt the map with it. For Meridian's
*tactical operational* use this is not a corner case; it is the threat model. So
this section spends as much effort on **gating and switchability** (§7.7–§7.9 —
how a bad fix is prevented from corrupting the estimate) as it does on the
residual itself (§7.4–§7.6).

A note on grounding. The FAST-LIO / FAST-LIVO2 / Point-LIO reference code on
disk does **not** consume GNSS: FAST-LIO's own paper explicitly lists "GPS"
among the sensors it deliberately does *not* fuse
([`slam-reference/papers/2010.08196.txt`](../../../../slam-reference/papers/2010.08196.txt),
the sensor-list in the introduction). So the residual *mechanics* below are
grounded in the shared MAP / iEKF framework that those codebases *do* implement —
the same $\boxplus$-error-state update that FAST-LIO's `esekfom` toolkit runs for
LiDAR (§09) — while the GNSS-specific geometry, frames, and robustness draw on
general SOTA practice and Meridian's own design intent (frames and back-end
placement per [`docs/specs/00_architecture.md`](../../specs/00_architecture.md)).
Where a claim is GNSS-specific and not in the reference code, it is stated as
SOTA practice, not attributed to the code.

---

### 7.1 What a GNSS receiver actually gives you

Be precise about the *output product* of a receiver, because the residual you
write depends entirely on which product you choose to trust.

For each tracked satellite $s$, the receiver observes a **pseudorange**
$\rho^s$ (a biased, noisy range to the satellite), and usually a **carrier phase**
$\phi^s$ and **Doppler** $\dot\rho^s$. From a set of these *raw observables* plus
the broadcast satellite ephemerides, the receiver's internal least-squares solver
produces a **Position–Velocity–Time (PVT)** solution: a geodetic position
$(\varphi,\lambda,h)=(\text{lat},\text{lon},\text{alt})$, often a velocity, a
time, and — crucially — **quality metadata**:

- a **fix type** (no-fix / 2D / 3D / DGPS / RTK-float / RTK-fixed),
- the number of satellites used and their geometry, summarized as **DOP**
  (Dilution Of Precision: GDOP, PDOP, HDOP, VDOP),
- a **reported covariance** $\Sigma_{\text{gnss}}$ (or per-axis standard
  deviations) in a local frame,
- carrier-to-noise density $C/N_0$ per satellite.

This yields a **coupling spectrum**, exactly parallel to the loose/tight coupling
spectrum introduced in §01:

```
 LOOSE  ----------------------------------------------------->  TIGHT
   |                          |                          |
 PVT position fix        PVT pos + vel              raw pseudorange /
 (lat,lon,alt)           (adds Doppler)             carrier phase
 receiver solves         receiver solves            WE solve, per-satellite,
 everything;             pos+vel; we add            inside our estimator,
 we add ONE factor       TWO factors                + receiver clock state
```

- **Loose coupling** (§7.4): take the receiver's PVT *position* (optionally
  velocity) as a single measurement and add **one** absolute factor per fix.
  Simple, robust, and the correct first target. This is Meridian's first-pass plan.
- **Tight coupling** (§7.6): ignore the receiver's PVT and feed the **raw
  pseudoranges** (one residual *per satellite*) directly into the estimator,
  jointly estimating the receiver clock bias. This survives with fewer than four
  satellites and enables per-satellite outlier rejection. Deferred, but specified
  here because it is the natural home of the spoof defence.

§7.4–§7.5 develop the loose case in full; §7.6 shows what changes for tight.

---

### 7.2 Frames: LLA → ECEF → ENU

GNSS lives in global, Earth-fixed coordinates; Meridian's estimator lives in a
**local tangent frame**. We must bridge the two cleanly and *once*. The frame
graph and naming are normative in
[`docs/specs/00_architecture.md`](../../specs/00_architecture.md); we restate the
math here.

#### 7.2.1 The three coordinate systems

1. **Geodetic / LLA** $(\varphi,\lambda,h)$ — latitude $\varphi$, longitude
   $\lambda$, ellipsoidal height $h$. This is what the receiver reports. It is
   *not* Cartesian: you cannot subtract two LLA triples to get a displacement.

2. **ECEF** (Earth-Centred, Earth-Fixed) — a right-handed Cartesian frame with
   origin at Earth's centre of mass, $x$ through the intersection of the equator
   and the prime meridian, $z$ through the (conventional) North pole, $y$
   completing the triad. Rigid to the Earth, rotates with it. WGS-84 datum.

3. **ENU** (East–North–Up) — a *local* Cartesian tangent plane fixed at a chosen
   **anchor** $(\varphi_0,\lambda_0,h_0)$. **This is Meridian's estimator world
   frame $W$** (the `map`/global frame; introduced in §01 as the ENU world).
   Axes: $x=$ East, $y=$ North, $z=$ Up.

```
   LLA (lat,lon,alt)          ECEF (x,y,z)               ENU (E,N,U) = W (map)
   geodetic, on ellipsoid     Cartesian, Earth-centre    local tangent @ anchor
        |                          |                           |
        |  geodetic->Cartesian     |  rotate by (phi0,lam0)    |
        |   (closed form, 7.1)     |  + translate by anchor    |
        +------------------------->+-------------------------->+
                                   <--- both exact & invertible --->
```

#### 7.2.2 LLA → ECEF (WGS-84)

Let the WGS-84 ellipsoid have semi-major axis $a = 6\,378\,137.0\ \mathrm{m}$ and
first-eccentricity-squared $e^2 = 6.694\,379\,990\times 10^{-3}$. The **prime
vertical radius of curvature** at latitude $\varphi$ is

$$
N(\varphi) \;=\; \frac{a}{\sqrt{1 - e^2 \sin^2\varphi}}.
$$

The closed-form geodetic-to-Cartesian map is then

$$
\mathbf{p}_{\text{ecef}}(\varphi,\lambda,h) \;=\;
\begin{bmatrix}
\big(N(\varphi)+h\big)\cos\varphi\cos\lambda \\[2pt]
\big(N(\varphi)+h\big)\cos\varphi\sin\lambda \\[2pt]
\big(N(\varphi)(1-e^2)+h\big)\sin\varphi
\end{bmatrix}.
\tag{7.1}
$$

This is *exact* (no series approximation) and trivially differentiable. The
inverse (ECEF → LLA) has no elementary closed form for $\varphi$; use Bowring's
or Olson's iteration (a few steps to machine precision). We rarely need the
inverse on the hot path — the anchor is computed once and frozen.

#### 7.2.3 ECEF → ENU (the anchor and the rotation)

Fix an **anchor** $(\varphi_0,\lambda_0,h_0)$ with ECEF position
$\mathbf{p}_{\text{ecef}}^0 = \mathbf{p}_{\text{ecef}}(\varphi_0,\lambda_0,h_0)$.
The rotation from ECEF to the local ENU triad depends *only* on the anchor's
longitude and latitude:

$$
R_{\text{enu}\,\text{ecef}}(\varphi_0,\lambda_0) =
\begin{bmatrix}
-\sin\lambda_0 & \cos\lambda_0 & 0 \\
-\sin\varphi_0\cos\lambda_0 & -\sin\varphi_0\sin\lambda_0 & \cos\varphi_0 \\
\cos\varphi_0\cos\lambda_0 & \cos\varphi_0\sin\lambda_0 & \sin\varphi_0
\end{bmatrix}.
\tag{7.2}
$$

The three rows are the East, North, Up unit vectors expressed in ECEF: row 1
(East) lies in the equatorial-tangent direction, row 3 (Up) is the ellipsoidal
normal. A geodetic point's ENU coordinates are

$$
\mathbf{p}_{\text{enu}} \;=\; R_{\text{enu}\,\text{ecef}}
\big(\mathbf{p}_{\text{ecef}} - \mathbf{p}_{\text{ecef}}^0\big).
\tag{7.3}
$$

Composing (7.1)→(7.3) gives a single function $\mathbf{p}_{\text{enu}} =
g(\varphi,\lambda,h)$ that turns any receiver fix into a point in $W$.

> **Discipline — freeze the anchor.** The anchor and its ENU↔ECEF transform are
> computed *once* (at the first good fix, or from a surveyed point) and reused for
> the whole session; do **not** recompute per fix. Re-deriving
> $R_{\text{enu}\,\text{ecef}}$ and $\mathbf{p}^0_{\text{ecef}}$ from each new fix
> injects sub-millimetre but *correlated* jitter into every absolute factor — a
> classic, avoidable, self-inflicted bias. Store the anchor with the map so a
> session can be resumed in the same global frame.

#### 7.2.4 Why ENU and not raw ECEF

Two reasons. **Numerics:** ECEF coordinates are $\sim 6.4\times 10^{6}\ \mathrm{m}$;
representing a robot moving over a $100\ \mathrm{m}$ site as the difference of two
seven-significant-digit numbers squanders `float`/`double` precision exactly
where the estimator needs it. ENU keeps coordinates small and centred near the
operating area. **Gravity & intuition:** in ENU, gravity is (very nearly) $-z$,
which aligns with the IMU gravity state $g$ (§04) and makes the estimator's
bookkeeping — and every rviz visualization — read naturally. Over Meridian's
operational footprint (kilometres, not hundreds of km) the flat-tangent-plane
approximation error is negligible; for continental-scale operation one would
re-anchor or move to a curved local frame, which is out of first-pass scope.

---

### 7.3 The lever arm: the antenna is not the body

The estimator's state $x = (R, p, v, b_g, b_a)$ (§01, §04) describes the **body
frame $B$**, which Meridian takes to be the **IMU frame**. The GNSS antenna's phase
centre is a *different* physical point, rigidly mounted elsewhere on the
platform. The offset is the **lever arm**, and it comes from offline calibration
(the antenna→body extrinsic is *position-only* — an antenna has no orientation).
Denote the lever arm expressed in the body frame as $\mathbf{t}^{B}_{a}$.

To compare a GNSS fix against the body state we must predict where the *antenna*
is, not where the *IMU* is. With $R$ the body (IMU) orientation in $W$ and $p$
the body position in $W$:

$$
\boxed{\;\mathbf{p}_a(x) \;=\; p \;+\; R\,\mathbf{t}^{B}_{a}\;}
\tag{7.4}
$$

The key consequence: **the GNSS position residual depends on orientation $R$, not
just position $p$.** That coupling is what lets a *moving* single antenna weakly
*observe yaw* (§7.4.2); it is also why a careless implementation that *drops* the
lever arm injects an orientation-coupled bias of magnitude $\|\mathbf{t}^B_a\|$
(easily tens of centimetres) into every fix — a bias that rotates with the
robot, so it cannot be calibrated away as a constant offset.

If $\mathbf{t}^B_a$ is itself poorly known, it can be promoted to a graph variable
and refined online — exactly as Meridian treats all extrinsics, and as §11 (online
calibration / observability) develops. For the loose residual below we treat it
as a known constant unless flagged otherwise.

---

### 7.4 The loose GNSS position residual

This is the workhorse: one fix → one factor.

#### 7.4.1 Measurement model and residual

The receiver reports a geodetic position; convert it through §7.2 to a point in
$W$:

$$
\mathbf{z}_{\text{gnss}} \;=\; g(\varphi,\lambda,h)\ \in\ \mathbb{R}^3 \qquad (\text{frame } W=\text{ENU}).
$$

The prediction is the antenna position (7.4). In the MAP / least-squares language
of §03, the **measurement model** is $h(x) = \mathbf{p}_a(x)$ and the residual is
a plain Euclidean difference — position lives in $\mathbb{R}^3$, so no manifold
$\boxminus$ is needed:

$$
\boxed{\;
r_{\text{gnss}}(x) \;=\; \mathbf{p}_a(x) - \mathbf{z}_{\text{gnss}}
\;=\; \big(p + R\,\mathbf{t}^B_a\big) - g(\varphi,\lambda,h)
\;\in\ \mathbb{R}^3.}
\tag{7.5}
$$

Its contribution to the global cost (the $J(x)$ of §03) is the
information-weighted, robustified quadratic

$$
J_{\text{gnss}} \;=\; s\,\rho\!\Big(\big\lVert r_{\text{gnss}}(x)\big\rVert^2_{\Sigma_{\text{gnss}}}\Big),
\qquad
\lVert r\rVert^2_{\Sigma}= r^\top \Omega\, r,\quad \Omega = \Sigma_{\text{gnss}}^{-1},
\tag{7.6}
$$

with $\rho$ a robust kernel (§7.8, §11), $\Omega$ the information matrix (§03),
and $s$ a **switch** — binary $s\in\{0,1\}$ for a hard gate or continuous
$s\in[0,1]$ for a switchable constraint (§7.9). Setting $\rho(\cdot)=(\cdot)$ and
$s=1$ recovers a plain Gaussian factor. The weighting $\Sigma_{\text{gnss}}$ is
seeded from the receiver-reported covariance, *inflated* by fix-type/DOP
heuristics (§7.7) so a marginal fix contributes proportionally less.

#### 7.4.2 The Jacobian

For the solvers of §08 (Gauss–Newton / Levenberg–Marquardt) and §09 (iEKF) we
need $H = \partial r_{\text{gnss}}/\partial\delta x$, where $\delta x$ is the
error state in the tangent space (§02). Only the rotation and position blocks are
non-zero. Using the right SO(3) perturbation $R\leftarrow R\,\mathrm{Exp}(\delta\theta)$
(§02) and the first-order $\mathrm{Exp}(\delta\theta)\approx I + (\delta\theta)^\wedge$:

$$
\mathbf{p}_a\big(R\,\mathrm{Exp}(\delta\theta),\,p+\delta p\big)
\approx p + \delta p + R\big(I+(\delta\theta)^\wedge\big)\mathbf{t}^B_a
= \mathbf{p}_a + \delta p - R\,(\mathbf{t}^B_a)^\wedge\,\delta\theta,
$$

using the identity $a^\wedge b = -\,b^\wedge a$. Hence, ordering the error state
as $\delta x = [\,\delta\theta,\ \delta p,\ \delta v,\ \delta b_g,\ \delta b_a\,]$,

$$
H_{\text{gnss}}
= \frac{\partial r_{\text{gnss}}}{\partial \delta x}
= \big[\;
\underbrace{-\,R\,(\mathbf{t}^B_a)^\wedge}_{\partial / \partial\delta\theta}
\;\;\;
\underbrace{I_{3}}_{\partial / \partial\delta p}
\;\;\; \mathbf{0} \;\;\; \mathbf{0} \;\;\; \mathbf{0}
\;\big] \ \in\ \mathbb{R}^{3\times 15}.
\tag{7.7}
$$

Two things to read off (7.7):

- If the lever arm is **zero**, the rotation block vanishes and the factor
  observes *position only* — pure translation pinning, no heading information.
  This is why a GNSS-only system cannot fix yaw at standstill.
- With a non-zero lever arm and **motion** (changing $R$), the
  $-R(\mathbf{t}^B_a)^\wedge$ term lets a *sequence* of position fixes weakly
  observe yaw — the same geometric effect by which a moving offset antenna acts
  as a crude compass. This couples directly into the observability analysis of
  §11 (it is one of the dimensions that GNSS can render observable).

Structurally this is identical to how the LiDAR point-to-plane Jacobian (§05) and
the IMU residual Jacobian (§04) are assembled and stacked into the normal
equations of §08: the GNSS factor is just another block row contributing
$H^\top\Omega\,H$ to the information matrix and $-H^\top\Omega\,r$ to the
gradient.

#### 7.4.3 Where the factor lives: back-end, not front-end

A subtle but important architectural choice. In FAST-LIO the exteroceptive
update happens inside the high-rate iEKF — the `h_share` measurement model handed
to the `esekfom` toolkit and called from the mapping loop (§05, §09). One *could*
add a GNSS update there too, and §7.5 gives the equations for exactly that,
because the loose residual is a perfectly ordinary iEKF measurement. But Meridian
places the **GNSS factor in the back-end factor graph (GTSAM / iSAM2), in the
`map`/ENU frame**, not in the front-end odometry. Three reasons:

1. **Rate mismatch.** GNSS arrives at 1–10 Hz, far slower than the IMU-rate
   front-end. It belongs with the other slow, global factors — loop closures and
   the like (§11) — anchored on keyframes.
2. **Latency tolerance.** A fix can be matched to the *nearest keyframe by
   PTP-synced timestamp* and added even if it arrives late; iSAM2 re-linearizes
   only the affected sub-graph. A late, out-of-order measurement is awkward in a
   forward-only filter.
3. **Frame separation (REP-105).** The back-end lives in `map`; a GNSS
   correction is exactly the kind of discontinuous global *snap* that the
   `map`→`odom` transform is designed to absorb. The smooth `odom`→`base`
   front-end never jumps, so the operator sees a continuous trajectory even when
   GNSS yanks the global frame. (See the frame graph in
   [`docs/specs/00_architecture.md`](../../specs/00_architecture.md).)

So: the residual (7.5) and Jacobian (7.7) are written once and consumed by the
back-end Gauss–Newton / iSAM2 solver of §08. The next subsection gives the
recursive (iEKF) form for completeness and for the optional case where a single
very high-quality fix is fused at front-end rate.

---

### 7.5 The same residual as a recursive update (iEKF view)

§09 establishes the equivalence between batch MAP / Gauss–Newton and the iterated
EKF. The loose GNSS factor drops into the iEKF measurement update with no new
machinery — worth seeing concretely because it reuses the *exact* error-state
update structure that the reference `esekfom` toolkit implements for LiDAR.

Given the IMU-propagated prior $(\hat x, \hat P)$ (§04), the GNSS measurement
gives innovation $r = r_{\text{gnss}}(\hat x)$ (7.5) and Jacobian
$H = H_{\text{gnss}}$ (7.7). The standard ESIKF / iEKF update (§09) is

$$
S = H \hat P H^\top + \Sigma_{\text{gnss}}, \qquad
K = \hat P H^\top S^{-1},
$$
$$
\delta x = -K\, r, \qquad
\hat x \;\leftarrow\; \hat x \boxplus \delta x, \qquad
\hat P \;\leftarrow\; (I - K H)\,\hat P,
\tag{7.8}
$$

iterated (re-evaluating $r,H$ at the updated $\hat x$) until convergence, with the
$\boxplus$ of §02 applying $\delta\theta$ on $SO(3)$ and the remaining blocks
additively. This is identical in *shape* to the LiDAR update the reference filter
runs; only $h(x)$, $r$, and $H$ change. The robust / switchable handling of
§7.8–§7.9 enters here by *inflating* $\Sigma_{\text{gnss}}$ or by gating: a
rejected fix simply sets $K=0$ (no update), which is exactly $s=0$ in (7.6).

> **Why the back-end is safer anyway.** Because the iEKF marginalizes
> immediately, a bad fix that slips past the gate is hard to *undo* — there is no
> later opportunity to re-weight it. In the smoothing back-end, a subsequent
> contradicting measurement can re-linearize and down-weight the offending factor
> (and PCM, §7.9 / §11, can reject it as part of an inconsistent set). This
> asymmetry is the estimation-theoretic argument, on top of the three
> architectural ones in §7.4.3, for keeping GNSS in the back-end. Reserve the
> iEKF form (7.8) for fixes you trust strongly (RTK-fixed, low DOP).

---

### 7.6 Tight coupling: the raw-pseudorange residual

Loose coupling discards information: it commits to the receiver's least-squares
PVT, which needs $\ge 4$ satellites and degrades silently (or extrapolates) in
urban canyons, under canopy, or under partial jamming. **Tight coupling** keeps
the raw observables and lets the *estimator's own* state — backed by IMU and
LiDAR — help resolve the navigation solution. Meridian defers this (it is real
work), but the residual belongs in this section because it is where spoof /
multipath rejection becomes *per-satellite* rather than all-or-nothing.

#### 7.6.1 The pseudorange model

For satellite $s$ at known ECEF position $\mathbf{x}^s$ (from ephemeris), with the
predicted antenna ECEF position $\mathbf{x}_a = \mathbf{p}^0_{\text{ecef}} +
R_{\text{enu}\,\text{ecef}}^\top\,\mathbf{p}_a(x)$ (invert (7.3) to push the
predicted ENU antenna position back to ECEF), the measured pseudorange is

$$
\rho^s = \big\lVert \mathbf{x}^s - \mathbf{x}_a \big\rVert
\;+\; c\,\delta t_r
\;-\; c\,\delta t^s
\;+\; I^s + T^s + \varepsilon^s,
\tag{7.9}
$$

where $c\,\delta t_r$ is the **receiver clock bias** (a *new state* we must
estimate — identical for all satellites at one epoch), $c\,\delta t^s$ the
satellite clock bias (from ephemeris, known), $I^s, T^s$ the ionospheric /
tropospheric delays (modelled, or differenced away with a base station), and
$\varepsilon^s$ measurement noise plus *multipath*. Augment the state with the
clock bias: $x^+ = (x,\ c\,\delta t_r)$.

#### 7.6.2 The residual and its Jacobian

The per-satellite residual is **scalar**:

$$
\boxed{\;
r^s_{\text{prange}}(x^+) =
\Big(\big\lVert \mathbf{x}^s - \mathbf{x}_a(x)\big\rVert + c\,\delta t_r - c\,\delta t^s + \hat I^s + \hat T^s\Big) - \rho^s
\;\in\ \mathbb{R}.}
\tag{7.10}
$$

Let $\mathbf{u}^s = \dfrac{\mathbf{x}_a - \mathbf{x}^s}{\lVert \mathbf{x}_a - \mathbf{x}^s\rVert}$
be the **line-of-sight unit vector** (from satellite toward antenna). The
position Jacobian is the LOS direction rotated into the ENU state frame, the
clock Jacobian is unity, and the rotation block again enters through the lever
arm:

$$
\frac{\partial r^s}{\partial \delta p} = (\mathbf{u}^s)^\top R_{\text{enu}\,\text{ecef}}, \qquad
\frac{\partial r^s}{\partial \delta\theta} = -\,(\mathbf{u}^s)^\top R_{\text{enu}\,\text{ecef}}\, R\,(\mathbf{t}^B_a)^\wedge, \qquad
\frac{\partial r^s}{\partial (c\,\delta t_r)} = 1.
\tag{7.11}
$$

Each satellite contributes one scalar residual; the shared clock bias couples
them. This is precisely why tight coupling can keep navigating with **fewer than
four** satellites: the IMU/LiDAR-supported state plus the few available LOS
constraints jointly pin position and clock, where the standalone receiver would
report *no fix*.

#### 7.6.3 Why tight coupling helps robustness

Because each satellite is now a *separate* residual, a single multipath-corrupted
or spoofed ray shows up as **one** large residual against a state already
well-constrained by IMU + LiDAR + the other satellites. The robust machinery of
§7.8 / §11 — a per-satellite GNC weight, an M-estimator, or RAIM-style residual
testing — can down-weight *that one ray* while keeping the good ones. That is
impossible in loose coupling, where the receiver has already blended everything
into a single contaminated PVT. This is the deepest reason the SOTA migrates to
tight coupling in adversarial environments, and it is the natural slot for the
spoof defences of §7.7.

---

### 7.7 Multipath and spoofing: the threat, and what to gate on

For tactical use the adversary is *active*. Distinguish three failure modes, in
increasing severity:

1. **Multipath / NLOS** (passive, environmental). The signal reflects off
   buildings or terrain before reaching the antenna, so the pseudorange is *too
   long*. The bias is per-satellite, sporadic, often several metres in urban
   canyons. Symptoms: low $C/N_0$ on the affected satellite, high HDOP, residual
   outliers on a *subset* of rays.
2. **Jamming** (active, denial). Broadband noise raises the floor; the receiver
   loses lock. Symptoms: $C/N_0$ collapse across the board, fix downgrades
   (RTK→3D→2D→no-fix), satellite count drops. The honest failure mode: *no* fix.
3. **Spoofing** (active, deception). The adversary transmits counterfeit signals
   so the receiver computes a *plausible but wrong* PVT — the dangerous case,
   because the fix *looks healthy*. Symptoms are subtle: a fix that is internally
   self-consistent yet **inconsistent with the IMU/LiDAR-propagated state**, an
   abnormally *clean* and uniform $C/N_0$, a sudden position jump, or a slow,
   deliberate "walk-off" drag.

The defence is **layered**, and each layer uses the information available at a
different stage:

- **Receiver-reported quality (cheap, first line).** Reject or inflate on: fix
  type below threshold, satellite count below a floor, DOP above a ceiling, and
  abnormal $C/N_0$ patterns. Catches jamming and gross multipath; a competent
  spoofer can fake all of it.
- **Innovation consistency against the fused state (the real defence).** The one
  test that catches a *healthy-looking* spoof — and it is available *only*
  because we hold a tightly-coupled IMU+LiDAR estimate to test against.
  Formalized in §7.8.
- **Per-satellite RAIM (tight coupling only, §7.6).** Statistical testing of
  individual pseudorange residuals to identify and exclude a faulty / spoofed
  satellite; requires redundancy (more satellites than the minimum).

The guiding principle for Meridian: **GNSS is never trusted unconditionally; it is
admitted only insofar as it agrees with the dead-reckoned estimate it is meant to
correct.** A self-consistent but estimate-contradicting fix is the signature of a
spoof, and the gate below is built precisely to refuse it.

---

### 7.8 The consistency gate: normalized innovation (NIS / Mahalanobis test)

The mathematically principled gate is the **innovation test** familiar from §09,
generalized here to the smoothing back-end. Before admitting a fix, evaluate the
innovation $r = r_{\text{gnss}}(\hat x)$ (7.5) against the current best estimate
$\hat x$ (with prior covariance $\hat P$ from IMU/LiDAR propagation), and form the
**innovation covariance** and the **Normalized Innovation Squared**:

$$
S = H \hat P H^\top + \Sigma_{\text{gnss}}, \qquad
\chi^2 \;=\; r^\top S^{-1} r .
\tag{7.12}
$$

Under the null hypothesis (the fix is good and the models correct),
$\chi^2 \sim \chi^2_d$ with $d=3$ degrees of freedom (3-D position) — or $d=1$
per satellite in the tight case. Pick a confidence level (e.g. 99%) giving a
threshold $\gamma$ ($\chi^2_{3,\,0.99}\approx 11.34$):

$$
\text{admit if } \chi^2 \le \gamma; \qquad
\text{reject (or down-weight) if } \chi^2 > \gamma.
\tag{7.13}
$$

This single test does the heavy lifting against spoofing: a counterfeit fix may
be internally self-consistent, but unless the spoofer also knows *and matches*
the robot's true IMU/LiDAR-tracked trajectory, the *innovation* against the fused
estimate blows past $\gamma$. The IMU's bounded short-term drift (§04) gives a
tight, trustworthy $\hat P$ over the seconds between fixes — exactly the window a
walk-off spoof tries to exploit, and exactly the window in which inertial dead
reckoning is most reliable.

**Hard gate vs. soft down-weight.** Equation (7.13) is a *hard* binary gate
evaluated on a *linearization*, before the solver runs. A softer, usually
better-behaved alternative lets the gate **inflate the covariance** instead of
discarding — scaling $\Sigma_{\text{gnss}}$ by a factor that grows with $\chi^2$;
or, equivalently, applying a robust kernel $\rho$ (§11) so the cost (7.6)
*saturates* for large residuals. The robust-kernel view writes the factor cost as

$$
J_{\text{gnss}} = s\,\rho\!\big(\lVert r_{\text{gnss}}\rVert_{\Sigma}\big),
$$

with $\rho$ a Huber or a redescending (Geman–McClure / Cauchy) kernel (§11). A
Huber kernel grows linearly past its threshold (bounded influence, but keeps some
pull); a redescending kernel drives the weight toward *zero* for gross outliers
(a spoof gets fully ignored). For the adversarial setting a redescending kernel,
ramped in by **GNC** (Graduated Non-Convexity, §11 — start convex to find a good
basin, then sharpen so true outliers are rejected without local-minimum risk), is
the SOTA choice. Gate and kernel are complementary: the $\chi^2$ gate is a fast
first cut on the obviously bad; the kernel handles the marginal cases gracefully
*inside* the optimizer.

---

### 7.9 Switchable constraints: the principled "off switch"

The hard gate (7.13) makes a binary, non-differentiable decision *before* the
solver runs, on a *linearization* of the residual. **Switchable constraints**
(Sünderhauf & Protzel) push that decision *inside* the optimization and make it
**continuous and self-adjusting**, so the optimizer itself decides how much to
trust each GNSS factor, jointly with the trajectory.

Augment the state with a **switch variable** $s\in\mathbb{R}$ (mapped through a
sigmoid or clamp to $[0,1]$) *per GNSS factor*. The factor cost becomes

$$
\boxed{\;
J_{\text{gnss}} = \big\lVert \Psi(s)\, r_{\text{gnss}}(x) \big\rVert^2_{\Sigma_{\text{gnss}}}
\;+\; \big\lVert\, s - s_{\text{prior}} \,\big\rVert^2_{\Sigma_s}\;}
\tag{7.14}
$$

where $\Psi(s)$ is a switch function (commonly $\Psi(s)=s$ clamped to $[0,1]$, or
a sigmoid) that scales the residual, and the **second term is a prior pulling $s$
toward "on"** ($s_{\text{prior}}=1$) with strength $\Sigma_s$. The mechanism is an
elegant tug-of-war:

- If the GNSS factor *agrees* with the rest of the graph, keeping $s\approx 1$
  costs nothing in the data term and satisfies the prior → the switch stays on.
- If the GNSS factor *fights* the graph, the optimizer can **turn it down**
  ($s\to 0$), paying only the *bounded* prior penalty
  $\lVert s-1\rVert^2_{\Sigma_s}$ while *escaping* the *unbounded* data-term
  penalty → a bad fix de-weights itself, smoothly and automatically.

$\Sigma_s$ tunes the credulity: a tight $\Sigma_s$ makes the system reluctant to
disable a fix (trust GNSS); a loose $\Sigma_s$ makes it quick to disable
(suspicious of GNSS — the tactical default). The Jacobians extend (7.7) with a
column $\partial r/\partial s = \Psi'(s)\,r_{\text{gnss}}(x)$ plus the trivial
prior block, and the whole thing drops straight into the Gauss–Newton normal
equations of §08 / iSAM2.

**Relationship to the other tools.** Switchable constraints, robust kernels, and
the $\chi^2$ gate are not rivals — they are the same idea at different commitment
levels. A switchable constraint is provably equivalent to optimizing a particular
robust kernel (the Black–Rangarajan duality: the switch $s$ *is* the IRLS weight
of that kernel's $\rho$). Meridian's layered stance, consistent with §11:

```
  cheap, pre-solve          continuous, in-solver           global, multi-factor
  +-----------------+       +-----------------------+        +--------------------+
  | receiver flags  |  -->  | switchable constraint |  -->   | PCM: reject the    |
  | + chi^2 gate    |       | + robust kernel (GNC) |        | INCONSISTENT SET   |
  | (§7.7, §7.8)    |       | (§7.9, §11)           |        | (§11, place rec.)  |
  +-----------------+       +-----------------------+        +--------------------+
```

**PCM** (Pairwise Consistency Maximization, §11 and the L5 place-recognition
stack) is the last line of defence. Where switchable constraints judge each GNSS
factor in *isolation*, PCM finds the largest *mutually consistent* set of
absolute / loop constraints and rejects the rest as a group — defeating a
coordinated spoof that injects several mutually-consistent-but-globally-wrong
fixes. GNSS factors are first-class citizens of that consistency graph alongside
loop closures.

---

### 7.10 GNSS-derived heading and velocity (briefly)

Two absolute residuals beyond position deserve mention; both follow the same
template and reuse the notation above.

- **Doppler / GNSS velocity.** Many receivers report a velocity from Doppler,
  which is *more robust to multipath than position* (Doppler is harder to spoof
  consistently). The residual is
  $r_v = v + R\,(\boldsymbol\omega^\wedge \mathbf{t}^B_a) - \mathbf{z}_v$ — body
  velocity plus the lever-arm rotational term, against the reported ENU velocity,
  with $\boldsymbol\omega$ the bias-corrected body angular rate (§04). It directly
  observes the velocity state $v$ and is a cheap, valuable second factor.

- **Dual-antenna / moving-baseline heading.** Two antennas give an absolute *yaw*
  (and pitch) from their baseline vector, observing $R$ *directly* rather than
  weakly through a single lever arm (§7.4.2). The residual compares the predicted
  baseline direction $R\,\mathbf{b}^B$ against the measured ENU baseline. This is
  the clean way to bound the one degree of freedom — heading — that a
  single-antenna LiDAR-inertial system is otherwise weakest at, and it is a strong
  spoof-resistance asset, since faking a *consistent baseline across two antennas*
  is far harder than faking a single position.

Both are out of first-pass scope but slot onto the same back-end with the same
gating / switching machinery; they are flagged here so the interfaces leave room
for them.

---

### 7.11 Putting it together: the GNSS factor lifecycle

End to end, what happens when a fix arrives (the concrete contract the L0/L2/L3
modules implement; cf. the full estimator step in §12):

```
1. RECEIVE   raw fix (lat,lon,alt, fix-type, #sats, DOP, C/N0, cov), PTP-stamped
2. QUALITY   reject / flag on fix-type, #sats, DOP, C/N0   ........... §7.7
3. PROJECT   LLA -> ECEF -> ENU via the FROZEN anchor      ........... §7.2  (7.1)-(7.3)
4. ASSOCIATE match to nearest keyframe by PTP timestamp    ........... back-end / time-sync
5. PREDICT   antenna position  p_a(x) = p + R t^B_a        ........... §7.3  (7.4)
6. RESIDUAL  r = p_a(x) - z_gnss ;  build Jacobian H       ........... §7.4  (7.5),(7.7)
7. GATE      chi^2 = r^T S^-1 r   vs threshold gamma       ........... §7.8  (7.12)-(7.13)
8. ADD       switchable constraint + robust kernel into    ........... §7.9  (7.14)
             the GTSAM / iSAM2 graph (map / ENU frame)            §7.4.3
9. SOLVE     back-end re-optimizes; map->odom snaps         .......... §08, frame graph
10. DEBUG    publish residual, chi^2, switch value s, fix-type, ...... cross-cutting
             #sats, covariance ellipse as rviz markers + topics       (debug principle)
```

Step 10 is non-negotiable under Meridian's engineering principles: the operator and
developer must be able to *see* the GNSS subsystem's decisions — the live
$\chi^2$, the per-factor switch values $s$, which fixes were admitted vs.
rejected, and the covariance ellipse in rviz — so a spoof or a multipath storm is
*visible*, not silent. A GNSS factor that quietly de-weights itself ($s\to 0$)
while the operator watches is exactly the introspection the system is built to
provide.

---

### 7.12 Key takeaways

- GNSS is the only **absolute, non-drifting** sensor in Meridian; one good fix per
  minute bounds the LiDAR-inertial drift of §04–§06 over long traverses.
- It is also the most **dangerous** sensor: a single bad / spoofed fix can corrupt
  the whole graph, so gating and switchability matter as much as the residual.
- Frames: **LLA → ECEF → ENU** (eqs. 7.1–7.3) with a **frozen anchor**; ENU is
  the estimator world frame $W$ / `map` (architecture spec).
- The **lever arm** (7.4) makes the position residual depend on $R$, weakly
  observing yaw under motion; never drop it.
- **Loose** coupling = one position factor per fix (7.5), Jacobian (7.7), placed
  in the **back-end** graph (§7.4.3) — the correct first target.
- **Tight** coupling = per-satellite pseudorange residual (7.10) with a clock
  state, surviving < 4 satellites and enabling per-ray outlier rejection
  (deferred, but the natural home of the spoof defence).
- Robustness is **layered**: receiver flags → $\chi^2$ consistency gate
  (7.12–7.13) → switchable constraint + robust kernel / GNC (7.14) → PCM — the
  same defence-in-depth philosophy as the loop-closure robustness of §11.
- Everything is **introspectable**: residual, $\chi^2$, switch value, and
  covariance are first-class debug outputs.

> **Next:** §08 shows how this factor — and all the others — is *solved* in the
> batch normal equations (sparsity, Schur complement); §09 gives the recursive
> (iEKF) equivalent used at front-end rate; §11 develops the robust kernels, GNC,
> and PCM this section leans on; §12 walks one full tightly-coupled estimator step
> end to end.
