# Architecture Decision Log

A running log of load-bearing design choices in Meridian, with the *why* and the
*evidence* preserved so the reasoning survives the code and can later seed a paper. Each
entry is an ADR (Architecture Decision Record): a decision, the options weighed, the
empirical result that settled it, and the mechanism that explains the result.

Scope note for the paper: the target platform is a **Shark Robotics Barracuda 4×4
offroad wheeled robot** (small-car class, heavy terrain-induced vibration) plus handheld
operation, on a **go-out-and-return, GPS-denied, long-trajectory** mission. "Offroad" in
this log means *aggressive, high-frequency motion over unstructured, often vegetated
terrain* — the regime that distinguishes our design choices from benchmark-leaderboard
ones tuned on smooth indoor/urban data.

Evaluation harness for every number below: deterministic offline replay (`replay_runner`,
synchronous, lossless), ATE via Umeyama-aligned RMSE against `gt/tum_asimu`, on the Newer
College Dataset (2021). Standins for the target regime: **math-medium** (a 2.6 Hz handheld
yaw sway = the aggressive-motion stressor) and **park** (parkland/vegetation, 571 s = the
offroad/unstructured + long-trajectory-drift stressor). quad-easy/quad-hard bracket gentle
and aggressive structured motion.

---

## ADR-0001 — Point-to-point over point-to-plane registration for offroad LiDAR-inertial odometry

**Status:** Accepted (2026-06-12). Point-to-point is the production registration residual;
point-to-plane is implemented but gated off (kept only for a possible gentle/structured
deployment).

### Context

The LIO frontend aligns each deskewed sweep to a local voxel-hash map by Gauss–Newton
ICP. The residual model is the central choice. The two canonical options:

- **Point-to-point:** residual `r = T·p − q` (3-vector) pulls each scan point `p` to its
  nearest map point `q`. Each correspondence constrains all three translation directions.
- **Point-to-plane:** residual `r = n·(T·p − c)` (scalar) pulls each point onto the local
  surface plane `(n, c)` fitted to its map neighbours. Each correspondence constrains
  motion only along the surface normal `n`.

Point-to-plane (and its feature cousins in LOAM and FAST-LIO, and the
distribution forms in GICP/NDT) is the leaderboard default for indoor/urban LiDAR
odometry: it converges in far fewer iterations and is more accurate **when the
environment is planar and the per-scan initial guess is good.** The question for Meridian
was whether that advantage survives the offroad regime.

### Decision

Use **point-to-point** as the default registration residual. Point-to-plane is retained
behind a config flag for structured/gentle deployments only.

### Evidence

ATE (m), point-to-point baseline vs point-to-plane variants:

| Variant | quad-easy (gentle) | math-medium (2.6 Hz sway) | quad-hard (aggressive) | park (vegetation, 571 s) |
|---|---|---|---|---|
| **point-to-point (baseline)** | 0.085 | 0.144 | 0.123 | 0.64 |
| point-to-plane, pure | **0.081** | **1537** 💥 | **6091** 💥 | **129017** 💥 |
| point-to-plane, point-to-point warmup 8 | 0.083 | 87 💥 | 5.8 💥 | 1013 💥 |
| point-to-plane, warmup 20 (≈full p2p first) | 0.086 | 29.7 💥 | 0.30 (degraded) | diverged |

Supporting telemetry: pure point-to-plane converged ~3× faster on quad-easy (registration
~23 ms vs ~73 ms; far fewer GN iterations) and was *more accurate there* (0.081 vs 0.086).
On math-medium it produced a **median of 0 matched correspondences** — it tracked the calm
opening, then lost the map at the onset of the sway and never re-acquired. A
point-to-point→point-to-plane warmup hybrid (with a per-iteration fallback to
point-to-point when too few planes are found) reduced but never removed the divergence,
**even at warmup 20**, where point-to-point has already fully converged before
point-to-plane engages.

### Mechanism (why point-to-plane fails offroad)

Three causes compound, each tied to a property of the regime:

1. **Narrow convergence basin (vs aggressive motion).** A scalar normal-direction residual
   constrains 1 DoF per correspondence and is silent tangentially. From the rough per-scan
   prior that high-frequency motion produces, early iterations associate points to the
   *wrong* planes, and those wrong 1-DoF constraints reinforce a wrong direction rather
   than averaging out. Point-to-point's 3-DoF residual always pulls toward the correct
   region, so bad associations self-correct — a far wider basin of attraction.

2. **Hypersensitivity to normal error (vs deskew residual).** Constant-screw deskew leaves
   a small residual intra-sweep motion that slightly rotates each local point cluster and
   therefore tilts its fitted normal. Point-to-plane constrains *exactly along that normal*,
   so a systematic normal tilt becomes a systematic pose error. Point-to-point's
   full-vector residual averages the tilt away. This is why warmup-20 still diverged on
   math-medium: starting from an already-converged pose, the point-to-plane refinement
   pulled it back off. The failure is in the residual model, not the initialization — no
   warmup schedule fixes it.

3. **Absent / false planes (vs vegetation).** Parkland foliage has few true planes, and
   leaf clusters occasionally pass a planarity gate as *false* planes with arbitrary
   normals. park diverged to ~1 km even with a point-to-point fallback, because enough
   false planes cleared the fallback's correspondence-count threshold.

The unifying statement: **point-to-plane trades convergence-basin width for local
sharpness, and the offroad regime is precisely where basin width matters most.**

### Scope / when this reverses

This decision is *use-case-specific*, not a claim that point-to-point is universally
superior. On gentle, structured motion (quad-easy) point-to-plane was both faster and more
accurate. For a well-initialized indoor/urban deployment it would likely be the better
choice — consistent with the LiDAR-odometry literature. The gated implementation is kept
for that case. The decision is driven entirely by the **aggressive-motion + unstructured-
terrain** target regime.

### Consequences

- Robustness is prioritised over peak structured-scene accuracy and over the ~3× register
  speedup point-to-plane offered on easy data.
- The *robust* path to point-to-plane's accuracy is a distribution-to-distribution residual
  (GICP/NDT-style: match local point covariances, not hard plane fits), which degrades
  gracefully where structure is weak and keeps a wide basin. That is the open accuracy
  lever, deferred to validation on real offroad (Barracuda) data — see ADR-0004.
- General methodological finding (see ADR-0003): registration-robustness features cannot be
  validated on a clean benchmark the baseline already aces; they need adversarial data.

---

## ADR-0002 — Online gyro-bias estimation as a slow observer on the registration result

**Status:** Accepted (2026-06-12); enabled in the eval configs.

**Context.** Static initialisation fixes the gyro bias once (mean rate at standstill) and
never updates it, so on long trajectories the bias drifts (temperature, etc.) and the
orientation slowly walks — the dominant error on park (571 s). The rko-style architecture
uses the IMU as a motion *prior*, not a fused measurement, so there is no filter state to
estimate bias in.

**Decision.** Keep the IMU as a prior, but make the *registration result* a slow observer
of gyro bias: each scan, fold the body-frame attitude discrepancy
`e = log(R_measured⁻¹ · R_imu_predicted)/dt` into `b_g` with a low gain. Random ICP
rotation noise averages out across scans; only the consistent bias error accumulates.

**Evidence (gain 0.05, vs point-to-point baseline).** quad-easy 0.085→0.086 and math-medium
0.144→0.146 (both neutral — the feared absorption of systematic deskew error did *not*
occur), quad-hard 0.123→0.118 (−4 %), **park 0.64→0.517 (−19 %, max 1.46→0.95 m).** A gain
sweep confirmed 0.05 as the optimum (0.1 over-corrects: park 0.576, quad-hard regresses).

**Mechanism / why it is safe.** The low gain is the crux: a single interval's discrepancy
is mostly registration noise; only the bias error is correlated across intervals, so a slow
filter is an unbiased estimator of the drift and leaves the noise-dominated clean sequences
untouched. It improves exactly the long-trajectory drift it targets, with no basin-width
cost (it rides on top of robust point-to-point) — the opposite profile to point-to-plane.

**Consequence.** Accel bias is deliberately *not* estimated online (prior art — STEAM-LIO —
shows accel-in-state hurts reliability under aggressive motion; the gravity regulariser
already handles the at-rest accel component).

---

## ADR-0003 — Tune the convergence threshold, never cap the iteration count

**Status:** Accepted (2026-06-12); `convergence_eps` 1e-5→1e-3 committed.

**Context.** Per-phase timing showed ~90 % of the frontend is the per-iteration
nearest-neighbour association in the GN solve; the linear solve itself is ~0.03 ms. The
solve was running ~32 iterations because the convergence threshold (1e-5, ~10 µm step) was
far tighter than needed.

**Decision & evidence.** Loosen `convergence_eps` to 1e-3 (~1 mm step): iterations 32→20,
frontend 116→77 ms (dev host), **zero ATE change** (the discarded iterations moved the pose
< 1 mm). A hard iteration cap was tested and **catastrophically rejected**: cap=10 diverged
on every sequence (thousands of metres) — point-to-point genuinely needs its ~20 iterations
to converge, and truncating *below* convergence leaves each pose far enough off that the
next scan's association fails and the system diverges.

**Principle (paper-relevant).** Terminate at *natural convergence* (a step threshold);
never truncate *below* it (a hard cap). The two look similar but the second removes
robustness. Also established here: **a benchmark the baseline already passes cannot validate
robustness features** — both the fixed-scale robust kernel (rejected: starved sparse
vegetation, park 0.64→7.8 m) and point-to-plane looked fine on the clean sequences and
failed on the stressors. Robustness must be validated on adversarial (Barracuda) data.

---

## Open decisions (to be logged when settled)

- **ADR-0004** — Distribution-to-distribution (GICP/NDT-style) registration as the robust
  route to plane-level accuracy; needs offroad data to validate (see ADR-0001 consequences).
- **ADR-0005** — Adaptive robust kernel (residual-statistic-scaled) vs the rejected
  fixed-scale kernel; needs offroad data.
- **ADR-0006** — Degeneracy handling in open/feature-poor terrain via the `H` spectrum.
- **ADR-0007** — Loop closure / submapping for residual global drift on go-out-and-return.
