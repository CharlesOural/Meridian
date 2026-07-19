# Global implementation provenance

This package is an original Meridian implementation. The reference code below was read to
fix frame conventions, identify observability gates, and avoid common integration mistakes.
No GPL source was copied or translated.

## LiDAR place retrieval and geometric verification

`include/meridian/global/lidar_place.hpp`, `src/lidar_place.cpp`, and
`test/test_lidar_place.cpp` are an original Meridian implementation grounded in
the MIT-licensed checked-in MapClosures snapshot. The complete algorithmic
context read for this seam was:

- `../slam-reference/MapClosures/cpp/map_closures/DensityMap.{hpp,cpp}`;
- `../slam-reference/MapClosures/cpp/map_closures/GroundAlign.{hpp,cpp}`;
- `../slam-reference/MapClosures/cpp/map_closures/MapClosures.{hpp,cpp}`;
- `../slam-reference/MapClosures/cpp/map_closures/AlignRansac2D.{hpp,cpp}`.

Meridian preserves the density-valued BEV, ORB, self-similarity pruning,
binary-descriptor voting, deterministic metric SE(2) verification, and full
seed-recovery structure. Sparse submap frames are already gravity aligned by
the binding system API, so Meridian does not repeat MapClosures' local-ground
roll/pitch estimation; it records a robust low-height statistic only for the
seed's vertical translation. Density uses a log-count image rather than a
binary occupancy image, retaining density while bounding the image range.

MapClosures obtains HBST through a network `FetchContent` declaration and the
HBST source is not present in the checked-in reference. This is the exact HBST
departure: Meridian uses one bounded, deterministic, 16-table binary
multi-index owned by `LidarPlaceIndex`, then performs exact 256-bit Hamming
validation. It never has a hidden sequential-ID exclusion rule or a retriever
that mutates the database. Identity, overlap/adjacency exclusions, temporal
separation, TTL, top-K, model/config revisions, comparison work, and capacity
failures are explicit reports. A complete query that would exceed its bound is
deferred with an error rather than prefix-truncated. Retrieval score schedules
verification only and is absent from graph information.

The independent verifier was informed by the MIT-licensed
`../slam-reference/gtsam_points/include/gtsam_points/factors/impl/integrated_vgicp_factor_impl.hpp`
and by the bounded registration/overlap sequence in GLIM
`../slam-reference/glim/src/glim/mapping/global_mapping_pose_graph.cpp` (GPL-3.0,
research context only). No GLIM code was copied. Meridian implements a fresh
symmetric directional surfel-GICP objective over immutable registration
proxies, robust damped right-local refinement, and fresh final
correspondences. Normal-inconsistent nearest neighbours remain visible in raw
overlap reports but are excluded from the optimizer and final information.
Acceptance separately gates bidirectional overlap, distributed support,
residual median/upper quantile, normal consistency, correction from the
independent seed, gravity tilt, and supported rank. It deliberately never
reads current corrected XY distance. Final graph information is recomputed in
the right, translation-first tangent from robust normal-projected rows, uses a
declared correlation inflation and eigenvalue cap, and leaves unsupported
directions exactly zero. `RegistrationProxy` currently has no semantic dynamic
label, so dynamic fraction is reported as unavailable and is never fabricated
from its generic weight.

## Visual place retrieval and geometric verification

`include/meridian/global/visual_place.hpp`, `src/visual_place.cpp`, and
`test/test_visual_place.cpp` are an original bounded implementation informed by
the BSD-3-Clause OKVIS/OKVIS2-X DBoW2 and loop-verification paths. In
particular, the following local reference seams were inspected:

- `../slam-reference/OKVIS2-X/okvis_apps/src/dbow2_test.cpp`;
- `../slam-reference/OKVIS2-X/okvis_multisensor_processing/include/okvis/ThreadedSlam.hpp`;
- the OKVIS/OKVIS2-X loop-detection and multi-camera frame records reachable
  from those application and processing seams.

Meridian owns a bounded fixed-vocabulary TF-IDF/L1 core for 64-byte binary
descriptors. Vocabulary model revision, checksum, and configuration revision
are mandatory in records, seeds, reports, and verification. Vocabulary
training is offline only; runtime mutation/training is absent. The OKVIS
`small_voc` artifact is deliberately not imported: it is a 48-byte FBrisk
vocabulary, while Meridian's current OpenCV BRISK records are 64 bytes, so
loading it would be a silent descriptor-model incompatibility rather than
reuse.

The geometric lane uses one OpenCV AP3P seed per central camera and never pools
observations from different camera centres into a fictitious central camera.
Accepted seeds enter one common-pose refinement over calibrated bearings, with
fresh outlier classification and rank-aware Hessian output. This is an
explicit departure from OKVIS' OpenGV generalized-P3P path because OpenGV is
not an available build dependency in the current framework. The per-camera
seed plus joint calibrated-bearing refinement is the single implemented path,
not a fallback. Retrieval score remains scheduling-only and does not scale the
geometric information.

## Sparse submap lifecycle and condensation seam

`include/meridian/global/sparse_submap.hpp` and `src/sparse_submap.cpp` are an
original Meridian implementation of section 13's finalized sparse boundary.
The following reference files were inspected as behavioral context:

- GLIM `src/glim/mapping/sub_mapping_passthrough.cpp` (GPL-3.0, research
  context only): bounded keyframe/voxel triggers, a fixed local submap origin,
  one-time proxy construction, and delivery of completed submaps to a separate
  global mapper;
- GLIM `src/glim/odometry/odometry_estimation_gpu.cpp` (GPL-3.0, research
  context only): overlap/displacement keyframe policies and the explicit handoff
  of marginalized frames;
- OKVIS2-X
  `okvis_multisensor_processing/src/SubmappingInterface.cpp` and
  `include/okvis/SubmappingInterface.hpp` (BSD-3-Clause): keyframe-count and
  modality-overlap split inputs, submap-local sensor poses, and completed-map
  alignment records.

Meridian does not copy those lifecycle implementations. In particular, it does
not make a mutable dense volume the sparse identity, does not use a live
keyframe pose as global evidence, and does not export optimizer types. It adds a
half-open terminal-time owner, an explicit out-of-lag state finality barrier,
split abort/reassignment, exact raw-factor partition checks that exclude the
incoming marginal prior, gravity-aligned immutable origins, joint-endpoint
covariance propagation into translation-first rank-aware information, bounded
in-memory registration/place records, and canonical SHA-256 seal identity.
The lifecycle deliberately labels its direct output `VolatileInProcessOnly`;
durability begins only after the separate `SealSpool::enqueue` operation has
returned success.

## Durable sparse-seal spool and outbox

Package placement is provisional in this implementation snapshot. The final
ROS-free owner is `meridian_local_rt`, beside sparse finality; global consumes
the durable outbox and will own its own seal cache instead of owning this spool.
No new dependency from the spool to global graph internals is introduced here.

`include/meridian/global/seal_spool.hpp`, `src/seal_spool.cpp`,
`src/persistence_internal.{hpp,cpp}`, and
`test/test_seal_spool.cpp` are an original Linux implementation of the
crash-ordering and replay rules bound by SYSTEM_SPECS section 6.6. No external
persistence implementation was copied. Canonical records use an explicit
versioned, big-endian field encoding and SHA-256 frame checksum. Publication
uses a same-directory temporary file, file `fsync`, Linux atomic no-replace
rename, and directory `fsync`; recovery removes only torn temporary files and
validates every seal, outbox entry, acknowledgement, byte count, canonical
round trip, checksum, and sequence relationship.

The private persistence core is shared with the graph journal and sparse seal
identity path so SHA-256, canonical framing, bounded file reads, directory
sync, temporary cleanup, and immutable no-replace writes have one behavior.
Its startup known-answer check covers the empty, `abc`, and standard
multi-block vectors. An identical destination is reconciled by reading exact
bytes and re-syncing the directory; an unreadable destination is an explicit
integrity ambiguity, never mislabeled as a content conflict.

The implementation is deliberately a bounded correctness spool, not a map
store. It never removes an unacknowledged seal and stops new growth with a
typed capacity error. A place payload is accepted only when it already names a
lease-free `DurableSpool` `BlobRef` whose identity, layout, bytes, and checksum
are valid. Process-local `ImmutableBytes`, shared-memory leases, and ordinary
in-process references fail before disk mutation: this component does not claim
to have persisted child bytes it cannot resolve. A future blob-copy owner may
materialize those child closures before calling this API without changing the
outbox format. That durable child-blob owner and the global seal cache do not
exist yet and remain required integration work.

## Durable graph journal and checkpoints

`src/graph_journal.cpp` is an original Meridian implementation of the crash
ordering specified in SYSTEM_SPECS section 6.6. It uses a single versioned,
canonical big-endian encoding for complete `GlobalGraphCommit` checkpoints,
PREPARED transaction inputs/dispositions, and COMMITTED revision markers.
SHA-256 content identities, no-replace rename, file fsync, and directory fsync
use the shared private persistence core and documented Linux/POSIX primitives.
Recovery resolves the contiguous committed-marker closure before classifying
child records, retires losing prepared siblings and unreferenced checkpoints,
and exposes a last verified prefix read-only when a later marker is corrupt.
Corruption in the first/only committed closure still fails opening. There is
no delta codec or lossy graph compaction.

The journal currently recovers immutable commit snapshots for inspection and
replay only. `GlobalGraph` does not yet expose a restore/install API for its
complete factor and solver state, so the journal cannot independently resume
the active optimizer or make graph installation atomic with the durable commit
marker. That integration remains required before restart recovery is complete.

## Private full-boundary adjacent factor adapter

`src/adjacent_boundary_factor_internal.{hpp,cpp}` and
`test/test_adjacent_boundary_factor.cpp` are an original Meridian adapter for
the solver-neutral `CondensedBoundaryTransition`. The implementation keeps all
thirty canonical endpoint columns `(A,V,Bg,Ba)_i,(A,V,Bg,Ba)_j`, evaluates the
frozen nonlinear pose charts, changes the velocity basis through the immutable
gravity-preserving epoch placement, and applies the stored square-root rows
without replacing their null space with a covariance or diagonal precision.
The GTSAM pose perturbation is converted explicitly from rotation-first to the
Meridian right translation-first tangent, including the endpoint-frame
adjoint. Analytic Jacobians are tested against central differences away from
the stored chart centers, and their Hessian is checked against the canonical
row Hessian under the exact coordinate basis change.

The following checked-in implementations were read for mathematical and
integration context:

- GLIM `src/glim/mapping/global_mapping.cpp` and
  `src/glim/mapping/sub_mapping.cpp` (GPL-3.0, research context only): separate
  endpoint pose, navigation velocity, and IMU-bias variables and explicit
  velocity-frame rotation across submap boundaries;
- gtsam_points
  `include/gtsam_points/factors/rotate_vector3_factor.hpp` and
  `src/gtsam_points/factors/integrated_matching_cost_factor.cpp` (MIT):
  analytic GTSAM chart Jacobians and direct Hessian-factor construction;
- GVINS `estimator/src/factor/imu_factor.h` and
  `estimator/src/factor/integration_base.h` (GPL-3.0, research context only):
  the coupled pose/velocity/accelerometer-bias/gyroscope-bias residual layout,
  fixed bias linearization centers, and analytic block Jacobians.

No source was copied. Unlike the reference systems, this adapter does not
reintegrate IMU measurements, add endpoint priors, infer covariance, or
reconstruct discarded raw factors. It consumes only the already-condensed,
checksum-bearing canonical rows. It remains source-private and is not wired
into the current public pose-only `GlobalGraph`.

`meridian_core` currently validates that the transition, frozen rows, lineage,
and input partition carry nonzero checksums, but it does not expose the
canonical checksum domain or a recomputation API for a complete condensed
transition. This adapter therefore rejects missing checksums and leaves
cryptographic content verification to the future checksum-verifying global
seal-cache boundary. It does not invent a second incompatible encoding.

## Loop-consensus context

`src/loop_consensus.cpp` is a Meridian-owned implementation informed by the
PCM/max-clique separation in the BSD-licensed `../slam-reference/Kimera-RPGO`
and by the deterministic tie-breaking and cycle-covariance guards at the named
legacy Meridian baseline. Meridian accepts only an explicit upstream
rank-aware cycle residual/covariance input, checks transitive ancestry through
a replaceable API, solves a complete component with deterministic exact
maximum clique, and defers the whole component when a resource bound is hit.
It does not copy either implementation, reconstruct missing chain covariance,
or fall back to a heuristic partial clique.

## Robust global-graph transaction context

`src/gnc_tls.cpp` and the robust loop transaction in `src/graph.cpp` are original Meridian
implementations informed by Yang et al., *Graduated Non-Convexity for Robust Spatial
Perception*, and by the BSD-licensed Kimera-RPGO known-inlier split in
`../slam-reference/Kimera-RPGO/src/RobustSolver.cpp`. The reference keeps odometry and special
factors at unit GNC weight and classifies loop closures. Meridian preserves that mathematical
separation but uses its own factor-dimension-normalized TLS controller and a disposable copy of
the complete bounded active graph on every solver call. Existing robust loops are reevaluated;
the mission gauge, adjacent-submap factors, and GNSS factors remain explicitly identified unit
inliers. Only binary TLS inliers are passed to the ordinary rank, condition, marginal-covariance,
and connectedness validation before one atomic graph revision is installed. A stale parent,
solver failure, all-outlier batch, or failed final validation leaves the committed state untouched.

## BSD-3-Clause implementation context

OKVIS2-X is available locally under `../slam-reference/OKVIS2-X` and is BSD-3-Clause.
The following files were inspected:

- `okvis_ceres/include/okvis/ceres/GpsErrorSynchronous.hpp`
- `okvis_ceres/src/GpsErrorSynchronous.cpp`
- `okvis_ceres/test/TestGpsErrorSynchronous.cpp`
- `okvis_ceres/test/TestEstimatorGpsError.cpp`

The important convention carried into Meridian is the synchronous antenna prediction:
the antenna phase centre is the IMU/sensor origin plus the rotated sensor-to-antenna lever
arm, followed by the world-to-GNSS/global transform. Meridian narrows that global transform
to the gravity-observable four degrees of freedom and exposes its Sophus-right Jacobians.
The code in `src/gnss.cpp` was written for Meridian's API and tangent conventions; it is not
a copy of the Ceres implementation.

`src/gnss_interpolation.cpp` implements the SYSTEM_SPECS binding for finalized
discrete trajectory brackets: shortest-branch SO(3) interpolation, cubic
Hermite translation using endpoint velocities, the declared 18-dimensional
right-local endpoint covariance order, and analytic antenna-position
Jacobians. Its formulas were independently derived and are checked against
central differences for every endpoint block; no continuous-time state is
introduced.

`src/gnss_fsm.cpp` implements the section 7.2 admission states and section
14.5 shadow-reacquisition boundary. GLIM's GNSS extension was read for the
global-layer ownership lesson; Meridian replaces direct unstateful admission
with receiver-metadata and covariance gates, NIS hysteresis, bounded quarantine
windows, and complete graph-batch shadow validation. The implementation is
original and has no API capable of feeding a correction into local odometry.

## Research context only (GPL-3.0)

GVINS is available locally under `../slam-reference/GVINS` and is GPL-3.0. The following
files were read as research context only:

- `estimator/src/initial/gnss_vi_initializer.h`
- `estimator/src/initial/gnss_vi_initializer.cpp`
- `estimator/src/estimator.cpp`, especially `processGNSS()` and `GNSSVIAlign()`

GVINS explicitly estimates the gravity-aligned local-to-ENU yaw, checks horizontal motion
excitation before alignment, refines the global anchor, rejects measurements without valid
tracking history, and freezes yaw when excitation disappears. Meridian uses those system
lessons but implements a different solution-level estimator: deterministic robust 4-DoF
alignment of timestamped antenna-position correspondences, with covariance, conditioning,
time-span and excitation diagnostics. The health monitor similarly requires a consistent
batch after a suspect period or outage, but its innovation-clustering state machine is an
original implementation and no GVINS code was copied.

## Geodesy and robust estimation

The WGS84 forward/inverse and ECEF/ENU equations are implemented directly from the public
WGS84 ellipsoid definition (`a = 6378137 m`, `1/f = 298.257223563`). Alignment uses one
deterministic path: pair-supported robust initial yaw, covariance-weighted median
translation, then Huber iteratively reweighted Gauss-Newton. No third-party geodesy or
robust-estimation source code is embedded in this package.
