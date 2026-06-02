# Meridian — Development & Evaluation Dataset

> Decision doc. Requirement (from the project owner): one dataset that exercises **LiDAR + camera + IMU + GNSS simultaneously**, so the *full* fused pipeline runs end‑to‑end ("do it all at once, not split"). Ground truth required for objective drift/ATE evaluation.

## Why one all‑modality dataset matters

If you develop LiDAR‑IMU on dataset A, vision on dataset B, and GNSS on dataset C, you never test the thing that actually matters for a tightly‑coupled system: the **joint** behaviour — cross‑modality observability hand‑off, time‑sync interaction, degeneracy recovery where one modality saves another. A single synchronized LiDAR+camera+IMU+GNSS dataset lets every phase (Phase 1 LiDAR‑IMU through Phase 6 full LIVO+GNSS) be validated on the *same* sequences, so improvements are measured against a fixed baseline rather than confounded by changing data.

## Candidates (all carry LiDAR + camera + IMU + GNSS with ground truth)

| Dataset | LiDAR | Camera | IMU | GNSS | Ground truth | Platform | Notes for Meridian |
|---|---|---|---|---|---|---|---|
| **FusionPortable** (HKUST) | **Ouster OS1‑128** | 20 Hz stereo (pinhole) + stereo event | 200 Hz | 10 Hz | prior‑map ICP (indoor) + GNSS (outdoor) | handheld / quadruped / UGV / vehicle | **Best rig match** — Ouster like the planned rig, pinhole stereo ideal for direct‑photometric visual, **hardware‑synced**. Campus scale. |
| **M2DGR** (SJTU) | Velodyne VLP‑32C | 6 fish‑eye + 1 sky RGB + VI‑sensor | yes + nav‑grade | consumer GNSS + **RTK** | motion capture / 3D laser tracker / RTK | ground robot | The de‑facto multi‑sensor ground‑robot SLAM **benchmark** (RA‑L'21/ICRA'22). Best for comparable published numbers + GNSS‑factor testing. Caveat: fish‑eye cams are awkward for direct visual (need rectification/pinhole crop). |
| **UrbanNav** (PolyU) | LiDAR | camera | yes | GNSS (raw) | SPAN‑CPT INS/RTK | vehicle, urban canyon | The **GNSS‑degraded stress test** — multipath, urban canyon. Use to harden the switchable‑GNSS / spoofing logic. |
| **MARS‑LVIG** (HKU‑MARS) | Livox Avia (solid‑state) | global‑shutter RGB | yes | GNSS raw | RTK | aerial (UAV) | **Same lab as FAST‑LIO2** (Fu Zhang group). LiDAR‑Visual‑Inertial‑GNSS by design, IJRR 2024. Aerial + non‑repetitive Livox scan differs from spinning Ouster, but a high‑quality, on‑ecosystem cross‑check. |

## Recommendation

- **Primary (development):** **FusionPortable.** It matches the target hardware (Ouster OS1‑128), gives pinhole stereo that the FAST‑LIVO2‑style direct visual front‑end wants, is hardware‑time‑synced (so you validate the estimator before you own the sync problem), and spans ground platforms close to the tactical use case. Every Meridian phase can run on the same sequences.
- **Co‑primary (benchmark + GNSS):** **M2DGR.** Use it for comparable, widely‑published accuracy numbers and for exercising the GNSS factor with real RTK ground truth. Treat the fish‑eye cameras as a "rectify‑then‑use" case (good stress for the camera abstraction).
- **Robustness gate:** **UrbanNav** for GNSS‑denied/multipath, and the **Hilti‑Oxford** sequences (LiDAR+camera+IMU, no GNSS) for geometric degeneracy — run these before declaring any phase "done".

Start on **FusionPortable**; bring in **M2DGR** the moment the GNSS factor lands (Phase 6).

## How it plugs into the build

- **Format / ROS 2:** these ship as ROS 1 bags. Convert with `rosbags` (the Python tool, `rosbags-convert`) or `ros1_bridge`. Meridian's sensor‑source layer (L0) reads the converted ROS 2 bags through the same `ISensorSource` interface used live — so "replay a bag" and "run on the robot" are the same code path. This is also the offline test harness for the ROS‑agnostic core.
- **Calibration:** all four datasets ship Kalibr‑format intrinsics + extrinsics + IMU noise. These seed the offline calibration stage; Meridian's back‑end then refines extrinsics online (so the dataset calibration is a *prior*, not a hard constant — itself a useful test of the online‑refinement path).
- **Evaluation:** trajectory error via **evo** (`evo_ape`, `evo_rpe`) against the provided ground truth. Track translational drift % over distance and absolute trajectory error (ATE). Map quality (later) via cloud‑to‑cloud distance against the survey/prior map where provided.

## Per‑phase test mapping

| Phase | Dataset slice | Metric / acceptance |
|---|---|---|
| 1 — LiDAR‑IMU odometry | FusionPortable indoor + outdoor | drift % vs GT; no divergence |
| 2 — back‑end + loop closure | FusionPortable sequences with revisits | ATE drop after loop closure; no false‑loop corruption |
| 3 — dense map + mesh | FusionPortable campus | mesh vs prior‑map cloud‑to‑cloud error |
| 4 — robustness | Hilti degenerate + UrbanNav canyon | survives degeneracy/GNSS‑denial without divergence |
| 5 — CT front‑end | FusionPortable fast‑motion + M2DGR | ≥ parity with v1 ATE, better on aggressive motion |
| 6 — full LIVO + GNSS | M2DGR (RTK) + FusionPortable | global consistency vs RTK; graceful GNSS drop/spoof handling |

## Sources

- [M2DGR](https://github.com/SJTU-ViSYS/M2DGR) ([paper](https://arxiv.org/pdf/2112.13659))
- [FusionPortable](https://arxiv.org/pdf/2208.11865)
- [UrbanNav](https://github.com/IPNL-POLYU/UrbanNavDataset) ([paper](https://navi.ion.org/content/70/4/navi.602))
- [MARS‑LVIG](https://journals.sagepub.com/doi/abs/10.1177/02783649241227968)
- [KAIST Complex Urban](https://irap.kaist.ac.kr/dataset/download.html) (LiDAR+stereo+IMU+GPS, large‑scale driving — alternative)
- evo evaluation toolkit: `pip install evo`
