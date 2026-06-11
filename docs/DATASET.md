# Newer College (multi-cam 2021) — benchmark data setup

Reproducible setup for the Newer College Dataset, 2021 collection (Ouster OS0-128
+ Alphasense multi-camera, hardware-synchronized, cm-accurate ICP ground truth).
This is the benchmark set: three tuning sequences + one **holdout**. No GNSS on
this rig (`sensors.gnss.enable: false`).

**Current scope: LIO benchmark.** All Newer College cameras are fisheye
(Kalibr `equidistant`), which the front-end's `CameraModel` deliberately rejects
(pinhole+radtan only; rectification is an unwired seam in `CameraPreprocessor`),
so the visual stage auto-disables on these bags (`frontend/visual/disabled`,
`cam_valid=0`). The configs still carry the full camera calibration — the moment
equidistant projection or the rectify map lands, the visual stage lights up with
no config change. Until then, ATE numbers here are LiDAR-inertial only.

The `bags/` tree is **not** committed; the three Meridian configs and the import
tools are. Follow this doc to rebuild the identical local layout.

| sequence | collection | role | raw size |
|---|---|---|---|
| quad-easy | 1 (newer college) | bring-up baseline — gentle motion, open quad | 10 GB |
| math-medium | 3 (maths institute) | mid-band tuning — mixed open/building | 8.5 GB |
| park (bags 0–2 of 8) | 2 (newer college) | vegetation/parkland tuning — the off-road analog | 28 GB |
| quad-hard | 1 | **HOLDOUT — do not tune on it.** Blind validation only | 9.5 GB |

## 1. Download

Everything lives under the dataset's public Drive folder
([root](https://drive.google.com/drive/folders/15lTH5osZzZlDpcW7oXfR_2t8TNssNARS) →
`2021-ouster-os0-128-alphasense`). Per-file quota errors ("download quota
exceeded") are per-file and reset on a ~24 h rolling window; copying a file into
your own Drive and downloading the copy bypasses it.

Bags (download individually):

- [quad-easy `2021-07-01-10-37-38-quad-easy.bag`](https://drive.google.com/file/d/1hF2h83E1THbFAvs7wpR6ORmrscIHxKMo/view)
- [quad-hard `2021-07-01-11-35-14_0-quad-hard.bag`](https://drive.google.com/file/d/1ss6KPSTZ4CRS7uHAMgqnd4GQ6tKEEiZD/view)
- [math-medium `2021-04-07-13-55-18-math-medium.bag`](https://drive.google.com/file/d/1IOq2e8Nfx79YFauBHCgdBSW9Ur5710GD/view)
- [park_0 `2021-11-30-17-09-49_0-park.bag`](https://drive.google.com/file/d/1KZo-gPVQTMJ4hRaiaqV3hfuVNaONScDp/view)
- [park_1 `2021-11-30-17-13-13_1-park.bag`](https://drive.google.com/file/d/1eGVPwFSaG0M2M7Lci6IBjKQrEf1uqtVn/view)
- [park_2 `2021-11-30-17-16-38_2-park.bag`](https://drive.google.com/file/d/1nhuoH0OcLbovbXkq3eW6whk6TIKk2SEu/view)

Calibration + ground truth (take the whole folders; megabytes):

- root files: [`beam_intrinsics_os0-128.json`](https://drive.google.com/file/d/1bRdAg8gpXaOYBOMrO6cxpPYbBux4H982/view),
  [`os_imu_lidar_transforms.yaml`](https://drive.google.com/file/d/1jgmKmqTzZ-oF2PVUwoAEAmaXlqrt_rhq/view)
- collection 1: [`ground_truth`](https://drive.google.com/drive/folders/1LDbFd3Yzeg1AvLabbez3Qa90hepFwki1),
  [`cam_calibration/cam0-1`](https://drive.google.com/drive/folders/1qOYSm1h2m299BhKWSiKviN5JDSRfzPwL)
- collection 2: [`ground_truth`](https://drive.google.com/drive/folders/1PhtFzlf9sS6jmmze2NfhNvKlzlM4ncxR),
  [`cam_calibration/cam0-1`](https://drive.google.com/drive/folders/1ug8fqXc6OzCFs-_aHCU3d8ZH0sDw9gUM)
- collection 3: [`ground_truth`](https://drive.google.com/drive/folders/1aId52hIyj4Scm14O03cknEjqT-S5SSSa)
  (includes the `maths-institute.ply` prior map),
  [`cam_calibration/cam0-1`](https://drive.google.com/drive/folders/1rHOMq3W311XeHTccIxtmajMtcLP0Boqy)

Optional, for later (same collection folders): park bags 3–7, cloister, math-easy/hard,
quad-medium, stairs. Keep them under `ros1-extra/` unconverted.

## 2. Target layout

```
bags/newer-college/
├── calib/
│   ├── beam_intrinsics_os0-128.json     # OS0-128 beam table (only needed for raw packets)
│   ├── os_imu_lidar_transforms.yaml     # rig chain: os_sensor/os_imu <-> alphasense imu <-> base
│   ├── collection1/cam0-1/…             # Kalibr camchain-imucam + reports (July 2021)
│   ├── collection2/cam0-1/…             # (November 2021 — the rig was recalibrated!)
│   └── collection3/cam0-1/…             # (March/April 2021)
├── gt/
│   ├── tum/        # dataset TUM GT, base frame; park clipped to bags 0-2 (full = *-full8.csv)
│   ├── tum_asimu/  # SAME poses re-expressed in the Alphasense-IMU frame — USE THIS FOR ATE
│   └── state/      # dataset full-state CSVs (sec,nsec,...), base frame
├── prior_map/maths-institute.ply
├── quad-easy/      # ROS2 bags (sqlite3), ready for ros2 bag play
├── math-medium/
├── park/           # park bags 0-2 merged into ONE bag (one continuous run)
├── quad-hard/      # HOLDOUT
└── ros1-extra/     # raw ROS1 extras, unconverted (park 3-7, cloister, math-easy/hard)
```

The prepared ROS2 bags contain: `/os_cloud_node/points` (PointCloud2, 10 Hz,
os_sensor frame, per-point `t`), `/alphasense_driver_ros/imu` (200 Hz),
`/alphasense_driver_ros/cam{0,1,3,4}/compressed` (mono8 JPEG, ~30 Hz),
`/os_cloud_node/imu` (Ouster internal IMU, unused).

## 3. Preparation pipeline (inside the `meridian` distrobox)

```bash
cd ~/Meridian/bags/newer-college
# 1) ROS1 -> ROS2 (park: three sources merge chronologically into one bag)
rosbags-convert --src ros1/2021-07-01-10-37-38-quad-easy.bag   --dst ros2-raw/quad-easy   --dst-typestore ros2_humble
rosbags-convert --src ros1/2021-04-07-13-55-18-math-medium.bag --dst ros2-raw/math-medium --dst-typestore ros2_humble
rosbags-convert --src ros1/2021-11-30-17-09-49_0-park.bag ros1/2021-11-30-17-13-13_1-park.bag \
                --src ros1/2021-11-30-17-16-38_2-park.bag      --dst ros2-raw/park        --dst-typestore ros2_humble
rosbags-convert --src ros1/2021-07-01-11-35-14_0-quad-hard.bag --dst ros2-raw/quad-hard   --dst-typestore ros2_humble

# 2) one-time post-conversion pass (ALREADY APPLIED to the bags in bags/): verify
#    per-point timestamps (NC's column clock is CLEAN — 0 scans needed repair on all
#    four bags), strip ouster_ros/PacketMsg (~GBs, no Humble typesupport), and rewrite
#    metadata v9 -> v5 (Humble's player refuses v9). The throwaway script that did this
#    has been removed; redoing it from scratch means: copy PointCloud2/Imu/CompressedImage
#    topics into a fresh bag and patch metadata version — no timestamp surgery needed.
rm -rf ros2-raw ros1   # keep ros1-extra

# 3) clip the park GT to the bags-0..2 window (the full-run GT covers all 8 bags).
#    Window from park/metadata.yaml: start 1638292236.000718678, duration 571.053319292 s.
#    (Already-clipped files keep the originals as gt/{tum,state}/gt-nc-park-full8.csv.)

# 4) re-express GT into the estimation frame (GT is the *base* frame; /meridian/odom is
#    the Alphasense-IMU frame; the constant ~7.6 cm body offset does NOT wash out under
#    Umeyama alignment):
cd gt && mkdir -p tum_asimu
for f in gt-nc-quad-easy gt-nc-quad-hard gt-nc-math-medium gt-nc-park; do
  python3 ~/Meridian/tools/ncd_gt_to_imu_frame.py tum/$f.csv tum_asimu/$f.csv
done
```

**Camera-included quad-easy bag.** The benchmark bags above keep only LiDAR + IMU
(the camera was dropped at conversion since the visual stage was off). To exercise the
camera path (L1 undistortion, image telemetry) regenerate a bag that also carries cam0
straight from the ROS1 source — `--include-topic` does the topic filtering in one pass,
no post-conversion script:

```bash
cd ~/Meridian/bags/newer-college
rosbags-convert --src ros1/2021-07-01-10-37-38-quad-easy.bag --dst quad-easy-cam \
  --src-typestore ros1_noetic --dst-typestore ros2_humble --dst-version 5 \
  --include-topic /os_cloud_node/points /alphasense_driver_ros/imu \
                  /alphasense_driver_ros/cam0/compressed
```

The other three cams (`cam{1,3,4}`) are available the same way; cam0 is the one the
config points at. View both feeds live: `replay_runner <config> bags/newer-college/quad-easy-cam
<out>.tum 0 --realtime 1 --viz` then open `/meridian/image/preprocess_camera_{raw,intensity}`
in Foxglove.

## 4. Configs (committed)

One config per collection — the camera was recalibrated between collections
(intrinsics and the FPGA cam→imu timeshift differ), so park must NOT run with the
quad calibration:

- `src/meridian_ros/config/newer-college-quad.yaml` — quad-easy / quad-hard (collection 1)
- `src/meridian_ros/config/newer-college-park.yaml` — park (collection 2)
- `src/meridian_ros/config/newer-college-math.yaml` — math-medium (collection 3)

Calibration blocks are machine-written by `tools/import_ncd.py` (LiDAR extrinsic
from `os_sensor_to_as_imu`, camera intrinsics/extrinsic/timeshift from the Kalibr
camchain, IMU noise densities squared into `cov_*`). To regenerate:

```bash
python3 tools/import_ncd.py \
    --camchain "bags/newer-college/calib/collection1/cam0-1/_2021-07-01-13-36-53-cam0-1-camchain-imucam.yaml" \
    --config src/meridian_ros/config/newer-college-quad.yaml
# collection2 camchain -> newer-college-park.yaml, collection3 -> newer-college-math.yaml
```

Known caveats baked into the configs:

- **The dataset's Kalibr gyro noise is wrong — do not re-import it.** The dataset
  file duplicates the accel density into the gyro field (both 0.019). A real
  BMI085 gyro is ~2e-3 rad/s/√Hz (OKVIS2's Hilti-2022 config for the same IMU);
  importing the duplicated value de-weights the gyro ~90× in variance, window
  rotation floats, and the estimator diverges at km scale (measured: quad-easy
  ATE 5018 m → **0.066 m** after the fix). The gyro **random walk** is equally
  unusable: the dataset's 2.66e-4 rad/s²/√Hz lets the bias absorb LiDAR misfit
  within one window — it pegs the `bias.gyr_max` box during brisk spans and the
  heading diverges past ~90 s (measured: quad-easy full bag **15.99 m → 0.193 m**
  rmse at 4e-6, i.e. `b_gyr_cov: 1.6e-11`). `import_ncd.py` defaults to the
  correct values; the accel numbers are kept from the dataset.
- **camera.time_offset_ms (+1.8…+2.0 ms)** comes from the Kalibr session. Same
  FPGA sync path as the recordings so it should transfer, but A/B against 0
  before trusting sub-centimeter ATE deltas (Kalibr session timeshifts have
  failed to transfer to recordings before).
- `preprocess.blind: 1.0` (operator's body is in the scan) and
  `preprocess.det_range: 50.0` (OS0 range) differ from the ledger's earlier
  baselines — see the OPTIMIZE.md rows.

## 5. Running the benchmark

```bash
# inside the distrobox, after colcon build + source install/setup.bash
CONFIG=src/meridian_ros/config/newer-college-quad.yaml BAG=bags/newer-college/quad-easy \
  tools/run_bag_headless.sh /tmp/nc_quad_easy
python3 tools/eval_ate.py bags/newer-college/gt/tum_asimu/gt-nc-quad-easy.csv /tmp/nc_quad_easy/traj_tum.txt
```

Same pattern for `math-medium` (math config) and `park` (park config). ATE must
be evaluated against **`gt/tum_asimu/`**, not `gt/tum/`.

**Holdout discipline:** `quad-hard` is for blind validation after a tuning round
converges on the other three. If a change is tuned *on* quad-hard, it stops
measuring generalization — record any quad-hard run in OPTIMIZE.md as a
validation result, never as a tuning input.

## 6. Provenance / sanity facts (verified during setup)

- Per-point `t` (uint32, ns): zero-based per sweep, monotonic, ~97.7 µs column
  cadence, ~100 ms span — **0 scans needed repair** (quad-easy 1991, math-medium
  1770, park 5711, quad-hard 1880).
- Clouds are stamped in the **os_sensor** frame (so `lidar.extrinsic_*` =
  `os_sensor_to_as_imu` directly).
- The transforms file quaternions are **(qx, qy, qz, qw)** — qx first. The
  os→Alphasense rotation is a 180° flip about X, not identity.
- IMU stream is **200 Hz** (the Kalibr file says 400 — that's the calibration
  bag, not these recordings).
- GT: ICP against a Leica BLK360 prior map, <1 cm, 10 Hz, **base frame**
  ([dataset GT page](https://ori-drs.github.io/newer-college-dataset/ground-truth/)).
- Dataset page: [Newer College multi-cam](https://ori-drs.github.io/newer-college-dataset/multi-cam/),
  paper: [arXiv 2112.08854](https://arxiv.org/abs/2112.08854). License CC BY-NC-SA 4.0 —
  non-commercial benchmark use.
