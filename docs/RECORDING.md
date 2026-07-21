# Recording a dataset on the Jetson rig

How to capture a loss-free bag from the barracuda rig (Orin AGX + Ouster OS-1-128 +
SBG Ellipse-D + ZED 2i). For the Newer College benchmark data see `docs/DATASET.md`;
this doc is about recording our own.

The goal is **zero dropped messages**. Every setting below was measured, not assumed —
the before/after numbers are inline, from 30 s bags checked with `tools/check_bag.py`
against nominal rate. Baseline before any of it: 8.1/10 Hz on the LiDAR (19% of scans
lost) at ~216 MB/s offered load; after all of it, 0% loss on every topic.

## Rig as measured

| sensor | topic | rate | size/msg | stream |
| --- | --- | --- | --- | --- |
| Ouster OS-1-128 (`192.168.100.10`, PTP-synced) | `/ouster/points` | 10 Hz | 6.29 MB | 53 MB/s |
| " | `/ouster/imu` | 100 Hz | 0.32 KB | — |
| SBG Ellipse-D (`/dev/ttyUSB0`) | `/imu/data` | 200 Hz | small | — |
| " GNSS/RTK | `/imu/nav_sat_fix`, `/sbg/gps_pos` | 5 Hz | small | — |
| ZED 2i (USB3, UVC) | `/zed/image_raw` | 15 Hz | 10.97 MB | 165 MB/s |

Full set ≈ **216 MB/s ≈ 780 GB/h**. The NVMe sustains ~2.3 GB/s and has ~1.6 TB free,
so the disk is never the constraint — **capacity is**: roughly **2 h** of full-rate
recording. `record_bag.sh` refuses to start below 20 GB free.

## Record

```bash
# 1. Bring the rig up (ptp -> ouster -> sbg; add the camera with --profile camera)
docker compose -f compose.jetson.yaml --profile camera up -d ptp ouster sbg zed

# 2. Record. Bags land on the 2 TB SSD (/media/agx/ssd/bags), never the eMMC.
tools/record_bag.sh lio  campus      # LiDAR + IMU only    (~55 MB/s, ~200 GB/h)
tools/record_bag.sh full campus      # + camera/GNSS/RTK   (~216 MB/s, ~780 GB/h)
# Ctrl-C stops, finalizes, and prints a per-topic loss check.
```

The loss check must end in `PASS`. Anything else means the bag has real loss — fix it
and re-record rather than keeping it, because loss is not recoverable afterwards. Re-run
the check on any older bag with:

```bash
python3 tools/check_bag.py <bag>/bag_info.txt
```

## Replay and inspect

```bash
# The live drivers publish the same topic names as the bag -- play_bag.sh refuses to
# start while they are up, because a replay alongside them interleaves recorded and
# live messages on one topic and you end up inspecting neither.
docker compose -f compose.jetson.yaml --profile camera stop ouster sbg zed

tools/play_bag.sh                       # newest bag, loops at 1x
tools/play_bag.sh campus_20260717_1503  # a named bag, or an absolute path
RATE=0.5 tools/play_bag.sh              # half speed
LOOP=0 tools/play_bag.sh                # once through, then exit
```

**Foxglove is the practical viewer** and needs no setup — the `foxglove` service is
already running and picks the replay up off the DDS graph. From any machine on the LAN
(including a Mac, where RViz is not realistic):

> Foxglove → Open connection → Foxglove WebSocket → `ws://11.0.0.68:8765`

Set the 3D panel's fixed frame to `os_sensor` and add the `/ouster/points` topic.

### RViz

`meridian-rviz:humble` (`docker/jetson/rviz/`) carries rviz2 plus the rig's message
packages. RViz needs a real X display, which the rig does not have by default — the
Jetson's X server is only GDM's login screen, so `DISPLAY=:0` fails until someone is
physically logged into the desktop. Options, best first:

1. **Foxglove instead.** Works from anywhere, already running, handles the 6.3 MB clouds.
2. **A Linux box on the LAN.** Run rviz2 there with `ROS_DOMAIN_ID=0`; it joins over the
   profile's UDP fallback (shared memory is same-host only). Budget ~53 MB/s of network
   for the cloud stream — fine on gigabit, hopeless on WiFi.
3. **The Jetson's own monitor**, once logged into the desktop:
   ```bash
   xhost +local:docker
   docker run --rm --network host --ipc host \
     -e DISPLAY=:0 -e QT_X11_NO_MITSHM=1 \
     -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds.xml \
     -v ./docker/jetson/dds/fastdds_large.xml:/dds.xml:ro \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     meridian-rviz:humble
   ```
   `ssh -X` also works but tunnels OpenGL over the network; expect it to crawl on point
   clouds.

## The four things that make it lossless

Each of these was, on its own, enough to corrupt a capture silently — the bag still
looked valid, just short. In rough order of impact:

1. **The FastDDS profile** (`docker/jetson/dds/fastdds_large.xml`, 512 MB segment).
   FastDDS' 536 KB default segment cannot hold a 6.3 MB scan or an 11 MB stereo frame;
   those fall back to fragmented UDP loopback and drop under load — 19% of LiDAR scans
   lost. It must be loaded by **every participant**: compose wires it in via the
   `x-dds-env`/`x-dds-volume` anchors, and `record_bag.sh` mounts it into the recorder.
   A single container missing it costs messages on *both* sides.
2. **RELIABLE QoS on the Ouster** (`use_system_default_qos:=true`, now the compose
   default). The sensor-data default is BEST_EFFORT, which silently discards scans to a
   subscriber that falls behind: 5.5% lost, with nothing logged.
3. **The recorder runs as root.** The drivers are root and FastDDS creates its
   `/dev/shm` segments mode 0644 root-owned, so a `--user`-mapped recorder cannot map
   them: every topic from a root driver records **zero messages** while the bag still
   looks valid. The bag is chowned back to the caller afterwards.
4. **MAXN + `jetson_clocks`.** 30 W mode caps the CPU at 1.73 GHz and the ZED at
   14.1/15 fps; MAXN gives 2.2 GHz and the full 15.0 fps.

```bash
sudo nvpmodel -m 0     # MAXN. Persists across reboot.
sudo jetson_clocks     # Lock clocks. Does NOT persist -- re-run after every boot.
```

### The rest of the tuning

| constant | value | trades | effect |
| --- | --- | --- | --- |
| `proc_mask` (compose, ouster) | `PCL\|IMU\|TLM` | loses the range/signal/nearir/reflec images + LaserScan vs per-scan CPU | drops 5 derived topics nothing consumes and all reconstructible from the cloud offline; frees driver CPU for the writer. Restore `IMG\|PCL\|IMU\|SCAN\|TLM` to preview them. |
| `--max-cache-size` (`record_bag.sh`) | 512 MiB | RAM vs tolerance to a write stall | rosbag2's 100 MiB default is ~16 clouds at 6.3 MB/scan, so one SSD stall spills into dropped messages; 512 MiB absorbs a multi-second one. Not binding here (SSD sustains 2.3 GB/s vs 216 MB/s offered). |
| storage plugin (`record_bag.sh`) | `mcap`, uncompressed | disk footprint vs CPU and write stalls | sqlite3 wraps each message in a transaction against a single-writer B-tree and stalls at this rate. Compression is off deliberately — the SSD is 10x faster than the stream, so zstd would only spend CPU the drivers need. |

## Before a campaign

- `sudo jetson_clocks` — it is lost on reboot and nothing warns you.
- Check PTP has converged: `docker compose -f compose.jetson.yaml logs ptp | grep phc2sys | tail`.
  The `offset` should settle to tens of ns. It starts in the tens of µs and takes ~90 s;
  recording before it converges puts a drifting clock in the dataset.
- Check GNSS has a fix if the run needs RTK ground truth:
  `ros2 topic echo /imu/nav_sat_fix --once` — `status: -1` is **no fix** (normal indoors,
  and then `/ntrip_client/rtcm` stays empty because no GGA goes up to the caster).
- Do a 30 s throwaway recording and confirm `PASS` before the real run.

## /dev/shm hygiene

The 512 MB segments are real tmpfs pages, not sparse, and FastDDS does not remove them
when a container dies — so they accumulate across restarts (`/dev/shm` is 31 GB; a
day of `compose up/down` cycles can leave tens of GB of orphans, and DDS fails once it
fills). A reboot clears them. To reclaim without one, stop everything first — deleting a
live participant's segment breaks it:

```bash
docker compose -f compose.jetson.yaml --profile camera down
sudo rm -f /dev/shm/fastrtps_*
```

Check with `df -h /dev/shm` before a long run.

## Known gaps

- **The ZED is uncalibrated.** `/zed/camera_info` carries v4l2_camera's default (an
  identity-ish guess), not a real intrinsic calibration, and `/zed/image_raw` is a
  side-by-side stereo frame that no consumer splits yet. The frames are recorded raw
  (`yuv422_yuy2`, no CPU conversion) so a calibration can be applied offline — but a
  stereo calibration has to happen before the camera is usable for the visual stage.
- **`os_sensor -> zed_camera` is an identity placeholder** in `zed_uvc.launch.py`, not a
  measured extrinsic.
- The odometry config reads GNSS from `/gnss/fix`, which nothing publishes — the SBG
  driver puts it on `/imu/nav_sat_fix`. Irrelevant to recording (raw topics are captured
  either way) but it needs a remap before the GNSS stage runs live.
