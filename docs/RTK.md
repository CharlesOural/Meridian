# RTK on the Jetson rig

Centimetre-accurate GNSS on the SBG Ellipse-D, using free RTK corrections from the
[Centipede](https://docs.centipede.fr) network. `docker compose -f compose.jetson.yaml
up -d sbg` brings up everything; this file covers the two things that are **not**
automatic and what to do when RTK does not engage.

## What runs

```
Centipede base station (~17 km away)
      │ RTCM3 over NTRIP
crtk.net:80
      │
      ▼
ntrip_client ──/ntrip_client/rtcm──▶ sbg_driver ──serial──▶ Ellipse-D
              (rtcm_msgs/Message)                            │ RTK solved here,
                                                             │ in the receiver
                                    ┌────────────────────────┘
                                    ▼
                     /sbg/gps_pos ──▶ rtk_monitor ──▶ /rtk/path, /rtk/track,
                     /imu/nav_sat_fix                 /diagnostics
```

The RTK computation happens **inside the Ellipse's GNSS receiver**. Nothing here solves
carrier-phase ambiguities; the containers only carry correction bytes to the unit and
report what it decides.

## The one thing you must change when the rig moves: the base station

`NTRIP_MOUNTPOINT` in `compose.jetson.yaml` names a **specific** Centipede base. RTK
error grows about 1 ppm with baseline, so a base under ~30 km is fine and one in another
region is useless.

Centipede offers a `NEAR` mountpoint that auto-selects the nearest base, which would
avoid this entirely — but it does not work with this client. `NEAR` requires the rover's
GGA position in the **HTTP request header**, and `ntrip_client` only writes GGA to the
socket after connecting, which the caster ignores. Measured against `crtk.net`:

| request | bytes in 8 s |
|---|---|
| `HTTP/1.0`, no version header, GGA over socket (what `ntrip_client` sends) | 358 (headers only, no RTCM) |
| `HTTP/1.0`, `Ntrip-Version: Ntrip/2.0`, GGA in header | 12831 |

The symptom is `ntrip_client` logging `Reconnecting because we received 0 bytes from the
socket even though it said there was data available` every ~60 s, and no RTCM. It is not
a network fault.

To pick a base for a new site, list the sourcetable and sort by distance:

```bash
curl -s http://crtk.net/ > /tmp/sourcetable.txt

# nearest bases to LAT/LON, with a rough km distance
awk -F';' -v lat=48.711 -v lon=2.219 '/^STR;/ && $10 != "" {
  dlat=($10-lat)*111.32; dlon=($11-lon)*73.4;   # 73.4 km/deg lon at ~48N
  d=sqrt(dlat*dlat+dlon*dlon);
  if (d<45) printf "  %-6s %-26s %6.1f km\n", $2, $3, d
}' /tmp/sourcetable.txt | sort -k3 -n | head
```

Bases go up and down without notice, so confirm the one you pick actually streams:

```bash
curl -s -m 8 -u centipede:centipede http://crtk.net/EVCC | wc -c   # expect thousands
```

Then set it — no rebuild, the launch file is bind-mounted:

```bash
NTRIP_MOUNTPOINT=XXXX docker compose -f compose.jetson.yaml up -d sbg
```

`NTRIP_PORT` defaults to **80**, not the conventional NTRIP 2101: 2101 times out from
this site's network while 80 is open, and the caster serves NTRIP/2.0 on both. On a
network that allows 2101, either works.

## The other thing that is not automatic: sky

The receiver needs a clear view. Indoors it streams at 5 Hz forever with
`INSUFFICIENT_OBS` and zero satellites — healthy, just blind. That state is
indistinguishable in software from a disconnected antenna.

## Reading the verdict

Foxglove (`ws://<jetson>:8765`) → **Diagnostics** panel. Or from the shell:

```bash
docker exec meridian-sbg bash -lc \
  'source /opt/ros/humble/setup.bash && source /opt/sbg_ws/install/setup.bash &&
   ros2 topic echo --once /diagnostics'
```

| message | level | meaning |
|---|---|---|
| `RTK FIXED, corrections N s old` | OK | ~2 cm. Integer ambiguities resolved. |
| `RTK FLOAT` | WARN | ~20 cm. Corrections arriving, ambiguities unresolved. Often transient. |
| `SINGLE -- no RTK corrections applied` | WARN | ~1-2 m. Corrections are not reaching the receiver. |
| `no GNSS solution` | ERROR | No satellites. Sky or antenna. |
| `corrections N s old -- NTRIP stream stalled?` | WARN | Was RTK, stream died, now coasting. |
| `RTK_FIXED claimed but static scatter N m` | ERROR | The receiver is lying — wrong ambiguity resolution. |

That last one is the reason this monitor exists: a wrongly-resolved fix reports success
while being decimetres off. It is only checked while stationary, where scatter is all
error. Park for ~10 s (50 fixes at 5 Hz) and `static scatter` should be a couple of
centimetres for a genuine fix.

Other panels: **Map** on `/imu/nav_sat_fix` (OpenStreetMap, needs internet on the
viewing machine); **3D** on `/rtk/track` in frame `rtk_enu`, coloured green/orange/red
for fixed/float/single, which shows *where* RTK drops out along a route.

## Why the monitor exists at all

`sensor_msgs/NavSatFix` cannot express RTK. The driver maps `RTK_INT`, `RTK_FLOAT` and
`SINGLE` all onto `STATUS_FIX` (`message_wrapper.cpp`), so a 2 cm fix and a 2 m fix are
indistinguishable on `/imu/nav_sat_fix` — the topic Meridian consumes. Only the
SBG-native `/sbg/gps_pos` carries `status.type`, `diff_age` and `base_station_id`. Any
honest answer to "is RTK valid?" has to read that topic.

The covariance on `NavSatFix` *is* honest and does tighten under RTK, which is why the
back-end can still weight fixes correctly despite the lost label.

## Expected bring-up sequence

Outdoors, from cold, watch `/diagnostics` walk:

```
no GNSS solution        →  SINGLE (~1 m)  →  RTK FLOAT (~20 cm)  →  RTK FIXED (~1.5 cm)
   satellites appear        first fix          RTCM applied          ambiguities resolved
```

Re-acquisition after a restart takes seconds, not minutes — the receiver keeps its
ephemeris.

## Device configuration

See the header of `docker/jetson/sbg/sbg_config.yaml`. In short: with `confWithRos: true`
that file is a **complete image** of the device's stored settings, not a patch. Any key
absent from it is written to flash as the driver's compiled-in default and the unit
reboots. Read the device before editing it:

```bash
docker run --rm --device /dev/ttyUSB0 --group-add dialout \
  --entrypoint sbg_probe meridian-sbg:humble /dev/ttyUSB0 921600
```

**Stop the `sbg` service first.** A serial port is not exclusive on Linux, so the probe
will happily open `/dev/ttyUSB0` while the driver is streaming and the two then steal
each other's bytes — both sides log `SBG_INVALID_FRAME` and read garbage. It recovers
when the probe exits, but do not do it to a live session:

```bash
docker compose -f compose.jetson.yaml down sbg    # then probe, then bring it back up
```

`sbg_probe` never writes: it issues only `_GET` commands. A correctly-matched config also
writes nothing on startup — if the log says `Settings saved and device rebooted`,
something in the file disagrees with the device and you should find out what before
shipping it.
