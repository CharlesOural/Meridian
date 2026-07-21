#!/usr/bin/env python3
"""Verdict on GNSS/RTK quality, from a bag or from the live graph.

Reads /sbg/gps_pos, not /imu/nav_sat_fix: NavSatFix has no RTK enum, so the driver
collapses RTK_FIXED, RTK_FLOAT and SINGLE all onto STATUS_FIX and the standard topic
cannot answer whether RTK worked.

  tools/check_rtk.py /path/to/bag           # audit a recording
  tools/check_rtk.py --live                 # watch the running rig
  tools/check_rtk.py --live --duration 30   # sample, then report

Exits 0 if RTK held for --min-fixed-frac of the run, 1 otherwise, so it can gate a
recording in a script.
"""
import argparse
import math
import sys
from collections import Counter

FIX = {0: "NO_SOLUTION", 1: "UNKNOWN", 2: "SINGLE", 3: "DGPS", 4: "SBAS", 5: "OMNISTAR",
       6: "RTK_FLOAT", 7: "RTK_FIXED", 8: "PPP_FLOAT", 9: "PPP_INT", 10: "FIXED"}
SOL = {0: "SOL_COMPUTED", 1: "INSUFFICIENT_OBS", 2: "INTERNAL_ERROR", 3: "HEIGHT_LIMIT"}
RTK_FIXED, RTK_FLOAT, NO_SOLUTION = 7, 6, 0
DIFF_AGE_NA, SV_NA, BASE_NA = 0xFFFF, 0xFF, 0xFFFF

WGS84_A, WGS84_E2 = 6378137.0, (1 / 298.257223563) * (2 - 1 / 298.257223563)


def lla_to_enu(lat, lon, alt, origin):
    lat0, lon0, alt0 = origin

    def ecef(la, lo, al):
        la, lo = math.radians(la), math.radians(lo)
        s = math.sin(la)
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * s * s)
        return ((n + al) * math.cos(la) * math.cos(lo),
                (n + al) * math.cos(la) * math.sin(lo),
                (n * (1 - WGS84_E2) + al) * s)

    x, y, z = ecef(lat, lon, alt)
    x0, y0, z0 = ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    la0, lo0 = math.radians(lat0), math.radians(lon0)
    sl, cl, so, co = math.sin(la0), math.cos(la0), math.sin(lo0), math.cos(lo0)
    return (-so * dx + co * dy,
            -sl * co * dx - sl * so * dy + cl * dz,
            cl * co * dx + cl * so * dy + sl * dz)


def read_bag(path):
    import rclpy.serialization as ser
    import rosbag2_py
    from sbg_driver.msg import SbgGpsPos

    r = rosbag2_py.SequentialReader()
    # mcap is what the rig records; sqlite3 bags still open via the default probe
    try:
        r.open(rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
               rosbag2_py.ConverterOptions("", ""))
    except RuntimeError:
        r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
               rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=["/sbg/gps_pos", "/ntrip_client/rtcm"]))

    fixes, rtcm = [], 0
    while r.has_next():
        topic, data, t = r.read_next()
        if topic == "/ntrip_client/rtcm":
            rtcm += 1
        else:
            fixes.append((t * 1e-9, ser.deserialize_message(data, SbgGpsPos)))
    return fixes, rtcm


def read_live(duration, topic):
    import rclpy
    from rclpy.node import Node
    from sbg_driver.msg import SbgGpsPos

    rclpy.init()
    node = Node("check_rtk")
    fixes = []
    node.create_subscription(
        SbgGpsPos, topic,
        lambda m: fixes.append((node.get_clock().now().nanoseconds * 1e-9, m)), 50)

    end = node.get_clock().now().nanoseconds * 1e-9 + duration
    last = None
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < end:
        rclpy.spin_once(node, timeout_sec=0.2)
        if fixes and fixes[-1][1] is not last:
            last = fixes[-1][1]
            name = FIX.get(last.status.type, "?")
            age = "n/a" if last.diff_age == DIFF_AGE_NA else f"{last.diff_age*0.01:.1f}s"
            sv = "n/a" if last.num_sv_used == SV_NA else last.num_sv_used
            sys.stderr.write(
                f"\r  {name:11} H {last.position_accuracy.x:6.3f} m  age {age:>5}  "
                f"sats {sv}   ")
            sys.stderr.flush()
    sys.stderr.write("\r" + " " * 70 + "\r")
    node.destroy_node()
    rclpy.shutdown()
    return fixes, 0


def report(fixes, rtcm, args):
    if not fixes:
        print("  no /sbg/gps_pos messages -- is the sbg container up?")
        return 1

    dur = fixes[-1][0] - fixes[0][0]
    n = len(fixes)
    types = Counter(FIX.get(m.status.type, str(m.status.type)) for _, m in fixes)
    rate = n / dur if dur > 0 else 0.0

    print(f"  {n} fixes over {dur:.1f} s ({rate:.2f} Hz)"
          + (f", {rtcm} RTCM msgs ({rtcm/dur:.2f} Hz)" if rtcm else ""))
    print()
    for k, v in types.most_common():
        bar = "#" * int(40 * v / n)
        print(f"    {k:12} {v:5d}  {100.0*v/n:5.1f} %  {bar}")

    sols = Counter(SOL.get(m.status.status, "?") for _, m in fixes)
    if set(sols) - {"SOL_COMPUTED"}:
        print(f"  solution status  {dict(sols)}")

    rtk = [m for _, m in fixes if m.status.type == RTK_FIXED]
    if rtk:
        hs = [m.position_accuracy.x for m in rtk]
        vs = [m.position_accuracy.z for m in rtk]
        ages = [m.diff_age * 0.01 for m in rtk if m.diff_age != DIFF_AGE_NA]
        svs = [m.num_sv_used for m in rtk if m.num_sv_used != SV_NA]
        bases = {m.base_station_id for m in rtk if m.base_station_id != BASE_NA}
        print()
        print(f"  RTK_FIXED  H {sum(hs)/len(hs):.3f} m mean ({min(hs):.3f}-{max(hs):.3f})"
              f"   V {sum(vs)/len(vs):.3f} m mean ({min(vs):.3f}-{max(vs):.3f})")
        if ages:
            print(f"             corrections {sum(ages)/len(ages):.2f} s mean "
                  f"(max {max(ages):.2f})")
        if svs:
            print(f"             satellites {min(svs)}-{max(svs)}   base {bases or 'n/a'}")

    seq = [m.status.type for _, m in fixes]
    trans = [(FIX.get(a), FIX.get(b)) for a, b in zip(seq, seq[1:]) if a != b]
    if trans:
        print()
        print(f"  {len(trans)} fix-type transitions:")
        for a, b in trans[:6]:
            print(f"    {a} -> {b}")
        if len(trans) > 6:
            print(f"    ... and {len(trans)-6} more")

    pos = [m for _, m in fixes if m.status.type != NO_SOLUTION]
    scatter = None
    if pos:
        o = (pos[0].latitude, pos[0].longitude, pos[0].altitude + pos[0].undulation)
        enu = [lla_to_enu(m.latitude, m.longitude, m.altitude + m.undulation, o) for m in pos]
        es, ns = [p[0] for p in enu], [p[1] for p in enu]
        span = math.hypot(max(es) - min(es), max(ns) - min(ns))
        steps = [math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(enu, enu[1:])]
        print()
        print(f"  trajectory {span:.2f} m span, max {max(steps)*rate:.1f} m/s"
              if steps else f"  trajectory {span:.2f} m span")
        # Scatter is an accuracy estimate only when parked; under motion it is the path.
        if span < 0.5:
            me, mn = sum(es)/len(es), sum(ns)/len(ns)
            scatter = math.sqrt(sum((e-me)**2 + (nn-mn)**2 for e, nn in zip(es, ns))/len(es))
            print(f"  STATIC log: scatter {scatter:.3f} m")

    frac = types.get("RTK_FIXED", 0) / n
    print()
    ok = frac >= args.min_fixed_frac
    # A fix claiming centimetres while scattering decimetres has resolved its
    # ambiguities wrongly -- it reports success and is the dangerous failure.
    if scatter is not None and frac > 0.5 and scatter > args.scatter_max:
        print(f"  FAIL  RTK_FIXED claimed but {scatter:.3f} m static scatter "
              f"(> {args.scatter_max}) -- not trustworthy")
        return 1
    if ok:
        print(f"  PASS  RTK_FIXED for {100*frac:.1f}% of fixes "
              f"(>= {100*args.min_fixed_frac:.0f}%)")
        return 0
    print(f"  FAIL  RTK_FIXED for only {100*frac:.1f}% of fixes "
          f"(< {100*args.min_fixed_frac:.0f}%)")
    return 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", nargs="?", help="bag directory to audit")
    p.add_argument("--live", action="store_true", help="read the running graph instead")
    p.add_argument("--duration", type=float, default=15.0, help="live sample seconds")
    p.add_argument("--topic", default="/sbg/gps_pos")
    p.add_argument("--min-fixed-frac", type=float, default=0.95,
                   help="fraction of fixes that must be RTK_FIXED to pass")
    p.add_argument("--scatter-max", type=float, default=0.10,
                   help="max static scatter [m] tolerated while claiming RTK_FIXED")
    args = p.parse_args()

    if args.live:
        fixes, rtcm = read_live(args.duration, args.topic)
    elif args.bag:
        fixes, rtcm = read_bag(args.bag)
    else:
        p.error("give a bag path or --live")
    return report(fixes, rtcm, args)


if __name__ == "__main__":
    sys.exit(main())
