#!/usr/bin/env python3
"""Check a recorded bag's per-topic message counts against each sensor's nominal rate.

`ros2 bag record` reports success as long as it wrote *something*: the clouds publish
BEST_EFFORT, so a subscriber that cannot keep up silently loses messages and the bag
looks fine. This compares actual count against duration * expected_rate and flags the
gap, which is the only cheap way to catch loss while the rig is still set up.

Usage: check_bag.py <bag_info.txt> [expected_topic ...]

<bag_info.txt> is the output of `ros2 bag info`. Any expected topics listed after it are
additionally required to be present, which catches a driver that never published at all.
"""
import re
import sys

# Nominal publish rate per topic, Hz, as measured on the rig. Latched/aperiodic topics map
# to None: they are reported but never rate-checked, since a count of 1 for
# /ouster/metadata is correct and /tf_static has no meaningful rate.
EXPECTED_HZ = {
    "/ouster/points": 10.0,
    "/ouster/imu": 100.0,
    "/imu/data": 200.0,
    "/imu/mag": 25.0,          # the SBG publishes mag at 25 Hz, not the IMU's 200
    "/zed/image_raw": 15.0,    # 4416x1242 is a 15 fps mode; raise if ZED_WIDTH changes
    "/zed/camera_info": 15.0,
    "/imu/nav_sat_fix": 5.0,
    "/sbg/gps_pos": 5.0,
    "/sbg/gps_vel": 5.0,
    "/sbg/imu_data": 200.0,
    "/sbg/ekf_quat": 200.0,
    "/ouster/metadata": None,
    "/tf_static": None,
    "/tf": None,
    "/diagnostics": None,
    "/ntrip_client/rtcm": None,
    "/ntrip_client/nmea": None,
    "/sbg/status": None,
    "/sbg/utc_time": None,
    "/imu/utc_ref": None,
}

# Topics that can legitimately record nothing, so an empty one is reported but is not a
# failure. A checker that cries wolf on a normal indoor run teaches the operator to ignore
# it, which costs more than it catches:
#   ntrip/utc  -- no GNSS fix means no GGA uplink, so no corrections and no GPS time
#   /tf        -- only the odometry node publishes dynamic transforms; a raw sensor
#                 capture has none
OPTIONAL = {
    "/ntrip_client/rtcm",
    "/ntrip_client/nmea",
    "/imu/utc_ref",
    "/sbg/utc_time",
    "/tf",
}

# A tolerance this wide is not a quality bar -- it is a loss alarm. Start/stop truncate
# partial messages at both ends and the rig's rates are nominal, not disciplined, so a
# few percent is normal; >5% missing means messages were actually dropped.
TOLERANCE = 0.05


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    text = open(sys.argv[1]).read()

    m = re.search(r"Duration:\s+([\d.]+)s", text)
    if not m:
        print("check_bag: no Duration in bag info -- cannot rate-check")
        return 2
    duration = float(m.group(1))

    # Topic lines look like:
    #   Topic: /ouster/points | Type: sensor_msgs/msg/PointCloud2 | Count: 300 | ...
    rows = re.findall(r"Topic:\s+(\S+)\s+\|.*?\|\s+Count:\s+(\d+)", text)
    if not rows:
        print("check_bag: no topics found in bag info")
        return 2

    # A topic whose driver never came up is absent from the bag entirely rather than
    # present with a zero count, so scanning only the recorded rows would report a bag
    # with no LiDAR as clean. Anything named on the command line but missing here is a
    # failure, not an omission.
    recorded = {t for t, _ in rows}
    missing = [t for t in sys.argv[2:] if t not in recorded and t not in OPTIONAL]

    print(f"\n=== Loss check ({duration:.1f}s) ===")
    worst = 0.0
    empty = []
    for topic in missing:
        print(f"  {topic:<24} ABSENT -- topic never appeared in the bag")
    for topic, count_s in sorted(rows):
        count = int(count_s)
        if count == 0:
            if topic in OPTIONAL:
                print(f"  {topic:<24} empty (optional -- not a failure)")
            else:
                empty.append(topic)
                print(f"  {topic:<24} EMPTY -- no messages recorded")
            continue
        hz = EXPECTED_HZ.get(topic)
        if hz is None:
            print(f"  {topic:<24} {count:>7} msgs   ({count/duration:5.1f} Hz, not checked)")
            continue
        expected = duration * hz
        loss = max(0.0, 1.0 - count / expected)
        worst = max(worst, loss)
        flag = "OK  " if loss <= TOLERANCE else "LOSS"
        print(
            f"  {topic:<24} {count:>7} msgs   ({count/duration:5.1f} Hz, "
            f"expect {hz:5.1f}) {flag} {loss*100:4.1f}% missing"
        )

    print()
    failed = False
    if missing:
        print(f"FAIL: {len(missing)} topic(s) absent from the bag: {', '.join(missing)}")
        failed = True
    if empty:
        print(f"FAIL: {len(empty)} topic(s) recorded nothing: {', '.join(empty)}")
        failed = True
    if worst > TOLERANCE:
        print(f"FAIL: up to {worst*100:.1f}% of messages missing -- bag has real loss.")
        failed = True
    if failed:
        return 1
    print("PASS: all rate-checked topics within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
