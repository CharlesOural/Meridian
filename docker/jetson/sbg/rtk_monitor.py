#!/usr/bin/env python3
"""Turn /sbg/gps_pos into something a human can judge at a glance in Foxglove.

sensor_msgs/NavSatFix cannot express RTK: the driver collapses RTK_INT, RTK_FLOAT and
SINGLE all onto STATUS_FIX, so the standard topic cannot answer "is RTK actually
working". Only the SBG-native SbgGpsPos carries status.type, the correction age and the
base station id, so this node reads that and republishes:

  /rtk/path    nav_msgs/Path            trajectory in local ENU metres
  /rtk/track   visualization_msgs/Marker  same points, coloured by fix type
  /diagnostics diagnostic_msgs/DiagnosticArray  decoded status + a verdict

Degrees are useless for judging centimetres, so everything is projected into a local ENU
tangent plane anchored on the first fix with a solution -- the same construction the
back-end uses for its datum. The frame is self-rooted (rtk_enu -> rtk_antenna) and is
deliberately NOT tied to the SLAM map frame: the two differ by an unknown yaw until the
back-end locks its datum, so overlaying them here would draw a misalignment that is not
real.
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from sbg_driver.msg import SbgGpsPos
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

NO_SOLUTION, UNKNOWN_TYPE, SINGLE, PSRDIFF, SBAS, OMNISTAR = 0, 1, 2, 3, 4, 5
RTK_FLOAT, RTK_INT, PPP_FLOAT, PPP_INT, FIXED = 6, 7, 8, 9, 10

FIX_NAMES = {
    NO_SOLUTION: "NO_SOLUTION", UNKNOWN_TYPE: "UNKNOWN", SINGLE: "SINGLE",
    PSRDIFF: "DGPS", SBAS: "SBAS", OMNISTAR: "OMNISTAR", RTK_FLOAT: "RTK_FLOAT",
    RTK_INT: "RTK_FIXED", PPP_FLOAT: "PPP_FLOAT", PPP_INT: "PPP_INT", FIXED: "FIXED",
}

# Green only for a true integer-ambiguity fix; float RTK is decimetre-class and must not
# look like success. Anything without corrections is red.
FIX_COLORS = {
    RTK_INT: (0.10, 0.85, 0.25), RTK_FLOAT: (1.00, 0.60, 0.00),
    PSRDIFF: (1.00, 0.85, 0.10), SBAS: (1.00, 0.85, 0.10),
    SINGLE: (0.95, 0.20, 0.20),
}
DEFAULT_COLOR = (0.55, 0.55, 0.55)

# The receiver reports "not available" as an all-ones field rather than omitting it, so
# these must be decoded or the panel shows 255 satellites and 655 s of correction age
# when it means "nothing yet".
SV_NA = 0xFF
DIFF_AGE_NA = 0xFFFF
BASE_ID_NA = 0xFFFF

SOLUTION_STATUS = {0: "SOL_COMPUTED", 1: "INSUFFICIENT_OBS", 2: "INTERNAL_ERROR", 3: "HEIGHT_LIMIT"}
IFM_STATUS = {0: "ERROR", 1: "UNKNOWN", 2: "CLEAN", 3: "MITIGATED", 4: "CRITICAL"}
SPOOF_STATUS = {0: "ERROR", 1: "UNKNOWN", 2: "CLEAN", 3: "SINGLE_METHOD", 4: "CONFIRMED"}


def lla_to_enu(lat_deg, lon_deg, alt_m, origin):
    """WGS84 geodetic -> local ENU metres about `origin` (lat, lon, alt)."""
    lat0, lon0, alt0 = origin
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    lat0r, lon0r = math.radians(lat0), math.radians(lon0)

    def ecef(latr, lonr, alt):
        s = math.sin(latr)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s * s)
        return ((n + alt) * math.cos(latr) * math.cos(lonr),
                (n + alt) * math.cos(latr) * math.sin(lonr),
                (n * (1.0 - WGS84_E2) + alt) * s)

    x, y, z = ecef(lat, lon, alt_m)
    x0, y0, z0 = ecef(lat0r, lon0r, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0

    sl, cl = math.sin(lat0r), math.cos(lat0r)
    so, co = math.sin(lon0r), math.cos(lon0r)
    return (-so * dx + co * dy,
            -sl * co * dx - sl * so * dy + cl * dz,
            cl * co * dx + cl * so * dy + sl * dz)


class RtkMonitor(Node):
    def __init__(self):
        super().__init__("rtk_monitor")

        self.declare_parameter("gps_pos_topic", "/sbg/gps_pos")
        self.declare_parameter("frame_id", "rtk_enu")
        self.declare_parameter("max_points", 20000)
        # A fix claiming RTK_FIXED while scattering more than a few centimetres has
        # resolved its ambiguities wrongly -- the dangerous failure, since it still
        # reports success. Judged only while stationary, where scatter is all error.
        self.declare_parameter("static_window", 50)
        self.declare_parameter("static_drift_max", 0.30)
        self.declare_parameter("rtk_scatter_max", 0.10)
        # Corrections older than this mean the NTRIP stream has stalled and the solution
        # is coasting back toward metre-class accuracy while still labelled RTK.
        self.declare_parameter("diff_age_max", 10.0)

        self.frame_id = self.get_parameter("frame_id").value
        self.max_points = int(self.get_parameter("max_points").value)

        self.origin = None
        self.points = []
        self.colors = []
        self.poses = []
        self.recent = deque(maxlen=int(self.get_parameter("static_window").value))
        self.stamps = deque(maxlen=64)
        self.counts = {}
        self.total = 0

        self.pub_path = self.create_publisher(Path, "/rtk/path", 1)
        self.pub_track = self.create_publisher(Marker, "/rtk/track", 1)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf = TransformBroadcaster(self)

        self.create_subscription(
            SbgGpsPos, self.get_parameter("gps_pos_topic").value, self.on_fix, 10)
        self.create_timer(1.0, self.publish_diagnostics)

        self.last = None
        self.get_logger().info(
            f"rtk_monitor: {self.get_parameter('gps_pos_topic').value} -> /rtk/path, "
            f"/rtk/track, /diagnostics (frame {self.frame_id})")

    def on_fix(self, msg: SbgGpsPos):
        self.last = msg
        self.total += 1
        name = FIX_NAMES.get(msg.status.type, f"type_{msg.status.type}")
        self.counts[name] = self.counts.get(name, 0) + 1
        self.stamps.append(self.get_clock().now().nanoseconds * 1e-9)

        if msg.status.type == NO_SOLUTION:
            return

        # SbgGpsPos.altitude is above MSL; undulation lifts it to the ellipsoid, matching
        # what the driver puts in NavSatFix and what the back-end expects.
        alt = msg.altitude + msg.undulation

        if self.origin is None:
            self.origin = (msg.latitude, msg.longitude, alt)
            self.get_logger().info(
                f"ENU origin set at {msg.latitude:.7f}, {msg.longitude:.7f}, {alt:.2f} m "
                f"({name})")

        e, n, u = lla_to_enu(msg.latitude, msg.longitude, alt, self.origin)
        self.recent.append((e, n, u))
        self.append_point(msg, e, n, u)
        self.publish_track(msg, e, n, u)

    def append_point(self, msg, e, n, u):
        # Drop the oldest half at the cap rather than per-point, so a long run degrades
        # into a coarser record instead of silently becoming a short one.
        if len(self.points) >= self.max_points:
            half = len(self.points) // 2
            del self.points[:half]
            del self.colors[:half]
            del self.poses[:half]

        p = Point(x=e, y=n, z=u)
        self.points.append(p)
        r, g, b = FIX_COLORS.get(msg.status.type, DEFAULT_COLOR)
        self.colors.append(ColorRGBA(r=r, g=g, b=b, a=1.0))

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose.position = p
        pose.pose.orientation.w = 1.0
        self.poses.append(pose)

    def publish_track(self, msg, e, n, u):
        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = self.frame_id
        path.poses = self.poses
        self.pub_path.publish(path)

        marker = Marker()
        marker.header = path.header
        marker.ns = "rtk"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.pose.orientation.w = 1.0
        marker.points = self.points
        marker.colors = self.colors
        self.pub_track.publish(marker)

        # Roots the ENU frame in TF so the 3D panel has something to render against, and
        # marks where the antenna is right now.
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = "rtk_antenna"
        t.transform.translation.x = e
        t.transform.translation.y = n
        t.transform.translation.z = u
        t.transform.rotation.w = 1.0
        self.tf.sendTransform(t)

    def measured_rate(self):
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def static_scatter(self):
        """Horizontal scatter over the recent window, or None if the antenna moved.

        Returns (scatter_m, drift_m). Scatter estimates accuracy only while stationary;
        under motion the spread is the trajectory, not error.

        Motion is judged by drift -- the distance between the mean of the window's first
        half and that of its second half -- not by how far the points spread. Spread
        cannot be used: a wrongly-resolved RTK fix scatters widely while going nowhere, so
        a spread-based gate would call it "moving" and skip the one check meant to catch
        it. Jitter has no trend and cancels between the halves; real motion does not.
        """
        if len(self.recent) < self.recent.maxlen:
            return None
        es = [p[0] for p in self.recent]
        ns = [p[1] for p in self.recent]

        half = len(es) // 2
        m1e, m1n = sum(es[:half]) / half, sum(ns[:half]) / half
        m2e, m2n = sum(es[half:]) / (len(es) - half), sum(ns[half:]) / (len(ns) - half)
        drift = math.hypot(m2e - m1e, m2n - m1n)
        if drift > float(self.get_parameter("static_drift_max").value):
            return None

        me, mn = sum(es) / len(es), sum(ns) / len(ns)
        var = sum((e - me) ** 2 + (n - mn) ** 2 for e, n in zip(es, ns)) / len(es)
        return math.sqrt(var), drift

    def publish_diagnostics(self):
        st = DiagnosticStatus(name="rtk_monitor: GNSS/RTK", hardware_id="sbg_ellipse_d")

        if self.last is None:
            st.level = DiagnosticStatus.ERROR
            st.message = "no SbgGpsPos received"
            self.pub_diag.publish(
                DiagnosticArray(header=self._stamp(), status=[st]))
            return

        m = self.last
        fix = m.status.type
        name = FIX_NAMES.get(fix, f"type_{fix}")
        rate = self.measured_rate()
        diff_age = None if m.diff_age == DIFF_AGE_NA else m.diff_age * 0.01
        scatter = self.static_scatter()
        solved = fix != NO_SOLUTION

        # Reported independently: the receiver can give a real "0 used" while reporting
        # tracked as not-available, and collapsing both to n/a hides that it is trying.
        used = "n/a" if m.num_sv_used == SV_NA else str(m.num_sv_used)
        tracked = "n/a" if m.num_sv_tracked == SV_NA else str(m.num_sv_tracked)
        sats = f"{used} used, {tracked} tracked"
        # Reported accuracy is meaningless without a solution, and comes through as a
        # huge sentinel rather than zero.
        accuracy = (f"{m.position_accuracy.x:.3f} / {m.position_accuracy.y:.3f} / "
                    f"{m.position_accuracy.z:.3f}") if solved else "n/a"

        kv = [
            KeyValue(key="fix type", value=f"{name} ({fix})"),
            KeyValue(key="solution status",
                     value=SOLUTION_STATUS.get(m.status.status, str(m.status.status))),
            KeyValue(key="satellites used", value=sats),
            KeyValue(key="accuracy E/N/U [m]", value=accuracy),
            KeyValue(key="correction age [s]",
                     value="n/a (no corrections)" if diff_age is None else f"{diff_age:.2f}"),
            KeyValue(key="base station id",
                     value="n/a" if m.base_station_id == BASE_ID_NA else str(m.base_station_id)),
            KeyValue(key="rate [Hz]", value=f"{rate:.2f}"),
            KeyValue(key="interference", value=IFM_STATUS.get(m.status.ifm, "?")),
            KeyValue(key="spoofing", value=SPOOF_STATUS.get(m.status.spoofing, "?")),
            KeyValue(key="fixes seen", value=f"{self.total} " + str(self.counts)),
            KeyValue(key="static scatter [m]",
                     value=f"{scatter[0]:.3f} (drift {scatter[1]:.3f})" if scatter
                     else ("moving" if self.recent else "n/a (no positioned fix)")),
        ]
        st.values = kv

        rtk = fix in (RTK_INT, RTK_FLOAT)
        corrections_stale = (rtk and diff_age is not None
                             and diff_age > float(self.get_parameter("diff_age_max").value))
        scatter_bad = (scatter is not None and fix == RTK_INT
                       and scatter[0] > float(self.get_parameter("rtk_scatter_max").value))

        if fix == NO_SOLUTION:
            st.level = DiagnosticStatus.ERROR
            st.message = f"no GNSS solution ({SOLUTION_STATUS.get(m.status.status, '?')})"
        elif scatter_bad:
            # Claims centimetres, scatters decimetres: wrong ambiguity resolution.
            st.level = DiagnosticStatus.ERROR
            st.message = f"RTK_FIXED claimed but static scatter {scatter[0]:.2f} m -- not trustworthy"
        elif corrections_stale:
            st.level = DiagnosticStatus.WARN
            st.message = f"{name} but corrections {diff_age:.1f} s old -- NTRIP stream stalled?"
        elif fix == RTK_INT:
            st.level = DiagnosticStatus.OK
            age = "age n/a" if diff_age is None else f"corrections {diff_age:.1f} s old"
            st.message = f"RTK FIXED, {age}, base {m.base_station_id}"
        elif fix == RTK_FLOAT:
            st.level = DiagnosticStatus.WARN
            st.message = "RTK FLOAT -- decimetre class, ambiguities not resolved"
        else:
            st.level = DiagnosticStatus.WARN
            st.message = f"{name} -- no RTK corrections applied"

        self.pub_diag.publish(DiagnosticArray(header=self._stamp(), status=[st]))

    def _stamp(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        return h


def main():
    rclpy.init()
    node = RtkMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
