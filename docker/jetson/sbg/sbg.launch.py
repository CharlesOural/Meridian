import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

# Centipede is a free, open RTK base-station network.
#
# A NAMED base, not NEAR. NEAR would auto-select the nearest live base, but it requires
# the rover's GGA in the HTTP request header, and ntrip_client only writes GGA to the
# socket after connecting -- which this caster ignores. Measured against crtk.net: a NEAR
# request with header-GGA streams ~13 kB in 8 s, the same request with socket-GGA returns
# only the 358-byte HTTP response and no RTCM ever follows, which is why the client loops
# on "received 0 bytes ... even though it said there was data available".
#
# So pick the base by hand. RTK error grows ~1 ppm with baseline, so anything under ~30 km
# is fine; list nearby ones from the sourcetable at http://crtk.net/ and see docs/RTK.md.
# The driver still publishes GGA (nmea.publish) -- harmless, and needed if NEAR ever works.
NTRIP_HOST = os.environ.get("NTRIP_HOST", "crtk.net")
# 80 rather than the conventional NTRIP 2101: 2101 times out from this network while 80
# is open, and the caster serves NTRIP/2.0 on both.
NTRIP_PORT = int(os.environ.get("NTRIP_PORT", "80"))
NTRIP_MOUNTPOINT = os.environ.get("NTRIP_MOUNTPOINT", "EVCC")
NTRIP_USERNAME = os.environ.get("NTRIP_USERNAME", "centipede")
NTRIP_PASSWORD = os.environ.get("NTRIP_PASSWORD", "centipede")
# Sent as the Ntrip-Version header. Left empty the client omits the header entirely and
# the caster may answer with chunked HTTP/1.1 framing that the client's hardcoded
# HTTP/1.0 request does not expect -- the socket reports data available and reads zero
# bytes, so it reconnects in a loop and no corrections ever arrive.
NTRIP_VERSION = os.environ.get("NTRIP_VERSION", "Ntrip/2.0")


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="sbg_driver",
            executable="sbg_device",
            name="sbg_device",
            output="screen",
            parameters=["/config/sbg_config.yaml"],
        ),
        # Feeds RTCM corrections to the INS. rtcm_message_package MUST be rtcm_msgs: this
        # node defaults to mavros_msgs, which the sbg driver does not subscribe to, and
        # the mismatch is silent -- corrections never arrive and the fix stays SINGLE.
        # Both nodes default to the ntrip_client namespace, so the topics line up.
        Node(
            package="ntrip_client",
            executable="ntrip_ros.py",
            name="ntrip_client",
            namespace="ntrip_client",
            output="screen",
            parameters=[{
                "host": NTRIP_HOST,
                "port": NTRIP_PORT,
                "mountpoint": NTRIP_MOUNTPOINT,
                "authenticate": True,
                "username": NTRIP_USERNAME,
                "password": NTRIP_PASSWORD,
                "rtcm_message_package": "rtcm_msgs",
                "ntrip_version": NTRIP_VERSION,
            }],
        ),
        # Not a colcon package, so it is run directly rather than through launch_ros.
        ExecuteProcess(
            cmd=["python3", "/rtk_monitor.py"],
            name="rtk_monitor",
            output="screen",
        ),
        # Measured extrinsic, IMU as the tree root: os_sensor pose expressed in sbg_imu
        # (parent sbg_imu, child os_sensor). Bench-measured on the Taurus/Mobilex mount;
        # roll ~pi + yaw ~pi is the LiDAR's ~180deg flip relative to the IMU. RPY is applied
        # Rz(yaw)Ry(pitch)Rx(roll), which is the convention static_transform_publisher uses.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="sbg_static_tf",
            arguments=[
                "--x", "-0.454687", "--y", "0.013142", "--z", "0.257027",
                "--roll", "3.14", "--pitch", "0.001", "--yaw", "3.043",
                "--frame-id", "sbg_imu", "--child-frame-id", "os_sensor",
            ],
        ),
    ])
