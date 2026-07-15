import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

# Centipede is a free, open RTK base-station network. NEAR selects the nearest live base
# rather than a named one, which is why the driver must publish GGA back to the caster
# (nmea.publish in sbg_config.yaml) -- without a position the caster cannot choose a base.
# NEAR4 serves MSM4 instead of MSM7: less bandwidth, for receivers that need it.
NTRIP_HOST = os.environ.get("NTRIP_HOST", "crtk.net")
# 80 rather than the conventional NTRIP 2101: 2101 times out from this network while 80
# is open, and the caster serves NTRIP/2.0 on both.
NTRIP_PORT = int(os.environ.get("NTRIP_PORT", "80"))
NTRIP_MOUNTPOINT = os.environ.get("NTRIP_MOUNTPOINT", "NEAR")
NTRIP_USERNAME = os.environ.get("NTRIP_USERNAME", "centipede")
NTRIP_PASSWORD = os.environ.get("NTRIP_PASSWORD", "centipede")


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
            }],
        ),
        # Not a colcon package, so it is run directly rather than through launch_ros.
        ExecuteProcess(
            cmd=["python3", "/rtk_monitor.py"],
            name="rtk_monitor",
            output="screen",
        ),
        # Identity placeholder so the IMU frame exists in the TF tree for preview.
        # Replace with the measured os_sensor<-sbg_imu extrinsic at calibration.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="sbg_static_tf",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "os_sensor", "--child-frame-id", "sbg_imu",
            ],
        ),
    ])
