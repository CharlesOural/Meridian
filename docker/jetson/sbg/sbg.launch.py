from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="sbg_driver",
            executable="sbg_device",
            name="sbg_device",
            output="screen",
            parameters=["/config/sbg_config.yaml"],
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
