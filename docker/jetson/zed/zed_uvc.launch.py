import os

from launch import LaunchDescription
from launch_ros.actions import Node

# ZED 2i UVC modes (one side-by-side stereo frame, left|right). Per-eye resolution is
# half the width; YUYV is the device's only format, so bytes/frame = width * height * 2.
#   4416x1242 @15    2208x1242/eye   10.97 MB/frame   165 MB/s
#   3840x1080 @30    1920x1080/eye    8.29 MB/frame   249 MB/s
#   2560x720  @60    1280x720/eye     3.69 MB/frame   221 MB/s
#   1344x376  @100    672x376/eye     1.01 MB/frame   101 MB/s
WIDTH = int(os.environ.get("ZED_WIDTH", "4416"))
HEIGHT = int(os.environ.get("ZED_HEIGHT", "1242"))


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="zed_uvc",
            output="screen",
            parameters=[{
                "video_device": "/dev/video0",
                "pixel_format": "YUYV",
                "image_size": [WIDTH, HEIGHT],
                # Passthrough: publish the device's YUYV bytes untouched. Requesting rgb8
                # makes v4l2_camera convert on the CPU, which cannot keep up at these
                # resolutions -- it caps 2560x720 at ~24 of 30 fps and drops the rest.
                # Colour conversion is the consumer's job; the dataset stores what the
                # sensor sent.
                "output_encoding": "yuv422_yuy2",
                "camera_frame_id": "zed_camera",
            }],
            remappings=[
                ("/image_raw", "/zed/image_raw"),
                ("/camera_info", "/zed/camera_info"),
            ],
        ),
        # Measured extrinsic: zed_camera_link (ZED mounting-screw base) expressed in sbg_imu
        # (parent sbg_imu, child zed_camera). Bench-measured on the Taurus/Mobilex mount.
        # This is the camera base, not an optical center; the left-optical offset is applied
        # downstream from the calibration. RPY applied Rz(yaw)Ry(pitch)Rx(roll).
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="zed_static_tf",
            arguments=[
                "--x", "0.399036", "--y", "-0.015032", "--z", "-0.213000",
                "--roll", "3.14", "--pitch", "0.002", "--yaw", "-0.010",
                "--frame-id", "sbg_imu", "--child-frame-id", "zed_camera",
            ],
        ),
    ])
