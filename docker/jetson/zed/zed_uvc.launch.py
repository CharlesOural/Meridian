from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ZED 2i UVC stream is a single side-by-side stereo frame (left|right).
        # YUYV is the only device format; convert to rgb8 so any viewer renders it.
        Node(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name="zed_uvc",
            output="screen",
            parameters=[{
                "video_device": "/dev/video0",
                "pixel_format": "YUYV",
                "image_size": [1344, 376],
                "output_encoding": "rgb8",
                "camera_frame_id": "zed_camera",
            }],
            remappings=[
                ("/image_raw", "/zed/image_raw"),
                ("/camera_info", "/zed/camera_info"),
            ],
        ),
        # Identity placeholder so the camera frame exists in the TF tree for preview.
        # Replace with the measured os_sensor<-zed_camera extrinsic at calibration.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="zed_static_tf",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--roll", "0", "--pitch", "0", "--yaw", "0",
                "--frame-id", "os_sensor", "--child-frame-id", "zed_camera",
            ],
        ),
    ])
