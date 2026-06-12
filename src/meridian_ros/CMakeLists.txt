cmake_minimum_required(VERSION 3.22)
project(meridian_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(meridian_msgs REQUIRED)
find_package(meridian_pipeline REQUIRED)
find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_sensors REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)

# msg <-> core translation + the ROS telemetry sink. The only code that sees both
# worlds; -Werror is relaxed here because middleware headers own their warnings.
add_library(meridian_ros_support
  src/conversions/ros2core.cpp
  src/conversions/core2ros.cpp
  src/debug/ros_telemetry_sink.cpp)
target_include_directories(meridian_ros_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(meridian_ros_support PUBLIC cxx_std_20)
target_link_libraries(meridian_ros_support meridian_pipeline::meridian_pipeline
  opencv_core opencv_imgcodecs)
ament_target_dependencies(meridian_ros_support
  rclcpp builtin_interfaces sensor_msgs nav_msgs geometry_msgs std_msgs
  visualization_msgs meridian_msgs)

add_executable(odometry_node src/odometry_node.cpp)
target_link_libraries(odometry_node meridian_ros_support)
ament_target_dependencies(odometry_node
  rclcpp tf2_ros geometry_msgs sensor_msgs meridian_msgs)

find_package(rosbag2_cpp REQUIRED)
add_executable(replay_runner src/replay_runner.cpp)
target_link_libraries(replay_runner meridian_ros_support meridian_debug::meridian_debug)
ament_target_dependencies(replay_runner rclcpp sensor_msgs rosbag2_cpp tf2_ros geometry_msgs)

install(TARGETS odometry_node replay_runner DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

ament_package()
