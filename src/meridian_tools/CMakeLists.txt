cmake_minimum_required(VERSION 3.22)
project(meridian_tools VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(meridian_core REQUIRED)
find_package(meridian_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(meridian_tools
  src/bag_replay.cpp
)
add_library(meridian::tools ALIAS meridian_tools)
target_include_directories(meridian_tools
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(meridian_tools
  PUBLIC
    meridian_core
    meridian_ros
    rclcpp
    rosbag2_cpp
    rosbag2_storage
    sensor_msgs
)
meridian_apply_warnings(meridian_tools)

install(TARGETS meridian_tools
  EXPORT export_meridian_tools
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_tools HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_core
  meridian_ros
  rclcpp
  rosbag2_cpp
  rosbag2_storage
  sensor_msgs
)

meridian_add_gtest(test_bag_replay test/test_bag_replay.cpp)
if(TARGET test_bag_replay)
  target_link_libraries(test_bag_replay meridian_tools)
endif()

meridian_add_gtest(test_scheduled_bag_replay
  test/test_scheduled_bag_replay.cpp
)
if(TARGET test_scheduled_bag_replay)
  target_link_libraries(test_scheduled_bag_replay meridian_tools)
endif()

ament_package()
