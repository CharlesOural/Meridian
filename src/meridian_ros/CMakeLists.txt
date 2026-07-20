cmake_minimum_required(VERSION 3.22)
project(meridian_ros LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
find_package(meridian_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

add_library(meridian_ros_conversion
  src/conversion.cpp
)
add_library(meridian::ros_conversion ALIAS meridian_ros_conversion)
target_compile_features(meridian_ros_conversion PUBLIC cxx_std_20)
target_include_directories(meridian_ros_conversion
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(meridian_ros_conversion meridian_core::meridian_core)
ament_target_dependencies(meridian_ros_conversion sensor_msgs)
meridian_apply_warnings(meridian_ros_conversion)

add_library(meridian_ros_ingress
  src/sensor_ingress_node.cpp
)
add_library(meridian::ros_ingress ALIAS meridian_ros_ingress)
target_compile_features(meridian_ros_ingress PUBLIC cxx_std_20)
target_include_directories(meridian_ros_ingress
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(meridian_ros_ingress
  PUBLIC
    meridian_ros_conversion
    meridian_core::meridian_core
)
ament_target_dependencies(meridian_ros_ingress
  PUBLIC
    rclcpp
    sensor_msgs
)
meridian_apply_warnings(meridian_ros_ingress)

install(
  TARGETS meridian_ros_conversion meridian_ros_ingress
  EXPORT export_meridian_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  meridian_add_gtest(test_conversion test/test_conversion.cpp)
  target_link_libraries(test_conversion meridian_ros_conversion meridian_core::meridian_core)
  ament_target_dependencies(test_conversion sensor_msgs)

  meridian_add_gtest(test_sensor_ingress_node test/test_sensor_ingress_node.cpp)
  target_link_libraries(test_sensor_ingress_node meridian_ros_ingress meridian_core::meridian_core)
  ament_target_dependencies(test_sensor_ingress_node
    rclcpp
    sensor_msgs
  )
endif()

ament_export_targets(export_meridian_ros HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_core
  rclcpp
  sensor_msgs
)
ament_package()
