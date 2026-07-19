cmake_minimum_required(VERSION 3.22)
project(meridian_ros VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(meridian_core REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgcodecs)
find_package(yaml-cpp REQUIRED)

add_library(meridian_ros
  src/newer_college_calibration.cpp
  src/pipeline_diagnostics.cpp
  src/sensor_conversions.cpp
)
add_library(meridian::ros ALIAS meridian_ros)
target_include_directories(meridian_ros
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(meridian_ros PUBLIC
  meridian_core
  builtin_interfaces
  diagnostic_msgs
  rclcpp
  sensor_msgs
)
target_link_libraries(meridian_ros PUBLIC ${OpenCV_LIBRARIES} yaml-cpp)
target_include_directories(meridian_ros SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
meridian_apply_warnings(meridian_ros)

install(TARGETS meridian_ros
  EXPORT export_meridian_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_ros HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_core
  builtin_interfaces
  diagnostic_msgs
  rclcpp
  sensor_msgs
  OpenCV
  yaml-cpp
)

meridian_add_gtest(test_sensor_conversions test/test_sensor_conversions.cpp)
if(TARGET test_sensor_conversions)
  target_link_libraries(test_sensor_conversions meridian_ros)
endif()

meridian_add_gtest(test_newer_college_calibration
  test/test_newer_college_calibration.cpp
)
if(TARGET test_newer_college_calibration)
  target_link_libraries(test_newer_college_calibration meridian_ros)
endif()

meridian_add_gtest(test_pipeline_diagnostics
  test/test_pipeline_diagnostics.cpp
)
if(TARGET test_pipeline_diagnostics)
  target_link_libraries(test_pipeline_diagnostics meridian_ros)
endif()

ament_package()
