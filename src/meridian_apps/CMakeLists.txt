cmake_minimum_required(VERSION 3.22)
project(meridian_apps LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
find_package(meridian_core REQUIRED)
find_package(meridian_ros REQUIRED)
find_package(rclcpp REQUIRED)
include(MeridianToolchain)

include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  rerun_sdk
  URL https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip
)
FetchContent_MakeAvailable(rerun_sdk)

add_executable(meridian_ingress
  src/meridian_ingress_main.cpp
  src/rerun_debug_sink.cpp
)
target_include_directories(meridian_ingress
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(meridian_ingress
  meridian_core::meridian_core
  meridian_ros::meridian_ros_ingress
  rerun_sdk
)
ament_target_dependencies(meridian_ingress rclcpp)
meridian_apply_warnings(meridian_ingress)

install(TARGETS meridian_ingress DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

ament_package()
