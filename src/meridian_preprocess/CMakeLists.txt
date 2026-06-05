cmake_minimum_required(VERSION 3.22)
project(meridian_preprocess LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
set(MERIDIAN_NEED_OPENCV ON)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus, OpenCV

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)

add_library(meridian_preprocess
  src/lidar_validity_filter.cpp
  src/imu_init.cpp
  src/imu_only_deskew.cpp
  src/gnss_gate.cpp
  src/camera_preprocess.cpp)

target_include_directories(meridian_preprocess PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_preprocess PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_preprocess)

target_link_libraries(meridian_preprocess PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  ${OpenCV_LIBS})


ament_export_targets(meridian_preprocessTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config meridian_debug meridian_calib OpenCV)

install(TARGETS meridian_preprocess EXPORT meridian_preprocessTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_lidar_validity_filter test/test_lidar_validity_filter.cpp)
  target_link_libraries(test_lidar_validity_filter meridian_preprocess)
  meridian_add_gtest(test_imu_init test/test_imu_init.cpp)
  target_link_libraries(test_imu_init meridian_preprocess)
  meridian_add_gtest(test_imu_only_deskew test/test_imu_only_deskew.cpp)
  target_link_libraries(test_imu_only_deskew meridian_preprocess)
  meridian_add_gtest(test_gnss_gate test/test_gnss_gate.cpp)
  target_link_libraries(test_gnss_gate meridian_preprocess)
  meridian_add_gtest(test_camera_preprocess test/test_camera_preprocess.cpp)
  target_link_libraries(test_camera_preprocess meridian_preprocess)
endif()

ament_package()
