cmake_minimum_required(VERSION 3.22)
project(meridian_calib LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)

add_library(meridian_calib
  src/calibration_from_config.cpp
  src/calibration_store.cpp
  src/camera_model.cpp)

target_include_directories(meridian_calib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_calib PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_calib)

target_link_libraries(meridian_calib PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  Eigen3::Eigen Sophus::Sophus)


install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_calib EXPORT meridian_calibTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

ament_export_targets(meridian_calibTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_calibration_store test/test_calibration_store.cpp)
  target_link_libraries(test_calibration_store meridian_calib)
  meridian_add_gtest(test_camera_model test/test_camera_model.cpp)
  target_link_libraries(test_camera_model meridian_calib)
endif()

ament_package()
