cmake_minimum_required(VERSION 3.22)
project(meridian_frontend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)
find_package(OpenMP)

add_library(meridian_frontend SHARED
  src/frontend_factory.cpp
  src/lio/voxel_grid_map.cpp
  src/lio/imu_tracker.cpp
  src/lio/scan_registration.cpp
  src/lio/lio_frontend.cpp
)
target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Internal estimator headers live under src/ and are not exported.
target_include_directories(meridian_frontend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(meridian_frontend PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_frontend)
target_link_libraries(meridian_frontend PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  Eigen3::Eigen
  Sophus::Sophus
)

# Parallel processing
if(OpenMP_CXX_FOUND)
  target_link_libraries(meridian_frontend PRIVATE OpenMP::OpenMP_CXX)
endif()

ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_common meridian_config meridian_debug meridian_calib
  Eigen3 Sophus
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_frontend
  EXPORT meridian_frontendTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  include(MeridianTesting)
  # Tests exercise internal estimator classes, so they see src/ directly.

  meridian_add_gtest(test_voxel_grid_map test/test_voxel_grid_map.cpp)
  meridian_add_gtest(test_imu_tracker test/test_imu_tracker.cpp)
  meridian_add_gtest(test_scan_registration test/test_scan_registration.cpp)
  # The end-to-end suite runs whole synthetic streams per test; give it headroom
  # over ctest's 60 s default.
  meridian_add_gtest(test_lio_frontend test/test_lio_frontend.cpp TIMEOUT 180)
  foreach(t test_voxel_grid_map test_imu_tracker test_scan_registration test_lio_frontend)
    target_link_libraries(${t} meridian_frontend)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()
endif()

ament_package()
