cmake_minimum_required(VERSION 3.22)
project(meridian_debug LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)

add_library(meridian_debug
  src/telemetry.cpp
  src/log.cpp)

target_include_directories(meridian_debug PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_debug PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_debug)

target_link_libraries(meridian_debug PUBLIC
  meridian_common::meridian_common
  Eigen3::Eigen Sophus::Sophus)


install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_debug EXPORT meridian_debugTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_export_targets(meridian_debugTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_telemetry test/test_telemetry.cpp)
  target_link_libraries(test_telemetry meridian_debug)
  meridian_add_gtest(test_sinks test/test_sinks.cpp)
  target_link_libraries(test_sinks meridian_debug)
endif()

ament_package()
