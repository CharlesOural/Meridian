cmake_minimum_required(VERSION 3.22)
project(meridian_sensors LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)
find_package(meridian_time REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_config REQUIRED)

add_library(meridian_sensors
  src/source_base.cpp
  src/sources.cpp
  src/aggregator.cpp)

target_include_directories(meridian_sensors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(meridian_sensors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(meridian_sensors PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_sensors)

target_link_libraries(meridian_sensors PUBLIC
  meridian_common::meridian_common
  meridian_time::meridian_time
  meridian_config::meridian_config
  Eigen3::Eigen)
target_link_libraries(meridian_sensors PRIVATE
  meridian_debug::meridian_debug)


ament_export_targets(meridian_sensorsTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_time meridian_config meridian_debug Eigen3)

install(TARGETS meridian_sensors EXPORT meridian_sensorsTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  include(MeridianTesting)

  meridian_add_gtest(test_aggregator test/test_aggregator.cpp)
  target_link_libraries(test_aggregator meridian_sensors)

  meridian_add_gtest(test_sources test/test_sources.cpp)
  target_link_libraries(test_sources meridian_sensors)
endif()

ament_package()
