cmake_minimum_required(VERSION 3.22)
project(meridian_pipeline LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_time REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)
find_package(meridian_sensors REQUIRED)
find_package(meridian_preprocess REQUIRED)
find_package(meridian_frontend REQUIRED)
find_package(meridian_backend REQUIRED)
find_package(meridian_map REQUIRED)
find_package(meridian_place REQUIRED)

add_library(meridian_pipeline
  src/meridian_pipeline.cpp)

target_include_directories(meridian_pipeline PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_pipeline PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_pipeline)

target_link_libraries(meridian_pipeline PUBLIC
  meridian_common::meridian_common meridian_config::meridian_config meridian_time::meridian_time meridian_debug::meridian_debug
  meridian_calib::meridian_calib meridian_sensors::meridian_sensors meridian_preprocess::meridian_preprocess
  meridian_frontend::meridian_frontend meridian_backend::meridian_backend
  meridian_map::meridian_map
  meridian_place::meridian_place
  Eigen3::Eigen Sophus::Sophus)


install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_pipeline EXPORT meridian_pipelineTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_export_targets(meridian_pipelineTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config meridian_time meridian_debug
                          meridian_calib meridian_sensors meridian_preprocess
                          meridian_frontend meridian_backend meridian_map meridian_place
                          Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_bounded_queue test/test_bounded_queue.cpp)
  target_link_libraries(test_bounded_queue meridian_pipeline)
  meridian_add_gtest(test_pipeline test/test_pipeline.cpp)
  target_link_libraries(test_pipeline meridian_pipeline)
endif()

ament_package()
