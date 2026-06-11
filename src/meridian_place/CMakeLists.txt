cmake_minimum_required(VERSION 3.22)
project(meridian_place LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)

add_library(meridian_place
  src/keyframe_store.cpp
  src/sc_descriptor.cpp
  src/submap_accumulator.cpp
  src/loop_detector_factory.cpp)

target_include_directories(meridian_place PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_place PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_place)

target_link_libraries(meridian_place PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  Eigen3::Eigen Sophus::Sophus)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_place EXPORT meridian_placeTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

ament_export_targets(meridian_placeTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_keyframe_store test/test_keyframe_store.cpp)
  target_link_libraries(test_keyframe_store meridian_place)
  # The descriptor / submap engines are private (src/) headers; tests reach them directly.
  meridian_add_gtest(test_sc_descriptor test/test_sc_descriptor.cpp)
  target_include_directories(test_sc_descriptor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_sc_descriptor meridian_place)
  meridian_add_gtest(test_submap_accumulator test/test_submap_accumulator.cpp)
  target_include_directories(test_submap_accumulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_submap_accumulator meridian_place)
endif()

ament_package()
