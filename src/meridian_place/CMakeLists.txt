cmake_minimum_required(VERSION 3.22)
project(meridian_place LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
set(MERIDIAN_NEED_SMALL_GICP ON)  # Stage-C geometric verification
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus, small_gicp::small_gicp

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)

add_library(meridian_place
  src/keyframe_store.cpp
  src/sc_descriptor.cpp
  src/submap_accumulator.cpp
  src/gicp_verify.cpp
  src/loop_cov.cpp
  src/pcm_self_test.cpp
  src/hierarchical_loop_detector.cpp
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
# small_gicp is an implementation detail of the Stage-C verifier (private src/ header), so
# it is not exported on the public interface.
target_link_libraries(meridian_place PRIVATE small_gicp::small_gicp)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_place EXPORT meridian_placeTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

ament_export_targets(meridian_placeTargets HAS_LIBRARY_TARGET)
# small_gicp is a private link dependency, but a static library propagates it as a link
# requirement, so downstream packages must be able to resolve the imported target.
ament_export_dependencies(meridian_common meridian_config Eigen3 Sophus small_gicp)

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
  meridian_add_gtest(test_gicp_verify test/test_gicp_verify.cpp)
  target_include_directories(test_gicp_verify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_gicp_verify meridian_place small_gicp::small_gicp)
  meridian_add_gtest(test_loop_cov test/test_loop_cov.cpp)
  target_include_directories(test_loop_cov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_loop_cov meridian_place)
  meridian_add_gtest(test_pcm_self_test test/test_pcm_self_test.cpp)
  target_include_directories(test_pcm_self_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_pcm_self_test meridian_place)
  meridian_add_gtest(test_loop_detector_e2e test/test_loop_detector_e2e.cpp)
  target_include_directories(test_loop_detector_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_loop_detector_e2e meridian_place)
  meridian_add_gtest(test_determinism test/test_determinism.cpp)
  target_include_directories(test_determinism PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_determinism meridian_place)
endif()

ament_package()
