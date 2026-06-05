cmake_minimum_required(VERSION 3.22)
project(meridian_common LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

# Header-only value-type / math library.
add_library(meridian_common INTERFACE)
target_include_directories(meridian_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(meridian_common INTERFACE cxx_std_20)
target_link_libraries(meridian_common INTERFACE Eigen3::Eigen Sophus::Sophus)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_common EXPORT meridian_commonTargets)
ament_export_targets(meridian_commonTargets)
ament_export_dependencies(Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_pose test/test_pose.cpp)
  target_link_libraries(test_pose meridian_common)
  meridian_add_gtest(test_nav_state test/test_nav_state.cpp)
  target_link_libraries(test_nav_state meridian_common)
  meridian_add_gtest(test_keyframe_packet test/test_keyframe_packet.cpp)
  target_link_libraries(test_keyframe_packet meridian_common)
  meridian_add_gtest(test_ring_buffer test/test_ring_buffer.cpp)
  target_link_libraries(test_ring_buffer meridian_common)
endif()

ament_package()
