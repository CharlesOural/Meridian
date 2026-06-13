cmake_minimum_required(VERSION 3.22)
project(meridian_map LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)  # Eigen3::Eigen, Sophus::Sophus

# Drop the whole map layer on a minimal non-map build. When OFF this package contributes
# nothing (no IMapLayer); meridian_pipeline/meridian_ros compile the map wiring out.
if(NOT MERIDIAN_WITH_MAP)
  ament_package()
  return()
endif()

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)

# Backend-independent core. The cpu surface backend, Tier R, and the factory are added
# alongside this list; the nvblox (CUDA) backend is gated by MERIDIAN_MAP_NVBLOX below.
add_library(meridian_map
  src/keyframe_store.cpp
  src/voxel_hash_map.cpp
  src/cpu/cpu_surface_map.cpp
  src/layered_map.cpp
  src/map_factory.cpp)

target_include_directories(meridian_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(meridian_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(meridian_map PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_map)

target_link_libraries(meridian_map PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  Eigen3::Eigen Sophus::Sophus)

# nvblox (GPU) surface backend — compiled only where CUDA + nvblox are present. The cpu
# backend (above) always builds, so a non-CUDA box still gets a working map.
if(MERIDIAN_MAP_NVBLOX)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(nvblox REQUIRED)
  target_sources(meridian_map PRIVATE
    src/nvblox/nvblox_surface_map.cpp
    src/nvblox/nvblox_integrate.cu)
  set_target_properties(meridian_map PROPERTIES
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 87)
  target_link_libraries(meridian_map PUBLIC nvblox::nvblox CUDA::cudart)
  target_compile_definitions(meridian_map PUBLIC MERIDIAN_MAP_HAS_NVBLOX)
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_map EXPORT meridian_mapTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)

ament_export_targets(meridian_mapTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common meridian_config meridian_debug meridian_calib
                          Eigen3 Sophus)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_keyframe_store test/test_keyframe_store.cpp)
  target_link_libraries(test_keyframe_store meridian_map)
  # Tier R is a private (src/) header; the test reaches it directly.
  meridian_add_gtest(test_voxel_hash_map test/test_voxel_hash_map.cpp)
  target_include_directories(test_voxel_hash_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_voxel_hash_map meridian_map)
  meridian_add_gtest(test_cpu_surface_map test/test_cpu_surface_map.cpp)
  target_include_directories(test_cpu_surface_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_cpu_surface_map meridian_map)
  # Façade + factory + loop-rebuild conformance (public interface only).
  meridian_add_gtest(test_layered_map test/test_layered_map.cpp)
  target_link_libraries(test_layered_map meridian_map)
endif()

ament_package()
