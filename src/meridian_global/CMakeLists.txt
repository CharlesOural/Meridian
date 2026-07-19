cmake_minimum_required(VERSION 3.22)
project(meridian_global VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(meridian_core REQUIRED)
# Keep both graph owners on the same minimum. Earlier ISAM2 leaf
# marginalization grows fixed-lag factor storage instead of reusing slots.
find_package(GTSAM 4.2.1 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS calib3d core features2d imgproc)

add_library(meridian_global
  src/adjacent_boundary_factor_internal.cpp
  src/graph_journal.cpp
  src/global_coordinator.cpp
  src/gnc_tls.cpp
  src/gnss.cpp
  src/gnss_fsm.cpp
  src/gnss_interpolation.cpp
  src/graph.cpp
  src/graph_checkpoint.cpp
  src/lidar_place.cpp
  src/loop_consensus.cpp
  src/persistence_internal.cpp
  src/seal_spool.cpp
  src/sparse_submap.cpp
  src/visual_place.cpp
)
add_library(meridian::global ALIAS meridian_global)
target_include_directories(meridian_global
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(meridian_global PUBLIC meridian_core)
target_link_libraries(meridian_global PRIVATE gtsam ${OpenCV_LIBRARIES})
target_include_directories(meridian_global SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
meridian_apply_warnings(meridian_global)

install(TARGETS meridian_global
  EXPORT export_meridian_global
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_global HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_core GTSAM)

meridian_add_gtest(test_gnss test/test_gnss.cpp)
if(TARGET test_gnss)
  target_link_libraries(test_gnss meridian_global)
endif()

meridian_add_gtest(test_gnss_interpolation test/test_gnss_interpolation.cpp)
if(TARGET test_gnss_interpolation)
  target_link_libraries(test_gnss_interpolation meridian_global)
endif()

meridian_add_gtest(test_gnss_fsm test/test_gnss_fsm.cpp)
if(TARGET test_gnss_fsm)
  target_link_libraries(test_gnss_fsm meridian_global)
endif()

meridian_add_gtest(test_global_graph test/test_graph.cpp)
if(TARGET test_global_graph)
  target_link_libraries(test_global_graph meridian_global)
endif()

meridian_add_gtest(test_global_graph_checkpoint test/test_graph_checkpoint.cpp)
if(TARGET test_global_graph_checkpoint)
  target_link_libraries(test_global_graph_checkpoint meridian_global)
endif()

meridian_add_gtest(test_adjacent_boundary_factor test/test_adjacent_boundary_factor.cpp)
if(TARGET test_adjacent_boundary_factor)
  target_link_libraries(test_adjacent_boundary_factor meridian_global gtsam)
endif()

meridian_add_gtest(test_graph_journal test/test_graph_journal.cpp)
if(TARGET test_graph_journal)
  target_link_libraries(test_graph_journal meridian_global)
endif()

meridian_add_gtest(test_loop_consensus test/test_loop_consensus.cpp)
if(TARGET test_loop_consensus)
  target_link_libraries(test_loop_consensus meridian_global)
endif()

meridian_add_gtest(test_gnc_tls test/test_gnc_tls.cpp)
if(TARGET test_gnc_tls)
  target_link_libraries(test_gnc_tls meridian_global)
endif()

meridian_add_gtest(test_sparse_submap test/test_sparse_submap.cpp)
if(TARGET test_sparse_submap)
  target_link_libraries(test_sparse_submap meridian_global)
endif()

meridian_add_gtest(test_seal_spool test/test_seal_spool.cpp)
if(TARGET test_seal_spool)
  target_link_libraries(test_seal_spool meridian_global)
endif()

meridian_add_gtest(test_global_coordinator test/test_global_coordinator.cpp)
if(TARGET test_global_coordinator)
  target_link_libraries(test_global_coordinator meridian_global)
endif()

meridian_add_gtest(test_lidar_place test/test_lidar_place.cpp)
if(TARGET test_lidar_place)
  target_link_libraries(test_lidar_place meridian_global)
endif()

meridian_add_gtest(test_visual_place test/test_visual_place.cpp)
if(TARGET test_visual_place)
  target_link_libraries(test_visual_place meridian_global)
endif()

ament_package()
