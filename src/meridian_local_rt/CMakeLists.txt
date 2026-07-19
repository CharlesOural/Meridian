cmake_minimum_required(VERSION 3.22)
project(meridian_local_rt VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(meridian_core REQUIRED)
# GTSAM 4.2.1 is the first release whose ISAM2 leaf marginalization reuses
# vacant nonlinear-factor slots instead of growing fixed-lag factor storage.
find_package(GTSAM 4.2.1 REQUIRED)
find_package(OpenCV 4.5 REQUIRED COMPONENTS core features2d imgproc video)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(meridian_local_rt
  src/candidate_isolated_isam2.cpp
  src/equidistant_camera.cpp
  src/event_scheduler.cpp
  src/factor_allocator.cpp
  src/direct_lidar_factor.cpp
  src/lidar_composite_target.cpp
  src/lidar_registration.cpp
  src/graph.cpp
  src/imu.cpp
  src/lidar_deskew.cpp
  src/finalized_lidar_target_map.cpp
  src/lidar_map_payload.cpp
  src/lidar_bootstrap.cpp
  src/lidar_registration_cloud.cpp
  src/local_estimator.cpp
  src/map_admission_gate.cpp
  src/motion_initializer.cpp
  src/pipeline_observability.cpp
  src/rolling_lidar_target.cpp
  src/sensor_health.cpp
  src/visual_factor.cpp
  src/visual_frontend.cpp
  src/visual_lane.cpp
)
add_library(meridian::local_rt ALIAS meridian_local_rt)
target_include_directories(meridian_local_rt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(meridian_local_rt PUBLIC meridian_core)
# Enable OpenMP only in the two explicitly indexed LiDAR stages. Applying the
# imported target's compile options to the whole estimator would define
# _OPENMP in graph/visual translation units and can silently enable unrelated
# Eigen parallelism. The runtime itself remains a normal private static-library
# dependency and is exported as a link-only requirement by CMake.
set_property(SOURCE
  src/lidar_registration.cpp
  APPEND PROPERTY COMPILE_OPTIONS ${OpenMP_CXX_FLAGS}
)
target_link_libraries(meridian_local_rt PRIVATE
  gtsam
  ${OpenCV_LIBRARIES}
  ${OpenMP_CXX_LIBRARIES}
)
target_include_directories(meridian_local_rt SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
meridian_apply_warnings(meridian_local_rt)

install(TARGETS meridian_local_rt
  EXPORT export_meridian_local_rt
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_local_rt HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_core GTSAM OpenCV OpenMP)

meridian_add_gtest(test_imu_and_deskew test/test_imu_and_deskew.cpp)
if(TARGET test_imu_and_deskew)
  target_link_libraries(test_imu_and_deskew meridian_local_rt)
endif()

meridian_add_gtest(test_graph test/test_graph.cpp)
if(TARGET test_graph)
  target_include_directories(test_graph PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_graph meridian_local_rt gtsam)
endif()

meridian_add_gtest(test_factor_batch_graph test/test_factor_batch_graph.cpp)
if(TARGET test_factor_batch_graph)
  target_link_libraries(test_factor_batch_graph meridian_local_rt)
endif()

meridian_add_gtest(test_event_scheduler test/test_event_scheduler.cpp)
if(TARGET test_event_scheduler)
  target_link_libraries(test_event_scheduler meridian_local_rt)
endif()

meridian_add_gtest(test_factor_allocator test/test_factor_allocator.cpp)
if(TARGET test_factor_allocator)
  target_link_libraries(test_factor_allocator meridian_local_rt)
endif()

meridian_add_gtest(test_sensor_health test/test_sensor_health.cpp)
if(TARGET test_sensor_health)
  target_link_libraries(test_sensor_health meridian_local_rt)
endif()

meridian_add_gtest(test_map_admission_gate test/test_map_admission_gate.cpp)
if(TARGET test_map_admission_gate)
  target_link_libraries(test_map_admission_gate meridian_local_rt)
endif()

meridian_add_gtest(test_lidar_registration_cloud
  test/test_lidar_registration_cloud.cpp
)
if(TARGET test_lidar_registration_cloud)
  target_link_libraries(test_lidar_registration_cloud
    meridian_local_rt OpenMP::OpenMP_CXX
  )
endif()

meridian_add_gtest(test_lidar_composite_target
  test/test_lidar_composite_target.cpp
)
if(TARGET test_lidar_composite_target)
  target_link_libraries(test_lidar_composite_target meridian_local_rt)
endif()

meridian_add_gtest(test_lidar_map_payload test/test_lidar_map_payload.cpp)
if(TARGET test_lidar_map_payload)
  target_link_libraries(test_lidar_map_payload meridian_local_rt)
endif()

meridian_add_gtest(test_finalized_lidar_target_map
  test/test_finalized_lidar_target_map.cpp
)
if(TARGET test_finalized_lidar_target_map)
  target_link_libraries(test_finalized_lidar_target_map meridian_local_rt)
endif()

meridian_add_gtest(test_lidar_registration test/test_lidar_registration.cpp)
if(TARGET test_lidar_registration)
  target_link_libraries(test_lidar_registration
    meridian_local_rt OpenMP::OpenMP_CXX
  )
endif()

meridian_add_gtest(test_direct_lidar_factor test/test_direct_lidar_factor.cpp)
if(TARGET test_direct_lidar_factor)
  target_include_directories(test_direct_lidar_factor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(test_direct_lidar_factor meridian_local_rt gtsam)
endif()

meridian_add_gtest(test_direct_lidar_factor_sufficient_statistics
  test/test_direct_lidar_factor_sufficient_statistics.cpp
)
if(TARGET test_direct_lidar_factor_sufficient_statistics)
  target_include_directories(test_direct_lidar_factor_sufficient_statistics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(test_direct_lidar_factor_sufficient_statistics
    meridian_local_rt gtsam
  )
endif()

meridian_add_gtest(test_lidar_bootstrap test/test_lidar_bootstrap.cpp)
if(TARGET test_lidar_bootstrap)
  target_link_libraries(test_lidar_bootstrap meridian_local_rt)
endif()

meridian_add_gtest(test_rolling_lidar_target test/test_rolling_lidar_target.cpp)
if(TARGET test_rolling_lidar_target)
  target_include_directories(test_rolling_lidar_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(test_rolling_lidar_target meridian_local_rt)
endif()

meridian_add_gtest(test_local_estimator test/test_local_estimator.cpp)
if(TARGET test_local_estimator)
  target_include_directories(test_local_estimator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(test_local_estimator meridian_local_rt)
endif()

meridian_add_gtest(test_local_estimator_imu_spine
  test/test_local_estimator_imu_spine.cpp
)
if(TARGET test_local_estimator_imu_spine)
  target_link_libraries(test_local_estimator_imu_spine meridian_local_rt)
endif()

meridian_add_gtest(test_local_estimator_visual
  test/test_local_estimator_visual.cpp
)
if(TARGET test_local_estimator_visual)
  target_link_libraries(test_local_estimator_visual
    meridian_local_rt ${OpenCV_LIBRARIES}
  )
  target_include_directories(test_local_estimator_visual SYSTEM PRIVATE
    ${OpenCV_INCLUDE_DIRS}
  )
endif()

meridian_add_gtest(test_local_estimator_motion_initialization
  test/test_local_estimator_motion_initialization.cpp
)
if(TARGET test_local_estimator_motion_initialization)
  target_link_libraries(test_local_estimator_motion_initialization
    meridian_local_rt gtsam
  )
endif()

meridian_add_gtest(test_motion_initializer test/test_motion_initializer.cpp)
if(TARGET test_motion_initializer)
  target_link_libraries(test_motion_initializer meridian_local_rt)
endif()

meridian_add_gtest(test_pipeline_observability test/test_pipeline_observability.cpp)
if(TARGET test_pipeline_observability)
  target_link_libraries(test_pipeline_observability meridian_local_rt)
endif()

meridian_add_gtest(test_visual_frontend test/test_visual_frontend.cpp)
if(TARGET test_visual_frontend)
  target_link_libraries(test_visual_frontend meridian_local_rt ${OpenCV_LIBRARIES})
  target_include_directories(test_visual_frontend SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

meridian_add_gtest(test_visual_factor test/test_visual_factor.cpp)
if(TARGET test_visual_factor)
  target_include_directories(test_visual_factor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  target_link_libraries(test_visual_factor meridian_local_rt gtsam)
endif()

meridian_add_gtest(test_visual_lane test/test_visual_lane.cpp)
if(TARGET test_visual_lane)
  target_link_libraries(test_visual_lane meridian_local_rt ${OpenCV_LIBRARIES})
  target_include_directories(test_visual_lane SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

ament_package()
