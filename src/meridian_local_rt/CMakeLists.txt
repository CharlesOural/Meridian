cmake_minimum_required(VERSION 3.22)
project(meridian_local_rt LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
find_package(meridian_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres 2.1 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Sophus REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

add_library(meridian_local_rt
  src/combined_imu_cost.cpp
  src/combined_preintegration.cpp
  src/config.cpp
  src/estimator/fixed_lag_estimator.cpp
  src/estimator/fixed_linearization_marginal_prior.cpp
  src/estimator/scan_to_map_cost.cpp
  src/estimator/square_root_marginalization.cpp
  src/imu_buffer.cpp
  src/imu_propagator.cpp
  src/initialization/bootstrap_odometry.cpp
  src/initialization/dynamic_initializer.cpp
  src/initialization/static_initializer.cpp
  src/lidar/point_to_point_registration.cpp
  src/lidar/scan_preprocessor.cpp
  src/lidar/scan_to_map.cpp
  src/lidar/voxel_grid.cpp
  src/lidar/voxel_target.cpp
  src/pipeline.cpp
  src/right_se3_manifold.cpp
)
target_compile_features(meridian_local_rt PUBLIC cxx_std_20)
target_include_directories(meridian_local_rt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(meridian_local_rt
  PUBLIC
    meridian_core::meridian_core
    Eigen3::Eigen
    Ceres::ceres
    Sophus::Sophus
  PRIVATE
    gtsam
)
meridian_apply_warnings(meridian_local_rt)

install(
  TARGETS meridian_local_rt
  EXPORT export_meridian_local_rt
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  meridian_add_gtest(test_config test/test_config.cpp)
  target_link_libraries(test_config meridian_local_rt)

  meridian_add_gtest(test_imu_buffer test/test_imu_buffer.cpp)
  target_link_libraries(test_imu_buffer meridian_local_rt)

  meridian_add_gtest(test_inertial_foundation test/test_inertial_foundation.cpp)
  target_link_libraries(test_inertial_foundation meridian_local_rt gtsam)

  meridian_add_gtest(test_static_initializer test/initialization/test_static_initializer.cpp)
  target_link_libraries(test_static_initializer meridian_local_rt)

  meridian_add_gtest(test_dynamic_initializer test/initialization/test_dynamic_initializer.cpp)
  target_link_libraries(test_dynamic_initializer meridian_local_rt)

  meridian_add_gtest(test_pipeline test/test_pipeline.cpp)
  target_link_libraries(test_pipeline meridian_local_rt)

  meridian_add_gtest(test_bootstrap_odometry test/initialization/test_bootstrap_odometry.cpp)
  target_link_libraries(test_bootstrap_odometry meridian_local_rt)

  meridian_add_gtest(test_point_to_point_registration
    test/lidar/test_point_to_point_registration.cpp
  )
  target_link_libraries(test_point_to_point_registration meridian_local_rt)

  meridian_add_gtest(test_voxel_grid test/lidar/test_voxel_grid.cpp)
  target_link_libraries(test_voxel_grid meridian_local_rt)

  meridian_add_gtest(test_voxel_target test/lidar/test_voxel_target.cpp)
  target_link_libraries(test_voxel_target meridian_local_rt)

  meridian_add_gtest(test_scan_preprocessor test/lidar/test_scan_preprocessor.cpp)
  target_link_libraries(test_scan_preprocessor meridian_local_rt)

  meridian_add_gtest(test_scan_to_map_target test/lidar/test_scan_to_map_target.cpp)
  target_link_libraries(test_scan_to_map_target meridian_local_rt)

  meridian_add_gtest(test_scan_to_map_cost test/estimator/test_scan_to_map_cost.cpp)
  target_link_libraries(test_scan_to_map_cost meridian_local_rt)

  meridian_add_gtest(test_fixed_linearization_marginal_prior
    test/estimator/test_fixed_linearization_marginal_prior.cpp
  )
  target_link_libraries(test_fixed_linearization_marginal_prior meridian_local_rt)

  meridian_add_gtest(test_square_root_marginalization
    test/estimator/test_square_root_marginalization.cpp
  )
  target_link_libraries(test_square_root_marginalization meridian_local_rt)

  meridian_add_gtest(test_fixed_lag_estimator
    test/estimator/test_fixed_lag_estimator.cpp
  )
  target_link_libraries(test_fixed_lag_estimator meridian_local_rt)
endif()

ament_export_targets(export_meridian_local_rt HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_core Eigen3 Ceres GTSAM Sophus)
ament_package()
