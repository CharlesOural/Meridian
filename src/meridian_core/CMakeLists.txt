cmake_minimum_required(VERSION 3.22)
project(meridian_core VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

add_library(meridian_core
  src/blob.cpp
  src/canonical_bytes.cpp
  src/canonical_records.cpp
  src/canonical_verification.cpp
  src/clock.cpp
  src/factor_batch_api.cpp
  src/local_graph_api.cpp
  src/observation_lineage.cpp
  src/pipeline_observability.cpp
  src/sensor_observations.cpp
  src/sha256.cpp
  src/sparse_map_api.cpp
)
add_library(meridian::core ALIAS meridian_core)

target_include_directories(meridian_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(meridian_core PUBLIC Eigen3::Eigen Sophus::Sophus fmt::fmt)
meridian_apply_warnings(meridian_core)

install(TARGETS meridian_core
  EXPORT export_meridian_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_core HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 Sophus fmt)

meridian_add_gtest(test_core_api test/test_core_api.cpp)
if(TARGET test_core_api)
  target_link_libraries(test_core_api meridian_core)
endif()

meridian_add_gtest(test_calibration test/test_calibration.cpp)
if(TARGET test_calibration)
  target_link_libraries(test_calibration meridian_core)
endif()

meridian_add_gtest(test_local_graph_api test/test_local_graph_api.cpp)
if(TARGET test_local_graph_api)
  target_link_libraries(test_local_graph_api meridian_core)
endif()

meridian_add_gtest(test_factor_batch_api test/test_factor_batch_api.cpp)
if(TARGET test_factor_batch_api)
  target_link_libraries(test_factor_batch_api meridian_core)
endif()

meridian_add_gtest(test_blob_api test/test_blob_api.cpp)
if(TARGET test_blob_api)
  target_link_libraries(test_blob_api meridian_core)
endif()

meridian_add_gtest(test_sparse_map_api test/test_sparse_map_api.cpp)
if(TARGET test_sparse_map_api)
  target_link_libraries(test_sparse_map_api meridian_core)
endif()

meridian_add_gtest(test_sensor_observations test/test_sensor_observations.cpp)
if(TARGET test_sensor_observations)
  target_link_libraries(test_sensor_observations meridian_core)
endif()

meridian_add_gtest(test_canonical_bytes test/test_canonical_bytes.cpp)
if(TARGET test_canonical_bytes)
  target_link_libraries(test_canonical_bytes meridian_core)
endif()

meridian_add_gtest(test_canonical_records test/test_canonical_records.cpp)
if(TARGET test_canonical_records)
  target_link_libraries(test_canonical_records meridian_core)
endif()

meridian_add_gtest(test_canonical_verification test/test_canonical_verification.cpp)
if(TARGET test_canonical_verification)
  target_link_libraries(test_canonical_verification meridian_core)
endif()

meridian_add_gtest(test_pipeline_observability
  test/test_pipeline_observability.cpp
)
if(TARGET test_pipeline_observability)
  target_link_libraries(test_pipeline_observability meridian_core)
endif()

ament_package()
