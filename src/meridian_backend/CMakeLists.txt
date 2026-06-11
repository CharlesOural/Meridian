cmake_minimum_required(VERSION 3.22)
project(meridian_backend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

set(MERIDIAN_NEED_GTSAM ON)
set(MERIDIAN_NEED_YAMLCPP ON)
include(MeridianDeps)

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)

# SHARED so GTSAM is resolved into this library and stays out of the exported link
# interface — downstream packages see only the meridian targets.
add_library(meridian_backend SHARED
  src/gtsam_adapter.cpp
  src/gauge_damping_factor.cpp
  src/observability_inflation.cpp
  src/chain_covariance.cpp
  src/max_clique.cpp
  src/pcm.cpp
  src/gnc_consolidation.cpp
  src/pim_from_summary.cpp
  src/geodetic.cpp
  src/robust_kernels.cpp
  src/gnss_factor.cpp
  src/gnss_factor_refined.cpp
  src/datum.cpp
  src/gnss_gate.cpp
  src/isam2_backend.cpp
  src/graph_markers.cpp
  src/backend_factory.cpp
)
target_include_directories(meridian_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Internal graph headers live under src/ and are not exported.
target_include_directories(meridian_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(meridian_backend PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_backend)
target_link_libraries(meridian_backend PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  Eigen3::Eigen
  Sophus::Sophus
)
target_link_libraries(meridian_backend PRIVATE
  gtsam
)

ament_export_targets(meridian_backendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_common meridian_config meridian_debug meridian_calib Eigen3 Sophus
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_backend
  EXPORT meridian_backendTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Offline iteration harness: packet log -> IBackEnd -> TUM, no pipeline, no ROS.
add_executable(backend_runner src/backend_runner.cpp)
target_compile_features(backend_runner PUBLIC cxx_std_20)
meridian_apply_warnings(backend_runner)
target_link_libraries(backend_runner meridian_backend yaml-cpp)
install(TARGETS backend_runner DESTINATION lib/meridian_backend)

if(BUILD_TESTING)
  include(MeridianTesting)
  # Tests exercise the internal graph classes directly, so they see src/ and link
  # GTSAM themselves (it is private to the library and not in its link interface).
  foreach(t test_gtsam_adapter test_tangent_ordering test_gauge_damping
            test_observability_inflation test_chain_covariance test_backend_spine
            test_determinism test_backend_recovery test_pim_from_summary
            test_restart_bridge test_geodetic test_robust_kernels test_gnss_factor
            test_datum test_gnss_gate test_backend_gnss test_max_clique test_pcm
            test_gnc_consolidation test_backend_loops test_gnss_factor_refined
            test_backend_extrinsic test_graph_markers)
    meridian_add_gtest(${t} test/${t}.cpp)
    target_link_libraries(${t} meridian_backend gtsam)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()
endif()

ament_package()
