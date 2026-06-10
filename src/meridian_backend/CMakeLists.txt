cmake_minimum_required(VERSION 3.22)
project(meridian_backend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

set(MERIDIAN_NEED_GTSAM ON)
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
  src/isam2_backend.cpp
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

if(BUILD_TESTING)
  include(MeridianTesting)
  # Tests exercise the internal graph classes directly, so they see src/ and link
  # GTSAM themselves (it is private to the library and not in its link interface).
  foreach(t test_gtsam_adapter test_tangent_ordering test_gauge_damping
            test_observability_inflation test_chain_covariance test_backend_spine
            test_determinism)
    meridian_add_gtest(${t} test/${t}.cpp)
    target_link_libraries(${t} meridian_backend gtsam)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()
endif()

ament_package()
