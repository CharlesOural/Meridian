cmake_minimum_required(VERSION 3.22)
project(meridian_core LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

add_library(meridian_core
  src/geometry.cpp
  src/ids.cpp
  src/initialization.cpp
  src/navigation.cpp
  src/observations.cpp
  src/time.cpp
)
add_library(meridian::core ALIAS meridian_core)
target_compile_features(meridian_core PUBLIC cxx_std_20)
target_include_directories(meridian_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
meridian_apply_warnings(meridian_core)

install(
  TARGETS meridian_core
  EXPORT export_meridian_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  meridian_add_gtest(test_time test/test_time.cpp)
  target_link_libraries(test_time meridian_core)

  meridian_add_gtest(test_geometry test/test_geometry.cpp)
  target_link_libraries(test_geometry meridian_core)

  meridian_add_gtest(test_navigation test/test_navigation.cpp)
  target_link_libraries(test_navigation meridian_core)

  meridian_add_gtest(test_initialization test/test_initialization.cpp)
  target_link_libraries(test_initialization meridian_core)

  meridian_add_gtest(test_observations test/test_observations.cpp)
  target_link_libraries(test_observations meridian_core)

  meridian_add_gtest(test_debug_sink test/test_debug_sink.cpp)
  target_link_libraries(test_debug_sink meridian_core)
endif()

ament_export_targets(export_meridian_core HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_cmake)
ament_package()
