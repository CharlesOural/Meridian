cmake_minimum_required(VERSION 3.22)
project(meridian_apps VERSION 0.1.0 LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianTesting)

find_package(meridian_core REQUIRED)
find_package(meridian_local_rt REQUIRED)
find_package(meridian_ros REQUIRED)
find_package(meridian_tools REQUIRED)

add_library(meridian_apps
  src/bag_localize.cpp
  src/process_report_normalizer.cpp
)
add_library(meridian::apps ALIAS meridian_apps)
target_include_directories(meridian_apps
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(meridian_apps PUBLIC
  meridian_core
  meridian_local_rt
  meridian_ros
  meridian_tools
)
meridian_apply_warnings(meridian_apps)

add_executable(meridian_bag_localize
  src/meridian_bag_localize_main.cpp
)
target_link_libraries(meridian_bag_localize PRIVATE meridian_apps)
meridian_apply_warnings(meridian_bag_localize)

install(TARGETS meridian_apps meridian_bag_localize
  EXPORT export_meridian_apps
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_meridian_apps HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_core
  meridian_local_rt
  meridian_ros
  meridian_tools
)

meridian_add_gtest(test_bag_localize test/test_bag_localize.cpp)
if(TARGET test_bag_localize)
  target_link_libraries(test_bag_localize meridian_apps)
  target_include_directories(test_bag_localize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  set_tests_properties(test_bag_localize PROPERTIES TIMEOUT 120)
endif()

ament_package()
