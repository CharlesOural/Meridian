cmake_minimum_required(VERSION 3.22)
project(meridian_config LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

set(MERIDIAN_NEED_YAMLCPP ON)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus, yaml-cpp

find_package(meridian_common REQUIRED)

add_library(meridian_config
  src/config_loader.cpp)

target_include_directories(meridian_config PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_config PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_config)

target_link_libraries(meridian_config PUBLIC
  meridian::common
  Eigen3::Eigen Sophus::Sophus
  PRIVATE yaml-cpp)

add_library(meridian::config ALIAS meridian_config)

ament_export_targets(meridian_configTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common Eigen3 Sophus)
install(TARGETS meridian_config EXPORT meridian_configTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_config test/test_config.cpp)
  target_link_libraries(test_config meridian_config)
endif()

ament_package()
