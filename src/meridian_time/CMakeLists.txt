cmake_minimum_required(VERSION 3.22)
project(meridian_time LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)
include(MeridianDeps)            # Eigen3::Eigen, Sophus::Sophus

find_package(meridian_common REQUIRED)

add_library(meridian_time
  src/clock_model.cpp)

target_include_directories(meridian_time PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_compile_features(meridian_time PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_time)

target_link_libraries(meridian_time PUBLIC
  meridian::common)

add_library(meridian::time ALIAS meridian_time)

ament_export_targets(meridian_timeTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(meridian_common)

install(TARGETS meridian_time EXPORT meridian_timeTargets
        ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  include(MeridianTesting)
  meridian_add_gtest(test_clock_model test/test_clock_model.cpp)
  target_link_libraries(test_clock_model meridian_time)
endif()

ament_package()
