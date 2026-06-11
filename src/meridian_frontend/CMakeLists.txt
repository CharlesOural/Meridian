cmake_minimum_required(VERSION 3.22)
project(meridian_frontend LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(meridian_cmake REQUIRED)
include(MeridianToolchain)

set(MERIDIAN_NEED_CERES ON)
set(MERIDIAN_NEED_PCL ON)   # ikd-Tree's header needs pcl/point_types.h
set(MERIDIAN_NEED_OPENCV ON)  # ImagePyramidView samples the cv::Mat pyramid
include(MeridianDeps)
include(MeridianVendored)

find_package(meridian_common REQUIRED)
find_package(meridian_config REQUIRED)
find_package(meridian_debug REQUIRED)
find_package(meridian_calib REQUIRED)
find_package(meridian_preprocess REQUIRED)

# OpenMP drives the per-point LiDAR association loop. It is optional: the loop
# guards its pragma so a toolchain without OpenMP still compiles and runs serially.
find_package(OpenMP QUIET)

# The keyframe-cov worker runs on its own std::thread.
find_package(Threads REQUIRED)

# SHARED so the privately linked vendored kernels (basalt spline, ikd-Tree) are
# resolved into this library and never appear in the exported link interface —
# downstream packages need only the meridian targets.
add_library(meridian_frontend SHARED
  src/frontend_factory.cpp
  src/ct/spline_window.cpp
  src/ct/residuals_imu.cpp
  src/ct/residuals_lidar.cpp
  src/ct/image_pyramid_view.cpp
  src/ct/visual_map.cpp
  src/ct/residuals_visual.cpp
  src/ct/residuals_gnss.cpp
  src/ct/marginalization.cpp
  src/ct/pose_marginal.cpp
  src/ct/window_problem.cpp
  src/ct/keyframe_finalizer.cpp
  src/ct/ct_frontend.cpp
)
target_include_directories(meridian_frontend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# Internal estimator headers live under src/ and are not exported.
target_include_directories(meridian_frontend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(meridian_frontend PUBLIC cxx_std_20)
meridian_apply_warnings(meridian_frontend)
target_link_libraries(meridian_frontend PUBLIC
  meridian_common::meridian_common
  meridian_config::meridian_config
  meridian_debug::meridian_debug
  meridian_calib::meridian_calib
  meridian_preprocess::meridian_preprocess
  Eigen3::Eigen
  Sophus::Sophus
  Ceres::ceres
  ${OpenCV_LIBS}
)
target_link_libraries(meridian_frontend PRIVATE
  meridian::vendor_basalt
  meridian::vendor_ikdtree
  Threads::Threads
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(meridian_frontend PRIVATE OpenMP::OpenMP_CXX)
endif()

ament_export_targets(meridian_frontendTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(
  meridian_common meridian_config meridian_debug meridian_calib
  meridian_preprocess Eigen3 Sophus Ceres OpenCV
)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS meridian_frontend
  EXPORT meridian_frontendTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  include(MeridianTesting)
  # Tests exercise internal estimator classes, so they see src/ directly.

  # The spline test compiles spline_window.cpp into its own target rather than
  # linking meridian_frontend, so it builds and runs independently of the other
  # estimator translation units in the library.
  meridian_add_gtest(test_spline_window
    test/test_spline_window.cpp
    src/ct/spline_window.cpp
  )
  target_link_libraries(test_spline_window
    meridian_common::meridian_common
    Eigen3::Eigen
    Sophus::Sophus
    meridian::vendor_basalt
  )
  target_include_directories(test_spline_window PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # The image-pyramid sampler depends only on OpenCV + Eigen, so its test compiles
  # the one translation unit directly and links neither the estimator library nor
  # the vendored kernels.
  meridian_add_gtest(test_image_pyramid_view
    test/test_image_pyramid_view.cpp
    src/ct/image_pyramid_view.cpp
  )
  target_link_libraries(test_image_pyramid_view
    Eigen3::Eigen
    ${OpenCV_LIBS}
  )
  target_include_directories(test_image_pyramid_view PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  meridian_add_gtest(test_residuals_imu test/test_residuals_imu.cpp)
  meridian_add_gtest(test_residuals_lidar test/test_residuals_lidar.cpp)
  meridian_add_gtest(test_residuals_gnss test/test_residuals_gnss.cpp)
  meridian_add_gtest(test_marginalization test/test_marginalization.cpp)
  # The end-to-end suite runs whole synthetic streams per test; give it headroom
# over ctest's 60 s default.
meridian_add_gtest(test_ct_frontend test/test_ct_frontend.cpp TIMEOUT 180)
  meridian_add_gtest(test_keyframe_finalizer test/test_keyframe_finalizer.cpp TIMEOUT 180)
  foreach(t test_residuals_imu test_residuals_lidar test_residuals_gnss test_marginalization
            test_ct_frontend test_keyframe_finalizer)
    # OpenCV is linked too: the CT front-end test renders synthetic camera images to
    # exercise the visual stage.
    target_link_libraries(${t} meridian_frontend meridian::vendor_basalt ${OpenCV_LIBS})
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  endforeach()

  # The photometric residual test renders synthetic image pyramids directly, so it
  # links OpenCV on top of the estimator library and the vendored spline kernels.
  meridian_add_gtest(test_residuals_visual test/test_residuals_visual.cpp)
  target_link_libraries(test_residuals_visual
    meridian_frontend
    meridian::vendor_basalt
    ${OpenCV_LIBS}
  )
  target_include_directories(test_residuals_visual PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

  # The visual-map test renders synthetic textured pyramids and drives the patch
  # lifecycle. It compiles the visual-map and image-pyramid translation units
  # directly so it builds independently of the rest of the estimator library, and
  # links the camera model, the vendored spline kernel (a header dependency through
  # residuals_lidar.hpp), and OpenCV.
  meridian_add_gtest(test_visual_map
    test/test_visual_map.cpp
    src/ct/visual_map.cpp
    src/ct/image_pyramid_view.cpp
  )
  target_link_libraries(test_visual_map
    meridian_common::meridian_common
    meridian_config::meridian_config
    meridian_calib::meridian_calib
    Eigen3::Eigen
    Sophus::Sophus
    meridian::vendor_basalt
    ${OpenCV_LIBS}
  )
  target_include_directories(test_visual_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()

# ThreadSanitizer probe for the parallel-association ikd-Tree data race. Off by default
# (it instruments its own copy of residuals_lidar.cpp and the vendored ikd-Tree source
# with -fsanitize=thread, which must not leak into the production library). Build with
# -DMERIDIAN_TSAN_PROBE=ON to get two executables: tsan_lidar_map (the fixed path, with
# the pre-flush) and tsan_lidar_map_noflush (the flush compiled out, which must report
# the race -- proof the probe bites).
option(MERIDIAN_TSAN_PROBE "Build the LidarLocalMap ThreadSanitizer probe" OFF)
if(MERIDIAN_TSAN_PROBE)
  find_package(Threads REQUIRED)
  set(_tsan_ikd "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree/ikd_Tree.cpp")
  foreach(variant tsan_lidar_map tsan_lidar_map_noflush)
    add_executable(${variant}
      test/test_lidar_map_tsan.cpp
      src/ct/residuals_lidar.cpp
      src/ct/spline_window.cpp
      "${_tsan_ikd}"
    )
    target_include_directories(${variant} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(${variant} SYSTEM PRIVATE
      "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree"
      ${PCL_INCLUDE_DIRS}
    )
    target_compile_features(${variant} PRIVATE cxx_std_20)
    target_compile_options(${variant} PRIVATE -fsanitize=thread -g -O1)
    target_link_options(${variant} PRIVATE -fsanitize=thread)
    target_link_libraries(${variant}
      meridian_common::meridian_common
      meridian_config::meridian_config
      Eigen3::Eigen
      Sophus::Sophus
      Ceres::ceres
      meridian::vendor_basalt
      Threads::Threads
    )
  endforeach()
  target_compile_definitions(tsan_lidar_map_noflush PRIVATE TSAN_PROBE_NO_FLUSH)
endif()

ament_package()
