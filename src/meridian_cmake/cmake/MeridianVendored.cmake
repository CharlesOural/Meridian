# INTERFACE/STATIC wrappers around the vendor/ git submodules. Each is guarded so
# a not-yet-fetched submodule does not break configuration for packages that do not
# link it. MERIDIAN_VENDOR_DIR may be set by the caller; otherwise it is resolved
# relative to a package source dir at <workspace>/src/<pkg>.

if(NOT DEFINED MERIDIAN_VENDOR_DIR)
  get_filename_component(MERIDIAN_VENDOR_DIR "${CMAKE_SOURCE_DIR}/../../vendor" ABSOLUTE)
endif()

# basalt-headers: header-only CT spline kernel; needs Eigen + Sophus.
if(NOT TARGET meridian::vendor_basalt AND EXISTS "${MERIDIAN_VENDOR_DIR}/basalt-headers/include")
  add_library(meridian_vendor_basalt INTERFACE)
  target_include_directories(meridian_vendor_basalt INTERFACE
    "${MERIDIAN_VENDOR_DIR}/basalt-headers/include")
  target_link_libraries(meridian_vendor_basalt INTERFACE Eigen3::Eigen Sophus::Sophus)
  add_library(meridian::vendor_basalt ALIAS meridian_vendor_basalt)
endif()

# ikd-Tree: small header + one .cpp, built as a tiny static lib (registration oracle).
if(NOT TARGET meridian::vendor_ikdtree AND EXISTS "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd_Tree.cpp")
  add_library(meridian_vendor_ikdtree STATIC "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd_Tree.cpp")
  target_include_directories(meridian_vendor_ikdtree PUBLIC "${MERIDIAN_VENDOR_DIR}/ikd-Tree")
  target_link_libraries(meridian_vendor_ikdtree PUBLIC Eigen3::Eigen)
  set_target_properties(meridian_vendor_ikdtree PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(meridian::vendor_ikdtree ALIAS meridian_vendor_ikdtree)
endif()

# Scan Context++: header-only loop descriptor.
if(NOT TARGET meridian::vendor_scancontext AND EXISTS "${MERIDIAN_VENDOR_DIR}/scancontext")
  add_library(meridian_vendor_scancontext INTERFACE)
  target_include_directories(meridian_vendor_scancontext INTERFACE
    "${MERIDIAN_VENDOR_DIR}/scancontext")
  target_link_libraries(meridian_vendor_scancontext INTERFACE Eigen3::Eigen)
  add_library(meridian::vendor_scancontext ALIAS meridian_vendor_scancontext)
endif()
