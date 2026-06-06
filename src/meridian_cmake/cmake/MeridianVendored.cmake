# INTERFACE/STATIC wrappers around the vendor/ git submodules.
#
# A vendored target is created whenever its submodule is present, so a package that
# links it just works once the submodules are fetched. The hazard this module guards
# against is a *missing* submodule: previously target creation was silently skipped,
# so a consumer that linked the target only learned of the problem via a cryptic
# "unknown target meridian::vendor_*" error much later.
#
# Guard policy: a consuming package declares the vendored targets it links by setting
# the matching MERIDIAN_NEED_VENDOR_* flag before include(MeridianVendored). If a
# needed submodule is absent the build fails immediately, naming the submodule and the
# fetch command. The fatal is gated on the package having requested the target (the
# NEED flag) AND the target not already existing, so packages that link no vendored
# target — or only ones that are present — still configure cleanly when an unrelated
# submodule is missing. MERIDIAN_VENDOR_DIR may be set by the caller; otherwise it is
# resolved relative to a package source dir at <workspace>/src/<pkg>.

if(NOT DEFINED MERIDIAN_VENDOR_DIR)
  get_filename_component(MERIDIAN_VENDOR_DIR "${CMAKE_SOURCE_DIR}/../../vendor" ABSOLUTE)
endif()

# Fail with a uniform, actionable message when a requested submodule is not fetched.
function(_meridian_vendor_missing submodule)
  message(FATAL_ERROR
    "meridian: vendored submodule '${submodule}' is required by this package but was "
    "not found under '${MERIDIAN_VENDOR_DIR}/${submodule}'. Fetch it with: "
    "git submodule update --init --recursive vendor/${submodule}")
endfunction()

# basalt-headers: header-only CT spline kernel; needs Eigen + Sophus.
if(NOT TARGET meridian::vendor_basalt)
  if(EXISTS "${MERIDIAN_VENDOR_DIR}/basalt-headers/include")
    add_library(meridian_vendor_basalt INTERFACE)
    target_include_directories(meridian_vendor_basalt INTERFACE
      "${MERIDIAN_VENDOR_DIR}/basalt-headers/include")
    target_link_libraries(meridian_vendor_basalt INTERFACE Eigen3::Eigen Sophus::Sophus)
    add_library(meridian::vendor_basalt ALIAS meridian_vendor_basalt)
  elseif(MERIDIAN_NEED_VENDOR_BASALT)
    _meridian_vendor_missing("basalt-headers")
  endif()
endif()

# ikd-Tree: header + one .cpp (nested ikd-Tree/ dir upstream), built as a tiny static
# lib (registration oracle). Its header includes pcl/point_types.h, so the consumer
# must have found PCL (MERIDIAN_NEED_PCL) before this module runs; the includes are
# SYSTEM so the vendored code is exempt from the house -Werror warning set.
if(NOT TARGET meridian::vendor_ikdtree)
  if(EXISTS "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree/ikd_Tree.cpp" AND PCL_FOUND)
    add_library(meridian_vendor_ikdtree STATIC
      "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree/ikd_Tree.cpp")
    target_include_directories(meridian_vendor_ikdtree SYSTEM PUBLIC
      "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree"
      ${PCL_INCLUDE_DIRS})
    target_link_libraries(meridian_vendor_ikdtree PUBLIC Eigen3::Eigen)
    find_package(Threads REQUIRED)
    target_link_libraries(meridian_vendor_ikdtree PUBLIC Threads::Threads)
    set_target_properties(meridian_vendor_ikdtree PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(meridian::vendor_ikdtree ALIAS meridian_vendor_ikdtree)
  elseif(MERIDIAN_NEED_VENDOR_IKDTREE
         AND NOT EXISTS "${MERIDIAN_VENDOR_DIR}/ikd-Tree/ikd-Tree/ikd_Tree.cpp")
    _meridian_vendor_missing("ikd-Tree")
  endif()
endif()

# Scan Context++: the C++ module inside the upstream evaluation repo.
if(NOT TARGET meridian::vendor_scancontext)
  if(EXISTS "${MERIDIAN_VENDOR_DIR}/scancontext/cpp/module/Scancontext")
    add_library(meridian_vendor_scancontext INTERFACE)
    target_include_directories(meridian_vendor_scancontext INTERFACE
      "${MERIDIAN_VENDOR_DIR}/scancontext/cpp/module/Scancontext")
    target_link_libraries(meridian_vendor_scancontext INTERFACE Eigen3::Eigen)
    add_library(meridian::vendor_scancontext ALIAS meridian_vendor_scancontext)
  elseif(MERIDIAN_NEED_VENDOR_SCANCONTEXT)
    _meridian_vendor_missing("scancontext")
  endif()
endif()
