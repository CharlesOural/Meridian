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
