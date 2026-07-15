# Shared C++/CUDA toolchain settings + the per-target warning helper.
# include()d once per package after find_package(meridian_cmake).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# An unset build type compiles without optimization, which the per-scan hot loops
# cannot afford; default to Release and let an explicit -DCMAKE_BUILD_TYPE override.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Static archives must stay linkable into shared libraries (and Python bindings),
# so everything compiles position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA settings take effect only in a package that enables CUDA. Deployment
# presets choose the architecture; this shared helper assumes no build host.
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(MERIDIAN_WERROR "Treat warnings as errors in core packages" ON)

# Apply the house warning set to a target. The CXX generator guard keeps the
# host-only flags off CUDA device compilation, which nvcc would reject.
function(meridian_apply_warnings _target)
  set(_flags -Wall -Wextra -Wpedantic)
  if(MERIDIAN_WERROR)
    list(APPEND _flags -Werror)
  endif()
  target_compile_options(${_target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_flags}>)
endfunction()
