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

# CUDA settings only take effect in a package that enables the CUDA language.
# Default arch is SM 8.7; override per build target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA SM architecture (default 8.7)")
endif()
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(MERIDIAN_WERROR "Treat warnings as errors in core packages" ON)

# L4 map build switches. MERIDIAN_WITH_MAP gates the whole map layer; MERIDIAN_MAP_NVBLOX
# gates the CUDA/nvblox surface backend within it. The cpu surface backend always builds
# when the map layer does (no special toolchain), so a non-CUDA box still gets a working
# map. Default the nvblox backend ON only when a CUDA compiler is present, so a non-CUDA
# dev box configures cleanly without manual flags.
option(MERIDIAN_WITH_MAP "Build the L4 map layer (meridian_map)" ON)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  option(MERIDIAN_MAP_NVBLOX "Compile the nvblox (CUDA/GPU) surface backend" ON)
else()
  option(MERIDIAN_MAP_NVBLOX "Compile the nvblox (CUDA/GPU) surface backend" OFF)
endif()

# Apply the house warning set to a target. The CXX generator guard keeps the
# host-only flags off CUDA device compilation, which nvcc would reject.
function(meridian_apply_warnings _target)
  set(_flags -Wall -Wextra -Wpedantic)
  if(MERIDIAN_WERROR)
    list(APPEND _flags -Werror)
  endif()
  target_compile_options(${_target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_flags}>)
endfunction()
