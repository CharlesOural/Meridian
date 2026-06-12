# meridian_add_gtest(<name> <sources...>): register a GoogleTest under BUILD_TESTING
# and apply the house warning flags. The caller links the package library afterwards.
macro(meridian_add_gtest _name)
  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(${_name} ${ARGN})
    if(TARGET ${_name})
      meridian_apply_warnings(${_name})
    endif()
  endif()
endmacro()
