cmake_minimum_required(VERSION 3.22)
project(meridian_cmake NONE)

find_package(ament_cmake REQUIRED)

# Ship the shared .cmake modules; consumers reach them through the CONFIG_EXTRAS
# below, which puts this directory on CMAKE_MODULE_PATH at find_package() time.
install(DIRECTORY cmake/ DESTINATION share/${PROJECT_NAME}/cmake)

ament_package(CONFIG_EXTRAS "meridian_cmake-extras.cmake")
