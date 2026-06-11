# Centralised find_package() calls so every package discovers deps identically.
# Eigen + Sophus are universal; heavier deps are gated by MERIDIAN_NEED_* flags the
# consuming package sets before include(MeridianDeps).

find_package(Eigen3 3.4 REQUIRED NO_MODULE)    # -> Eigen3::Eigen
find_package(Sophus REQUIRED)                   # -> Sophus::Sophus (header-only, needs Eigen)

if(MERIDIAN_NEED_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video calib3d)
endif()
if(MERIDIAN_NEED_GTSAM)
  find_package(GTSAM 4.2 REQUIRED)              # -> gtsam
endif()
if(MERIDIAN_NEED_PCL)
  find_package(PCL 1.12 REQUIRED COMPONENTS common io filters)
endif()
if(MERIDIAN_NEED_YAMLCPP)
  find_package(yaml-cpp REQUIRED)               # -> yaml-cpp
endif()
