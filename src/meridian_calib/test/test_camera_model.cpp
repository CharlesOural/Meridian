#include "meridian/calib/camera_model.hpp"

#include <cmath>

#include <gtest/gtest.h>

#include "meridian/calib/intrinsics.hpp"

using meridian::CameraModel;
using meridian::IntrinsicsCamera;

namespace {

IntrinsicsCamera pinholeIntrinsics() {
  IntrinsicsCamera k;
  k.fx = 400.0;
  k.fy = 410.0;
  k.cx = 320.0;
  k.cy = 240.0;
  k.model = IntrinsicsCamera::Distortion::None;
  k.coeffs = {0, 0, 0, 0, 0};
  k.width = 640;
  k.height = 480;
  return k;
}

IntrinsicsCamera radtanIntrinsics() {
  IntrinsicsCamera k = pinholeIntrinsics();
  k.model = IntrinsicsCamera::Distortion::RadTan;
  // Moderate barrel + small tangential, the regime real wide lenses live in.
  k.coeffs = {-0.28, 0.07, 0.0006, -0.0004, 0.001};
  return k;
}

// Numeric 2x3 projection Jacobian by central differences.
Eigen::Matrix<double, 2, 3> numericJacobian(const CameraModel& cam,
                                            const Eigen::Vector3d& p) {
  Eigen::Matrix<double, 2, 3> J;
  const double h = 1e-6;
  for (int c = 0; c < 3; ++c) {
    Eigen::Vector3d pp = p, pm = p;
    pp[c] += h;
    pm[c] -= h;
    Eigen::Vector2d up, um;
    EXPECT_TRUE(cam.project(pp, &up));
    EXPECT_TRUE(cam.project(pm, &um));
    J.col(c) = (up - um) / (2.0 * h);
  }
  return J;
}

}  // namespace

TEST(CameraModel, DefaultIsInvalid) {
  CameraModel cam;
  EXPECT_FALSE(cam.valid());
  Eigen::Vector2d uv;
  EXPECT_FALSE(cam.project(Eigen::Vector3d(0, 0, 1), &uv));
}

TEST(CameraModel, ZeroFocalIsInvalidGate) {
  IntrinsicsCamera k = pinholeIntrinsics();
  k.fx = 0.0;  // the visual-disable gate keys off this exact condition
  CameraModel cam(k);
  EXPECT_FALSE(cam.valid());
}

TEST(CameraModel, ValidWhenFocalsPositive) {
  CameraModel cam(pinholeIntrinsics());
  EXPECT_TRUE(cam.valid());
  EXPECT_EQ(cam.width(), 640);
  EXPECT_EQ(cam.height(), 480);
}

TEST(CameraModel, PinholeProjectMatchesClosedForm) {
  CameraModel cam(pinholeIntrinsics());
  const Eigen::Vector3d p(0.3, -0.2, 2.0);
  Eigen::Vector2d uv;
  ASSERT_TRUE(cam.project(p, &uv));
  EXPECT_NEAR(uv.x(), 400.0 * (0.3 / 2.0) + 320.0, 1e-9);
  EXPECT_NEAR(uv.y(), 410.0 * (-0.2 / 2.0) + 240.0, 1e-9);
}

TEST(CameraModel, RejectsPointsBehindAndOffImage) {
  CameraModel cam(pinholeIntrinsics());
  Eigen::Vector2d uv;
  EXPECT_FALSE(cam.project(Eigen::Vector3d(0, 0, -1), &uv));   // behind
  EXPECT_FALSE(cam.project(Eigen::Vector3d(0, 0, 0), &uv));    // at plane
  EXPECT_FALSE(cam.project(Eigen::Vector3d(10, 0, 1), &uv));   // far off right edge
}

TEST(CameraModel, PinholeRoundTrip) {
  CameraModel cam(pinholeIntrinsics());
  for (double x : {-0.4, -0.1, 0.0, 0.2, 0.5}) {
    for (double y : {-0.3, 0.0, 0.25}) {
      const Eigen::Vector3d p(x, y, 1.0);
      Eigen::Vector2d uv;
      ASSERT_TRUE(cam.project(p, &uv));
      const Eigen::Vector3d ray = cam.unproject(uv);
      ASSERT_GT(ray.z(), 0.0);
      EXPECT_NEAR(ray.x() / ray.z(), x, 1e-9);
      EXPECT_NEAR(ray.y() / ray.z(), y, 1e-9);
    }
  }
}

TEST(CameraModel, RadtanRoundTrip) {
  CameraModel cam(radtanIntrinsics());
  for (double x : {-0.35, -0.1, 0.0, 0.15, 0.4}) {
    for (double y : {-0.3, -0.05, 0.0, 0.2}) {
      const Eigen::Vector3d p(x, y, 1.0);
      Eigen::Vector2d uv;
      ASSERT_TRUE(cam.project(p, &uv));
      const Eigen::Vector3d ray = cam.unproject(uv);
      ASSERT_GT(ray.z(), 0.0);
      // project->unproject must recover the original normalized ray direction.
      EXPECT_NEAR(ray.x() / ray.z(), x, 1e-7);
      EXPECT_NEAR(ray.y() / ray.z(), y, 1e-7);
    }
  }
}

TEST(CameraModel, RadtanActuallyDistorts) {
  CameraModel cam(radtanIntrinsics());
  CameraModel pin(pinholeIntrinsics());
  // A point well off-axis must land at a different pixel than the pinhole model;
  // otherwise the distortion branch is silently a no-op.
  const Eigen::Vector3d p(0.4, 0.3, 1.0);
  Eigen::Vector2d uv_d, uv_p;
  ASSERT_TRUE(cam.project(p, &uv_d));
  ASSERT_TRUE(pin.project(p, &uv_p));
  EXPECT_GT((uv_d - uv_p).norm(), 1.0);
}

TEST(CameraModel, JacobianPinholeVsNumeric) {
  CameraModel cam(pinholeIntrinsics());
  for (const Eigen::Vector3d& p :
       {Eigen::Vector3d(0.2, -0.1, 2.0), Eigen::Vector3d(-0.3, 0.25, 3.5),
        Eigen::Vector3d(0.05, 0.05, 1.2)}) {
    const auto Ja = cam.projectJacobian(p);
    const auto Jn = numericJacobian(cam, p);
    EXPECT_LT((Ja - Jn).cwiseAbs().maxCoeff(), 1e-4)
        << "analytic\n" << Ja << "\nnumeric\n" << Jn;
  }
}

TEST(CameraModel, JacobianRadtanVsNumeric) {
  CameraModel cam(radtanIntrinsics());
  for (const Eigen::Vector3d& p :
       {Eigen::Vector3d(0.2, -0.1, 2.0), Eigen::Vector3d(-0.3, 0.25, 3.5),
        Eigen::Vector3d(0.3, 0.2, 1.5), Eigen::Vector3d(0.05, 0.05, 1.2)}) {
    const auto Ja = cam.projectJacobian(p);
    const auto Jn = numericJacobian(cam, p);
    EXPECT_LT((Ja - Jn).cwiseAbs().maxCoeff(), 1e-3)
        << "at " << p.transpose() << "\nanalytic\n" << Ja << "\nnumeric\n" << Jn;
  }
}

TEST(CameraModel, EquidistantIsInvalidNotThrow) {
  // An equidistant/fisheye lens is unusable by this pinhole projector. It must
  // gate the visual stage off (valid()==false), never crash construction.
  IntrinsicsCamera k = pinholeIntrinsics();  // positive focals
  k.model = IntrinsicsCamera::Distortion::Equidistant;
  CameraModel cam(k);
  EXPECT_FALSE(cam.valid());
  Eigen::Vector2d uv;
  EXPECT_FALSE(cam.project(Eigen::Vector3d(0, 0, 1), &uv));
}

TEST(CameraModel, ZeroCoeffRadtanDegradesToPinhole) {
  IntrinsicsCamera k = pinholeIntrinsics();
  k.model = IntrinsicsCamera::Distortion::RadTan;  // RadTan tag but zero coeffs
  CameraModel cam(k);
  CameraModel pin(pinholeIntrinsics());
  const Eigen::Vector3d p(0.4, 0.3, 1.7);
  Eigen::Vector2d a, b;
  ASSERT_TRUE(cam.project(p, &a));
  ASSERT_TRUE(pin.project(p, &b));
  EXPECT_LT((a - b).norm(), 1e-12);
}
