#pragma once

namespace meridian::core {

struct Vec3d final {
  double x{};
  double y{};
  double z{};

  [[nodiscard]] bool isFinite() const noexcept;
  friend bool operator==(const Vec3d&, const Vec3d&) noexcept = default;
};

// Unit quaternion whose rotation maps vectors from a child frame into its
// parent frame. Construction rejects non-finite and non-unit coefficients.
class Quaterniond final {
public:
  constexpr Quaterniond() noexcept = default;
  Quaterniond(double w, double x, double y, double z);

  [[nodiscard]] constexpr double w() const noexcept { return w_; }
  [[nodiscard]] constexpr double x() const noexcept { return x_; }
  [[nodiscard]] constexpr double y() const noexcept { return y_; }
  [[nodiscard]] constexpr double z() const noexcept { return z_; }
  [[nodiscard]] bool isFinite() const noexcept;
  [[nodiscard]] double squaredNorm() const noexcept;

  friend bool operator==(const Quaterniond&, const Quaterniond&) noexcept = default;

private:
  double w_{1.0};
  double x_{};
  double y_{};
  double z_{};
};

// A rigid transform from a child frame into its parent frame. Translation is
// expressed in the parent frame and rotation follows Quaterniond's convention.
class Pose3d final {
public:
  constexpr Pose3d() noexcept = default;
  Pose3d(Vec3d translation, Quaterniond rotation);

  [[nodiscard]] constexpr const Vec3d& translation() const noexcept { return translation_; }
  [[nodiscard]] constexpr const Quaterniond& rotation() const noexcept { return rotation_; }

  friend bool operator==(const Pose3d&, const Pose3d&) noexcept = default;

private:
  Vec3d translation_{};
  Quaterniond rotation_{};
};

}  // namespace meridian::core
