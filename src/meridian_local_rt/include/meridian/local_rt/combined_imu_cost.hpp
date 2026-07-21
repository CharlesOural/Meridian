#pragma once

#include <ceres/sized_cost_function.h>

#include <memory>

#include "meridian/local_rt/combined_preintegration.hpp"

namespace meridian::local_rt {

// Private-backend adapter for the 15D GTSAM combined IMU residual.
// Pose blocks are [p, qxyzw]; motion blocks are [v, gyro_bias, accel_bias].
class CombinedImuCost final : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
  explicit CombinedImuCost(const CombinedPreintegration& preintegration);
  ~CombinedImuCost() override;

  CombinedImuCost(const CombinedImuCost&) = delete;
  CombinedImuCost& operator=(const CombinedImuCost&) = delete;

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian::local_rt
