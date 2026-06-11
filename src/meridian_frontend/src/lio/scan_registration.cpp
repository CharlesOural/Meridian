#include "lio/scan_registration.hpp"

namespace meridian::lio {

ScanRegistration::ScanRegistration(const FrontendLio& cfg) : cfg_(cfg) {}

std::vector<Eigen::Vector3d> ScanRegistration::deskew(const LidarScan& scan,
                                                      const Pose& T_body_lidar,
                                                      const Eigen::Vector3d& omega_body,
                                                      const Eigen::Vector3d& vel_body,
                                                      Timestamp t_end) const {
  // TODO(lio): implemented in a later lane
  (void)scan;
  (void)T_body_lidar;
  (void)omega_body;
  (void)vel_body;
  (void)t_end;
  return {};
}

RegistrationResult ScanRegistration::registerScan(
    const std::vector<Eigen::Vector3d>& keypoints_body, const VoxelGridMap& map,
    const Pose& T_world_body_guess, const MotionPrior& prior) const {
  // TODO(lio): implemented in a later lane
  (void)keypoints_body;
  (void)map;
  (void)prior;
  RegistrationResult out;
  out.pose = T_world_body_guess;
  return out;
}

}  // namespace meridian::lio
