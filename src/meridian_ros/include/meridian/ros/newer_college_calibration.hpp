#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "meridian/core/calibration.hpp"
#include "meridian/core/result.hpp"

namespace meridian::ros {

enum class NewerCollegeCollection {
  Collection1,
  Collection2,
};

struct NewerCollegeCalibrationFiles {
  std::vector<std::filesystem::path> camera_chain_files;
  std::filesystem::path lidar_imu_transforms_file;
};

enum class CalibrationLoadErrorCode {
  FileNotFound,
  AmbiguousFile,
  YamlParseFailure,
  MissingValue,
  InvalidValue,
  UnsupportedCameraModel,
  UnsupportedDistortionModel,
  InvalidTransform,
  DuplicateCamera,
  InconsistentExtrinsics,
  InvalidBundle,
};

struct CalibrationLoadError {
  CalibrationLoadErrorCode code{CalibrationLoadErrorCode::InvalidValue};
  std::filesystem::path file;
  std::string field;
  std::string detail;
};

// Resolves the three Kalibr camera-chain files and the shared lidar/IMU
// transform file in the official Newer College multicamera calibration tree.
[[nodiscard]] core::Result<NewerCollegeCalibrationFiles, CalibrationLoadError>
resolveNewerCollegeCalibrationFiles(
    const std::filesystem::path& calibration_root,
    NewerCollegeCollection collection);

// The official NCD calibration archive does not contain an IMU Allan-variance
// noise profile.  Requiring a typed profile here prevents silently inventing
// dataset-specific noise numbers; the estimator configuration owns that
// separately measured/tuned input.
[[nodiscard]] core::Result<core::CalibrationBundle, CalibrationLoadError>
loadNewerCollegeCalibration(
    const NewerCollegeCalibrationFiles& files, core::CalibrationEpoch epoch,
    const core::ImuCalibration& imu_calibration);

[[nodiscard]] core::Result<core::CalibrationBundle, CalibrationLoadError>
loadNewerCollegeCalibration(
    const std::filesystem::path& calibration_root,
    NewerCollegeCollection collection, core::CalibrationEpoch epoch,
    const core::ImuCalibration& imu_calibration);

}  // namespace meridian::ros
