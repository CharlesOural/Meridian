#include "meridian/ros/newer_college_calibration.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

namespace meridian::ros {
namespace {

using core::BaseFromImuTransform;
using core::CalibrationBundle;
using core::CalibrationEpoch;
using core::CameraCalibration;
using core::CameraId;
using core::CameraTimestampReference;
using core::CameraTimingCalibration;
using core::Duration;
using core::ImageDimensions;
using core::ImuCalibration;
using core::ImuFromCameraTransform;
using core::ImuFromLidarTransform;
using core::LidarCalibration;
using core::LidarId;
using core::LidarPointTimeConvention;
using core::LidarSweepTimestampReference;
using core::LidarTimingCalibration;
using core::PinholeEquidistantCameraModel;
using core::Pose3d;
using core::Result;

struct DecodeFailure {
  CalibrationLoadError error;
};

[[noreturn]] void fail(CalibrationLoadErrorCode code,
                       const std::filesystem::path& file, std::string field,
                       std::string detail) {
  throw DecodeFailure{
      CalibrationLoadError{code, file, std::move(field), std::move(detail)}};
}

[[nodiscard]] CalibrationLoadError yamlError(
    const std::filesystem::path& file, const YAML::Exception& exception) {
  return CalibrationLoadError{CalibrationLoadErrorCode::YamlParseFailure, file,
                              {}, exception.what()};
}

[[nodiscard]] YAML::Node requiredNode(const YAML::Node& parent,
                                      const std::filesystem::path& file,
                                      const std::string& scope,
                                      const std::string& key) {
  const YAML::Node value = parent[key];
  if (!value.IsDefined()) {
    fail(CalibrationLoadErrorCode::MissingValue, file, scope + "." + key,
         "required YAML value is missing");
  }
  return value;
}

template <typename Value>
[[nodiscard]] Value scalar(const YAML::Node& parent,
                           const std::filesystem::path& file,
                           const std::string& scope, const std::string& key) {
  const YAML::Node value = requiredNode(parent, file, scope, key);
  if (!value.IsScalar()) {
    fail(CalibrationLoadErrorCode::InvalidValue, file, scope + "." + key,
         "expected a scalar value");
  }
  try {
    return value.as<Value>();
  } catch (const YAML::Exception& exception) {
    fail(CalibrationLoadErrorCode::InvalidValue, file, scope + "." + key,
         exception.what());
  }
}

template <std::size_t Size>
[[nodiscard]] std::array<double, Size> doubleArray(
    const YAML::Node& parent, const std::filesystem::path& file,
    const std::string& scope, const std::string& key) {
  const YAML::Node values = requiredNode(parent, file, scope, key);
  if (!values.IsSequence() || values.size() != Size) {
    fail(CalibrationLoadErrorCode::InvalidValue, file, scope + "." + key,
         "expected a sequence with " + std::to_string(Size) + " entries");
  }

  std::array<double, Size> decoded{};
  for (std::size_t index = 0; index < Size; ++index) {
    try {
      decoded[index] = values[index].as<double>();
    } catch (const YAML::Exception& exception) {
      fail(CalibrationLoadErrorCode::InvalidValue, file,
           scope + "." + key + "[" + std::to_string(index) + "]",
           exception.what());
    }
    if (!std::isfinite(decoded[index])) {
      fail(CalibrationLoadErrorCode::InvalidValue, file,
           scope + "." + key + "[" + std::to_string(index) + "]",
           "value must be finite");
    }
  }
  return decoded;
}

[[nodiscard]] Pose3d matrixPose(const YAML::Node& parent,
                                const std::filesystem::path& file,
                                const std::string& scope,
                                const std::string& key) {
  const YAML::Node rows = requiredNode(parent, file, scope, key);
  if (!rows.IsSequence() || rows.size() != 4U) {
    fail(CalibrationLoadErrorCode::InvalidTransform, file, scope + "." + key,
         "expected a 4x4 transform matrix");
  }

  Eigen::Matrix4d matrix;
  for (std::size_t row = 0; row < 4U; ++row) {
    if (!rows[row].IsSequence() || rows[row].size() != 4U) {
      fail(CalibrationLoadErrorCode::InvalidTransform, file,
           scope + "." + key, "expected a 4x4 transform matrix");
    }
    for (std::size_t column = 0; column < 4U; ++column) {
      try {
        matrix(static_cast<Eigen::Index>(row),
               static_cast<Eigen::Index>(column)) =
            rows[row][column].as<double>();
      } catch (const YAML::Exception& exception) {
        fail(CalibrationLoadErrorCode::InvalidTransform, file,
             scope + "." + key, exception.what());
      }
    }
  }

  if (!matrix.allFinite() ||
      (matrix.row(3) - Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0)).norm() >
          1.0e-8) {
    fail(CalibrationLoadErrorCode::InvalidTransform, file, scope + "." + key,
         "transform contains non-finite values or an invalid homogeneous row");
  }

  const Eigen::Matrix3d rotation = matrix.block<3, 3>(0, 0);
  if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() >
          1.0e-5 ||
      std::abs(rotation.determinant() - 1.0) > 1.0e-5) {
    fail(CalibrationLoadErrorCode::InvalidTransform, file, scope + "." + key,
         "rotation is not a proper orthonormal matrix");
  }

  return Pose3d(Sophus::SO3d(rotation), matrix.block<3, 1>(0, 3));
}

[[nodiscard]] Pose3d quaternionPose(const YAML::Node& parent,
                                    const std::filesystem::path& file,
                                    const std::string& scope) {
  const std::array<double, 3> translation =
      doubleArray<3>(parent, file, scope, "translation");
  const std::array<double, 4> xyzw =
      doubleArray<4>(parent, file, scope, "rotation");

  Eigen::Quaterniond quaternion(xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  const double norm = quaternion.norm();
  if (!std::isfinite(norm) || norm < 1.0e-12 || std::abs(norm - 1.0) > 1.0e-4) {
    fail(CalibrationLoadErrorCode::InvalidTransform, file, scope + ".rotation",
         "quaternion must be finite and unit length");
  }
  quaternion.normalize();
  return Pose3d(Sophus::SO3d(quaternion),
                Eigen::Vector3d(translation[0], translation[1],
                                translation[2]));
}

[[nodiscard]] CameraId cameraIdFromTopic(
    const std::string& topic, const std::filesystem::path& file,
    const std::string& scope) {
  const std::size_t marker = topic.rfind("cam");
  if (marker == std::string::npos || marker + 3U == topic.size()) {
    fail(CalibrationLoadErrorCode::InvalidValue, file, scope + ".rostopic",
         "camera topic must end in cam<integer>");
  }

  const char* begin = topic.data() + marker + 3U;
  const char* end = topic.data() + topic.size();
  std::uint64_t value{};
  const std::from_chars_result parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc{} || parsed.ptr != end) {
    fail(CalibrationLoadErrorCode::InvalidValue, file, scope + ".rostopic",
         "camera topic must end in cam<integer>");
  }
  return CameraId(value);
}

[[nodiscard]] std::string cameraName(CameraId id) {
  return "cam" + std::to_string(id.value());
}

[[nodiscard]] Result<std::vector<CameraCalibration>, CalibrationLoadError>
decodeCameraFile(const std::filesystem::path& file) {
  try {
    YAML::Node root;
    try {
      root = YAML::LoadFile(file.string());
    } catch (const YAML::Exception& exception) {
      return Result<std::vector<CameraCalibration>,
                    CalibrationLoadError>::failure(yamlError(file, exception));
    }
    if (!root.IsMap()) {
      return Result<std::vector<CameraCalibration>,
                    CalibrationLoadError>::failure(CalibrationLoadError{
          CalibrationLoadErrorCode::InvalidValue, file, {},
          "camera calibration root must be a YAML map"});
    }

    std::vector<CameraCalibration> cameras;
    for (const auto& entry : root) {
      const std::string scope = entry.first.as<std::string>();
      const YAML::Node camera = entry.second;
      if (!camera.IsMap()) {
        fail(CalibrationLoadErrorCode::InvalidValue, file, scope,
             "camera entry must be a YAML map");
      }

      const std::string projection_model =
          scalar<std::string>(camera, file, scope, "camera_model");
      if (projection_model != "pinhole") {
        fail(CalibrationLoadErrorCode::UnsupportedCameraModel, file,
             scope + ".camera_model", projection_model);
      }
      const std::string distortion_model =
          scalar<std::string>(camera, file, scope, "distortion_model");
      if (distortion_model != "equidistant") {
        fail(CalibrationLoadErrorCode::UnsupportedDistortionModel, file,
             scope + ".distortion_model", distortion_model);
      }

      const std::array<double, 4> intrinsics =
          doubleArray<4>(camera, file, scope, "intrinsics");
      const std::array<double, 4> distortion =
          doubleArray<4>(camera, file, scope, "distortion_coeffs");
      const YAML::Node resolution =
          requiredNode(camera, file, scope, "resolution");
      if (!resolution.IsSequence() || resolution.size() != 2U) {
        fail(CalibrationLoadErrorCode::InvalidValue, file,
             scope + ".resolution", "expected [width, height]");
      }
      const std::uint32_t width = resolution[0].as<std::uint32_t>();
      const std::uint32_t height = resolution[1].as<std::uint32_t>();
      const std::string topic =
          scalar<std::string>(camera, file, scope, "rostopic");
      const CameraId id = cameraIdFromTopic(topic, file, scope);

      // Kalibr stores T_cam_imu. Meridian exposes the opposite direction;
      // the inversion here is intentional and covered by the dataset tests.
      const Pose3d T_cam_imu = matrixPose(camera, file, scope, "T_cam_imu");
      const double offset_seconds =
          scalar<double>(camera, file, scope, "timeshift_cam_imu");
      if (!std::isfinite(offset_seconds) ||
          std::abs(offset_seconds) >
              static_cast<double>(std::numeric_limits<std::int64_t>::max()) /
                  1.0e9) {
        fail(CalibrationLoadErrorCode::InvalidValue, file,
             scope + ".timeshift_cam_imu", "time offset is not representable");
      }
      const auto offset_nanoseconds =
          static_cast<std::int64_t>(std::llround(offset_seconds * 1.0e9));

      cameras.emplace_back(
          id, cameraName(id), topic,
          PinholeEquidistantCameraModel(
              intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
              distortion, ImageDimensions{width, height}),
          ImuFromCameraTransform(T_cam_imu.inverse()),
          CameraTimingCalibration(CameraTimestampReference::SensorReportedImageTime,
                                  Duration{offset_nanoseconds}));
    }
    if (cameras.empty()) {
      fail(CalibrationLoadErrorCode::MissingValue, file, {},
           "camera file contains no cameras");
    }
    return Result<std::vector<CameraCalibration>,
                  CalibrationLoadError>::success(std::move(cameras));
  } catch (const DecodeFailure& failure) {
    return Result<std::vector<CameraCalibration>,
                  CalibrationLoadError>::failure(failure.error);
  } catch (const YAML::Exception& exception) {
    return Result<std::vector<CameraCalibration>,
                  CalibrationLoadError>::failure(yamlError(file, exception));
  }
}

struct RigExtrinsics {
  Pose3d T_imu_lidar;
  Pose3d T_base_imu;
  Pose3d T_base_lidar;
};

[[nodiscard]] Result<RigExtrinsics, CalibrationLoadError> decodeRigExtrinsics(
    const std::filesystem::path& file) {
  try {
    YAML::Node root;
    try {
      root = YAML::LoadFile(file.string());
    } catch (const YAML::Exception& exception) {
      return Result<RigExtrinsics, CalibrationLoadError>::failure(
          yamlError(file, exception));
    }
    const Pose3d T_imu_lidar = quaternionPose(
        requiredNode(root, file, {}, "os_sensor_to_as_imu"), file,
        "os_sensor_to_as_imu");
    const Pose3d T_base_imu = quaternionPose(
        requiredNode(root, file, {}, "as_imu_to_base"), file,
        "as_imu_to_base");
    const Pose3d T_base_lidar = quaternionPose(
        requiredNode(root, file, {}, "os_sensor_to_base"), file,
        "os_sensor_to_base");

    const Pose3d composed = T_base_imu * T_imu_lidar;
    if ((composed.matrix() - T_base_lidar.matrix()).norm() > 1.0e-7) {
      return Result<RigExtrinsics, CalibrationLoadError>::failure(
          CalibrationLoadError{
              CalibrationLoadErrorCode::InconsistentExtrinsics, file,
              "os_sensor_to_base",
              "T_base_imu * T_imu_lidar disagrees with T_base_lidar"});
    }
    return Result<RigExtrinsics, CalibrationLoadError>::success(
        RigExtrinsics{T_imu_lidar, T_base_imu, T_base_lidar});
  } catch (const DecodeFailure& failure) {
    return Result<RigExtrinsics, CalibrationLoadError>::failure(failure.error);
  } catch (const YAML::Exception& exception) {
    return Result<RigExtrinsics, CalibrationLoadError>::failure(
        yamlError(file, exception));
  }
}

[[nodiscard]] bool hasSuffix(const std::string& value,
                             const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

[[nodiscard]] Result<std::filesystem::path, CalibrationLoadError>
findCameraChain(const std::filesystem::path& directory) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error)) {
    return Result<std::filesystem::path, CalibrationLoadError>::failure(
        CalibrationLoadError{CalibrationLoadErrorCode::FileNotFound, directory,
                             {}, "camera calibration directory is missing"});
  }

  std::vector<std::filesystem::path> matches;
  for (std::filesystem::directory_iterator iterator(directory, error), end;
       iterator != end && !error; iterator.increment(error)) {
    if (iterator->is_regular_file(error) &&
        hasSuffix(iterator->path().filename().string(),
                  "camchain-imucam.yaml")) {
      matches.push_back(iterator->path());
    }
  }
  if (error) {
    return Result<std::filesystem::path, CalibrationLoadError>::failure(
        CalibrationLoadError{CalibrationLoadErrorCode::FileNotFound, directory,
                             {}, error.message()});
  }
  std::sort(matches.begin(), matches.end());
  if (matches.empty()) {
    return Result<std::filesystem::path, CalibrationLoadError>::failure(
        CalibrationLoadError{CalibrationLoadErrorCode::FileNotFound, directory,
                             {}, "no *camchain-imucam.yaml file found"});
  }
  if (matches.size() != 1U) {
    return Result<std::filesystem::path, CalibrationLoadError>::failure(
        CalibrationLoadError{CalibrationLoadErrorCode::AmbiguousFile, directory,
                             {}, "multiple *camchain-imucam.yaml files found"});
  }
  return Result<std::filesystem::path, CalibrationLoadError>::success(
      matches.front());
}

[[nodiscard]] std::string collectionDirectory(
    NewerCollegeCollection collection) {
  switch (collection) {
    case NewerCollegeCollection::Collection1:
      return "collection1";
    case NewerCollegeCollection::Collection2:
      return "collection2";
  }
  return {};
}

}  // namespace

Result<NewerCollegeCalibrationFiles, CalibrationLoadError>
resolveNewerCollegeCalibrationFiles(
    const std::filesystem::path& calibration_root,
    NewerCollegeCollection collection) {
  NewerCollegeCalibrationFiles files;
  const std::filesystem::path collection_root =
      calibration_root / collectionDirectory(collection);
  for (const char* camera_group : {"cam0-1", "cam3", "cam4"}) {
    auto camera_file = findCameraChain(collection_root / camera_group);
    if (!camera_file) {
      return Result<NewerCollegeCalibrationFiles,
                    CalibrationLoadError>::failure(camera_file.error());
    }
    files.camera_chain_files.push_back(std::move(camera_file).value());
  }
  files.lidar_imu_transforms_file =
      calibration_root / "os_imu_lidar_transforms.yaml";
  std::error_code error;
  if (!std::filesystem::is_regular_file(files.lidar_imu_transforms_file,
                                        error)) {
    return Result<NewerCollegeCalibrationFiles,
                  CalibrationLoadError>::failure(CalibrationLoadError{
        CalibrationLoadErrorCode::FileNotFound,
        files.lidar_imu_transforms_file, {},
        error ? error.message() : "lidar/IMU transform file is missing"});
  }
  return Result<NewerCollegeCalibrationFiles, CalibrationLoadError>::success(
      std::move(files));
}

Result<CalibrationBundle, CalibrationLoadError> loadNewerCollegeCalibration(
    const NewerCollegeCalibrationFiles& files, CalibrationEpoch epoch,
    const ImuCalibration& imu_calibration) {
  if (files.camera_chain_files.empty()) {
    return Result<CalibrationBundle, CalibrationLoadError>::failure(
        CalibrationLoadError{CalibrationLoadErrorCode::MissingValue, {},
                             "camera_chain_files",
                             "at least one camera chain is required"});
  }

  std::vector<CameraCalibration> cameras;
  for (const std::filesystem::path& camera_file : files.camera_chain_files) {
    auto decoded = decodeCameraFile(camera_file);
    if (!decoded) {
      return Result<CalibrationBundle, CalibrationLoadError>::failure(
          decoded.error());
    }
    std::vector<CameraCalibration> file_cameras = std::move(decoded).value();
    for (CameraCalibration& camera : file_cameras) {
      const auto duplicate = std::find_if(
          cameras.begin(), cameras.end(), [&camera](const auto& existing) {
            return existing.id() == camera.id();
          });
      if (duplicate != cameras.end()) {
        return Result<CalibrationBundle, CalibrationLoadError>::failure(
            CalibrationLoadError{CalibrationLoadErrorCode::DuplicateCamera,
                                 camera_file, camera.name(),
                                 "camera id appears in more than one file"});
      }
      cameras.push_back(std::move(camera));
    }
  }
  std::sort(cameras.begin(), cameras.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.id().value() < rhs.id().value();
            });

  auto extrinsics = decodeRigExtrinsics(files.lidar_imu_transforms_file);
  if (!extrinsics) {
    return Result<CalibrationBundle, CalibrationLoadError>::failure(
        extrinsics.error());
  }
  const RigExtrinsics rig = std::move(extrinsics).value();

  LidarCalibration lidar(
      LidarId(0U), "os_sensor", "/os_cloud_node/points",
      ImuFromLidarTransform(rig.T_imu_lidar),
      LidarTimingCalibration(LidarSweepTimestampReference::SweepStart,
                             LidarPointTimeConvention::OffsetFromSweepTimestamp));
  auto bundle = CalibrationBundle::create(
      epoch, imu_calibration, BaseFromImuTransform(rig.T_base_imu),
      std::move(lidar), std::move(cameras));
  if (!bundle) {
    return Result<CalibrationBundle, CalibrationLoadError>::failure(
        CalibrationLoadError{
            CalibrationLoadErrorCode::InvalidBundle, {}, {},
            "core calibration validation failed with code " +
                std::to_string(static_cast<int>(bundle.error()))});
  }
  return Result<CalibrationBundle, CalibrationLoadError>::success(
      std::move(bundle).value());
}

Result<CalibrationBundle, CalibrationLoadError> loadNewerCollegeCalibration(
    const std::filesystem::path& calibration_root,
    NewerCollegeCollection collection, CalibrationEpoch epoch,
    const ImuCalibration& imu_calibration) {
  auto files = resolveNewerCollegeCalibrationFiles(calibration_root, collection);
  if (!files) {
    return Result<CalibrationBundle, CalibrationLoadError>::failure(
        files.error());
  }
  return loadNewerCollegeCalibration(std::move(files).value(), epoch,
                                     imu_calibration);
}

}  // namespace meridian::ros
