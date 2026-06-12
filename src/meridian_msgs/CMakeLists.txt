cmake_minimum_required(VERSION 3.22)
project(meridian_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Telemetry.msg"
  "msg/StageTiming.msg"
  "msg/Event.msg"
  "srv/SetDebugKey.srv"
  "srv/SetLogLevel.srv"
  "srv/SetTelemetryRate.srv"
  "srv/ResetTiming.srv"
  "srv/SnapshotForensics.srv"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
