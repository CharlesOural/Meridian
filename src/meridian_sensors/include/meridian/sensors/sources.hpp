#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "meridian/common/sample.hpp"
#include "meridian/common/time.hpp"
#include "meridian/config/config.hpp"
#include "meridian/sensors/health.hpp"
#include "meridian/sensors/isensor_source.hpp"
#include "meridian/sensors/raw_frames.hpp"
#include "meridian/sensors/sensor_info.hpp"

namespace meridian {

class ClockModel;
class TelemetrySink;

// The live sources convert wire-free POD frames to typed samples and stamp them on the
// Meridian timeline. Each is fed by the wrapper through an ingest_raw(...) call: the
// wrapper's subscription callback builds the POD and hands it over; the source applies
// its per-sensor stamping policy via the borrowed ClockModel, enforces per-sensor
// monotonicity, updates health, and invokes the callback.
//
// Thread-confined: set_callback/start/stop are called from the owning thread, and
// ingest_raw + the callback fire on the wrapper's subscription thread. ClockModel,
// HealthSink, and TelemetrySink pointers are borrowed (non-owning), owned by the
// pipeline.
//
// Confinement invariant: all sources share the single executor ingest thread, so no two
// ingest_raw calls run concurrently. ClockModel access is relaxed from this (it locks
// internally), but per-source state still assumes serialized ingest.

// Single LiDAR source. Stamps from the device's PTP-disciplined clock when locked,
// degrading to the software offset otherwise. Per-point time is copied losslessly with
// no scaling; no range/decimation filtering happens here.
class OusterLidarSource final : public ISensorSource<LidarScan> {
 public:
  OusterLidarSource(SensorInfo info, const LidarSensorConfig& cfg, ClockModel* clock,
                    HealthSink* health, TelemetrySink* telemetry,
                    const TimeHealth& health_cfg,
                    const ValidatorConfig& validator_cfg = ValidatorConfig{});
  ~OusterLidarSource() override;

  void set_callback(Callback cb) override;
  void start() override;
  void stop() override;
  SensorInfo info() const override;

  // Fed by the wrapper with a wire-free sweep. Converts, stamps, validates, and pushes.
  void ingest_raw(const RawLidarFrame& f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Single IMU source. Stamps from a disciplined device clock when available, else falls
// back to host arrival (degraded, ImuNoDeviceClock). Applies the interval-end shift so
// the stamp denotes the end of the integration interval. Acc/gyro pass through raw.
class ImuSource final : public ISensorSource<ImuSample> {
 public:
  ImuSource(SensorInfo info, const ImuSensorConfig& cfg, ClockModel* clock,
            HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
            const ValidatorConfig& validator_cfg = ValidatorConfig{});
  ~ImuSource() override;

  void set_callback(Callback cb) override;
  void start() override;
  void stop() override;
  SensorInfo info() const override;

  void ingest_raw(const RawImuFrame& f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Single camera source. Stamps at mid-exposure: trigger_edge + exposure/2 when a GPIO
// trigger edge matches within the gate, else device/host time + exposure/2. The
// trigger schedule lives in meridian_time; this source consults it through the
// ClockModel/TriggerSchedule seam.
//
// The trigger-schedule lookup is not yet wired (meridian_time does not expose a
// scheduled-edge query); until it is, the source stamps on SwOffset/ArrivalOnly and
// applies the mid-exposure shift. This is the camera-trigger seam.
class CameraSource final : public ISensorSource<CameraFrame> {
 public:
  CameraSource(SensorInfo info, const CameraSensorConfig& cfg, ClockModel* clock,
               HealthSink* health, TelemetrySink* telemetry,
               const TimeHealth& health_cfg,
               const ValidatorConfig& validator_cfg = ValidatorConfig{});
  ~CameraSource() override;

  void set_callback(Callback cb) override;
  void start() override;
  void stop() override;
  SensorInfo info() const override;

  void ingest_raw(const RawCameraFrame& f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Single GNSS source. Stamps the fix to the PPS edge it references (HwPps); without a
// PPS reference it degrades to the software offset against the host clock and raises
// GnssPpsLost (the rig loses its absolute time anchor). Reports the raw geodetic fix
// and ENU covariance; no metric/ENU conversion and no fix-quality gating here.
class GnssSource final : public ISensorSource<GnssFix> {
 public:
  GnssSource(SensorInfo info, const GnssSensorConfig& cfg, ClockModel* clock,
             HealthSink* health, TelemetrySink* telemetry, const TimeHealth& health_cfg,
             const ValidatorConfig& validator_cfg = ValidatorConfig{});
  ~GnssSource() override;

  void set_callback(Callback cb) override;
  void start() override;
  void stop() override;
  SensorInfo info() const override;

  void ingest_raw(const RawGnssFrame& f);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace meridian
