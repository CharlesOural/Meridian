// Read-only dump of an SBG Ellipse's stored configuration.
//
// The ROS driver applies its YAML as a complete device image: any key absent from
// the file is written to flash as the driver's compiled-in default. Recovering from
// that requires knowing what the device actually holds, which no driver log reports.
// This issues only SBG_ECOM_CMD_*_GET commands -- it never sets, never saves, and so
// never reboots the unit.
//
//   sbg_probe [port] [baud]        defaults: /dev/ttyUSB0 921600

#include <stdio.h>
#include <stdlib.h>

#include <sbgCommon.h>
#include <version/sbgVersion.h>
#include <sbgEComLib.h>

static const char *gnssModelName(SbgEComGnssModelsStdIds id)
{
  switch (id)
  {
    case SBG_ECOM_GNSS_MODEL_INTERNAL:            return "INTERNAL (internal receiver; default for ELLIPSE-N/-D)";
    case SBG_ECOM_GNSS_MODEL_NMEA:                return "NMEA (external GNSS via NMEA; for ELLIPSE-E)";
    case SBG_ECOM_GNSS_MODEL_UBLOX_GPS_BEIDOU:    return "UBLOX_GPS_BEIDOU";
    case SBG_ECOM_GNSS_MODEL_UBLOX_EXTERNAL:      return "UBLOX_EXTERNAL (passive)";
    case SBG_ECOM_GNSS_MODEL_NOVATEL_EXTERNAL:    return "NOVATEL_EXTERNAL (passive)";
    case SBG_ECOM_GNSS_MODEL_SEPTENTRIO_EXTERNAL: return "SEPTENTRIO_EXTERNAL (passive)";
    case SBG_ECOM_GNSS_MODEL_RESERVED_01:
    case SBG_ECOM_GNSS_MODEL_RESERVED_02:
    case SBG_ECOM_GNSS_MODEL_RESERVED_03:
    case SBG_ECOM_GNSS_MODEL_RESERVED_04:         return "RESERVED (do not use)";
    default:                                      return "<unknown>";
  }
}

static const char *portName(SbgEComModulePortAssignment port)
{
  switch (port)
  {
    case SBG_ECOM_MODULE_PORT_A:   return "PORT_A";
    case SBG_ECOM_MODULE_PORT_B:   return "PORT_B";
    case SBG_ECOM_MODULE_PORT_C:   return "PORT_C";
    case SBG_ECOM_MODULE_PORT_D:   return "PORT_D";
    case SBG_ECOM_MODULE_PORT_E:   return "PORT_E";
    case SBG_ECOM_MODULE_INTERNAL: return "INTERNAL";
    case SBG_ECOM_MODULE_DISABLED: return "DISABLED";
    default:                       return "<unknown>";
  }
}

static const char *installModeName(SbgEComGnssInstallationMode mode)
{
  switch (mode)
  {
    case SBG_ECOM_GNSS_INSTALLATION_MODE_SINGLE:       return "SINGLE (secondary antenna unused -> no GNSS heading)";
    case SBG_ECOM_GNSS_INSTALLATION_MODE_DUAL_AUTO:    return "DUAL_AUTO (reserved)";
    case SBG_ECOM_GNSS_INSTALLATION_MODE_DUAL_ROUGH:   return "DUAL_ROUGH (deprecated)";
    case SBG_ECOM_GNSS_INSTALLATION_MODE_DUAL_PRECISE: return "DUAL_PRECISE";
    default:                                           return "<unknown>";
  }
}

static const char *motionProfileName(SbgEComMotionProfileStdIds id)
{
  switch (id)
  {
    case SBG_ECOM_MOTION_PROFILE_GENERAL_PURPOSE: return "GENERAL_PURPOSE";
    case SBG_ECOM_MOTION_PROFILE_AUTOMOTIVE:      return "AUTOMOTIVE";
    case SBG_ECOM_MOTION_PROFILE_MARINE:          return "MARINE";
    case SBG_ECOM_MOTION_PROFILE_AIRPLANE:        return "AIRPLANE";
    case SBG_ECOM_MOTION_PROFILE_HELICOPTER:      return "HELICOPTER";
    default:                                      return "<other/unknown>";
  }
}

static const char *axisName(SbgEComAxisDirection dir)
{
  switch (dir)
  {
    case SBG_ECOM_ALIGNMENT_FORWARD:  return "FORWARD";
    case SBG_ECOM_ALIGNMENT_BACKWARD: return "BACKWARD";
    case SBG_ECOM_ALIGNMENT_LEFT:     return "LEFT";
    case SBG_ECOM_ALIGNMENT_RIGHT:    return "RIGHT";
    case SBG_ECOM_ALIGNMENT_UP:       return "UP";
    case SBG_ECOM_ALIGNMENT_DOWN:     return "DOWN";
    default:                          return "<unknown>";
  }
}

static const char *magModelName(SbgEComMagModelsStdId id)
{
  switch (id)
  {
    case SBG_ECOM_MAG_MODEL_INTERNAL_NORMAL:   return "INTERNAL_NORMAL";
    case SBG_ECOM_MAG_MODEL_INTERNAL_RESERVED: return "INTERNAL_RESERVED (fw v2.x compat)";
    case SBG_ECOM_MAG_MODEL_ECOM_NORMAL:       return "ECOM_NORMAL (external)";
    default:                                   return "<unknown>";
  }
}

static const char *rejectionName(SbgEComRejectionMode mode)
{
  switch (mode)
  {
    case SBG_ECOM_NEVER_ACCEPT_MODE:  return "NEVER_ACCEPT";
    case SBG_ECOM_AUTOMATIC_MODE:     return "AUTOMATIC";
    case SBG_ECOM_ALWAYS_ACCEPT_MODE: return "ALWAYS_ACCEPT";
    default:                          return "<unknown>";
  }
}

// A GET that the firmware does not implement returns an error rather than filling the
// struct; report it and carry on so one unsupported command can't blank the whole dump.
static bool ok(SbgErrorCode code, const char *what)
{
  if (code != SBG_NO_ERROR)
  {
    printf("  !! %-22s GET failed: %s\n", what, sbgErrorCodeToString(code));
    return false;
  }
  return true;
}

static void dumpDeviceInfo(SbgEComHandle *pHandle)
{
  SbgEComDeviceInfo info;
  char              fw[32];
  char              hw[32];
  char              cal[32];

  printf("\n-- device --------------------------------------------------\n");

  if (!ok(sbgEComCmdGetInfo(pHandle, &info), "device info"))
  {
    return;
  }

  sbgVersionToStringEncoded(info.firmwareRev, fw, sizeof(fw));
  sbgVersionToStringEncoded(info.hardwareRev, hw, sizeof(hw));
  sbgVersionToStringEncoded(info.calibationRev, cal, sizeof(cal));

  printf("  product code         %s\n", info.productCode);
  printf("  serial number        %u\n", info.serialNumber);
  printf("  firmware             %s\n", fw);
  printf("  hardware             %s\n", hw);
  printf("  calibration          %s  (%04u-%02u-%02u)\n",
         cal, info.calibrationYear, info.calibrationMonth, info.calibrationDay);
}

static void dumpInitCondition(SbgEComHandle *pHandle)
{
  SbgEComInitConditionConf conf;

  printf("\n-- init conditions -----------------------------------------\n");

  if (!ok(sbgEComCmdSensorGetInitCondition(pHandle, &conf), "init condition"))
  {
    return;
  }

  printf("  latitude             %.6f deg\n", conf.latitude);
  printf("  longitude            %.6f deg\n", conf.longitude);
  printf("  altitude             %.2f m (above ellipsoid)\n", conf.altitude);
  printf("  date                 %04u-%02u-%02u\n", conf.year, conf.month, conf.day);
}

static void dumpMotionProfile(SbgEComHandle *pHandle)
{
  SbgEComMotionProfileStdIds id;

  printf("\n-- motion profile ------------------------------------------\n");

  if (!ok(sbgEComCmdSensorGetMotionProfileId(pHandle, &id), "motion profile"))
  {
    return;
  }

  printf("  motion profile       %u = %s\n", (unsigned)id, motionProfileName(id));
}

static void dumpImuAlignment(SbgEComHandle *pHandle)
{
  SbgEComSensorAlignmentInfo align;
  float                      leverArm[3];

  printf("\n-- IMU alignment / lever arm -------------------------------\n");

  if (!ok(sbgEComCmdSensorGetAlignmentAndLeverArm(pHandle, &align, leverArm), "imu alignment"))
  {
    return;
  }

  printf("  axisDirectionX       %u = %s\n", (unsigned)align.axisDirectionX, axisName(align.axisDirectionX));
  printf("  axisDirectionY       %u = %s\n", (unsigned)align.axisDirectionY, axisName(align.axisDirectionY));
  printf("  misRoll/Pitch/Yaw    [%.6f, %.6f, %.6f] rad\n", align.misRoll, align.misPitch, align.misYaw);
  printf("  lever arm            [%.4f, %.4f, %.4f] m\n", leverArm[0], leverArm[1], leverArm[2]);
}

static void dumpMagnetometer(SbgEComHandle *pHandle)
{
  SbgEComMagModelsStdId   model;
  SbgEComMagRejectionConf reject;

  printf("\n-- magnetometer --------------------------------------------\n");

  if (ok(sbgEComCmdMagGetModelId(pHandle, &model), "mag model"))
  {
    printf("  model                %u = %s\n", (unsigned)model, magModelName(model));
  }

  if (ok(sbgEComCmdMagGetRejection(pHandle, &reject), "mag rejection"))
  {
    printf("  reject magnetometer  %s\n", rejectionName(reject.magneticField));
  }
}

static void dumpAidingAssignment(SbgEComHandle *pHandle)
{
  SbgEComAidingAssignConf conf;

  printf("\n-- aiding assignment ---------------------------------------\n");

  if (!ok(sbgEComCmdSensorGetAidingAssignment(pHandle, &conf), "aiding assignment"))
  {
    return;
  }

  printf("  gps1Port             %s\n", portName(conf.gps1Port));
  printf("  gps1Sync             %u\n", (unsigned)conf.gps1Sync);
  printf("  rtcmPort             %s\n", portName(conf.rtcmPort));
  printf("  dvlPort              %s\n", portName(conf.dvlPort));
  printf("  dvlSync              %u\n", (unsigned)conf.dvlSync);
  printf("  airDataPort          %s\n", portName(conf.airDataPort));
  printf("  odometerPinsConf     %u\n", (unsigned)conf.odometerPinsConf);
}

static void dumpOdometer(SbgEComHandle *pHandle)
{
  SbgEComOdoConf          conf;
  SbgEComOdoRejectionConf reject;
  float                   leverArm[3];

  printf("\n-- odometer ------------------------------------------------\n");

  if (ok(sbgEComCmdOdoGetConf(pHandle, &conf), "odometer conf"))
  {
    printf("  gain                 %.4f pulses/m\n", conf.gain);
    printf("  gain error           %u %%\n", (unsigned)conf.gainError);
    printf("  reverse mode         %s\n", conf.reverseMode ? "true" : "false");
  }

  if (ok(sbgEComCmdOdoGetLeverArm(pHandle, leverArm), "odometer lever arm"))
  {
    printf("  lever arm            [%.4f, %.4f, %.4f] m\n", leverArm[0], leverArm[1], leverArm[2]);
  }

  if (ok(sbgEComCmdOdoGetRejection(pHandle, &reject), "odometer rejection"))
  {
    printf("  reject velocity      %s\n", rejectionName(reject.velocity));
  }
}

static void dumpGnss(SbgEComHandle *pHandle)
{
  SbgEComGnssModelsStdIds  model;
  SbgEComGnssInstallation  install;
  SbgEComGnssRejectionConf reject;

  printf("\n-- GNSS ----------------------------------------------------\n");

  if (ok(sbgEComCmdGnss1GetModelId(pHandle, &model), "gnss model"))
  {
    printf("  model                %u = %s\n", (unsigned)model, gnssModelName(model));
  }

  if (ok(sbgEComCmdGnss1InstallationGet(pHandle, &install), "gnss installation"))
  {
    printf("  primary lever arm    [%.4f, %.4f, %.4f] m\n",
           install.leverArmPrimary[0], install.leverArmPrimary[1], install.leverArmPrimary[2]);
    printf("  primary precise      %s\n", install.leverArmPrimaryPrecise ? "true" : "false");
    printf("  secondary lever arm  [%.4f, %.4f, %.4f] m\n",
           install.leverArmSecondary[0], install.leverArmSecondary[1], install.leverArmSecondary[2]);
    printf("  secondary mode       %u = %s\n",
           (unsigned)install.leverArmSecondaryMode, installModeName(install.leverArmSecondaryMode));
  }

  if (ok(sbgEComCmdGnss1GetRejection(pHandle, &reject), "gnss rejection"))
  {
    printf("  reject position      %s\n", rejectionName(reject.position));
    printf("  reject velocity      %s\n", rejectionName(reject.velocity));
    printf("  reject heading       %s\n", rejectionName(reject.hdt));
  }
}

int main(int argc, char **argv)
{
  const char   *port = (argc > 1) ? argv[1] : "/dev/ttyUSB0";
  uint32_t      baud = (argc > 2) ? (uint32_t)atoi(argv[2]) : 921600;
  SbgInterface  iface;
  SbgEComHandle handle;
  SbgErrorCode  code;

  printf("sbg_probe: %s @ %u bps  (read-only; no settings are written)\n", port, baud);

  code = sbgInterfaceSerialCreate(&iface, port, baud);

  if (code != SBG_NO_ERROR)
  {
    fprintf(stderr, "cannot open %s at %u: %s\n", port, baud, sbgErrorCodeToString(code));
    return 1;
  }

  code = sbgEComInit(&handle, &iface);

  if (code != SBG_NO_ERROR)
  {
    fprintf(stderr, "sbgEComInit failed: %s\n", sbgErrorCodeToString(code));
    sbgInterfaceDestroy(&iface);
    return 1;
  }

  dumpDeviceInfo(&handle);
  dumpInitCondition(&handle);
  dumpMotionProfile(&handle);
  dumpImuAlignment(&handle);
  dumpMagnetometer(&handle);
  dumpAidingAssignment(&handle);
  dumpOdometer(&handle);
  dumpGnss(&handle);

  printf("\n");

  sbgEComClose(&handle);
  sbgInterfaceDestroy(&iface);

  return 0;
}
