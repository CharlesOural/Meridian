#include <memory>
#include <stdexcept>

#include "ct/ct_frontend.hpp"
#include "iekf/iekf_frontend.hpp"
#include "meridian/frontend/ifrontend.hpp"

namespace meridian {

std::unique_ptr<IFrontEnd> makeFrontEnd(const FrontendConfig& cfg,
                                        std::shared_ptr<const CalibrationSet> calib,
                                        TelemetrySink* telemetry, bool deterministic) {
  switch (cfg.kind) {
    case FrontEndKind::IekfOracle: {
      // The iEKF runs a fixed iteration schedule with no wall-clock deadline, so the
      // determinism flag is already its only mode and need not be threaded further.
      auto fe = std::make_unique<IekfFrontEnd>(cfg, telemetry);
      if (calib) {
        fe->set_calibration(std::move(calib));
      }
      return fe;
    }
    case FrontEndKind::CtLivo:
      return std::make_unique<CtFrontEnd>(cfg, std::move(calib), telemetry, deterministic);
  }
  throw std::runtime_error("makeFrontEnd: unknown FrontEndKind");
}

}  // namespace meridian
