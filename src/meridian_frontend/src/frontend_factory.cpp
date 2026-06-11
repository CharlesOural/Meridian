#include <memory>
#include <stdexcept>

#include "lio/lio_frontend.hpp"
#include "meridian/frontend/ifrontend.hpp"

namespace meridian {

std::unique_ptr<IFrontEnd> makeFrontEnd(const FrontendConfig& cfg,
                                        std::shared_ptr<const CalibrationSet> calib,
                                        TelemetrySink* telemetry, bool deterministic) {
  switch (cfg.kind) {
    case FrontEndKind::Lio:
      return std::make_unique<lio::LioFrontEnd>(cfg, std::move(calib), telemetry, deterministic);
  }
  throw std::runtime_error("makeFrontEnd: unknown FrontEndKind");
}

}  // namespace meridian
