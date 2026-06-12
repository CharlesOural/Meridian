#include <memory>
#include <stdexcept>
#include <utility>

#include "isam2_backend.hpp"
#include "meridian/backend/ibackend.hpp"

namespace meridian {

std::unique_ptr<IBackEnd> makeBackEnd(const BackendConfig& cfg,
                                      std::shared_ptr<const CalibrationSet> calib,
                                      TelemetrySink* telemetry, bool deterministic) {
  switch (cfg.kind) {
    case BackEndKind::Isam2:
      return std::make_unique<backend::Isam2BackEnd>(cfg, std::move(calib), telemetry,
                                                     deterministic);
  }
  throw std::invalid_argument("makeBackEnd: unknown BackEndKind");
}

}  // namespace meridian