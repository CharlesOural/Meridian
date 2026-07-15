#include <memory>
#include <stdexcept>
#include <utility>

#include "cpu/cpu_surface_map.hpp"
#include "layered_map.hpp"
#include "meridian/map/imap_layer.hpp"
#include "voxel_hash_map.hpp"

#ifdef MERIDIAN_MAP_HAS_NVBLOX
#include "nvblox/nvblox_surface_map.hpp"
#endif

namespace meridian {

namespace {

std::unique_ptr<ISurfaceMap> make_surface_backend(const MapConfig& cfg,
                                                  std::shared_ptr<const CalibrationSet> calib) {
  switch (cfg.backend) {
    case MapBackend::Cpu:
      return std::make_unique<map::CpuSurfaceMap>(cfg, std::move(calib));
    case MapBackend::Nvblox:
#ifdef MERIDIAN_MAP_HAS_NVBLOX
      return std::make_unique<map::NvbloxSurfaceMap>(cfg, std::move(calib));
#else
      throw std::invalid_argument(
          "map.backend=nvblox is not compiled into this build (MERIDIAN_MAP_NVBLOX=OFF); "
          "select map.backend=cpu or build with CUDA + nvblox");
#endif
    case MapBackend::Vulkan:
      throw std::invalid_argument("map.backend=vulkan is not implemented yet (deferred)");
  }
  throw std::invalid_argument("makeMapLayer: unknown map.backend");
}

}  // namespace

IMapLayer::Ptr makeMapLayer(const MapConfig& cfg, std::shared_ptr<const CalibrationSet> calib,
                            std::shared_ptr<KeyframeStore> store, TelemetrySink* telemetry) {
  auto reg = std::make_unique<map::VoxelHashMap>(cfg);
  auto surf = make_surface_backend(cfg, calib);
  return std::make_unique<map::LayeredMap>(cfg, std::move(calib), std::move(store), std::move(reg),
                                           std::move(surf), telemetry);
}

}  // namespace meridian
