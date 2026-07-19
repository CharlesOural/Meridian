#pragma once

// Private GTSAM seam. This header is intentionally not installed.

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <cstdint>

#include "meridian/core/api.hpp"

namespace meridian::local::gtsam_api {

struct CandidateCacheStamp {
  std::uint64_t revision{};
  core::ContentHash semantic_hash{};

  bool operator==(const CandidateCacheStamp&) const = default;
};

// Stateful nonlinear factors implement this API so a copied iSAM2 candidate
// can own a distinct mutable cache handle while initially sharing the exact
// immutable cache contents represented by CandidateCacheStamp.
class CandidateIsolationApi {
public:
  virtual ~CandidateIsolationApi() = default;

  [[nodiscard]] virtual gtsam::NonlinearFactor::shared_ptr cloneForCandidate() const = 0;
  [[nodiscard]] virtual CandidateCacheStamp candidateCacheStamp() const = 0;
  [[nodiscard]] virtual const void* candidateCacheHandleIdentity() const noexcept = 0;
};

}  // namespace meridian::local::gtsam_api
