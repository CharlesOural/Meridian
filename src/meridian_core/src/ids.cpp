#include "meridian/core/ids.hpp"

#include <stdexcept>
#include <utility>

namespace meridian::core {
namespace {

void requireNonEmpty(std::string_view value, std::string_view kind) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(kind) + " cannot be empty");
  }
}

}  // namespace

SensorId::SensorId(std::string value) : value_(std::move(value)) {
  requireNonEmpty(value_, "SensorId");
}

CalibrationId::CalibrationId(std::string value) : value_(std::move(value)) {
  requireNonEmpty(value_, "CalibrationId");
}

}  // namespace meridian::core
