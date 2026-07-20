#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace meridian::core {

class SensorId final {
public:
  explicit SensorId(std::string value);

  [[nodiscard]] std::string_view value() const noexcept { return value_; }
  friend bool operator==(const SensorId&, const SensorId&) noexcept = default;

private:
  std::string value_;
};

class CalibrationId final {
public:
  explicit CalibrationId(std::string value);

  [[nodiscard]] std::string_view value() const noexcept { return value_; }
  friend bool operator==(const CalibrationId&, const CalibrationId&) noexcept = default;

private:
  std::string value_;
};

class MeasurementId final {
public:
  constexpr explicit MeasurementId(std::uint64_t value) noexcept : value_(value) {}

  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }
  friend constexpr bool operator==(MeasurementId, MeasurementId) noexcept = default;
  friend constexpr auto operator<=>(MeasurementId, MeasurementId) noexcept = default;

private:
  std::uint64_t value_;
};

}  // namespace meridian::core
