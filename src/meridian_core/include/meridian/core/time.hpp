#pragma once

#include <cstdint>
#include <optional>

namespace meridian::core {

// Signed, integer nanoseconds on one explicitly selected measurement timeline.
// TimeNs deliberately exposes no unchecked arithmetic operators.
class TimeNs final {
public:
  constexpr explicit TimeNs(std::int64_t nanoseconds) noexcept : nanoseconds_(nanoseconds) {}

  [[nodiscard]] constexpr std::int64_t count() const noexcept { return nanoseconds_; }

  [[nodiscard]] static std::optional<TimeNs> fromSecNanosec(std::int64_t seconds,
                                                            std::uint32_t nanoseconds) noexcept;
  [[nodiscard]] static std::optional<TimeNs> checkedAdd(TimeNs time,
                                                        std::int64_t delta_ns) noexcept;
  [[nodiscard]] static std::optional<std::int64_t> checkedDifference(TimeNs lhs,
                                                                     TimeNs rhs) noexcept;

  friend constexpr bool operator==(TimeNs, TimeNs) noexcept = default;
  friend constexpr auto operator<=>(TimeNs, TimeNs) noexcept = default;

private:
  std::int64_t nanoseconds_;
};

}  // namespace meridian::core
