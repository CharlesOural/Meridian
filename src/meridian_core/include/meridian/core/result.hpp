#pragma once

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

namespace meridian::core {

// Small value-or-error type for expected runtime outcomes.  It deliberately
// does not perform implicit conversions: callers must choose success/failure
// and inspect the result before accessing either branch.
template <typename Value, typename Error>
class Result {
 public:
  [[nodiscard]] static Result success(Value value) {
    return Result(std::in_place_index<0>, std::move(value));
  }

  [[nodiscard]] static Result failure(Error error) {
    return Result(std::in_place_index<1>, std::move(error));
  }

  [[nodiscard]] bool hasValue() const noexcept { return storage_.index() == 0U; }
  [[nodiscard]] explicit operator bool() const noexcept { return hasValue(); }

  [[nodiscard]] Value& value() & {
    assert(hasValue());
    return std::get<0>(storage_);
  }
  [[nodiscard]] const Value& value() const& {
    assert(hasValue());
    return std::get<0>(storage_);
  }
  [[nodiscard]] Value&& value() && {
    assert(hasValue());
    return std::get<0>(std::move(storage_));
  }

  [[nodiscard]] Error& error() & {
    assert(!hasValue());
    return std::get<1>(storage_);
  }
  [[nodiscard]] const Error& error() const& {
    assert(!hasValue());
    return std::get<1>(storage_);
  }

 private:
  template <std::size_t Index, typename Item>
  explicit Result(std::in_place_index_t<Index> index, Item&& item)
      : storage_(index, std::forward<Item>(item)) {}

  std::variant<Value, Error> storage_;
};

}  // namespace meridian::core
