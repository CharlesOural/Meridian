#pragma once

#include <string>

#include "meridian/config/config.hpp"

namespace meridian {

// Parses a YAML document at `path` into a Config, layering values over the struct
// defaults, then runs Config::validate(). Throws std::runtime_error on a missing
// file, a YAML parse error, an unknown enum/kind string, or a failed validation;
// the exception message names the offending key.
Config load_config_yaml(const std::string& path);

}  // namespace meridian
