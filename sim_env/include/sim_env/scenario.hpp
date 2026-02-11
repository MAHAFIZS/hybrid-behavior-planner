#pragma once
#include <string>
#include "sim_env/world.hpp"

namespace sim {

class Scenario {
public:
  static World load_from_file(const std::string& path);
};

} // namespace sim
