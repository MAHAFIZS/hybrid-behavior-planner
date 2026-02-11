#pragma once
#include "sim_env/vehicle.hpp"

namespace planner {

struct SafetyLimits {
  float hard_stop = 1.0f; // meters
  float slow_zone = 2.0f; // meters
};

sim::Action apply_safety(sim::Action proposed, float front_dist, const SafetyLimits& lim);

} // namespace planner
