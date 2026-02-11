#pragma once
#include "sim_env/world.hpp"

namespace sim {

struct SensorReadings {
  float front = 999.f;
  float left  = 999.f;
  float right = 999.f;

  float goal_dist = 0.f;
  float goal_dir_sin = 0.f; // direction to goal in vehicle frame
  float goal_dir_cos = 1.f;
};

class Sensors {
public:
  // simple ray distances to closest obstacle along 3 directions
  static SensorReadings sense(const World& w);
};

} // namespace sim
