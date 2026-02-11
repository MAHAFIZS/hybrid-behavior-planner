#pragma once
#include <vector>
#include "sim_env/vehicle.hpp"

namespace sim {

struct Circle {
  float x=0.f, y=0.f, r=0.5f;
};

struct Goal {
  float x=0.f, y=0.f, r=0.6f;
};

struct StepResult {
  bool collided = false;
  bool reached_goal = false;
};

class World {
public:
  float dt = 0.05f;
  int max_steps = 600;

  Vehicle vehicle;
  Goal goal;
  std::vector<Circle> obstacles;

  int step_count = 0;

  StepResult step(Action a);
  float goal_dist() const;
};

} // namespace sim
