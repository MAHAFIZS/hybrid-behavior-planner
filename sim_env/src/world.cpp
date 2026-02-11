#include "sim_env/world.hpp"
#include <cmath>

namespace sim {

static float dist2(float ax,float ay,float bx,float by){
  float dx=ax-bx, dy=ay-by;
  return dx*dx + dy*dy;
}

float World::goal_dist() const {
  return std::sqrt(dist2(vehicle.s.x, vehicle.s.y, goal.x, goal.y));
}

StepResult World::step(Action a) {
  StepResult r;

  vehicle.step(a, dt);
  step_count++;

  // goal reached
  if (goal_dist() <= goal.r) r.reached_goal = true;

  // collision check (vehicle circle vs obstacle circles)
  for (const auto& o : obstacles) {
    float d = std::sqrt(dist2(vehicle.s.x, vehicle.s.y, o.x, o.y));
    if (d <= (vehicle.p.radius + o.r)) {
      r.collided = true;
      break;
    }
  }
  return r;
}

} // namespace sim
