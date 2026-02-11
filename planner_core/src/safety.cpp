#include "planner_core/safety.hpp"

namespace planner {

sim::Action apply_safety(sim::Action proposed, float front_dist, const SafetyLimits& lim) {
  // If very close: disallow forward motion, but allow turning to escape.
  if (front_dist < lim.hard_stop) {
    if (proposed == sim::Action::TURN_LEFT || proposed == sim::Action::TURN_RIGHT)
      return proposed;
    return sim::Action::STOP;
  }

  // In slow zone: only override GO_FORWARD to SLOW_DOWN. Allow turning.
  if (front_dist < lim.slow_zone) {
    if (proposed == sim::Action::GO_FORWARD) return sim::Action::SLOW_DOWN;
    return proposed;
  }

  return proposed;
}

} // namespace planner
