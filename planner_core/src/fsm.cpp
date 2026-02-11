#include "planner_core/fsm.hpp"
#include <cmath>

namespace planner {

static sim::Action turn_toward_goal(float goal_sin) {
  // goal_sin > 0 means goal is to the left in vehicle frame
  if (goal_sin > 0.15f) return sim::Action::TURN_LEFT;
  if (goal_sin < -0.15f) return sim::Action::TURN_RIGHT;
  return sim::Action::GO_FORWARD;
}

sim::Action Fsm::act(const sim::SensorReadings& s) {
  // State transitions (deterministic)
  if (s.front < safety.hard_stop) state = State::STOP;
  else if (s.front < cfg.obstacle_trigger) state = State::AVOID_OBSTACLE;
  else if (state == State::AVOID_OBSTACLE && s.front > cfg.obstacle_clear) state = State::FOLLOW_GOAL;
  else if (state == State::STOP && s.front > cfg.obstacle_clear) state = State::FOLLOW_GOAL;

  sim::Action proposed = sim::Action::GO_FORWARD;

  switch (state) {
    case State::FOLLOW_GOAL:
      proposed = turn_toward_goal(s.goal_dir_sin);
      break;

    case State::AVOID_OBSTACLE:
      // turn away from closer side
      if (s.left < s.right) proposed = sim::Action::TURN_RIGHT;
      else proposed = sim::Action::TURN_LEFT;
      break;

   case State::STOP:
  // Escape: rotate away from the closer side to increase front clearance
      if (s.left < s.right) proposed = sim::Action::TURN_RIGHT;
      else proposed = sim::Action::TURN_LEFT;
      break;


    case State::SEARCH:
    default:
      // simple search: rotate slowly
      proposed = sim::Action::TURN_LEFT;
      break;
  }

  // Safety override
  return apply_safety(proposed, s.front, safety);
}

} // namespace planner
