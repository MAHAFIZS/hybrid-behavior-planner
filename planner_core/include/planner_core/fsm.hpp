#pragma once
#include "sim_env/sensors.hpp"
#include "planner_core/safety.hpp"

namespace planner {

enum class State { SEARCH=0, FOLLOW_GOAL=1, AVOID_OBSTACLE=2, STOP=3 };

struct FsmConfig {
  float obstacle_trigger = 2.5f;
  float obstacle_clear   = 3.0f;
};

class Fsm {
public:
  State state = State::FOLLOW_GOAL;
  FsmConfig cfg;
  SafetyLimits safety;

  sim::Action act(const sim::SensorReadings& s);
};

} // namespace planner
