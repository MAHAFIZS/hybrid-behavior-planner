#include <iostream>
#include <string>
#include <iomanip>

#include "sim_env/scenario.hpp"
#include "sim_env/sensors.hpp"
#include "planner_core/fsm.hpp"

static const char* action_name(sim::Action a) {
  switch (a) {
    case sim::Action::STOP: return "STOP";
    case sim::Action::GO_FORWARD: return "GO_FORWARD";
    case sim::Action::TURN_LEFT: return "TURN_LEFT";
    case sim::Action::TURN_RIGHT: return "TURN_RIGHT";
    case sim::Action::SLOW_DOWN: return "SLOW_DOWN";
  }
  return "UNKNOWN";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: run_fsm <scenario.json>\n";
    return 1;
  }
  std::string path = argv[1];

  sim::World w = sim::Scenario::load_from_file(path);
  planner::Fsm fsm;

  bool done = false;
  bool collided = false;
  bool success = false;

  std::cout << "Running scenario: " << path << "\n";
  for (int t = 0; t < w.max_steps && !done; ++t) {
    auto sens = sim::Sensors::sense(w);
    sim::Action a = fsm.act(sens);
    auto r = w.step(a);

    collided = r.collided;
    success = r.reached_goal;
    done = collided || success;

    if (t % 10 == 0 || done) {
      std::cout << std::fixed << std::setprecision(2)
        << "t=" << t
        << " x=" << w.vehicle.s.x
        << " y=" << w.vehicle.s.y
        << " v=" << w.vehicle.s.v
        << " front=" << sens.front
        << " goal_d=" << sens.goal_dist
        << " a=" << action_name(a)
        << (collided ? "  [COLLISION]" : "")
        << (success ? "  [GOAL]" : "")
        << "\n";
    }
  }

  std::cout << "\nResult: "
            << (success ? "SUCCESS" : collided ? "COLLISION" : "TIMEOUT")
            << "\n";
  return 0;
}
