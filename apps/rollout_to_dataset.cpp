#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sim_env/scenario.hpp"
#include "sim_env/sensors.hpp"
#include "planner_core/fsm.hpp"

// CSV columns:
// scenario,t,dt,x,y,heading,v,front,left,right,goal_dist,goal_sin,goal_cos,action,done,done_reason
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

static int action_id(sim::Action a) {
  return static_cast<int>(a); // STOP=0, GO_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, SLOW_DOWN=4
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: rollout_to_dataset <out.csv> <scenario1.json> [scenario2.json ...]\n";
    return 1;
  }

  std::string out_path = argv[1];

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 1;
  }

  out << "scenario,t,dt,x,y,heading,v,front,left,right,goal_dist,goal_sin,goal_cos,action_id,action_str,done,done_reason\n";

  for (int i = 2; i < argc; ++i) {
    std::string scen_path = argv[i];

    sim::World w = sim::Scenario::load_from_file(scen_path);
    planner::Fsm fsm;

    for (int t = 0; t < w.max_steps; ++t) {
      auto sens = sim::Sensors::sense(w);
      sim::Action a = fsm.act(sens);

      auto r = w.step(a);

      bool done = r.collided || r.reached_goal || (t == w.max_steps - 1);
      std::string reason = r.reached_goal ? "goal" : r.collided ? "collision" : done ? "timeout" : "none";

      out
        << scen_path << ","
        << t << ","
        << w.dt << ","
        << w.vehicle.s.x << ","
        << w.vehicle.s.y << ","
        << w.vehicle.s.heading << ","
        << w.vehicle.s.v << ","
        << sens.front << ","
        << sens.left << ","
        << sens.right << ","
        << sens.goal_dist << ","
        << sens.goal_dir_sin << ","
        << sens.goal_dir_cos << ","
        << action_id(a) << ","
        << action_name(a) << ","
        << (done ? 1 : 0) << ","
        << reason
        << "\n";

      if (done) break;
    }

    std::cout << "Rolled out: " << scen_path << "\n";
  }

  std::cout << "Wrote dataset: " << out_path << "\n";
  return 0;
}
