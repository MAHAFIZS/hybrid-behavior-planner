#include "sim_env/scenario.hpp"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace sim {

World Scenario::load_from_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Failed to open scenario: " + path);

  nlohmann::json j;
  f >> j;

  World w;
  w.dt = j.value("dt", 0.05);
  w.max_steps = j.value("max_steps", 600);

  auto v = j.at("vehicle");
  w.vehicle.s.x = v.value("x", 0.0);
  w.vehicle.s.y = v.value("y", 0.0);
  w.vehicle.s.heading = v.value("heading", 0.0);
  w.vehicle.s.v = v.value("v", 0.0);
  w.vehicle.p.radius = v.value("radius", 0.35);

  auto g = j.at("goal");
  w.goal.x = g.value("x", 10.0);
  w.goal.y = g.value("y", 0.0);
  w.goal.r = g.value("radius", 0.6);

  w.obstacles.clear();
  for (auto& o : j.at("obstacles")) {
    Circle c;
    c.x = o.value("x", 0.0);
    c.y = o.value("y", 0.0);
    c.r = o.value("radius", 0.45);
    w.obstacles.push_back(c);
  }
  return w;
}

} // namespace sim
