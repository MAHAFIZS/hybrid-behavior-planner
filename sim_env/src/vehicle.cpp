#include "sim_env/vehicle.hpp"

namespace sim {

static float clamp(float v, float lo, float hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

void Vehicle::step(Action a, float dt) {
  // speed update
  if (a == Action::STOP) {
    s.v -= p.decel * dt;
  } else if (a == Action::SLOW_DOWN) {
    s.v -= (0.7f * p.decel) * dt;
  } else if (a == Action::GO_FORWARD || a == Action::TURN_LEFT || a == Action::TURN_RIGHT) {
    s.v += p.accel * dt;
  }
  s.v = clamp(s.v, 0.f, p.v_max);

  // heading update
  if (a == Action::TURN_LEFT)  s.heading += p.yaw_rate * dt;
  if (a == Action::TURN_RIGHT) s.heading -= p.yaw_rate * dt;

  // position update (simple kinematics)
  s.x += std::cos(s.heading) * s.v * dt;
  s.y += std::sin(s.heading) * s.v * dt;
}

} // namespace sim
