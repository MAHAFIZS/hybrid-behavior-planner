#pragma once
#include <cmath>

namespace sim {

struct VehicleState {
  float x = 0.f;
  float y = 0.f;
  float heading = 0.f; // radians
  float v = 0.f;       // m/s
};

struct VehicleParams {
  float radius = 0.35f;
  float v_max = 2.0f;
  float accel = 2.0f;        // m/s^2
  float decel = 3.0f;        // m/s^2
  float yaw_rate = 1.2f;     // rad/s for turning actions
};

enum class Action { STOP=0, GO_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, SLOW_DOWN=4 };

class Vehicle {
public:
  VehicleState s;
  VehicleParams p;

  void step(Action a, float dt);
};

} // namespace sim
