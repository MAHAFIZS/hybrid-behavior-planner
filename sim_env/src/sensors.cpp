#include "sim_env/sensors.hpp"
#include <cmath>
#include <algorithm>

namespace sim {

static float norm_angle(float a) {
  while (a >  M_PI) a -= 2*M_PI;
  while (a < -M_PI) a += 2*M_PI;
  return a;
}

static float ray_circle_distance(
  float ox, float oy, float dirx, float diry,
  float cx, float cy, float r
) {
  // Ray: o + t*d, t>=0. Circle center c.
  float rx = cx - ox;
  float ry = cy - oy;

  float proj = rx*dirx + ry*diry; // projection onto ray direction
  if (proj < 0.f) return 999.f;   // circle behind ray

  // perpendicular distance squared
  float perp2 = (rx*rx + ry*ry) - proj*proj;
  float r2 = r*r;
  if (perp2 > r2) return 999.f;

  // intersection distance along ray to first hit
  float thc = std::sqrt(std::max(0.f, r2 - perp2));
  float t0 = proj - thc;
  if (t0 < 0.f) t0 = proj + thc;
  return (t0 >= 0.f) ? t0 : 999.f;
}

SensorReadings Sensors::sense(const World& w) {
  SensorReadings s;

  const float x = w.vehicle.s.x;
  const float y = w.vehicle.s.y;
  const float th = w.vehicle.s.heading;

  auto eval_dir = [&](float ang)->float{
    float dx = std::cos(ang);
    float dy = std::sin(ang);
    float best = 999.f;
    for (const auto& o : w.obstacles) {
      // inflate obstacle by vehicle radius to approximate vehicle as point
      float d = ray_circle_distance(x, y, dx, dy, o.x, o.y, o.r + w.vehicle.p.radius);
      best = std::min(best, d);
    }
    return best;
  };

  s.front = eval_dir(th);
  s.left  = eval_dir(th + 0.6f);
  s.right = eval_dir(th - 0.6f);

  // goal direction in vehicle frame
  float gx = w.goal.x - x;
  float gy = w.goal.y - y;
  s.goal_dist = std::sqrt(gx*gx + gy*gy);

  float goal_world = std::atan2(gy, gx);
  float rel = norm_angle(goal_world - th);

  s.goal_dir_sin = std::sin(rel);
  s.goal_dir_cos = std::cos(rel);

  return s;
}

} // namespace sim
