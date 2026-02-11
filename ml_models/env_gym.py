import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# Action IDs must match C++:
# STOP=0, GO_FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3, SLOW_DOWN=4
STOP, GO_FWD, TL, TR, SLOW = 0, 1, 2, 3, 4

def _wrap_pi(a: float) -> float:
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def _ray_circle_distance(ox, oy, dirx, diry, cx, cy, r):
    rx, ry = cx - ox, cy - oy
    proj = rx * dirx + ry * diry
    if proj < 0.0:
        return 999.0
    perp2 = (rx * rx + ry * ry) - proj * proj
    r2 = r * r
    if perp2 > r2:
        return 999.0
    thc = math.sqrt(max(0.0, r2 - perp2))
    t0 = proj - thc
    if t0 < 0.0:
        t0 = proj + thc
    return t0 if t0 >= 0.0 else 999.0

@dataclass
class VehicleParams:
    radius: float = 0.35
    v_max: float = 2.0
    accel: float = 2.0
    decel: float = 3.0
    yaw_rate: float = 1.2

class Simple2DEnv:
    """
    Lightweight python replica of your C++ sim + sensors.
    Obs vector = [v, front, left, right, goal_dist, goal_sin, goal_cos]
    """
    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
        self.p = VehicleParams()
        self.dt = 0.05
        self.max_steps = 700

        self.x = self.y = self.heading = self.v = 0.0
        self.goal_x = self.goal_y = self.goal_r = 0.6
        self.obstacles: List[Tuple[float, float, float]] = []
        self.t = 0

        self.reset()

    def reset(self):
        j = json.loads(Path(self.scenario_path).read_text())
        self.dt = float(j.get("dt", 0.05))
        self.max_steps = int(j.get("max_steps", 700))

        v = j["vehicle"]
        self.x = float(v.get("x", 0.0))
        self.y = float(v.get("y", 0.0))
        self.heading = float(v.get("heading", 0.0))
        self.v = float(v.get("v", 0.0))
        self.p.radius = float(v.get("radius", 0.35))

        g = j["goal"]
        self.goal_x = float(g.get("x", 10.0))
        self.goal_y = float(g.get("y", 0.0))
        self.goal_r = float(g.get("radius", 0.6))

        self.obstacles = []
        for o in j["obstacles"]:
            self.obstacles.append((float(o["x"]), float(o["y"]), float(o["radius"])))

        self.t = 0
        return self.obs()

    def goal_dist(self) -> float:
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def collided(self) -> bool:
        for (ox, oy, r) in self.obstacles:
            d = math.hypot(self.x - ox, self.y - oy)
            if d <= (self.p.radius + r):
                return True
        return False

    def _sense_dir(self, ang: float) -> float:
        dx, dy = math.cos(ang), math.sin(ang)
        best = 999.0
        for (ox, oy, r) in self.obstacles:
            d = _ray_circle_distance(self.x, self.y, dx, dy, ox, oy, r + self.p.radius)
            best = min(best, d)
        return best

    def sensors(self):
        front = self._sense_dir(self.heading)
        left  = self._sense_dir(self.heading + 0.6)
        right = self._sense_dir(self.heading - 0.6)

        gx = self.goal_x - self.x
        gy = self.goal_y - self.y
        gd = math.hypot(gx, gy)
        goal_world = math.atan2(gy, gx)
        rel = _wrap_pi(goal_world - self.heading)

        return front, left, right, gd, math.sin(rel), math.cos(rel)

    def obs(self) -> np.ndarray:
        front, left, right, gd, gs, gc = self.sensors()

        # clip like training
        front = min(max(front, 0.0), 20.0)
        left  = min(max(left,  0.0), 20.0)
        right = min(max(right, 0.0), 20.0)
        gd    = min(max(gd,    0.0), 50.0)
        return np.array([self.v, front, left, right, gd, gs, gc], dtype=np.float32)

    def step(self, action_id: int):
        # speed
        if action_id == STOP:
            self.v -= self.p.decel * self.dt
        elif action_id == SLOW:
            self.v -= 0.7 * self.p.decel * self.dt
        else:
            self.v += self.p.accel * self.dt

        self.v = float(np.clip(self.v, 0.0, self.p.v_max))

        # heading
        if action_id == TL:
            self.heading += self.p.yaw_rate * self.dt
        elif action_id == TR:
            self.heading -= self.p.yaw_rate * self.dt

        # position
        self.x += math.cos(self.heading) * self.v * self.dt
        self.y += math.sin(self.heading) * self.v * self.dt

        self.t += 1

        reached = self.goal_dist() <= self.goal_r
        coll = self.collided()
        timeout = self.t >= self.max_steps

        done = reached or coll or timeout
        reason = "goal" if reached else "collision" if coll else "timeout" if timeout else "none"
        info = {"reason": reason}

        return self.obs(), done, info
