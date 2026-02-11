import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from ml_models.env_gym import Simple2DEnv
from ml_models.fsm_policy import FsmPolicy
from ml_models.il_policy import ILPolicy
from ml_models.ppo_policy import PPOPolicy
from ml_models.ppo_shield_policy import PPOShieldPolicy


# Keep in sync with your action ids
ACTION_NAMES = {
    0: "STOP",
    1: "GO_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
    4: "SLOW_DOWN",
}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_policy(name: str):
    name = name.lower()
    if name == "fsm":
        return FsmPolicy(), "FSM"
    if name == "il":
        return ILPolicy("ml_models/il_policy.pt"), "IL"
    if name == "ppo":
        # PPOPolicy should load with CPU (see note below), but even if not, it still works
        return PPOPolicy("ml_models/ppo_policy.zip"), "PPO"
    if name in ("ppo_shield", "ppo+shield", "shield"):
        return PPOShieldPolicy("ml_models/ppo_policy.zip"), "PPO+Shield"
    raise ValueError(f"Unknown policy '{name}'. Use: fsm | il | ppo | ppo_shield")


def rollout(scenario_path: str, policy_name: str):
    policy, title = _load_policy(policy_name)

    env = Simple2DEnv(scenario_path)
    obs = env.reset()

    xs, ys, hs, vs, fronts, actions = [], [], [], [], [], []
    reason = "none"
    done = False

    steps = 0
    while not done:
        a = policy.act(obs)
        obs, done, info = env.step(a)

        # obs = [v, front, left, right, goal_dist, goal_sin, goal_cos]
        v = float(obs[0])
        front = float(obs[1])

        # Your env stores pose internally (you already used it in eval)
        x = float(env.x)
        y = float(env.y)
        h = float(env.heading)

        xs.append(x)
        ys.append(y)
        hs.append(h)
        vs.append(v)
        fronts.append(front)
        actions.append(int(a))

        reason = info.get("reason", "none")
        steps += 1
        if steps > 5000:
            reason = "forced_stop"
            break

    return title, np.array(xs), np.array(ys), np.array(hs), np.array(vs), np.array(fronts), np.array(actions), reason


def circle_xy(x, y, r, n=128):
    th = np.linspace(0, 2 * np.pi, n)
    return x + r * np.cos(th), y + r * np.sin(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="scenarios/test_unseen_001.json")
    ap.add_argument("--policy", type=str, default="ppo_shield", help="fsm | il | ppo | ppo_shield")
    ap.add_argument("--out", type=str, default="evaluation/demo.gif")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    sc = load_json(args.scenario)
    veh_r = float(sc["vehicle"]["radius"])
    goal = sc["goal"]
    obstacles = sc["obstacles"]

    title, xs, ys, hs, vs, fronts, actions, reason = rollout(args.scenario, args.policy)

    pad = 2.0
    xmin = min(xs.min(), goal["x"]) - pad
    xmax = max(xs.max(), goal["x"]) + pad
    ymin = min(ys.min(), goal["y"]) - pad
    ymax = max(ys.max(), goal["y"]) + pad

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.2)

    # Draw obstacles (static)
    for ob in obstacles:
        cx, cy = circle_xy(ob["x"], ob["y"], ob["radius"])
        ax.plot(cx, cy, linewidth=2)

    # Draw goal (static)
    gx, gy = circle_xy(goal["x"], goal["y"], goal["radius"])
    ax.plot(gx, gy, linewidth=2)

    # Dynamic artists
    traj_line, = ax.plot([], [], linewidth=2)
    veh_body, = ax.plot([], [], linewidth=3)
    head_line, = ax.plot([], [], linewidth=2)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def draw(i):
        # Vehicle circle
        cx, cy = circle_xy(xs[i], ys[i], veh_r, n=64)
        veh_body.set_data(cx, cy)

        # Heading line
        hx = xs[i] + veh_r * math.cos(hs[i])
        hy = ys[i] + veh_r * math.sin(hs[i])
        head_line.set_data([xs[i], hx], [ys[i], hy])

        # Trajectory
        traj_line.set_data(xs[: i + 1], ys[: i + 1])

        a = int(actions[i])
        txt.set_text(
            f"{title} | step={i}/{len(xs)-1} | done={reason}\n"
            f"action={ACTION_NAMES.get(a, str(a))}\n"
            f"v={vs[i]:.2f}  front={fronts[i]:.2f}"
        )

        return traj_line, veh_body, head_line, txt

    ani = FuncAnimation(fig, draw, frames=len(xs), interval=1000 / args.fps, blit=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=PillowWriter(fps=args.fps))
    plt.close(fig)

    print(f"Saved demo: {out_path}")


if __name__ == "__main__":
    main()
