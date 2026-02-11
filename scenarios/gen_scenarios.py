import json
import math
import random
from pathlib import Path

def sample_obstacles(rng: random.Random, n: int):
    obs = []
    for _ in range(n):
        # Keep obstacles mostly in the corridor between start and goal
        x = rng.uniform(2.0, 9.0)
        y = rng.uniform(-2.0, 2.0)
        r = rng.uniform(0.35, 0.60)
        obs.append({"x": round(x, 3), "y": round(y, 3), "radius": round(r, 3)})
    return obs

def make_scenario(seed: int, difficulty: str):
    rng = random.Random(seed)

    dt = 0.05
    max_steps = 700

    # Start near origin with slight heading noise
    start_heading = rng.uniform(-0.2, 0.2)
    vehicle = {"x": 0.0, "y": 0.0, "heading": round(start_heading, 3), "v": 0.0, "radius": 0.35}

    # Goal around x=10 with lateral offset
    goal_y = rng.uniform(-1.0, 1.0) if difficulty == "train" else rng.uniform(-2.0, 2.0)
    goal = {"x": 10.0, "y": round(goal_y, 3), "radius": 0.6}

    if difficulty == "train":
        n_obs = rng.randint(2, 5)
        y_span = 2.0
    else:
        # unseen: more obstacles + wider spread (distribution shift)
        n_obs = rng.randint(5, 9)
        y_span = 3.0

    obstacles = []
    for _ in range(n_obs):
        x = rng.uniform(1.5, 9.2)
        y = rng.uniform(-y_span, y_span)
        r = rng.uniform(0.35, 0.65)
        obstacles.append({"x": round(x, 3), "y": round(y, 3), "radius": round(r, 3)})

    return {
        "seed": seed,
        "dt": dt,
        "max_steps": max_steps,
        "vehicle": vehicle,
        "goal": goal,
        "obstacles": obstacles
    }

def main():
    out_dir = Path("scenarios")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train set
    for i in range(1, 51):
        scen = make_scenario(seed=1000 + i, difficulty="train")
        p = out_dir / f"train_{i:03d}.json"
        p.write_text(json.dumps(scen, indent=2))

    # Unseen test set
    for i in range(1, 16):
        scen = make_scenario(seed=9000 + i, difficulty="unseen")
        p = out_dir / f"test_unseen_{i:03d}.json"
        p.write_text(json.dumps(scen, indent=2))

    print("Generated:")
    print(" - 50 train scenarios: scenarios/train_001.json ... train_050.json")
    print(" - 15 test scenarios : scenarios/test_unseen_001.json ... test_unseen_015.json")

if __name__ == "__main__":
    main()
