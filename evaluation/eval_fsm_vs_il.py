import argparse
import glob
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from ml_models.env_gym import Simple2DEnv
from ml_models.fsm_policy import FsmPolicy
from ml_models.il_policy import ILPolicy

def run_policy(policy, scen_paths, max_steps=None):
    results = []
    for sp in scen_paths:
        env = Simple2DEnv(sp)
        obs = env.reset()

        t0 = time.perf_counter()
        steps = 0
        min_front = 999.0

        done = False
        reason = "none"
        while not done:
            a = policy.act(obs)
            obs, done, info = env.step(a)
            steps += 1
            min_front = min(min_front, float(obs[1]))
            if max_steps is not None and steps >= max_steps:
                done = True
                reason = "timeout_forced"
                break
            reason = info["reason"]

        t1 = time.perf_counter()

        results.append({
            "scenario": sp,
            "reason": reason,
            "success": 1 if reason == "goal" else 0,
            "collision": 1 if reason == "collision" else 0,
            "steps": steps,
            "min_front": min_front,
            "wall_time_s": (t1 - t0),
            "avg_step_ms": 1000.0 * (t1 - t0) / max(1, steps),
        })
    return pd.DataFrame(results)

def summarize(df, name):
    return {
        "policy": name,
        "episodes": len(df),
        "success_rate": df["success"].mean(),
        "collision_rate": df["collision"].mean(),
        "avg_steps": df["steps"].mean(),
        "min_front_mean": df["min_front"].mean(),
        "avg_step_ms": df["avg_step_ms"].mean(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="scenarios/test_unseen_*.json")
    ap.add_argument("--out_csv", default="evaluation/results_fsm_vs_il.csv")
    args = ap.parse_args()

    scen_paths = sorted(glob.glob(args.glob))
    assert len(scen_paths) > 0, f"No scenarios found for {args.glob}"

    fsm = FsmPolicy()
    il = ILPolicy("ml_models/il_policy.pt")

    df_fsm = run_policy(fsm, scen_paths)
    df_fsm["policy"] = "FSM"

    df_il = run_policy(il, scen_paths)
    df_il["policy"] = "IL"

    all_df = pd.concat([df_fsm, df_il], ignore_index=True)
    all_df.to_csv(args.out_csv, index=False)

    sum_df = pd.DataFrame([summarize(df_fsm, "FSM"), summarize(df_il, "IL")])
    print("\n=== Summary (unseen test) ===")
    print(sum_df.to_string(index=False))
    print(f"\nWrote: {args.out_csv}")

if __name__ == "__main__":
    main()
