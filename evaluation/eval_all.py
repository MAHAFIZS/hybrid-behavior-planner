import glob
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_models.env_gym import Simple2DEnv
from ml_models.fsm_policy import FsmPolicy
from ml_models.il_policy import ILPolicy
from ml_models.ppo_policy import PPOPolicy
from ml_models.ppo_shield_policy import PPOShieldPolicy
from ml_models.hybrid_policy import HybridPolicy


@dataclass
class EpisodeStats:
    scenario: str
    policy: str
    steps: int
    success: int
    collision: int
    timeout: int
    done_reason: str
    min_front_mean: float
    avg_step_ms: float


def _list_scenarios(pattern_or_list: List[str]) -> List[str]:
    """
    Accepts:
      - ["scenarios/test_unseen_*.json"]  (glob pattern)
      - ["scenarios/a.json", "scenarios/b.json"] (explicit list)
    """
    paths: List[str] = []
    for p in pattern_or_list:
        if "*" in p or "?" in p or "[" in p:
            paths.extend(sorted(glob.glob(p)))
        else:
            paths.append(p)
    # de-dup, keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _env_step(env: Simple2DEnv, action: int):
    """
    Normalize env.step return to: (obs, done, info)
    Some gym envs return (obs, reward, terminated, truncated, info),
    but our Simple2DEnv returns (obs, done, info).
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 3:
        obs, done, info = out
        return obs, bool(done), info
    if isinstance(out, tuple) and len(out) == 5:
        obs, _reward, terminated, truncated, info = out
        return obs, bool(terminated or truncated), info
    raise RuntimeError(f"Unexpected env.step() output format: len={len(out)}")


def run_policy(policy_obj, scen_paths: List[str], policy_name: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for sp in scen_paths:
        env = Simple2DEnv(sp)
        obs = env.reset()

        # If a policy has stats, reset them per episode or keep global?
        # For reporting fallback/override rates globally, keep global.
        # But it’s still nice to reset any per-episode internal state if exists.
        if hasattr(policy_obj, "reset_episode"):
            try:
                policy_obj.reset_episode()
            except Exception:
                pass

        step_times_ms: List[float] = []
        fronts: List[float] = []

        done = False
        done_reason = "none"
        steps = 0

        while not done:
            t0 = time.perf_counter()
            a = int(policy_obj.act(obs))
            obs, done, info = _env_step(env, a)
            t1 = time.perf_counter()

            step_times_ms.append((t1 - t0) * 1000.0)

            # obs layout: [v, front, left, right, goal_dist, goal_sin, goal_cos]
            if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) >= 2:
                fronts.append(float(obs[1]))

            done_reason = info.get("reason", done_reason)
            steps += 1

            # hard safety break to avoid infinite loops
            if steps > 10000:
                done = True
                done_reason = "forced_stop"

        success = int(done_reason == "goal")
        collision = int(done_reason == "collision")
        timeout = int(done_reason in ("timeout", "forced_stop"))

        min_front_mean = float(np.mean(fronts)) if len(fronts) else float("nan")
        avg_step_ms = float(np.mean(step_times_ms)) if len(step_times_ms) else float("nan")

        rows.append(
            dict(
                scenario=sp,
                policy=policy_name,
                steps=steps,
                success=success,
                collision=collision,
                timeout=timeout,
                done_reason=done_reason,
                min_front_mean=min_front_mean,
                avg_step_ms=avg_step_ms,
            )
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("policy", as_index=False).agg(
        episodes=("scenario", "count"),
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
        avg_steps=("steps", "mean"),
        min_front_mean=("min_front_mean", "mean"),
        avg_step_ms=("avg_step_ms", "mean"),
    )
    # nice formatting in print
    return grp.sort_values("policy")


def main():
    # unseen test set
    scen_paths = _list_scenarios(["scenarios/test_unseen_*.json"])
    if not scen_paths:
        raise RuntimeError("No scenarios found for: scenarios/test_unseen_*.json")

    fsm = FsmPolicy()
    il = ILPolicy("ml_models/il_policy.pt")
    ppo = PPOPolicy("ml_models/ppo_policy.zip")
    ppo_shield = PPOShieldPolicy("ml_models/ppo_policy.zip")

    # Hybrid: PPO+Shield primary + fallback to FSM if front < threshold
    hybrid = HybridPolicy(
        ppo_path="ml_models/ppo_policy.zip",
        use_shield=True,
        fallback_front_threshold=0.9,
    )

    df_all = []
    df_all.append(run_policy(fsm, scen_paths, "FSM"))
    df_all.append(run_policy(il, scen_paths, "IL"))
    df_all.append(run_policy(ppo, scen_paths, "PPO"))
    df_all.append(run_policy(ppo_shield, scen_paths, "PPO+Shield"))
    df_all.append(run_policy(hybrid, scen_paths, "Hybrid"))

    df = pd.concat(df_all, ignore_index=True)

    summ = summarize(df)

    print("\n=== Summary (unseen test) ===")
    print(summ.to_string(index=False))

    out_csv = "evaluation/results_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    # Optional: policy-specific extra stats
    if hasattr(ppo_shield, "override_rate"):
        try:
            print(f"\nPPO+Shield override_rate = {float(ppo_shield.override_rate):.4f}")
        except Exception:
            pass

    if hasattr(hybrid, "fallback_rate"):
        try:
            print(f"Hybrid fallback_rate = {float(hybrid.fallback_rate):.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
