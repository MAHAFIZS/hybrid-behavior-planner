import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from ml_models.gymnasium_env import ScenarioGymEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="ml_models/ppo_policy.zip")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    env = DummyVecEnv([lambda: ScenarioGymEnv("scenarios/train_*.json", seed=args.seed)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        seed=args.seed,
        device="cpu",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="ml_models/checkpoints",
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=args.steps, callback=ckpt_cb)
    model.save(str(out_path))
    print(f"Saved PPO policy: {out_path}")

if __name__ == "__main__":
    main()
