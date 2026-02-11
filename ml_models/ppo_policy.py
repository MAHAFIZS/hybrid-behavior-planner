from stable_baselines3 import PPO
import numpy as np

class PPOPolicy:
    def __init__(self, path="ml_models/ppo_policy.zip"):
        self.model = PPO.load(path, device="cpu")  # IMPORTANT

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
