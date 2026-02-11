import numpy as np

from ml_models.fsm_policy import FsmPolicy
from ml_models.ppo_policy import PPOPolicy
from ml_models.ppo_shield_policy import PPOShieldPolicy


class HybridPolicy:
    """
    Hybrid behavior policy:
      - Primary: PPO (or PPO+Shield)
      - If risk is high (front distance small), fallback to FSM
      - Optional: always shield the final action (recommended)
    """

    def __init__(
        self,
        ppo_path="ml_models/ppo_policy.zip",
        use_shield=True,
        fallback_front_threshold=0.9,   # tune later
    ):
        self.fsm = FsmPolicy()
        self.use_shield = use_shield
        self.fallback_front_threshold = float(fallback_front_threshold)

        # Primary policy
        if use_shield:
            self.primary = PPOShieldPolicy(ppo_path)
        else:
            self.primary = PPOPolicy(ppo_path)

        # Stats
        self.fallback_count = 0
        self.steps = 0

    def reset_stats(self):
        self.fallback_count = 0
        self.steps = 0

    @property
    def fallback_rate(self) -> float:
        return (self.fallback_count / self.steps) if self.steps > 0 else 0.0

    def act(self, obs: np.ndarray) -> int:
        """
        obs layout:
          [v, front, left, right, goal_dist, goal_sin, goal_cos]
        """
        self.steps += 1

        front = float(obs[1])

        # High-risk: fallback to deterministic FSM
        if front < self.fallback_front_threshold:
            self.fallback_count += 1
            return self.fsm.act(obs)

        # Otherwise use PPO (or PPO+Shield)
        return self.primary.act(obs)
