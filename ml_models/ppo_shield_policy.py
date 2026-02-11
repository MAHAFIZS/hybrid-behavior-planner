import numpy as np
from ml_models.ppo_policy import PPOPolicy
from ml_models.safety_shield import SafetyShield

class PPOShieldPolicy:
    def __init__(self, ppo_path="ml_models/ppo_policy.zip", hard_stop=0.9, slow_zone=1.8):
        self.ppo = PPOPolicy(ppo_path)
        self.shield = SafetyShield(hard_stop=hard_stop, slow_zone=slow_zone)
        self.overrides = 0
        self.total = 0

    def act(self, obs: np.ndarray):
        a = self.ppo.act(obs)
        a2, overridden = self.shield.override(a, obs)

        self.total += 1
        if overridden:
            self.overrides += 1

        # return action + info for animations/evaluation
        return int(a2), {"overridden": bool(overridden), "raw_action": int(a)}

    def override_rate(self) -> float:
        return self.overrides / max(1, self.total)
