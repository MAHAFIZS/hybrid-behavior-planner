import glob
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ml_models.env_gym import Simple2DEnv

class ScenarioGymEnv(gym.Env):
    """
    Gymnasium wrapper around Simple2DEnv.
    - Each episode uses one scenario JSON.
    - Scenarios are cycled or randomly sampled.
    - Observation: 7D float vector
    - Action: Discrete(5)
    """
    metadata = {"render_modes": []}

    def __init__(self, scenario_glob="scenarios/train_*.json", seed=0):
        super().__init__()
        self.scenario_paths = sorted(glob.glob(scenario_glob))
        assert len(self.scenario_paths) > 0, f"No scenarios found: {scenario_glob}"
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self._env = None
        self._cur_path = None

    def _pick_scenario(self):
        # random sampling helps RL generalization
        idx = int(self.rng.integers(0, len(self.scenario_paths)))
        return self.scenario_paths[idx]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._cur_path = self._pick_scenario()
        self._env = Simple2DEnv(self._cur_path)
        obs = self._env.reset()
        info = {"scenario": self._cur_path}
        return obs, info

    def step(self, action):
        obs, done, info = self._env.step(int(action))
        # Reward shaping (simple, works well enough)
        # - encourage reducing goal distance
        # - big penalty on collision
        # - bonus on success
        gd = float(obs[4])
        # approximate progress reward: negative goal distance
        reward = -0.01 * gd

        if info["reason"] == "collision":
            reward -= 10.0
        elif info["reason"] == "goal":
            reward += 10.0

        terminated = info["reason"] in ("goal", "collision")
        truncated = info["reason"] == "timeout"
        return obs, reward, terminated, truncated, info
