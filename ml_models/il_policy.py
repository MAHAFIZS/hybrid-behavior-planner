import numpy as np
import torch
from .models import MLPPolicy


class ILPolicy:
    def __init__(self, ckpt_path="ml_models/il_policy.pt"):
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.features = payload["features"]
        self.mu = payload["mu"].astype(np.float32)
        self.sigma = payload["sigma"].astype(np.float32)

        self.model = MLPPolicy(in_dim=len(self.features), hidden=64, out_dim=5)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def act(self, obs: np.ndarray) -> int:
        # obs already in the same feature order used in training:
        # [v, front, left, right, goal_dist, goal_sin, goal_cos]
        x = obs.astype(np.float32)

        # normalize using stored stats
        xn = (x - self.mu) / (self.sigma + 1e-6)

        with torch.no_grad():
            logits = self.model(torch.from_numpy(xn).unsqueeze(0))
            a = int(torch.argmax(logits, dim=1).item())
        return a
