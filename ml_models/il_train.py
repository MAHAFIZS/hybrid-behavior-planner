import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from ml_models.models import MLPPolicy


FEATURES = ["v", "front", "left", "right", "goal_dist", "goal_sin", "goal_cos"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data_train.csv")
    ap.add_argument("--out", type=str, default="ml_models/il_policy.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)

    # Build X, y
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["action_id"].to_numpy(dtype=np.int64)

    # Simple normalization (helps IL + later RL baseline)
    # front/left/right can be 999; clip then scale
    X_clip = X.copy()
    X_clip[:, 1:4] = np.clip(X_clip[:, 1:4], 0.0, 20.0)  # distances
    X_clip[:, 4] = np.clip(X_clip[:, 4], 0.0, 50.0)      # goal_dist

    # z-score normalize
    mu = X_clip.mean(axis=0)
    sigma = X_clip.std(axis=0) + 1e-6
    Xn = (X_clip - mu) / sigma

    # Train/val split
    n = len(Xn)
    idx = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    Xtr, ytr = Xn[tr_idx], y[tr_idx]
    Xva, yva = Xn[val_idx], y[val_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPPolicy(in_dim=Xtr.shape[1], hidden=64, out_dim=5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch,
        shuffle=True,
    )

    def accuracy(Xa, ya):
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(Xa).to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        return float((pred == ya).mean())

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.shape[0]

        tr_acc = accuracy(Xtr, ytr)
        va_acc = accuracy(Xva, yva)
        print(f"epoch {ep:02d}  loss={total/len(Xtr):.4f}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}")

    # Save model + normalization stats (important!)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "features": FEATURES,
        "mu": mu,
        "sigma": sigma,
    }
    torch.save(payload, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
