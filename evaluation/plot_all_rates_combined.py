import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    df = pd.read_csv("evaluation/results_all.csv")

    summary = df.groupby("policy", as_index=False).agg(
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
    ).sort_values("policy")

    labels = summary["policy"].tolist()
    x = np.arange(len(labels))
    w = 0.38

    fig = plt.figure()
    plt.bar(x - w/2, summary["success_rate"].to_numpy(), width=w, label="success_rate")
    plt.bar(x + w/2, summary["collision_rate"].to_numpy(), width=w, label="collision_rate")

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("rate")
    plt.title("Success vs Collision (unseen test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation/plot_all_rates_combined.png", dpi=200)
    plt.close(fig)

    print("Wrote: evaluation/plot_all_rates_combined.png")


if __name__ == "__main__":
    main()
