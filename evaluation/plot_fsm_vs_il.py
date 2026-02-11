import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("evaluation/results_fsm_vs_il.csv")

    # Summary bars
    summary = df.groupby("policy").agg(
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
        avg_steps=("steps", "mean"),
        min_front_mean=("min_front", "mean"),
        avg_step_ms=("avg_step_ms", "mean"),
    ).reset_index()

    # Bar plot: success & collision
    ax = summary.set_index("policy")[["success_rate", "collision_rate"]].plot(kind="bar")
    ax.set_ylabel("rate")
    ax.set_title("Unseen test: success vs collision")
    plt.tight_layout()
    plt.savefig("evaluation/plot_rates.png", dpi=160)
    plt.close()

    # Box plot: min_front by policy
    ax = df.boxplot(column="min_front", by="policy")
    plt.title("Min obstacle clearance proxy (min_front)")
    plt.suptitle("")
    plt.ylabel("meters (clipped sensor)")
    plt.tight_layout()
    plt.savefig("evaluation/plot_min_front.png", dpi=160)
    plt.close()

    # Bar plot: avg_step_ms
    ax = summary.set_index("policy")[["avg_step_ms"]].plot(kind="bar")
    ax.set_ylabel("ms")
    ax.set_title("Decision latency (avg step wall time)")
    plt.tight_layout()
    plt.savefig("evaluation/plot_latency.png", dpi=160)
    plt.close()

    print("Wrote:")
    print(" - evaluation/plot_rates.png")
    print(" - evaluation/plot_min_front.png")
    print(" - evaluation/plot_latency.png")

if __name__ == "__main__":
    main()
