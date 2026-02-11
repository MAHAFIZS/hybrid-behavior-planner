import pandas as pd
import matplotlib.pyplot as plt


def _save_bar(summary: pd.DataFrame, ycol: str, out_path: str, title: str, ylabel: str):
    fig = plt.figure()
    plt.bar(summary["policy"], summary[ycol])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv("evaluation/results_all.csv")

    # Per-episode table -> per-policy summary
    summary = df.groupby("policy", as_index=False).agg(
        episodes=("scenario", "count"),
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
        avg_steps=("steps", "mean"),
        min_front_mean=("min_front_mean", "mean"),
        avg_step_ms=("avg_step_ms", "mean"),
    ).sort_values("policy")

    # --- Plots ---
    # 1) Success + collision rates
    fig = plt.figure()
    x = summary["policy"]
    plt.bar(x, summary["success_rate"])
    plt.title("Success rate (higher is better)")
    plt.ylabel("rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("evaluation/plot_all_rates_success.png", dpi=200)
    plt.close(fig)

    fig = plt.figure()
    plt.bar(x, summary["collision_rate"])
    plt.title("Collision rate (lower is better)")
    plt.ylabel("rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("evaluation/plot_all_rates_collision.png", dpi=200)
    plt.close(fig)

    # 2) Safety proxy: mean front distance (higher is better)
    _save_bar(
        summary,
        ycol="min_front_mean",
        out_path="evaluation/plot_all_min_front.png",
        title="Mean front distance (safety proxy)",
        ylabel="meters",
    )

    # 3) Latency
    _save_bar(
        summary,
        ycol="avg_step_ms",
        out_path="evaluation/plot_all_latency.png",
        title="Decision latency per step",
        ylabel="ms",
    )

    print("Wrote:")
    print(" - evaluation/plot_all_rates_success.png")
    print(" - evaluation/plot_all_rates_collision.png")
    print(" - evaluation/plot_all_min_front.png")
    print(" - evaluation/plot_all_latency.png")


if __name__ == "__main__":
    main()
