"""Generate the Pareto/ablation plots for weeks 13-14 from committed CSVs
in results/. No new measurements -- purely a plotting pass over data
already gathered by eval/run.py and bench_multistream.py.

Usage: python scripts/make_plots.py
Writes results/plots/*.png
"""

import csv
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "plots"
OUT.mkdir(exist_ok=True)


def read_csv(name: str) -> list[dict]:
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def plot_pipeline_pareto() -> None:
    rows = read_csv("pipeline_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [(axes[0], "hota", "HOTA"), (axes[1], "mota", "MOTA")]:
        for row in rows:
            fps = float(row["mean_fps"])
            val = float(row[metric])
            ax.scatter(fps, val, s=70)
            ax.annotate(row["pipeline"], (fps, val), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.set_xlabel("mean fps")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs throughput (MOT17 train, all 7 seqs)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_pareto.png", dpi=150)
    plt.close(fig)


def plot_ablation_curve() -> None:
    rows = read_csv("ablation_anchor_interval.csv")
    rows.sort(key=lambda r: int(r["anchor_interval"]))
    intervals = [int(r["anchor_interval"]) for r in rows]
    fps = [float(r["mean_fps"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for metric, label in [("hota", "HOTA"), ("mota", "MOTA"), ("idf1", "IDF1")]:
        ax.plot(intervals, [float(r[metric]) for r in rows], marker="o", label=label)
    ax.set_xlabel("anchor interval (frames)")
    ax.set_ylabel("score")
    ax.set_title("mv-fixed: accuracy vs anchor interval")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    mota = [float(r["mota"]) for r in rows]
    ax.plot(fps, mota, marker="o", color="tab:orange")
    for x, y, n in zip(fps, mota, intervals):
        ax.annotate(f"interval={n}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("mean fps")
    ax.set_ylabel("MOTA")
    ax.set_title("mv-fixed: MOTA/throughput tradeoff by anchor interval")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_anchor_interval.png", dpi=150)
    plt.close(fig)


def plot_multistream() -> None:
    pipelines = ["baseline", "mv-fixed", "mv-adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in pipelines:
        rows = read_csv(f"bench_multistream_{name}.csv")
        rows.sort(key=lambda r: int(r["n_streams"]))
        n = [int(r["n_streams"]) for r in rows]
        agg = [float(r["aggregate_fps"]) for r in rows]
        per = [float(r["per_stream_fps"]) for r in rows]
        axes[0].plot(n, agg, marker="o", label=name)
        axes[1].plot(n, per, marker="o", label=name)

    axes[0].set_xlabel("concurrent streams")
    axes[0].set_ylabel("aggregate fps")
    axes[0].set_title("Aggregate throughput vs stream count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axhline(25, color="gray", linestyle="--", linewidth=1, label="25fps target")
    axes[1].set_xlabel("concurrent streams")
    axes[1].set_ylabel("per-stream fps")
    axes[1].set_title("Per-stream fps vs stream count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "multistream_scaling.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_pipeline_pareto()
    plot_ablation_curve()
    plot_multistream()
    print(f"wrote plots to {OUT}")
