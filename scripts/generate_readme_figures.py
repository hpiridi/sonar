#!/usr/bin/env python
"""Generate figures for the README from benchmark results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"
ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "accent": "#059669",
    "warning": "#D97706",
    "muted": "#94A3B8",
    "bg": "#FFFFFF",
    "text": "#1E293B",
}

DETECTOR_COLORS = {
    "AdONE": "#2563EB",
    "DONE": "#3B82F6",
    "GAE": "#7C3AED",
    "ANOMALOUS": "#059669",
    "DOMINANT": "#10B981",
    "CONAD": "#D97706",
    "AnomalyDAE": "#F59E0B",
    "ONE": "#EF4444",
    "CoLA": "#F87171",
    "OCGNN": "#94A3B8",
}

SCALE_COLORS = {
    "Small\n(37K)": "#2563EB",
    "Medium\n(847K)": "#7C3AED",
    "Large\n(7.4M)": "#DC2626",
}


def load_results(name):
    with open(RESULTS_DIR / f"{name}_detectors_results.json") as f:
        return json.load(f)


def fig_benchmark_small():
    """Bar chart: all 10 detectors on small dataset, ROC-AUC + AP side by side."""
    results = load_results("small")
    results.sort(key=lambda r: r["roc_auc"], reverse=True)

    detectors = [r["algorithm"] for r in results]
    roc_auc = [r["roc_auc"] for r in results]
    ap = [r["average_precision"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(detectors))
    width = 0.35

    bars1 = ax.bar(x - width / 2, roc_auc, width, label="ROC-AUC",
                   color="#2563EB", edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + width / 2, ap, width, label="Avg Precision",
                   color="#7C3AED", edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels on ROC-AUC bars
    for bar, val in zip(bars1, roc_auc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#1E293B")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("SONAR Small — 10 PyGOD Detectors", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, fontsize=10, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "benchmark_small.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Saved assets/benchmark_small.png")


def fig_scalability():
    """Grouped bar chart: 3 scalable detectors across 3 scales."""
    small = load_results("small")
    medium = load_results("medium")
    large = load_results("large")

    scalable = ["CoLA", "GAE", "OCGNN"]

    def get_metric(results, algo, metric):
        for r in results:
            if r["algorithm"] == algo:
                return r[metric]
        return 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scales = ["Small\n(37K)", "Medium\n(847K)", "Large\n(7.4M)"]
    x = np.arange(len(scalable))
    width = 0.25

    for ax_idx, (metric, title) in enumerate([("roc_auc", "ROC-AUC"), ("average_precision", "Average Precision")]):
        ax = axes[ax_idx]
        for i, (scale_label, results) in enumerate(zip(scales, [small, medium, large])):
            vals = [get_metric(results, algo, metric) for algo in scalable]
            bars = ax.bar(x + i * width, vals, width, label=scale_label,
                         color=list(SCALE_COLORS.values())[i],
                         edgecolor="white", linewidth=0.5, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#1E293B")

        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scalable, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, title="Scale", title_fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Scalability — Only 3 of 10 Detectors Handle Medium & Large Graphs",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "scalability.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Saved assets/scalability.png")


def fig_scale_comparison():
    """Visual showing SONAR scale vs social network GAD benchmarks from the paper."""
    # Datasets from paper Table 1 (comparison with existing social network benchmarks)
    benchmarks = [
        ("Cresci-15", 5301),
        ("TwiBot-20", 229580),
        ("MGTAB", 410199),
        ("TwiBot-22", 1000000),
        ("SONAR-Small", 36860),
        ("SONAR-Medium", 846496),
        ("SONAR-Large", 3797980),
    ]

    names = [b[0] for b in benchmarks]
    users = [b[1] for b in benchmarks]
    relations = [1, 1, 4, 1, 7, 7, 7]

    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = ["#94A3B8", "#94A3B8", "#94A3B8", "#94A3B8",
              "#2563EB", "#7C3AED", "#DC2626"]
    bars = ax.barh(names, users, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    for bar, n, rel in zip(bars, users, relations):
        if n >= 1_000_000:
            label = f"{n / 1_000_000:.1f}M users, {rel} relation{'s' if rel > 1 else ''}"
        elif n >= 1_000:
            label = f"{n / 1_000:.0f}K users, {rel} relation{'s' if rel > 1 else ''}"
        else:
            label = f"{n} users, {rel} relation"
        ax.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2,
                label, ha="left", va="center", fontsize=9.5, fontweight="bold",
                color="#1E293B")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Users (log scale)", fontsize=11, fontweight="bold")
    ax.set_title("SONAR vs Existing Social Network Anomaly Detection Benchmarks",
                 fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(right=users[-1] * 8)

    # Add annotation
    ax.annotate("3.8x larger than TwiBot-22\n+ 7 relation types + dual labels",
                xy=(3797980, 6), fontsize=9.5, fontweight="bold",
                color="#DC2626", ha="left",
                xytext=(5500000, 4.3),
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "scale_comparison.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("  Saved assets/scale_comparison.png")


if __name__ == "__main__":
    print("Generating README figures...")
    fig_benchmark_small()
    fig_scalability()
    fig_scale_comparison()
    print("Done.")
