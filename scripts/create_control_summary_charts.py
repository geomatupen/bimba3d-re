from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPECTED_PIPELINE_ID = "pipeline_5df0ad97ef5a"
BETTER_COLOR = "#16835D"
WORSE_COLOR = "#D2604B"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#172033"
MODEL_TIME_COLOR = "#2563A6"
CONTROL_TIME_COLOR = "#E8A317"


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if not value:
        raise ValueError(f"Missing {key} for {row.get('project')}")
    return float(value)


def is_better(row: dict[str, str], prefix: str, metric: str) -> bool:
    control = number(row, f"{prefix}_{metric}")
    model = number(row, f"model_selected_{metric}")
    return control < model if metric == "lpips" else control > model


def chart_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "figure.titlesize": 25,
        "axes.titlesize": 19,
        "xtick.labelsize": 14,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#94A3B8",
    })


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def create_summary_chart(rows: list[dict[str, str]], output_dir: Path) -> None:
    experiments = (
        ("Time-controlled baseline", "time_control"),
        ("Gaussian-controlled baseline", "gaussian_control"),
    )
    metrics = ("psnr", "ssim", "lpips")
    metric_labels = ("PSNR", "SSIM", "LPIPS")

    result_sets = []
    for label, prefix in experiments:
        completed = [
            row for row in rows
            if row.get(f"{prefix}_status") == "success"
            and row.get(f"{prefix}_psnr")
            and row.get("model_selected_psnr")
        ]
        better = [sum(is_better(row, prefix, metric) for row in completed) for metric in metrics]
        worse = [len(completed) - count for count in better]
        result_sets.append((label, completed, better, worse))

    mean_times = []
    for (_, completed, _, _), (_, prefix) in zip(result_sets, experiments):
        mean_times.extend((
            float(np.mean([number(row, "model_selected_training_seconds") for row in completed])),
            float(np.mean([number(row, f"{prefix}_training_seconds") for row in completed])),
        ))
    time_axis_max = int(np.ceil(max(mean_times) / 200.0) * 200)

    fig, axes = plt.subplots(2, 1, figsize=(15.5, 12.4))
    fig.subplots_adjust(left=0.20, right=0.965, top=0.64, bottom=0.31, hspace=1.18)
    fig.suptitle("Controlled baseline vs model-selected quality", fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.905,
        "Completed projects with better or worse final quality",
        ha="center",
        fontsize=17,
        color="#475569",
    )

    for ax, (label, completed, better, worse), (_, prefix) in zip(axes, result_sets, experiments):
        y = np.arange(len(metrics))
        ax.barh(y, better, color=BETTER_COLOR, height=0.56)
        ax.barh(y, worse, left=better, color=WORSE_COLOR, height=0.56)

        for index, (better_count, worse_count) in enumerate(zip(better, worse)):
            if better_count:
                ax.text(
                    better_count / 2,
                    index,
                    str(better_count),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=17,
                    fontweight="bold",
                )
            if worse_count:
                ax.text(
                    better_count + worse_count / 2,
                    index,
                    str(worse_count),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=17,
                    fontweight="bold",
                )

        total = len(completed)
        ax.set_xlim(0, total)
        ax.set_xticks(range(0, total + 1, 2))
        ax.set_yticks(y, metric_labels)
        ax.invert_yaxis()
        ax.set_title(f"{label} (n={total})", loc="left", fontweight="bold", pad=42)
        ax.set_xlabel("Projects")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)

        model_time = float(np.mean([number(row, "model_selected_training_seconds") for row in completed]))
        control_time = float(np.mean([number(row, f"{prefix}_training_seconds") for row in completed]))
        time_ax = ax.twiny()
        time_ax.set_xlim(0, time_axis_max)
        time_ax.set_xticks(np.arange(0, time_axis_max + 1, 200))
        time_ax.tick_params(axis="x", colors="#64748B", labelsize=12, pad=3)
        time_ax.spines[["right", "left", "bottom"]].set_visible(False)
        time_ax.spines["top"].set_color("#94A3B8")
        time_ax.set_xlabel(
            f"Average training time (seconds)  |  M = {model_time:.1f}s  |  C = {control_time:.1f}s",
            color="#64748B",
            fontsize=13,
            labelpad=7,
        )

        for metric_y in y:
            time_ax.scatter(
                model_time,
                metric_y - 0.13,
                s=105,
                color=MODEL_TIME_COLOR,
                edgecolor="white",
                linewidth=1.3,
                zorder=5,
            )
            time_ax.scatter(
                control_time,
                metric_y + 0.13,
                s=115,
                marker="D",
                color=CONTROL_TIME_COLOR,
                edgecolor="white",
                linewidth=1.3,
                zorder=5,
            )

    fig.legend(
        handles=[
            Patch(facecolor=BETTER_COLOR, label="Better quality"),
            Patch(facecolor=WORSE_COLOR, label="Worse quality"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_TIME_COLOR,
                   markeredgecolor="white", markersize=10, label="M: Model-selected mean time"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=CONTROL_TIME_COLOR,
                   markeredgecolor="white", markersize=10, label="C: Controlled mean time"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.775),
        ncol=4,
        frameon=False,
    )
    time_model, time_control, gaussian_model, gaussian_control = mean_times
    fig.text(
        0.20,
        0.105,
        f"Summary: Time control vs model selected: {time_control - time_model:.1f}s additional mean training time "
        f"({time_control:.1f}s vs {time_model:.1f}s; {(time_control / time_model - 1) * 100:.1f}% more).\n"
        "PSNR split 6/6, while SSIM and LPIPS were worse in 11/12 projects.\n"
        f"Gaussian control vs model selected: {gaussian_control - gaussian_model:.1f}s additional mean training time "
        f"({gaussian_control:.1f}s vs {gaussian_model:.1f}s; {gaussian_control / gaussian_model:.1f}x as long).\n"
        "Despite the additional time, it was worse in 7/11 PSNR, 10/11 SSIM, and 9/11 LPIPS results.",
        fontsize=13.0,
        fontweight="bold",
        color=TEXT_COLOR,
        linespacing=1.45,
    )
    fig.text(
        0.20,
        0.025,
        "Times are mean training-loop seconds before final evaluation. Higher is better for PSNR and SSIM; lower is better for LPIPS. "
        "The unfinished DJI Gaussian-controlled run is excluded.",
        fontsize=11.8,
        color="#475569",
    )
    save_figure(fig, output_dir, "controlled_better_worse_summary")


def create_chart(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("pipeline_id") for row in rows} != {EXPECTED_PIPELINE_ID}:
        raise ValueError("Input CSV is not the Final_test_June_27 pipeline export")

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_style()
    create_summary_chart(rows, output_dir)


def main() -> None:
    default_dir = Path("outputs/1. ridge_regression/base_matched_time_and_gaussian_constrained_runs")
    parser = argparse.ArgumentParser(description="Create a simple controlled-run quality summary chart.")
    parser.add_argument("--input", type=Path, default=default_dir / "compare.csv")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    create_chart(args.input, args.output_dir)
    print(f"Summary chart written to {args.output_dir}")


if __name__ == "__main__":
    main()
