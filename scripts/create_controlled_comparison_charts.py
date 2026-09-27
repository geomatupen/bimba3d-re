from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


EXPECTED_PIPELINE_ID = "pipeline_5df0ad97ef5a"
TIME_COLOR = "#0F766E"
GAUSSIAN_COLOR = "#B45309"
BETTER_COLOR = "#047857"
WORSE_COLOR = "#C2410C"
PARTIAL_COLOR = "#64748B"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#172033"

SHORT_NAMES = {
    "12-Maisonneuve Market": "Maisonneuve Market",
    "4-Thomas-More-Church": "Thomas More Church",
    "Brasov": "Brasov",
    "Changu": "Changu",
    "Chatteau_circle_60_and_45_degrees": "Chateau",
    "DJI_202402171501_006_KTM20-oblique": "DJI KTM20",
    "Kninice_Church_of_the_Exaltation_of_the_Holy_Cross": "Kninice Church",
    "Pix4d_forensic": "Pix4D Forensic",
    "Telc_sv_Jakub_March_18": "Telc",
    "Tiburon_Angel_Islands_State_Park_Split_group_3": "Tiburon",
    "morice": "Morice",
    "uncovice": "Uncovice",
}

METRIC_CONFIG = {
    "psnr": ("PSNR difference (dB)", 3),
    "ssim": ("SSIM difference", 4),
    "lpips": ("LPIPS advantage", 4),
}


def number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key]) if row.get(key) else None
    except (TypeError, ValueError):
        return None


def quality_delta(row: dict[str, str], control: str, metric: str) -> float:
    control_value = number(row, f"{control}_{metric}")
    model_value = number(row, f"model_selected_{metric}")
    if control_value is None or model_value is None:
        raise ValueError(f"Missing {metric} for {row['project']} ({control})")
    return model_value - control_value if metric == "lpips" else control_value - model_value


def gaussian_ratio(row: dict[str, str]) -> float:
    actual = number(row, "gaussian_control_gaussian_count")
    target = number(row, "gaussian_control_target_count")
    if actual is None or target is None or target <= 0:
        raise ValueError(f"Missing Gaussian target data for {row['project']}")
    return actual / target


def gaussian_target_reached(row: dict[str, str]) -> bool:
    final_step = number(row, "gaussian_control_final_step")
    step_limit = number(row, "gaussian_control_step_limit")
    return (
        row.get("gaussian_control_status") == "success"
        and final_step is not None
        and step_limit is not None
        and final_step < step_limit
        and abs(gaussian_ratio(row) - 1.0) <= 0.05
    )


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "figure.titlesize": 24,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#94A3B8",
    })


def delta_chart(
    rows: list[dict[str, str]],
    control: str,
    metric: str,
    title: str,
    subtitle: str,
    output_dir: Path,
    filename: str,
) -> None:
    axis_label, decimals = METRIC_CONFIG[metric]
    plotted = sorted(
        [(SHORT_NAMES.get(row["project"], row["project"]), quality_delta(row, control, metric)) for row in rows],
        key=lambda item: item[1],
    )
    labels = [item[0] for item in plotted]
    values = [item[1] for item in plotted]
    height = max(5.5, 0.55 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(13.33, height), constrained_layout=True)
    y_positions = range(len(values))
    colors = [BETTER_COLOR if value >= 0 else WORSE_COLOR for value in values]
    ax.hlines(y_positions, 0, values, color=colors, linewidth=3, alpha=0.65)
    ax.scatter(values, y_positions, s=115, color=colors, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axvline(0, color=TEXT_COLOR, linewidth=1.5)
    ax.set_yticks(list(y_positions), labels)
    ax.set_xlabel(axis_label)
    ax.set_title(title, loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=15, color="#475569", va="bottom")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    span = max(max(values) - min(values), max(abs(value) for value in values), 0.01)
    ax.set_xlim(min(min(values), 0) - span * 0.16, max(max(values), 0) + span * 0.20)
    for y, value in zip(y_positions, values):
        offset = span * 0.025
        ax.text(value + (offset if value >= 0 else -offset), y, f"{value:+.{decimals}f}", va="center", ha="left" if value >= 0 else "right", fontsize=13, fontweight="bold")
    ax.text(0.01, -0.12, "Negative favors selected model", transform=ax.transAxes, color=WORSE_COLOR, fontsize=13, ha="left")
    ax.text(0.99, -0.12, "Positive favors baseline control", transform=ax.transAxes, color=BETTER_COLOR, fontsize=13, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, filename)


def time_budget_chart(rows: list[dict[str, str]], output_dir: Path) -> None:
    eligible = [row for row in rows if row.get("model_selected_status") == "success"]
    plotted = sorted([
        (
            SHORT_NAMES.get(row["project"], row["project"]),
            100 * number(row, "time_control_training_seconds") / number(row, "time_control_target_seconds"),
        )
        for row in eligible
    ], key=lambda item: item[1])
    labels = [item[0] for item in plotted]
    values = [item[1] for item in plotted]
    fig, ax = plt.subplots(figsize=(13.33, 8), constrained_layout=True)
    y_positions = range(len(values))
    ax.hlines(y_positions, 100, values, color=TIME_COLOR, linewidth=3, alpha=0.65)
    ax.scatter(values, y_positions, s=120, color=TIME_COLOR, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axvline(100, color=TEXT_COLOR, linewidth=1.8, linestyle="--")
    ax.set_yticks(list(y_positions), labels)
    ax.set_xlabel("Achieved time / selected-model time (%)")
    ax.set_title("Time-control budget verification", loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.01, "Trainer loop time before evaluation; DJI hard-cap reference excluded", transform=ax.transAxes, fontsize=15, color="#475569", va="bottom")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.set_xlim(98.8, max(values) + 1.6)
    for y, value in zip(y_positions, values):
        ax.text(value + 0.18, y, f"{value:.1f}%", va="center", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, output_dir, "time_budget_match")


def gaussian_attainment_chart(rows: list[dict[str, str]], output_dir: Path) -> None:
    plotted: list[tuple[str, float, str]] = []
    for row in rows:
        state = "partial" if row.get("gaussian_control_status") == "partial" else "reached" if gaussian_target_reached(row) else "ceiling"
        plotted.append((SHORT_NAMES.get(row["project"], row["project"]), 100 * gaussian_ratio(row), state))
    plotted.sort(key=lambda item: item[1])
    colors_by_state = {"reached": BETTER_COLOR, "ceiling": GAUSSIAN_COLOR, "partial": PARTIAL_COLOR}
    fig, ax = plt.subplots(figsize=(13.33, 8.6), constrained_layout=True)
    y_positions = range(len(plotted))
    for y, (_, value, state) in zip(y_positions, plotted):
        color = colors_by_state[state]
        ax.hlines(y, 0, value, color=color, linewidth=4, alpha=0.58)
        ax.scatter(value, y, s=125, color="white" if state == "partial" else color, edgecolor=color, linewidth=2, zorder=3)
        status_label = {"reached": "target reached", "ceiling": "15k ceiling", "partial": "partial at 10k"}[state]
        ax.text(value + 1.4, y, f"{value:.1f}%  {status_label}", va="center", fontsize=12.5, fontweight="bold", color=color)
    ax.axvline(100, color=TEXT_COLOR, linewidth=1.8, linestyle="--")
    ax.set_yticks(list(y_positions), [item[0] for item in plotted])
    ax.set_xlabel("Achieved Gaussian count / selected-model target (%)")
    ax.set_title("Gaussian-control target attainment", loc="left", fontweight="bold", pad=18)
    ax.text(0, 1.01, "Five completed controls reached and froze at the target; DJI is partial", transform=ax.transAxes, fontsize=15, color="#475569", va="bottom")
    ax.set_xlim(0, max(112, max(item[1] for item in plotted) + 18))
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BETTER_COLOR, markeredgecolor=BETTER_COLOR, markersize=10, label="Target reached"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GAUSSIAN_COLOR, markeredgecolor=GAUSSIAN_COLOR, markersize=10, label="Stopped at ceiling"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PARTIAL_COLOR, markeredgewidth=2, markersize=10, label="Partial run"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False)
    save_figure(fig, output_dir, "gaussian_target_attainment")


def combined_summary_chart(rows: list[dict[str, str]], output_dir: Path) -> None:
    eligible = [row for row in rows if row.get("model_selected_status") == "success"]
    matched = [row for row in eligible if gaussian_target_reached(row)]
    time_counts = [sum(quality_delta(row, "time_control", metric) >= 0 for row in eligible) for metric in ("psnr", "ssim", "lpips")]
    gaussian_counts = [sum(quality_delta(row, "gaussian_control", metric) >= 0 for row in matched) for metric in ("psnr", "ssim", "lpips")]
    time_percent = [100 * count / len(eligible) for count in time_counts]
    gaussian_percent = [100 * count / len(matched) for count in gaussian_counts]

    fig, (validity_ax, quality_ax) = plt.subplots(1, 2, figsize=(16, 8.6), gridspec_kw={"width_ratios": [0.82, 1.45]}, constrained_layout=True)
    fig.suptitle("Do equal time or equal Gaussian count reproduce selected-model quality?", fontweight="bold")

    validity_labels = ["Time budget matched", "Gaussian target reached"]
    validity_values = [100, 100 * len(matched) / len(eligible)]
    validity_colors = [TIME_COLOR, GAUSSIAN_COLOR]
    y = [1, 0]
    validity_ax.barh(y, validity_values, color=validity_colors, height=0.48)
    validity_ax.set_yticks(y, validity_labels)
    validity_ax.set_xlim(0, 108)
    validity_ax.set_xlabel("Eligible projects (%)")
    validity_ax.set_title("Control validity", loc="left", fontweight="bold")
    validity_ax.text(validity_values[0] - 2, y[0], f"{len(eligible)}/{len(eligible)}", color="white", ha="right", va="center", fontsize=16, fontweight="bold")
    validity_ax.text(validity_values[1] + 2, y[1], f"{len(matched)}/{len(eligible)}", color=GAUSSIAN_COLOR, ha="left", va="center", fontsize=16, fontweight="bold")
    validity_ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    validity_ax.set_axisbelow(True)
    validity_ax.spines[["top", "right"]].set_visible(False)

    metrics = ["PSNR", "SSIM", "LPIPS"]
    positions = [0, 1, 2]
    offset = 0.18
    quality_ax.barh([position + offset for position in positions], time_percent, height=0.32, color=TIME_COLOR, label=f"Time matched (n={len(eligible)})")
    quality_ax.barh([position - offset for position in positions], gaussian_percent, height=0.32, color=GAUSSIAN_COLOR, label=f"Gaussian matched (n={len(matched)})")
    quality_ax.set_yticks(positions, metrics)
    quality_ax.invert_yaxis()
    quality_ax.set_xlim(0, 108)
    quality_ax.set_xlabel("Projects where control met or exceeded selected model (%)")
    quality_ax.set_title("Quality outcome", loc="left", fontweight="bold")
    for index, (time_value, gaussian_value) in enumerate(zip(time_percent, gaussian_percent)):
        quality_ax.text(time_value + 1.5, index + offset, f"{time_counts[index]}/{len(eligible)}", va="center", fontsize=14, fontweight="bold", color=TIME_COLOR)
        quality_ax.text(gaussian_value + 1.5, index - offset, f"{gaussian_counts[index]}/{len(matched)}", va="center", fontsize=14, fontweight="bold", color=GAUSSIAN_COLOR)
    quality_ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.75)
    quality_ax.set_axisbelow(True)
    quality_ax.spines[["top", "right"]].set_visible(False)
    quality_ax.legend(loc="lower right", frameon=False)

    fig.text(0.5, -0.015, "Positive quality outcome means PSNR/SSIM was at least as high or LPIPS was no greater. DJI hard-cap/partial comparison excluded.", ha="center", fontsize=14, color="#475569")
    save_figure(fig, output_dir, "controlled_experiments_summary")


def create_charts(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    pipeline_ids = {row.get("pipeline_id") for row in rows}
    if pipeline_ids != {EXPECTED_PIPELINE_ID}:
        raise ValueError(f"Expected only {EXPECTED_PIPELINE_ID}, found {sorted(pipeline_ids)}")
    if len(rows) != 12:
        raise ValueError(f"Expected 12 projects, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    time_rows = [row for row in rows if row.get("model_selected_status") == "success"]
    gaussian_rows = [row for row in time_rows if gaussian_target_reached(row)]

    time_budget_chart(rows, output_dir)
    for metric in METRIC_CONFIG:
        delta_chart(
            time_rows,
            "time_control",
            metric,
            f"Time-matched baseline: {metric.upper()} difference",
            f"Controlled baseline minus selected model; {len(time_rows)} normally completed model references",
            output_dir,
            f"time_{metric}_delta",
        )
    gaussian_attainment_chart(rows, output_dir)
    for metric in METRIC_CONFIG:
        delta_chart(
            gaussian_rows,
            "gaussian_control",
            metric,
            f"Gaussian-matched baseline: {metric.upper()} difference",
            f"Only the {len(gaussian_rows)} controls that reached and froze at the selected-model target",
            output_dir,
            f"gaussian_{metric}_delta",
        )
    combined_summary_chart(rows, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create slide-ready charts for controlled experiments.")
    parser.add_argument("--input", type=Path, default=Path("docs/comparison/final_test_june_27/compare.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/comparison/final_test_june_27"))
    args = parser.parse_args()
    create_charts(args.input, args.output_dir)
    print(f"Charts written to {args.output_dir}")


if __name__ == "__main__":
    main()
