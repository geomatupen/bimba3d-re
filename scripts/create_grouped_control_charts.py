from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPECTED_PIPELINE_ID = "pipeline_5df0ad97ef5a"
BASELINE_COLOR = "#64748B"
MODEL_COLOR = "#2563EB"
TIME_COLOR = "#0F766E"
GAUSSIAN_COLOR = "#B45309"
PARTIAL_COLOR = "#94A3B8"
GRID_COLOR = "#CBD5E1"
TEXT_COLOR = "#172033"

SHORT_NAMES = {
    "12-Maisonneuve Market": "Maisonneuve",
    "4-Thomas-More-Church": "Thomas More",
    "Brasov": "Brasov",
    "Changu": "Changu",
    "Chatteau_circle_60_and_45_degrees": "Chateau",
    "DJI_202402171501_006_KTM20-oblique": "DJI KTM20*",
    "Kninice_Church_of_the_Exaltation_of_the_Holy_Cross": "Kninice",
    "Pix4d_forensic": "Pix4D",
    "Telc_sv_Jakub_March_18": "Telc",
    "Tiburon_Angel_Islands_State_Park_Split_group_3": "Tiburon",
    "morice": "Morice",
    "uncovice": "Uncovice",
}

METRICS = {
    "psnr": {"label": "PSNR (dB)", "decimals": 2, "better": "Higher is better"},
    "ssim": {"label": "SSIM", "decimals": 3, "better": "Higher is better"},
    "lpips": {"label": "LPIPS", "decimals": 3, "better": "Lower is better"},
}


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if not value:
        raise ValueError(f"Missing {key} for {row.get('project')}")
    return float(value)


def target_reached(row: dict[str, str]) -> bool:
    final_step = float(row["gaussian_control_final_step"]) if row.get("gaussian_control_final_step") else None
    step_limit = float(row["gaussian_control_step_limit"]) if row.get("gaussian_control_step_limit") else None
    target = number(row, "gaussian_control_target_count")
    actual = number(row, "gaussian_control_gaussian_count")
    return (
        row.get("gaussian_control_status") == "success"
        and final_step is not None
        and step_limit is not None
        and final_step < step_limit
        and abs(actual / target - 1) <= 0.05
    )


def chart_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "figure.titlesize": 27,
        "axes.titlesize": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": "#94A3B8",
        "axes.unicode_minus": False,
    })


def metric_limits(rows: list[dict[str, str]], metric: str) -> tuple[float, float]:
    values: list[float] = []
    for row in rows:
        values.extend([
            number(row, f"baseline_{metric}"),
            number(row, f"model_selected_{metric}"),
            number(row, f"time_control_{metric}"),
            number(row, f"gaussian_control_{metric}"),
        ])
    low, high = min(values), max(values)
    span = high - low
    if span == 0:
        span = max(abs(high) * 0.05, 0.01)
    return low - span * 0.10, high + span * 0.22


def format_seconds(value: float) -> str:
    return f"{value:,.1f}s"


def format_step(value: float) -> str:
    return f"{int(value):,}"


def format_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,.0f}"


def label_bars(ax: plt.Axes, bars, decimals: int, y_limits: tuple[float, float]) -> None:
    offset = (y_limits[1] - y_limits[0]) * 0.016
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=13,
            rotation=90,
            fontweight="bold",
            color=TEXT_COLOR,
        )


def resource_text(row: dict[str, str], control_kind: str) -> tuple[str, str, str]:
    model = (
        f"M  S:{format_step(number(row, 'model_selected_last_observed_step'))}\n"
        f"T:{format_seconds(number(row, 'model_selected_training_seconds'))}  "
        f"G:{format_count(number(row, 'model_selected_gaussian_count'))}"
    )
    prefix = f"{control_kind}_control"
    control = (
        f"C  S:{format_step(number(row, f'{prefix}_last_observed_step'))}\n"
        f"T:{format_seconds(number(row, f'{prefix}_training_seconds'))}  "
        f"G:{format_count(number(row, f'{prefix}_gaussian_count'))}"
    )
    if control_kind == "time":
        target = f"Target T: {format_seconds(number(row, 'time_control_target_seconds'))}"
    else:
        target = (
            f"Target G: {format_count(number(row, 'gaussian_control_target_count'))}\n"
            f"Step limit S: {format_step(number(row, 'gaussian_control_step_limit'))}"
        )
        if row.get("gaussian_control_status") == "partial":
            control = "C [unfinished]" + control[1:]
        elif target_reached(row):
            control = "C [reached]" + control[1:]
        else:
            control = "C [limit]" + control[1:]
    return target, model, control


def add_resource_band(ax: plt.Axes, rows: list[dict[str, str]], control_kind: str) -> None:
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, row in enumerate(rows):
        if index:
            ax.axvline(index - 0.5, ymin=0.08, ymax=0.92, color="#E2E8F0", linewidth=1)
        target, model, control = resource_text(row, control_kind)
        ax.text(
            index - 0.25,
            0.50,
            target,
            ha="center",
            va="center",
            fontsize=11.8,
            color=TEXT_COLOR,
            fontweight="bold",
            rotation=90,
        )
        ax.text(
            index,
            0.50,
            model,
            ha="center",
            va="center",
            fontsize=11.8,
            linespacing=1.25,
            color=MODEL_COLOR,
            fontweight="bold",
            rotation=90,
        )
        ax.text(
            index + 0.28,
            0.50,
            control,
            ha="center",
            va="center",
            fontsize=11.8,
            linespacing=1.25,
            color=TIME_COLOR if control_kind == "time" else GAUSSIAN_COLOR,
            fontweight="bold",
            rotation=90,
        )


def grouped_chart(
    rows: list[dict[str, str]],
    metric: str,
    control_kind: str,
    output_dir: Path,
    y_limits: tuple[float, float],
) -> None:
    config = METRICS[metric]
    x = np.arange(len(rows), dtype=float)
    width = 0.24
    baseline = [number(row, f"baseline_{metric}") for row in rows]
    model = [number(row, f"model_selected_{metric}") for row in rows]
    control_prefix = "time_control" if control_kind == "time" else "gaussian_control"
    control = [number(row, f"{control_prefix}_{metric}") for row in rows]
    control_color = TIME_COLOR if control_kind == "time" else GAUSSIAN_COLOR
    control_name = "Time control" if control_kind == "time" else "Gaussian control"

    fig, (resource_ax, quality_ax) = plt.subplots(
        2,
        1,
        figsize=(24, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.8, 4.5], "hspace": 0.02},
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.79, bottom=0.13)
    fig.suptitle(f"{control_name}: {config['label']} comparison", fontweight="bold", y=0.975)
    fig.text(
        0.5,
        0.923,
        f"Quality bars use the left axis; training-time dots use the right axis  |  {config['better']}  |  Truncated quality axis",
        ha="center",
        fontsize=17,
        color="#475569",
    )
    fig.text(
        0.5,
        0.885,
        f"M = model selected   C = {'time-controlled' if control_kind == 'time' else 'Gaussian-controlled'} baseline   "
        "S = final/observed step   T = training-loop time   G = Gaussian count",
        ha="center",
        fontsize=14.5,
        color=TEXT_COLOR,
        fontweight="bold",
    )

    baseline_bars = quality_ax.bar(x - width, baseline, width, color=BASELINE_COLOR)
    model_bars = quality_ax.bar(x, model, width, color=MODEL_COLOR)
    control_bars = quality_ax.bar(x + width, control, width, color=control_color)

    model_times = [number(row, "model_selected_training_seconds") for row in rows]
    control_times = [number(row, f"{control_prefix}_training_seconds") for row in rows]
    time_ax = quality_ax.twinx()
    time_ax.scatter(
        x,
        model_times,
        s=105,
        marker="o",
        facecolor=MODEL_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=6,
    )
    time_ax.scatter(
        x + width,
        control_times,
        s=110,
        marker="D",
        facecolor=control_color,
        edgecolor="white",
        linewidth=1.8,
        zorder=6,
    )
    time_ax.set_ylim(0, max(model_times + control_times) * 1.12)
    time_ax.set_ylabel("Training-loop time (seconds)", color="#475569", labelpad=15)
    time_ax.tick_params(axis="y", colors="#475569")
    time_ax.spines["top"].set_visible(False)
    time_ax.spines["right"].set_color("#94A3B8")
    time_ax.grid(False)

    if control_kind == "gaussian":
        for bar, row in zip(control_bars, rows):
            if row.get("gaussian_control_status") == "partial":
                bar.set_facecolor("white")
                bar.set_edgecolor(PARTIAL_COLOR)
                bar.set_linewidth(2)
            elif not target_reached(row):
                bar.set_facecolor("#FFEDD5")
                bar.set_hatch("//")
                bar.set_edgecolor(GAUSSIAN_COLOR)

    quality_ax.set_ylim(*y_limits)
    quality_ax.set_ylabel(config["label"])
    quality_ax.set_xticks(x, [SHORT_NAMES.get(row["project"], row["project"]) for row in rows], rotation=24, ha="right")
    quality_ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.8)
    quality_ax.set_axisbelow(True)
    quality_ax.spines[["top", "right"]].set_visible(False)
    label_bars(quality_ax, baseline_bars, config["decimals"], y_limits)
    label_bars(quality_ax, model_bars, config["decimals"], y_limits)
    label_bars(quality_ax, control_bars, config["decimals"], y_limits)

    legend_items = [
        Patch(facecolor=BASELINE_COLOR, label="Baseline"),
        Patch(facecolor=MODEL_COLOR, label="Model selected (M)"),
        Patch(facecolor=control_color, label=f"{control_name} (C)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=MODEL_COLOR, markeredgecolor="white", markersize=9, label="Model-selected training time"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=control_color, markeredgecolor="white", markersize=9, label="C training time"),
    ]
    if control_kind == "gaussian":
        legend_items.extend([
            Patch(facecolor="white", edgecolor=GAUSSIAN_COLOR, hatch="//", label="Stopped at step limit (15k)"),
            Patch(facecolor="white", edgecolor=PARTIAL_COLOR, linewidth=2, label="Unfinished control run"),
        ])
    fig.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.855), ncol=len(legend_items), frameon=False)

    add_resource_band(resource_ax, rows, control_kind)
    footnote = (
        "* Timing is recorded before evaluation at the reported step and excludes export, but includes earlier scheduled evaluations.\n"
        "  If an earlier evaluation crosses the target, stopping after the next step causes a time jump. "
        "DJI model-selected quality is from its last evaluation before the Gaussian hard cap."
    )
    if control_kind == "gaussian":
        footnote += " Hatched controls did not reach the target; the outlined DJI control run was unfinished."
    fig.text(0.065, 0.025, footnote, fontsize=13.0, color="#475569", linespacing=1.35)

    name = f"{control_kind}_{metric}_grouped_comparison"
    fig.savefig(output_dir / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def create_charts(input_csv: Path, output_dir: Path) -> None:
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("pipeline_id") for row in rows} != {EXPECTED_PIPELINE_ID}:
        raise ValueError("Input CSV is not the Final_test_June_27 pipeline export")
    if len(rows) != 12:
        raise ValueError(f"Expected 12 projects, found {len(rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_style()
    limits = {metric: metric_limits(rows, metric) for metric in METRICS}
    for control_kind in ("time", "gaussian"):
        for metric in METRICS:
            grouped_chart(rows, metric, control_kind, output_dir, limits[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create six grouped metric-and-cost comparison charts.")
    parser.add_argument("--input", type=Path, default=Path("docs/comparison/final_test_june_27/compare.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/comparison/final_test_june_27"))
    args = parser.parse_args()
    create_charts(args.input, args.output_dir)
    print(f"Six grouped charts written to {args.output_dir}")


if __name__ == "__main__":
    main()
