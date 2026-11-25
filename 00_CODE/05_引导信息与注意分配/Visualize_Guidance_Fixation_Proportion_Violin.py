"""Draw the split-violin figure for guidance fixation proportion.

This is a visualization-only program. It reads the trial-level
``GuideDwellProportion`` produced by ``Extract_Visual_Information_Features.py``
after the guidance AOI reconstruction pipeline has completed. It does not
reconstruct AOIs, remove outliers, or alter the processed analysis data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT = (
    ROOT
    / "Results_Guidance_Spatial_Causal_Analysis"
    / "data"
    / "ar_information_acquisition_features.csv"
)
RESULT_DIR = ROOT / "Results_Guidance_Fixation_Proportion_Violin"
DEFAULT_OUTPUT = RESULT_DIR / "Guidance_Fixation_Proportion_Violin.png"

METRIC = "GuideDwellProportion"
TITLE = "Guidance Fixation Proportion"
Y_LABEL = "Guidance Fixation Proportion"

SOURCE_GUIDANCE_ORDER = ["ArrowRibbon", "MiniMap", "TextBar", "Full"]
GUIDANCE_NAME_MAP = {
    "ArrowRibbon": "Arrow",
    "Arrow": "Arrow",
    "MiniMap": "MiniMap",
    "TextBar": "TextBar",
    "Full": "Multimodal",
    "Multimodal": "Multimodal",
}
GUIDANCE_ORDER = ["Arrow", "MiniMap", "TextBar", "Multimodal"]
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]

# The palette follows graph_0911/basic_indicator/basic_indicator.py exactly.
PALETTE = {
    "Arrow": {"LowSmoke": "#83E5FC", "HighSmoke": "#398BD1"},
    "MiniMap": {"LowSmoke": "#BDE6EA", "HighSmoke": "#357887"},
    "TextBar": {"LowSmoke": "#FDD8C7", "HighSmoke": "#E99E74"},
    "Multimodal": {"LowSmoke": "#FFC8C8", "HighSmoke": "#CF5149"},
}


def configure_style() -> None:
    """Apply the approved basic-indicator figure style."""
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16.5,
            "axes.linewidth": 1.2,
            "axes.labelsize": 16.5,
            "axes.titlesize": 19.5,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def load_plot_data(path: Path) -> pd.DataFrame:
    """Load and validate the current processed 0.7 m AOI feature table."""
    if not path.is_file():
        raise FileNotFoundError(f"Input feature table not found: {path}")

    frame = pd.read_csv(path, low_memory=False)
    required = {"Guidance", "Smoke", "ParticipantID", METRIC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = frame.loc[
        frame["Guidance"].isin(SOURCE_GUIDANCE_ORDER)
        & frame["Smoke"].isin(SMOKE_ORDER),
        ["Guidance", "Smoke", "ParticipantID", METRIC],
    ].copy()
    data[METRIC] = pd.to_numeric(data[METRIC], errors="coerce")
    data = data.loc[np.isfinite(data[METRIC])].copy()
    data["Guidance"] = data["Guidance"].map(GUIDANCE_NAME_MAP)

    outside = ~data[METRIC].between(0.0, 1.0)
    if outside.any():
        examples = data.loc[outside, METRIC].tolist()[:5]
        raise ValueError(f"{METRIC} must be within [0, 1]; examples: {examples}")

    counts = data.groupby(["Guidance", "Smoke"], observed=True).size()
    expected = pd.MultiIndex.from_product(
        [GUIDANCE_ORDER, SMOKE_ORDER], names=["Guidance", "Smoke"]
    )
    counts = counts.reindex(expected, fill_value=0)
    if (counts < 3).any():
        raise ValueError(f"Too few observations for violin KDE: {counts.to_dict()}")

    return data


def draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    center: float,
    side: int,
    color: str,
    width: float = 0.38,
    alpha: float = 0.92,
    linewidth: float = 1.1,
) -> None:
    """Draw a left (side=-1) or right (side=+1) half violin."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    if len(values) < 3 or np.ptp(values) == 0:
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if y_min == y_max:
            y_min -= 0.01
            y_max += 0.01
        y_grid = np.linspace(max(0.0, y_min), min(1.0, y_max), 50)
        density = np.ones_like(y_grid)
    else:
        kde = gaussian_kde(values)
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        padding = (y_max - y_min) * 0.03
        y_grid = np.linspace(
            max(0.0, y_min - padding),
            min(1.0, y_max + padding),
            300,
        )
        density = kde(y_grid)

    if float(np.max(density)) > 0:
        density = density / np.max(density)
    density = density * width
    x_center = np.full_like(y_grid, center)
    x_outer = center + side * density

    ax.fill(
        np.concatenate([x_center, x_outer[::-1]]),
        np.concatenate([y_grid, y_grid[::-1]]),
        facecolor=color,
        edgecolor="#555555",
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )


def draw_mean_label(
    ax: plt.Axes,
    values: np.ndarray,
    center: float,
    side: int,
    fontsize: float = 11.0,
    x_offset: float = 0.22,
) -> None:
    """Annotate the arithmetic mean on the corresponding violin half."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return

    ax.text(
        center + side * x_offset,
        float(np.mean(values)),
        f"{np.mean(values):.2f}",
        ha="center",
        va="center",
        fontsize=fontsize,
        color="black",
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#555555",
            "linewidth": 0.9,
            "alpha": 1.0,
        },
    )


def draw_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    position: float,
) -> None:
    """Overlay the approved white box-and-whisker summary."""
    ax.boxplot(
        values,
        positions=[position],
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        medianprops={"color": "black", "linewidth": 1.4},
        zorder=4,
    )


def plot(frame: pd.DataFrame, output: Path, dpi: int) -> Path:
    """Render and save the formal 600 dpi PNG."""
    configure_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.6), facecolor="white")

    for index, guidance in enumerate(GUIDANCE_ORDER):
        subset = frame.loc[frame["Guidance"].eq(guidance)]
        for smoke, side, offset in (
            ("LowSmoke", -1, -0.12),
            ("HighSmoke", 1, 0.12),
        ):
            values = subset.loc[subset["Smoke"].eq(smoke), METRIC].to_numpy(float)
            draw_half_violin(
                ax,
                values,
                center=float(index),
                side=side,
                color=PALETTE[guidance][smoke],
            )
            draw_mean_label(
                ax,
                values,
                center=float(index),
                side=side,
            )
            draw_boxplot(ax, values, position=index + offset)

    ax.set_facecolor("white")
    ax.set_title(TITLE, fontsize=21, fontweight="bold", loc="center", pad=12)
    ax.set_xlabel("Guidance", fontsize=16.5, fontweight="bold", labelpad=9, loc="center")
    ax.set_ylabel(Y_LABEL, fontsize=16.5, fontweight="bold", labelpad=9)
    ax.set_xlim(-0.52, len(GUIDANCE_ORDER) - 0.48)
    ax.set_ylim(0.0, min(1.0, max(0.80, float(frame[METRIC].max()) + 0.08)))
    ax.set_xticks(range(len(GUIDANCE_ORDER)))
    ax.set_xticklabels(GUIDANCE_ORDER, fontsize=15, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=15, width=1.0, length=5, pad=6)
    ax.tick_params(axis="y", labelsize=15, width=1.0, length=5)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 7},
    )
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_plot_data(args.input)
    summary = (
        frame.groupby(["Guidance", "Smoke"], observed=True)[METRIC]
        .agg(n="size", mean="mean", median="median", sd="std")
        .reindex(
            pd.MultiIndex.from_product(
                [GUIDANCE_ORDER, SMOKE_ORDER], names=["Guidance", "Smoke"]
            )
        )
    )
    print(f"Input: {args.input}")
    print(summary.round(4).to_string())
    print(f"Saved: {plot(frame, args.output, args.dpi)}")


if __name__ == "__main__":
    main()
