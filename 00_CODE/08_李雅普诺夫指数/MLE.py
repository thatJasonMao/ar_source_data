from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# =========================================================
# Path settings
#
# Expected directory structure:
#
# INDICATORS/
# └── Results_LyapunovMLE/
#     ├── AllSubjects_LyapunovMLE.csv
#     └── MLE_graph/
#         └── MLE.py
# =========================================================

# Current script directory:
# INDICATORS/Results_LyapunovMLE/MLE_graph
SCRIPT_DIR = Path(__file__).resolve().parent

# The raw CSV is located one level above MLE_graph.
DATA_FILE = (
    SCRIPT_DIR.parent
    / "AllSubjects_LyapunovMLE.csv"
)

# Save figures directly to MLE_graph.
OUTPUT_DIR = SCRIPT_DIR

# =========================================================
# Check input file
# =========================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cannot find the MLE input file.\n"
        f"Expected path: {DATA_FILE}\n"
        "Please confirm that AllSubjects_LyapunovMLE.csv "
        "is located in INDICATORS/Results_LyapunovMLE."
    )

print(f"Reading data from: {DATA_FILE}")
print(f"Saving figures to: {OUTPUT_DIR}")

# =========================================================
# SCI paper plotting parameters
# =========================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16.5,
    "axes.linewidth": 1.2,
    "axes.labelsize": 16.5,
    "axes.titlesize": 19.5,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================================================
# Read data
# =========================================================

df = pd.read_csv(DATA_FILE)

# =========================================================
# Check required columns
# =========================================================

required_columns = [
    "Scenario",
    "MLE",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns are missing from the CSV file: "
        + ", ".join(missing_columns)
    )

# =========================================================
# Convert MLE to numeric
# =========================================================

df["MLE"] = pd.to_numeric(
    df["MLE"],
    errors="coerce",
)

# =========================================================
# Extract Guidance and Smoke from Scenario
#
# Example:
# StaticSign-HighSmoke
#
# Extracted result:
# Guidance = StaticSign
# Smoke = HighSmoke
# =========================================================

scenario_info = (
    df["Scenario"]
    .astype(str)
    .str.strip()
    .str.extract(
        r"^(?P<Guidance>.+)-"
        r"(?P<Smoke>LowSmoke|HighSmoke)$"
    )
)

df["Guidance"] = scenario_info["Guidance"]
df["Smoke"] = scenario_info["Smoke"]

# =========================================================
# Standardize category names
#
# Internal names:
# ArrowRibbon
# Full
#
# X-axis display labels:
# Arrow
# Multimodal
# =========================================================

df["Guidance"] = df["Guidance"].replace({
    "Arrow": "ArrowRibbon",
    "Multimodal": "Full",
})

# =========================================================
# Clean invalid data
# =========================================================

before_clean = len(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan,
)

df = df.dropna(
    subset=[
        "MLE",
        "Guidance",
        "Smoke",
    ]
).copy()

removed_count = before_clean - len(df)

print(f"Invalid or missing rows removed: {removed_count}")
print(f"Final sample size: {len(df)}")

# =========================================================
# Guidance order
# =========================================================

guidance_order = [
    "StaticSign",
    "ArrowRibbon",
    "MiniMap",
    "TextBar",
    "Full",
]

# =========================================================
# X-axis display labels
# =========================================================

guidance_labels = [
    "StaticSign",
    "Arrow",
    "MiniMap",
    "TextBar",
    "Multimodal",
]

# =========================================================
# Smoke condition order
# =========================================================

smoke_order = [
    "LowSmoke",
    "HighSmoke",
]

# =========================================================
# Print category check information
# =========================================================

print("\nGuidance categories:")
print(df["Guidance"].value_counts())

print("\nSmoke categories:")
print(df["Smoke"].value_counts())

# =========================================================
# Print descriptive statistics
# =========================================================

print("\nMLE descriptive statistics:")

descriptive_statistics = (
    df.groupby(
        ["Guidance", "Smoke"]
    )["MLE"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
    ])
)

print(descriptive_statistics)

# =========================================================
# Color palette
# =========================================================

palette = {
    "StaticSign": {
        "LowSmoke": "#E6DAF5",
        "HighSmoke": "#6B5C89",
    },
    "ArrowRibbon": {
        "LowSmoke": "#83E5FC",
        "HighSmoke": "#398BD1",
    },
    "MiniMap": {
        "LowSmoke": "#BDE6EA",
        "HighSmoke": "#357887",
    },
    "TextBar": {
        "LowSmoke": "#FDD8C7",
        "HighSmoke": "#E99E74",
    },
    "Full": {
        "LowSmoke": "#FFC8C8",
        "HighSmoke": "#CF5149",
    },
}

# =========================================================
# Create canvas
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 6),
    facecolor="white",
)

ax.set_facecolor("white")

# =========================================================
# Plotting parameters
# =========================================================

half_width = 0.42
box_offset = 0.12
box_width = 0.18
kde_points = 300
violin_alpha = 0.92

# =========================================================
# Draw half violin
#
# side = -1: left half
# side =  1: right half
# =========================================================

def draw_half_violin(
    axis,
    values,
    center,
    side,
    color,
    width=0.42,
    alpha=0.92,
    linewidth=1.1,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    # KDE may be unstable when there are too few values or all values are identical.
    if (
        len(values) < 3
        or np.ptp(values) == 0
    ):
        y_min = float(np.min(values))
        y_max = float(np.max(values))

        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1

        y = np.linspace(
            y_min,
            y_max,
            50,
        )

        density = np.ones_like(y)

    else:
        try:
            kde = gaussian_kde(values)

            y_min = float(np.min(values))
            y_max = float(np.max(values))
            data_range = y_max - y_min

            padding = data_range * 0.03

            if padding == 0:
                padding = 0.1

            y = np.linspace(
                y_min - padding,
                y_max + padding,
                kde_points,
            )

            density = kde(y)

        except np.linalg.LinAlgError:
            y_min = float(np.min(values))
            y_max = float(np.max(values))

            if y_min == y_max:
                y_min -= 0.1
                y_max += 0.1

            y = np.linspace(
                y_min,
                y_max,
                50,
            )

            density = np.ones_like(y)

    # Normalize density to the specified width.
    maximum_density = np.max(density)

    if maximum_density > 0:
        density = density / maximum_density

    density = density * width

    # Center dividing line.
    x_center = np.full_like(
        y,
        center,
        dtype=float,
    )

    if side == -1:
        x_outer = center - density
    else:
        x_outer = center + density

    x_polygon = np.concatenate([
        x_center,
        x_outer[::-1],
    ])

    y_polygon = np.concatenate([
        y,
        y[::-1],
    ])

    axis.fill(
        x_polygon,
        y_polygon,
        facecolor=color,
        edgecolor="#555555",
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )

# =========================================================
# Draw mean marker
# =========================================================

def draw_mean_marker(
    axis,
    values,
    center,
    side,
    color="#D62728",
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    mean_value = float(
        np.mean(values)
    )

    if side == -1:
        x_start = center - 0.36
        x_end = center - 0.02
    else:
        x_start = center + 0.02
        x_end = center + 0.36

    axis.plot(
        [x_start, x_end],
        [mean_value, mean_value],
        color=color,
        linewidth=2.4,
        zorder=7,
        solid_capstyle="round",
    )

# =========================================================
# Draw split violin plots
#
# LowSmoke: left half
# HighSmoke: right half
# =========================================================

for index, guidance in enumerate(guidance_order):
    guidance_subset = df[
        df["Guidance"] == guidance
    ].copy()

    if guidance_subset.empty:
        print(
            "Warning: no data found for "
            f"Guidance={guidance}"
        )
        continue

    low_smoke_values = (
        guidance_subset[
            guidance_subset["Smoke"] == "LowSmoke"
        ]["MLE"]
        .dropna()
        .to_numpy()
    )

    if len(low_smoke_values) > 0:
        draw_half_violin(
            axis=ax,
            values=low_smoke_values,
            center=index,
            side=-1,
            color=palette[guidance]["LowSmoke"],
            width=half_width,
            alpha=violin_alpha,
            linewidth=1.1,
        )

        draw_mean_marker(
            axis=ax,
            values=low_smoke_values,
            center=index,
            side=-1,
            color="#D62728",
        )

    high_smoke_values = (
        guidance_subset[
            guidance_subset["Smoke"] == "HighSmoke"
        ]["MLE"]
        .dropna()
        .to_numpy()
    )

    if len(high_smoke_values) > 0:
        draw_half_violin(
            axis=ax,
            values=high_smoke_values,
            center=index,
            side=1,
            color=palette[guidance]["HighSmoke"],
            width=half_width,
            alpha=violin_alpha,
            linewidth=1.1,
        )

        draw_mean_marker(
            axis=ax,
            values=high_smoke_values,
            center=index,
            side=1,
            color="#D62728",
        )

# =========================================================
# Add box plots
#
# LowSmoke: left side
# HighSmoke: right side
# =========================================================

for index, guidance in enumerate(guidance_order):
    guidance_subset = df[
        df["Guidance"] == guidance
    ].copy()

    if guidance_subset.empty:
        continue

    for smoke in smoke_order:
        values = (
            guidance_subset[
                guidance_subset["Smoke"] == smoke
            ]["MLE"]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        if smoke == "LowSmoke":
            position = index - box_offset
        else:
            position = index + box_offset

        ax.boxplot(
            values,
            positions=[position],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.1,
            },
            whiskerprops={
                "color": "black",
                "linewidth": 1.0,
            },
            capprops={
                "color": "black",
                "linewidth": 1.0,
            },
            medianprops={
                "color": "black",
                "linewidth": 1.4,
            },
            zorder=4,
        )

# =========================================================
# Title
# =========================================================

ax.set_title(
    "Maximum Lyapunov Exponent",
    fontsize=19.5,
    fontweight="bold",
    loc="center",
    pad=14,
)

# =========================================================
# X-axis
# =========================================================

ax.set_xlabel(
    "Guidance",
    fontsize=16.5,
    fontweight="bold",
    labelpad=6,
)

ax.set_xticks(
    range(len(guidance_order))
)

ax.set_xticklabels(
    guidance_labels,
    fontsize=15,
    rotation=0,
    ha="center",
)

# =========================================================
# Y-axis
# =========================================================

ax.set_ylabel(
    "Maximum Lyapunov Exponent (MLE)",
    fontsize=16.5,
    fontweight="bold",
    labelpad=6,
)

# =========================================================
# Ticks
# =========================================================

ax.tick_params(
    axis="x",
    labelsize=15,
    width=1.0,
    length=4,
    pad=4,
)

ax.tick_params(
    axis="y",
    labelsize=15,
    width=1.0,
    length=4,
)

# =========================================================
# Grid lines
# =========================================================

ax.yaxis.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35,
)

ax.xaxis.grid(False)
ax.set_axisbelow(True)

# =========================================================
# Axis spines
# =========================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

# =========================================================
# X-axis limits
# =========================================================

ax.set_xlim(
    -0.5,
    len(guidance_order) - 0.5,
)

# =========================================================
# Legend
# =========================================================

legend_handles = [
    Patch(
        facecolor="#BDE6EA",
        edgecolor="#555555",
        linewidth=1.0,
        label="Low smoke",
    ),
    Patch(
        facecolor="#357887",
        edgecolor="#555555",
        linewidth=1.0,
        label="High smoke",
    ),
]

ax.legend(
    handles=legend_handles,
    title="Smoke condition",
    loc="upper right",
    frameon=False,
    fontsize=15,
    title_fontsize=16.5,
    handlelength=1.4,
    handleheight=1.2,
    borderaxespad=0.3,
)

# =========================================================
# Automatic layout
# =========================================================

plt.tight_layout()

# =========================================================
# Save figures
#
# All three files are saved to the MLE_graph directory where MLE.py is located.
# =========================================================

png_file = OUTPUT_DIR / "MLE_violin.png"
tiff_file = OUTPUT_DIR / "MLE_violin.tiff"
pdf_file = OUTPUT_DIR / "MLE_violin.pdf"

plt.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    tiff_file,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    pdf_file,
    bbox_inches="tight",
    facecolor="white",
)

print("\nFigures generated successfully:")
print(f"PNG:  {png_file}")
print(f"TIFF: {tiff_file}")
print(f"PDF:  {pdf_file}")

# =========================================================
# Show figure
# =========================================================

plt.show()