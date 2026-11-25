import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path


# =========================================================
# Script directory
# All input/output paths are anchored to the folder that contains
# this script, so it can be run from any working directory.
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent


# =========================================================
# SCI paper plot parameters
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
df = pd.read_csv(
    SCRIPT_DIR / "saccade_amplitude.csv"
)


# =========================================================
# Indicator
# =========================================================
ycol = "MedianSaccadeAmplitude_deg"
ylabel = "Median Saccade Amplitude (deg)"


# =========================================================
# Required columns
# =========================================================
required_columns = [
    ycol,
    "Scenario",
    "Guidance",
    "Smoke"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column not found: {col}"
        )


# =========================================================
# Normalize category names
#
# Use ArrowRibbon internally.
# Display it as Arrow only on the x-axis.
# =========================================================
df["Guidance"] = (
    df["Guidance"]
    .astype(str)
    .str.strip()
    .replace({
        "Arrow": "ArrowRibbon",
        "Multimodal": "Full"
    })
)

df["Smoke"] = (
    df["Smoke"]
    .astype(str)
    .str.strip()
)


# =========================================================
# Keep positive values only
# =========================================================
df = df[
    df[ycol] > 0
].copy()


# =========================================================
# Drop missing values
# =========================================================
before_nan = len(df)

df = df.dropna(
    subset=required_columns
).copy()

print(
    f"Missing values removed: "
    f"{before_nan - len(df)}"
)


# =========================================================
# Check actual category names
# =========================================================
print("\nGuidance categories:")
print(df["Guidance"].value_counts())

print("\nSmoke categories:")
print(df["Smoke"].value_counts())


# =========================================================
# IQR outlier removal
# Grouped by Scenario
# =========================================================
def remove_iqr_outliers(
    data,
    value_col,
    group_col="Scenario"
):
    clean_data = []
    removed_total = 0

    for name, group in data.groupby(group_col):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        group_clean = group[
            (group[value_col] >= lower) &
            (group[value_col] <= upper)
        ].copy()

        removed = (
            len(group) -
            len(group_clean)
        )

        removed_total += removed

        print(
            f"{name}: "
            f"Q1={Q1:.3f}, "
            f"Q3={Q3:.3f}, "
            f"IQR={IQR:.3f}, "
            f"Lower={lower:.3f}, "
            f"Upper={upper:.3f}, "
            f"Removed={removed}"
        )

        clean_data.append(
            group_clean
        )

    if len(clean_data) == 0:
        return data.copy(), 0

    clean_df = pd.concat(
        clean_data,
        ignore_index=True
    )

    return clean_df, removed_total


print("\n" + "=" * 60)
print("IQR outlier removal")
print("=" * 60)

before_iqr = len(df)

df, iqr_removed = remove_iqr_outliers(
    df,
    ycol,
    "Scenario"
)

after_iqr = len(df)

print("=" * 60)
print(f"IQR outliers removed: {iqr_removed}")
print(f"Sample size before IQR: {before_iqr}")
print(f"Final sample size: {after_iqr}")
print("=" * 60)


# =========================================================
# Final descriptive statistics
# =========================================================
print("\nFinal data:")

print(
    df.groupby(
        ["Guidance", "Smoke"]
    )[ycol].agg([
        "count",
        "mean",
        "median",
        "std"
    ])
)


# =========================================================
# Guidance order
#
# These are the original values used in the CSV.
# =========================================================
guidance_order = [
    "StaticSign",
    "ArrowRibbon",
    "MiniMap",
    "TextBar",
    "Full"
]


# =========================================================
# X-axis display labels
# =========================================================
guidance_labels = [
    "StaticSign",
    "Arrow",
    "MiniMap",
    "TextBar",
    "Multimodal"
]


smoke_order = [
    "LowSmoke",
    "HighSmoke"
]


# =========================================================
# Palette
# =========================================================
palette = {
    "StaticSign": {
        "LowSmoke": "#E6DAF5",
        "HighSmoke": "#6B5C89"
    },

    "ArrowRibbon": {
        "LowSmoke": "#83E5FC",
        "HighSmoke": "#398BD1"
    },

    "MiniMap": {
        "LowSmoke": "#BDE6EA",
        "HighSmoke": "#357887"
    },

    "TextBar": {
        "LowSmoke": "#FDD8C7",
        "HighSmoke": "#E99E74"
    },

    "Full": {
        "LowSmoke": "#FFC8C8",
        "HighSmoke": "#CF5149"
    }
}


# =========================================================
# Figure
# =========================================================
fig, ax = plt.subplots(
    figsize=(10, 6),
    facecolor="white"
)

ax.set_facecolor(
    "white"
)


# =========================================================
# Plot parameters
# =========================================================
half_width = 0.42
box_offset = 0.12
box_width = 0.18
kde_points = 300
violin_alpha = 0.92


# =========================================================
# Custom half-violin function
#
# side = -1: left half
# side =  1: right half
# =========================================================
def draw_half_violin(
    ax,
    values,
    center,
    side,
    color,
    width=0.42,
    alpha=0.92,
    linewidth=1.1
):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    # Use a simple density when KDE is unstable.
    if len(values) < 3 or np.ptp(values) == 0:
        y_min = np.min(values)
        y_max = np.max(values)

        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1

        y = np.linspace(
            y_min,
            y_max,
            50
        )

        density = np.ones_like(y)

    else:
        kde = gaussian_kde(values)

        y_min = np.min(values)
        y_max = np.max(values)
        data_range = y_max - y_min
        padding = data_range * 0.03

        if padding == 0:
            padding = 0.1

        y = np.linspace(
            y_min - padding,
            y_max + padding,
            kde_points
        )

        density = kde(y)

    # Normalize density.
    if np.max(density) > 0:
        density = (
            density /
            np.max(density)
        )

    density = density * width

    # The center forms the vertical split line.
    x_center = np.full_like(
        y,
        center,
        dtype=float
    )

    if side == -1:
        x_outer = center - density
    else:
        x_outer = center + density

    x_polygon = np.concatenate([
        x_center,
        x_outer[::-1]
    ])

    y_polygon = np.concatenate([
        y,
        y[::-1]
    ])

    ax.fill(
        x_polygon,
        y_polygon,
        facecolor=color,
        edgecolor="#555555",
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )


# =========================================================
# Mean label
# Text + opaque rounded background
# =========================================================
def draw_mean_label(
    ax,
    values,
    center,
    side,
    fontsize=11.0,
    x_offset=0.22
):
    """
    Annotate the group mean with a two-decimal value label
    drawn as text on an opaque rounded background.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return

    mean_val = float(
        np.mean(values)
    )

    if side == -1:
        x_pos = center - x_offset
    else:
        x_pos = center + x_offset

    ax.text(
        x_pos,
        mean_val,
        f"{mean_val:.2f}",
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
            "alpha": 1.0
        }
    )


# =========================================================
# Draw vertically split half-violins
#
# LowSmoke  = left
# HighSmoke = right
# =========================================================
for i, guidance in enumerate(guidance_order):
    sub = df[
        df["Guidance"] == guidance
    ].copy()

    if sub.empty:
        print(
            f"Warning: no data found for Guidance={guidance}"
        )
        continue

    # -----------------------------------------------------
    # LowSmoke: left half
    # -----------------------------------------------------
    low_values = (
        sub[
            sub["Smoke"] == "LowSmoke"
        ][ycol]
        .dropna()
        .values
    )

    if len(low_values) > 0:
        draw_half_violin(
            ax=ax,
            values=low_values,
            center=i,
            side=-1,
            color=palette[guidance]["LowSmoke"],
            width=half_width,
            alpha=violin_alpha,
            linewidth=1.1
        )

        draw_mean_label(
            ax=ax,
            values=low_values,
            center=i,
            side=-1
        )

    # -----------------------------------------------------
    # HighSmoke: right half
    # -----------------------------------------------------
    high_values = (
        sub[
            sub["Smoke"] == "HighSmoke"
        ][ycol]
        .dropna()
        .values
    )

    if len(high_values) > 0:
        draw_half_violin(
            ax=ax,
            values=high_values,
            center=i,
            side=1,
            color=palette[guidance]["HighSmoke"],
            width=half_width,
            alpha=violin_alpha,
            linewidth=1.1
        )

        draw_mean_label(
            ax=ax,
            values=high_values,
            center=i,
            side=1
        )


# =========================================================
# Add boxplots
#
# LowSmoke  = left
# HighSmoke = right
# =========================================================
for i, guidance in enumerate(guidance_order):
    sub = df[
        df["Guidance"] == guidance
    ].copy()

    if sub.empty:
        continue

    for smoke in smoke_order:
        values = (
            sub[
                sub["Smoke"] == smoke
            ][ycol]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        if smoke == "LowSmoke":
            position = i - box_offset
        else:
            position = i + box_offset

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
                "linewidth": 1.1
            },
            whiskerprops={
                "color": "black",
                "linewidth": 1.0
            },
            capprops={
                "color": "black",
                "linewidth": 1.0
            },
            medianprops={
                "color": "black",
                "linewidth": 1.4
            },
            zorder=4
        )


# =========================================================
# Title
# =========================================================
ax.set_title(
    "Median Saccade Amplitude",
    fontsize=21,
    fontweight="bold",
    loc="center",
    pad=12
)


# =========================================================
# X axis
# =========================================================
ax.set_xlabel(
    "Guidance",
    fontsize=16.5,
    fontweight="bold",
    labelpad=6
)

ax.set_xticks(
    range(len(guidance_order))
)

ax.set_xticklabels(
    guidance_labels,
    fontsize=15,
    rotation=0,
    ha="center"
)


# =========================================================
# Y axis
# =========================================================
ax.set_ylabel(
    ylabel,
    fontsize=16.5,
    fontweight="bold",
    labelpad=6
)


# =========================================================
# Tick parameters
# =========================================================
ax.tick_params(
    axis="x",
    labelsize=15,
    width=1.0,
    length=4,
    pad=4
)

ax.tick_params(
    axis="y",
    labelsize=15,
    width=1.0,
    length=4
)


# =========================================================
# Grid
# =========================================================
ax.yaxis.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

ax.xaxis.grid(False)
ax.set_axisbelow(True)


# =========================================================
# Spines
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
    len(guidance_order) - 0.5
)


# =========================================================
# Layout
# =========================================================
plt.tight_layout()


# =========================================================
# Save
# =========================================================
plt.savefig(
    SCRIPT_DIR / "saccade_amplitude.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    SCRIPT_DIR / "saccade_amplitude.tiff",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)


# =========================================================
# Display
# =========================================================
plt.show()