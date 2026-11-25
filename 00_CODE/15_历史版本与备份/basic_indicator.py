import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from pathlib import Path

# SCI paper global parameters
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Remove subjects containing 'TEST'
def remove_test_subject(df):
    if 'Subject' in df.columns:
        df = df[~df['Subject'].astype(str).str.contains('TEST', case=False, na=False)].copy()
    return df

# IQR outlier removal per Scenario group
def remove_iqr_outliers(df, value_col, group_col='Scenario'):
    clean_data = []
    for name, group in df.groupby(group_col):
        Q1 = group[value_col].quantile(0.25)
        Q3 = group[value_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        group_clean = group[(group[value_col] >= lower) & (group[value_col] <= upper)]
        clean_data.append(group_clean)
    return pd.concat(clean_data, ignore_index=True)

# Read four metric files (script-relative paths routed to the original results)
SCRIPT_DIR = Path(__file__).resolve().parent
INDICATORS_DIR = SCRIPT_DIR.parent.parent / "INDICATORS"

distance = pd.read_csv(INDICATORS_DIR / "Results_EvacuationDistance" / "AllSubjects_EvacuationDistance.csv")
rotation = pd.read_csv(INDICATORS_DIR / "Results_EvacuationRotation" / "AllSubjects_EvacuationRotation.csv")
speed = pd.read_csv(INDICATORS_DIR / "Results_EvacuationSpeed" / "AllSubjects_EvacuationSpeed.csv")
time = pd.read_csv(INDICATORS_DIR / "Results_EvacuationTime" / "AllSubjects_EvacuationTime.csv")

# Define metrics information
datasets = {
    "Distance": [distance, "EvacuationDistance", "Evacuation Distance (m)"],
    "Rotation": [rotation, "TotalRotation_deg", "Total Rotation (deg)"],
    "Speed": [speed, "AverageSpeed_m_s", "Average Speed (m/s)"],
    "Time": [time, "EvacuationTime_s", "Evacuation Time (s)"]
}

# Preprocess data: remove TEST + IQR outlier cleaning
for key, value in datasets.items():
    df = value[0].copy()
    col = value[1]
    original = len(df)

    df = remove_test_subject(df)
    after_test = len(df)

    before_iqr = len(df)
    df = remove_iqr_outliers(df, col, "Scenario")
    after_iqr = len(df)

    print(f"\n{'=' * 45}\n{key}\n{'=' * 45}")
    print(f"Original data: {original}")
    print(f"TEST removed: {original - after_test}")
    print(f"IQR outliers removed: {before_iqr - after_iqr}")
    print(f"Final data: {after_iqr}")

    datasets[key][0] = df

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor='white')
axes = axes.flatten()

palette = {
    "StaticSign": {
        "LowSmoke": "#D9E2E8",
        "HighSmoke": "#6F8797"
    },
    "Full": {
        "LowSmoke": "#FFCACA",
        "HighSmoke": "#D15850"
    },
    "ArrowRibbon": {
        "LowSmoke": "#88E6FC",
        "HighSmoke": "#418FD2"
    },
    "MiniMap": {
        "LowSmoke": "#BFE7EA",
        "HighSmoke": "#3D7D8C"
    },
    "TextBar": {
        "LowSmoke": "#FDD9C9",
        "HighSmoke": "#E9A279"
    }
}

# Subplot titles
titles = [
    "(a) Evacuation Distance",
    "(b) Total Rotation",
    "(c) Average Speed",
    "(d) Evacuation Time"
]

# Guidance order
guidance_order = ["StaticSign", "Full", "ArrowRibbon", "MiniMap", "TextBar"]
smoke_order = ["HighSmoke", "LowSmoke"]

# Draw half violin helper
def draw_half_violin(ax, values, center, side, color,
                     width=0.38, alpha=0.92, linewidth=1.1):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    if len(values) < 3 or np.ptp(values) == 0:
        ymin = np.min(values)
        ymax = np.max(values)
        if ymin == ymax:
            ymin -= 0.1
            ymax += 0.1
        y = np.linspace(ymin, ymax, 50)
        density = np.ones_like(y)
    else:
        kde = gaussian_kde(values)
        y_min = np.min(values)
        y_max = np.max(values)
        data_range = y_max - y_min
        padding = data_range * 0.03
        if padding == 0:
            padding = 0.1
        y = np.linspace(y_min - padding, y_max + padding, 300)
        density = kde(y)

    if np.max(density) > 0:
        density = density / np.max(density)
    density = density * width

    x_center = np.full_like(y, center)
    x_outer = center - density if side == -1 else center + density

    x_polygon = np.concatenate([x_center, x_outer[::-1]])
    y_polygon = np.concatenate([y, y[::-1]])

    ax.fill(
        x_polygon,
        y_polygon,
        facecolor=color,
        edgecolor="#555555",
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )

# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), facecolor="white")
axes = axes.flatten()

titles = [
    "(a) Evacuation Distance",
    "(b) Total Rotation",
    "(c) Average Speed",
    "(d) Evacuation Time"
]

guidance_order = ["StaticSign", "Full", "ArrowRibbon", "MiniMap", "TextBar"]
smoke_order = ["LowSmoke", "HighSmoke"]

for ax, (key, value), title in zip(axes, datasets.items(), titles):
    df = value[0].copy()
    ycol = value[1]
    ylabel = value[2]

    df["Smoke"] = df["Scenario"].apply(
        lambda x: "HighSmoke" if "HighSmoke" in str(x) else "LowSmoke"
    )
    df["Guidance"] = (
        df["Scenario"]
        .astype(str)
        .str.replace("-HighSmoke", "", regex=False)
        .str.replace("-LowSmoke", "", regex=False)
    )

    for i, guidance in enumerate(guidance_order):
        subset = df[df["Guidance"] == guidance].copy()

        if subset.empty:
            continue

        low_values = subset[
            subset["Smoke"] == "LowSmoke"
        ][ycol].dropna().values

        high_values = subset[
            subset["Smoke"] == "HighSmoke"
        ][ycol].dropna().values

        if len(low_values) > 0:
            draw_half_violin(
                ax=ax,
                values=low_values,
                center=i,
                side=-1,
                color=palette[guidance]["LowSmoke"]
            )

        if len(high_values) > 0:
            draw_half_violin(
                ax=ax,
                values=high_values,
                center=i,
                side=1,
                color=palette[guidance]["HighSmoke"]
            )

        # Add boxplots at the center of each corresponding half.
        for smoke, offset in [("LowSmoke", -0.12), ("HighSmoke", 0.12)]:
            values = subset[
                subset["Smoke"] == smoke
            ][ycol].dropna().values

            if len(values) == 0:
                continue

            ax.boxplot(
                values,
                positions=[i + offset],
                widths=0.16,
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

    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    ax.set_xticks(range(len(guidance_order)))
    ax.set_xticklabels(
        guidance_order,
        fontsize=10,
        rotation=20
    )
    ax.tick_params(
        axis="x",
        labelsize=10,
        width=1.0,
        length=4,
        pad=4
    )
    ax.tick_params(
        axis="y",
        labelsize=10,
        width=1.0,
        length=4
    )
    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

# Create unified legend
legend_handles = [
    Patch(
        facecolor="#BFE7EA",
        edgecolor="#555555",
        linewidth=1.0,
        label="Low smoke"
    ),
    Patch(
        facecolor="#3D7D8C",
        edgecolor="#555555",
        linewidth=1.0,
        label="High smoke"
    )
]

fig.legend(
    handles=legend_handles,
    title="Smoke condition",
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=11,
    title_fontsize=11,
    bbox_to_anchor=(0.5, 0.01)
)

# Adjust layout
plt.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.13, wspace=0.14, hspace=0.33)

# Save figures (optional, currently commented out)
plt.savefig(r"basic_indicator.png", dpi=600, bbox_inches="tight")
plt.savefig(r"basic_indicator.pdf", bbox_inches="tight")

plt.show()