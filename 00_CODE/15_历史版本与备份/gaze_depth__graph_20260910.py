import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


root_dir = r"depth"
ycol = "LocalPosZ"
xlabel = "Guidance"
ylabel = "Gaze Depth (LocalPosZ)"

guidance_order = [
    "StaticSign",
    "Full",
    "ArrowRibbon",
    "MiniMap",
    "TextBar"
]

smoke_order = [
    "LowSmoke",
    "HighSmoke"
]

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


if not os.path.exists(root_dir):
    raise FileNotFoundError(
        f"Folder not found: {root_dir}\n"
        f"Please check whether depth folder exists."
    )


def parse_scenario_folder(folder_name):
    guidance = None
    smoke = None

    for guidance_name in guidance_order:
        if folder_name.startswith(guidance_name):
            guidance = guidance_name
            break

    if "HighSmoke" in folder_name:
        smoke = "HighSmoke"
    elif "LowSmoke" in folder_name:
        smoke = "LowSmoke"

    return guidance, smoke


csv_files = glob.glob(
    os.path.join(
        root_dir,
        "*",
        "*",
        "RelativePosInfo.csv"
    )
)

print("=" * 60)
print(f"Number of RelativePosInfo.csv files found: {len(csv_files)}")
print("=" * 60)

if not csv_files:
    raise FileNotFoundError(
        "No RelativePosInfo.csv found.\n\n"
        "Expected directory structure:\n"
        "depth\\Scenario-Smoke\\Participant\\RelativePosInfo.csv"
    )


all_data = []

for file in csv_files:
    participant_dir = os.path.dirname(file)
    scenario_dir = os.path.dirname(participant_dir)
    scenario_folder = os.path.basename(scenario_dir)
    participant = os.path.basename(participant_dir)

    guidance, smoke = parse_scenario_folder(scenario_folder)

    if guidance is None or smoke is None:
        print(f"Cannot parse scenario folder: {scenario_folder}")
        print(f"File: {file}")
        continue

    if "TEST" in participant.upper():
        print(f"Skipping TEST participant: {participant}")
        continue

    try:
        temp = pd.read_csv(file)
    except Exception as e:
        print(f"CSV read failed: {file}")
        print(f"Error: {e}")
        continue

    if ycol not in temp.columns:
        print(f"Column {ycol} not found in: {file}")
        continue

    temp = temp[[ycol]].copy()
    temp[ycol] = pd.to_numeric(temp[ycol], errors="coerce")

    temp["Guidance"] = guidance
    temp["Smoke"] = smoke
    temp["ParticipantID"] = participant
    temp["Scenario"] = scenario_folder
    temp["File"] = file

    all_data.append(temp)


if not all_data:
    raise ValueError(
        "No valid data were read.\n"
        "Please check the scenario folder names and CSV files."
    )


df = pd.concat(all_data, ignore_index=True)

print("=" * 60)
print("Data reading completed")
print("=" * 60)
print(f"Total raw observations: {len(df)}")

before_nan = len(df)
df = df.dropna(
    subset=[ycol, "Guidance", "Smoke"]
).copy()
nan_removed = before_nan - len(df)
print(f"Missing values removed: {nan_removed}")

before_negative = len(df)
df = df[df[ycol] > 0].copy()
negative_removed = before_negative - len(df)
print(f"Non-positive LocalPosZ removed: {negative_removed}")

print("=" * 60)
print("LocalPosZ range after filtering")
print("=" * 60)
print(f"Minimum: {df[ycol].min():.3f}")
print(f"Maximum: {df[ycol].max():.3f}")
print(f"Mean: {df[ycol].mean():.3f}")
print(f"Median: {df[ycol].median():.3f}")

print("\nGuidance categories:")
print(df["Guidance"].value_counts())

print("\nSmoke categories:")
print(df["Smoke"].value_counts())

print("\nScenario sample sizes:")
print(df.groupby("Scenario")[ycol].count())


def remove_iqr_outliers(data, value_col, group_col="Scenario"):
    clean_data = []
    removed_total = 0

    for name, group in data.groupby(group_col):
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        group_clean = group[
            (group[value_col] >= lower) &
            (group[value_col] <= upper)
        ].copy()

        removed = len(group) - len(group_clean)
        removed_total += removed

        print(
            f"{name}: Q1={q1:.3f}, Q3={q3:.3f}, "
            f"IQR={iqr:.3f}, Lower={lower:.3f}, "
            f"Upper={upper:.3f}, Removed={removed}"
        )

        clean_data.append(group_clean)

    clean_df = pd.concat(clean_data, ignore_index=True)
    return clean_df, removed_total


print("\n" + "=" * 60)
print("IQR outlier removal")
print("=" * 60)

before_iqr = len(df)
df, iqr_removed = remove_iqr_outliers(df, ycol, "Scenario")
after_iqr = len(df)

print("=" * 60)
print(f"IQR outliers removed: {iqr_removed}")
print(f"Sample size before IQR: {before_iqr}")
print(f"Final sample size: {after_iqr}")
print("=" * 60)

print("\nFinal data:")
print(
    df.groupby(["Guidance", "Smoke"])[ycol]
    .agg(["count", "mean", "median", "std"])
)


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

        y = np.linspace(
            y_min - padding,
            y_max + padding,
            300
        )
        density = kde(y)

    if np.max(density) > 0:
        density = density / np.max(density)

    density = density * width

    x_center = np.full_like(y, center)

    if side == -1:
        x_outer = center - density
    else:
        x_outer = center + density

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


fig, ax = plt.subplots(
    figsize=(10, 6),
    facecolor="white"
)

ax.set_facecolor("white")

box_offset = 0.12
box_width = 0.16

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

    for smoke, offset in [
        ("LowSmoke", -box_offset),
        ("HighSmoke", box_offset)
    ]:
        values = subset[
            subset["Smoke"] == smoke
        ][ycol].dropna().values

        if len(values) == 0:
            continue

        ax.boxplot(
            values,
            positions=[i + offset],
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


ax.set_title(
    "Gaze Depth",
    fontsize=14,
    fontweight="bold",
    loc="left",
    pad=10
)

ax.set_xlabel("")
ax.set_xticks(range(len(guidance_order)))
ax.set_xticklabels(
    guidance_order,
    fontsize=10,
    rotation=20
)

ax.set_ylabel(
    ylabel,
    fontsize=11,
    labelpad=6
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

ax.set_xlim(
    -0.45,
    len(guidance_order) - 0.55
)


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

ax.legend(
    handles=legend_handles,
    title="Smoke condition",
    loc="upper left",
    bbox_to_anchor=(0.025, 0.98),
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    handlelength=1.4,
    handleheight=1.2,
    borderaxespad=0
)


plt.subplots_adjust(
    left=0.10,
    right=0.97,
    top=0.97,
    bottom=0.13
)


png_path = r"GazeDepth_SCI_BoxViolin.png"
pdf_path = r"GazeDepth_SCI_BoxViolin.pdf"

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

print("\n" + "=" * 60)
print("Plotting completed!")
print("=" * 60)
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
print("=" * 60)

plt.show()
