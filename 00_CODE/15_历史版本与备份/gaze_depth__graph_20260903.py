import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import glob

# SCI global parameters
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

# path
root_dir = "D:/depth"
ycol = "LocalPosZ"
xlabel = "Guidance"
ylabel = "Gaze Depth (LocalPosZ)"

# Guidance order
guidance_order = [
    "StaticSign",
    "Full",
    "ArrowRibbon",
    "MiniMap",
    "TextBar"
]

# Smoke order
smoke_order = [
    "HighSmoke",
    "LowSmoke"
]

# colors
palette = {
    "HighSmoke": "#5B7FA3",
    "LowSmoke": "#D69063"
}

if not os.path.exists(root_dir):
    raise FileNotFoundError(
        f"Folder not found: {root_dir}\n"
        f"Please check if D:/depth exists."
    )

# D:/depth/
#
#     StaticSign-HighSmoke/
#         Participant01/
#             RelativePosInfo.csv
#
#     StaticSign-LowSmoke/
#         Participant01/
#             RelativePosInfo.csv
#
#     Full-HighSmoke/
#         Participant01/
#             RelativePosInfo.csv
#
#     Full-LowSmoke/
#         Participant01/
#             RelativePosInfo.csv
#
#     ArrowRibbon-HighSmoke/
#         ...

def parse_scenario_folder(folder_name):
    guidance = None
    smoke = None

    # Identify Guidance
    for g in guidance_order:
        if folder_name.startswith(g):
            guidance = g
            break

    # Identify Smoke
    if "HighSmoke" in folder_name:
        smoke = "HighSmoke"
    elif "LowSmoke" in folder_name:
        smoke = "LowSmoke"

    return guidance, smoke

# Search all RelativePosInfo.csv files
csv_files = glob.glob(
    os.path.join(
        root_dir,
        "*",
        "*",
        "RelativePosInfo.csv"
    )
)

print("=" * 60)
print(
    f"Number of RelativePosInfo.csv files found: "
    f"{len(csv_files)}"
)
print("=" * 60)

# Check if any CSV found
if len(csv_files) == 0:
    raise FileNotFoundError(
        "No RelativePosInfo.csv found!\n\n"
        "Expected directory structure:\n"
        "D:/depth/Scenario-Smoke/Participant/RelativePosInfo.csv\n\n"
        "Please check the folder structure."
    )

# Read all participant data
all_data = []

for file in csv_files:
    # Get participant directory
    participant_dir = os.path.dirname(file)

    # Get scenario directory
    scenario_dir = os.path.dirname(
        participant_dir
    )

    # Get scenario folder name
    scenario_folder = os.path.basename(
        scenario_dir
    )

    # Get participant name
    participant = os.path.basename(
        participant_dir
    )

    # Parse Guidance and Smoke
    guidance, smoke = parse_scenario_folder(
        scenario_folder
    )

    # If scenario cannot be parsed
    if guidance is None or smoke is None:
        print(
            f"⚠ Cannot parse scenario folder: "
            f"{scenario_folder}"
        )
        print(
            f"   File: {file}"
        )
        continue

    # Skip TEST participants
    if "TEST" in participant.upper():
        print(
            f"Skipping TEST participant: "
            f"{participant}"
        )
        continue

    # Read CSV
    try:
        temp = pd.read_csv(
            file
        )
    except Exception as e:
        print(
            f"⚠ CSV read failed: "
            f"{file}"
        )
        print(
            f"Error: {e}"
        )
        continue

    # Check LocalPosZ
    if ycol not in temp.columns:
        print(
            f"⚠ Column {ycol} not found in: "
        )
        print(
            f"   {file}"
        )
        continue

    # Keep only LocalPosZ
    temp = temp[
        [ycol]
    ].copy()

    # Convert LocalPosZ to numeric, non-numeric become NaN
    temp[ycol] = pd.to_numeric(
        temp[ycol],
        errors="coerce"
    )

    # Add experimental info
    temp["Guidance"] = guidance
    temp["Smoke"] = smoke
    temp["ParticipantID"] = participant
    temp["Scenario"] = scenario_folder
    temp["File"] = file

    # Save
    all_data.append(
        temp
    )

# Check if any valid data read
if len(all_data) == 0:
    raise ValueError(
        "No valid data read.\n"
        "Please check scenario folder names and CSV files."
    )

# Concatenate all data
df = pd.concat(
    all_data,
    ignore_index=True
)

print("\n")
print("=" * 60)
print("Data reading completed")
print("=" * 60)
print(
    f"Total raw observations: "
    f"{len(df)}"
)

# Remove missing values
before_nan = len(df)

df = df.dropna(
    subset=[
        ycol,
        "Guidance",
        "Smoke"
    ]
).copy()

nan_removed = (
    before_nan -
    len(df)
)

print(
    f"Missing values removed: "
    f"{nan_removed}"
)

# Remove negative LocalPosZ (keep >= 0)
before_negative = len(df)

df = df[
    df[ycol] > 0
].copy()

negative_removed = (
    before_negative -
    len(df)
)

print(
    f"Negative LocalPosZ removed: "
    f"{negative_removed}"
)

# Check LocalPosZ range
print("\n")
print("=" * 60)
print(
    "LocalPosZ range after removing negatives:"
)
print(
    f"Minimum: "
    f"{df[ycol].min():.3f}"
)
print(
    f"Maximum: "
    f"{df[ycol].max():.3f}"
)
print(
    f"Mean: "
    f"{df[ycol].mean():.3f}"
)
print(
    f"Median: "
    f"{df[ycol].median():.3f}"
)
print("=" * 60)

# Check Guidance categories
print("\nGuidance categories:")
print(
    df["Guidance"].value_counts()
)

# Check Smoke categories
print("\nSmoke categories:")
print(
    df["Smoke"].value_counts()
)

# Check sample sizes per Scenario
print("\nScenario sample sizes:")
print(
    df.groupby(
        "Scenario"
    )[ycol].count()
)

# IQR outlier removal function (per Scenario)
def remove_iqr_outliers(
        data,
        value_col,
        group_col="Scenario"):
    clean_data = []
    removed_total = 0

    # Group by Scenario
    for name, group in data.groupby(
        group_col
    ):
        # Q1
        Q1 = group[
            value_col
        ].quantile(0.25)

        # Q3
        Q3 = group[
            value_col
        ].quantile(0.75)

        # IQR
        IQR = Q3 - Q1

        # Lower bound
        lower = (
            Q1 -
            1.5 * IQR
        )

        # Upper bound
        upper = (
            Q3 +
            1.5 * IQR
        )

        # Keep non-outliers
        group_clean = group[
            (group[value_col] >= lower)
            &
            (group[value_col] <= upper)
        ].copy()

        # Count removed
        removed = (
            len(group) -
            len(group_clean)
        )

        removed_total += removed

        # Print IQR range for each Scenario
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

    # Concatenate
    clean_df = pd.concat(
        clean_data,
        ignore_index=True
    )

    return clean_df, removed_total

# Perform IQR cleaning
print("\n")
print("=" * 60)
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
print(
    f"IQR outliers removed: "
    f"{iqr_removed}"
)
print(
    f"Sample size before IQR: "
    f"{before_iqr}"
)
print(
    f"Final sample size: "
    f"{after_iqr}"
)
print("=" * 60)

# Check final data summary
print("\nFinal data:")
print(
    df.groupby(
        ["Guidance", "Smoke"]
    )[ycol]
    .agg(
        ["count", "mean", "median", "std"]
    )
)

# Create figure
fig, ax = plt.subplots(
    figsize=(10, 6),
    facecolor="white"
)

# Draw violin plot
sns.violinplot(
    data=df,
    x="Guidance",
    y=ycol,
    hue="Smoke",
    order=guidance_order,
    hue_order=smoke_order,
    split=True,
    inner=None,
    linewidth=1.1,
    palette=palette,
    saturation=0.9,
    ax=ax
)

# Overlay box plot (consistent with earlier EyeGaze plot)
sns.boxplot(
    data=df,
    x="Guidance",
    y=ycol,
    hue="Smoke",
    order=guidance_order,
    hue_order=smoke_order,
    width=0.30,
    linewidth=1.3,
    showcaps=True,
    boxprops={
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    whiskerprops={
        'linewidth': 1.2,
        'color': 'black'
    },
    capprops={
        'linewidth': 1.2,
        'color': 'black'
    },
    medianprops={
        'color': 'black',
        'linewidth': 1.5
    },
    showfliers=False,
    ax=ax
)

# Set white background
ax.set_facecolor(
    "white"
)

# Title
ax.set_title(
    "Gaze Depth",
    fontsize=14,
    fontweight="bold",
    loc="left",
    pad=10
)

# X axis
ax.set_xlabel("")

ax.tick_params(
    axis="x",
    labelrotation=20,
    labelsize=10,
    width=1.0,
    length=4,
    pad=4
)

# Y axis
ax.set_ylabel(
    ylabel,
    fontsize=11,
    labelpad=6
)

ax.tick_params(
    axis="y",
    labelsize=10,
    width=1.0,
    length=4
)

# Horizontal grid lines
ax.yaxis.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

ax.xaxis.grid(
    False
)

ax.set_axisbelow(
    True
)

# Remove top and right spines
ax.spines["top"].set_visible(
    False
)
ax.spines["right"].set_visible(
    False
)

ax.spines["left"].set_linewidth(
    1.1
)
ax.spines["bottom"].set_linewidth(
    1.1
)

# Remove seaborn default legend
if ax.legend_:
    ax.legend_.remove()

# Create unified legend
legend_handles = [
    Patch(
        facecolor=palette["HighSmoke"],
        edgecolor="#444444",
        label="HighSmoke"
    ),
    Patch(
        facecolor=palette["LowSmoke"],
        edgecolor="#444444",
        label="LowSmoke"
    )
]

ax.legend(
    handles=legend_handles,
    title="Smoke",
    loc="upper right",
    frameon=False,
    fontsize=10,
    title_fontsize=10
)

# Adjust layout
plt.subplots_adjust(
    left=0.10,
    right=0.97,
    top=0.90,
    bottom=0.16
)

# Save PNG
png_path = (
    "D:/GazeDepth_SCI_BoxViolin.png"
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

# Save PDF vector
pdf_path = (
    "D:/GazeDepth_SCI_BoxViolin.pdf"
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white"
)

# Show plot
plt.show()

# Output save locations
print("\n")
print("=" * 60)
print("Plotting completed!")
print("=" * 60)
print(
    f"PNG: {png_path}"
)
print(
    f"PDF: {pdf_path}"
)
print("=" * 60)