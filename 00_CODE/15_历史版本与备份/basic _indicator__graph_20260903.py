import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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

# Read four metric files
distance = pd.read_csv("D:/AllSubjects_EvacuationDistance.csv")
rotation = pd.read_csv("D:/AllSubjects_EvacuationRotation.csv")
speed = pd.read_csv("D:/AllSubjects_EvacuationSpeed.csv")
time = pd.read_csv("D:/AllSubjects_EvacuationTime.csv")

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

# SCI color palette
palette = {
    "HighSmoke": "#5B7FA3",
    "LowSmoke": "#D69063"
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

# Plot each metric
for ax, (key, value), title in zip(axes, datasets.items(), titles):
    df = value[0].copy()
    ycol = value[1]
    ylabel = value[2]

    df["Smoke"] = df["Scenario"].apply(lambda x: "HighSmoke" if "HighSmoke" in str(x) else "LowSmoke")
    df["Guidance"] = (df["Scenario"].astype(str)
                      .str.replace("-HighSmoke", "", regex=False)
                      .str.replace("-LowSmoke", "", regex=False))

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

    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=8)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11, labelpad=6)
    ax.tick_params(axis='x', labelrotation=20, labelsize=10, width=1.0, length=4, pad=4)
    ax.tick_params(axis='y', labelsize=10, width=1.0, length=4)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    if ax.legend_:
        ax.legend_.remove()

# Create unified legend
legend_handles = [
    Patch(facecolor=palette["HighSmoke"], edgecolor="#444444", label="HighSmoke"),
    Patch(facecolor=palette["LowSmoke"], edgecolor="#444444", label="LowSmoke")
]
fig.legend(
    handles=legend_handles,
    title="Smoke",
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=11,
    title_fontsize=11,
    bbox_to_anchor=(0.5, 0.01)
)

# Adjust layout
plt.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.13, wspace=0.14, hspace=0.33)

# Save figures (optional, currently commented out in original)
# plt.savefig("...", dpi=600, bbox_inches='tight')
# plt.savefig("...pdf", bbox_inches='tight')

plt.show()