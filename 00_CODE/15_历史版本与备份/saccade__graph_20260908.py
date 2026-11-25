import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# SCI paper plot parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# Read data
df = pd.read_csv("eye_jump_2.csv")

# Remove outliers with SaccadeRate_per_min < 25
df = df[df['SaccadeRate_per_min'] >= 25]

# Set up canvas
plt.figure(figsize=(8, 5))

# SCI color palette
palette = {
    'LowSmoke': '#4C72B0',
    'HighSmoke': '#DD8452'
}

# Violin plot
sns.violinplot(
    data=df,
    x='Guidance',
    y='SaccadeRate_per_min',
    hue='Smoke',
    split=True,
    inner=None,
    linewidth=1.2,
    palette=palette
)

# Overlay box plot
sns.boxplot(
    data=df,
    x='Guidance',
    y='SaccadeRate_per_min',
    hue='Smoke',
    width=0.30,
    linewidth=1.3,
    showcaps=True,
    boxprops={
        'facecolor': 'white',
        'edgecolor': 'black'
    },
    whiskerprops={
        'linewidth': 1.2
    },
    capprops={
        'linewidth': 1.2
    },
    medianprops={
        'color': 'black',
        'linewidth': 1.5
    },
    showfliers=False
)

# Axis settings
plt.xlabel(
    'Guidance',
    fontsize=13,
    fontweight='bold'
)
plt.ylabel(
    'Saccade Rate (per min)',
    fontsize=13,
    fontweight='bold'
)

# X-axis ticks
plt.xticks(
    fontsize=11,
    rotation=20
)

# Y-axis ticks
plt.yticks(
    fontsize=11
)

# Legend optimization
handles, labels = plt.gca().get_legend_handles_labels()
n_smoke = df['Smoke'].nunique()
plt.legend(
    handles[:n_smoke],
    labels[:n_smoke],
    title='Smoke',
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc='upper right'
)

# Remove extra spines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save high-resolution figures
plt.tight_layout()

# PNG
plt.savefig(
    "eye_jump_SCI.png",
    dpi=600,
    bbox_inches='tight'
)

# TIFF (commonly used in papers)
plt.savefig(
    "eye_jump_SCI.tiff",
    dpi=600,
    bbox_inches='tight'
)

plt.show()