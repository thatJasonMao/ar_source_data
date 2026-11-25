import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============ SCI paper plot parameters ============
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 1.2,
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============ Read data ============
df = pd.read_csv("saccade_amplitude.csv")

# ============ Indicator ============
ycol = "MedianSaccadeAmplitude_deg"
ylabel = "Saccade Amplitude (deg)"

# Keep positive values only
df = df[df[ycol] > 0].copy()

# ============ Required columns ============
required_columns = [ycol, "Scenario", "Guidance", "Smoke"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")

# Drop missing values
before_nan = len(df)
df = df.dropna(subset=required_columns).copy()
print(f"Missing values removed: {before_nan - len(df)}")

# ============ IQR outlier removal (grouped by Scenario) ============
def remove_iqr_outliers(data, value_col, group_col="Scenario"):
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

        removed = len(group) - len(group_clean)
        removed_total += removed

        print(
            f"{name}: Q1={Q1:.3f}, Q3={Q3:.3f}, "
            f"IQR={IQR:.3f}, Lower={lower:.3f}, "
            f"Upper={upper:.3f}, Removed={removed}"
        )

        clean_data.append(group_clean)

    if len(clean_data) == 0:
        return data.copy(), 0

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

# ============ Order & palette ============
guidance_order = ["StaticSign", "Full", "ArrowRibbon", "MiniMap", "TextBar"]
smoke_order = ["LowSmoke", "HighSmoke"]

palette = {
    "StaticSign":  {"LowSmoke": "#D9E2E8", "HighSmoke": "#6F8797"},
    "Full":        {"LowSmoke": "#FFCACA", "HighSmoke": "#D15850"},
    "ArrowRibbon": {"LowSmoke": "#88E6FC", "HighSmoke": "#418FD2"},
    "MiniMap":     {"LowSmoke": "#BFE7EA", "HighSmoke": "#3D7D8C"},
    "TextBar":     {"LowSmoke": "#FDD9C9", "HighSmoke": "#E9A279"},
}

# ============ Figure ============
fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")

violin_offset = 0.15
violin_width = 0.34
box_width = 0.10

for i, guidance in enumerate(guidance_order):
    sub = df[df['Guidance'] == guidance]
    if sub.empty:
        continue

    for smoke, offset in [("LowSmoke", -violin_offset),
                          ("HighSmoke", violin_offset)]:
        vals = sub[sub['Smoke'] == smoke][ycol].dropna().values
        if len(vals) < 2:
            continue

        color = palette[guidance][smoke]

        # --- Violin (per-guidance color) ---
        vp = ax.violinplot(
            [vals],
            positions=[i + offset],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor("#555555")
            body.set_linewidth(1.1)
            body.set_alpha(0.92)
            body.set_zorder(1)

        # --- Box overlay ---
        ax.boxplot(
            vals,
            positions=[i + offset],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": "white",
                      "edgecolor": "black",
                      "linewidth": 1.1},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.4},
            zorder=4,
        )

# ============ Axis settings ============
ax.set_xlabel("Guidance", fontsize=11, labelpad=6)
ax.set_ylabel(ylabel, fontsize=11, labelpad=6)

ax.set_xticks(range(len(guidance_order)))
ax.set_xticklabels(guidance_order, fontsize=10, rotation=20)
ax.tick_params(axis="x", labelsize=10, width=1.0, length=4, pad=4)
ax.tick_params(axis="y", labelsize=10, width=1.0, length=4)

ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.1)
ax.spines["bottom"].set_linewidth(1.1)

ax.set_xlim(-0.5, len(guidance_order) - 0.5)

# ============ Legend ============
legend_handles = [
    Patch(facecolor="#BFE7EA", edgecolor="#555555",
          linewidth=1.0, label="Low smoke"),
    Patch(facecolor="#3D7D8C", edgecolor="#555555",
          linewidth=1.0, label="High smoke"),
]

ax.legend(
    handles=legend_handles,
    title="Smoke condition",
    loc="upper right",
    frameon=False,
    fontsize=10,
    title_fontsize=11,
    handlelength=1.4,
    handleheight=1.2,
    borderaxespad=0.3,
)

plt.tight_layout()

# ============ Save ============
plt.savefig("saccade_amplitude.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig("saccade_amplitude.tiff", dpi=600, bbox_inches="tight", facecolor="white")

plt.show()