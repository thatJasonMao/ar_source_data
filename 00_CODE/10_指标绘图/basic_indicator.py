import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from pathlib import Path
import sys


# =========================================================
# Script directory
# Read canonical indicator tables from the project; save figures beside this script.
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import common
from common import INDICATORS_DIR
from INDICATORS_SIGNIFICANCE.TrialInput import prepare_trial_data

# Shared geometry for all six metric figures: tick digit counts must not
# move the axes or the y-axis title. Keep these values identical in all three scripts.
FIGURE_LAYOUT = dict(left=0.14, right=0.9667, bottom=0.1847, top=0.8751)
Y_LABEL_X = -0.095  # Axes coordinates; a fixed distance from the y-axis spine.
# Fixed crop in inches, preserving the previous export extent without content-based trimming.
EXPORT_BBOX = Bbox.from_bounds(0.35, 0.35, 12.8, 7.7)

# Preserve the pre-change axes; limits only clip display, never select trials.
Y_LIMITS = {'EvacuationDistance': (111.85628056454159, 408.7246471639206), 'TotalRotation': (334.92754699999693, 6588.182752999989), 'AverageSpeed': (0.7362169773439893, 1.5424300157425277), 'EvacuationTime': (85.6107175, 308.2472525)}


# =========================================================
# SCI paper global parameters
# Font sizes increased another 10%; preserve the 13.5 x 8.4 inch canvas.
# =========================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 29.9475,
    'axes.linewidth': 1.2,
    'axes.labelsize': 29.9475,
    'axes.titlesize': 38.115,
    'xtick.labelsize': 27.225,
    'ytick.labelsize': 27.225,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# =========================================================
# Helper functions
# =========================================================
def normalize_guidance(x):

    """Normalize scenario guidance name to a unified label."""

    x = str(x).replace(
        "-HighSmoke",
        ""
    ).replace(
        "-LowSmoke",
        ""
    )

    mapping = {

        "StaticSign": "StaticSign",

        "ArrowRibbon": "Arrow",

        "Arrow": "Arrow",

        "MiniMap": "MiniMap",

        "TextBar": "TextBar",

        "Full": "Multimodal",

        "Multimodal": "Multimodal",

    }

    return mapping.get(x, x)


def draw_half_violin(
    ax,
    values,
    center,
    side,
    color,
    width=0.38,
    alpha=0.92,
    linewidth=1.1
):

    """
    Draw a half violin on the left (side=-1)
    or right (side=+1) of center.
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

    if (
        len(values) < 3
        or np.ptp(values) == 0
    ):

        ymin = np.min(values)
        ymax = np.max(values)

        if ymin == ymax:

            ymin -= 0.1
            ymax += 0.1

        y = np.linspace(
            ymin,
            ymax,
            50
        )

        density = np.ones_like(y)

    else:

        kde = gaussian_kde(values)

        y_min = np.min(values)
        y_max = np.max(values)

        data_range = y_max - y_min

        padding = (
            data_range * 0.03
            if data_range > 0
            else 0.1
        )

        y = np.linspace(
            y_min - padding,
            y_max + padding,
            300
        )

        density = kde(y)

    if np.max(density) > 0:

        density = (
            density /
            np.max(density)
        )

    density = density * width

    x_center = np.full_like(
        y,
        center
    )

    if side == -1:

        x_outer = (
            center - density
        )

    else:

        x_outer = (
            center + density
        )

    x_polygon = np.concatenate(
        [
            x_center,
            x_outer[::-1]
        ]
    )

    y_polygon = np.concatenate(
        [
            y,
            y[::-1]
        ]
    )

    ax.fill(
        x_polygon,
        y_polygon,
        facecolor=color,
        edgecolor="#555555",
        linewidth=linewidth,
        alpha=alpha,
        zorder=1
    )


def draw_mean_label(
    ax,
    values,
    center,
    side,
    fontsize=26.62,
    x_offset=0.25
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

    # -----------------------------------------------------
    # Label position: centered on the half violin
    # -----------------------------------------------------
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
            "edgecolor": "black",
            "linewidth": 1.2,
            "alpha": 1.0
        }
    )


def space_mean_labels_horizontally(ax):
    """Separate enlarged boxes while preserving their exact mean y positions."""
    from scipy.optimize import Bounds, LinearConstraint, minimize

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [label for label in ax.texts if label.get_bbox_patch() is not None]
    boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in labels]
    anchors = np.array([ax.transData.transform(label.get_position()) for label in labels])
    widths = np.array([box.width for box in boxes])
    gap = 4.0 * fig.dpi / 72.0
    lower = ax.bbox.x0 + widths / 2 + gap
    upper = ax.bbox.x1 - widths / 2 - gap
    rows, separations = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if boxes[i].y0 >= boxes[j].y1 + gap or boxes[j].y0 >= boxes[i].y1 + gap:
                continue
            left, right = (i, j) if anchors[i, 0] <= anchors[j, 0] else (j, i)
            row = np.zeros(len(labels))
            row[left], row[right] = -1, 1
            rows.append(row)
            separations.append((widths[i] + widths[j]) / 2 + gap)
    constraints = [LinearConstraint(np.array(rows), separations, np.inf)] if rows else []
    fit = minimize(lambda xs: np.sum((xs - anchors[:, 0]) ** 2),
                   np.clip(anchors[:, 0], lower, upper),
                   jac=lambda xs: 2 * (xs - anchors[:, 0]),
                   method="SLSQP", bounds=Bounds(lower, upper),
                   constraints=constraints, options={"ftol": 1e-7, "maxiter": 200})
    if not fit.success:
        raise RuntimeError(f"Mean-label spacing failed: {fit.message}")
    for label, x, anchor in zip(labels, fit.x, anchors):
        label.set_x(ax.transData.inverted().transform((x, anchor[1]))[0])


# =========================================================
# Read data
# =========================================================

distance = pd.read_csv(
    Path(INDICATORS_DIR) / "Results_EvacuationDistance" / "AllSubjects_EvacuationDistance.csv"
)

rotation = pd.read_csv(
    Path(INDICATORS_DIR) / "Results_EvacuationRotation" / "AllSubjects_EvacuationRotation.csv"
)

speed = pd.read_csv(
    Path(INDICATORS_DIR) / "Results_EvacuationSpeed" / "AllSubjects_EvacuationSpeed.csv"
)

time = pd.read_csv(
    Path(INDICATORS_DIR) / "Results_EvacuationTime" / "AllSubjects_EvacuationTime.csv"
)


datasets = {

    "EvacuationDistance": [
        distance,
        "EvacuationDistance",
        "Evacuation Distance (m)"
    ],

    "TotalRotation": [
        rotation,
        "TotalRotation_deg",
        "Total Rotation (deg)"
    ],

    "AverageSpeed": [
        speed,
        "AverageSpeed_m_s",
        "Average Speed (m/s)"
    ],

    "EvacuationTime": [
        time,
        "EvacuationTime_s",
        "Evacuation Time (s)"
    ]

}


# =========================================================
# Figure titles
# Removed (a), (b), (c), (d)
# =========================================================

titles = {

    "EvacuationDistance":
        "Evacuation Distance",

    "TotalRotation":
        "Total Rotation",

    "AverageSpeed":
        "Average Speed",

    "EvacuationTime":
        "Evacuation Time",

}


# =========================================================
# Validate the complete canonical cohort; no trial-level outlier removal.
# =========================================================

for key, value in datasets.items():
    _, validated, _ = prepare_trial_data(value[0], common, value[1])
    datasets[key][0] = validated
    print(f"{key}: all {len(validated)} canonical trials included; no IQR exclusion")


# =========================================================
# Palette
# =========================================================

palette = {

    "StaticSign": {

        "LowSmoke": "#E6DAF5",

        "HighSmoke": "#6B5C89"

    },

    "Arrow": {

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

    "Multimodal": {

        "LowSmoke": "#FFC8C8",

        "HighSmoke": "#CF5149"

    }

}


guidance_order = [

    "StaticSign",

    "Arrow",

    "MiniMap",

    "TextBar",

    "Multimodal"

]


# =========================================================
# Output files
# =========================================================

out_files = {

    "EvacuationDistance":
        SCRIPT_DIR / "EvacuationDistance.png",

    "TotalRotation":
        SCRIPT_DIR / "TotalRotation.png",

    "AverageSpeed":
        SCRIPT_DIR / "AverageSpeed.png",

    "EvacuationTime":
        SCRIPT_DIR / "EvacuationTime.png",

}


# =========================================================
# Draw four separate figures
# =========================================================

for key, (
    df_full,
    ycol,
    ylabel
) in datasets.items():

    df = df_full.copy()

    # Long rotation mean labels need extra horizontal room.
    mean_label_offset = 0.32 if key == "TotalRotation" else 0.25

    # =====================================================
    # Derive smoke condition and guidance
    # =====================================================

    df["Smoke"] = df["Scenario"].apply(
        lambda x:
        "HighSmoke"
        if "HighSmoke" in str(x)
        else "LowSmoke"
    )

    df["Guidance"] = df["Scenario"].apply(
        normalize_guidance
    )

    # =====================================================
    # Figure
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(13.5, 8.4),
        facecolor="white"
    )

    # =====================================================
    # Draw each guidance condition
    # =====================================================

    for i, guidance in enumerate(
        guidance_order
    ):

        subset = df[
            df["Guidance"] == guidance
        ].copy()

        if subset.empty:
            continue

        # -------------------------------------------------
        # Low smoke
        # -------------------------------------------------

        low_values = subset[
            subset["Smoke"] == "LowSmoke"
        ][ycol].dropna().values

        # -------------------------------------------------
        # High smoke
        # -------------------------------------------------

        high_values = subset[
            subset["Smoke"] == "HighSmoke"
        ][ycol].dropna().values

        # =================================================
        # Low smoke: left half violin
        # =================================================

        if len(low_values) > 0:

            draw_half_violin(
                ax,
                low_values,
                center=i,
                side=-1,
                color=palette[
                    guidance
                ]["LowSmoke"]
            )

            draw_mean_label(
                ax,
                low_values,
                center=i,
                side=-1,
                x_offset=mean_label_offset
            )

        # =================================================
        # High smoke: right half violin
        # =================================================

        if len(high_values) > 0:

            draw_half_violin(
                ax,
                high_values,
                center=i,
                side=+1,
                color=palette[
                    guidance
                ]["HighSmoke"]
            )

            draw_mean_label(
                ax,
                high_values,
                center=i,
                side=+1,
                x_offset=mean_label_offset
            )

        # =================================================
        # Boxplots
        # =================================================

        for smoke, offset in [

            ("LowSmoke", -0.12),

            ("HighSmoke", 0.12)

        ]:

            values = subset[
                subset["Smoke"] == smoke
            ][ycol].dropna().values

            if len(values) == 0:
                continue

            ax.boxplot(

                values,

                positions=[
                    i + offset
                ],

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

    # =====================================================
    # Figure formatting
    # =====================================================

    ax.set_facecolor("white")

    # -----------------------------------------------------
    # Title
    # Removed (a), (b), (c), (d)
    # -----------------------------------------------------

    ax.set_title(
        titles[key],
        fontsize=38.115,
        fontweight="bold",
        loc="center",
        pad=18
    )

    ax.set_xlabel(
    "Guidance",
    fontsize=29.9475,
    fontweight="bold",
    labelpad=13.5,
    loc="center"
)

    # =====================================================
    # X axis
    # Horizontal labels — no rotation
    # =====================================================

    ax.set_xticks(
        range(len(guidance_order))
    )

    ax.set_xticklabels(
        guidance_order,
        fontsize=27.225,
        rotation=0,
        ha="center"
    )

    # =====================================================
    # Y axis
    # =====================================================

    ax.set_ylabel(
        ylabel,
        fontsize=29.9475,
        fontweight="bold",
        labelpad=13.5
    )

    # =====================================================
    # Tick parameters
    # =====================================================

    ax.tick_params(
        axis="x",
        labelsize=27.225,
        width=1.0,
        length=5,
        pad=9
    )

    ax.tick_params(
        axis="y",
        labelsize=27.225,
        width=1.0,
        length=5
    )

    # =====================================================
    # Grid
    # =====================================================

    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35
    )

    ax.xaxis.grid(False)

    ax.set_axisbelow(True)

    # =====================================================
    # Spines
    # =====================================================

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_linewidth(1.1)

    ax.spines[
        "bottom"
    ].set_linewidth(1.1)

    # =====================================================
    # Layout
    # =====================================================

    ax.set_ylim(*Y_LIMITS[key])
    fig.subplots_adjust(**FIGURE_LAYOUT)
    ax.yaxis.set_label_coords(Y_LABEL_X, 0.5)
    fig.set_dpi(600)
    space_mean_labels_horizontally(ax)

    # =====================================================
    # Save
    # =====================================================

    fig.savefig(

        out_files[key],

        dpi=600,

        bbox_inches=EXPORT_BBOX,

        facecolor="white"

    )

    plt.close(fig)

    print(
        f"Saved: {out_files[key]}"
    )


print(
    "\nAll four figures have been generated successfully."
)
