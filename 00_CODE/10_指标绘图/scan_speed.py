"""Trial-level scan speed rebuilt from RAW_DATA Subject/EyeGaze recordings.

The core matches extract_timestamp_metrics.py: normalize world gaze, undo
Unity zxy head rotation, smooth local unit directions over 3 samples within
valid runs, then sum adjacent angles / sum valid timestamp intervals (deg/s).
Use complete recordings from common.load_trials() (45 people / 270 trials), without spatial
zones, cells, trial-window clipping or spatial support thresholds. Position
alignment and motion validity guards remain identical to the reference.
All 270 canonical trials contribute to the plot and its mean labels; no IQR exclusions.
KDE draws only the violin outline and is not part of metric computation.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import gaussian_kde
from scipy.spatial.transform import Rotation
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import common
from common import load_trials
from INDICATORS_SIGNIFICANCE.TrialInput import prepare_trial_data

# Shared geometry for all six metric figures: tick digit counts must not
# move the axes or the y-axis title. Keep these values identical in all three scripts.
FIGURE_LAYOUT = dict(left=0.14, right=0.9667, bottom=0.1847, top=0.8751)
Y_LABEL_X = -0.095  # Axes coordinates; a fixed distance from the y-axis spine.
# Fixed crop in inches, preserving the previous export extent without content-based trimming.
EXPORT_BBOX = Bbox.from_bounds(0.35, 0.35, 12.8, 7.7)

# Fixed pre-change axes: display clipping does not exclude observations.
Y_LIMITS = (9.859060015463225, 51.06151061629711)
EXTRACTOR_DIR = PROJECT_ROOT / "Results_Spatial_Metric_Surfaces" / "_timestamp_input"
sys.path.insert(0, str(EXTRACTOR_DIR))
from extract_timestamp_metrics import (read_raw, align_positions, unit_vectors,
                                       smooth_vectors, interval_angles, CONFIG)

STYLE = {"font.family": "Times New Roman", "font.size": 29.9475,
         "axes.linewidth": 1.2, "axes.labelsize": 29.9475, "axes.titlesize": 38.115,
         "xtick.labelsize": 27.225, "ytick.labelsize": 27.225,
         "pdf.fonttype": 42, "ps.fonttype": 42}
guidance_order = ["StaticSign", "Arrow", "MiniMap", "TextBar", "Multimodal"]

# Plot helpers and palette copied from basic_indicator.py.
palette = {'StaticSign': {'LowSmoke': '#E6DAF5', 'HighSmoke': '#6B5C89'}, 'Arrow': {'LowSmoke': '#83E5FC', 'HighSmoke': '#398BD1'}, 'MiniMap': {'LowSmoke': '#BDE6EA', 'HighSmoke': '#357887'}, 'TextBar': {'LowSmoke': '#FDD8C7', 'HighSmoke': '#E99E74'}, 'Multimodal': {'LowSmoke': '#FFC8C8', 'HighSmoke': '#CF5149'}}

def normalize_guidance(x):
    """Normalize scenario guidance name to a unified label."""
    x = str(x).replace('-HighSmoke', '').replace('-LowSmoke', '')
    mapping = {'StaticSign': 'StaticSign', 'ArrowRibbon': 'Arrow', 'Arrow': 'Arrow', 'MiniMap': 'MiniMap', 'TextBar': 'TextBar', 'Full': 'Multimodal', 'Multimodal': 'Multimodal'}
    return mapping.get(x, x)

def draw_half_violin(ax, values, center, side, color, width=0.38, alpha=0.92, linewidth=1.1):
    """
    Draw a half violin on the left (side=-1)
    or right (side=+1) of center.
    """
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
        padding = data_range * 0.03 if data_range > 0 else 0.1
        y = np.linspace(y_min - padding, y_max + padding, 300)
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
    ax.fill(x_polygon, y_polygon, facecolor=color, edgecolor='#555555', linewidth=linewidth, alpha=alpha, zorder=1)

def draw_mean_label(ax, values, center, side, fontsize=26.62, x_offset=0.25):
    """
    Annotate the group mean with a two-decimal value label
    drawn as text on an opaque rounded background.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    mean_val = float(np.mean(values))
    if side == -1:
        x_pos = center - x_offset
    else:
        x_pos = center + x_offset
    ax.text(x_pos, mean_val, f'{mean_val:.2f}', ha='center', va='center', fontsize=fontsize, color='black', zorder=8, bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 1.2, 'alpha': 1.0})

def summarize_scan_speed(times, positions, head, gaze):
    """Reference eye-angle numerator and exposure, without spatial assignment."""
    times = np.asarray(times, dtype=float)
    positions, head, gaze = [np.asarray(x, dtype=float) for x in (positions, head, gaze)]
    if len(times) < 2:
        raise ValueError("At least two timestamps are required")
    if any(x.shape != (len(times), 3) for x in (positions, head, gaze)):
        raise ValueError("Position, head and gaze arrays must have shape (N, 3)")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    dt = np.r_[np.nan, np.diff(times)]
    position_valid = np.isfinite(positions).all(axis=1)
    motion_valid = position_valid & np.r_[False, position_valid[:-1]]
    step = np.r_[np.nan, np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    motion_valid &= ((dt > 0) & (dt <= CONFIG["max_gap_s"])
                     & (step <= CONFIG["max_position_step_m"]))
    head_valid = np.isfinite(head).all(axis=1)
    rotations = Rotation.from_euler(
        "zxy", np.where(np.isfinite(head), head, 0)[:, [2, 0, 1]], degrees=True)
    gaze, gaze_valid = unit_vectors(gaze)
    local = rotations.inv().apply(np.nan_to_num(gaze))
    local_valid = gaze_valid & head_valid
    local = smooth_vectors(local, local_valid, times, CONFIG["gaze_smooth_samples"])
    angles = interval_angles(local, np.isfinite(local).all(axis=1), dt)
    valid = motion_valid & np.isfinite(angles)
    angle = float(angles[valid].sum())
    duration = float(dt[valid].sum())
    return {"EyeGazeScanpathRate_deg_s": angle / duration if duration > 0 else np.nan,
            "EyeAngle_deg": angle, "EyeTime_s": duration,
            "ValidFrames": int(np.isfinite(local).all(axis=1).sum()),
            "AlignedFrames": len(times), "ValidIntervals": int(valid.sum()),
            "RecordingDuration_s": float(times[-1] - times[0])}


def collect_trials():
    manifest = load_trials()
    rows = []
    for trial in tqdm(manifest.itertuples(index=False), total=len(manifest),
                      desc="Scan speed trials"):
        identity = {"Scenario": trial.Scenario, "Subject": trial.ParticipantID,
                    "FileName": trial.FileName}
        subject_path = Path(trial.SubjectFile)
        eye_path = Path(trial.EyeGazeFile)
        pos, pos_audit = read_raw(subject_path, ["TimeStamp", "PosX", "PosY", "PosZ"])
        eye, eye_audit = read_raw(eye_path, ["TimeStamp", "GazeDirX", "GazeDirY", "GazeDirZ",
                                          "HeadRotX", "HeadRotY", "HeadRotZ"])
        if len(pos) < 2 or len(eye) < 2:
            raise ValueError(f"Insufficient observations: {eye_path}")
        # pandas 3 may parse timestamps at microsecond resolution; explicitly
        # convert both streams to nanoseconds before dividing intervals by 1e9.
        ens = eye.ParsedTime.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        pns = pos.ParsedTime.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        positions, method = align_positions(ens, pns,
                                            pos[["PosX", "PosY", "PosZ"]].to_numpy(float),
                                            max_gap_s=CONFIG["max_gap_s"])
        summary = summarize_scan_speed((ens - ens[0]) / 1e9, positions,
                                  eye[["HeadRotX", "HeadRotY", "HeadRotZ"]].to_numpy(float),
                                  eye[["GazeDirX", "GazeDirY", "GazeDirZ"]].to_numpy(float))
        rows.append({**identity, "ParticipantID": trial.ParticipantID,
                     "Guidance": trial.Guidance,
                     "Smoke": trial.Smoke,
                     **summary, "ExactFrames": int((method == 1).sum()),
                     "InterpolatedFrames": int((method == 2).sum()),
                     "UnmatchedFrames": int((method == 0).sum()),
                     "SubjectFile": str(subject_path), "EyeGazeFile": str(eye_path),
                     **{"Subject_" + k: v for k, v in pos_audit.items()},
                     **{"EyeGaze_" + k: v for k, v in eye_audit.items()}})
    result = pd.DataFrame(rows)
    keys = ["Scenario", "ParticipantID", "FileName"]
    expected = set(manifest[keys].itertuples(index=False, name=None))
    actual = set(result[keys].itertuples(index=False, name=None))
    if (len(result) != 270 or actual != expected
            or result.duplicated(["Scenario", "ParticipantID"]).any()):
        raise RuntimeError("Calculated trials do not match common.load_trials(): expected 45 participants / 270 trials")
    return result



def prepare_plot_trials(trials):
    """Validate all canonical trials and include every trial in the plot."""
    prepare_trial_data(trials, common, "EyeGazeScanpathRate_deg_s")
    # Accept existing full tables, but discard obsolete display-only IQR metadata.
    result = trials.drop(columns=[c for c in trials.columns if c.startswith("IQR")]).copy()
    result["PlotIncluded"] = True
    result["PlotExclusionReason"] = ""
    return result


def plot_scan_speed(trials, output):
    plt.rcParams.update(STYLE)
    df = prepare_plot_trials(trials)
    if df.empty:
        raise ValueError("No trials remain for plotting")
    df["Guidance"] = df.Scenario.map(normalize_guidance)
    df["Smoke"] = df.Scenario.str.rsplit("-", n=1).str[-1]
    fig, ax = plt.subplots(figsize=(13.5, 8.4), facecolor="white")
    for i, guidance in enumerate(guidance_order):
        for smoke, side, offset in [("LowSmoke", -1, -.12), ("HighSmoke", 1, .12)]:
            values = df.loc[df.Guidance.eq(guidance) & df.Smoke.eq(smoke), "EyeGazeScanpathRate_deg_s"].to_numpy()
            if len(values) == 0:
                raise ValueError(f"Missing plot condition: {guidance}/{smoke}")
            draw_half_violin(ax, values, i, side, palette[guidance][smoke])
            draw_mean_label(ax, values, i, side)
            ax.boxplot(values, positions=[i + offset], widths=.16,
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
                       whiskerprops={"color": "black", "linewidth": 1.0},
                       capprops={"color": "black", "linewidth": 1.0},
                       medianprops={"color": "black", "linewidth": 1.4}, zorder=4)
    ax.set_title("Scan Speed", fontsize=38.115, fontweight="bold", pad=18)
    ax.set_xlabel("Guidance", fontsize=29.9475, fontweight="bold", labelpad=13.5)
    ax.set_ylabel("Eye Gaze Scanpath Rate (deg/s)", fontsize=29.9475, fontweight="bold", labelpad=13.5)
    ax.set_xticks(range(len(guidance_order)), guidance_order, fontsize=27.225, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=27.225, width=1, length=5, pad=9)
    ax.tick_params(axis="y", labelsize=27.225, width=1, length=5)
    ax.yaxis.grid(True, linestyle="--", linewidth=.6, alpha=.35)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.1)
    ax.set_ylim(*Y_LIMITS)
    fig.subplots_adjust(**FIGURE_LAYOUT)
    ax.yaxis.set_label_coords(Y_LABEL_X, 0.5)
    # Retain the existing output names so the old, incorrect figure is replaced.
    fig.savefig(output / "scan_speed.png", dpi=600, bbox_inches=EXPORT_BBOX, facecolor="white")
    fig.savefig(output / "scan_speed.pdf", bbox_inches=EXPORT_BBOX, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--no-plot", action="store_true", help="Write numerical outputs only")
    args = parser.parse_args()
    trials = collect_trials()
    trials = prepare_plot_trials(trials)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(output / "AllSubjects_ScanSpeed.csv", index=False, encoding="utf-8-sig")
    plotted = trials.copy()
    plotted.to_csv(output / "ScanSpeed_PlotData.csv", index=False, encoding="utf-8-sig")
    plotted.groupby("Scenario").EyeGazeScanpathRate_deg_s.agg(["count", "mean", "std", "median", "min", "max"]).to_csv(
        output / "ScanSpeed_Summary.csv", encoding="utf-8-sig")
    audit = {"definition": "sum of head-local adjacent gaze angles / valid exposure seconds (deg/s)",
             "aggregation": "one rate per trial: sum(EyeAngle_deg) / sum(EyeTime_s), not mean instantaneous speed",
             "window": "complete recorded trial; no spatial-zone mask or cell-support filter",
             "reference_extractor": str(EXTRACTOR_DIR / "extract_timestamp_metrics.py"),
             "gaze_smooth_samples": CONFIG["gaze_smooth_samples"], "max_gap_s": CONFIG["max_gap_s"],
             "max_position_step_m": CONFIG["max_position_step_m"],
             "reference_sha256": hashlib.sha256((EXTRACTOR_DIR / "extract_timestamp_metrics.py").read_bytes()).hexdigest(),
             "iqr": "disabled; all 270 canonical trials included",
             "plot_ylim": list(Y_LIMITS),
             "axis_clipping": "display only; means and distributions use all trials",
             "trials": len(trials), "participants": trials.ParticipantID.nunique(),
             "plotted_trials": len(plotted), "excluded_from_plot": len(trials) - len(plotted),
             "cohort_source": str(PROJECT_ROOT / "common.py"),
             "cohort_loader": "common.load_trials()",
             "cohort": "fixed primary cohort: 45 participants / 270 trials",
             "style_reference": "INDICATORS_GRAPH/basic_indicator/basic_indicator.py",
             "violin_sides": {"left": "LowSmoke", "right": "HighSmoke"}}
    (output / "scan_speed_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot_scan_speed(trials, output)
    print(json.dumps({"trials": len(trials), "participants": trials.ParticipantID.nunique(),
                      "plotted_trials": len(plotted), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
