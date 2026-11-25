"""Plot route efficiency for all guidance-by-smoke experimental groups.

The figure follows the visual language of
``graph_0911/basic_indicator/basic_indicator.py`` while placing all ten
experimental conditions on one axis.  Route efficiency is recomputed from
the raw trajectories and the formal spatial-validation cohort as

    NetworkShortestStartToEnd_m / NetworkObservedPathDistance_m

Both distances use the first and last matched trajectory nodes. The denominator
sums network shortest distances between temporally successive mapped nodes,
preserving backtracking. No substantive upper clipping is applied.
Point face colours match their half violins.  Green outlines mark
correct exits and red outlines mark wrong exits.

Only rows with ``AnalysisIncluded == True`` are plotted.  No IQR or other
outlier deletion is applied because the individual trial points and exit
outcomes are part of the intended result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from Route_Efficiency_Network import (
    LONG_GAP_S,
    MAP_THRESHOLD_M,
    SAMPLE_INTERVAL_S,
    add_network_route_metrics,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "Results_Guidance_Spatial_Causal_Analysis"
    / "data"
    / "analysis_trials.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "Results_Route_Efficiency_Raincloud"
DEFAULT_SPATIAL_DIR = ROOT / "SpatialSceneExport"
DEFAULT_RAW_DIR = ROOT / "RAW_DATA"

FIGURE_BASENAME = "Route_Efficiency_Raincloud_All_Conditions"
PLOT_DATA_NAME = "Route_Efficiency_Plot_Data.csv"
SUMMARY_NAME = "Route_Efficiency_Group_Summary.csv"
WRONG_EXIT_NAME = "Route_Efficiency_Wrong_Exit_Subjects.csv"
METADATA_NAME = "Route_Efficiency_Figure_Metadata.json"
NETWORK_TRANSITIONS_NAME = "Route_Efficiency_Network_Transitions.csv"

CORRECT_EDGE = "#00AD5F"
WRONG_EDGE = "#D62728"

# Colours copied from graph_0911/basic_indicator/basic_indicator.py.
PALETTE = {
    "StaticSign-LowSmoke": "#E6DAF5",
    "StaticSign-HighSmoke": "#6B5C89",
    "ArrowRibbon-LowSmoke": "#83E5FC",
    "ArrowRibbon-HighSmoke": "#398BD1",
    "MiniMap-LowSmoke": "#BDE6EA",
    "MiniMap-HighSmoke": "#357887",
    "TextBar-LowSmoke": "#FDD8C7",
    "TextBar-HighSmoke": "#E99E74",
    "Full-LowSmoke": "#FFC8C8",
    "Full-HighSmoke": "#CF5149",
}

SCENARIO_ORDER = list(PALETTE)
DISPLAY_LABELS = {
    "StaticSign-LowSmoke": "StaticSign\nLowSmoke",
    "StaticSign-HighSmoke": "StaticSign\nHighSmoke",
    "ArrowRibbon-LowSmoke": "Arrow\nLowSmoke",
    "ArrowRibbon-HighSmoke": "Arrow\nHighSmoke",
    "MiniMap-LowSmoke": "MiniMap\nLowSmoke",
    "MiniMap-HighSmoke": "MiniMap\nHighSmoke",
    "TextBar-LowSmoke": "TextBar\nLowSmoke",
    "TextBar-HighSmoke": "TextBar\nHighSmoke",
    "Full-LowSmoke": "Multimodal\nLowSmoke",
    "Full-HighSmoke": "Multimodal\nHighSmoke",
}

# Low/High smoke conditions stay close, with a larger gap between guidance
# families.  This remains one axis and one panel, as requested.
POSITIONS = np.array(
    [0.00, 0.82, 1.95, 2.77, 3.90, 4.72, 5.85, 6.67, 7.80, 8.62],
    dtype=float,
)

REQUIRED_COLUMNS = {
    "Scenario",
    "ParticipantID",
    "FileName",
    "ShortestStartToTarget_m",
    "ObservedPathDistance_m",
    "RouteEfficiency",
    "ChosenExit",
    "TargetExit",
    "ExitCorrect",
    "AnalysisIncluded",
    "AOICohortExcluded",
    "SpawnNode",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spatial-dir", type=Path, default=DEFAULT_SPATIAL_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_S)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to strict booleans."""
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = sorted(set(normalized.dropna()) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown}")
    return normalized.map(mapping).astype(bool)


def load_and_validate(
    input_path: Path,
    spatial_dir: Path = DEFAULT_SPATIAL_DIR,
    raw_dir: Path = DEFAULT_RAW_DIR,
    sample_interval_s: float = SAMPLE_INTERVAL_S,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {missing}")

    data["AnalysisIncluded"] = as_bool(data["AnalysisIncluded"])
    data["AOICohortExcluded"] = as_bool(data["AOICohortExcluded"])
    data = data.loc[
        data["AnalysisIncluded"] & ~data["AOICohortExcluded"]
    ].copy()
    if len(data) != 270:
        raise ValueError(
            "The formal primary-analysis input must contain exactly 270 "
            f"included trials; found {len(data)}"
        )
    data["ExitCorrect"] = pd.to_numeric(data["ExitCorrect"], errors="raise").astype(int)

    for column in [
        "ShortestStartToTarget_m",
        "ObservedPathDistance_m",
        "RouteEfficiency",
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    invalid_distance = (
        ~np.isfinite(data["ShortestStartToTarget_m"])
        | ~np.isfinite(data["ObservedPathDistance_m"])
        | data["ShortestStartToTarget_m"].le(0)
        | data["ObservedPathDistance_m"].le(1.0)
    )
    if invalid_distance.any():
        bad = data.loc[
            invalid_distance,
            [
                "Scenario",
                "ParticipantID",
                "ShortestStartToTarget_m",
                "ObservedPathDistance_m",
            ],
        ]
        raise ValueError(f"Invalid distances in included trials:\n{bad.to_string(index=False)}")

    # Validate the original target-based field before changing the metric.
    target_efficiency = (
        data["ShortestStartToTarget_m"] / data["ObservedPathDistance_m"]
    ).clip(lower=0.0, upper=1.0)

    if not np.allclose(
        target_efficiency,
        data["RouteEfficiency"],
        rtol=0.0,
        atol=1e-10,
    ):
        max_error = float(
            np.max(
                np.abs(
                    target_efficiency - data["RouteEfficiency"]
                )
            )
        )
        raise ValueError(
            "Recomputed target-based efficiency does not match the formal field; "
            f"maximum absolute error={max_error:.3g}"
        )

    data, transitions = add_network_route_metrics(
        data, spatial_dir, raw_dir, interval_s=sample_interval_s
    )
    data["RouteEfficiencyToTarget"] = data["RouteEfficiency"]
    data["RouteEfficiencyToChosenExitRaw"] = (
        data["ShortestStartToChosenExit_m"] / data["ObservedPathDistance_m"]
    )
    data["RouteEfficiencyCalculated"] = data["NetworkRouteEfficiency"]
    data["RouteEfficiency"] = data["RouteEfficiencyCalculated"]

    calculated_correct = (
        data["ChosenExit"].astype(str).str.strip()
        == data["TargetExit"].astype(str).str.strip()
    ).astype(int)
    if not calculated_correct.equals(data["ExitCorrect"]):
        raise ValueError("ExitCorrect is inconsistent with ChosenExit == TargetExit")

    unexpected = sorted(set(data["Scenario"]) - set(SCENARIO_ORDER))
    absent = sorted(set(SCENARIO_ORDER) - set(data["Scenario"]))
    if unexpected or absent:
        raise ValueError(
            f"Scenario mismatch. Unexpected={unexpected}; missing={absent}"
        )

    duplicates = data.duplicated(["Scenario", "ParticipantID"], keep=False)
    if duplicates.any():
        bad = data.loc[duplicates, ["Scenario", "ParticipantID", "FileName"]]
        raise ValueError(
            "Expected one trial per participant and condition, but found duplicates:\n"
            + bad.to_string(index=False)
        )

    if not data["ExitCorrect"].isin([0, 1]).all():
        raise ValueError("ExitCorrect must contain only 0/1")
    if not data["RouteEfficiencyCalculated"].between(0, 1).all():
        raise ValueError("Calculated route efficiency must be within [0, 1]")

    data["ExitStatus"] = np.where(
        data["ExitCorrect"].eq(1), "Correct exit", "Wrong exit"
    )
    data["Scenario"] = pd.Categorical(
        data["Scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    return data.sort_values(["Scenario", "ParticipantID"]).reset_index(drop=True), transitions


def draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    center: float,
    color: str,
    width: float = 0.34,
) -> None:
    """Draw a bounded left half violin without extending outside [0, 1]."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    if values.size < 3 or np.ptp(values) < 1e-12:
        # A zero-variance group is shown as a narrow triangular density stub,
        # not as an invented broad distribution.
        value = float(values[0])
        epsilon = 0.008
        y = np.array(
            [max(0.0, value - epsilon), value, min(1.0, value + epsilon)]
        )
        density = np.array([0.0, width, 0.0])
    else:
        kde = gaussian_kde(values)
        span = float(np.ptp(values))
        padding = max(0.008, span * 0.04)
        y = np.linspace(
            max(0.0, float(values.min()) - padding),
            min(1.0, float(values.max()) + padding),
            300,
        )
        density = kde(y)
        density = density / density.max() * width

    x_outer = center - density
    ax.fill(
        np.concatenate([np.full_like(y, center), x_outer[::-1]]),
        np.concatenate([y, y[::-1]]),
        facecolor=color,
        edgecolor="#555555",
        linewidth=1.05,
        alpha=0.92,
        zorder=1,
    )


def draw_mean_label(
    ax: plt.Axes,
    values: np.ndarray,
    center: float,
    x_offset: float = 0.245,
) -> None:
    """Draw the group mean in a white text box inside the half violin."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    mean_value = float(np.mean(values))
    ax.text(
        center - x_offset,
        mean_value,
        f"{mean_value:.2f}",
        ha="center",
        va="center",
        fontsize=18.0,
        color="black",
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.26",
            "facecolor": "white",
            "edgecolor": "#555555",
            "linewidth": 1.1,
            "alpha": 1.0,
        },
    )


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in SCENARIO_ORDER:
        part = data.loc[data["Scenario"].astype(str).eq(scenario)].copy()
        values = part["RouteEfficiencyCalculated"]
        correct_n = int(part["ExitCorrect"].sum())
        n = int(len(part))
        rows.append(
            {
                "Scenario": scenario,
                "DisplayScenario": DISPLAY_LABELS[scenario].replace("\n", "-"),
                "N": n,
                "CorrectExitN": correct_n,
                "WrongExitN": n - correct_n,
                "CorrectExitRate": correct_n / n,
                "RouteEfficiencyMean": float(values.mean()),
                "RouteEfficiencySD": float(values.std(ddof=1)),
                "RouteEfficiencyMedian": float(values.median()),
                "RouteEfficiencyQ1": float(values.quantile(0.25)),
                "RouteEfficiencyQ3": float(values.quantile(0.75)),
                "RouteEfficiencyMin": float(values.min()),
                "RouteEfficiencyMax": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def draw_figure(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    seed: int,
) -> tuple[plt.Figure, pd.DataFrame]:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 18.0,
            "axes.linewidth": 1.2,
            "axes.labelsize": 25.0,
            "axes.titlesize": 29.0,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 20.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(17.5, 9.4), facecolor="white")
    rng = np.random.default_rng(seed)
    plotted_parts: list[pd.DataFrame] = []

    for scenario, center in zip(SCENARIO_ORDER, POSITIONS):
        part = data.loc[data["Scenario"].astype(str).eq(scenario)].copy()
        values = part["RouteEfficiencyCalculated"].to_numpy(float)
        color = PALETTE[scenario]

        draw_half_violin(ax, values, center=center, color=color)
        draw_mean_label(ax, values, center=center)

        ax.boxplot(
            values,
            positions=[center + 0.015],
            widths=0.105,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.05,
            },
            whiskerprops={"color": "black", "linewidth": 0.95},
            capprops={"color": "black", "linewidth": 0.95},
            medianprops={"color": "black", "linewidth": 1.35},
            zorder=4,
        )

        # The deterministic rain jitter is saved with the source data so each
        # plotted point remains traceable to its participant.
        plot_x = center + rng.uniform(0.105, 0.365, len(part))
        part["PlotX"] = plot_x
        part["PlotPositionCenter"] = center
        plotted_parts.append(part)

        for correct, edge_color, linewidth in [
            (1, CORRECT_EDGE, 1.55),
            (0, WRONG_EDGE, 1.55),
        ]:
            mask = part["ExitCorrect"].eq(correct).to_numpy()
            if not mask.any():
                continue
            ax.scatter(
                plot_x[mask],
                values[mask],
                s=42,
                marker="o",
                facecolors=color,
                edgecolors=edge_color,
                linewidths=linewidth,
                alpha=0.96,
                zorder=6 if correct == 0 else 5,
            )

        row = summary.loc[summary["Scenario"].eq(scenario)].iloc[0]
        ax.text(
            center + 0.01,
            1.152,
            f"n={int(row['N'])}",
            ha="center",
            va="center",
            fontsize=18.0,
            color="#333333",
        )
        ax.text(
            center + 0.01,
            1.101,
            "Correct rate",
            ha="center",
            va="center",
            fontsize=17.5,
            fontweight="bold",
            color="#000000",
        )
        ax.text(
            center + 0.01,
            1.049,
            f"{float(row['CorrectExitRate']):.1%}",
            ha="center",
            va="center",
            fontsize=20.5,
            fontweight="bold",
            color="#000000",
        )

    plotted = pd.concat(plotted_parts, ignore_index=True)

    for left_index in [1, 3, 5, 7]:
        separator = (POSITIONS[left_index] + POSITIONS[left_index + 1]) / 2
        ax.axvline(
            separator,
            color="#B8B8B8",
            linestyle=(0, (2, 4)),
            linewidth=0.8,
            alpha=0.8,
            zorder=0,
        )

    ax.axhline(
        1.0,
        color="#555555",
        linestyle=(0, (5, 4)),
        linewidth=0.9,
        alpha=0.9,
        zorder=0,
    )
    ax.set_title(
        "Network Route Efficiency by Experimental Condition",
        fontsize=29.0,
        fontweight="bold",
        pad=51,
    )
    ax.set_xlabel(
        "Experimental Condition",
        fontsize=25.0,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_ylabel(
        "Network Route Efficiency",
        fontsize=25.0,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_xticks(POSITIONS)
    ax.set_xticklabels([DISPLAY_LABELS[x] for x in SCENARIO_ORDER])
    ax.set_xlim(POSITIONS[0] - 0.48, POSITIONS[-1] + 0.48)
    # The area above 1.0 provides room for the three annotation rows.
    ax.set_ylim(0.0, 1.185)
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    ax.tick_params(axis="x", width=1.0, length=5, pad=8)
    ax.tick_params(axis="y", width=1.0, length=5)
    ax.yaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
    )
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=CORRECT_EDGE,
            markeredgewidth=1.7,
            markersize=11,
            label="Correct exit (green outline)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=WRONG_EDGE,
            markeredgewidth=1.7,
            markersize=11,
            label="Wrong exit (red outline)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.112),
        ncol=2,
        frameon=False,
        fontsize=19.0,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.205, top=0.82)
    return fig, plotted


def save_outputs(
    fig: plt.Figure,
    plotted: pd.DataFrame,
    summary: pd.DataFrame,
    input_path: Path,
    output_dir: Path,
    dpi: int,
    seed: int,
    spatial_dir: Path,
    raw_dir: Path,
    sample_interval_s: float,
    transitions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{FIGURE_BASENAME}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    output_columns = [
        "Scenario",
        "ParticipantID",
        "FileName",
        "ShortestStartToTarget_m",
        "SpawnNode",
        "ShortestStartToChosenExit_m",
        "ObservedPathDistance_m",
        "RouteEfficiencyToTarget",
        "RouteEfficiencyToChosenExitRaw",
        "RouteEfficiency",
        "RouteEfficiencyCalculated",
        "NetworkShortestStartToEnd_m",
        "NetworkObservedPathDistance_m",
        "NetworkRouteEfficiency",
        "NetworkStartNode",
        "NetworkEndNode",
        "NetworkStartMatchesSpawn",
        "NetworkEndToChosenExit_m",
        "NetworkRawPointCount",
        "NetworkInvalidOrInitializationPointCount",
        "NetworkSampledPointCount",
        "NetworkMatchedPointCount",
        "NetworkMapMatchFraction3m",
        "NetworkTransitionCount",
        "NetworkIndirectTransitionCount",
        "NetworkLongGapCount",
        "NetworkInferredGapCount",
        "NetworkGapBridgeDistance_m",
        "NetworkMaxTimeGap_s",
        "NetworkMaxTransitionDistance_m",
        "NetworkMedianSnapDistance_m",
        "NetworkMaxSnapDistance_m",
        "NetworkLeadingUnmatchedSampleCount",
        "NetworkTrailingUnmatchedSampleCount",
        "NetworkFinalRecordedToMatchedDistance_m",
        "TargetExit",
        "ChosenExit",
        "ExitCorrect",
        "ExitStatus",
        "PlotPositionCenter",
        "PlotX",
    ]
    plotted.loc[:, output_columns].to_csv(
        output_dir / PLOT_DATA_NAME, index=False, encoding="utf-8-sig"
    )
    summary.to_csv(output_dir / SUMMARY_NAME, index=False, encoding="utf-8-sig")
    transitions.to_csv(
        output_dir / NETWORK_TRANSITIONS_NAME, index=False, encoding="utf-8-sig"
    )
    plotted.loc[plotted["ExitCorrect"].eq(0), output_columns].to_csv(
        output_dir / WRONG_EXIT_NAME, index=False, encoding="utf-8-sig"
    )

    metadata = {
        "input_file": str(input_path.resolve()),
        "reference_style_script": str(
            (ROOT / "graph_0911" / "basic_indicator" / "basic_indicator.py").resolve()
        ),
        "analysis_filter": (
            "AnalysisIncluded == True and AOICohortExcluded == False; "
            "formal 270-trial primary-analysis cohort"
        ),
        "route_efficiency_formula": (
            "NetworkShortestStartToEnd_m / NetworkObservedPathDistance_m"
        ),
        "shortest_distance_definition": (
            "Weighted network shortest distance between the first and last valid mapped "
            "trajectory nodes, using Dijkstra on the original undirected A* network"
        ),
        "raw_trajectory_directory": str(raw_dir.resolve()),
        "sampling_interval_s": sample_interval_s,
        "mapping_definition": (
            "Nearest 3D node within 3 m in the spawn node's connected component; "
            "non-finite coordinates/times and initialization positions are removed; "
            "the first and final valid observations are retained during sampling"
        ),
        "mapping_threshold_m": MAP_THRESHOLD_M,
        "spatial_source_directory": str(spatial_dir.resolve()),
        "spatial_source_files": ["astar_nodes.csv", "astar_edges.csv", "exits.csv"],
        "start_node_validation": (
            "SpawnNode-to-TargetExit distances reproduce ShortestStartToTarget_m "
            "within absolute tolerance 1e-8 m"
        ),
        "observed_distance_definition": (
            "Sum of weighted graph shortest distances between successive mapped nodes in "
            "temporal order. Consecutive duplicate nodes contribute zero; later returns "
            "remain. Nonadjacent nodes and missing intervals use inferred shortest connections."
        ),
        "endpoint_definition": (
            "Both distances use NetworkStartNode and NetworkEndNode. No assumed traversal "
            "to the exit centre is added. NetworkEndToChosenExit_m audits the remaining "
            "distance from the recorded matched endpoint to the labelled chosen exit."
        ),
        "wrong_exit_interpretation": (
            "All points measure network route efficiency to the recorded trajectory endpoint; "
            "red outlines indicate ChosenExit != TargetExit. Correct rates are reported separately."
        ),
        "original_target_efficiency_column": "RouteEfficiencyToTarget",
        "legacy_coordinate_ratio_column": "RouteEfficiencyToChosenExitRaw",
        "upper_clipped_trial_n": 0,
        "raw_efficiency_max": float(plotted["NetworkRouteEfficiency"].max()),
        "upper_clip_interpretation": (
            "No substantive clipping. Only machine-precision overshoot <= 1e-12 is "
            "rounded to 1; larger shortest-distance violations raise an error."
        ),
        "long_gap_threshold_s": LONG_GAP_S,
        "trial_n_with_long_mapping_gaps": int(plotted["NetworkLongGapCount"].gt(0).sum()),
        "long_mapping_gap_n": int(plotted["NetworkLongGapCount"].sum()),
        "trial_n_with_trimmed_final_observations": int(
            plotted["NetworkTrailingUnmatchedSampleCount"].gt(0).sum()
        ),
        "mapping_limitations": (
            "Nearest-node boundary switches can inflate reconstructed network distance. "
            "Shortest connections across missing intervals can omit unobserved detours. "
            "These are network-scale estimates, not exact physical travelled distances. "
            "All transitions, including mapping gaps, are supplied for audit."
        ),
        "outlier_removal": "none",
        "trial_n": int(len(plotted)),
        "participant_n": int(plotted["ParticipantID"].nunique()),
        "correct_exit_n": int(plotted["ExitCorrect"].sum()),
        "wrong_exit_n": int(plotted["ExitCorrect"].eq(0).sum()),
        "scenario_order": SCENARIO_ORDER,
        "palette": PALETTE,
        "correct_exit_outline": CORRECT_EDGE,
        "wrong_exit_outline": WRONG_EDGE,
        "jitter_seed": seed,
        "figure_size_inches": fig.get_size_inches().tolist(),
        "y_axis_limits": list(fig.axes[0].get_ylim()),
        "y_axis_truncated": False,
        "png_dpi": dpi,
        "outputs": {
            "png": f"{FIGURE_BASENAME}.png",
            "plot_data": PLOT_DATA_NAME,
            "group_summary": SUMMARY_NAME,
            "wrong_exit_subjects": WRONG_EXIT_NAME,
            "network_transitions": NETWORK_TRANSITIONS_NAME,
        },
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    spatial_dir = args.spatial_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    data, transitions = load_and_validate(
        input_path, spatial_dir, raw_dir, sample_interval_s=args.sample_interval
    )
    summary = summarize(data)
    fig, plotted = draw_figure(data, summary, seed=args.seed)
    save_outputs(
        fig,
        plotted,
        summary,
        input_path=input_path,
        output_dir=output_dir,
        dpi=args.dpi,
        seed=args.seed,
        spatial_dir=spatial_dir,
        raw_dir=raw_dir,
        sample_interval_s=args.sample_interval,
        transitions=transitions,
    )
    plt.close(fig)

    print(f"Input: {input_path}")
    print(f"Included trials: {len(plotted)}")
    print(f"Participants: {plotted['ParticipantID'].nunique()}")
    print(f"Correct exits: {int(plotted['ExitCorrect'].sum())}")
    print(f"Wrong exits: {int(plotted['ExitCorrect'].eq(0).sum())}")
    print(f"Output directory: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
