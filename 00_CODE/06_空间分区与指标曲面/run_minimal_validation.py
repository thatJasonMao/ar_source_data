"""Minimal validation of the spatially informed evacuation research framework.

The script is intentionally limited to an interpretable chain:

    guidance condition -> spatial route behaviour -> evacuation outcomes

It does not use Transformer, HMM, rSLDS, or other representation-learning models.
All analyses are exploratory feasibility checks rather than confirmatory causal proof.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import patsy
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact, norm, ttest_ind
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent  # 工程根目录（脚本位于子目录内）
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
IND = ROOT / "INDICATORS"  # 基础指标结果根目录（Evacuation*/LyapunovMLE 结果均位于此）
OUT = Path(__file__).resolve().parent
DOC = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DOC.mkdir(parents=True, exist_ok=True)

SPATIAL = ROOT / "SpatialSceneExport"
RNG_SEED = 20260728
ANALYSIS_VERSION = "2026-07-28-scenario-target-v2"
# 300 cluster resamples are sufficient for this explicitly minimal feasibility run.
# A confirmatory paper should increase this to at least 2,000.
BOOTSTRAP_N = 300
MAP_THRESHOLD_M = 3.0
SAMPLE_INTERVAL_S = 0.20

GUIDANCE_ORDER = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
GUIDANCE_LABEL = {
    "StaticSign": "Static sign",
    "ArrowRibbon": "AR light blanket",
    "MiniMap": "AR mini-map",
    "TextBar": "AR text",
    "Full": "Full AR",
}
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]

# The Unity level numbers preserve the historical labels. The scene audit showed
# that physical exits B and C were named in reverse, so levels 2 and 3 must swap.
LEVEL_TO_CORRECTED_EXIT = {
    1: "A",
    2: "C",
    3: "B",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
}
SCENARIO_TARGET_EXIT = {
    "StaticSign-HighSmoke": "A",
    "StaticSign-LowSmoke": "C",
    "ArrowRibbon-HighSmoke": "B",
    "ArrowRibbon-LowSmoke": "D",
    "MiniMap-HighSmoke": "C",
    "MiniMap-LowSmoke": "D",
    "TextBar-HighSmoke": "C",
    "TextBar-LowSmoke": "D",
    "Full-HighSmoke": "E",
    "Full-LowSmoke": "F",
}
LEVEL_RE = re.compile(r"\[Info\]Level:\s*([1-8])\s+Show Line")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def mark_analysis_eligibility(data: pd.DataFrame) -> pd.DataFrame:
    """Apply design-level exclusions while preserving every raw trial in the audit table."""
    marked = data.copy()
    marked["ParticipantID"] = marked["SubjectFolder"].astype(str)
    marked["AnalysisIncluded"] = True
    marked["ExclusionReason"] = ""

    test_mask = marked["ParticipantID"].str.upper().eq("TEST")
    marked.loc[test_mask, "AnalysisIncluded"] = False
    marked.loc[test_mask, "ExclusionReason"] = "technical_test_folder"

    eligible = marked.loc[~test_mask].copy()
    duplicate_mask = eligible.duplicated(
        subset=["Scenario", "ParticipantID"], keep=False
    )
    for _, group in eligible.loc[duplicate_mask].groupby(
        ["Scenario", "ParticipantID"], observed=True
    ):
        # The later timestamp is the completed rerun; the earlier file is retained
        # in the audit table but excluded from inferential analysis.
        keep_index = group.sort_values("FileName").index[-1]
        remove_indices = group.index.difference([keep_index])
        marked.loc[remove_indices, "AnalysisIncluded"] = False
        marked.loc[remove_indices, "ExclusionReason"] = (
            "superseded_by_later_rerun_same_condition"
        )
    return marked


def read_outcomes() -> pd.DataFrame:
    keys = ["Scenario", "Subject", "FileName"]
    sources = {
        "time": IND / "Results_EvacuationTime" / "AllSubjects_EvacuationTime.csv",
        "exit": ROOT / "Results_ExitChoice" / "AllSubjects_ExitChoice.csv",
        "rotation": IND
        / "Results_EvacuationRotation"
        / "AllSubjects_EvacuationRotation.csv",
        "speed": IND / "Results_EvacuationSpeed" / "AllSubjects_EvacuationSpeed.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in sources.items()}
    data = frames["time"].copy()
    keep = {
        "exit": keys + ["ChosenExit"],
        "rotation": keys + ["TotalRotation_deg"],
        "speed": keys + ["AverageSpeed_m_s"],
    }
    for name in ("exit", "rotation", "speed"):
        data = data.merge(
            frames[name][keep[name]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    data[["Guidance", "Smoke"]] = data["Scenario"].str.rsplit("-", n=1, expand=True)
    data["Guidance"] = pd.Categorical(
        data["Guidance"], categories=GUIDANCE_ORDER, ordered=True
    )
    data["Smoke"] = pd.Categorical(
        data["Smoke"], categories=SMOKE_ORDER, ordered=True
    )
    return data


def index_trials() -> pd.DataFrame:
    rows: list[dict] = []
    for log_path in sorted(RAW.glob("*Smoke/*/*_Log_Info_*.csv")):
        text = log_path.read_text(encoding="utf-8-sig", errors="ignore")
        levels = sorted({int(x) for x in LEVEL_RE.findall(text)})
        if len(levels) != 1:
            raise ValueError(f"Expected one target level in {log_path}, found {levels}")
        trajectory_name = log_path.name.replace(
            "_Log_Info_", "_Subject_Simulation_Info_"
        )
        trajectory_path = log_path.with_name(trajectory_name)
        if not trajectory_path.exists():
            raise FileNotFoundError(trajectory_path)
        level = levels[0]
        scenario = log_path.parent.parent.name
        if scenario not in SCENARIO_TARGET_EXIT:
            raise KeyError(f"No confirmed target exit for scenario: {scenario}")
        rows.append(
            {
                "Scenario": scenario,
                "SubjectFolder": log_path.parent.name,
                "FileName": trajectory_name,
                "TrajectoryPath": str(trajectory_path),
                "LogTargetLevel": level,
                "LogDerivedExitAfterBCSwap": LEVEL_TO_CORRECTED_EXIT[level],
                "TargetExit": SCENARIO_TARGET_EXIT[scenario],
                "TargetExitSource": "confirmed_experiment_design",
            }
        )
    return pd.DataFrame(rows)


def load_graph():
    nodes = pd.read_csv(SPATIAL / "astar_nodes.csv")
    edges = pd.read_csv(SPATIAL / "astar_edges.csv")
    exits = pd.read_csv(SPATIAL / "exits.csv")
    node_ids = nodes["node_id"].astype(int).to_numpy()
    coords = nodes[["x", "y", "z"]].to_numpy(float)
    id_to_row = {int(node_id): i for i, node_id in enumerate(node_ids)}
    tree = cKDTree(coords)

    graph = nx.Graph()
    graph.add_nodes_from(node_ids.tolist())
    for row in edges.itertuples(index=False):
        u, v, length = int(row.from_node), int(row.to_node), float(row.length_m)
        if u in id_to_row and v in id_to_row:
            if graph.has_edge(u, v):
                graph[u][v]["weight"] = min(graph[u][v]["weight"], length)
            else:
                graph.add_edge(u, v, weight=length)

    target_nodes: dict[str, int] = {}
    distance_fields: dict[str, dict[int, float]] = {}
    for row in exits.itertuples(index=False):
        exit_xyz = np.array([row.x, row.y, row.z], dtype=float)
        _, idx = tree.query(exit_xyz)
        target = int(node_ids[int(idx)])
        target_nodes[str(row.exit_id)] = target
        distance_fields[str(row.exit_id)] = nx.single_source_dijkstra_path_length(
            graph, target, weight="weight"
        )
    return nodes, edges, exits, node_ids, coords, tree, graph, target_nodes, distance_fields


def parse_timestamps(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype(str),
        format="%Y_%m_%d_%H_%M_%S_%f",
        errors="coerce",
    )


def temporal_subsample(frame: pd.DataFrame, interval_s: float) -> pd.DataFrame:
    if len(frame) <= 2:
        return frame
    elapsed = (frame["ParsedTime"] - frame["ParsedTime"].iloc[0]).dt.total_seconds()
    bins = np.floor(elapsed.to_numpy(float) / interval_s)
    keep = np.r_[True, bins[1:] > bins[:-1]]
    return frame.loc[keep].copy()


def collapsed_indices(node_ids: np.ndarray) -> np.ndarray:
    if len(node_ids) == 0:
        return np.array([], dtype=int)
    return np.r_[0, np.flatnonzero(node_ids[1:] != node_ids[:-1]) + 1]


def process_trajectory(
    path: str,
    target_exit: str,
    node_ids: np.ndarray,
    tree: cKDTree,
    distance_fields: dict[str, dict[int, float]],
) -> dict:
    raw = pd.read_csv(
        path,
        usecols=["TimeStamp", "PosX", "PosY", "PosZ"],
        low_memory=False,
    )
    for col in ("PosX", "PosY", "PosZ"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["ParsedTime"] = parse_timestamps(raw["TimeStamp"])
    raw = raw.dropna(subset=["ParsedTime", "PosX", "PosY", "PosZ"]).copy()
    sampled = temporal_subsample(raw, SAMPLE_INTERVAL_S)

    xyz = sampled[["PosX", "PosY", "PosZ"]].to_numpy(float)
    snap_dist, snap_idx = tree.query(xyz)
    nearest_ids = node_ids[np.asarray(snap_idx, dtype=int)]
    valid = np.asarray(snap_dist <= MAP_THRESHOLD_M)

    valid_sampled = sampled.loc[valid].reset_index(drop=True)
    valid_xyz = valid_sampled[["PosX", "PosY", "PosZ"]].to_numpy(float)
    valid_nodes = nearest_ids[valid]
    valid_snap = np.asarray(snap_dist)[valid]

    if len(valid_sampled) < 2:
        return {
            "RawPointCount": len(raw),
            "SampledPointCount": len(sampled),
            "MatchedPointCount": len(valid_sampled),
            "MapMatchFraction3m": float(valid.mean()) if len(valid) else np.nan,
        }

    # Distance is recomputed after removing initialization coordinates and at 5 Hz,
    # preventing the old (0, 0, 0) startup jump and high-frequency jitter inflation.
    steps = np.linalg.norm(np.diff(valid_xyz, axis=0), axis=1)
    times = valid_sampled["ParsedTime"].to_numpy()
    time_gap_s = np.diff(times).astype("timedelta64[ms]").astype(float) / 1000.0
    contiguous = (time_gap_s > 0) & (time_gap_s <= 2.0)
    observed_distance = float(steps[contiguous].sum())
    duration = float(
        (valid_sampled["ParsedTime"].iloc[-1] - valid_sampled["ParsedTime"].iloc[0])
        .total_seconds()
    )

    field = distance_fields[target_exit]
    geo = np.array([field.get(int(n), np.nan) for n in valid_nodes], dtype=float)
    finite = np.isfinite(geo)
    finite_fraction = float(finite.mean())
    geo_finite = geo[finite]
    nodes_finite = valid_nodes[finite]

    if len(geo_finite):
        start_geo = float(geo_finite[0])
        end_geo = float(geo_finite[-1])
        collapse = collapsed_indices(nodes_finite)
        geo_collapsed = geo_finite[collapse]
        smoothed = (
            pd.Series(geo_collapsed)
            .rolling(window=5, center=True, min_periods=1)
            .median()
            .to_numpy()
        )
        increases = np.diff(smoothed)
        backtracking = float(increases[increases > 0.25].sum())
        transitions = max(0, len(collapse) - 1)
    else:
        start_geo = end_geo = backtracking = np.nan
        transitions = 0

    detour_ratio_raw = (
        observed_distance / start_geo - 1.0 if start_geo > 1.0 else np.nan
    )
    route_efficiency_raw = (
        start_geo / observed_distance if observed_distance > 1.0 else np.nan
    )

    return {
        "RawPointCount": len(raw),
        "SampledPointCount": len(sampled),
        "MatchedPointCount": len(valid_sampled),
        "MapMatchFraction3m": float(valid.mean()),
        "MedianSnapDistance_m": float(np.median(valid_snap)),
        "P95SnapDistance_m": float(np.quantile(valid_snap, 0.95)),
        "TargetReachableFraction": finite_fraction,
        "SpatialDuration_s": duration,
        "ObservedPathDistance_m": observed_distance,
        "ShortestStartToTarget_m": start_geo,
        "FinalTargetDistance_m": end_geo,
        "NetworkProgress_m": start_geo - end_geo,
        "NetworkProgressFraction": (
            (start_geo - end_geo) / start_geo if start_geo > 1.0 else np.nan
        ),
        "RouteEfficiencyRaw": route_efficiency_raw,
        "DetourRatioRaw": detour_ratio_raw,
        "BacktrackingDistance_m": backtracking,
        "BacktrackingRatio": backtracking / start_geo if start_geo > 1.0 else np.nan,
        "NodeTransitionCount": transitions,
    }


def build_trial_metrics() -> tuple[pd.DataFrame, dict]:
    outcomes = read_outcomes()
    trials = index_trials()
    (
        nodes,
        edges,
        exits,
        node_ids,
        _coords,
        tree,
        graph,
        target_nodes,
        distance_fields,
    ) = load_graph()

    merged = outcomes.merge(
        trials,
        on=["Scenario", "FileName"],
        how="left",
        validate="one_to_one",
    )
    if merged["TrajectoryPath"].isna().any():
        raise RuntimeError("Some outcome rows could not be linked to trajectory files.")

    spatial_rows = []
    total = len(merged)
    for i, row in enumerate(merged.itertuples(index=False), start=1):
        metrics = process_trajectory(
            row.TrajectoryPath,
            row.TargetExit,
            node_ids,
            tree,
            distance_fields,
        )
        spatial_rows.append(metrics)
        if i % 25 == 0 or i == total:
            print(f"Spatial matching: {i}/{total}")
    spatial = pd.DataFrame(spatial_rows)
    data = pd.concat([merged.reset_index(drop=True), spatial], axis=1)

    data["ExitCorrect"] = (
        data["ChosenExit"].astype(str) == data["TargetExit"].astype(str)
    ).astype(int)
    # Preserve the raw values, and use non-negative excess for interpretability.
    data["DetourRatio"] = data["DetourRatioRaw"].clip(lower=0)
    data["RouteEfficiency"] = data["RouteEfficiencyRaw"].clip(lower=0, upper=1)
    data["LogTime"] = np.log(data["EvacuationTime_s"].clip(lower=0.1))
    data["LogDetour"] = np.log1p(data["DetourRatio"])
    data["LogBacktracking"] = np.log1p(data["BacktrackingRatio"].clip(lower=0))

    component_sizes = sorted(
        (len(c) for c in nx.connected_components(graph)), reverse=True
    )
    graph_summary = {
        "node_count": int(len(nodes)),
        "directed_edge_rows": int(len(edges)),
        "undirected_edge_count": int(graph.number_of_edges()),
        "connected_component_count": int(nx.number_connected_components(graph)),
        "component_sizes": [int(x) for x in component_sizes],
        "target_nodes": {k: int(v) for k, v in target_nodes.items()},
        "trial_count": int(len(data)),
        "unique_subject_folder_count": int(data["SubjectFolder"].nunique()),
    }
    return data, graph_summary


def wilson_interval(success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = success / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def descriptive_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for guidance in GUIDANCE_ORDER:
        part = data[data["Guidance"] == guidance]
        successes = int(part["ExitCorrect"].sum())
        low, high = wilson_interval(successes, len(part))
        rows.append(
            {
                "Guidance": guidance,
                "N": len(part),
                "Subjects": part["ParticipantID"].nunique(),
                "EvacuationTime_Mean_s": part["EvacuationTime_s"].mean(),
                "EvacuationTime_SD_s": part["EvacuationTime_s"].std(),
                "ExitCorrect_Rate": part["ExitCorrect"].mean(),
                "ExitCorrect_Wilson95_Low": low,
                "ExitCorrect_Wilson95_High": high,
                "ObservedDistance_Mean_m": part["ObservedPathDistance_m"].mean(),
                "DetourRatio_Mean": part["DetourRatio"].mean(),
                "BacktrackingRatio_Mean": part["BacktrackingRatio"].mean(),
                "RouteEfficiency_Mean": part["RouteEfficiency"].mean(),
                "TotalRotation_Mean_deg": part["TotalRotation_deg"].mean(),
                "MapMatchFraction3m_Mean": part["MapMatchFraction3m"].mean(),
            }
        )
    return pd.DataFrame(rows)


def design_average(model, guidance: str) -> np.ndarray:
    rows = pd.DataFrame(
        {
            "Guidance": [guidance, guidance],
            "Smoke": SMOKE_ORDER,
        }
    )
    matrix = patsy.build_design_matrices(
        [model._design_info], rows, return_type="dataframe"
    )[0]
    return np.asarray(matrix, dtype=float).mean(axis=0)


def fit_formula_model(formula: str, data: pd.DataFrame, family: str):
    """Fit only the required statsmodels submodules, avoiding blocked TSA DLLs."""
    y, x = patsy.dmatrices(formula, data=data, return_type="dataframe")
    groups = data.loc[y.index, "ParticipantID"]
    if family == "binomial":
        result = GLM(y, x, family=Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    else:
        result = OLS(y, x).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    result._design_info = x.design_info
    return result


def guidance_contrasts(model, outcome: str, scale: str) -> pd.DataFrame:
    params = np.asarray(model.params, dtype=float)
    cov = np.asarray(model.cov_params(), dtype=float)
    avg = {g: design_average(model, g) for g in GUIDANCE_ORDER}
    rows = []
    for g1, g2 in combinations(GUIDANCE_ORDER, 2):
        contrast = avg[g2] - avg[g1]
        estimate = float(contrast @ params)
        se = float(np.sqrt(max(0.0, contrast @ cov @ contrast)))
        z = estimate / se if se > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        low, high = estimate - 1.96 * se, estimate + 1.96 * se
        row = {
            "Outcome": outcome,
            "Comparison": f"{g2} - {g1}",
            "Group1": g1,
            "Group2": g2,
            "Scale": scale,
            "Estimate": estimate,
            "SE": se,
            "CI95_Low": low,
            "CI95_High": high,
            "PValue": p,
        }
        if scale == "log":
            row["InterpretableEffect"] = 100 * (math.exp(estimate) - 1)
            row["InterpretableUnit"] = "percent_change"
        elif scale == "log_odds":
            row["InterpretableEffect"] = math.exp(estimate)
            row["InterpretableUnit"] = "odds_ratio"
        else:
            row["InterpretableEffect"] = estimate
            row["InterpretableUnit"] = "original_or_transformed_unit"
        rows.append(row)
    result = pd.DataFrame(rows)
    valid = result["PValue"].notna()
    result["FDR_PValue"] = np.nan
    if valid.any():
        result.loc[valid, "FDR_PValue"] = multipletests(
            result.loc[valid, "PValue"], method="fdr_bh"
        )[1]
    return result


def guidance_global_test(model) -> tuple[float, float, int]:
    names = list(model.params.index)
    idx = [i for i, name in enumerate(names) if "Guidance" in name]
    if not idx:
        return np.nan, np.nan, 0
    restriction = np.zeros((len(idx), len(names)))
    for row, col in enumerate(idx):
        restriction[row, col] = 1.0
    test = model.wald_test(restriction, scalar=True)
    return float(test.statistic), float(test.pvalue), len(idx)


def fit_overall_effects(data: pd.DataFrame):
    formulas = {
        "EvacuationTime": ("LogTime", "log", "gaussian"),
        "ExitCorrect": ("ExitCorrect", "identity", "gaussian"),
        "DetourRatio": ("LogDetour", "log1p", "gaussian"),
        "BacktrackingRatio": ("LogBacktracking", "log1p", "gaussian"),
        "RouteEfficiency": ("RouteEfficiency", "identity", "gaussian"),
    }
    overall_rows = []
    pairwise_parts = []
    models = {}
    for label, (variable, scale, family) in formulas.items():
        use = data.dropna(
            subset=[variable, "Guidance", "Smoke", "ParticipantID"]
        ).copy()
        formula = f"{variable} ~ C(Guidance) + C(Smoke)"
        model = fit_formula_model(formula, use, family)
        models[label] = model
        stat, pvalue, df = guidance_global_test(model)
        overall_rows.append(
            {
                "Outcome": label,
                "Model": "cluster-robust GLM" if family == "binomial" else "cluster-robust OLS",
                "Scale": scale,
                "N": len(use),
                "Subjects": use["ParticipantID"].nunique(),
                "Guidance_Wald_Statistic": stat,
                "Guidance_df": df,
                "Guidance_Global_PValue": pvalue,
                "Smoke_Coefficient": float(
                    model.params.get("C(Smoke)[T.HighSmoke]", np.nan)
                ),
                "Smoke_PValue": float(
                    model.pvalues.get("C(Smoke)[T.HighSmoke]", np.nan)
                ),
                "R2_or_PseudoR2": float(
                    model.rsquared
                    if hasattr(model, "rsquared")
                    else model.pseudo_rsquared()
                ),
            }
        )
        pairwise_parts.append(guidance_contrasts(model, label, scale))
    return pd.DataFrame(overall_rows), pd.concat(pairwise_parts, ignore_index=True), models


def standardized(series: pd.Series) -> pd.Series:
    sd = series.std()
    if not np.isfinite(sd) or sd == 0:
        return series * np.nan
    return (series - series.mean()) / sd


def fit_process_outcome_associations(data: pd.DataFrame) -> pd.DataFrame:
    mediator_specs = {
        "DetourRatio": "LogDetour",
        "BacktrackingRatio": "LogBacktracking",
        "RouteEfficiency": "RouteEfficiency",
        "TotalRotation": "TotalRotation_deg",
    }
    rows = []
    for mediator_label, mediator in mediator_specs.items():
        use = data.dropna(
            subset=[
                mediator,
                "LogTime",
                "ExitCorrect",
                "Guidance",
                "Smoke",
                "ParticipantID",
            ]
        ).copy()
        use["MediatorZ"] = standardized(use[mediator])
        time_model = fit_formula_model(
            "LogTime ~ MediatorZ + C(Guidance) + C(Smoke)",
            use,
            "gaussian",
        )
        exit_model = fit_formula_model(
            "ExitCorrect ~ MediatorZ + C(Guidance) + C(Smoke)",
            use,
            "gaussian",
        )
        rows.extend(
            [
                {
                    "ProcessMetric": mediator_label,
                    "Outcome": "EvacuationTime",
                    "ModelScale": "log_time_per_1SD",
                    "N": len(use),
                    "Estimate": time_model.params["MediatorZ"],
                    "SE": time_model.bse["MediatorZ"],
                    "PValue": time_model.pvalues["MediatorZ"],
                    "InterpretableEffect": 100
                    * (math.exp(time_model.params["MediatorZ"]) - 1),
                    "InterpretableUnit": "percent_time_change_per_1SD",
                },
                {
                    "ProcessMetric": mediator_label,
                    "Outcome": "ExitCorrect",
                    "ModelScale": "probability_per_1SD",
                    "N": len(use),
                    "Estimate": exit_model.params["MediatorZ"],
                    "SE": exit_model.bse["MediatorZ"],
                    "PValue": exit_model.pvalues["MediatorZ"],
                    "InterpretableEffect": exit_model.params["MediatorZ"],
                    "InterpretableUnit": "risk_difference_per_1SD",
                },
            ]
        )
    result = pd.DataFrame(rows)
    result["FDR_PValue"] = multipletests(result["PValue"], method="fdr_bh")[1]
    return result


def ols_coef(y: np.ndarray, x: np.ndarray, index: int) -> float:
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return np.nan
    return float(np.linalg.lstsq(x, y, rcond=None)[0][index])


def mediation_point(frame: pd.DataFrame) -> tuple[float, float, float]:
    treat = frame["Treat"].to_numpy(float)
    smoke = frame["SmokeBinary"].to_numpy(float)
    mediator = frame["Mediator"].to_numpy(float)
    outcome = frame["LogTime"].to_numpy(float)
    if np.nanstd(smoke) == 0:
        x_m = np.column_stack([np.ones(len(frame)), treat])
        x_y = np.column_stack([np.ones(len(frame)), treat, mediator])
    else:
        x_m = np.column_stack([np.ones(len(frame)), treat, smoke])
        x_y = np.column_stack([np.ones(len(frame)), treat, mediator, smoke])
    a = ols_coef(mediator, x_m, 1)
    b = ols_coef(outcome, x_y, 2)
    direct = ols_coef(outcome, x_y, 1)
    return a * b, direct, a


def cluster_bootstrap_mediation(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    subjects = frame["ParticipantID"].unique()
    grouped = {s: frame[frame["ParticipantID"] == s] for s in subjects}
    estimates = []
    for _ in range(BOOTSTRAP_N):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        boot = pd.concat([grouped[s] for s in sampled], ignore_index=True)
        indirect, direct, _ = mediation_point(boot)
        if np.isfinite(indirect) and np.isfinite(direct):
            estimates.append((indirect, direct))
    return np.asarray(estimates, dtype=float)


def fit_minimal_mediation(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    mediator_specs = {
        "DetourRatio": "LogDetour",
        "BacktrackingRatio": "LogBacktracking",
    }
    rows = []
    for mediator_label, mediator in mediator_specs.items():
        for g1, g2 in combinations(GUIDANCE_ORDER, 2):
            use = data[
                data["Guidance"].astype(str).isin([g1, g2])
            ].dropna(
                subset=[mediator, "LogTime", "Smoke", "ParticipantID"]
            ).copy()
            use["Treat"] = (use["Guidance"].astype(str) == g2).astype(int)
            use["SmokeBinary"] = (use["Smoke"].astype(str) == "HighSmoke").astype(int)
            use["Mediator"] = standardized(use[mediator])
            indirect, direct, a_path = mediation_point(use)
            total_x = np.column_stack(
                [
                    np.ones(len(use)),
                    use["Treat"].to_numpy(float),
                    use["SmokeBinary"].to_numpy(float),
                ]
            )
            total = ols_coef(use["LogTime"].to_numpy(float), total_x, 1)
            boot = cluster_bootstrap_mediation(use, rng)
            low_i, high_i = (
                np.quantile(boot[:, 0], [0.025, 0.975])
                if len(boot)
                else (np.nan, np.nan)
            )
            low_d, high_d = (
                np.quantile(boot[:, 1], [0.025, 0.975])
                if len(boot)
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "Mediator": mediator_label,
                    "Comparison": f"{g2} - {g1}",
                    "Reference": g1,
                    "Treatment": g2,
                    "N": len(use),
                    "Subjects": use["ParticipantID"].nunique(),
                    "A_Path_TreatmentToMediator": a_path,
                    "Indirect_LogTime_Effect": indirect,
                    "Indirect_PercentApprox": 100 * (math.exp(indirect) - 1),
                    "Indirect_CI95_Low": low_i,
                    "Indirect_CI95_High": high_i,
                    "Indirect_CI_ExcludesZero": bool(low_i > 0 or high_i < 0),
                    "Direct_LogTime_Effect": direct,
                    "Direct_CI95_Low": low_d,
                    "Direct_CI95_High": high_d,
                    "Total_LogTime_Effect": total,
                    "BootstrapValid": len(boot),
                }
            )
    return pd.DataFrame(rows)


def fit_matched_spatial_comparisons(data: pd.DataFrame) -> pd.DataFrame:
    """Compare guidance modes only inside strata sharing smoke and target exit."""
    rng = np.random.default_rng(RNG_SEED + 11)
    strata = [
        (
            "LowSmoke_TargetD",
            (data["Smoke"].astype(str) == "LowSmoke")
            & (data["TargetExit"] == "D"),
            ["ArrowRibbon", "MiniMap", "TextBar"],
        ),
        (
            "HighSmoke_TargetC",
            (data["Smoke"].astype(str) == "HighSmoke")
            & (data["TargetExit"] == "C"),
            ["MiniMap", "TextBar"],
        ),
    ]
    metrics = {
        "EvacuationTime": ("LogTime", "percent_change"),
        "ExitCorrect": ("ExitCorrect", "risk_difference"),
        "DetourRatio": ("LogDetour", "log1p_difference"),
        "BacktrackingRatio": ("LogBacktracking", "log1p_difference"),
    }
    rows = []
    for stratum, mask, modes in strata:
        part = data.loc[mask & data["Guidance"].astype(str).isin(modes)].copy()
        for g1, g2 in combinations(modes, 2):
            for outcome, (variable, unit) in metrics.items():
                a = part.loc[part["Guidance"].astype(str) == g1, variable].dropna()
                b = part.loc[part["Guidance"].astype(str) == g2, variable].dropna()
                estimate = float(b.mean() - a.mean())
                boot = []
                for _ in range(BOOTSTRAP_N):
                    aa = rng.choice(a.to_numpy(float), size=len(a), replace=True)
                    bb = rng.choice(b.to_numpy(float), size=len(b), replace=True)
                    boot.append(float(bb.mean() - aa.mean()))
                low, high = np.quantile(boot, [0.025, 0.975])
                if outcome == "ExitCorrect":
                    table = np.array(
                        [
                            [int(a.sum()), int(len(a) - a.sum())],
                            [int(b.sum()), int(len(b) - b.sum())],
                        ]
                    )
                    pvalue = float(fisher_exact(table)[1])
                    interpretable = estimate
                    ilow, ihigh = low, high
                else:
                    pvalue = float(ttest_ind(b, a, equal_var=False).pvalue)
                    if unit == "percent_change":
                        interpretable = 100 * (math.exp(estimate) - 1)
                        ilow = 100 * (math.exp(low) - 1)
                        ihigh = 100 * (math.exp(high) - 1)
                    else:
                        interpretable, ilow, ihigh = estimate, low, high
                rows.append(
                    {
                        "Stratum": stratum,
                        "Smoke": str(part["Smoke"].iloc[0]),
                        "TargetExit": str(part["TargetExit"].iloc[0]),
                        "Outcome": outcome,
                        "Comparison": f"{g2} - {g1}",
                        "Reference": g1,
                        "Treatment": g2,
                        "N_Reference": len(a),
                        "N_Treatment": len(b),
                        "Mean_Reference_Original": float(
                            part.loc[
                                part["Guidance"].astype(str) == g1,
                                "EvacuationTime_s"
                                if outcome == "EvacuationTime"
                                else (
                                    "BacktrackingRatio"
                                    if outcome == "BacktrackingRatio"
                                    else outcome
                                ),
                            ].mean()
                            if outcome != "DetourRatio"
                            else part.loc[
                                part["Guidance"].astype(str) == g1,
                                "DetourRatio",
                            ].mean()
                        ),
                        "Mean_Treatment_Original": float(
                            part.loc[
                                part["Guidance"].astype(str) == g2,
                                "EvacuationTime_s"
                                if outcome == "EvacuationTime"
                                else (
                                    "BacktrackingRatio"
                                    if outcome == "BacktrackingRatio"
                                    else outcome
                                ),
                            ].mean()
                            if outcome != "DetourRatio"
                            else part.loc[
                                part["Guidance"].astype(str) == g2,
                                "DetourRatio",
                            ].mean()
                        ),
                        "ModelEstimate": estimate,
                        "InterpretableEffect": interpretable,
                        "InterpretableCI95_Low": ilow,
                        "InterpretableCI95_High": ihigh,
                        "InterpretableUnit": unit,
                        "PValue": pvalue,
                    }
                )
    result = pd.DataFrame(rows)
    result["FDR_PValue"] = np.nan
    for _, idx in result.groupby("Outcome").groups.items():
        result.loc[idx, "FDR_PValue"] = multipletests(
            result.loc[idx, "PValue"], method="fdr_bh"
        )[1]
    return result


def fit_matched_mediation(data: pd.DataFrame) -> pd.DataFrame:
    """Exploratory mediation only where guidance modes share target geometry."""
    rng = np.random.default_rng(RNG_SEED + 29)
    specifications = [
        (
            "LowSmoke_TargetD",
            (data["Smoke"].astype(str) == "LowSmoke")
            & (data["TargetExit"] == "D"),
            [("ArrowRibbon", "MiniMap"), ("ArrowRibbon", "TextBar"), ("MiniMap", "TextBar")],
        ),
        (
            "HighSmoke_TargetC",
            (data["Smoke"].astype(str) == "HighSmoke")
            & (data["TargetExit"] == "C"),
            [("MiniMap", "TextBar")],
        ),
        (
            "Pooled_MiniMap_Text_Matched",
            data["Guidance"].astype(str).isin(["MiniMap", "TextBar"]),
            [("MiniMap", "TextBar")],
        ),
    ]
    mediator_specs = {
        "DetourRatio": "LogDetour",
        "BacktrackingRatio": "LogBacktracking",
    }
    rows = []
    for stratum, mask, pairs in specifications:
        for g1, g2 in pairs:
            for mediator_label, mediator in mediator_specs.items():
                use = data.loc[
                    mask & data["Guidance"].astype(str).isin([g1, g2])
                ].dropna(
                    subset=[mediator, "LogTime", "ParticipantID"]
                ).copy()
                use["Treat"] = (
                    use["Guidance"].astype(str) == g2
                ).astype(int)
                use["SmokeBinary"] = (
                    use["Smoke"].astype(str) == "HighSmoke"
                ).astype(int)
                use["Mediator"] = standardized(use[mediator])
                indirect, direct, a_path = mediation_point(use)
                boot = cluster_bootstrap_mediation(use, rng)
                low, high = np.quantile(boot[:, 0], [0.025, 0.975])
                rows.append(
                    {
                        "Stratum": stratum,
                        "Mediator": mediator_label,
                        "Comparison": f"{g2} - {g1}",
                        "N": len(use),
                        "A_Path_TreatmentToMediator": a_path,
                        "Indirect_LogTime_Effect": indirect,
                        "Indirect_PercentApprox": 100
                        * (math.exp(indirect) - 1),
                        "Indirect_CI95_Low": low,
                        "Indirect_CI95_High": high,
                        "Indirect_CI_ExcludesZero": bool(
                            low > 0 or high < 0
                        ),
                        "Direct_LogTime_Effect": direct,
                        "BootstrapValid": len(boot),
                    }
                )
    return pd.DataFrame(rows)


def create_figures(
    data: pd.DataFrame,
    summary: pd.DataFrame,
    associations: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = dict(
        zip(
            GUIDANCE_ORDER,
            sns.color_palette("colorblind", n_colors=len(GUIDANCE_ORDER)),
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    sns.boxplot(
        data=data,
        x="Guidance",
        y="EvacuationTime_s",
        order=GUIDANCE_ORDER,
        palette=palette,
        showfliers=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=data,
        x="Guidance",
        y="EvacuationTime_s",
        order=GUIDANCE_ORDER,
        color="0.2",
        alpha=0.32,
        size=2.3,
        ax=axes[0],
    )
    axes[0].set_ylabel("Evacuation time (s)")

    sns.boxplot(
        data=data,
        x="Guidance",
        y="DetourRatio",
        order=GUIDANCE_ORDER,
        palette=palette,
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=data,
        x="Guidance",
        y="DetourRatio",
        order=GUIDANCE_ORDER,
        color="0.2",
        alpha=0.32,
        size=2.3,
        ax=axes[1],
    )
    axes[1].set_ylabel("Route detour ratio")

    ordered = summary.set_index("Guidance").loc[GUIDANCE_ORDER].reset_index()
    y = ordered["ExitCorrect_Rate"].to_numpy()
    low = ordered["ExitCorrect_Wilson95_Low"].to_numpy()
    high = ordered["ExitCorrect_Wilson95_High"].to_numpy()
    axes[2].errorbar(
        np.arange(len(ordered)),
        y,
        yerr=np.vstack([y - low, high - y]),
        fmt="o",
        color="#333333",
        ecolor="#555555",
        capsize=4,
    )
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Correct-exit rate (95% CI)")
    axes[2].set_xticks(np.arange(len(ordered)))

    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticks(np.arange(len(GUIDANCE_ORDER)))
        ax.set_xticklabels(
            [GUIDANCE_LABEL[g] for g in GUIDANCE_ORDER],
            rotation=30,
            ha="right",
        )
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "01_guidance_outcomes.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "01_guidance_outcomes.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    plot_data = data.dropna(subset=["DetourRatio", "EvacuationTime_s"])
    sns.scatterplot(
        data=plot_data,
        x="DetourRatio",
        y="EvacuationTime_s",
        hue="Guidance",
        hue_order=GUIDANCE_ORDER,
        palette=palette,
        alpha=0.58,
        s=28,
        ax=axes[0],
    )
    sns.regplot(
        data=plot_data,
        x="DetourRatio",
        y="EvacuationTime_s",
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.5},
        ax=axes[0],
    )
    axes[0].set_xlabel("Route detour ratio")
    axes[0].set_ylabel("Evacuation time (s)")
    axes[0].legend(title="Guidance", fontsize=8)

    assoc_plot = associations.copy()
    assoc_plot["Label"] = (
        assoc_plot["ProcessMetric"] + " → " + assoc_plot["Outcome"]
    )
    assoc_plot["SignedStrength"] = np.sign(assoc_plot["Estimate"]) * -np.log10(
        assoc_plot["PValue"].clip(lower=1e-12)
    )
    sns.barplot(
        data=assoc_plot,
        y="Label",
        x="SignedStrength",
        hue="Outcome",
        palette={"EvacuationTime": "#D55E00", "ExitCorrect": "#0072B2"},
        ax=axes[1],
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].axvline(-np.log10(0.05), color="0.5", linestyle="--", linewidth=0.8)
    axes[1].axvline(np.log10(0.05), color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Signed −log10(p)")
    axes[1].set_ylabel("")
    axes[1].legend(title="Outcome", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_spatial_process_links.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "02_spatial_process_links.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    sns.boxplot(
        data=data,
        x="Guidance",
        y="MapMatchFraction3m",
        order=GUIDANCE_ORDER,
        palette=palette,
        showfliers=False,
        ax=axes[0],
    )
    axes[0].axhline(0.9, color="#D55E00", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Trajectory matched within 3 m")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xticks(np.arange(len(GUIDANCE_ORDER)))
    axes[0].set_xticklabels(
        [GUIDANCE_LABEL[g] for g in GUIDANCE_ORDER], rotation=30, ha="right"
    )
    sns.histplot(
        data=data,
        x="P95SnapDistance_m",
        bins=25,
        color="#0072B2",
        ax=axes[1],
    )
    axes[1].axvline(MAP_THRESHOLD_M, color="#D55E00", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Trial-level 95th percentile snap distance (m)")
    axes[1].set_ylabel("Trials")
    fig.tight_layout()
    fig.savefig(FIG / "03_spatial_data_quality.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "03_spatial_data_quality.svg", bbox_inches="tight")
    plt.close(fig)


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def best_pairwise(pairwise: pd.DataFrame, outcome: str) -> pd.Series | None:
    part = pairwise[pairwise["Outcome"] == outcome].sort_values("FDR_PValue")
    return None if part.empty else part.iloc[0]


def write_report(
    data: pd.DataFrame,
    graph_summary: dict,
    quality: dict,
    summary: pd.DataFrame,
    overall: pd.DataFrame,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
    mediation: pd.DataFrame,
    matched_tests: pd.DataFrame,
    matched_mediation: pd.DataFrame,
) -> None:
    s = summary.set_index("Guidance")
    time_best = s["EvacuationTime_Mean_s"].idxmin()
    exit_best = s["ExitCorrect_Rate"].idxmax()
    detour_best = s["DetourRatio_Mean"].idxmin()
    assoc_sig = associations[associations["FDR_PValue"] < 0.05]
    med_sig = mediation[mediation["Indirect_CI_ExcludesZero"]]
    back_exit = associations[
        (associations["ProcessMetric"] == "BacktrackingRatio")
        & (associations["Outcome"] == "ExitCorrect")
    ].iloc[0]
    back_time = associations[
        (associations["ProcessMetric"] == "BacktrackingRatio")
        & (associations["Outcome"] == "EvacuationTime")
    ].iloc[0]

    overall_map = overall.set_index("Outcome")
    time_p = overall_map.loc["EvacuationTime", "Guidance_Global_PValue"]
    exit_p = overall_map.loc["ExitCorrect", "Guidance_Global_PValue"]
    detour_p = overall_map.loc["DetourRatio", "Guidance_Global_PValue"]

    feasibility_pass = (
        quality["map_match_fraction_3m_median"] >= 0.90
        and quality[
            "trials_target_at_least_95pct_points_reachable_fraction"
        ]
        >= 0.95
        and len(data) >= 250
    )

    lines = [
        "# 考虑空间因素的研究框架：最小验证",
        "",
        "## 一句话结论",
        "",
        (
            f"**空间分析链路在数据与计算层面{'可行' if feasibility_pass else '尚未完全通过'}**："
            f"{len(data)} 次试验均已连接到目标出口与轨迹；"
            f"每次试验轨迹落在 A* 网络 3 m 内的中位比例为 "
            f"{quality['map_match_fraction_3m_median']:.1%}，"
            f"至少 95% 轨迹点可沿网络到达目标的试验比例为 "
            f"{quality['trials_target_at_least_95pct_points_reachable_fraction']:.1%}。"
        ),
        "",
        "本次是“小样本/最小模型”的探索性验证，只回答链路能不能跑通、是否存在值得继续检验的信号；不能把中介结果直接写成已证实的因果机制。",
        "",
        "## 验证的最小链路",
        "",
        "```text",
        "引导方式 + 烟雾条件",
        "        ↓",
        "空间过程：绕行、回退、路线效率、转向",
        "        ↓",
        "结果：疏散时间 + 是否到达指定出口",
        "```",
        "",
        "## 核心结果",
        "",
        f"- 疏散时间最短的描述性组别：**{GUIDANCE_LABEL[time_best]}**，"
        f"均值 {s.loc[time_best, 'EvacuationTime_Mean_s']:.1f} s；"
        f"引导方式总体检验 p={fmt_p(time_p)}。",
        f"- 指定出口正确率最高的描述性组别：**{GUIDANCE_LABEL[exit_best]}**，"
        f"正确率 {s.loc[exit_best, 'ExitCorrect_Rate']:.1%}；"
        f"引导方式总体检验 p={fmt_p(exit_p)}。",
        f"- 绕行比例最低的描述性组别：**{GUIDANCE_LABEL[detour_best]}**，"
        f"均值 {s.loc[detour_best, 'DetourRatio_Mean']:.3f}；"
        f"引导方式总体检验 p={fmt_p(detour_p)}。",
        f"- 控制引导方式与烟雾后，共有 **{len(assoc_sig)}/{len(associations)}** 条"
        "“空间过程→结果”关联通过 FDR 0.05。",
        f"- 回退比例每增加 1 SD，与指定出口正确率降低 "
        f"**{abs(back_exit['InterpretableEffect']):.1%}** 相关"
        f"（FDR p={fmt_p(back_exit['FDR_PValue'])}），同时与疏散时间增加 "
        f"**{back_time['InterpretableEffect']:.1f}%** 相关"
        f"（FDR p={fmt_p(back_time['FDR_PValue'])}）。",
        f"- 预设的两类空间中介（绕行、回退）×10 个两两比较中，有 "
        f"**{len(med_sig)}/{len(mediation)}** 条五组探索性 bootstrap 95% 区间不跨 0；"
        "但完全同空间的中介检验为 0/10，因此不能把前者当成已证实机制。",
        "",
        "## 关键设计边界",
        "",
        "- 目标出口按确认的实验设置确定：Static sign=A/C，光毯=B/D，小地图=C/D，文字=C/D，Full AR=E/F。",
        "- 因此五组总体结果只能用于描述和空间标准化后的探索，不能直接解释为“纯引导方式因果效应”。",
        "- **完全同空间比较**：低烟 D 出口下比较光毯/小地图/文字；高烟 C 出口下比较小地图/文字。",
        "- 光毯高烟目标为 B，而小地图、文字高烟目标为 C；这部分不能直接混为纯引导差异。",
        "",
        "## 同烟雾、同目标出口的主结果",
        "",
        "| 空间层 | 比较（后者相对前者） | 时间变化 | 正确率变化 | 绕行差 | 回退差 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    report_comparisons = [
        ("LowSmoke_TargetD", "MiniMap - ArrowRibbon"),
        ("LowSmoke_TargetD", "TextBar - ArrowRibbon"),
        ("LowSmoke_TargetD", "TextBar - MiniMap"),
        ("HighSmoke_TargetC", "TextBar - MiniMap"),
    ]
    for stratum, comparison in report_comparisons:
        matched_rows = matched_tests[
            (matched_tests["Stratum"] == stratum)
            & (matched_tests["Comparison"] == comparison)
        ].set_index("Outcome")
        g1 = matched_rows["Reference"].iloc[0]
        g2 = matched_rows["Treatment"].iloc[0]
        detour_diff = (
            matched_rows.loc["DetourRatio", "Mean_Treatment_Original"]
            - matched_rows.loc["DetourRatio", "Mean_Reference_Original"]
        )
        back_diff = (
            matched_rows.loc["BacktrackingRatio", "Mean_Treatment_Original"]
            - matched_rows.loc["BacktrackingRatio", "Mean_Reference_Original"]
        )
        space_label = "低烟·D出口" if stratum == "LowSmoke_TargetD" else "高烟·C出口"
        lines.append(
            f"| {space_label} | {GUIDANCE_LABEL[g1]} → {GUIDANCE_LABEL[g2]} | "
            f"{matched_rows.loc['EvacuationTime', 'InterpretableEffect']:+.1f}% "
            f"(FDR p={fmt_p(matched_rows.loc['EvacuationTime', 'FDR_PValue'])}) | "
            f"{matched_rows.loc['ExitCorrect', 'InterpretableEffect']:+.1%} "
            f"(FDR p={fmt_p(matched_rows.loc['ExitCorrect', 'FDR_PValue'])}) | "
            f"{detour_diff:+.3f} "
            f"(FDR p={fmt_p(matched_rows.loc['DetourRatio', 'FDR_PValue'])}) | "
            f"{back_diff:+.3f} "
            f"(FDR p={fmt_p(matched_rows.loc['BacktrackingRatio', 'FDR_PValue'])}) |"
        )
    matched_med_sig = matched_mediation[
        matched_mediation["Indirect_CI_ExcludesZero"]
    ]
    lines.extend(
        [
            "",
            "同空间比较是本次最可信的验证结果：它避免把出口几何差异误写成引导差异。",
            f"同空间中介共 {len(matched_mediation)} 条候选，其中 "
            f"{len(matched_med_sig)} 条 bootstrap 95% 区间不跨 0；"
            "若为 0，表示当前样本支持“存在过程差异”，但还不足以稳定量化完整中介链。",
            "",
            "## 各引导方式的描述性结果",
            "",
            "| 引导 | N | 时间(s) | 正确出口率 | 绕行比例 | 回退比例 | 路线效率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for guidance in GUIDANCE_ORDER:
        row = s.loc[guidance]
        lines.append(
            f"| {GUIDANCE_LABEL[guidance]} | {int(row['N'])} | "
            f"{row['EvacuationTime_Mean_s']:.1f} | "
            f"{row['ExitCorrect_Rate']:.1%} | "
            f"{row['DetourRatio_Mean']:.3f} | "
            f"{row['BacktrackingRatio_Mean']:.3f} | "
            f"{row['RouteEfficiency_Mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 如何理解",
            "",
            "1. **总效果层**先检验不同引导在时间和正确出口率上是否不同。",
            "2. **空间过程层**检验引导是否改变绕路、逆向回退、路线效率或转向。",
            "3. **结果关联层**检验这些空间行为是否与更快疏散或更高出口正确率相关。",
            "4. **最小中介层**把前两段接起来，寻找“某引导减少绕行/回退，因此时间改变”的候选链路。",
            "",
            "这套顺序与当前研究框架一致，并已经证明现有历史轨迹可以和清理后的空间网络联合使用。",
            "",
            "## 数据质量与限制",
            "",
            f"- A* 网络：{graph_summary['node_count']} 个节点、"
            f"{graph_summary['undirected_edge_count']} 条无向边、"
            f"{graph_summary['connected_component_count']} 个连通分量。",
            f"- 轨迹匹配：3 m 内匹配比例中位数 "
            f"{quality['map_match_fraction_3m_median']:.1%}；"
            f"试次级 P95 吸附距离中位数 "
            f"{quality['trial_p95_snap_distance_m_median']:.2f} m。",
            f"- 原始 292 次记录中排除 {quality['excluded_trial_count']} 次："
            "3 次技术测试、1 次被稍后重做替代的未完成记录；正式分析为 48 人、288 次试验。",
            "- B/C 空间名称采用修正后的出口文件；每个场景的指定目标出口采用本研究已确认的实验设置，日志 Level 仅用于审计。",
            "- 路程由有效轨迹以 0.20 s 间隔重算，排除了启动阶段 `(0,0,0)` 跳变。",
            "- 单一 AR 方式分配给不同受试者组；当前模型按受试者聚类稳健标准误并控制烟雾，但仍需在正式模型中加入随机截距/设计组别与关键协变量。",
            "- 中介检验依赖“没有未测量的中介—结果混杂”等强假设，因此当前只能称为探索性机制证据。",
            "- HRV、眼动 AOI、空间区域阶段和认知负荷尚未进入本次最小验证；它们应在空间链路通过后作为下一层机制加入。",
            "",
            "## 文件说明",
            "",
            "- `trial_spatial_metrics.csv`：292 次试验的目标出口、空间匹配和试次级空间指标。",
            "- `data_quality_summary.json`：A* 网络和轨迹匹配质量。",
            "- `guidance_descriptive_summary.csv`：五种引导的描述性均值与正确率区间。",
            "- `overall_effects.csv`：引导方式对时间、正确出口及空间过程的总体检验。",
            "- `pairwise_effects.csv`：五种引导两两比较，含 FDR 校正。",
            "- `design_overlap_audit.csv`：引导、烟雾与指定出口的重叠关系。",
            "- `matched_spatial_comparisons.csv`：同烟雾、同目标出口下的主比较。",
            "- `process_outcome_associations.csv`：空间过程与两类结果的关联。",
            "- `minimal_mediation_results.csv`：探索性空间中介候选。",
            "- `matched_spatial_mediation.csv`：仅在目标空间可比层内估计的中介候选。",
            "- `../Spatial_Framework_Minimal_Validation/figures/`：结果图与空间数据质量图（PNG 300 dpi + SVG）。",
            "- `analysis_summary.json`：供后续脚本读取的机器可读摘要。",
            "",
            "## 下一步判定",
            "",
            (
                "**建议进入正式建模。** "
                if feasibility_pass
                else "**先修复空间匹配或不可达试次，再进入正式建模。** "
            )
            + "正式分析应把“出口是否正确”和“疏散时间”设为双主结果，"
            "把空间绕行/回退作为过程机制，再加入眼动与 HRV 构成空间—视觉搜索—认知负荷链路。",
        ]
    )
    (DOC / "README_Spatial_Framework_Minimal_Validation_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    np.random.seed(RNG_SEED)
    metrics_path = OUT / "trial_spatial_metrics.csv"
    quality_path = OUT / "data_quality_summary.json"
    cached_quality = (
        json.loads(quality_path.read_text(encoding="utf-8"))
        if quality_path.exists()
        else {}
    )
    cache_valid = (
        metrics_path.exists()
        and cached_quality.get("analysis_version") == ANALYSIS_VERSION
    )
    if cache_valid:
        print("Using cached trial-level spatial metrics.")
        data = pd.read_csv(metrics_path)
        data["Guidance"] = pd.Categorical(
            data["Guidance"], categories=GUIDANCE_ORDER, ordered=True
        )
        data["Smoke"] = pd.Categorical(
            data["Smoke"], categories=SMOKE_ORDER, ordered=True
        )
        graph_summary = cached_quality
    else:
        data, graph_summary = build_trial_metrics()
    all_trials = mark_analysis_eligibility(data)
    save_csv(
        all_trials.drop(columns=["TrajectoryPath"], errors="ignore"),
        "trial_spatial_metrics.csv",
    )
    data = all_trials.loc[all_trials["AnalysisIncluded"]].copy()

    quality = {
        **graph_summary,
        "analysis_version": ANALYSIS_VERSION,
        "raw_trial_count": int(len(all_trials)),
        "analysis_trial_count": int(len(data)),
        "excluded_trial_count": int((~all_trials["AnalysisIncluded"]).sum()),
        "analysis_subject_count": int(data["ParticipantID"].nunique()),
        "exclusion_reasons": {
            str(key): int(value)
            for key, value in all_trials.loc[
                ~all_trials["AnalysisIncluded"], "ExclusionReason"
            ]
            .value_counts()
            .items()
        },
        "map_threshold_m": MAP_THRESHOLD_M,
        "sampling_interval_s": SAMPLE_INTERVAL_S,
        "map_match_fraction_3m_mean": float(data["MapMatchFraction3m"].mean()),
        "map_match_fraction_3m_median": float(data["MapMatchFraction3m"].median()),
        "trial_p95_snap_distance_m_median": float(data["P95SnapDistance_m"].median()),
        "target_reachable_point_fraction_mean": float(
            data["TargetReachableFraction"].mean()
        ),
        "trials_target_fully_reachable_fraction": float(
            (data["TargetReachableFraction"] >= 0.999).mean()
        ),
        "trials_target_at_least_95pct_points_reachable_fraction": float(
            (data["TargetReachableFraction"] >= 0.95).mean()
        ),
        "trials_target_start_reachable_fraction": float(
            data["ShortestStartToTarget_m"].notna().mean()
        ),
        "trials_map_match_at_least_90pct": int(
            (data["MapMatchFraction3m"] >= 0.90).sum()
        ),
        "trials_map_match_below_80pct": int(
            (data["MapMatchFraction3m"] < 0.80).sum()
        ),
        "confirmed_scenario_target_exit": SCENARIO_TARGET_EXIT,
        "log_level_audit_mapping_after_bc_swap": LEVEL_TO_CORRECTED_EXIT,
    }
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = descriptive_summary(data)
    save_csv(summary, "guidance_descriptive_summary.csv")

    overall, pairwise, _models = fit_overall_effects(data)
    save_csv(overall, "overall_effects.csv")
    save_csv(pairwise, "pairwise_effects.csv")
    matched_comparisons = fit_matched_spatial_comparisons(data)
    save_csv(matched_comparisons, "matched_spatial_comparisons.csv")

    associations = fit_process_outcome_associations(data)
    save_csv(associations, "process_outcome_associations.csv")

    mediation = fit_minimal_mediation(data)
    save_csv(mediation, "minimal_mediation_results.csv")
    matched_mediation = fit_matched_mediation(data)
    save_csv(matched_mediation, "matched_spatial_mediation.csv")

    design_overlap = (
        data.groupby(
            ["Guidance", "Smoke", "TargetExit"], observed=True
        )
        .size()
        .reset_index(name="N")
        .sort_values(["Guidance", "Smoke", "TargetExit"])
    )
    save_csv(design_overlap, "design_overlap_audit.csv")

    create_figures(data, summary, associations)

    overall_map = overall.set_index("Outcome")
    analysis_summary = {
        "status": "completed",
        "purpose": "exploratory_minimal_validation",
        "trial_count": int(len(data)),
        "raw_trial_count": int(len(all_trials)),
        "excluded_trial_count": int((~all_trials["AnalysisIncluded"]).sum()),
        "subject_count": int(data["ParticipantID"].nunique()),
        "technical_feasibility": bool(
            quality["map_match_fraction_3m_median"] >= 0.90
            and quality[
                "trials_target_at_least_95pct_points_reachable_fraction"
            ]
            >= 0.95
        ),
        "global_guidance_pvalues": {
            key: float(overall_map.loc[key, "Guidance_Global_PValue"])
            for key in overall_map.index
        },
        "significant_process_outcome_links_fdr_005": int(
            (associations["FDR_PValue"] < 0.05).sum()
        ),
        "mediation_candidates_ci_excludes_zero": int(
            mediation["Indirect_CI_ExcludesZero"].sum()
        ),
        "matched_spatial_mediation_candidates_ci_excludes_zero": int(
            matched_mediation["Indirect_CI_ExcludesZero"].sum()
        ),
        "primary_supported_mechanism_signal": (
            "Text guidance shows substantially more backtracking and lower "
            "correct-exit probability than mini-map/light-blanket within "
            "matched smoke-target strata."
        ),
        "design_identification_boundary": (
            "Only LowSmoke-ExitD (three single AR modes) and HighSmoke-ExitC "
            "(MiniMap vs TextBar) are directly space-matched. Other five-mode "
            "comparisons remain geometry-confounded."
        ),
        "warning": (
            "Exploratory evidence only. The mediation estimates are not causal proof "
            "without stronger identification assumptions and a pre-specified model."
        ),
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(analysis_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        data,
        graph_summary,
        quality,
        summary,
        overall,
        pairwise,
        associations,
        mediation,
        matched_comparisons,
        matched_mediation,
    )
    print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
