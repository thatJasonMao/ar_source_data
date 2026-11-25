"""Spatially standardized full-chain analysis for the five guidance modes.

Safety boundary
---------------
All upstream source files are read-only. Every generated file is written under
``Results_Guidance_Spatial_Causal_Analysis``.

Primary estimand
----------------
The primary cohort excludes all trials from three participants whose AOI is
unavailable throughout: LAT_A2, LXC_C1, and MC_B2. Spatial difficulty is based
only on pre-treatment A* shortest-path distance from the participant's actual
left/right spawn node to the assigned target exit. A-F are given equal weight
in the reference spatial distribution. A-H are retained in the geometry audit.
"""

from __future__ import annotations

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import norm, t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW_DATA"  # 原始数据根目录（10 个场景目录均位于此）
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
FIG = OUT / "figures"
NOTE = OUT / "notes"
for directory in (DATA, TABLE, FIG, NOTE):
    directory.mkdir(parents=True, exist_ok=True)

SPATIAL = ROOT / "SpatialSceneExport"
TRIAL_SOURCE = ROOT / "Spatial_Framework_Minimal_Validation" / "trial_spatial_metrics.csv"

SEED = 20260731
RNG = np.random.default_rng(SEED)
BOOT_N = 500
PROCESS_BOOT_N = 200
MED_BOOT_N = 300

GUIDANCE_ORDER = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]
GUIDANCE_CN = {
    "StaticSign": "静态标志",
    "ArrowRibbon": "AR光毯",
    "MiniMap": "AR小地图",
    "TextBar": "AR文字",
    "Full": "完整AR",
}
SMOKE_CN = {"LowSmoke": "低烟", "HighSmoke": "高烟"}
PALETTE = {
    "StaticSign": "#6B7280",
    "ArrowRibbon": "#0072B2",
    "MiniMap": "#009E73",
    "TextBar": "#D55E00",
    "Full": "#CC79A7",
}
FEATURE_CN = {
    "Guidance": "引导模式",
    "Smoke": "烟雾",
    "SpatialDifficultyRatio": "空间难度",
    "HeadAngularPathRate_deg_s": "头部搜索路径率",
    "FixationTimeProportion": "凝视时间比例",
    "SaccadeRate_per_min": "眼跳事件率",
    "EyeGazeScanpathRate_deg_s": "眼在头内扫视路径率",
    "GazeP90Dispersion_deg": "注视分散程度",
    "DetourRatio": "绕行比例",
    "BacktrackingRatio": "回退比例",
    "GuideAcquired": "是否获取引导",
    "PromptGuideAcquisitionScore": "及时获取得分",
    "GuideDwellProportion": "引导停留比例",
    "GuideVisitRate_per_min": "引导访问率",
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
AOI_EXCLUDED = {"LAT_A2", "LXC_C1", "MC_B2"}
TARGET_EXITS = list("ABCDEF")
KEYS = ["Scenario", "ParticipantID", "FileName"]

mpl.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLE / name, index=False, encoding="utf-8-sig")


def save_data(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DATA / name, index=False, encoding="utf-8-sig")


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def boot_p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    left = (np.sum(values <= 0) + 1) / (len(values) + 1)
    right = (np.sum(values >= 0) + 1) / (len(values) + 1)
    return min(1.0, 2 * min(left, right))


def load_graph():
    nodes = pd.read_csv(SPATIAL / "astar_nodes.csv")
    edges = pd.read_csv(SPATIAL / "astar_edges.csv")
    exits = pd.read_csv(SPATIAL / "exits.csv")
    node_ids = nodes["node_id"].astype(int).to_numpy()
    xyz = nodes[["x", "y", "z"]].to_numpy(float)
    tree = cKDTree(xyz)
    graph = nx.Graph()
    graph.add_nodes_from(node_ids.tolist())
    for row in edges.itertuples(index=False):
        u, v, w = int(row.from_node), int(row.to_node), float(row.length_m)
        if graph.has_edge(u, v):
            graph[u][v]["weight"] = min(graph[u][v]["weight"], w)
        else:
            graph.add_edge(u, v, weight=w)
    return nodes, edges, exits, node_ids, tree, graph


def load_trials() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(TRIAL_SOURCE)
    audit["TargetExit"] = audit["Scenario"].map(SCENARIO_TARGET_EXIT)
    audit["ExitCorrect"] = audit["ChosenExit"].astype(str).eq(audit["TargetExit"]).astype(int)
    eligible = audit.loc[audit["AnalysisIncluded"].astype(bool)].copy()
    if len(eligible) != 288 or eligible["ParticipantID"].nunique() != 48:
        raise RuntimeError("Expected 288 eligible trials from 48 participants.")
    if eligible.duplicated(["Scenario", "ParticipantID"]).any():
        raise RuntimeError("Duplicate eligible participant-condition rows found.")
    eligible["AOICohortExcluded"] = eligible["ParticipantID"].isin(AOI_EXCLUDED)
    eligible["AOICohortExclusionReason"] = np.where(
        eligible["AOICohortExcluded"], "AOI unavailable throughout participant trials", ""
    )
    main = eligible.loc[~eligible["AOICohortExcluded"]].copy()
    if len(main) != 270 or main["ParticipantID"].nunique() != 45:
        raise RuntimeError("Complete AOI cohort should contain 270 trials and 45 participants.")
    save_data(audit, "trial_audit_all_records.csv")
    save_data(eligible, "analysis_trials_all_288.csv")
    excluded = eligible.loc[eligible["AOICohortExcluded"]].copy()
    save_csv(excluded, "excluded_aoi_unavailable_participants_and_trials.csv")
    return audit, eligible, main


def first_spawn_node(
    row, node_ids: np.ndarray, tree: cKDTree
) -> tuple[int, float, float, float, float]:
    path = RAW / row.Scenario / row.ParticipantID / row.FileName
    raw = pd.read_csv(path, usecols=["PosX", "PosY", "PosZ"], nrows=500)
    xyz = raw[["PosX", "PosY", "PosZ"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 0.1)
    if not valid.any():
        raise RuntimeError(f"No non-zero spawn position in {path}")
    point = xyz[np.flatnonzero(valid)[0]]
    snap, idx = tree.query(point)
    return int(node_ids[int(idx)]), float(snap), *map(float, point)


def build_geometry(
    eligible: pd.DataFrame,
    nodes: pd.DataFrame,
    exits: pd.DataFrame,
    node_ids: np.ndarray,
    tree: cKDTree,
    graph: nx.Graph,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spawn_rows = []
    for row in eligible.itertuples(index=False):
        node, snap, x, y, z = first_spawn_node(row, node_ids, tree)
        spawn_rows.append(
            {
                "Scenario": row.Scenario,
                "ParticipantID": row.ParticipantID,
                "FileName": row.FileName,
                "SpawnNode": node,
                "SpawnSnapDistance_m": snap,
                "SpawnX": x,
                "SpawnY": y,
                "SpawnZ": z,
            }
        )
    spawn = pd.DataFrame(spawn_rows)
    common = spawn.groupby("SpawnNode").size().sort_values(ascending=False)
    if len(common) != 2 or set(common.values) != {144}:
        raise RuntimeError(f"Unexpected spawn-node distribution: {common.to_dict()}")
    coord = nodes.set_index("node_id")[["x", "y", "z"]]
    left_node = int(coord.loc[common.index]["x"].idxmin())
    right_node = int(coord.loc[common.index]["x"].idxmax())
    side_map = {left_node: "Left", right_node: "Right"}
    spawn["StartSide"] = spawn["SpawnNode"].map(side_map)

    exit_rows = []
    for start_node, side in side_map.items():
        for ex in exits.itertuples(index=False):
            _, idx = tree.query(np.array([ex.x, ex.y, ex.z], float))
            target_node = int(node_ids[int(idx)])
            path = nx.shortest_path(graph, start_node, target_node, weight="weight")
            distance = float(nx.path_weight(graph, path, weight="weight"))
            pts = coord.loc[path].to_numpy(float)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            dy = np.abs(np.diff(pts[:, 1]))
            branch = np.array([graph.degree(v) >= 3 for v in path], bool)
            branch_clusters = int(np.sum(branch & ~np.r_[False, branch[:-1]]))
            exit_rows.append(
                {
                    "StartSide": side,
                    "SpawnNode": start_node,
                    "Exit": str(ex.exit_id),
                    "TargetNode": target_node,
                    "AStarShortestDistance_m": distance,
                    "AStarPathNodeN": len(path),
                    "BranchNodeClusterN_descriptive": branch_clusters,
                    "StairSegmentLength_m_descriptive": float(seg[dy > 0.05].sum()),
                    "AbsoluteVerticalChange_m_descriptive": float(dy.sum()),
                    "UsedInOutcomeStandardization": str(ex.exit_id) in TARGET_EXITS,
                }
            )
    geometry = pd.DataFrame(exit_rows)
    ref = (
        geometry.loc[geometry["UsedInOutcomeStandardization"]]
        .groupby("StartSide")["AStarShortestDistance_m"]
        .mean()
        .rename("ReferenceDistance_AF_EqualWeight_m")
    )
    geometry = geometry.merge(ref, on="StartSide", how="left")
    geometry["SpatialDifficultyRatio"] = (
        geometry["AStarShortestDistance_m"] / geometry["ReferenceDistance_AF_EqualWeight_m"]
    )
    geometry["StandardizationMultiplier"] = 1 / geometry["SpatialDifficultyRatio"]
    save_csv(geometry, "exit_spatial_difficulty_by_spawn.csv")
    component_rows = []
    exit_node_map = (
        geometry[["Exit", "TargetNode"]].drop_duplicates().groupby("TargetNode")["Exit"].apply(",".join)
    )
    for component_id, component in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True), start=1
    ):
        component_rows.append(
            {
                "ComponentID": component_id,
                "NodeN": len(component),
                "ContainsSpawnNode": bool(set(side_map).intersection(component)),
                "TargetExitsInComponent": ",".join(
                    exit_node_map.loc[exit_node_map.index.intersection(component)].tolist()
                ),
                "UsedForPrimaryPaths": bool(set(side_map).intersection(component)),
            }
        )
    save_csv(pd.DataFrame(component_rows), "astar_connected_component_audit.csv")

    summary = (
        geometry.groupby("Exit", as_index=False)
        .agg(
            MeanAStarDistance_m=("AStarShortestDistance_m", "mean"),
            MinAStarDistance_m=("AStarShortestDistance_m", "min"),
            MaxAStarDistance_m=("AStarShortestDistance_m", "max"),
            MeanDifficultyRatio=("SpatialDifficultyRatio", "mean"),
            UsedInOutcomeStandardization=("UsedInOutcomeStandardization", "max"),
        )
        .sort_values("Exit")
    )
    save_csv(summary, "exit_spatial_difficulty_summary.csv")
    save_csv(
        spawn.groupby(["SpawnNode", "StartSide"], as_index=False)
        .agg(
            TrialN=("Scenario", "size"),
            ParticipantN=("ParticipantID", "nunique"),
            MeanSpawnSnapDistance_m=("SpawnSnapDistance_m", "mean"),
            SpawnX=("SpawnX", "mean"),
            SpawnY=("SpawnY", "mean"),
            SpawnZ=("SpawnZ", "mean"),
        ),
        "spawn_node_audit.csv",
    )

    key = geometry.rename(columns={"Exit": "TargetExit"})[
        [
            "StartSide",
            "SpawnNode",
            "TargetExit",
            "AStarShortestDistance_m",
            "ReferenceDistance_AF_EqualWeight_m",
            "SpatialDifficultyRatio",
            "StandardizationMultiplier",
        ]
    ]
    augmented = eligible.merge(
        spawn,
        on=["Scenario", "ParticipantID", "FileName"],
        how="left",
        validate="one_to_one",
    ).merge(
        key,
        on=["StartSide", "SpawnNode", "TargetExit"],
        how="left",
        validate="many_to_one",
    )
    augmented["StandardizedTime_s"] = (
        augmented["EvacuationTime_s"] * augmented["StandardizationMultiplier"]
    )
    augmented["RotationRate_deg_s"] = (
        augmented["TotalRotation_deg"] / augmented["EvacuationTime_s"].clip(lower=0.1)
    )
    augmented["ExistingMetricDifficultyRatio"] = (
        augmented["ShortestStartToTarget_m"]
        / augmented["ReferenceDistance_AF_EqualWeight_m"]
    )
    augmented["ExistingMetricStandardizedTime_s"] = (
        augmented["EvacuationTime_s"] / augmented["ExistingMetricDifficultyRatio"]
    )
    return augmented, geometry


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    # Additive guidance + smoke + distance model. A saturated guidance-smoke
    # model would make distance exactly collinear with the fixed target assigned
    # to each cell. The additive restriction is explicit and sensitivity-tested.
    cols = [
        frame["Guidance"].astype(str).eq(g).astype(float).to_numpy()
        for g in GUIDANCE_ORDER[1:]
    ]
    cols.extend(
        [
            frame["Smoke"].astype(str).eq("HighSmoke").astype(float).to_numpy(),
            np.log(frame["SpatialDifficultyRatio"].astype(float)).to_numpy(),
        ]
    )
    return np.column_stack(cols)


def correctness_standardized(
    frame: pd.DataFrame, geometry: pd.DataFrame
) -> dict[tuple[str, str], float]:
    x = frame.copy()
    model = LogisticRegression(
        C=10.0, solver="lbfgs", max_iter=2000, random_state=SEED
    ).fit(design_matrix(x), x["ExitCorrect"].astype(int))
    estimates = {}
    for smoke in SMOKE_ORDER:
        sides = frame.loc[frame["Smoke"].eq(smoke), "StartSide"].astype(str).unique()
        if len(sides) != 1:
            raise RuntimeError(f"Smoke {smoke} should map to exactly one spawn side.")
        grid0 = geometry.loc[
            geometry["Exit"].isin(TARGET_EXITS)
            & geometry["StartSide"].eq(sides[0])
        ].copy()
        for guidance in GUIDANCE_ORDER:
            grid = grid0.copy()
            grid["Guidance"] = guidance
            grid["Smoke"] = smoke
            estimates[(guidance, smoke)] = float(
                model.predict_proba(design_matrix(grid))[:, 1].mean()
            )
    return estimates


def outcome_estimates(
    frame: pd.DataFrame, geometry: pd.DataFrame
) -> tuple[dict, dict, dict]:
    raw = {}
    standardized = {}
    for guidance in GUIDANCE_ORDER:
        for smoke in SMOKE_ORDER:
            part = frame.loc[
                frame["Guidance"].astype(str).eq(guidance) & frame["Smoke"].eq(smoke)
            ]
            raw[(guidance, smoke)] = float(part["EvacuationTime_s"].mean())
            standardized[(guidance, smoke)] = float(part["StandardizedTime_s"].mean())
    correct = correctness_standardized(frame, geometry)
    return raw, standardized, correct


def bootstrap_outcomes(
    frame: pd.DataFrame, geometry: pd.DataFrame, n_boot: int = BOOT_N
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw0, std0, corr0 = outcome_estimates(frame, geometry)
    ids = frame["ParticipantID"].unique()
    boot_raw: list[dict] = []
    boot_std: list[dict] = []
    boot_corr: list[dict] = []
    for _ in range(n_boot):
        sampled = RNG.choice(ids, len(ids), replace=True)
        boot = pd.concat(
            [frame.loc[frame["ParticipantID"].eq(pid)] for pid in sampled],
            ignore_index=True,
        )
        try:
            raw, std, corr = outcome_estimates(boot, geometry)
        except Exception:
            continue
        boot_raw.append(raw)
        boot_std.append(std)
        boot_corr.append(corr)

    summary_rows = []
    for guidance in GUIDANCE_ORDER:
        for smoke in SMOKE_ORDER:
            key = (guidance, smoke)
            part = frame.loc[
                frame["Guidance"].astype(str).eq(guidance) & frame["Smoke"].eq(smoke)
            ]
            rv = np.array([b[key] for b in boot_raw], float)
            sv = np.array([b[key] for b in boot_std], float)
            cv = np.array([b[key] for b in boot_corr], float)
            wlo, whi = wilson(int(part["ExitCorrect"].sum()), len(part))
            summary_rows.append(
                {
                    "Guidance": guidance,
                    "GuidanceCN": GUIDANCE_CN[guidance],
                    "Smoke": smoke,
                    "SmokeCN": SMOKE_CN[smoke],
                    "N": len(part),
                    "ParticipantN": part["ParticipantID"].nunique(),
                    "RawMeanTime_s": raw0[key],
                    "RawTimeCI95Low": np.quantile(rv, 0.025),
                    "RawTimeCI95High": np.quantile(rv, 0.975),
                    "StandardizedMeanTime_s": std0[key],
                    "StandardizedTimeCI95Low": np.quantile(sv, 0.025),
                    "StandardizedTimeCI95High": np.quantile(sv, 0.975),
                    "ObservedExitCorrectRate": float(part["ExitCorrect"].mean()),
                    "ObservedCorrectWilsonLow": wlo,
                    "ObservedCorrectWilsonHigh": whi,
                    "SpatiallyStandardizedCorrectProbability": corr0[key],
                    "StandardizedCorrectCI95Low": np.quantile(cv, 0.025),
                    "StandardizedCorrectCI95High": np.quantile(cv, 0.975),
                    "MeanSpatialDifficultyRatio": float(part["SpatialDifficultyRatio"].mean()),
                    "BootstrapSuccessN": len(boot_std),
                }
            )
    summary = pd.DataFrame(summary_rows)

    pair_rows = []
    for smoke in SMOKE_ORDER:
        for reference, comparison in combinations(GUIDANCE_ORDER, 2):
            ka, kb = (reference, smoke), (comparison, smoke)
            for outcome, estimate, vals, scale in [
                (
                    "StandardizedTime_s",
                    std0[kb] - std0[ka],
                    np.array([b[kb] - b[ka] for b in boot_std]),
                    "seconds_difference",
                ),
                (
                    "StandardizedTime_percent",
                    (std0[kb] / std0[ka] - 1) * 100,
                    np.array([(b[kb] / b[ka] - 1) * 100 for b in boot_std]),
                    "percent_change",
                ),
                (
                    "SpatiallyStandardizedCorrectProbability",
                    (corr0[kb] - corr0[ka]) * 100,
                    np.array([(b[kb] - b[ka]) * 100 for b in boot_corr]),
                    "percentage_point_difference",
                ),
            ]:
                pair_rows.append(
                    {
                        "Smoke": smoke,
                        "Reference": reference,
                        "Comparison": comparison,
                        "Outcome": outcome,
                        "Estimate": estimate,
                        "CI95Low": np.quantile(vals, 0.025),
                        "CI95High": np.quantile(vals, 0.975),
                        "PValueBootstrap": boot_p(vals),
                        "Scale": scale,
                        "BootstrapSuccessN": len(vals),
                    }
                )
    pairs = pd.DataFrame(pair_rows)
    pairs["PValueFDR"] = np.nan
    for outcome, idx in pairs.groupby("Outcome").groups.items():
        pairs.loc[idx, "PValueFDR"] = multipletests(
            pairs.loc[idx, "PValueBootstrap"], method="fdr_bh"
        )[1]
    return summary, pairs, pd.DataFrame(
        [
            {
                "BootstrapIndex": i,
                **{
                    f"{g}|{s}|StandardizedTime_s": boot_std[i][(g, s)]
                    for s in SMOKE_ORDER
                    for g in GUIDANCE_ORDER
                },
                **{
                    f"{g}|{s}|StandardizedCorrect": boot_corr[i][(g, s)]
                    for s in SMOKE_ORDER
                    for g in GUIDANCE_ORDER
                },
            }
            for i in range(len(boot_std))
        ]
    )


def descriptive_all_smoke(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for guidance in GUIDANCE_ORDER:
        part = frame.loc[frame["Guidance"].astype(str).eq(guidance)]
        lo, hi = wilson(int(part["ExitCorrect"].sum()), len(part))
        rows.append(
            {
                "Guidance": guidance,
                "GuidanceCN": GUIDANCE_CN[guidance],
                "N": len(part),
                "ParticipantN": part["ParticipantID"].nunique(),
                "MeanRawTime_s": part["EvacuationTime_s"].mean(),
                "MeanStandardizedTime_s": part["StandardizedTime_s"].mean(),
                "MedianStandardizedTime_s": part["StandardizedTime_s"].median(),
                "ObservedExitCorrectRate": part["ExitCorrect"].mean(),
                "ObservedCorrectWilsonLow": lo,
                "ObservedCorrectWilsonHigh": hi,
                "MeanDetourRatio": part["DetourRatio"].mean(),
                "MeanBacktrackingRatio": part["BacktrackingRatio"].mean(),
                "MeanRotationRate_deg_s": part["RotationRate_deg_s"].mean(),
            }
        )
    return pd.DataFrame(rows)


def joint_outcome_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe time conditional on correctness without treating it as causal."""
    rows = []
    for (guidance, smoke), part in frame.groupby(["Guidance", "Smoke"], observed=True):
        correct = part.loc[part["ExitCorrect"].eq(1), "StandardizedTime_s"]
        wrong = part.loc[part["ExitCorrect"].eq(0), "StandardizedTime_s"]
        rows.append(
            {
                "Guidance": str(guidance),
                "Smoke": smoke,
                "TrialN": len(part),
                "CorrectTrialN": len(correct),
                "WrongTrialN": len(wrong),
                "AllTrialMeanStandardizedTime_s": part["StandardizedTime_s"].mean(),
                "CorrectOnlyMeanStandardizedTime_s": correct.mean(),
                "WrongOnlyMeanStandardizedTime_s": wrong.mean(),
                "Interpretation": (
                    "descriptive_sensitivity_only_conditioning_on_post_treatment_"
                    "correctness_can_induce_selection_bias"
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, "joint_time_correctness_sensitivity.csv")
    return out


def process_curve_summary(
    movement: pd.DataFrame,
    visual: pd.DataFrame,
    aoi: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def summarize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        rows = []
        for metric in metrics:
            for keys, part in frame.groupby(
                ["Guidance", "Smoke", "ProgressBin", "ProgressMid"], observed=True
            ):
                values = part[metric].dropna().to_numpy(float)
                if not len(values):
                    continue
                boots = [
                    RNG.choice(values, len(values), replace=True).mean()
                    for _ in range(PROCESS_BOOT_N)
                ]
                rows.append(
                    {
                        "Guidance": keys[0],
                        "Smoke": keys[1],
                        "ProgressBin": keys[2],
                        "ProgressMid": keys[3],
                        "Metric": metric,
                        "Mean": values.mean(),
                        "CI95Low": np.quantile(boots, 0.025),
                        "CI95High": np.quantile(boots, 0.975),
                        "ParticipantN": part["ParticipantID"].nunique(),
                    }
                )
        return pd.DataFrame(rows)

    move = summarize(
        movement, ["MeanSpeed_m_s", "MovementTurnPathRate_deg_s", "BacktrackIncrement_m"]
    )
    # Visual and AOI phases use normalized trial time so wrong-exit trials are
    # not selectively removed from late target-route bins.
    visual2 = visual.rename(
        columns={"TrialProgressBin": "ProgressBin", "TrialProgressMid": "ProgressMid"}
    )
    aoi2 = aoi.rename(
        columns={"TrialProgressBin": "ProgressBin", "TrialProgressMid": "ProgressMid"}
    )
    visual_out = summarize(
        visual2,
        [
            "HeadAngularPathRate_deg_s",
            "FixationTimeProportion",
            "SaccadeRate_per_min",
            "EyeGazeScanpathRate_deg_s",
            "GazeP90Dispersion_deg",
        ],
    )
    aoi_out = summarize(aoi2, ["GuideDwellProportion"])
    save_csv(move, "movement_progress_curve_summary.csv")
    save_csv(visual_out, "visual_information_progress_curve_summary.csv")
    save_csv(aoi_out, "ar_information_progress_curve_summary.csv")
    return move, visual_out, aoi_out


def earliest_process_difference(
    movement: pd.DataFrame, visual: pd.DataFrame, aoi: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    visual2 = visual.rename(columns={"TrialProgressBin": "ProgressBin"})
    aoi2 = aoi.rename(columns={"TrialProgressBin": "ProgressBin"})
    sources = [
        (movement, "MeanSpeed_m_s", GUIDANCE_ORDER, "spatial_behavior", "target_route_progress"),
        (movement, "MovementTurnPathRate_deg_s", GUIDANCE_ORDER, "spatial_behavior", "normalized_trial_time"),
        (movement, "BacktrackIncrement_m", GUIDANCE_ORDER, "spatial_behavior", "target_route_progress"),
        (visual2, "HeadAngularPathRate_deg_s", GUIDANCE_ORDER, "visual_search", "normalized_trial_time"),
        (visual2, "FixationTimeProportion", GUIDANCE_ORDER, "visual_search", "normalized_trial_time"),
        (visual2, "SaccadeRate_per_min", GUIDANCE_ORDER, "visual_search", "normalized_trial_time"),
        (visual2, "EyeGazeScanpathRate_deg_s", GUIDANCE_ORDER, "visual_search", "normalized_trial_time"),
        (visual2, "GazeP90Dispersion_deg", GUIDANCE_ORDER, "visual_search", "normalized_trial_time"),
        (aoi2, "GuideDwellProportion", ["ArrowRibbon", "MiniMap", "TextBar", "Full"], "AR_information", "normalized_trial_time"),
    ]
    for source, metric, modes, layer, axis in sources:
        for smoke in SMOKE_ORDER:
            for reference, comparison in combinations(modes, 2):
                bin_rows = []
                for bin_id in range(1, 11):
                    part = source.loc[
                        source["Smoke"].eq(smoke)
                        & source["ProgressBin"].eq(bin_id)
                        & source["Guidance"].astype(str).isin([reference, comparison])
                    ]
                    xa = part.loc[
                        part["Guidance"].astype(str).eq(reference), metric
                    ].dropna().to_numpy(float)
                    xb = part.loc[
                        part["Guidance"].astype(str).eq(comparison), metric
                    ].dropna().to_numpy(float)
                    if len(xa) < 2 or len(xb) < 2:
                        continue
                    est = float(xb.mean() - xa.mean())
                    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
                    se = math.sqrt(va + vb)
                    den = va * va / max(len(xa) - 1, 1) + vb * vb / max(len(xb) - 1, 1)
                    df = (va + vb) ** 2 / den if den > 0 else np.inf
                    crit = t.ppf(0.975, df) if np.isfinite(df) else 1.96
                    lo, hi = est - crit * se, est + crit * se
                    bin_rows.append((bin_id, est, lo, hi, lo > 0 or hi < 0))
                sustained = np.nan
                for i in range(len(bin_rows) - 1):
                    if (
                        bin_rows[i][4]
                        and bin_rows[i + 1][4]
                        and bin_rows[i + 1][0] == bin_rows[i][0] + 1
                    ):
                        sustained = bin_rows[i][0]
                        break
                rows.append(
                    {
                        "Smoke": smoke,
                        "Reference": reference,
                        "Comparison": comparison,
                        "Metric": metric,
                        "Layer": layer,
                        "ProgressAxis": axis,
                        "EarliestTwoConsecutiveSeparatedBin": sustained,
                        "EarliestProgressPercent": (
                            (sustained - 1) * 10 if np.isfinite(sustained) else np.nan
                        ),
                        "Rule": "two_consecutive_pointwise_Welch_CIs_exclude_zero",
                        "Status": "exploratory_GAMM_like_screen_not_formal_GAMM",
                    }
                )
    out = pd.DataFrame(rows)
    save_csv(out, "earliest_sustained_process_difference.csv")
    return out


def process_outcome_associations(
    frame: pd.DataFrame, visual: pd.DataFrame, aoi: pd.DataFrame
) -> pd.DataFrame:
    common_mediators = [
        "HeadAngularPathRate_deg_s",
        "FixationTimeProportion",
        "SaccadeRate_per_min",
        "EyeGazeScanpathRate_deg_s",
        "GazeP90Dispersion_deg",
        "GazeDirectionEntropy",
        "BacktrackingRatio",
        "DetourRatio",
        "RotationRate_deg_s",
    ]
    ar_mediators = [
        "GuideAcquired",
        "PromptGuideAcquisitionScore",
        "GuideDwellProportion",
        "GuideVisitRate_per_min",
    ]
    rows = []
    x_common = frame.merge(visual[KEYS + [c for c in common_mediators if c in visual]], on=KEYS, how="left", validate="one_to_one")
    x_ar = x_common.merge(aoi[KEYS + ar_mediators], on=KEYS, how="inner", validate="one_to_one")
    for layer, x, mediators in [
        ("common_visual_and_spatial", x_common, common_mediators),
        ("AR_information", x_ar, ar_mediators),
    ]:
      for mediator in mediators:
        model_data = x.dropna(subset=[mediator]).copy()
        zcol = f"Z_{mediator}"
        sd = model_data[mediator].std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            continue
        model_data[zcol] = (model_data[mediator] - model_data[mediator].mean()) / sd
        for outcome in ["StandardizedTime_s", "ExitCorrect"]:
            y = f"np.log(StandardizedTime_s)" if outcome == "StandardizedTime_s" else outcome
            model = ols(
                f"{y} ~ {zcol} + C(Guidance)*C(Smoke) + "
                "SpatialDifficultyRatio",
                data=model_data,
            ).fit(cov_type="cluster", cov_kwds={"groups": model_data["ParticipantID"]})
            coef = model.params[zcol]
            lo, hi = model.conf_int().loc[zcol]
            if outcome == "StandardizedTime_s":
                coef, lo, hi = (np.exp(v) - 1 for v in (coef, lo, hi))
                scale = "proportional_time_change_per_1SD"
            else:
                coef, lo, hi = coef * 100, lo * 100, hi * 100
                scale = "correctness_percentage_points_per_1SD"
            rows.append(
                {
                    "MediatorCandidate": mediator,
                    "Layer": layer,
                    "N": len(model_data),
                    "Outcome": outcome,
                    "Estimate": coef,
                    "CI95Low": lo,
                    "CI95High": hi,
                    "PValue": model.pvalues[zcol],
                    "Scale": scale,
                    "Model": "cluster_robust_OLS_or_LPM_adjusted_guidance_smoke_geometry",
                }
            )
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    for (_, outcome), idx in out.groupby(["Layer", "Outcome"]).groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValue"], method="fdr_bh")[1]
    save_csv(out, "adjusted_process_outcome_associations.csv")
    return out


def grouped_ebm(
    trials: pd.DataFrame, visual: pd.DataFrame, aoi: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visual_features = [
        "HeadAngularPathRate_deg_s",
        "FixationTimeProportion",
        "SaccadeRate_per_min",
        "EyeGazeScanpathRate_deg_s",
        "GazeP90Dispersion_deg",
    ]
    aoi_features = [
        "GuideAcquired",
        "PromptGuideAcquisitionScore",
        "GuideDwellProportion",
        "GuideVisitRate_per_min",
    ]
    merged = trials.merge(
        visual[KEYS + visual_features], on=KEYS, how="left", validate="one_to_one"
    )
    merged_ar = merged.merge(
        aoi[KEYS + aoi_features], on=KEYS, how="inner", validate="one_to_one"
    )
    base_features = ["Guidance", "Smoke", "SpatialDifficultyRatio"]
    common_visual = base_features + visual_features
    integrated = common_visual + ["DetourRatio", "BacktrackingRatio"]
    specs = [
        ("pretreatment", trials.copy(), base_features),
        ("visual_information_all_modes", merged.copy(), common_visual),
        ("visual_spatial_process_all_modes", merged.copy(), integrated),
        ("AR_visual_spatial_without_AOI", merged_ar.copy(), integrated),
        ("AR_information_integrated", merged_ar.copy(), integrated + aoi_features),
    ]
    perf_rows, importance_rows = [], []
    for stage, frame, features in specs:
        frame = frame.dropna(subset=["Guidance", "Smoke"]).copy()
        X = frame[features].copy()
        for c in ["Guidance", "Smoke"]:
            if c in X:
                X[c] = X[c].astype(str)
        types = [
            "nominal" if c in {"Guidance", "Smoke"} else "continuous"
            for c in features
        ]
        groups = frame["ParticipantID"].to_numpy()
        splitter = GroupKFold(n_splits=5)
        for outcome in ["StandardizedTime_s", "ExitCorrect"]:
            pred = np.full(len(frame), np.nan)
            for train, test in splitter.split(X, frame[outcome], groups):
                common = dict(
                    feature_types=types,
                    interactions=0,
                    outer_bags=8,
                    max_rounds=2500,
                    learning_rate=0.03,
                    min_samples_leaf=4,
                    random_state=SEED,
                    n_jobs=1,
                )
                if outcome == "StandardizedTime_s":
                    model = ExplainableBoostingRegressor(**common)
                    model.fit(X.iloc[train], frame[outcome].iloc[train])
                    pred[test] = model.predict(X.iloc[test])
                else:
                    model = ExplainableBoostingClassifier(**common)
                    model.fit(X.iloc[train], frame[outcome].iloc[train])
                    pred[test] = model.predict_proba(X.iloc[test])[:, 1]
            if outcome == "StandardizedTime_s":
                perf_rows.append(
                    {
                        "Stage": stage,
                        "Outcome": outcome,
                        "N": len(frame),
                        "ParticipantN": frame["ParticipantID"].nunique(),
                        "MAE": mean_absolute_error(frame[outcome], pred),
                        "R2": r2_score(frame[outcome], pred),
                        "ROC_AUC": np.nan,
                        "Brier": np.nan,
                    }
                )
                full = ExplainableBoostingRegressor(
                    feature_types=types,
                    interactions=0,
                    outer_bags=8,
                    max_rounds=2500,
                    learning_rate=0.03,
                    min_samples_leaf=4,
                    random_state=SEED,
                    n_jobs=1,
                )
            else:
                perf_rows.append(
                    {
                        "Stage": stage,
                        "Outcome": outcome,
                        "N": len(frame),
                        "ParticipantN": frame["ParticipantID"].nunique(),
                        "MAE": np.nan,
                        "R2": np.nan,
                        "ROC_AUC": roc_auc_score(frame[outcome], pred),
                        "Brier": brier_score_loss(frame[outcome], pred),
                    }
                )
                full = ExplainableBoostingClassifier(
                    feature_types=types,
                    interactions=0,
                    outer_bags=8,
                    max_rounds=2500,
                    learning_rate=0.03,
                    min_samples_leaf=4,
                    random_state=SEED,
                    n_jobs=1,
                )
            full.fit(X, frame[outcome])
            for feature, value in zip(full.term_names_, full.term_importances()):
                importance_rows.append(
                    {
                        "Stage": stage,
                        "Outcome": outcome,
                        "Feature": feature,
                        "Importance": float(value),
                    }
                )
    perf = pd.DataFrame(perf_rows)
    importance = pd.DataFrame(importance_rows)
    save_csv(perf, "ebm_grouped_cv_performance.csv")
    save_csv(importance, "ebm_feature_importance.csv")
    return perf, importance


def linear_predict(beta: np.ndarray, a, m, logdiff) -> np.ndarray:
    a, m, logdiff = np.broadcast_arrays(a, m, logdiff)
    X = np.column_stack(
        [
            np.ones(a.size),
            a.ravel(),
            m.ravel(),
            logdiff.ravel(),
        ]
    )
    return X @ beta


def mediation_one_fit(
    data: pd.DataFrame,
    geometry: pd.DataFrame,
    reference: str,
    comparison: str,
    mediator: str,
    outcome: str,
) -> np.ndarray:
    x = data.copy()
    a = x["Guidance"].astype(str).eq(comparison).astype(float).to_numpy()
    logdiff = np.log(x["SpatialDifficultyRatio"].to_numpy(float))
    m = x[mediator].to_numpy(float)
    XM = np.column_stack([np.ones(len(x)), a, logdiff])
    bm = np.linalg.lstsq(XM, m, rcond=None)[0]
    y = (
        np.log(x["StandardizedTime_s"].clip(lower=0.1).to_numpy(float))
        if outcome == "StandardizedTime_s"
        else x["ExitCorrect"].to_numpy(float)
    )
    XY = np.column_stack([np.ones(len(x)), a, m, logdiff])
    by = np.linalg.lstsq(XY, y, rcond=None)[0]

    sides = x["StartSide"].astype(str).unique()
    if len(sides) != 1:
        raise RuntimeError("Pairwise smoke stratum should have one spawn side.")
    grid = geometry.loc[
        geometry["Exit"].isin(TARGET_EXITS) & geometry["StartSide"].eq(sides[0])
    ]
    gd = np.log(grid["SpatialDifficultyRatio"].to_numpy(float))
    m0 = np.column_stack([np.ones(len(grid)), np.zeros(len(grid)), gd]) @ bm
    m1 = np.column_stack([np.ones(len(grid)), np.ones(len(grid)), gd]) @ bm
    y00 = linear_predict(by, 0, m0, gd).mean()
    y10 = linear_predict(by, 1, m0, gd).mean()
    y11 = linear_predict(by, 1, m1, gd).mean()
    total, direct, indirect = y11 - y00, y10 - y00, y11 - y10
    if outcome == "StandardizedTime_s":
        return (np.exp([total, direct, indirect]) - 1) * 100
    return np.array([total, direct, indirect]) * 100


def gcomp_mediation(
    trials: pd.DataFrame,
    geometry: pd.DataFrame,
    visual: pd.DataFrame,
    aoi: pd.DataFrame,
) -> pd.DataFrame:
    common = trials.merge(
        visual[
            KEYS
            + [
                "HeadAngularPathRate_deg_s",
                "FixationTimeProportion",
                "SaccadeRate_per_min",
                "EyeGazeScanpathRate_deg_s",
                "GazeP90Dispersion_deg",
            ]
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    ar = common.merge(
        aoi[
            KEYS
            + [
                "PromptGuideAcquisitionScore",
                "GuideDwellProportion",
                "GuideVisitRate_per_min",
            ]
        ],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    specs = [
        (
            "common_visual",
            "all_five_modes",
            common,
            GUIDANCE_ORDER,
            [
                "HeadAngularPathRate_deg_s",
                "FixationTimeProportion",
                "SaccadeRate_per_min",
                "EyeGazeScanpathRate_deg_s",
                "GazeP90Dispersion_deg",
            ],
        ),
        (
            "spatial_behavior",
            "all_five_modes",
            common,
            GUIDANCE_ORDER,
            ["BacktrackingRatio", "DetourRatio", "RotationRate_deg_s"],
        ),
        (
            "AR_information",
            "AR_modes_only",
            ar,
            ["ArrowRibbon", "MiniMap", "TextBar", "Full"],
            [
                "PromptGuideAcquisitionScore",
                "GuideDwellProportion",
                "GuideVisitRate_per_min",
            ],
        ),
    ]
    rows = []
    for layer, scope, source, modes, mediators in specs:
      for smoke in SMOKE_ORDER:
        for reference, comparison in combinations(modes, 2):
            sub0 = source.loc[
                source["Smoke"].eq(smoke)
                & source["Guidance"].astype(str).isin([reference, comparison])
            ].copy()
            for mediator in mediators:
              sub = sub0.dropna(subset=[mediator]).copy()
              ids = sub["ParticipantID"].unique()
              if sub["Guidance"].astype(str).nunique() < 2:
                continue
              id_positions = {
                  pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid)
                  for pid in ids
              }
              for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                    estimate = mediation_one_fit(
                        sub, geometry, reference, comparison, mediator, outcome
                    )
                    boots = []
                    for _ in range(MED_BOOT_N):
                        sampled = RNG.choice(ids, len(ids), replace=True)
                        # Position-based sampling is equivalent to participant
                        # cluster bootstrap but avoids thousands of expensive
                        # DataFrame filter/concat operations.
                        take = np.concatenate([id_positions[pid] for pid in sampled])
                        boot = sub.iloc[take]
                        if boot["Guidance"].astype(str).nunique() < 2:
                            continue
                        try:
                            value = mediation_one_fit(
                                boot,
                                geometry,
                                reference,
                                comparison,
                                mediator,
                                outcome,
                            )
                            if np.isfinite(value).all():
                                boots.append(value)
                        except Exception:
                            continue
                    arr = np.asarray(boots)
                    for idx, effect in enumerate(["Total", "Direct", "Indirect"]):
                        vals = arr[:, idx] if len(arr) else np.array([])
                        rows.append(
                            {
                                "Smoke": smoke,
                                "Layer": layer,
                                "Scope": scope,
                                "Reference": reference,
                                "Comparison": comparison,
                                "Mediator": mediator,
                                "Outcome": outcome,
                                "Effect": effect,
                                "Estimate": estimate[idx],
                                "CI95Low": np.quantile(vals, 0.025) if len(vals) else np.nan,
                                "CI95High": np.quantile(vals, 0.975) if len(vals) else np.nan,
                                "PValueBootstrap": boot_p(vals),
                                "Scale": (
                                    "percent_standardized_time_change"
                                    if outcome == "StandardizedTime_s"
                                    else "correctness_percentage_points"
                                ),
                                "BootstrapSuccessN": len(arr),
                                "CausalStatus": (
                                    "exploratory_randomized_guidance_but_mediator_"
                                    "outcome_exchangeability_unverifiable"
                                ),
                            }
                        )
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    indirect = out["Effect"].eq("Indirect")
    for (_, outcome), idx in out.loc[indirect].groupby(["Layer", "Outcome"]).groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(
            out.loc[idx, "PValueBootstrap"], method="fdr_bh"
        )[1]
    save_csv(out, "gcomputation_multimediator_pairwise.csv")
    # Backward-compatible filename now contains the all-pair backtracking subset.
    save_csv(
        out.loc[out["Mediator"].eq("BacktrackingRatio")],
        "gcomputation_backtracking_mediation.csv",
    )
    return out


def sensitivity_tables(
    main: pd.DataFrame,
    eligible: pd.DataFrame,
    geometry: pd.DataFrame,
    main_summary: pd.DataFrame,
) -> pd.DataFrame:
    eligible_summary, _, _ = bootstrap_outcomes(eligible, geometry, n_boot=200)
    save_csv(eligible_summary, "sensitivity_all_288_outcome_summary.csv")
    rows = []
    for cohort_name, frame in [("Primary45", main), ("All48Sensitivity", eligible)]:
        for smoke in SMOKE_ORDER:
            for guidance in GUIDANCE_ORDER:
                part = frame.loc[
                    frame["Smoke"].eq(smoke)
                    & frame["Guidance"].astype(str).eq(guidance)
                ]
                rows.extend(
                    [
                        {
                            "Cohort": cohort_name,
                            "Smoke": smoke,
                            "Guidance": guidance,
                            "Method": "RawTime",
                            "MeanTime_s": part["EvacuationTime_s"].mean(),
                        },
                        {
                            "Cohort": cohort_name,
                            "Smoke": smoke,
                            "Guidance": guidance,
                            "Method": "FixedSpawn_AStarDistance",
                            "MeanTime_s": part["StandardizedTime_s"].mean(),
                        },
                        {
                            "Cohort": cohort_name,
                            "Smoke": smoke,
                            "Guidance": guidance,
                            "Method": "ExistingTrialStartMetric",
                            "MeanTime_s": part["ExistingMetricStandardizedTime_s"].mean(),
                        },
                    ]
                )
    out = pd.DataFrame(rows)
    out["RankWithinSmokeMethod"] = out.groupby(
        ["Cohort", "Smoke", "Method"]
    )["MeanTime_s"].rank(method="min")
    save_csv(out, "spatial_standardization_sensitivity_and_ranking.csv")
    return out


def correctness_model_sensitivity(
    frame: pd.DataFrame, geometry: pd.DataFrame
) -> pd.DataFrame:
    """Expose dependence of binary-outcome standardization on model assumptions."""

    def matrix(x: pd.DataFrame, specification: str) -> np.ndarray:
        cols = [
            x["Guidance"].astype(str).eq(g).astype(float).to_numpy()
            for g in GUIDANCE_ORDER[1:]
        ]
        cols.append(x["Smoke"].astype(str).eq("HighSmoke").astype(float).to_numpy())
        if specification != "GuidanceSmokeOnly":
            cols.append(np.log(x["SpatialDifficultyRatio"].astype(float)).to_numpy())
        return np.column_stack(cols)

    rows = []
    specs = [
        ("GuidanceSmokeOnly", 10.0),
        ("AdditiveDistance_C0.1", 0.1),
        ("AdditiveDistance_C1", 1.0),
        ("AdditiveDistance_C10_primary", 10.0),
        ("AdditiveDistance_C100", 100.0),
    ]
    for spec, c_value in specs:
        model = LogisticRegression(
            C=c_value, solver="lbfgs", max_iter=5000, random_state=SEED
        ).fit(matrix(frame, spec), frame["ExitCorrect"].astype(int))
        for smoke in SMOKE_ORDER:
            side = frame.loc[frame["Smoke"].eq(smoke), "StartSide"].astype(str).unique()[0]
            grid0 = geometry.loc[
                geometry["Exit"].isin(TARGET_EXITS)
                & geometry["StartSide"].eq(side)
            ].copy()
            for guidance in GUIDANCE_ORDER:
                grid = grid0.assign(Guidance=guidance, Smoke=smoke)
                rows.append(
                    {
                        "Specification": spec,
                        "C": c_value,
                        "Smoke": smoke,
                        "Guidance": guidance,
                        "StandardizedCorrectProbability": model.predict_proba(
                            matrix(grid, spec)
                        )[:, 1].mean(),
                    }
                )
    out = pd.DataFrame(rows)
    save_csv(out, "correctness_standardization_model_sensitivity.csv")
    return out


def visual_quality_and_correlation_sensitivity(
    visual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = pd.read_csv(TABLE / "eye_tracking_data_quality_audit.csv")
    quality = quality.loc[~quality["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    merged = visual.merge(
        quality[KEYS + ["ValidEyeSampleProportion", "ValidDurationCoverage"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    metrics = [
        "HeadAngularPathRate_deg_s",
        "FixationTimeProportion",
        "SaccadeRate_per_min",
        "EyeGazeScanpathRate_deg_s",
        "GazeP90Dispersion_deg",
        "GazeDirectionEntropy",
    ]
    rows = []
    for cohort, source in [
        ("PrimaryAll270", merged),
        ("CoverageAtLeast80pct", merged.loc[merged["ValidDurationCoverage"].ge(0.80)]),
    ]:
        for (guidance, smoke), part in source.groupby(["Guidance", "Smoke"], observed=True):
            for metric in metrics:
                rows.append(
                    {
                        "Cohort": cohort,
                        "Guidance": guidance,
                        "Smoke": smoke,
                        "Metric": metric,
                        "TrialN": len(part),
                        "Mean": part[metric].mean(),
                    }
                )
    coverage = pd.DataFrame(rows)
    save_csv(coverage, "visual_quality_coverage_sensitivity.csv")
    corr = merged[metrics].corr(method="spearman")
    corr_rows = []
    for a, b in combinations(metrics, 2):
        corr_rows.append(
            {
                "FeatureA": a,
                "FeatureB": b,
                "SpearmanRho": corr.loc[a, b],
                "HighRedundancyAbsRhoAtLeast0.80": abs(corr.loc[a, b]) >= 0.80,
            }
        )
    corr_out = pd.DataFrame(corr_rows).sort_values("SpearmanRho", key=lambda x: x.abs(), ascending=False)
    save_csv(corr_out, "visual_feature_correlation_audit.csv")
    return coverage, corr_out


def make_figures(
    trials: pd.DataFrame,
    geometry: pd.DataFrame,
    summary: pd.DataFrame,
    pairs: pd.DataFrame,
    move_curve: pd.DataFrame,
    visual_features: pd.DataFrame,
    aoi_features: pd.DataFrame,
    visual_curve: pd.DataFrame,
    aoi_curve: pd.DataFrame,
    ebm_imp: pd.DataFrame,
    mediation: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    # 01. Geometry audit.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for side, marker in [("Left", "o"), ("Right", "s")]:
        part = geometry.loc[geometry["StartSide"].eq(side)]
        ax.plot(
            part["Exit"],
            part["AStarShortestDistance_m"],
            marker=marker,
            lw=1.5,
            label=f"{'左' if side == 'Left' else '右'}出生点",
        )
    for x in ["G", "H"]:
        ax.axvspan(ord(x) - ord("A") - 0.35, ord(x) - ord("A") + 0.35, color="#E5E7EB", alpha=0.5)
    ax.set_ylabel("A*最短距离（m）")
    ax.set_xlabel("出口（G、H未作为正式目标）")
    ax.set_title("出口空间难度审计：按两个实际出生点计算")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, "01_exit_spatial_difficulty")

    # 02. Raw and standardized time, with all participant points visible.
    long = trials.melt(
        id_vars=["Guidance", "Smoke", "ParticipantID"],
        value_vars=["EvacuationTime_s", "StandardizedTime_s"],
        var_name="Metric",
        value_name="Time_s",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for col, smoke in enumerate(SMOKE_ORDER):
        for row, metric in enumerate(["EvacuationTime_s", "StandardizedTime_s"]):
            ax = axes[row, col]
            part = long.loc[long["Smoke"].eq(smoke) & long["Metric"].eq(metric)]
            sns.stripplot(
                data=part,
                x="Guidance",
                y="Time_s",
                order=GUIDANCE_ORDER,
                palette=PALETTE,
                hue="Guidance",
                hue_order=GUIDANCE_ORDER,
                jitter=0.18,
                alpha=0.55,
                size=4,
                legend=False,
                ax=ax,
            )
            sns.pointplot(
                data=part,
                x="Guidance",
                y="Time_s",
                order=GUIDANCE_ORDER,
                color="black",
                errorbar=("ci", 95),
                markers="D",
                linestyle="none",
                ax=ax,
            )
            ax.set_xticks(np.arange(len(GUIDANCE_ORDER)))
            ax.set_xticklabels([GUIDANCE_CN[g] for g in GUIDANCE_ORDER], rotation=20)
            ax.set_xlabel("")
            ax.set_ylabel("疏散时间（s）" if col == 0 else "")
            title = "原始时间" if metric == "EvacuationTime_s" else "空间难度标准化时间"
            ax.set_title(f"{SMOKE_CN[smoke]}：{title}")
    fig.suptitle("标准化前后比较（每个点为一次试次）", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "02_raw_vs_spatially_standardized_time")

    # 03. Primary standardized outcomes.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    offsets = np.linspace(-0.18, 0.18, len(GUIDANCE_ORDER))
    for ax, outcome in zip(
        axes, ["StandardizedMeanTime_s", "SpatiallyStandardizedCorrectProbability"]
    ):
        for i, guidance in enumerate(GUIDANCE_ORDER):
            part = summary.loc[summary["Guidance"].eq(guidance)].set_index("Smoke").loc[SMOKE_ORDER]
            x = np.arange(2) + offsets[i]
            if outcome == "StandardizedMeanTime_s":
                y = part[outcome].to_numpy()
                lo = part["StandardizedTimeCI95Low"].to_numpy()
                hi = part["StandardizedTimeCI95High"].to_numpy()
                ylabel = "标准化疏散时间（s）"
            else:
                y = part[outcome].to_numpy() * 100
                lo = part["StandardizedCorrectCI95Low"].to_numpy() * 100
                hi = part["StandardizedCorrectCI95High"].to_numpy() * 100
                ylabel = "标准化出口正确概率（%）"
            ax.errorbar(
                x,
                y,
                yerr=[y - lo, hi - y],
                fmt="o",
                capsize=3,
                color=PALETTE[guidance],
                label=GUIDANCE_CN[guidance],
            )
        ax.set_xticks([0, 1], [SMOKE_CN[s] for s in SMOKE_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    axes[0].set_title("效率：越低越好")
    axes[1].set_title("目标出口正确性：越高越好")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.03, 1), loc="upper left")
    fig.suptitle("五种引导在同一参考空间分布下的结果", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(fig, "03_primary_standardized_outcomes")

    # 04. Pairwise time differences.
    p = pairs.loc[pairs["Outcome"].eq("StandardizedTime_s")].copy()
    p["Pair"] = p["Comparison"].map(GUIDANCE_CN) + " − " + p["Reference"].map(GUIDANCE_CN)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True)
    for ax, smoke in zip(axes, SMOKE_ORDER):
        part = p.loc[p["Smoke"].eq(smoke)].reset_index(drop=True)
        y = np.arange(len(part))
        colors = np.where(part["PValueFDR"].lt(0.05), "#D55E00", "#6B7280")
        ax.axvline(0, color="black", lw=1)
        for i, row in part.iterrows():
            ax.plot([row.CI95Low, row.CI95High], [i, i], color=colors[i], lw=2)
            ax.scatter(row.Estimate, i, color=colors[i], s=35)
        ax.set_yticks(y, part["Pair"])
        ax.invert_yaxis()
        ax.set_xlabel("比较组 − 参照组（标准化秒；负值更快）")
        ax.set_title(SMOKE_CN[smoke])
    fig.suptitle("五种引导两两比较：空间标准化疏散时间", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "04_pairwise_standardized_time")

    # 05. Five-mode common visual information metrics. Raw trial points are
    # retained because the single-guidance cells contain only 15 participants.
    visual_plot = visual_features.copy()
    visual_metrics = [
        ("HeadAngularPathRate_deg_s", "头部搜索路径率（°/s）"),
        ("FixationTimeProportion", "凝视时间比例"),
        ("SaccadeRate_per_min", "眼跳事件率（次/min）"),
        ("GazeP90Dispersion_deg", "注视分散程度（°）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (metric, ylabel) in zip(axes.flat, visual_metrics):
        sns.stripplot(
            data=visual_plot, x="Guidance", y=metric, hue="Smoke",
            order=GUIDANCE_ORDER, hue_order=SMOKE_ORDER, dodge=True,
            jitter=0.16, alpha=0.38, size=3.5,
            palette=["#56B4E9", "#E69F00"], ax=ax,
        )
        sns.pointplot(
            data=visual_plot, x="Guidance", y=metric, hue="Smoke",
            order=GUIDANCE_ORDER, hue_order=SMOKE_ORDER, dodge=0.42,
            errorbar=("ci", 95), markers=["o", "s"], linestyles="none",
            palette=["#0072B2", "#D55E00"], ax=ax,
        )
        ax.set_xticks(np.arange(len(GUIDANCE_ORDER)))
        ax.set_xticklabels([GUIDANCE_CN[g] for g in GUIDANCE_ORDER], rotation=18)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[0, 1]:
            ax.legend(handles[-2:], [SMOKE_CN[s] for s in SMOKE_ORDER], frameon=False)
        else:
            ax.get_legend().remove()
    fig.suptitle("五种引导的视觉搜索模式（点为试次，黑边点为均值±95% CI）", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "05_visual_information_by_guidance")

    # 06. When common visual processes diverge over normalized trial time.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, metric in enumerate(["HeadAngularPathRate_deg_s", "EyeGazeScanpathRate_deg_s"]):
        for col, smoke in enumerate(SMOKE_ORDER):
            ax = axes[row, col]
            part = visual_curve.loc[visual_curve["Metric"].eq(metric) & visual_curve["Smoke"].eq(smoke)]
            for guidance in GUIDANCE_ORDER:
                q = part.loc[part["Guidance"].astype(str).eq(guidance)]
                ax.plot(q["ProgressMid"] * 100, q["Mean"], color=PALETTE[guidance], label=GUIDANCE_CN[guidance])
                ax.fill_between(q["ProgressMid"] * 100, q["CI95Low"], q["CI95High"], color=PALETTE[guidance], alpha=0.09)
            ax.set_title(SMOKE_CN[smoke])
            ax.set_xlabel("试次时间进度（%）")
            ax.set_ylabel("头部路径率（°/s）" if row == 0 else "眼内扫视路径率（°/s）")
    axes[0, 1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("动态视觉过程：不同引导何时开始分化（阴影为95% bootstrap CI）", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "06_visual_information_progress")

    # 07. AR guidance information acquisition (StaticSign intentionally absent).
    ar_metrics = [
        ("GuideAcquired", "获取引导的试次比例"),
        ("PromptGuideAcquisitionScore", "及时获取得分"),
        ("GuideDwellProportion", "引导停留比例"),
        ("GuideVisitRate_per_min", "引导访问率（次/min）"),
    ]
    ar_order = ["ArrowRibbon", "MiniMap", "TextBar", "Full"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, ylabel) in zip(axes.flat, ar_metrics):
        sns.stripplot(
            data=aoi_features, x="Guidance", y=metric, hue="Smoke",
            order=ar_order, hue_order=SMOKE_ORDER, dodge=True, jitter=0.16,
            alpha=0.38, size=3.5, palette=["#56B4E9", "#E69F00"], ax=ax,
        )
        sns.pointplot(
            data=aoi_features, x="Guidance", y=metric, hue="Smoke",
            order=ar_order, hue_order=SMOKE_ORDER, dodge=0.42,
            errorbar=("ci", 95), markers=["o", "s"], linestyles="none",
            palette=["#0072B2", "#D55E00"], ax=ax,
        )
        ax.set_xticks(np.arange(len(ar_order)))
        ax.set_xticklabels([GUIDANCE_CN[g] for g in ar_order], rotation=15)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        handles, labels = ax.get_legend_handles_labels()
        if ax is axes[0, 1]:
            ax.legend(handles[-2:], [SMOKE_CN[s] for s in SMOKE_ORDER], frameon=False)
        else:
            ax.get_legend().remove()
    fig.suptitle("四种AR模式的信息获取（StaticSign无重构AOI，不进入本图）", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "07_ar_information_acquisition")

    # 08. EBM integrated-process importance.
    imp = ebm_imp.loc[
        ebm_imp["Stage"].eq("visual_spatial_process_all_modes")
        & ebm_imp["Outcome"].isin(["StandardizedTime_s", "ExitCorrect"])
    ].copy()
    imp["FeatureCN"] = imp["Feature"].map(FEATURE_CN).fillna(imp["Feature"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
        part = imp.loc[imp["Outcome"].eq(outcome)].nlargest(10, "Importance").sort_values("Importance")
        ax.barh(part["FeatureCN"], part["Importance"], color="#0072B2" if outcome == "StandardizedTime_s" else "#009E73")
        ax.set_xlabel("EBM全局重要性（预测贡献，不等于因果效应）")
        ax.set_title("空间标准化时间" if outcome == "StandardizedTime_s" else "出口正确性")
    fig.suptitle("EBM：信息获取、视觉搜索与空间行为谁更能解释结果", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(fig, "08_ebm_layered_importance")

    # 09. Layered mediation overview for four decision-relevant contrasts.
    med = mediation.loc[
        mediation["Effect"].eq("Indirect")
        & mediation["Reference"].isin(["StaticSign", "TextBar"])
        & mediation["Comparison"].isin(["ArrowRibbon", "Full"])
    ].copy()
    med["Contrast"] = med["Comparison"].map(GUIDANCE_CN) + " − " + med["Reference"].map(GUIDANCE_CN) + " / " + med["Smoke"].map(SMOKE_CN)
    med["MediatorCN"] = med["Mediator"].map(FEATURE_CN).fillna(med["Mediator"])
    selected = [
        "HeadAngularPathRate_deg_s", "FixationTimeProportion", "SaccadeRate_per_min",
        "EyeGazeScanpathRate_deg_s", "GazeP90Dispersion_deg", "BacktrackingRatio", "DetourRatio",
    ]
    med = med.loc[med["Mediator"].isin(selected)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
        pivot = med.loc[med["Outcome"].eq(outcome)].pivot_table(index="MediatorCN", columns="Contrast", values="Estimate")
        sig = med.loc[med["Outcome"].eq(outcome)].assign(Significant=lambda x: (x.CI95Low > 0) | (x.CI95High < 0)).pivot_table(index="MediatorCN", columns="Contrast", values="Significant", aggfunc="max")
        pivot = pivot.reindex([FEATURE_CN.get(x, x) for x in selected])
        lim = np.nanmax(np.abs(pivot.to_numpy()))
        sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-lim, vmax=lim, ax=ax, cbar_kws={"label": "间接效应"})
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                if pd.notna(sig.reindex_like(pivot).iloc[i, j]) and bool(sig.reindex_like(pivot).iloc[i, j]):
                    ax.text(j + 0.5, i + 0.5, "●", ha="center", va="center", color="black", fontsize=6)
        ax.set_title("标准化时间（%）" if outcome == "StandardizedTime_s" else "出口正确性（百分点）")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=55)
    fig.suptitle("探索性中介：引导差异经哪些内部过程传递（●=单项95% CI不跨0）", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "09_gcomp_layered_mediation")

    # 10. Spatial behavior progress remains a separate downstream layer.
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, metric in enumerate(["MeanSpeed_m_s", "BacktrackIncrement_m"]):
        for col, smoke in enumerate(SMOKE_ORDER):
            ax = axes[row, col]
            part = move_curve.loc[move_curve["Metric"].eq(metric) & move_curve["Smoke"].eq(smoke)]
            for guidance in GUIDANCE_ORDER:
                q = part.loc[part["Guidance"].astype(str).eq(guidance)]
                ax.plot(q["ProgressMid"] * 100, q["Mean"], color=PALETTE[guidance], label=GUIDANCE_CN[guidance])
                ax.fill_between(q["ProgressMid"] * 100, q["CI95Low"], q["CI95High"], color=PALETTE[guidance], alpha=0.09)
            ax.set_title(SMOKE_CN[smoke])
            ax.set_xlabel("目标路线进度（%）")
            ax.set_ylabel("平均速度（m/s）" if row == 0 else "该阶段回退距离（m）")
    axes[0, 1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("下游空间行为：速度与回退在目标路线上的分化", y=1.01, fontsize=14)
    fig.tight_layout()
    savefig(fig, "10_movement_progress_curves")

    # 11. Standardization robustness.
    sens = sensitivity.loc[sensitivity["Cohort"].eq("Primary45")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    style = {
        "RawTime": ("原始时间", "o", "-"),
        "FixedSpawn_AStarDistance": ("固定出生点A*标准化", "s", "--"),
        "ExistingTrialStartMetric": ("既有试次起点A*标准化", "^", ":"),
    }
    for ax, smoke in zip(axes, SMOKE_ORDER):
        part = sens.loc[sens["Smoke"].eq(smoke)]
        for method, (label, marker, ls) in style.items():
            q = part.loc[part["Method"].eq(method)].set_index("Guidance").loc[GUIDANCE_ORDER]
            ax.plot(
                [GUIDANCE_CN[g] for g in GUIDANCE_ORDER],
                q["MeanTime_s"],
                marker=marker,
                ls=ls,
                label=label,
            )
        ax.set_title(SMOKE_CN[smoke])
        ax.set_ylabel("平均疏散时间（s）")
        ax.tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("敏感性分析：不同空间折算定义是否改变结论排序", y=1.02, fontsize=14)
    fig.tight_layout()
    savefig(fig, "11_spatial_standardization_sensitivity")


def main() -> None:
    print("1/9 Loading source trials and A* graph")
    audit, eligible, main_raw = load_trials()
    nodes, edges, exits, node_ids, tree, graph = load_graph()

    print("2/9 Building pre-treatment spatial difficulty")
    eligible_aug, geometry = build_geometry(
        eligible, nodes, exits, node_ids, tree, graph
    )
    main = eligible_aug.loc[~eligible_aug["AOICohortExcluded"]].copy()
    save_data(eligible_aug, "analysis_trials_all_288.csv")
    save_data(main, "analysis_trials.csv")
    save_csv(
        main[
            [
                "Scenario",
                "ParticipantID",
                "Guidance",
                "Smoke",
                "TargetExit",
                "StartSide",
                "SpawnNode",
                "AStarShortestDistance_m",
                "ReferenceDistance_AF_EqualWeight_m",
                "SpatialDifficultyRatio",
                "StandardizationMultiplier",
                "EvacuationTime_s",
                "StandardizedTime_s",
                "ExitCorrect",
            ]
        ],
        "spatial_standardization_trial_audit.csv",
    )

    print("3/9 Extracting visual-search and AR-information features")
    from Extract_Visual_Information_Features import main as extract_visual_information
    from Analyze_Information_Contrasts import main as analyze_information_contrasts

    extract_visual_information()
    analyze_information_contrasts()

    print("4/9 Bootstrapping standardized outcomes and pairwise contrasts")
    outcome_summary, pairs, boot = bootstrap_outcomes(main, geometry)
    save_csv(outcome_summary, "five_mode_spatial_standardized_summary.csv")
    save_csv(pairs, "five_mode_pairwise_spatial_standardized_effects.csv")
    save_data(boot, "primary_outcome_bootstrap_draws.csv")
    save_csv(descriptive_all_smoke(main), "five_mode_descriptive_summary.csv")
    joint_outcome_sensitivity(main)

    print("5/9 Loading process data for the 45-participant cohort")
    movement = pd.read_csv(DATA / "movement_progress_bins.csv")
    movement = movement.loc[~movement["ParticipantID"].isin(AOI_EXCLUDED)].copy()
    visual_features = pd.read_csv(DATA / "visual_information_trial_features.csv")
    visual_progress = pd.read_csv(DATA / "visual_information_progress_bins.csv")
    aoi_features = pd.read_csv(DATA / "ar_information_acquisition_features.csv")
    aoi_progress = pd.read_csv(DATA / "ar_information_acquisition_progress_bins.csv")
    save_data(movement, "movement_progress_bins_complete_case.csv")
    save_csv(
        aoi_features.groupby(["Guidance", "Smoke"], as_index=False).agg(
            TrialN=("Scenario", "size"),
            ValidAOITrialN=("AOIAvailable", "sum"),
            NeverGazeTrialN=("GuideAcquired", lambda x: int(pd.to_numeric(x, errors="coerce").eq(0).sum())),
            MeanGuideDwellProportion=("GuideDwellProportion", "mean"),
            MeanGuideVisitRate_per_min=("GuideVisitRate_per_min", "mean"),
            MeanFirstGuideLatency_s=("FirstGuideLatency_s", "mean"),
            MeanPromptGuideAcquisitionScore=("PromptGuideAcquisitionScore", "mean"),
        ),
        "guidance_attention_complete_case_summary.csv",
    )
    save_csv(
        aoi_features.loc[
            pd.to_numeric(aoi_features["AOIAvailable"], errors="coerce").eq(1)
            & pd.to_numeric(aoi_features["GuideAcquired"], errors="coerce").eq(0)
        ],
        "valid_aoi_trials_never_gazed.csv",
    )
    move_curve, visual_curve, aoi_curve = process_curve_summary(
        movement, visual_progress, aoi_progress
    )
    earliest_process_difference(movement, visual_progress, aoi_progress)
    process_outcome_associations(main, visual_features, aoi_features)

    print("6/9 Fitting grouped-CV EBM models")
    ebm_perf, ebm_imp = grouped_ebm(main, visual_features, aoi_features)

    print("7/9 Running pairwise layered g-computation")
    mediation = gcomp_mediation(main, geometry, visual_features, aoi_features)

    print("8/9 Running sensitivity analyses")
    sensitivity = sensitivity_tables(main, eligible_aug, geometry, outcome_summary)
    correctness_model_sensitivity(main, geometry)
    visual_quality_and_correlation_sensitivity(visual_features)
    save_csv(
        mediation.loc[
            mediation["Effect"].eq("Indirect")
            & ((mediation["CI95Low"] > 0) | (mediation["CI95High"] < 0))
        ].sort_values(["Layer", "Outcome", "Smoke", "Reference", "Comparison", "Mediator"]),
        "layered_mediation_indirect_ci_excludes_zero.csv",
    )

    print("9/9 Rendering figures and integrity summary")
    make_figures(
        main,
        geometry,
        outcome_summary,
        pairs,
        move_curve,
        visual_features,
        aoi_features,
        visual_curve,
        aoi_curve,
        ebm_imp,
        mediation,
        sensitivity,
    )
    aoi_unavailable = int(aoi_features["AOIAvailable"].fillna(0).le(0).sum())
    never_gaze = int(
        (
            aoi_features["AOIAvailable"].fillna(0).gt(0)
            & aoi_features["GuideAcquired"].fillna(0).eq(0)
        ).sum()
    )
    summary = {
        "generated_at": "2026-08-02",
        "seed": SEED,
        "primary_trial_n": int(len(main)),
        "primary_participant_n": int(main["ParticipantID"].nunique()),
        "sensitivity_trial_n": int(len(eligible_aug)),
        "sensitivity_participant_n": int(eligible_aug["ParticipantID"].nunique()),
        "excluded_aoi_participants": sorted(AOI_EXCLUDED),
        "excluded_trial_n": int(eligible_aug["AOICohortExcluded"].sum()),
        "primary_visual_trial_n": int(len(visual_features)),
        "primary_ar_aoi_trial_n": int(len(aoi_features)),
        "primary_aoi_unavailable_trial_n": aoi_unavailable,
        "primary_valid_aoi_never_gaze_trial_n": never_gaze,
        "raw_eye_trial_matched_n": 288,
        "median_valid_eye_sample_proportion": 0.9887280946044049,
        "median_eye_sample_interval_ms": 22.0,
        "ivt_primary_velocity_threshold_deg_s": 30.0,
        "ivt_primary_fixation_min_duration_ms": 100.0,
        "visual_phase_axis": "normalized_trial_time_10_bins",
        "movement_trial_n": int(
            movement[["Scenario", "ParticipantID"]].drop_duplicates().shape[0]
        ),
        "spawn_nodes": sorted(main["SpawnNode"].unique().astype(int).tolist()),
        "target_exits_in_primary_outcomes": TARGET_EXITS,
        "geometry_audit_exits": sorted(exits["exit_id"].astype(str).tolist()),
        "astar_node_n": int(len(nodes)),
        "astar_edge_row_n": int(len(edges)),
        "astar_connected_components": int(nx.number_connected_components(graph)),
        "standardization": (
            "trial_time * equal_weight_mean_A-F_distance_for_spawn_side / "
            "Astar_distance_from_actual_spawn_side_to_target"
        ),
        "correctness_standardization": (
            "additive_ridge_logistic_g_computation_over_equal_A-F_within_"
            "the_smoke-specific_observed_spawn_side"
        ),
        "main_identification_limit": (
            "Target exit remains structurally tied to guidance-smoke cells. "
            "Standardization removes measured shortest-distance difficulty only; "
            "unmeasured visibility, gate asymmetry, and semantic differences remain."
        ),
    }
    (DATA / "analysis_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
