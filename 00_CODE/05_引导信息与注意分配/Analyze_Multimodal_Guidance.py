"""Analyze the key eye, HRV, AOI and spatial-process factors.

This module deliberately keeps only theory-distinct primary factors in EBM,
dynamic tests and mediation. More redundant eye metrics remain in extraction
audit files but no longer compete in the primary models.
"""

from __future__ import annotations

import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
FIG = OUT / "figures"
NOTE = OUT / "notes"
for directory in (DATA, TABLE, FIG, NOTE):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 20260803
RNG = np.random.default_rng(SEED)
BOOT_N = 200
GUIDANCE_ORDER = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]
KEYS = ["Scenario", "ParticipantID", "FileName"]
PALETTE = {
    "StaticSign": "#6B7280",
    "ArrowRibbon": "#0072B2",
    "MiniMap": "#009E73",
    "TextBar": "#D55E00",
    "Full": "#CC79A7",
}

KEY_VISUAL = ["HeadAngularPathRate_deg_s", "FixationTimeProportion", "GazeDirectionEntropy"]
KEY_HRV = ["MeanHRBaselineDelta_bpm", "LogRMSSDBaselineDelta"]
KEY_SPATIAL = ["DetourRatio", "BacktrackingRatio"]
KEY_AOI = ["PromptGuideAcquisitionScore", "GuideDwellProportion"]

FEATURE_LABEL = {
    "Guidance": "Guidance mode",
    "Smoke": "Smoke",
    "SpatialDifficultyRatio": "A* spatial difficulty",
    "HeadAngularPathRate_deg_s": "Head search path rate",
    "FixationTimeProportion": "Fixation-time proportion",
    "GazeDirectionEntropy": "Gaze-direction entropy",
    "MeanHRBaselineDelta_bpm": "Heart-rate change from resting baseline",
    "LogRMSSDBaselineDelta": "log(RMSSD) change from resting baseline",
    "DetourRatio": "Detour ratio",
    "BacktrackingRatio": "Backtracking ratio",
    "PromptGuideAcquisitionScore": "Prompt guide acquisition",
    "GuideDwellProportion": "Guide dwell proportion",
    "EarlyInformationBurden": "Early visual-search burden",
    "MidAutonomicLoad": "Mid-trial autonomic load",
    "LateSpatialInefficiency": "Late spatial inefficiency",
}

mpl.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLE / name, index=False, encoding="utf-8-sig")


def save_data(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(DATA / name, index=False, encoding="utf-8-sig")


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = numeric.std(ddof=0)
    return (numeric - numeric.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=series.index)


def row_mean_with_min(frame: pd.DataFrame, columns: list[str], minimum: int = 2) -> pd.Series:
    return frame[columns].mean(axis=1, skipna=True).where(frame[columns].notna().sum(axis=1) >= minimum)


def load_and_build_features() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    trials = pd.read_csv(DATA / "analysis_trials.csv")
    trials["StartParsed"] = pd.to_datetime(trials["StartTime"], format="%Y_%m_%d_%H_%M_%S_%f", errors="coerce")
    trials["TrialOrder"] = trials.groupby("ParticipantID")["StartParsed"].rank(method="first").astype(int)
    visual = pd.read_csv(DATA / "visual_information_trial_features.csv")
    visual_bins = pd.read_csv(DATA / "visual_information_progress_bins.csv").rename(
        columns={"TrialProgressBin": "ProgressBin", "TrialProgressMid": "ProgressMid"}
    )
    hrv = pd.read_csv(DATA / "hrv_trial_features.csv")
    hrv_bins = pd.read_csv(DATA / "hrv_progress_bins.csv")
    movement = pd.read_csv(DATA / "movement_progress_bins.csv")
    aoi = pd.read_csv(DATA / "ar_information_acquisition_features.csv")
    aoi_bins = pd.read_csv(DATA / "ar_information_acquisition_progress_bins.csv").rename(
        columns={"TrialProgressBin": "ProgressBin", "TrialProgressMid": "ProgressMid"}
    )

    merged = trials.merge(visual[KEYS + KEY_VISUAL], on=KEYS, how="left", validate="one_to_one")
    merged = merged.merge(
        hrv[KEYS + KEY_HRV + ["MeanHR_bpm", "RMSSD_ms", "SDNN_ms", "HRVTimeCoverage", "HRVPrimaryUsable"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    merged = merged.merge(aoi[KEYS + KEY_AOI], on=KEYS, how="left", validate="one_to_one")

    early = visual_bins.loc[visual_bins["ProgressBin"].between(1, 3)].groupby(KEYS, as_index=False)[KEY_VISUAL].mean()
    early = early.rename(columns={c: f"Early_{c}" for c in KEY_VISUAL})
    mid = hrv_bins.loc[hrv_bins["ProgressBin"].between(4, 6)].groupby(KEYS, as_index=False)[KEY_HRV].mean()
    mid = mid.rename(columns={c: f"Mid_{c}" for c in KEY_HRV})
    # Stop at 90% so the spatial mediator precedes the final exit outcome.
    late_source = movement.loc[movement["ProgressBin"].between(7, 9)].copy()
    late = late_source.groupby(KEYS, as_index=False).agg(
        Late_MovementTurnPathRate_deg_s=("MovementTurnPathRate_deg_s", "mean"),
        Late_BacktrackIncrement_m=("BacktrackIncrement_m", "sum"),
    )
    merged = merged.merge(early, on=KEYS, how="left", validate="one_to_one")
    merged = merged.merge(mid, on=KEYS, how="left", validate="one_to_one")
    merged = merged.merge(late, on=KEYS, how="left", validate="one_to_one")

    info_parts = []
    for source, sign in [
        ("Early_HeadAngularPathRate_deg_s", 1),
        ("Early_FixationTimeProportion", -1),
        ("Early_GazeDirectionEntropy", 1),
    ]:
        col = f"Z_{source}"
        merged[col] = zscore(merged[source]) * sign
        info_parts.append(col)
    hrv_parts = []
    for source, sign in [
        ("Mid_MeanHRBaselineDelta_bpm", 1),
        ("Mid_LogRMSSDBaselineDelta", -1),
    ]:
        col = f"Z_{source}"
        merged[col] = zscore(merged[source]) * sign
        hrv_parts.append(col)
    space_parts = []
    for source in [
        "Late_MovementTurnPathRate_deg_s",
        "Late_BacktrackIncrement_m",
    ]:
        col = f"Z_{source}"
        merged[col] = zscore(merged[source])
        space_parts.append(col)
    merged["EarlyInformationBurden"] = row_mean_with_min(merged, info_parts, 2)
    merged["MidAutonomicLoad"] = row_mean_with_min(merged, hrv_parts, 2)
    merged["LateSpatialInefficiency"] = row_mean_with_min(merged, space_parts, 2)
    save_data(merged, "multimodal_trial_features.csv")
    order_audit = merged.groupby(["Guidance", "Smoke"], as_index=False).agg(
        TrialN=("Scenario", "size"),
        ParticipantN=("ParticipantID", "nunique"),
        MinimumTrialOrder=("TrialOrder", "min"),
        MaximumTrialOrder=("TrialOrder", "max"),
        MeanTrialOrder=("TrialOrder", "mean"),
    )
    order_audit["IdentificationStatus"] = np.where(
        order_audit["Guidance"].isin(["ArrowRibbon", "MiniMap", "TextBar"]),
        "same_order_randomized_single_guidance_arm",
        "guidance_structurally_tied_to_trial_order",
    )
    save_csv(order_audit, "trial_order_design_audit.csv")
    return merged, {
        "visual": visual,
        "visual_bins": visual_bins,
        "hrv": hrv,
        "hrv_bins": hrv_bins,
        "movement": movement,
        "aoi": aoi,
        "aoi_bins": aoi_bins,
    }


def curve_summary(source: pd.DataFrame, metrics: list[str], layer: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for (guidance, smoke, bin_id, mid), part in source.groupby(
            ["Guidance", "Smoke", "ProgressBin", "ProgressMid"], observed=True
        ):
            part = part.dropna(subset=[metric])
            ids = part["ParticipantID"].unique()
            if len(part) == 0:
                continue
            values = part[metric].to_numpy(float)
            participant_values = part.groupby("ParticipantID", observed=True)[metric].mean().to_numpy(float)
            boots = RNG.choice(participant_values, (BOOT_N, len(participant_values)), replace=True).mean(axis=1)
            rows.append({
                "Layer": layer,
                "Metric": metric,
                "Guidance": guidance,
                "Smoke": smoke,
                "ProgressBin": int(bin_id),
                "ProgressMid": float(mid),
                "TrialN": int(len(part)),
                "ParticipantN": int(len(ids)),
                "Mean": float(values.mean()),
                "CI95Low": float(np.quantile(boots, 0.025)),
                "CI95High": float(np.quantile(boots, 0.975)),
            })
    return pd.DataFrame(rows)


def dynamic_analysis(merged: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve = pd.concat([
        curve_summary(sources["visual_bins"], KEY_VISUAL, "visual_information"),
        curve_summary(sources["hrv_bins"], KEY_HRV, "autonomic_load"),
        curve_summary(sources["movement"], ["MovementTurnPathRate_deg_s", "BacktrackIncrement_m"], "spatial_behavior"),
        curve_summary(sources["aoi_bins"], ["GuideDwellProportion"], "AR_information"),
    ], ignore_index=True)
    save_csv(curve, "normalized_time_curve_summary.csv")

    source_specs = [
        (sources["visual_bins"], KEY_VISUAL, "visual_information", GUIDANCE_ORDER),
        (sources["hrv_bins"], KEY_HRV, "autonomic_load", GUIDANCE_ORDER),
        (sources["movement"], ["MovementTurnPathRate_deg_s", "BacktrackIncrement_m"], "spatial_behavior", GUIDANCE_ORDER),
        (sources["aoi_bins"], ["GuideDwellProportion"], "AR_information", ["ArrowRibbon", "MiniMap", "TextBar", "Full"]),
    ]
    earliest_rows = []
    for source, metrics, layer, modes in source_specs:
        for metric in metrics:
            for smoke in SMOKE_ORDER:
                for reference, comparison in combinations(modes, 2):
                    bin_results = []
                    for bin_id in range(1, 11):
                        part = source.loc[
                            source["Smoke"].eq(smoke)
                            & source["ProgressBin"].eq(bin_id)
                            & source["Guidance"].astype(str).isin([reference, comparison])
                        ].dropna(subset=[metric])
                        ids = part["ParticipantID"].unique()
                        if len(ids) < 3 or part["Guidance"].nunique() < 2:
                            continue
                        pivot = part.pivot_table(index="ParticipantID", columns="Guidance", values=metric, aggfunc="mean").reindex(columns=[reference, comparison])
                        est = float(pivot[comparison].mean() - pivot[reference].mean())
                        boots = []
                        arr = pivot.to_numpy(float)
                        for _ in range(BOOT_N):
                            sampled = arr[RNG.integers(0, len(arr), len(arr))]
                            val = float(np.nanmean(sampled[:, 1]) - np.nanmean(sampled[:, 0]))
                            if np.isfinite(val):
                                boots.append(val)
                        lo, hi = (np.quantile(boots, [0.025, 0.975]) if len(boots) else [np.nan, np.nan])
                        bin_results.append((bin_id, est, lo, hi, bool(lo > 0 or hi < 0)))
                    sustained = np.nan
                    for i in range(len(bin_results) - 1):
                        if bin_results[i][4] and bin_results[i + 1][4] and bin_results[i + 1][0] == bin_results[i][0] + 1:
                            sustained = bin_results[i][0]
                            break
                    earliest_rows.append({
                        "Layer": layer,
                        "Metric": metric,
                        "Smoke": smoke,
                        "Reference": reference,
                        "Comparison": comparison,
                        "EarliestTwoConsecutiveSeparatedBin": sustained,
                        "EarliestProgressPercent": (sustained - 1) * 10 if np.isfinite(sustained) else np.nan,
                        "Axis": "normalized_trial_time_0_100_percent",
                        "Rule": "two consecutive participant-cluster-bootstrap 95% CIs exclude zero",
                        "ComparisonDesignStatus": (
                            "same_trial_order_randomized_single_guidance_comparison"
                            if {reference, comparison}.issubset({"ArrowRibbon", "MiniMap", "TextBar"})
                            else "guidance_confounding_with_trial_order_cannot_be_excluded"
                        ),
                    })
    earliest = pd.DataFrame(earliest_rows)
    save_csv(earliest, "earliest_sustained_process_difference.csv")

    spline_rows = []
    trial_controls = merged[KEYS + ["SpatialDifficultyRatio"]]
    for source, metrics, layer, modes in source_specs:
        frame0 = source.merge(trial_controls, on=KEYS, how="left", validate="many_to_one")
        scopes = [("all_available_modes_descriptive", modes)]
        same_order = [m for m in modes if m in {"ArrowRibbon", "MiniMap", "TextBar"}]
        if len(same_order) >= 2:
            scopes.append(("randomized_single_modes_same_trial_order", same_order))
        for scope, scope_modes in scopes:
          frame = frame0.loc[frame0["Guidance"].astype(str).isin(scope_modes)].copy()
          for metric in metrics:
            for smoke in SMOKE_ORDER:
                x = frame.loc[frame["Smoke"].eq(smoke)].dropna(subset=[metric, "ProgressMid", "SpatialDifficultyRatio"]).copy()
                if len(x) < 100 or x["ParticipantID"].nunique() < 10:
                    continue
                safe = "MetricValue"
                x[safe] = x[metric]
                full_formula = f"{safe} ~ C(Guidance) * bs(ProgressMid, df=4, degree=3) + np.log(SpatialDifficultyRatio)"
                try:
                    # Participant fixed intercepts and cluster-robust covariance
                    # prevent the ten bins per trial from creating pseudo-N.
                    full_formula += " + C(ParticipantID)"
                    full = ols(full_formula, x).fit(cov_type="cluster", cov_kwds={"groups": x["ParticipantID"]})
                    names = list(full.params.index)
                    term_idx = [i for i, name in enumerate(names) if "C(Guidance)" in name and ":bs(ProgressMid" in name]
                    restriction = np.zeros((len(term_idx), len(names)))
                    for r, idx in enumerate(term_idx):
                        restriction[r, idx] = 1.0
                    test = full.wald_test(restriction, scalar=True)
                    lr = float(test.statistic)
                    df = len(term_idx)
                    p = float(test.pvalue)
                    status = "ok"
                except Exception as exc:
                    lr, df, p, status = np.nan, np.nan, np.nan, f"failed:{type(exc).__name__}"
                spline_rows.append({
                    "Scope": scope,
                    "Layer": layer,
                    "Metric": metric,
                    "Smoke": smoke,
                    "NRows": len(x),
                    "ParticipantN": x["ParticipantID"].nunique(),
                    "GuidanceBySmoothTime_Wald": lr,
                    "DFDifference": df,
                    "PValue": p,
                    "Status": status,
                    "Model": "cubic_spline_participant_fixed_intercept_cluster_robust",
                })
    spline = pd.DataFrame(spline_rows)
    spline["PValueFDR"] = np.nan
    valid = spline["PValue"].notna()
    if valid.any():
        spline.loc[valid, "PValueFDR"] = multipletests(spline.loc[valid, "PValue"], method="fdr_bh")[1]
    save_csv(spline, "normalized_time_mixed_spline_tests.csv")
    return curve, earliest, spline


def grouped_ebm(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ["Guidance", "Smoke", "SpatialDifficultyRatio"]
    hrv_complete = merged.dropna(subset=KEY_HRV).copy()
    ar = hrv_complete.loc[hrv_complete["Guidance"].ne("StaticSign")].copy()
    specs = [
        ("pretreatment_all_trials", merged, base),
        ("pretreatment_hrv_subset", hrv_complete, base),
        ("key_visual_hrv_subset", hrv_complete, base + KEY_VISUAL),
        ("key_visual_plus_hrv", hrv_complete, base + KEY_VISUAL + KEY_HRV),
        ("all_key_processes", hrv_complete, base + KEY_VISUAL + KEY_HRV + KEY_SPATIAL),
        ("AR_key_processes_without_AOI", ar, base + KEY_VISUAL + KEY_HRV + KEY_SPATIAL),
        ("AR_key_processes_with_AOI", ar, base + KEY_VISUAL + KEY_HRV + KEY_SPATIAL + KEY_AOI),
    ]
    perf_rows, importance_rows = [], []
    for stage, source, features in specs:
        frame = source.dropna(subset=features + ["StandardizedTime_s", "ExitCorrect"]).copy()
        if frame["ParticipantID"].nunique() < 5:
            continue
        X = frame[features].copy()
        for col in ["Guidance", "Smoke"]:
            X[col] = X[col].astype(str)
        types = ["nominal" if c in {"Guidance", "Smoke"} else "continuous" for c in features]
        groups = frame["ParticipantID"].to_numpy()
        n_splits = min(5, frame["ParticipantID"].nunique())
        splitter = GroupKFold(n_splits=n_splits)
        for outcome in ["StandardizedTime_s", "ExitCorrect"]:
            pred = np.full(len(frame), np.nan)
            common = dict(
                feature_types=types,
                interactions=0,
                outer_bags=8,
                max_rounds=1800,
                learning_rate=0.03,
                min_samples_leaf=4,
                random_state=SEED,
                n_jobs=1,
            )
            for train, test in splitter.split(X, frame[outcome], groups):
                model = ExplainableBoostingRegressor(**common) if outcome == "StandardizedTime_s" else ExplainableBoostingClassifier(**common)
                model.fit(X.iloc[train], frame[outcome].iloc[train])
                pred[test] = model.predict(X.iloc[test]) if outcome == "StandardizedTime_s" else model.predict_proba(X.iloc[test])[:, 1]
            if outcome == "StandardizedTime_s":
                perf_rows.append({
                    "Stage": stage, "Outcome": outcome, "N": len(frame), "ParticipantN": frame["ParticipantID"].nunique(),
                    "MAE": mean_absolute_error(frame[outcome], pred), "R2": r2_score(frame[outcome], pred), "ROC_AUC": np.nan, "Brier": np.nan,
                })
                full = ExplainableBoostingRegressor(**common)
            else:
                perf_rows.append({
                    "Stage": stage, "Outcome": outcome, "N": len(frame), "ParticipantN": frame["ParticipantID"].nunique(),
                    "MAE": np.nan, "R2": np.nan, "ROC_AUC": roc_auc_score(frame[outcome], pred), "Brier": brier_score_loss(frame[outcome], pred),
                })
                full = ExplainableBoostingClassifier(**common)
            full.fit(X, frame[outcome])
            for feature, value in zip(full.term_names_, full.term_importances()):
                importance_rows.append({"Stage": stage, "Outcome": outcome, "Feature": feature, "Importance": float(value)})
    perf = pd.DataFrame(perf_rows)
    importance = pd.DataFrame(importance_rows)
    save_csv(perf, "ebm_grouped_cv_performance.csv")
    save_csv(importance, "ebm_feature_importance.csv")
    return perf, importance


def key_factor_summaries_and_associations(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors_by_layer = {
        "visual_information": KEY_VISUAL,
        "autonomic_load": KEY_HRV,
        "spatial_behavior": KEY_SPATIAL,
        "AR_information": KEY_AOI,
        "temporal_layer_composites": ["EarlyInformationBurden", "MidAutonomicLoad", "LateSpatialInefficiency"],
    }
    summary_rows = []
    for layer, factors in factors_by_layer.items():
        for factor in factors:
            for (guidance, smoke), part in merged.groupby(["Guidance", "Smoke"], observed=True):
                values = pd.to_numeric(part[factor], errors="coerce").dropna()
                summary_rows.append({
                    "Layer": layer,
                    "Factor": factor,
                    "Guidance": guidance,
                    "Smoke": smoke,
                    "TrialN": len(part),
                    "AvailableN": len(values),
                    "Mean": values.mean(),
                    "SD": values.std(ddof=1),
                    "Median": values.median(),
                })
    summary = pd.DataFrame(summary_rows)
    save_csv(summary, "key_multimodal_factor_summary.csv")

    association_rows = []
    for layer, factors in factors_by_layer.items():
        for factor in factors:
            data = merged.dropna(subset=[factor, "SpatialDifficultyRatio", "StandardizedTime_s", "ExitCorrect"]).copy()
            data["ZFactor"] = zscore(data[factor])
            for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                lhs = "np.log(StandardizedTime_s)" if outcome == "StandardizedTime_s" else "ExitCorrect"
                model = ols(
                    f"{lhs} ~ ZFactor + C(Guidance)*C(Smoke) + np.log(SpatialDifficultyRatio)",
                    data=data,
                ).fit(cov_type="cluster", cov_kwds={"groups": data["ParticipantID"]})
                estimate = float(model.params["ZFactor"])
                lo, hi = model.conf_int().loc["ZFactor"].astype(float)
                if outcome == "StandardizedTime_s":
                    estimate, lo, hi = ((np.exp(v) - 1) * 100 for v in (estimate, lo, hi))
                    scale = "percent_standardized_time_change_per_1SD"
                else:
                    estimate, lo, hi = (v * 100 for v in (estimate, lo, hi))
                    scale = "correctness_percentage_points_per_1SD"
                association_rows.append({
                    "Layer": layer,
                    "Factor": factor,
                    "Outcome": outcome,
                    "N": len(data),
                    "ParticipantN": data["ParticipantID"].nunique(),
                    "Estimate": estimate,
                    "CI95Low": lo,
                    "CI95High": hi,
                    "PValue": model.pvalues["ZFactor"],
                    "Scale": scale,
                    "Model": "one_factor_at_a_time_cluster_robust_adjusted_guidance_smoke_Astar",
                })
    associations = pd.DataFrame(association_rows)
    associations["PValueFDR"] = np.nan
    for (layer, outcome), idx in associations.groupby(["Layer", "Outcome"]).groups.items():
        associations.loc[idx, "PValueFDR"] = multipletests(associations.loc[idx, "PValue"], method="fdr_bh")[1]
    save_csv(associations, "adjusted_process_outcome_associations.csv")

    contrast_rows = []
    for smoke in SMOKE_ORDER:
        for reference, comparison in combinations(GUIDANCE_ORDER, 2):
            for factor in KEY_HRV:
                part = merged.loc[
                    merged["Smoke"].eq(smoke)
                    & merged["Guidance"].astype(str).isin([reference, comparison])
                ].dropna(subset=[factor, "SpatialDifficultyRatio"]).copy()
                if part["Guidance"].nunique() < 2:
                    continue
                part["A"] = part["Guidance"].astype(str).eq(comparison).astype(float)
                model = ols(f"{factor} ~ A + np.log(SpatialDifficultyRatio)", data=part).fit(
                    cov_type="cluster", cov_kwds={"groups": part["ParticipantID"]}
                )
                lo, hi = model.conf_int().loc["A"].astype(float)
                contrast_rows.append({
                    "Smoke": smoke,
                    "Reference": reference,
                    "Comparison": comparison,
                    "Factor": factor,
                    "Estimate": model.params["A"],
                    "CI95Low": lo,
                    "CI95High": hi,
                    "PValue": model.pvalues["A"],
                    "N": len(part),
                    "ParticipantN": part["ParticipantID"].nunique(),
                    "ComparisonDesignStatus": (
                        "same_trial_order_randomized_single_guidance_comparison"
                        if {reference, comparison}.issubset({"ArrowRibbon", "MiniMap", "TextBar"})
                        else "guidance_confounding_with_trial_order_cannot_be_excluded"
                    ),
                })
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["PValueFDR"] = multipletests(contrasts["PValue"].fillna(1), method="fdr_bh")[1]
    save_csv(contrasts, "hrv_pairwise_adjusted_contrasts.csv")
    return summary, associations, contrasts


def boot_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(2 * min((values <= 0).mean(), (values >= 0).mean()))


def scale_effect(value: float, outcome: str) -> float:
    return float((np.exp(value) - 1) * 100) if outcome == "StandardizedTime_s" else float(value * 100)


def parallel_fit(data: pd.DataFrame, mediators: list[str], outcome: str, comparison: str) -> dict[str, float]:
    x = data.copy()
    a = x["Guidance"].astype(str).eq(comparison).astype(float).to_numpy()
    c = np.log(x["SpatialDifficultyRatio"].to_numpy(float))
    m = x[mediators].to_numpy(float)
    y = np.log(x["StandardizedTime_s"].clip(lower=0.1).to_numpy(float)) if outcome == "StandardizedTime_s" else x["ExitCorrect"].to_numpy(float)
    total_beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), a, c]), y, rcond=None)[0]
    a_paths = []
    for j in range(len(mediators)):
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), a, c]), m[:, j], rcond=None)[0]
        a_paths.append(beta[1])
    y_beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), a, m, c]), y, rcond=None)[0]
    specific = np.asarray(a_paths) * y_beta[2:2 + len(mediators)]
    result = {"Total": total_beta[1], "Direct": y_beta[1], "JointIndirect": float(specific.sum())}
    result.update({f"Specific:{name}": value for name, value in zip(mediators, specific)})
    return result


def parallel_mediation(merged: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("key_visual_parallel", KEY_VISUAL, GUIDANCE_ORDER),
        ("HRV_parallel", KEY_HRV, GUIDANCE_ORDER),
        ("spatial_parallel", KEY_SPATIAL, GUIDANCE_ORDER),
        ("three_layer_parallel", ["EarlyInformationBurden", "MidAutonomicLoad", "LateSpatialInefficiency"], GUIDANCE_ORDER),
        ("AR_information_parallel", KEY_AOI, ["ArrowRibbon", "MiniMap", "TextBar", "Full"]),
    ]
    rows = []
    for model_name, mediators, modes in specs:
        for smoke in SMOKE_ORDER:
            for reference, comparison in combinations(modes, 2):
                sub = merged.loc[
                    merged["Smoke"].eq(smoke)
                    & merged["Guidance"].astype(str).isin([reference, comparison])
                ].dropna(subset=mediators + ["SpatialDifficultyRatio", "StandardizedTime_s", "ExitCorrect"]).copy()
                if sub["Guidance"].nunique() < 2:
                    continue
                for mediator in mediators:
                    sub[mediator] = zscore(sub[mediator])
                ids = sub["ParticipantID"].unique()
                if len(ids) < 8:
                    continue
                id_rows = {pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid) for pid in ids}
                for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                    estimate = parallel_fit(sub, mediators, outcome, comparison)
                    boot_values = {key: [] for key in estimate}
                    for _ in range(BOOT_N):
                        sampled = RNG.choice(ids, len(ids), replace=True)
                        take = np.concatenate([id_rows[pid] for pid in sampled])
                        boot = sub.iloc[take]
                        if boot["Guidance"].nunique() < 2:
                            continue
                        try:
                            values = parallel_fit(boot, mediators, outcome, comparison)
                            for key, value in values.items():
                                if np.isfinite(value):
                                    boot_values[key].append(value)
                        except Exception:
                            continue
                    for effect, raw in estimate.items():
                        vals = np.asarray(boot_values[effect], float)
                        scaled_vals = np.asarray([scale_effect(v, outcome) for v in vals])
                        rows.append({
                            "Model": model_name,
                            "Smoke": smoke,
                            "Reference": reference,
                            "Comparison": comparison,
                            "Outcome": outcome,
                            "Effect": effect,
                            "Estimate": scale_effect(raw, outcome),
                            "CI95Low": float(np.quantile(scaled_vals, 0.025)) if len(scaled_vals) else np.nan,
                            "CI95High": float(np.quantile(scaled_vals, 0.975)) if len(scaled_vals) else np.nan,
                            "PValueBootstrap": boot_p(vals),
                            "N": len(sub),
                            "ParticipantN": len(ids),
                            "Scale": "percent_standardized_time_change" if outcome == "StandardizedTime_s" else "correctness_percentage_points",
                            "Status": "exploratory_multiple_parallel_mediation_cluster_bootstrap",
                            "ComparisonDesignStatus": (
                                "same_trial_order_randomized_single_guidance_comparison"
                                if {reference, comparison}.issubset({"ArrowRibbon", "MiniMap", "TextBar"})
                                else "guidance_confounding_with_trial_order_cannot_be_excluded"
                            ),
                        })
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    indirect = out["Effect"].str.contains("Indirect|Specific", regex=True)
    for (model, outcome), idx in out.loc[indirect].groupby(["Model", "Outcome"]).groups.items():
        p = out.loc[idx, "PValueBootstrap"].fillna(1.0)
        out.loc[idx, "PValueFDR"] = multipletests(p, method="fdr_bh")[1]
    save_csv(out, "gcomputation_multifactor_parallel_mediation.csv")
    # This is now the primary multimediator table; retain the familiar name.
    save_csv(out, "gcomputation_multimediator_pairwise.csv")
    return out


def serial_fit(data: pd.DataFrame, outcome: str, comparison: str) -> dict[str, float]:
    a = data["Guidance"].astype(str).eq(comparison).astype(float).to_numpy()
    c = np.log(data["SpatialDifficultyRatio"].to_numpy(float))
    m1 = data["EarlyInformationBurden"].to_numpy(float)
    m2 = data["MidAutonomicLoad"].to_numpy(float)
    m3 = data["LateSpatialInefficiency"].to_numpy(float)
    y = np.log(data["StandardizedTime_s"].clip(lower=0.1).to_numpy(float)) if outcome == "StandardizedTime_s" else data["ExitCorrect"].to_numpy(float)
    b1 = np.linalg.lstsq(np.column_stack([np.ones(len(data)), a, c]), m1, rcond=None)[0]
    b2 = np.linalg.lstsq(np.column_stack([np.ones(len(data)), a, m1, c]), m2, rcond=None)[0]
    b3 = np.linalg.lstsq(np.column_stack([np.ones(len(data)), a, m1, m2, c]), m3, rcond=None)[0]
    by = np.linalg.lstsq(np.column_stack([np.ones(len(data)), a, m1, m2, m3, c]), y, rcond=None)[0]
    total = np.linalg.lstsq(np.column_stack([np.ones(len(data)), a, c]), y, rcond=None)[0][1]
    a1, a2, a3 = b1[1], b2[1], b3[1]
    d21, d31, d32 = b2[2], b3[2], b3[3]
    y1, y2, y3 = by[2], by[3], by[4]
    paths = {
        "A→Info→Y": a1 * y1,
        "A→HRV→Y": a2 * y2,
        "A→Spatial→Y": a3 * y3,
        "A→Info→HRV→Y": a1 * d21 * y2,
        "A→Info→Spatial→Y": a1 * d31 * y3,
        "A→HRV→Spatial→Y": a2 * d32 * y3,
        "A→Info→HRV→Spatial→Y": a1 * d21 * d32 * y3,
    }
    result = {"Total": total, "Direct": by[1], "JointSerialIndirect": float(sum(paths.values()))}
    result.update(paths)
    return result


def serial_mediation(merged: pd.DataFrame) -> pd.DataFrame:
    mediators = ["EarlyInformationBurden", "MidAutonomicLoad", "LateSpatialInefficiency"]
    rows = []
    for smoke in SMOKE_ORDER:
        for reference, comparison in combinations(GUIDANCE_ORDER, 2):
            sub = merged.loc[
                merged["Smoke"].eq(smoke)
                & merged["Guidance"].astype(str).isin([reference, comparison])
            ].dropna(subset=mediators + ["SpatialDifficultyRatio", "StandardizedTime_s", "ExitCorrect"]).copy()
            if sub["Guidance"].nunique() < 2:
                continue
            for mediator in mediators:
                sub[mediator] = zscore(sub[mediator])
            ids = sub["ParticipantID"].unique()
            if len(ids) < 8:
                continue
            id_rows = {pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid) for pid in ids}
            for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                estimate = serial_fit(sub, outcome, comparison)
                boot_values = {key: [] for key in estimate}
                for _ in range(BOOT_N):
                    sampled = RNG.choice(ids, len(ids), replace=True)
                    take = np.concatenate([id_rows[pid] for pid in sampled])
                    boot = sub.iloc[take]
                    if boot["Guidance"].nunique() < 2:
                        continue
                    try:
                        values = serial_fit(boot, outcome, comparison)
                        for key, value in values.items():
                            if np.isfinite(value):
                                boot_values[key].append(value)
                    except Exception:
                        continue
                for effect, raw in estimate.items():
                    vals = np.asarray(boot_values[effect], float)
                    scaled = np.asarray([scale_effect(v, outcome) for v in vals])
                    rows.append({
                        "Smoke": smoke,
                        "Reference": reference,
                        "Comparison": comparison,
                        "Outcome": outcome,
                        "Path": effect,
                        "Estimate": scale_effect(raw, outcome),
                        "CI95Low": float(np.quantile(scaled, 0.025)) if len(scaled) else np.nan,
                        "CI95High": float(np.quantile(scaled, 0.975)) if len(scaled) else np.nan,
                        "PValueBootstrap": boot_p(vals),
                        "N": len(sub),
                        "ParticipantN": len(ids),
                        "Scale": "percent_standardized_time_change" if outcome == "StandardizedTime_s" else "correctness_percentage_points",
                        "TemporalDefinition": "information 0-30%; HRV 30-60%; spatial behavior 60-90%; outcome at 100%",
                        "Status": "exploratory_temporally_ordered_serial_mediation",
                        "ComparisonDesignStatus": (
                            "same_trial_order_randomized_single_guidance_comparison"
                            if {reference, comparison}.issubset({"ArrowRibbon", "MiniMap", "TextBar"})
                            else "guidance_confounding_with_trial_order_cannot_be_excluded"
                        ),
                    })
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    indirect = ~out["Path"].isin(["Total", "Direct"])
    for outcome, idx in out.loc[indirect].groupby("Outcome").groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValueBootstrap"].fillna(1), method="fdr_bh")[1]
    save_csv(out, "gcomputation_temporal_serial_mediation.csv")
    return out


def make_figures(curve: pd.DataFrame, ebm_importance: pd.DataFrame, parallel: pd.DataFrame, serial: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> None:
    visual = sources["visual"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, KEY_VISUAL):
        sns.pointplot(data=visual, x="Guidance", y=metric, hue="Smoke", order=GUIDANCE_ORDER, errorbar=("ci", 95), dodge=0.25, ax=ax)
        ax.set_title(FEATURE_LABEL[metric])
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=22)
        if ax is not axes[0] and ax.legend_:
            ax.legend_.remove()
    axes[0].legend(frameon=False, title="Smoke")
    fig.suptitle("Primary visual-information factors (redundant eye metrics removed)", y=1.02)
    fig.tight_layout()
    savefig(fig, "05_visual_information_by_guidance")

    visual_curve = curve.loc[curve["Layer"].eq("visual_information")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    for row, metric in enumerate(KEY_VISUAL):
        for col, smoke in enumerate(SMOKE_ORDER):
            ax = axes[row, col]
            part = visual_curve.loc[visual_curve["Metric"].eq(metric) & visual_curve["Smoke"].eq(smoke)]
            for guidance in GUIDANCE_ORDER:
                g = part.loc[part["Guidance"].eq(guidance)].sort_values("ProgressMid")
                ax.plot(g["ProgressMid"] * 100, g["Mean"], label=guidance, color=PALETTE[guidance], marker="o", ms=3)
                ax.fill_between(g["ProgressMid"] * 100, g["CI95Low"], g["CI95High"], color=PALETTE[guidance], alpha=0.10)
            ax.set_title(f"{FEATURE_LABEL[metric]} — {smoke}")
            ax.set_xlabel("Normalized trial time (%)")
    axes[0, 1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Visual processes over normalized trial time", y=1.01)
    fig.tight_layout()
    savefig(fig, "06_visual_information_progress")

    hrv_curve = curve.loc[curve["Layer"].eq("autonomic_load")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row, metric in enumerate(KEY_HRV):
        for col, smoke in enumerate(SMOKE_ORDER):
            ax = axes[row, col]
            part = hrv_curve.loc[hrv_curve["Metric"].eq(metric) & hrv_curve["Smoke"].eq(smoke)]
            for guidance in GUIDANCE_ORDER:
                g = part.loc[part["Guidance"].eq(guidance)].sort_values("ProgressMid")
                ax.plot(g["ProgressMid"] * 100, g["Mean"], label=guidance, color=PALETTE[guidance], marker="o", ms=3)
                ax.fill_between(g["ProgressMid"] * 100, g["CI95Low"], g["CI95High"], color=PALETTE[guidance], alpha=0.10)
            ax.axhline(0, color="#374151", lw=0.8, ls="--")
            ax.set_title(f"{FEATURE_LABEL[metric]} — {smoke}")
            ax.set_xlabel("Normalized trial time (%)")
    axes[0, 1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Heart-rate and RMSSD changes over normalized trial time", y=1.01)
    fig.tight_layout()
    savefig(fig, "12_hrv_progress_curves")

    movement_curve = curve.loc[curve["Layer"].eq("spatial_behavior")]
    metrics = ["MovementTurnPathRate_deg_s", "BacktrackIncrement_m"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row, metric in enumerate(metrics):
        for col, smoke in enumerate(SMOKE_ORDER):
            ax = axes[row, col]
            part = movement_curve.loc[movement_curve["Metric"].eq(metric) & movement_curve["Smoke"].eq(smoke)]
            for guidance in GUIDANCE_ORDER:
                g = part.loc[part["Guidance"].eq(guidance)].sort_values("ProgressMid")
                ax.plot(g["ProgressMid"] * 100, g["Mean"], label=guidance, color=PALETTE[guidance], marker="o", ms=3)
            ax.set_title(f"{metric} — {smoke}")
            ax.set_xlabel("Normalized trial time (%)")
    axes[0, 1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Spatial behavior over normalized trial time", y=1.01)
    fig.tight_layout()
    savefig(fig, "10_movement_progress_curves")

    stage = "all_key_processes"
    imp = ebm_importance.loc[ebm_importance["Stage"].eq(stage)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
        part = imp.loc[imp["Outcome"].eq(outcome)].nlargest(9, "Importance").sort_values("Importance")
        ax.barh([FEATURE_LABEL.get(x, x) for x in part["Feature"]], part["Importance"], color="#0072B2")
        ax.set_title(outcome)
        ax.set_xlabel("EBM global importance (predictive, not causal)")
    fig.suptitle("EBM with non-redundant visual, HRV and spatial factors", y=1.01)
    fig.tight_layout()
    savefig(fig, "08_ebm_layered_importance")

    selected = parallel.loc[
        parallel["Model"].eq("three_layer_parallel")
        & parallel["Effect"].eq("JointIndirect")
        & parallel["ComparisonDesignStatus"].eq("same_trial_order_randomized_single_guidance_comparison")
    ].copy()
    selected["Contrast"] = selected["Comparison"] + " vs " + selected["Reference"] + " / " + selected["Smoke"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(selected) * 0.22)), sharey=False)
    for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
        part = selected.loc[selected["Outcome"].eq(outcome)].sort_values("Estimate")
        y = np.arange(len(part))
        ax.errorbar(part["Estimate"], y, xerr=[part["Estimate"] - part["CI95Low"], part["CI95High"] - part["Estimate"]], fmt="o", color="#0072B2", ecolor="#9CA3AF")
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_yticks(y, part["Contrast"])
        ax.set_title(outcome)
        ax.set_xlabel("Joint indirect effect (%)" if outcome == "StandardizedTime_s" else "Joint indirect effect (percentage points)")
    fig.suptitle("Parallel joint mediation by information, autonomic and spatial layers", y=1.01)
    fig.tight_layout()
    savefig(fig, "09_gcomp_layered_mediation")

    selected = serial.loc[
        serial["Path"].eq("JointSerialIndirect")
        & serial["ComparisonDesignStatus"].eq("same_trial_order_randomized_single_guidance_comparison")
    ].copy()
    selected["Contrast"] = selected["Comparison"] + " vs " + selected["Reference"] + " / " + selected["Smoke"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(selected) * 0.22)))
    for ax, outcome in zip(axes, ["StandardizedTime_s", "ExitCorrect"]):
        part = selected.loc[selected["Outcome"].eq(outcome)].sort_values("Estimate")
        y = np.arange(len(part))
        ax.errorbar(part["Estimate"], y, xerr=[part["Estimate"] - part["CI95Low"], part["CI95High"] - part["Estimate"]], fmt="o", color="#D55E00", ecolor="#9CA3AF")
        ax.axvline(0, color="#374151", lw=0.8)
        ax.set_yticks(y, part["Contrast"])
        ax.set_title(outcome)
    fig.suptitle("Temporally ordered serial mediation: 0–30% information → 30–60% HRV → 60–90% spatial behavior", y=1.01)
    fig.tight_layout()
    savefig(fig, "13_temporal_serial_mediation")


def main() -> None:
    print("A/6 Building key multimodal factors")
    merged, sources = load_and_build_features()
    print("B/6 Running normalized-time curves and mixed splines")
    curve, earliest, spline = dynamic_analysis(merged, sources)
    print("C/6 Fitting grouped-CV EBM")
    perf, importance = grouped_ebm(merged)
    key_factor_summaries_and_associations(merged)
    print("D/6 Running multiple parallel mediation")
    parallel = parallel_mediation(merged)
    print("E/6 Running temporally ordered serial mediation")
    serial = serial_mediation(merged)
    print("F/6 Rendering updated figures")
    make_figures(curve, importance, parallel, serial, sources)
    summary = {
        "generated_at": "2026-08-03",
        "primary_trial_n": int(len(merged)),
        "primary_participant_n": int(merged["ParticipantID"].nunique()),
        "hrv_complete_trial_n": int(merged[KEY_HRV].notna().all(axis=1).sum()),
        "serial_complete_trial_n": int(merged[["EarlyInformationBurden", "MidAutonomicLoad", "LateSpatialInefficiency"]].notna().all(axis=1).sum()),
        "primary_visual_factors": KEY_VISUAL,
        "primary_hrv_factors": KEY_HRV,
        "primary_spatial_factors": KEY_SPATIAL,
        "primary_aoi_factors": KEY_AOI,
        "all_dynamic_axes": "normalized_trial_time_0_100_percent_in_10_bins",
        "multi_mediation": ["multiple_parallel", "temporally_ordered_serial"],
        "causal_boundary": "randomized guidance supports treatment contrasts; mediator-outcome exchangeability remains unverified",
    }
    (DATA / "multimodal_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run_path = DATA / "analysis_run_summary.json"
    run_summary = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    hrv_summary_path = DATA / "hrv_run_summary.json"
    hrv_summary = json.loads(hrv_summary_path.read_text(encoding="utf-8")) if hrv_summary_path.exists() else {}
    run_summary.update({
        "generated_at": "2026-08-03",
        "ivt_primary_velocity_threshold_deg_s": 30.0,
        "ivt_primary_fixation_min_duration_ms": 100.0,
        "visual_phase_axis": "normalized_trial_time_10_bins",
        "movement_phase_axis": "normalized_trial_time_10_bins",
        "hrv_phase_axis": "normalized_trial_time_10_bins",
        "primary_hrv_usable_trial_n": hrv_summary.get("primary_hrv_usable_trial_n"),
        "primary_hrv_usable_participant_n": hrv_summary.get("primary_hrv_usable_participant_n"),
        "serial_complete_trial_n": summary["serial_complete_trial_n"],
        "primary_visual_factors": KEY_VISUAL,
        "primary_hrv_factors": KEY_HRV,
        "primary_spatial_factors": KEY_SPATIAL,
        "trial_order_identification_limit": (
            "StaticSign is fixed at orders 1/2, randomized single guidance at 3/4, "
            "and Full at 5/6; practice/order confounding cannot be separated for five-mode contrasts."
        ),
    })
    run_path.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
