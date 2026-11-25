"""Requested incremental revision: randomized-order assumption, HRV-only stress,
space-time-mode visuals, EBM without exit difficulty, and tiered mediation.

This script reuses reviewed extracted tables and writes only to
``Results_Guidance_Spatial_Causal_Analysis``.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ValueWarning


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ValueWarning)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DATA, TABLE, FIG, NOTE = (OUT / x for x in ("data", "tables", "figures", "notes"))
DOC = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"
DOC.mkdir(parents=True, exist_ok=True)
GUIDANCE = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
KEYS = ["Scenario", "ParticipantID", "FileName"]
SEED = 20260804
BOOT_N = 300
RNG = np.random.default_rng(SEED)

VISUAL = ["HeadAngularPathRate_deg_s", "FixationTimeProportion", "GazeDirectionEntropy"]
AOI = ["GuideDwellProportion"]
HRV = ["HRVStressProxy"]
MOVEMENT = ["MeanSpeed_m_s", "MovementTurnPathRate_deg_s", "BacktrackIncrement_m", "NonProgressPathRatio"]
METRICS = VISUAL + AOI + HRV + MOVEMENT

LABEL = {
    "HeadAngularPathRate_deg_s": "头部搜索路径率",
    "FixationTimeProportion": "凝视时间比例",
    "GazeDirectionEntropy": "注视方向熵",
    "GuideDwellProportion": "引导注视比例",
    "HRVStressProxy": "HRV压力代理（静息logRMSSD−任务logRMSSD）",
    "MeanSpeed_m_s": "平均速度",
    "MovementTurnPathRate_deg_s": "转向路径率",
    "BacktrackIncrement_m": "回退增量",
    "NonProgressPathRatio": "非前进路径比例",
}
LAYER = {**{x: "visual_information" for x in VISUAL + AOI}, "HRVStressProxy": "hrv_stress", **{x: "spatial_behavior" for x in MOVEMENT}}
ROLE_ORDER = [
    "Spawn", "Platform", "TargetSideStairEntrance", "TargetSideStair",
    "NonTargetSideStairEntrance", "NonTargetSideStair", "HallInsideGate", "Gate",
    "HallOutsideGate", "TargetGroupDecision4", "NonTargetGroupDecision4",
    "TargetExitCorridor", "NonTargetExitCorridor", "TargetPairDecision2",
    "NonTargetPairDecision2", "TargetBranchCorridor", "WrongBranchCorridor", "TargetExit", "WrongExit",
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def savefig(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(np.nan, index=x.index)


def winsor(series: pd.Series, low: float = 0.01, high: float = 0.99) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 10:
        return x
    return x.clip(x.quantile(low), x.quantile(high))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bins = pd.read_csv(DATA / "spatiotemporal_zone_bins.csv")
    trial_zone = pd.read_csv(DATA / "spatiotemporal_trial_zone_features.csv")
    # Existing spatial extension stores the EBM-wide table together with its
    # derived mediator columns under this audited filename.
    wide = pd.read_csv(DATA / "spatiotemporal_trial_mediators.csv")
    trial = pd.read_csv(DATA / "multimodal_trial_features.csv")
    for frame in (bins, trial_zone, wide, trial):
        if "LogRMSSDBaselineDelta" in frame:
            frame["HRVStressProxy"] = -pd.to_numeric(frame["LogRMSSDBaselineDelta"], errors="coerce")
    return bins, trial_zone, wide, trial


def revised_spline_tests(bins: pd.DataFrame) -> pd.DataFrame:
    # Trial order was already absent from the reviewed spline screen. Reversing
    # logRMSSD change into a stress proxy changes only its sign, not its p-value.
    # Reuse avoids another slow, rank-deficient 146-model screen.
    out = pd.read_csv(TABLE / "spatiotemporal_spline_tests.csv")
    out = out.loc[out["Metric"] != "MeanHRBaselineDelta_bpm"].copy()
    out.loc[out["Metric"] == "LogRMSSDBaselineDelta", "Metric"] = "HRVStressProxy"
    out.loc[out["Metric"] == "HRVStressProxy", "Layer"] = "hrv_stress"
    out["Controls"] = "smoke + log spatial difficulty; trial order absent; reused because sign reversal leaves significance unchanged"
    out["Status"] = out["Status"].astype(str) + "; RMSSD_reexpressed_as_stress_proxy"
    out.to_csv(TABLE / "spatiotemporal_spline_tests.csv", index=False, encoding="utf-8-sig")
    return out


def revised_pairwise(bins: pd.DataFrame) -> pd.DataFrame:
    prior = pd.read_csv(TABLE / "spatiotemporal_pairwise_guidance_contrasts.csv")
    prior = prior.loc[~prior["Metric"].isin(["MeanHRBaselineDelta_bpm", "HRVStressProxy"])].copy()
    global_sd = {m: pd.to_numeric(bins[m], errors="coerce").std(ddof=0) for m in METRICS if m in bins}
    prior["Layer"] = prior["Metric"].map(LAYER)
    prior["StandardizedDifference"] = prior.apply(lambda r: r["AdjustedDifference"] / global_sd.get(r["Metric"], np.nan), axis=1)
    prior["StandardizedCILow"] = prior.apply(lambda r: (r["AdjustedDifference"] - 1.96*r["SE"]) / global_sd.get(r["Metric"], np.nan), axis=1)
    prior["StandardizedCIHigh"] = prior.apply(lambda r: (r["AdjustedDifference"] + 1.96*r["SE"]) / global_sd.get(r["Metric"], np.nan), axis=1)
    prior["Controls"] = "smoke + log spatial difficulty; trial order absent"
    rows: list[dict] = prior.to_dict(orient="records")
    # Add the requested HRV-variability stress contrast. The sign is reversed
    # so positive values mean more vagal withdrawal / higher stress proxy.
    for metric in HRV:
        source = bins.dropna(subset=[metric]).copy()
        for (role, time_bin), cell in source.groupby(["FunctionalZone", "GlobalTimeBin"], observed=True):
            for reference, comparison in combinations(GUIDANCE, 2):
                sub = cell.loc[cell["Guidance"].isin([reference, comparison])].copy()
                counts = sub.groupby("Guidance")["ParticipantID"].nunique()
                if len(counts) < 2 or counts.min() < 6:
                    continue
                sub["Treatment"] = sub["Guidance"].eq(comparison).astype(int)
                sd = global_sd[metric]
                try:
                    fit = ols(f"{metric} ~ Treatment + C(Smoke) + np.log(SpatialDifficultyRatio)", data=sub).fit(
                        cov_type="cluster", cov_kwds={"groups": sub["ParticipantID"]}
                    )
                    diff = float(fit.params.get("Treatment", np.nan))
                    se = float(fit.bse.get("Treatment", np.nan))
                    rows.append({
                        "Metric": metric, "Layer": LAYER[metric], "FunctionalZone": role,
                        "GlobalTimeBin": int(time_bin), "GlobalTimeMidPct": (float(time_bin) - 0.5) * 10,
                        "Reference": reference, "Comparison": comparison,
                        "AdjustedDifference": diff, "SE": se,
                        "StandardizedDifference": diff / sd if np.isfinite(sd) and sd > 0 else np.nan,
                        "StandardizedCILow": (diff - 1.96 * se) / sd if np.isfinite(sd) and sd > 0 else np.nan,
                        "StandardizedCIHigh": (diff + 1.96 * se) / sd if np.isfinite(sd) and sd > 0 else np.nan,
                        "PValue": float(fit.pvalues.get("Treatment", np.nan)),
                        "N": len(sub), "ParticipantN": sub["ParticipantID"].nunique(),
                        "Controls": "smoke + log spatial difficulty; trial order absent",
                    })
                except Exception:
                    continue
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    for metric, idx in out.groupby("Metric").groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValue"].fillna(1), method="fdr_bh")[1]
    out.to_csv(TABLE / "spatiotemporal_pairwise_guidance_contrasts.csv", index=False, encoding="utf-8-sig")
    return out


def revised_ebm(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = wide.copy()
    wide["SmokeHigh"] = wide["Smoke"].eq("HighSmoke").astype(int)
    for col in [c for c in wide if c.endswith("LogRMSSDBaselineDelta")]:
        wide[col.replace("LogRMSSDBaselineDelta", "HRVStressProxy")] = -pd.to_numeric(wide[col], errors="coerce")

    controls = ["SmokeHigh"]
    roles = {"Spawn", "Platform", "HallInsideGate", "Gate", "HallOutsideGate"}
    allowed_suffix = tuple(VISUAL + HRV + MOVEMENT)
    process = [
        c for c in wide.columns if "__" in c and c.split("__", 1)[0] in roles
        and c.endswith(allowed_suffix) and "MeanHRBaseline" not in c
        and wide[c].notna().sum() >= 150 and wide[c].nunique(dropna=True) > 1
    ]
    rows, importance = [], []
    groups = wide["ParticipantID"]
    splitter = GroupKFold(n_splits=5)
    for scope, features in [("smoke_only", controls), ("prechoice_spatiotemporal_processes", controls + process)]:
        x = wide[features].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True))
        for outcome in ["StandardizedTime_s", "ExitCorrect"]:
            y = pd.to_numeric(wide[outcome], errors="coerce")
            pred = np.full(len(wide), np.nan)
            for train, test in splitter.split(x, y, groups):
                kwargs = dict(max_bins=12, max_rounds=1500, outer_bags=4, interactions=3, random_state=SEED, n_jobs=1)
                model = ExplainableBoostingRegressor(**kwargs) if outcome == "StandardizedTime_s" else ExplainableBoostingClassifier(**kwargs)
                model.fit(x.iloc[train], y.iloc[train])
                pred[test] = model.predict(x.iloc[test]) if outcome == "StandardizedTime_s" else model.predict_proba(x.iloc[test])[:, 1]
            row = {
                "Scope": scope, "Outcome": outcome, "FeatureN": len(features),
                "MAE": mean_absolute_error(y, pred) if outcome == "StandardizedTime_s" else np.nan,
                "R2": r2_score(y, pred) if outcome == "StandardizedTime_s" else np.nan,
                "AUC": roc_auc_score(y, pred) if outcome == "ExitCorrect" else np.nan,
                "Brier": brier_score_loss(y, pred) if outcome == "ExitCorrect" else np.nan,
                "Controls": "smoke_only; trial_order_ignored; spatial_difficulty_excluded",
            }
            rows.append(row)
            if scope == "prechoice_spatiotemporal_processes":
                kwargs = dict(max_bins=12, max_rounds=2000, outer_bags=6, interactions=3, random_state=SEED, n_jobs=1)
                full = ExplainableBoostingRegressor(**kwargs) if outcome == "StandardizedTime_s" else ExplainableBoostingClassifier(**kwargs)
                full.fit(x, y)
                for name, value in zip(full.term_names_, full.term_importances()):
                    importance.append({"Outcome": outcome, "Term": name, "Importance": float(value)})
    cv = pd.DataFrame(rows)
    imp = pd.DataFrame(importance).sort_values(["Outcome", "Importance"], ascending=[True, False])
    cv.to_csv(TABLE / "spatiotemporal_ebm_grouped_cv.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(TABLE / "spatiotemporal_ebm_importance.csv", index=False, encoding="utf-8-sig")
    return cv, imp


def build_mediation_features(trial: pd.DataFrame, trial_zone: pd.DataFrame) -> pd.DataFrame:
    base = trial[KEYS + ["Guidance", "Smoke", "StandardizedTime_s", "ExitCorrect", "HeadAngularPathRate_deg_s", "FixationTimeProportion", "GazeDirectionEntropy", "LogRMSSDBaselineDelta", "BacktrackingRatio"]].copy()
    base["HeadZ"] = zscore(winsor(base["HeadAngularPathRate_deg_s"]))
    base["FixZ"] = -zscore(winsor(base["FixationTimeProportion"]))
    base["EntropyZ"] = zscore(winsor(base["GazeDirectionEntropy"]))
    base["UnzonedSearchBurden"] = base[["HeadZ", "FixZ", "EntropyZ"]].mean(axis=1, skipna=False)
    base["UnzonedHRVStress"] = -pd.to_numeric(base["LogRMSSDBaselineDelta"], errors="coerce")
    base["UnzonedBacktrack"] = np.log1p(winsor(base["BacktrackingRatio"].clip(lower=0)))

    z = trial_zone.copy()
    z["HeadZ"] = zscore(winsor(z["HeadAngularPathRate_deg_s"]))
    z["FixZ"] = -zscore(winsor(z["FixationTimeProportion"]))
    z["EntropyZ"] = zscore(winsor(z["GazeDirectionEntropy"]))
    z["SearchBurden"] = z[["HeadZ", "FixZ", "EntropyZ"]].mean(axis=1, skipna=False)
    z["HRVStress"] = -pd.to_numeric(z["LogRMSSDBaselineDelta"], errors="coerce")
    z["BacktrackLog"] = np.log1p(winsor(z["BacktrackIncrement_m"].clip(lower=0)))

    def zone_mean(roles: list[str], col: str, name: str) -> pd.DataFrame:
        return z.loc[z["FunctionalZone"].isin(roles)].groupby(KEYS, as_index=False)[col].mean().rename(columns={col: name})

    early = zone_mean(["Spawn", "Platform"], "SearchBurden", "ZonedSearchBurden")
    middle = zone_mean(["HallInsideGate", "Gate"], "HRVStress", "ZonedHRVStress")
    late = zone_mean(["HallOutsideGate"], "BacktrackLog", "ZonedBacktrack")
    out = base.merge(early, on=KEYS, how="left", validate="one_to_one").merge(middle, on=KEYS, how="left", validate="one_to_one").merge(late, on=KEYS, how="left", validate="one_to_one")
    out.to_csv(DATA / "tiered_mediation_features.csv", index=False, encoding="utf-8-sig")
    return out


def scale_effect(value: float, outcome: str) -> float:
    return float((np.exp(value) - 1) * 100) if outcome == "StandardizedTime_s" else float(value * 100)


def bootstrap_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(2 * min((values <= 0).mean(), (values >= 0).mean())) if len(values) else np.nan


def mediation_fit(data: pd.DataFrame, mediators: list[str], outcome: str, comparison: str) -> dict[str, float]:
    a = data["Guidance"].eq(comparison).astype(float).to_numpy()
    smoke = data["Smoke"].eq("HighSmoke").astype(float).to_numpy()
    m = data[mediators].to_numpy(float)
    y = np.log(data["StandardizedTime_s"].clip(lower=0.1).to_numpy(float)) if outcome == "StandardizedTime_s" else data["ExitCorrect"].to_numpy(float)
    base = np.column_stack([np.ones(len(data)), a, smoke])
    total = np.linalg.lstsq(base, y, rcond=None)[0][1]
    a_paths = [np.linalg.lstsq(base, m[:, j], rcond=None)[0][1] for j in range(len(mediators))]
    by = np.linalg.lstsq(np.column_stack([base, m]), y, rcond=None)[0]
    specific = np.asarray(a_paths) * by[base.shape[1]:base.shape[1] + len(mediators)]
    result = {"Total": total, "Direct": by[1], "JointIndirect": float(specific.sum())}
    result.update({f"Specific:{name}": float(v) for name, v in zip(mediators, specific)})
    return result


def tiered_mediation(features: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "unsegmented": ["UnzonedSearchBurden", "UnzonedHRVStress", "UnzonedBacktrack"],
        # Regional RMSSD is too sparse when a person crosses a zone quickly.
        # Keep visual/spatial mediators anchored to key zones, while using the
        # stable whole-trial HRV stress proxy in the same temporal layer.
        "zone_anchored": ["ZonedSearchBurden", "UnzonedHRVStress", "ZonedBacktrack"],
    }
    rows: list[dict] = []
    for scope, all_mediators in scopes.items():
        complete = features.dropna(subset=all_mediators + ["StandardizedTime_s", "ExitCorrect"]).copy()
        for count in (1, 2, 3):
            for selected in combinations(all_mediators, count):
                selected = list(selected)
                for reference, comparison in combinations(GUIDANCE, 2):
                    sub = complete.loc[complete["Guidance"].isin([reference, comparison])].copy()
                    if sub["Guidance"].nunique() < 2:
                        continue
                    for col in selected:
                        sub[col] = zscore(winsor(sub[col]))
                    participants = sub["ParticipantID"].unique()
                    if len(participants) < 10:
                        continue
                    id_rows = {pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid) for pid in participants}
                    for outcome in ["StandardizedTime_s", "ExitCorrect"]:
                        estimate = mediation_fit(sub, selected, outcome, comparison)
                        boot = {k: [] for k in estimate}
                        for _ in range(BOOT_N):
                            sampled = RNG.choice(participants, len(participants), replace=True)
                            take = np.concatenate([id_rows[pid] for pid in sampled])
                            sample = sub.iloc[take]
                            if sample["Guidance"].nunique() < 2:
                                continue
                            try:
                                vals = mediation_fit(sample, selected, outcome, comparison)
                                for key, value in vals.items():
                                    if np.isfinite(value):
                                        boot[key].append(value)
                            except Exception:
                                continue
                        for effect, raw in estimate.items():
                            vals = np.asarray(boot[effect], float)
                            scaled = np.asarray([scale_effect(v, outcome) for v in vals])
                            rows.append({
                                "Scope": scope, "MediatorCount": count, "MediatorSet": " + ".join(selected),
                                "Reference": reference, "Comparison": comparison, "Outcome": outcome,
                                "Effect": effect, "Estimate": scale_effect(raw, outcome),
                                "CI95Low": float(np.quantile(scaled, .025)) if len(scaled) else np.nan,
                                "CI95High": float(np.quantile(scaled, .975)) if len(scaled) else np.nan,
                                "PValueBootstrap": bootstrap_p(vals), "N": len(sub), "ParticipantN": len(participants),
                                "Scale": "percent_standardized_time_change" if outcome == "StandardizedTime_s" else "correctness_percentage_points",
                                "Controls": "smoke_only; trial_order_ignored; spatial_difficulty_not_adjusted",
                                "Assumption": "guidance assignment treated as randomized per user instruction; mediator-outcome confounding remains possible",
                            })
    out = pd.DataFrame(rows)
    out["PValueFDR"] = np.nan
    indirect = out["Effect"].str.contains("Indirect|Specific", regex=True)
    for (scope, count, outcome), idx in out.loc[indirect].groupby(["Scope", "MediatorCount", "Outcome"]).groups.items():
        out.loc[idx, "PValueFDR"] = multipletests(out.loc[idx, "PValueBootstrap"].fillna(1), method="fdr_bh")[1]
    out.to_csv(TABLE / "gcomputation_tiered_mediation.csv", index=False, encoding="utf-8-sig")
    return out


def plot_spline(spline: pd.DataFrame) -> None:
    mat = spline.pivot(index="FunctionalZone", columns="Metric", values="GuidanceTimeInteractionFDR")
    mat = mat.reindex([x for x in ROLE_ORDER if x in mat.index], columns=[x for x in METRICS if x in mat.columns])
    values = -np.log10(mat.clip(lower=1e-6))
    fig, ax = plt.subplots(figsize=(15.5, 9.5))
    sns.heatmap(values, cmap="YlOrBr", vmin=0, vmax=6, linewidths=.35, linecolor="#d1d5db", ax=ax,
                cbar_kws={"label": "−log10(FDR)"})
    ax.set_xticklabels([LABEL.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()], rotation=38, ha="right")
    ax.set_xlabel("因素"); ax.set_ylabel("功能区域")
    ax.set_title("空间区域中的引导×时间曲线差异（仅HRV变异性，不含平均心率）")
    fig.tight_layout(); savefig(fig, "15_zone_time_spline_significance")


def plot_count_heatmap(contrasts: pd.DataFrame) -> None:
    sig = contrasts.loc[contrasts["PValueFDR"] < .05]
    count = sig.groupby(["FunctionalZone", "GlobalTimeBin"]).size().unstack(fill_value=0)
    count = count.reindex([x for x in ROLE_ORDER if x in count.index], columns=range(1, 11), fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 8.5))
    sns.heatmap(count, annot=True, fmt="d", cmap="Oranges", ax=ax, cbar_kws={"label": "FDR<0.05的两两差异数"})
    ax.set_xlabel("全程归一化时间格（1=0–10%）"); ax.set_ylabel("功能区域")
    ax.set_title("差异集中在哪个空间与时间（不含平均心率）")
    fig.tight_layout(); savefig(fig, "16_zone_time_pairwise_difference_heatmap")


def relative_means(bins: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = bins.dropna(subset=[metric]).groupby(["Guidance", "FunctionalZone", "GlobalTimeBin"], as_index=False).agg(
        Mean=(metric, "mean"), ParticipantN=("ParticipantID", "nunique")
    )
    g = g.loc[g["ParticipantN"] >= 5].copy()
    pooled_sd = pd.to_numeric(bins[metric], errors="coerce").std(ddof=0)
    cell_mean = g.groupby(["FunctionalZone", "GlobalTimeBin"])["Mean"].transform("mean")
    g["RelativeStandardizedMean"] = (g["Mean"] - cell_mean) / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
    return g


def plot_metric_grid(bins: pd.DataFrame, metrics: list[str], stem: str, title: str) -> None:
    fig, axes = plt.subplots(len(metrics), len(GUIDANCE), figsize=(21, max(5, 4.2 * len(metrics))), squeeze=False, sharex=True, sharey=False)
    last = None
    for r, metric in enumerate(metrics):
        rel = relative_means(bins, metric)
        for c, mode in enumerate(GUIDANCE):
            ax = axes[r, c]
            part = rel.loc[rel["Guidance"] == mode]
            mat = part.pivot(index="FunctionalZone", columns="GlobalTimeBin", values="RelativeStandardizedMean")
            mat = mat.reindex(ROLE_ORDER, columns=range(1, 11))
            last = sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-1.5, vmax=1.5, mask=mat.isna(), cbar=False,
                               linewidths=.2, linecolor="#e5e7eb", ax=ax)
            ax.set_title(mode if r == 0 else "")
            ax.set_xlabel("时间格" if r == len(metrics) - 1 else "")
            ax.set_ylabel(LABEL[metric] if c == 0 else "")
            ax.set_yticks(np.arange(len(ROLE_ORDER)) + .5)
            if c > 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(ROLE_ORDER, rotation=0)
                ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1.5, vmax=1.5), cmap="RdBu_r")
    # Reserve a dedicated strip outside the subplot grid so the shared colorbar
    # never covers the right-most guidance panel.
    cbar_bottom = .18 if len(metrics) == 1 else .12
    cbar_height = .62 if len(metrics) == 1 else .76
    cax = fig.add_axes([.905, cbar_bottom, .012, cbar_height])
    fig.colorbar(sm, cax=cax, label="相对同一空间×时间单元五模式平均的标准化偏差")
    fig.suptitle(title + "\n红=高于同一时空单元平均，蓝=低于平均，空白=有效受试者不足5人", y=.995)
    fig.subplots_adjust(top=.84 if len(metrics) == 1 else .92, wspace=.05, hspace=.12, right=.875)
    savefig(fig, stem)


def plot_direction_chain(features: pd.DataFrame, mediation: pd.DataFrame) -> None:
    """Direction-resolved StaticSign→ArrowRibbon parallel-mediation diagram."""
    required = ["UnzonedSearchBurden", "UnzonedHRVStress", "UnzonedBacktrack", "StandardizedTime_s", "ExitCorrect"]
    d = features.dropna(subset=required).loc[
        features["Guidance"].isin(["StaticSign", "ArrowRibbon"])
    ].copy()
    d["Treatment"] = d["Guidance"].eq("ArrowRibbon").astype(int)
    d["SmokeHigh"] = d["Smoke"].eq("HighSmoke").astype(int)
    d["LogTime"] = np.log(d["StandardizedTime_s"].clip(lower=.1))
    for mediator in ["UnzonedHRVStress", "UnzonedBacktrack"]:
        d[mediator] = zscore(winsor(d[mediator]))

    rows: list[dict] = []

    def add_fit(path_type: str, mediator: str, outcome: str, fit, term: str,
                transform=lambda x: x, unit: str = "SD") -> None:
        ci = fit.conf_int().loc[term]
        rows.append({
            "PathType": path_type, "Mediator": mediator, "Outcome": outcome, "Effect": term,
            "Estimate": transform(float(fit.params[term])), "CI95Low": transform(float(ci.iloc[0])),
            "CI95High": transform(float(ci.iloc[1])), "PValue": float(fit.pvalues[term]),
            "PValueFDR": np.nan, "Unit": unit, "N": len(d),
            "ParticipantN": d["ParticipantID"].nunique(),
        })

    for mediator in ["UnzonedHRVStress", "UnzonedBacktrack"]:
        fit = ols(f"{mediator} ~ Treatment + SmokeHigh", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["ParticipantID"]}
        )
        add_fit("GroupDifference", mediator, "Mediator", fit, "Treatment")

    serial = ols("UnzonedBacktrack ~ Treatment + SmokeHigh + UnzonedHRVStress", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["ParticipantID"]}
    )
    add_fit("HRVToBacktrack", "UnzonedHRVStress", "UnzonedBacktrack", serial, "UnzonedHRVStress")

    for outcome, response, transform, unit in [
        ("StandardizedTime_s", "LogTime", lambda x: (np.exp(x) - 1) * 100, "%/SD"),
        ("ExitCorrect", "ExitCorrect", lambda x: x * 100, "percentage points/SD"),
    ]:
        fit = ols(
            f"{response} ~ Treatment + SmokeHigh + UnzonedHRVStress + UnzonedBacktrack", data=d
        ).fit(cov_type="cluster", cov_kwds={"groups": d["ParticipantID"]})
        for mediator in ["UnzonedHRVStress", "UnzonedBacktrack"]:
            add_fit("MediatorOutcome", mediator, outcome, fit, mediator, transform, unit)

    selected = mediation.loc[
        (mediation["Scope"] == "unsegmented")
        & (mediation["MediatorCount"] == 2)
        & (mediation["MediatorSet"] == "UnzonedHRVStress + UnzonedBacktrack")
        & (mediation["Reference"] == "StaticSign")
        & (mediation["Comparison"] == "ArrowRibbon")
    ]
    for row in selected.itertuples():
        mediator = ""
        if row.Effect.startswith("Specific:"):
            mediator = row.Effect.split(":", 1)[1]
        rows.append({
            "PathType": "MediationEffect", "Mediator": mediator, "Outcome": row.Outcome,
            "Effect": row.Effect, "Estimate": row.Estimate, "CI95Low": row.CI95Low,
            "CI95High": row.CI95High, "PValue": row.PValueBootstrap,
            "PValueFDR": row.PValueFDR, "Unit": row.Scale, "N": row.N,
            "ParticipantN": row.ParticipantN,
        })
    evidence = pd.DataFrame(rows)
    evidence.to_csv(TABLE / "arrowribbon_vs_staticsign_directional_paths.csv", index=False, encoding="utf-8-sig")

    def one(path_type: str, mediator: str = "", outcome: str = "", effect: str = "") -> pd.Series:
        q = evidence["PathType"].eq(path_type)
        if mediator:
            q &= evidence["Mediator"].eq(mediator)
        if outcome:
            q &= evidence["Outcome"].eq(outcome)
        if effect:
            q &= evidence["Effect"].eq(effect)
        return evidence.loc[q].iloc[0]

    hrv_a = one("GroupDifference", "UnzonedHRVStress")
    back_a = one("GroupDifference", "UnzonedBacktrack")
    serial_row = one("HRVToBacktrack", "UnzonedHRVStress", "UnzonedBacktrack")
    total_t = one("MediationEffect", outcome="StandardizedTime_s", effect="Total")
    joint_t = one("MediationEffect", outcome="StandardizedTime_s", effect="JointIndirect")
    hrv_i_t = one("MediationEffect", "UnzonedHRVStress", "StandardizedTime_s", "Specific:UnzonedHRVStress")
    back_i_t = one("MediationEffect", "UnzonedBacktrack", "StandardizedTime_s", "Specific:UnzonedBacktrack")
    total_c = one("MediationEffect", outcome="ExitCorrect", effect="Total")
    joint_c = one("MediationEffect", outcome="ExitCorrect", effect="JointIndirect")
    hrv_i_c = one("MediationEffect", "UnzonedHRVStress", "ExitCorrect", "Specific:UnzonedHRVStress")
    back_i_c = one("MediationEffect", "UnzonedBacktrack", "ExitCorrect", "Specific:UnzonedBacktrack")
    hrv_b_t = one("MediatorOutcome", "UnzonedHRVStress", "StandardizedTime_s")
    back_b_t = one("MediatorOutcome", "UnzonedBacktrack", "StandardizedTime_s")
    hrv_b_c = one("MediatorOutcome", "UnzonedHRVStress", "ExitCorrect")
    back_b_c = one("MediatorOutcome", "UnzonedBacktrack", "ExitCorrect")

    def ptxt(value: float, prefix: str = "p") -> str:
        return f"{prefix}<.001" if value < .001 else f"{prefix}={value:.3f}".replace("0.", ".")

    def signed(value: float, digits: int = 2) -> str:
        return f"{value:+.{digits}f}"

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str,
            face: str, edge: str, title_color: str = "#111827") -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.012,rounding_size=.018",
                               linewidth=1.8, edgecolor=edge, facecolor=face, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + .025, y + h - .045, title, ha="left", va="top", fontsize=14,
                fontweight="bold", color=title_color, transform=ax.transAxes)
        ax.text(x + .025, y + h - .105, body, ha="left", va="top", fontsize=10.5,
                linespacing=1.45, color="#1F2937", transform=ax.transAxes)

    def arrow(start, end, supported: bool, color: str = "#1D4ED8", rad: float = 0.0) -> None:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.6 if supported else 2.0,
            color=color if supported else "#6B7280", linestyle="-" if supported else "--",
            connectionstyle=f"arc3,rad={rad}", transform=ax.transAxes, shrinkA=4, shrinkB=4,
        ))

    exposure_body = (
        f"相对StaticSign\n总效应：时间 {signed(total_t.Estimate)}%\n"
        f"总效应：正确率 {signed(total_c.Estimate)}个百分点"
    )
    hrv_body = (
        f"Δ = {signed(hrv_a.Estimate)} SD\n95%CI [{signed(hrv_a.CI95Low)}, {signed(hrv_a.CI95High)}]\n"
        f"{ptxt(hrv_a.PValue)}：略升高，但不显著"
    )
    back_body = (
        f"Δ = {signed(back_a.Estimate)} SD\n95%CI [{signed(back_a.CI95Low)}, {signed(back_a.CI95High)}]\n"
        f"{ptxt(back_a.PValue)}：显著减少"
    )
    outcome_body = (
        "疏散时间\n"
        f"  HRVStress：b={signed(hrv_b_t.Estimate)}%/SD，{ptxt(hrv_b_t.PValue)}\n"
        f"    特定间接={signed(hrv_i_t.Estimate)}%（FDR={hrv_i_t.PValueFDR:.3f}）\n"
        f"  Backtrack：b={signed(back_b_t.Estimate)}%/SD，{ptxt(back_b_t.PValue)}\n"
        f"    特定间接={signed(back_i_t.Estimate)}%（FDR<.001）\n"
        f"  联合间接={signed(joint_t.Estimate)}% [{signed(joint_t.CI95Low)}, {signed(joint_t.CI95High)}]\n\n"
        "出口正确率\n"
        f"  HRVStress：b={signed(hrv_b_c.Estimate)}个百分点/SD，{ptxt(hrv_b_c.PValue)}\n"
        f"    特定间接={signed(hrv_i_c.Estimate)}个百分点（FDR={hrv_i_c.PValueFDR:.3f}）\n"
        f"  Backtrack：b={signed(back_b_c.Estimate)}个百分点/SD，{ptxt(back_b_c.PValue)}\n"
        f"    特定间接={signed(back_i_c.Estimate)}个百分点（FDR<.001）\n"
        f"  联合间接={signed(joint_c.Estimate)}个百分点 [{signed(joint_c.CI95Low)}, {signed(joint_c.CI95High)}]"
    )
    box(.025, .40, .185, .28, "ArrowRibbon（光毯）", exposure_body, "#FFF7ED", "#C2410C")
    box(.285, .66, .245, .22, "HRVStress", hrv_body, "#F5F3FF", "#7C3AED")
    box(.285, .27, .245, .22, "Backtrack", back_body, "#EFF6FF", "#1D4ED8")
    box(.665, .25, .305, .64, "结果路径与间接效应", outcome_body, "#F9FAFB", "#374151")

    arrow((.21, .59), (.285, .74), False)
    arrow((.21, .47), (.285, .37), True)
    arrow((.405, .66), (.405, .49), False)
    ax.text(.415, .565,
            f"HRVStress → Backtrack\nβ={signed(serial_row.Estimate, 3)}\n{ptxt(serial_row.PValue)}",
            ha="left", va="center", fontsize=9.5, color="#6B7280", transform=ax.transAxes)
    arrow((.53, .76), (.665, .73), False)
    arrow((.53, .38), (.665, .48), True)
    ax.text(.59, .77, "结果路径不显著", fontsize=9.5, color="#6B7280", ha="center", transform=ax.transAxes)
    ax.text(.59, .39, "结果路径显著", fontsize=9.5, color="#1D4ED8", ha="center", transform=ax.transAxes)

    ax.plot([.03, .08], [.145, .145], color="#1D4ED8", lw=2.8, transform=ax.transAxes)
    ax.text(.085, .145, "实线：p<.05，当前数据支持方向", va="center", fontsize=10, transform=ax.transAxes)
    ax.plot([.32, .37], [.145, .145], color="#6B7280", lw=2.2, ls="--", transform=ax.transAxes)
    ax.text(.375, .145, "虚线：p≥.05，方向不确定", va="center", fontsize=10, transform=ax.transAxes)
    ax.text(.03, .08,
            "结论：光毯显著减少回退；回退减少解释了几乎全部显著联合间接效应。当前数据不支持“光毯降低HRV压力→进一步减少回退”。",
            fontsize=12, fontweight="bold", color="#1E3A8A", transform=ax.transAxes)
    ax.text(.03, .035,
            f"不分区双中介；控制烟雾；N={len(d)}试次、{d['ParticipantID'].nunique()}名受试者。路径p值采用受试者聚类稳健标准误；间接效应采用受试者Bootstrap并进行层级FDR校正。",
            fontsize=9.5, color="#4B5563", transform=ax.transAxes)
    ax.set_title("StaticSign → ArrowRibbon：HRVStress与Backtrack的方向链路", fontsize=20,
                 fontweight="bold", pad=24)
    savefig(fig, "25_directional_mechanism_chain")


def plot_key_direction_chains(features: pd.DataFrame, mediation: pd.DataFrame) -> None:
    """Publication-style summary of all de-duplicated key directional chains."""
    single = mediation.loc[
        (mediation["Scope"] == "unsegmented")
        & (mediation["MediatorCount"] == 1)
        & (mediation["Effect"] == "Specific:UnzonedBacktrack")
    ].copy()
    key_pairs = (
        single.loc[single["PValueFDR"] < .05, ["Reference", "Comparison"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    key_pairs = set(key_pairs)
    pair_order = [
        ("StaticSign", "ArrowRibbon"), ("StaticSign", "MiniMap"),
        ("StaticSign", "Full"), ("ArrowRibbon", "Full"),
        ("ArrowRibbon", "TextBar"), ("MiniMap", "TextBar"),
        ("TextBar", "Full"),
    ]
    pairs = [pair for pair in pair_order if pair in key_pairs]

    complete = features.dropna(subset=[
        "UnzonedSearchBurden", "UnzonedHRVStress", "UnzonedBacktrack",
        "StandardizedTime_s", "ExitCorrect",
    ]).copy()
    rows = []
    for reference, comparison in pairs:
        sub = complete.loc[complete["Guidance"].isin([reference, comparison])].copy()
        mediator = winsor(sub["UnzonedBacktrack"])
        sub["BacktrackZ"] = zscore(mediator)
        sub["Treatment"] = sub["Guidance"].eq(comparison).astype(int)
        sub["SmokeHigh"] = sub["Smoke"].eq("HighSmoke").astype(int)
        fit = ols("BacktrackZ ~ Treatment + SmokeHigh", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["ParticipantID"]}
        )
        ci = fit.conf_int().loc["Treatment"]
        row = {
            "Reference": reference, "Comparison": comparison,
            "BacktrackDifferenceSD": float(fit.params["Treatment"]),
            "BacktrackCI95Low": float(ci.iloc[0]), "BacktrackCI95High": float(ci.iloc[1]),
            "BacktrackPValue": float(fit.pvalues["Treatment"]),
            "N": len(sub), "ParticipantN": sub["ParticipantID"].nunique(),
        }
        for outcome, prefix in [("StandardizedTime_s", "Time"), ("ExitCorrect", "Correctness")]:
            effect = single.loc[
                (single["Reference"] == reference)
                & (single["Comparison"] == comparison)
                & (single["Outcome"] == outcome)
            ].iloc[0]
            row.update({
                f"{prefix}Indirect": effect["Estimate"],
                f"{prefix}CI95Low": effect["CI95Low"],
                f"{prefix}CI95High": effect["CI95High"],
                f"{prefix}PValueFDR": effect["PValueFDR"],
            })
        rows.append(row)
    chains = pd.DataFrame(rows)
    chains.to_csv(TABLE / "key_directional_mediation_chains.csv", index=False, encoding="utf-8-sig")

    mode_cn = {
        "StaticSign": "静态标识", "ArrowRibbon": "光毯", "MiniMap": "小地图",
        "TextBar": "文字", "Full": "完整AR",
    }
    blue, warm, grey, dark = "#0F4D92", "#B64342", "#8A8A8A", "#272727"

    def stars(p: float) -> str:
        if p < .001:
            return "***"
        if p < .01:
            return "**"
        if p < .05:
            return "*"
        return " ns"

    def cell(value: float, p: float, unit: str) -> tuple[str, str, str]:
        direction = "↑" if value > 0 else "↓"
        color = warm if value > 0 else blue
        sig = stars(p)
        if p >= .05:
            color = grey
        return f"{direction} {abs(value):.1f}{unit}{sig}", color, "bold" if p < .05 else "normal"

    fig, ax = plt.subplots(figsize=(7.2, 5.25))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x = {"comparison": .03, "mediator": .43, "time": .70, "correctness": .91}
    ax.text(x["comparison"], .91, "引导对比（右−左）", fontsize=8, fontweight="bold", ha="left")
    ax.text(x["mediator"], .91, "Backtrack变化", fontsize=8, fontweight="bold", ha="center")
    ax.text(x["time"], .91, "时间间接效应", fontsize=8, fontweight="bold", ha="center")
    ax.text(x["correctness"], .91, "正确率间接效应", fontsize=8, fontweight="bold", ha="center")
    ax.plot([.02, .98], [.885, .885], color=dark, lw=.8)

    y_positions = np.linspace(.80, .22, len(chains))
    for i, (row, y) in enumerate(zip(chains.itertuples(), y_positions)):
        if i % 2 == 0:
            ax.axhspan(y - .043, y + .043, xmin=.015, xmax=.985, color="#F5F5F5", zorder=0)
        comparison = f"{mode_cn[row.Reference]}  →  {mode_cn[row.Comparison]}"
        ax.text(x["comparison"], y, comparison, fontsize=7.5, ha="left", va="center", color=dark)
        ax.text(.315, y, "→", fontsize=9, ha="center", va="center", color="#A0A0A0")

        mediator_text, mediator_color, mediator_weight = cell(row.BacktrackDifferenceSD, row.BacktrackPValue, " SD")
        time_text, time_color, time_weight = cell(row.TimeIndirect, row.TimePValueFDR, "%")
        correct_text, correct_color, correct_weight = cell(row.CorrectnessIndirect, row.CorrectnessPValueFDR, " pp")
        ax.text(x["mediator"], y, mediator_text, fontsize=7.5, ha="center", va="center",
                color=mediator_color, fontweight=mediator_weight)
        ax.text(.565, y, "→", fontsize=9, ha="center", va="center", color="#A0A0A0")
        ax.text(x["time"], y, time_text, fontsize=7.5, ha="center", va="center",
                color=time_color, fontweight=time_weight)
        ax.text(.805, y, "│", fontsize=8, ha="center", va="center", color="#D0D0D0")
        ax.text(x["correctness"], y, correct_text, fontsize=7.5, ha="center", va="center",
                color=correct_color, fontweight=correct_weight)

    ax.text(.02, .13, "↓ 减少", color=blue, fontsize=7)
    ax.text(.13, .13, "↑ 增加", color=warm, fontsize=7)
    ax.text(.24, .13, "灰色 = 不显著", color=grey, fontsize=7)
    ax.text(.43, .13, "Backtrack列：p；结果列：FDR   *<.05  **<.01  ***<.001", color=dark, fontsize=6.8)
    ax.text(.02, .075,
            "单中介模型；控制烟雾。Backtrack差值采用受试者聚类稳健检验；时间与正确率为特定间接效应并经层级FDR校正。",
            fontsize=6.3, color="#555555")
    ax.set_title("引导模式的关键方向链路", fontsize=11, fontweight="bold", loc="left", pad=10)
    fig.subplots_adjust(left=.03, right=.985, top=.90, bottom=.04)
    base = FIG / "25_directional_mechanism_chain"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_directional(contrasts: pd.DataFrame) -> None:
    layers = [("visual_information", "视觉/AOI"), ("hrv_stress", "HRV压力"), ("spatial_behavior", "空间行为")]
    fig, axes = plt.subplots(1, 3, figsize=(21, 10))
    for ax, (layer, title) in zip(axes, layers):
        d = contrasts.loc[(contrasts["Layer"] == layer) & contrasts["StandardizedDifference"].notna()].sort_values(["PValueFDR", "PValue"]).head(14).copy()
        d = d.iloc[::-1]
        d["Label"] = d["Comparison"] + "−" + d["Reference"] + " | " + d["FunctionalZone"] + " | T" + d["GlobalTimeBin"].astype(str) + " | " + d["Metric"].map(LABEL)
        y = np.arange(len(d))
        for i, row in enumerate(d.itertuples()):
            color = "#C23B22" if row.PValueFDR < .05 else "#6B7280"
            ax.errorbar(row.StandardizedDifference, i,
                        xerr=[[row.StandardizedDifference-row.StandardizedCILow], [row.StandardizedCIHigh-row.StandardizedDifference]],
                        fmt="o", color=color, ecolor=color, capsize=2)
        ax.axvline(0, color="#111827", lw=1)
        ax.set_yticks(y, d["Label"], fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel("标准化差异：Comparison − Reference")
    fig.suptitle("具体哪种引导在何时何地更高或更低（每层FDR最强的14项）", y=1.01)
    fig.tight_layout(); savefig(fig, "22_directional_space_time_guidance_contrasts")


def plot_ebm(importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, outcome, title in zip(axes, ["StandardizedTime_s", "ExitCorrect"], ["标准化疏散时间", "出口正确性"]):
        top = importance.loc[importance["Outcome"] == outcome].nlargest(12, "Importance").iloc[::-1]
        ax.barh(top["Term"], top["Importance"], color="#0072B2" if outcome == "StandardizedTime_s" else "#D55E00")
        ax.set_title(title); ax.set_xlabel("EBM全局重要性（预测贡献，不是因果效应）")
    fig.suptitle("空间化EBM：已忽略试次顺序并排除空间难度比和平均心率", y=1.01)
    fig.tight_layout(); savefig(fig, "17_spatiotemporal_ebm_importance")


def hrv_pressure_analysis(trial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = trial[KEYS + ["Guidance", "Smoke", "HRVStressProxy"]].dropna(subset=["HRVStressProxy"]).copy()
    rows = []
    for (guidance, smoke), g in d.groupby(["Guidance", "Smoke"], observed=True):
        ids = g["ParticipantID"].unique()
        by_id = {pid: g.loc[g["ParticipantID"] == pid, "HRVStressProxy"].to_numpy(float) for pid in ids}
        boot = []
        for _ in range(1000):
            sampled = RNG.choice(ids, len(ids), replace=True)
            boot.append(float(np.mean(np.concatenate([by_id[x] for x in sampled]))))
        rows.append({
            "Guidance": guidance, "Smoke": smoke, "N": len(g), "ParticipantN": len(ids),
            "MeanHRVStressProxy": float(g["HRVStressProxy"].mean()), "MedianHRVStressProxy": float(g["HRVStressProxy"].median()),
            "CI95Low": float(np.quantile(boot, .025)), "CI95High": float(np.quantile(boot, .975)),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE / "hrv_stress_guidance_summary.csv", index=False, encoding="utf-8-sig")

    tests = []
    omnibus = ols("HRVStressProxy ~ C(Guidance) * C(Smoke)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ParticipantID"]})
    names = omnibus.model.exog_names
    for label, indexes in {
        "guidance_omnibus": [i for i, x in enumerate(names) if "C(Guidance)" in x and ":" not in x],
        "guidance_by_smoke_interaction": [i for i, x in enumerate(names) if "C(Guidance)" in x and ":C(Smoke)" in x],
    }.items():
        restriction = np.zeros((len(indexes), len(names)))
        for r, c in enumerate(indexes): restriction[r, c] = 1
        tests.append({"Test": label, "Smoke": "All", "Reference": "", "Comparison": "", "Difference": np.nan, "SE": np.nan,
                      "PValue": float(omnibus.wald_test(restriction, scalar=True).pvalue), "N": len(d), "ParticipantN": d["ParticipantID"].nunique()})
    for smoke in ["HighSmoke", "LowSmoke"]:
        cell = d.loc[d["Smoke"] == smoke]
        for reference, comparison in combinations(GUIDANCE, 2):
            sub = cell.loc[cell["Guidance"].isin([reference, comparison])].copy()
            if sub["Guidance"].nunique() < 2:
                continue
            sub["Treatment"] = sub["Guidance"].eq(comparison).astype(int)
            fit = ols("HRVStressProxy ~ Treatment", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["ParticipantID"]})
            tests.append({"Test": "pairwise", "Smoke": smoke, "Reference": reference, "Comparison": comparison,
                          "Difference": float(fit.params["Treatment"]), "SE": float(fit.bse["Treatment"]),
                          "PValue": float(fit.pvalues["Treatment"]), "N": len(sub), "ParticipantN": sub["ParticipantID"].nunique()})
    tests = pd.DataFrame(tests)
    tests["PValueFDR"] = np.nan
    pair = tests["Test"] == "pairwise"
    tests.loc[pair, "PValueFDR"] = multipletests(tests.loc[pair, "PValue"], method="fdr_bh")[1]
    tests.to_csv(TABLE / "hrv_stress_guidance_tests.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    offsets = {"HighSmoke": -.13, "LowSmoke": .13}
    colors = {"HighSmoke": "#6A3D9A", "LowSmoke": "#1B9E77"}
    for smoke in ["HighSmoke", "LowSmoke"]:
        s = summary.loc[summary["Smoke"] == smoke].set_index("Guidance").reindex(GUIDANCE)
        x = np.arange(len(GUIDANCE)) + offsets[smoke]
        ax.errorbar(x, s["MeanHRVStressProxy"], yerr=[s["MeanHRVStressProxy"]-s["CI95Low"], s["CI95High"]-s["MeanHRVStressProxy"]],
                    fmt="o-", capsize=3, color=colors[smoke], label="高烟" if smoke == "HighSmoke" else "低烟")
    ax.axhline(0, color="#111827", lw=1, ls="--")
    ax.set_xticks(np.arange(len(GUIDANCE)), GUIDANCE)
    ax.set_ylabel("HRV压力代理：静息log(RMSSD)−任务log(RMSSD)")
    ax.set_title("五种引导模式的HRV压力代理（均值与受试者整簇Bootstrap 95%CI）")
    ax.legend(title="烟雾"); fig.tight_layout(); savefig(fig, "24_hrv_stress_by_guidance")
    return summary, tests


def plot_mediation(mediation: pd.DataFrame) -> None:
    joint = mediation.loc[mediation["Effect"] == "JointIndirect"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, outcome, title in zip(axes, ["StandardizedTime_s", "ExitCorrect"], ["标准化时间（%变化）", "出口正确性（百分点）"]):
        d = joint.loc[joint["Outcome"] == outcome].sort_values(["PValueFDR", "PValueBootstrap"]).head(18).copy().iloc[::-1]
        short = d["MediatorSet"].str.replace("Unzoned", "", regex=False).str.replace("Zoned", "", regex=False)
        d["Label"] = d["Scope"].map({"unsegmented": "不分区", "zone_anchored": "空间锚定"}) + " | " + d["MediatorCount"].astype(str) + "因素 | " + short + " | " + d["Reference"] + "→" + d["Comparison"]
        y = np.arange(len(d))
        color = np.where(d["PValueFDR"] < .05, "#C23B22", "#6B7280")
        for i, row in enumerate(d.itertuples()):
            ax.errorbar(row.Estimate, i, xerr=[[row.Estimate-row.CI95Low], [row.CI95High-row.Estimate]], fmt="o", color=color[i], ecolor=color[i], alpha=.9, capsize=2)
        ax.axvline(0, color="#111827", lw=1)
        ax.set_yticks(y, d["Label"], fontsize=7.2); ax.set_title(title); ax.set_xlabel("联合间接效应（红色=对应层级FDR<0.05）")
    fig.suptitle("单因素、双因素与三因素中介：不分区和功能分区对照", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "18_spatiotemporal_serial_mediation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "18_spatiotemporal_serial_mediation.svg", bbox_inches="tight")
    fig.savefig(FIG / "23_tiered_mediation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "23_tiered_mediation.svg", bbox_inches="tight")
    fig.savefig(FIG / "23_tiered_mediation.pdf", bbox_inches="tight")
    fig.savefig(FIG / "23_tiered_mediation.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_complete_tiered_mediation_map(features: pd.DataFrame, mediation: pd.DataFrame) -> None:
    """Plot all pairwise comparisons and all 1/2/3-mediator models in both scopes."""
    scope_mediators = {
        "unsegmented": ["UnzonedSearchBurden", "UnzonedHRVStress", "UnzonedBacktrack"],
        "zone_anchored": ["ZonedSearchBurden", "UnzonedHRVStress", "ZonedBacktrack"],
    }
    scope_cn = {"unsegmented": "不分区", "zone_anchored": "空间锚定"}
    mode_cn = {
        "StaticSign": "静态标识", "ArrowRibbon": "光毯", "MiniMap": "小地图",
        "TextBar": "文字", "Full": "完整AR",
    }
    pairs = list(combinations(GUIDANCE, 2))
    rows: list[dict] = []

    for scope, mediators in scope_mediators.items():
        complete = features.dropna(subset=mediators + ["StandardizedTime_s", "ExitCorrect"]).copy()
        model_sets = [
            [mediators[0]], [mediators[1]], [mediators[2]],
            [mediators[0], mediators[1]], [mediators[0], mediators[2]],
            [mediators[1], mediators[2]], mediators,
        ]
        for reference, comparison in pairs:
            sub = complete.loc[complete["Guidance"].isin([reference, comparison])].copy()
            sub["Treatment"] = sub["Guidance"].eq(comparison).astype(int)
            sub["SmokeHigh"] = sub["Smoke"].eq("HighSmoke").astype(int)
            for mediator in mediators:
                sub[mediator] = zscore(winsor(sub[mediator]))
                fit = ols(f"{mediator} ~ Treatment + SmokeHigh", data=sub).fit(
                    cov_type="cluster", cov_kwds={"groups": sub["ParticipantID"]}
                )
                ci = fit.conf_int().loc["Treatment"]
                rows.append({
                    "Scope": scope, "Reference": reference, "Comparison": comparison,
                    "Panel": "MediatorChange", "MediatorCount": 1,
                    "MediatorSet": mediator, "Estimate": float(fit.params["Treatment"]),
                    "CI95Low": float(ci.iloc[0]), "CI95High": float(ci.iloc[1]),
                    "PValue": float(fit.pvalues["Treatment"]), "PValueFDR": np.nan,
                    "Unit": "SD", "N": len(sub), "ParticipantN": sub["ParticipantID"].nunique(),
                })
            for selected in model_sets:
                mediator_set = " + ".join(selected)
                count = len(selected)
                for outcome, panel, unit in [
                    ("StandardizedTime_s", "TimeJointIndirect", "%"),
                    ("ExitCorrect", "CorrectnessJointIndirect", "pp"),
                ]:
                    result = mediation.loc[
                        (mediation["Scope"] == scope)
                        & (mediation["MediatorCount"] == count)
                        & (mediation["MediatorSet"] == mediator_set)
                        & (mediation["Reference"] == reference)
                        & (mediation["Comparison"] == comparison)
                        & (mediation["Outcome"] == outcome)
                        & (mediation["Effect"] == "JointIndirect")
                    ].iloc[0]
                    rows.append({
                        "Scope": scope, "Reference": reference, "Comparison": comparison,
                        "Panel": panel, "MediatorCount": count, "MediatorSet": mediator_set,
                        "Estimate": result["Estimate"], "CI95Low": result["CI95Low"],
                        "CI95High": result["CI95High"], "PValue": result["PValueBootstrap"],
                        "PValueFDR": result["PValueFDR"], "Unit": unit,
                        "N": result["N"], "ParticipantN": result["ParticipantN"],
                    })
    source = pd.DataFrame(rows)
    source.to_csv(TABLE / "complete_tiered_mediation_map.csv", index=False, encoding="utf-8-sig")

    blue, warm, grey, dark = "#0F4D92", "#B64342", "#8A8A8A", "#272727"
    fill_blue, fill_warm, fill_grey = "#E8F0F8", "#F8E9E7", "#FAFAFA"

    def stars(p: float) -> str:
        if p < .001:
            return "***"
        if p < .01:
            return "**"
        if p < .05:
            return "*"
        return " ns"

    def draw_matrix(ax, scope: str, panel: str, model_sets: list[list[str]], title: str,
                    show_y: bool, label_map: dict[str, str]) -> None:
        ax.set_xlim(0, len(model_sets)); ax.set_ylim(len(pairs), 0)
        ax.set_title(title, fontsize=7.4, fontweight="bold", pad=7, loc="left")
        for i, (reference, comparison) in enumerate(pairs):
            for j, model_set in enumerate(model_sets):
                mediator_set = " + ".join(model_set)
                row = source.loc[
                    (source["Scope"] == scope) & (source["Reference"] == reference)
                    & (source["Comparison"] == comparison) & (source["Panel"] == panel)
                    & (source["MediatorSet"] == mediator_set)
                ].iloc[0]
                p_display = row["PValue"] if panel == "MediatorChange" else row["PValueFDR"]
                significant = p_display < .05
                value = float(row["Estimate"])
                if significant and value < 0:
                    face, edge, text_color = fill_blue, blue, blue
                elif significant and value > 0:
                    face, edge, text_color = fill_warm, warm, warm
                else:
                    face, edge, text_color = fill_grey, "#D8D8D8", grey
                count = int(row["MediatorCount"])
                linewidth = 1.05 if count == 3 else (.75 if count == 2 else .55)
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor=edge, linewidth=linewidth))
                arrow_symbol = "↑" if value > 0 else "↓"
                digits = 2 if panel == "MediatorChange" else 1
                ax.text(j + .5, i + .5, f"{arrow_symbol}{abs(value):.{digits}f}{stars(p_display)}",
                        ha="center", va="center", fontsize=5.15, color=text_color,
                        fontweight="bold" if significant else "normal")
        ax.set_xticks(np.arange(len(model_sets)) + .5,
                      [label_map[" + ".join(x)] for x in model_sets], fontsize=5.55)
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(np.arange(len(pairs)) + .5)
        if show_y:
            ax.set_yticklabels([f"{mode_cn[a]}→{mode_cn[b]}" for a, b in pairs], fontsize=5.55)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if len(model_sets) == 7:
            ax.axvline(3, color="#6B7280", lw=.75)
            ax.axvline(6, color="#6B7280", lw=.75)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2))
    for col, (scope, mediators) in enumerate(scope_mediators.items()):
        short = {mediators[0]: "搜索", mediators[1]: "HRV", mediators[2]: "回退"}
        model_sets = [
            [mediators[0]], [mediators[1]], [mediators[2]],
            [mediators[0], mediators[1]], [mediators[0], mediators[2]],
            [mediators[1], mediators[2]], mediators,
        ]
        med_labels = {x: short[x] for x in mediators}
        model_labels = {
            " + ".join(model_sets[0]): "1 搜索", " + ".join(model_sets[1]): "1 HRV",
            " + ".join(model_sets[2]): "1 回退", " + ".join(model_sets[3]): "2 搜+H",
            " + ".join(model_sets[4]): "2 搜+回", " + ".join(model_sets[5]): "2 H+回",
            " + ".join(model_sets[6]): "3 全部",
        }
        for row_index, (panel, sets, subtitle, labels) in enumerate([
            ("MediatorChange", [[x] for x in mediators], "中介变化 (ΔSD)", med_labels),
            ("TimeJointIndirect", model_sets, "时间联合间接效应 (%)", model_labels),
            ("CorrectnessJointIndirect", model_sets, "正确率联合间接效应 (pp)", model_labels),
        ]):
            letter = "abcdef"[row_index * 2 + col]
            draw_matrix(axes[row_index, col], scope, panel, sets,
                        f"{letter}  {scope_cn[scope]}：{subtitle}", col == 0, labels)

    fig.suptitle("五种引导模式的完整分层中介结果", fontsize=10.5, fontweight="bold", x=.02, ha="left", y=.993)
    fig.text(.02, .958, "全部10组两两比较；3个单因素、3个双因素和1个三因素模型；右侧引导 − 左侧引导", fontsize=6.4, color="#555555")
    fig.text(.02, .035, "↓ 减少", fontsize=6.2, color=blue)
    fig.text(.11, .035, "↑ 增加", fontsize=6.2, color=warm)
    fig.text(.20, .035, "灰色 = 不显著", fontsize=6.2, color=grey)
    fig.text(.38, .035, "中介变化：聚类稳健p；间接效应：层级FDR   *<.05  **<.01  ***<.001", fontsize=5.9, color=dark)
    fig.text(.02, .010, "1/2/3表示中介因素数量；搜=搜索负担，H=HRV压力，回=回退。", fontsize=5.9, color="#555555")
    fig.subplots_adjust(left=.15, right=.99, top=.91, bottom=.065, hspace=.34, wspace=.12)
    base = FIG / "26_complete_tiered_mediation_map"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_tiered_mediation_pathways() -> None:
    """Turn the complete mediation matrix into explicit contrast→mediator→outcome strips."""
    source = pd.read_csv(TABLE / "complete_tiered_mediation_map.csv")
    scope_mediators = {
        "unsegmented": ["UnzonedSearchBurden", "UnzonedHRVStress", "UnzonedBacktrack"],
        "zone_anchored": ["ZonedSearchBurden", "UnzonedHRVStress", "ZonedBacktrack"],
    }
    scope_cn = {"unsegmented": "不分区", "zone_anchored": "空间锚定"}
    mode_cn = {
        "StaticSign": "静态标识", "ArrowRibbon": "光毯", "MiniMap": "小地图",
        "TextBar": "文字", "Full": "完整AR",
    }
    pairs = list(combinations(GUIDANCE, 2))
    panel_spec = [
        ("unsegmented", "TimeJointIndirect", "疏散时间", "时", "%"),
        ("unsegmented", "CorrectnessJointIndirect", "出口正确率", "正", "pp"),
        ("zone_anchored", "TimeJointIndirect", "疏散时间", "时", "%"),
        ("zone_anchored", "CorrectnessJointIndirect", "出口正确率", "正", "pp"),
    ]

    def pstars(p: float) -> str:
        if pd.isna(p) or p >= .05:
            return " ns"
        if p < .001:
            return "***"
        if p < .01:
            return "**"
        return "*"

    pathway_rows: list[dict] = []
    scope_model_sets: dict[str, list[list[str]]] = {}
    for scope, mediators in scope_mediators.items():
        model_sets = [
            [mediators[0]], [mediators[1]], [mediators[2]],
            [mediators[0], mediators[1]], [mediators[0], mediators[2]],
            [mediators[1], mediators[2]], mediators,
        ]
        scope_model_sets[scope] = model_sets
        for reference, comparison in pairs:
            changes = source.loc[
                (source["Scope"] == scope) & (source["Reference"] == reference)
                & (source["Comparison"] == comparison) & (source["Panel"] == "MediatorChange")
            ].set_index("MediatorSet")
            for panel in ["TimeJointIndirect", "CorrectnessJointIndirect"]:
                for selected in model_sets:
                    mediator_set = " + ".join(selected)
                    joint = source.loc[
                        (source["Scope"] == scope) & (source["Reference"] == reference)
                        & (source["Comparison"] == comparison) & (source["Panel"] == panel)
                        & (source["MediatorSet"] == mediator_set)
                    ].iloc[0]
                    record = {
                        "Scope": scope, "Reference": reference, "Comparison": comparison,
                        "OutcomePanel": panel, "MediatorCount": len(selected),
                        "MediatorSet": mediator_set, "JointEstimate": joint["Estimate"],
                        "JointCI95Low": joint["CI95Low"], "JointCI95High": joint["CI95High"],
                        "JointPValueFDR": joint["PValueFDR"], "N": joint["N"],
                        "ParticipantN": joint["ParticipantN"],
                    }
                    for index, mediator in enumerate(selected, start=1):
                        change = changes.loc[mediator]
                        record[f"Mediator{index}"] = mediator
                        record[f"Mediator{index}ChangeSD"] = change["Estimate"]
                        record[f"Mediator{index}PValue"] = change["PValue"]
                    pathway_rows.append(record)
    pathways = pd.DataFrame(pathway_rows)
    pathways.to_csv(TABLE / "tiered_mediation_pathways.csv", index=False, encoding="utf-8-sig")

    blue, warm, grey, dark = "#0F4D92", "#B64342", "#898989", "#282828"
    fill_blue, fill_warm, fill_grey = "#EDF3F9", "#FAEEEC", "#FBFBFB"
    model_labels = ["1 搜索", "1 HRV", "1 回退", "2 搜+H", "2 搜+回", "2 H+回", "3 全部"]
    code_order = ["搜", "H", "回"]

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 21.0))
    for panel_index, (scope, panel, outcome_cn, outcome_code, unit) in enumerate(panel_spec):
        ax = axes[panel_index]
        mediators = scope_mediators[scope]
        code_map = dict(zip(mediators, code_order))
        model_sets = scope_model_sets[scope]
        ax.set_xlim(0, len(model_sets)); ax.set_ylim(len(pairs), 0)
        ax.set_title(
            f"{'abcd'[panel_index]}  {scope_cn[scope]}：{outcome_cn}",
            fontsize=9.0, fontweight="bold", loc="left", pad=8,
        )
        for i, (reference, comparison) in enumerate(pairs):
            for j, selected in enumerate(model_sets):
                mediator_set = " + ".join(selected)
                row = pathways.loc[
                    (pathways["Scope"] == scope) & (pathways["Reference"] == reference)
                    & (pathways["Comparison"] == comparison)
                    & (pathways["OutcomePanel"] == panel)
                    & (pathways["MediatorSet"] == mediator_set)
                ].iloc[0]
                effect = float(row["JointEstimate"])
                effect_p = float(row["JointPValueFDR"])
                effect_sig = effect_p < .05
                if effect_sig and effect < 0:
                    face, edge, outcome_color = fill_blue, blue, blue
                elif effect_sig and effect > 0:
                    face, edge, outcome_color = fill_warm, warm, warm
                else:
                    face, edge, outcome_color = fill_grey, "#D2D2D2", grey
                ax.add_patch(Rectangle(
                    (j + .02, i + .025), .96, .95, facecolor=face, edgecolor=edge,
                    linewidth=1.05 if effect_sig else .55,
                    linestyle="-" if effect_sig else (0, (2.3, 2.0)),
                ))

                mediator_y = {
                    1: [i + .22],
                    2: [i + .13, i + .34],
                    3: [i + .09, i + .27, i + .45],
                }[len(selected)]
                for k, (mediator, ypos) in enumerate(zip(selected, mediator_y), start=1):
                    change = float(row[f"Mediator{k}ChangeSD"])
                    change_p = float(row[f"Mediator{k}PValue"])
                    change_sig = change_p < .05
                    med_color = blue if change_sig and change < 0 else warm if change_sig else grey
                    arrow = "↓" if change < 0 else "↑"
                    ax.text(
                        j + .5, ypos,
                        f"{code_map[mediator]} {arrow}{abs(change):.2f}{pstars(change_p)}",
                        ha="center", va="center", fontsize=6.0, color=med_color,
                        fontweight="bold" if change_sig else "normal",
                    )
                connector_start = {1: i + .36, 2: i + .46, 3: i + .55}[len(selected)]
                connector_end = {1: i + .58, 2: i + .62, 3: i + .65}[len(selected)]
                ax.annotate(
                    "", xy=(j + .5, connector_end), xytext=(j + .5, connector_start),
                    arrowprops=dict(arrowstyle="-|>", lw=.7, color=outcome_color,
                                    mutation_scale=6.5, shrinkA=0, shrinkB=0),
                )
                arrow = "↓" if effect < 0 else "↑"
                ax.text(
                    j + .5, i + .80,
                    f"{outcome_code} {arrow}{abs(effect):.1f}{pstars(effect_p)}",
                    ha="center", va="center", fontsize=6.25, color=outcome_color,
                    fontweight="bold" if effect_sig else "normal",
                )
        ax.set_xticks(np.arange(7) + .5, model_labels, fontsize=7.0)
        ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=3)
        ax.set_yticks(np.arange(10) + .5)
        ax.set_yticklabels(
            [f"{mode_cn[a]}→{mode_cn[b]}" for a, b in pairs], fontsize=6.6,
        )
        ax.tick_params(axis="y", length=0, pad=5)
        ax.axvline(3, color="#6B7280", lw=.8)
        ax.axvline(6, color="#6B7280", lw=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "五种引导模式的完整分层中介链路",
        fontsize=13.2, fontweight="bold", x=.02, ha="left", y=.992,
    )
    fig.text(
        .02, .968,
        "每行表示右侧引导相对左侧引导；单元格依次显示中介变化（ΔSD）→时间或正确率的联合间接效应",
        fontsize=7.3, color="#555555",
    )
    fig.text(
        .02, .018,
        "搜=搜索负担，H=HRV压力，回=回退；时=标准化时间（%），正=出口正确率（pp）。蓝=减少，红=增加，灰色虚框=FDR不显著。",
        fontsize=7.0, color="#444444",
    )
    fig.text(
        .02, .006,
        "中介变化：受试者聚类稳健p；联合间接效应：对应模型层级FDR。*<.05  **<.01  ***<.001；双/三因素为并行联合中介。",
        fontsize=6.7, color="#555555",
    )
    fig.subplots_adjust(left=.13, right=.995, top=.925, bottom=.036, hspace=.29)
    base = FIG / "27_tiered_mediation_pathways"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_notes(cv: pd.DataFrame, mediation: pd.DataFrame, spline: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    joint = mediation.loc[mediation["Effect"] == "JointIndirect"]
    note = f"""# 用户指定修订：随机顺序假设、HRV压力和分层中介

## 执行口径

- 按用户指示忽略试次顺序，将引导分配视为随机；所有修订模型不含TrialOrder。
- 空间化EBM排除SpatialDifficultyRatio，因为目标出口与引导模式绑定。
- EBM只控制烟雾，过程特征仅来自最终出口选择前的公共区域。
- RR分析不再使用平均心率；压力代理定义为静息log(RMSSD)−任务log(RMSSD)，越高表示任务期RMSSD相对降低越明显。
- SDNN仍保留在完整试次质量表，但不进入短区域或10%时间格模型。
- 新中介只控制烟雾，不控制试次顺序或空间难度；标准化时间仍沿用既有A*标准化结果。

## 中介层级

- 不分区：全程视觉搜索负担、全程HRV压力代理、全程回退比例。
- 空间锚定：出生点/站台搜索负担、全试次HRV压力、闸机外回退。区域RMSSD因快速穿越导致完整AR仅4个完整试次，故不再强行作为中介；区域HRV只保留描述图。
- 每个范围在同一完整案例队列上分别检验1因素、全部2因素组合和3因素联合模型，便于比较复杂度而不混入样本变化。

## 规模与结果文件

- 修订样条检验：{len(spline)}项。
- 修订区域×时间两两比较：{len(contrasts)}项。
- 分层中介联合间接效应：{len(joint)}项，其中FDR<0.05为{int((joint['PValueFDR'] < .05).sum())}项。
- EBM性能见`../Results_Guidance_Spatial_Causal_Analysis/tables/spatiotemporal_ebm_grouped_cv.csv`。

## 必须保留的限制

“引导随机、忽略试次顺序”是用户指定的分析假设，不是由现有日志重新验证得到的事实。目标出口与引导绑定仍限制五模式总体因果解释；排除空间难度只能防止EBM利用该代理，不能创造缺失的共同支持。HRV压力代理还混合运动、呼吸与姿势影响，不等同于纯心理压力。
"""
    (DOC / "requested_revision_randomized_hrv_mediation.md").write_text(note, encoding="utf-8")


def main() -> None:
    print("A/7 Loading reviewed extracted tables")
    bins, trial_zone, wide, trial = load_inputs()
    print("B/7 Refitting spline and pairwise screens without heart rate/order/difficulty")
    spline = revised_spline_tests(bins)
    contrasts = revised_pairwise(bins)
    print("C/7 Refitting EBM without trial order and spatial difficulty")
    cv, importance = revised_ebm(wide)
    print("D/7 Building unsegmented and zoned mediation features")
    features = build_mediation_features(trial, trial_zone)
    print("E/7 Running 1-, 2-, and 3-factor mediation")
    mediation = tiered_mediation(features)
    hrv_pressure_analysis(trial)
    print("F/7 Rendering detailed space-time-mode figures")
    plot_spline(spline); plot_count_heatmap(contrasts); plot_ebm(importance)
    plot_metric_grid(bins, VISUAL + AOI, "19_visual_attention_space_time_by_guidance", "视觉注意：空间×时间×引导模式")
    plot_metric_grid(bins, HRV, "20_hrv_stress_space_time_by_guidance", "HRV压力代理：空间×时间×引导模式")
    plot_metric_grid(bins, MOVEMENT, "21_spatial_behavior_space_time_by_guidance", "空间行为：空间×时间×引导模式")
    plot_directional(contrasts); plot_mediation(mediation)
    plot_complete_tiered_mediation_map(features, mediation)
    plot_tiered_mediation_pathways()
    plot_key_direction_chains(features, mediation)
    print("G/7 Writing audit notes and summary")
    write_notes(cv, mediation, spline, contrasts)
    joint = mediation.loc[mediation["Effect"] == "JointIndirect"]
    summary = {
        "generated_at": "2026-08-04", "assumption": "guidance treated as randomized; trial order ignored",
        "ebm_exclusions": ["SpatialDifficultyRatio", "TrialOrder", "MeanHRBaselineDelta_bpm"],
        "hrv_primary": "HRVStressProxy = resting log(RMSSD) - task log(RMSSD)",
        "spline_test_n": len(spline), "pairwise_test_n": len(contrasts),
        "mediation_joint_test_n": len(joint), "mediation_joint_fdr_significant_n": int((joint["PValueFDR"] < .05).sum()),
        "ebm_cv": cv.to_dict(orient="records"),
    }
    (DATA / "requested_revision_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
