"""Cluster-bootstrap descriptive contrasts for visual and AOI information metrics."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
SEED = 20260802
BOOT_N = 1000
GUIDANCE_ORDER = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar", "Full"]
AR_ORDER = ["ArrowRibbon", "MiniMap", "TextBar", "Full"]
SMOKE_ORDER = ["LowSmoke", "HighSmoke"]


def boot_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return min(1.0, 2 * min((np.sum(values <= 0) + 1) / (len(values) + 1), (np.sum(values >= 0) + 1) / (len(values) + 1)))


def contrast_table(source: pd.DataFrame, modes: list[str], metrics: list[str], scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    pair_rows, mean_rows = [], []
    for smoke in SMOKE_ORDER:
        smoke_data = source.loc[source["Smoke"].eq(smoke)].copy()
        for metric in metrics:
            for guidance in modes:
                values = smoke_data.loc[smoke_data["Guidance"].astype(str).eq(guidance), metric].dropna().to_numpy(float)
                ids_frame = smoke_data.loc[smoke_data["Guidance"].astype(str).eq(guidance)].dropna(subset=[metric])
                ids = ids_frame["ParticipantID"].unique()
                positions = {pid: np.flatnonzero(ids_frame["ParticipantID"].to_numpy() == pid) for pid in ids}
                boots = []
                for _ in range(BOOT_N):
                    sampled = rng.choice(ids, len(ids), replace=True)
                    take = np.concatenate([positions[pid] for pid in sampled])
                    boots.append(ids_frame.iloc[take][metric].mean())
                mean_rows.append({
                    "Scope": scope, "Smoke": smoke, "Guidance": guidance, "Metric": metric,
                    "TrialN": len(values), "ParticipantN": len(ids), "Mean": values.mean(),
                    "CI95Low": np.quantile(boots, 0.025), "CI95High": np.quantile(boots, 0.975),
                })
            for reference, comparison in combinations(modes, 2):
                sub = smoke_data.loc[smoke_data["Guidance"].astype(str).isin([reference, comparison])].dropna(subset=[metric]).copy()
                estimate = sub.loc[sub["Guidance"].astype(str).eq(comparison), metric].mean() - sub.loc[sub["Guidance"].astype(str).eq(reference), metric].mean()
                ids = sub["ParticipantID"].unique()
                positions = {pid: np.flatnonzero(sub["ParticipantID"].to_numpy() == pid) for pid in ids}
                boots = []
                for _ in range(BOOT_N):
                    sampled = rng.choice(ids, len(ids), replace=True)
                    take = np.concatenate([positions[pid] for pid in sampled])
                    boot = sub.iloc[take]
                    means = boot.groupby("Guidance", observed=True)[metric].mean()
                    if reference in means.index and comparison in means.index:
                        boots.append(means[comparison] - means[reference])
                arr = np.asarray(boots, float)
                pair_rows.append({
                    "Scope": scope, "Smoke": smoke, "Reference": reference, "Comparison": comparison,
                    "Metric": metric, "Estimate": estimate, "CI95Low": np.quantile(arr, 0.025),
                    "CI95High": np.quantile(arr, 0.975), "PValueBootstrap": boot_p(arr),
                    "BootstrapSuccessN": len(arr),
                    "Interpretation": "descriptive_process_contrast_not_spatially_identified_causal_effect",
                })
    pairs = pd.DataFrame(pair_rows)
    pairs["PValueFDR"] = np.nan
    for (smoke, metric), idx in pairs.groupby(["Smoke", "Metric"]).groups.items():
        pairs.loc[idx, "PValueFDR"] = multipletests(pairs.loc[idx, "PValueBootstrap"], method="fdr_bh")[1]
    return pairs, pd.DataFrame(mean_rows)


def main() -> None:
    visual = pd.read_csv(DATA / "visual_information_trial_features.csv")
    aoi = pd.read_csv(DATA / "ar_information_acquisition_features.csv")
    visual_metrics = [
        "HeadAngularPathRate_deg_s", "FixationTimeProportion", "SaccadeRate_per_min",
        "EyeGazeScanpathRate_deg_s", "GazeP90Dispersion_deg", "GazeDirectionEntropy",
    ]
    aoi_metrics = [
        "GuideAcquired", "PromptGuideAcquisitionScore", "GuideDwellProportion", "GuideVisitRate_per_min",
    ]
    visual_pairs, visual_means = contrast_table(visual, GUIDANCE_ORDER, visual_metrics, "all_five_modes")
    aoi_pairs, aoi_means = contrast_table(aoi, AR_ORDER, aoi_metrics, "AR_modes_only")
    visual_pairs.to_csv(TABLE / "five_mode_visual_information_pairwise.csv", index=False, encoding="utf-8-sig")
    aoi_pairs.to_csv(TABLE / "ar_information_acquisition_pairwise.csv", index=False, encoding="utf-8-sig")
    pd.concat([visual_means, aoi_means], ignore_index=True).to_csv(
        TABLE / "information_metric_group_means_bootstrap.csv", index=False, encoding="utf-8-sig"
    )
    print({
        "visual_FDR_lt_0.05": int(visual_pairs["PValueFDR"].lt(0.05).sum()),
        "aoi_FDR_lt_0.05": int(aoi_pairs["PValueFDR"].lt(0.05).sum()),
    })


if __name__ == "__main__":
    main()
