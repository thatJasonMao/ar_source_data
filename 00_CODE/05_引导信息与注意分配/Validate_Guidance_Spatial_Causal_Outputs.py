"""Deterministic integrity checks for the final multimodal analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Results_Guidance_Spatial_Causal_Analysis"
DOC = ROOT / "Documentation_Guidance_Spatial_Causal_Analysis"
DATA = OUT / "data"
TABLE = OUT / "tables"
FIG = OUT / "figures"
NOTE = OUT / "notes"


def main() -> None:
    trials = pd.read_csv(DATA / "analysis_trials.csv")
    all_trials = pd.read_csv(DATA / "analysis_trials_all_288.csv")
    geometry = pd.read_csv(TABLE / "exit_spatial_difficulty_by_spawn.csv")
    outcomes = pd.read_csv(TABLE / "five_mode_spatial_standardized_summary.csv")
    visual = pd.read_csv(DATA / "visual_information_trial_features.csv")
    aoi = pd.read_csv(DATA / "ar_information_acquisition_features.csv")
    quality = pd.read_csv(TABLE / "eye_tracking_data_quality_audit.csv")
    thresholds = pd.read_csv(TABLE / "eye_event_threshold_sensitivity.csv")
    hrv = pd.read_csv(DATA / "hrv_trial_features.csv")
    hrv_bins = pd.read_csv(DATA / "hrv_progress_bins.csv")
    movement_bins = pd.read_csv(DATA / "movement_progress_bins.csv")
    multimodal = pd.read_csv(DATA / "multimodal_trial_features.csv")
    ebm = pd.read_csv(TABLE / "ebm_grouped_cv_performance.csv")
    serial = pd.read_csv(TABLE / "gcomputation_temporal_serial_mediation.csv")
    order = pd.read_csv(TABLE / "trial_order_design_audit.csv")
    sensitivity = pd.read_csv(TABLE / "spatial_standardization_sensitivity_and_ranking.csv")
    extension = json.loads((DATA / "multimodal_extension_summary.json").read_text(encoding="utf-8"))
    rest = pd.read_csv(TABLE / "hrv_rest_baseline_quality_audit.csv")
    zone_bins = pd.read_csv(DATA / "spatiotemporal_zone_bins.csv")
    spatial_ebm = pd.read_csv(TABLE / "spatiotemporal_ebm_grouped_cv.csv")
    spatial_mediation = pd.read_csv(TABLE / "gcomputation_spatiotemporal_multifactor_mediation.csv")
    spatial_extension = json.loads((DATA / "spatiotemporal_extension_summary.json").read_text(encoding="utf-8"))
    revision = json.loads((DATA / "requested_revision_summary.json").read_text(encoding="utf-8"))
    tiered = pd.read_csv(TABLE / "gcomputation_tiered_mediation.csv")
    hrv_stress_tests = pd.read_csv(TABLE / "hrv_stress_guidance_tests.csv")
    spatial_importance = pd.read_csv(TABLE / "spatiotemporal_ebm_importance.csv")

    checks: dict[str, bool] = {}
    checks["primary_270_trials_45_participants"] = len(trials) == 270 and trials["ParticipantID"].nunique() == 45
    checks["six_trials_each_no_duplicates"] = (
        trials.groupby("ParticipantID").size().eq(6).all()
        and not trials.duplicated(["Scenario", "ParticipantID"]).any()
    )
    checks["sensitivity_288_trials_48_participants"] = len(all_trials) == 288 and all_trials["ParticipantID"].nunique() == 48
    checks["excluded_18_aoi_trials"] = int(all_trials["AOICohortExcluded"].sum()) == 18
    checks["visual_primary_factors_complete"] = len(visual) == 270 and not visual[
        ["HeadAngularPathRate_deg_s", "FixationTimeProportion", "GazeDirectionEntropy"]
    ].isna().any().any()
    checks["aoi_180_available_two_never_acquired"] = (
        len(aoi) == 180
        and pd.to_numeric(aoi["AOIAvailable"], errors="coerce").eq(1).all()
        and pd.to_numeric(aoi["GuideAcquired"], errors="coerce").eq(0).sum() == 2
    )
    checks["all_eye_files_matched_quality_ok"] = (
        len(quality) == 288
        and pd.to_numeric(quality["EyeFileMatched"], errors="coerce").eq(1).all()
        and quality["ValidEyeSampleProportion"].median() > 0.98
    )
    checks["ivt_threshold_grid_3x3"] = (
        set(thresholds["VelocityThreshold_deg_s"].unique()) == {30.0, 60.0, 100.0}
        and set(thresholds["MinimumFixationDuration_ms"].unique()) == {80.0, 100.0, 120.0}
    )
    checks["hrv_270_trials_251_usable_45_people"] = (
        len(hrv) == 270
        and int(pd.to_numeric(hrv["HRVPrimaryUsable"], errors="coerce").sum()) == 251
        and hrv.loc[pd.to_numeric(hrv["HRVPrimaryUsable"], errors="coerce").eq(1), "ParticipantID"].nunique() == 45
    )
    checks["rest_baseline_48_of_48_usable"] = (
        len(rest) == 48
        and rest["ParticipantID"].nunique() == 48
        and pd.to_numeric(rest["RestPrimaryUsable"], errors="coerce").eq(1).all()
        and pd.to_numeric(rest["RestObservedSpan_s"], errors="coerce").min() >= 299
    )
    checks["hrv_uses_resting_baseline_deltas"] = all(
        c in hrv.columns for c in ["MeanHRBaselineDelta_bpm", "LogRMSSDBaselineDelta", "SDNNBaselineDelta_ms"]
    ) and hrv.loc[pd.to_numeric(hrv["HRVPrimaryUsable"], errors="coerce").eq(1), "MeanHRBaselineDelta_bpm"].notna().all()
    checks["ten_bin_multimodal_axes"] = (
        len(hrv_bins) == 2700
        and len(movement_bins) == 2700
        and hrv_bins.groupby(["Scenario", "ParticipantID"]).size().eq(10).all()
        and movement_bins.groupby(["Scenario", "ParticipantID"]).size().eq(10).all()
    )
    checks["multimodal_270_serial_243"] = len(multimodal) == 270 and extension["serial_complete_trial_n"] == 243
    zone_trial_n = zone_bins[
        ["Scenario", "ParticipantID", "FileName"]
    ].drop_duplicates().shape[0]
    checks["spatiotemporal_extension_dimensions"] = (
        len(zone_bins) == spatial_extension["zone_time_row_n"]
        and zone_trial_n == 270
        and zone_bins["FunctionalZone"].nunique() == 19
        and spatial_extension["functional_zone_n"] == 19
        and spatial_extension["physical_zone_n"] == 39
        and 0 < spatial_extension["mediation_complete_trial_n"] <= 270
    )
    checks["spatial_ebm_no_outcome_leakage"] = (
        spatial_ebm["Scope"].nunique() == 2
        and spatial_ebm.groupby("Scope")["Outcome"].nunique().eq(2).all()
        and float(spatial_ebm.loc[(spatial_ebm["Scope"] == "prechoice_spatiotemporal_processes") & (spatial_ebm["Outcome"] == "ExitCorrect"), "AUC"].iloc[0]) < 0.98
        and not spatial_importance["Term"].str.contains("SpatialDifficultyRatio|TrialOrder|MeanHRBaseline", regex=True).any()
    )
    checks["spatial_mediation_fdr_complete"] = (
        len(spatial_mediation) > 0
        and spatial_mediation.loc[spatial_mediation["Effect"] == "JointIndirect", "PValueFDR"].notna().all()
    )
    joint = tiered.loc[tiered["Effect"] == "JointIndirect"]
    significant_specific = tiered.loc[tiered["Effect"].str.startswith("Specific:") & (tiered["PValueFDR"] < .05), "Effect"]
    checks["tiered_mediation_summary_consistent"] = (
        len(joint) == revision["mediation_joint_test_n"] == 280
        and int((joint["PValueFDR"] < .05).sum())
        == revision["mediation_joint_fdr_significant_n"]
        and significant_specific.str.contains("Backtrack").all()
    )
    checks["hrv_stress_no_pairwise_fdr_signal"] = (
        (hrv_stress_tests.loc[hrv_stress_tests["Test"] == "pairwise", "PValueFDR"] >= .05).all()
        and revision["hrv_primary"].startswith("HRVStressProxy")
    )
    expected_order = {
        ("StaticSign", "HighSmoke"): 1,
        ("StaticSign", "LowSmoke"): 2,
        ("ArrowRibbon", "HighSmoke"): 3,
        ("MiniMap", "HighSmoke"): 3,
        ("TextBar", "HighSmoke"): 3,
        ("ArrowRibbon", "LowSmoke"): 4,
        ("MiniMap", "LowSmoke"): 4,
        ("TextBar", "LowSmoke"): 4,
        ("Full", "HighSmoke"): 5,
        ("Full", "LowSmoke"): 6,
    }
    observed_order = {(r.Guidance, r.Smoke): int(r.MeanTrialOrder) for r in order.itertuples()}
    checks["fixed_trial_order_audited"] = observed_order == expected_order
    checks["ebm_seven_stages_two_outcomes"] = (
        ebm["Stage"].nunique() == 7 and ebm.groupby("Stage")["Outcome"].nunique().eq(2).all()
    )
    checks["serial_mediation_has_design_status_and_fdr"] = (
        len(serial) > 0
        and serial["ComparisonDesignStatus"].notna().all()
        and serial.loc[serial["Path"].str.contains("Indirect", na=False), "PValueFDR"].notna().all()
    )
    formula = trials["EvacuationTime_s"] * trials["ReferenceDistance_AF_EqualWeight_m"] / trials["AStarShortestDistance_m"]
    checks["spatial_time_formula_exact"] = float(np.max(np.abs(formula - trials["StandardizedTime_s"]))) < 1e-9
    af = geometry.loc[geometry["Exit"].isin(list("ABCDEF"))]
    checks["spatial_difficulty_centered_by_start_side"] = af.groupby("StartSide")["SpatialDifficultyRatio"].mean().sub(1).abs().max() < 1e-12
    checks["standardized_outcomes_complete"] = not outcomes[
        ["StandardizedMeanTime_s", "SpatiallyStandardizedCorrectProbability", "StandardizedTimeCI95Low", "StandardizedTimeCI95High"]
    ].isna().any().any()
    ranks = sensitivity.loc[sensitivity["Cohort"].eq("Primary45")].pivot_table(
        index=["Smoke", "Guidance"], columns="Method", values="RankWithinSmokeMethod"
    )
    checks["time_rank_robust_all_methods"] = ranks.nunique(axis=1).eq(1).all()
    checks["all_13_png_svg_and_14_18_png_figures"] = all(
        (FIG / f"{i:02d}_{name}.png").exists() and (FIG / f"{i:02d}_{name}.svg").exists()
        for i, name in [
            (1, "exit_spatial_difficulty"), (2, "raw_vs_spatially_standardized_time"),
            (3, "primary_standardized_outcomes"), (4, "pairwise_standardized_time"),
            (5, "visual_information_by_guidance"), (6, "visual_information_progress"),
            (7, "ar_information_acquisition"), (8, "ebm_layered_importance"),
            (9, "gcomp_layered_mediation"), (10, "movement_progress_curves"),
            (11, "spatial_standardization_sensitivity"), (12, "hrv_progress_curves"),
            (13, "temporal_serial_mediation"),
        ]
    ) and all(
        (FIG / f"{i:02d}_{name}.png").exists()
        for i, name in [
            (14, "functional_zone_visit_heatmap"), (15, "zone_time_spline_significance"),
            (16, "zone_time_pairwise_difference_heatmap"), (17, "spatiotemporal_ebm_importance"),
            (18, "spatiotemporal_serial_mediation"),
        ]
    )
    checks["all_19_27_revision_figures"] = all(
        (FIG / f"{i:02d}_{name}.png").exists() and (FIG / f"{i:02d}_{name}.svg").exists()
        for i, name in [
            (19, "visual_attention_space_time_by_guidance"), (20, "hrv_stress_space_time_by_guidance"),
            (21, "spatial_behavior_space_time_by_guidance"), (22, "directional_space_time_guidance_contrasts"),
            (23, "tiered_mediation"), (24, "hrv_stress_by_guidance"),
            (25, "directional_mechanism_chain"),
            (26, "complete_tiered_mediation_map"),
            (27, "tiered_mediation_pathways"),
        ]
    )
    checks["directional_paths_table_complete"] = (
        (TABLE / "arrowribbon_vs_staticsign_directional_paths.csv").exists()
        and len(pd.read_csv(TABLE / "arrowribbon_vs_staticsign_directional_paths.csv")) >= 15
    )
    checks["key_directional_chain_publication_bundle"] = (
        (TABLE / "key_directional_mediation_chains.csv").exists()
        and len(pd.read_csv(TABLE / "key_directional_mediation_chains.csv")) == 7
        and all((FIG / f"25_directional_mechanism_chain.{ext}").exists() for ext in ["png", "svg", "pdf", "tiff"])
    )
    checks["complete_tiered_mediation_bundle"] = (
        (TABLE / "complete_tiered_mediation_map.csv").exists()
        and len(pd.read_csv(TABLE / "complete_tiered_mediation_map.csv")) == 340
        and all((FIG / f"23_tiered_mediation.{ext}").exists() for ext in ["png", "svg", "pdf", "tiff"])
        and all((FIG / f"26_complete_tiered_mediation_map.{ext}").exists() for ext in ["png", "svg", "pdf", "tiff"])
    )
    checks["tiered_mediation_pathway_bundle_no_pdf"] = (
        (TABLE / "tiered_mediation_pathways.csv").exists()
        and len(pd.read_csv(TABLE / "tiered_mediation_pathways.csv")) == 280
        and all((FIG / f"27_tiered_mediation_pathways.{ext}").exists() for ext in ["png", "svg", "tiff"])
        and not (FIG / "27_tiered_mediation_pathways.pdf").exists()
    )
    report = (DOC / "完整中文工作报告.md").read_text(encoding="utf-8")
    checks["report_utf8_and_current"] = all(
        token in report for token in ["251个", "48/48", "HRV压力代理", "空间锚定", "43个", "5158", "排除", "RMSSD"]
    ) and "\ufffd" not in report
    html = (DOC / "report.html").read_text(encoding="utf-8")
    checks["html_utf8_and_current"] = all(token in html for token in ["251个", "48/48", "HRV压力代理", "空间锚定", "43个", "5158"]) and "\ufffd" not in html

    result = {
        "all_passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "primary_trials": len(trials),
        "primary_participants": trials["ParticipantID"].nunique(),
        "hrv_usable_trials": int(pd.to_numeric(hrv["HRVPrimaryUsable"], errors="coerce").sum()),
        "serial_complete_trials": extension["serial_complete_trial_n"],
        "spatial_mediation_complete_trials": spatial_extension["mediation_complete_trial_n"],
        "figure_png_n": len(list(FIG.glob("*.png"))),
        "figure_svg_n": len(list(FIG.glob("*.svg"))),
    }
    (NOTE / "validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
