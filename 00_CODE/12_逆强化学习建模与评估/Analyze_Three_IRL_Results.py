"""Audit three saved IRL results without fitting models or rebuilding caches.

Natural-frame metrics are recalculated on identical saved labels. Official V6
metrics retain their hierarchical weighting and are reported separately.
Participant block bootstrap retains every frame of each sampled participant.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = {
    "V6": "20261003_014907_277732_V6",
    "V6_1": "20261003_020827_592825_V6_1",
    "V3OptimizeBank": "20260927_200838_252398_V3OptimizeBank",
}
CHANNELS = ["StaticSign", "ArrowRibbon", "MiniMap", "TextBar"]
CONDITIONS = [*CHANNELS, "Full"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc_parts(y, score, person):
    """Positive/negative person-pair contributions, including half credit for ties."""
    people = np.unique(person)
    _, bins = np.unique(score, return_inverse=True)
    p = np.searchsorted(people, person)
    positive = np.zeros((bins.max() + 1, len(people)), dtype=float)
    negative = np.zeros_like(positive)
    np.add.at(positive, (bins[y == 1], p[y == 1]), 1)
    np.add.at(negative, (bins[y == 0], p[y == 0]), 1)
    pairs = positive.T @ (np.cumsum(negative, axis=0) - .5 * negative)
    return people, pairs, positive.sum(axis=0), negative.sum(axis=0)


def bootstrap(parts, counts):
    _, pairs, positive, negative = parts
    denominator = (counts @ positive) * (counts @ negative)
    numerator = np.einsum("bi,ij,bj->b", counts, pairs, counts)
    return np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)


def metrics(y, probability, utility, baseline):
    p = np.clip(probability.astype(float), 1e-7, 1-1e-7)
    b = np.clip(baseline.astype(float), 1e-7, 1-1e-7)
    loss = lambda q: float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
    brier = float(np.mean((p-y)**2))
    baseline_brier = float(np.mean((b-y)**2))
    return dict(n=len(y), observed_fraction=float(y.mean()), predicted_fraction=float(p.mean()),
                policy_auc=float(roc_auc_score(y, p)), spatial_auc=float(roc_auc_score(y, utility)),
                policy_ap=float(average_precision_score(y, p)), spatial_ap=float(average_precision_score(y, utility)),
                ap_constant_reference=float(y.mean()), nll=loss(p), baseline_nll=loss(b),
                nll_gain=loss(b)-loss(p), brier=brier, baseline_brier=baseline_brier,
                brier_skill=1-brier/baseline_brier)


def run(output, replicates=500):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new --output directory")
    rng = np.random.default_rng(20261003)
    loaded, source_hashes, selections, official, source_splits = {}, {}, {}, [], {}
    for model, run_name in RUNS.items():
        folder = HERE / "Results_IRL" / run_name
        best = folder / "Best_Model"
        evaluation = json.loads((best / "evaluation.json").read_text(encoding="utf-8"))
        config = json.loads((best / "config.json").read_text(encoding="utf-8"))
        manifest_path = Path(config["dataset"]) / "manifest.json"
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes[str(manifest_path)] = sha(manifest_path)
        source_splits[model] = source_manifest["participant_splits"]
        data = dict(np.load(best / "evaluation_predictions.npz"))
        loaded[model] = (evaluation, data, best)
        selections[model] = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        files = [best / n for n in ["config.json", "evaluation.json", "evaluation_predictions.npz", "reward_rules.csv"]]
        files += list(best.rglob("reward.pt"))
        source_hashes.update({str(f): sha(f) for f in files})
        for split in ["train", "val", "test"]:
            report = evaluation[split]
            for channel in CHANNELS if model != "V3OptimizeBank" else [*CHANNELS[1:], "Full"]:
                d = (report["by_channel_condition"]["StaticSign/StaticSign"] if channel == "StaticSign"
                     else report["by_guidance"][channel])
                official.append(dict(model=model, split=split, channel=channel, n=d["n"],
                                     policy_auc=d["auc"], spatial_auc=d["spatial_reward_auc"],
                                     nll=d.get("balanced_nll", d["participant_macro_nll"]),
                                     weighting="hierarchical" if model != "V3OptimizeBank" else "natural-frame AUC; participant NLL"))
    if any(splits != source_splits["V6"] for splits in source_splits.values()):
        raise ValueError("Source participant splits differ")
    geometries = [json.loads((best / "config.json").read_text())["geometry_sha256"] for _, _, best in loaded.values()]
    if len(set(geometries)) != 1:
        raise ValueError("Source geometry hashes differ")
    rows, interval_rows, difference_rows = [], [], []
    scopes = [("baseline_static", "StaticSign"), *[("own_plus_full", g) for g in CHANNELS[1:]],
              *[("own_condition", g) for g in CHANNELS[1:]], ("full_any_guidance", "Full")]
    for split in tqdm(["val", "test"], desc="Audit saved predictions"):
        for scope, channel in scopes:
            local = {}
            for model, (evaluation, data, best) in loaded.items():
                old = model == "V3OptimizeBank"
                if old and scope not in ["own_condition", "full_any_guidance"]:
                    continue
                if not old and channel == "Full":
                    continue
                gi = ([*CHANNELS[1:], "Full"] if old else CHANNELS).index(channel)
                mask = data[split+"_guidance"] == gi
                if scope in ["baseline_static", "own_condition"] and not old:
                    mask &= data[split+"_condition"] == CONDITIONS.index(channel)
                y, p, u, person = [data[split+"_"+key][mask] for key in ["action", "probability", "utility", "person"]]
                # Each condition's natural training prevalence; no validation/test fitting.
                if old:
                    b = np.full(len(y), evaluation["train"]["by_guidance"][channel]["observed_fraction"])
                else:
                    cond = data[split+"_condition"][mask]
                    b = np.empty(len(y), dtype=float)
                    for ci in np.unique(cond):
                        b[cond == ci] = evaluation["train"]["by_channel_condition"][channel+"/"+CONDITIONS[ci]]["observed_fraction"]
                rows.append(dict(model=model, split=split, scope=scope, channel=channel,
                                 people=len(np.unique(person)), **metrics(y, p, u, b)))
                local[model] = (y, p, u, person)
            if not local:
                continue
            reference = next(iter(local.values()))
            for model, values in local.items():
                if not np.array_equal(values[0], reference[0]) or not np.array_equal(values[3], reference[3]):
                    raise ValueError("Saved labels/participant order differ in "+scope+"/"+channel)
            people = np.unique(reference[3])
            draws = rng.integers(len(people), size=(replicates, len(people)))
            counts = np.stack([np.bincount(d, minlength=len(people)) for d in draws])
            distributions = {}
            for model, (y, p, u, person) in local.items():
                for metric, score in [("policy_auc", p), ("spatial_auc", u)]:
                    parts = auc_parts(y, score, person)
                    point = bootstrap(parts, np.ones((1, len(people))))[0]
                    if abs(point-roc_auc_score(y, score)) > 1e-10:
                        raise AssertionError("Person-pair bootstrap does not reproduce ROC AUC")
                    values = bootstrap(parts, counts)
                    distributions[(model, metric)] = values
                    ci = np.nanquantile(values, [.025, .975])
                    interval_rows.append(dict(model=model, split=split, scope=scope, channel=channel,
                                              metric=metric, estimate=point, low=ci[0], high=ci[1],
                                              people=len(people), replicates=replicates,
                                              valid_replicates=int(np.isfinite(values).sum())))
            if "V6_1" in local:
                for other in ["V6", "V3OptimizeBank"]:
                    if other not in local:
                        continue
                    for metric in ["policy_auc", "spatial_auc"]:
                        delta = distributions[("V6_1", metric)]-distributions[(other, metric)]
                        ci = np.nanquantile(delta, [.025, .975])
                        difference_rows.append(dict(split=split, scope=scope, channel=channel, metric=metric,
                                                    contrast="V6_1 - "+other, low=ci[0], high=ci[1], people=len(people),
                                                    valid_replicates=int(np.isfinite(delta).sum())))
    # Pure geometry-grid comparisons are descriptive, not independent observations.
    fields, reconstruction_errors = {}, {}
    sys.path.insert(0, str(HERE))
    from Spatial_IRL_Rule import SpatialIRLRule as LegacyRule
    from Spatial_IRL_Rule_V6 import SpatialIRLRule as ChannelRule
    grid_rows = []
    for model, (_, _, best) in loaded.items():
        version = "V4" if model == "V3OptimizeBank" else "V6"
        f = best / ("Results_Global_Spatial_Field_"+version) / "global_spatial_rule_field.csv.gz"
        source_hashes[str(f)] = sha(f)
        fields[model] = pd.read_csv(f)
        rule = (LegacyRule if model == "V3OptimizeBank" else ChannelRule)(best)
        errors = []
        for channel, part in fields[model].groupby("guidance"):
            pred = rule.query_features(part, channel)
            errors.append(float(np.max(np.abs(pred-part.relative_information_utility.to_numpy()))))
            if not np.allclose(pred, part.relative_information_utility, atol=2e-5, rtol=1e-5):
                raise ValueError("Saved field differs from fitted weights")
        reconstruction_errors[model] = max(errors)
    keys = ["guidance", "smoke", "floor", "x", "z"]
    for a, b in [("V6", "V6_1"), ("V3OptimizeBank", "V6_1")]:
        common = fields[a].merge(fields[b], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        for g, part in common.groupby("guidance"):
            x, y = part.relative_information_utility_a.to_numpy(), part.relative_information_utility_b.to_numpy()
            top_a, top_b = x >= np.quantile(x, .9), y >= np.quantile(y, .9)
            grid_rows.append(dict(contrast=b+" vs "+a, channel=g, cells=len(part),
                                  spearman=float(spearmanr(x,y).statistic),
                                  top10_retention=float((top_a & top_b).sum()/top_a.sum()),
                                  mean_delta=float((y-x).mean())))
    frame = pd.DataFrame(rows)
    common_macro = frame[frame.scope.isin(["baseline_static", "own_plus_full"])].groupby(["model","split"])[["policy_auc","spatial_auc","nll"]].mean().reset_index()
    own_macro = frame[frame.scope.eq("own_condition")].groupby(["model","split"])[["policy_auc","spatial_auc","nll"]].mean().reset_index()
    tables = dict(natural_frame_metrics=frame, participant_auc_intervals=pd.DataFrame(interval_rows),
                  paired_auc_differences=pd.DataFrame(difference_rows), common_scope_natural_macro=common_macro,
                  own_condition_natural_macro=own_macro, official_scope_metrics=pd.DataFrame(official),
                  raw_grid_comparison=pd.DataFrame(grid_rows))
    for f, checksum in source_hashes.items():
        if sha(f) != checksum:
            raise RuntimeError("Source artifact changed: "+f)
    output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(output/(name+".csv"), index=False, encoding="utf-8-sig")
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman","DejaVu Serif"], "font.size":10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = [("Common baseline / own + Full", ["baseline_static", "own_plus_full"], CHANNELS, ["V6", "V6_1"]),
              ("Same standalone-condition frames", ["own_condition"], CHANNELS[1:], list(RUNS))]
    colours = {"V6":"#31688e", "V6_1":"#b35806", "V3OptimizeBank":"#539b59"}
    for ax, (title, scope_names, channel_names, models) in zip(axes, groups):
        for mi, model in enumerate(models):
            part = frame[frame.model.eq(model) & frame.split.eq("test") & frame.scope.isin(scope_names)].set_index("channel").loc[channel_names]
            ax.scatter(part.spatial_auc, np.arange(len(channel_names))+(mi-(len(models)-1)/2)*.14,
                       label=model, color=colours[model], s=32)
        ax.set_yticks(range(len(channel_names)), channel_names)
        ax.invert_yaxis(); ax.set_xlim(.45,.95); ax.axvline(.5,color="gray",linestyle=":")
        ax.set_xlabel("Spatial reward AUC | natural frames"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"test_spatial_auc_comparison.png", dpi=300); plt.close(fig)
    manifest = dict(status="complete", source_hashes=source_hashes, script_sha256=sha(__file__),
                    no_training=True, evaluation_weighting="natural-frame; official hierarchical metrics kept separately",
                    bootstrap_unit="participant retaining all frames; paired draws for common model scopes", replicates=replicates,
                    frame_alignment="source provenance plus exact saved action and participant array equality",
                    participant_splits_equal=True, geometry_hashes_equal=True,
                    fitted_field_reconstruction_max_abs_error=reconstruction_errors,
                    baseline="condition-specific natural training prevalence; no validation/test fit",
                    limitations=["One reused participant split", "No adjustment for candidate selection or development reuse",
                                 "Standalone new-channel subsets contain only 2 or 3 test participants",
                                 "V6/V6_1 cached state/frame folders referenced in selection.json are absent",
                                 "Full union label cannot be compared with separate binary channel labels"],
                    outputs=list(tables))
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMMON_NATURAL", common_macro.to_json(orient="records"))
    print("OWN_CONDITION_NATURAL", own_macro.to_json(orient="records"))
    print("TEST_BASELINES", frame[(frame.split.eq("test")) & frame.scope.isin(["baseline_static","own_plus_full"])].to_json(orient="records"))
    print("OUTPUT", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"Results_Three_Model_Assessment_20261003")
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    run(args.output, args.replicates)


if __name__ == "__main__":
    main()
