"""Dispatch upstream data/assumptions to single-factor tests and reports.

No raw-data discovery, aggregation, normality testing or automatic assumption
selection occurs here. Repeated observations must already be correctly paired.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import warnings

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway


def _result(method, statistic, p, **extra):
    statistic, p = float(statistic), float(p)
    if np.isnan(statistic) or not np.isfinite(p):
        raise ValueError(f"{method}: undefined statistic/p-value; inspect constant or degenerate data")
    return dict(method=method, statistic=statistic, p_value=p, **extra)


def _pair(a, b, distribution, design, equal_var):
    if distribution == "parametric":
        if design == "repeated":
            r = stats.ttest_rel(a, b)
            method = "paired_t"
        else:
            r = stats.ttest_ind(a, b, equal_var=equal_var)
            method = "student_t" if equal_var else "welch_t"
        return _result(method, r.statistic, r.pvalue, df=float(r.df))
    if design == "independent":
        # Asymptotic method handles ties; recorded explicitly for reproducibility.
        r = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        return _result("mann_whitney_u", r.statistic, r.pvalue,
                       calculation="asymptotic, tie correction, continuity correction")
    if np.all(a == b):
        return _result("wilcoxon_signed_rank", 0, 1,
                       calculation="all paired differences zero: p=1 convention")
    r = stats.wilcoxon(a - b, alternative="two-sided", zero_method="wilcox", method="approx")
    return _result("wilcoxon_signed_rank", r.statistic, r.pvalue,
                   calculation="normal approximation; zero_method=wilcox; no continuity correction")


def _rm_anova(arrays, sphericity):
    x = np.column_stack(arrays)
    n, k = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    effects = centered.mean(axis=0)
    ss_effect = n * np.sum(effects ** 2)
    ss_error = np.sum((centered - effects) ** 2)
    if ss_error <= np.finfo(float).eps * np.sum(centered ** 2):
        raise ValueError("Repeated-measures ANOVA has zero/numerically degenerate residual variance")
    df1, df2 = k - 1, (n - 1) * (k - 1)
    f = (ss_effect / df1) / (ss_error / df2)
    # GG epsilon from the double-centered within-subject covariance matrix.
    covariance = np.cov(centered, rowvar=False, ddof=1)
    epsilon = float(np.clip(np.trace(covariance) ** 2 /
                           ((k - 1) * np.sum(covariance ** 2)), 1 / (k - 1), 1))
    correction = 1.0 if sphericity is True else epsilon
    return _result("repeated_measures_anova", f,
                   stats.f.sf(f, df1 * correction, df2 * correction),
                   df1=df1 * correction, df2=df2 * correction,
                   df1_uncorrected=df1, df2_uncorrected=df2,
                   p_uncorrected=float(stats.f.sf(f, df1, df2)),
                   epsilon_gg=epsilon,
                   correction="none" if sphericity is True else "greenhouse_geisser",
                   partial_eta_squared=float(ss_effect / (ss_effect + ss_error)))


def _dunn(arrays):
    values = np.concatenate(arrays)
    ranks = stats.rankdata(values)
    n = len(values)
    _, ties = np.unique(values, return_counts=True)
    variance = n * (n + 1) / 12 - np.sum(ties ** 3 - ties) / (12 * (n - 1))
    if variance <= 0:
        raise ValueError("Dunn test is undefined when all observations are equal")
    offsets = np.cumsum([0] + [len(a) for a in arrays])
    means = [ranks[offsets[i]:offsets[i + 1]].mean() for i in range(len(arrays))]
    rows = []
    for i, j in combinations(range(len(arrays)), 2):
        z = (means[i] - means[j]) / np.sqrt(variance * (1 / len(arrays[i]) + 1 / len(arrays[j])))
        rows.append((i, j, _result("dunn", z, 2 * stats.norm.sf(abs(z)))))
    return rows


def analyze_significance(groups, *, distribution, design, group_count=None,
                         equal_var=None, sphericity=None, subject_ids=None,
                         assume_aligned=False, missing="raise", alpha=0.05,
                         posthoc="auto", correction="holm", metric="value",
                         metadata=None):
    """Return a JSON-compatible report for one metric / one factor.

    groups: ordered mapping {group_name: one-dimensional numeric sequence}.
    distribution: 'parametric' or 'nonparametric' (upstream decision).
    design: 'independent' or 'repeated'. Repeated arrays need identical ordered
        subject_ids per group, or an explicit assume_aligned=True assertion.
    equal_var: required bool for independent parametric tests.
    sphericity: True uses uncorrected RM ANOVA; False/None uses GG correction.
    missing: 'raise' or 'omit'; repeated designs omit entire subject rows.
    posthoc: 'auto' (only after significant omnibus), 'always', or 'none'.
    correction: 'holm' or 'bonferroni'; Tukey uses its own family correction.
    """
    if distribution not in ("parametric", "nonparametric"):
        raise ValueError("distribution must be parametric or nonparametric")
    if design not in ("independent", "repeated"):
        raise ValueError("design must be independent or repeated; mixed designs are unsupported")
    if not isinstance(groups, dict) or len(groups) < 2:
        raise ValueError("groups must be a mapping with at least two groups")
    names = list(groups)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("group names must be nonempty strings")
    if group_count is not None and (type(group_count) is not int or group_count != len(names)):
        raise ValueError("group_count does not match groups")
    if equal_var is not None and type(equal_var) is not bool:
        raise ValueError("equal_var must be bool or None")
    if sphericity is not None and type(sphericity) is not bool:
        raise ValueError("sphericity must be bool or None")
    if type(assume_aligned) is not bool:
        raise ValueError("assume_aligned must be bool")
    if design == "independent" and distribution == "parametric" and equal_var is None:
        raise ValueError("independent parametric tests require explicit equal_var=True/False")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if missing not in ("raise", "omit") or posthoc not in ("auto", "always", "none"):
        raise ValueError("invalid missing or posthoc policy")
    if correction not in ("holm", "bonferroni"):
        raise ValueError("correction must be holm or bonferroni")
    arrays = [np.asarray(groups[name], dtype=float) for name in names]
    if any(a.ndim != 1 for a in arrays):
        raise ValueError("each group must be a one-dimensional numeric array")
    if any(np.isinf(a).any() for a in arrays):
        raise ValueError("infinite observations are not allowed")
    n_input = [len(a) for a in arrays]
    retained_ids = None
    if design == "repeated":
        if len(set(n_input)) != 1:
            raise ValueError("repeated arrays must have equal lengths before missing-value removal")
        if subject_ids is None:
            if not assume_aligned:
                raise ValueError("repeated design requires subject_ids or assume_aligned=True")
        else:
            if not isinstance(subject_ids, dict) or set(subject_ids) != set(names):
                raise ValueError("subject_ids must contain exactly the group names")
            ids = [list(subject_ids[name]) for name in names]
            for v in ids:
                if len(v) != n_input[0] or any(not isinstance(i, (str, int)) or isinstance(i, bool) for i in v):
                    raise ValueError("subject IDs must be strings/integers and match array length")
                if len(set(v)) != len(v):
                    raise ValueError("duplicate subject IDs")
            if any(v != ids[0] for v in ids[1:]):
                raise ValueError("subject IDs differ in membership/order; align upstream first")
            retained_ids = ids[0]
    elif subject_ids is not None:
        raise ValueError("subject_ids is only supported for repeated designs")
    if any(np.isnan(a).any() for a in arrays):
        if missing == "raise":
            raise ValueError("missing observations; use missing='omit' explicitly")
        if design == "repeated":
            keep = ~np.any(np.isnan(np.column_stack(arrays)), axis=1)
            arrays = [a[keep] for a in arrays]
            if retained_ids is not None:
                retained_ids = [i for i, valid in zip(retained_ids, keep) if valid]
        else:
            arrays = [a[~np.isnan(a)] for a in arrays]
    if any(len(a) < 2 for a in arrays):
        raise ValueError("each group needs at least two usable observations")
    notes = ["Distribution/design/variance assumptions are supplied by upstream; not tested here.",
             "Tests are two-sided; one metric and one factor per call. No mixed-design inference.",
             "Posthoc adjustment controls comparisons within this call only, not across metrics."]
    if distribution == "nonparametric":
        notes.append("Rank tests do not generally test means or medians without additional distributional assumptions.")
        notes.append("Rank-test p-values use asymptotic approximations; small samples require caution.")
    if design == "repeated" and subject_ids is None:
        notes.append("Pairing is positional and explicitly asserted by caller; subject identity was not verified.")
    if design == "repeated" and distribution == "parametric":
        notes.append("Paired t assumes normal differences; RM ANOVA requires appropriate within-subject residual assumptions.")
    if sphericity is None and design == "repeated" and distribution == "parametric" and len(arrays) > 2:
        notes.append("Sphericity unspecified: conservatively using GG correction without Mauchly testing.")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if len(arrays) == 2:
            omnibus = _pair(*arrays, distribution, design, equal_var)
        elif distribution == "parametric" and design == "independent":
            r = anova_oneway(arrays, use_var="equal" if equal_var else "unequal")
            omnibus = _result("one_way_anova" if equal_var else "welch_anova", r.statistic,
                               r.pvalue, df1=float(r.df[0]), df2=float(r.df[1]))
        elif distribution == "parametric":
            omnibus = _rm_anova(arrays, sphericity)
        elif design == "independent":
            r = stats.kruskal(*arrays)
            omnibus = _result("kruskal_wallis", r.statistic, r.pvalue, df=len(arrays) - 1)
        else:
            r = stats.friedmanchisquare(*arrays)
            omnibus = _result("friedman", r.statistic, r.pvalue, df=len(arrays) - 1,
                               kendalls_w=float(r.statistic / (len(arrays[0]) * (len(arrays) - 1))))
        omnibus["significant"] = omnibus["p_value"] < alpha
        rows = []
        do_posthoc = len(arrays) > 2 and (posthoc == "always" or (posthoc == "auto" and omnibus["significant"]))
        if do_posthoc:
            if distribution == "parametric" and design == "independent" and equal_var:
                r = stats.tukey_hsd(*arrays)
                for i, j in combinations(range(len(arrays)), 2):
                    rows.append(dict(group_a=names[i], group_b=names[j], method="tukey_hsd",
                                     statistic=float(r.statistic[i, j]), statistic_kind="mean_difference",
                                     p_adjusted=float(r.pvalue[i, j]), correction="tukey_hsd"))
            else:
                if distribution == "nonparametric" and design == "independent":
                    pairs = _dunn(arrays)
                else:
                    pairs = [(i, j, _pair(arrays[i], arrays[j], distribution, design, equal_var))
                             for i, j in combinations(range(len(arrays)), 2)]
                adjusted = multipletests([r["p_value"] for _, _, r in pairs], alpha=alpha, method=correction)[1]
                for (i, j, r), p in zip(pairs, adjusted):
                    rows.append(dict(group_a=names[i], group_b=names[j], **r,
                                     p_adjusted=float(p), correction=correction))
            for row in rows:
                if not np.isfinite(row["p_adjusted"]):
                    raise ValueError("posthoc comparison has undefined p-value")
                row["significant"] = row["p_adjusted"] < alpha
        notes.extend(str(w.message) for w in caught)
    descriptive = [dict(group=name, n_input=before, n=len(a), n_removed=before - len(a),
                        mean=float(a.mean()), sd=float(a.std(ddof=1)), median=float(np.median(a)),
                        q1=float(np.quantile(a, .25)), q3=float(np.quantile(a, .75)))
                   for name, before, a in zip(names, n_input, arrays)]
    return dict(schema_version="1.0", metric=metric,
                parameters=dict(distribution=distribution, design=design, group_count=len(arrays),
                                equal_var=equal_var, sphericity=sphericity, alpha=alpha,
                                missing=missing, posthoc=posthoc, correction=correction),
                descriptive=descriptive, omnibus=omnibus, posthoc=rows,
                posthoc_status="performed" if do_posthoc else ("not_applicable" if len(arrays) == 2
                                else "disabled" if posthoc == "none" else "omnibus_not_significant"),
                retained_subject_ids=retained_ids, notes=notes, metadata=metadata or {},
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, statsmodels=statsmodels.__version__))


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else None
    return value


def write_report(report, output, *, force=False):
    """Write report.json, report.md and posthoc.csv; refuse overwrite by default."""
    import csv
    import io
    output = Path(output)
    paths = [output / name for name in ("report.json", "report.md", "posthoc.csv")]
    if not force and any(p.exists() for p in paths):
        raise FileExistsError(f"Report exists in {output}; choose another directory or force=True")
    r = report["omnibus"]
    lines = [f"# 显著性分析：{report['metric']}", "", f"- 方法：{r['method']}",
             f"- 统计量：{r['statistic']:.8g}", f"- p 值：{r['p_value']:.8g}",
             f"- α：{report['parameters']['alpha']}",
             f"- 结论：{'拒绝原假设' if r['significant'] else '未拒绝原假设（不等于证明无差异）'}", "",
             "## 参数与检验详情", "", "```json",
             json.dumps(_json_safe(dict(parameters=report['parameters'], test=r)), ensure_ascii=False, indent=2),
             "```", "", "## 描述统计", "",
             "| 组 | 输入 n | 有效 n | 删除 n | 均值 | SD | 中位数 | Q1 | Q3 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["descriptive"]:
        name = row['group'].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {name} | " + " | ".join(f"{row[k]:.8g}" for k in
                     ("n_input", "n", "n_removed", "mean", "sd", "median", "q1", "q3")) + " |")
    lines += ["", "## 事后比较", "", f"状态：{report['posthoc_status']}", "",
              "| A | B | 方法 | 原始 p | 校正 p | 校正方法 | 显著 |", "|---|---|---|---:|---:|---|---|"]
    for row in report["posthoc"]:
        cells = [row['group_a'], row['group_b'], row['method'], row.get('p_value', '—'),
                 row['p_adjusted'], row['correction'], row['significant']]
        lines.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in cells) + ' |')
    lines += ["", "## 口径与提示", ""] + [f"- {n}" for n in report["notes"]]
    lines += ["", "## 上游元数据", "", "```json", json.dumps(report['metadata'], ensure_ascii=False, indent=2), "```"]
    json_text = json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
    buffer = io.StringIO(newline="")
    fields = ["group_a", "group_b", "method", "statistic", "p_value", "p_adjusted", "correction", "significant"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report['posthoc'])
    output.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, (json_text, '\n'.join(lines) + '\n', buffer.getvalue())):
        with path.open('w' if force else 'x', encoding='utf-8-sig' if path.suffix == '.csv' else 'utf-8', newline='') as f:
            f.write(content)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='JSON object matching analyze_significance arguments')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'Results_Significance')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    from tqdm import tqdm
    with tqdm(total=3, desc='Significance', unit='step') as progress:
        request = json.loads(args.input.read_text(encoding='utf-8-sig'))
        progress.update()
        report = analyze_significance(**request)
        progress.update()
        paths = write_report(report, args.output, force=args.force)
        progress.update()
    print('\n'.join(str(p) for p in paths))


if __name__ == '__main__':
    main()
