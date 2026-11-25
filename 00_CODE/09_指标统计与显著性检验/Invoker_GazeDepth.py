"""Prepare gaze depth values and invoke seven single-factor comparisons.

Run from any working directory. Paths are derived from this file's hierarchy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

try:
    from .Analyze_Significance import analyze_significance, write_report
    from .TrialInput import prepare_trial_data
except ImportError:
    from Analyze_Significance import analyze_significance, write_report
    from TrialInput import prepare_trial_data

METRIC = 'GazeDepth_m'
GUIDANCE = ('StaticSign', 'ArrowRibbon', 'MiniMap', 'TextBar', 'Full')
SPECIFIC = ('ArrowRibbon', 'MiniMap', 'TextBar')
SMOKES = ('HighSmoke', 'LowSmoke')
LABELS = dict(StaticSign='传统标识', ArrowRibbon='光毯', MiniMap='小地图', TextBar='文字', Full='全量信息')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_common(root):
    spec = importlib.util.spec_from_file_location('a5_invoker_common', Path(root) / 'common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common


def prepare_data(raw, common):
    """Align values and groups exclusively to common.load_trials()."""
    return prepare_trial_data(raw, common, METRIC)


def normality(values, label, alpha):
    values = np.asarray(values, dtype=float)
    entry = dict(target=label, test='shapiro_wilk', n=len(values), statistic=None,
                 p_value=None, passed=False, status='tested')
    if len(values) < 3 or len(values) > 5000 or np.ptp(values) == 0:
        entry['status'] = 'not_testable_small_large_or_constant_sample'
        return entry
    result = stats.shapiro(values)
    entry.update(statistic=float(result.statistic), p_value=float(result.pvalue),
                 passed=bool(result.pvalue >= alpha))
    return entry


def build_jobs(data, assumption_alpha):
    jobs = []
    for mode in GUIDANCE:
        subset = data[data.Guidance == mode]
        wide = subset.pivot(index='SubjectID', columns='Smoke', values=METRIC).reindex(columns=SMOKES)
        wide = wide.sort_index()
        unmatched = wide[wide.isna().any(axis=1)]
        pairs = wide.dropna()
        groups = {smoke: pairs[smoke].to_numpy() for smoke in SMOKES}
        check = normality(groups['HighSmoke'] - groups['LowSmoke'], 'HighSmoke-LowSmoke', assumption_alpha)
        jobs.append(dict(id=f'smoke_{mode}', title=f'{LABELS[mode]}：高烟雾 vs 低烟雾',
                         design='repeated', groups=groups,
                         subject_ids={smoke: pairs.index.tolist() for smoke in SMOKES},
                         distribution='parametric' if check['passed'] else 'nonparametric', equal_var=None,
                         assumptions=[check], paired_data=pairs.reset_index(),
                         unmatched=unmatched.reset_index(),
                         selection_reason='对高烟雾减低烟雾的配对差值作 Shapiro-Wilk；通过选配对 t，否则选 Wilcoxon。'))
    for smoke in SMOKES:
        subset = data[(data.Smoke == smoke) & data.Guidance.isin(SPECIFIC)]
        parts = {mode: subset[subset.Guidance == mode].sort_values('SubjectID') for mode in SPECIFIC}
        ids = {mode: set(part.SubjectID) for mode, part in parts.items()}
        if any(ids[a] & ids[b] for a, b in combinations(SPECIFIC, 2)):
            raise ValueError(f'Overlapping subjects in independent comparison: {smoke}')
        groups = {mode: part[METRIC].to_numpy() for mode, part in parts.items()}
        checks = [normality(v, mode, assumption_alpha) for mode, v in groups.items()]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lev = stats.levene(*groups.values(), center='median')
        lev_valid = bool(np.isfinite(lev.statistic) and np.isfinite(lev.pvalue))
        equal = bool(lev_valid and lev.pvalue >= assumption_alpha)
        checks.append(dict(target='all_three_groups', test='levene_median',
                           n=sum(len(v) for v in groups.values()),
                           statistic=float(lev.statistic) if lev_valid else None,
                           p_value=float(lev.pvalue) if lev_valid else None, passed=equal,
                           status='tested' if lev_valid else 'undefined_use_unequal_variance'))
        jobs.append(dict(id=f'guidance_{smoke}', title=f'{smoke}：光毯 vs 小地图 vs 文字',
                         design='independent', groups=groups, subject_ids=None,
                         distribution='parametric' if all(c['passed'] for c in checks[:3]) else 'nonparametric',
                         equal_var=equal, assumptions=checks, sample_data=subset,
                         selection_reason='三组均通过 Shapiro-Wilk 才选参数检验；Levene 中位数检验决定 ANOVA/Welch；否则选 Kruskal-Wallis。'))
    return jobs


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(root, output, *, alpha=.05, assumption_alpha=.05, force=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not 0 < alpha < 1 or not 0 < assumption_alpha < 1:
        raise ValueError('Alpha values must be between zero and one')
    common = load_common(root)
    source = root / 'INDICATORS_GRAPH' / 'gaze_depth' / 'AllSubjects_GazeDepth.csv'
    raw = pd.read_csv(source)
    prepared, audit, evidence = prepare_data(raw, common)
    if not force and (output / 'manifest.json').exists():
        old = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if (old.get('input_sha256') == file_hash(source)
                and old.get('invoker_sha256') == file_hash(__file__)
                and old.get('common_sha256') == file_hash(root / 'common.py')
                and old.get('trial_input_sha256') == file_hash(Path(__file__).with_name('TrialInput.py'))
                and old.get('core_sha256') == file_hash(Path(__file__).with_name('Analyze_Significance.py'))
                and old.get('alpha') == alpha and old.get('assumption_alpha') == assumption_alpha
                and old.get('artifacts') and all((output / p).is_file() for p in old['artifacts'])):
            print(f'Skip completed analysis: {output}; use --force to recompute')
            return pd.read_csv(output / 'summary.csv')
        raise FileExistsError('Existing results differ from this request; use a new output or --force')
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError('Output directory is nonempty; use another output or --force')
    with tqdm(total=10, desc='Gaze depth', unit='step') as progress:
        progress.update()
        jobs = build_jobs(prepared, assumption_alpha)
        progress.update()
        reports = []
        # Compute all seven first; fail before creating a partial result tree.
        for job in jobs:
            report = analyze_significance(
                job['groups'], distribution=job['distribution'], design=job['design'],
                equal_var=job['equal_var'], subject_ids=job['subject_ids'],
                group_count=len(job['groups']), alpha=alpha, posthoc='auto', correction='holm',
                metric=f'{METRIC}: {job["title"]}',
                metadata=dict(unit='m', source=str(source),
                              cohort='common.load_trials(): fixed 45 participants / 270 trials',
                              selection_reason=job['selection_reason'], assumptions=job['assumptions']))
            reports.append(report)
            progress.update()
        p_adjusted = multipletests([r['omnibus']['p_value'] for r in reports], method='holm')[1]
        output.mkdir(parents=True, exist_ok=True)
        (output / 'manifest.json').unlink(missing_ok=True)
        audit.to_csv(output / 'source_row_audit.csv', index=False, encoding='utf-8-sig')
        prepared.to_csv(output / 'prepared_subject_conditions.csv', index=False, encoding='utf-8-sig')
        summary, assumptions, descriptive, posthoc = [], [], [], []
        for job, report, adjusted in zip(jobs, reports, p_adjusted):
            family = dict(method='holm', family_size=7, p_adjusted=float(adjusted), significant=bool(adjusted < alpha))
            report['metadata']['primary_family_correction'] = family
            report['notes'].append('Individual significant uses raw p; use primary_family_correction for the seven-test Holm decision. Posthoc p adjusts only the three within-smoke pairs.')
            dest = output / job['id']
            write_report(report, dest, force=force)
            request = dict(groups={k: v.tolist() for k, v in job['groups'].items()},
                           distribution=job['distribution'], design=job['design'], group_count=len(job['groups']),
                           equal_var=job['equal_var'], subject_ids=job['subject_ids'], alpha=alpha,
                           posthoc='auto', correction='holm', metric=report['metric'], metadata=report['metadata'])
            save_json(dest / 'request.json', request)
            if job['design'] == 'repeated':
                job['paired_data'].to_csv(dest / 'paired_samples.csv', index=False, encoding='utf-8-sig')
                job['unmatched'].to_csv(dest / 'unmatched_subjects.csv', index=False, encoding='utf-8-sig')
            else:
                job['sample_data'].to_csv(dest / 'independent_samples.csv', index=False, encoding='utf-8-sig')
            summary.append(dict(comparison=job['id'], title=job['title'], design=job['design'],
                                group_count=len(job['groups']), sample_sizes='/'.join(str(len(v)) for v in job['groups'].values()),
                                distribution=job['distribution'], method=report['omnibus']['method'],
                                statistic=report['omnibus']['statistic'], p_raw=report['omnibus']['p_value'],
                                p_holm_7=float(adjusted), significant_raw=report['omnibus']['significant'],
                                significant_holm_7=bool(adjusted < alpha)))
            assumptions.extend(dict(comparison=job['id'], **c) for c in job['assumptions'])
            descriptive.extend(dict(comparison=job['id'], **d) for d in report['descriptive'])
            posthoc.extend(dict(comparison=job['id'], primary_passes_holm_7=bool(adjusted < alpha), **p) for p in report['posthoc'])
        summary = pd.DataFrame(summary)
        summary.to_csv(output / 'summary.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(assumptions).to_csv(output / 'assumptions.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(descriptive).to_csv(output / 'descriptive.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(posthoc, columns=['comparison','primary_passes_holm_7','group_a','group_b','method','statistic','p_value','p_adjusted','correction','significant']).to_csv(output / 'posthoc_all.csv', index=False, encoding='utf-8-sig')
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), input=str(source),
                        input_sha256=file_hash(source), invoker_sha256=file_hash(__file__),
                        common_sha256=file_hash(root / 'common.py'),
                        trial_input_sha256=file_hash(Path(__file__).with_name('TrialInput.py')),
                        cohort='common.load_trials(): 45 participants / 270 trials',
                        core_sha256=file_hash(Path(__file__).with_name('Analyze_Significance.py')),
                        alpha=alpha, assumption_alpha=assumption_alpha, input_rows=len(raw),
                        excluded_rows=int((audit.Status == 'excluded').sum()),
                        prepared_rows=len(prepared), subjects=int(prepared.SubjectID.nunique()),
                        duplicate_evidence=evidence, versions=reports[0]['versions'],
                        pandas_version=pd.__version__, primary_correction='Holm across all seven tests')
        write_overview(output, manifest, summary, reports, jobs, audit, assumptions)
        manifest['artifacts'] = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json')
        save_json(output / 'manifest.json', manifest)  # completion marker written last
        progress.update()
    print(summary[['comparison', 'method', 'p_raw', 'p_holm_7']].to_string(index=False))
    print(f'Report: {output / "report.md"}')
    return summary


def write_overview(output, manifest, summary, reports, jobs, audit, assumptions):
    lines = ['# 注视深度显著性分析', '',
             f'输入：`{manifest["input"]}`', '',
             f'源表 {manifest["input_rows"]} 行；排除 {manifest["excluded_rows"]} 行；整理后 {manifest["subjects"]} 人、{manifest["prepared_rows"]} 个受试者×场景观测。', '',
             '## 数据口径', '',
             '- 全部试次及 A/B/C、引导方式、烟雾分组来自 common.load_trials()，固定 45 人 / 270 试次。',
             '- 按 Scenario、ParticipantID、FileName 精确对齐；不推断会话身份，不合并或平均重复记录。',
             '- 源表缺失、重复、队列外试次或无效指标均报错终止，不自行剔除后继续统计。',
             '- 按 SubjectID 显式配对，未配对记录列于各比较目录；三类独立组验证受试者集合互不重叠。',
             '- 注视深度使用原始 GazeDepth_m，不截尾、不标准化、不依据离群程度剔除。单位为 m。PlotIncluded、PlotExclusionReason 和 IQR 界限仅作审计，不参与检验筛选。', '',
             '## 方法选择及多重比较', '',
             '- 使用 CSV 第 E 列 GazeDepth_m；按列名读取以避免列位置变化时取错指标。缺失、非有限或负值直接报错，未重算上游注视深度。',
             f'- 正态性阈值 {manifest["assumption_alpha"]}：配对检验检查高减低差值；三组检验分别检查各组 Shapiro-Wilk。',
             '- 未拒绝正态性不等于证明正态；此处按预设自动选型规则执行，小样本诊断能力有限。',
             '- 三组均通过正态性时，Levene（median）未拒绝等方差选 ANOVA，否则选 Welch ANOVA；任一组未通过正态性选 Kruskal-Wallis。',
             f'- 显著性阈值 α={manifest["alpha"]}，全部双侧；主结论按全部 7 次主检验的 Holm 校正判定，同时保留原始 p；不跨指标合并校正。',
             '- 三组主检验原始 p 显著时执行事后比较：ANOVA→Tukey；Welch→两两 Welch+Holm；Kruskal→Dunn+Holm。',
             '- 事后校正仅覆盖同一烟雾内的 3 对比较，不代表对全部主检验和全部事后比较统一控制错误率；未通过主检验 Holm 的事后结果仅作探索性参考。', '',
             '## 七次主检验', '',
             '| 对比 | 样本量 | 方法 | 原始 p | Holm p（7次） | 校正后显著 |',
             '|---|---|---|---:|---:|---|']
    for row in summary.itertuples():
        lines.append(f'| {row.title} | {row.sample_sizes} | {row.method} | {row.p_raw:.6g} | {row.p_holm_7:.6g} | {"是" if row.significant_holm_7 else "否"} |')
    lines += ['', '## 描述统计与两两比较', '']
    for job, report in zip(jobs, reports):
        lines += [f'### {job["title"]}', '', '| 组 | n | 均值 ± SD | 中位数 [Q1, Q3] |', '|---|---:|---:|---:|']
        for d in report['descriptive']:
            lines.append(f'| {d["group"]} | {d["n"]} | {d["mean"]:.3f} ± {d["sd"]:.3f} | {d["median"]:.3f} [{d["q1"]:.3f}, {d["q3"]:.3f}] |')
        if job['design'] == 'repeated':
            diff = job['groups']['HighSmoke'] - job['groups']['LowSmoke']
            lines += ['', f'高−低：平均差 {np.mean(diff):.3f}，配对差值中位数 {np.median(diff):.3f}。未完整配对人数：{len(job["unmatched"])}。']
        if report['posthoc']:
            lines += ['', '| A | B | 方法 | 校正 p | 组内校正后显著 |', '|---|---|---|---:|---|']
            for p in report['posthoc']:
                lines.append(f'| {p["group_a"]} | {p["group_b"]} | {p["method"]} | {p["p_adjusted"]:.6g} | {"是" if p["significant"] else "否"} |')
        lines += ['', f'[完整检验报告]({job["id"]}/report.md) · [实际调用参数]({job["id"]}/request.json)', '']
    lines += ['## 分布与方差检查', '', '| 对比 | 检查对象 | 方法 | n | p | 未拒绝假设 |', '|---|---|---|---:|---:|---|']
    for c in assumptions:
        p = f'{c["p_value"]:.6g}' if c['p_value'] is not None else '不可检验'
        lines.append(f'| {c["comparison"]} | {c["target"]} | {c["test"]} | {c["n"]} | {p} | {c["passed"]} |')
    lines += ['', '## 排除审计', '', '| CSV 行号 | 原始身份 | 场景 | 原因 |', '|---:|---|---|---|']
    for row in audit[audit.Status == 'excluded'].itertuples():
        lines.append(f'| {row.SourceRow} | {row.Subject} | {row.Scenario} | {row.Reason} |')
    lines += ['', '## 解释边界', '',
              '- 秩检验不直接证明均值差异；描述统计的方向与检验结论应分开理解。Wilcoxon 使用公共框架的渐近近似，小样本存在近似误差。',
              '- 本分析比较原始注视深度；未调整路线/出口差异。高低烟雾比较也可能受固定试次顺序影响，不能据此直接认定因果。', '',
              '## 文件与复现', '',
              '- summary.csv：七次主检验；assumptions.csv：正态性/方差检查；descriptive.csv：描述统计；posthoc_all.csv：全部事后比较。',
              '- source_row_audit.csv：每条源记录的处理；prepared_subject_conditions.csv：实际受试者×条件数据。',
              '- 各比较目录保存 request.json、报告、配对/独立样本；manifest.json 保存输入、脚本与公共模块哈希。',
              '- 从任意目录执行 `python <工程根>/INDICATORS_SIGNIFICANCE/Invoker_GazeDepth.py`；重算加 `--force`。', '',
              '方法参考：[SciPy Shapiro-Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)、[SciPy Levene](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html)。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--assumption-alpha', type=float, default=.05)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    output = args.output or args.root / 'INDICATORS_SIGNIFICANCE' / 'Results_GazeDepth'
    run(args.root, output, alpha=args.alpha, assumption_alpha=args.assumption_alpha, force=args.force)


if __name__ == '__main__':
    main()
