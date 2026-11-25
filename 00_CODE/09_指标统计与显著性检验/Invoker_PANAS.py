"""Independent paired PA/NA invoker for the existing significance framework."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

SCORES = ['PA_Pre', 'NA_Pre', 'PA_Post', 'NA_Post']


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(raw, trials):
    """Keep complete participants, never expand questionnaires to trial rows."""
    data = raw.copy()
    if data.ParticipantID.isna().any() or data.ParticipantID.duplicated().any():
        raise ValueError('Missing or duplicate participant ID')
    if set(data.ParticipantID) != set(trials.ParticipantID):
        raise ValueError('Summary must cover exactly the common.load_trials participant set')
    counts = trials.groupby('ParticipantID').size()
    if not data.IncludedTrialCount.eq(data.ParticipantID.map(counts)).all():
        raise ValueError('Trial counts do not match common.load_trials')
    groups = trials.drop_duplicates('ParticipantID').set_index('ParticipantID').Group
    if not data.Group.eq(data.ParticipantID.map(groups)).all():
        raise ValueError('Group mismatch')
    if not data.MeasurementLevel.eq('participant_experiment_pre_post').all():
        raise ValueError('Incorrect measurement level')
    data[SCORES] = data[SCORES].apply(pd.to_numeric, errors='raise')
    for col in SCORES:
        if not data[col].dropna().isin(range(9, 46)).all():
            raise ValueError(f'Invalid PANAS score in {col}')
    complete = data[SCORES].notna().all(axis=1)
    audit = data[['ParticipantID', 'Group'] + SCORES].copy()
    audit['Status'] = np.where(complete, 'included_complete_pair', 'excluded_missing_measurement')
    audit['MissingFields'] = data[SCORES].apply(lambda r: ';'.join(r.index[r.isna()]), axis=1)
    paired = data.loc[complete].sort_values('ParticipantID').reset_index(drop=True)
    if len(paired) < 3:
        raise ValueError('At least three complete paired participants required')
    for metric in ['PA', 'NA']:
        paired[f'{metric}_Delta_PostMinusPre'] = paired[f'{metric}_Post'] - paired[f'{metric}_Pre']
    return paired, audit


def normality(diff, alpha):
    if np.ptp(diff) == 0 or len(diff) > 5000:
        return dict(test='shapiro_wilk', target='Post-Pre', n=len(diff), statistic=None,
                    p_value=None, passed=False, status='not_testable_constant_or_large_sample')
    result = stats.shapiro(diff)
    return dict(test='shapiro_wilk', target='Post-Pre', n=len(diff), statistic=float(result.statistic),
                p_value=float(result.pvalue), passed=bool(result.pvalue >= alpha), status='tested')


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(root, output, alpha=.05, assumption_alpha=.05, force=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not 0 < alpha < 1 or not 0 < assumption_alpha < 1:
        raise ValueError('Alpha must be between zero and one')
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError('Existing results: select a new --output or pass --force')
    common = load_module('panas_common', root/'common.py')
    core_path = root/'INDICATORS_SIGNIFICANCE'/'Analyze_Significance.py'
    core = load_module('panas_significance_core', core_path)
    folder = root/'INDICATORS'/'questionnaire_emotion'
    source = folder/'PANAS_participant_summary.csv'
    paired, audit = prepare_data(pd.read_csv(source), common.load_trials())
    extractor = load_module('panas_item_definitions', folder/'Extract_PANAS_Summary.py')
    books = {phase: pd.read_excel(folder/file) for phase, file in
             [('Pre', 'emotion_before.xlsx'), ('Post', 'emotion_after.xlsx')]}
    for _, row in tqdm(paired.iterrows(), total=len(paired), desc='Verify PANAS source'):
        for phase in ['Pre', 'Post']:
            raw = books[phase].iloc[int(row[f'SourceExcelRow_{phase}'])-2]
            if extractor.normalize_name(raw['1、您的姓名']) != row.ParticipantID.rsplit('_', 1)[0].upper():
                raise ValueError('Source questionnaire identity mismatch')
            for metric, items in [('PA', extractor.PA_ITEMS), ('NA', extractor.NA_ITEMS)]:
                values = pd.to_numeric(raw[items], errors='raise')
                if not values.isin(range(1, 6)).all() or values.sum() != row[f'{metric}_{phase}']:
                    raise ValueError('Source questionnaire score mismatch')
    # Compare with the existing figure when present, without importing plotting code.
    plot_source = root/'INDICATORS_GRAPH'/'PA&NA'/'PA_NA_paired_data.csv'
    if plot_source.exists():
        plotted = pd.read_csv(plot_source).sort_values('ParticipantID').reset_index(drop=True)
        pd.testing.assert_frame_equal(paired[['ParticipantID']+SCORES],
                                      plotted[['ParticipantID']+SCORES], check_dtype=False)
    reports, requests, rows, checks = [], [], [], []
    for metric in tqdm(['PA', 'NA'], desc='Paired tests'):
        diff = paired[f'{metric}_Delta_PostMinusPre'].to_numpy(dtype=float)
        check = normality(diff, assumption_alpha)
        checks.append(dict(metric=metric, **check))
        distribution = 'parametric' if check['passed'] else 'nonparametric'
        ids = paired.ParticipantID.tolist()
        request = dict(groups={phase: paired[f'{metric}_{phase}'].tolist() for phase in ['Post', 'Pre']},
                       distribution=distribution, design='repeated', group_count=2,
                       subject_ids={phase: ids for phase in ['Post', 'Pre']}, missing='raise',
                       alpha=alpha, posthoc='none', correction='holm', metric=f'PANAS {metric}: Post vs Pre',
                       metadata=dict(unit='9-item sum (9-45)', source=str(source),
                                     measurement_level='participant_experiment_pre_post',
                                     difference_direction='Post-Pre', assumptions=check,
                                     selection_rule='Shapiro on paired differences: p>=threshold -> paired t; otherwise Wilcoxon',
                                     assumption_alpha=assumption_alpha))
        report = core.analyze_significance(**request)
        reports.append(report)
        requests.append(request)
        row = dict(metric=metric, n_pairs=len(paired), method=report['omnibus']['method'],
                   pre_mean=float(paired[f'{metric}_Pre'].mean()), post_mean=float(paired[f'{metric}_Post'].mean()),
                   mean_delta_post_minus_pre=float(diff.mean()), median_delta=float(np.median(diff)),
                   zero_difference_n=int((diff==0).sum()), shapiro_p=check['p_value'],
                   statistic=report['omnibus']['statistic'], df=report['omnibus'].get('df'),
                   p_raw=report['omnibus']['p_value'], significant_raw=report['omnibus']['significant'])
        if distribution == 'parametric':
            sd = float(diff.std(ddof=1))
            half = float(stats.t.ppf(1-alpha/2, len(diff)-1)*stats.sem(diff))
            row.update(mean_delta_ci_low=float(diff.mean()-half), mean_delta_ci_high=float(diff.mean()+half),
                       confidence_level=1-alpha, cohens_dz=float(diff.mean()/sd))
        rows.append(row)
    adjusted = multipletests([r['p_raw'] for r in rows], method='holm')[1]
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').unlink(missing_ok=True)
    for metric, report, request, row, p in zip(['PA','NA'], reports, requests, rows, adjusted):
        row.update(p_holm_2=float(p), significant_holm_2=bool(p<alpha))
        report['metadata']['primary_family_correction'] = dict(method='holm', family_size=2, p_adjusted=float(p), significant=bool(p<alpha))
        report['notes'].append('Use primary_family_correction for the PA/NA two-test family decision; omnibus significance uses raw p.')
        core.write_report(report, output/metric, force=force)
        save_json(output/metric/'request.json', request)
    result = pd.DataFrame(rows)
    result.to_csv(output/'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(output/'assumptions.csv', index=False, encoding='utf-8-sig')
    paired.to_csv(output/'paired_samples.csv', index=False, encoding='utf-8-sig')
    audit.to_csv(output/'source_row_audit.csv', index=False, encoding='utf-8-sig')
    lines = ['# PANAS 前后测显著性检验', '',
             f'正式队列 45 人中纳入 {len(paired)} 名完整配对受试者；与现有 PA/NA 图一致。',
             '前后测按 ParticipantID 配对，未复制为试次观测；同名时间推断标记保留于 paired_samples.csv。', '',
             f'双侧检验，α={alpha}；对 Post−Pre 差值作 Shapiro–Wilk（阈值 {assumption_alpha}）。',
             '依照现有 Invoker 规则：未拒绝正态性选配对 t，否则选 Wilcoxon。未拒绝不等于证明正态。',
             '主结论使用 PA 与 NA 两次检验的 Holm 校正；原始 p 同时保留。不与其他指标合并校正。', '',
             '| 指标 | 配对人数 | 方法 | 前测均值 | 后测均值 | 后−前 | 原始 p | Holm p | 校正后显著 |',
             '|---|---:|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["metric"]} | {r["n_pairs"]} | {r["method"]} | {r["pre_mean"]:.4f} | {r["post_mean"]:.4f} | {r["mean_delta_post_minus_pre"]:.4f} | {r["p_raw"]:.6g} | {r["p_holm_2"]:.6g} | {"是" if r["significant_holm_2"] else "否"} |')
    lines += ['', 'summary.csv 的均值差置信区间为单项、未作多重校正的配对 t 区间；Cohen dz=差值均值/差值样本标准差。',
              '若选 Wilcoxon，使用公共框架的正态近似、去除零差值、不作连续性校正；不把秩检验直接称为均值检验。',
              '未显著不表示证明前后相等。缺失者清单见 source_row_audit.csv。', '',
              '参考：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html',
              '参考：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html']
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    inputs = [source, root/'common.py', core_path, folder/'Extract_PANAS_Summary.py',
              folder/'emotion_before.xlsx', folder/'emotion_after.xlsx']
    if plot_source.exists():
        inputs.append(plot_source)
    save_json(output/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        invoker_sha256=file_hash(__file__), input_sha256={str(p.relative_to(root)):file_hash(p) for p in inputs},
        alpha=alpha, assumption_alpha=assumption_alpha, paired_n=len(paired),
        family_correction='Holm across two PA/NA tests', versions=reports[0]['versions']))
    print(result.to_string(index=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--assumption-alpha', type=float, default=.05)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    run(args.root, args.output or args.root/'INDICATORS_SIGNIFICANCE'/'Results_PANAS',
        args.alpha, args.assumption_alpha, args.force)
