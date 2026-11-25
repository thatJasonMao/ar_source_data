"""Three within-group HRV repeated-measures tests and nine planned paired comparisons."""
from pathlib import Path
from itertools import combinations
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
try:
    from .HRVInput import prepare_hrv, group_wide, load_module, MODES
except ImportError:
    from HRVInput import prepare_hrv, group_wide, load_module, MODES


def save_json(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def run(root,output,mapping=None,force=False):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError('Results exist; use --force or another output')
    data,audit,coverage,source,mapping_path=prepare_hrv(root,mapping)
    core_path=root/'INDICATORS_SIGNIFICANCE'/'Analyze_Significance.py'
    core=load_module('hrv_significance_core',core_path)
    reports=[];requests=[];checks=[];wides=[]
    for group in tqdm(['A','B','C'],desc='HRV repeated comparisons'):
        wide=group_wide(data,group);wides.append(wide)
        group_checks=[]
        for earlier,later in combinations(MODES[group],2):
            diff=np.round(wide[later].to_numpy()-wide[earlier].to_numpy(),6)
            result=stats.shapiro(diff) if np.ptp(diff)>0 else None
            entry=dict(Group=group,Contrast=f'{later}-{earlier}',Test='Shapiro-Wilk',
                Statistic=float(result.statistic) if result else None,
                p_value=float(result.pvalue) if result else None,
                Passed=bool(result and result.pvalue>=.05))
            checks.append(entry);group_checks.append(entry)
        distribution='parametric' if all(c['Passed'] for c in group_checks) else 'nonparametric'
        request=dict(groups={m:np.round(wide[m].to_numpy(),6).tolist() for m in MODES[group]},
            distribution=distribution,design='repeated',sphericity=None,
            subject_ids={m:wide.index.tolist() for m in MODES[group]},alpha=.05,missing='raise',
            posthoc='always',correction='holm',metric=f'HRV stress-index change: Group {group}',
            metadata=dict(source=str(source),mapping=str(mapping_path),
                scope='Within-group three chronological guidance stages; not independent samples',
                aggregation='Use existing lower three-stage blocks in HRV_2.xlsx unchanged',
                assumptions=group_checks,
                method_rule='All three paired contrasts pass Shapiro -> repeated-measures ANOVA with GG; otherwise Friedman',
                posthoc_rule='All three within-panel pairs requested; Holm within panel, plus Holm across all nine reported separately',
                precision='Source values rounded to 6 decimals before tests; all observed source precision retained',
                interpretation='Guidance and chronological order effects are not separable'))
        reports.append(core.analyze_significance(**request));requests.append(request)
    adjusted=multipletests([r['omnibus']['p_value'] for r in reports],method='holm')[1]
    output.mkdir(parents=True,exist_ok=True)
    (output/'manifest.json').unlink(missing_ok=True)
    summary=[];pairs=[]
    for group,report,request,wide,p in zip(['A','B','C'],reports,requests,wides,adjusted):
        report['metadata']['omnibus_family_holm_3']=dict(p_adjusted=float(p),significant=bool(p<.05))
        report['notes'].append('Posthoc Holm controls three pairs within this panel; all nine-pair Holm is separately in posthoc_all.csv. Nonsignificant omnibus does not prohibit reporting requested planned contrasts, but do not call them omnibus-supported posthoc findings.')
        core.write_report(report,output/f'group_{group}',force=force)
        save_json(output/f'group_{group}'/'request.json',request)
        wide.to_csv(output/f'group_{group}'/'paired_samples.csv',encoding='utf-8-sig')
        o=report['omnibus']
        summary.append(dict(Group=group,N=len(wide),Method=o['method'],Statistic=o['statistic'],
            p_raw=o['p_value'],p_holm_3=float(p),SignificantHolm=bool(p<.05),
            EpsilonGG=o.get('epsilon_gg'),KendallsW=o.get('kendalls_w'),PartialEtaSquared=o.get('partial_eta_squared')))
        for pair in report['posthoc']:
            delta=wide[pair['group_b']]-wide[pair['group_a']]
            pairs.append(dict(Group=group,N=len(wide),**pair,
                MeanDeltaLaterMinusEarlier=float(delta.mean()),
                OmnibusPassesHolm=bool(p<.05)))
    for pair,p in zip(pairs,multipletests([r['p_value'] for r in pairs],method='holm')[1]):
        pair['p_holm_9']=float(p);pair['significant_holm_9']=bool(p<.05)
    summary=pd.DataFrame(summary);pair_table=pd.DataFrame(pairs)
    summary.to_csv(output/'summary.csv',index=False,encoding='utf-8-sig')
    pair_table.to_csv(output/'posthoc_all.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(output/'assumptions.csv',index=False,encoding='utf-8-sig')
    data.to_csv(output/'prepared_subject_stages.csv',index=False,encoding='utf-8-sig')
    audit.to_csv(output/'stage_value_audit.csv',index=False,encoding='utf-8-sig')
    coverage.to_csv(output/'cohort_coverage.csv',index=False,encoding='utf-8-sig')
    lines=['# HRV 三阶段组内显著性分析','','每组同一批受试者经历传统→单一引导→全量；实际每组人数见下表。',
        '读取原表已有压力指数变化量，不再对阶段作差后充当原始观测。组间不作独立比较。',
        '全部双侧，α=0.05。三项组内配对差值均未拒绝正态性时用重复测量 ANOVA（GG），否则 Friedman。',
        '三组总检验作 Holm 校正；每组的三个两两比较作组内 Holm。另列全九次 Holm 供整体图版推断使用。',
        '按照用户要求，无论总检验是否显著均报告预先指定的三对比较；总检验不显著时不称其为总检验支持的事后发现。',
        '固定阶段顺序使引导模态与时间/练习效应混杂；不能把差异全部解释为引导的因果效应。','',
        '|组|人数|方法|原始p|三组Holm p|','|---|---:|---|---:|---:|']
    for r in summary.itertuples():lines.append(f'|{r.Group}|{r.N}|{r.Method}|{r.p_raw:.6g}|{r.p_holm_3:.6g}|')
    lines+=['','|组|比较|原始p|组内Holm p|九次Holm p|','|---|---|---:|---:|---:|']
    for r in pair_table.itertuples():lines.append(f'|{r.Group}|{r.group_a} vs {r.group_b}|{r.p_value:.6g}|{r.p_adjusted:.6g}|{r.p_holm_9:.6g}|')
    lines+=['','差值均值方向统一为后阶段减前阶段；公共框架的配对 t 统计量方向为 group_a 减 group_b。',
        'Wilcoxon/Friedman 沿用公共框架渐近 p，Wilcoxon 去零差值；小样本近似结果需谨慎解释。',
        '完整映射、源单元格及汇总差异见配套 CSV；未作插补、IQR剔除或显著性驱动筛选。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths=[source,mapping_path,root/'common.py',core_path,Path(__file__),Path(__file__).with_name('HRVInput.py')]
    save_json(output/'manifest.json',dict(inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        alpha=.05,versions=reports[0]['versions'],participants=int(data.ParticipantID.nunique())))
    print(summary.to_string(index=False));print(pair_table[['Group','group_a','group_b','p_value','p_adjusted','p_holm_9']].to_string(index=False))
    return summary,pair_table


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output',type=Path)
    p.add_argument('--mapping',type=Path)
    p.add_argument('--force',action='store_true')
    a=p.parse_args();run(a.root,a.output or a.root/'INDICATORS_SIGNIFICANCE'/'Results_HRV',a.mapping,a.force)
