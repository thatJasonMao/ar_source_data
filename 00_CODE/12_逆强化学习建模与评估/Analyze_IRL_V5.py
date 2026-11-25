"""Post-selection V5 comparisons and seed diagnostics; never selects a model."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from irl_search_v4 import write_json,digest
from irl_frame_optimization import frame_report
from Spatial_IRL_Rule import SpatialIRLRule,SPATIAL
from Analyze_IRL_V3_Optimization import participant_bootstrap

HERE=Path(__file__).resolve().parent
GUIDANCE=['ArrowRibbon','MiniMap','TextBar']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def analyze(out):
    out=Path(out);model=out/'Best_Model';evaluation=read(model/'evaluation.json');config=read(model/'config.json')
    arrays=dict(np.load(model/'evaluation_predictions.npz',allow_pickle=False))
    baseline=HERE/'Results_IRL/20260927_200838_252398_V3OptimizeBank/Best_Model'
    comparison=dict(test_used_for_selection=False,test_previously_inspected=True,baseline=str(baseline),same_three_guidance_rows=False)
    if (baseline/'config.json').exists() and read(baseline/'config.json').get('manifest_sha256')==config['parent_manifest_sha256']:
        oldconfig=read(baseline/'config.json')
        if oldconfig['manifest_sha256']!=config['parent_manifest_sha256']:raise ValueError('Baseline source differs')
        old=dict(np.load(baseline/'evaluation_predictions.npz',allow_pickle=False));reports={}
        for split in ['val','test']:
            ix=old[split+'_guidance']<3
            for n in ['action','guidance']:
                if not np.array_equal(arrays[split+'_'+n],old[split+'_'+n][ix]):raise ValueError('Baseline observation alignment differs')
            data={n:arrays[split+'_'+n] for n in ['action','person','guidance']}
            reports[split]=frame_report(data,old[split+'_probability'][ix],old[split+'_utility'][ix])
        comparison.update(same_three_guidance_rows=True,baseline_reports=reports,v5_reports={s:evaluation[s] for s in ['val','test']})
    else:comparison['baseline_unavailable_or_incompatible']=True
    write_json(out/'comparison_previous.json',comparison)
    rows=[]
    for split in ['val','test']:
        for i,g in enumerate(GUIDANCE):
            ix=arrays[split+'_guidance']==i
            for mode,n in [('policy','probability'),('spatial_reward','utility')]:
                r=participant_bootstrap(arrays[split+'_action'][ix],arrays[split+'_'+n][ix],arrays[split+'_person'][ix])
                rows.append(dict(r,split=split,guidance=g,mode=mode))
    write_json(out/'participant_bootstrap.json',dict(rows=rows,test_used_for_selection=False,test_previously_inspected=True,
        limitation='Fixed split with small per-guidance participant counts; exploratory reuse, not independent blind validation'))
    field=model/'Results_Global_Spatial_Field_V4';stability=[];verification={}
    if (field/'manifest.json').exists():
        fm=read(field/'manifest.json');frame=pd.read_csv(field/'global_spatial_rule_field.csv.gz');rule=SpatialIRLRule(model)
        if len(list(field.glob('*.png')))!=3 or list(model.rglob('*.pdf')):raise ValueError('Expected exactly three PNGs and no PDF')
        errors={}
        for g,part in frame.groupby('guidance'):
            if g not in GUIDANCE:raise ValueError('Excluded guidance in field')
            err=float(np.abs(rule.query_features(part,g)-part.relative_information_utility.to_numpy()).max())
            if err>1e-5:raise ValueError('Rule/CSV disagreement')
            errors[g]=err
        if fm['raw_csv_sha256']!=digest(field/'global_spatial_rule_field.csv.gz'):raise ValueError('Field checksum differs')
        verification=dict(png_count=3,pdf_count=0,raw_grid_smoothed=False,max_query_csv_error=errors)
        for q in read(out/'seed_checks.json'):
            if q['status']!='complete':continue
            folder=Path(q['output'])
            if not (folder/'encoder.json').exists():shutil.copy2(Path(q['cache'])/'encoder.json',folder/'encoder.json')
            if not (folder/'config.json').exists():
                c=dict(config,**q['config']);c.pop('model_bank',None);write_json(folder/'config.json',c)
            other=SpatialIRLRule(folder);g=q['guidance_component'];part=frame.loc[frame.guidance.eq(g)]
            a=part.relative_information_utility.to_numpy();b=other.query_features(part,g);n=max(1,int(np.ceil(.1*len(a))))
            topa=set(np.argsort(a)[-n:]);topb=set(np.argsort(b)[-n:])
            stability.append(dict(guidance=g,seed=q['config']['seed'],rank_correlation=float(spearmanr(a,b).statistic),top10_overlap=len(topa&topb)/n))
        selection=read(out/'selection.json');r=selection['candidate'];train=np.load(Path(r['frames'])/'train_x.npy',mmap_mode='r')
        names=read(Path(r['frames'])/'metadata.json')['spatial_input_names'];mask=rule.mask.astype(bool)
        from irl_data_v3 import NUMERIC
        query=frame.copy()
        for name in NUMERIC[5:]:query[name]=0.
        query['previous_action']=-1;x=rule.encoder.transform(query)[:,mask];numeric=[names.index('scaled_'+n) for n in NUMERIC[:5]]
        lo=train[:,numeric].min(0);hi=train[:,numeric].max(0)
        outside=((x[:,numeric]<lo-1e-6)|(x[:,numeric]>hi+1e-6)).any(1)
        write_json(out/'spatial_support_audit.json',dict(outside_training_numeric_range_fraction=float(outside.mean()),
            meaning='Marginal range only; not joint coverage or external validation'))
    write_json(out/'seed_stability.json',dict(rows=stability,test_used=False,grid='same unsmoothed 0.4m geometry grid'))
    m=read(Path(config['dataset'])/'manifest.json')
    assert set(r['guidance'] for r in m['trials'])==set(GUIDANCE)
    for g,relative in config['model_bank'].items():
        child=model/relative;state=__import__('torch').load(child/'reward.pt',map_location='cpu',weights_only=True)
        if state['deviation'].shape[0]!=3:raise ValueError('Unused Full reward branch remains')
        enc=read(child/'encoder.json')
        if any(k.split('/')[0] not in GUIDANCE for k in enc['contexts']):raise ValueError('Excluded guidance in serialized MDP')
    verification.update(status='passed',active_guidance=GUIDANCE,full_in_dataset=False,full_in_model=False,train_rows=evaluation['train']['n'],val_rows=evaluation['val']['n'],test_rows=evaluation['test']['n'])
    write_json(out/'delivery_verification.json',verification)
    table=['| 范式 | 上版策略 AUC | V5 策略 AUC | 上版空间 AUC | V5 空间 AUC |','|---|---:|---:|---:|---:|']
    for g in GUIDANCE:
        a=comparison.get('baseline_reports',{}).get('test',{}).get('by_guidance',{}).get(g);b=evaluation['test']['by_guidance'][g]
        table.append(f"| {g} | {a['auc']:.3f} | {b['auc']:.3f} | {a['spatial_reward_auc']:.3f} | {b['spatial_reward_auc']:.3f} |" if a else f"| {g} | unavailable | {b['auc']:.3f} | unavailable | {b['spatial_reward_auc']:.3f} |")
    selections=read(out/'selection.json')['by_guidance']
    lines=[f"- {g}：候选 {s['id']}，验证空间 AUC={s['validation']['spatial_reward_auc']:.3f}、验证人均 NLL={s['validation']['participant_macro_nll']:.3f}。" for g,s in selections.items()]
    outcome=''
    if comparison['same_three_guidance_rows']:
        previous=comparison['baseline_reports']['test'];current=evaluation['test']
        outcome=(f"配对三范式整体策略 AUC：{previous['auc']:.3f} → {current['auc']:.3f}；"
                 f"范式宏平均 AUC：{previous['guidance_macro_auc']:.3f} → {current['guidance_macro_auc']:.3f}；"
                 f"NLL：{previous['nll']:.3f} → {current['nll']:.3f}。"
                 "排除 Full 并重新选参不保证泛化改善，以以下分范式结果为准。"
                 "本对比同时重新拟合编码器、基函数和参数，不能单独归因于 Full 的正／负迁移。")
    report=f'''# V5 三种单一引导形式：训练结果

{outcome}

本轮从原始 V3 全量帧数据中排除全部 Full，重新拟合编码器、聚类、经验转移、空间基函数、三分支共享奖励与范式偏差。未复用四范式训练权重或基函数。

保留 {m['cohort_trials']} 个试次、{m['cohort_participants']} 人。原始人员 split 不变；LJN_C12 的 TextBar 原已排除，去掉 Full 后不再具有有效试次。训练 {evaluation['train']['n']} 帧，验证 {evaluation['val']['n']} 帧，测试 {evaluation['test']['n']} 帧。

{chr(10).join(table)}

上版来自已交付四范式优化规则库，但指标只重算同一组三范式观察帧；不把删除 Full 后的合并 AUC 和原四范式 AUC 直接比较。策略分数含动态继续价值，空间分数只用即时奖励差；最终地图是空间奖励差而非 gaze 概率。

V5 三范式测试合并 AUC={evaluation['test']['auc']:.3f}，范式宏平均 AUC={evaluation['test']['guidance_macro_auc']:.3f}，NLL={evaluation['test']['nll']:.3f}。是否变好以各范式结果为准，排除 Full 不保证每项提升。

## 验证选择

{chr(10).join(lines)}

每种范式在验证人均 NLL 最低值 +0.005 内选空间奖励 AUC 最大的候选。原始试次标签和时间戳不变，10s 曝光封顶沿用此前优化设置；每 5 轮验证，按验证损失降学习率和早停。种子检查不参与择优。

最终按范式选取三范式共同拟合的候选组成规则库，不是重新联合拟合的单一共享权重模型。跨范式奖励绝对幅度应谨慎比较。

## 交付与限制

模型：`{model}`。默认自动输出 3 张 PNG，无 PDF；数值格网 0.4m 不平滑，0.6m 高斯仅显示。点查询只需空间指标／决断距离和空间上下文，Full 查询明确拒绝。全局未观测位置仍是空间规则外推。

测试集此前已经被查看，本轮是探索性复用，不是独立盲测。Arrow/MiniMap 验证和测试各 3 人，TextBar 测试 2 人；大量相关帧不能代替独立受试者。按人 bootstrap 与种子空间场稳定性另存 JSON。训练和验证没有任何 Full 帧；原数据和上版结果完整保留。没有安装包、修改环境或提交代码。
'''
    (out/'TRAINING_REPORT.md').write_text(report,encoding='utf-8')
    print('V5_ANALYSIS='+str(out),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();analyze(args.run)

if __name__=='__main__':main()
