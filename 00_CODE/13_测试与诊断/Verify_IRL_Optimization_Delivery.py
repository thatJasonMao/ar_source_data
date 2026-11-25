"""Read-only numerical and artifact checks for an exported IRL rule bank."""
from pathlib import Path
import argparse,json,sys,tempfile,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent/'0_RL_FRAMEWORK'
sys.path.insert(0,str(ROOT))
from Spatial_IRL_Rule import SpatialIRLRule,GUIDANCE
from irl_search_v4 import digest,write_json
from Finalize_IRL_V3_Optimize import choose

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();best=args.run/'Best_Model';config=read(best/'config.json')
    field=best/'Results_Global_Spatial_Field_V4';manifest=read(field/'manifest.json')
    assert len(list(field.glob('*.png')))==4 and len(list(best.rglob('*.pdf')))==0
    csv=field/'global_spatial_rule_field.csv.gz'
    assert digest(csv)==manifest['raw_csv_sha256']
    assert not manifest['raw_csv_smoothed'] and not manifest['query_smoothed'] and not manifest['trajectory_input']
    assert digest(Path(config['dataset'])/'manifest.json')==config['manifest_sha256']
    for n,h in config['source_code_sha256'].items():assert digest(ROOT/n)==h
    frame=pd.read_csv(csv);rule=SpatialIRLRule(best);errors={}
    for g in GUIDANCE:
        part=frame.loc[frame.guidance.eq(g)]
        actual=rule.query_features(part,g)
        error=float(np.abs(actual-part.relative_information_utility.to_numpy()).max())
        assert error<1e-5,(g,error)
        errors[g]=error
        child=best/config['model_bank'][g]
        assert digest(child/'reward.pt')==manifest['reward_sha256'][g]
        assert digest(child/'mdp.npz')==read(child/'mdp_metadata.json')['files_sha256']['mdp.npz']
    # A portable rule-only copy must return identical values with no MDP or trial files.
    with tempfile.TemporaryDirectory(prefix='irl_rule_query_') as tmp:
        portable=Path(tmp);shutil.copy2(best/'config.json',portable/'config.json')
        for g,relative in config['model_bank'].items():
            target=portable/relative;target.mkdir(parents=True)
            for n in ['config.json','encoder.json','reward.pt','reward_basis.json']:
                source=best/relative/n
                if source.exists():shutil.copy2(source,target/n)
        query=SpatialIRLRule(portable)
        for g in GUIDANCE:
            part=frame.loc[frame.guidance.eq(g)].iloc[::101]
            np.testing.assert_allclose(query.query_features(part,g),rule.query_features(part,g),rtol=0,atol=0)
    selected,_=choose(ROOT/'Results_IRL/20260927_200339_538355_V3Resolution96')
    frozen=read(args.run/'selection.json')['by_guidance']
    assert all(selected[g]['id']==frozen[g]['id'] for g in GUIDANCE)
    assert read(best/'evaluation.json')['test_previously_inspected'] is True
    result=dict(status='passed',raw_field_rows=len(frame),png_count=4,pdf_count=0,
        max_query_csv_error=errors,portable_rule_without_mdp=True,selection_unchanged=True,
        training=False,dataset_manifest_unchanged=True)
    write_json(args.run/'delivery_verification.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
