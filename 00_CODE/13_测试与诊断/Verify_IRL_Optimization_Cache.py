"""Verify full-frame cache provenance and training likelihood mass."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from irl_search_v4 import load_cache,digest,write_json
from irl_frame_optimization import load_frames

def main():
    p=argparse.ArgumentParser();p.add_argument('--search',type=Path,required=True);args=p.parse_args();rows=[]
    for f in args.search.glob('Frame_Caches/*/metadata.json'):
        meta=json.loads(f.read_text(encoding='utf-8'));mdp,_,_=load_cache(meta['cache'])
        data=load_frames(f.parent,'train');counts=np.zeros_like(mdp['counts'],dtype=np.float64)
        np.add.at(counts,(data['state'],data['action'].astype(int)),data['weight'])
        error=float(np.abs(counts-mdp['counts']).max())
        assert np.allclose(counts,mdp['counts'],atol=1e-6)
        assert np.isfinite(data['x']).all() and np.isin(data['action'],[0,1]).all()
        assert not any('speed' in n or 'previous_action' in n or 'elapsed' in n for n in meta['spatial_input_names'])
        for split,m in meta['splits'].items():
            obs=load_frames(f.parent,split)
            assert len(obs['action'])==m['rows']
            if split!='test':assert abs(float(obs['weight'].sum())-len(m['people']))<1e-4
        rows.append(dict(cache=str(f.parent),max_count_error=error,train_rows=len(data['action']),checks='passed'))
    write_json(args.search/'frame_cache_verification.json',dict(rows=rows,status='passed'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
