"""Reproduce state-count consistency failures without fitting reward models."""
from pathlib import Path
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from irl_data_v4 import build_mdp
from irl_search_v4 import pack_observations,run_duration_table,write_json

def main():
    dataset=ROOT/'0_RL_SOURCE/Results_IRL_Dataset_V3/20260927_165422_574880'
    with threadpool_limits(limits=2):
        mdp=build_mdp(dataset,48,20260926,.5,run_cap_s=5.)
        obs,_=pack_observations(dataset,mdp,['train'],5.,run_duration_table(dataset))
    diff=obs['train']['balanced'].sum(0)-mdp['counts']
    result=dict(max_absolute_count_difference=float(np.abs(diff).max()),absolute_mass_difference=float(np.abs(diff).sum()),
        affected_state_actions=int((np.abs(diff)>1e-6).sum()),states=len(mdp['centers']),passes=bool(np.allclose(diff,0,atol=1e-6)))
    print(json.dumps(result),flush=True)
    folder=ROOT/'TEST/Results_IRL_State_Diagnostic';folder.mkdir(exist_ok=True);write_json(folder/'diagnostic.json',result)
    np.savez_compressed(folder/'mdp_debug.npz',**{k:mdp[k] for k in ['centers','counts','raw_counts','guidance_index']})
    write_json(folder/'encoder.json',dict(encoder=mdp['encoder'].as_dict(),contexts=mdp['contexts']))

if __name__=='__main__':main()
