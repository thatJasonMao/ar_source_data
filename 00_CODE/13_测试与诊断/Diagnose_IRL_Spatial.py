"""Training/validation-only spatial information diagnostic. Not an IRL replacement."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits

ROOT=Path(r'F:\ExperimentData\Experiment_Data_A5')
sys.path[:0]=[str(ROOT/'0_RL_FRAMEWORK'),str(ROOT/'0_RL_SOURCE')]
from irl_data_v3 import iter_trials,StateEncoder,GUIDANCE
from Spatial_IRL_Rule import reward_mask,SpatialIRLRule
from irl_search_v4 import write_json

def metrics(y,p):
    return dict(auc=float(roc_auc_score(y,p)),nll=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)))

def main():
    started=time.time()
    dataset=ROOT/'0_RL_SOURCE/Results_IRL_Dataset_V3/20260927_165422_574880'
    result=ROOT/'TEST/Results_IRL_Spatial_Diagnostic';result.mkdir(exist_ok=True)
    frames={s:[] for s in ['train','val']}
    for rec,df in iter_trials(dataset,{'train','val'}):frames[rec['split']].append(df.loc[df.valid.eq(1)])
    frames={s:pd.concat(parts,ignore_index=True) for s,parts in frames.items()}
    encoder=StateEncoder().fit(frames['train']);mask=reward_mask(encoder.names).astype(bool)
    x={s:encoder.transform(df)[:,mask] for s,df in frames.items()}
    output=[]
    with threadpool_limits(limits=2):
        for guidance in GUIDANCE:
            tr=frames['train'].guidance.eq(guidance).to_numpy();va=frames['val'].guidance.eq(guidance).to_numpy()
            y=frames['train'].action.to_numpy()[tr];yv=frames['val'].action.to_numpy()[va]
            for label,model in [('linear',LogisticRegression(C=1.,max_iter=300)),('nonlinear_diagnostic',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,max_bins=64,l2_regularization=10,learning_rate=.07,early_stopping=False,random_state=20260926))]:
                model.fit(x['train'][tr],y)
                row=dict(guidance=guidance,model=label,train=metrics(y,model.predict_proba(x['train'][tr])[:,1]),val=metrics(yv,model.predict_proba(x['val'][va])[:,1]))
                output.append(row);print(json.dumps(row),flush=True)
    write_json(result/'diagnostic.json',dict(rows=output,inputs='four isovist metrics + decision distance + floor/stage/smoke ONLY',test_used=False,elapsed_seconds=time.time()-started,interpretation='diagnostic feasibility probe, not a replacement for IRL or a proven ceiling'))

if __name__=='__main__':main()
