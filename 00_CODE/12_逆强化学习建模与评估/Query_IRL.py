"""Query a completed IRL reward model; no training and no policy optimization."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch

from irl_data import StateEncoder,GUIDANCE,NUMERIC,assign_states
from irl_core import LinearReward,soft_policy,expected_burden


class IRLQuery:
    def __init__(self, model_dir):
        model_dir=Path(model_dir)
        self.config=json.loads((model_dir/'config.json').read_text(encoding='utf-8'))
        if self.config.get('status')!='complete':raise ValueError('Model is not complete')
        encoding=json.loads((model_dir/'encoder.json').read_text(encoding='utf-8'))
        self.encoder=StateEncoder()
        self.encoder.median=np.asarray(encoding['encoder']['median'])
        self.encoder.scale=np.asarray(encoding['encoder']['scale'])
        if self.encoder.names!=encoding['encoder']['names']:raise ValueError('Encoder schema mismatch')
        self.contexts=encoding['contexts']
        with np.load(model_dir/'mdp.npz',allow_pickle=False) as data:
            self.mdp={k:data[k] for k in data.files}
        model=LinearReward(len(self.encoder.names))
        model.load_state_dict(torch.load(model_dir/'reward.pt',map_location='cpu',weights_only=True))
        deviation=model.deviation.detach().numpy()
        self.weights=model.shared.detach().numpy()[None,:]+deviation-deviation.mean(0,keepdims=True)
        with torch.no_grad():
            features=torch.from_numpy(self.mdp['centers'])
            guidance=torch.from_numpy(self.mdp['guidance_index']).long()
            transitions=torch.from_numpy(self.mdp['transitions'])
            reward=model(features,guidance)
            terminal=int(self.mdp['terminal_index']) if 'terminal_index' in self.mdp else None
            logp,q=soft_policy(reward,transitions,self.config['horizon'],self.config['gamma'],terminal)
            prob,burden=expected_burden(logp.exp(),transitions,torch.from_numpy(self.mdp['event_cost']),self.config['horizon'],encoding['step_s'])
        self.reward,self.policy,self.q=reward.numpy(),logp.exp().numpy(),q.numpy()
        self.prob,self.burden=prob.numpy(),burden.numpy()

    def query_frame(self, frame):
        required=NUMERIC+['guidance','smoke','previous_action','floor','stage']
        if not set(required)<=set(frame.columns):raise ValueError('Missing state fields: '+str(sorted(set(required)-set(frame.columns))))
        if not frame.previous_action.isin([-1,0,1]).all():raise ValueError('Invalid previous_action')
        x=self.encoder.transform(frame)
        states,distance=assign_states(frame,x,self.contexts,self.mdp['centers'])
        result=[]
        for i,s in enumerate(states):
            if s<0:
                result.append(dict(status='unsupported_context'))
                continue
            n=float(self.mdp['transition_counts'][s].min())
            supported=n>=self.config['min_support'] and distance[i]<=1.5*self.mdp['radius'][s]
            utility=float(x[i]@self._weights(frame.iloc[i].guidance))
            result.append(dict(status='ok' if supported else 'low_support',state_id=int(s),
                state_distance=float(distance[i]),transition_support_min=n,
                # The linear rule is pointwise; policy/diagnosis use the fitted state abstraction.
                reward_contrast=utility,p_look=float(self.policy[s,1]),q_contrast=float(self.q[s,1]-self.q[s,0]),
                trap_probability=float(self.prob[s]) if supported else None,
                trap_burden_s=float(self.burden[s]) if supported else None))
        return pd.DataFrame(result)

    def _weights(self, guidance):
        return self.weights[GUIDANCE.index(guidance)]

    def query_position(self,x,y,z,*,guidance,smoke,elapsed_s,previous_action,history):
        """History contains speed_prev etc.; missing optional dynamics stay NaN.

        No invented typical person is silently supplied for a spatial map.
        """
        if guidance not in GUIDANCE or smoke not in {'HighSmoke','LowSmoke'}:
            return dict(status='unsupported_context')
        source=Path(__file__).resolve().parent.parent/'0_RL_SOURCE'
        if str(source) not in sys.path:sys.path.insert(0,str(source))
        from spatial_isovist_core import get_engine
        from Analyze_NextDecisionDistance import get_engine as decision_engine
        r=get_engine().query(x,y,z,7.5 if smoke=='HighSmoke' else 12.5)
        d=decision_engine().query(x,y,z)
        if r.status!='ok' or d.status!='ok':return dict(status=r.status if r.status!='ok' else d.status)
        values={name:float(history.get(name,np.nan)) for name in NUMERIC}
        values.update(concavity=r.concavity,distance_variation=r.distance_variation,
            elongation=r.elongation,horizontal_size=r.horizontal_size,
            next_decision_distance=d.next_decision_distance,guidance=guidance,smoke=smoke,
            elapsed_s=elapsed_s,previous_action=previous_action,floor=d.floor,stage=d.stage)
        return self.query_frame(pd.DataFrame([values])).iloc[0].to_dict()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='CSV with explicit state fields')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose a new filename')
    model=IRLQuery(args.model_dir)
    frame=pd.read_csv(args.input)
    result=model.query_frame(frame)
    pd.concat([frame.reset_index(drop=True),result],axis=1).to_csv(args.output,index=False)


if __name__=='__main__':main()
